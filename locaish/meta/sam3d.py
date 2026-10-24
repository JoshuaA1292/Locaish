"""SAM 3D Objects and SAM 3D Body, over HTTP from the Modal app.

The two open-weight SAM 3D models need a CUDA GPU with more memory than any
laptop has, so they run in `locaish/meta/modal_app.py` on Modal and Locaish
calls them here. Two verbs:

- `lift_object(image, mask_png)`: SAM 3D Objects turns one masked object in
  a frame into a textured gaussian splat (and a vertex-coloured GLB mesh when
  the model produced one). The splat comes back as a standard 3DGS PLY and is
  decoded with `video.splat.read_gaussian_ply`, so a lifted sofa is the same
  kind of thing as the Brush layer the renderer already composites.
  `fit_to_box` then maps the splat's bounding box onto a measured footprint
  and `to_field_npz` writes the field the set-pieces module consumes.
- `lift_body(image)`: SAM 3D Body turns one person in a frame into a Momentum
  Human Rig mesh. The server already stands the figure up (+Z up, feet at
  z=0, facing +Y); `to_body_npz` re-checks that from the vertex extents and
  writes the cast module's contract.

Configuration: `LOCAISH_SAM3D_URL` and `LOCAISH_SAM3D_TOKEN`. Modal gives
every `fastapi_endpoint` its own hostname, so the URL is a template with an
`{endpoint}` slot, e.g. `https://ws--locaish-sam3d-{endpoint}.modal.run`; a
URL without the slot is treated as a base and `/object`, `/body` appended.
`LOCAISH_SAM3D_OBJECT_URL` / `LOCAISH_SAM3D_BODY_URL` override one endpoint.

Nothing here retries: a lift is a 30-120 s GPU job and the caller decides
whether a failure is fatal. Non-2xx answers and malformed bodies raise
`Sam3DError`, a `MetaError`, so one `except MetaError` covers the whole
`locaish.meta` package. Tests pass an `httpx` `transport=`; no network.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .client import MetaError

__all__ = [
    "Sam3DError",
    "ObjectLift",
    "BodyLift",
    "configured",
    "endpoint_url",
    "lift_object",
    "lift_body",
    "gaussian_cov",
    "fit_to_box",
    "to_field_npz",
    "upright_body",
    "to_body_npz",
    "decode_gaussian_ply",
]

_DEFAULT_TIMEOUT_S = 300.0


class Sam3DError(MetaError):
    """The Modal app answered with an error, did not answer, or answered nonsense."""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def _token() -> str | None:
    return os.environ.get("LOCAISH_SAM3D_TOKEN") or None


def configured() -> bool:
    """Whether the Modal app can be called from this process."""
    return bool(os.environ.get("LOCAISH_SAM3D_URL")) and bool(_token())


def endpoint_url(name: str) -> str:
    """The URL for `object` or `body`, from the env (see module docstring)."""
    if name not in ("object", "body"):
        raise ValueError(f"unknown SAM 3D endpoint {name!r}")
    override = os.environ.get(f"LOCAISH_SAM3D_{name.upper()}_URL")
    if override:
        return override
    base = os.environ.get("LOCAISH_SAM3D_URL")
    if not base:
        raise Sam3DError("LOCAISH_SAM3D_URL is not set")
    if "{endpoint}" in base:
        return base.format(endpoint=name)
    return base.rstrip("/") + "/" + name


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@dataclass
class ObjectLift:
    """One object lifted to 3D.

    `gaussians` is the dict `read_gaussian_ply` returns (`xyz`, `rgb01`,
    `opacity`, `scale`, `quat`), in the model's own object-local frame: a
    unit-ish box, Z up (the notebook's `_fix_gaussian_alignment` swaps Y and
    Z only to render with a Y-up camera, which is why `fit_to_box` needs no
    more than a yaw). `None` when the server returned no splat.
    """

    gaussians: dict[str, np.ndarray] | None
    mesh_glb: bytes | None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BodyLift:
    """One person lifted to a Momentum Human Rig mesh.

    `vertices` (V, 3) and `keypoints` (K, 3) are in the model's units with
    the figure stood up by the server (+Z up, feet at z=0, facing +Y);
    `height_units` is the server's vertical extent in those units. The cast
    module rescales to the typed height, so the units never need to be known.
    """

    vertices: np.ndarray
    faces: np.ndarray
    keypoints: np.ndarray | None
    height_units: float
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------


def _post(name: str, body: dict[str, Any], *, timeout_s: float, transport: Any) -> dict[str, Any]:
    import httpx

    token = _token()
    if not token:
        raise Sam3DError("LOCAISH_SAM3D_TOKEN is not set")
    url = endpoint_url(name)
    # Modal cuts a web request at 150 s and answers 303 to a result URL that
    # blocks until the job is done; httpx turns the 303 into a GET of that
    # URL, which is exactly the continuation Modal intends.
    try:
        with httpx.Client(timeout=timeout_s, transport=transport, follow_redirects=True) as http:
            r = http.post(url, json=body, headers={"Authorization": f"Bearer {token}"})
    except httpx.HTTPError as exc:
        raise Sam3DError(f"{name}: {type(exc).__name__}: {exc}") from exc
    if r.status_code >= 400:
        raise Sam3DError(f"{name} returned {r.status_code}: {r.text[:400]}",
                         status=r.status_code, body=r.text)
    try:
        data = r.json()
    except ValueError as exc:
        raise Sam3DError(f"{name}: the answer was not JSON: {r.text[:200]}") from exc
    if not isinstance(data, dict):
        raise Sam3DError(f"{name}: expected a JSON object, got {type(data).__name__}")
    if data.get("error"):
        raise Sam3DError(f"{name}: {data['error']}", body=json.dumps(data)[:400])
    return data


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(s: Any, what: str) -> bytes:
    if not isinstance(s, str):
        raise Sam3DError(f"{what}: expected a base64 string")
    try:
        return base64.b64decode(s, validate=True)
    except (ValueError, TypeError) as exc:
        raise Sam3DError(f"{what}: not valid base64") from exc


# ---------------------------------------------------------------------------
# objects
# ---------------------------------------------------------------------------


def decode_gaussian_ply(data: bytes) -> dict[str, np.ndarray]:
    """A 3DGS PLY held in memory, decoded by `video.splat.read_gaussian_ply`.

    That reader takes a path (it streams a 6M-point Brush file), so the bytes
    go through a temporary file; SAM 3D Objects splats are tens of thousands
    of gaussians, so the copy costs nothing worth avoiding.
    """
    from ..video.splat import read_gaussian_ply

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "lift.ply"
        p.write_bytes(data)
        try:
            return read_gaussian_ply(p)
        except (ValueError, KeyError) as exc:
            raise Sam3DError(f"gaussian PLY could not be decoded: {exc}") from exc


def lift_object(
    image: bytes,
    mask_png: bytes,
    *,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    transport: Any = None,
    seed: int = 42,
) -> ObjectLift:
    """SAM 3D Objects: one frame plus one object mask in, a gaussian splat out.

    `image` is PNG or JPEG bytes of the whole frame; `mask_png` is a PNG of
    the same size whose non-zero pixels are the object (the server embeds
    it as the alpha channel, which is how the model takes its mask).
    """
    data = _post("object", {"image_png_b64": _b64(image), "mask_png_b64": _b64(mask_png), "seed": seed},
                 timeout_s=timeout_s, transport=transport)
    gauss = None
    if data.get("gaussians_ply_b64"):
        gauss = decode_gaussian_ply(_unb64(data["gaussians_ply_b64"], "gaussians_ply_b64"))
    mesh = _unb64(data["mesh_glb_b64"], "mesh_glb_b64") if data.get("mesh_glb_b64") else None
    meta = dict(data.get("meta") or {})
    if gauss is not None:
        meta.setdefault("n_gaussians", int(len(gauss["xyz"])))
    return ObjectLift(gaussians=gauss, mesh_glb=mesh, meta=meta)


def gaussian_cov(scale: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """World covariance Σ = R S Sᵀ Rᵀ per gaussian, as `film.render` builds it.

    `scale` (N, 3) standard deviations along the gaussian's own axes, `quat`
    (N, 4) wxyz. The columns of R are the gaussian's axes, so scaling the
    columns gives RS and the product is the covariance.
    """
    from ..film.render import _quat_to_rot

    q = np.asarray(quat, dtype=np.float64).reshape(-1, 4)
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    R = _quat_to_rot(q)                                      # (N, 3, 3)
    RS = R * np.asarray(scale, dtype=np.float64).reshape(-1, 1, 3)
    return RS @ np.transpose(RS, (0, 2, 1))


def _box(xyz: np.ndarray, opacity: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """The bounding box of the solid part of a splat: gaussians with opacity
    above 0.5 when there are enough of them, every gaussian otherwise. The
    notebook's own normaliser (`normalized_gaussian`) uses opacity > 0.9 for
    the same reason: a few faint stragglers would otherwise set the size."""
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if opacity is not None:
        solid = np.asarray(opacity).reshape(-1) > 0.5
        if solid.sum() >= 16:
            pts = pts[solid]
    if len(pts) == 0:
        raise Sam3DError("the lift holds no gaussians to fit")
    return pts.min(axis=0), pts.max(axis=0)


def _yaw(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def fit_to_box(
    gaussians: dict[str, np.ndarray],
    size_xyz_m: tuple[float, float, float] | np.ndarray,
    centre_xyz_m: tuple[float, float, float] | np.ndarray,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """A 4x4 that puts the lift inside a measured box in the twin's frame.

    Uniform scale, chosen as the smallest of the three per-axis ratios so no
    axis exceeds the box (an object's aspect ratio is the model's, not the
    label's); then a yaw about Z; then a translation that puts the scaled,
    rotated bounding-box centre at `centre_xyz_m`... except vertically: the
    base of the object lands on the base of the box, because a sofa sits on
    the floor rather than floating at the centroid of its label.
    Returns M with p_twin = M @ [p_local, 1].
    """
    lo, hi = _box(gaussians["xyz"], gaussians.get("opacity"))
    extent = np.maximum(hi - lo, 1e-9)
    size = np.asarray(size_xyz_m, dtype=np.float64).reshape(3)
    if np.any(size <= 0):
        raise ValueError("size_xyz_m must be positive")
    s = float(np.min(size / extent))
    R = _yaw(yaw_deg)
    centre_local = (lo + hi) / 2.0
    centre = np.asarray(centre_xyz_m, dtype=np.float64).reshape(3)
    # The yaw turns the local centre too; the translation is what is left.
    t = centre - s * (R @ centre_local)
    # Base on the floor of the box: the box's bottom is centre_z - size_z/2,
    # and the object's bottom after scale is s * lo_z above its centre's shift.
    t[2] = (centre[2] - size[2] / 2.0) - s * lo[2]
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = t
    return M


def to_field_npz(lift: ObjectLift, path: str | Path, *, transform: np.ndarray | None = None) -> Path:
    """Write the gaussian field the set-pieces module and `film.render` read:
    `xyz` (N, 3), `rgb` (N, 3) float 0-1, `opacity` (N,), `cov` (N, 3, 3).

    `transform` is a 4x4 (from `fit_to_box` or anything else) applied before
    writing: positions through the affine, covariances through the linear
    part A as A Σ Aᵀ, which carries a uniform scale and a rotation exactly.
    """
    if lift.gaussians is None:
        raise Sam3DError("the lift holds no gaussians")
    g = lift.gaussians
    xyz = np.asarray(g["xyz"], dtype=np.float64)
    cov = gaussian_cov(g["scale"], g["quat"])
    if transform is not None:
        M = np.asarray(transform, dtype=np.float64)
        if M.shape != (4, 4):
            raise ValueError("transform must be 4x4")
        A, t = M[:3, :3], M[:3, 3]
        xyz = xyz @ A.T + t
        cov = A[None] @ cov @ A.T[None]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        xyz=xyz.astype(np.float64),
        rgb=np.asarray(g["rgb01"], dtype=np.float32).reshape(-1, 3),
        opacity=np.asarray(g["opacity"], dtype=np.float32).reshape(-1),
        cov=cov.astype(np.float32),
        source=np.array("sam-3d-objects"),
        meta=np.array(json.dumps(lift.meta, default=str)),
    )
    return path


# ---------------------------------------------------------------------------
# body
# ---------------------------------------------------------------------------


def lift_body(
    image: bytes,
    *,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    transport: Any = None,
) -> BodyLift:
    """SAM 3D Body: one frame with one person in, an upright MHR mesh out.

    When the frame shows several people the server returns the one with the
    largest detector box and says so in `meta["n_people"]`.
    """
    data = _post("body", {"image_png_b64": _b64(image)}, timeout_s=timeout_s, transport=transport)
    try:
        v = np.asarray(data["vertices"], dtype=np.float64).reshape(-1, 3)
        f = np.asarray(data["faces"], dtype=np.int64).reshape(-1, 3)
    except (KeyError, TypeError, ValueError) as exc:
        raise Sam3DError(f"body: malformed vertices/faces: {exc}") from exc
    if len(v) < 4 or len(f) < 1:
        raise Sam3DError(f"body: {len(v)} vertices and {len(f)} faces is not a mesh")
    if f.min() < 0 or f.max() >= len(v):
        raise Sam3DError("body: face indices out of range")
    kp = data.get("keypoints_3d")
    keypoints = None
    if kp:
        try:
            keypoints = np.asarray(kp, dtype=np.float64).reshape(-1, 3)
        except (TypeError, ValueError) as exc:
            raise Sam3DError(f"body: malformed keypoints_3d: {exc}") from exc
    height = data.get("height_units")
    if height is None:
        height = float(v[:, 2].max() - v[:, 2].min())
    return BodyLift(vertices=v, faces=f, keypoints=keypoints,
                    height_units=float(height), meta=dict(data.get("meta") or {}))


_AXIS_ROTS = {
    # Rotations that bring axis k of the input onto +Z.
    0: np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),   # x -> z (about Y)
    1: np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),   # y -> z (about X)
    2: np.eye(3),
}


def upright_body(vertices: np.ndarray, keypoints: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, dict[str, Any]]:
    """Re-check the server's orientation from the extents and fix it if wrong.

    The contract is +Z up, feet at z=0, facing +Y. The server orients from
    the joints, but a body is also self-describing: a standing figure is
    taller than it is wide, and wider (shoulder to shoulder, X) than it is
    deep (chest to back, Y). So: the longest extent goes to Z if it is not
    there already; if depth exceeds width the figure is yawed 90 degrees;
    the feet are dropped to z=0 and the xy centroid moved to the origin.
    Returns (vertices, keypoints, R_applied, notes). Facing +Y versus -Y
    cannot be told from extents alone and is left to the server's joints.
    """
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    kp = None if keypoints is None else np.asarray(keypoints, dtype=np.float64).reshape(-1, 3)
    notes: dict[str, Any] = {}
    ext = v.max(axis=0) - v.min(axis=0)
    R = np.eye(3)
    tallest = int(np.argmax(ext))
    if tallest != 2:
        R = _AXIS_ROTS[tallest] @ R
        notes["rotated_up_axis_from"] = "xyz"[tallest]
        v = v @ R.T
        kp = None if kp is None else kp @ R.T
        ext = v.max(axis=0) - v.min(axis=0)
    if ext[1] > ext[0] * 1.15:
        # Deeper than wide: turned sideways. Quarter turn about Z.
        Ry = _yaw(90.0)
        R = Ry @ R
        v = v @ Ry.T
        kp = None if kp is None else kp @ Ry.T
        notes["yawed_90"] = True
    shift = np.array([*(v[:, :2].min(axis=0) + v[:, :2].max(axis=0)) / 2.0, v[:, 2].min()])
    v = v - shift
    kp = None if kp is None else kp - shift
    notes["shift"] = shift.tolist()
    return v, kp, R, notes


def to_body_npz(lift: BodyLift, path: str | Path) -> Path:
    """Write the cast module's contract: `vertices` (V, 3) float upright
    (+Z up, feet at z=0, facing +Y), `faces` (F, 3) int, `height_m` (the
    lift's height in its own units, passed through: the cast module
    rescales to the typed height so the units cancel), plus `keypoints`
    when the server gave them and `rgb` when a colour is known.
    """
    v, kp, R, notes = upright_body(lift.vertices, lift.keypoints)
    height = float(v[:, 2].max() - v[:, 2].min())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "vertices": v.astype(np.float64),
        "faces": np.asarray(lift.faces, dtype=np.int64).reshape(-1, 3),
        "height_m": np.array(height, dtype=np.float64),
        "source": np.array("sam-3d-body"),
        "meta": np.array(json.dumps({**lift.meta, "client_orientation": notes,
                                     "server_height_units": lift.height_units}, default=str)),
    }
    if kp is not None:
        arrays["keypoints"] = kp.astype(np.float64)
    rgb = lift.meta.get("rgb")
    if rgb is not None:
        arrays["rgb"] = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)
    np.savez_compressed(path, **arrays)
    return path
