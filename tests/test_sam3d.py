"""The SAM 3D client and the Modal app's plain helpers: no modal, no network.

The HTTP side runs through `httpx.MockTransport`; the gaussian PLY the
server would return is written here with the attribute names
`video.splat.read_gaussian_ply` expects (x y z nx ny nz f_dc_* opacity
scale_* rot_*), so the decode path is the real one end to end.
"""

from __future__ import annotations

import base64
import importlib
import io
import json
import sys

import httpx
import numpy as np
import pytest
from PIL import Image

from locaish.meta import MetaError
from locaish.meta import sam3d
from locaish.video.splat import SH_C0


# ---------------------------------------------------------------------------
# fixtures and builders
# ---------------------------------------------------------------------------


def _ply_bytes(xyz, rgb01, opacity, scale, quat) -> bytes:
    """A standard 3DGS binary PLY (what `GaussianModel.save_ply` writes)."""
    n = len(xyz)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n"
    header += "".join(f"property float {nm}\n" for nm in names) + "end_header\n"
    f_dc = (np.asarray(rgb01) - 0.5) / SH_C0
    op = np.asarray(opacity, dtype=np.float64)
    logit = np.log(op / (1.0 - op))
    cols = np.concatenate([
        np.asarray(xyz), np.zeros((n, 3)), f_dc, logit[:, None], np.log(np.asarray(scale)), np.asarray(quat),
    ], axis=1).astype("<f4")
    return header.encode("ascii") + cols.tobytes()


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _unit_quat(axis, deg):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    h = np.radians(deg) / 2.0
    return np.array([np.cos(h), *(np.sin(h) * axis)])


def _rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("LOCAISH_SAM3D_URL", "https://ws--locaish-sam3d-{endpoint}.modal.run")
    monkeypatch.setenv("LOCAISH_SAM3D_TOKEN", "s3cret")
    monkeypatch.delenv("LOCAISH_SAM3D_OBJECT_URL", raising=False)
    monkeypatch.delenv("LOCAISH_SAM3D_BODY_URL", raising=False)


@pytest.fixture
def splat():
    rng = np.random.default_rng(0)
    n = 40
    xyz = rng.uniform(-0.5, 0.5, size=(n, 3))
    rgb = rng.uniform(0.05, 0.95, size=(n, 3))
    opacity = rng.uniform(0.6, 0.99, size=n)
    scale = rng.uniform(0.01, 0.1, size=(n, 3))
    quat = rng.normal(size=(n, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return {"xyz": xyz, "rgb01": rgb, "opacity": opacity, "scale": scale, "quat": quat}


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def test_configured_needs_both_url_and_token(monkeypatch):
    monkeypatch.delenv("LOCAISH_SAM3D_URL", raising=False)
    monkeypatch.delenv("LOCAISH_SAM3D_TOKEN", raising=False)
    assert sam3d.configured() is False
    monkeypatch.setenv("LOCAISH_SAM3D_URL", "https://x.modal.run")
    assert sam3d.configured() is False
    monkeypatch.setenv("LOCAISH_SAM3D_TOKEN", "t")
    assert sam3d.configured() is True


def test_endpoint_url_template_base_and_override(monkeypatch, env):
    assert sam3d.endpoint_url("object") == "https://ws--locaish-sam3d-object.modal.run"
    assert sam3d.endpoint_url("body") == "https://ws--locaish-sam3d-body.modal.run"
    monkeypatch.setenv("LOCAISH_SAM3D_URL", "https://proxy.example/sam3d/")
    assert sam3d.endpoint_url("body") == "https://proxy.example/sam3d/body"
    monkeypatch.setenv("LOCAISH_SAM3D_BODY_URL", "https://elsewhere.test/b")
    assert sam3d.endpoint_url("body") == "https://elsewhere.test/b"
    with pytest.raises(ValueError):
        sam3d.endpoint_url("hands")
    monkeypatch.delenv("LOCAISH_SAM3D_URL")
    with pytest.raises(sam3d.Sam3DError):
        sam3d.endpoint_url("object")


# ---------------------------------------------------------------------------
# objects
# ---------------------------------------------------------------------------


def test_lift_object_decodes_ply_sends_auth_and_payload(env, splat):
    ply = _ply_bytes(**splat)
    image = _png(np.full((8, 6, 3), 200, np.uint8))
    mask = _png(np.full((8, 6), 255, np.uint8))
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "gaussians_ply_b64": base64.b64encode(ply).decode(),
            "mesh_glb_b64": base64.b64encode(b"glTF-fake").decode(),
            "meta": {"seconds": 31.2, "pose": {"scale": [[0.4, 0.4, 0.4]]}},
        })

    lift = sam3d.lift_object(image, mask, transport=httpx.MockTransport(handler))
    assert seen["url"] == "https://ws--locaish-sam3d-object.modal.run"
    assert seen["auth"] == "Bearer s3cret"
    assert base64.b64decode(seen["body"]["image_png_b64"]) == image
    assert base64.b64decode(seen["body"]["mask_png_b64"]) == mask
    assert seen["body"]["seed"] == 42
    g = lift.gaussians
    assert set(g) == {"xyz", "rgb01", "opacity", "scale", "quat"}
    np.testing.assert_allclose(g["xyz"], splat["xyz"], atol=1e-6)
    np.testing.assert_allclose(g["rgb01"], splat["rgb01"], atol=1e-5)
    np.testing.assert_allclose(g["opacity"], splat["opacity"], atol=1e-5)
    np.testing.assert_allclose(g["scale"], splat["scale"], rtol=1e-5)
    np.testing.assert_allclose(g["quat"], splat["quat"], atol=1e-6)
    assert lift.mesh_glb == b"glTF-fake"
    assert lift.meta["seconds"] == 31.2 and lift.meta["n_gaussians"] == 40


def test_lift_object_non_200_raises_meta_error(env):
    def handler(request):
        return httpx.Response(503, text="cold start exploded")

    with pytest.raises(sam3d.Sam3DError) as ei:
        sam3d.lift_object(b"img", b"mask", transport=httpx.MockTransport(handler))
    assert ei.value.status == 503 and "cold start" in str(ei.value)
    assert isinstance(ei.value, MetaError)


def test_lift_object_error_field_and_bad_json_raise(env):
    def err(request):
        return httpx.Response(200, json={"error": "the mask is empty"})

    with pytest.raises(sam3d.Sam3DError, match="mask is empty"):
        sam3d.lift_object(b"i", b"m", transport=httpx.MockTransport(err))

    def junk(request):
        return httpx.Response(200, text="<html>not json</html>")

    with pytest.raises(sam3d.Sam3DError, match="not JSON"):
        sam3d.lift_object(b"i", b"m", transport=httpx.MockTransport(junk))


def test_lift_object_without_token_raises(monkeypatch):
    monkeypatch.setenv("LOCAISH_SAM3D_URL", "https://x/{endpoint}")
    monkeypatch.delenv("LOCAISH_SAM3D_TOKEN", raising=False)
    with pytest.raises(sam3d.Sam3DError, match="TOKEN"):
        sam3d.lift_object(b"i", b"m", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))


def test_lift_object_follows_modal_303_to_result_url(env, splat):
    """Modal cuts a web request at 150 s and answers 303 to a result URL
    that blocks until the job is done; the client must follow it."""
    ply = _ply_bytes(**splat)
    calls = []

    def handler(request):
        calls.append((request.method, str(request.url)))
        if request.method == "POST":
            return httpx.Response(303, headers={"location": "https://ws--locaish-sam3d-object.modal.run/?__modal_function_id=abc"})
        return httpx.Response(200, json={"gaussians_ply_b64": base64.b64encode(ply).decode(), "mesh_glb_b64": None, "meta": {}})

    lift = sam3d.lift_object(b"i", b"m", transport=httpx.MockTransport(handler))
    assert [m for m, _ in calls] == ["POST", "GET"]
    assert lift.gaussians is not None and lift.mesh_glb is None


def test_gaussian_cov_is_R_S_St_Rt():
    q = _unit_quat([0, 0, 1], 90.0)
    s = np.array([[1.0, 2.0, 3.0]])
    cov = sam3d.gaussian_cov(s, q[None])
    R = _rot(q)
    expected = R @ np.diag(s[0] ** 2) @ R.T
    np.testing.assert_allclose(cov[0], expected, atol=1e-12)
    # A 90 degree yaw swaps the x and y variances.
    np.testing.assert_allclose(np.diag(cov[0]), [4.0, 1.0, 9.0], atol=1e-12)
    # Unnormalised quaternions are normalised first.
    np.testing.assert_allclose(sam3d.gaussian_cov(s, 3.0 * q[None]), cov, atol=1e-12)


def test_to_field_npz_round_trip_and_transform(tmp_path, splat):
    lift = sam3d.ObjectLift(gaussians=splat, mesh_glb=None, meta={"seconds": 1})
    p = sam3d.to_field_npz(lift, tmp_path / "sofa.npz")
    d = np.load(p)
    assert set(d.files) >= {"xyz", "rgb", "opacity", "cov"}
    n = len(splat["xyz"])
    assert d["xyz"].shape == (n, 3) and d["rgb"].shape == (n, 3) and d["opacity"].shape == (n,)
    assert d["cov"].shape == (n, 3, 3) and d["cov"].dtype == np.float32
    assert d["rgb"].min() >= 0 and d["rgb"].max() <= 1
    np.testing.assert_allclose(d["xyz"], splat["xyz"])
    expected = sam3d.gaussian_cov(splat["scale"], splat["quat"])
    np.testing.assert_allclose(d["cov"], expected, rtol=1e-5, atol=1e-7)
    assert json.loads(str(d["meta"])) == {"seconds": 1}

    # Pre-transformed: uniform scale 2, yaw 90, translation.
    M = np.eye(4)
    M[:3, :3] = 2.0 * _rot(_unit_quat([0, 0, 1], 90.0))
    M[:3, 3] = [1.0, -2.0, 0.5]
    p2 = sam3d.to_field_npz(lift, tmp_path / "sofa_t.npz", transform=M)
    d2 = np.load(p2)
    A = M[:3, :3]
    np.testing.assert_allclose(d2["xyz"], splat["xyz"] @ A.T + M[:3, 3], atol=1e-9)
    np.testing.assert_allclose(d2["cov"], A[None] @ expected @ A.T[None], rtol=1e-4, atol=1e-6)

    with pytest.raises(sam3d.Sam3DError):
        sam3d.to_field_npz(sam3d.ObjectLift(None, None, {}), tmp_path / "none.npz")
    with pytest.raises(ValueError):
        sam3d.to_field_npz(lift, tmp_path / "bad.npz", transform=np.eye(3))


def test_fit_to_box_unit_cube_into_measured_box():
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    g = {"xyz": corners, "opacity": np.ones(8)}
    size, centre, yaw = (2.0, 1.0, 0.8), (3.0, 4.0, 1.0), 90.0
    M = sam3d.fit_to_box(g, size, centre, yaw)
    assert M.shape == (4, 4) and np.allclose(M[3], [0, 0, 0, 1])
    out = corners @ M[:3, :3].T + M[:3, 3]
    lo, hi = out.min(axis=0), out.max(axis=0)
    # Uniform scale = min(2/1, 1/1, 0.8/1) = 0.8: nothing exceeds the box.
    np.testing.assert_allclose(hi - lo, [0.8, 0.8, 0.8], atol=1e-12)
    assert np.all(hi - lo <= np.asarray(size) + 1e-12)
    # Centred in x and y, standing on the bottom of the box.
    np.testing.assert_allclose((lo + hi)[:2] / 2, centre[:2], atol=1e-12)
    np.testing.assert_allclose(lo[2], centre[2] - size[2] / 2, atol=1e-12)
    # The linear part is 0.8 * Rz(90): the local +x edge now runs along +y.
    np.testing.assert_allclose(M[:3, :3], 0.8 * _rot(_unit_quat([0, 0, 1], 90.0)), atol=1e-12)
    np.testing.assert_allclose(M[:3, :3] @ [1, 0, 0], [0, 0.8, 0], atol=1e-12)


def test_fit_to_box_ignores_faint_stragglers_and_rejects_bad_size():
    rng = np.random.default_rng(1)
    solid = rng.uniform(-0.5, 0.5, size=(64, 3))
    far = np.array([[10.0, 10.0, 10.0]])
    g = {"xyz": np.vstack([solid, far]), "opacity": np.r_[np.ones(64), 0.05]}
    M = sam3d.fit_to_box(g, (1.0, 1.0, 1.0), (0, 0, 0.5), 0.0)
    s = M[0, 0]
    assert 0.9 < s < 1.2  # fitted to the solid metre-ish cloud, not the 20-unit outlier
    with pytest.raises(ValueError):
        sam3d.fit_to_box(g, (0.0, 1.0, 1.0), (0, 0, 0), 0.0)


# ---------------------------------------------------------------------------
# body
# ---------------------------------------------------------------------------


def _body_payload(verts, faces, kp=None, height=None):
    return {
        "vertices": verts.tolist(), "faces": faces.tolist(),
        "keypoints_3d": None if kp is None else kp.tolist(),
        "height_units": float(verts[:, 2].max() - verts[:, 2].min()) if height is None else height,
        "meta": {"n_people": 1, "orientation": {"method": "keypoints"}},
    }


def _standing_body(n=200, seed=0):
    """A tall (z), shoulder-wide (x), thin (y) blob with its feet at z=0."""
    rng = np.random.default_rng(seed)
    v = rng.uniform([-0.25, -0.12, 0.0], [0.25, 0.12, 1.75], size=(n, 3))
    v[0] = [0, 0, 0.0]
    v[1] = [0, 0, 1.75]
    faces = np.stack([np.arange(n - 2), np.arange(1, n - 1), np.arange(2, n)], axis=1)
    return v, faces


def test_lift_body_parses_and_to_body_npz_writes_contract(env, tmp_path):
    v, f = _standing_body()
    kp = np.zeros((70, 3))
    kp[13] = [-0.1, 0, 0.05]
    kp[14] = [0.1, 0, 0.05]
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_body_payload(v, f, kp))

    lift = sam3d.lift_body(b"headshot", transport=httpx.MockTransport(handler))
    assert seen["url"] == "https://ws--locaish-sam3d-body.modal.run" and seen["auth"] == "Bearer s3cret"
    assert base64.b64decode(seen["body"]["image_png_b64"]) == b"headshot"
    assert lift.vertices.shape == v.shape and lift.faces.shape == f.shape and lift.faces.dtype == np.int64
    assert lift.keypoints.shape == (70, 3)
    assert abs(lift.height_units - 1.75) < 1e-9
    assert lift.meta["n_people"] == 1

    p = sam3d.to_body_npz(lift, tmp_path / "actor.npz")
    d = np.load(p)
    assert {"vertices", "faces", "height_m"} <= set(d.files)
    assert d["vertices"].shape == v.shape and d["vertices"].dtype == np.float64
    assert d["faces"].dtype.kind == "i" and d["faces"].shape == f.shape
    assert abs(float(d["height_m"]) - 1.75) < 1e-9
    assert abs(d["vertices"][:, 2].min()) < 1e-12                      # feet at z=0
    ext = d["vertices"].max(axis=0) - d["vertices"].min(axis=0)
    assert ext[2] > ext[0] > ext[1]                                    # tall, then wide, then deep
    assert d["keypoints"].shape == (70, 3)
    meta = json.loads(str(d["meta"]))
    assert meta["server_height_units"] == pytest.approx(1.75)
    assert "client_orientation" in meta and "rotated_up_axis_from" not in meta["client_orientation"]


def test_upright_body_stands_up_a_lying_or_sideways_figure():
    v, _ = _standing_body()
    # Lying along x: swap z and x.
    lying = v[:, [2, 1, 0]]
    up, _, R, notes = sam3d.upright_body(lying)
    ext = up.max(axis=0) - up.min(axis=0)
    assert np.argmax(ext) == 2 and notes["rotated_up_axis_from"] == "x"
    assert abs(up[:, 2].min()) < 1e-12
    assert np.isclose(np.linalg.det(R), 1.0)
    # Turned sideways: swap x and y so the figure is deeper than wide.
    side = v[:, [1, 0, 2]]
    up2, _, R2, notes2 = sam3d.upright_body(side)
    ext2 = up2.max(axis=0) - up2.min(axis=0)
    assert notes2.get("yawed_90") is True and ext2[0] > ext2[1]
    assert np.isclose(np.linalg.det(R2), 1.0)


def test_lift_body_rejects_malformed_meshes(env):
    v, f = _standing_body()

    def no_faces(request):
        return httpx.Response(200, json={"vertices": v.tolist(), "faces": [], "height_units": 1.7})

    with pytest.raises(sam3d.Sam3DError, match="not a mesh"):
        sam3d.lift_body(b"x", transport=httpx.MockTransport(no_faces))

    def bad_index(request):
        return httpx.Response(200, json={"vertices": v.tolist(), "faces": [[0, 1, 10_000]], "height_units": 1.7})

    with pytest.raises(sam3d.Sam3DError, match="out of range"):
        sam3d.lift_body(b"x", transport=httpx.MockTransport(bad_index))

    def not_found(request):
        return httpx.Response(401, json={"detail": "bad or missing bearer token"})

    with pytest.raises(sam3d.Sam3DError) as ei:
        sam3d.lift_body(b"x", transport=httpx.MockTransport(not_found))
    assert ei.value.status == 401


# ---------------------------------------------------------------------------
# the Modal app's plain helpers
# ---------------------------------------------------------------------------


def test_modal_app_imports_without_modal():
    """`import locaish.meta.modal_app` must work on a machine without modal
    (this one), exposing a clear message instead of raising."""
    had = sys.modules.get("modal")
    sys.modules["modal"] = None  # makes `import modal` raise ImportError
    try:
        import locaish.meta.modal_app as m
        m = importlib.reload(m)
        assert m.modal is None and m.available() is False
        assert "modal" in m.MODAL_IMPORT_ERROR and "modal deploy" in m.MODAL_IMPORT_ERROR
        assert not hasattr(m, "app")
        assert len(m.OBJECTS_COMMIT) == 40 and len(m.BODY_COMMIT) == 40
    finally:
        if had is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = had
        importlib.reload(sys.modules["locaish.meta.modal_app"])


def test_check_bearer():
    from locaish.meta.modal_app import check_bearer

    assert check_bearer("Bearer abc", "abc") is True
    assert check_bearer("bearer abc", "abc") is True
    assert check_bearer("Bearer abd", "abc") is False
    assert check_bearer("Basic abc", "abc") is False
    assert check_bearer(None, "abc") is False
    assert check_bearer("Bearer abc", None) is False     # no secret deployed: closed, not open
    assert check_bearer("Bearer abc", "") is False
    assert check_bearer("Bearer", "abc") is False


def test_decode_image_and_mask_b64():
    from locaish.meta.modal_app import decode_image_b64, decode_mask_b64

    rgb = np.zeros((10, 12, 3), np.uint8)
    rgb[..., 0] = 255
    im = decode_image_b64(base64.b64encode(_png(rgb)).decode())
    assert im.shape == (10, 12, 3) and im.dtype == np.uint8 and im[0, 0].tolist() == [255, 0, 0]
    # An RGBA mask: the alpha channel is the mask, and a half-size mask is resized.
    rgba = np.zeros((5, 6, 4), np.uint8)
    rgba[:, 3:, 3] = 255
    m = decode_mask_b64(base64.b64encode(_png(rgba)).decode(), (10, 12))
    assert m.shape == (10, 12) and m.dtype == bool
    assert m[:, :6].sum() == 0 and m[:, 6:].all()


def _mhr_keypoints_camera_frame(R=None):
    """70 MHR keypoints of a person standing upright and facing the camera,
    in the OpenCV camera frame (+Y down, +Z forward). Their left is at +X."""
    kp = np.zeros((70, 3))
    # Head and feet groups have zero mean depth so "up" is exactly -Y and
    # the rotation can be compared with the fixed camera-frame mapping.
    kp[0] = [0.0, -0.80, 0.0]           # nose
    kp[1], kp[2] = [0.03, -0.83, -0.02], [-0.03, -0.83, -0.02]
    kp[3], kp[4] = [0.07, -0.82, 0.02], [-0.07, -0.82, 0.02]
    kp[5], kp[6] = [0.20, -0.55, 0.0], [-0.20, -0.55, 0.0]      # shoulders
    kp[9], kp[10] = [0.10, 0.0, 0.0], [-0.10, 0.0, 0.0]         # hips
    kp[13], kp[14] = [0.10, 0.85, 0.0], [-0.10, 0.85, 0.0]      # ankles
    kp[15], kp[16] = [0.11, 0.90, -0.10], [0.14, 0.90, -0.08]   # left toes (in front: -Z)
    kp[17] = [0.10, 0.90, 0.18]                                 # left heel (behind: +Z)
    kp[18], kp[19] = [-0.11, 0.90, -0.10], [-0.14, 0.90, -0.08]
    kp[20] = [-0.10, 0.90, 0.18]
    if R is not None:
        kp = kp @ R.T
    return kp


def test_upright_from_keypoints_matches_camera_fallback_for_a_facing_person():
    from locaish.meta.modal_app import CAMERA_TO_UPRIGHT_FALLBACK, upright_from_keypoints

    kp = _mhr_keypoints_camera_frame()
    verts = np.vstack([kp, [[0, -0.9, 0.0], [0, 0.9, 0.0], [0.3, 0.0, -0.15]]])   # head top, sole, a hand in front
    v, k, R, notes = upright_from_keypoints(verts, kp)
    assert notes["method"] == "keypoints" and notes["head_above_feet"] is True
    assert np.isclose(np.linalg.det(R), 1.0)
    np.testing.assert_allclose(R, CAMERA_TO_UPRIGHT_FALLBACK, atol=1e-9)
    assert abs(v[:, 2].min()) < 1e-12                                      # feet at z=0
    assert k[0, 2] > k[13, 2]                                              # nose above ankle
    assert v[-1, 1] > 0.1                                                  # the hand in front points +Y: facing +Y
    assert k[9, 0] < 0 < k[10, 0]                                          # left hip at -X, right hip at +X
    np.testing.assert_allclose(k[[13, 14]].mean(axis=0)[:2], [0, 0], atol=1e-12)   # standing on the mark
    assert notes["height_units"] == pytest.approx(1.8)


def test_upright_from_keypoints_stands_up_a_rolled_camera_and_falls_back_without_joints():
    from locaish.meta.modal_app import CAMERA_TO_UPRIGHT_FALLBACK, upright_from_keypoints

    # The same person seen by a camera rolled 90 degrees about its optical axis.
    c, s = 0.0, 1.0
    roll = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    kp = _mhr_keypoints_camera_frame(roll)
    verts = kp.copy()
    v, k, R, notes = upright_from_keypoints(verts, kp)
    assert notes["method"] == "keypoints"
    assert k[list(range(0, 5))][:, 2].mean() > k[list(range(13, 21))][:, 2].mean()
    ext = v.max(axis=0) - v.min(axis=0)
    assert np.argmax(ext) == 2                                             # tall along Z again
    # The rotation undoes the roll: R_rolled == R_upright @ roll^T.
    np.testing.assert_allclose(R @ roll, CAMERA_TO_UPRIGHT_FALLBACK, atol=1e-9)

    # No keypoints: the fixed camera-frame mapping, centred on the bounding box.
    v2, k2, R2, notes2 = upright_from_keypoints(verts, None)
    assert notes2["method"] == "fallback_camera_frame" and k2 is None
    np.testing.assert_allclose(R2, CAMERA_TO_UPRIGHT_FALLBACK)
    assert abs(v2[:, 2].min()) < 1e-12
    # Too few keypoints for the joint indices also falls back.
    _, _, R3, notes3 = upright_from_keypoints(verts, kp[:10])
    assert notes3["method"] == "fallback_camera_frame"
