"""Turn a `Twin` into one self-contained HTML page you can open from disk.

The viewer is the part of Phase 1 that has to convince a human. A QA report can
assert that a room is 1:1; only a labelled metre grid under the geometry and a
measure tool that hits the actual surface can make someone believe it. So this
module is written as an instrument rather than a renderer: everything the page
draws is a number that came out of the pipeline, and nothing is invented in the
browser except the grid, which is by definition exact.

Two constraints shape every decision here.

The page must open from `file://` with no network, which rules out three.js and
every CDN, so `template.html` carries hand-written WebGL2 and this module only
injects data into it. And a real sweep is 1-20M points, which rules out JSON
arrays of numbers -- a million floats spelled out in decimal is roughly 20 MB of
text that the browser then has to parse one token at a time. Geometry therefore
travels as base64 of the raw little-endian buffer and is turned back into a
typed array with one `atob` and one `new Float32Array(bytes.buffer)`.

Positions are float32 relative to `payload["origin"]`, the twin's XY centre with
Z left alone. Centring in XY is what keeps float32 honest for a twin whose
origin was never canonicalised and sits 30 km away (the `tilted` fixture
translates by (31.2, -18.7, 4.4), and a georeferenced export can be far worse);
leaving Z alone is what keeps the floor at exactly z = 0, which is where the
grid is drawn and where every sill height is measured from. float32 at room
scale resolves about 0.5 um, three orders of magnitude below the noise floor of
any consumer scanner, so nothing is lost.
"""

from __future__ import annotations

import base64
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..geom import shell as shellmod
from ..types import Mesh, Opening, Plane, Twin, chunked

PAYLOAD_VERSION = 1

_TEMPLATE_PATH = Path(__file__).with_name("template.html")
_TITLE_TOKEN = "__LOCAISH_TITLE__"
_PAYLOAD_TOKEN = "__LOCAISH_PAYLOAD__"

# Okabe-Ito, the standard eight-colour set that survives deuteranopia,
# protanopia and tritanopia. The plane kinds have to be told apart by hue alone
# because they overlap in space, so a palette that collapses for one in twelve
# men is not an option.
PLANE_TINTS: dict[str, str] = {
    "floor": "#e69f00",
    "wall": "#009e73",
    "ceiling": "#cc79a7",
    "other": "#f0e442",
}

# How many points to *query* for the spacing estimate. The tree is always built
# over the full drawn cloud (see `_point_spacing`); the median converges long
# before twenty thousand queries, so asking more is wasted time.
_SPACING_SAMPLE = 20_000
_DECIMATION_SEED = 20240816


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------


def _b64(array: np.ndarray, dtype: str) -> str:
    """Base64 of the raw buffer in an explicitly little-endian dtype.

    The dtype string carries the byte order (`<f4`, not `f4`) because the
    browser reinterprets these bytes with a typed array, and typed arrays are
    always host order. Every platform that runs a browser is little-endian, but
    a big-endian numpy build writing `>f4` here would produce a page that is
    silently, spectacularly wrong rather than one that fails, so we pin it.
    """
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def _jsonable(value: Any) -> Any:
    """Recursively coerce numpy scalars and arrays into plain JSON types.

    Non-finite floats become null rather than raising: a NaN in `QAReport.metrics`
    is a defect upstream, but it must not take the viewer down with it -- the
    whole point of the panel is to show the operator what the pipeline thought,
    including when the pipeline thought something broken.
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _embed_json(obj: Any) -> str:
    """JSON escaped so it cannot terminate the script element that contains it.

    A twin name of `</script>` would otherwise end the program and dump the rest
    of the payload into the document as text. Escaping the three characters that
    can start an HTML token, plus the two line separators that are newlines to a
    JS parser but not to JSON, makes the literal inert in any HTML context.
    """
    text = json.dumps(obj, separators=(",", ":"), allow_nan=False)
    for raw, escaped in (
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("&", "\\u0026"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        text = text.replace(raw, escaped)
    return text


# ---------------------------------------------------------------------------
# non-finite geometry
#
# A depth frame with an invalid pixel, a reconstruction that divided by a zero
# area, a PLY written by a tool that spells "unknown" as NaN: every one of them
# reaches the viewer as a coordinate that is not a number. `_jsonable` already
# argues that a NaN in the QA metrics must not take the page down with it, and
# a NaN in a *coordinate* deserves the same treatment for a stronger reason --
# it is the last stage of the pipeline, so crashing here throws away a twin
# that is otherwise entirely usable. The rule is therefore: drop the bad rows,
# count them, and put the count where the operator will see it, because a
# silently shortened cloud would be its own kind of lie.
# ---------------------------------------------------------------------------


def _finite_rows(xyz: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Mask of rows whose three coordinates are all finite, and how many are not.

    Returns `None` for the mask when every row is finite, which lets the caller
    skip the fancy-index copy in the overwhelmingly common case -- on a 20M
    point sweep that copy is half a gigabyte of pointless traffic. The scan is
    chunked for the same reason: `np.isfinite` on the whole (N, 3) array
    allocates an (N, 3) bool temporary, and we only ever need one column of it.
    """
    n = len(xyz)
    if n == 0:
        return None, 0
    mask = np.empty(n, dtype=bool)
    for sl in chunked(n, 1_000_000):
        mask[sl] = np.isfinite(xyz[sl]).all(axis=1)
    bad = int(n - int(mask.sum()))
    return (None, 0) if bad == 0 else (mask, bad)


def _finite_mesh(mesh: Mesh | None) -> tuple[Mesh | None, int]:
    """The mesh with every face touching a non-finite vertex removed.

    Faces rather than vertices are the unit of removal because an index buffer
    that survives its vertex is a hole in the page's memory rather than a hole
    in the room. The kept vertices are compacted so the buffers we ship carry
    no NaN at all: a single one would poison the vertex normals through the
    area weighting and stripe the whole mesh black.
    """
    if mesh is None or len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        return mesh, 0
    good_vertex = np.isfinite(mesh.vertices).all(axis=1)
    if bool(good_vertex.all()):
        return mesh, 0
    keep_face = good_vertex[mesh.faces].all(axis=1)
    dropped = int(len(mesh.faces) - int(keep_face.sum()))
    faces = mesh.faces[keep_face]
    if len(faces) == 0:
        return None, dropped
    used = np.unique(faces)
    remap = np.zeros(len(mesh.vertices), dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return (
        Mesh(
            vertices=mesh.vertices[used],
            faces=remap[faces],
            vertex_colors=None if mesh.vertex_colors is None else mesh.vertex_colors[used],
            uv=None if mesh.uv is None else mesh.uv[used],
            texture=mesh.texture,
            texture_format=mesh.texture_format,
        ),
        dropped,
    )


# ---------------------------------------------------------------------------
# point cloud
# ---------------------------------------------------------------------------


def _decimation_index(count: int, budget: int) -> np.ndarray | None:
    """Indices of a jittered-stride subsample, or None when everything fits.

    The obvious `rng.choice(n, k, replace=False)` allocates and shuffles an
    n-element permutation, which for a 20M-point sweep is 160 MB of scratch to
    pick 900k survivors. A stride with a seeded jitter inside each stride window
    is O(k) in both time and memory, uniform over the cloud, and -- unlike a
    plain stride -- immune to aliasing against clouds whose point order is
    periodic, which is exactly what a raster-scanned depth sensor produces.
    """
    if budget <= 0 or count <= budget:
        return None
    rng = np.random.default_rng(_DECIMATION_SEED)
    step = count / float(budget)
    base = np.arange(budget, dtype=np.float64) * step
    idx = (base + rng.random(budget) * step).astype(np.int64)
    return np.clip(idx, 0, count - 1)


def _point_spacing(xyz: np.ndarray) -> float:
    """Median nearest-neighbour distance, in metres, of the points as drawn.

    This is what sets the on-screen point size: a splat about one spacing wide
    closes the gaps between neighbours and the cloud reads as a surface, while
    anything much smaller reads as confetti.

    The subtlety is that the tree must be built over *every* drawn point while
    only a sample is queried against it. Thinning the tree as well as the query
    inflates the answer by roughly the square root of the thinning factor --
    points on a scan lie on 2D surfaces, so dropping 44 in 45 pushes neighbours
    about seven times further apart -- and the page would then draw splats seven
    times too fat. Building the whole tree is bounded work regardless of the
    source cloud, because this runs after decimation and the drawn cloud never
    exceeds `max_points`.
    """
    if len(xyz) < 2:
        return 0.02
    from scipy.spatial import cKDTree

    full = np.asarray(xyz, dtype=np.float64)
    # cKDTree rejects a non-finite coordinate with a ValueError. The caller has
    # already dropped the twin's own non-finite points, so what is defended
    # against here is the cast to float32 above: a coordinate of 1e39 is finite
    # in the twin and infinite in the buffer we ship. Two lines to keep the
    # last stage of the pipeline alive is a trade worth making twice.
    keep = np.isfinite(full).all(axis=1)
    if not bool(keep.all()):
        full = full[keep]
        if len(full) < 2:
            return 0.02
    tree = cKDTree(full)
    stride = max(1, len(full) // _SPACING_SAMPLE)
    dist, _ = tree.query(full[::stride], k=2, workers=-1)
    near = dist[:, 1]
    near = near[np.isfinite(near) & (near > 0)]
    if not len(near):
        return 0.02
    return float(np.clip(np.median(near), 0.001, 0.5))


# ---------------------------------------------------------------------------
# overlay geometry
# ---------------------------------------------------------------------------


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic orthonormal (u, v) spanning the plane.

    Deterministic matters because the polygon we emit has to be identical for
    identical input; the seed-everything rule applies to basis choices too. We
    prefer u horizontal so a wall's u runs along the wall and v runs up, which
    is the frame a human reads dimensions in.
    """
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n = n / max(np.linalg.norm(n), 1e-12)
    u = np.cross(np.array([0.0, 0.0, 1.0]), n)
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(np.array([0.0, 1.0, 0.0]), n)
    u = u / max(np.linalg.norm(u), 1e-12)
    v = np.cross(n, u)
    v = v / max(np.linalg.norm(v), 1e-12)
    return u, v


def _clip_to_box(poly: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of a convex polygon against an axis-aligned box.

    A `Plane` is infinite, and drawing an infinite plane as a fixed-size quad
    either buries the room or floats in space next to it. `Plane.extent_2d` would
    bound it, but the contract in types.py does not say which 2D basis those
    numbers are in, so consuming them as a rectangle would risk drawing a
    confidently wrong quad. Clipping against the twin's own bounding box is a
    claim we can always defend: this is where the plane passes through the room.
    """
    for axis in range(3):
        for sign, bound in ((1.0, lo[axis]), (-1.0, -hi[axis])):
            if len(poly) < 3:
                return np.zeros((0, 3))
            d = sign * poly[:, axis] - bound
            nxt = np.roll(poly, -1, axis=0)
            d_next = np.roll(d, -1)
            out: list[np.ndarray] = []
            for i in range(len(poly)):
                if d[i] >= 0.0:
                    out.append(poly[i])
                if (d[i] >= 0.0) != (d_next[i] >= 0.0):
                    denom = d[i] - d_next[i]
                    if abs(denom) > 1e-12:
                        t = d[i] / denom
                        out.append(poly[i] + t * (nxt[i] - poly[i]))
            poly = np.asarray(out, dtype=np.float64).reshape(-1, 3)
    return poly


def _plane_polygon(plane: Plane, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """The plane's footprint inside the twin bounding box, as a convex polygon."""
    span = float(np.linalg.norm(hi - lo))
    if span <= 0:
        return np.zeros((0, 3))
    centre = (lo + hi) / 2.0
    on_plane = centre - (float(np.dot(plane.normal, centre)) - plane.offset) * plane.normal
    u, v = _plane_basis(plane.normal)
    r = span
    quad = np.stack(
        [
            on_plane - r * u - r * v,
            on_plane + r * u - r * v,
            on_plane + r * u + r * v,
            on_plane - r * u + r * v,
        ]
    )
    return _clip_to_box(quad, lo, hi)


def _opening_corners(opening: Opening) -> np.ndarray:
    """The four corners of the opening rectangle, counter-clockwise from below.

    An `Opening` is stored as a centre plus a width, a height and the wall
    normal, which is three quarters of a frame. We complete it by insisting the
    in-wall horizontal axis is perpendicular to world up, so `height` is drawn
    vertically and the reveal reads as a window rather than a lozenge. A wall
    normal that is itself vertical (a skylight in a `Structure` that called it a
    wall) falls back to a Y-derived axis instead of dividing by zero.
    """
    n = opening.normal
    if np.linalg.norm(n) < 1e-9:
        n = np.array([0.0, 1.0, 0.0])
    n = n / np.linalg.norm(n)
    u = np.cross(np.array([0.0, 0.0, 1.0]), n)
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(np.array([0.0, 1.0, 0.0]), n)
    u = u / max(np.linalg.norm(u), 1e-12)
    v = np.cross(n, u)
    v = v / max(np.linalg.norm(v), 1e-12)
    hw = float(opening.width) / 2.0
    hh = float(opening.height) / 2.0
    c = opening.center
    return np.stack([c - hw * u - hh * v, c + hw * u - hh * v, c + hw * u + hh * v, c - hw * u + hh * v])


# ---------------------------------------------------------------------------
# mesh
# ---------------------------------------------------------------------------


def _vertex_normals(mesh: Mesh) -> np.ndarray:
    """Area-weighted vertex normals, computed here so the page never has to.

    `np.bincount` rather than `np.add.at`: the latter is a scattered
    read-modify-write that runs at Python-loop speed on a million faces, and a
    mesh out of a reconstruction routine is routinely that big.
    """
    verts = mesh.vertices
    faces = mesh.faces
    if len(faces) == 0 or len(verts) == 0:
        return np.zeros((len(verts), 3), dtype=np.float64)
    weighted = mesh.face_normals * mesh.face_areas[:, None]
    flat = faces.ravel()
    acc = np.empty((len(verts), 3), dtype=np.float64)
    for c in range(3):
        acc[:, c] = np.bincount(flat, weights=np.repeat(weighted[:, c], 3), minlength=len(verts))
    length = np.linalg.norm(acc, axis=1, keepdims=True)
    return acc / np.where(length < 1e-12, 1.0, length)


def _mesh_payload(mesh: Mesh, origin: np.ndarray) -> dict[str, Any] | None:
    """Interleave-free mesh buffers, or None if there is nothing to draw."""
    if mesh is None or len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        return None
    verts = (mesh.vertices - origin).astype(np.float32)
    payload: dict[str, Any] = {
        "vertex_count": int(len(verts)),
        "index_count": int(mesh.faces.size),
        "face_count": int(len(mesh.faces)),
        "area_m2": float(mesh.area),
        "xyz": _b64(verts, "<f4"),
        "normal": _b64(_vertex_normals(mesh), "<f4"),
        "index": _b64(mesh.faces.astype(np.uint32), "<u4"),
        "rgb": None,
        "uv": None,
        "texture": None,
    }
    if mesh.vertex_colors is not None and len(mesh.vertex_colors) == len(verts):
        payload["rgb"] = _b64(mesh.vertex_colors, "u1")
    if mesh.uv is not None and len(mesh.uv) == len(verts) and mesh.texture:
        fmt = (mesh.texture_format or "png").lower()
        mime = "image/jpeg" if fmt in {"jpg", "jpeg"} else f"image/{fmt}"
        payload["uv"] = _b64(mesh.uv, "<f4")
        payload["texture"] = f"data:{mime};base64," + base64.b64encode(mesh.texture).decode("ascii")
    return payload


def _in_room_mask(xyz: np.ndarray, s) -> np.ndarray | None:
    """Which points stand inside the solved room volume, or None if unknowable.

    Even-odd polygon test against the footprint plus a z gate at the drawn
    ceiling. Used only to *order* the drawn cloud so the solid view can stop
    at the room's contents: the fringe of returns behind walls and through
    doorways is real evidence and stays in the evidence views, but drawn
    inside a photographed room it reads as noise stuck to the outside of the
    walls. A small margin keeps the returns sitting ON the walls.
    """
    if s.footprint is None or len(s.footprint) < 3 or s.footprint_source == "raster-sparse":
        return None
    poly = np.asarray(s.footprint, dtype=np.float64)
    x, y = xyz[:, 0], xyz[:, 1]
    inside = np.zeros(len(xyz), dtype=bool)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i - 1]
        x1, y1 = poly[i]
        dy = y0 - y1
        if abs(dy) < 1e-12:
            continue
        cross = ((y1 > y) != (y0 > y)) & (x < (x0 - x1) * (y - y1) / dy + x1)
        inside ^= cross
    # points on the walls themselves sit within noise of the boundary; keep an
    # apron of a few centimetres -- the fit tolerance plus range noise -- so
    # the wall returns survive the cut without dragging in the smear that
    # sits behind the wall
    margin = 0.05
    for i in range(n):
        a = poly[i - 1]
        b = poly[i]
        seg = b - a
        length = float(np.hypot(*seg))
        if length < 1e-9:
            continue
        d = seg / length
        rel_x = x - a[0]
        rel_y = y - a[1]
        along = rel_x * d[0] + rel_y * d[1]
        perp = np.abs(rel_x * -d[1] + rel_y * d[0])
        inside |= (along >= -margin) & (along <= length + margin) & (perp <= margin)
    cap = s.drawable_ceiling_z
    top = (cap if cap is not None else s.floor_z + 2.40) + margin
    z = xyz[:, 2]
    inside &= (z >= float(s.floor_z) - margin) & (z <= top)
    return inside


def _shell_atlas_uv(built, twin: Twin) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Per-vertex atlas texcoords for the shell, from the baked layout.

    The baker laid panels out by the same part keys and metre-uv convention
    the shell carries, so this is bookkeeping and not geometry: look the
    vertex's panel up in the layout, map its (u, v) into the panel's atlas
    rect. Vertices of panels the bake declared untextured get weight 0 and
    keep the provenance tint -- the mask is how "the video never saw this
    wall" survives into the picture.

    The page uploads textures with UNPACK_FLIP_Y_WEBGL, so texcoord t counts
    from the *bottom* of the atlas image while the baker's rows count from
    the top; the `1 -` below is that flip and nothing else.
    """
    layout = twin.shell_texture_layout
    if (
        twin.shell_texture is None
        or not layout
        or built.uv is None
        or len(built.uv_part) != len(built.vertices)
    ):
        return None, None
    panels = layout.get("panels") or {}
    atlas_w = float(layout.get("atlas_w", 0))
    atlas_h = float(layout.get("atlas_h", 0))
    if atlas_w <= 0 or atlas_h <= 0 or not any(p.get("textured") for p in panels.values()):
        return None, None

    n = len(built.vertices)
    uv_out = np.zeros((n, 2), dtype=np.float32)
    mask = np.zeros(n, dtype=np.uint8)
    for j in range(n):
        entry = panels.get(built.uv_part[j])
        if not entry or not entry.get("textured"):
            continue
        x, y, w, h = entry["rect"]
        u0, v0, u1, v1 = entry["uv_bounds"]
        du = max(u1 - u0, 1e-9)
        dv = max(v1 - v0, 1e-9)
        u, v = float(built.uv[j, 0]), float(built.uv[j, 1])
        px = x + (u - u0) / du * w
        row = y + (v1 - v) / dv * h
        uv_out[j, 0] = px / atlas_w
        uv_out[j, 1] = 1.0 - row / atlas_h
        mask[j] = 255
    return uv_out, mask


def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a proper rotation matrix."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(R[i, i] - R[j, j] - R[k, k] + 1.0, 1e-12)) * 2.0
    q = np.empty(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a x b, (w,x,y,z); a is (4,), b is (N,4)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
    )


# Splat payload budget: enough gaussians to carry the whole room, few enough
# that the page stays loadable.
_SPLAT_MAX = 900_000
_SPLAT_OPACITY_MIN = 0.04
_SPLAT_S1_CAP_M = 0.35


def _splat_payload(twin: Twin, origin: np.ndarray) -> dict[str, Any] | None:
    """The trained gaussians, transformed to view frame, packed for the page.

    Rendering the gaussians themselves is what makes the room read as
    continuous surface: a sampled point cloud always has gaps at some zoom,
    an alpha-blended gaussian field does not. The transform chain is the
    twin's own: metric scale then the canonical (and wall-squared) rotation,
    the same numbers that placed every point in the twin.
    """
    path = (twin.provenance or {}).get("splat_ply")
    if not path or not Path(path).exists():
        return None
    steps = (twin.provenance or {}).get("steps", {})
    factor = ((steps.get("video") or {}).get("scale") or {}).get("factor")
    if not factor:
        return None
    try:
        from ..video.splat import read_gaussian_ply

        g = read_gaussian_ply(path)
    except Exception:
        return None
    M = np.asarray(twin.canonical_transform, dtype=np.float64)
    R, t = M[:3, :3], M[:3, 3]
    xyz = (g["xyz"] * float(factor)) @ R.T + t
    scale = g["scale"] * float(factor)

    s1 = scale.max(axis=1)
    lo = twin.points.xyz.min(axis=0) - 0.4
    hi = twin.points.xyz.max(axis=0) + 0.4
    keep = (
        (g["opacity"] >= _SPLAT_OPACITY_MIN)
        & (s1 <= _SPLAT_S1_CAP_M)
        & np.all((xyz >= lo) & (xyz <= hi), axis=1)
    )
    idx = np.flatnonzero(keep)
    if len(idx) < 1000:
        return None
    if len(idx) > _SPLAT_MAX:
        order = np.argsort(-g["opacity"][idx], kind="stable")
        idx = idx[order[:_SPLAT_MAX]]
    quat = _quat_mul(_quat_from_matrix(R), g["quat"][idx])
    rgba = np.empty((len(idx), 4), dtype=np.uint8)
    rgba[:, :3] = np.clip(g["rgb01"][idx] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgba[:, 3] = np.clip(g["opacity"][idx] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return {
        "count": int(len(idx)),
        "xyz": _b64((xyz[idx] - origin).astype(np.float32), "<f4"),
        "rgba": _b64(rgba, "u1"),
        "scale": _b64(scale[idx].astype(np.float32), "<f4"),
        "quat": _b64(quat.astype(np.float32), "<f4"),
    }


# ---------------------------------------------------------------------------
# the payload
# ---------------------------------------------------------------------------


def twin_to_payload(twin: Twin, *, max_points: int = 900_000) -> dict[str, Any]:
    """Everything the page consumes: metadata as JSON, geometry as base64.

    Exposed separately from `render_html` so a later FastAPI server can hand the
    same dict to a browser over the wire without ever writing a file, and so
    tests can assert on the numbers without parsing HTML. The dict is pure JSON
    types -- no numpy survives the boundary -- and is stable for a given twin.

    All coordinates in the returned buffers are metres in *view frame*, which is
    twin space minus `origin`. Convert a readout back to twin space by adding
    `origin`; the page does exactly that when it reports a picked point.

    Points and faces carrying a non-finite coordinate are dropped rather than
    propagated, counted in `points["dropped_nonfinite"]` and
    `mesh["dropped_faces_nonfinite"]`, and described in `warnings`, which the
    page shows at the top of its panel. Rendering what is intact and saying so
    beats refusing to render anything, which is what handing a NaN to the
    spacing estimate used to do.
    """
    # -- reject non-finite geometry before anything measures it ---------
    xyz = twin.points.xyz
    rgb = twin.points.rgb
    inferred = twin.points.inferred
    finite, dropped_points = _finite_rows(xyz)
    if finite is not None:
        xyz = xyz[finite]
        rgb = None if rgb is None else rgb[finite]
        inferred = None if inferred is None else inferred[finite]
    mesh, dropped_faces = _finite_mesh(twin.mesh)

    warnings: list[str] = []
    if dropped_points:
        warnings.append(
            f"{dropped_points:,} non-finite point{'' if dropped_points == 1 else 's'} "
            f"dropped, of {len(twin.points):,}"
        )
    if dropped_faces:
        warnings.append(
            f"{dropped_faces:,} mesh face{'' if dropped_faces == 1 else 's'} dropped "
            "for touching a non-finite vertex"
        )

    # Bounds come from what survived, not from `twin.bounds`: one NaN makes
    # that property non-finite in all three axes, and framing the camera on it
    # would blank a twin that is 99.999% intact.
    corners = []
    if len(xyz):
        corners.append(np.stack([xyz.min(axis=0), xyz.max(axis=0)]))
    if mesh is not None and len(mesh.vertices):
        corners.append(np.stack([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)]))
    empty = not corners
    if empty:
        lo = np.zeros(3)
        hi = np.zeros(3)
    else:
        stacked = np.stack(corners)
        lo = stacked[:, 0].min(axis=0)
        hi = stacked[:, 1].max(axis=0)
    origin = np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, 0.0])

    # -- points ---------------------------------------------------------
    index = _decimation_index(len(xyz), max_points)
    if index is not None:
        xyz = xyz[index]
        rgb = None if rgb is None else rgb[index]
        inferred = None if inferred is None else inferred[index]
    # In-room points first, so the solid view can draw a prefix of the same
    # buffer and show only the room's contents; the evidence views keep
    # drawing everything. A stable partition, not a filter: nothing is lost.
    solid_count = None
    mask = _in_room_mask(xyz, twin.structure)
    if mask is not None:
        order = np.argsort(~mask, kind="stable")
        xyz = xyz[order]
        rgb = None if rgb is None else rgb[order]
        inferred = None if inferred is None else inferred[order]
        solid_count = int(mask.sum())
    drawn = (xyz - origin).astype(np.float32)
    points = {
        "count": int(len(drawn)),
        "solid_count": solid_count,
        # A very dense cloud is meant to read as surface, not as confetti:
        # open it with fatter splats so the walls close up, exactly what the
        # size slider does by hand.
        "default_boost": 1.8 if len(drawn) > 4_000_000 else 1.0,
        "source_count": int(len(twin.points)),
        "dropped_nonfinite": dropped_points,
        "decimated": bool(index is not None),
        "spacing_m": _point_spacing(drawn),
        "xyz": _b64(drawn, "<f4"),
        "rgb": None if rgb is None else _b64(rgb, "u1"),
        # Per-point inferred weight, same contract as the mesh's `filled`:
        # a viewer that reads it can fade or toggle invented surface, and one
        # that does not still shows it desaturated because the colours already
        # are. None when the question was never asked.
        "inferred": None if inferred is None else _b64(
            np.clip(np.asarray(inferred, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8),
            "u1",
        ),
    }

    # -- structure ------------------------------------------------------
    s = twin.structure
    planes: list[dict[str, Any]] = []
    for plane in s.planes:
        poly = _plane_polygon(plane, lo, hi) if not empty else np.zeros((0, 3))
        planes.append(
            {
                "kind": plane.kind,
                "normal": plane.normal.tolist(),
                "offset": float(plane.offset),
                "area_m2": float(plane.area),
                "inlier_count": int(plane.inlier_count),
                "extent_2d": None if plane.extent_2d is None else [float(x) for x in plane.extent_2d],
                "polygon": (poly - origin).tolist() if len(poly) >= 3 else [],
                "tint": PLANE_TINTS.get(plane.kind, PLANE_TINTS["other"]),
            }
        )

    openings: list[dict[str, Any]] = []
    for op in s.openings:
        corners = _opening_corners(op)
        openings.append(
            {
                "kind": op.kind,
                "width_m": float(op.width),
                "height_m": float(op.height),
                "area_m2": float(op.area),
                "sill_height_m": float(op.sill_height),
                "confidence": float(op.confidence),
                "center": (op.center - origin).tolist(),
                "center_twin": op.center.tolist(),
                "normal": op.normal.tolist(),
                "corners": (corners - origin).tolist(),
            }
        )

    footprint = None
    if s.footprint is not None and len(s.footprint) >= 3:
        footprint = (s.footprint - origin[:2]).tolist()

    # -- the room as a surface ------------------------------------------
    #
    # The point cloud is the evidence; this is the reading of it. Sent as its
    # own buffer rather than folded into `mesh` because the two answer
    # different questions -- the mesh is the surface of everything that was
    # scanned, contents included, and this is only the architecture -- and
    # because the page has to be able to draw one without the other.
    shell_payload = None
    # A shell is only drawn from a footprint whose edges mean something: the
    # cell-derived outline, or a rastered one from a dense pose-less import.
    # "raster-sparse" is a video capture whose room solve declined -- extruding
    # that contour is how a twin ends up looking like a carved cylinder, so
    # the honest picture there is points plus capture bounds and no shell.
    drawable_footprint = (
        not empty
        and s.footprint is not None
        and len(s.footprint) >= 3
        and s.footprint_source != "raster-sparse"
    )
    if drawable_footprint:
        built = shellmod.shell_from_structure(s)
        if built is not None and len(built.faces):
            shell_mesh = Mesh(vertices=built.vertices, faces=built.faces)
            # Measured surface is drawn in the room's own warm grey; inferred
            # surface in a cold, desaturated blue. The colour is the label: a
            # viewer should be able to see, without reading a panel, which
            # walls are a measurement and which are the fitter's reading of
            # where the room must continue.
            tint = np.empty((len(built.vertices), 3), dtype=np.uint8)
            w = np.clip(built.inferred, 0.0, 1.0)[:, None]
            tint[:] = ((1.0 - w) * np.array([196, 192, 184]) + w * np.array([86, 104, 132])).astype(
                np.uint8
            )
            shell_payload = {
                "xyz": _b64((built.vertices - origin).astype(np.float32), "<f4"),
                "normal": _b64(_vertex_normals(shell_mesh), "<f4"),
                "index": _b64(built.faces.astype(np.uint32), "<u4"),
                "rgb": _b64(tint, "u1"),
                "inferred": _b64(
                    np.clip(built.inferred * 255.0, 0, 255).astype(np.uint8), "u1"
                ),
                "vertex_count": int(len(built.vertices)),
                "index_count": int(built.faces.size),
                "face_count": int(len(built.faces)),
                "area_m2": float(built.area),
                "measured_fraction": float(built.measured_fraction()),
                "ceiling_source": s.ceiling_source,
                # the height the shell was actually capped at, which is the
                # fitted ceiling when there is one and a drawing default when
                # there is not -- either way it is what the picture shows
                "ceiling_drawn_z": float(built.vertices[:, 2].max() - origin[2]),
                "ceiling_measured": s.ceiling_z is not None,
                "parts": built.parts,
                "uv": None,
                "texture": None,
                "textured": None,
            }
            uv_atlas, textured_mask = _shell_atlas_uv(built, twin)
            if uv_atlas is not None:
                fmt = (twin.shell_texture_format or "jpg").lower()
                mime = "image/jpeg" if fmt in {"jpg", "jpeg"} else f"image/{fmt}"
                shell_payload["uv"] = _b64(uv_atlas, "<f4")
                shell_payload["textured"] = _b64(textured_mask, "u1")
                shell_payload["texture"] = f"data:{mime};base64," + base64.b64encode(
                    twin.shell_texture
                ).decode("ascii")
                cov = [
                    p.get("coverage", 0.0)
                    for p in (twin.shell_texture_layout or {}).get("panels", {}).values()
                    if p.get("textured")
                ]
                shell_payload["textured_coverage"] = (
                    float(np.mean(cov)) if cov else 0.0
                )

    capture = None
    cb = twin.capture_bounds
    if cb is not None and len(cb.hull_xy) >= 3:
        capture = {
            "hull_xy": (cb.hull_xy - origin[:2]).tolist(),
            "z_range": [float(cb.z_range[0]), float(cb.z_range[1])],
            "area_m2": float(cb.area),
            "source": cb.source,
            "has_camera_positions": cb.camera_positions is not None,
        }

    mesh_payload = _mesh_payload(mesh, origin) if mesh is not None else None
    if mesh_payload is not None:
        mesh_payload["dropped_faces_nonfinite"] = dropped_faces

    extent = (hi - lo).astype(float)
    payload: dict[str, Any] = {
        "schema": PAYLOAD_VERSION,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "name": twin.name,
        "empty": bool(empty),
        "origin": origin.tolist(),
        "bounds_twin": [lo.tolist(), hi.tolist()],
        "bounds_view": [(lo - origin).tolist(), (hi - origin).tolist()],
        "extent_m": extent.tolist(),
        "points": points,
        "mesh": mesh_payload,
        "warnings": warnings,
        "structure": {
            "floor_z": float(s.floor_z),
            "ceiling_z": None if s.ceiling_z is None else float(s.ceiling_z),
            "ceiling_height_m": s.ceiling_height,
            "floor_area_m2": float(s.floor_area),
            "planes": planes,
            "openings": openings,
            "footprint": footprint,
            "footprint_source": s.footprint_source,
            "fixture_count": len(s.fixtures),
            "wall_count": len(s.walls()),
            "shell": shell_payload,
            "ceiling_z_inferred": (
                None if s.ceiling_z_inferred is None else float(s.ceiling_z_inferred)
            ),
            "ceiling_source": s.ceiling_source,
        },
        "capture_bounds": capture,
        "splat": _splat_payload(twin, origin) if not empty else None,
        "qa": twin.qa.to_dict(),
        "georeference": None if twin.georeference is None else twin.georeference.to_dict(),
        "provenance": twin.provenance,
        "summary": twin.summary(),
        "plane_tints": PLANE_TINTS,
    }
    return _jsonable(payload)


def render_html(
    twin: Twin,
    out_path: str | Path,
    *,
    max_points: int = 900_000,
    title: str | None = None,
) -> Path:
    """Write the twin as one self-contained HTML file and return its path.

    The result has no external references of any kind -- no script src, no
    stylesheet link, no image URL -- because the person checking a twin against
    a tape measure is often standing in the room they scanned, on a laptop with
    no signal. Anything the page needs is inside the page.
    """
    out_path = Path(out_path)
    if out_path.suffix.lower() not in {".html", ".htm"}:
        out_path = out_path.with_suffix(".html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    if _PAYLOAD_TOKEN not in template or _TITLE_TOKEN not in template:
        raise RuntimeError(f"{_TEMPLATE_PATH.name} is missing its substitution tokens")

    payload = twin_to_payload(twin, max_points=max_points)
    heading = title or twin.name or "untitled twin"
    # The title is substituted into text nodes only, but it comes from a file
    # name we did not choose, so it is escaped rather than trusted.
    safe_title = (
        heading.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )
    html = template.replace(_TITLE_TOKEN, safe_title).replace(_PAYLOAD_TOKEN, _embed_json(payload))
    out_path.write_text(html, encoding="utf-8")
    return out_path
