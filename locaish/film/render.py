"""Render what a proposed camera would actually see, from the twin itself.

Not a visualisation of the answer -- the answer. A row in the shot table says
"50 mm from (2.1, 0.4) at 1.55 m reads as a medium close-up"; this module
points a pinhole camera with exactly those numbers at the twin and produces
the frame, so the claim can be checked by looking at it. The projection is
the same thin-lens arithmetic the sweep scored with, which is the point: if
the render looks wrong, the table *is* wrong, and no amount of prose in
between can hide it.

Two sources can stand behind the frame. When the twin was reconstructed from
video it carries a trained gaussian field (see `video/splat.py`), and that is
rendered here as the field it is: every gaussian projected to an ellipse,
weighted, alpha-composited front to back. The field closes the blank walls a
point cloud leaves open, and the rendered frame looks like the room rather
than like a survey of it -- which matters twice over, because a director is
going to judge the frame by eye, and so is Gemini. A twin without a field
(a scan-file import) falls back to the measured points.

The compositing is an approximation a CPU can afford: gaussians are sorted
by depth into thin slabs, blended order-independently within a slab, and the
slabs composited in order. Within-slab order is the only thing lost, and a
slab is thin enough that what it contains is one surface.

A stand-in for the subject is drawn at the mark -- a simple figure of the
standard 1.75 m stature the sweep framed against -- because a frame of an
empty room does not show what "waist up" means.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..types import Twin
from . import optics

WIDTH_PX = 960
BACKGROUND = (14, 16, 19)
SUBJECT_HEIGHT_M = 1.75
EYELINE_RATIO = 0.94

# Gaussian rasterisation. The radius cap bounds the work per gaussian: a
# gaussian wider than this on screen is a wall filler with many neighbours,
# and truncating its tails costs nothing visible. Slabs are in log depth so
# the near field, where one surface is a metre from the next, gets thin ones.
_GAUSS_RADIUS_CAP_PX = 9
_GAUSS_SIGMA_CUT = 2.5
_GAUSS_SLABS = 32
_GAUSS_OPACITY_MIN = 0.04
_GAUSS_S1_CAP_M = 0.35
_GAUSS_ALPHA_MAX = 0.99
_GAUSS_LEVELS = 4


@dataclass(frozen=True)
class Camera:
    """A pinhole placed in the twin: position, basis and pixel focal lengths."""

    pos: np.ndarray
    fwd: np.ndarray
    right: np.ndarray
    down: np.ndarray
    fx: float
    fy: float
    width_px: int
    height_px: int

    def project(self, p) -> tuple[float, float] | None:
        rel = np.asarray(p, dtype=np.float64) - self.pos
        z = float(rel @ self.fwd)
        if z <= 0.05:
            return None
        return (
            float(rel @ self.right) / z * self.fx + self.width_px / 2.0,
            float(rel @ self.down) / z * self.fy + self.height_px / 2.0,
        )


def camera_at(
    cam_pos, aim_at, focal_mm: float, *, sensor_key: str = optics.DEFAULT_SENSOR,
    width_px: int = WIDTH_PX,
) -> Camera:
    """A pinhole at `cam_pos` looking at `aim_at`, with the sensor's aspect."""
    sensor = optics.SENSORS[sensor_key]
    cam = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    aim = np.asarray(aim_at, dtype=np.float64).reshape(3)
    fwd = aim - cam
    n = float(np.linalg.norm(fwd))
    if n < 1e-6:
        raise ValueError("camera and aim point coincide")
    fwd = fwd / n
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    rn = float(np.linalg.norm(right))
    right = right / rn if rn > 1e-9 else np.array([1.0, 0.0, 0.0])
    down = np.cross(fwd, right)
    height_px = int(round(width_px * sensor.height_mm / sensor.width_mm))
    fx = focal_mm / sensor.width_mm * width_px
    fy = focal_mm / sensor.height_mm * height_px
    return Camera(cam, fwd, right, down, fx, fy, width_px, height_px)


def render_shot(
    twin: Twin,
    cam_pos,
    subject_xy,
    focal_mm: float,
    *,
    sensor_key: str = optics.DEFAULT_SENSOR,
    width_px: int = WIDTH_PX,
    out: str | Path | None = None,
    draw_subject: bool = True,
    subject_marks=None,
):
    """Project the twin through a pinhole at the given setup. Returns a PIL image.

    The camera is aimed at the subject's eyeline, exactly as the sweep assumed
    when it scored the row being rendered. `subject_marks` may list further
    (x, y) marks to draw stand-ins at -- the other actor in a two-shot.
    """
    from PIL import Image, ImageDraw

    floor_z = float(twin.structure.floor_z)
    subj = np.asarray(subject_xy, dtype=np.float64).reshape(2)
    eye = np.array([subj[0], subj[1], floor_z + EYELINE_RATIO * SUBJECT_HEIGHT_M])
    cam = camera_at(cam_pos, eye, focal_mm, sensor_key=sensor_key, width_px=width_px)

    field = gaussian_field(twin)
    if field is not None:
        img = _render_gaussians(field, cam)
    else:
        img = _render_points(twin, cam)

    im = Image.fromarray(img)
    if draw_subject:
        draw = ImageDraw.Draw(im, "RGBA")
        marks = [subj] + [np.asarray(m, dtype=np.float64).reshape(2) for m in (subject_marks or [])]
        for i, m in enumerate(marks):
            _draw_subject(draw, cam, m, floor_z, primary=(i == 0))

    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        im.save(out)
    return im


# ---------------------------------------------------------------------------
# the gaussian field, in the twin's frame
# ---------------------------------------------------------------------------


@dataclass
class GaussianField:
    xyz: np.ndarray        # (N, 3) twin frame
    rgb: np.ndarray        # (N, 3) float 0-1
    opacity: np.ndarray    # (N,)
    cov: np.ndarray        # (N, 3, 3) world-frame covariance


_FIELD_CACHE: dict[str, GaussianField | None] = {}
_FIELD_LOCK = threading.Lock()


def gaussian_field(twin: Twin) -> GaussianField | None:
    """The twin's trained gaussians transformed into its own frame, cached.

    The transform chain is the one the viewer uses (`viewer/build.py`): metric
    scale, then the canonical rotation that levelled and squared the room. The
    cache is keyed on the field file, so a planner rendering thirty frames
    pays the decode once.
    """
    path = (twin.provenance or {}).get("splat_ply")
    if not path or not Path(path).exists():
        return None
    steps = (twin.provenance or {}).get("steps", {})
    factor = ((steps.get("video") or {}).get("scale") or {}).get("factor")
    if not factor:
        return None
    key = f"{path}:{os.path.getmtime(path)}:{twin.name}"
    with _FIELD_LOCK:
        if key in _FIELD_CACHE:
            return _FIELD_CACHE[key]
        # Decode under the lock: two concurrent renders of one twin would
        # otherwise both pay for the decode, and the second wins nothing.
        return _decode_field(twin, path, factor, key)


def _decode_field(twin: Twin, path: str, factor: float, key: str) -> GaussianField | None:
    try:
        from ..video.splat import read_gaussian_ply

        g = read_gaussian_ply(path)
    except Exception:
        _FIELD_CACHE[key] = None
        return None

    M = np.asarray(twin.canonical_transform, dtype=np.float64)
    R, t = M[:3, :3], M[:3, 3]
    xyz = (g["xyz"] * float(factor)) @ R.T + t
    scale = g["scale"] * float(factor)
    s1 = scale.max(axis=1)
    lo = twin.points.xyz.min(axis=0) - 0.4
    hi = twin.points.xyz.max(axis=0) + 0.4
    keep = (
        (g["opacity"] >= _GAUSS_OPACITY_MIN)
        & (s1 <= _GAUSS_S1_CAP_M)
        & np.all((xyz >= lo) & (xyz <= hi), axis=1)
    )
    idx = np.flatnonzero(keep)
    if len(idx) < 1000:
        _FIELD_CACHE[key] = None
        return None

    # Σ_world = R Rq S Sᵀ Rqᵀ Rᵀ, with Rq the gaussian's own rotation.
    Rq = _quat_to_rot(g["quat"][idx])                # (n, 3, 3)
    RS = (R[None] @ Rq) * scale[idx][:, None, :]     # (n, 3, 3): columns scaled
    cov = RS @ np.transpose(RS, (0, 2, 1))
    field = GaussianField(
        xyz=xyz[idx].astype(np.float64),
        rgb=g["rgb01"][idx].astype(np.float32),
        opacity=g["opacity"][idx].astype(np.float32),
        cov=cov.astype(np.float32),
    )
    _FIELD_CACHE[key] = field
    return field


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


# ---------------------------------------------------------------------------
# rasterisers
# ---------------------------------------------------------------------------


def _render_gaussians(field: GaussianField, cam: Camera) -> np.ndarray:
    W, H = cam.width_px, cam.height_px
    basis = np.stack([cam.right, cam.down, cam.fwd])          # rows: camera axes
    rel = field.xyz - cam.pos
    pc = rel @ basis.T                                        # (N, 3) camera coords
    z = pc[:, 2]
    infront = z > 0.05
    pc, z = pc[infront], z[infront]
    cov = field.cov[infront]
    rgb = field.rgb[infront]
    opa = field.opacity[infront]

    u = pc[:, 0] / z * cam.fx + W / 2.0
    v = pc[:, 1] / z * cam.fy + H / 2.0
    # Generous frustum cull: the widest gaussian drawn spans this many pixels
    # at the coarsest level, so anything farther out cannot touch the frame.
    m = _GAUSS_RADIUS_CAP_PX * (1 << (_GAUSS_LEVELS - 1)) + 1
    inview = (u > -m) & (u < W + m) & (v > -m) & (v < H + m)
    pc, z, u, v = pc[inview], z[inview], u[inview], v[inview]
    cov, rgb, opa = cov[inview], rgb[inview], opa[inview]
    n = len(z)
    if n == 0:
        return np.full((H, W, 3), BACKGROUND, dtype=np.uint8)

    # 2D covariance: Σ' = J B Σ Bᵀ Jᵀ, J the projection jacobian at the centre.
    covc = basis[None] @ cov.astype(np.float64) @ basis.T[None]   # camera-frame Σ
    x, y = pc[:, 0], pc[:, 1]
    J = np.zeros((n, 2, 3))
    J[:, 0, 0] = cam.fx / z
    J[:, 0, 2] = -cam.fx * x / (z * z)
    J[:, 1, 1] = cam.fy / z
    J[:, 1, 2] = -cam.fy * y / (z * z)
    c2 = J @ covc @ np.transpose(J, (0, 2, 1))                   # (n, 2, 2)
    # A third of a pixel of low-pass, as every splat renderer adds, so a
    # gaussian smaller than a pixel still lands on one.
    c2[:, 0, 0] += 0.3
    c2[:, 1, 1] += 0.3
    det = c2[:, 0, 0] * c2[:, 1, 1] - c2[:, 0, 1] * c2[:, 1, 0]
    ok = det > 1e-9
    c2, det = c2[ok], det[ok]
    u, v, z, rgb, opa = u[ok], v[ok], z[ok], rgb[ok], opa[ok]
    n = len(z)
    if n == 0:
        return np.full((H, W, 3), BACKGROUND, dtype=np.uint8)
    inv = np.empty_like(c2)
    inv[:, 0, 0] = c2[:, 1, 1] / det
    inv[:, 1, 1] = c2[:, 0, 0] / det
    inv[:, 0, 1] = -c2[:, 0, 1] / det
    inv[:, 1, 0] = -c2[:, 1, 0] / det
    # Extent: the larger eigenvalue's standard deviation, in pixels. A
    # gaussian is rasterised at the resolution level where that extent fits
    # under the radius cap, so a wall-filling gaussian is drawn coarse and
    # whole rather than fine and truncated to a square.
    tr = c2[:, 0, 0] + c2[:, 1, 1]
    sigma = np.sqrt(0.5 * tr + np.sqrt(np.maximum(0.25 * tr * tr - det, 0.0)))
    sigma_fit = _GAUSS_RADIUS_CAP_PX / _GAUSS_SIGMA_CUT
    level = np.clip(np.ceil(np.log2(np.maximum(sigma / sigma_fit, 1e-6))), 0, _GAUSS_LEVELS - 1).astype(np.int64)

    # Depth slabs, log-spaced between the nearest and farthest gaussian.
    zlo, zhi = float(z.min()), float(z.max())
    if zhi <= zlo * 1.001:
        slab = np.zeros(n, dtype=np.int64)
    else:
        f = (np.log(z) - math.log(zlo)) / (math.log(zhi) - math.log(zlo))
        slab = np.clip((f * _GAUSS_SLABS).astype(np.int64), 0, _GAUSS_SLABS - 1)
    K = int(slab.max()) + 1
    npx = W * H

    acc_c = np.zeros((K, H, W, 3), dtype=np.float64)
    acc_a = np.zeros((K, H, W), dtype=np.float64)
    acc_logt = np.zeros((K, H, W), dtype=np.float64)

    for L in range(_GAUSS_LEVELS):
        at = np.flatnonzero(level == L)
        if not len(at):
            continue
        s = 1 << L
        WL, HL = -(-W // s), -(-H // s)
        npxL = WL * HL
        uL, vL = u[at] / s, v[at] / s
        invL = inv[at] * (s * s)
        rad = np.clip(np.ceil(_GAUSS_SIGMA_CUT * sigma[at] / s), 1, _GAUSS_RADIUS_CAP_PX).astype(np.int32)
        slabL, rgbL, opaL = slab[at], rgb[at], opa[at]
        a_c = np.zeros((K * npxL, 3), dtype=np.float64)
        a_a = np.zeros(K * npxL, dtype=np.float64)
        a_t = np.zeros(K * npxL, dtype=np.float64)
        for r in np.unique(rad):
            sel = np.flatnonzero(rad == r)
            foot = (2 * r + 1) ** 2
            step = max(1, 30_000_000 // foot)
            offs = np.arange(-r, r + 1)
            du, dv = np.meshgrid(offs, offs, indexing="xy")
            du, dv = du.ravel().astype(np.float64), dv.ravel().astype(np.float64)
            for s0 in range(0, len(sel), step):
                g = sel[s0:s0 + step]
                cu, cv = uL[g], vL[g]
                px = np.floor(cu)[:, None] + du[None, :]        # (m, foot)
                py = np.floor(cv)[:, None] + dv[None, :]
                dx = px + 0.5 - cu[:, None]
                dy = py + 0.5 - cv[:, None]
                a, b, c = invL[g, 0, 0][:, None], invL[g, 0, 1][:, None], invL[g, 1, 1][:, None]
                w = np.exp(-0.5 * (a * dx * dx + 2 * b * dx * dy + c * dy * dy))
                alpha = np.minimum(opaL[g][:, None] * w, _GAUSS_ALPHA_MAX)
                inb = (px >= 0) & (px < WL) & (py >= 0) & (py < HL) & (alpha > 1.0 / 255.0)
                pix = (slabL[g][:, None] * npxL + py.astype(np.int64) * WL + px.astype(np.int64))[inb]
                al = alpha[inb]
                col = np.repeat(rgbL[g][:, None, :], foot, axis=1)[inb]
                a_a += np.bincount(pix, weights=al, minlength=K * npxL)
                for ch in range(3):
                    a_c[:, ch] += np.bincount(pix, weights=al * col[:, ch], minlength=K * npxL)
                a_t += np.bincount(pix, weights=np.log1p(-al), minlength=K * npxL)
        if s == 1:
            acc_a += a_a.reshape(K, H, W)
            acc_logt += a_t.reshape(K, H, W)
            acc_c += a_c.reshape(K, H, W, 3)
        else:
            stacked = np.concatenate(
                [a_a.reshape(K, HL, WL, 1), a_t.reshape(K, HL, WL, 1), a_c.reshape(K, HL, WL, 3)],
                axis=3,
            ).astype(np.float32)
            live = np.flatnonzero(np.bincount(slabL, minlength=K))
            up = _upsample(stacked[live], s, H, W)
            acc_a[live] += up[..., 0]
            acc_logt[live] += up[..., 1]
            acc_c[live] += up[..., 2:]

    # Composite the slabs front to back.
    out = np.zeros((H, W, 3), dtype=np.float64)
    T = np.ones((H, W), dtype=np.float64)
    for k in range(K):
        A = 1.0 - np.exp(acc_logt[k])
        den = acc_a[k]
        colour = np.where(den[..., None] > 0, acc_c[k] / np.maximum(den, 1e-12)[..., None], 0.0)
        out += (T * A)[..., None] * colour
        T *= 1.0 - A
    bg = np.asarray(BACKGROUND, dtype=np.float64) / 255.0
    out += T[..., None] * bg[None, None, :]
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _upsample(arr: np.ndarray, s: int, H: int, W: int) -> np.ndarray:
    """Upsample (K, h, w, C) by an integer factor, cropped to (H, W).

    Nearest-neighbour repeat followed by two box filters of the same width,
    which is a triangle filter -- bilinear in effect -- and runs in scipy's
    separable C loops instead of four fancy-index gathers per slab.
    """
    from scipy import ndimage

    K, h, w, C = arr.shape
    out = np.empty((K, H, W, C), dtype=np.float32)
    for k in range(K):
        rep = np.repeat(np.repeat(arr[k], s, axis=0), s, axis=1)
        rep = ndimage.uniform_filter(rep, size=(s, s, 1), mode="nearest")
        out[k] = ndimage.uniform_filter(rep, size=(s, s, 1), mode="nearest")[:H, :W]
    return out


def _render_points(twin: Twin, cam: Camera) -> np.ndarray:
    W, H = cam.width_px, cam.height_px
    xyz = np.asarray(twin.points.xyz, dtype=np.float64)
    rgb = twin.points.rgb
    if rgb is None:
        rgb = np.full((len(xyz), 3), 170, dtype=np.uint8)

    rel = xyz - cam.pos
    z = rel @ cam.fwd
    infront = z > 0.05
    rel, z = rel[infront], z[infront]
    cols = rgb[infront]
    u = (rel @ cam.right) / z * cam.fx + W / 2.0
    v = (rel @ cam.down) / z * cam.fy + H / 2.0
    inframe = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    u, v, z, cols = u[inframe], v[inframe], z[inframe], cols[inframe]

    img = np.full((H, W, 3), BACKGROUND, dtype=np.uint8)
    if len(z):
        # Painter's algorithm: far points first, near points overwrite them.
        order = np.argsort(-z)
        ui = u[order].astype(np.int32)
        vi = v[order].astype(np.int32)
        zo = z[order]
        cz = cols[order]
        # Splat size scales with closeness, like a real surface would, and
        # the base scale adapts to how densely this twin was sampled.
        density_scale = float(np.clip(2.2 * math.sqrt(600_000 / max(len(xyz), 1)), 1.6, 6.0))
        radius = np.clip(np.round(density_scale / zo).astype(np.int32), 1, 5)
        for r in np.unique(radius):
            sel = radius == r
            us, vs, cs = ui[sel], vi[sel], cz[sel]
            for du in range(-int(r) + 1, int(r)):
                for dv in range(-int(r) + 1, int(r)):
                    uu = np.clip(us + du, 0, W - 1)
                    vv = np.clip(vs + dv, 0, H - 1)
                    img[vv, uu] = cs
    return img


# ---------------------------------------------------------------------------
# the stand-in
# ---------------------------------------------------------------------------


def _draw_subject(draw, cam: Camera, subj_xy, floor_z: float, *, primary: bool = True):
    """A stand-in figure at the mark: feet, stature line, head, shoulder bar."""
    x, y = float(subj_xy[0]), float(subj_xy[1])
    feet = cam.project((x, y, floor_z))
    head = cam.project((x, y, floor_z + SUBJECT_HEIGHT_M))
    if feet is None or head is None:
        return
    shoulder_z = floor_z + 0.82 * SUBJECT_HEIGHT_M
    lat = np.cross(cam.fwd, np.array([0.0, 0.0, 1.0]))
    n = float(np.linalg.norm(lat))
    lat = lat / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    half = 0.23
    s1 = cam.project((x - lat[0] * half, y - lat[1] * half, shoulder_z))
    s2 = cam.project((x + lat[0] * half, y + lat[1] * half, shoulder_z))

    colour = (110, 168, 254, 230) if primary else (229, 192, 123, 210)
    draw.line([feet, head], fill=colour, width=3)
    if s1 and s2:
        draw.line([s1, s2], fill=colour, width=3)
    # The head is a real 0.11 m radius at the head's own depth -- projected,
    # not scaled from the stature line, which runs off the bottom of a low
    # angle and would make the head the size of the frame.
    crown = floor_z + SUBJECT_HEIGHT_M - 0.11
    side = cam.project((x + lat[0] * 0.11, y + lat[1] * 0.11, crown))
    mid = cam.project((x, y, crown))
    r = max(3.0, math.hypot(side[0] - mid[0], side[1] - mid[1])) if side and mid else 6.0
    cy = mid[1] if mid else head[1] + r
    draw.ellipse([head[0] - r, cy - r, head[0] + r, cy + r], outline=colour, width=3)
