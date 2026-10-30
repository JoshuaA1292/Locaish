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

People in the frame are solids, not splats. A `Figure` with a `mesh` (see
`product/figures.py`, or `stand_in` here) is rasterised after the room:
flat-shaded triangles under a fixed key light, z-buffered against each
other and against a depth map of the twin's *measured* points, outlined
along the silhouette. Drawing an actor as a cloud of opaque gaussians gave a
flat blob with no limbs and no edges, and a blob a metre from the lens is a
smear; a shaded body with a lit face reads as a person facing somewhere.
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
# The stand-in's colour when no cast colour is known: the stick figure's blue.
STAND_IN_RGB = (110, 168, 254)

# Solid figures. The key light sits up and to the camera's left, a little
# behind the lens, so a face turned to camera is lit and its far cheek is
# not. The floor tolerance keeps the feet from being clipped by the floor's
# own points; the outline darkens one pixel of silhouette so a figure reads
# as a body against a wall of the same tone.
_FIG_AMBIENT = 0.45
_FIG_KEY = (-0.45, -0.75, -0.5)      # in (right, down, fwd) camera axes
_FIG_FLOOR_TOL_M = 0.05
_FIG_FLOOR_TOL_PER_M = 0.02      # the splats widen with range; so does the tolerance
_FIG_SPLAT_COVER = 0.7           # splat radius as a fraction of the sample spacing on screen
_FIG_SPLAT_MAX_PX = 14
_FIG_OUTLINE = 0.45
_FIG_EDGE_DEPTH_M = 0.08
_FIG_BUCKETS = (4, 8, 16, 32, 64)
_FIG_NEAR_M = 0.05

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


@dataclass
class Figure:
    """Something put into the room for this frame and nothing else: an actor
    on a mark, a set piece moved from where the scan found it. Points with
    colours, drawn as opaque gaussians of `radius_m`; or, when `cov` is given,
    a real gaussian splat (what SAM 3D Objects emits), drawn as it is. In the
    twin's frame, in metres. A figure never enters a measurement: it is drawn
    into the frame so the frame shows what the plan claims, which is the
    renderer's one job.

    With `mesh` = (vertices (V, 3) float64, faces (F, 3) int, outward
    winding) the figure is a solid body: `render_shot` rasterises it as
    flat-shaded triangles with a z-buffer over the twin's render, occluded
    where the twin's measured geometry is nearer. `mesh_rgb` is an optional
    (F, 3) float 0-1 colour per face (the face patch, the shoes); without it
    every face takes the figure's colour. `xyz`/`rgb` stay what they were --
    for a mesh figure, surface samples of the mesh -- so anything that only
    knows points (the viewer's figure layer, `as_gaussians`) still gets a
    body. `product/figures.stand_in` builds one from a height.
    """

    xyz: np.ndarray
    rgb: np.ndarray
    radius_m: float = 0.02
    cov: np.ndarray | None = None
    opacity: np.ndarray | None = None
    label: str = ""
    mesh: tuple[np.ndarray, np.ndarray] | None = None
    mesh_rgb: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=np.float64).reshape(-1, 3)
        rgb = np.asarray(self.rgb)
        if rgb.ndim == 1:
            rgb = np.broadcast_to(rgb.reshape(1, 3), (len(self.xyz), 3))
        if rgb.dtype.kind in "ui" or rgb.max(initial=0.0) > 1.0:
            rgb = rgb.astype(np.float32) / 255.0
        self.rgb = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)
        if len(self.rgb) != len(self.xyz):
            raise ValueError("a figure needs one colour per point")
        if self.mesh is not None:
            V = np.asarray(self.mesh[0], dtype=np.float64).reshape(-1, 3)
            F = np.asarray(self.mesh[1], dtype=np.int64).reshape(-1, 3)
            if len(F) and (F.min() < 0 or F.max() >= len(V)):
                raise ValueError("mesh faces index outside the vertices")
            self.mesh = (V, F)
            if self.mesh_rgb is not None:
                mr = np.asarray(self.mesh_rgb, dtype=np.float32).reshape(-1, 3)
                if mr.max(initial=0.0) > 1.0:
                    mr = mr / 255.0
                if len(mr) != len(F):
                    raise ValueError("mesh_rgb needs one colour per face")
                self.mesh_rgb = mr

    @property
    def colour(self) -> np.ndarray:
        """One colour for the whole figure, 0-1: the mean of its points'."""
        return self.rgb.mean(axis=0) if len(self.rgb) else np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def as_gaussians(self) -> GaussianField:
        n = len(self.xyz)
        if self.cov is not None:
            cov = np.asarray(self.cov, dtype=np.float32).reshape(n, 3, 3)
        else:
            cov = np.zeros((n, 3, 3), dtype=np.float32)
            cov[:, 0, 0] = cov[:, 1, 1] = cov[:, 2, 2] = float(self.radius_m) ** 2
        opa = (np.asarray(self.opacity, dtype=np.float32).reshape(n) if self.opacity is not None
               else np.full(n, 0.98, dtype=np.float32))
        return GaussianField(xyz=self.xyz, rgb=self.rgb, opacity=opa, cov=cov)

    def translated(self, dx: float, dy: float, dz: float = 0.0) -> "Figure":
        d = np.array([dx, dy, dz])
        mesh = None if self.mesh is None else (self.mesh[0] + d, self.mesh[1])
        return Figure(self.xyz + d, self.rgb, self.radius_m, self.cov,
                      self.opacity, self.label, mesh, self.mesh_rgb)


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
    figures: list["Figure"] | None = None,
    subject_height_m: float = SUBJECT_HEIGHT_M,
    exclude_points: np.ndarray | None = None,
):
    """Project the twin through a pinhole at the given setup. Returns a PIL image.

    The camera is aimed at the subject's eyeline, exactly as the sweep assumed
    when it scored the row being rendered. `subject_marks` may list further
    (x, y) marks to draw stand-ins at -- the other actor in a two-shot.

    `figures` are drawn into the room as part of it (composited with the
    gaussian field, or painted with the points), so an actor or a moved sofa
    occludes and is occluded like the walls are. A figure carrying a `mesh`
    is instead rasterised as a shaded solid over the room's render, z-tested
    against the twin's measured points (`draw_figures`). `subject_height_m`
    is the stature the stand-in and the eyeline use; the sweep's 1.75 m
    unless the cast is known. `exclude_points` is a boolean mask over
    `twin.points` naming points to leave out of the frame -- the sofa where
    it used to be, once a set piece has been moved; it applies to the point
    render and the figures' depth test only, the gaussian field being a view
    layer that is masked in `figures` space.

    To draw a proper person rather than the stick stand-in, hand in
    `stand_in(height_m, xy, facing_deg, floor_z)` (or a cast actor via
    `product.figures.figure_for_actor`) as one of `figures` and pass
    `draw_subject=False`.
    """
    from PIL import Image, ImageDraw

    floor_z = float(twin.structure.floor_z)
    subj = np.asarray(subject_xy, dtype=np.float64).reshape(2)
    eye = np.array([subj[0], subj[1], floor_z + EYELINE_RATIO * float(subject_height_m)])
    cam = camera_at(cam_pos, eye, focal_mm, sensor_key=sensor_key, width_px=width_px)

    solids = [f for f in (figures or []) if f.mesh is not None]
    splats = [f for f in (figures or []) if f.mesh is None and len(f.xyz)]
    field = gaussian_field(twin)
    if field is not None:
        if splats:
            field = merge_fields([field] + [f.as_gaussians() for f in splats])
        img = _render_gaussians(field, cam)
    else:
        extra = [(f.xyz, (f.rgb * 255.0 + 0.5).astype(np.uint8)) for f in splats]
        img = _render_points(twin, cam, extra=extra, exclude=exclude_points)
    if solids:
        img = draw_figures(img, twin, cam, solids, exclude=exclude_points)

    im = Image.fromarray(img)
    if draw_subject:
        draw = ImageDraw.Draw(im, "RGBA")
        marks = [subj] + [np.asarray(m, dtype=np.float64).reshape(2) for m in (subject_marks or [])]
        for i, m in enumerate(marks):
            _draw_subject(draw, cam, m, floor_z, primary=(i == 0), height_m=float(subject_height_m))

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


def merge_fields(fields: list[GaussianField]) -> GaussianField:
    """Several gaussian fields in one frame become one field: the room and
    whatever was put into it. Concatenation, nothing else, which is the
    whole reason set pieces and the room share a representation."""
    fields = [f for f in fields if f is not None and len(f.xyz)]
    if not fields:
        raise ValueError("nothing to merge")
    if len(fields) == 1:
        return fields[0]
    return GaussianField(
        xyz=np.concatenate([np.asarray(f.xyz, dtype=np.float64) for f in fields]),
        rgb=np.concatenate([np.asarray(f.rgb, dtype=np.float32).reshape(-1, 3) for f in fields]),
        opacity=np.concatenate([np.asarray(f.opacity, dtype=np.float32).reshape(-1) for f in fields]),
        cov=np.concatenate([np.asarray(f.cov, dtype=np.float32).reshape(-1, 3, 3) for f in fields]),
    )


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
    except MemoryError:
        # Transient by nature -- another render or a build was holding the
        # memory. Do not poison the cache: the next frame should try again
        # rather than condemn the whole plan to point renders.
        return None
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


def _render_points(twin: Twin, cam: Camera, *, extra=None, exclude: np.ndarray | None = None) -> np.ndarray:
    W, H = cam.width_px, cam.height_px
    xyz = np.asarray(twin.points.xyz, dtype=np.float64)
    rgb = twin.points.rgb
    if rgb is None:
        rgb = np.full((len(xyz), 3), 170, dtype=np.uint8)
    if exclude is not None and len(exclude) == len(xyz):
        keep = ~np.asarray(exclude, dtype=bool)
        xyz, rgb = xyz[keep], rgb[keep]
    if extra:
        xyz = np.concatenate([xyz] + [np.asarray(e[0], dtype=np.float64).reshape(-1, 3) for e in extra])
        rgb = np.concatenate([np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)]
                             + [np.asarray(e[1], dtype=np.uint8).reshape(-1, 3) for e in extra])

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
# solid figures: a depth map of the twin, triangles z-tested against it
# ---------------------------------------------------------------------------


def stand_in(
    height_m: float, xy, facing_deg: float, floor_z: float, *,
    pose: str = "standing", phase: float = 0.0, rgb=None, label: str = "",
) -> "Figure":
    """A proper person at the mark, as a mesh figure: `figures.figure_mesh`
    at the given height, pose (`standing`, `walking` with `phase` in cycles,
    `sitting`) and facing (degrees anticlockwise from +X in plan). `rgb` is
    the body colour, 0-1 or 0-255; the stick figure's blue by default.

    One call per actor per frame is the whole integration::

        figs = [render.stand_in(h, (x, y), facing, floor_z,
                                pose="walking", phase=figures.walk_phase(t, speed), rgb=colour)
                for (x, y, facing) in marks]
        render.render_shot(twin, cam_pos, subj_xy, focal, figures=figs, draw_subject=False)

    Callers that build their own camera and call the rasterisers directly
    (the sim's frame render) composite with `draw_figures(img, twin, cam, figs)`.
    """
    from ..product import figures

    return figures.stand_in(height_m, xy, facing_deg, floor_z, pose=pose, phase=phase, rgb=rgb, label=label)


def twin_depth_map(twin: Twin, cam: Camera, *, exclude: np.ndarray | None = None) -> np.ndarray:
    """Depth (metres along the camera axis) of the twin's nearest *measured*
    point at each pixel, +inf where nothing was seen. Painter's splats as in
    `_render_points`, writing nearest depth instead of colour, with the
    splat sized to the cloud's sample spacing so a sofa's returns occlude as
    a sofa. The gaussian field is never consulted, even when it painted the
    frame: it is a view layer, and figures hide behind what was measured,
    not behind what was painted in."""
    from scipy import ndimage

    W, H = cam.width_px, cam.height_px
    pts = twin.points
    keep = np.ones(len(pts.xyz), dtype=bool)
    if pts.inferred is not None:
        keep &= np.asarray(pts.inferred) <= 0.5
    if exclude is not None and len(exclude) == len(keep):
        keep &= ~np.asarray(exclude, dtype=bool)
    xyz = np.asarray(pts.xyz, dtype=np.float64)[keep]
    n_all = max(len(xyz), 1)

    rel = xyz - cam.pos
    z = rel @ cam.fwd
    infront = z > _FIG_NEAR_M
    rel, z = rel[infront], z[infront]
    u = (rel @ cam.right) / z * cam.fx + W / 2.0
    v = (rel @ cam.down) / z * cam.fy + H / 2.0
    inframe = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    u, v, z = u[inframe], v[inframe], z[inframe]

    depth = np.full((H, W), np.inf, dtype=np.float64)
    if not len(z):
        return depth
    ui = u.astype(np.int64)
    vi = v.astype(np.int64)
    # Each point stands for a patch of surface about one sample spacing
    # across, so its splat must cover that spacing on screen or a sofa at
    # range is a sieve and the figure behind it shows through the holes.
    # The spacing is estimated from the cloud's bounding surface over its
    # count -- an overestimate for a furnished room, which errs generous.
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    ext = np.maximum(hi - lo, 0.1)
    area = 2.0 * (ext[0] * ext[1] + ext[0] * ext[2] + ext[1] * ext[2])
    spacing_m = math.sqrt(area / n_all)
    radius = np.clip(np.ceil(_FIG_SPLAT_COVER * spacing_m * cam.fx / z).astype(np.int64), 1, _FIG_SPLAT_MAX_PX)
    for r in np.unique(radius):
        sel = radius == r
        layer = np.full(H * W, np.inf, dtype=np.float64)
        np.minimum.at(layer, vi[sel] * W + ui[sel], z[sel])
        layer = layer.reshape(H, W)
        if r > 1:
            layer = ndimage.minimum_filter(layer, size=int(2 * r - 1), mode="nearest")
        np.minimum(depth, layer, out=depth)
    return depth


def draw_figures(img: np.ndarray, twin: Twin, cam: Camera, figures: list["Figure"], *,
                 exclude: np.ndarray | None = None, depth: np.ndarray | None = None) -> np.ndarray:
    """Composite mesh figures over a rendered frame: flat-shaded triangles,
    z-buffered among themselves and against the twin's measured depth (with
    a floor tolerance), outlined along silhouettes. Returns a new (H, W, 3)
    uint8 image. `depth` may be passed to reuse `twin_depth_map`."""
    solids = [f for f in figures if f.mesh is not None and len(f.mesh[1])]
    if not solids:
        return img
    H, W = img.shape[:2]
    if depth is None:
        depth = twin_depth_map(twin, cam, exclude=exclude)
    zbuf = np.full((H, W), np.inf, dtype=np.float64)
    colour = np.asarray(img, dtype=np.float32) / 255.0
    key = np.asarray(_FIG_KEY, dtype=np.float64)
    L = key[0] * cam.right + key[1] * cam.down + key[2] * cam.fwd
    L /= np.linalg.norm(L)

    for fig in solids:
        V, F = fig.mesh
        face_rgb = fig.mesh_rgb if fig.mesh_rgb is not None else np.repeat(fig.colour[None], len(F), axis=0)
        _raster_mesh(V, F, face_rgb, cam, L, depth, zbuf, colour)

    # Outline: a drawn pixel beside an undrawn one, or beside one more than
    # a few centimetres farther (an arm in front of the torso).
    drawn = np.isfinite(zbuf)
    if drawn.any():
        edge = np.zeros_like(drawn)
        zfill = np.where(drawn, zbuf, np.inf)
        with np.errstate(invalid="ignore"):
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nb = np.roll(zfill, (dy, dx), axis=(0, 1))
                nbd = np.roll(drawn, (dy, dx), axis=(0, 1))
                edge |= drawn & (~nbd | (nb - zfill > _FIG_EDGE_DEPTH_M))
        # Wrap-around from np.roll must not paint the far border.
        edge[0, :] |= drawn[0, :]
        edge[-1, :] |= drawn[-1, :]
        edge[:, 0] |= drawn[:, 0]
        edge[:, -1] |= drawn[:, -1]
        colour[edge] *= _FIG_OUTLINE
    return np.clip(colour * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _raster_mesh(V, F, face_rgb, cam: Camera, L, depth, zbuf, colour) -> None:
    """Rasterise one mesh into `zbuf`/`colour` in place. Per-triangle
    barycentric fill over the triangle's screen bbox, vectorised by bbox
    size, perspective-correct depth, nearest wins."""
    H, W = zbuf.shape
    tri = V[F]                                               # (F, 3, 3)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nlen = np.linalg.norm(nrm, axis=1)
    ok = nlen > 1e-12
    nrm = nrm / np.maximum(nlen, 1e-12)[:, None]
    # Back faces: the normal points away from the camera.
    front = np.einsum("ij,ij->i", nrm, tri[:, 0] - cam.pos) < 0
    ok &= front
    rel = V - cam.pos
    zc = rel @ cam.fwd
    uc = (rel @ cam.right) / np.maximum(zc, 1e-9) * cam.fx + W / 2.0
    vc = (rel @ cam.down) / np.maximum(zc, 1e-9) * cam.fy + H / 2.0
    ok &= np.all(zc[F] > _FIG_NEAR_M, axis=1)
    if not ok.any():
        return
    shade = _FIG_AMBIENT + (1.0 - _FIG_AMBIENT) * np.maximum(0.0, nrm @ L)
    col = np.asarray(face_rgb, dtype=np.float32) * shade[:, None].astype(np.float32)

    idx = np.flatnonzero(ok)
    su, sv, sz = uc[F[idx]], vc[F[idx]], zc[F[idx]]          # (n, 3)
    x0 = np.clip(np.floor(su.min(axis=1)), 0, W - 1).astype(np.int64)
    x1 = np.clip(np.ceil(su.max(axis=1)), 0, W - 1).astype(np.int64)
    y0 = np.clip(np.floor(sv.min(axis=1)), 0, H - 1).astype(np.int64)
    y1 = np.clip(np.ceil(sv.max(axis=1)), 0, H - 1).astype(np.int64)
    inframe = (su.max(axis=1) >= 0) & (su.min(axis=1) < W) & (sv.max(axis=1) >= 0) & (sv.min(axis=1) < H)
    size = np.maximum(x1 - x0, y1 - y0) + 1
    # Signed doubled area; its sign is the screen winding, which the
    # inside test is made relative to.
    area = (su[:, 1] - su[:, 0]) * (sv[:, 2] - sv[:, 0]) - (su[:, 2] - su[:, 0]) * (sv[:, 1] - sv[:, 0])
    inframe &= np.abs(area) > 1e-9
    idx, su, sv, sz, x0, y0, size, area, col = (
        idx[inframe], su[inframe], sv[inframe], sz[inframe], x0[inframe], y0[inframe],
        size[inframe], area[inframe], col[idx][inframe],
    )
    if not len(idx):
        return
    inv_z = 1.0 / sz

    def fill(sel, S):
        """Candidates for triangles `sel` on an S x S grid from each bbox min."""
        offs = np.arange(S)
        gx, gy = np.meshgrid(offs, offs, indexing="xy")
        gx, gy = gx.ravel(), gy.ravel()
        px = x0[sel][:, None] + gx[None]                       # (m, S*S)
        py = y0[sel][:, None] + gy[None]
        cx = px + 0.5
        cy = py + 0.5
        a_u, a_v = su[sel], sv[sel]
        w0 = (a_u[:, 1, None] - cx) * (a_v[:, 2, None] - cy) - (a_u[:, 2, None] - cx) * (a_v[:, 1, None] - cy)
        w1 = (a_u[:, 2, None] - cx) * (a_v[:, 0, None] - cy) - (a_u[:, 0, None] - cx) * (a_v[:, 2, None] - cy)
        w2 = area[sel][:, None] - w0 - w1
        sgn = np.sign(area[sel])[:, None]
        inside = (w0 * sgn >= 0) & (w1 * sgn >= 0) & (w2 * sgn >= 0) & (px < W) & (py < H)
        if not inside.any():
            return None
        b0 = w0 / area[sel][:, None]
        b1 = w1 / area[sel][:, None]
        b2 = 1.0 - b0 - b1
        iz = inv_z[sel]
        zpix = 1.0 / (b0 * iz[:, 0, None] + b1 * iz[:, 1, None] + b2 * iz[:, 2, None])
        flat = (py * W + px)[inside]
        zp = zpix[inside]
        cp = np.repeat(col[sel][:, None, :], S * S, axis=1)[inside]
        return flat, zp, cp

    flats, zs, cs = [], [], []
    lo = 0
    for S in _FIG_BUCKETS:
        sel = np.flatnonzero((size > lo) & (size <= S))
        lo = S
        if len(sel):
            got = fill(sel, S)
            if got is not None:
                flats.append(got[0]); zs.append(got[1]); cs.append(got[2])
    big = np.flatnonzero(size > _FIG_BUCKETS[-1])
    for t in big:
        S = int(size[t])
        got = fill(np.array([t]), S)
        if got is not None:
            flats.append(got[0]); zs.append(got[1]); cs.append(got[2])
    if not flats:
        return
    flat = np.concatenate(flats)
    zp = np.concatenate(zs)
    cp = np.concatenate(cs)
    # Nearest candidate per pixel: a sort by depth, then the first of each pixel.
    order = np.argsort(zp, kind="stable")
    flat, zp, cp = flat[order], zp[order], cp[order]
    _, first = np.unique(flat, return_index=True)
    flat, zp, cp = flat[first], zp[first], cp[first]
    zb = zbuf.reshape(-1)
    dp = depth.reshape(-1)
    win = (zp < zb[flat]) & (zp < dp[flat] + _FIG_FLOOR_TOL_M + _FIG_FLOOR_TOL_PER_M * zp)
    flat, zp, cp = flat[win], zp[win], cp[win]
    zb[flat] = zp
    colour.reshape(-1, 3)[flat] = cp


# ---------------------------------------------------------------------------
# the stand-in
# ---------------------------------------------------------------------------


def _draw_subject(draw, cam: Camera, subj_xy, floor_z: float, *, primary: bool = True,
                  height_m: float = SUBJECT_HEIGHT_M):
    """A stand-in figure at the mark: feet, stature line, head, shoulder bar."""
    x, y = float(subj_xy[0]), float(subj_xy[1])
    feet = cam.project((x, y, floor_z))
    head = cam.project((x, y, floor_z + height_m))
    if feet is None or head is None:
        return
    shoulder_z = floor_z + 0.82 * height_m
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
    crown = floor_z + height_m - 0.11
    side = cam.project((x + lat[0] * 0.11, y + lat[1] * 0.11, crown))
    mid = cam.project((x, y, crown))
    r = max(3.0, math.hypot(side[0] - mid[0], side[1] - mid[1])) if side and mid else 6.0
    cy = mid[1] if mid else head[1] + r
    draw.ellipse([head[0] - r, cy - r, head[0] + r, cy + r], outline=colour, width=3)
