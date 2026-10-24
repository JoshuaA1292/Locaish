"""Learned geometry for the video front-end: metres, and surface where stereo has none.

Two things a classical reconstruction of a room cannot do, and a monocular
geometry model can:

**Say how big the room is.** Parallax recovers shape, never size (see
`metric.py`); the pipeline's scale came from two physical priors -- how high a
person holds a phone, how tall a door is -- and on a real capture the door is
often not in shot, which leaves the twin's size resting on one prior with a
twelve-percent error bar. A metric monocular model has learned what size
things are from millions of rooms, and it answers in metres per frame. Aligned
against the sparse points COLMAP triangulated in that same frame, each frame
yields an independent reading of "metres per reconstruction unit". Tens of
frames give a median and a spread. That estimate shares no evidence with the
camera-height prior -- one is anatomy, the other is appearance -- which is the
property `metric.combine_scales` exists to exploit: when they agree the twin's
size is finally a measurement, and when they disagree the bar widens.

**Put surface on a blank wall.** Patch-match stereo returns nothing where
there is nothing to match, and most of a room is painted plaster. The model
predicts depth for every pixel; after per-frame alignment those depths are
checked against each other across neighbouring views -- a predicted wall that
reprojects into the next frame at the same depth is a wall, a hallucination
does not survive the second view -- and only the survivors are kept, and only
where the stereo cloud left a gap. Stereo keeps authority wherever it has an
answer; the model fills what it could not measure, and the twin records how
much of its surface came from which.

The model is MoGe-2 (Microsoft, MIT licence; weights on Hugging Face), chosen
over its peers for the only two things that matter here: it is metric out of
the box, and it is free to use commercially. `LOCAISH_METRIC_MODEL` swaps the
checkpoint; `LOCAISH_LEARNED=0` turns the stage off and the pipeline reverts to
its priors, warnings and all. Everything here degrades to "not available"
rather than to a wrong answer: no torch, no weights, no frames that align --
the caller gets None and says so.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .metric import ScaleEstimate, _confidence

# The checkpoint. ViT-L is the accurate one and costs ~1.5 s per frame on an
# Apple GPU, ~8 s on a laptop CPU; `moge-2-vitb-normal` is three times faster
# and a little less accurate, which is the right trade on a CPU-only host.
MODEL_ID = os.environ.get("LOCAISH_METRIC_MODEL", "Ruicheng/moge-2-vitl")

# Frames are handed to the model at this long side. The model's own token
# budget caps what it can use, and 1024 is already past it; decoding larger
# frames buys nothing but transfer time.
INFER_LONG_SIDE = 1024

# How many of the registered views are run through the model, spread evenly
# over the sweep. The scale anchor is a median over frames and is solid by a
# few dozen; the dense fill wants coverage, and every view costs a model call.
MAX_FRAMES_GPU = 64
MAX_FRAMES_CPU = 24

# Alignment of a frame's predicted depth to the sparse points COLMAP saw in
# it. Below this many points the ratio is a guess; above this much scatter
# the frame is not describing the same surfaces (motion blur, a mirror, a
# window) and is left out of the anchor rather than averaged in.
ALIGN_MIN_POINTS = 40
ALIGN_MAX_LOG_MAD = 0.25

# The metric head's own bias, as a prior on the log of the factor. MoGe-2's
# reported in-the-wild metric-scale error is on the order of ten percent, and
# it is systematic per scene rather than per frame -- a room full of
# oversized furniture reads oversized in every frame -- so averaging frames
# does not remove it. This is what keeps the anchor honest when it is the only
# one that fired: a single estimator inherits its prior's bias, not zero.
MODEL_LOG_BIAS = math.log(1.12)

# Cross-view consistency for the dense fill. A predicted point is kept when
# at least `MIN_VIEWS` other frames predict the same depth along its ray to
# within `REL` of the distance. The first version used 4% and two views and
# measured 3 cm RMS wall planarity against stereo's 2 cm on two real rooms;
# three views at 3% is the setting that keeps the fill from thickening the
# walls it was meant to complete.
CONSISTENCY_REL = 0.03
CONSISTENCY_MIN_VIEWS = 3
NEIGHBOURS = 4
PIXEL_STRIDE = 2

# A prediction is dropped where the ray meets the predicted surface at
# nearly grazing incidence (|cos| below this): monocular depth is at its
# worst along a floor or wall seen edge-on, and those pixels are exactly the
# far floor that bows a twin's gravity. 0.2 is about 78 degrees off the
# normal, which still admits the floor a walking phone sees.
INCIDENCE_MIN_COS = 0.2

# The verified fill is a shell a few centimetres thick around the true
# surface -- measured at about 5 cm RMS against stereo-fitted planes, with
# under 3 cm of bias and under 0.4 degrees of tilt -- because the model's
# per-pixel depth noise is a few percent of distance and the cross-view vote
# cannot thin what every view agrees on. Before the fill meets the stereo
# cloud it is collapsed onto its own mid-surface by moving least squares
# over a wide neighbourhood; what was a shell becomes a sheet sitting where
# the shell's median sat. Metres.
COLLAPSE_K = 24
COLLAPSE_PLANAR_M = 0.08
COLLAPSE_REJECT_M = 0.12

# Alignment is a single scale per frame, the median ratio of COLMAP depth to
# model depth at the sparse points. A scale-plus-shift fit was tried and
# reverted: on frames with a narrow depth range the two are collinear, the
# fitted shifts reached 13% of depth, and whole depth maps moved.

# Bumped whenever alignment or fusion changes behaviour, so a cached
# reconstruction built with the old fill cannot be mistaken for this one.
FUSION_VERSION = 7

# How the fill meets the stereo cloud. Stereo has authority where it
# answered; the model is for where it did not. A learned point within
# `REDUNDANT_M` of a stereo point is dropped outright. One with measured
# surface within `CONFORM_RADIUS_M` is *conformed*: projected onto the plane
# its stereo neighbours define, so it extends the measured wall laterally
# at the measured wall's position instead of sitting a few centimetres off
# it (which thickened walls from 2 cm RMS to 3.5 cm on both test rooms when
# the sheet was merely kept, and starved the floor when it was merely
# dropped past 20 cm). A point whose stereo neighbourhood is not planar, or
# that stands further than `CONFORM_MAX_OFF_M` from the measured plane, is
# dropped as clutter the model smoothed over. Beyond the radius the point
# stands as predicted.
REDUNDANT_M = 0.06
CONFORM_RADIUS_M = 0.20
CONFORM_NEIGHBOURS = 24
CONFORM_MIN_NEIGHBOURS = 12
CONFORM_PLANAR_M = 0.03
CONFORM_MAX_OFF_M = 0.12
# The stereo cloud is voxel-thinned to this pitch before the neighbour
# search, so the 24 nearest stereo points span a hand's width laterally
# whatever the stereo density. On a cloud at 20k points per square metre the
# 24 nearest sit inside a two-centimetre patch, and a plane fitted there has
# a normal set by sensor noise, not by the wall.
CONFORM_THIN_M = 0.03
FUSE_VOXEL_M = 0.015

# What the model may put where stereo has nothing at all. Horizontal
# surface -- the ceiling nobody films, the floor between the scraps stereo
# found -- is where the fill earns its keep, and the carve-based completion
# cannot help there. A stand-alone *vertical* sheet is a wall the model
# invented: the completion already closes the room where free space stops,
# and a learned wall a few centimetres off, with no measurement behind it,
# was found to swing the wall-arrangement vote by several degrees between
# otherwise identical runs. So stand-alone sheets are kept only when they
# lie within this angle of horizontal; a learned wall survives only by
# extending a measured one (the conform branch above).
STANDALONE_MAX_TILT_COS = 0.7


@dataclass
class FrameDepth:
    """One frame's prediction, at inference resolution, before alignment."""

    name: str
    depth: np.ndarray        # (h, w) metres, model's own scale
    mask: np.ndarray         # (h, w) bool, where the model trusts itself
    K: np.ndarray            # (3, 3) pixels at (w, h)
    size: tuple[int, int]    # (w, h)


@dataclass
class AlignedFrame:
    """A frame whose metric depth has been tied to the reconstruction's unit."""

    index: int               # row in the sparse model
    frame: FrameDepth
    units_per_metre: float   # multiply model depth by this to get COLMAP units
    log_mad: float
    points_used: int
    shift: float = 0.0       # added after scaling, COLMAP units


@dataclass
class LearnedResult:
    scale: ScaleEstimate | None
    dense: np.ndarray                 # (N, 6) xyz+rgb, COLMAP units
    frames_run: int
    frames_aligned: int
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# availability
# ---------------------------------------------------------------------------


def enabled() -> bool:
    return os.environ.get("LOCAISH_LEARNED", "1") != "0"


def fill_enabled() -> bool:
    """Whether the verified depth fill joins the measured cloud.

    The scale anchor is always on when the stack is; the fill is a separate
    switch because its two halves pull apart: it finds the ceiling nobody
    filmed and doubles floor coverage, and on captures whose camera solve
    already drifts it also moves the room-arrangement vote by a few degrees
    between otherwise identical runs. `LOCAISH_LEARNED_FILL=0` keeps the
    anchor and leaves the cloud exactly as stereo measured it.
    """
    return enabled() and os.environ.get("LOCAISH_LEARNED_FILL", "1") != "0"


def available() -> bool:
    """Whether the model stack is importable. Cheap; does not load weights."""
    if not enabled():
        return False
    try:
        import torch  # noqa: F401
        from moge.model import v2  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means "not here"
        return False
    return True


def device_name() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_MODEL: Any = None


def load_model(progress: Callable[[str], None] | None = None):
    """The model, loaded once per process. Weights are fetched on first use."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import torch
    from moge.model.v2 import MoGeModel

    if progress:
        progress(f"loading metric geometry model ({MODEL_ID.split('/')[-1]})")
    dev = device_name()
    model = MoGeModel.from_pretrained(MODEL_ID).to(dev).eval()
    _MODEL = model
    return model


# ---------------------------------------------------------------------------
# inference, cached per frame
# ---------------------------------------------------------------------------


def _stamp() -> dict:
    return {"model": MODEL_ID, "long_side": INFER_LONG_SIDE, "version": 1}


def infer_frame(
    model, path: Path, *, fov_x_deg: float | None = None
) -> FrameDepth:
    """Run the model on one frame. `fov_x_deg` is the calibrated horizontal
    field of view when the solve knows it; a model that can take it produces
    a depth whose metres are tied to the real focal rather than a guessed
    one, which is the difference between a scale anchor and a scale opinion."""
    import inspect

    import torch
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((INFER_LONG_SIDE, INFER_LONG_SIDE))
        arr = np.asarray(im)
    h, w = arr.shape[:2]
    x = torch.tensor(arr / 255.0, dtype=torch.float32, device=model.device).permute(2, 0, 1)
    kwargs: dict[str, Any] = {}
    try:
        params = inspect.signature(model.infer).parameters
    except (TypeError, ValueError):
        params = {}
    if fov_x_deg is not None and "fov_x" in params:
        kwargs["fov_x"] = float(fov_x_deg)
    if "use_fp16" in params and model.device.type != "cuda":
        kwargs["use_fp16"] = False
    with torch.no_grad():
        out = model.infer(x, **kwargs)
    depth = out["depth"].float().cpu().numpy().astype(np.float32)
    mask = out["mask"].cpu().numpy().astype(bool)
    Kn = out["intrinsics"].float().cpu().numpy()
    K = np.array(
        [[Kn[0, 0] * w, 0.0, Kn[0, 2] * w], [0.0, Kn[1, 1] * h, Kn[1, 2] * h], [0.0, 0.0, 1.0]]
    )
    mask &= np.isfinite(depth) & (depth > 0)
    return FrameDepth(name=path.name, depth=depth, mask=mask, K=K, size=(w, h))


def infer_frames(
    paths: list[Path],
    cache_dir: Path,
    *,
    fov_x_deg: float | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[FrameDepth]:
    """Predictions for every path, read from the cache where the stamp matches."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamp_path = cache_dir / "stamp.json"
    stamp = _stamp()
    if fov_x_deg is not None:
        stamp["fov_x_deg"] = round(float(fov_x_deg), 3)
    fresh = stamp_path.exists() and json.loads(stamp_path.read_text()) == stamp
    out: list[FrameDepth] = []
    model = None
    for n, p in enumerate(paths):
        npz = cache_dir / (Path(p).stem + ".npz")
        if fresh and npz.exists():
            try:
                d = np.load(npz)
                out.append(
                    FrameDepth(
                        name=Path(p).name,
                        depth=d["depth"].astype(np.float32),
                        mask=d["mask"].astype(bool),
                        K=d["K"].astype(np.float64),
                        size=(int(d["size"][0]), int(d["size"][1])),
                    )
                )
                continue
            except (KeyError, ValueError, OSError):
                pass
        if model is None:
            model = load_model(progress)
            if not fresh:
                for old in cache_dir.glob("*.npz"):
                    old.unlink()
                stamp_path.write_text(json.dumps(stamp))
        if progress and n % 8 == 0:
            progress(f"metric depth {n + 1}/{len(paths)}")
        fd = infer_frame(model, Path(p), fov_x_deg=fov_x_deg)
        np.savez_compressed(
            npz,
            depth=fd.depth.astype(np.float16),
            mask=fd.mask,
            K=fd.K,
            size=np.array(fd.size),
        )
        out.append(fd)
    return out


# ---------------------------------------------------------------------------
# alignment: metres -> reconstruction units, one frame at a time
# ---------------------------------------------------------------------------


def align_frame(
    fd: FrameDepth,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    frame_size: tuple[int, int],
    sparse_xyz: np.ndarray,
) -> tuple[float, float, int] | None:
    """Tie one frame's metric depth to the sparse model's unit.

    Projects every sparse point in front of the camera into the frame, reads
    the model's depth at that pixel, and takes the median ratio. The median,
    not a fit: sparse points include a few that are behind glass or on a thin
    rail the model smoothed over, and those are outliers to be ignored, not
    residuals to be minimised. Returns (units per metre, log-MAD, n) or None
    when the frame has too little to align on or disagrees with itself.
    """
    W0, H0 = frame_size
    w, h = fd.size
    E = np.asarray(extrinsic, dtype=np.float64)
    pc = (E[:, :3] @ np.asarray(sparse_xyz, dtype=np.float64).T).T + E[:, 3]
    z = pc[:, 2]
    front = z > 1e-6
    if front.sum() < ALIGN_MIN_POINTS:
        return None
    uv = (np.asarray(intrinsic, dtype=np.float64) @ pc[front].T).T
    uv = uv[:, :2] / uv[:, 2:3]
    u = np.floor(uv[:, 0] * (w / W0)).astype(int)
    v = np.floor(uv[:, 1] * (h / H0)).astype(int)
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if inb.sum() < ALIGN_MIN_POINTS:
        return None
    zc = z[front][inb]
    zm = fd.depth[v[inb], u[inb]].astype(np.float64)
    ok = fd.mask[v[inb], u[inb]] & np.isfinite(zm) & (zm > 0)
    if ok.sum() < ALIGN_MIN_POINTS:
        return None
    logs = np.log(zc[ok] / zm[ok])
    med = float(np.median(logs))
    mad = float(np.median(np.abs(logs - med)) * 1.4826)
    if not np.isfinite(med) or mad > ALIGN_MAX_LOG_MAD:
        return None
    return float(math.exp(med)), mad, int(ok.sum())


def scale_from_metric_depth(aligned: list[AlignedFrame], rejected: int = 0) -> ScaleEstimate | None:
    """The anchor: metres per reconstruction unit, from the aligned frames.

    Each frame's ratio is an independent reading; the spread between them is
    the measured part of the error bar and the model's own bias is the prior
    part, combined exactly as the other anchors combine theirs. Fewer than
    three frames is not a distribution and returns None.
    """
    if len(aligned) < 3:
        return None
    factors = np.array([1.0 / a.units_per_metre for a in aligned], dtype=np.float64)
    logs = np.log(factors)
    med = float(np.median(logs))
    spread = float(np.median(np.abs(logs - med)) * 1.4826)
    # the median's own standard error, not the per-frame scatter: thirty
    # frames that scatter 5% pin the median far better than 5%
    spread_of_median = spread * 1.2533 / math.sqrt(len(aligned))
    warnings: list[str] = []
    if spread > 0.15:
        warnings.append(
            f"the metric depth model's per-frame scale readings scatter by "
            f"+/-{np.expm1(spread):.0%}; the sweep contains surfaces it reads "
            "inconsistently (glass, mirrors, or featureless depth) and its anchor "
            "is weighted accordingly"
        )
    return ScaleEstimate(
        factor=float(math.exp(med)),
        confidence=_confidence(math.hypot(spread_of_median, MODEL_LOG_BIAS)),
        log_spread=spread_of_median,
        source="metric-depth-model",
        prior_bias=MODEL_LOG_BIAS,
        per_frame=[float(f) for f in factors],
        frames_used=len(aligned),
        frames_rejected=int(rejected),
        model=MODEL_ID,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# dense fill: predicted depth, verified across views
# ---------------------------------------------------------------------------


def _point_map(fd: FrameDepth, scale: float, shift: float) -> np.ndarray:
    """Every pixel lifted into the camera frame, (h, w, 3) in COLMAP units."""
    h, w = fd.depth.shape
    vs, us = np.mgrid[0:h, 0:w]
    d = fd.depth.astype(np.float64) * scale + shift
    fx, fy, cx, cy = fd.K[0, 0], fd.K[1, 1], fd.K[0, 2], fd.K[1, 2]
    x = (us + 0.5 - cx) / fx * d
    y = (vs + 0.5 - cy) / fy * d
    return np.stack([x, y, d], axis=-1)


def _incidence_cos(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """|cos| between each pixel's ray and the predicted surface normal there,
    the normal taken from the point map's own finite differences. Zero where
    a neighbour is invalid, so the filter drops the edge rather than guesses."""
    dpu = np.zeros_like(points)
    dpv = np.zeros_like(points)
    dpu[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dpv[1:-1, :] = points[2:, :] - points[:-2, :]
    n = np.cross(dpu, dpv)
    nn = np.linalg.norm(n, axis=-1)
    pn = np.linalg.norm(points, axis=-1)
    cos = np.abs((n * points).sum(-1)) / np.maximum(nn * pn, 1e-12)
    ok = mask.copy()
    ok[:, [0, -1]] = False
    ok[[0, -1], :] = False
    ok[:, 1:-1] &= mask[:, 2:] & mask[:, :-2]
    ok[1:-1, :] &= mask[2:, :] & mask[:-2, :]
    cos[~ok] = 0.0
    return cos


def _backproject(
    fd: FrameDepth, scale: float, shift: float, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel grid -> camera-frame points (COLMAP units), plus the pixel
    coords, keeping only pixels the model trusts and whose predicted surface
    the ray meets at a usable angle."""
    pm = _point_map(fd, scale, shift)
    cos = _incidence_cos(pm, fd.mask)
    h, w = fd.depth.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    us = us.ravel()
    vs = vs.ravel()
    ok = fd.mask[vs, us] & (cos[vs, us] >= INCIDENCE_MIN_COS) & (pm[vs, us, 2] > 1e-6)
    us, vs = us[ok], vs[ok]
    return pm[vs, us], us, vs


def _sample_depth(fd: FrameDepth, scale: float, shift: float, pts_cam: np.ndarray) -> np.ndarray:
    """Depth this frame predicts along the rays through `pts_cam` (its own
    camera frame, COLMAP units); NaN where the ray leaves the frame or the
    model declined."""
    z = pts_cam[:, 2]
    out = np.full(len(z), np.nan)
    front = z > 1e-6
    if not front.any():
        return out
    fx, fy, cx, cy = fd.K[0, 0], fd.K[1, 1], fd.K[0, 2], fd.K[1, 2]
    u = np.floor(pts_cam[front, 0] / z[front] * fx + cx).astype(int)
    v = np.floor(pts_cam[front, 1] / z[front] * fy + cy).astype(int)
    h, w = fd.depth.shape
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    idx = np.flatnonzero(front)[inb]
    uu, vv = u[inb], v[inb]
    valid = fd.mask[vv, uu]
    vals = fd.depth[vv, uu].astype(np.float64) * scale + shift
    vals[~valid] = np.nan
    out[idx] = vals
    return out


def fuse_depth_maps(
    aligned: list[AlignedFrame],
    extrinsics: np.ndarray,
    image_dir: Path | None,
    *,
    stride: int = PIXEL_STRIDE,
    rel: float = CONSISTENCY_REL,
    min_views: int = CONSISTENCY_MIN_VIEWS,
    neighbours: int = NEIGHBOURS,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Every frame's predicted surface, kept where other frames agree.

    For each aligned frame the depth map is lifted to world points; each point
    is then projected into the neighbouring aligned frames and compared with
    the depth they predict along that ray. Agreement within `rel` in at least
    `min_views` other frames keeps the point. This is the same test stereo
    fusion applies to its own depth maps -- a surface is seen again and again
    as the camera walks past, a mistake is seen once -- applied to predictions
    instead of matches. Returns (N, 6) xyz+rgb in COLMAP units.
    """
    if len(aligned) < min_views + 1:
        return np.zeros((0, 6))
    E = np.asarray(extrinsics, dtype=np.float64)
    out: list[np.ndarray] = []
    order = sorted(range(len(aligned)), key=lambda i: aligned[i].index)
    for n, i in enumerate(order):
        a = aligned[i]
        if progress and n % 8 == 0:
            progress(f"verifying predicted surface {n + 1}/{len(order)}")
        cam_pts, us, vs = _backproject(a.frame, a.units_per_metre, a.shift, stride)
        if len(cam_pts) == 0:
            continue
        Ei = E[a.index]
        world = (Ei[:, :3].T @ (cam_pts - Ei[:, 3]).T).T
        agree = np.zeros(len(world), dtype=np.int32)
        lo, hi = max(0, n - neighbours), min(len(order), n + neighbours + 1)
        for m in range(lo, hi):
            if m == n:
                continue
            b = aligned[order[m]]
            Ej = E[b.index]
            pj = (Ej[:, :3] @ world.T).T + Ej[:, 3]
            pred = _sample_depth(b.frame, b.units_per_metre, b.shift, pj)
            z = pj[:, 2]
            ok = np.isfinite(pred) & (z > 0) & (np.abs(pred - z) <= rel * z)
            agree += ok
        keep = agree >= min_views
        if not keep.any():
            continue
        rgb = _colours(image_dir, a.frame, us[keep], vs[keep])
        out.append(np.hstack([world[keep], rgb]))
    if not out:
        return np.zeros((0, 6))
    return np.concatenate(out)


def _colours(image_dir: Path | None, fd: FrameDepth, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
    if image_dir is None:
        return np.full((len(us), 3), 180.0)
    try:
        from PIL import Image

        with Image.open(Path(image_dir) / fd.name) as im:
            im = im.convert("RGB").resize(fd.size)
            arr = np.asarray(im)
        return arr[vs, us].astype(np.float64)
    except (OSError, ValueError):
        return np.full((len(us), 3), 180.0)


def fuse_voxels(pts: np.ndarray, voxel: float) -> np.ndarray:
    """One averaged point per voxel: divides the per-view noise by root-N and
    leaves spacing and noise at the same scale, which the plane fitter assumes."""
    if voxel <= 0 or len(pts) == 0:
        return pts
    key = np.floor(pts[:, :3] / voxel).astype(np.int64)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    n = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=n).astype(np.float64)
    fused = np.empty((n, pts.shape[1]))
    for c in range(pts.shape[1]):
        fused[:, c] = np.bincount(inverse, weights=pts[:, c], minlength=n) / counts
    return fused


def collapse_shell(pts: np.ndarray, units_per_m: float) -> np.ndarray:
    """Project the verified fill onto its own mid-surface.

    Moving least squares over a wide neighbourhood: each point moves onto the
    plane its 24 nearest neighbours define when that neighbourhood is a
    sheet no thicker than `COLLAPSE_PLANAR_M`, and is dropped when it stands
    further than `COLLAPSE_REJECT_M` off such a sheet. Curved and cluttered
    regions, whose neighbourhoods do not vote for a plane, are left alone.
    Lengths are metres, converted with the solved scale; (N, 6) in and out.
    """
    if len(pts) < COLLAPSE_K * 10:
        return pts
    from .reconstruct import _refine_mls

    xyz, rgb, _ = _refine_mls(
        pts[:, :3],
        pts[:, 3:6],
        planar_m=COLLAPSE_PLANAR_M * units_per_m,
        reject_m=COLLAPSE_REJECT_M * units_per_m,
        k=COLLAPSE_K,
    )
    return np.hstack([xyz, rgb])


def fill_gaps(learned: np.ndarray, stereo_xyz: np.ndarray, gap: float) -> np.ndarray:
    """Learned points farther than `gap` (same units) from every stereo point."""
    if len(learned) == 0:
        return learned
    if stereo_xyz is None or len(stereo_xyz) == 0:
        return learned
    from scipy.spatial import cKDTree

    d, _ = cKDTree(np.asarray(stereo_xyz, dtype=np.float64)).query(
        learned[:, :3], k=1, distance_upper_bound=gap, workers=-1
    )
    return learned[~np.isfinite(d)]


def local_normals(xyz: np.ndarray, k: int = COLLAPSE_K) -> np.ndarray:
    """Unit normal of each point's k-neighbourhood plane (sign arbitrary)."""
    from scipy.spatial import cKDTree

    n = len(xyz)
    out = np.zeros((n, 3))
    if n <= k:
        return out
    tree = cKDTree(xyz)
    batch = 400_000
    for i0 in range(0, n, batch):
        sl = slice(i0, min(i0 + batch, n))
        _, idx = tree.query(xyz[sl], k=k + 1, workers=-1)
        nb = xyz[idx]
        diff = nb - nb.mean(axis=1)[:, None, :]
        cov = np.einsum("bki,bkj->bij", diff, diff) / (k + 1)
        _, evecs = np.linalg.eigh(cov)
        out[sl] = evecs[:, :, 0]
    return out


def drop_standalone_walls(
    learned: np.ndarray, standalone: np.ndarray, up: np.ndarray | None
) -> tuple[np.ndarray, int]:
    """Remove stand-alone learned sheets that are not horizontal.

    `standalone` marks the rows that neither duplicated nor extended a
    measured surface. Without a gravity direction nothing can be classified
    and everything is kept. Returns the surviving rows and how many went.
    """
    if up is None or len(learned) == 0 or not standalone.any():
        return learned, 0
    u = np.asarray(up, dtype=np.float64).reshape(3)
    nu = float(np.linalg.norm(u))
    if nu < 1e-9:
        return learned, 0
    u /= nu
    normals = local_normals(learned[:, :3])
    horizontal = np.abs(normals @ u) >= STANDALONE_MAX_TILT_COS
    drop = standalone & ~horizontal
    return learned[~drop], int(drop.sum())


def conform_to_stereo(
    learned: np.ndarray, stereo_xyz: np.ndarray | None, units_per_m: float
) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """Let the measured cloud decide where the learned fill stands.

    Per learned point, by what the stereo cloud has nearby: within
    `REDUNDANT_M` of a stereo point, dropped (stereo already has it); with at
    least `CONFORM_MIN_NEIGHBOURS` stereo points inside `CONFORM_RADIUS_M`
    that form a plane (thin, and wide enough that the normal is the wall's
    and not the noise's), projected onto that plane -- or dropped if it
    stands further than `CONFORM_MAX_OFF_M` from it, since a plane is there
    and the model put the point somewhere else; otherwise kept as predicted.
    Returns the surviving (N, 6) rows, a tally of what happened, and a mask
    over the survivors marking the stand-alone ones (kept as predicted).
    """
    tally = {"redundant": 0, "conformed": 0, "kept": 0, "dropped": 0}
    if len(learned) == 0:
        return learned, tally, np.zeros(0, dtype=bool)
    if stereo_xyz is None or len(stereo_xyz) == 0:
        tally["kept"] = int(len(learned))
        return learned, tally, np.ones(len(learned), dtype=bool)
    from scipy.spatial import cKDTree

    stereo = fuse_voxels(
        np.asarray(stereo_xyz, dtype=np.float64)[:, :3], CONFORM_THIN_M * units_per_m
    )
    tree = cKDTree(stereo)
    pts = learned[:, :3].astype(np.float64)
    out = np.array(learned, dtype=np.float64, copy=True)
    keep = np.ones(len(pts), dtype=bool)
    standalone = np.zeros(len(pts), dtype=bool)
    radius = CONFORM_RADIUS_M * units_per_m
    k = CONFORM_NEIGHBOURS
    batch = 200_000
    for i0 in range(0, len(pts), batch):
        sl = slice(i0, min(i0 + batch, len(pts)))
        d, idx = tree.query(pts[sl], k=k, distance_upper_bound=radius, workers=-1)
        valid = np.isfinite(d)
        redundant = d[:, 0] < REDUNDANT_M * units_per_m
        count = valid.sum(axis=1)
        conform = (~redundant) & (count >= CONFORM_MIN_NEIGHBOURS)
        keep[sl] &= ~redundant
        standalone[sl] = ~redundant & ~conform
        tally["redundant"] += int(redundant.sum())
        tally["kept"] += int((~redundant & ~conform).sum())
        if not conform.any():
            continue
        rows = np.flatnonzero(conform)
        idx_c = np.where(valid[rows], idx[rows], 0)
        nb = stereo[idx_c]                                  # (m, k, 3)
        w = valid[rows].astype(np.float64)[:, :, None]      # (m, k, 1)
        n_valid = w.sum(axis=1)                             # (m, 1)
        centre = (nb * w).sum(axis=1) / n_valid
        diff = (nb - centre[:, None, :]) * w
        cov = np.einsum("bki,bkj->bij", diff, diff) / n_valid[:, :, None]
        evals, evecs = np.linalg.eigh(cov)
        normal = evecs[:, :, 0]
        thickness = np.sqrt(np.maximum(evals[:, 0], 0.0))
        width = np.sqrt(np.maximum(evals[:, 1], 0.0))
        dist = np.einsum("bi,bi->b", pts[sl][rows] - centre, normal)
        # A neighbourhood is a plane only if it is thin AND wide: stereo
        # points strung along a hole's edge are a line, whose "normal" is
        # whichever direction the noise happened to pick.
        planar = (thickness < CONFORM_PLANAR_M * units_per_m) & (width > 4.0 * thickness)
        close = np.abs(dist) <= CONFORM_MAX_OFF_M * units_per_m
        ok = planar & close
        bad = planar & ~close          # a plane is there and the point is not on it
        g = rows[ok] + i0
        out[g, :3] = pts[g] - dist[ok][:, None] * normal[ok]
        keep[rows[bad] + i0] = False
        standalone[rows[~planar] + i0] = True
        tally["conformed"] += int(ok.sum())
        tally["dropped"] += int(bad.sum())
        tally["kept"] += int((~planar).sum())   # no plane to conform to: stands as predicted
    return out[keep], tally, standalone[keep]


# ---------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------


def choose_frames(n_registered: int, limit: int) -> list[int]:
    """Indices spread evenly over the registered views."""
    if n_registered <= limit:
        return list(range(n_registered))
    return sorted(set(int(round(x)) for x in np.linspace(0, n_registered - 1, limit)))


def learned_geometry(
    model,
    image_dir: Path,
    workdir: Path,
    *,
    stereo_xyz: np.ndarray | None,
    up_hint: np.ndarray | None = None,
    dense: bool = True,
    progress: Callable[[str], None] | None = None,
) -> LearnedResult | None:
    """Run the model over a spread of registered frames; return the scale anchor
    and the verified dense fill. None when the stack is unavailable. `up_hint`
    is gravity in the model's frame (from the camera poses), used to tell a
    ceiling the model may fill from a wall it may not invent."""
    if not available():
        return None
    image_dir = Path(image_dir)
    workdir = Path(workdir)
    limit = int(
        os.environ.get(
            "LOCAISH_LEARNED_FRAMES",
            MAX_FRAMES_CPU if device_name() == "cpu" else MAX_FRAMES_GPU,
        )
    )
    idx = choose_frames(len(model.names), limit)
    paths = [image_dir / model.names[i] for i in idx]
    W0, H0 = model.image_size
    fov = None
    if W0 > 0 and len(model.intrinsics):
        fx = float(model.intrinsics[idx[0]][0, 0])
        if fx > 0:
            fov = float(np.degrees(2.0 * math.atan(W0 / (2.0 * fx))))
    frames = infer_frames(paths, workdir / "learned", fov_x_deg=fov, progress=progress)

    aligned: list[AlignedFrame] = []
    rejected = 0
    for i, fd in zip(idx, frames):
        got = align_frame(fd, model.extrinsics[i], model.intrinsics[i], (W0, H0), model.points)
        if got is None:
            rejected += 1
            continue
        s, mad, n = got
        aligned.append(AlignedFrame(index=i, frame=fd, units_per_metre=s, log_mad=mad, points_used=n))

    warnings: list[str] = []
    scale = scale_from_metric_depth(aligned, rejected)
    if scale is None:
        warnings.append(
            f"the metric depth model could not be aligned to the reconstruction "
            f"({len(aligned)} of {len(frames)} frames had enough sparse points in "
            "agreement), so it contributes no scale anchor"
        )

    fill = np.zeros((0, 6))
    tally: dict[str, int] | None = None
    dense = dense and fill_enabled()
    if dense and scale is not None and len(aligned) >= CONSISTENCY_MIN_VIEWS + 1:
        if progress:
            progress("fusing predicted depth across views")
        raw = fuse_depth_maps(aligned, model.extrinsics, image_dir, progress=progress)
        units_per_m = 1.0 / scale.factor
        raw = fuse_voxels(raw, FUSE_VOXEL_M * units_per_m)
        raw = collapse_shell(raw, units_per_m)
        fill, tally, standalone = conform_to_stereo(raw, stereo_xyz, units_per_m)
        fill, walls_dropped = drop_standalone_walls(fill, standalone, up_hint)
        tally["standalone_walls_dropped"] = walls_dropped
        tally["kept"] -= walls_dropped
        if len(raw):
            warnings.append(
                f"{len(fill):,} points of surface came from a metric depth model "
                f"({MODEL_ID.split('/')[-1]}), each confirmed by at least "
                f"{CONSISTENCY_MIN_VIEWS} other views to within "
                f"{CONSISTENCY_REL:.0%} of its distance: {tally['kept']:,} stand "
                f"as predicted on floor or ceiling where stereo found nothing, "
                f"{tally['conformed']:,} extend a measured surface and were moved "
                f"onto it; {tally['redundant']:,} duplicated stereo, "
                f"{tally['dropped']:,} contradicted it, and {walls_dropped:,} "
                "would have been walls with no measurement behind them -- the "
                "room's completion closes those from where the camera could see "
                "-- and were discarded"
            )

    summary = {
        "model": MODEL_ID,
        "device": device_name(),
        "frames_run": len(frames),
        "frames_aligned": len(aligned),
        "frames_rejected": rejected,
        "scale": None if scale is None else scale.to_dict(),
        "fill_points": int(len(fill)),
        "fill_enabled": bool(dense),
        "fill_tally": tally,
        "median_align_log_mad": (
            float(np.median([a.log_mad for a in aligned])) if aligned else None
        ),
        "fusion": {
            "version": FUSION_VERSION,
            "consistency_rel": CONSISTENCY_REL,
            "min_views": CONSISTENCY_MIN_VIEWS,
            "incidence_min_cos": INCIDENCE_MIN_COS,
            "collapse_planar_m": COLLAPSE_PLANAR_M,
        },
    }
    (workdir / "learned").mkdir(parents=True, exist_ok=True)
    (workdir / "learned" / "summary.json").write_text(json.dumps(summary, indent=2))
    return LearnedResult(
        scale=scale,
        dense=fill,
        frames_run=len(frames),
        frames_aligned=len(aligned),
        summary=summary,
        warnings=warnings,
    )


__all__ = [
    "AlignedFrame",
    "FrameDepth",
    "LearnedResult",
    "align_frame",
    "available",
    "choose_frames",
    "collapse_shell",
    "conform_to_stereo",
    "drop_standalone_walls",
    "enabled",
    "fill_gaps",
    "fuse_depth_maps",
    "fuse_voxels",
    "infer_frames",
    "learned_geometry",
    "scale_from_metric_depth",
]
