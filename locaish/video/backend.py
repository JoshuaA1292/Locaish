"""The reconstruction network: frames in, a dense point cloud and poses out.

Classical structure-from-motion (COLMAP and its descendants) solves this by
matching hand-designed features between images and bundle-adjusting the result.
It is accurate and it is the right answer when it works, but it fails on
exactly the footage a phone sweep of a room produces: blank painted walls with
no features to match, a rotation-dominant trajectory with little parallax, and
-- on any machine without an NVIDIA GPU -- no dense stereo stage at all, which
leaves a few thousand feature points where a room needs a few million.

So the backend here is a feed-forward reconstruction transformer (VGGT). It
takes all the frames at once and directly predicts, for every pixel of every
frame, a depth and a camera pose in one shared coordinate frame. No matching,
no incremental bundle adjustment, no failure mode where a blank wall produces
nothing. It runs on Apple Silicon through Metal, which is the only reason a
dense room reconstruction is possible on this machine at all.

Two consequences shape everything downstream, and both are honest limitations
rather than bugs:

*The output has no scale.* The network sees pixels; pixels do not encode
metres. `metric.solve_scale` supplies that separately.

*Attention is global across frames*, which is what makes the frames mutually
consistent -- and also means memory grows with the square of the frame count.
Twenty-odd frames is the working range on a 32 GB machine; the caller is told
what it cost rather than left to discover it by swapping.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

MODEL_ID = "facebook/VGGT-1B"

# Confidence below this quantile of a frame's own distribution is discarded.
# The network is well calibrated in the sense that its low-confidence pixels
# really are the bad ones -- window glass, specular floor, the blurred edge of
# a pan -- so a per-frame quantile removes them without needing an absolute
# threshold that would depend on the room.
DEFAULT_CONF_QUANTILE = 0.35

# Largest per-pixel depth step, as a fraction of the depth itself, that can still
# be a surface rather than an edge.
#
# Predicted depth at a silhouette does not jump -- it *ramps*, interpolating
# between the object and whatever is behind it over a few pixels. Unprojected,
# that ramp becomes a sheet of points hanging in mid-air between the two, and it
# is the single most visible artefact in a video twin: the curtains and streaks
# that drip off every furniture edge and doorframe. The network's own confidence
# does not catch them, because it is confident, and it is wrong.
#
# The threshold is set by geometry rather than taste. At 518 px across roughly a
# 60 degree field, one pixel subtends about 0.12 degrees, so a real surface seen
# at 80 degrees of grazing incidence changes depth by about 1.2% per pixel and
# one at 85 degrees by about 2.4%. A genuine silhouette jumps by tens of percent.
# Three percent therefore keeps every surface short of about 87 degrees -- past
# which a surface is edge-on and contributing nothing anyway -- and removes the
# ramps.
MAX_DEPTH_STEP_RATIO = 0.03


@dataclass
class Reconstruction:
    """A multi-frame reconstruction in an arbitrary (unscaled) unit."""

    points: np.ndarray            # (M, 3) world points, network units
    colors: np.ndarray            # (M, 3) uint8
    confidence: np.ndarray        # (M,)
    depths: np.ndarray            # (N, H, W) per-frame depth, network units
    depth_conf: np.ndarray        # (N, H, W)
    frame_rgb: np.ndarray         # (N, H, W, 3) uint8, exactly what the network saw
    frame_valid: np.ndarray       # (N, H, W) bool, False where the square was padded
    frame_box: np.ndarray         # (N, 4) int: top, left, height, width of the real pixels
    camera_centers: np.ndarray    # (N, 3) world positions of each frame
    extrinsics: np.ndarray        # (N, 3, 4) world-to-camera
    intrinsics: np.ndarray        # (N, 3, 3)
    up_hint: np.ndarray | None = None   # (3,) world direction away from the floor
    up_coherence: float = 0.0           # how much the frames agreed about it
    device: str = "cpu"
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "frames": int(len(self.depths)),
            "points": int(len(self.points)),
            "device": self.device,
            "seconds": round(self.seconds, 1),
            "up_from_cameras": self.up_hint is not None,
            "up_coherence": round(float(self.up_coherence), 3),
        }


def pick_device(prefer: str | None = None) -> str:
    """Metal if it exists, then CUDA, then CPU.

    The MPS fallback flag is set here rather than left to the user: a handful
    of the network's operators have no Metal kernel, and without the fallback
    the run dies several minutes in with an unhelpful message instead of
    quietly finishing those ops on the CPU.
    """
    import torch

    if prefer:
        return prefer
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(device: str | None = None, progress=None):
    import torch
    from vggt.models.vggt import VGGT

    device = device or pick_device()
    if progress:
        progress("load model")
    dtype = torch.float32 if device == "mps" else torch.float32
    model = VGGT.from_pretrained(MODEL_ID)
    model = model.to(device=device, dtype=dtype).eval()
    return model, device


def edge_mask(depth: np.ndarray, max_step_ratio: float = MAX_DEPTH_STEP_RATIO) -> np.ndarray:
    """True where a depth map is locally smooth enough to be a surface.

    Compares each pixel against its four neighbours and rejects it when the
    largest step exceeds `max_step_ratio` of its own depth. Relative rather than
    absolute because a 5 cm step is a silhouette at arm's length and is
    invisible on the far wall of a hall.

    The border is rejected outright: a pixel with neighbours off the edge of the
    frame cannot be tested, and the frame edge is exactly where the network's
    depth is least constrained.
    """
    d = np.asarray(depth, dtype=np.float64)
    keep = np.zeros(d.shape, dtype=bool)
    if d.shape[0] < 3 or d.shape[1] < 3:
        return keep

    core = d[1:-1, 1:-1]
    steps = np.maximum.reduce([
        np.abs(d[:-2, 1:-1] - core),
        np.abs(d[2:, 1:-1] - core),
        np.abs(d[1:-1, :-2] - core),
        np.abs(d[1:-1, 2:] - core),
    ])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = steps / np.maximum(np.abs(core), 1e-9)
    keep[1:-1, 1:-1] = np.isfinite(ratio) & (ratio <= max_step_ratio)
    return keep


def reconstruct(
    frame_paths,
    *,
    device: str | None = None,
    conf_quantile: float = DEFAULT_CONF_QUANTILE,
    max_points: int = 2_000_000,
    max_depth_step: float = MAX_DEPTH_STEP_RATIO,
    model=None,
    progress=None,
) -> Reconstruction:
    """Run the network over every frame at once and fuse the result."""
    import time

    import torch
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    paths = [str(p) for p in frame_paths]
    if len(paths) < 2:
        raise ValueError("reconstruction needs at least two frames")

    if model is None:
        model, device = load_model(device, progress=progress)
    else:
        device = device or pick_device()

    if progress:
        progress(f"reconstruct {len(paths)} frames")
    t0 = time.perf_counter()
    batch, valid, boxes = preprocess(paths)
    images = batch.to(device)
    with torch.no_grad():
        preds = model(images)
        extri, intri = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
    seconds = time.perf_counter() - t0

    depth = _np(preds["depth"])[0]           # (N, H, W, 1)
    depth_conf = _np(preds["depth_conf"])[0]  # (N, H, W)
    extrinsics = _np(extri)[0]                # (N, 3, 4)
    intrinsics = _np(intri)[0]                # (N, 3, 3)
    rgb = _np(images)                         # (N, 3, H, W) in 0..1

    world = unproject_depth_map_to_point_map(depth, extrinsics, intrinsics)  # (N,H,W,3)

    # Camera centre from a world-to-camera [R|t] is -R^T t. Downstream uses
    # these to orient surface normals outward-facing, which is what stops the
    # mesher from turning the room inside out.
    centers = np.einsum("nij,nj->ni", extrinsics[:, :, :3].transpose(0, 2, 1), -extrinsics[:, :, 3])
    up_hint, up_coherence, up_note = _up_from_cameras(extrinsics)

    pts = world.reshape(-1, 3)
    conf = depth_conf.reshape(-1)
    warn_up = [up_note] if up_note else []
    frame_rgb = (np.clip(rgb.transpose(0, 2, 3, 1), 0, 1) * 255).astype(np.uint8)
    cols = (np.clip(rgb.transpose(0, 2, 3, 1).reshape(-1, 3), 0, 1) * 255).astype(np.uint8)

    smooth = np.stack([edge_mask(depth[i, ..., 0], max_depth_step) for i in range(len(depth))])

    # The padding is white pixels the camera never saw; the network dutifully
    # predicts a depth for them, and every one of those predictions is fiction.
    finite = (
        np.isfinite(pts).all(axis=1)
        & np.isfinite(conf)
        & valid.reshape(-1)
        & smooth.reshape(-1)
    )
    warnings: list[str] = list(warn_up)
    edges = int((valid & ~smooth).sum())
    if edges:
        warnings.append(
            f"{edges:,} points ({edges / max(int(valid.sum()), 1):.0%} of the frame "
            "area) were dropped as silhouette ramps -- depth predicted across an "
            "object's edge lands in mid-air between it and the wall behind it"
        )
    pts, conf, cols = pts[finite], conf[finite], cols[finite]

    if conf.size:
        thresh = float(np.quantile(conf, conf_quantile))
        keep = conf >= thresh
        if keep.sum() < 1000:  # pathological -- keep everything rather than nothing
            keep = np.ones_like(conf, dtype=bool)
            warnings.append(
                "confidence filtering would have removed almost every point, so "
                "it was skipped; treat this twin's geometry as unverified"
            )
        pts, conf, cols = pts[keep], conf[keep], cols[keep]

    if len(pts) > max_points:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(pts), max_points, replace=False)
        sel.sort()
        pts, conf, cols = pts[sel], conf[sel], cols[sel]

    return Reconstruction(
        points=pts.astype(np.float64),
        colors=cols,
        confidence=conf.astype(np.float64),
        depths=depth[..., 0].astype(np.float64),
        depth_conf=depth_conf.astype(np.float64),
        frame_rgb=frame_rgb,
        frame_valid=valid,
        frame_box=boxes,
        camera_centers=centers.astype(np.float64),
        up_hint=up_hint,
        up_coherence=up_coherence,
        extrinsics=extrinsics.astype(np.float64),
        intrinsics=intrinsics.astype(np.float64),
        device=str(device),
        seconds=seconds,
        warnings=warnings,
    )


# The network's input resolution. Every depth map, confidence map and mask in a
# Reconstruction is at this size, and so is anything that wants to be compared
# against them pixel for pixel.
TARGET_SIZE = 518
PATCH = 14


def preprocess(paths, target: int = TARGET_SIZE) -> tuple["object", np.ndarray, np.ndarray]:
    """Letterbox each frame into the network's square input, keeping all of it.

    Written out here rather than taken from the upstream helper for one reason:
    that helper defaults to *cropping* height to reach a square, and on portrait
    phone video that throws away the top and bottom third of every frame -- which
    is to say the ceiling and the floor. For a twin whose entire job is to know
    where the floor and ceiling are, cropping them out of the input is not a
    detail. Padding keeps the full field of view at the cost of some wasted
    pixels, and the mask returned alongside says which pixels those are, so
    nothing downstream mistakes the padding for geometry.

    The third return value is where each frame's real pixels landed in the
    square, as (top, left, height, width). Anything that needs to run a *second*
    model on these frames needs that box, because the right thing to feed a
    second model is the original image -- a letterboxed photo with white bars
    down both sides is not a photograph of anything, and a network asked how far
    away it is will answer something that is true of neither the room nor the
    bars.
    """
    import torch
    from PIL import Image

    canvases, masks, boxes = [], [], []
    for path in paths:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if w >= h:
                nw = target
                nh = max(PATCH, round(h * (nw / w) / PATCH) * PATCH)
            else:
                nh = target
                nw = max(PATCH, round(w * (nh / h) / PATCH) * PATCH)
            arr = np.asarray(im.resize((nw, nh), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0

        canvas = np.ones((target, target, 3), dtype=np.float32)
        mask = np.zeros((target, target), dtype=bool)
        top, left = (target - nh) // 2, (target - nw) // 2
        canvas[top : top + nh, left : left + nw] = arr
        mask[top : top + nh, left : left + nw] = True
        canvases.append(canvas)
        masks.append(mask)
        boxes.append((top, left, nh, nw))

    batch = torch.from_numpy(np.stack(canvases)).permute(0, 3, 1, 2).contiguous()
    return batch, np.stack(masks), np.array(boxes, dtype=int)


# Below this agreement between frames, the operator was not holding the phone
# in any consistent orientation -- filming straight down at the floor, or
# rotating through portrait and landscape -- and the median up direction stops
# meaning anything.
UP_COHERENCE_FLOOR = 0.75


def _up_from_cameras(extrinsics: np.ndarray) -> tuple[np.ndarray | None, float, str | None]:
    """Recover which way is up from how the phone was held.

    Nothing in the reconstruction itself knows about gravity -- the network sees
    pixels, and a room photographed upside down reconstructs perfectly happily
    upside down. But the *poses* carry the answer for free: a person filming a
    room holds the phone roughly upright, so the camera's own down axis is
    roughly gravity, in every frame, and the frames were solved in a single
    shared world frame. Averaging the per-frame down axes therefore recovers
    gravity directly, without a single assumption about furniture, ceilings or
    what a room looks like.

    The mean resultant length of those directions is returned alongside, and it
    is the whole safety net: it is near 1 when the phone was held consistently
    and collapses toward 0 when it was not, which is exactly when this estimate
    deserves to be ignored. Below the floor the hint is withheld rather than
    downweighted, because a confidently wrong up is worse for the pipeline than
    no up at all.

    With OpenCV extrinsics `x_cam = R x_world + t`, a camera-frame direction d
    maps back to the world as `R^T d`; camera down is +Y, so world down is the
    second row of R.
    """
    if extrinsics is None or len(extrinsics) == 0:
        return None, 0.0, None
    downs = np.asarray(extrinsics, dtype=np.float64)[:, 1, :3]
    norms = np.linalg.norm(downs, axis=1, keepdims=True)
    good = norms[:, 0] > 1e-9
    if good.sum() < 2:
        return None, 0.0, None
    ups = -downs[good] / norms[good]
    mean = ups.mean(axis=0)
    coherence = float(np.linalg.norm(mean))
    if coherence < UP_COHERENCE_FLOOR:
        return (
            None,
            coherence,
            f"the camera was not held in a consistent orientation (frames agree "
            f"on which way is up only {coherence:.0%}), so gravity was left to be "
            "recovered from the room's own geometry",
        )
    return mean / coherence, coherence, None


def reconstruct_chunked(
    frame_paths,
    *,
    chunk: int | None = None,
    overlap: int | None = None,
    device: str | None = None,
    conf_quantile: float = DEFAULT_CONF_QUANTILE,
    max_points: int = 2_000_000,
    max_depth_step: float = MAX_DEPTH_STEP_RATIO,
    progress=None,
) -> Reconstruction:
    """Reconstruct more frames than fit at once, and register the pieces.

    Falls straight through to `reconstruct` when the frames fit in one window,
    so the single-window path is not a special case of anything -- it is the
    same code it always was.

    The windows are solved in order and each is carried onto the accumulated
    frame by the similarity its shared poses imply, which means error
    accumulates along the chain rather than being distributed over it. That is
    the honest limitation of doing this without a bundle adjustment, and the
    per-join residual is recorded so the size of it is visible rather than
    inferred from a twin that looks slightly bent.
    """
    from . import chunks as chunkmod

    paths = [str(p) for p in frame_paths]
    chunk = chunkmod.DEFAULT_CHUNK if chunk is None else int(chunk)
    overlap = chunkmod.DEFAULT_OVERLAP if overlap is None else int(overlap)

    spans = chunkmod.windows(len(paths), chunk, overlap)
    if len(spans) == 1:
        return reconstruct(
            paths,
            device=device,
            conf_quantile=conf_quantile,
            max_points=max_points,
            max_depth_step=max_depth_step,
            progress=progress,
        )

    model, device = load_model(device, progress=progress)
    warnings: list[str] = []
    seconds = 0.0

    merged_points: list[np.ndarray] = []
    merged_colors: list[np.ndarray] = []
    merged_conf: list[np.ndarray] = []
    # Per-frame arrays are kept for the first window that saw each frame; a
    # second opinion on the same frame adds nothing but memory.
    seen: dict[int, None] = {}
    depths: list[np.ndarray] = []
    depth_conf: list[np.ndarray] = []
    frame_rgb: list[np.ndarray] = []
    frame_valid: list[np.ndarray] = []
    frame_box: list[np.ndarray] = []
    extrinsics: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    frame_order: list[int] = []

    previous: tuple[tuple[int, int], Reconstruction] | None = None
    residuals: list[float] = []


    for w, (lo, hi) in enumerate(spans):
        if progress:
            progress(f"reconstruct window {w + 1}/{len(spans)} (frames {lo}-{hi - 1})")
        piece = reconstruct(
            paths[lo:hi],
            device=device,
            conf_quantile=conf_quantile,
            max_points=max_points,
            max_depth_step=max_depth_step,
            model=model,
            progress=None,
        )
        seconds += piece.seconds
        warnings += piece.warnings

        if previous is None:
            transform = chunkmod.Similarity(1.0, np.eye(3), np.zeros(3), shared=0)
        else:
            (plo, phi), prev = previous
            shared = sorted(set(range(lo, hi)) & set(range(plo, phi)))
            if len(shared) < chunkmod.MIN_OVERLAP:
                warnings.append(
                    f"window {w + 1} shares only {len(shared)} frames with the one "
                    "before it, too few to register against; the rest of the sweep "
                    "was dropped rather than joined on at a guessed position"
                )
                break
            # `prev` is already expressed in the accumulated frame, because it
            # was transformed on the iteration that admitted it.
            step = chunkmod.register(
                prev.extrinsics[[i - plo for i in shared]],
                piece.extrinsics[[i - lo for i in shared]],
            )
            residuals.append(step.residual_m)
            transform = step

        if transform.shared:
            piece.points = transform.apply(piece.points)
            piece.camera_centers = transform.apply(piece.camera_centers)
            piece.extrinsics = transform.apply_extrinsics(piece.extrinsics)
            piece.depths = piece.depths * transform.scale
            if piece.up_hint is not None:
                piece.up_hint = transform.apply_direction(piece.up_hint)

        merged_points.append(piece.points)
        merged_colors.append(piece.colors)
        merged_conf.append(piece.confidence)
        for i in range(lo, hi):
            if i in seen:
                continue
            seen[i] = None
            k = i - lo
            depths.append(piece.depths[k])
            depth_conf.append(piece.depth_conf[k])
            frame_rgb.append(piece.frame_rgb[k])
            frame_valid.append(piece.frame_valid[k])
            frame_box.append(piece.frame_box[k])
            extrinsics.append(piece.extrinsics[k])
            intrinsics.append(piece.intrinsics[k])
            frame_order.append(i)

        previous = ((lo, hi), piece)

    order = np.argsort(frame_order)
    extr = np.stack(extrinsics)[order]
    up_hint, up_coherence, up_note = _up_from_cameras(extr)
    if up_note:
        warnings.append(up_note)

    points = np.concatenate(merged_points)
    colors = np.concatenate(merged_colors)
    conf = np.concatenate(merged_conf)
    if len(points) > max_points:
        rng = np.random.default_rng(0)
        sel = np.sort(rng.choice(len(points), max_points, replace=False))
        points, colors, conf = points[sel], colors[sel], conf[sel]

    if residuals:
        worst = max(residuals)
        warnings.append(
            f"the sweep was reconstructed in {len(spans)} overlapping windows and "
            f"joined through their shared views; the worst join left the shared "
            f"cameras {worst * 100:.1f} cm apart, and that error accumulates along "
            "the sweep rather than being spread over it"
        )

    return Reconstruction(
        points=points,
        colors=colors,
        confidence=conf,
        depths=np.stack(depths)[order],
        depth_conf=np.stack(depth_conf)[order],
        frame_rgb=np.stack(frame_rgb)[order],
        frame_valid=np.stack(frame_valid)[order],
        frame_box=np.stack(frame_box)[order],
        camera_centers=chunkmod.camera_centres(extr),
        extrinsics=extr,
        intrinsics=np.stack(intrinsics)[order],
        up_hint=up_hint,
        up_coherence=up_coherence,
        device=str(device),
        seconds=seconds,
        warnings=warnings,
    )


def _np(t) -> np.ndarray:
    import torch

    if isinstance(t, torch.Tensor):
        return t.detach().float().cpu().numpy()
    return np.asarray(t)
