"""Video in, ScanImport out -- the orchestrator for the video front-end.

The sequence is fixed and each step depends on the one before it:

    frames -> reconstruction -> scale -> clean -> subsample -> ScanImport

Scale has to come after reconstruction because it is solved by comparing the
reconstruction's own depth against a metric prior, and cleaning has to come
after scale because every threshold in it is a distance in metres -- "drop
points more than 12 cm from any neighbour" is meaningless in the network's
arbitrary unit.

The result deliberately stops at a `ScanImport`, the same object the PLY and
OBJ readers produce. Everything after that -- gravity, yaw, planes, openings,
QA -- is the existing pipeline, unchanged and unaware that the points came from
video. That is the point: a twin from a phone sweep gets audited by exactly the
same code as a twin from a LiDAR export, and gets to fail the same checks.

The intermediate artefacts (chosen frames, raw cloud, manifest) are written to
a working directory rather than a temp dir that vanishes. When a twin comes out
wrong, the first question is always whether the frames were any good, and that
question cannot be answered from a point cloud.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..formats import ScanImport, voxel_subsample_indices, write_ply
from ..types import PointCloud
from .frames import FrameSet, VideoError, VideoInfo, extract_frames

# Default number of frames handed to the network. Global attention across
# frames makes memory grow quadratically, and on a 32 GB Apple Silicon machine
# this is comfortably inside the envelope while still covering a room sweep.
DEFAULT_FRAMES = 24

# Raising `frames` past a window's worth is what engages the chunked path: the
# frame budget stops being a hardware limit and becomes a choice about how much
# of the room to reconstruct.
DEFAULT_FRAMES_CHUNKED = 24

# Neighbourhood used by the outlier trim. Small enough to be cheap on a
# multi-million point cloud, large enough that a genuine thin surface (a
# curtain, a lampshade) survives it.
OUTLIER_K = 8
OUTLIER_SIGMA = 3.0


@dataclass
class VideoReconstruction:
    """A reconstructed sweep, with every intermediate kept for inspection."""

    scan: ScanImport
    frames: FrameSet
    scale: Any
    recon_summary: dict
    workdir: Path
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def manifest(self) -> dict:
        return {
            "video": self.frames.info.summary(),
            "frames": {
                "used": len(self.frames),
                "candidates": self.frames.candidates_considered,
                "timestamps_s": [round(t, 3) for t in self.frames.timestamps],
            },
            "reconstruction": self.recon_summary,
            "scale": self.scale.to_dict() if self.scale is not None else None,
            "points": len(self.scan.points),
            "timings_s": {k: round(v, 2) for k, v in self.timings.items()},
            "warnings": self.warnings,
        }


def reconstruct_video(
    path: str | Path,
    *,
    workdir: str | Path | None = None,
    frames: int = DEFAULT_FRAMES,
    device: str | None = None,
    max_points: int = 1_500_000,
    scale_factor: float | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    conf_quantile: float | None = None,
    chunk: int | None = None,
    overlap: int | None = None,
    keep_frames: bool = True,
    refresh: bool = False,
    extra_scales=None,
    progress=None,
) -> VideoReconstruction:
    """Reconstruct a hand-held video sweep into a metric `ScanImport`.

    The expensive half of this -- decode, network, scale -- depends only on the
    video and the handful of options listed in the cache key, so its result is
    stored next to the twin and reused. That matters more than it sounds: the
    reconstruction is minutes and everything downstream is seconds, so without
    a cache, adjusting a voxel size means re-running a billion-parameter network
    to get back a cloud that was never going to change. `refresh=True` forces
    the work to happen again.
    """
    import time

    from . import backend as backendmod
    from . import metric as metricmod

    src = Path(path)
    workdir = Path(workdir) if workdir else src.parent / f"{src.stem}.recon"
    workdir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    warnings: list[str] = []

    def _step(name):
        return _Clock(timings, name, progress)

    key = _cache_key(
        src, frames, max_points, scale_factor, start_s, end_s, conf_quantile, chunk, overlap
    )
    # Extra estimators change the scale but nothing the network computes, so
    # they are kept out of the cache key: a second pass carrying a door anchor
    # must still reuse the first pass's reconstruction, or the anchor costs a
    # full re-run of the network to change one number.
    if not refresh and not extra_scales:
        cached = _load_cache(workdir, key, src, progress=progress)
        if cached is not None:
            return cached

    # -- frames -----------------------------------------------------------
    with _step("frames"):
        fs = extract_frames(
            src, workdir, count=frames, start_s=start_s, end_s=end_s, progress=progress
        )
    warnings += fs.warnings
    if len(fs) < 2:
        raise VideoError(
            f"{src.name} yielded only {len(fs)} usable frame(s); a reconstruction "
            "needs at least two viewpoints of the room"
        )

    # -- reconstruction ---------------------------------------------------
    with _step("reconstruct"):
        kw = {} if conf_quantile is None else {"conf_quantile": conf_quantile}
        recon = backendmod.reconstruct_chunked(
            fs.paths,
            chunk=chunk,
            overlap=overlap,
            device=device,
            max_points=max_points * 2,
            progress=progress,
            **kw,
        )
    warnings += recon.warnings

    # -- scale ------------------------------------------------------------
    scale = None
    if scale_factor is not None:
        factor = float(scale_factor)
        warnings.append(
            f"scale was supplied as {factor:.6g} m per reconstruction unit rather "
            "than solved, so the twin is exactly as accurate as that number"
        )
    else:
        with _step("scale"):
            # Two estimators, deliberately sharing no evidence: one reads the
            # room's appearance, the other reads where the operator's hand was.
            # Either alone can be confidently wrong; only together do they say
            # anything about accuracy rather than repeatability.
            estimates = []
            from_cameras = metricmod.scale_from_camera_height(
                recon.points, recon.camera_centers, recon.up_hint
            )
            if from_cameras is not None:
                estimates.append(from_cameras)
            estimates.append(
                metricmod.solve_scale(
                    fs.paths,
                    recon.depths,
                    recon.depth_conf,
                    boxes=recon.frame_box,
                    progress=progress,
                )
            )
            for extra in extra_scales or []:
                estimates.append(extra)
            scale = metricmod.combine_scales(estimates)
        factor = scale.factor
        warnings += scale.warnings

    pts = recon.points * factor
    cams = recon.camera_centers * factor

    # -- clean ------------------------------------------------------------
    with _step("clean"):
        pts, cols, dropped = _trim_outliers(pts, recon.colors)
    if dropped:
        warnings.append(
            f"{dropped:,} isolated points ({dropped / (len(pts) + dropped):.1%}) were "
            "removed as reconstruction floaters -- depth predicted at an object "
            "silhouette lands in mid-air, and left in place it would be fitted as "
            "a wall"
        )

    # -- subsample --------------------------------------------------------
    if len(pts) > max_points:
        with _step("subsample"):
            keep = voxel_subsample_indices(pts, max_points)
        pts, cols = pts[keep], cols[keep]

    cloud = PointCloud(xyz=pts, rgb=cols)

    # The raw cloud is written before any of the pipeline's own processing so
    # that a bad twin can be replayed from exactly this geometry.
    with _step("write"):
        ply_path = write_ply(workdir / "cloud.ply", points=cloud)

    scan = ScanImport(
        points=cloud,
        mesh=None,
        source_path=src,
        source_format="video",
        software="locaish-video/vggt",
        camera_positions=cams,
        # Gravity, measured rather than inferred: see backend._up_from_cameras.
        up_hint=recon.up_hint,
        # Declared, not assumed: the cloud really has been converted to metres
        # by the scale solve, so the unit inference downstream must not run its
        # plausibility priors over it a second time.
        unit_hint="m",
        warnings=list(warnings),
        raw_header={
            "video": fs.info.summary(),
            "frames_used": len(fs),
            "scale_factor_m_per_unit": factor,
            "scale_source": "supplied" if scale_factor is not None else "metric-depth",
            "scale_confidence": None if scale is None else scale.confidence,
            "device": recon.device,
            "up_coherence": recon.up_coherence,
        },
    )

    result = VideoReconstruction(
        scan=scan,
        frames=fs,
        scale=scale,
        recon_summary=recon.summary(),
        workdir=workdir,
        timings=timings,
        warnings=warnings,
    )
    (workdir / "manifest.json").write_text(json.dumps(result.manifest(), indent=2))
    _save_cache(workdir, key, result)
    if not keep_frames:
        import shutil

        shutil.rmtree(workdir / "frames", ignore_errors=True)
    if progress:
        progress(f"wrote {ply_path.name} ({len(cloud):,} points)")
    return result


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


CACHE_NAME = "reconstruction.npz"


def _cache_key(
    src: Path, frames, max_points, scale_factor, start_s, end_s, conf_quantile,
    chunk=None, overlap=None,
) -> dict:
    """Everything that would change the reconstructed cloud, and nothing else.

    The video is identified by size and modification time rather than a hash of
    its contents: a room sweep is hundreds of megabytes, hashing it costs more
    than it saves, and the failure mode of the cheap test -- a file edited
    within the same second at exactly the same length -- does not happen to
    video files in practice.
    """
    st = src.stat()
    return {
        "version": 2,
        "source": str(src.resolve()),
        "size": st.st_size,
        "mtime": int(st.st_mtime),
        "frames": frames,
        "max_points": max_points,
        "scale_factor": scale_factor,
        "start_s": start_s,
        "end_s": end_s,
        "conf_quantile": conf_quantile,
        "chunk": chunk,
        "overlap": overlap,
    }


def _save_cache(workdir: Path, key: dict, result: "VideoReconstruction") -> None:
    scan = result.scan
    np.savez_compressed(
        workdir / CACHE_NAME,
        key=json.dumps(key),
        xyz=scan.points.xyz.astype(np.float32),
        rgb=(scan.points.rgb if scan.points.rgb is not None else np.zeros((0, 3), np.uint8)),
        cams=(scan.camera_positions if scan.camera_positions is not None else np.zeros((0, 3))),
        up_hint=(scan.up_hint if scan.up_hint is not None else np.zeros(0)),
        header=json.dumps(scan.raw_header),
        warnings=json.dumps(result.warnings),
        recon_summary=json.dumps(result.recon_summary),
        scale=json.dumps(result.scale.to_dict() if result.scale is not None else None),
        timings=json.dumps(result.timings),
        frame_paths=json.dumps([str(p) for p in result.frames.paths]),
        frame_stamps=json.dumps(result.frames.timestamps),
        frame_sharpness=json.dumps(result.frames.sharpness),
        frame_info=json.dumps(result.frames.info.summary()),
        frame_candidates=result.frames.candidates_considered,
    )


def _load_cache(workdir: Path, key: dict, src: Path, progress=None) -> "VideoReconstruction | None":
    path = workdir / CACHE_NAME
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        if json.loads(str(data["key"])) != key:
            return None
        rgb = data["rgb"]
        cams = data["cams"]
        up = data["up_hint"]
        info = json.loads(str(data["frame_info"]))
        scale_dict = json.loads(str(data["scale"]))
        scan = ScanImport(
            points=PointCloud(
                xyz=data["xyz"].astype(np.float64),
                rgb=rgb if len(rgb) else None,
            ),
            mesh=None,
            source_path=src,
            source_format="video",
            software="locaish-video/vggt",
            camera_positions=cams if len(cams) else None,
            up_hint=up if len(up) == 3 else None,
            unit_hint="m",
            warnings=json.loads(str(data["warnings"])),
            raw_header=json.loads(str(data["header"])),
        )
        fs = FrameSet(
            paths=[Path(p) for p in json.loads(str(data["frame_paths"]))],
            timestamps=json.loads(str(data["frame_stamps"])),
            sharpness=json.loads(str(data["frame_sharpness"])),
            info=_info_from_summary(src, info),
            candidates_considered=int(data["frame_candidates"]),
            warnings=[],
        )
    except (KeyError, ValueError, OSError):
        # A cache that cannot be read is not an error -- it is a cache miss.
        return None

    if progress:
        progress(f"reusing cached reconstruction ({len(scan.points):,} points)")
    return VideoReconstruction(
        scan=scan,
        frames=fs,
        scale=_ScaleView(scale_dict) if scale_dict else None,
        recon_summary=json.loads(str(data["recon_summary"])),
        workdir=workdir,
        timings=json.loads(str(data["timings"])),
        warnings=list(scan.warnings),
    )


def _info_from_summary(src: Path, s: dict) -> VideoInfo:
    w, _, h = str(s.get("resolution", "0x0")).partition("x")
    return VideoInfo(
        path=src,
        duration_s=float(s.get("duration_s") or 0.0),
        fps=float(s.get("fps") or 0.0),
        width=int(w or 0),
        height=int(h or 0),
        rotation=float(s.get("rotation_deg") or 0.0),
        codec=str(s.get("codec") or "unknown"),
        frame_count=int(s.get("frames") or 0),
    )


class _ScaleView:
    """A restored scale estimate: the numbers, without the model that made them."""

    def __init__(self, d: dict):
        self._d = dict(d)
        self.factor = float(d.get("factor", 1.0))
        self.confidence = float(d.get("confidence", 0.0))
        self.warnings: list[str] = []

    def to_dict(self) -> dict:
        return dict(self._d)


def _trim_outliers(xyz: np.ndarray, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop points whose local neighbourhood is far away.

    Predicted depth at a silhouette edge interpolates between the object and
    whatever is behind it, so it lands in empty space. Those floaters are few
    but they are catastrophic for plane fitting, which will happily fit a wall
    through a cloud of them. The test is on the mean distance to the k nearest
    neighbours, cut at a robust sigma rather than a fixed threshold so that it
    adapts to how densely this particular sweep sampled the room.
    """
    from scipy.spatial import cKDTree

    n = len(xyz)
    if n < OUTLIER_K * 4:
        return xyz, rgb, 0
    tree = cKDTree(xyz)
    dist, _ = tree.query(xyz, k=OUTLIER_K + 1, workers=-1)
    mean_d = dist[:, 1:].mean(axis=1)
    med = float(np.median(mean_d))
    mad = float(np.median(np.abs(mean_d - med))) * 1.4826
    if not np.isfinite(mad) or mad <= 0:
        return xyz, rgb, 0
    keep = mean_d <= med + OUTLIER_SIGMA * mad
    return xyz[keep], rgb[keep], int((~keep).sum())


class _Clock:
    def __init__(self, store, name, progress):
        self.store, self.name, self.progress = store, name, progress

    def __enter__(self):
        import time

        if self.progress:
            self.progress(self.name)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        import time

        self.store[self.name] = time.perf_counter() - self.t0
        return False
