"""Video in, ScanImport out -- the orchestrator for the video front-end.

The sequence is fixed and each step depends on the one before it:

    frames -> structure from motion -> dense stereo -> scale -> clean -> ScanImport

Reconstruction is classical: SIFT features matched between frames, bundle
adjustment, and photometric stereo to densify -- see `colmap.py` for why that
choice is load-bearing rather than aesthetic. Scale has to come after
reconstruction because it is solved from the camera trajectory, and cleaning
has to come after scale because every threshold in it is a distance in metres
-- "drop points more than 12 cm from any neighbour" is meaningless in the
reconstruction's arbitrary unit.

The result deliberately stops at a `ScanImport`, the same object the PLY and
OBJ readers produce. Everything after that -- gravity, yaw, planes, openings,
QA -- is the existing pipeline, unchanged and unaware that the points came from
video. That is the point: a twin from a phone sweep gets audited by exactly the
same code as a twin from a LiDAR export, and gets to fail the same checks.

The intermediate artefacts (chosen frames, raw cloud, manifest, COLMAP logs)
are written to a working directory rather than a temp dir that vanishes. When a
twin comes out wrong, the first question is always whether the frames were any
good, and that question cannot be answered from a point cloud.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..formats import ScanImport, voxel_subsample_indices, write_ply
from ..types import PointCloud
from .frames import FrameSet, VideoError, VideoInfo, extract_frames, probe

# Ceiling on how many frames are decoded for matching. Classical SfM wants
# temporal density -- correspondences chain frame to frame -- but past a few
# hundred frames the matching cost grows without the room gaining coverage.
MAX_FRAMES = 300

# Sparse points kept for the final cloud must have been seen from at least this
# many views with at most this reprojection error. A two-view point with a
# three-pixel residual is a matching accident, not a surface.
SPARSE_MIN_TRACK = 3
SPARSE_MAX_ERROR_PX = 2.0

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
    fps: float | None = None,
    max_points: int = 1_500_000,
    scale_factor: float | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    keep_frames: bool = True,
    refresh: bool = False,
    extra_scales=None,
    progress=None,
) -> VideoReconstruction:
    """Reconstruct a hand-held video sweep into a metric `ScanImport`.

    The expensive half of this -- decode, matching, bundle adjustment, stereo --
    depends only on the video and the handful of options in the cache key, so
    its result is stored next to the twin and reused. That matters more than it
    sounds: the reconstruction is minutes and everything downstream is seconds,
    so without a cache, adjusting a voxel size means re-matching a few hundred
    frames to get back a cloud that was never going to change. `refresh=True`
    forces the work to happen again.
    """
    import time

    from . import colmap as colmapmod
    from . import dense as densemod
    from . import metric as metricmod

    src = Path(path)
    workdir = Path(workdir) if workdir else src.parent / f"{src.stem}.recon"
    workdir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    warnings: list[str] = []

    def _step(name):
        return _Clock(timings, name, progress)

    fps = float(fps) if fps else colmapmod.CLASSICAL_FPS
    key = _cache_key(src, fps, max_points, scale_factor, start_s, end_s)
    if not refresh and not extra_scales:
        cached = _load_cache(workdir, key, src, progress=progress)
        if cached is not None:
            return cached

    # -- frames -----------------------------------------------------------
    #
    # Dense in time, because correspondence is the binding constraint: SIFT
    # matches a frame against its neighbours, and a gap in time where the
    # camera kept moving is a break in the chain. The sharpest frame per
    # timeline slice is still chosen -- motion blur costs matches too.
    info = probe(src)
    span = info.duration_s or 0.0
    if start_s is not None or end_s is not None:
        lo = start_s or 0.0
        hi = end_s if end_s is not None else span
        span = max(hi - lo, 0.0)
    count = int(math.ceil(max(span, 1.0) * fps))
    if count > MAX_FRAMES:
        warnings.append(
            f"the sweep offers {count} frames at {fps:g} fps and {MAX_FRAMES} were "
            "kept, spread across the whole timeline; a very long walk is better "
            "reconstructed as two shorter clips"
        )
        count = MAX_FRAMES
    with _step("frames"):
        fs = extract_frames(
            src,
            workdir,
            count=count,
            candidate_fps=min(max(fps * 1.5, fps + 1.0), 30.0),
            long_side=colmapmod.CLASSICAL_LONG_SIDE,
            start_s=start_s,
            end_s=end_s,
            progress=progress,
        )
    warnings += fs.warnings
    if len(fs) < 2:
        raise VideoError(
            f"{src.name} yielded only {len(fs)} usable frame(s); a reconstruction "
            "needs at least two viewpoints of the room"
        )
    image_dir = fs.paths[0].parent

    # -- structure from motion --------------------------------------------
    with _step("reconstruct"):
        colmap_dir = workdir / "colmap"
        model_dir = colmapmod.run_sfm(
            image_dir, colmap_dir, reuse=not refresh, progress=progress
        )
        model = colmapmod.read_model(model_dir)
    warnings += model.warnings
    if len(model) < 0.8 * len(fs):
        warnings.append(
            f"only {len(model)} of {len(fs)} frames could be registered into one "
            "model; the parts of the sweep that broke the chain are simply absent "
            "from this twin -- whatever the missing frames saw, the twin does not "
            "know about"
        )

    up_hint, up_coherence, up_note = colmapmod.up_from_cameras(model.extrinsics)
    if up_note:
        warnings.append(up_note)
    centres = densemod._camera_centres(model.extrinsics)

    # -- dense stereo ------------------------------------------------------
    #
    # Three implementations of the same classical job, best available wins:
    # COLMAP's CUDA PatchMatch on a GPU host, OpenMVS's CPU patch-match where
    # the binary exists, and semi-global block matching as the floor that runs
    # anywhere. The gap between the first two and the third is large, which is
    # why OpenMVS is worth the install.
    with _step("stereo"):
        if colmapmod.supports_cuda():
            dense_pts = densemod.densify_patchmatch(
                image_dir, model_dir, colmap_dir, progress=progress
            )
            stereo = "patchmatch"
        elif densemod.openmvs_binary():
            dense_pts = densemod.densify_openmvs(
                image_dir, model_dir, colmap_dir, progress=progress
            )
            stereo = "openmvs"
        else:
            dense_pts, dense_warnings = densemod.densify_sgbm(
                model, image_dir, progress=progress
            )
            warnings += dense_warnings
            stereo = "sgbm"
            warnings.append(
                "dense stereo ran on the block-matching fallback; installing "
                "OpenMVS (DensifyPointCloud on PATH) upgrades this stage "
                "substantially on machines without CUDA"
            )

    keep_sparse = (model.errors <= SPARSE_MAX_ERROR_PX) & (
        model.track_lengths >= SPARSE_MIN_TRACK
    )
    sparse_pts = np.hstack(
        [model.points[keep_sparse], model.colors[keep_sparse].astype(np.float64)]
    )
    if len(dense_pts):
        # Stereo on a degenerate pair triangulates fog far outside the room.
        # The sparse model is bundle-adjusted and trustworthy about where the
        # room *is*, so anything the dense stage puts well outside it is a
        # matching artefact, not a discovery.
        if len(sparse_pts):
            # Percentile bounds, not min/max: even the filtered sparse set
            # keeps a few long-range triangulations, and one of them would
            # inflate the box to cover the fog it exists to exclude.
            lo = np.percentile(sparse_pts[:, :3], 1, axis=0)
            hi = np.percentile(sparse_pts[:, :3], 99, axis=0)
            margin = 0.25 * (hi - lo).max()
            ok = np.all(
                (dense_pts[:, :3] >= lo - margin) & (dense_pts[:, :3] <= hi + margin),
                axis=1,
            )
            dense_pts = dense_pts[ok]
        if len(dense_pts) > max_points * 2:
            keep = voxel_subsample_indices(dense_pts[:, :3], max_points * 2)
            dense_pts = dense_pts[keep]
    merged = (
        np.concatenate([sparse_pts, dense_pts]) if len(dense_pts) else sparse_pts
    )
    if len(merged) == 0:
        raise VideoError(
            "the reconstruction produced no points at all; the sweep carries too "
            "little texture or too little parallax to triangulate"
        )
    raw_xyz = merged[:, :3]
    raw_rgb = np.clip(merged[:, 3:6], 0, 255).astype(np.uint8)

    # How thick this reconstruction's surfaces come out, measured during
    # development against planes fitted to real captures: ~2 cm RMS for the
    # patch-match densifiers, worse for block matching. Downstream plane
    # thresholds widen to this instead of starving on millimetre spacing.
    noise_hint = 0.03 if stereo in ("patchmatch", "openmvs") else 0.045

    recon_summary = {
        "backend": "colmap",
        "stereo": stereo,
        **model.summary(),
        "dense_points": int(len(dense_pts)),
        "up_coherence": round(float(up_coherence), 3),
    }

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
            # Parallax recovers the room's shape, never its size, so the metres
            # come from evidence outside the geometry: how high the phone rode
            # above the floor, and -- on a second pass -- the height of any
            # doorway found in the room. Both are physical priors, not models.
            estimates = []
            from_cameras = metricmod.scale_from_camera_height(
                raw_xyz, centres, up_hint
            )
            if from_cameras is not None:
                estimates.append(from_cameras)
            for extra in extra_scales or []:
                estimates.append(extra)
            if estimates:
                scale = metricmod.combine_scales(estimates)
            else:
                scale = metricmod.ScaleEstimate(
                    factor=1.0,
                    confidence=0.05,
                    log_spread=math.log(2.0),
                    source="unresolved",
                    warnings=[
                        "no scale evidence survived -- the camera path gave no "
                        "usable height above the floor and no doorway anchored "
                        "it -- so the twin keeps the reconstruction's arbitrary "
                        "unit; tape-measure one length and pass --scale-factor"
                    ],
                )
        factor = scale.factor
        warnings += scale.warnings

    pts = raw_xyz * factor
    cams = centres * factor

    # -- clean ------------------------------------------------------------
    with _step("clean"):
        pts, cols, dropped = _trim_outliers(pts, raw_rgb)
    if dropped:
        warnings.append(
            f"{dropped:,} isolated points ({dropped / (len(pts) + dropped):.1%}) were "
            "removed as reconstruction floaters -- a mismatched patch triangulates "
            "into mid-air, and left in place it would be fitted as a wall"
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
        software="locaish-video/colmap",
        camera_positions=cams,
        # Gravity, measured rather than inferred: see colmap.up_from_cameras.
        up_hint=up_hint,
        # Declared, not assumed: the cloud really has been converted to metres
        # by the scale solve, so the unit inference downstream must not run its
        # plausibility priors over it a second time.
        unit_hint="m",
        noise_hint_m=noise_hint,
        warnings=list(warnings),
        raw_header={
            "video": fs.info.summary(),
            "frames_used": len(fs),
            "frames_registered": len(model),
            "noise_hint_m": noise_hint,
            "scale_factor_m_per_unit": factor,
            "scale_source": "supplied" if scale_factor is not None else "camera-path",
            "scale_confidence": None if scale is None else scale.confidence,
            "up_coherence": up_coherence,
        },
    )

    result = VideoReconstruction(
        scan=scan,
        frames=fs,
        scale=scale,
        recon_summary=recon_summary,
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


def _cache_key(src: Path, fps, max_points, scale_factor, start_s, end_s) -> dict:
    """Everything that would change the reconstructed cloud, and nothing else.

    The video is identified by size and modification time rather than a hash of
    its contents: a room sweep is hundreds of megabytes, hashing it costs more
    than it saves, and the failure mode of the cheap test -- a file edited
    within the same second at exactly the same length -- does not happen to
    video files in practice.
    """
    st = src.stat()
    return {
        # Bumped whenever the reconstruction itself changes behaviour, so a
        # cache written by an older pipeline cannot masquerade as this one's
        # output. 4: single-model SfM chain + OpenMVS-preferred dense stage.
        "version": 4,
        "source": str(src.resolve()),
        "size": st.st_size,
        "mtime": int(st.st_mtime),
        "fps": fps,
        "max_points": max_points,
        "scale_factor": scale_factor,
        "start_s": start_s,
        "end_s": end_s,
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
            software="locaish-video/colmap",
            camera_positions=cams if len(cams) else None,
            up_hint=up if len(up) == 3 else None,
            unit_hint="m",
            noise_hint_m=json.loads(str(data["header"])).get("noise_hint_m"),
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
    """A restored scale estimate: the numbers, without the solver that made them."""

    def __init__(self, d: dict):
        self._d = dict(d)
        self.factor = float(d.get("factor", 1.0))
        self.confidence = float(d.get("confidence", 0.0))
        self.warnings: list[str] = []

    def to_dict(self) -> dict:
        return dict(self._d)


def _trim_outliers(xyz: np.ndarray, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop points whose local neighbourhood is far away.

    A patch matched wrongly between two views triangulates into empty space.
    Those floaters are few but they are catastrophic for plane fitting, which
    will happily fit a wall through a cloud of them. The test is on the mean
    distance to the k nearest neighbours, cut at a robust sigma rather than a
    fixed threshold so that it adapts to how densely this particular sweep
    sampled the room.
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
