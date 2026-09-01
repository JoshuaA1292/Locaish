"""The ingest pipeline: an export file goes in, a Twin comes out.

This is the orchestrator. Every hard decision lives in a module underneath it --
what unit the file is in, which way is up, where the walls are, whether the scan
is good enough to trust. What happens here is sequencing and bookkeeping, plus
one editorial job the modules can't do for themselves: deciding what to record
about how the twin was made, so that a number in the final report can always be
traced back to the step that produced it.

The order is not arbitrary and should not be rearranged casually:

    read -> normals -> planes -> scale -> gravity -> yaw -> origin
         -> [grid, mesh, capture bounds, structure] -> QA

Scale has to be settled before gravity, because the plausibility priors that
identify the unit are metric ("a ceiling is 2.4 m, not 240"). Gravity has to be
settled before structure, because "floor" means "the horizontal plane at the
bottom" and neither word means anything until we know which way is down. QA runs
last because it is the only step allowed to look at everything at once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..types import (
    CaptureBounds,
    Georeference,
    Mesh,
    PointCloud,
    QAReport,
    Twin,
)


@dataclass
class IngestOptions:
    """Everything the caller can steer, with defaults that suit an interior.

    `max_points` exists because a Polycam room export can be 20M points and the
    QA pass is O(n log n) on a kd-tree; nothing in Phase 1 gets more accurate
    above a couple of million, and the subsample is spatially uniform so it
    costs coverage rather than geometry.

    `skip_mesh_reconstruction` is for the common case where the import already
    carried faces. `force_mesh` re-derives them anyway, which is occasionally
    what you want when the scanner's own mesh is decimated to uselessness.
    """

    name: str | None = None
    max_points: int | None = 3_000_000
    voxel_xy: float = 0.05
    voxel_z: float = 0.05
    mesh_voxel: float = 0.04
    unit_hint: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    heading_deg: float | None = None
    elevation_m: float = 0.0
    skip_mesh_reconstruction: bool = False
    force_mesh: bool = False
    skip_openings: bool = False
    # Completing the mesh against the swept volume. On by default, but it only
    # ever engages when the import carried camera poses -- see `geom.infill` for
    # why an unposed cloud cannot tell a hole from an open door.
    fill_holes: bool = True
    fill_radius_m: float | None = None
    # Believe the capture device about which way is up: forces the gravity
    # axis onto the camera-derived hint. Off by default -- see align.canonicalize.
    trust_up_hint: bool = False
    # Resampling points onto detected wall planes where camera rays prove the
    # wall was observed and unbroken -- see `geom.planefill`. The added points
    # are tagged `inferred` and excluded from every measurement; the flag
    # exists for callers who want the raw capture and nothing else.
    fill_planes: bool = True
    seed: int = 0
    progress: Callable[[str], None] | None = None

    # -- video front-end ---------------------------------------------------
    # Only consulted when the source is a video. Kept on the same options
    # object rather than a parallel one because from the caller's point of view
    # this is still "ingest a scan"; the source just happens to be footage.
    video_fps: float | None = None
    video_scale_factor: float | None = None
    video_workdir: Path | None = None
    video_start_s: float | None = None
    video_end_s: float | None = None
    video_refresh: bool = False
    # Extra scale estimators handed to the video front-end. Populated by the
    # second pass below, never by a caller.
    video_extra_scales: tuple = ()
    # Whether a doorway found in the first pass may be used to re-anchor scale.
    video_door_anchor: bool = True


@dataclass
class IngestResult:
    """The twin plus the receipts: timings, warnings, and what each step decided."""

    twin: Twin
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    steps: dict[str, Any] = field(default_factory=dict)

    @property
    def total_seconds(self) -> float:
        return float(sum(self.timings.values()))


class _Timer:
    def __init__(self, result: IngestResult, name: str, progress) -> None:
        self.result, self.name, self.progress = result, name, progress

    def __enter__(self):
        if self.progress:
            self.progress(self.name)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.result.timings[self.name] = time.perf_counter() - self.t0
        return False


def ingest(
    path: str | Path,
    options: IngestOptions | None = None,
    *,
    _door_pass: int = 0,
) -> IngestResult:
    """Turn a scanner export into a canonical, QA'd Twin.

    Raises `IngestError` only for things that make a twin impossible (an
    unreadable file, an empty cloud). Everything else -- an unknown unit, a
    missing ceiling, a scan too sparse to trust -- comes back as a twin whose
    QA report says so, because a location manager is better served by a twin
    labelled "do not trust the ceiling height" than by a stack trace.
    """
    # Imports are local so that a partially built tree still lets you import
    # locaish.types; during development that mattered more than it looks.
    from ..formats import read_scan
    from ..geom import align, grid as gridmod, hull, mesher, normals as normmod, planes as planemod
    from . import qa as qamod, structure as structmod

    opts = options or IngestOptions()
    src = Path(path)
    result = IngestResult(twin=None)  # type: ignore[arg-type]
    prog = opts.progress

    def step(name: str) -> _Timer:
        return _Timer(result, name, prog)

    # -- read -------------------------------------------------------------
    #
    # A video is not a scan file, so it cannot go through the format readers --
    # it has to be reconstructed into a cloud first. That reconstruction is the
    # only branch in this function: past this point nothing downstream knows or
    # cares whether the points came from a LiDAR export or from footage, which
    # is deliberate. A twin built from video earns its numbers by passing the
    # same QA as a twin built from a laser scanner, not by being special-cased.
    frames_dir: Path | None = None
    extra_xyz: np.ndarray | None = None
    extra_rgb: np.ndarray | None = None
    with step("read"):
        if is_video(src):
            from ..video import reconstruct_video

            video = reconstruct_video(
                src,
                workdir=opts.video_workdir,
                fps=opts.video_fps,
                max_points=opts.max_points or 3_000_000,
                scale_factor=opts.video_scale_factor,
                start_s=opts.video_start_s,
                end_s=opts.video_end_s,
                refresh=opts.video_refresh,
                extra_scales=list(opts.video_extra_scales),
                progress=prog,
            )
            scan = video.scan
            result.steps["video"] = video.manifest()
            frames_dir = Path(video.workdir) / "frames"
            # The splat view layer rides outside the scan so every fitter
            # below sees only the measured cloud; appended after the solve.
            extra_xyz = video.extra_xyz
            extra_rgb = video.extra_rgb
        else:
            scan = read_scan(src, max_points=opts.max_points)
    if len(scan.points) == 0:
        raise IngestError(f"{src.name} contained no points")
    result.warnings += list(scan.warnings)
    result.steps["read"] = {
        "format": scan.source_format,
        "software": scan.software,
        "points": len(scan.points),
        "had_mesh": scan.mesh is not None,
        "had_poses": scan.camera_positions is not None,
        "had_up_hint": scan.up_hint is not None,
    }

    cloud = scan.points
    mesh = scan.mesh

    # -- normals and planes, on the raw cloud -----------------------------
    #
    # Both are scale-invariant in direction, so running them before we know the
    # unit is safe; only the distance thresholds care, and those get scaled.
    with step("normals"):
        raw_normals = normmod.estimate_normals(cloud)
        raw_normals = normmod.orient_normals(
            raw_normals, cloud.xyz, viewpoints=scan.camera_positions
        )
        cloud = PointCloud(xyz=cloud.xyz, rgb=cloud.rgb, normals=raw_normals)

    with step("planes"):
        # The inlier threshold cannot be a fixed 3 cm here, because at this
        # point we do not yet know what a centimetre is. A file exported in
        # centimetres would get a threshold a hundred times too tight, RANSAC
        # would find no walls at all, and the scale inference that depends on
        # those walls would then have nothing to reason about -- so the unit
        # error and the geometry error would conspire to produce a twin that is
        # silently the wrong size and the wrong way round. Scaling the
        # threshold to the cloud's own point spacing makes it unit-free, and is
        # the better rule regardless: what counts as "on the wall" depends on
        # how finely the wall was sampled, not on the choice of unit.
        spacing = _median_spacing(cloud.xyz, seed=opts.seed)
        # The threshold has to cover whichever is larger: how far apart the
        # points are, or how thick the producer says its surfaces come out.
        # Stereo reconstruction packs millimetre spacing onto centimetre-noisy
        # surfaces, and the spacing rule alone would find fragments of every
        # plane and the whole of none.
        plane_thresh = float(np.clip(max(spacing, scan.noise_hint_m or 0.0), 1e-9, None))
        raw_planes = planemod.detect_planes(
            cloud, normals=raw_normals, distance_thresh=plane_thresh, seed=opts.seed
        )
    result.steps["planes"] = {
        "count": len(raw_planes),
        "median_spacing_source_units": spacing,
        "distance_thresh_source_units": plane_thresh,
    }

    # -- canonicalise ------------------------------------------------------
    with step("canonicalize"):
        canon = align.canonicalize(
            cloud,
            mesh=mesh,
            planes=raw_planes,
            normals=raw_normals,
            camera_positions=scan.camera_positions,
            up_hint=scan.up_hint,
            trust_up_hint=opts.trust_up_hint,
            # The video front-end measures how consistently the device was
            # held (its `up_coherence`); the sign vote scales the hint's
            # authority with it -- see align._sign_hint_weight.
            up_hint_coherence=(scan.raw_header or {}).get("up_coherence"),
            unit_hint=opts.unit_hint or scan.unit_hint,
            unit_hint_confidence=_hint_confidence(scan, opts),
            unit_hint_evidence=_hint_evidence(scan),
            seed=opts.seed,
        )
        cloud, mesh = align.apply(cloud, mesh, canon)
        cams = None
        if scan.camera_positions is not None:
            m = canon.transform
            cams = scan.camera_positions @ m[:3, :3].T + m[:3, 3]
        if extra_xyz is not None:
            m = canon.transform
            extra_xyz = (
                extra_xyz.astype(np.float64) @ m[:3, :3].T + m[:3, 3]
            ).astype(np.float32)
    result.warnings += list(canon.warnings)
    result.steps["canonicalize"] = {
        "unit": canon.scale.unit,
        "scale_factor": canon.scale.factor,
        "scale_confidence": canon.scale.confidence,
        "scale_evidence": list(canon.scale.evidence),
        "up_residual_deg": canon.up_residual_deg,
        "yaw_residual_deg": canon.yaw_residual_deg,
        "gravity_axis_margin": canon.gravity_axis_margin,
        "gravity_sign_margin": canon.gravity_sign_margin,
        "method": dict(canon.method),
    }

    # planes were fitted in source space; refit in canonical space rather than
    # transforming them, because the transform carries a scale and a plane's
    # offset does not survive that as cleanly as a refit does
    with step("planes_canonical"):
        canon_normals = cloud.normals
        canon_planes = planemod.detect_planes(
            cloud,
            normals=canon_normals,
            distance_thresh=max(0.03, scan.noise_hint_m or 0.0),
            seed=opts.seed,
        )

    # -- derived representations ------------------------------------------
    with step("grid"):
        grid = gridmod.build_grid(cloud, voxel_xy=opts.voxel_xy, voxel_z=opts.voxel_z)

    with step("structure"):
        structure_notes: list[str] = []
        structure = structmod.analyze(
            cloud,
            planes=canon_planes,
            normals=canon_normals,
            grid=grid,
            camera_positions=cams,
            notes=structure_notes,
            seed=opts.seed,
        )
        if opts.skip_openings:
            structure.openings = []
    result.warnings += structure_notes

    # -- square the twin to the solved walls --------------------------------
    #
    # The canonicaliser's yaw is a vote over RANSAC wall planes, and a dense
    # cloud stuffs that ballot with cabinet fronts, counter sides and
    # through-door geometry -- this capture came out 22 degrees off before
    # this pass existed. The cell-solved footprint IS the wall arrangement,
    # so once it exists the twin is rotated to lay its dominant edge on the
    # axis. A pure yaw about the origin: gravity, the floor at z=0 and every
    # plane offset are all invariant under it, and everything yaw-dependent
    # downstream (capture bounds, mesh, QA, plane fill) runs after this.
    if (
        structure.footprint_source == "cells"
        and structure.footprint is not None
        and len(structure.footprint) >= 3
    ):
        theta = align.footprint_yaw(structure.footprint)
        if abs(theta) > np.radians(1.0):
            R3 = align.yaw_matrix(-theta)
            R2 = R3[:2, :2]
            cloud = PointCloud(
                xyz=(cloud.xyz @ R3.T).astype(np.float32),
                rgb=cloud.rgb,
                normals=(
                    None if cloud.normals is None else (cloud.normals @ R3.T).astype(np.float32)
                ),
                inferred=cloud.inferred,
            )
            if mesh is not None:
                mesh.vertices = mesh.vertices @ R3.T
            if cams is not None:
                cams = cams @ R3.T
            if extra_xyz is not None:
                extra_xyz = (extra_xyz @ R3.T).astype(np.float32)
            structure.footprint = structure.footprint @ R2.T
            for p in structure.planes:
                p.normal = R3 @ p.normal
            for o in structure.openings:
                o.center = R3 @ o.center
                o.normal = R3 @ o.normal
            R4 = np.eye(4)
            R4[:3, :3] = R3
            canon.transform = R4 @ canon.transform
            canon_normals = cloud.normals
            grid = gridmod.build_grid(cloud, voxel_xy=opts.voxel_xy, voxel_z=opts.voxel_z)
            result.steps["squared_to_walls_deg"] = round(float(np.degrees(theta)), 2)
            result.warnings.append(
                f"the twin was rotated {abs(float(np.degrees(theta))):.1f} deg to lay "
                "the solved walls on the axes; the initial yaw vote over fitted "
                "planes was pulled off-axis by furniture and through-door surfaces"
            )

    with step("capture_bounds"):
        bounds = hull.capture_bounds(
            cloud, camera_positions=cams, floor_z=structure.floor_z
        )

    if mesh is None or opts.force_mesh:
        if opts.skip_mesh_reconstruction and mesh is None:
            result.warnings.append(
                "the import carried no faces and mesh reconstruction was skipped, "
                "so this twin is points only"
            )
        else:
            with step("mesh"):
                try:
                    mesh = mesher.reconstruct_mesh(
                        cloud,
                        voxel=opts.mesh_voxel,
                        fill_holes=opts.fill_holes,
                        camera_positions=cams,
                        fill_radius_m=opts.fill_radius_m,
                        seed=opts.seed,
                    )
                    if mesh.filled is not None:
                        share = float((mesh.filled > 0.5).mean())
                        result.steps["mesh_fill"] = {"inferred_vertex_fraction": share}
                        if share > 0:
                            result.warnings.append(
                                f"{share:.0%} of the reconstructed surface was completed "
                                "from where the camera could see rather than from returns; "
                                "it is labelled in the twin and no measurement is taken "
                                "from it"
                            )
                except Exception as exc:  # reconstruction is best-effort
                    result.warnings.append(f"mesh reconstruction failed: {exc}")

    # -- assemble ----------------------------------------------------------
    geo = None
    if opts.latitude is not None and opts.longitude is not None:
        geo = Georeference(
            latitude=opts.latitude,
            longitude=opts.longitude,
            heading_deg=opts.heading_deg or 0.0,
            elevation_m=opts.elevation_m,
            heading_source="user" if opts.heading_deg is not None else "assumed",
        )
        if opts.heading_deg is None:
            result.warnings.append(
                "no heading was given, so +X is assumed to point north; every "
                "solar result downstream inherits that assumption"
            )

    twin = Twin(
        name=opts.name or _slug(src.stem),
        points=cloud,
        mesh=mesh,
        structure=structure,
        georeference=geo,
        capture_bounds=bounds,
        canonical_transform=canon.transform,
        provenance={
            "source_path": str(src.resolve()),
            "source_format": scan.source_format,
            "software": scan.software,
            "source_points": result.steps["read"]["points"],
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "locaish_version": _version(),
            "options": {
                "voxel_xy": opts.voxel_xy,
                "voxel_z": opts.voxel_z,
                "mesh_voxel": opts.mesh_voxel,
                "max_points": opts.max_points,
                "seed": opts.seed,
            },
            "steps": result.steps,
            "warnings": result.warnings,
        },
    )

    with step("qa"):
        twin.qa = qamod.assess(twin, grid=grid, normals=canon_normals, seed=opts.seed)
        twin.qa.metrics.setdefault("scale_confidence", canon.scale.confidence)
        twin.qa.metrics.setdefault("gravity_residual_deg", canon.up_residual_deg)
        _check_gravity_decision(twin.qa, canon)
        twin.qa.finalize()

    # -- plane-guided completion -------------------------------------------
    #
    # Deliberately after QA and after every measurement: the points added here
    # are resampled from the fitted wall planes, not observed, so structure,
    # openings, footprint and every QA metric were taken from the capture
    # alone. What the fill changes is only what a viewer sees -- complete
    # walls where the camera proved wall -- and each added point carries
    # `inferred = 1.0` so no later consumer can mistake it for measurement.
    if opts.fill_planes and cams is not None:
        with step("plane_fill"):
            from ..geom import planefill

            try:
                fill_xyz, fill_rgb, fill_nrm, fill_stats = planefill.fill_wall_planes(
                    cloud, structure, grid, cams, seed=opts.seed
                )
            except Exception as exc:  # completion is best-effort, like the mesh
                fill_xyz = np.zeros((0, 3))
                fill_stats = {"filled": False, "reason": f"plane fill failed: {exc}"}
        result.steps["plane_fill"] = {
            k: v for k, v in fill_stats.items() if k != "walls"
        }
        if len(fill_xyz):
            inferred = np.concatenate([
                np.zeros(len(cloud), dtype=np.float32)
                if cloud.inferred is None else cloud.inferred,
                np.ones(len(fill_xyz), dtype=np.float32),
            ])
            twin.points = PointCloud(
                xyz=np.concatenate([cloud.xyz, fill_xyz]),
                rgb=None if cloud.rgb is None
                else np.concatenate([cloud.rgb, fill_rgb]),
                normals=None if cloud.normals is None
                else np.concatenate([cloud.normals, fill_nrm]),
                inferred=inferred,
            )
            area = sum(w["filled_area_m2"] for w in fill_stats.get("walls", []))
            result.warnings.append(
                f"{len(fill_xyz):,} points ({area:.1f} m2 of wall) were resampled "
                "onto detected wall planes where camera rays prove the wall was "
                "seen and unbroken; they are tagged inferred, drawn desaturated, "
                "and excluded from every measurement"
            )

    # -- append the splat view layer ---------------------------------------
    #
    # Optimiser-derived density covering the surfaces stereo cannot measure.
    # It joins the twin only here, after every fit and every measurement is
    # done: exclusion by order, the same guarantee the plane fill has. And
    # because the room is already solved, the solve disciplines the layer:
    # a splat's soft fringe sprays feathered wings past the walls and stacks
    # layers behind them, so everything outside the solved volume is deleted
    # and everything within a handsbreadth of a solved wall is snapped flat
    # onto it. The optimiser proposes, the measured room disposes.
    if extra_xyz is not None and len(extra_xyz):
        extra_xyz, extra_rgb = _discipline_view_layer(
            extra_xyz, extra_rgb, structure
        )
    if extra_xyz is not None and len(extra_xyz):
        base = twin.points
        inferred = (
            None
            if base.inferred is None
            else np.concatenate(
                [base.inferred, np.zeros(len(extra_xyz), dtype=np.float32)]
            )
        )
        twin.points = PointCloud(
            xyz=np.concatenate([base.xyz, extra_xyz.astype(base.xyz.dtype)]),
            rgb=None
            if base.rgb is None
            else np.concatenate([base.rgb, extra_rgb.astype(np.uint8)]),
            normals=None
            if base.normals is None
            else np.concatenate(
                [base.normals, np.zeros((len(extra_xyz), 3), dtype=np.float32)]
            ),
            inferred=inferred,
        )
        result.steps["splat_points"] = int(len(extra_xyz))
        sp = (scan.raw_header or {}).get("splat_ply")
        if sp:
            twin.provenance["splat_ply"] = sp

    result.twin = twin
    result.timings["total"] = result.total_seconds

    # -- second pass: re-anchor the scale on a doorway ---------------------
    #
    # Runs at most once, and only for video, where the scale was estimated
    # rather than read off a header. Everything expensive is cached, so the
    # cost is the tail of the pipeline rather than the network.
    if _door_pass == 0 and is_video(src) and opts.video_scale_factor is None:
        anchor = _door_anchor(twin, result, opts)
        if anchor is not None:
            from dataclasses import replace

            if prog:
                prog(f"re-anchoring scale on {anchor.frames_used} doorway(s)")
            second = ingest(
                path,
                replace(opts, video_extra_scales=(anchor,)),
                _door_pass=1,
            )
            second.warnings.append(
                f"scale was re-solved against {anchor.frames_used} doorway(s) found "
                f"in the first pass; a door leaf is a manufactured standard and is "
                f"the tightest anchor available in a room nobody measured"
            )
            second.steps["door_anchor"] = anchor.to_dict()
            _semantic_crosscheck(second, frames_dir, prog)
            return second

    # -- Gemini cross-check, on the final twin only -------------------------
    #
    # An independent reading of the raw frames, compared against the geometry
    # after the fact -- see scan.semantic. Gated to the outermost pass so the
    # door-anchor recursion does not pay for it twice, and best-effort like
    # every other advisory stage.
    if _door_pass == 0:
        _semantic_crosscheck(result, frames_dir, prog)

    return result


def _discipline_view_layer(
    xyz: np.ndarray, rgb: np.ndarray, structure
) -> tuple[np.ndarray, np.ndarray]:
    """Cut a view-density layer to the solved room and flatten it onto walls.

    Everything below runs on the solved footprint and ceiling, which the
    layer itself never influenced -- so this is measurement disciplining
    inference, never inference voting on itself.
    """
    fp = structure.footprint
    if fp is None or len(fp) < 3 or structure.footprint_source not in {"cells", "raster"}:
        return xyz, rgb
    poly = np.asarray(fp, dtype=np.float64)
    pts = xyz.astype(np.float64)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    inside = np.zeros(len(pts), dtype=bool)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i - 1]
        x1, y1 = poly[i]
        dy = y0 - y1
        if abs(dy) < 1e-12:
            continue
        inside ^= ((y1 > y) != (y0 > y)) & (x < (x0 - x1) * (y - y1) / dy + x1)

    # walls: within the band around each solved wall, every 2 cm cell keeps
    # only the points that agree with that cell's median depth -- the
    # dominant layer. A splat stacks translucent layers through a wall, and
    # simply flattening them puts the dim back layers on the same plane as
    # the true surface, which reads as pepper; the consensus filter keeps
    # the layer most of the evidence voted for, whatever its colour, and
    # snaps it flat. A drifted minority layer is deleted, not averaged.
    band_out = 0.05
    band_in = 0.08
    agree = 0.010
    cell = 0.02
    near_wall = np.zeros(len(pts), dtype=bool)
    drop_wall = np.zeros(len(pts), dtype=bool)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        seg = b - a
        length = float(np.hypot(*seg))
        if length < 1e-9:
            continue
        d = seg / length
        inward = np.array([-d[1], d[0]])
        rel_x, rel_y = x - a[0], y - a[1]
        along = rel_x * d[0] + rel_y * d[1]
        d_in = rel_x * inward[0] + rel_y * inward[1]
        band = (
            (along >= -0.05)
            & (along <= length + 0.05)
            & (d_in >= -band_out)
            & (d_in <= band_in)
        )
        idx = np.flatnonzero(band)
        if not len(idx):
            continue
        cells = (
            np.floor(along[idx] / cell).astype(np.int64) * 4096
            + np.floor(z[idx] / cell).astype(np.int64)
        )
        order = np.argsort(cells, kind="stable")
        cs = cells[order]
        starts = np.flatnonzero(np.concatenate([[True], cs[1:] != cs[:-1]]))
        din_sorted = d_in[idx][order]
        med = np.empty(len(cs))
        for s0, s1 in zip(starts, np.concatenate([starts[1:], [len(cs)]])):
            med[s0:s1] = np.median(din_sorted[s0:s1])
        ok_sorted = np.abs(din_sorted - med) <= agree
        ok = np.zeros(len(idx), dtype=bool)
        ok[order] = ok_sorted
        keep_ids = idx[ok]
        x[keep_ids] -= d_in[keep_ids] * inward[0]
        y[keep_ids] -= d_in[keep_ids] * inward[1]
        near_wall[keep_ids] = True
        drop_wall[idx[~ok]] = True

    floor = float(structure.floor_z)
    cap = structure.drawable_ceiling_z
    top = (cap if cap is not None else floor + 2.40) + 0.08
    below = (z > floor - 0.06) & (z < floor + 0.01)
    z[below] = floor
    keep = (inside | near_wall) & ~(drop_wall & ~near_wall) & (z >= floor - 0.01) & (z <= top)
    if cap is not None:
        high = keep & (z > cap - 0.05)
        z[high] = np.minimum(z[high], cap)

    pts[:, 0], pts[:, 1], pts[:, 2] = x, y, z
    return pts[keep].astype(np.float32), rgb[keep]


def _semantic_crosscheck(result: IngestResult, frames_dir: Path | None, prog) -> None:
    """Run `scan.semantic` against the frames, best-effort, and record it."""
    from . import semantic

    if frames_dir is None or result.twin is None or not semantic.available():
        return
    if prog:
        prog("semantic")
    t0 = time.perf_counter()
    try:
        observation = semantic.crosscheck(frames_dir)
    except Exception as exc:  # advisory: a network failure must not cost a twin
        result.warnings.append(f"the Gemini frame cross-check was skipped: {exc}")
        return
    finally:
        result.timings["semantic"] = time.perf_counter() - t0
    if observation is None:
        return
    semantic.apply(result.twin, observation)
    result.steps["semantic"] = observation
    result.twin.provenance["semantic"] = observation


class IngestError(RuntimeError):
    """A scan that cannot become a twin at all, as opposed to one we distrust."""


#: Below this relative margin the vertical axis was effectively a coin toss.
#: Observed on a 3.6 x 5.9 x 2.4 m room where the axis was chosen by a 2% margin,
#: lost, and produced a twin 1185 mm wrong in two dimensions that passed every
#: other check cleanly.
GRAVITY_AXIS_MARGIN_FAIL = 0.10
GRAVITY_AXIS_MARGIN_WARN = 0.25
GRAVITY_SIGN_MARGIN_WARN = 0.10


def _door_anchor(twin, result, opts) -> object | None:
    """A scale estimate from any doorway the first pass found, or None.

    This exists as a second pass rather than as a third voice in the original
    vote because of an ordering that cannot be undone: apertures are found in
    the canonical frame, the canonical frame is built from a scale, and the
    scale is what we are trying to improve. So the first pass measures the
    doorways in provisional metres, and this converts "that door came out 3.4 m
    tall" into "then the provisional metre was 1.7 times too long".

    Cheap, because the reconstruction is cached: the second pass skips the
    network entirely and only redoes the arithmetic downstream of it.
    """
    from ..video import metric as metricmod

    if not opts.video_door_anchor or opts.video_scale_factor is not None:
        return None
    video = (result.steps or {}).get("video") or {}
    scale = video.get("scale") or {}
    current = scale.get("factor")
    if not current or not np.isfinite(current):
        return None
    openings = getattr(twin.structure, "openings", None)
    if not openings:
        return None
    return metricmod.scale_from_doors(openings, float(current))


def is_video(path: str | Path) -> bool:
    """Whether this source has to be reconstructed before it can be read.

    Extension only. Unlike the scan formats -- where the bytes are trusted over
    the filename because share sheets rename things -- there is nothing to gain
    from sniffing here: ffmpeg's own demuxer probe is far better than any header
    check we would write, and it runs a moment later anyway.
    """
    from ..video.frames import VIDEO_EXTENSIONS

    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def _hint_confidence(scan, opts) -> float | None:
    """How far to trust a declared unit, when the declarer said.

    Returns None for every ordinary import, which leaves the unit inference at
    its usual behaviour of believing a header outright. A front-end that
    *computed* the unit rather than reading it -- currently only video, whose
    metres come from the camera path -- records its own confidence, and
    that number has to survive all the way to QA or the twin will claim a
    certainty nobody established.
    """
    if opts.unit_hint:  # an explicit --unit from the operator is a decision
        return None
    value = (scan.raw_header or {}).get("scale_confidence")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _hint_evidence(scan) -> list[str] | None:
    header = scan.raw_header or {}
    source = header.get("scale_source")
    if not source or source == "header":
        return None
    factor = header.get("scale_factor_m_per_unit")
    if source == "supplied":
        return [f"scale supplied by the operator as {factor:.6g} m per source unit"]
    return [
        f"scale solved as {factor:.6g} m per source unit from the camera path's "
        "height above the floor (and any doorway found in the room); the room's "
        "shape is measured but its size is inferred"
    ]


def _check_gravity_decision(report: QAReport, canon) -> None:
    """Let the QA report see how close the gravity decision actually was.

    Every other check in the report measures the finished twin, which works for
    every error except this one. A twin built on the wrong vertical axis is
    internally perfect -- the walls are square, the surfaces are planar, the
    floor is level -- because it is a real room, merely lying on its side with
    its height and one plan dimension exchanged. No amount of inspecting the
    result recovers that; the only evidence is how narrowly the axis won at the
    moment it was chosen, and that evidence exists nowhere else.
    """
    axis = float(getattr(canon, "gravity_axis_margin", 1.0))
    sign = float(getattr(canon, "gravity_sign_margin", 1.0))
    report.metrics["gravity_axis_margin"] = axis
    report.metrics["gravity_sign_margin"] = sign

    if axis < GRAVITY_AXIS_MARGIN_FAIL:
        report.add(
            "gravity_axis",
            "fail",
            f"The vertical axis beat the runner-up by only {axis:.0%}, under the "
            f"{GRAVITY_AXIS_MARGIN_FAIL:.0%} floor, so which way is up was close to a "
            "coin toss. If it went the wrong way the room is standing on its side "
            "with its height and one plan dimension exchanged, and every other "
            "check in this report would still pass. Confirm the ceiling height "
            "against the real room before using any dimension.",
        )
    elif axis < GRAVITY_AXIS_MARGIN_WARN:
        report.add(
            "gravity_axis",
            "warn",
            f"The vertical axis beat the runner-up by {axis:.0%}, under the "
            f"{GRAVITY_AXIS_MARGIN_WARN:.0%} comfort threshold. This usually means a "
            "sparsely furnished or nearly cubical space, where little in the "
            "geometry distinguishes up from sideways. Check the ceiling height "
            "looks right.",
        )
    else:
        report.add(
            "gravity_axis",
            "pass",
            f"The vertical axis was chosen over the runner-up by a {axis:.0%} margin, "
            f"clear of the {GRAVITY_AXIS_MARGIN_WARN:.0%} threshold.",
        )

    if sign < GRAVITY_SIGN_MARGIN_WARN:
        report.add(
            "gravity_sign",
            "warn",
            f"Floor and ceiling were told apart by only a {sign:.0%} margin, under "
            f"{GRAVITY_SIGN_MARGIN_WARN:.0%}. If they are swapped the room is the right "
            "shape but upside down, so sill heights and camera heights would be "
            "measured from the ceiling.",
        )


def _median_spacing(xyz: np.ndarray, *, sample: int = 40_000, seed: int = 0) -> float:
    """Median nearest-neighbour distance, in whatever unit the file is in.

    This is the one length scale available before the unit is known, which is
    what makes it useful: every threshold expressed as a multiple of it is
    automatically unit-free. Sampled rather than exhaustive because the median
    of forty thousand spacings is indistinguishable from the median of twenty
    million and costs a thousandth as much.
    """
    from scipy.spatial import cKDTree

    if len(xyz) < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    idx = (
        rng.choice(len(xyz), size=sample, replace=False)
        if len(xyz) > sample
        else np.arange(len(xyz))
    )
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz[idx], k=2, workers=-1)
    spacing = float(np.median(d[:, 1]))
    return spacing if spacing > 0 else 1.0


def _slug(text: str) -> str:
    out = "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out or "twin"


def _version() -> str:
    from .. import __version__

    return __version__
