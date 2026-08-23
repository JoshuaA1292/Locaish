"""Every camera setup the room physically allows, scored and made searchable.

A shot brief -- "a clean close-up, long lens, no window behind her" -- is
answered on a real recce by a scout standing in the room mentally sweeping the
camera through it. This module does that sweep exhaustively: every standable
camera position, at several working heights, on every prime in the case,
against every plausible subject mark, with each combination checked against the
twin's actual geometry. The result is not a recommendation but a *table* --
hundreds of thousands of rows, one per physically-possible setup -- built for a
database whose job is selective filters over big tables, because that is what a
shot brief is: a filter.

Every column is a measurement or a direct consequence of one. Line of sight is
a ray marched through the occupancy grid the twin's own points built; clearance
and headroom come from the floor maps; framing and depth of field are thin-lens
arithmetic. Nothing here is an aesthetic judgement -- the one exception is
`score`, which is labelled as a tie-breaker and computed from stated,
inspectable preferences (level camera, room to work, measured ground), so that
ordering by it is a convenience rather than an oracle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..types import Twin
from . import optics
from . import space as spacemod

# Camera positions are drawn from the standable map at this spacing. Finer
# placement than this is below the twin's own accuracy for a video capture,
# and the table doubles in size for nothing.
CAMERA_SPACING_M = 0.30

# Subject marks are sparser: a scene has a handful of places the action can
# be, not a continuum, and each mark multiplies the whole table.
SUBJECT_SPACING_M = 0.70

# Working heights for the camera: low mode, slider/dolly, eye level. Each is
# kept only where the room's actual headroom allows it plus a margin.
CAMERA_HEIGHTS_M = (0.45, 1.05, 1.55)
HEIGHT_MARGIN_M = 0.15

# The primes actually swept. A subset of the full case: neighbouring focal
# lengths produce near-identical rows, and six lenses already span the range
# from wide establishing to compressed close-up.
SWEEP_PRIMES_MM = (16.0, 25.0, 35.0, 50.0, 75.0, 100.0)

# Setups closer than this are inside the actor's personal space and closer
# than any lens here can focus comfortably; farther than this is outside any
# room this pipeline is used on.
MIN_DISTANCE_M = 0.6
MAX_DISTANCE_M = 30.0

# Subject framings outside this band are dropped: below, the lens sees a
# button; above, the subject is a speck. Both are real shots in cinema and
# neither is answerable usefully from a room twin.
MIN_SUBJECT_FILL = 0.04
MAX_SUBJECT_FILL = 3.0

# Working aperture the depth-of-field columns are computed at. T2.8 is the
# conventional interior working stop; the agent can re-derive any other stop
# from distance and focal length, which are both columns.
WORKING_APERTURE_F = 2.8

SUBJECT_HEIGHT_M = 1.75
EYELINE_RATIO = 0.94

# Caps that keep the sweep bounded on a big room. Applied by even subsampling,
# never by truncation, so coverage of the room survives the cap.
MAX_CAMERA_CELLS = 700
MAX_SUBJECT_MARKS = 36


@dataclass
class ShotSweep:
    """The full sweep as parallel column arrays, ready for a columnar store."""

    location: str
    columns: dict[str, np.ndarray]
    subject_marks: np.ndarray          # (S, 2) the marks that were swept
    camera_cells: int
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        first = next(iter(self.columns.values()))
        return int(len(first))

    def summary(self) -> dict:
        vis = self.columns["visible"]
        return {
            "rows": len(self),
            "visible_rows": int(vis.sum()),
            "subject_marks": int(len(self.subject_marks)),
            "camera_cells": int(self.camera_cells),
            "lenses": sorted(set(np.unique(self.columns["focal_mm"]).tolist())),
            "warnings": self.warnings,
        }


def sweep(
    twin: Twin,
    *,
    cell: float = spacemod.DEFAULT_CELL_M,
    primes_mm: tuple[float, ...] = SWEEP_PRIMES_MM,
    heights_m: tuple[float, ...] = CAMERA_HEIGHTS_M,
    sensor_key: str = optics.DEFAULT_SENSOR,
    progress=None,
) -> ShotSweep:
    """Sweep every camera setup the twin's geometry allows, and score it."""
    warnings: list[str] = []
    sensor = optics.SENSORS[sensor_key]
    maps = spacemod.floor_maps(twin, cell=cell)
    standable = maps.standable()
    if not standable.any():
        # A thin or rough twin -- a sparse video reconstruction, usually --
        # can fail the full standing test everywhere while still knowing
        # perfectly well where its floor is. Degrade rather than refuse: the
        # sweep that comes back is labelled as resting on relaxed criteria,
        # which is a caveat, where an empty product is an outage.
        standable = (
            maps.inside
            & (maps.headroom_m >= 1.4)
            & (maps.clearance_m >= 0.15)
        )
        warnings.append(
            "no floor cell passed the full standing test (1.9 m headroom, "
            "0.28 m clearance); positions below rest on relaxed criteria and "
            "this twin is too rough to promise any of them physically works"
        )
    if not standable.any():
        raise ValueError(
            "no standable floor at all -- the twin has no cell with even "
            "crouching headroom and slim clearance, so there is nowhere to "
            "put a camera or an actor"
        )

    if progress:
        progress("sweep positions")
    cam_ij = _subsample(standable, maps.cell, CAMERA_SPACING_M, MAX_CAMERA_CELLS)
    subj_ij = _subsample(
        standable & maps.surveyed if (standable & maps.surveyed).any() else standable,
        maps.cell,
        SUBJECT_SPACING_M,
        MAX_SUBJECT_MARKS,
    )
    if not len(cam_ij) or not len(subj_ij):
        raise ValueError("the standable area is too small to sweep")

    cam_xy = maps.world_of(cam_ij)                      # (C, 2)
    subj_xy = maps.world_of(subj_ij)                    # (S, 2)
    floor_z = maps.floor_z

    cam_clear = maps.clearance_m[cam_ij[:, 0], cam_ij[:, 1]]
    cam_head = maps.headroom_m[cam_ij[:, 0], cam_ij[:, 1]]
    cam_surv = maps.surveyed[cam_ij[:, 0], cam_ij[:, 1]]

    # -- pair geometry, lens-independent ----------------------------------
    if progress:
        progress("sweep sightlines")
    grid, gorigin, gcell = spacemod.occupancy(twin)
    eye_z = floor_z + EYELINE_RATIO * SUBJECT_HEIGHT_M

    pairs = []
    for hz in heights_m:
        ok_h = cam_head >= hz + HEIGHT_MARGIN_M
        ci = np.flatnonzero(ok_h)
        if not len(ci):
            continue
        c_idx, s_idx = np.meshgrid(ci, np.arange(len(subj_xy)), indexing="ij")
        c_idx, s_idx = c_idx.ravel(), s_idx.ravel()
        a = np.column_stack([cam_xy[c_idx], np.full(len(c_idx), floor_z + hz)])
        b = np.column_stack([subj_xy[s_idx], np.full(len(s_idx), eye_z)])
        dist = np.linalg.norm(b - a, axis=1)
        keep = (dist >= MIN_DISTANCE_M) & (dist <= MAX_DISTANCE_M)
        c_idx, s_idx, a, b, dist = c_idx[keep], s_idx[keep], a[keep], b[keep], dist[keep]
        if not len(c_idx):
            continue
        vis = _visible_batch(grid, gorigin, gcell, a, b)
        pairs.append((hz, c_idx, s_idx, a, b, dist, vis))

    if not pairs:
        raise ValueError("no camera-subject pair survived the distance and headroom cuts")

    # -- assemble the per-pair rows, then cross with the lens set ----------
    if progress:
        progress("sweep scoring")
    h_all = np.concatenate([np.full(len(p[1]), p[0]) for p in pairs])
    c_all = np.concatenate([p[1] for p in pairs])
    s_all = np.concatenate([p[2] for p in pairs])
    a_all = np.concatenate([p[3] for p in pairs])
    b_all = np.concatenate([p[4] for p in pairs])
    d_all = np.concatenate([p[5] for p in pairs])
    v_all = np.concatenate([p[6] for p in pairs])

    view = b_all - a_all
    yaw = np.degrees(np.arctan2(view[:, 1], view[:, 0]))
    pitch = np.degrees(np.arcsin(np.clip(view[:, 2] / d_all, -1.0, 1.0)))

    openings = getattr(twin.structure, "openings", None) or []
    win_in_frame, win_behind_subj = _window_flags(openings, a_all, b_all, sensor)

    n_pairs = len(d_all)
    n_lens = len(primes_mm)
    total = n_pairs * n_lens

    def tile(x):
        return np.repeat(np.asarray(x), n_lens)

    focal = np.tile(np.asarray(primes_mm, dtype=np.float64), n_pairs)
    dist = tile(d_all)

    framed_h = sensor.height_mm * dist / focal
    fill = SUBJECT_HEIGHT_M / framed_h
    fov_h = 2.0 * np.degrees(np.arctan(sensor.width_mm / (2.0 * focal)))

    # Thin-lens depth of field at the working stop, vectorised.
    coc = sensor.circle_of_confusion_mm
    hyper_mm = focal * focal / (WORKING_APERTURE_F * coc) + focal
    d_mm = dist * 1000.0
    near = d_mm * (hyper_mm - focal) / (hyper_mm + d_mm - 2.0 * focal) / 1000.0
    far = np.where(
        d_mm >= hyper_mm,
        np.inf,
        d_mm * (hyper_mm - focal) / np.maximum(hyper_mm - d_mm, 1e-9) / 1000.0,
    )

    size_keys = np.array([s.key for s in optics.SHOT_SIZES])
    size_logs = np.log([s.framed_height_m for s in optics.SHOT_SIZES])
    log_fh = np.log(np.maximum(framed_h, 1e-9))
    nearest = np.argmin(np.abs(log_fh[:, None] - size_logs[None, :]), axis=1)
    shot_size = size_keys[nearest]
    # How cleanly the framing lands on the named size, 1 at dead-on.
    size_fit = np.exp(-np.abs(log_fh - size_logs[nearest]) * 3.0)

    keep = (fill >= MIN_SUBJECT_FILL) & (fill <= MAX_SUBJECT_FILL)

    clear = tile(cam_clear[c_all])
    head = tile(cam_head[c_all])
    surv = tile(cam_surv[c_all]).astype(np.uint8)
    visible = tile(v_all).astype(np.uint8)

    # The tie-breaker, from stated preferences: a level camera reads natural,
    # clearance is where the crew works, measured ground beats reconstructed
    # ground, and a framing that lands on a named size is one a director can
    # ask for by name. Blocked sightlines score zero -- they are kept in the
    # table because "how many setups did the pillar cost" is a real question.
    level = np.exp(-np.abs(tile(pitch)) / 18.0)
    room = np.clip(clear / 1.2, 0.0, 1.0)
    score = (
        100.0
        * np.where(visible > 0, 1.0, 0.0)
        * (0.40 * size_fit + 0.25 * level + 0.20 * room + 0.15 * surv)
    )

    columns = {
        "location": np.full(total, twin.name, dtype=object),
        "setup_id": np.arange(total, dtype=np.uint32),
        "cam_x": tile(a_all[:, 0]).astype(np.float32),
        "cam_y": tile(a_all[:, 1]).astype(np.float32),
        "cam_z": tile(h_all).astype(np.float32),
        "subj_x": tile(b_all[:, 0]).astype(np.float32),
        "subj_y": tile(b_all[:, 1]).astype(np.float32),
        "distance_m": dist.astype(np.float32),
        "yaw_deg": tile(yaw).astype(np.float32),
        "pitch_deg": tile(pitch).astype(np.float32),
        "focal_mm": focal.astype(np.float32),
        "sensor": np.full(total, sensor_key, dtype=object),
        "shot_size": shot_size.astype(object),
        "size_fit": size_fit.astype(np.float32),
        "framed_height_m": framed_h.astype(np.float32),
        "subject_fill": fill.astype(np.float32),
        "fov_h_deg": fov_h.astype(np.float32),
        "dof_near_m": near.astype(np.float32),
        "dof_far_m": np.where(np.isfinite(far), far, 0.0).astype(np.float32),
        "dof_infinite": (~np.isfinite(far)).astype(np.uint8),
        "visible": visible,
        "surveyed": surv,
        "clearance_m": clear.astype(np.float32),
        "headroom_m": head.astype(np.float32),
        "window_in_frame": tile(win_in_frame).astype(np.uint8),
        "window_behind_subject": tile(win_behind_subj).astype(np.uint8),
        "score": score.astype(np.float32),
    }
    columns = {k: v[keep] for k, v in columns.items()}
    columns["setup_id"] = np.arange(len(columns["setup_id"]), dtype=np.uint32)

    dropped = int(total - keep.sum())
    if dropped:
        warnings.append(
            f"{dropped:,} of {total:,} setups were dropped for framing outside "
            f"{MIN_SUBJECT_FILL}-{MAX_SUBJECT_FILL} of frame height"
        )

    return ShotSweep(
        location=twin.name,
        columns=columns,
        subject_marks=subj_xy,
        camera_cells=int(len(cam_xy)),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _subsample(mask: np.ndarray, cell: float, spacing_m: float, cap: int) -> np.ndarray:
    """Cells of `mask`, thinned to a spacing and then evenly capped."""
    stride = max(1, int(round(spacing_m / cell)))
    thin = np.zeros_like(mask)
    thin[::stride, ::stride] = True
    idx = np.argwhere(mask & thin)
    if len(idx) > cap:
        pick = np.linspace(0, len(idx) - 1, cap).round().astype(int)
        idx = idx[np.unique(pick)]
    return idx


def _visible_batch(grid, origin, cell, a, b, *, slack: float = 0.15) -> np.ndarray:
    """`space.visible` over N segments at once: one march, N rays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    seg = b - a
    length = np.linalg.norm(seg, axis=1)
    longest = float(length.max()) if len(length) else 0.0
    if longest < 1e-6:
        return np.ones(len(a), dtype=bool)
    steps = max(2, int(math.ceil(longest / (cell * 0.5))))
    t = np.linspace(0.0, 1.0, steps)                      # (T,)
    pts = a[:, None, :] + t[None, :, None] * seg[:, None, :]   # (N, T, 3)
    run = t[None, :] * length[:, None]
    inside = (run > slack) & ((length[:, None] - run) > slack)

    idx = np.floor((pts - origin) / cell).astype(np.int64)
    hi = np.array(grid.shape) - 1
    inb = np.all((idx >= 0) & (idx <= hi), axis=2)
    idx = np.clip(idx, 0, hi)
    hit = grid[idx[..., 0], idx[..., 1], idx[..., 2]] & inb & inside
    return ~hit.any(axis=1)


def _window_flags(openings, a, b, sensor) -> tuple[np.ndarray, np.ndarray]:
    """Per pair: is any opening in shot, and is one behind the subject.

    "Behind the subject" is the flag that matters on a recce: a window past the
    actor's shoulder means shooting into the light, and either a silhouette or
    a fight with HDR. Both tests are pure angles -- the opening's centre
    against the lens axis and its horizontal field of view.
    """
    n = len(a)
    in_frame = np.zeros(n, dtype=bool)
    behind = np.zeros(n, dtype=bool)
    if not openings:
        return in_frame, behind

    view = b - a
    view_n = view / np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-9)
    # The widest lens in the sweep sets the cone that could ever see it.
    half_fov = math.atan(sensor.width_mm / (2.0 * min(SWEEP_PRIMES_MM)))
    subj_dist = np.linalg.norm(view, axis=1)

    for o in openings:
        centre = np.asarray(getattr(o, "center", None), dtype=np.float64).reshape(3)
        rel = centre[None, :] - a
        d = np.linalg.norm(rel, axis=1)
        ok = d > 1e-6
        cosang = np.einsum("ij,ij->i", rel, view_n) / np.maximum(d, 1e-9)
        seen = ok & (cosang >= math.cos(half_fov))
        in_frame |= seen
        behind |= seen & (d > subj_dist)
    return in_frame, behind
