"""Quality assessment: the evidence behind the phrase "1:1".

A twin is a claim about a real room, and this module is where the claim is
made falsifiable. It measures the twin against itself and against a handful of
architectural priors, and it reports two different kinds of thing side by side:
numbers that were *measured* off the geometry, and numbers that were *inferred*
because the geometry did not contain the answer. The failure mode we exist to
prevent is a plausible number with nothing behind it. A wrong number that says
"the floor is only 43% covered, do not trust this footprint" is recoverable; a
wrong number that arrives silently is not.

Two rules follow from that and are worth stating because both have been broken
here before. A number that describes a way the twin can be wrong has to gate a
check: `opposite_wall_parallelism_deg` was measured for a long time and read by
nothing, so a room bent into a trapezoid came back "pass" with the evidence
sitting in the metrics table. And the canonical frame has to be verified on
every axis it constrains: the floor was re-measured against +Z while nothing
measured the walls against +X, and a twin rotated 15 degrees off its own walls
passed, because the one wall number we did have was a comparison of the walls
with each other and is unchanged by rotating the room.

Three things live here. `assess` builds the `QAReport`. `format_report` renders
it for a terminal so a location manager, not a graphics engineer, can read it.
`verify_measurement` answers the only question that really settles the argument:
the user holds a tape measure against a wall, and wants to know whether the twin
agrees and by how much it is entitled to disagree.

Deliberate dependency choice: this module imports numpy, scipy and
`locaish.types` and nothing else from the package. The occupancy grid and the
per-point normals produced upstream are accepted as optional arguments and used
when they are handed over, but QA never *requires* another analysis stage to
have succeeded. A quality report that cannot run when the pipeline is degraded
is a quality report that is missing exactly when it is needed.
"""

from __future__ import annotations

import math
import os
import textwrap
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ..types import Opening, Plane, QAReport, Twin, chunked

# ---------------------------------------------------------------------------
# thresholds
#
# Named constants rather than literals buried in the checks, because every
# message quotes the threshold it compared against and the two must never drift
# apart. The values are architectural, not statistical: they come from what a
# camera crew can work with, not from what a scanner happens to achieve.
# ---------------------------------------------------------------------------

SCALE_CONFIDENCE_FAIL = 0.40
GRAVITY_WARN_DEG = 0.25
GRAVITY_FAIL_DEG = 1.00
YAW_WARN_DEG = 0.50
YAW_FAIL_DEG = 2.00
PARALLEL_WARN_DEG = 0.50
PARALLEL_FAIL_DEG = 2.00
# Two planes further than this from facing each other are adjacent facets of a
# polygon, not a pair of walls that was ever meant to be parallel. Without the
# cap the check reads the 45 degrees between two facets of a round room as a
# 45-degree parallelism error, which is an opinion about the building.
PARALLEL_PAIR_MAX_DEG = 15.0
# The band a ceiling has to leave before we say so. It reads like an opinion
# about architecture and is really a check on scale: a room whose unit was
# misread comes back with a ceiling 2.54 or 3.28 times wrong, and that lands
# outside this band long before anything else in the report notices.
#
# Widening it to 14 m to stop it warning on genuinely tall locations was tried
# and reverted. It does warn on real 5 m rooms whose height we had recovered to
# 2 mm, which is a false alarm and a cost. But it was simultaneously the only
# check catching several twins whose unit was misread, and promoting those to
# `pass` broke the one invariant that must hold -- that nothing is wrong
# quietly. A warning an operator must clear by hand on a tall location is a
# fair price for that, and the message says which of the two it might be rather
# than asserting the space is implausible.
CEILING_MIN_M = 2.10
CEILING_MAX_M = 5.00
FLOOR_COVERAGE_FAIL = 0.50
FLOOR_COVERAGE_WARN = 0.80
DENSITY_WARN_PER_M2 = 300.0
SPACING_WARN_M = 0.05
DRIFT_WARN_M = 0.02
DRIFT_FAIL_M = 0.05
PLANARITY_WARN_M = 0.02
CAPTURE_RATIO_WARN = 0.50
DUPLICATE_WARN = 0.15

DUPLICATE_RADIUS_M = 0.001
WALL_BAND_M = (0.5, 2.0)
DRIFT_TILE_M = 0.5
WALL_INLIER_TOL_M = 0.06
NEIGHBOUR_SAMPLE = 200_000
DENSITY_K = 9

# A wall is only allowed to speak for the room's shape if it is sampled at
# something like the scan's own resolution; see `_structural_walls`.
WALL_SUPPORT_REL = 0.50
# How far a wall may sit from the frame its neighbours agree on and still count
# as part of that frame. Five degrees is well past any plaster or fitting error
# and well inside a deliberate splay, so it separates "a rectangular room" from
# "a room with an angled wall" without adjudicating either.
AXIS_TOL_DEG = 5.0
# Wall area that has to agree on one right-angled frame before a yaw offset is
# read as our error rather than as the building's floor plan.
RECTILINEAR_MIN_FRACTION = 0.60
# Below this the room is square enough that which side is "long" is not a
# meaningful claim, so the +X convention is not tested against it.
LONG_AXIS_RATIO_MIN = 1.05

_STATUS_ORDER = ("pass", "info", "warn", "fail")


# ---------------------------------------------------------------------------
# small geometry helpers
# ---------------------------------------------------------------------------


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares plane through `points` as (unit normal, offset).

    Plain PCA on the centred coordinates: the smallest singular direction is
    the normal. We use SVD on the 3x3 covariance rather than on the (N, 3)
    matrix so the cost is independent of how many inliers a wall has.
    """
    c = points.mean(axis=0)
    cov = np.cov((points - c).T)
    _, _, vt = np.linalg.svd(cov)
    n = vt[-1]
    n = n / np.linalg.norm(n)
    return n, float(np.dot(n, c))


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))
    return math.degrees(math.acos(d))


def _inside_polygon(polygon: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    """Ray-crossing point-in-polygon, vectorised over points.

    Duplicated from `CaptureBounds.contains` rather than borrowed, because the
    footprint is a bare (M, 2) array and wrapping it in a CaptureBounds to
    borrow the method would put a capture claim on a polygon that is not one.
    """
    poly = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(poly) < 3:
        return np.zeros(len(pts), dtype=bool)
    inside = np.zeros(len(pts), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        straddles = (yi > pts[:, 1]) != (yj > pts[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (xj - xi) * (pts[:, 1] - yi) / (yj - yi) + xi
        inside ^= straddles & (pts[:, 0] < x_cross)
        j = i
    return inside


def _wall_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """In-plane axes for a wall: `u` horizontal along the wall, `v` up it.

    Tiling a wall's residual needs axes that mean something architecturally --
    "along the wall" and "up the wall" -- so that a drift tile is a patch of
    wall rather than an arbitrary parallelogram. Degenerate for a horizontal
    plane, which is why only walls are ever passed in.
    """
    up = np.array([0.0, 0.0, 1.0])
    u = np.cross(up, normal)
    n = np.linalg.norm(u)
    if n < 1e-9:
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = u / n
    v = np.cross(normal, u)
    return u, v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# point statistics
# ---------------------------------------------------------------------------


def _neighbour_stats(xyz: np.ndarray, seed: int) -> dict[str, float]:
    """Median nearest-neighbour spacing, surface density and duplicate fraction.

    All three come from one k-nearest-neighbour query, so we pay for the tree
    once. The tree is built on every point (densities must be true) but queried
    from a bounded random subsample, because the median of 200k samples is the
    same number as the median of 20M and costs two orders of magnitude less.

    Density is estimated per point as (k-1) / (pi * r_k^2), the standard
    unbiased kernel for a locally planar Poisson sample: the k-th neighbour
    distance describes the disc of surface that k points had to share. Taking
    the median over points, rather than dividing the total by the mesh area,
    means a scan that is dense in one corner and empty elsewhere reports the
    density a user would actually experience in a typical spot.

    The planar assumption is the estimator's limit: once range noise grows to
    the same size as the spacing, the surface is a thin slab rather than a
    sheet and the discs overlap in three dimensions, which under-reports the
    density. It stays monotone in the true density, so the thresholds still
    discriminate, but a very dense and very noisy scan will read lower than its
    nominal sampling rate.

    Spacing deliberately ignores neighbours closer than 1 mm. A re-registered
    second pass lays a near-copy of every surface on top of itself; counting
    those pairs would report a spacing ten times finer than the scanner can
    resolve. The duplicates are not discarded, they are reported separately as
    `duplicate_fraction` so the inflation is visible rather than absorbed.
    """
    out = {
        "median_spacing_m": float("nan"),
        "density_per_m2": float("nan"),
        "duplicate_fraction": float("nan"),
    }
    n = len(xyz)
    if n < 8:
        return out

    rng = np.random.default_rng(seed)
    tree = cKDTree(xyz)
    if n > NEIGHBOUR_SAMPLE:
        idx = rng.choice(n, NEIGHBOUR_SAMPLE, replace=False)
    else:
        idx = np.arange(n)
    k = min(DENSITY_K, n)
    dist, _ = tree.query(xyz[idx], k=k, workers=-1)
    dist = np.atleast_2d(dist)

    # column 0 is the query point finding itself
    near = dist[:, 1:]
    if near.size == 0:
        return out

    out["duplicate_fraction"] = float(np.mean(near[:, 0] < DUPLICATE_RADIUS_M))

    distinct = np.where(near >= DUPLICATE_RADIUS_M, near, np.inf)
    first = distinct.min(axis=1)
    valid = np.isfinite(first)
    if valid.any():
        out["median_spacing_m"] = float(np.median(first[valid]))

    r_k = near[:, -1]
    good = r_k > 0
    if good.any():
        density = (near.shape[1] - 1) / (math.pi * r_k[good] ** 2)
        out["density_per_m2"] = float(np.median(density))
    return out


# ---------------------------------------------------------------------------
# coverage lattice
#
# Floor, ceiling, wall band and hole fraction are all the same question asked
# of different parts of the shell -- "is there a point where the architecture
# says there should be one" -- so they share one lattice. Anything else lets
# the four numbers disagree with each other.
# ---------------------------------------------------------------------------


class _Lattice:
    """A horizontal cell lattice over the footprint plus its boundary ring."""

    def __init__(self, cell: float, lo: np.ndarray, shape: tuple[int, int], inside: np.ndarray):
        self.cell = cell
        self.lo = lo
        self.shape = shape
        self.inside = inside
        eroded = ndimage.binary_erosion(inside, iterations=1, border_value=0)
        self.boundary = inside & ~eroded
        if not self.boundary.any():
            self.boundary = inside.copy()

    def index(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ix = np.clip(((xy[:, 0] - self.lo[0]) / self.cell).astype(np.int64), 0, self.shape[0] - 1)
        iy = np.clip(((xy[:, 1] - self.lo[1]) / self.cell).astype(np.int64), 0, self.shape[1] - 1)
        return ix, iy

    def centres(self, mask: np.ndarray) -> np.ndarray:
        ix, iy = np.nonzero(mask)
        return np.stack(
            [self.lo[0] + (ix + 0.5) * self.cell, self.lo[1] + (iy + 0.5) * self.cell], axis=1
        )


def _coverage_cell(spacing: float, grid: Any) -> float:
    """Cell size for every coverage number, in metres.

    A cell has to be comfortably larger than the point spacing or a perfectly
    scanned surface reads as full of holes purely from Poisson luck; four times
    the spacing puts roughly sixteen points in a full cell, so an empty cell is
    evidence of a real gap rather than of sparse sampling. That decoupling is
    the point: coverage should measure occlusion, and density should measure
    density, and neither should quietly report the other.

    If the caller handed us the occupancy grid from `geom.grid` we adopt its
    cell size so the two modules describe the same room in the same units. We
    read it defensively through `getattr` and sanity-bound it, because QA must
    keep working against whatever that module's field is eventually called.
    """
    if grid is not None:
        for attr in ("cell_size", "cell", "resolution", "voxel_size"):
            value = getattr(grid, attr, None)
            if isinstance(value, (int, float)) and 0.01 <= float(value) <= 0.5:
                return float(value)
    if not math.isfinite(spacing) or spacing <= 0:
        return 0.10
    return float(np.clip(4.0 * spacing, 0.05, 0.20))


def _build_lattice(xyz: np.ndarray, footprint: np.ndarray | None, cell: float) -> _Lattice | None:
    """Rasterise the footprint, or fall back to where the points actually are.

    The polygon is preferred because it is the structural claim about the room.
    When `scan.structure` could not produce one we rasterise the cloud itself
    and close it up, which over-reports coverage slightly (a region nobody
    scanned is simply not part of the footprint) -- so the fallback is recorded
    in the coverage message rather than passed off as equivalent.
    """
    if len(xyz) == 0:
        return None
    lo = xyz[:, :2].min(axis=0) - cell
    hi = xyz[:, :2].max(axis=0) + cell
    span = np.maximum(hi - lo, cell)
    shape = tuple(int(max(2, math.ceil(s / cell))) for s in span)
    if shape[0] * shape[1] > 4_000_000:
        return None

    inside = None
    if footprint is not None and len(footprint) >= 3:
        gx = lo[0] + (np.arange(shape[0]) + 0.5) * cell
        gy = lo[1] + (np.arange(shape[1]) + 0.5) * cell
        mx, my = np.meshgrid(gx, gy, indexing="ij")
        centres = np.stack([mx.ravel(), my.ravel()], axis=1)
        inside = _inside_polygon(footprint, centres).reshape(shape)
        if inside.sum() < 4:
            inside = None

    if inside is None:
        occ = np.zeros(shape, dtype=bool)
        ix = np.clip(((xyz[:, 0] - lo[0]) / cell).astype(np.int64), 0, shape[0] - 1)
        iy = np.clip(((xyz[:, 1] - lo[1]) / cell).astype(np.int64), 0, shape[1] - 1)
        occ[ix, iy] = True
        occ = ndimage.binary_closing(occ, structure=np.ones((3, 3), bool), border_value=0)
        inside = ndimage.binary_fill_holes(occ)
        if inside is None or inside.sum() < 4:
            return None
    return _Lattice(cell, lo, shape, inside)


def _slab_occupancy(lat: _Lattice, xyz: np.ndarray, z_lo: float, z_hi: float) -> np.ndarray:
    sel = (xyz[:, 2] >= z_lo) & (xyz[:, 2] < z_hi)
    occ = np.zeros(lat.shape, dtype=bool)
    if sel.any():
        ix, iy = lat.index(xyz[sel, :2])
        occ[ix, iy] = True
    return occ


def _opening_probe_mask(
    probes: np.ndarray, openings: Sequence[Opening], cell: float
) -> np.ndarray:
    """Which wall probes sit inside a detected opening.

    A window is a hole that we have already accounted for and named. Counting
    it again as missing geometry would mean every honestly scanned room with
    windows reported itself as full of holes, and the number would stop meaning
    "unscanned".
    """
    mask = np.zeros(len(probes), dtype=bool)
    up = np.array([0.0, 0.0, 1.0])
    for op in openings:
        n = op.normal
        if np.linalg.norm(n) < 1e-9:
            continue
        u = np.cross(up, n)
        nu = np.linalg.norm(u)
        if nu < 1e-9:
            continue
        u = u / nu
        d = probes - op.center
        mask |= (
            (np.abs(d @ u) <= op.width / 2.0 + cell)
            & (np.abs(d @ up) <= op.height / 2.0 + cell)
            & (np.abs(d @ n) <= 0.35)
        )
    return mask


def _coverage(
    xyz: np.ndarray,
    lat: _Lattice,
    floor_z: float,
    ceiling_z: float | None,
    openings: Sequence[Opening],
) -> dict[str, float]:
    """Floor, ceiling and wall-band coverage, plus the overall hole fraction."""
    cell = lat.cell
    slab = max(0.10, 1.5 * cell)
    inside_n = int(lat.inside.sum())
    out = {
        "floor_coverage": float("nan"),
        "ceiling_coverage": float("nan"),
        "wall_coverage": float("nan"),
        "hole_fraction": float("nan"),
    }
    if inside_n == 0:
        return out

    floor_occ = _slab_occupancy(lat, xyz, floor_z - 0.5 * cell, floor_z + slab)
    out["floor_coverage"] = float(floor_occ[lat.inside].mean())
    shell_total = inside_n
    shell_filled = int(floor_occ[lat.inside].sum())

    if ceiling_z is not None and math.isfinite(ceiling_z):
        ceil_occ = _slab_occupancy(lat, xyz, ceiling_z - slab, ceiling_z + 0.5 * cell)
        out["ceiling_coverage"] = float(ceil_occ[lat.inside].mean())
        shell_total += inside_n
        shell_filled += int(ceil_occ[lat.inside].sum())

    # walls: one horizontal slice per cell of height, dilated by a cell before
    # testing. A wall surface lies exactly on the footprint boundary, so which
    # side of a grid line its points land on is an accident of where we happened
    # to put the lattice origin and must not be read as a hole.
    top = ceiling_z if (ceiling_z is not None and math.isfinite(ceiling_z)) else float(
        xyz[:, 2].max()
    )
    boundary_n = int(lat.boundary.sum())
    boundary_xy = lat.centres(lat.boundary)
    band_hits = band_total = 0
    shell_wall_hits = shell_wall_total = 0
    z = floor_z + 0.5 * cell
    struct = np.ones((3, 3), bool)
    while z < top - 0.25 * cell and boundary_n:
        occ = _slab_occupancy(lat, xyz, z - 0.5 * cell, z + 0.5 * cell)
        filled = ndimage.binary_dilation(occ, structure=struct, border_value=0)[lat.boundary]

        probes = np.concatenate([boundary_xy, np.full((boundary_n, 1), z)], axis=1)
        counted = ~_opening_probe_mask(probes, openings, cell)
        shell_wall_total += int(counted.sum())
        shell_wall_hits += int((filled & counted).sum())

        if WALL_BAND_M[0] <= z - floor_z <= WALL_BAND_M[1]:
            band_total += boundary_n
            band_hits += int(filled.sum())
        z += cell

    if band_total:
        out["wall_coverage"] = band_hits / band_total
    shell_total += shell_wall_total
    shell_filled += shell_wall_hits
    if shell_total:
        out["hole_fraction"] = 1.0 - shell_filled / shell_total
    return out


# ---------------------------------------------------------------------------
# surfaces: floor level, wall planarity, drift
# ---------------------------------------------------------------------------


@dataclass
class _WallFit:
    """One wall as QA measured it, rather than as the pipeline declared it.

    Every wall-shaped question -- is the room bent, is it square to the axes,
    do facing walls stay the same distance apart -- has to be asked of the same
    re-measured surface. Reading the answers off `Plane.normal` instead would
    mean the yaw check graded the plane detector's output rather than the
    geometry, and would return whatever the detector believed even when the
    points underneath it say something else.
    """

    normal: np.ndarray
    rms: float
    drift: float
    span_u: float
    weight: float

    @property
    def azimuth_rad(self) -> float:
        return math.atan2(float(self.normal[1]), float(self.normal[0]))


def _wall_residual_analysis(
    plane: Plane,
    xyz: np.ndarray,
    normals: np.ndarray | None,
    openings: Sequence[Opening] = (),
    tile: float = DRIFT_TILE_M,
) -> _WallFit | None:
    """Re-fit one wall from the points and describe how flat and how bent it is.

    Returns None when the wall has too few points left, after the contaminants
    below are removed, to say anything at all -- an absent measurement rather
    than a zero, because a zero here reads as "flat" and would be the most
    expensive lie in the module.

    Drift is not noise and must not be measured like noise. A SLAM solution
    that loses a little pose accuracy over a sweep does not scatter points, it
    *bends* the room: a wall that is physically flat comes back as a shallow
    curve, and the error a user meets is the difference between two ends of
    that curve. Raw RMS about the best-fit plane cannot see this, because it is
    dominated by per-point range noise -- a scanner with 12 mm noise and no
    drift and a scanner with 3 mm noise and a 60 mm bow can report the same RMS
    while only one of them will disagree with a tape measure.

    So we tile the wall at roughly half a metre, take the *median* residual per
    tile (which averages the noise down while leaving any bow intact) and
    report the range across tiles. That range is what a long measurement across
    the wall is wrong by.

    Contaminants have to be kept out of the tile medians or they invent drift
    that is not there. The worst offender is the reveal around a doorway: the
    120 mm depth of the jamb sits comfortably inside any sane inlier band, and
    a tile that lands in the middle of a door contains nothing *but* reveal, so
    no amount of robust averaging inside that tile can save it. Three defences,
    in order of directness: detected openings are cut out of the wall along with
    a margin; per-point normals, when the caller has them, drop anything facing
    more than 30 degrees away from the wall; and tiles left holding far fewer
    points than a typical tile are discarded, because a tile that is mostly hole
    is a tile whose median describes the hole's edges. What survives is trimmed
    a few robust sigma about the *local* median rather than about the plane, so
    the trim follows a genuine bow instead of flattening it.
    """
    cuts = _openings_on(plane, openings)
    u_axis = _wall_frame(plane.normal)[0]
    cos_limit = math.cos(math.radians(30.0))
    chosen: list[np.ndarray] = []
    for block in chunked(len(xyz), 1_000_000):
        pts = xyz[block]
        keep = np.abs(pts @ plane.normal - plane.offset) < WALL_INLIER_TOL_M
        if normals is not None:
            keep &= np.abs(normals[block] @ plane.normal) > cos_limit
        idx = np.nonzero(keep)[0]
        if len(idx) and cuts:
            near = pts[idx]
            drop = np.zeros(len(idx), dtype=bool)
            for cut in cuts:
                d = near - cut.center
                drop |= (np.abs(d @ u_axis) <= cut.width / 2.0 + 0.08) & (
                    np.abs(d[:, 2]) <= cut.height / 2.0 + 0.08
                )
            idx = idx[~drop]
        if len(idx):
            chosen.append(idx + block.start)
    if not chosen:
        return None
    sel = np.concatenate(chosen)
    if len(sel) < 50:
        return None

    pts = xyz[sel]
    normal, offset = _fit_plane(pts)
    if np.dot(normal, plane.normal) < 0:
        normal, offset = -normal, -offset
    r = pts @ normal - offset

    u_axis, v_axis = _wall_frame(normal)
    u = pts @ u_axis
    v = pts @ v_axis
    span_u = float(u.max() - u.min())
    weight = float(plane.area) if plane.area > 0 else float(len(sel))

    def fit(rms: float, drift: float) -> _WallFit:
        return _WallFit(
            normal=normal, rms=rms, drift=drift, span_u=span_u, weight=weight
        )

    iu = ((u - u.min()) / tile).astype(np.int64)
    iv = ((v - v.min()) / tile).astype(np.int64)
    key = iu * (iv.max() + 1) + iv
    uniq, inv = np.unique(key, return_inverse=True)
    if len(uniq) < 4:
        return fit(float(np.sqrt(np.mean(r**2))), float("nan"))

    def tile_medians(keep: np.ndarray) -> np.ndarray:
        """Median residual per tile, or nan for tiles too thin to trust one.

        The threshold is relative to the typical tile as well as absolute: an
        edge tile with a third of the usual points is a partly-scanned patch of
        wall and its median is not evidence.
        """
        med = np.full(len(uniq), np.nan)
        tiles = inv[keep]
        order = np.argsort(tiles, kind="stable")
        sorted_tiles = tiles[order]
        sorted_r = r[keep][order]
        edges = np.searchsorted(sorted_tiles, np.arange(len(uniq) + 1))
        counts = np.diff(edges)
        floor_count = max(12.0, 0.35 * float(np.median(counts[counts >= 12])) if (counts >= 12).any() else 12.0)
        for i in range(len(uniq)):
            lo, hi = edges[i], edges[i + 1]
            if hi - lo >= floor_count:
                med[i] = np.median(sorted_r[lo:hi])
        return med

    med = tile_medians(np.ones(len(r), dtype=bool))
    local = med[inv]
    known = np.isfinite(local)
    detrended = r - np.where(known, local, 0.0)
    sigma = 1.4826 * float(np.median(np.abs(detrended[known]))) if known.any() else 0.0
    keep = known & (np.abs(detrended) < max(4.0 * sigma, 0.004))
    if keep.sum() < 50:
        keep = np.ones(len(r), dtype=bool)

    med = tile_medians(keep)
    filled = med[np.isfinite(med)]
    rms = float(np.sqrt(np.mean(r[keep] ** 2)))
    drift = float(filled.max() - filled.min()) if len(filled) >= 4 else float("nan")
    return fit(rms, drift)


def _measure_floor(
    xyz: np.ndarray, floor_z: float, spacing: float, seed: int
) -> tuple[np.ndarray, float] | None:
    """Refit the floor from the points and return its (normal, offset).

    This deliberately re-measures instead of reading the floor plane out of
    `Structure`. That plane is usually the very one the canonicaliser levelled
    the twin against, so comparing it with +Z would report the aligner's
    intention rather than the geometry's behaviour and would come back as
    exactly zero no matter how badly the room was bent. An independent fit to
    the points near floor level can disagree with the alignment, which is the
    only way the disagreement ever gets seen.

    One robust trim removes the skirting, the bottom edges of furniture and
    anything else sitting in the band that is not the floor.

    The band the points are drawn from is then re-cut about the fitted plane
    rather than about z = floor_z, and the fit repeated. Selecting on height
    alone is a horizontal window, so on the very twin this check exists to
    catch -- one whose floor is not level -- it keeps the middle of the floor
    and throws away both ends, and the surviving strip fits far flatter than
    the floor really is. A one degree tilt across a 10 m room moves the floor
    by 175 mm, well outside any sane band, so the check would have quietly
    reported a fraction of the error it was measuring.
    """
    band = max(0.12, 3.0 * spacing if math.isfinite(spacing) else 0.06)
    rng = np.random.default_rng(seed)

    def robust_fit(pts: np.ndarray) -> tuple[np.ndarray, float] | None:
        if len(pts) < 256:
            return None
        if len(pts) > 500_000:
            pts = pts[rng.choice(len(pts), 500_000, replace=False)]
        normal, offset = _fit_plane(pts)
        if normal[2] < 0:
            normal, offset = -normal, -offset
        r = pts @ normal - offset
        sigma = 1.4826 * float(np.median(np.abs(r - np.median(r))))
        keep = np.abs(r) < 3.0 * max(sigma, 0.005)
        if keep.sum() >= 256:
            normal, offset = _fit_plane(pts[keep])
            if normal[2] < 0:
                normal, offset = -normal, -offset
        return normal, float(offset)

    fit = robust_fit(xyz[np.abs(xyz[:, 2] - floor_z) < band])
    if fit is None:
        return None
    for _ in range(2):
        normal, offset = fit
        # the re-cut is the same width as the original band and starts from a
        # plane already fitted to the floor, so what it adds is the far ends of
        # a tilted floor rather than a new surface; the robust trim inside the
        # fit still throws out whatever furniture the wider cut sweeps up
        near = np.abs(xyz @ normal - offset) < band
        if int(near.sum()) < 256:
            break
        again = robust_fit(xyz[near])
        if again is None:
            break
        fit = again
    return fit


def _openings_on(plane: Plane, openings: Sequence[Opening]) -> list[Opening]:
    """The openings that belong to this wall, by orientation and by proximity.

    An opening carries its own normal but not the identity of the wall it was
    cut from, so we re-associate here rather than trusting an index that no
    contract guarantees.
    """
    out = []
    for op in openings:
        if abs(float(np.dot(op.normal, plane.normal))) < 0.8:
            continue
        if abs(float(np.dot(plane.normal, op.center)) - plane.offset) > 0.4:
            continue
        out.append(op)
    return out


def _large_walls(walls: Sequence[Plane]) -> list[Plane]:
    """Walls big enough that a bow in them means something.

    A 0.3 m^2 sliver of plane picked up beside a doorway has a residual range
    driven entirely by which few hundred points fell in it, so letting it decide
    the drift verdict would make the verdict a coin toss.
    """
    if not walls:
        return []
    areas = np.array([max(w.area, 0.0) for w in walls])
    counts = np.array([max(w.inlier_count, 0) for w in walls])
    if areas.max() > 0:
        keep = areas >= max(2.0, 0.15 * areas.max())
    elif counts.max() > 0:
        keep = counts >= max(500, int(0.15 * counts.max()))
    else:
        keep = np.ones(len(walls), dtype=bool)
    chosen = [w for w, k in zip(walls, keep) if k]
    return chosen or list(walls)


def _wall_support(plane: Plane, spacing: float) -> float:
    """Points per square metre on a plane, in units of the scan's own resolution.

    A surface that was really scanned carries about one point per `spacing^2`
    of its area, and the constant is the same whether the scan is dense or thin
    because both the numerator and the denominator move with it: for a Poisson
    sample of density lambda the median nearest-neighbour distance satisfies
    lambda * spacing^2 = ln 2 / pi, about 0.22, for any lambda. So this ratio is
    near a fixed value for every genuine wall in every room, and far below it
    for a plane the detector scraped together out of leftovers.

    That distinction matters because those leftovers are not harmless. A thin
    plane cutting obliquely through a real wall's points collects a slanted slab
    of them, which reads as a bowed wall, an out-of-parallel pair and a tilted
    frame all at once -- the observed case was a room correct to 2 mm failed for
    51 mm of tracking drift that only a 600-point ghost could see.
    """
    if plane.area <= 0 or not math.isfinite(spacing) or spacing <= 0:
        return float("nan")
    return float(plane.inlier_count) * spacing * spacing / float(plane.area)


def _structural_walls(walls: Sequence[Plane], spacing: float) -> list[Plane]:
    """The walls entitled to make a claim about the room's shape.

    Large enough to mean something (`_large_walls`) and solid enough to be a
    surface rather than a residue. The support floor is relative to the best
    wall in the same room rather than absolute, so scanner density, unit and
    noise all cancel; an absolute threshold would have to be re-tuned for every
    scanner, and would be exactly the kind of constant that quietly stops
    discriminating on hardware nobody tested it against.

    When no plane records an area there is nothing to judge and every candidate
    is kept, because dropping walls we cannot measure would silently narrow the
    evidence rather than report that it was missing.

    The cost of the filter is that a real wall seen only at grazing incidence
    could be dropped, and a bow or a splay that lives on that wall alone would
    then go unmeasured. That is why every check built on this set quotes how
    many of the room's wall planes survived it: a narrowed set of evidence is
    something the reader is told about rather than something they infer.
    """
    big = _large_walls(walls)
    if len(big) < 2:
        return big
    support = np.array([_wall_support(w, spacing) for w in big])
    if not np.isfinite(support).any():
        return big
    floor = WALL_SUPPORT_REL * float(np.nanmax(support))
    keep = [w for w, s in zip(big, support) if math.isfinite(s) and s >= floor]
    return keep if len(keep) >= 2 else big


def _wall_frame_stats(fits: Sequence[_WallFit]) -> dict[str, float]:
    """Where the walls put their right-angled frame, and how far that is from ours.

    Wall azimuths in a rectangular room are four values 90 degrees apart, so the
    quantity with a single well-defined mean is the azimuth taken modulo 90
    degrees; averaging unit vectors at four times the angle is the standard way
    to do that without the wrap at the fold ruining the mean. Each wall is
    weighted by its area, so the room's big walls decide the frame.

    Three numbers come out of it and they answer three different questions.
    `manhattan_residual_deg` is how far the worst wall sits from the frame its
    neighbours agree on -- a property of the building, and never a failure.
    `wall_frame_offset_deg` is how far that whole frame sits from the twin's own
    X and Y axes -- a property of our canonicalisation, and nothing to do with
    the building, because rotating a room does not change its floor plan.
    `wall_rectilinear_fraction` is the share of wall area that lies within
    `AXIS_TOL_DEG` of the frame, which is what says whether the offset means
    anything: in a round or splayed room there is no frame to be offset from.
    """
    out = {
        "manhattan_residual_deg": float("nan"),
        "wall_frame_offset_deg": float("nan"),
        "wall_rectilinear_fraction": float("nan"),
    }
    if len(fits) < 2:
        return out
    theta = np.array([f.azimuth_rad for f in fits])
    weight = np.array([max(f.weight, 1e-9) for f in fits])
    mean = float(np.angle(np.sum(weight * np.exp(4j * theta)))) / 4.0
    dev = (theta - mean + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0
    dev_deg = np.abs(np.degrees(dev))

    out["manhattan_residual_deg"] = float(dev_deg.max())
    out["wall_frame_offset_deg"] = math.degrees(mean)
    out["wall_rectilinear_fraction"] = float(
        weight[dev_deg <= AXIS_TOL_DEG].sum() / weight.sum()
    )
    return out


def _opposite_wall_parallelism(fits: Sequence[_WallFit]) -> tuple[float, float]:
    """Worst facing pair's departure from parallel, in degrees and in metres.

    Two walls that face each other should have exactly opposed normals. When
    they do not, the room is a trapezoid: every width measured between them
    depends on where along the wall you measure it, which is the single most
    common way a twin disagrees with a tape. Reporting the angle alone leaves a
    reader with no way to judge whether it matters, so the second return value
    turns it into the thing they would actually observe -- the difference in
    width between the two ends of the longer of the two walls.

    Only pairs already close to facing each other are considered. Two facets of
    a round room are 45 degrees apart and were never a facing pair; measuring
    them here would turn a curved wall into a 45-degree parallelism error, which
    is the check developing an opinion about the building's floor plan instead
    of about the scan.
    """
    limit = -math.cos(math.radians(PARALLEL_PAIR_MAX_DEG))
    worst = float("nan")
    worst_gap = float("nan")
    for i in range(len(fits)):
        for j in range(i + 1, len(fits)):
            a, b = fits[i], fits[j]
            d = float(np.dot(a.normal, b.normal))
            if d >= limit:
                continue
            angle = 180.0 - _angle_deg(a.normal, b.normal)
            if math.isfinite(worst) and angle <= worst:
                continue
            worst = angle
            run = max(a.span_u, b.span_u)
            worst_gap = run * math.tan(math.radians(angle))
    return worst, worst_gap


# ---------------------------------------------------------------------------
# scale
# ---------------------------------------------------------------------------


def _architectural_scale_prior(
    twin: Twin, ceiling_height: float, spacing: float
) -> tuple[float, str]:
    """Independent evidence that this twin is in metres, and what it rests on.

    Architectural priors: ceiling heights cluster hard around 2.2-3.6 m, doors
    are close to 2 m tall, and a handheld scanner resolves somewhere between
    2 mm and 80 mm. A room whose "ceiling" is 285 m tall is in centimetres and
    nothing else. The result is capped below 1.0 because a prior is not a
    measurement, and returns 0 with an empty note when the twin offers no
    evidence at all -- which is a different thing from evidence against, and the
    caller is given the note so it can say which it had.
    """
    scores: list[tuple[float, float]] = []
    notes: list[str] = []
    if math.isfinite(ceiling_height) and ceiling_height > 0:
        scores.append((_plateau(ceiling_height, 1.6, 2.2, 3.6, 6.0), 2.0))
        notes.append(f"a {ceiling_height:.2f} m ceiling")
    doors = [o for o in twin.structure.openings if o.sill_height < 0.15 and o.height > 0]
    if doors:
        tallest = max(o.height for o in doors)
        scores.append((_plateau(tallest, 1.4, 1.9, 2.4, 3.2), 1.0))
        notes.append(f"a {tallest:.2f} m door")
    if math.isfinite(spacing) and spacing > 0:
        scores.append((_plateau(spacing, 0.0005, 0.002, 0.05, 0.30), 1.0))
        notes.append(f"{spacing * 1000:.0f} mm between neighbouring points")
    if not scores:
        return 0.0, ""
    total = sum(w for _, w in scores)
    return float(0.75 * sum(s * w for s, w in scores) / total), ", ".join(notes)


def _scale_confidence(twin: Twin, ceiling_height: float, spacing: float) -> tuple[float, bool]:
    """How sure we are that a metre in the twin is a metre in the room.

    The unit-scale stage is the one that actually knows, so its confidence is
    taken from provenance whenever it recorded one. When it did not -- an
    import that skipped inference, or a twin assembled by hand -- we fall back
    to the architectural priors above and report the answer as inferred, because
    the check message has to say which of the two it had.
    """
    prov = twin.provenance or {}
    steps = prov.get("steps")
    candidates: list[Any] = [prov.get("scale_confidence"), prov.get("unit_scale_confidence")]
    scale = prov.get("scale")
    if isinstance(scale, dict):
        candidates.append(scale.get("confidence"))
    if isinstance(steps, dict):
        for step in steps.values():
            if isinstance(step, dict) and "scale_confidence" in step:
                candidates.append(step["scale_confidence"])
    for value in candidates:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return float(value), True

    return _architectural_scale_prior(twin, ceiling_height, spacing)[0], False


def _plateau(x: float, lo_zero: float, lo_one: float, hi_one: float, hi_zero: float) -> float:
    """Trapezoidal plausibility: 1 inside the prior, tapering to 0 outside it."""
    if x <= lo_zero or x >= hi_zero:
        return 0.0
    if x < lo_one:
        return (x - lo_zero) / (lo_one - lo_zero)
    if x > hi_one:
        return (hi_zero - x) / (hi_zero - hi_one)
    return 1.0


def _boundary_edge_fraction(twin: Twin) -> float:
    """Fraction of mesh edges used by exactly one face.

    A closed solid has none. An interior room scan has a rim of them wherever
    the sweep stopped, so this is the cheapest honest description of how open
    the shell is without pretending to run a manifold check.
    """
    mesh = twin.mesh
    if mesh is None or len(mesh.faces) == 0:
        return float("nan")
    f = mesh.faces
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return float(np.mean(counts == 1))


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------


def assess(twin: Twin, *, grid: Any = None, normals: np.ndarray | None = None, seed: int = 0) -> QAReport:
    """Measure a twin's trustworthiness and return the filled-in `QAReport`.

    `grid` is the occupancy grid from `geom.grid` and `normals` the per-point
    normals from `geom.normals`; both are optional. Passing them makes the
    coverage lattice agree with the grid and lets wall analysis reject window
    reveals by orientation, but their absence only widens error bars, it never
    stops the report -- a degraded pipeline is exactly when QA has to run.

    `seed` fixes the subsample used for neighbour statistics so the same twin
    always produces the same report, byte for byte.
    """
    report = QAReport()
    m = report.metrics
    xyz = twin.points.xyz
    if normals is None:
        normals = twin.points.normals
    if normals is not None and len(normals) != len(xyz):
        normals = None

    m["point_count"] = float(len(xyz))
    m["mesh_face_count"] = float(0 if twin.mesh is None else len(twin.mesh))

    if len(xyz) < 32:
        report.add(
            "input",
            "fail",
            f"The twin holds {len(xyz)} points, which is far too few to measure anything "
            "from; no other check could be run.",
        )
        return report.finalize()

    # -- raw point statistics ------------------------------------------------
    m.update(_neighbour_stats(xyz, seed))
    spacing = m["median_spacing_m"]

    extent = twin.extent
    m["extent_x_m"], m["extent_y_m"], m["extent_z_m"] = (float(v) for v in extent)

    struct = twin.structure
    floor_z = float(struct.floor_z)
    ceiling_z = struct.ceiling_z
    ceiling_height = struct.ceiling_height
    m["ceiling_height_m"] = float("nan") if ceiling_height is None else float(ceiling_height)

    # -- footprint, area, volume --------------------------------------------
    cell = _coverage_cell(spacing, grid)
    lat = _build_lattice(xyz, struct.footprint, cell)
    footprint_source = "structure footprint polygon"
    floor_area = struct.floor_area
    if floor_area <= 0 and lat is not None:
        floor_area = float(lat.inside.sum()) * cell * cell
        footprint_source = "the occupied floor cells, since no footprint polygon was available"
    m["floor_area_m2"] = float(floor_area)

    usable_height = (
        float(ceiling_height)
        if ceiling_height is not None
        else float(xyz[:, 2].max() - floor_z)
    )
    m["volume_m3"] = float(floor_area * usable_height)

    # -- coverage ------------------------------------------------------------
    if lat is not None:
        m.update(_coverage(xyz, lat, floor_z, ceiling_z, struct.openings))
    else:
        m.update(
            {
                "floor_coverage": float("nan"),
                "ceiling_coverage": float("nan"),
                "wall_coverage": float("nan"),
                "hole_fraction": float("nan"),
            }
        )

    # -- gravity -------------------------------------------------------------
    fitted_floor = _measure_floor(xyz, floor_z, spacing, seed)
    if fitted_floor is None:
        declared = next((p for p in struct.planes if p.kind == "floor"), None)
        fitted_floor = None if declared is None else (declared.normal, declared.offset)
    m["gravity_residual_deg"] = (
        float("nan")
        if fitted_floor is None
        else _angle_deg(fitted_floor[0], np.array([0.0, 0.0, 1.0]))
    )

    # -- walls ---------------------------------------------------------------
    #
    # One re-measured set of walls answers every wall-shaped question below, so
    # the flatness, the squareness and the frame can never describe different
    # rooms. A plane the points do not support is dropped here rather than in
    # each check, because a ghost admitted once is a ghost admitted everywhere.
    all_walls = struct.walls()
    walls = _structural_walls(all_walls, spacing)
    m["wall_count"] = float(len(all_walls))
    m["structural_wall_count"] = float(len(walls))

    fits = [
        f
        for f in (_wall_residual_analysis(w, xyz, normals, struct.openings) for w in walls)
        if f is not None
    ]
    m.update(_wall_frame_stats(fits))
    parallelism, parallel_gap = _opposite_wall_parallelism(fits)
    m["opposite_wall_parallelism_deg"] = parallelism
    m["opposite_wall_gap_m"] = parallel_gap

    rms_all = [f.rms for f in fits if math.isfinite(f.rms)]
    drift_all = [f.drift for f in fits if math.isfinite(f.drift)]
    m["wall_planarity_rms_m"] = max(rms_all) if rms_all else float("nan")
    m["drift_estimate_m"] = max(drift_all) if drift_all else float("nan")

    # the frozen frame puts the room's long side on X; a twin turned a quarter
    # turn satisfies every mod-90 measurement above and still breaks it
    ex, ey = m["extent_x_m"], m["extent_y_m"]
    m["long_axis_ratio"] = float(max(ex, ey) / min(ex, ey)) if min(ex, ey) > 0 else float("nan")
    m["long_axis_is_x"] = float(ex >= ey)

    # -- openings ------------------------------------------------------------
    m["opening_count"] = float(len(struct.openings))
    m["opening_area_m2"] = float(sum(o.area for o in struct.openings))

    # -- capture bounds ------------------------------------------------------
    cb = twin.capture_bounds
    m["capture_bounds_area_m2"] = float("nan") if cb is None else float(cb.area)
    m["capture_coverage_ratio"] = (
        float(cb.area / floor_area) if (cb is not None and floor_area > 0) else float("nan")
    )

    # -- scale ---------------------------------------------------------------
    #
    # The prior is computed whether or not the unit-scale stage answered,
    # because when that stage reports a low confidence the reader's next
    # question is whether anything else in the twin corroborates the unit, and
    # a fail message that cannot answer it sends someone back to the building
    # for a rescan they may not need.
    confidence, measured_scale = _scale_confidence(twin, m["ceiling_height_m"], spacing)
    prior, prior_note = _architectural_scale_prior(twin, m["ceiling_height_m"], spacing)
    m["scale_confidence"] = confidence
    m["scale_confidence_prior"] = prior

    m["mesh_boundary_edge_fraction"] = _boundary_edge_fraction(twin)

    _add_checks(
        report,
        twin,
        measured_scale=measured_scale,
        prior_note=prior_note,
        footprint_source=footprint_source,
        cell=cell,
    )
    return report.finalize()


def _fmt(value: float, digits: int = 3, unit: str = "") -> str:
    if value is None or not math.isfinite(value):
        return "unknown"
    return f"{value:.{digits}f}{unit}"


def _add_checks(
    report: QAReport,
    twin: Twin,
    *,
    measured_scale: bool,
    prior_note: str,
    footprint_source: str,
    cell: float,
) -> None:
    """Turn the metrics into verdicts a location manager can act on.

    Every message names the number and the threshold it was compared against,
    because "FAIL: 0.31" tells a reader that something is wrong without telling
    them what to do about it, and the whole value of this report is that the
    person holding it can decide whether to reshoot.
    """
    m = report.metrics

    # -- scale ---------------------------------------------------------------
    conf = m["scale_confidence"]
    provenance_note = (
        "measured by the unit-scale stage"
        if measured_scale
        else "inferred here from architectural priors rather than measured, because the import "
        "recorded no scale confidence"
    )
    if conf < SCALE_CONFIDENCE_FAIL:
        prior = m.get("scale_confidence_prior", float("nan"))
        if measured_scale and prior_note and prior >= SCALE_CONFIDENCE_FAIL:
            corroboration = (
                f" The twin's own architecture is separately consistent with metres ({prior_note}, "
                f"which score {prior:.2f} against the same {SCALE_CONFIDENCE_FAIL:.2f} floor), so a "
                "rescan may not be what this needs -- but a prior is not a measurement, and nothing "
                "here establishes the unit of the source file."
            )
        elif prior_note:
            corroboration = (
                f" Nothing else in the twin rescues it: its own architecture ({prior_note}) scores "
                f"only {prior:.2f} on the same {SCALE_CONFIDENCE_FAIL:.2f} floor."
            )
        else:
            corroboration = (
                " The twin carries no ceiling, no door and no measurable point spacing either, so "
                "there is no second opinion to be had from its architecture."
            )
        report.add(
            "metric_scale",
            "fail",
            f"Scale confidence is {conf:.2f} ({provenance_note}), below the {SCALE_CONFIDENCE_FAIL:.2f} "
            "floor, so no distance in this twin can be quoted in metres until the unit of the "
            f"source file is established.{corroboration}",
        )
    else:
        report.add(
            "metric_scale",
            "pass",
            f"Scale confidence is {conf:.2f} ({provenance_note}), above the {SCALE_CONFIDENCE_FAIL:.2f} "
            "floor, so distances in this twin are in real metres.",
        )

    # -- gravity -------------------------------------------------------------
    grav = m["gravity_residual_deg"]
    if not math.isfinite(grav):
        report.add(
            "gravity",
            "warn",
            "No floor plane could be identified, so the twin's alignment with gravity is "
            "unverified and every height in it is unsupported. This warns rather than passes "
            "because an unmeasurable check is not a satisfied one.",
        )
    elif grav > GRAVITY_FAIL_DEG:
        report.add(
            "gravity",
            "fail",
            f"The floor is {grav:.2f} deg off level, past the {GRAVITY_FAIL_DEG:.2f} deg limit. "
            f"Over a 10 m room that tilts one end by {10.0 * math.tan(math.radians(grav)) * 100:.0f} cm, "
            "so ceiling clearances and eye-line heights will be wrong.",
        )
    elif grav > GRAVITY_WARN_DEG:
        report.add(
            "gravity",
            "warn",
            f"The floor is {grav:.2f} deg off level, past the {GRAVITY_WARN_DEG:.2f} deg warning "
            f"threshold but inside the {GRAVITY_FAIL_DEG:.2f} deg limit; heights are usable but "
            "a long dolly move planned in this twin may creep.",
        )
    else:
        report.add(
            "gravity",
            "pass",
            f"The floor is level to {grav:.2f} deg against the {GRAVITY_WARN_DEG:.2f} deg threshold, "
            "so heights measured in this twin are true verticals.",
        )

    # -- manhattan -----------------------------------------------------------
    man = m["manhattan_residual_deg"]
    walls_used = int(m.get("structural_wall_count", 0))
    walls_found = int(m.get("wall_count", 0))
    if math.isfinite(man):
        report.add(
            "manhattan",
            "info",
            f"The walls sit up to {man:.2f} deg away from a single right-angled frame, measured "
            f"over the {walls_used} of {walls_found} wall planes solid enough to be re-fitted from "
            "the points. This is reported for information only and can never fail, because a bay, "
            "a splay or a curved wall is real architecture rather than a scanning error.",
        )
    else:
        report.add(
            "manhattan",
            "info",
            f"Fewer than two of the {walls_found} wall planes could be re-fitted from the points, "
            "so there is nothing to compare against a right-angled frame. This check never fails; "
            "an angled or round room is legitimate.",
        )

    # -- yaw -----------------------------------------------------------------
    #
    # The frozen frame says the dominant wall runs parallel to +X. Nothing else
    # in this report tests that: `manhattan` compares the walls with each other
    # and is unchanged by rotating the whole room, which is precisely the error
    # being looked for here. The two numbers together are what separates "these
    # walls are not square to the axes", which is ours to fix, from "this room
    # is not rectangular", which is none of our business -- so the offset is
    # only ever read as a fault when most of the wall area agrees on a frame for
    # it to be offset from, and the message says which of the two we believe.
    offset = m["wall_frame_offset_deg"]
    rect = m["wall_rectilinear_fraction"]
    if not math.isfinite(offset):
        report.add(
            "yaw",
            "warn",
            f"Fewer than two of the {walls_found} wall planes could be re-fitted, so the twin's "
            "rotation about the vertical was never verified against its own walls and the +X "
            "convention is unsupported. This warns rather than passes because the check was "
            "skipped, not satisfied.",
        )
    elif not math.isfinite(rect) or rect < RECTILINEAR_MIN_FRACTION:
        report.add(
            "yaw",
            "info",
            f"Only {rect:.0%} of the wall area lies within {AXIS_TOL_DEG:.0f} deg of any single "
            f"right-angled frame, under the {RECTILINEAR_MIN_FRACTION:.0%} needed for the idea of "
            f"'square to the axes' to mean anything, so the {abs(offset):.2f} deg between the walls "
            "and the X/Y axes is not evidence of a bad alignment. We read this as a room that is "
            "genuinely not rectangular, which is the building's business and not ours, so this "
            "check cannot fail.",
        )
    else:
        # what the offset costs whoever measures the room off its own axes
        ex, ey = m["extent_x_m"], m["extent_y_m"]
        skew_mm = 1000.0 * (
            ex * abs(math.cos(math.radians(offset)) - 1.0)
            + ey * abs(math.sin(math.radians(offset)))
        )
        turned = (
            ""
            if m.get("long_axis_is_x", 1.0) > 0.5
            or not math.isfinite(m.get("long_axis_ratio", float("nan")))
            or m["long_axis_ratio"] < LONG_AXIS_RATIO_MIN
            else (
                f" The room is also lying across its axes: its long side is {ey:.2f} m on Y "
                f"against {ex:.2f} m on X, a quarter turn from the convention that the dominant "
                "wall runs along +X, which no modulo-90 measurement above can see. Nothing "
                "measured in twin space is wrong because of it, but a compass heading given for "
                "+X is 90 deg out."
            )
        )
        square = (
            f"{rect:.0%} of the wall area is square to that frame, so this room is rectilinear "
            "and the offset is ours rather than the building's"
        )
        if abs(offset) > YAW_FAIL_DEG:
            report.add(
                "yaw",
                "fail",
                f"The walls are square to each other but sit {abs(offset):.2f} deg away from the "
                f"X/Y axes, past the {YAW_FAIL_DEG:.2f} deg limit: {square}. A plan taken off this "
                f"twin's own axes is about {skew_mm:.0f} mm too big, and every bearing derived from "
                f"+X is out by the same {abs(offset):.2f} deg. Re-run the yaw normalisation.{turned}",
            )
        elif abs(offset) > YAW_WARN_DEG:
            report.add(
                "yaw",
                "warn",
                f"The walls sit {abs(offset):.2f} deg away from the X/Y axes, past the "
                f"{YAW_WARN_DEG:.2f} deg warning threshold but inside the {YAW_FAIL_DEG:.2f} deg "
                f"limit: {square}. Expect about {skew_mm:.0f} mm of inflation in anything measured "
                f"off the axis-aligned bounding box rather than off the walls themselves.{turned}",
            )
        else:
            report.add(
                "yaw",
                "pass",
                f"The dominant walls run parallel to the axes to {abs(offset):.2f} deg, inside the "
                f"{YAW_WARN_DEG:.2f} deg threshold, and {rect:.0%} of the wall area agrees with "
                f"that frame, so the twin is squared up the way the canonical frame promises.{turned}",
            )

    # -- opposite walls ------------------------------------------------------
    #
    # This is the check that catches a room bent into a trapezoid, which the
    # width of a room is the first casualty of. It deliberately stops at 'warn'
    # unless the twin's own walls are also bowed, because a splayed room is a
    # real thing a location scout will walk into and a report that fails it is
    # asserting an opinion about the building it has no way to support.
    par = m["opposite_wall_parallelism_deg"]
    gap_mm = 1000.0 * m.get("opposite_wall_gap_m", float("nan"))
    drift_now = m["drift_estimate_m"]
    bent = math.isfinite(drift_now) and drift_now > DRIFT_WARN_M
    if math.isfinite(rect) and rect < RECTILINEAR_MIN_FRACTION:
        report.add(
            "wall_parallelism",
            "info",
            f"Only {rect:.0%} of the wall area agrees on one right-angled frame, under the "
            f"{RECTILINEAR_MIN_FRACTION:.0%} this check needs, so 'opposite walls should be "
            f"parallel' is not a claim that can be made about this room. The nearest thing to a "
            f"facing pair sits {_fmt(par, 2, ' deg')} from parallel, and in a room that is round, "
            "splayed or many-sided that is the floor plan rather than an error, so this cannot "
            "fail here.",
        )
    elif not math.isfinite(par):
        report.add(
            "wall_parallelism",
            "info",
            f"No two of the {walls_used} re-fitted wall planes face each other within "
            f"{PARALLEL_PAIR_MAX_DEG:.0f} deg, so no width across this room could be checked for "
            "depending on where it is measured.",
        )
    elif par <= PARALLEL_WARN_DEG:
        report.add(
            "wall_parallelism",
            "pass",
            f"Facing walls are parallel to {par:.2f} deg, inside the {PARALLEL_WARN_DEG:.2f} deg "
            f"threshold, so a width measured across this room changes by at most {gap_mm:.0f} mm "
            "depending on where along the wall the tape goes.",
        )
    else:
        cause = (
            f"The walls themselves bow by {drift_now * 1000:.0f} mm, past the "
            f"{DRIFT_WARN_M * 1000:.0f} mm drift threshold, so we read this as the scan bending the "
            "room rather than as the room being that shape"
            if bent
            else f"Each wall is individually flat to {drift_now * 1000:.0f} mm, under the "
            f"{DRIFT_WARN_M * 1000:.0f} mm drift threshold, so we read this as the building really "
            "being splayed rather than as a scanning error"
        )
        report.add(
            "wall_parallelism",
            "fail" if (bent and par > PARALLEL_FAIL_DEG) else "warn",
            f"Two facing walls are {par:.2f} deg from parallel, past the {PARALLEL_WARN_DEG:.2f} deg "
            f"threshold, so the room is a trapezoid: its width differs by about {gap_mm:.0f} mm "
            f"between the two ends of that wall and no single number can be quoted for it. {cause}.",
        )

    # -- ceiling height ------------------------------------------------------
    height = m["ceiling_height_m"]
    if not math.isfinite(height):
        report.add(
            "ceiling_height",
            "info",
            "No ceiling was captured, so ceiling height is unknown and this twin cannot answer "
            "questions about lighting rigs, boom clearance or overhead fixtures.",
        )
    elif height < CEILING_MIN_M or height > CEILING_MAX_M:
        report.add(
            "ceiling_height",
            "warn",
            f"Ceiling height measures {height:.2f} m, outside the plausible {CEILING_MIN_M:.2f}-"
            f"{CEILING_MAX_M:.2f} m range for a habitable room. Either the space really is unusual "
            "or the ceiling plane has latched onto a beam, a soffit or the underside of a mezzanine.",
        )
    else:
        report.add(
            "ceiling_height",
            "pass",
            f"Ceiling height measures {height:.2f} m, inside the plausible {CEILING_MIN_M:.2f}-"
            f"{CEILING_MAX_M:.2f} m range.",
        )

    # -- coverage ------------------------------------------------------------
    floor_cov = m["floor_coverage"]
    detail = (
        f"Ceiling coverage is {_fmt(m['ceiling_coverage'], 2)} and the "
        f"{WALL_BAND_M[0]:.1f}-{WALL_BAND_M[1]:.1f} m wall band is {_fmt(m['wall_coverage'], 2)} filled "
        f"(windows and doors legitimately account for part of any wall shortfall). Coverage was "
        f"measured on {cell * 100:.0f} cm cells over the {footprint_source}."
    )
    if not math.isfinite(floor_cov):
        report.add(
            "coverage",
            "warn",
            "Floor coverage could not be measured because no footprint could be established, so "
            "there is no way to tell how much of this room was actually scanned.",
        )
    elif floor_cov < FLOOR_COVERAGE_FAIL:
        report.add(
            "coverage",
            "fail",
            f"Only {floor_cov:.0%} of the footprint has floor beneath it, below the "
            f"{FLOOR_COVERAGE_FAIL:.0%} minimum. More than half this room was never scanned, so the "
            f"footprint and floor area are guesses. {detail}",
        )
    elif floor_cov < FLOOR_COVERAGE_WARN:
        report.add(
            "coverage",
            "warn",
            f"Floor coverage is {floor_cov:.0%}, under the {FLOOR_COVERAGE_WARN:.0%} target though "
            f"above the {FLOOR_COVERAGE_FAIL:.0%} failure line; expect gaps where furniture or a "
            f"person blocked the sweep. {detail}",
        )
    else:
        report.add(
            "coverage",
            "pass",
            f"Floor coverage is {floor_cov:.0%} of the footprint, above the {FLOOR_COVERAGE_WARN:.0%} "
            f"target. {detail}",
        )

    # -- density -------------------------------------------------------------
    density = m["density_per_m2"]
    spacing = m["median_spacing_m"]
    thin = (math.isfinite(density) and density < DENSITY_WARN_PER_M2) or (
        math.isfinite(spacing) and spacing > SPACING_WARN_M
    )
    if not math.isfinite(density) and not math.isfinite(spacing):
        report.add("density", "warn", "Point spacing could not be measured, so sampling quality is unknown.")
    elif thin:
        report.add(
            "density",
            "warn",
            f"The scan carries {density:,.0f} points per square metre with {spacing * 1000:.0f} mm "
            f"between neighbours, against targets of {DENSITY_WARN_PER_M2:,.0f} per square metre and "
            f"{SPACING_WARN_M * 1000:.0f} mm. Anything smaller than roughly {spacing * 2000:.0f} mm -- a "
            "skirting board, a socket, a door reveal -- may be missing entirely.",
        )
    else:
        report.add(
            "density",
            "pass",
            f"The scan carries {density:,.0f} points per square metre with {spacing * 1000:.0f} mm "
            f"between neighbours, comfortably past the {DENSITY_WARN_PER_M2:,.0f} per square metre and "
            f"{SPACING_WARN_M * 1000:.0f} mm thresholds.",
        )

    # -- drift ---------------------------------------------------------------
    drift = m["drift_estimate_m"]
    if not math.isfinite(drift):
        report.add(
            "drift",
            "warn",
            "No wall was large enough to test for tracking drift, so a slow bend across this twin "
            "would not have been detected. This warns rather than passes because the check was "
            "skipped, not satisfied.",
        )
    elif drift > DRIFT_FAIL_M:
        report.add(
            "drift",
            "fail",
            f"The worst wall bows by {drift * 1000:.0f} mm once sensor noise is averaged out over "
            f"{DRIFT_TILE_M:.1f} m tiles, past the {DRIFT_FAIL_M * 1000:.0f} mm limit. This is the "
            "signature of tracking drift: a physically flat wall came back curved, so any dimension "
            "measured across the room is wrong by about that much. Rescan in a tighter loop.",
        )
    elif drift > DRIFT_WARN_M:
        report.add(
            "drift",
            "warn",
            f"The worst wall bows by {drift * 1000:.0f} mm after noise is averaged out over "
            f"{DRIFT_TILE_M:.1f} m tiles, past the {DRIFT_WARN_M * 1000:.0f} mm warning threshold but "
            f"inside the {DRIFT_FAIL_M * 1000:.0f} mm limit. Treat long measurements as accurate to "
            "centimetres, not millimetres.",
        )
    else:
        report.add(
            "drift",
            "pass",
            f"The worst wall is flat to {drift * 1000:.0f} mm after noise is averaged out over "
            f"{DRIFT_TILE_M:.1f} m tiles, inside the {DRIFT_WARN_M * 1000:.0f} mm threshold, so there "
            "is no sign of tracking drift bending the room.",
        )

    # -- wall planarity ------------------------------------------------------
    rms = m["wall_planarity_rms_m"]
    if not math.isfinite(rms):
        report.add(
            "wall_planarity",
            "info",
            "No wall plane was available to measure surface noise against.",
        )
    elif rms > PLANARITY_WARN_M:
        report.add(
            "wall_planarity",
            "warn",
            f"The worst large wall scatters {rms * 1000:.0f} mm RMS about its own best-fit plane, "
            f"past the {PLANARITY_WARN_M * 1000:.0f} mm threshold. Either the scanner was noisy or "
            "the surface is not flat; expect roughly that much slop when placing anything against "
            "this wall.",
        )
    else:
        report.add(
            "wall_planarity",
            "pass",
            f"The worst large wall sits {rms * 1000:.0f} mm RMS from its own best-fit plane, inside "
            f"the {PLANARITY_WARN_M * 1000:.0f} mm threshold.",
        )

    # -- capture bounds ------------------------------------------------------
    ratio = m["capture_coverage_ratio"]
    if not math.isfinite(ratio):
        report.add(
            "capture_bounds",
            "warn",
            "No capture bounds were established, so we cannot say which parts of this twin were "
            "walked and which were reconstructed from across the room. Every standpoint in it is "
            f"unverified, which is treated the same as falling under the {CAPTURE_RATIO_WARN:.2f} "
            "coverage target.",
        )
    elif ratio < CAPTURE_RATIO_WARN:
        report.add(
            "capture_bounds",
            "warn",
            f"The scanner covered {m['capture_bounds_area_m2']:.1f} m2 of the {m['floor_area_m2']:.1f} m2 "
            f"floor, a ratio of {ratio:.2f} against the {CAPTURE_RATIO_WARN:.2f} target. The rest of "
            "the room was reconstructed from a distance and should not be trusted for tripod "
            "placement or for anything measured at grazing incidence.",
        )
    else:
        report.add(
            "capture_bounds",
            "pass",
            f"The scanner covered {m['capture_bounds_area_m2']:.1f} m2 of the {m['floor_area_m2']:.1f} m2 "
            f"floor, a ratio of {ratio:.2f} against the {CAPTURE_RATIO_WARN:.2f} target, so most "
            "standpoints in this twin were physically occupied during the scan.",
        )

    # -- duplicates ----------------------------------------------------------
    dupes = m["duplicate_fraction"]
    if not math.isfinite(dupes):
        report.add("duplicates", "info", "Duplicate points could not be measured on this cloud.")
    elif dupes > DUPLICATE_WARN:
        report.add(
            "duplicates",
            "warn",
            f"{dupes:.0%} of points sit within 1 mm of another point, past the {DUPLICATE_WARN:.0%} "
            "threshold. That is the signature of a second pass registered on top of the first: the "
            "surface is doubled, so the density figure above overstates how much of the room was "
            "really seen, and any doubled surface is also slightly thickened.",
        )
    else:
        report.add(
            "duplicates",
            "pass",
            f"{dupes:.1%} of points sit within 1 mm of a neighbour, under the {DUPLICATE_WARN:.0%} "
            "threshold, so the density figure is not inflated by a re-registered second pass.",
        )

    # -- openings ------------------------------------------------------------
    count = int(m["opening_count"])
    has_ceiling = math.isfinite(m["ceiling_height_m"])
    if count == 0 and has_ceiling:
        report.add(
            "openings",
            "warn",
            "No windows or doors were found in a room that has a ceiling. A sealed interior box is "
            "possible but rare, so this is more likely a detection failure than a real room, and "
            "any daylight or access planning based on this twin would be wrong.",
        )
    elif count == 0:
        report.add(
            "openings",
            "info",
            "No openings were found. The capture has no ceiling, so this is most likely an open or "
            "partial sweep rather than a windowless room.",
        )
    else:
        report.add(
            "openings",
            "info",
            f"{count} opening(s) totalling {m['opening_area_m2']:.2f} m2 were detected. Openings are "
            "inferred from missing geometry rather than seen directly, so check the count against "
            "what you remember of the room before planning daylight around it.",
        )

    # -- watertight ----------------------------------------------------------
    hole = m["hole_fraction"]
    edge = m.get("mesh_boundary_edge_fraction", float("nan"))
    mesh_note = (
        f" The mesh leaves {edge:.0%} of its edges open at a boundary."
        if math.isfinite(edge)
        else " No mesh was reconstructed, so this describes the point cloud only."
    )
    report.add(
        "watertight",
        "info",
        f"An interior scan is a shell, not a solid, and is never watertight by design: it is the "
        f"inside surface of a room with a hole wherever the sweep stopped. Ignoring windows and "
        f"doors, {_fmt(hole, 2)} of the expected floor, ceiling and wall surface has no points on "
        f"it at all.{mesh_note}",
    )


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

_ANSI = {
    "pass": "\x1b[32m",
    "info": "\x1b[36m",
    "warn": "\x1b[33m",
    "fail": "\x1b[31m",
    "bold": "\x1b[1m",
    "reset": "\x1b[0m",
}

_METRIC_FORMAT: dict[str, tuple[int, str]] = {
    "point_count": (0, ""),
    "mesh_face_count": (0, ""),
    "opening_count": (0, ""),
    "density_per_m2": (0, " /m2"),
    "median_spacing_m": (4, " m"),
    "extent_x_m": (3, " m"),
    "extent_y_m": (3, " m"),
    "extent_z_m": (3, " m"),
    "ceiling_height_m": (3, " m"),
    "floor_area_m2": (2, " m2"),
    "volume_m3": (2, " m3"),
    "opening_area_m2": (2, " m2"),
    "capture_bounds_area_m2": (2, " m2"),
    "gravity_residual_deg": (2, " deg"),
    "manhattan_residual_deg": (2, " deg"),
    "wall_frame_offset_deg": (2, " deg"),
    "opposite_wall_parallelism_deg": (2, " deg"),
    "opposite_wall_gap_m": (3, " m"),
    "wall_planarity_rms_m": (4, " m"),
    "drift_estimate_m": (4, " m"),
    "wall_count": (0, ""),
    "structural_wall_count": (0, ""),
    "long_axis_is_x": (0, ""),
}


def _metric_text(name: str, value: float) -> str:
    digits, unit = _METRIC_FORMAT.get(name, (3, ""))
    if value is None or not math.isfinite(value):
        return "unknown"
    if digits == 0:
        return f"{value:,.0f}{unit}"
    return f"{value:,.{digits}f}{unit}"


def format_report(report: QAReport, *, width: int = 78, color: bool = True) -> str:
    """Render a `QAReport` for a terminal.

    Checks are grouped pass, info, warn, fail so the eye travels down to the
    problems and stops there; the reverse order would bury a failure under a
    wall of green. Colour is dropped when `color` is false or when NO_COLOR is
    set in the environment, since this output is routinely piped into a log or
    a ticket where escape codes are noise.
    """
    width = max(48, int(width))
    use_color = bool(color) and not os.environ.get("NO_COLOR")

    def paint(text: str, key: str) -> str:
        if not use_color:
            return text
        return f"{_ANSI[key]}{text}{_ANSI['reset']}"

    verdict = report.verdict
    counts = {s: sum(1 for c in report.checks if c["status"] == s) for s in _STATUS_ORDER}
    tally = ", ".join(f"{counts[s]} {s}" for s in _STATUS_ORDER)

    lines: list[str] = []
    rule = "=" * width
    lines.append(rule)
    banner = f"  VERDICT: {verdict.upper()}"
    lines.append(paint(banner, verdict if verdict in _ANSI else "bold"))
    lines.append(f"  {tally}")
    lines.append(rule)
    lines.append("")

    label_w = max([len(c["name"]) for c in report.checks], default=8) + 2
    for status in _STATUS_ORDER:
        group = [c for c in report.checks if c["status"] == status]
        if not group:
            continue
        lines.append(paint(f"{status.upper()} ({len(group)})", status))
        for check in group:
            body = textwrap.wrap(check["message"], width=max(20, width - label_w - 4)) or [""]
            head = f"  {check['name']:<{label_w}}"
            lines.append(f"{head}{body[0]}")
            for extra in body[1:]:
                lines.append(f"{' ' * (label_w + 2)}{extra}")
        lines.append("")

    lines.append(paint("METRICS", "bold"))
    items = [(k, _metric_text(k, v)) for k, v in report.metrics.items()]
    if items:
        name_w = max(len(k) for k, _ in items)
        value_w = max(len(v) for _, v in items)
        col = 2 + name_w + 2 + value_w
        columns = 2 if 2 * col <= width else 1
        rows = math.ceil(len(items) / columns)
        for i in range(rows):
            cells = []
            for j in range(i, len(items), rows):
                k, v = items[j]
                cells.append(f"  {k:<{name_w}}  {v:>{value_w}}".ljust(col))
            lines.append("".join(cells).rstrip())
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# measuring against the real room
# ---------------------------------------------------------------------------


def verify_measurement(
    twin: Twin, a: np.ndarray, b: np.ndarray, *, seed: int = 0
) -> dict[str, Any]:
    """Distance between two twin-space points, with an honest error bar.

    This is the function that settles an argument with a tape measure, so it
    has to quote an uncertainty rather than six decimal places of a number that
    is only good to a centimetre. Three error sources are combined in
    quadrature:

      * where each endpoint really is, which is bounded by half the median
        point spacing -- you cannot land on a surface more precisely than the
        surface is sampled;
      * tracking drift, scaled by how far apart the endpoints are relative to
        the room, because a bow in the room barely affects two points a hand's
        width apart and affects opposite corners fully;
      * residual uncertainty in the unit scale, which is proportional to the
        distance itself and is the one term that gets worse the longer the
        measurement.

    Every one of those three is *measured*, off `twin.qa.metrics` when a report
    has been run and off the twin's own geometry when it has not, and a term
    that cannot be measured is returned as unknown instead of as a plausible
    number. An earlier version filled the gaps with 20 mm of spacing, 20 mm of
    drift and a scale confidence of 0.5, and returned them in a dict with no
    field saying they were invented -- an error bar with nothing behind it,
    quoted to a person holding a tape, which is the exact failure this module
    exists to prevent. Measuring here costs a plane fit per wall on a twin that
    never had a QA pass; that is worth paying once for an answer someone is
    about to act on.

    `uncertainty_m` is therefore NaN when any term is unmeasurable, with
    `uncertainty_known` false, `unmeasured_terms` naming what is missing and
    `uncertainty_floor_m` giving the error bar implied by the terms that did
    survive -- a floor, never a total, because an unknown term can only make it
    worse.

    `within_capture_bounds` is true only when *both* endpoints lie inside the
    area the scanner actually walked. Outside it the geometry is extrapolation
    and the error bar above does not apply, which is precisely the case where
    quoting a confident number would be dishonest.
    """
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    distance = float(np.linalg.norm(b - a))

    metrics = dict(twin.qa.metrics or {})
    xyz = twin.points.xyz

    # -- spacing ---------------------------------------------------------
    spacing = float(metrics.get("median_spacing_m", float("nan")))
    spacing_basis = "measured by the QA report"
    if not math.isfinite(spacing) and len(xyz) >= 8:
        spacing = _neighbour_stats(xyz, seed)["median_spacing_m"]
        spacing_basis = "measured here from the point cloud"
    if not math.isfinite(spacing):
        spacing_basis = "unknown: too few points to measure a spacing"

    # -- drift -----------------------------------------------------------
    drift = float(metrics.get("drift_estimate_m", float("nan")))
    drift_basis = "measured by the QA report"
    if not math.isfinite(drift) and math.isfinite(spacing):
        walls = _structural_walls(twin.structure.walls(), spacing)
        measured = [
            f.drift
            for f in (
                _wall_residual_analysis(w, xyz, twin.points.normals, twin.structure.openings)
                for w in walls
            )
            if f is not None and math.isfinite(f.drift)
        ]
        if measured:
            drift = max(measured)
            drift_basis = f"measured here from {len(measured)} wall plane(s)"
    if not math.isfinite(drift):
        drift_basis = "unknown: no wall was large enough to measure a bow across"

    # -- scale -----------------------------------------------------------
    confidence = float(metrics.get("scale_confidence", float("nan")))
    scale_basis = "measured by the unit-scale stage"
    if not math.isfinite(confidence):
        ceiling = twin.structure.ceiling_height
        confidence, from_stage = _scale_confidence(
            twin, float("nan") if ceiling is None else float(ceiling), spacing
        )
        scale_basis = (
            "measured by the unit-scale stage"
            if from_stage
            else "inferred here from architectural priors rather than measured"
        )
        if not from_stage and confidence <= 0.0:
            confidence = float("nan")
            scale_basis = "unknown: nothing in this twin says what a metre is"

    diagonal = float(np.linalg.norm(twin.extent))
    span = 1.0 if diagonal <= 0 else min(1.0, distance / diagonal)

    terms = {
        "endpoint_m": math.sqrt(2.0) * 0.5 * spacing,
        "drift_m": drift * span,
        "scale_m": 0.01 * (1.0 - confidence) * distance,
    }
    unmeasured = [k for k, v in terms.items() if not math.isfinite(v)]
    known = [v for v in terms.values() if math.isfinite(v)]
    combined = math.sqrt(sum(v * v for v in known)) if known else float("nan")

    # A tolerance quoted finer than a millimetre is theatre for a room scanned
    # at centimetre spacing, so both numbers are rounded to millimetres and the
    # error bar is never allowed below one.
    def to_mm(value: float) -> float:
        return float("nan") if not math.isfinite(value) else max(0.001, round(value, 3))

    floor_m = to_mm(combined)
    return {
        "distance_m": round(distance, 3),
        "uncertainty_m": float("nan") if unmeasured else floor_m,
        "uncertainty_known": not unmeasured,
        "uncertainty_floor_m": floor_m,
        "unmeasured_terms": unmeasured,
        "uncertainty_terms_m": {k: to_mm(v) for k, v in terms.items()},
        "uncertainty_basis": {
            "endpoint_m": spacing_basis,
            "drift_m": drift_basis,
            "scale_m": scale_basis,
        },
        "within_capture_bounds": _inside_bounds(twin, a, b),
    }


def _inside_bounds(twin: Twin, a: np.ndarray, b: np.ndarray) -> bool:
    """Whether both endpoints lie in the area the scanner physically walked.

    False when there are no capture bounds at all: an unproven standpoint and a
    standpoint proven to be outside the sweep are equally unsafe to quote a
    distance from, and the two must not be told apart by an optimistic default.
    """
    cb = twin.capture_bounds
    if cb is None:
        return False
    return bool(np.all(cb.contains(np.stack([a[:2], b[:2]]))))
