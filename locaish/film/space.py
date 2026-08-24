"""Where a body, a tripod or a dolly can actually be, and what can be seen from there.

A location scout's real work is spatial bookkeeping: standing in a room and
knowing that the camera cannot go *there* because of the radiator, that the
operator cannot back up past *this* line, that the dolly will not turn at that
end. All of it is answerable from geometry the twin already holds, and none of
it is answerable from a photograph, which is why the trip happens.

Three things are computed here and everything else in `film` is built on them.

**Headroom** -- for every square of floor, the height of clear air above it.
Not the ceiling height: the underside of whatever is actually over that square,
which may be a beam, a shelf or a light fitting.

**Clearance** -- for every square of floor, how far to the nearest obstruction
at working height. This is what decides whether a footprint fits, and it is
measured in the horizontal plane through the equipment rather than over the
whole column, because a dolly does not care about a picture rail.

**Visibility** -- whether a straight line between two points passes through
anything. Cheap, and the basis of every sightline question: can this camera see
that actor, is the doorway in shot, does the pillar cut the eyeline.

The load-bearing honesty here is `capture_bounds`. The twin knows where the
scanner actually walked, and everything outside it is reconstruction rather
than measurement. A position proposed out there is a guess dressed as a
measurement, so placements are marked with whether they stand on ground the
sweep really covered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from ..types import Twin

# Floor resolution for every map here. Ten centimetres is finer than the
# decisions being made -- nobody positions a dolly to the centimetre on a
# recce -- and coarse enough that a room is a few thousand cells.
DEFAULT_CELL_M = 0.10

# The band of height a standing person occupies, used for clearance. Measuring
# obstruction over the whole floor-to-ceiling column would call a room with a
# picture rail impassable; measuring it at ankle height would miss a table.
BODY_BAND_M = (0.30, 1.80)

# A person needs this much clear air overhead to stand up in, and this much
# radius to stand in at all.
STANDING_HEADROOM_M = 1.90
STANDING_RADIUS_M = 0.28

# Occupancy: a floor cell counts as blocked when this many points sit in the
# band above it. One stray reconstruction floater is not a chair.
MIN_POINTS_FOR_OBSTRUCTION = 4


@dataclass
class FloorMaps:
    """Rasters over the room's floor, all sharing one grid.

    `origin` is the XY of cell (0, 0)'s corner and `cell` its size, so cell
    (i, j) covers `origin + (i, j) * cell` to `origin + (i+1, j+1) * cell`.
    """

    origin: np.ndarray            # (2,)
    cell: float
    inside: np.ndarray            # (nx, ny) bool: within the room footprint
    surveyed: np.ndarray          # (nx, ny) bool: within the capture bounds
    headroom_m: np.ndarray        # (nx, ny) float: clear height above the floor
    clearance_m: np.ndarray       # (nx, ny) float: distance to nearest obstruction
    floor_rise_m: np.ndarray      # (nx, ny) float: floor height above floor_z, NaN if unseen
    floor_z: float = 0.0
    ceiling_z: float | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.inside.shape[0]), int(self.inside.shape[1]))

    def world_of(self, ij) -> np.ndarray:
        """Centre of a cell, or of an array of cells, in twin XY."""
        idx = np.asarray(ij, dtype=np.float64).reshape(-1, 2)
        return self.origin + (idx + 0.5) * self.cell

    def index_of(self, xy) -> np.ndarray:
        pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        return np.floor((pts - self.origin) / self.cell).astype(np.int64)

    def sample(self, grid: np.ndarray, xy) -> np.ndarray:
        """Read a map at world XY, clamped to the grid."""
        ij = self.index_of(xy)
        ij[:, 0] = np.clip(ij[:, 0], 0, self.shape[0] - 1)
        ij[:, 1] = np.clip(ij[:, 1], 0, self.shape[1] - 1)
        return grid[ij[:, 0], ij[:, 1]]

    def standable(self, headroom_m: float = STANDING_HEADROOM_M,
                  radius_m: float = STANDING_RADIUS_M) -> np.ndarray:
        """Cells a person could stand on: inside, tall enough, wide enough."""
        return self.inside & (self.headroom_m >= headroom_m) & (self.clearance_m >= radius_m)


def floor_maps(twin: Twin, *, cell: float = DEFAULT_CELL_M) -> FloorMaps:
    """Raster the room into the maps every other question is answered from."""
    xyz = np.asarray(twin.points.xyz, dtype=np.float64)
    if len(xyz) == 0:
        raise ValueError("twin has no points")
    floor_z = float(twin.structure.floor_z)
    ceiling_z = twin.structure.ceiling_z

    lo = xyz[:, :2].min(axis=0)
    hi = xyz[:, :2].max(axis=0)
    dims = np.maximum(np.ceil((hi - lo) / cell).astype(int), 1)
    nx, ny = int(dims[0]), int(dims[1])

    ij = np.floor((xyz[:, :2] - lo) / cell).astype(np.int64)
    ij[:, 0] = np.clip(ij[:, 0], 0, nx - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, ny - 1)
    flat = ij[:, 0] * ny + ij[:, 1]
    height = xyz[:, 2] - floor_z

    # What counts as "enough points to be an obstacle" has to scale with how
    # densely this twin was sampled. Four points is right for a LiDAR export at
    # a few hundred points per cell; a stereo reconstruction packs thousands
    # into the same cell and scatters a percent of them as mid-air speckle, so
    # a fixed four would wall off the whole room.
    occupied = np.bincount(flat, minlength=nx * ny)
    per_cell = float(np.median(occupied[occupied > 0])) if (occupied > 0).any() else 0.0
    min_pts = max(MIN_POINTS_FOR_OBSTRUCTION, int(0.02 * per_cell))

    inside = _footprint_mask(twin, lo, cell, nx, ny)
    surveyed = _surveyed_mask(twin, lo, cell, nx, ny)
    headroom = _headroom(flat, height, nx, ny, ceiling_z, floor_z, min_pts)
    clearance = _clearance(flat, height, nx, ny, cell, min_pts)
    rise = _floor_rise(flat, height, nx, ny)

    return FloorMaps(
        origin=lo,
        cell=float(cell),
        inside=inside,
        surveyed=surveyed,
        headroom_m=headroom,
        clearance_m=clearance,
        floor_rise_m=rise,
        floor_z=floor_z,
        ceiling_z=None if ceiling_z is None else float(ceiling_z),
    )


def _footprint_mask(twin: Twin, lo, cell, nx, ny) -> np.ndarray:
    """Cells inside the room outline, by even-odd crossing of the polygon."""
    poly = twin.structure.footprint
    if poly is None or len(poly) < 3:
        return np.ones((nx, ny), dtype=bool)
    xs = lo[0] + (np.arange(nx) + 0.5) * cell
    ys = lo[1] + (np.arange(ny) + 0.5) * cell
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return _points_in_polygon(gx.ravel(), gy.ravel(), poly).reshape(nx, ny)


def _points_in_polygon(px, py, poly) -> np.ndarray:
    """Ray-crossing test, vectorised over the query points."""
    poly = np.asarray(poly, dtype=np.float64)
    inside = np.zeros(px.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        straddles = (y0 > py) != (y1 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x1 - x0) * (py - y0) / np.where(y1 == y0, np.nan, y1 - y0) + x0
        crosses = straddles & np.isfinite(xint) & (px < xint)
        inside ^= crosses
    return inside


def _surveyed_mask(twin: Twin, lo, cell, nx, ny) -> np.ndarray:
    bounds = twin.capture_bounds
    if bounds is None or bounds.hull_xy is None or len(bounds.hull_xy) < 3:
        return np.zeros((nx, ny), dtype=bool)
    xs = lo[0] + (np.arange(nx) + 0.5) * cell
    ys = lo[1] + (np.arange(ny) + 0.5) * cell
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return _points_in_polygon(gx.ravel(), gy.ravel(), bounds.hull_xy).reshape(nx, ny)


def _headroom(flat, height, nx, ny, ceiling_z, floor_z, min_pts: int = MIN_POINTS_FOR_OBSTRUCTION) -> np.ndarray:
    """Clear height over each cell: the lowest thing above head height.

    Looks only above `BODY_BAND_M[1]`, because the question is what a standing
    person's head meets. A table at 0.75 m blocks a dolly and is not a headroom
    problem, and conflating the two makes every furnished room read as a crawl
    space.
    """
    ceiling = (
        float(ceiling_z - floor_z) if ceiling_z is not None else float(np.percentile(height, 99.5))
    )
    out = np.full(nx * ny, ceiling, dtype=np.float64)

    overhead = height > BODY_BAND_M[1]
    if overhead.any():
        f = flat[overhead]
        h = height[overhead]
        counts = np.bincount(f, minlength=nx * ny)
        # Lowest overhead return per cell, via a sort-and-take-first.
        order = np.lexsort((h, f))
        fs, hs = f[order], h[order]
        first = np.concatenate([[True], fs[1:] != fs[:-1]])
        cells, lowest = fs[first], hs[first]
        solid = counts[cells] >= min_pts
        out[cells[solid]] = np.minimum(out[cells[solid]], lowest[solid])

    return out.reshape(nx, ny)


def _floor_rise(flat, height, nx, ny) -> np.ndarray:
    """How high the actual floor sits over each cell, relative to the fitted plane.

    Taken as a low percentile of the returns in the ankle band rather than the
    minimum, because the minimum is a reconstruction floater under the boards.
    NaN where no returns fell low enough to see the floor at all -- under a sofa,
    say -- and NaN is the right answer there rather than zero: track cannot be
    levelled against a floor nobody measured.
    """
    out = np.full(nx * ny, np.nan, dtype=np.float64)
    low = height < BODY_BAND_M[0]
    if not low.any():
        return out.reshape(nx, ny)
    f, h = flat[low], height[low]
    order = np.lexsort((h, f))
    fs, hs = f[order], h[order]
    starts = np.concatenate([[0], np.flatnonzero(fs[1:] != fs[:-1]) + 1])
    ends = np.concatenate([starts[1:], [len(fs)]])
    counts = ends - starts
    keep = counts >= MIN_POINTS_FOR_OBSTRUCTION
    # 20th percentile within each cell: above the floaters, below the clutter.
    picks = starts[keep] + (counts[keep] * 0.2).astype(np.int64)
    out[fs[starts[keep]]] = hs[np.clip(picks, 0, len(hs) - 1)]
    return out.reshape(nx, ny)


def _clearance(flat, height, nx, ny, cell, min_pts: int = MIN_POINTS_FOR_OBSTRUCTION) -> np.ndarray:
    """Distance from each cell to the nearest thing standing in the body band."""
    band = (height >= BODY_BAND_M[0]) & (height <= BODY_BAND_M[1])
    blocked = np.zeros(nx * ny, dtype=bool)
    if band.any():
        counts = np.bincount(flat[band], minlength=nx * ny)
        blocked = counts >= min_pts
    blocked = blocked.reshape(nx, ny)
    # A single blocked cell with no blocked neighbour is reconstruction
    # speckle, not furniture -- nothing a camera crew cares about is 10 cm
    # across in every direction. A neighbour test rather than a morphological
    # opening, because a wall shows up here as a one-cell-thick *line*, and an
    # opening would erase the wall along with the speckle; every cell of a
    # line has neighbours, an isolated fleck has none.
    if blocked.any():
        neighbours = ndimage.convolve(
            blocked.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant"
        ) - blocked.astype(np.uint8)
        blocked &= neighbours > 0
    if not blocked.any():
        return np.full((nx, ny), float(max(nx, ny)) * cell)
    return ndimage.distance_transform_edt(~blocked, sampling=(cell, cell)).astype(np.float64)


# ---------------------------------------------------------------------------
# fitting things into the room
# ---------------------------------------------------------------------------


@dataclass
class Placement:
    """Somewhere a specific piece of gear will physically go."""

    xy: np.ndarray
    clearance_m: float
    headroom_m: float
    surveyed: bool
    notes: list[str] = field(default_factory=list)


def fits(maps: FloorMaps, footprint_m: tuple[float, float], xy) -> bool:
    """Whether a footprint centred at `xy` clears everything in the body band.

    Tested against the clearance map with the footprint's circumscribed radius,
    which is orientation-free: a rectangle that fits at *some* rotation fits
    inside a circle of that radius, and a scout who has to rotate the dolly to
    the degree to make it fit does not have room for the dolly.
    """
    w, l = footprint_m
    radius = 0.5 * float(np.hypot(w, l))
    return bool(maps.sample(maps.clearance_m, xy)[0] >= radius)


def fits_mask(
    maps: FloorMaps,
    footprint_m: tuple[float, float],
    *,
    min_headroom_m: float = STANDING_HEADROOM_M,
    surveyed_only: bool = False,
) -> np.ndarray:
    """Boolean map of every cell the gear fits in.

    Separate from `placements` so that counting does not require building a
    dataclass per cell -- a large room has tens of thousands of them, and the
    report only ever wants the count and the best few.
    """
    radius = 0.5 * float(np.hypot(*footprint_m))
    ok = maps.inside & (maps.clearance_m >= radius) & (maps.headroom_m >= min_headroom_m)
    if surveyed_only:
        ok &= maps.surveyed
    return ok


def placements(
    maps: FloorMaps,
    footprint_m: tuple[float, float],
    *,
    min_headroom_m: float = STANDING_HEADROOM_M,
    surveyed_only: bool = False,
    limit: int | None = None,
) -> list[Placement]:
    """Every cell the gear fits in, most clearance first.

    Sorted by clearance because the best position for a piece of equipment is
    the one with the most room around it -- that is where the crew can work,
    and where a small error in the twin cannot make the answer wrong.
    """
    ok = fits_mask(maps, footprint_m, min_headroom_m=min_headroom_m,
                   surveyed_only=surveyed_only)
    idx = np.argwhere(ok)
    if not len(idx):
        return []
    order = np.argsort(-maps.clearance_m[ok])
    idx = idx[order]
    if limit:
        idx = idx[:limit]

    out: list[Placement] = []
    for i, j in idx:
        surveyed = bool(maps.surveyed[i, j])
        notes: list[str] = []
        if not surveyed:
            notes.append(
                "outside the surveyed area: this position rests on reconstructed "
                "geometry rather than on anything the sweep actually saw"
            )
        out.append(
            Placement(
                xy=maps.world_of((i, j))[0],
                clearance_m=float(maps.clearance_m[i, j]),
                headroom_m=float(maps.headroom_m[i, j]),
                surveyed=surveyed,
                notes=notes,
            )
        )
    return out


# ---------------------------------------------------------------------------
# sightlines
# ---------------------------------------------------------------------------


def occupancy(twin: Twin, *, cell: float = 0.08) -> tuple[np.ndarray, np.ndarray, float]:
    """A 3D occupancy volume for line-of-sight tests. Returns (grid, origin, cell)."""
    xyz = np.asarray(twin.points.xyz, dtype=np.float64)
    lo = xyz.min(axis=0) - cell
    dims = np.maximum(np.ceil((xyz.max(axis=0) + cell - lo) / cell).astype(int), 1)
    idx = np.floor((xyz - lo) / cell).astype(np.int64)
    idx = np.clip(idx, 0, dims - 1)
    flat = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    counts = np.bincount(flat, minlength=int(np.prod(dims)))
    grid = (counts >= MIN_POINTS_FOR_OBSTRUCTION).reshape(tuple(dims))
    return grid, lo, float(cell)


def visible(grid: np.ndarray, origin: np.ndarray, cell: float, a, b, *, slack: float = 0.12) -> bool:
    """Whether the segment a->b is clear of occupied voxels.

    `slack` is how much of each end to ignore. Both endpoints are usually *on*
    something -- a camera on a tripod, an actor's head on an actor -- and a
    strict test would report every sightline blocked by its own subject.
    """
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(b - a))
    if length < 1e-6:
        return True
    steps = max(2, int(np.ceil(length / (cell * 0.5))))
    t = np.linspace(0.0, 1.0, steps)
    keep = (t * length > slack) & ((1.0 - t) * length > slack)
    pts = a[None, :] + t[keep, None] * (b - a)[None, :]
    if not len(pts):
        return True
    idx = np.floor((pts - origin) / cell).astype(np.int64)
    hi = np.array(grid.shape) - 1
    inb = np.all((idx >= 0) & (idx <= hi), axis=1)
    idx = idx[inb]
    if not len(idx):
        return True
    return not bool(grid[idx[:, 0], idx[:, 1], idx[:, 2]].any())
