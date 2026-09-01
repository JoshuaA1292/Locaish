"""Reading a room off the evidence rather than off the returns.

Plane RANSAC over a point cloud asks "where are there enough points to fit a
surface?", and on a LiDAR export that question is a good proxy for "where are
the walls?". On a video reconstruction it is not, and the gap between the two
is where this module lives.

Dense stereo can only triangulate texture. A kitchen filmed on a phone comes
back with every jar and handle in it and almost nothing on the white cabinet
doors, the white walls or the plain ceiling -- the very surfaces that *are* the
room. Fitting planes to what is left finds the clutter: a real capture measured
here returned six coplanar slabs all labelled floor, twelve walls whose largest
was 0.79 m2 in a room whose walls are seven, no ceiling at all, and no
doorway, from a sweep that walked through one.

The evidence for those surfaces was in the data the whole time, in a form
RANSAC cannot read. Every camera-to-point ray proves the corridor it travelled
along was empty, and the union of those corridors is a solid volume whose
boundary is the room -- including the parts of it no stereo pair could ever
match, because **a blank white wall is not a place with no evidence, it is the
place where free space stops**. `geom.infill` already carves that volume; it
was being used to filter floaters and to close a mesh. Here it becomes the
thing the architecture is read from.

The division of labour is the whole design, and it is deliberate:

    the carve says *which* surfaces exist and roughly where
    the points say *exactly* where each one is

Carving alone is capped at the voxel -- 5 cm, where the dimension tolerance
this pipeline is held to is 15 mm. So every wall the carve discovers is refit
by total least squares to the returns that lie along it, which recovers the
millimetres; and a wall with too few returns to refit keeps its carved position
and is marked inferred, carrying a voxel-sized error bar instead of a
fabricated one. Nothing here lets an inferred surface pass for a measured one.

What this module will not do is invent a surface the sweep never bounded. A
ceiling is reported only when returns support it at the height the carve says
free space stops; an open-topped capture still comes back with `ceiling_z`
None, because the top of a swept volume is a record of where the phone was
pointed and not of where the architecture is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from ..types import Opening, Plane, PointCloud
from . import infill
from .planes import plane_frame

# The voxel the room is read at when no grid is offered. Matched to
# `geom.grid.ROOM_VOXEL` on purpose: the ingest pipeline has usually built one
# already at that size and reusing it saves a pass over the cloud.
ROOM_VOXEL = 0.05

# A column counts as room interior when the carve cleared this much height in
# it. One metre is chosen to sit above the furniture and below the architecture:
# the free space over a worktop is half a metre, the free space in a room a
# person walked through is the better part of two.
INTERIOR_MIN_FREE_M = 1.0

# Walls are looked for in a slab that excludes this much at the floor and at the
# ceiling. Skirting, cornice and the floor's own returns all sit within a
# handful of centimetres of the horizontal surfaces and, projected downward,
# they smear the wall evidence into a band the line fit then has to fight.
SLAB_MARGIN_M = 0.25

# A column is wall evidence when this many voxels in the slab are occupied.
# This is the discriminator that separates architecture from contents without
# knowing what either is: a worktop, a shelf edge or a table top occupies one
# or two voxels of vertical extent, and a wall occupies the whole slab. Three
# voxels is 15 cm of standing surface.
WALL_MIN_VERTICAL_VOXELS = 3

# How far outside the carved interior wall evidence is still taken to be this
# room's boundary. Beyond half a metre it is the neighbouring room, seen
# through a doorway.
WALL_SEARCH_M = 0.60

# Distance from a candidate line within which a cell is its inlier. Six
# centimetres is a little over a voxel, so a wall that crosses a cell boundary
# is not split in two by the grid it was found on.
WALL_INLIER_M = 0.06

# The shortest run of evidence that is allowed to become a wall, and the gap
# along a line that splits one run into two. A 0.7 m wall is a jamb or a pier;
# below that the fit is reading furniture. The gap is set above a doorway's
# reveal depth and below a doorway's width, so a wall interrupted by a door
# stays one wall.
WALL_MIN_LENGTH_M = 0.70
WALL_GAP_M = 0.90

# Two lines this close, with normals this well aligned, are the same wall found
# twice -- once on each face of it, or once per RANSAC pass through a thick
# stereo surface.
WALL_MERGE_OFFSET_M = 0.22
WALL_MERGE_ANGLE_DEG = 10.0

# Refitting a carved wall to the returns along it needs this many of them, and
# is rejected if they scatter further than this about the line or pull it
# further than this off the carved direction. The point count is low on purpose:
# a textureless wall that caught two hundred returns along a skirting board
# still locates itself far better than the voxel does.
REFIT_MIN_POINTS = 150
REFIT_MAX_RMS_M = 0.06
REFIT_MAX_TURN_DEG = 8.0

# The ceiling plateau: how flat the top of the carve has to be to be called a
# surface, over how much area, and how much of the interior has to carry
# returns at that height before it is a measurement rather than the top of the
# sweep.
CEILING_PLATEAU_TOL_M = 0.08
CEILING_MIN_AREA_M2 = 1.0
# What fraction of the swept room the plateau has to cap. This is the test
# that separates a ceiling from a camera cone: a real ceiling stops the free
# space in nearly every column at the same height, while a sweep that only
# glanced upward leaves a top surface that climbs smoothly with wherever the
# phone was aimed. Measured on a kitchen whose ceiling was filmed but returned
# almost nothing: the best candidate capped 21% of the columns, and the
# distribution of column tops ran 1.94 m at the median to 2.44 m at the
# maximum with no step in it anywhere. There was no ceiling in that data to
# find, and reporting one would have been invention.
CEILING_PLATEAU_FRACTION = 0.45
# Returns at the plateau height, standing over the interior, promote it from
# inferred to measured.
CEILING_MIN_RETURN_COVER = 0.12
# ... and this much says the cap is at least made of something, which is what
# separates an inferrable ceiling from the ragged top of a truncated capture.
CEILING_MIN_INFERRED_COVER = 0.01
CEILING_MIN_HEIGHT_M = 1.80
CEILING_MAX_HEIGHT_M = 6.00

# Snapping a rastered footprint onto the fitted walls. An edge is claimed by a
# wall when it is no further than this and no more turned than this; the
# distance is three cells, which is the most a marching-squares staircase can
# stand off the line it is approximating.
SNAP_MAX_DIST_M = 0.18
SNAP_MAX_ANGLE_DEG = 14.0

# A corner is completed by intersecting two walls only across a gap shorter
# than this. Beyond it the outline is not a missed corner but a real feature --
# the mouth of an alcove, the open side of an L -- and cutting across it would
# annex floor the room does not have.
CORNER_MAX_GAP_M = 1.20


@dataclass
class WallLine:
    """One wall, as a line in plan with a vertical extent and a provenance.

    `normal` and `offset` describe the line as `dot(normal, p) = offset` with
    the normal pointing into the room, matching `Plane`'s convention so that
    the sign of a distance means the same thing at every level of the stack.
    `t0`/`t1` are the extent along the wall, measured on the in-plane direction
    `cross(up, normal)`.
    """

    normal: np.ndarray
    offset: float
    t0: float
    t1: float
    cells: int = 0
    points: int = 0
    rms: float = 0.0
    source: str = "carve"  # carve | returns

    @property
    def length(self) -> float:
        return float(self.t1 - self.t0)

    @property
    def direction(self) -> np.ndarray:
        return np.array([-self.normal[1], self.normal[0]])

    @property
    def measured(self) -> bool:
        return self.source == "returns"


@dataclass
class RoomFit:
    """The room the evidence supports, and how well it supports each part."""

    floor_z: float
    ceiling_z: float | None
    walls: list[Plane] = field(default_factory=list)
    lines: list[WallLine] = field(default_factory=list)
    interior_area: float = 0.0
    # The footprint as the boundary of the interior cells of the wall-line
    # arrangement -- straight edges and closed corners by construction, never
    # a raster contour. None when the cells could not be labelled, in which
    # case the caller must fall back and say so.
    footprint: np.ndarray | None = None
    # One entry per footprint edge (edge i runs footprint[i] -> footprint[i+1]):
    # "returns" for a wall the points located, "carve" for one only the free
    # space placed, "frontier" for the limit of what was seen -- no wall there
    # at all, just the end of the evidence.
    edge_sources: list[str] = field(default_factory=list)
    # A ceiling the carve is sure of but the returns never landed on. Kept
    # apart from `ceiling_z` on purpose: everything that measures reads
    # `ceiling_z` and gets None, everything that draws or bounds the room can
    # use this and say where it came from. A blank white ceiling is the single
    # most common surface a phone sweep fails to return, and capping the room
    # at the top of the points instead -- which is what happens without this --
    # is a guess with no error bar at all.
    ceiling_z_inferred: float | None = None
    openings: list[Opening] = field(default_factory=list)
    ceiling_source: str = "none"  # returns | carve | supplied | none
    carved_fraction: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def measured_wall_area(self) -> float:
        return float(sum(p.area for p, l in zip(self.walls, self.lines) if l.measured))

    @property
    def inferred_wall_area(self) -> float:
        return float(sum(p.area for p, l in zip(self.walls, self.lines) if not l.measured))


# ---------------------------------------------------------------------------
# the volumes
# ---------------------------------------------------------------------------


def _volumes(
    xyz: np.ndarray,
    cameras: np.ndarray,
    grid: Any | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """`(solid, free, origin, voxel)`, reusing a prebuilt grid when it fits.

    The grid ingest already built is preferred over rebuilding one, but only
    when its cells are cubic: the carve steps along rays in units of the
    smallest voxel and marks whole cells, so an anisotropic grid would clear a
    different amount of space in Z than in X for reasons that have nothing to
    do with the room.
    """
    origin = voxel = None
    solid = None
    if grid is not None:
        occ = getattr(grid, "occupied", None)
        gor = getattr(grid, "origin", None)
        gvx = getattr(grid, "voxel", None)
        if occ is not None and gor is not None and gvx is not None:
            occ = np.asarray(occ)
            gvx = np.asarray(gvx, dtype=np.float64).ravel()
            if occ.ndim == 3 and len(gvx) == 3 and np.ptp(gvx) < 1e-9 and gvx[0] > 0:
                solid = occ.astype(bool)
                origin = np.asarray(gor, dtype=np.float64).reshape(3)
                voxel = gvx

    if solid is None:
        from .grid import build_grid

        built = build_grid(
            PointCloud(xyz=xyz), voxel_xy=ROOM_VOXEL, voxel_z=ROOM_VOXEL
        )
        solid = np.asarray(built.occupied, dtype=bool)
        origin = np.asarray(built.origin, dtype=np.float64).reshape(3)
        voxel = np.asarray(built.voxel, dtype=np.float64).reshape(3)

    if not solid.any():
        return None

    free = infill.carve_free_space(
        solid.shape, origin, voxel, xyz, cameras, solid=solid, seed=seed
    )
    return solid, free, origin, voxel


# ---------------------------------------------------------------------------
# interior, floor and ceiling
# ---------------------------------------------------------------------------


def _interior_columns(free: np.ndarray, voxel_z: float, res: float) -> np.ndarray:
    """Columns the sweep cleared floor-to-head-height, as one connected region.

    Filling holes is what makes this the *room* rather than the swept part of
    it: a table the rays never got under leaves an island of unknown in the
    middle of a region that is otherwise all free, and the room plainly
    continues underneath it.
    """
    cleared = free.sum(axis=2) * voxel_z >= INTERIOR_MIN_FREE_M
    if not cleared.any():
        return cleared

    span = max(1, int(round(0.15 / res)))
    struct = np.ones((span, span), dtype=bool)
    mask = ndimage.binary_closing(cleared, structure=struct, border_value=0)
    mask = ndimage.binary_fill_holes(mask)

    labels, n = ndimage.label(mask)
    if n > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        mask = labels == int(sizes.argmax())
    return mask


def _ceiling_from_carve(
    free: np.ndarray,
    interior: np.ndarray,
    xyz: np.ndarray,
    origin: np.ndarray,
    voxel: np.ndarray,
    floor_z: float,
    normals: np.ndarray | None,
) -> tuple[float | None, str, dict[str, Any]]:
    """The height at which free space stops, and how much to believe it.

    Three things are asked, in order, and each one can end the answer.

    *Is there a plateau?* The top of the swept volume has to sit at one height
    across most of the room. The candidate is taken from the upper tail of the
    column tops rather than from their mode, because a sweep that filmed the
    ceiling over half the room and not the other half has its mode down in the
    half that never looked up. A capture with no plateau has no ceiling in it,
    however much free space it carved.

    *Is anything up there?* A plateau with no returns on it at all is the
    frontier of the sweep -- the shape of where the phone pointed. A very
    little evidence is enough to say the cap is real, because a blank white
    ceiling is exactly the surface stereo cannot see: on a real kitchen the
    ceiling returned 1.4% of the interior, essentially the light fitting alone.

    *Is it measured?* Only if returns cover enough of it to have located it
    themselves. Otherwise the height comes from the carve, is quantised to the
    voxel, and comes back labelled inferred -- good enough to cap a room in a
    viewer and to bound a volume, never good enough to quote.
    """
    diag: dict[str, Any] = {}
    if not interior.any():
        return None, "none", diag

    has_free = free.any(axis=2)
    top_idx = free.shape[2] - 1 - np.argmax(free[:, :, ::-1], axis=2)
    top_z = origin[2] + (top_idx + 0.5) * voxel[2]
    tops = top_z[interior & has_free]
    if len(tops) < 8:
        return None, "none", diag

    tail = tops[tops >= np.percentile(tops, 75)]
    edges = np.arange(tail.min() - voxel[2], tail.max() + 2 * voxel[2], voxel[2])
    if len(edges) < 3:
        return None, "none", diag
    hist, _ = np.histogram(tail, bins=edges)
    peak = int(np.argmax(hist))
    centre = 0.5 * (edges[peak] + edges[peak + 1])

    near = np.abs(tops - centre) <= CEILING_PLATEAU_TOL_M
    if not near.any():
        return None, "none", diag
    plateau_z = float(tops[near].mean())
    cell_area = float(voxel[0] * voxel[1])
    plateau_area = float(near.sum() * cell_area)
    fraction = float(near.sum() / max(1, len(tops)))
    height = plateau_z - floor_z
    diag["ceiling_plateau_z"] = round(plateau_z, 4)
    diag["ceiling_plateau_area_m2"] = round(plateau_area, 3)
    diag["ceiling_plateau_fraction"] = round(fraction, 4)

    if not (CEILING_MIN_HEIGHT_M <= height <= CEILING_MAX_HEIGHT_M):
        diag["ceiling_rejected"] = (
            f"free space stops at {plateau_z:.2f} m, which implies a "
            f"{height:.2f} m room; that is not a ceiling"
        )
        return None, "none", diag
    if plateau_area < CEILING_MIN_AREA_M2 or fraction < CEILING_PLATEAU_FRACTION:
        diag["ceiling_rejected"] = (
            f"the top of the sweep only levels off over {fraction * 100:.0f}% "
            f"of the room ({plateau_area:.1f} m2); that is the shape of where "
            "the camera pointed, not a ceiling -- film the ceiling for a few "
            "seconds and it becomes measurable"
        )
        return None, "none", diag

    cover = _return_cover(xyz, normals, interior, origin, voxel, plateau_z)
    diag["ceiling_return_cover"] = round(cover, 4)
    if cover < CEILING_MIN_INFERRED_COVER:
        diag["ceiling_rejected"] = (
            f"free space is capped at {plateau_z:.2f} m but nothing was ever "
            "returned from up there; the capture was cut off rather than roofed"
        )
        return None, "none", diag
    if cover < CEILING_MIN_RETURN_COVER:
        return plateau_z, "carve", diag

    band = np.abs(xyz[:, 2] - plateau_z) <= CEILING_PLATEAU_TOL_M
    refined = float(np.median(xyz[band, 2])) if int(band.sum()) >= 50 else plateau_z
    if abs(refined - plateau_z) > 3 * float(voxel[2]):
        refined = plateau_z
    return refined, "returns", diag


def _return_cover(
    xyz: np.ndarray,
    normals: np.ndarray | None,
    interior: np.ndarray,
    origin: np.ndarray,
    voxel: np.ndarray,
    z: float,
) -> float:
    """Fraction of interior columns carrying a return within a tolerance of `z`.

    Restricting to the interior is what makes this mean "ceiling" rather than
    "anything at that height": the top of a wall, the top of a bookcase and the
    top of a doorframe all sit at ceiling height, and none of them stands over
    the middle of the room.
    """
    band = np.abs(xyz[:, 2] - z) <= CEILING_PLATEAU_TOL_M
    if normals is not None and len(normals) == len(xyz):
        # a ceiling faces down; the top of a shelf faces up
        band = band & (normals[:, 2] <= -0.5)
    if not band.any():
        return 0.0
    ij = np.floor((xyz[band, :2] - origin[:2]) / voxel[:2]).astype(np.int64)
    ok = np.all((ij >= 0) & (ij < np.array(interior.shape)), axis=1)
    ij = ij[ok]
    if not len(ij):
        return 0.0
    hit = np.zeros_like(interior)
    hit[ij[:, 0], ij[:, 1]] = True
    return float((hit & interior).sum() / max(1, int(interior.sum())))


# ---------------------------------------------------------------------------
# walls
# ---------------------------------------------------------------------------


def _wall_cells(
    solid: np.ndarray,
    interior: np.ndarray,
    k0: int,
    k1: int,
    res: float,
) -> np.ndarray:
    """Boolean plan mask of columns that look like standing surface near the room."""
    band = solid[:, :, k0:k1]
    if band.shape[2] <= 0:
        return np.zeros(interior.shape, dtype=bool)
    standing = band.sum(axis=2) >= WALL_MIN_VERTICAL_VOXELS

    reach = max(1, int(round(WALL_SEARCH_M / res)))
    near = ndimage.binary_dilation(interior, iterations=reach)
    return standing & near


def _dominant_angle(xy: np.ndarray, res: float) -> float:
    """The yaw at which the wall evidence stacks up into the fewest, sharpest lines.

    Scoring a candidate direction by the sum of the squares of its projection
    histogram is the whole trick: a direction aligned with the walls puts every
    cell of a wall into one bin and the sum of squares goes up quadratically
    with how well they agree, while an off-axis direction smears the same
    cells across many bins. It finds a room's own orientation without assuming
    the canonicaliser got the yaw exactly right, and without assuming the room
    is rectangular -- it only assumes two of its walls are parallel or square,
    which is a far weaker claim.
    """

    def score(theta: float) -> float:
        c, s = math.cos(theta), math.sin(theta)
        total = 0.0
        for n in ((c, s), (-s, c)):
            p = xy @ np.array(n)
            bins = np.floor((p - p.min()) / res).astype(np.int64)
            h = np.bincount(bins)
            total += float(np.dot(h, h))
        return total

    coarse = np.arange(0.0, 90.0, 0.5)
    best = max(coarse, key=lambda d: score(math.radians(d)))
    fine = np.arange(best - 0.5, best + 0.5, 0.05)
    best = max(fine, key=lambda d: score(math.radians(d)))
    return math.radians(float(best))


def _lines_along(
    xy: np.ndarray,
    normal: np.ndarray,
    res: float,
    min_cells: int,
) -> list[WallLine]:
    """Every wall lying perpendicular to `normal`, as offset peaks in projection."""
    p = xy @ normal
    if len(p) == 0:
        return []
    edges = np.arange(p.min() - res, p.max() + 2 * res, res)
    if len(edges) < 3:
        return []
    hist, _ = np.histogram(p, bins=edges)
    smooth = ndimage.uniform_filter1d(hist.astype(np.float64), size=3)

    out: list[WallLine] = []
    order = np.argsort(smooth)[::-1]
    claimed = np.zeros(len(p), dtype=bool)
    for b in order:
        if smooth[b] < min_cells:
            break
        centre = 0.5 * (edges[b] + edges[b + 1])
        near = (np.abs(p - centre) <= WALL_INLIER_M) & ~claimed
        if near.sum() < min_cells:
            continue
        offset = float(p[near].mean())
        # re-select about the refined offset so the extent is measured on the
        # line rather than on the bin that found it
        near = (np.abs(p - offset) <= WALL_INLIER_M) & ~claimed
        if near.sum() < min_cells:
            continue
        claimed |= near

        direction = np.array([-normal[1], normal[0]])
        t = np.sort(xy[near] @ direction)
        for lo, hi, count in _runs(t, WALL_GAP_M):
            if hi - lo >= WALL_MIN_LENGTH_M:
                out.append(
                    WallLine(
                        normal=normal.copy(),
                        offset=offset,
                        t0=float(lo),
                        t1=float(hi),
                        cells=int(count),
                    )
                )
    return out


def _runs(t: np.ndarray, gap: float) -> list[tuple[float, float, int]]:
    """Split a sorted 1D set into runs separated by more than `gap`."""
    if len(t) == 0:
        return []
    breaks = np.nonzero(np.diff(t) > gap)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(t) - 1]])
    return [(float(t[a]), float(t[b]), int(b - a + 1)) for a, b in zip(starts, ends)]


def _merge_lines(lines: list[WallLine]) -> list[WallLine]:
    """One line per physical wall, keeping the longest of each duplicate group.

    Two candidates are the same wall only if they are collinear *and* cover the
    same stretch of it. Dropping the second half of that test is a mistake that
    reads as a wall: the bottom edges of two separate cupboards standing against
    opposite ends of a room are collinear to the centimetre, and absorbing one
    into the other invents a 5.2 m surface spanning the room through thin air,
    which then shadows the real wall behind them. Runs further apart than a
    doorway stay separate lines and are judged on their own length.
    """
    cos_tol = math.cos(math.radians(WALL_MERGE_ANGLE_DEG))
    kept: list[WallLine] = []
    for line in sorted(lines, key=lambda l: -l.length):
        duplicate = False
        for other in kept:
            align = float(np.dot(line.normal, other.normal))
            if abs(align) < cos_tol:
                continue
            offset = line.offset if align > 0 else -line.offset
            if abs(offset - other.offset) > WALL_MERGE_OFFSET_M:
                continue
            t0 = line.t0 if align > 0 else -line.t1
            t1 = line.t1 if align > 0 else -line.t0
            gap = max(t0 - other.t1, other.t0 - t1)
            if gap > WALL_GAP_M:
                continue  # collinear, but a different stretch of the room
            other.t0 = min(other.t0, t0)
            other.t1 = max(other.t1, t1)
            duplicate = True
            break
        if not duplicate:
            kept.append(line)
    return kept


#: A line is the room's boundary rather than its contents when this much of
#: what lies just behind it is outside the swept interior. The distance is one
#: step beyond a stereo surface's own thickness, so it lands behind the wall
#: rather than inside it.
BOUNDARY_MIN_FRACTION = 0.6
BOUNDARY_PROBE_M = 0.20


def _is_boundary(
    line: WallLine, interior: np.ndarray, origin: np.ndarray, voxel: np.ndarray
) -> float:
    """How much of what stands behind this line is outside the room.

    This is the test that separates architecture from furniture without
    measuring either. Walk a short step through a wall and you are out of the
    room; walk the same step through the back of a sofa and you are still in
    it. A kitchen island is rejected by it, an appliance pushed against a wall
    is not -- and that is right, because the surface a light can stand against
    is the appliance, not the plaster behind it.
    """
    n = max(4, int(round(line.length / max(voxel[0], 1e-6))))
    t = np.linspace(line.t0, line.t1, n)
    d = line.direction
    pts = (
        line.offset * line.normal[None, :]
        + t[:, None] * d[None, :]
        - BOUNDARY_PROBE_M * line.normal[None, :]
    )
    ij = np.floor((pts - origin[:2]) / voxel[:2]).astype(np.int64)
    shape = np.array(interior.shape)
    inside_grid = np.all((ij >= 0) & (ij < shape), axis=1)
    behind_interior = np.zeros(len(pts), dtype=bool)
    if inside_grid.any():
        sel = ij[inside_grid]
        behind_interior[inside_grid] = interior[sel[:, 0], sel[:, 1]]
    return float(1.0 - behind_interior.mean())


#: Two parallel surfaces competing to be the same stretch of boundary have to
#: overlap along their shared direction by at least this much of the outer one,
#: and the inner one has to be at least this fraction of its length, before the
#: outer is discarded. Below either bar they are not rivals: they are the two
#: arms of an L, or a wall and the fridge standing proud of it. Without the
#: length bar a 0.73 m appliance front deleted the 3.07 m wall behind it.
PARALLEL_OVERLAP = 0.5
PARALLEL_MIN_LENGTH_RATIO = 0.6


def _resolve_parallel(lines: list[WallLine]) -> list[WallLine]:
    """Keep the innermost of any parallel surfaces covering the same stretch.

    A kitchen wall arrives three times: the run of cabinet doors, the worktop
    edge standing proud of them, and the plaster behind both. All three are
    real, and all three are parallel and within 30 cm of each other, so the
    duplicate merge cannot touch them -- it would be wrong to, they are
    different surfaces.

    Only one of them is the room, though, and it is the nearest one. The
    boundary of a room is the first thing you meet going outward, which is also
    the only one of the three answers a filmmaker can use: it is the surface a
    light stands against and the one a dolly stops at. The others become
    contents.

    Overlap is what makes this safe on an L-shaped room, where two parallel
    walls at different offsets are both boundary and never cover the same span.
    """
    kept: list[WallLine] = []
    cos_tol = math.cos(math.radians(WALL_MERGE_ANGLE_DEG))
    # innermost first: with normals pointing inward, the largest offset is the
    # plane closest to the room
    for line in sorted(lines, key=lambda l: -l.offset):
        shadowed = False
        for other in kept:
            if float(np.dot(line.normal, other.normal)) < cos_tol:
                continue
            if other.length < PARALLEL_MIN_LENGTH_RATIO * line.length:
                # the inner surface is a protrusion, not a rival boundary
                continue
            lo = max(line.t0, other.t0)
            hi = min(line.t1, other.t1)
            if line.length > 0 and (hi - lo) / line.length >= PARALLEL_OVERLAP:
                shadowed = True
                break
        if not shadowed:
            kept.append(line)
    return kept


def _orient(lines: list[WallLine], inside: np.ndarray) -> None:
    """Point every normal into the room, in place."""
    for line in lines:
        if float(np.dot(line.normal, inside) - line.offset) < 0:
            line.normal = -line.normal
            line.offset = -line.offset
            line.t0, line.t1 = -line.t1, -line.t0


def _refit(line: WallLine, xy: np.ndarray, z: np.ndarray, zlo: float, zhi: float) -> None:
    """Pull a carved wall onto the returns that lie along it, in place.

    Total least squares rather than a regression of one coordinate on the
    other: a wall running along Y has no slope in the least-squares sense, and
    fitting `y = mx + c` to it divides by zero. The eigenvector of the
    scatter's smaller eigenvalue is the same fit taken symmetrically, and it
    behaves identically at every orientation.

    The result is rejected outright unless it agrees with the carve. A refit
    that turns the wall eight degrees or scatters six centimetres has not found
    the wall; it has found the sofa in front of it, and the carved position --
    coarse, but derived from where the room stops -- is the better answer.
    """
    d = line.direction
    t = xy @ d
    s = xy @ line.normal - line.offset
    band = (
        (np.abs(s) <= 0.12)
        & (t >= line.t0 - 0.10)
        & (t <= line.t1 + 0.10)
        & (z >= zlo)
        & (z <= zhi)
    )
    n = int(band.sum())
    line.points = n
    if n < REFIT_MIN_POINTS:
        return

    pts = xy[band]
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    evals, evecs = np.linalg.eigh(cov)
    normal = evecs[:, int(np.argmin(evals))]
    normal = normal / np.linalg.norm(normal)
    if float(np.dot(normal, line.normal)) < 0:
        normal = -normal
    turn = math.degrees(math.acos(min(1.0, abs(float(np.dot(normal, line.normal))))))
    offset = float(np.dot(normal, mean))
    rms = float(np.sqrt(np.mean((pts @ normal - offset) ** 2)))
    if turn > REFIT_MAX_TURN_DEG or rms > REFIT_MAX_RMS_M:
        return

    direction = np.array([-normal[1], normal[0]])
    t2 = pts @ direction
    line.normal = normal
    line.offset = offset
    # The extent is the union of the two: the returns can only be found inside
    # the band the carve proposed, and the carve can reach along a stretch of
    # blank wall that caught no returns at all. Both are the wall.
    line.t0 = float(min(line.t0, t2.min()))
    line.t1 = float(max(line.t1, t2.max()))
    line.rms = rms
    line.source = "returns"


def _plane_from_line(line: WallLine, floor_z: float, ceiling_z: float | None) -> Plane:
    """A `WallLine` as the full-height architectural plane it stands for."""
    top = ceiling_z if ceiling_z is not None else floor_z + 2.4
    plane = Plane(
        normal=np.array([line.normal[0], line.normal[1], 0.0]),
        offset=float(line.offset),
        kind="wall",
        inlier_count=int(line.points),
    )
    u, v = plane_frame(plane)
    d = line.direction
    corners = np.array(
        [
            [line.offset * line.normal[0] + t * d[0], line.offset * line.normal[1] + t * d[1], z]
            for t in (line.t0, line.t1)
            for z in (floor_z, top)
        ]
    )
    uu, vv = corners @ u, corners @ v
    plane.extent_2d = (float(uu.min()), float(vv.min()), float(uu.max()), float(vv.max()))
    plane.area = float(line.length * max(0.0, top - floor_z))
    return plane


# ---------------------------------------------------------------------------
# openings
# ---------------------------------------------------------------------------

# How far to step either side of a wall plane when asking whether the sweep saw
# through it. Inward is a short step, because the room is right there; outward
# has to clear the wall's own thickness, and a stereo-reconstructed wall is a
# slab several centimetres deep rather than a surface.
OPENING_PROBE_IN_M = 0.15
OPENING_PROBE_OUT_M = 0.30
# Material within this distance of the plane counts as the wall being present.
OPENING_MATERIAL_M = 0.12
# The smallest hole worth reporting. Below this it is a gap in the returns.
OPENING_MIN_W_M = 0.45
OPENING_MIN_H_M = 0.55
OPENING_MIN_AREA_M2 = 0.35
# A hole at floor level is a doorway only if something spans it. Without this a
# wall that simply stops -- the end of a run, the mouth of an alcove -- reads as
# the widest door in the building.
OPENING_LINTEL_COVER = 0.45
OPENING_LINTEL_M = 0.35


def _openings_from_carve(
    lines: list[WallLine],
    solid: np.ndarray,
    free: np.ndarray,
    origin: np.ndarray,
    voxel: np.ndarray,
    floor_z: float,
    ceiling_z: float,
) -> list[Opening]:
    """Holes in the walls, found by looking through them.

    Hole detection from returns alone has to reason from absence -- a patch of
    wall with no points on it -- and absence is exactly what a textureless
    surface produces anyway. So it either misses real doors or invents them
    wherever the stereo failed, and on a video reconstruction it does both.

    Seeing through is a positive test instead. A doorway has carved free space
    on both sides of the wall plane, because the camera stood in the room and
    looked into the hall; a blank stretch of wall has free space on one side
    only, however few points landed on it. The test needs no threshold on point
    density and cannot be fooled by a wall the stereo could not see.
    """
    res = float(voxel[0])
    shape = np.array(solid.shape)
    top = ceiling_z

    def probe(pts: np.ndarray, volume: np.ndarray) -> np.ndarray:
        ijk = np.floor((pts - origin) / voxel).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < shape), axis=-1)
        out = np.zeros(pts.shape[:-1], dtype=bool)
        sel = ijk[ok]
        out[ok] = volume[sel[..., 0], sel[..., 1], sel[..., 2]]
        return out

    found: list[Opening] = []
    for line in lines:
        height = top - floor_z
        if line.length < OPENING_MIN_W_M or height < OPENING_MIN_H_M:
            continue
        nu = max(2, int(round(line.length / res)))
        nv = max(2, int(round(height / res)))
        u = np.linspace(line.t0, line.t1, nu)
        v = np.linspace(floor_z, top, nv)
        uu, vv = np.meshgrid(u, v, indexing="ij")

        n3 = np.array([line.normal[0], line.normal[1], 0.0])
        d3 = np.array([line.direction[0], line.direction[1], 0.0])
        base = (
            line.offset * n3[None, None, :]
            + uu[..., None] * d3[None, None, :]
            + vv[..., None] * np.array([0.0, 0.0, 1.0])[None, None, :]
        )
        base[..., 2] = vv

        inside = probe(base + OPENING_PROBE_IN_M * n3, free)
        beyond = probe(base - OPENING_PROBE_OUT_M * n3, free)
        material = np.zeros(uu.shape, dtype=bool)
        steps = np.arange(-OPENING_MATERIAL_M, OPENING_MATERIAL_M + 1e-9, res)
        for s in steps:
            material |= probe(base + s * n3, solid)

        holes = inside & beyond & ~material
        if not holes.any():
            continue

        labels, count = ndimage.label(holes)
        for label in range(1, count + 1):
            comp = labels == label
            iu, iv = np.nonzero(comp)
            u0, u1 = float(u[iu.min()]), float(u[iu.max()])
            v0, v1 = float(v[iv.min()]), float(v[iv.max()])
            width, tall = u1 - u0, v1 - v0
            if width < OPENING_MIN_W_M or tall < OPENING_MIN_H_M:
                continue
            if width * tall < OPENING_MIN_AREA_M2:
                continue
            fill = float(comp.sum()) / max(1.0, (iu.max() - iu.min() + 1) * (iv.max() - iv.min() + 1))
            if fill < 0.45:
                continue

            sill = v0 - floor_z
            if sill <= 0.30:
                # a doorway needs a lintel; a wall that merely stops does not
                lint = (v >= v1) & (v <= v1 + OPENING_LINTEL_M)
                if not lint.any():
                    continue
                span = material[iu.min() : iu.max() + 1][:, lint]
                if span.size == 0 or float(span.mean()) < OPENING_LINTEL_COVER:
                    continue

            centre_u = 0.5 * (u0 + u1)
            centre = np.array(
                [
                    line.offset * line.normal[0] + centre_u * line.direction[0],
                    line.offset * line.normal[1] + centre_u * line.direction[1],
                    0.5 * (v0 + v1),
                ]
            )
            found.append(
                Opening(
                    center=centre,
                    width=float(width),
                    height=float(tall),
                    normal=n3,
                    sill_height=float(max(0.0, sill)),
                    kind=_classify(sill, tall),
                    confidence=float(min(0.95, 0.45 + 0.5 * fill)),
                )
            )
    return found


def _classify(sill: float, height: float) -> str:
    if sill <= 0.30 and height >= 1.60:
        return "door"
    if sill >= 0.40:
        return "window"
    return "opening"


# ---------------------------------------------------------------------------
# footprint
# ---------------------------------------------------------------------------


def straighten_footprint(
    poly: np.ndarray, lines: list[WallLine]
) -> tuple[np.ndarray, np.ndarray]:
    """Put a rastered outline back onto the walls it is an approximation of.

    A marching-squares outline of an occupancy raster is a staircase with a
    half-cell bias, and it wanders wherever the capture was thin. The walls are
    known to millimetres by this point, so every edge that is plainly an
    approximation of one gets replaced by the line itself, and consecutive
    edges belonging to different walls meet at the intersection of their lines
    -- which is where the corner of the room is, whether or not the sweep ever
    reached it.

    Corners are only closed across a short gap. A metre of unsupported outline
    between two walls is a missed corner; three metres is the open side of an
    L-shaped room or the mouth of an alcove, and cutting the chord would annex
    floor that is not there.

    Returns the polygon and a per-vertex mask saying which vertices came from
    inference rather than from the raster.
    """
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(poly) < 3 or not lines:
        return poly, np.zeros(len(poly), dtype=bool)

    n = len(poly)
    assigned: list[WallLine | None] = []
    cos_tol = math.cos(math.radians(SNAP_MAX_ANGLE_DEG))
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        seg = b - a
        length = float(np.linalg.norm(seg))
        if length < 1e-9:
            assigned.append(None)
            continue
        edge_n = np.array([-seg[1], seg[0]]) / length
        mid = 0.5 * (a + b)
        best, best_d = None, SNAP_MAX_DIST_M
        for line in lines:
            if abs(float(np.dot(edge_n, line.normal))) < cos_tol:
                continue
            d = abs(float(np.dot(mid, line.normal) - line.offset))
            if d < best_d:
                best, best_d = line, d
        assigned.append(best)

    if not any(a is not None for a in assigned):
        return poly, np.zeros(n, dtype=bool)

    out: list[np.ndarray] = []
    inferred: list[bool] = []
    for i in range(n):
        line = assigned[i]
        if line is None:
            # unsupported edge: keep its start vertex, it is measured raster
            out.append(poly[i])
            inferred.append(False)
            continue

        # find the next edge on a *different* wall, and the gap in between
        j, gap = i + 1, 0.0
        while j < i + n:
            nxt = assigned[j % n]
            if nxt is not None and nxt is not line:
                break
            gap += float(np.linalg.norm(poly[(j + 1) % n] - poly[j % n]))
            j += 1
        nxt = assigned[j % n] if j < i + n else None

        start = _project(poly[i], line)
        if not out or np.linalg.norm(out[-1] - start) > 1e-6:
            out.append(start)
            inferred.append(False)

        if nxt is None or gap > CORNER_MAX_GAP_M:
            end = _project(poly[(i + 1) % n], line)
            out.append(end)
            inferred.append(False)
            continue

        corner = _intersect(line, nxt)
        if corner is None:
            end = _project(poly[(i + 1) % n], line)
            out.append(end)
            inferred.append(False)
            continue
        out.append(corner)
        inferred.append(gap > 1e-6)

    keep = [0]
    for k in range(1, len(out)):
        if np.linalg.norm(out[k] - out[keep[-1]]) > 1e-6:
            keep.append(k)
    poly2 = np.array([out[k] for k in keep])
    inf2 = np.array([inferred[k] for k in keep], dtype=bool)
    if len(poly2) < 3:
        return poly, np.zeros(n, dtype=bool)
    return poly2, inf2


def _project(p: np.ndarray, line: WallLine) -> np.ndarray:
    return p - (float(np.dot(p, line.normal)) - line.offset) * line.normal


def _intersect(a: WallLine, b: WallLine) -> np.ndarray | None:
    m = np.stack([a.normal, b.normal])
    det = float(np.linalg.det(m))
    if abs(det) < 0.2:  # under ~11 degrees apart the corner is unstable
        return None
    return np.linalg.solve(m, np.array([a.offset, b.offset]))


# ---------------------------------------------------------------------------
# the footprint, from the cells of the wall arrangement
# ---------------------------------------------------------------------------
#
# The old way to get a footprint was to raster the occupied columns, close the
# raster, trace its contour, and then try to snap the trace back onto the
# fitted walls. Every failure of that chain shipped: a thin sweep rounds the
# raster off, the contour is a 200-vertex staircase, and when the snap declined
# the twin was drawn as the extrusion of a blob -- the "carved cylinder" look.
#
# Here the walls themselves *are* the outline. The fitted lines cut the plan
# into convex cells; a cell is room when the cameras stood in it or the carve
# cleared it; the footprint is the boundary of the union of room cells. Every
# edge of that boundary lies on a fitted wall or on the frontier of the
# evidence, so the polygon has straight walls and closed corners by
# construction, and each edge knows which of the two it is.

# Margin around the swept interior that the arrangement covers. Half a metre
# reaches the wall behind any plausible skirting of unscanned floor without
# annexing the neighbouring room.
CELL_MARGIN_M = 0.50

# Two fitted lines closer than this with normals this aligned are one carrier
# in the arrangement. Looser than WALL_MERGE_OFFSET_M would double-cut cells;
# tighter than a voxel would keep genuine duplicates apart.
CELL_LINE_TOL_M = 0.05
CELL_LINE_COS = math.cos(math.radians(5.0))

# A boundary edge is claimed by a wall run when its midpoint projects inside
# the run's extent, extended by this much slack -- half a doorway jamb.
CELL_RUN_SLACK_M = 0.12

# A cell with no camera in it is annexed into the room only when the carve
# cleared at least this fraction of it. Below that it is the hallway seen
# through a doorway, or the void beyond the frontier.
CELL_FREE_MIN = 0.30

# Sliver protection: intervals and loops shorter than this are noise from
# near-coincident carriers, not architecture.
CELL_EPS = 1e-6


def _clip_half(poly: np.ndarray, normal: np.ndarray, offset: float, keep: float) -> np.ndarray | None:
    """The part of a convex CCW polygon with `keep * (n.p - c) >= 0`, or None."""
    out: list[np.ndarray] = []
    n = len(poly)
    s = (poly @ normal - offset) * keep
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        sa, sb = s[i], s[(i + 1) % n]
        if sa >= -CELL_EPS:
            out.append(a)
        if (sa < -CELL_EPS and sb > CELL_EPS) or (sa > CELL_EPS and sb < -CELL_EPS):
            t = sa / (sa - sb)
            out.append(a + t * (b - a))
    if len(out) < 3:
        return None
    p = np.array(out)
    # drop consecutive near-duplicates the intersection arithmetic produces
    keep_idx = [0]
    for k in range(1, len(p)):
        if np.linalg.norm(p[k] - p[keep_idx[-1]]) > CELL_EPS:
            keep_idx.append(k)
    if np.linalg.norm(p[keep_idx[-1]] - p[keep_idx[0]]) <= CELL_EPS:
        keep_idx.pop()
    if len(keep_idx) < 3:
        return None
    p = p[keep_idx]
    if abs(_poly_area(p)) < 1e-6:
        return None
    return p


def _poly_area(poly: np.ndarray) -> float:
    q = np.roll(poly, -1, axis=0)
    return float(np.sum(poly[:, 0] * q[:, 1] - q[:, 0] * poly[:, 1]) / 2.0)


def _unique_carriers(lines: list[WallLine]) -> list[tuple[np.ndarray, float]]:
    """The distinct infinite lines the walls lie on, sign-normalised."""
    out: list[tuple[np.ndarray, float]] = []
    for line in lines:
        n, c = line.normal, float(line.offset)
        # normalise the sign so one physical line has one representation
        if n[0] < 0 or (abs(n[0]) < 1e-12 and n[1] < 0):
            n, c = -n, -c
        matched = False
        for un, uc in out:
            if float(n @ un) >= CELL_LINE_COS and abs(c - uc) <= CELL_LINE_TOL_M:
                matched = True
                break
        if not matched:
            out.append((n.copy(), c))
    return out


def _cell_complex(
    carriers: list[tuple[np.ndarray, float]], lo: np.ndarray, hi: np.ndarray
) -> list[np.ndarray]:
    """Convex CCW cells of the arrangement of `carriers` over the box lo..hi."""
    base = np.array(
        [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]], dtype=np.float64
    )
    cells = [base]
    for n, c in carriers:
        nxt: list[np.ndarray] = []
        for cell in cells:
            for keep in (1.0, -1.0):
                part = _clip_half(cell, n, c, keep)
                if part is not None:
                    nxt.append(part)
        cells = nxt
    return cells


def _in_convex(pts: np.ndarray, poly: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Which of `pts` lie inside the convex CCW polygon `poly`."""
    inside = np.ones(len(pts), dtype=bool)
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        inside &= (e[0] * (pts[:, 1] - a[1]) - e[1] * (pts[:, 0] - a[0])) >= -tol
    return inside


def _cell_free_fraction(
    poly: np.ndarray, interior: np.ndarray, origin: np.ndarray, voxel: np.ndarray
) -> float:
    """Fraction of the cell's plan cells the carve cleared."""
    lo = poly.min(axis=0)
    hi = poly.max(axis=0)
    i0 = np.maximum(np.floor((lo - origin[:2]) / voxel[:2]).astype(np.int64), 0)
    i1 = np.minimum(
        np.ceil((hi - origin[:2]) / voxel[:2]).astype(np.int64), np.array(interior.shape)
    )
    if np.any(i1 <= i0):
        return 0.0
    ii, jj = np.meshgrid(
        np.arange(i0[0], i1[0]), np.arange(i0[1], i1[1]), indexing="ij"
    )
    centres = origin[:2] + (np.stack([ii.ravel(), jj.ravel()], axis=1) + 0.5) * voxel[:2]
    mask = _in_convex(centres, poly)
    if not mask.any():
        return 0.0
    vals = interior[ii.ravel()[mask], jj.ravel()[mask]]
    return float(vals.mean())


def _carrier_of(
    a: np.ndarray, b: np.ndarray, carriers: list[tuple[np.ndarray, float]]
) -> int:
    """Which carrier the edge a->b lies on, or -1 for a box edge."""
    mid = 0.5 * (a + b)
    d = b - a
    length = float(np.linalg.norm(d))
    if length < CELL_EPS:
        return -1
    d = d / length
    for k, (n, c) in enumerate(carriers):
        if abs(float(d @ n)) > 0.01:
            continue
        if abs(float(mid @ n) - c) <= 1e-6:
            return k
    return -1


def _edge_source(
    mid: np.ndarray, direction: np.ndarray, lines: list[WallLine]
) -> str:
    """What put this stretch of boundary here: a measured wall, a carved one,
    or nothing but the end of the evidence."""
    best = "frontier"
    for line in lines:
        if abs(float(direction @ line.normal)) > 0.3:
            continue
        if abs(float(mid @ line.normal) - line.offset) > CELL_LINE_TOL_M + WALL_INLIER_M:
            continue
        t = float(mid @ line.direction)
        if line.t0 - CELL_RUN_SLACK_M <= t <= line.t1 + CELL_RUN_SLACK_M:
            if line.source == "returns":
                return "returns"
            best = "carve"
    return best


def footprint_from_cells(
    lines: list[WallLine],
    interior: np.ndarray,
    origin: np.ndarray,
    voxel: np.ndarray,
    cameras_xy: np.ndarray,
) -> tuple[np.ndarray | None, list[str], dict[str, Any]]:
    """The room outline as the boundary of the interior cells of the wall
    arrangement.

    Returns `(polygon, edge_sources, diagnostics)`. The polygon is CCW;
    `edge_sources[i]` labels the edge from vertex i to vertex i+1. Returns
    `(None, [], diag)` when the evidence cannot be read as a room -- the caller
    is expected to degrade loudly, not to substitute a blob.
    """
    diag: dict[str, Any] = {}
    if not lines or not interior.any():
        diag["footprint_declined"] = "no wall lines or no swept interior"
        return None, [], diag

    ij = np.argwhere(interior)
    lo = origin[:2] + ij.min(axis=0) * voxel[:2] - CELL_MARGIN_M
    hi = origin[:2] + (ij.max(axis=0) + 1) * voxel[:2] + CELL_MARGIN_M

    carriers = _unique_carriers(lines)
    cells = _cell_complex(carriers, lo, hi)
    diag["cells"] = len(cells)
    if not cells:
        diag["footprint_declined"] = "the arrangement produced no cells"
        return None, [], diag

    free = np.array(
        [_cell_free_fraction(c, interior, origin, voxel) for c in cells]
    )

    def locate(p: np.ndarray) -> int:
        for k, cell in enumerate(cells):
            if bool(_in_convex(p[None, :], cell, tol=1e-7)[0]):
                return k
        return -1

    cams = np.asarray(cameras_xy, dtype=np.float64).reshape(-1, 2)
    seeds = {k for k in (locate(c) for c in cams) if k >= 0}
    if not seeds:
        # a capture whose cameras all fall outside the arrangement is not a
        # room this solver can label; better no footprint than a guessed one
        diag["footprint_declined"] = "no camera stands inside any cell"
        return None, [], diag

    # region-grow from the cells the operator stood in, through edges that are
    # not sealed by a wall, into cells the carve cleared. A doorway is an
    # unsealed gap, but the hallway beyond it fails the free-fraction bar
    # because only its sliver near the door was ever swept.
    def edge_sealed(a: np.ndarray, b: np.ndarray) -> bool:
        d = b - a
        length = float(np.linalg.norm(d))
        if length < CELL_EPS:
            return False
        d = d / length
        count = max(3, int(length / 0.15) + 1)
        ts = (np.arange(count) + 0.5) / count
        pts = a[None, :] + ts[:, None] * (b - a)[None, :]
        hit = np.zeros(count, dtype=bool)
        for line in lines:
            if abs(float(d @ line.normal)) > 0.3:
                continue
            dist = np.abs(pts @ line.normal - line.offset)
            t = pts @ line.direction
            hit |= (
                (dist <= CELL_LINE_TOL_M + WALL_INLIER_M)
                & (t >= line.t0 - CELL_RUN_SLACK_M)
                & (t <= line.t1 + CELL_RUN_SLACK_M)
            )
        return bool(hit.mean() >= 0.5)

    kept = set(seeds)
    queue = list(seeds)
    while queue:
        k = queue.pop()
        poly = cells[k]
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            e = b - a
            length = float(np.linalg.norm(e))
            if length < CELL_EPS:
                continue
            if edge_sealed(a, b):
                continue
            outward = np.array([e[1], -e[0]]) / length  # right of a CCW edge
            probe = 0.5 * (a + b) + 0.02 * outward
            j = locate(probe)
            if j >= 0 and j not in kept and free[j] >= CELL_FREE_MIN:
                kept.add(j)
                queue.append(j)

    diag["cells_kept"] = len(kept)

    # -- boundary, by interval bookkeeping on each carrier -----------------
    #
    # Cells on opposite sides of a line are cut by different crossing lines,
    # so their edges subdivide the line differently (T-junctions), and naive
    # edge-twin matching falls apart. Projecting every kept-cell edge onto its
    # carrier as an interval and taking the symmetric difference of the two
    # sides is exact: boundary is where exactly one side is room.
    box_sides = [
        (np.array([0.0, 1.0]), float(lo[1])),   # bottom
        (np.array([1.0, 0.0]), float(hi[0])),   # right
        (np.array([0.0, 1.0]), float(hi[1])),   # top
        (np.array([1.0, 0.0]), float(lo[0])),   # left
    ]
    all_carriers = carriers + box_sides
    n_walls = len(carriers)

    # per carrier: list of (t0, t1, side) with side = +1 for the +n half
    spans: dict[int, list[tuple[float, float, int]]] = {}
    for k in kept:
        poly = cells[k]
        centroid = poly.mean(axis=0)
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            ci = _carrier_of(a, b, all_carriers)
            if ci < 0:
                continue
            cn, cc = all_carriers[ci]
            direction = np.array([-cn[1], cn[0]])
            ta, tb = float(a @ direction), float(b @ direction)
            side = 1 if float(centroid @ cn) - cc > 0 else -1
            spans.setdefault(ci, []).append((min(ta, tb), max(ta, tb), side))

    segments: list[tuple[np.ndarray, np.ndarray, str]] = []
    for ci, ivals in spans.items():
        cn, cc = all_carriers[ci]
        direction = np.array([-cn[1], cn[0]])
        cuts = sorted({round(v, 9) for t0, t1, _ in ivals for v in (t0, t1)})
        for t0, t1 in zip(cuts[:-1], cuts[1:]):
            if t1 - t0 < CELL_EPS:
                continue
            mid_t = 0.5 * (t0 + t1)
            pos = any(s > 0 and a - 1e-9 <= mid_t <= b + 1e-9 for a, b, s in ivals)
            neg = any(s < 0 and a - 1e-9 <= mid_t <= b + 1e-9 for a, b, s in ivals)
            if pos == neg:
                continue  # both sides room (internal) or neither (outside)
            p0 = cc * cn + t0 * direction
            p1 = cc * cn + t1 * direction
            # interior must lie on the LEFT of the directed edge for a CCW loop
            if pos:
                p0, p1 = p1, p0
            mid = 0.5 * (p0 + p1)
            src = (
                _edge_source(mid, (p1 - p0) / max(np.linalg.norm(p1 - p0), CELL_EPS), lines)
                if ci < n_walls
                else "frontier"
            )
            segments.append((p0, p1, src))

    if len(segments) < 3:
        diag["footprint_declined"] = "the interior cells left no traceable boundary"
        return None, [], diag

    # -- chain the segments into loops -------------------------------------
    def key(p: np.ndarray) -> tuple[float, float]:
        return (round(float(p[0]), 6), round(float(p[1]), 6))

    by_start: dict[tuple[float, float], list[int]] = {}
    for idx, (p0, _, _) in enumerate(segments):
        by_start.setdefault(key(p0), []).append(idx)

    used = np.zeros(len(segments), dtype=bool)
    loops: list[tuple[np.ndarray, list[str]]] = []
    for start in range(len(segments)):
        if used[start]:
            continue
        chain = [start]
        used[start] = True
        while True:
            _, p1, _ = segments[chain[-1]]
            nxt = None
            for cand in by_start.get(key(p1), []):
                if not used[cand]:
                    nxt = cand
                    break
            if nxt is None:
                break
            used[nxt] = True
            chain.append(nxt)
            if key(segments[nxt][1]) == key(segments[chain[0]][0]):
                verts = np.array([segments[c][0] for c in chain])
                srcs = [segments[c][2] for c in chain]
                loops.append((verts, srcs))
                break

    if not loops:
        diag["footprint_declined"] = "the boundary segments did not close"
        return None, [], diag

    poly, srcs = max(loops, key=lambda l: abs(_poly_area(l[0])))
    if _poly_area(poly) < 0:
        # a CW largest loop is the outside of a hole; the room is not that
        diag["footprint_declined"] = "the largest boundary loop was a hole"
        return None, [], diag
    diag["footprint_loops"] = len(loops)

    # -- merge collinear runs, carrying the dominant provenance ------------
    merged_v: list[np.ndarray] = []
    merged_s: list[str] = []
    n = len(poly)
    i = 0
    order = {"returns": 0, "carve": 1, "frontier": 2}
    while i < n:
        j = i
        seg_len: dict[str, float] = {}
        a = poly[i % n]
        while True:
            b = poly[(j + 1) % n]
            src = srcs[j % n]
            seg_len[src] = seg_len.get(src, 0.0) + float(np.linalg.norm(b - poly[j % n]))
            nxt = poly[(j + 2) % n]
            e1 = b - a
            e2 = nxt - b
            cross = e1[0] * e2[1] - e1[1] * e2[0]
            straight = abs(cross) <= 1e-9 * max(1.0, float(np.linalg.norm(e1) * np.linalg.norm(e2)))
            if straight and j + 1 < i + n:
                j += 1
            else:
                break
        merged_v.append(a)
        merged_s.append(min(seg_len, key=lambda s: (-seg_len[s], order[s])))
        i = j + 1
    poly = np.array(merged_v)
    srcs = merged_s

    if len(poly) < 3:
        diag["footprint_declined"] = "the merged outline degenerated"
        return None, [], diag

    area = _poly_area(poly)
    interior_area = float(interior.sum() * voxel[0] * voxel[1])
    diag["footprint_area_m2"] = round(area, 3)
    diag["footprint_vertices"] = len(poly)
    diag["footprint_edges"] = {
        s: sum(1 for x in srcs if x == s) for s in ("returns", "carve", "frontier")
    }
    if area < max(1.0, 0.4 * interior_area):
        diag["footprint_declined"] = (
            f"the labelled cells cover {area:.1f} m2 against {interior_area:.1f} m2 "
            "of swept interior; the arrangement did not capture the room"
        )
        return None, [], diag
    return poly, srcs, diag


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def fit_room(
    cloud: PointCloud | np.ndarray,
    camera_positions: np.ndarray,
    *,
    floor_z: float = 0.0,
    ceiling_z: float | None = None,
    grid: Any | None = None,
    normals: np.ndarray | None = None,
    notes: list[str] | None = None,
    seed: int = 0,
) -> RoomFit | None:
    """Fit floor, ceiling and walls to what the sweep proved about the room.

    Returns None -- rather than a bad room -- when the capture cannot support
    the question: too few cameras to carve with, a carve that cleared almost
    nothing, or no standing surface anywhere near the space that was cleared.
    The caller is expected to fall back to fitting planes to the returns, which
    is the right answer for a LiDAR export and the wrong one only here.
    """
    xyz = np.asarray(getattr(cloud, "xyz", cloud), dtype=np.float64).reshape(-1, 3)
    if normals is None:
        normals = getattr(cloud, "normals", None)
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        if len(normals) != len(xyz):
            normals = None

    cams = np.asarray(camera_positions, dtype=np.float64).reshape(-1, 3)
    if len(cams) < 6 or len(xyz) < 2000:
        return None

    got = _volumes(xyz, cams, grid, seed)
    if got is None:
        return None
    solid, free, origin, voxel = got
    res = float(voxel[0])
    carved = float(free.mean())

    interior = _interior_columns(free, float(voxel[2]), res)
    interior_area = float(interior.sum() * voxel[0] * voxel[1])
    if interior_area < 1.0:
        if notes is not None:
            notes.append(
                "the sweep cleared less than a square metre of room; the walls "
                "were read from the returns alone"
            )
        return None

    ceiling_source = "supplied" if ceiling_z is not None else "none"
    ceiling_inferred: float | None = None
    diag: dict[str, Any] = {}
    if ceiling_z is None:
        found, ceiling_source, diag = _ceiling_from_carve(
            free, interior, xyz, origin, voxel, floor_z, normals
        )
        if ceiling_source == "returns":
            ceiling_z = found
        elif ceiling_source == "carve":
            ceiling_inferred = found
            if notes is not None:
                notes.append(
                    f"free space is capped at {found:.2f} m across the room, so "
                    "that is where the ceiling is drawn -- but almost nothing "
                    "was returned from it, so no measurement is taken off it"
                )
        elif notes is not None and diag.get("ceiling_rejected"):
            notes.append(str(diag["ceiling_rejected"]))

    zlo = floor_z + SLAB_MARGIN_M
    cap = ceiling_z if ceiling_z is not None else ceiling_inferred
    zhi = (cap - SLAB_MARGIN_M) if cap is not None else float(xyz[:, 2].max())
    if zhi - zlo < 0.40:
        zhi = zlo + 0.40
    k0 = max(0, int(math.floor((zlo - origin[2]) / voxel[2])))
    k1 = min(solid.shape[2], int(math.ceil((zhi - origin[2]) / voxel[2])))

    cells = _wall_cells(solid, interior, k0, k1, res)
    if cells.sum() < 3 * int(WALL_MIN_LENGTH_M / res):
        if notes is not None:
            notes.append(
                "no standing surface was found around the swept space, so the "
                "room's walls were read from the returns alone"
            )
        return None

    ij = np.argwhere(cells)
    xy = origin[:2] + (ij + 0.5) * voxel[:2]
    min_cells = max(4, int(round(WALL_MIN_LENGTH_M / res)) // 2)

    theta = _dominant_angle(xy, res)
    axes = [
        np.array([math.cos(theta), math.sin(theta)]),
        np.array([-math.sin(theta), math.cos(theta)]),
    ]
    lines: list[WallLine] = []
    for axis in axes:
        lines += _lines_along(xy, axis, res, min_cells)

    # A second pass on whatever the room's own axes did not explain, so a
    # canted wall or a bay is found rather than smeared across two Manhattan
    # lines that do not exist.
    residual = np.ones(len(xy), dtype=bool)
    for line in lines:
        d = np.abs(xy @ line.normal - line.offset)
        t = xy @ line.direction
        residual &= ~((d <= WALL_INLIER_M) & (t >= line.t0 - res) & (t <= line.t1 + res))
    if residual.sum() >= 4 * min_cells:
        rxy = xy[residual]
        rtheta = _dominant_angle(rxy, res)
        for axis in (
            np.array([math.cos(rtheta), math.sin(rtheta)]),
            np.array([-math.sin(rtheta), math.cos(rtheta)]),
        ):
            lines += _lines_along(rxy, axis, res, min_cells)

    lines = _merge_lines(lines)
    if not lines:
        return None

    inside = origin[:2] + (np.argwhere(interior).mean(axis=0) + 0.5) * voxel[:2]
    _orient(lines, inside)
    dropped = len(lines)
    lines = [
        l
        for l in lines
        if _is_boundary(l, interior, origin, voxel) >= BOUNDARY_MIN_FRACTION
    ]
    dropped -= len(lines)
    if not lines:
        return None
    for line in lines:
        _refit(line, xyz[:, :2], xyz[:, 2], zlo, zhi)
    lines = _merge_lines(lines)
    _orient(lines, inside)
    lines = _resolve_parallel(lines)
    lines.sort(key=lambda l: -l.length)

    walls = [_plane_from_line(l, floor_z, cap) for l in lines]
    # A capture with no ceiling still has doorways in it, and a doorway is
    # under 2.1 m whatever the room's height turns out to be, so the search
    # falls back to the top of the wall evidence rather than declining.
    search_top = cap if cap is not None else max(zhi, floor_z + 2.20)
    openings = _openings_from_carve(
        lines, solid, free, origin, voxel, floor_z, search_top
    )

    footprint, edge_sources, fp_diag = footprint_from_cells(
        lines, interior, origin, voxel, cams[:, :2]
    )
    diag.update(fp_diag)
    if footprint is None and notes is not None and fp_diag.get("footprint_declined"):
        notes.append(
            "the room outline could not be read from the wall arrangement: "
            + str(fp_diag["footprint_declined"])
        )

    diag.update(
        {
            "interior_area_m2": round(interior_area, 3),
            "carved_fraction": round(carved, 4),
            "wall_cells": int(cells.sum()),
            "manhattan_yaw_deg": round(math.degrees(theta), 3),
            "surfaces_dropped_as_contents": int(dropped),
            "openings_found": len(openings),
            "walls_measured": sum(1 for l in lines if l.measured),
            "walls_inferred": sum(1 for l in lines if not l.measured),
        }
    )
    return RoomFit(
        floor_z=float(floor_z),
        ceiling_z=None if ceiling_z is None else float(ceiling_z),
        ceiling_z_inferred=None if ceiling_inferred is None else float(ceiling_inferred),
        walls=walls,
        lines=lines,
        openings=openings,
        interior_area=interior_area,
        footprint=footprint,
        edge_sources=edge_sources,
        ceiling_source=ceiling_source,
        carved_fraction=carved,
        diagnostics=diag,
    )


__all__ = [
    "RoomFit",
    "WallLine",
    "fit_room",
    "footprint_from_cells",
    "straighten_footprint",
]
