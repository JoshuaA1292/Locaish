"""Reading a canonical point cloud as architecture rather than as points.

The rest of the pipeline hands us a cloud that is metric, gravity aligned and
sitting with its floor near z = 0, plus a set of fitted planes. This module
decides which of those planes is the floor, which is the ceiling and which are
walls, traces the outline of the floor, and finds the holes in the walls. The
result is a `Structure`, which is what turns a heap of triangles into a room
that Phase 3 can shine the sun through.

Three ideas do most of the work here.

The first is that a plane's *label* is a statement about the cloud, not about
the plane. A horizontal plane is only a floor if the room is above it, and only
a ceiling if the room is below it and it covers most of the floor's footprint.
The top of a wardrobe passes the "horizontal and high" test and fails both of
the others, which is exactly why it is not allowed to become the ceiling of the
room. Both of those questions are answered as covered *area* rather than as a
share of the points, because a share of the points measures how the room was
scanned: the same floor scores differently depending on how long the operator
lingered on it, and on one shipped fixture that was enough to lose the floor.

The second is that the floor outline must come from the whole occupied column,
not from floor-height points. Furniture stands *on* the floor: sample only the
points near z = floor and every sofa punches a hole in the footprint, and the
polygon that comes out has the topology of a doily.

The third, and the one that decides whether opening detection is useful or
embarrassing, is that a hole in a wall raster is ambiguous. A window, a
wardrobe standing in front of the wall, and the shadow of the person holding
the scanner all produce the same thing: cells with no wall material in them.
Telling them apart needs evidence from *off* the plane -- what is in front of
the aperture, what is behind it, and whether the scanner caught the reveal, the
few centimetres of wall thickness that a genuine opening has and a shadow does
not. `detect_openings` tests all three and reports how well they separated as
the opening's confidence, because an opening is inferred from absence and
downstream code deserves to know how sure we are.

The reveal is the one that carries the weight, and it only carries it if the
depth band it is measured in is set relative to the wall's own noise. A wall's
points are a cloud a centimetre thick, so a band fixed at fifteen millimetres
is comfortably outside that cloud on a careful scan and inside it on a hasty
one -- at which point a tenth of the wall is "reveal", every hole has one, and
the discriminator has quietly stopped discriminating on exactly the rooms it
was never tested against.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Iterable

import numpy as np
from scipy import ndimage
from skimage import measure

from ..geom.planes import detect_planes, plane_frame
from ..types import Opening, Plane, PointCloud, Structure, chunked

# ---------------------------------------------------------------------------
# tuning constants
#
# These are architectural priors, not free parameters. Each one is here because
# some real failure mode needs a line drawn somewhere, and the comment says
# which one.
# ---------------------------------------------------------------------------

#: How far a normal may sit from the vertical (or the horizontal) and still
#: count as level. A phone's gravity vector is good to well under a degree
#: after canonicalisation; 15 degrees is slack for a bowed wall, not for a
#: sloping ceiling.
AXIS_TOL_DEG = 15.0

#: Band around a plane that counts as "on" it. Wide enough for 6 mm range
#: noise plus the 35 mm bow that SLAM drift puts into a long wall, narrow
#: enough that a wardrobe 150 mm off the wall is never mistaken for wall.
PLANE_BAND_M = 0.06

#: A ceiling candidate must roof at least this fraction of the room's
#: footprint, as area rather than as sampled cells. A wardrobe top covers a
#: couple of percent.
CEILING_COVERAGE = 0.35

#: How much of a horizontal plane's own footprint is allowed to have material
#: on the far side of it before it stops being a boundary of the room. A floor
#: has nothing underneath it and a ceiling has nothing above it; the top of a
#: wardrobe has the wardrobe and then the floor, over every square centimetre
#: of itself. The few percent of slack is for the stray returns any scanner
#: puts through a surface and for a scan that saw a sliver of the next room
#: through a doorway. See `_outside_coverage` for why this is measured as
#: covered area rather than as a share of the points. Measured over the
#: catalogue and 50 unseen rooms, real floors and ceilings score 0.000 to 0.178
#: -- the top of that range is `drifty`, whose ceiling genuinely bows through
#: itself -- and furniture tops score 0.50 to 1.00, so the line goes between
#: them with a factor of nearly three either way.
MAX_OUTSIDE_COVERAGE = 0.30

#: A floor or a ceiling has to be a large piece of the room, expressed as a
#: fraction of the room's own XY footprint. This is what stops a large desk in
#: a small room -- which has nothing under it that the scanner ever saw -- from
#: being labelled the floor. On the same rooms, real boundaries measure 0.64 to
#: 1.00 and furniture tops 0.11 to 0.27, and the line sits deliberately near
#: the bottom of that gap rather than in the middle: the outside-coverage test
#: above already separates the two by a factor of three on its own, so this one
#: is the redundant guard, and the expensive mistake for a redundant guard to
#: make is throwing away a floor that is simply half hidden under furniture.
MIN_BOUNDARY_COVERAGE = 0.30

#: Finest cell the coverage rasters above will use. These are ratios of area,
#: not measurements, so 100 mm cells are ample and cost almost nothing. It is a
#: floor on the cell size rather than the cell size: a thinly sampled surface
#: gets coarser cells, because a raster whose cells are mostly empty measures
#: the sampling and not the surface -- see `_footprint_area`.
COVERAGE_RES_M = 0.10

#: The smallest hole we are willing to call an opening, and the most lopsided.
MIN_OPENING_AREA_M2 = 0.15
MAX_OPENING_ASPECT = 6.0

#: Below this area a void has to show its reveal before we will call it an
#: opening. A sensor shadow and a small window are the same shape and have the
#: same emptiness behind them; only the wall's own turned edge tells them
#: apart. Set above a typical shadow (a body at arm's length masks a few tenths
#: of a square metre) and below the smallest aperture worth reporting to a
#: daylight study.
SHADOW_SUSPECT_AREA_M2 = 0.60

#: Depth bands used to interrogate the space through an aperture, in metres of
#: signed distance from the wall plane (positive is into the room). The reveal
#: band is the wall's own thickness, the occluder band is everything standing
#: in the room in front of the aperture, and anything negative is behind the
#: wall. The two positive bands must not overlap: a wardrobe pushed 150 mm off
#: the wall would otherwise read as a 150 mm reveal and turn its own shadow
#: into a window, which is precisely the mistake this function exists to avoid.
#: The reveal band is kept tight against the wall surface for the same reason:
#: a reveal is a surface running *out of* the wall, so its points start at the
#: plaster and continue outwards, whereas furniture is a surface parallel to
#: the wall with a clear gap between it and the plaster.
REVEAL_BAND_M = (0.015, 0.08)
OCCLUDER_BAND_M = (0.08, 2.00)
BEHIND_M = -0.10

#: The reveal band's near edge is pushed out to this many standard deviations
#: of the wall's own range scatter, and never nearer than `REVEAL_BAND_M[0]`.
#:
#: This is the single line that decides whether opening detection generalises.
#: A wall's points do not lie *on* the plane, they lie in a Gaussian cloud
#: around it, so a fixed 15 mm near edge is 5 sigma out on a quiet scan and 1.3
#: sigma out on a noisy one. At 1.3 sigma a tenth of every wall point in the
#: cloud falls inside the "reveal" band, each raster cell beside a hole holds a
#: couple of those, and the reveal -- the one piece of evidence that separates
#: a window from the round shadow a person's body leaves -- reads as present
#: for any hole at all. Measured over 38 unseen rooms, the round occlusion
#: shadows scored 0.12 to 0.22 reveal under the fixed edge, straddling the 0.15
#: acceptance line, and 0.00 to 0.04 under this one, while real openings barely
#: moved (0.32 to 0.85). Four sigma is where a Gaussian has 0.003% of its mass
#: left, which is to say: past here, a point is a surface and not a wall that
#: was measured badly.
REVEAL_NOISE_SIGMAS = 4.0
#: ... and what is left of the band after that has to be wide enough to hold a
#: reveal. On a wall whose own points scatter by two centimetres, "fifteen
#: millimetres in front of the plaster" is not a place, and no depth evidence
#: taken there means anything: such a wall reports the reveal as unmeasurable
#: and can only produce an opening the scanner actually saw through.
#:
#: This doubles as the guard against a badly fitted plane. Across the catalogue
#: and 38 unseen rooms, the 130 planes that are genuinely a wall of the room
#: measure 3.1 to 12.6 mm of scatter, tracking their scan's range noise almost
#: exactly, while every spurious extra plane -- the diagonal slabs RANSAC cuts
#: through a wall it has already found -- measures 17.6 to 41.6 mm. Those
#: slabs are where the last false window came from: tilted a degree or two out
#: of the wall, they sweep the real plaster through the reveal band, and every
#: hole in them arrives with a reveal. The exception is `drifty`, whose real
#: walls bow to 18-34 mm, and which consequently stops reporting openings --
#: correctly, because on a wall that is genuinely 35 mm from where we say it
#: is, we cannot tell a jamb from the wall.
MIN_REVEAL_BAND_M = 0.02

#: An aperture whose near field is this full is a thing standing in front of
#: the wall, not a hole through it.
MAX_OCCLUDER_FRACTION = 0.60

#: The smallest plane we are prepared to search for openings in, measured
#: along the wall and up it. A vertical plane smaller than this is furniture --
#: the side of a wardrobe, the back of a bookcase -- and the gaps in it are not
#: windows. `label_planes` still calls it a wall, because geometrically it is
#: one and a consumer may want it; this is opening detection's own filter.
MIN_WALL_SPAN_M = 1.2

#: How much of each end of a wall is treated as corner rather than wall. Where
#: two walls meet, the plane fit runs out, the neighbouring wall's points bleed
#: into the band, and that neighbour -- being perpendicular -- offers a perfect
#: imitation of a reveal to anything sitting in the corner. A void that reaches
#: into that strip is the join between two walls, not a hole in one.
CORNER_MARGIN_M = 0.12

#: A genuine opening must either show its reveal or show something behind it.
#: A scanner shadow shows neither, which is the whole discriminator.
MIN_REVEAL_FRACTION = 0.15
MIN_BEHIND_FRACTION = 0.25

#: Expected points per raster cell. Below this the raster is measuring Poisson
#: gaps rather than geometry, so the cell size is grown until it is met -- see
#: `_adaptive_resolution`.
POINTS_PER_CELL = 3.0

#: Percentile used to place an opening's edge against the wall points that
#: stop at it. Range noise scatters a thin tail of wall points across the edge
#: into the aperture, so the extreme value reads outside the wall and any
#: estimator has to step over that tail; step too far and the edge is dragged
#: into the material instead.
EDGE_PERCENTILE = 1.0

#: Statistics (what fraction of the cloud is above a plane, and so on) are
#: computed on a deterministic stride subsample. Half a million points pins a
#: fraction to four decimal places; twenty million just costs more.
STAT_SAMPLE_LIMIT = 500_000

_UP = np.array([0.0, 0.0, 1.0])
_CHUNK = 1_000_000


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------


def _xyz(cloud: PointCloud | np.ndarray) -> np.ndarray:
    """Accept either a PointCloud or a bare (N, 3) array.

    Callers inside the pipeline always have a PointCloud, but the QA module and
    the tests often hold a raw array, and refusing it buys nothing.
    """
    if isinstance(cloud, PointCloud):
        return cloud.xyz
    return np.asarray(cloud, dtype=np.float64).reshape(-1, 3)


def _stat_sample(xyz: np.ndarray, limit: int = STAT_SAMPLE_LIMIT) -> np.ndarray:
    """A deterministic stride subsample, used wherever we only need a ratio.

    Stride rather than random choice because the answer must be identical
    between runs and between machines, and because a stride over a cloud that
    is written surface by surface still keeps every surface in proportion.
    """
    if len(xyz) <= limit:
        return xyz
    step = int(math.ceil(len(xyz) / limit))
    return xyz[::step]


def _signed(xyz: np.ndarray, plane: Plane) -> np.ndarray:
    """Signed distance to a plane, evaluated in chunks.

    The naive form allocates a temporary as large as the cloud; at twenty
    million points that is 160 MB per call and we make several per plane.
    """
    out = np.empty(len(xyz), dtype=np.float64)
    n = plane.normal
    for sl in chunked(len(xyz), _CHUNK):
        out[sl] = xyz[sl] @ n - plane.offset
    return out


def _oriented(plane: Plane, sample: np.ndarray) -> Plane:
    """A copy of `plane` whose normal points into the room.

    Every downstream sign convention -- clearance, "above the floor", "in front
    of the wall" -- assumes the normal faces the interior. A plane fitted by
    SVD has an arbitrary sign, so we resolve it against the cloud: for any
    boundary surface of a room, the room is the side the bulk of the points are
    on. Interior clutter is ambiguous under this rule and is labelled `other`
    anyway, so the ambiguity never reaches a consumer.
    """
    d = sample @ plane.normal - plane.offset
    if np.count_nonzero(d > 0) >= np.count_nonzero(d < 0):
        return replace(plane)
    return replace(plane, normal=-plane.normal, offset=-plane.offset)


def _raster(
    a: np.ndarray, b: np.ndarray, a0: float, b0: float, res: float, na: int, nb: int
) -> np.ndarray:
    """Count points into an (na, nb) cell grid. Vectorised, no accumulation loop."""
    ia = np.floor((a - a0) / res).astype(np.int64)
    ib = np.floor((b - b0) / res).astype(np.int64)
    ok = (ia >= 0) & (ia < na) & (ib >= 0) & (ib < nb)
    if not np.any(ok):
        return np.zeros((na, nb), dtype=np.int64)
    flat = ia[ok] * nb + ib[ok]
    return np.bincount(flat, minlength=na * nb).reshape(na, nb)


def _adaptive_resolution(requested: float, count: int, area: float) -> float:
    """Coarsen a raster until its cells are expected to contain points.

    A raster whose cells are usually empty does not measure geometry, it
    measures the Poisson gaps between samples, and every one of those gaps
    looks like a small window. At 900 points/m2 -- a careful iPhone sweep -- a
    40 mm cell holds 1.4 points and one cell in four is empty by chance; on the
    `sparse` fixture it is three in four and the empty phase percolates. So the
    caller's resolution is treated as a floor on the detail we ask for, not a
    promise, and openings are measured from point positions afterwards anyway.
    """
    if count <= 0 or area <= 0:
        return requested
    density = count / area
    needed = math.sqrt(POINTS_PER_CELL / density)
    return float(max(requested, needed))


# ---------------------------------------------------------------------------
# labelling
# ---------------------------------------------------------------------------


def label_planes(planes: list[Plane], cloud: PointCloud) -> list[Plane]:
    """Assign `kind` to every plane, returning new objects.

    The geometric test (is the normal near +Z, near -Z, or near horizontal) is
    necessary but nowhere near sufficient: a desk, a shelf and the top of a
    wardrobe are all "near +/-Z". What separates a floor from a desk is that
    the room sits above the floor and nothing sits below it, and what separates
    a ceiling from a wardrobe top is the same statement upside down.

    That question used to be answered with a point fraction -- "at least 85% of
    the cloud is above this plane" -- and a point fraction is a statement about
    how the room was sampled, not about what the plane is. It changes when the
    scanner spends longer on the floor, when the ceiling is missing from the
    capture, when a wall is decimated harder than the rest. It failed exactly
    that way on the `noceiling` fixture: removing the ceiling raised the
    floor's own share of the cloud until the floor's inliers, half of which sit
    a hair *below* their own plane, dragged the "above" fraction to 0.845 and
    the room's only floor was labelled `other`. So the test is now geometric.
    A boundary of the room is a large horizontal surface with nothing on its
    far side, both of which are measured as covered area over the plane's own
    footprint -- see `_outside_coverage` -- and neither of which cares how
    densely anything was sampled.

    Vertical planes are labelled `wall` on the geometry alone, deliberately:
    the side of a tall bookcase really is wall-like, and pruning it here would
    hide it from consumers that want it. `detect_openings` does its own size
    filtering rather than trusting this label to have done it.
    """
    xyz = _xyz(cloud)
    if len(xyz) == 0:
        return [replace(p) for p in planes]

    sample = _stat_sample(xyz)
    z = sample[:, 2]
    z_high = float(np.percentile(z, 97.0))
    z_low = float(np.percentile(z, 3.0))
    span = max(z_high - z_low, 1e-6)

    # The room's own footprint, which the "large" test is a fraction of. It is
    # taken over the whole occupied column rather than at any one height, so
    # every point in the cloud votes and the raster is never short of material.
    room_area = max(_footprint_area(sample), COVERAGE_RES_M**2)

    cos_tol = math.cos(math.radians(AXIS_TOL_DEG))
    sin_tol = math.sin(math.radians(AXIS_TOL_DEG))

    out: list[Plane] = []
    for plane in planes:
        p = _oriented(plane, sample)
        nz = float(p.normal @ _UP)

        kind = "other"
        if abs(nz) <= sin_tol:
            kind = "wall"
        elif abs(nz) >= cos_tol:
            outside, footprint_area, plane_z = _outside_coverage(sample, p)
            large = footprint_area / room_area >= MIN_BOUNDARY_COVERAGE
            clear = outside <= MAX_OUTSIDE_COVERAGE
            # "low" and "high" are relative to the cloud, not to zero: a twin
            # that skipped canonicalisation still has a floor, just not at z = 0
            if nz > 0 and large and clear and plane_z <= z_low + 0.25 * span:
                kind = "floor"
            elif nz < 0 and large and clear and plane_z >= z_high - 0.25 * span:
                kind = "ceiling"

        out.append(replace(p, kind=kind))
    return out


def _outside_coverage(
    sample: np.ndarray, plane: Plane, res: float = COVERAGE_RES_M
) -> tuple[float, float, float]:
    """How much of a horizontal plane has the room on the wrong side of it.

    Returns the covered fraction, the plane's footprint in square metres, and
    the median height of its inliers.

    The measurement is a raster over the plane's own footprint: what fraction
    of the cells the plane occupies also contain a point on the far side of it,
    where "far side" means against its normal, which by convention points into
    the room. A floor scores zero because there is nothing under a floor; the
    top of a wardrobe scores one because the wardrobe's own sides and then the
    floor sit under every cell of it. The point of counting cells rather than
    points is that the answer is then a fact about the room: doubling the
    sampling density of the wardrobe changes the point counts and not this
    number, whereas the point-fraction test it replaces moved far enough on an
    ordinary change of capture to lose a floor entirely.

    The plane band is excluded from both sides so that the plane's own inliers,
    half of which land on either side of it under range noise, cannot vote.

    The cells are grown until they are expected to contain a point, and the
    footprint comes back as an area rather than as a count, because otherwise
    this function would smuggle the sampling dependence it exists to remove
    back in through the raster: a ceiling the scanner only glanced at leaves
    three quarters of its own 100 mm cells empty and would measure as a quarter
    of the room, which is how you erase a real ceiling by decimating it.

    The height comes back from here because the median of the inliers is the
    honest height of a surface, whereas `offset / n_z` is only the height of
    the fitted plane directly above the origin and tilts away from it.
    """
    d = sample @ plane.normal - plane.offset
    on = np.abs(d) <= PLANE_BAND_M
    n_on = int(np.count_nonzero(on))
    if n_on < 10:
        return 1.0, 0.0, float("nan")
    inliers = sample[on]
    plane_z = float(np.median(inliers[:, 2]))

    lo = inliers[:, :2].min(axis=0)
    hi = inliers[:, :2].max(axis=0)
    res = _adaptive_resolution(res, n_on, float((hi[0] - lo[0]) * (hi[1] - lo[1])))
    nx = int(math.ceil((hi[0] - lo[0]) / res)) + 1
    ny = int(math.ceil((hi[1] - lo[1]) / res)) + 1
    foot = _raster(inliers[:, 0], inliers[:, 1], lo[0], lo[1], res, nx, ny) > 0
    cells = int(np.count_nonzero(foot))
    area = cells * res * res
    if cells == 0:
        return 1.0, 0.0, plane_z

    beyond = sample[d < -PLANE_BAND_M]
    if len(beyond) == 0:
        return 0.0, area, plane_z
    occupied = _raster(beyond[:, 0], beyond[:, 1], lo[0], lo[1], res, nx, ny) > 0
    return float(np.count_nonzero(foot & occupied) / cells), area, plane_z


# ---------------------------------------------------------------------------
# floor and ceiling heights
# ---------------------------------------------------------------------------


def _column_mask(
    xyz: np.ndarray, res: float, z_lo: float, z_hi: float, pad_cells: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Occupancy of the XY cells that have anything in them between two heights.

    Returns the boolean mask and the world XY of cell (0, 0)'s lower corner.
    """
    keep = (xyz[:, 2] >= z_lo) & (xyz[:, 2] <= z_hi)
    pts = xyz[keep]
    if len(pts) == 0:
        return np.zeros((1, 1), dtype=bool), np.zeros(2)
    lo = pts[:, :2].min(axis=0) - pad_cells * res
    hi = pts[:, :2].max(axis=0) + pad_cells * res
    nx = int(math.ceil((hi[0] - lo[0]) / res)) + 1
    ny = int(math.ceil((hi[1] - lo[1]) / res)) + 1
    counts = _raster(pts[:, 0], pts[:, 1], lo[0], lo[1], res, nx, ny)
    return counts > 0, lo


def find_floor_ceiling(
    cloud: PointCloud, planes: list[Plane]
) -> tuple[float, float | None]:
    """Robust floor and ceiling heights, with `None` for an unroofed capture.

    Two things are deliberately not done. The plane's own offset is not used as
    the height: a plane fitted to a floor that includes the bottom faces of the
    furniture standing on it is pulled a few millimetres off, and the median of
    its inliers is not. And a ceiling candidate is not accepted on height
    alone. It has to roof a real fraction of the floor's footprint, because the
    top of a wardrobe is horizontal, faces down from the room's point of view,
    and sits near the top of the cloud -- every test but the coverage test
    passes, and a room whose ceiling height is the height of its wardrobe is
    worse than a room with no ceiling height at all.

    An open-topped capture returns `None`. Inventing a number there would put a
    roof into the daylight study that does not exist in the building.
    """
    xyz = _xyz(cloud)
    if len(xyz) == 0:
        return 0.0, None

    floor_candidates = [p for p in planes if p.kind == "floor"]
    if floor_candidates:
        best = max(floor_candidates, key=lambda p: _inlier_count(xyz, p))
        z_in = xyz[np.abs(_signed(xyz, best)) <= PLANE_BAND_M, 2]
        floor_z = float(np.median(z_in)) if len(z_in) else float(best.offset)
    else:
        floor_z = _modal_floor_z(xyz)

    ceiling_z = _pick_ceiling(xyz, planes, floor_z)
    return floor_z, ceiling_z


def _inlier_count(xyz: np.ndarray, plane: Plane) -> int:
    sample = _stat_sample(xyz)
    return int(np.count_nonzero(np.abs(sample @ plane.normal - plane.offset) <= PLANE_BAND_M))


def _modal_floor_z(xyz: np.ndarray) -> float:
    """Floor height with no floor plane to lean on: the lowest dense z band.

    A percentile of z would be dragged down by the handful of stray returns
    every scanner puts below the floor, so we look for the lowest 20 mm bin
    that holds a serious share of the points instead.
    """
    z = _stat_sample(xyz)[:, 2]
    lo, hi = float(z.min()), float(z.max())
    if hi - lo < 1e-6:
        return lo
    bins = max(8, int(math.ceil((hi - lo) / 0.02)))
    hist, edges = np.histogram(z, bins=bins, range=(lo, hi))
    strong = np.flatnonzero(hist >= 0.25 * hist.max())
    if len(strong) == 0:
        return float(np.percentile(z, 1.0))
    b = int(strong[0])
    top = float(edges[min(b + 2, len(edges) - 1)])
    band = z[(z >= edges[b]) & (z <= top)]
    return float(np.median(band)) if len(band) else float(edges[b])


def _footprint_area(xyz: np.ndarray, res: float = COVERAGE_RES_M) -> float:
    """The XY area a set of points covers, in square metres.

    The cell size is grown until a cell is expected to hold points, and the
    answer is an area rather than a cell count, so that the same surface
    measures the same whether the scanner swept it once or four times. Counting
    cells at a fixed size instead makes this a measure of sampling density
    wearing a geometry costume, and every threshold built on it then moves when
    the capture changes.
    """
    if len(xyz) == 0:
        return 0.0
    lo = xyz[:, :2].min(axis=0)
    hi = xyz[:, :2].max(axis=0)
    res = _adaptive_resolution(res, len(xyz), float((hi[0] - lo[0]) * (hi[1] - lo[1])))
    nx = int(math.ceil((hi[0] - lo[0]) / res)) + 1
    ny = int(math.ceil((hi[1] - lo[1]) / res)) + 1
    counts = _raster(xyz[:, 0], xyz[:, 1], lo[0], lo[1], res, nx, ny)
    return float(np.count_nonzero(counts) * res * res)


def _pick_ceiling(xyz: np.ndarray, planes: list[Plane], floor_z: float) -> float | None:
    candidates = [p for p in planes if p.kind == "ceiling"]
    if not candidates:
        return None

    sample = _stat_sample(xyz)
    # the footprint the ceiling has to compete against, taken over the whole
    # occupied column so that every point in the room votes for it
    keep = (sample[:, 2] >= floor_z - 0.1) & (sample[:, 2] <= float(sample[:, 2].max()) + 0.1)
    room_area = _footprint_area(sample[keep])
    if room_area <= 0:
        return None

    best: tuple[float, float] | None = None  # (coverage, z)
    for p in candidates:
        d = sample @ p.normal - p.offset
        inl = sample[np.abs(d) <= PLANE_BAND_M]
        if len(inl) == 0:
            continue
        z_med = float(np.median(inl[:, 2]))
        if z_med - floor_z < 1.6:
            # nothing below head height is a ceiling; it is a soffit or a shelf
            continue
        coverage = _footprint_area(inl) / room_area
        if coverage >= CEILING_COVERAGE and (best is None or coverage > best[0]):
            best = (coverage, z_med)

    return None if best is None else best[1]


# ---------------------------------------------------------------------------
# floor footprint
# ---------------------------------------------------------------------------


def floor_footprint(
    cloud: PointCloud,
    floor_z: float,
    *,
    ceiling_z: float | None = None,
    resolution: float = 0.05,
    grid: Any | None = None,
) -> np.ndarray:
    """The floor outline as an (M, 2) CCW polygon.

    Rastering the *whole occupied column* rather than the floor-height points
    is the point of this function. Furniture stands on the floor and hides it,
    so a floor-height raster is a floor with sofa-shaped holes in it; the
    column raster says "something is here, therefore the floor reaches here",
    which is the question a footprint is asking.

    The rest is damage control on the raster. Closing bridges the gaps that
    occlusion leaves along the skirting, hole filling deals with an unscanned
    patch in the middle of the room, and keeping the largest connected
    component discards the points a window let the scanner see outside. The
    outline is then traced with marching squares and simplified, which is what
    lets an L-shaped or bay-windowed room come back as its own shape -- a
    convex hull would quietly fill the L in and overstate the area.

    The traced outline is finally pulled in by half a cell. Marching squares
    puts its 0.5 contour midway between the last occupied cell centre and the
    first empty one, so the raw trace sits half a cell outside the wall on
    every side -- a fixed, one-sided error that inflates a 21 m2 room by about
    half a square metre. Offsetting the polygon along its own edge normals
    removes exactly that bias and, unlike shrinking about the centroid, stays
    correct at the reflex corner of an L-shaped room.
    """
    xyz = _xyz(cloud)
    mask, lo, res = _column_mask_for_footprint(xyz, floor_z, ceiling_z, resolution, grid)
    if not np.any(mask):
        return np.zeros((0, 2))

    close_cells = max(1, int(round(0.12 / res)))
    mask = _closed(mask, close_cells)
    mask = ndimage.binary_fill_holes(mask)

    labels, n = ndimage.label(mask)
    if n > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        mask = labels == int(sizes.argmax())

    contours = measure.find_contours(mask.astype(np.float64), 0.5)
    if not contours:
        return np.zeros((0, 2))
    contour = max(contours, key=len)
    if len(contour) > 1 and np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]

    poly = np.stack(
        [lo[0] + (contour[:, 0] + 0.5) * res, lo[1] + (contour[:, 1] + 0.5) * res], axis=1
    )
    # The polygon is simplified only far enough to drop collinear runs; a
    # quarter of a cell is below anything the raster can resolve. Simplifying
    # harder gives a prettier outline and a worse number, because
    # Douglas-Peucker anchors on extreme vertices: smoothing away the one-cell
    # staircase where a wall crosses a cell boundary skews the whole edge and
    # loses about a percent of the area, which is the one quantity a footprint
    # exists to provide.
    poly = _ccw(_simplify(poly, tolerance=0.25 * res))
    return _offset_inward(poly, 0.5 * res)


def _column_mask_for_footprint(
    xyz: np.ndarray,
    floor_z: float,
    ceiling_z: float | None,
    resolution: float,
    grid: Any | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Column occupancy, taken from a prebuilt grid when one is offered.

    `geom.grid` already pays for this raster during ingest, so reusing it saves
    a pass over the cloud; but the grid is optional and may have been built
    with a different voxel size, so anything unexpected about it falls back to
    rastering here rather than guessing at its contract.
    """
    if grid is not None:
        got = _grid_column_mask(grid)
        if got is not None:
            return got[0], got[1], got[2]
    z_hi = float(xyz[:, 2].max()) if len(xyz) else 0.0
    if ceiling_z is not None:
        # trim the ceiling itself: it spans the whole room and would hide a
        # genuinely unscanned corner, and anything above it is outside
        z_hi = ceiling_z - 0.05
    mask, lo = _column_mask(xyz, resolution, floor_z - 0.10, z_hi)
    return mask, lo, resolution


def _grid_column_mask(grid: Any) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Column occupancy from an `OccupancyGrid`, or None if it cannot be used.

    Duck-typed rather than isinstance-checked so that a caller can pass any
    object carrying the same three attributes, and so that a grid built with
    settings this raster cannot honour is declined instead of misread.
    """
    occupied = getattr(grid, "occupied", None)
    origin = getattr(grid, "origin", None)
    voxel = getattr(grid, "voxel", None)
    if occupied is None or origin is None or voxel is None:
        return None
    occupied = np.asarray(occupied)
    if occupied.ndim != 3:
        return None
    voxel_xy = np.atleast_1d(np.asarray(voxel, dtype=np.float64)).ravel()
    res = float(voxel_xy[0])
    if not np.isfinite(res) or res <= 0:
        return None
    if len(voxel_xy) > 1 and abs(float(voxel_xy[1]) - res) > 1e-9:
        # a grid with different X and Y voxel sizes cannot be rastered as one
        # square-celled mask, and silently using the X size would stretch the
        # footprint; rasterise here instead
        return None
    lo = np.asarray(origin, dtype=np.float64).ravel()[:2]
    return occupied.any(axis=2), lo, res


def _simplify(poly: np.ndarray, tolerance: float) -> np.ndarray:
    """Ramer-Douglas-Peucker on a closed polygon, iterative to avoid recursion
    limits on a contour that can carry a few thousand vertices."""
    if len(poly) <= 4:
        return poly
    closed = np.vstack([poly, poly[:1]])
    keep = np.zeros(len(closed), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(closed) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        seg = closed[j] - closed[i]
        length = float(np.linalg.norm(seg))
        pts = closed[i + 1 : j]
        if length < 1e-12:
            dist = np.linalg.norm(pts - closed[i], axis=1)
        else:
            rel = pts - closed[i]
            # 2D cross product written out: numpy 2 dropped the 2-vector form
            dist = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / length
        k = int(dist.argmax())
        if dist[k] > tolerance:
            keep[i + 1 + k] = True
            stack.append((i, i + 1 + k))
            stack.append((i + 1 + k, j))
    out = closed[keep][:-1]
    return out if len(out) >= 3 else poly


def _offset_inward(poly: np.ndarray, distance: float) -> np.ndarray:
    """Move every edge of a CCW polygon `distance` towards the interior.

    A mitred offset: each edge line is translated along its inward normal and
    the new vertices are the intersections of consecutive translated lines.
    Consecutive edges that are almost collinear have no usable intersection, so
    those vertices are simply translated, which is the limit of the mitre
    anyway. Any offset that turns the polygon inside out is discarded and the
    original returned -- a footprint that is roughly the size of a raster cell
    has bigger problems than a half-cell bias.
    """
    if len(poly) < 3 or distance <= 0:
        return poly
    nxt = np.roll(poly, -1, axis=0)
    edge = nxt - poly
    length = np.linalg.norm(edge, axis=1, keepdims=True)
    length = np.where(length == 0, 1.0, length)
    d = edge / length
    # interior is to the left of a CCW edge
    inward = np.stack([-d[:, 1], d[:, 0]], axis=1)
    base = poly + inward * distance

    out = np.empty_like(poly)
    prev = np.roll(np.arange(len(poly)), 1)
    for i in range(len(poly)):
        j = prev[i]
        cross = d[j, 0] * d[i, 1] - d[j, 1] * d[i, 0]
        if abs(cross) < 1e-9:
            out[i] = base[i]
            continue
        diff = base[i] - base[j]
        t = (diff[0] * d[i, 1] - diff[1] * d[i, 0]) / cross
        out[i] = base[j] + d[j] * t
    q = np.roll(out, -1, axis=0)
    area = float(np.sum(out[:, 0] * q[:, 1] - q[:, 0] * out[:, 1]) / 2.0)
    return out if area > 0 else poly


def _closed(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Morphological closing that does not eat the array border.

    `binary_closing` erodes with a background border, so material sitting on
    the edge of the array is quietly removed and reappears as a void. Padding
    first keeps the erosion's border effect out in the padding where it belongs.
    """
    pad = iterations + 1
    big = np.zeros((mask.shape[0] + 2 * pad, mask.shape[1] + 2 * pad), dtype=bool)
    big[pad:-pad, pad:-pad] = mask
    big = ndimage.binary_closing(
        big, structure=np.ones((3, 3), bool), iterations=iterations
    )
    return big[pad:-pad, pad:-pad]


def _ccw(poly: np.ndarray) -> np.ndarray:
    if len(poly) < 3:
        return poly
    q = np.roll(poly, -1, axis=0)
    signed = float(np.sum(poly[:, 0] * q[:, 1] - q[:, 0] * poly[:, 1]) / 2.0)
    return poly if signed >= 0 else poly[::-1].copy()


# ---------------------------------------------------------------------------
# openings
# ---------------------------------------------------------------------------


def detect_openings(
    cloud: PointCloud,
    wall_planes: Iterable[Plane],
    *,
    floor_z: float,
    ceiling_z: float | None,
    resolution: float = 0.04,
    seed: int = 0,
) -> list[Opening]:
    """Find the windows, doors and pass-throughs in a set of wall planes.

    A window in a LiDAR scan is an absence: the beam went through the wall and
    either hit nothing or hit something several metres beyond it. So we project
    each wall's inliers into the wall's own frame, raster them, and look for
    connected runs of empty cells. That much a naive hole finder does too, and
    it reports the bookshelf as a window.

    What makes the difference is that an aperture is interrogated in depth
    before it is believed. Anything standing in front of the wall, within a
    couple of metres of it and filling most of the aperture, is an occluder:
    the hole exists because the scanner could not see past the wardrobe. Points
    behind the wall plane are the opposite evidence, since only a real opening
    lets the beam through. And between the two lies the reveal -- the wall's own
    thickness, seen as a thin band of points hugging the plaster just inside the
    aperture edge -- which a real opening has and a shadow does not, and which
    is the only signal that separates a window in a room with nothing outside
    it from the round hole the person holding the scanner left behind.

    A hole is accepted when the near field is not full and either the reveal or
    the behind-the-wall evidence is there. All three measurements then go into
    `confidence`, so a caller can tell a window we watched daylight through from
    one we inferred.

    Whether that works on a room nobody tuned it against comes down to where
    the reveal band starts. It is not a fixed distance from the plane: it is
    four standard deviations of the wall's own scatter, because a wall's points
    are a cloud and not a surface, and a band that starts inside that cloud
    fills up with wall. See `REVEAL_NOISE_SIGMAS`, which is the line between
    finding the windows in an unseen room and finding the scanner's shadow.

    The aperture is a rectangle fitted to the void rather than the void's
    bounding box, because a hole in a wall raster is very often a window with
    an occlusion shadow fused to one edge, and a bounding box reports the union.
    See `_straight_rect`.

    Widths and heights are then not read off the raster at all. Its cells are
    40 mm at best and coarser on a thinly sampled wall, so each edge is
    remeasured against the geometry: against the reveal running out of the wall
    at that edge where there is one, and against the wall points that stop at
    it otherwise. Edges with neither -- the bottom of a door, the top of an
    opening that runs into the ceiling -- are snapped to the floor or ceiling
    instead. See `_refine_edges`.

    `seed` is accepted so callers can thread one seed through the whole scan
    stage; nothing here samples randomly, so the result is identical for every
    value of it.
    """
    xyz = _xyz(cloud)
    if len(xyz) == 0:
        return []

    found: list[Opening] = []
    for plane in wall_planes:
        found.extend(
            _openings_in_wall(
                xyz, plane, floor_z=floor_z, ceiling_z=ceiling_z, resolution=resolution
            )
        )
    return _dedupe(found)


def _openings_in_wall(
    xyz: np.ndarray,
    plane: Plane,
    *,
    floor_z: float,
    ceiling_z: float | None,
    resolution: float,
) -> list[Opening]:
    """Every opening in one wall plane. See `detect_openings` for the reasoning."""
    # `plane_frame` is the shared convention: v is the in-plane steepest
    # ascent, which on a wall is +Z, and u = v x n runs along the wall. Every
    # width, height and sill below is expressed in it, so it is imported rather
    # than reimplemented -- two copies of a sign convention is how a window
    # ends up on its side.
    u_ax, v_ax = plane_frame(plane)

    # A plane detector run on a furnished room returns the side of every
    # wardrobe as well as the four walls, and each of those costs a pass over
    # the whole cloud to find out it is too small. When the detector has
    # already recorded the inliers' extent -- in this same frame -- the small
    # ones can be dropped before that pass rather than after it.
    if plane.extent_2d is not None:
        umin, vmin, umax, vmax = plane.extent_2d
        if umax - umin < MIN_WALL_SPAN_M or vmax - vmin < MIN_WALL_SPAN_M:
            return []

    s = _signed(xyz, plane)
    near = np.abs(s) <= PLANE_BAND_M
    if np.count_nonzero(near) < 200:
        return []
    wall_pts = xyz[near]
    wu = wall_pts @ u_ax
    wv = wall_pts @ v_ax

    # How thick this wall's own point cloud is, as a robust standard deviation
    # of the inliers' signed distance. Everything the depth evidence does is
    # relative to this: a "point in front of the plaster" only means something
    # once we know how far in front of the plaster the plaster itself lands.
    # The median absolute deviation rather than np.std because a wall's inliers
    # include the odd picture frame and the skirting board.
    sigma = float(np.median(np.abs(s[near] - np.median(s[near]))) * 1.4826)
    reveal_lo = max(REVEAL_BAND_M[0], REVEAL_NOISE_SIGMAS * sigma)
    reveal_band: tuple[float, float] | None = (
        (float(reveal_lo), REVEAL_BAND_M[1])
        if REVEAL_BAND_M[1] - reveal_lo >= MIN_REVEAL_BAND_M
        else None
    )

    # the wall's own extent, trimmed of the few stragglers that any plane fit
    # picks up from the perpendicular walls meeting it at the corners
    u0, u1 = (float(x) for x in np.percentile(wu, [0.3, 99.7]))
    v_lo = floor_z
    v_hi = float(np.percentile(wv, 99.7)) if ceiling_z is None else ceiling_z
    if u1 - u0 < MIN_WALL_SPAN_M or v_hi - v_lo < MIN_WALL_SPAN_M:
        # too small to be a room wall: the side of a wardrobe, or a sliver of a
        # plane fit that happened to be vertical
        return []

    res = _adaptive_resolution(resolution, len(wall_pts), (u1 - u0) * (v_hi - v_lo))
    # round down, never up: a final row that reaches above the ceiling holds no
    # material, so it would read as a void strip running the width of the wall
    # and would swallow any opening that reaches the ceiling, such as an arch
    nu = max(3, int((u1 - u0) / res))
    nv = max(3, int((v_hi - v_lo) / res))
    counts = _raster(wu, wv, u0, v_lo, res, nu, nv)
    material_raw = counts > 0
    if np.count_nonzero(material_raw) < 0.25 * material_raw.size:
        # a wall we barely saw; every gap in it would be a "window"
        return []
    # Close voids narrower than roughly 250 mm. Two things force this. The
    # sampling is Poisson, so a few percent of cells are empty by luck, and a
    # chain of those cells will bridge a real opening to the unscanned strip at
    # the end of the wall -- once that happens the opening touches the border
    # and is thrown away, which is how the door in the fixture goes missing.
    # And a square structuring element leaves a rectangular void untouched, so
    # closing costs the aperture nothing while removing every bridge.
    close_cells = max(1, int(round(0.12 / res)))
    material = _closed(material_raw, close_cells)
    # Closing pulls the void in by up to its own radius, and the raster
    # quantises the edge by another cell, so anything that wants to look at the
    # aperture's true edge -- the reveal, above all -- has to reach this far
    # outside the void the raster reports.
    edge_pad = close_cells + 1

    # Wrap the raster in a ring that says what we know about the wall's edges.
    # The floor is a hard boundary, so a door that runs down to it is still
    # fully enclosed; the ceiling is only a boundary when we found one; the two
    # ends run into the corners of the room, which the plane fit does not
    # reliably reach, so a void that touches them is unscanned wall and not an
    # opening.
    padded = np.zeros((nu + 2, nv + 2), dtype=bool)
    padded[1:-1, 1:-1] = material
    padded[:, 0] = True
    if ceiling_z is not None:
        padded[:, -1] = True

    void = ~padded
    labels, n_void = ndimage.label(void, structure=ndimage.generate_binary_structure(2, 1))
    if n_void == 0:
        return []

    ring = np.zeros_like(void)
    ring[0, :] = ring[-1, :] = True
    ring[:, 0] = ring[:, -1] = True
    open_to_edge = set(np.unique(labels[ring & void]).tolist()) - {0}

    # Everything we might see through an aperture, projected once per wall.
    # Only points inside the depth bands can contribute to the evidence, and
    # dropping the rest first keeps three cloud-sized temporaries from being
    # allocated for a wall that cares about a slab two metres deep.
    near_edge = OCCLUDER_BAND_M[0] if reveal_band is None else reveal_band[0]
    relevant = ((s > near_edge) & (s < OCCLUDER_BAND_M[1])) | (s < BEHIND_M)
    rs = s[relevant]
    rp = xyz[relevant]
    pu = rp @ u_ax
    pv = rp @ v_ax

    out: list[Opening] = []
    for lab in range(1, n_void + 1):
        if lab in open_to_edge:
            continue
        comp = labels == lab
        cells = int(np.count_nonzero(comp))
        if cells * res * res < MIN_OPENING_AREA_M2:
            continue
        # back out of the padded frame into raster indices
        i0, i1, j0, j1, straight = _straight_rect(comp)
        i0, i1, j0, j1 = i0 - 1, i1 - 1, j0 - 1, j1 - 1
        if i0 * res < CORNER_MARGIN_M or (nu - 1 - i1) * res < CORNER_MARGIN_M:
            continue
        box_u = (i1 - i0 + 1) * res
        box_v = (j1 - j0 + 1) * res
        if max(box_u, box_v) > MAX_OPENING_ASPECT * max(min(box_u, box_v), 1e-6):
            continue
        rect_fill = min(1.0, cells / float((i1 - i0 + 1) * (j1 - j0 + 1)))
        if cells > 0.6 * material_raw.size:
            # more absence than wall: this is a wall we failed to scan
            continue

        void_area = cells * res * res

        aperture = (u0 + i0 * res, v_lo + j0 * res, u0 + (i1 + 1) * res, v_lo + (j1 + 1) * res)
        depth = _depth_evidence(
            rs, pu, pv, aperture, res, (i1 - i0 + 1), (j1 - j0 + 1),
            floor_z=floor_z, ceiling_z=ceiling_z, reveal_band=reveal_band,
            pad=edge_pad, jamb_limits=(u0 + CORNER_MARGIN_M, u1 - CORNER_MARGIN_M),
        )
        if depth["front"] > MAX_OCCLUDER_FRACTION:
            continue
        if depth["reveal"] < MIN_REVEAL_FRACTION and depth["behind"] < MIN_BEHIND_FRACTION:
            continue

        # A small void needs its reveal specifically, not merely an absence
        # behind it. Where the beam simply never reached a patch of wall -- a
        # body in the way, a grazing angle, a dark or glossy surface -- what is
        # left is a hole with nothing behind it and nothing in front of it,
        # which is geometrically indistinguishable from a small window looking
        # out at open sky. No amount of depth reasoning separates those two,
        # and shape does not either: the void is as rectangular as a window
        # once the raster is closed.
        #
        # What a real aperture has and a shadow cannot is a reveal -- the
        # returns off the jambs, head and sill where the wall turns through its
        # own thickness. Requiring it below the area limit costs us small
        # windows whose reveal was never captured, which is the right trade:
        # an opening the daylight study will shine sun through had better be
        # one we can show a wall around. Above the limit the reveal is not
        # required, because a large void with nothing behind it is a doorway or
        # a missing wall either way, and neither is safe to ignore.
        if void_area < SHADOW_SUSPECT_AREA_M2 and depth["reveal"] < MIN_REVEAL_FRACTION:
            continue

        edges = _refine_edges(
            wu, wv, aperture, res, floor_z=floor_z, ceiling_z=ceiling_z, v_top=v_hi,
            reveal=(rs, pu, pv), reveal_band=reveal_band, sigma=sigma, pad=edge_pad,
        )
        width = edges[2] - edges[0]
        height = edges[3] - edges[1]
        if width <= 0 or height <= 0 or width * height < MIN_OPENING_AREA_M2:
            continue
        if max(width, height) > MAX_OPENING_ASPECT * min(width, height):
            continue

        enclosure = _enclosure(comp, material_raw)
        separation = max(depth["reveal"], depth["behind"])
        confidence = (
            0.25 * enclosure
            + 0.35 * min(1.0, separation / 0.5)
            + 0.20 * rect_fill
            + 0.20 * straight
        ) * (1.0 - 0.5 * depth["front"])

        uc = 0.5 * (edges[0] + edges[2])
        vc = 0.5 * (edges[1] + edges[3])
        center = plane.offset * plane.normal + uc * u_ax + vc * v_ax
        sill = edges[1] - floor_z
        out.append(
            Opening(
                center=center,
                width=float(width),
                height=float(height),
                normal=plane.normal,
                sill_height=float(sill),
                kind=_classify(sill, height),
                confidence=float(np.clip(confidence, 0.0, 1.0)),
            )
        )
    return out


def _straight_rect(comp: np.ndarray) -> tuple[int, int, int, int, float]:
    """The rectangle a void would have if its four edges were straight.

    Returns the rectangle as inclusive index bounds in the padded frame, plus
    how straight-edged the void actually is, from 0 to 1.

    An aperture is bounded by jambs, a head and a sill, all of them straight
    and all of them running the full span of the opening. The bounding box of a
    void component is therefore the right answer for an aperture and the wrong
    answer for anything else, because a bounding box is decided entirely by
    four extreme cells. That is the failure this function exists to remove: the
    fixtures punch round occlusion shadows wherever the scanner's own body
    stood, and one of those touching the edge of a window fuses with it, adding
    a lobe that no jamb bounds. The bounding box then reports the window plus
    the lobe -- up to 500 mm too wide on the rooms measured here -- and every
    edge measurement downstream inherits the error, because the search windows
    that look for the jamb are aimed at where the lobe reached instead.

    So each edge is taken as the *median* over the rows (or columns) the void
    occupies, rather than the extreme. A straight edge puts every row at the
    same index, so the median is that index and nothing changes. A lobe moves a
    minority of the rows, so the median ignores it and the rectangle stays on
    the jamb. It degrades gracefully rather than failing: a lobe covering more
    than half the rows would move the answer, but by then there is more shadow
    in the void than window and no estimator is going to save it.

    The straightness that comes back is the mean over the four edges of the
    fraction of rows sitting within one cell of that edge's median. It is
    reported rather than enforced. A gate on it was measured and rejected: real
    openings with a shadow fused to them score as low as 0.60 on their worst
    edge, and a round shadow rasterised at the 60-80 mm cells a real wall
    forces is only a handful of cells across and scores as high as 1.00 on its
    best, so the two populations overlap. It is honest as a confidence term and
    dishonest as a filter.
    """
    n_u, n_v = comp.shape
    rows_v = np.flatnonzero(comp.any(axis=0))  # the v indices the void occupies
    rows_u = np.flatnonzero(comp.any(axis=1))  # the u indices the void occupies
    if len(rows_v) == 0 or len(rows_u) == 0:
        return 0, 0, 0, 0, 0.0

    first_i = np.argmax(comp, axis=0)[rows_v]
    last_i = (n_u - 1 - np.argmax(comp[::-1, :], axis=0))[rows_v]
    first_j = np.argmax(comp, axis=1)[rows_u]
    last_j = (n_v - 1 - np.argmax(comp[:, ::-1], axis=1))[rows_u]

    i0, i1 = int(np.median(first_i)), int(np.median(last_i))
    j0, j1 = int(np.median(first_j)), int(np.median(last_j))

    # Re-median inside the rectangle we just found, twice. A lobe as tall as
    # the aperture is as wide contributes as many rows as the aperture does,
    # and the first median then lands between the two rather than on either;
    # restricting to the rows the rectangle already claims lets the aperture
    # decide its own edges. Two rounds is enough to settle every case measured
    # and cannot wander, because each round can only look at rows the previous
    # round kept.
    for _ in range(2):
        inside_v = (rows_v >= j0) & (rows_v <= j1)
        inside_u = (rows_u >= i0) & (rows_u <= i1)
        if np.count_nonzero(inside_v) >= 3:
            i0 = int(np.median(first_i[inside_v]))
            i1 = int(np.median(last_i[inside_v]))
        if np.count_nonzero(inside_u) >= 3:
            j0 = int(np.median(first_j[inside_u]))
            j1 = int(np.median(last_j[inside_u]))

    inside_v = (rows_v >= j0) & (rows_v <= j1)
    inside_u = (rows_u >= i0) & (rows_u <= i1)
    pairs = [
        (first_i[inside_v], i0), (last_i[inside_v], i1),
        (first_j[inside_u], j0), (last_j[inside_u], j1),
    ]
    straight = float(
        np.mean([
            np.count_nonzero(np.abs(e - m) <= 1) / len(e) if len(e) else 0.0
            for e, m in pairs
        ])
    )

    # A void whose medians cross -- possible for a ring or an L -- has no
    # straight rectangle to speak of, so fall back to the bounding box and let
    # the low straightness score say so.
    if i1 < i0 or j1 < j0:
        return (
            int(first_i.min()), int(last_i.max()),
            int(first_j.min()), int(last_j.max()), straight,
        )
    return i0, i1, j0, j1, straight


def _depth_evidence(
    s: np.ndarray,
    pu: np.ndarray,
    pv: np.ndarray,
    aperture: tuple[float, float, float, float],
    res: float,
    nu: int,
    nv: int,
    *,
    floor_z: float,
    ceiling_z: float | None,
    reveal_band: tuple[float, float] | None,
    pad: int = 1,
    jamb_limits: tuple[float, float] | None = None,
) -> dict[str, float]:
    """What the scanner saw through the aperture, as three cell coverages.

    Coverages rather than point counts, because a point count is a statement
    about sampling density and a coverage is a statement about geometry: a
    wardrobe sampled twice as densely does not occlude twice as much.

    The reveal is measured *along* the aperture's boundary rather than in it:
    the boundary is walked position by position -- each row down the left jamb,
    each column along the head -- and a position counts as revealed if any
    reveal-band point sits in the `pad + 1` cells that straddle the aperture
    edge there. Two things force that thickness. The reveal lies exactly on the
    edge, so which side of a cell boundary it lands on is a lottery; and the
    morphological closing that removes the sampling bridges pulls the void in
    by up to its own radius, which puts the true edge outside a one-cell ring
    entirely. That last effect is not hypothetical: it is what dropped a real
    1.4 m window on one unseen room to a reveal of 0.14 against a 0.15 line,
    while the window beside it, whose edge happened to land a cell further out,
    scored 0.53. Reducing each boundary position to "revealed or not" also
    stops a densely sampled patch of jamb from covering for a missing one.

    Boundary positions that coincide with the floor or the ceiling are struck
    out first. The strip of floor running along the base of a wall sits a few
    centimetres in front of it and inside that band, so without this every
    scanner shadow that happens to reach the skirting board would come with a
    full reveal along its bottom edge and be promoted to a doorway.

    `reveal_band` is passed in rather than read from the module because its near
    edge depends on how noisy this particular wall is; see
    `REVEAL_NOISE_SIGMAS` for why a fixed near edge turns every hole in a noisy
    wall into a window. `None` means this wall cannot be located precisely
    enough for the reveal to mean anything, and the reveal comes back as zero
    rather than as a guess.
    """
    umin, vmin, umax, vmax = aperture
    inside = (pu >= umin) & (pu <= umax) & (pv >= vmin) & (pv <= vmax)
    total = float(nu * nv)

    def coverage(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        c = _raster(pu[mask], pv[mask], umin, vmin, res, nu, nv)
        return float(np.count_nonzero(c) / total)

    front = coverage(inside & (s > OCCLUDER_BAND_M[0]) & (s < OCCLUDER_BAND_M[1]))
    behind = coverage(inside & (s < BEHIND_M))

    if reveal_band is None:
        return {"front": front, "behind": behind, "reveal": 0.0}

    pad = max(1, int(pad))
    reveal = _reveal_coverage(
        s, pu, pv, aperture, res, nu, nv,
        floor_z=floor_z, ceiling_z=ceiling_z, reveal_band=reveal_band, pad=pad,
        jamb_limits=jamb_limits,
    )
    return {"front": front, "behind": behind, "reveal": reveal}


def _reveal_coverage(
    s: np.ndarray,
    pu: np.ndarray,
    pv: np.ndarray,
    aperture: tuple[float, float, float, float],
    res: float,
    nu: int,
    nv: int,
    *,
    floor_z: float,
    ceiling_z: float | None,
    reveal_band: tuple[float, float],
    pad: int,
    jamb_limits: tuple[float, float] | None = None,
) -> float:
    """The fraction of the aperture's boundary that has a reveal behind it.

    Split out of `_depth_evidence` because it is the one number that decides
    whether a hole is an opening, and it deserves to be readable on its own.

    `jamb_limits` is the stretch of the wall over which a jamb can be believed,
    normally the wall's own extent less a corner margin. Where two walls meet,
    the neighbour is a plane standing at right angles to this one, and its
    points sit in this wall's reveal depth band along the whole height of the
    corner: geometrically it is indistinguishable from a jamb, so a search band
    that reaches the corner reports one. Rather than push every opening away
    from the corner -- which costs real windows, and on the rooms measured here
    cost three of them -- the side that reaches the corner is simply struck off
    the ballot. An opening beside a corner still has its other jamb, its head
    and its sill to make the case with, and a shadow beside a corner has
    nothing left.
    """
    umin, vmin, umax, vmax = aperture
    u_org, v_org = umin - pad * res, vmin - pad * res
    reach = pad * res
    keep = (
        (pu >= u_org) & (pu <= umax + reach) & (pv >= v_org) & (pv <= vmax + reach)
        & (s > reveal_band[0]) & (s < reveal_band[1])
    )
    n_u, n_v = nu + 2 * pad, nv + 2 * pad

    # which boundary positions are allowed to vote; a jamb row level with the
    # floor is looking at the floor, not at a reveal
    row_v = vmin + res * (np.arange(nv) + 0.5)
    row_ok = np.abs(row_v - floor_z) >= res
    if ceiling_z is not None:
        row_ok &= np.abs(row_v - ceiling_z) >= res
    # The sill and head bands look `pad` cells further out than the jamb bands
    # do, so they have to stand that much clear of the floor and the ceiling.
    # Both of those meet the wall in a strip of points a few centimetres proud
    # of the plaster, which is indistinguishable from a reveal and belongs to
    # every hole in the wall equally; a shadow ten centimetres below the
    # ceiling would otherwise arrive with a full head reveal.
    clear = (pad + 1) * res
    sill_ok = vmin - floor_z >= clear
    head_ok = ceiling_z is None or ceiling_z - vmax >= clear
    left_ok = right_ok = True
    if jamb_limits is not None:
        left_ok = umin - reach >= jamb_limits[0]
        right_ok = umax + reach <= jamb_limits[1]

    n_rows = int(np.count_nonzero(row_ok))
    eligible = n_rows * (int(left_ok) + int(right_ok)) + nu * (int(sill_ok) + int(head_ok))
    if eligible == 0 or not np.any(keep):
        return 0.0

    hit = _raster(pu[keep], pv[keep], u_org, v_org, res, n_u, n_v) > 0
    inner_u = slice(pad, pad + nu)
    inner_v = slice(pad, pad + nv)
    covered = 0
    if left_ok:
        covered += int(np.count_nonzero(hit[: pad + 1, inner_v].any(axis=0) & row_ok))
    if right_ok:
        covered += int(np.count_nonzero(hit[pad + nu - 1 :, inner_v].any(axis=0) & row_ok))
    if sill_ok:
        covered += int(np.count_nonzero(hit[inner_u, : pad + 1].any(axis=1)))
    if head_ok:
        covered += int(np.count_nonzero(hit[inner_u, pad + nv - 1 :].any(axis=1)))
    return float(covered / eligible)


def _refine_edges(
    wu: np.ndarray,
    wv: np.ndarray,
    aperture: tuple[float, float, float, float],
    res: float,
    *,
    floor_z: float,
    ceiling_z: float | None,
    v_top: float,
    reveal: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    reveal_band: tuple[float, float] | None = REVEAL_BAND_M,
    sigma: float = 0.0,
    pad: int = 2,
) -> tuple[float, float, float, float]:
    """Move each aperture edge from the raster grid onto the geometry.

    The raster locates an opening but cannot measure it: its bounding box runs
    to the cell boundaries, so it is up to two cells too wide and too tall,
    which at a 40 mm cell is already half the 80 mm accuracy budget and at the
    coarser cells a sparse wall forces is all of it.

    Two estimators are used, in order of how directly they see the edge.

    The reveal is the better one wherever it exists, and it is used first. The
    reveal is the surface running out of the wall at the aperture boundary --
    the plaster returning into the jamb -- so its intersection with the wall
    plane *is* the edge, and the median position of its points is that edge
    with no bias at all. What makes this worth the trouble is the alternative's
    bias, which is not small: the wall points beside an edge are contaminated by
    that same reveal, which piles a tenth of the window's points into a line
    exactly on the boundary, and a percentile then steps over that pile instead
    of over the thin noise tail it was designed for. Measured against truth on
    38 unseen rooms, that put every jambed width short by 10 mm on a quiet scan
    and 45 mm on a noisy one, growing as 1.3 sigma per edge -- a bias that
    reads as a tuning constant and is really the reveal being counted twice.
    The reveal is only trusted when its points are tightly clustered across the
    edge, which is what says the reveal runs square out of the wall rather than
    splaying; a splayed reveal falls back rather than reporting its midpoint.

    Otherwise the wall points beside the edge are used. The search window
    straddles the raster edge, reaching two cells *inside* the aperture, so that
    it contains the whole material-to-void transition wherever the cell boundary
    happened to fall relative to it. A low percentile rather than the extreme
    value keeps the thin tail of range noise, which always scatters some wall
    points into the aperture, from dragging the edge past the material.

    An edge with neither is not measurable and is snapped to the architecture
    instead: the bottom of a door is the floor.
    """
    umin, vmin, umax, vmax = aperture
    reach = 6.0 * res
    straddle = 2.0 * res
    inset_v = 0.15 * (vmax - vmin)
    inset_u = 0.15 * (umax - umin)

    band_v = (wv >= vmin + inset_v) & (wv <= vmax - inset_v)
    band_u = (wu >= umin + inset_u) & (wu <= umax - inset_u)

    left = wu[band_v & (wu < umin + straddle) & (wu >= umin - reach)]
    right = wu[band_v & (wu > umax - straddle) & (wu <= umax + reach)]
    below = wv[band_u & (wv < vmin + straddle) & (wv >= vmin - reach)]
    above = wv[band_u & (wv > vmax - straddle) & (wv <= vmax + reach)]

    jamb = _reveal_edges(aperture, res, reveal, reveal_band, sigma, pad)

    n_min = 20
    q = EDGE_PERCENTILE
    if jamb[0] is not None:
        u_lo = jamb[0]
    elif len(left) >= n_min:
        u_lo = float(np.percentile(left, 100.0 - q))
    else:
        u_lo = umin
    if jamb[2] is not None:
        u_hi = jamb[2]
    elif len(right) >= n_min:
        u_hi = float(np.percentile(right, q))
    else:
        u_hi = umax
    if jamb[1] is not None:
        v_lo = jamb[1]
    elif len(below) >= n_min:
        v_lo = float(np.percentile(below, 100.0 - q))
    else:
        v_lo = floor_z if vmin - floor_z < 3.0 * res else vmin
    if jamb[3] is not None:
        v_hi = jamb[3]
    elif len(above) >= n_min:
        v_hi = float(np.percentile(above, q))
    elif ceiling_z is not None and ceiling_z - vmax < 3.0 * res:
        v_hi = ceiling_z
    else:
        v_hi = min(vmax, v_top)
    return u_lo, v_lo, u_hi, v_hi


def _reveal_edges(
    aperture: tuple[float, float, float, float],
    res: float,
    reveal: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    reveal_band: tuple[float, float] | None,
    sigma: float,
    pad: int,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Each aperture edge read straight off the reveal, where there is one.

    Returns four edges in aperture order (u_lo, v_lo, u_hi, v_hi), with `None`
    wherever the reveal did not give a usable answer -- either too few points
    to average or a spread across the edge too wide to be a square reveal.

    The search window around each edge reaches as far as the raster's own
    uncertainty about where that edge is, `pad` cells, and is then capped at
    40% of the aperture's own span so that a narrow window's two jambs cannot
    be averaged together into a single edge halfway between them.
    """
    if reveal is None or reveal_band is None:
        return (None, None, None, None)
    s, pu, pv = reveal
    umin, vmin, umax, vmax = aperture
    in_band = (s > reveal_band[0]) & (s < reveal_band[1])
    if not np.any(in_band):
        return (None, None, None, None)
    bu, bv = pu[in_band], pv[in_band]

    n_min = 30
    # a reveal running square out of the wall lands within range noise of one
    # line; anything wider is a splay, a chamfer or something that is not a
    # reveal at all, and its middle is not the edge
    max_spread = max(2.0 * sigma, 0.015)
    span_u = min(pad * res, 0.40 * (umax - umin))
    span_v = min(pad * res, 0.40 * (vmax - vmin))

    def edge(vals: np.ndarray, other: np.ndarray, lo: float, hi: float,
             centre: float, span: float) -> float | None:
        sel = (np.abs(vals - centre) <= span) & (other >= lo) & (other <= hi)
        if int(np.count_nonzero(sel)) < n_min:
            return None
        picked = vals[sel]
        m = float(np.median(picked))
        spread = float(np.median(np.abs(picked - m)) * 1.4826)
        return m if spread <= max_spread else None

    return (
        edge(bu, bv, vmin, vmax, umin, span_u),
        edge(bv, bu, umin, umax, vmin, span_v),
        edge(bu, bv, vmin, vmax, umax, span_u),
        edge(bv, bu, umin, umax, vmax, span_v),
    )


def _enclosure(comp: np.ndarray, material_raw: np.ndarray) -> float:
    """How much of the aperture's border is wall we actually measured.

    A void is only accepted when it is fully enclosed, so enclosure is never a
    gate -- it is the honesty term in the confidence. A border made of real
    returns scores 1; one that leans on the virtual floor line at the bottom of
    a doorway, or on cells that morphological closing invented, scores less.
    """
    border = ndimage.binary_dilation(
        comp, structure=ndimage.generate_binary_structure(2, 2)
    ) & ~comp
    total = int(np.count_nonzero(border))
    if total == 0:
        return 0.0
    # `comp` is indexed in the padded frame, so the raw material has to be put
    # back into that frame before the two can be compared cell for cell
    raw = np.zeros_like(comp)
    raw[1:-1, 1:-1] = material_raw
    return float(np.count_nonzero(border & raw) / total)


def _classify(sill_height: float, height: float) -> str:
    """Door, window or neither, from the two facts that separate them.

    A door reaches the floor and is tall enough to walk through; a window sits
    on a sill well clear of the floor. Anything else -- a serving hatch, a
    knocked-through arch, a hole where the scan simply failed -- is left as the
    honest `opening`, because guessing between the two here would put a door
    into a wall that has none.
    """
    if sill_height <= 0.15 and height >= 1.6:
        return "door"
    if sill_height > 0.4:
        return "window"
    return "opening"


def _dedupe(openings: list[Opening], radius: float = 0.30) -> list[Opening]:
    """Drop duplicates from coincident wall planes, keeping the surer one.

    Plane detection can return the same wall twice -- once per RANSAC pass over
    a wall that noise split in two -- and reporting a window twice would double
    the glazing area in the daylight study.
    """
    kept: list[Opening] = []
    for o in sorted(openings, key=lambda x: -x.confidence):
        if any(np.linalg.norm(o.center - k.center) < radius for k in kept):
            continue
        kept.append(o)
    # a stable, frame-independent order so two runs agree byte for byte
    kept.sort(key=lambda x: (round(float(x.center[0]), 4), round(float(x.center[1]), 4),
                             round(float(x.center[2]), 4)))
    return kept


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


#: Two wall planes are the same wall if their normals agree to within this
#: angle and their offsets to within this distance. Both are loose on purpose:
#: the duplicates that matter are the ones close enough to be confused for the
#: same surface, and a genuine parallel wall in a room is metres away, not
#: centimetres.
DUPLICATE_WALL_ANGLE_DEG = 6.0
DUPLICATE_WALL_OFFSET_M = 0.20


def merge_duplicate_walls(walls: list[Plane]) -> list[Plane]:
    """Collapse the several planes RANSAC returns for one physical wall.

    Iterative RANSAC does not partition a room into surfaces; it removes the
    inliers it found and fits again, so a long wall that its first fit did not
    quite capture comes back two or three more times as near-copies, each
    tilted a fraction of a degree differently. A 12.6 x 8.3 m room produced
    fifteen planes labelled wall.

    That is not merely wasteful, it invents openings. Hole detection rasters
    the points lying within a fixed band of the plane, and a duplicate tilted
    even a couple of degrees has that band walk off the real wall as it runs
    along it -- so the far end of the copy has no points near it, and the empty
    region reads as a full-height aperture. One such copy produced a 4.63 m
    wide, 2.06 m tall "door" in a room with no openings at all, at confidence
    0.85, and every other check on that twin passed.

    Keeping the best-supported representative of each group is the fix: the
    real wall is the one the first RANSAC pass found, and it is the one with
    the inliers.
    """
    if not walls:
        return []

    cos_tol = math.cos(math.radians(DUPLICATE_WALL_ANGLE_DEG))
    ordered = sorted(walls, key=lambda p: -p.inlier_count)
    kept: list[Plane] = []
    for plane in ordered:
        duplicate = False
        for other in kept:
            # compare the plane to the one already kept, allowing for the two
            # having been fitted with opposite normal signs
            alignment = float(np.dot(plane.normal, other.normal))
            if abs(alignment) < cos_tol:
                continue
            offset = plane.offset if alignment > 0 else -plane.offset
            if abs(offset - other.offset) <= DUPLICATE_WALL_OFFSET_M:
                duplicate = True
                break
        if not duplicate:
            kept.append(plane)
    return kept


def analyze(
    cloud: PointCloud,
    *,
    planes: list[Plane] | None = None,
    normals: np.ndarray | None = None,
    grid: Any | None = None,
    seed: int = 0,
) -> Structure:
    """Read a canonical cloud as a room.

    `planes`, `normals` and `grid` are all optional inputs that the ingest
    pipeline has usually computed already; passing them in avoids paying for
    them twice, and leaving them out makes this callable on a bare cloud, which
    is what the tests and the CLI's one-off inspection mode want.

    Nothing here re-detects planes when it is given them, and nothing
    re-labels them either: `Structure.planes` is the labelled list, so the
    twin carries one set of planes with one set of kinds rather than two that
    can disagree.
    """
    if planes is None:
        planes = detect_planes(cloud, normals=normals, seed=seed)

    labelled = label_planes(list(planes), cloud)
    floor_z, ceiling_z = find_floor_ceiling(cloud, labelled)
    footprint = floor_footprint(cloud, floor_z, ceiling_z=ceiling_z, grid=grid)
    openings = detect_openings(
        cloud,
        merge_duplicate_walls([p for p in labelled if p.kind == "wall"]),
        floor_z=floor_z,
        ceiling_z=ceiling_z,
        seed=seed,
    )
    return Structure(
        floor_z=float(floor_z),
        ceiling_z=None if ceiling_z is None else float(ceiling_z),
        planes=labelled,
        openings=openings,
        footprint=footprint if len(footprint) >= 3 else None,
    )


__all__ = [
    "analyze",
    "detect_openings",
    "find_floor_ceiling",
    "floor_footprint",
    "label_planes",
]
