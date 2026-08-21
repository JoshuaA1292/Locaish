"""Putting an arbitrary export into the one frame everything else assumes.

A scanner hands us a room in a pose that means nothing: the phone was never
level, the SLAM origin is wherever the app happened to start, the yaw is
whichever way the user was facing, and the numbers might be inches. Canonical
twin space is the opposite of all that -- metres, Z up, floor at z = 0, the
dominant wall parallel to +X, origin in the middle of the floor. Everything
downstream, from clearance queries to sun angles to comparing two twins of the
same room, is written against that frame, so this module is the point where a
model becomes a measurement.

The order is forced. Scale comes first, because the priors that identify a unit
are metric ones about real rooms and they have to see the raw numbers. Gravity
comes second, because "horizontal" and "vertical" are meaningless until we know
which way is down. Yaw comes third, because it is a rotation *about* the up
axis. The origin comes last, because it is defined by the floor, which we only
know once gravity is settled.

Four decisions here are less obvious than they look.

Gravity is decided in two separate arguments, an axis and a sign, because the
two fail in different ways and a single score that conflates them hides which
one went wrong. The axis is one of the three directions of the room's own
Manhattan frame, recovered from the surface normals; the sign is settled by the
fact that furniture stands on the floor and nothing hangs off the ceiling.

Neither argument is allowed to rest on how big a surface is. Area was the old
signal and it is not evidence about gravity at all: in a room taller than it is
narrow a wall carries more square metres than the floor, so an area-led vote
canonicalises a stairwell, a corridor or an atrium lying on its side, and every
height and every width downstream comes back transposed. What the shape of a
room does tell us is only the *triad* -- an empty rectangular box is symmetric
under permuting its three axes and no purely geometric statistic can say which
member of the triad is up. The asymmetries that can are the ones an interior
actually has: clutter rests on the floor and hangs off it into the room, the
reveals of doors reach the floor, a floor-to-ceiling distance is 2 to 6 m
whatever the plan dimensions are, and a person carrying a scanner walks metres
horizontally while holding it at a nearly constant height.

Which horizontal plane is the floor is settled by weight and not by height. The
lowest horizontal surface in a dense scan is very often a small tilted scrap
that the detector split off the real slab, and letting a scrap a centimetre low
and nine degrees out define gravity tilts the entire twin. The two heaviest
horizontal surfaces are the floor and the ceiling; between those two, height
decides which is which.

Yaw is a Manhattan-frame estimate taken as a circular mean of four times the
wall azimuth, not as the argmax of a histogram. A histogram quantises the answer
to its bin width and makes the result jump when a wall crosses a bin boundary;
the circular mean is continuous in the input, which matters because two scans of
the same room must canonicalise to the same pose if their twins are to be
comparable at all. For the same reason the final quarter-turn is chosen by the
room's own shape -- longest horizontal extent along +X -- rather than by
whichever wall the plane detector happened to find first.

Every residual reported here is measured after the fact and by a route that does
not reuse the thing it is checking. The gravity residual in particular is a fresh
fit of the floor to the transformed points, not the tilt of the plane the twin
was levelled against, because that second quantity is zero by construction and a
check that cannot fail is not a check. The same reasoning is why the origin's Z
is read after the XY move rather than before it: a number that happens to be
small today is still the wrong number, and it grows with the size of the space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..scan.scale import ScaleEstimate, infer_unit_scale
from ..types import (
    Mesh,
    Plane,
    PointCloud,
    homogeneous,
    rotation_between,
    rotation_z,
)
from .normals import estimate_normals, orient_normals
from .planes import detect_planes, plane_inliers

_UP = np.array([0.0, 0.0, 1.0])

# A plane counts as horizontal if its normal is within this of the up candidate,
# and as a wall if its normal is within this of perpendicular. The gap between
# the two is deliberately left as "oblique" -- a ramp or a pitched roof should
# vote for neither.
_HORIZONTAL_TOL_DEG = 12.0
_WALL_TOL_DEG = 15.0

# Cameras ride on a person; a handheld scanner sits about this far above the
# floor. Used only to break the floor/ceiling tie, and only weakly.
_CAMERA_HEIGHT_M = 1.45

# Clutter lives in a metric band above the floor, not in a fixed fraction of the
# room's height: a sofa is 0.9 m tall in a flat and 0.9 m tall in an atrium, so a
# relative band would dilute the signal to nothing in exactly the tall rooms that
# need it. The skin keeps the bounding surface itself out of the band, since the
# floor slab is not evidence about what is standing on it.
_CLUTTER_BAND_M = 1.20
_CLUTTER_SKIN_M = 0.18
# How far a point has to sit off every large plane before it counts as clutter
# rather than as part of the room's shell.
_SHELL_SKIN_M = 0.08
# Raster and tolerance for the "is the clutter standing on this surface" test. A
# 25 cm column is wide enough to hold a few points off a chair leg and narrow
# enough that a table and the floor under it stay separate columns. The tolerance
# is measured from the inner face of the shell slab, not from the surface itself,
# since the slab has already taken the bottom few centimetres of everything.
_CONTACT_CELL_M = 0.25
_CONTACT_TOL_M = 0.10

# Floor-to-ceiling distances live in this band whatever the plan dimensions are,
# and a twin whose vertical span falls outside it says so.
_VERTICAL_MIN_M = 2.05
_VERTICAL_MAX_M = 6.50
# The same idea used as a prior over the three candidate axes, deliberately
# slacker at the bottom than the reporting band above. Its job is to refuse a
# 20 m axis, not to referee between a 2.2 m ceiling and a 2.5 m one -- a knee
# tight enough to do the latter would demote the genuine vertical of a low room
# and hand the decision to a wall, which is the failure being fixed. It only ever
# attenuates a candidate, never vetoes one, hence the floor below.
_VERTICAL_PRIOR_MIN_M = 1.80
_VERTICAL_PRIOR_MAX_M = 6.50
_VERTICAL_PRIOR_FLOOR = 0.25

# How much each axis signal is worth relative to the others. The ordering is not
# a guess: over 43 unseen rooms, each statistic used on its own picked the wrong
# axis on 1 (clutter band over the whole cloud), 0 (the same band over the
# clutter alone), and 11 (contact) of them, so the weights follow that and the
# fragile one is left able to break a tie and nothing more.
_W_CLUTTER_BAND = 1.0
_W_CLUTTER_FREE = 0.5
_W_CLUTTER_CONTACT = 0.3
_W_CAMERA_AXIS = 1.0

# Relative margins below which the winning axis or sign is reported as contested
# rather than as a decision. A room where gravity is genuinely ambiguous -- an
# empty cubical box, a capture with no furniture and no openings -- must say so.
_AXIS_MARGIN_WARN = 0.15
_SIGN_MARGIN_WARN = 0.10

_FLOOR_INLIER_THRESH_M = 0.03
_SAMPLE_POINTS = 200_000

# A plane has to carry this fraction of the heaviest wall's weight before it is
# allowed to vote on yaw, and this much before it counts towards the reported
# yaw residual.
_YAW_MIN_REL_WEIGHT = 0.05
_RESIDUAL_MIN_REL_WEIGHT = 0.30

# How far off-centre the room's contents have to sit, as a fraction of its
# longest side, before that asymmetry is allowed to settle the half-turn.
_SKEW_DEADBAND = 0.002

# When only one horizontal surface exists, it is the floor if it sits within
# this fraction of the cloud's height from the bottom, and a ceiling otherwise.
_FLOOR_BAND = 0.4

# A candidate ceiling has to be at least this high, and has to cover the top of
# the point distribution rather than sit halfway up it.
_CEILING_MIN_HEIGHT_M = 1.6
_CEILING_MIN_COVERAGE = 0.8

# How far the floor refit is allowed to move an already-decided axis. The refit
# exists to buy precision, and on a healthy scan it moves the axis by five to ten
# hundredths of a degree; a surface that wants to move it by a degree is not the
# floor but a ramp, a rug, or a scrap the plane detector split off, and following
# it would trade a good estimate for a bad one. So the ceiling on the correction
# is set an order of magnitude above what a real refinement costs and no more.
_REFINE_MAX_DEG = 0.5

# When the floor and the walls disagree by more than `_REFINE_MAX_DEG`, the
# floor is still accepted if the walls are too imprecise to have an opinion at
# that scale -- specifically if the disagreement is inside this multiple of the
# wall family's own scatter about the chosen axis.
#
# The scatter is used as it stands, not divided by the root of the wall count.
# That was the first version and it was wrong: dividing assumes the walls are
# independent samples of one truth, so sixteen of them would pin the vertical
# four times better than one. What actually bends the walls of a video
# reconstruction is drift, which bends them all the same way at once. Correlated
# error does not average down, and treating it as though it does made the walls
# look four times more certain than they were and kept the floor overruled.
#
# The point is that the fixed half-degree limit assumes the wall normals are a
# sharp instrument, which they are on a laser scan of a flat-plastered room and
# are not on a video reconstruction, where the walls bow by centimetres and
# three of them may be all there is. Refusing to level in that case leaves the
# twin a degree and a half out of true and blames the floor for it. Measuring
# the walls' scatter turns "which do I trust" into something the data answers:
# on a clean capture the scatter is a hundredth of a degree and this changes
# nothing, and on a drifty one it correctly steps aside.
_REFINE_SIGMA_K = 1.0

# However badly conditioned the walls are, a floor further off than this is not
# a levelling disagreement -- it is a fitting failure somewhere -- and gets
# refused and reported rather than obeyed.
_REFINE_MAX_TOLERANCE_DEG = 3.0

# Grid cell and trim used by the independent post-transform floor refit. The cell
# is coarse enough that most cells see the real floor rather than the underside
# of the furniture standing on it.
_GROUND_CELL_M = 0.30
_GROUND_QUANTILE = 0.10
_GROUND_TRIM_M = 0.05

# How far the points fitted as "floor" may scatter about their own best-fit
# plane before we stop believing they are a floor at all.
#
# The column-bottom method assumes each cell's lowest points sit on the ground.
# On a complete scan they do. On a video sweep that saw 40% of the room, most
# cells bottom out on a chair, a wall base, or the frontier of the completion,
# and the "floor" fitted through them scatters half a metre -- measured at
# 0.492 m on a real capture, against 0.008 m for a full scan of the same kind of
# room. A plane fitted through that is not a measurement of anything, and the
# angle it reports is clutter, not tilt. Six centimetres sits an order of
# magnitude clear of both.
_GROUND_MAX_SCATTER_M = 0.06


@dataclass
class Canonicalization:
    """The transform into canonical twin space, plus how much to believe it.

    `transform` is applied to the *source* geometry. `yaw_residual_deg` is
    measured after the fact on the transformed planes, and `up_residual_deg` is
    measured after the fact on the transformed *points*, by an estimator that
    shares nothing with the one that produced the transform -- so both are a
    check on the result rather than a restatement of the input, and both are
    capable of coming back large.
    `floor_plane` and `ceiling_plane` are returned already in canonical
    coordinates -- the floor's normal is +Z and its offset is 0 to within the
    residual -- because everything that consumes them works in that frame.
    """

    transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    up_residual_deg: float = 0.0
    yaw_residual_deg: float = 0.0
    floor_plane: Plane | None = None
    ceiling_plane: Plane | None = None
    scale: ScaleEstimate = field(default_factory=ScaleEstimate)
    method: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # Carried out of GravitySolution so that QA can gate on them. A contested
    # axis is the one failure that produces a twin which is wrong in a
    # completely self-consistent way -- every surface is planar, every wall is
    # square, the room is simply standing on its side with its height and its
    # depth exchanged -- so no downstream measurement can detect it and the
    # margin at the moment of the decision is the only evidence there will ever
    # be. It stops here unless something reads it.
    gravity_axis_margin: float = 1.0
    gravity_sign_margin: float = 1.0

    def __post_init__(self) -> None:
        self.transform = np.asarray(self.transform, dtype=np.float64).reshape(4, 4)

    @property
    def ceiling_height(self) -> float | None:
        if self.ceiling_plane is None:
            return None
        return abs(float(self.ceiling_plane.offset))

    def to_dict(self) -> dict[str, object]:
        return {
            "transform": self.transform.tolist(),
            "up_residual_deg": float(self.up_residual_deg),
            "yaw_residual_deg": float(self.yaw_residual_deg),
            "scale": self.scale.to_dict(),
            "method": dict(self.method),
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------


def _plane_weight(plane: Plane) -> float:
    """How much a plane's opinion is worth.

    Area if the detector measured it, inlier count otherwise. Reading a missing
    area as 0.0 would make every plane weightless and let a cabinet door outvote
    a wall, which is the exact failure the weighting exists to prevent.
    """
    if plane.area and plane.area > 0:
        return float(plane.area)
    return float(plane.inlier_count)


def _sample_indices(n: int, seed: int = 0) -> np.ndarray:
    """Deterministic index subsample.

    Every use of the raw points here is a statistic -- a quantile, a centroid, a
    mass ratio -- so a couple of hundred thousand points answers the question as
    well as twenty million and keeps the whole canonicalisation interactive.
    """
    if n <= _SAMPLE_POINTS:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=_SAMPLE_POINTS, replace=False))


def _sample_cloud(cloud: PointCloud, seed: int = 0) -> PointCloud:
    idx = _sample_indices(len(cloud.xyz), seed)
    if len(idx) == len(cloud.xyz):
        return cloud
    return cloud.subset(idx)


def _spread(values: np.ndarray) -> float:
    """Robust spread of a 1D projection, as a median absolute deviation.

    A handful of stray poses at the start of a capture, before tracking settles,
    is enough to make a standard deviation say the phone climbed a metre.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if len(v) == 0:
        return 0.0
    return float(np.median(np.abs(v - np.median(v))))


def _axis_coordinate(plane: Plane, axis: np.ndarray) -> float:
    """Where a near-perpendicular plane crosses the given axis."""
    c = float(np.dot(plane.normal, axis))
    if abs(c) < 1e-6:
        return math.nan
    return plane.offset / c


def _oriented(plane: Plane, direction: np.ndarray, kind: str) -> Plane:
    """Copy of a plane with its normal forced onto `direction`'s side.

    The plane detector is free to return either sign; the twin contract is that
    a normal points into the room, and every clearance query downstream depends
    on that sign being stable.
    """
    flip = float(np.dot(plane.normal, direction)) < 0
    return Plane(
        normal=-plane.normal if flip else plane.normal,
        offset=-plane.offset if flip else plane.offset,
        kind=kind,
        inlier_count=plane.inlier_count,
        area=plane.area,
        extent_2d=plane.extent_2d,
    )


def transform_plane(plane: Plane, matrix: np.ndarray) -> Plane:
    """Carry a plane through a similarity transform (uniform scale + rotation).

    Planes transform by the inverse transpose, which for a uniform scale is the
    rotation divided by the scale; renormalising the normal absorbs the scale
    and leaves the offset carrying it, which is why the offset is multiplied by
    the scale factor and the areas by its square.
    """
    matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    linear = matrix[:3, :3]
    scale = float(np.cbrt(abs(np.linalg.det(linear))))
    rot = linear / scale if scale else linear
    normal = rot @ plane.normal
    normal = normal / np.linalg.norm(normal)
    offset = scale * plane.offset + float(np.dot(normal, matrix[:3, 3]))
    extent = None
    if plane.extent_2d is not None:
        extent = tuple(float(v) * scale for v in plane.extent_2d)
    return Plane(
        normal=normal,
        offset=offset,
        kind=plane.kind,
        inlier_count=plane.inlier_count,
        area=float(plane.area) * scale * scale,
        extent_2d=extent,
    )


# ---------------------------------------------------------------------------
# gravity
# ---------------------------------------------------------------------------


@dataclass
class GravitySolution:
    """Which way is up, and the two separate arguments that decided it.

    `axis_margin` and `sign_margin` are relative margins in [0, 1] over the
    runner-up. They are kept apart because the two decisions fail apart: an axis
    error stands the room on its side and transposes every dimension, while a
    sign error leaves the room the right shape and hangs it from its ceiling. A
    caller that only reads one number cannot tell those two apart, and a twin
    that scored badly on either has no business reporting a confident height.
    """

    up: np.ndarray = field(default_factory=lambda: _UP.copy())
    axis_margin: float = 0.0
    sign_margin: float = 0.0
    pole: np.ndarray | None = None
    method: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.up = np.asarray(self.up, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(self.up))
        if n == 0:
            raise ValueError("up vector must be non-zero")
        self.up = self.up / n


def _taper(x: float, lo: float, hi: float, width: float = 0.10) -> float:
    """Logistic band-pass in log space: 1 inside [lo, hi], decaying outside.

    Log space rather than linear because the question "is this a plausible
    floor-to-ceiling distance" is a question about ratios -- 6.5 m is as far
    above the band as 1.3 m is below it -- and a linear taper would make the
    penalty for a 20 m span indistinguishable from the penalty for a 200 m one.
    """
    if x <= 0:
        return 0.0
    lx = math.log(x)
    below = 1.0 / (1.0 + math.exp(-(lx - math.log(lo)) / width))
    above = 1.0 / (1.0 + math.exp(-(math.log(hi) - lx) / width))
    return below * above


def manhattan_axes(planes: list[Plane], normals: np.ndarray | None = None) -> list[np.ndarray]:
    """The room's three structural axes, as an unordered orthogonal triad.

    Deliberately says nothing about which of the three is up. In a rectangular
    room the plane normals fall into three mutually perpendicular families and
    that is all the shape of the room knows; deciding gravity here, by picking
    the heaviest family, is precisely the bug that stands a tall room on its
    side. So this returns all three and lets `find_up` argue about them.

    The triad is grown from the families rather than read off the eigenvectors
    of the normal scatter matrix, because two of those eigenvalues coincide
    whenever a room has two similar face areas -- a 5 x 3 x 4 m room has
    2*5*3 = 30 and 2*3*4 = 24 -- and eigenvectors inside a near-degenerate
    eigenspace are an arbitrary rotation of the axes we actually want. Growing
    from the heaviest family and then taking the heaviest family perpendicular
    to it has no such degeneracy.
    """
    strong = sorted((p for p in planes if _plane_weight(p) > 0), key=_plane_weight, reverse=True)
    directions: list[np.ndarray] = []
    weights: list[float] = []
    same = math.cos(math.radians(10.0))
    for p in strong[:24]:
        for k, direction in enumerate(directions):
            if abs(float(np.dot(p.normal, direction))) >= same:
                weights[k] += _plane_weight(p)
                break
        else:
            directions.append(p.normal.copy())
            weights.append(_plane_weight(p))
    order = sorted(range(len(directions)), key=lambda k: -weights[k])

    first: np.ndarray | None = None
    second: np.ndarray | None = None
    if order:
        first = directions[order[0]]
        perp = math.sin(math.radians(20.0))
        for k in order[1:]:
            if abs(float(np.dot(directions[k], first))) <= perp:
                second = directions[k]
                break

    if first is None:
        # No planes at all. The point normals still carry the triad, and their
        # scatter is the only thing left to read it from.
        if normals is None or len(normals) < 64:
            return []
        _, vecs = np.linalg.eigh(normals.T @ normals)
        return [vecs[:, k] / np.linalg.norm(vecs[:, k]) for k in range(3)]

    if second is None:
        # A capture that only ever saw one wall family. Any perpendicular pair
        # completes the frame; the scoring below still has to choose between
        # them, and it will be told the choice was underdetermined.
        helper = np.array([1.0, 0.0, 0.0]) if abs(first[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        second = np.cross(first, helper)
    second = second - float(np.dot(second, first)) * first
    length = float(np.linalg.norm(second))
    if length < 1e-9:
        return [first / np.linalg.norm(first)]
    second = second / length
    first = first / float(np.linalg.norm(first))
    return [first, second, np.cross(first, second)]


def normal_pole(
    normals: np.ndarray, axis: np.ndarray, *, tol_deg: float = _WALL_TOL_DEG
) -> tuple[np.ndarray, bool]:
    """Refine one candidate axis into the pole of its own great circle of normals.

    Every surface perpendicular to a candidate axis has a normal lying on the
    great circle whose pole is that axis, so the least-squares pole of those
    normals is the candidate re-estimated from every point on every such surface
    at once. For the true vertical those surfaces are the walls, which is why
    this still answers when the floor is completely buried under furniture, and
    why it is a genuinely independent second opinion on gravity: it never looks
    at the floor plane, or at any plane, only at the normals.

    The pole is the smallest eigenvector of the normals' scatter matrix, which
    is the direction of least projection and therefore the axis of the circle.
    Membership of the band and the pole are refined together for a few rounds
    because the initial axis is only as good as the plane family that seeded it,
    and each round admits the surfaces the previous one clipped.

    What this cannot do, and must not be asked to do, is choose the axis. In a
    rectangular room all three of the Manhattan directions are poles of a great
    circle of normals -- the floor, the ceiling and one pair of walls are exactly
    as coplanar-in-normal-space as the four walls are -- so each candidate
    refines to itself and all three are equally self-consistent. Taking the
    smallest eigenvector of *all* the normals instead does pick one, but it picks
    the axis with the least surface facing it, which is the area argument in
    disguise and fails on the same tall rooms. Hence the split of labour: this
    function makes a candidate precise, and the interior asymmetries in `find_up`
    decide which candidate is gravity.

    Returns (pole, converged). `converged` is False when too few normals sit on
    the circle to fit one, in which case the seed axis is handed back unchanged
    and the caller must not treat it as a measurement.
    """
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    pole = np.asarray(axis, dtype=np.float64).reshape(3)
    pole = pole / np.linalg.norm(pole)
    if len(normals) < 64:
        return pole, False
    limit = math.sin(math.radians(tol_deg))
    ok = False
    for _ in range(4):
        band = np.abs(normals @ pole) <= limit
        if int(np.count_nonzero(band)) < 64:
            return pole, ok
        sub = normals[band]
        _, vecs = np.linalg.eigh(sub.T @ sub)
        refined = vecs[:, 0]
        if float(np.dot(refined, pole)) < 0:
            refined = -refined
        shift = float(np.linalg.norm(refined - pole))
        pole, ok = refined, True
        if shift < 1e-12:
            break
    return pole, ok


def _axis_extremes(xyz: np.ndarray, axes: list[np.ndarray]) -> list[tuple[float, float]]:
    """Robust near-extreme coordinate of the cloud along each candidate axis.

    Quantiles rather than the true minimum and maximum, because a scan taken
    through a doorway catches a few square metres of the corridor beyond and a
    single stray point must not define where the room stops.
    """
    out: list[tuple[float, float]] = []
    for axis in axes:
        t = xyz @ axis
        out.append((float(np.quantile(t, 0.002)), float(np.quantile(t, 0.998))))
    return out


def _clutter_mask(
    xyz: np.ndarray, axes: list[np.ndarray], extremes: list[tuple[float, float]]
) -> np.ndarray:
    """Points that lie on none of the room's six bounding slabs.

    What is left after the shell is subtracted is the furniture, the reveals
    around doors and windows, and whatever else is standing in the room -- the
    only part of an interior that knows where the floor is.

    The shell is taken as a slab at each end of each axis of the triad rather
    than as a list of detected planes, and that is a deliberate retreat from the
    plane detector. A wall commonly comes back as a large fit plus a couple of
    small fragments a few degrees off it, and any rule that nominates one plane
    per side of the room then leaves the fragments -- or worse, the main fit --
    classified as furniture. Six slabs cannot fragment. They also make this
    signal independent of the plane detector altogether, which is worth having
    in the one place where a wrong answer stands the whole twin on its side.

    Deliberately computed from the triad, which all three gravity hypotheses
    share, so the clutter is the same set of points whichever hypothesis is being
    scored; a clutter set that moved with the hypothesis would let each
    hypothesis choose the evidence for itself.
    """
    if not axes:
        return np.zeros(len(xyz), dtype=bool)
    keep = np.ones(len(xyz), dtype=bool)
    for axis, (lo, hi) in zip(axes, extremes):
        t = xyz @ axis
        keep &= (t > lo + _SHELL_SKIN_M) & (t < hi - _SHELL_SKIN_M)
    return keep


def _band_asymmetry(t: np.ndarray, lo: float, hi: float, skin: float) -> float:
    """Signed imbalance between metric bands at the two ends of an axis, [-1, 1].

    Positive means the low end carries more geometry, i.e. the low end is the
    floor. The band is metric and not a fraction of the span, which is the whole
    point: furniture is about a metre tall in a flat and about a metre tall in an
    atrium, so a band scaled to the room's height dilutes the signal to nothing
    in exactly the tall rooms where the old area-based vote already failed.

    Along the true vertical the low band holds the furniture and the bottoms of
    the walls while the high band holds only the tops of the walls. Along either
    horizontal axis both bands hold a strip of floor, a strip of ceiling, a
    perpendicular wall and roughly half the furniture, so they balance. Measured
    over 43 unseen rooms this is the single most reliable statistic available,
    which is why it leads.

    `skin` excludes the bounding surface itself, because the floor slab is a
    statement about the room's shape and not about what is standing on it. It is
    passed in rather than fixed because the caller running this over clutter has
    already removed the shell and must not pay for it twice.
    """
    span = hi - lo
    if span <= 0 or len(t) == 0:
        return 0.0
    band = min(_CLUTTER_BAND_M, 0.45 * span)
    skin = min(skin, 0.08 * span)
    low = float(np.count_nonzero((t > lo + skin) & (t < lo + skin + band)))
    high = float(np.count_nonzero((t < hi - skin) & (t > hi - skin - band)))
    if low + high <= 0:
        return 0.0
    return (low - high) / (low + high)


def _contact_fractions(
    xyz: np.ndarray, clutter: np.ndarray, axis: np.ndarray, lo: float, hi: float
) -> tuple[float, float]:
    """How much of the clutter is standing *on* each end of the axis, in [0, 1].

    The sharpest thing an interior knows about gravity, and the one signal here
    that a lopsided room cannot fake. Furniture does not merely sit near the
    floor, it rests on it: every object's lowest point is the floor's height, to
    within the thickness of a castor. A wardrobe pushed against a wall does not
    do the same thing to that wall -- it stops wherever its own depth stops --
    so along a horizontal axis the clutter's near extreme is scattered through
    the room, while along the vertical every last piece of it bottoms out on one
    plane.

    Measured by rastering the plane perpendicular to the candidate axis and
    asking, per column, whether the clutter in it reaches the boundary. Columns
    rather than raw points, because otherwise a single large object with a lot of
    surface near the floor would outvote five small ones that are not, and the
    question is how many separate things rest there.

    It is weighted lightly, and the reason is worth recording: columns do not
    fully solve that problem. A wardrobe standing against a wall fills a large
    patch of the perpendicular raster with columns that all bottom out on that
    wall, and anything behind it in projection is hidden by it, so a room with
    one big object pushed against one wall reports near-perfect contact on a
    horizontal axis. Measured alone over 43 unseen rooms this statistic picked
    the wrong axis on 11 of them, against 0 for the clutter band, so it corrects
    the band rather than the other way round.

    Returns (low_end, high_end); the axis cares about the larger and the sign
    cares about the difference.
    """
    if int(np.count_nonzero(clutter)) < 64 or hi <= lo:
        return 0.0, 0.0
    reference = _UP if abs(float(axis[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, reference)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)

    points = xyz[clutter]
    t = points @ axis
    grid = np.stack([points @ u, points @ v], axis=1)
    cells = np.floor((grid - grid.min(axis=0)) / _CONTACT_CELL_M).astype(np.int64)
    key = cells[:, 0] * (int(cells[:, 1].max()) + 1) + cells[:, 1]
    uniq, inverse = np.unique(key, return_inverse=True)

    order = np.lexsort((t, inverse))
    sorted_cell = inverse[order]
    sorted_t = t[order]
    starts = np.searchsorted(sorted_cell, np.arange(len(uniq)), side="left")
    ends = np.searchsorted(sorted_cell, np.arange(len(uniq)), side="right")
    populated = (ends - starts) >= 4
    if int(np.count_nonzero(populated)) < 8:
        return 0.0, 0.0
    bottoms = sorted_t[starts[populated]]
    tops = sorted_t[ends[populated] - 1]
    tol = _SHELL_SKIN_M + _CONTACT_TOL_M
    return (
        float(np.mean(bottoms <= lo + tol)),
        float(np.mean(tops >= hi - tol)),
    )


# How much a declared up direction counts. The two decisions want very
# different answers, which is why there are two weights.
#
# For the *axis*, the hint is worth the least of the geometric terms. A room
# full of furniture already knows which way is up to a fraction of a degree,
# while a hint is whatever angle a person happened to be holding a phone at; a
# hint strong enough to beat that evidence would lay a good scan on its side
# whenever someone filmed pointing at the floor. So it breaks ties and does
# nothing else.
#
# For the *sign*, it is worth twice the strongest single vote -- but still less
# than the four geometric votes combined, so it flips a split decision and
# loses to a unanimous one. This asymmetry is the point: the room's geometry is
# excellent at telling a vertical axis from a horizontal one and frequently
# hopeless at telling a floor from a ceiling, which is exactly the question a
# device that knows which way it was pointing can answer outright.
#
# Both terms scale with |cos| to the candidate axis, so a hint at right angles
# to the axis under test contributes nothing rather than contributing noise.
_W_UP_HINT = 0.3
_W_SIGN_UP_HINT = 2.0


def find_up(
    cloud: PointCloud,
    planes: list[Plane],
    *,
    camera_positions: np.ndarray | None = None,
    up_hint: np.ndarray | None = None,
) -> GravitySolution:
    """Recover the up vector in (already metric) source coordinates.

    The axis and the sign are settled by separate arguments.

    The axis is chosen from the room's three Manhattan directions, each one
    first re-estimated as the pole of its own great circle of surface normals so
    that the candidates are precise before they are compared. They are then
    scored on evidence that has nothing to do with how much surface faces which
    way: how much clutter piles up against one end of the axis rather than being
    spread along it, whether the span along the axis is a believable
    floor-to-ceiling distance, and how little the camera path moved along it.
    Surface area is deliberately absent. It was the old primary signal and it is
    simply wrong -- in a room taller than it is narrow the walls out-cover the
    floor, and an area-led vote lays the room on its side.

    The sign then points the axis from floor to ceiling, using the same clutter
    imbalance read with its sign kept, the relative weight of the two horizontal
    surfaces, and the camera's height above the floor.

    Both decisions come back with a margin over their runner-up, and a contested
    decision is reported as contested. A perfectly empty cubical box genuinely
    does not know which way is up, and the honest answer there is a warning
    rather than a confident guess.
    """
    if up_hint is not None:
        up_hint = np.asarray(up_hint, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(up_hint))
        up_hint = up_hint / n if n > 1e-9 else None

    sample = _sample_cloud(cloud)
    xyz = sample.xyz
    normals = sample.normals
    if normals is not None and len(normals) == len(xyz):
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.where(lengths == 0, 1.0, lengths)
    else:
        normals = None

    usable = [p for p in planes if _plane_weight(p) > 0]
    axes = manhattan_axes(usable, normals)
    if not axes:
        return GravitySolution(
            up=_UP.copy(),
            method={"gravity_axis": "assumed +Z", "gravity_sign": "assumed +Z"},
            warnings=["gravity assumed to be +Z; the scan had no usable surfaces"],
        )

    poles: list[np.ndarray] = []
    converged: list[bool] = []
    for axis in axes:
        if normals is None:
            poles.append(axis / np.linalg.norm(axis))
            converged.append(False)
            continue
        pole, ok = normal_pole(normals, axis)
        poles.append(pole)
        converged.append(ok)

    cams = None
    if camera_positions is not None and len(camera_positions) >= 8:
        cams = np.asarray(camera_positions, dtype=np.float64).reshape(-1, 3)
    extremes = _axis_extremes(xyz, poles)
    clutter = _clutter_mask(xyz, poles, extremes)

    spreads = [_spread(cams @ p) for p in poles] if cams is not None else []
    widest = max(spreads) if spreads else 0.0

    scores: list[float] = []
    detail: list[str] = []
    for i, pole in enumerate(poles):
        t = xyz @ pole
        lo, hi = extremes[i]
        band = _band_asymmetry(t, lo, hi, _CLUTTER_SKIN_M)
        free = _band_asymmetry(t[clutter], lo, hi, 0.0)
        low_contact, high_contact = _contact_fractions(xyz, clutter, pole, lo, hi)
        evidence = (
            _W_CLUTTER_BAND * abs(band)
            + _W_CLUTTER_FREE * abs(free)
            + _W_CLUTTER_CONTACT * max(low_contact, high_contact)
        )
        if cams is not None and widest > 0:
            evidence += _W_CAMERA_AXIS * (1.0 - spreads[i] / widest)
        if up_hint is not None:
            evidence += _W_UP_HINT * abs(float(np.dot(pole, up_hint)))
        prior = _VERTICAL_PRIOR_FLOOR + (1.0 - _VERTICAL_PRIOR_FLOOR) * _taper(
            hi - lo, _VERTICAL_PRIOR_MIN_M, _VERTICAL_PRIOR_MAX_M
        )
        scores.append(evidence * prior)
        detail.append(
            f"span {hi - lo:.2f} m, clutter {band:+.3f}/{free:+.3f}, "
            f"resting on it {max(low_contact, high_contact):.2f}"
        )

    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    best = order[0]
    runner = scores[order[1]] if len(order) > 1 else 0.0
    axis_margin = 0.0 if scores[best] <= 0 else float(1.0 - runner / scores[best])

    axis = poles[best]
    warnings: list[str] = []
    if scores[best] <= 0:
        warnings.append(
            "no clutter, no camera path and no plausible ceiling height, so "
            "nothing in this capture distinguishes up from sideways; the twin "
            "may be lying on its side"
        )
    elif axis_margin < _AXIS_MARGIN_WARN:
        warnings.append(
            f"the vertical axis beat the runner-up by only {axis_margin:.0%}; "
            f"this room is close to symmetric under swapping two of its axes, so "
            f"its width, depth and height may be transposed"
        )
    if not converged[best]:
        warnings.append(
            "too few surface normals lie on the great circle around the chosen "
            "axis to refine it, so gravity is only as good as the plane fit that "
            "seeded it"
        )

    axis, sign_margin, sign_how = _resolve_sign(
        axis, xyz, clutter, usable, cams, up_hint=up_hint
    )
    if sign_margin < _SIGN_MARGIN_WARN:
        warnings.append(
            f"floor and ceiling are nearly indistinguishable ({sign_how}), so the "
            f"twin may be upside down"
        )

    method = {
        "gravity_axis": (
            f"manhattan candidate {best + 1} of {len(poles)} refined on the "
            f"great circle of wall normals ({detail[best]}), margin "
            f"{axis_margin:.0%}"
        ),
        "gravity_sign": f"{sign_how}, margin {sign_margin:.0%}",
    }
    return GravitySolution(
        up=axis,
        axis_margin=axis_margin,
        sign_margin=sign_margin,
        pole=poles[best] if converged[best] else None,
        method=method,
        warnings=warnings,
    )


def _resolve_sign(
    axis: np.ndarray,
    xyz: np.ndarray,
    clutter: np.ndarray,
    planes: list[Plane],
    camera_positions: np.ndarray | None,
    up_hint: np.ndarray | None = None,
) -> tuple[np.ndarray, float, str]:
    """Point a known axis from floor to ceiling. Returns (axis, margin, how).

    Up to five votes, none of which is a statement about how big a surface is.
    Furniture stands on the floor and nothing hangs off the ceiling, so the
    clutter-heavy end is the floor; that is the strongest evidence and it is read
    twice, once over the whole cloud and once over the clutter alone. Clutter
    physically resting on a surface says the same thing more directly but is
    fooled by a wardrobe against a wall, so it is worth half. The floor is
    usually the better covered of the two horizontal surfaces, which is weaker
    still because a capture that stared at the ceiling inverts it. And the
    scanner rides at about chest height, which only discriminates in a room
    appreciably taller than a person.

    Kept apart from the axis decision on purpose. A sign error is a different
    failure -- the room comes out the right shape, hanging from its ceiling --
    and a single conflated score would report one confidence for two answers.
    """
    axis = axis / np.linalg.norm(axis)
    t = xyz @ axis
    lo, hi = float(np.quantile(t, 0.002)), float(np.quantile(t, 0.998))
    if hi - lo <= 0:
        return axis, 0.0, "degenerate span"

    votes: list[tuple[float, float, str]] = []

    if up_hint is not None:
        # The one vote here that is a measurement rather than an inference: a
        # device that reported which way it was pointing was reading an
        # accelerometer, and gravity is the only thing an accelerometer at rest
        # can see. It still cannot outvote the room unaided -- a phone waved at
        # the ceiling reports a tilted up -- so it is worth twice a clutter vote
        # and no more.
        votes.append((float(np.dot(axis, up_hint)), _W_SIGN_UP_HINT, "declared camera orientation"))

    votes.append((_band_asymmetry(t, lo, hi, _CLUTTER_SKIN_M), 1.0, "clutter band"))
    votes.append((_band_asymmetry(t[clutter], lo, hi, 0.0), 1.0, "clutter mass"))
    low_contact, high_contact = _contact_fractions(xyz, clutter, axis, lo, hi)
    votes.append((low_contact - high_contact, 0.5, "clutter resting on the surface"))

    horiz = math.cos(math.radians(_HORIZONTAL_TOL_DEG))
    family = [p for p in planes if abs(float(np.dot(p.normal, axis))) >= horiz]
    if len(family) >= 2:
        pair = sorted(family, key=_plane_weight, reverse=True)[:2]
        coords = [_axis_coordinate(p, axis) for p in pair]
        if all(math.isfinite(c) for c in coords):
            first, second = (0, 1) if coords[0] < coords[1] else (1, 0)
            w_lo, w_hi = _plane_weight(pair[first]), _plane_weight(pair[second])
            if w_lo + w_hi > 0:
                votes.append(((w_lo - w_hi) / (w_lo + w_hi), 0.5, "surface coverage"))

    if camera_positions is not None and len(camera_positions):
        cam_t = float(np.median(np.asarray(camera_positions).reshape(-1, 3) @ axis))
        err_lo = abs((cam_t - lo) - _CAMERA_HEIGHT_M)
        err_hi = abs((hi - cam_t) - _CAMERA_HEIGHT_M)
        if err_lo + err_hi > 1e-6:
            votes.append(((err_hi - err_lo) / (err_hi + err_lo), 0.5, "camera height"))

    total = sum(w for _, w, _ in votes)
    vote = sum(v * w for v, w, _ in votes)
    margin = abs(vote) / total if total > 0 else 0.0
    agreeing = ", ".join(
        name for v, _, name in votes if (v >= 0) == (vote >= 0) and abs(v) > 1e-9
    )
    how = f"{agreeing or 'no evidence'} put the floor at the {'low' if vote >= 0 else 'high'} end"
    return (axis if vote >= 0 else -axis), float(margin), how


def _wall_scatter_deg(planes: list[Plane], axis: np.ndarray) -> tuple[float, int]:
    """How far the wall normals scatter off perpendicular to `axis`, in degrees.

    Every wall in a building is vertical, so each wall normal should sit exactly
    on the great circle perpendicular to gravity. How far they actually sit off
    it is the wall family's own noise, measured without reference to the floor
    or to anything else the axis was derived from -- which is what lets it
    arbitrate a disagreement with the floor rather than merely restate one side
    of it.

    Weighted by plane size, because a 20 m2 wall and a 0.3 m2 reveal are not
    equally informative about which way is up.
    """
    vert = math.sin(math.radians(_WALL_TOL_DEG))
    walls = [
        p
        for p in planes
        if abs(float(np.dot(p.normal, axis))) <= vert and _plane_weight(p) > 0
    ]
    if len(walls) < 2:
        return float("nan"), len(walls)
    tilts = np.array(
        [math.degrees(math.asin(min(1.0, abs(float(np.dot(p.normal, axis)))))) for p in walls]
    )
    weights = np.array([_plane_weight(p) for p in walls])
    rms = float(np.sqrt(np.sum(weights * tilts**2) / np.sum(weights)))
    return rms, len(walls)


def _refine_on_floor(
    axis: np.ndarray, planes: list[Plane], warnings: list[str]
) -> tuple[np.ndarray, float]:
    """Sharpen a settled axis on the heaviest surface perpendicular to it.

    A floor slab is tens of thousands of points spread over several metres and
    its least-squares normal is good to hundredths of a degree, which is better
    than any vote can do. The reason this is safe here and was not safe as the
    primary signal is that it can no longer change the answer: the axis has
    already been chosen, and a plane that wants to move it by more than
    `_REFINE_MAX_DEG` is not the floor -- it is a ramp, a domed scrap the plane
    detector split off, or a table top -- so it is refused and the axis the
    surface normals produced stands.

    Returns (axis, swing_deg), where the swing is how far the two disagreed. That
    number is the honest cross-check on gravity and the caller is expected to
    record it: the floor plane and the great circle of wall normals are two
    independent estimates of the same direction, so when they part company by a
    degree one of them is wrong and the twin has no business claiming otherwise.
    NaN when there was no perpendicular surface to compare against at all.
    """
    horiz = math.cos(math.radians(_HORIZONTAL_TOL_DEG))
    family = [p for p in planes if abs(float(np.dot(p.normal, axis))) >= horiz]
    if not family:
        warnings.append(
            "no surface lies perpendicular to the recovered vertical, so gravity "
            "rests on the surface normals alone and is a degree-scale estimate"
        )
        return axis, float("nan")
    heaviest = max(family, key=_plane_weight)
    refined = heaviest.normal.copy()
    if float(np.dot(refined, axis)) < 0:
        refined = -refined
    swing = math.degrees(math.acos(float(np.clip(np.dot(refined, axis), -1.0, 1.0))))
    if swing <= _REFINE_MAX_DEG:
        return refined, swing

    scatter, n_walls = _wall_scatter_deg(planes, axis)
    tolerance = _REFINE_MAX_DEG
    if math.isfinite(scatter) and n_walls >= 2:
        tolerance = float(
            np.clip(_REFINE_SIGMA_K * scatter, _REFINE_MAX_DEG, _REFINE_MAX_TOLERANCE_DEG)
        )

    if swing <= tolerance:
        warnings.append(
            f"the heaviest horizontal surface sits {swing:.2f} deg off the vertical "
            f"the wall normals recover, but those {n_walls} walls scatter "
            f"{scatter:.2f} deg about that vertical themselves, so they cannot "
            f"resolve a disagreement this small; gravity was levelled on the "
            f"surface, which is the better-conditioned of the two"
        )
        return refined, swing

    warnings.append(
        f"the heaviest horizontal surface sits {swing:.2f} deg off the vertical "
        f"the wall normals recover, further than a floor and a wall can honestly "
        f"disagree; gravity was left on the surface-normal estimate rather than "
        f"levelled to that surface, and one of the two is wrong"
    )
    return axis, swing


def _wall_planes(
    planes: list[Plane], axis: np.ndarray, min_rel_weight: float
) -> list[Plane]:
    """Planes vertical enough and big enough to be treated as walls.

    The relative weight floor is what keeps a 12 cm window reveal or a chair
    back out of the frame estimate. Those surfaces really are vertical, so an
    angle test alone admits them, and being small they are also the worst
    fitted -- exactly the planes whose azimuth error is largest.
    """
    vert = math.sin(math.radians(_WALL_TOL_DEG))
    candidates = [
        p
        for p in planes
        if abs(float(np.dot(p.normal, axis))) <= vert and _plane_weight(p) > 0
    ]
    if not candidates:
        return []
    heaviest = max(_plane_weight(p) for p in candidates)
    return [p for p in candidates if _plane_weight(p) >= min_rel_weight * heaviest]



# ---------------------------------------------------------------------------
# yaw
# ---------------------------------------------------------------------------


def find_yaw(cloud: PointCloud, planes: list[Plane], up: np.ndarray) -> tuple[float, str]:
    """Rotation about `up`, in radians, that lays the walls on the axes.

    Wall azimuths in a rectangular room are four values 90 degrees apart, so the
    quantity with a single well-defined mean is the azimuth taken modulo 90
    degrees. Multiplying the angle by four maps those four clusters onto one and
    turns the mod-90 wrap into an ordinary circular mean, which a histogram
    argmax cannot do without quantising the answer to a bin. Each wall is
    weighted by its area so that the room's long walls decide the frame and a
    cabinet door does not.
    """
    axis = np.asarray(up, dtype=np.float64).reshape(3)
    axis = axis / np.linalg.norm(axis)
    to_z = rotation_between(axis, _UP)

    sin_sum = 0.0
    cos_sum = 0.0
    used = 0
    for p in _wall_planes(planes, axis, _YAW_MIN_REL_WEIGHT):
        n = to_z @ p.normal
        n = n - float(n[2]) * _UP
        length = float(np.linalg.norm(n))
        if length < 1e-9:
            continue
        w = _plane_weight(p)
        theta = math.atan2(float(n[1]), float(n[0]))
        sin_sum += w * math.sin(4.0 * theta)
        cos_sum += w * math.cos(4.0 * theta)
        used += 1

    if used == 0 or (abs(sin_sum) + abs(cos_sum)) < 1e-12:
        return 0.0, "none"
    residual = math.atan2(sin_sum, cos_sum) / 4.0
    return -residual, f"manhattan_circular_mean({used} walls)"


# ---------------------------------------------------------------------------
# the whole job
# ---------------------------------------------------------------------------


def canonicalize(
    cloud: PointCloud,
    *,
    mesh: Mesh | None = None,
    planes: list[Plane] | None = None,
    normals: np.ndarray | None = None,
    camera_positions: np.ndarray | None = None,
    up_hint: np.ndarray | None = None,
    unit_hint: str | None = None,
    unit_hint_confidence: float | None = None,
    unit_hint_evidence: list[str] | None = None,
    seed: int = 0,
) -> Canonicalization:
    """Scale, level, square up and centre a scan in one transform.

    Nothing here is allowed to run out of order. The unit priors are metric, so
    they must see the raw coordinates. Plane detection thresholds are metric
    too, so planes are detected *after* scaling -- a 3 cm inlier band applied to
    a cloud that is really in centimetres is a 0.3 mm band, tighter than the
    sensor noise, and the detector would come back with nothing. Callers who
    pass planes in are assumed to have them in source units and they are carried
    through the scale with the geometry.
    """
    warnings: list[str] = []
    method: dict[str, str] = {}

    scale = infer_unit_scale(
        cloud,
        planes=planes,
        hint=unit_hint,
        hint_confidence=unit_hint_confidence,
        hint_evidence=unit_hint_evidence,
    )
    method["scale"] = f"{scale.unit} (confidence {scale.confidence:.2f})"
    if not scale.is_known:
        warnings.append(
            "unit could not be inferred; geometry left in source units and every "
            "downstream measurement is therefore unitless"
        )
    elif scale.confidence < 0.5:
        warnings.append(
            f"unit {scale.unit!r} inferred with low confidence {scale.confidence:.2f}"
        )

    factor = scale.factor
    scale_matrix = homogeneous(np.eye(3) * factor)
    metric = cloud.transformed(scale_matrix)
    cams = None
    if camera_positions is not None and len(camera_positions):
        cams = np.asarray(camera_positions, dtype=np.float64).reshape(-1, 3) * factor

    # normals are directions, so a uniform scale leaves them alone
    if normals is None:
        normals = cloud.normals
    if normals is not None and len(normals) == len(cloud.xyz):
        metric.normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    else:
        est = estimate_normals(metric)
        metric.normals = orient_normals(est, metric.xyz, viewpoints=cams)

    if planes is None:
        planes = detect_planes(metric, normals=metric.normals, seed=seed)
    else:
        planes = [transform_plane(p, scale_matrix) for p in planes]
    if not planes:
        warnings.append("no planes detected; falling back to an assumed up vector")

    sample_cloud = _sample_cloud(metric, seed=seed)

    gravity = find_up(metric, planes, camera_positions=cams, up_hint=up_hint)
    up = gravity.up
    method.update(gravity.method)
    warnings += gravity.warnings

    # The axis is now settled, so the most precise estimate of gravity available
    # is the normal of the heaviest surface perpendicular to it -- the floor in
    # almost every capture, fitted over tens of thousands of points. That refit
    # is a *precision* step and never a decision: it is only ever allowed to move
    # the axis by a fraction of a degree, and if it wants to move it further the
    # plane it came from is not the floor and is ignored.
    #
    # The size of that move is also the only honest cross-check on gravity we
    # have, because the two things being compared -- a plane fitted to one
    # surface and the pole of the great circle of every wall normal in the room --
    # share no arithmetic. The old cross-check compared the pole against a
    # quantity derived from the floor plane it was supposed to be auditing, and
    # duly reported agreement to 0.08 degrees with an answer that was 90 degrees
    # wrong; a check that confirms whatever it is given is worse than none.
    up, floor_swing = _refine_on_floor(up, planes, warnings)
    if gravity.pole is not None and math.isfinite(floor_swing):
        method["gravity_crosscheck"] = (
            f"the heaviest surface perpendicular to gravity and the great circle "
            f"of wall normals, which share no arithmetic, differ by "
            f"{floor_swing:.2f} deg"
        )

    yaw, yaw_method = find_yaw(metric, planes, up)
    method["yaw"] = yaw_method
    if yaw_method == "none":
        warnings.append("no wall planes found; yaw left unrotated")

    rotation = rotation_z(yaw) @ rotation_between(up, _UP)

    # identify floor and ceiling in the levelled frame
    levelled = [transform_plane(p, homogeneous(rotation)) for p in planes]
    horiz = math.cos(math.radians(_HORIZONTAL_TOL_DEG))
    family = [
        (i, p) for i, p in enumerate(levelled) if abs(float(p.normal[2])) >= horiz
    ]
    family.sort(key=lambda ip: _plane_weight(ip[1]), reverse=True)
    sample = sample_cloud.xyz @ rotation.T
    z_lo = float(np.quantile(sample[:, 2], 0.002))
    z_hi = float(np.quantile(sample[:, 2], 0.998))

    # The floor and the ceiling are the two *heaviest* horizontal surfaces, not
    # the lowest and the highest. On a dense scan the detector returns a tail of
    # small near-horizontal patches -- a slightly domed scrap of floor the refit
    # did not absorb, the top of a door frame -- and taking the lowest of those
    # as the floor picks whichever fragment happens to sit a centimetre low and
    # tilted by degrees, which then defines gravity for the whole twin.
    floor_index: int | None = None
    ceiling_index: int | None = None
    if len(family) >= 2:
        pair = sorted(family[:2], key=lambda ip: _axis_coordinate(ip[1], _UP))
        floor_index, ceiling_index = pair[0][0], pair[1][0]
    elif len(family) == 1:
        index, plane = family[0]
        height = _axis_coordinate(plane, _UP)
        if height - z_lo <= _FLOOR_BAND * max(z_hi - z_lo, 1e-9):
            floor_index = index
        else:
            ceiling_index = index
            warnings.append(
                "the only horizontal surface found sits at the top of the cloud; "
                "treating it as a ceiling and taking the floor from the points"
            )
    else:
        warnings.append("no horizontal plane found; floor left at the cloud minimum")

    if floor_index is None:
        floor_z = z_lo
        footprint = sample
    else:
        floor_z = float(_axis_coordinate(levelled[floor_index], _UP))
        # gate the footprint on the normals as well as the distance, so the
        # bottom courses of the walls and the feet of the furniture do not get
        # counted as floor and drag the origin around
        mask = plane_inliers(
            sample_cloud,
            planes[floor_index],
            distance_thresh=_FLOOR_INLIER_THRESH_M,
            normals=sample_cloud.normals,
            normal_thresh_deg=_HORIZONTAL_TOL_DEG,
        )
        footprint = sample[mask] if np.count_nonzero(mask) >= 32 else sample
        if np.count_nonzero(mask) < 32:
            warnings.append("floor plane has too few inliers to place the origin")

    # the quarter turn: the room's own shape decides, not the detector's luck
    quarter, xy_centre, decisive = _choose_quadrant(sample, footprint)
    rotation = rotation_z(0.5 * math.pi * quarter) @ rotation
    method["quarter_turn"] = f"{quarter * 90} deg to put the long axis on +X"
    if not decisive:
        warnings.append(
            "the room is too symmetric to choose between the two half-turns; "
            "another scan of the same space may come out rotated 180 degrees"
        )

    # The floor's height has to be read at the origin the twin will actually
    # have, which is why the XY move is applied first and the Z intercept second.
    # Taken at the old origin instead -- while the same step slides XY to the
    # room's centre -- the floor lands off z = 0 by the XY offset times the
    # tangent of whatever tilt survived levelling. That is millimetres in a small
    # room and grows with the size of the space, and it is the wrong order of
    # operations regardless.
    xy_shift = np.array([-xy_centre[0], -xy_centre[1], 0.0])
    if floor_index is not None:
        floor_z = float(
            _axis_coordinate(
                transform_plane(planes[floor_index], homogeneous(rotation, xy_shift)),
                _UP,
            )
        )

    translation = np.array([-xy_centre[0], -xy_centre[1], -floor_z])
    # the planes were detected on the already-scaled cloud, so they take the
    # rotation and translation but must not be scaled a second time
    to_canonical = homogeneous(rotation, translation)
    transform = homogeneous(rotation * factor, translation)

    final = [transform_plane(p, to_canonical) for p in planes]
    floor_plane = None
    ceiling_plane = None
    if floor_index is not None:
        floor_plane = _oriented(final[floor_index], _UP, "floor")
    if ceiling_index is not None:
        candidate = _oriented(final[ceiling_index], -_UP, "ceiling")
        # A capture that never looked up leaves the highest horizontal plane
        # somewhere mid-room -- a table top, the top of a wardrobe. Calling that
        # a ceiling would report a 1.15 m room, so a candidate has to actually
        # cap the point distribution to be believed, and an open-topped capture
        # is supposed to come back with no ceiling at all.
        top_z = float(np.quantile(sample[:, 2], 0.995)) - floor_z
        height = abs(float(candidate.offset))
        if height >= _CEILING_MIN_HEIGHT_M and height >= _CEILING_MIN_COVERAGE * top_z:
            ceiling_plane = candidate
        else:
            warnings.append(
                f"highest horizontal plane sits {height:.2f} m up but the cloud "
                f"reaches {top_z:.2f} m; treating the capture as open-topped"
            )

    # a last sanity check on gravity that does not depend on any plane: whatever
    # a room's plan dimensions are, its floor-to-ceiling distance is a couple of
    # metres to about six, so a vertical span outside that band is evidence that
    # a horizontal axis has been elected as up
    canonical = sample_cloud.xyz @ rotation.T + translation
    extent = np.quantile(canonical, 0.999, axis=0) - np.quantile(canonical, 0.001, axis=0)
    if not _VERTICAL_MIN_M <= extent[2] <= _VERTICAL_MAX_M:
        warnings.append(
            f"the recovered up axis spans {extent[2]:.2f} m, outside the "
            f"{_VERTICAL_MIN_M:.2f}-{_VERTICAL_MAX_M:.2f} m band a floor-to-ceiling "
            f"distance lives in (the horizontal axes span {extent[0]:.2f} m and "
            f"{extent[1]:.2f} m); gravity may be wrong"
        )

    # The gravity residual is re-derived from the transformed points by a route
    # that shares nothing with the plane that produced the transform -- see
    # `independent_up_residual`. Measuring it off `floor_plane` instead, as this
    # did until it was caught, made it arithmetically incapable of being anything
    # but zero, which turned both the warning below and the QA gate downstream
    # into decoration.
    up_residual, ground_cells = independent_up_residual(canonical)
    method["up_residual"] = (
        f"floor refitted from {ground_cells} ground cells of the canonical cloud, "
        f"independently of the plane the twin was levelled on"
    )
    yaw_residual = _yaw_residual(final)

    if math.isnan(up_residual):
        warnings.append(
            "the canonical cloud has no surface flat enough to refit as a floor, "
            "so the twin's alignment with gravity could not be verified "
            "independently of the plane it was levelled on; treat its levelling "
            "as unchecked rather than as confirmed"
        )
    elif up_residual > 0.25:
        warnings.append(f"gravity residual {up_residual:.3f} deg exceeds 0.25 deg")
    if math.isnan(yaw_residual):
        warnings.append(
            "no wall planes were found, so the room's rotation about the vertical "
            "axis is unverified; the twin is metric and level but may not be "
            "squared up to its own walls"
        )
    elif yaw_residual > 0.5:
        warnings.append(f"yaw residual {yaw_residual:.3f} deg exceeds 0.5 deg")

    return Canonicalization(
        transform=transform,
        up_residual_deg=up_residual,
        yaw_residual_deg=yaw_residual,
        floor_plane=floor_plane,
        ceiling_plane=ceiling_plane,
        scale=scale,
        method=method,
        warnings=warnings,
        gravity_axis_margin=float(gravity.axis_margin),
        gravity_sign_margin=float(gravity.sign_margin),
    )


def independent_up_residual(xyz: np.ndarray) -> tuple[float, int]:
    """Angle between +Z and a floor re-derived from canonical points, in degrees.

    The point of this function is that it must be able to come back large. The
    number it replaces was read off the same floor plane that defined the
    rotation, so it was arithmetically incapable of being anything but zero and
    the checks that consumed it -- a warning here, a gate in `scan.qa` and an
    assertion in the accuracy suite -- were all inert. A residual that cannot
    fail is worse than no residual, because it reports a verified alignment
    precisely when nothing was verified.

    So the floor is found again, from the transformed points, by a route that
    reuses none of the arithmetic that produced the transform: no RANSAC, no
    surface normals, no plane list. The cloud is rastered in XY and the low
    quantile of each column is taken, which is the classic ground extraction and
    is the only estimator here that does not need to be told what a floor looks
    like; the points sitting on those column bottoms are then fitted by total
    least squares and trimmed once about their own median. If the twin was
    levelled on a tilted scrap, or stood on its side, this fit follows the real
    geometry and disagrees, which is the whole purpose.

    A column whose bottom is far above the room's own floor level is a patch the
    sweep never reached, so its lowest point is a table top or a ceiling and it
    is dropped -- otherwise a single occluded corner would rotate the fit.

    Returns (degrees, cells_used), with NaN when too little floor survives to
    fit anything *or* when what survives is not flat enough to be a floor,
    because an unmeasurable check must not report zero -- and must not report a
    plausible-looking non-zero either. A partial capture whose column bottoms
    are furniture will otherwise hand back a degree and a half of "tilt" that is
    really the height of a chair, and that number then straddles a QA gate and
    contradicts the better-conditioned estimate `scan.qa` makes from the same
    twin.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(xyz) < 256:
        return float("nan"), 0
    z = xyz[:, 2]
    span = float(np.quantile(z, 0.999) - np.quantile(z, 0.001))
    if span <= 0:
        return float("nan"), 0

    lo_xy = xyz[:, :2].min(axis=0)
    cells = np.floor((xyz[:, :2] - lo_xy) / _GROUND_CELL_M).astype(np.int64)
    width = int(cells[:, 1].max()) + 1
    key = cells[:, 0] * width + cells[:, 1]
    uniq, inverse = np.unique(key, return_inverse=True)

    order = np.lexsort((z, inverse))
    sorted_cell = inverse[order]
    sorted_z = z[order]
    starts = np.searchsorted(sorted_cell, np.arange(len(uniq)), side="left")
    counts = np.searchsorted(sorted_cell, np.arange(len(uniq)), side="right") - starts
    with np.errstate(invalid="ignore"):
        offset = np.minimum(
            (counts * _GROUND_QUANTILE).astype(np.int64), np.maximum(counts - 1, 0)
        )
    bottom = np.where(counts > 0, sorted_z[np.clip(starts + offset, 0, len(sorted_z) - 1)], np.inf)

    # a column has to hold enough points for its low quantile to mean anything,
    # and has to bottom out near the room's own floor rather than on a table top
    floor_level = float(np.quantile(bottom[np.isfinite(bottom)], 0.10))
    reach = max(0.20, min(1.0, 0.35 * span))
    good = (counts >= 4) & (bottom <= floor_level + reach)
    if int(np.count_nonzero(good)) < 12:
        return float("nan"), int(np.count_nonzero(good))

    on_floor = good[inverse] & (z <= bottom[inverse] + _GROUND_TRIM_M)
    if int(np.count_nonzero(on_floor)) < 64:
        return float("nan"), int(np.count_nonzero(good))

    points = xyz[on_floor]
    normal, offset_d = _total_least_squares(points)
    residual = points @ normal - offset_d
    sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    keep = np.abs(residual) < max(3.0 * sigma, 0.005)
    if int(np.count_nonzero(keep)) >= 64:
        points = points[keep]
        normal, offset_d = _total_least_squares(points)
        residual = points @ normal - offset_d
        sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))

    if sigma > _GROUND_MAX_SCATTER_M:
        return float("nan"), int(np.count_nonzero(good))

    angle = math.degrees(math.acos(float(np.clip(abs(normal[2]), -1.0, 1.0))))
    return angle, int(np.count_nonzero(good))


def _total_least_squares(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Perpendicular-distance plane fit, as (unit normal, offset).

    Written out here rather than borrowed from `geom.planes` so that the
    independent residual really is independent: sharing the estimator with the
    module that produced the transform would leave one common failure mode
    between the measurement and the thing it is measuring.
    """
    centroid = points.mean(axis=0)
    centred = points - centroid
    _, vectors = np.linalg.eigh(centred.T @ centred)
    normal = vectors[:, 0]
    normal = normal / np.linalg.norm(normal)
    return normal, float(normal @ centroid)


def _choose_quadrant(
    sample: np.ndarray, footprint: np.ndarray
) -> tuple[int, np.ndarray, bool]:
    """Pick one of the four 90-degree yaws, the XY origin, and whether we mean it.

    Manhattan yaw is only defined modulo 90 degrees, so without this step the
    same room canonicalises four different ways depending on which wall the
    plane detector hit first, and two twins of one room stop being comparable.
    The long axis of the footprint picks the axis; the room's own mass asymmetry
    picks which end of it, since a rectangle is symmetric under a half turn and
    the contents of the room are not. A room whose contents are symmetric too
    has no half-turn evidence at all, and the third return value says so rather
    than pretending the coin toss was a decision.
    """
    lo = np.quantile(footprint[:, :2], 0.002, axis=0)
    hi = np.quantile(footprint[:, :2], 0.998, axis=0)
    centre = 0.5 * (lo + hi)
    # which axis is longest is asked of the whole room, not of the floor: a
    # floor slab that has been sliced at a slight angle, or half hidden under
    # furniture, is a much less reliable read on the room's proportions than the
    # walls are, and getting this backwards transposes width and depth
    span_lo = np.quantile(sample[:, :2], 0.002, axis=0)
    span_hi = np.quantile(sample[:, :2], 0.998, axis=0)
    extent = span_hi - span_lo

    quarter = 0 if extent[0] >= extent[1] else 1
    turned = sample[:, :2] @ rotation_z(0.5 * math.pi * quarter)[:2, :2].T
    centre_t = centre @ rotation_z(0.5 * math.pi * quarter)[:2, :2].T
    skew = turned.mean(axis=0) - centre_t
    span = max(float(extent.max()), 1e-9)
    deadband = _SKEW_DEADBAND * span
    if abs(skew[0]) > deadband:
        if skew[0] < 0:
            quarter += 2
    elif skew[1] < 0:
        quarter += 2
    decisive = abs(skew[0]) > deadband or abs(skew[1]) > deadband

    turn = rotation_z(0.5 * math.pi * quarter)[:2, :2]
    return quarter % 4, centre @ turn.T, decisive


def _yaw_residual(planes: list[Plane]) -> float:
    """Area-weighted RMS deviation of the wall azimuths from the nearest axis.

    RMS rather than the mean, because the mean of a symmetric set of errors is
    zero however badly the walls are squared up, and this number is supposed to
    be able to fail.

    With no walls to measure the answer is NaN, not zero. Returning zero here
    would report a perfectly squared-up room precisely when we found nothing to
    square it up against -- the failure that produced a twin rotated 15 degrees
    off its own walls while its own diagnostics said the yaw was exact.
    """
    total = 0.0
    acc = 0.0
    for p in _wall_planes(planes, _UP, _RESIDUAL_MIN_REL_WEIGHT):
        w = _plane_weight(p)
        theta = math.atan2(float(p.normal[1]), float(p.normal[0]))
        wrapped = math.atan2(math.sin(4.0 * theta), math.cos(4.0 * theta)) / 4.0
        acc += w * math.degrees(wrapped) ** 2
        total += w
    if total <= 0:
        return float("nan")
    return math.sqrt(acc / total)


def apply(
    cloud: PointCloud, mesh: Mesh | None, canon: Canonicalization
) -> tuple[PointCloud, Mesh | None]:
    """Move geometry into canonical space with the transform we just solved.

    Kept separate from `canonicalize` so the decision can be inspected, logged
    or overridden before twenty million points are rewritten.
    """
    moved = cloud.transformed(canon.transform)
    moved_mesh = None if mesh is None else mesh.transformed(canon.transform)
    return moved, moved_mesh
