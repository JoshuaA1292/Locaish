"""Deciding what one unit of an export actually means on the ground.

A scan file is a pile of numbers with no dimension attached. Polycam writes
metres, some Matterport and CAD exports write centimetres, a lot of OBJ files
that have been through SketchUp write inches, and plenty of formats write
nothing at all in the header. Getting this wrong is the one error that poisons
every downstream number at once: a room in centimetres read as metres is a
520-metre hall whose "windows" are 140 m wide, and every clearance, every lens
choice and every sun angle computed on top of it is nonsense. So this module
refuses to guess quietly. It either has a header that says the unit, or it
scores each candidate unit by how architecturally plausible the resulting room
is, or it says "unknown" out loud and lets the caller stop.

The floor-to-ceiling distance is by far the sharpest signal. Building codes and
human bodies mean rooms cluster hard around 2.4-3.0 m, and essentially nothing
habitable is under 2.0 m or over about 5.5 m. That band spans a factor of about
2.7, which is the reason this works at all: the two candidate units that are
closest together, centimetres and inches, differ by 2.54, so at most one of them
can land inside the band. If a candidate makes the ceiling 2.85 m, the same
geometry read in the neighbouring unit is either 1.12 m or 7.24 m, and both are
architecturally impossible. The band therefore has to stay tight; widening it to
be generous with cathedral ceilings would immediately re-open the cm/in
ambiguity, so the prior is a sharp core around 2.75 m plus a low, broad shoulder
that keeps genuinely tall rooms plausible rather than a flat window.

The catch is that this module runs before gravity is known -- it has to, because
the priors that identify a unit are metric and the geometry that identifies
gravity is not -- so we cannot point at the floor-to-ceiling distance. What we
can measure is the set of distances between facing surfaces, one of which is the
ceiling height and the rest of which are plan dimensions. Every assignment is
therefore scored as a whole and mixed by how much surface each pair accounts
for. Scoring the whole assignment is what makes the answer stable: a wrong unit
can nearly always make *some* dimension look like a convincing ceiling, but only
the right one leaves the remaining dimensions looking like a room, and the
strongest single check is that a room is wider than it is tall.

The total diagonal extent and the median nearest-neighbour spacing are the two
corroborators. Spacing is the more interesting of them because it is the only
evidence here that is not a statement about the shape of the room: a consumer
LiDAR sweep lands points 3-60 mm apart in the real world whatever unit the file
happens to be written in, so a candidate factor implying 0.2 mm or 400 mm
between neighbours is refuted no matter how convincing a ceiling it produces.
That independence is only worth having if the prior is broad enough to mean it.
A narrow one does not refute, it referees, and it refereed badly: a large hall
scanned at 40 mm was read 2.54x too small because the wrong unit happened to put
the spacing nearer the peak, which is a fact about the scanner's settings and
says nothing whatever about the room.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from ..geom.normals import estimate_normals
from ..types import Plane, PointCloud

# ---------------------------------------------------------------------------
# the candidates
# ---------------------------------------------------------------------------

CANDIDATE_UNITS: dict[str, float] = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "in": 0.0254,
    "ft": 0.3048,
}

# Whatever a reader scrapes out of a header, normalised. Anything not in here is
# treated as no hint at all rather than as an error, because a reader inventing
# a unit name must not be able to abort a scan.
_UNIT_ALIASES: dict[str, str] = {
    "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "cm": "cm", "centimeter": "cm", "centimeters": "cm",
    "centimetre": "cm", "centimetres": "cm",
    "mm": "mm", "millimeter": "mm", "millimeters": "mm",
    "millimetre": "mm", "millimetres": "mm",
    "in": "in", "inch": "in", "inches": "in", '"': "in",
    "ft": "ft", "foot": "ft", "feet": "ft", "'": "ft",
}

# Prior shape constants, named so the report can quote them and so a future
# tuning pass changes one number rather than hunting through expressions.
_CEIL_CORE_M = 2.75      # centre of the sharp core, metres
_CEIL_CORE_LOGSIGMA = 0.16   # ~1.17x either side
_CEIL_WIDE_M = 3.05      # centre of the tall-room shoulder
_CEIL_WIDE_LOGSIGMA = 0.40
_CEIL_WIDE_WEIGHT = 0.15
_CEIL_MIN_M = 2.00       # soft cutoffs; a crawl space is not a room
_CEIL_MAX_M = 5.80
# The two edges of that window are deliberately not the same shape. The lower one
# is the sharpest fact in this module -- a habitable room is essentially never
# under 2.1 m, and building codes make that a hard floor rather than a tendency --
# and it is also the edge that does the work, because every way of reading a room
# too small hands the ceiling prior a sub-2 m number and hopes it is forgiven. A
# lower edge as soft as the upper one leaves a 1.77 m "ceiling" a sixth of full
# credit, which is enough for the rest of a wrongly scaled room to carry it. The
# upper edge stays soft because tall rooms are real and a 6 m atrium must degrade
# rather than be ruled impossible.
_CEIL_EDGE_LO = 0.030
_CEIL_EDGE_HI = 0.060

_PLAN_CENTRE_M = 4.5
_PLAN_LOGSIGMA = 0.55
_PLAN_MIN_M = 1.6
_PLAN_MAX_M = 40.0

_DIAG_CENTRE_M = 9.0
_DIAG_LOGSIGMA = 0.60
_DIAG_MIN_M = 1.5
_DIAG_MAX_M = 80.0

# A consumer sweep lands points roughly 3 to 60 mm apart in the real world, and
# anywhere in that range is ordinary. The prior is therefore broad on purpose:
# its job is to refute a candidate that implies a fifth of a millimetre or half a
# metre between neighbouring points, not to referee between two readings that are
# both plausible sampling rates. A narrow prior does referee, and it refereed
# wrongly -- a large hall scanned at 40 mm was read 2.54x too small because the
# wrong unit put the spacing nearer the peak, which is the sampler's business and
# says nothing about the room.
_SPACING_CENTRE_M = 0.014
_SPACING_LOGSIGMA = 0.75
_SPACING_MIN_M = 0.0015
_SPACING_MAX_M = 0.12
_SPACING_EDGE = 0.18

# How much each signal is trusted, as an exponent on its probability.
_W_CEILING = 1.0
_W_DIAG = 0.5
_W_SPACING = 0.35

# Below this weighted-geometric-mean plausibility, no candidate is good enough
# to name and we return "unknown". Re-measured after the ceiling and spacing
# priors were retuned: an ordinary room at its true scale scores 0.81, the
# hardest legitimate one (a 9.5 x 7.2 x 4.6 m hall, whose proportions are
# unusual enough that the priors genuinely struggle) scores 0.107, and the same
# room at a nonsense 7.3x scale scores below 1e-4. The floor sits between the
# last two with more than a factor of three of headroom on each side.
_PLAUSIBILITY_FLOOR = 0.03
# Plausibility at which confidence saturates, so an odd-but-real room reports
# reduced confidence instead of a flat 1.0.
_PLAUSIBILITY_FULL = 0.12
# Ceiling on confidence when the cloud yielded no pair of facing surfaces.
_NO_GAP_MAX_CONFIDENCE = 0.4
# How far a floor/ceiling spike has to stand above the surrounding histogram
# before it counts as a real surface rather than a lump in a smooth cloud.
_SLAB_PROMINENCE = 2.5
# Exponent applied to slab mass before it becomes a mixture weight.
_MASS_SHARPNESS = 3.0
# Where the "rooms are wider than they are tall" penalty turns over, expressed
# as the ratio of the narrowest plan dimension to the ceiling height, and how
# abruptly it does so.
_ASPECT_KNEE = 0.95
_ASPECT_WIDTH = 0.16

_SUBSAMPLE = 120_000       # points used for the normal/histogram work
_NN_TREE_MAX = 400_000     # points used to build the spacing KD-tree
_NN_QUERIES = 20_000


@dataclass
class ScaleEstimate:
    """The unit decision, with the reasoning kept attached to it.

    `factor` multiplies source coordinates to give metres. `evidence` is printed
    verbatim by the CLI, because a user whose room came out 2.54x wrong needs to
    see which prior was responsible without reading this file.
    """

    factor: float = 1.0
    unit: str = "unknown"
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.factor = float(self.factor)
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))

    @property
    def is_known(self) -> bool:
        return self.unit != "unknown"

    def to_dict(self) -> dict[str, object]:
        return {
            "factor": self.factor,
            "unit": self.unit,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }


# ---------------------------------------------------------------------------
# priors
# ---------------------------------------------------------------------------


def _soft_window(
    x: float, lo: float, hi: float, width: float = 0.06, width_hi: float | None = None
) -> float:
    """Logistic band-pass in log space, so the cutoffs are ratios not offsets.

    A hard window would make the whole unit decision discontinuous at the edge:
    a 1.99 m ceiling would be impossible and a 2.01 m ceiling certain. This
    decays over a few percent instead, which keeps a marginal room scoring
    marginally rather than flipping to the next unit.

    The two edges take separate widths because the priors here are not symmetric.
    A ceiling is bounded hard below by what a person can stand up in and only
    loosely above by architectural fashion, so the same softness on both sides
    either forgives an impossible crawl space or rules out a real atrium.
    """
    if x <= 0:
        return 0.0
    if width_hi is None:
        width_hi = width
    lx = math.log(x)
    a = 1.0 / (1.0 + math.exp(-(lx - math.log(lo)) / width))
    b = 1.0 / (1.0 + math.exp(-(math.log(hi) - lx) / width_hi))
    return a * b


def _lognormal(x: float, centre: float, log_sigma: float) -> float:
    if x <= 0:
        return 0.0
    z = (math.log(x) - math.log(centre)) / log_sigma
    return math.exp(-0.5 * z * z)


def ceiling_prior(height_m: float) -> float:
    """Plausibility in [0, 1] of a floor-to-ceiling distance, unnormalised.

    Sharp core plus a low shoulder: the core does the cm-versus-inch separation,
    the shoulder is what stops a 4.6 m warehouse ceiling from being scored as
    impossible and thrown to "unknown".
    """
    core = _lognormal(height_m, _CEIL_CORE_M, _CEIL_CORE_LOGSIGMA)
    wide = _CEIL_WIDE_WEIGHT * _lognormal(height_m, _CEIL_WIDE_M, _CEIL_WIDE_LOGSIGMA)
    return (core + wide) * _soft_window(
        height_m, _CEIL_MIN_M, _CEIL_MAX_M, _CEIL_EDGE_LO, _CEIL_EDGE_HI
    )


def plan_prior(span_m: float) -> float:
    """Plausibility of a horizontal room dimension, unnormalised.

    Much broader than the ceiling prior -- rooms are 2 m to 30 m across and
    nothing in the physics narrows that -- but it is not flat, and that matters.
    It is what stops a large room read in the wrong unit from scoring well by
    handing its *width* to the ceiling prior: reading a 9.5 x 7.2 x 4.6 m hall
    as feet makes a very plausible 2.9 m ceiling, and the only thing wrong with
    that story is that the room is then 2.2 m by 1.4 m in plan, which is a
    cupboard, not a hall.
    """
    return _lognormal(span_m, _PLAN_CENTRE_M, _PLAN_LOGSIGMA) * _soft_window(
        span_m, _PLAN_MIN_M, _PLAN_MAX_M, width=0.25
    )


def diagonal_prior(diagonal_m: float) -> float:
    """Plausibility of the room's overall diagonal. A captured space is a couple
    of metres to a few tens of metres across -- never millimetres, never a
    kilometre."""
    return _lognormal(diagonal_m, _DIAG_CENTRE_M, _DIAG_LOGSIGMA) * _soft_window(
        diagonal_m, _DIAG_MIN_M, _DIAG_MAX_M, width=0.25
    )


def spacing_prior(spacing_m: float) -> float:
    """Plausibility of the median nearest-neighbour spacing.

    The signal that knows the difference between a room and a scale model of a
    room, and the only axis of evidence here that is independent of the room's
    shape: the sampler's resolution is a property of the hardware, not of the
    subject, so a candidate unit implying a fifth of a millimetre or half a metre
    between neighbouring points is refuted whatever the ceiling prior thinks. It
    is broad on purpose -- see `_SPACING_LOGSIGMA` -- because a sweep at 40 mm and
    a sweep at 8 mm are both perfectly ordinary and preferring one of them turns
    a scanner setting into an argument about units.
    """
    return _lognormal(spacing_m, _SPACING_CENTRE_M, _SPACING_LOGSIGMA) * _soft_window(
        spacing_m, _SPACING_MIN_M, _SPACING_MAX_M, width=_SPACING_EDGE
    )


# ---------------------------------------------------------------------------
# measurements taken in source units
# ---------------------------------------------------------------------------


def _subsample(xyz: np.ndarray, limit: int, seed: int = 0) -> np.ndarray:
    """Deterministic index subsample. Every measurement below is a statistic, so
    a few tens of thousands of points is as good as twenty million and the whole
    routine stays interactive on a large export."""
    n = len(xyz)
    if n <= limit:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=limit, replace=False))


def _parallel_plane_gaps(planes: list[Plane]) -> list[tuple[float, float, str]]:
    """Every facing-pair distance in the scan, weighted by how much surface it
    accounts for. Returns (gap, weight, label) triples.

    The floor-to-ceiling distance is one of these, but so is the distance
    between two facing walls, and which is which is not knowable before gravity
    is solved -- and gravity cannot be solved before the units are, because
    "horizontal" is a metric idea. Rather than commit to the heaviest pair and
    be silently wrong in a tall narrow room where a wall pair carries more
    surface than the floor, the caller is handed all of them and scores the
    candidate units against the mixture.
    """
    strong = [p for p in planes if _plane_weight(p) > 0]
    if len(strong) < 2:
        return []
    strong.sort(key=_plane_weight, reverse=True)
    strong = strong[:16]
    families: list[list[Plane]] = []
    for p in strong:
        for fam in families:
            if abs(float(np.dot(p.normal, fam[0].normal))) >= math.cos(math.radians(7.5)):
                fam.append(p)
                break
        else:
            families.append([p])
    out: list[tuple[float, float, str]] = []
    for i, fam in enumerate(families):
        if len(fam) < 2:
            continue
        a, b = fam[0], fam[1]
        c = float(np.dot(a.normal, b.normal))
        gap = abs(a.offset - math.copysign(1.0, c) * b.offset)
        if gap <= 0:
            continue
        out.append((gap, _plane_weight(a) + _plane_weight(b), f"plane pair {i}"))
    return out


def _plane_weight(plane: Plane) -> float:
    """Prefer measured area, fall back to inlier count.

    The plane detector is written by another module and may or may not populate
    `area`; a weighting scheme that silently reads 0.0 for every plane would
    make every plane equally important, which is exactly the failure that lets a
    cabinet door outvote a wall.
    """
    if plane.area and plane.area > 0:
        return float(plane.area)
    return float(plane.inlier_count)


def _axis_slab_gaps(
    xyz: np.ndarray, normals: np.ndarray
) -> list[tuple[float, float, str]]:
    """Candidate room dimensions taken straight from the points.

    In a room the sum of outer products of the surface normals has the three
    room axes as its eigenvectors: the floor and ceiling pile mass onto gravity,
    each pair of facing walls piles mass onto one horizontal axis. That gives us
    the axis triad but not which member of it is up, and no purely metric
    statistic can tell us before the units are known. So all three are returned,
    each weighted by how much of the cloud lies in its two slabs, and the caller
    scores every candidate unit against the whole mixture. A tall narrow room
    where the walls out-cover the floor then costs a little confidence instead
    of flipping the answer between centimetres and inches.
    """
    if len(xyz) < 64:
        return []
    moment = normals.T @ normals
    _, eigvecs = np.linalg.eigh(moment)
    proj = xyz @ eigvecs
    lo = np.quantile(proj, 0.001, axis=0)
    hi = np.quantile(proj, 0.999, axis=0)
    spans = hi - lo
    if not np.all(spans > 0):
        return []
    # one bin width for all three axes: per-axis binning would give the longest
    # axis the fattest bins and therefore the highest apparent slab mass, which
    # is a measurement artefact and not evidence about the room
    bin_width = float(spans.max()) / 400.0
    out: list[tuple[float, float, str]] = []
    for k in range(3):
        found = _slab_pair(proj[:, k], bin_width)
        if found is None:
            continue
        mass, gap = found
        out.append((gap, mass, f"normal-moment axis {k}"))
    return out


def _slab_pair(t: np.ndarray, bin_width: float) -> tuple[float, float] | None:
    """Two dominant parallel slabs along a 1D projection: (mass, separation).

    Histogram rather than clustering because the interesting structure is two
    spikes on a broad background of walls, and a spike detector is both cheaper
    and less likely to merge the ceiling into the top of a wall. Both spikes
    have to stand well clear of that background: a cloud that is not a built
    space at all -- a scanned object, a terrain patch, a blob of noise -- has no
    such spikes, and it is much better to report no floor-to-ceiling evidence
    than to invent a room dimension out of a smooth distribution.
    """
    lo = float(np.quantile(t, 0.001))
    hi = float(np.quantile(t, 0.999))
    span = hi - lo
    if span <= 0 or bin_width <= 0:
        return None
    bins = int(np.clip(round(span / bin_width), 16, 4000))
    counts, edges = np.histogram(t, bins=bins, range=(lo, hi))
    kernel = np.ones(3) / 3.0
    smooth = np.convolve(counts.astype(np.float64), kernel, mode="same")
    first = int(np.argmax(smooth))
    guard = max(1, int(0.15 * bins))
    masked = smooth.copy()
    masked[max(0, first - guard) : first + guard + 1] = -1.0
    second = int(np.argmax(masked))
    if masked[second] <= 0:
        return None
    occupied = smooth[smooth > 0]
    if len(occupied) == 0:
        return None
    background = float(np.median(occupied))
    if background > 0 and smooth[second] < _SLAB_PROMINENCE * background:
        return None
    centres = 0.5 * (edges[:-1] + edges[1:])
    gap = abs(float(centres[second] - centres[first]))
    if gap < 0.2 * span:
        return None
    return float(smooth[first] + smooth[second]), gap


def _mixture_weights(
    gaps: list[tuple[float, float, str]]
) -> list[tuple[float, float, str]]:
    """Turn raw slab masses into the probability that each pair is floor/ceiling.

    Sharpened, because the floor and the ceiling really do out-cover any one
    pair of facing walls in most rooms, and near-uniform weights would let the
    parts of the mixture we believe least drown out the part we believe most.
    """
    if not gaps:
        return []
    weights = [w ** _MASS_SHARPNESS for _, w, _ in gaps]
    total = float(sum(weights))
    if total <= 0:
        return []
    return [
        (gap, weight / total, label)
        for (gap, _, label), weight in zip(gaps, weights)
    ]


def _dimension_term(gaps: list[tuple[float, float, str]], factor: float) -> float:
    """How plausible the whole set of facing-surface separations is at one scale.

    Exactly one of the separations is the floor-to-ceiling distance and the rest
    are plan dimensions, but which one cannot be known before the units are, so
    every assignment is scored and the results are mixed in proportion to how
    much surface each pair accounts for. Scoring the assignment as a whole --
    ceiling prior on one, plan prior on the others -- is what makes the answer
    robust: a wrong unit can usually make *some* dimension look like a ceiling,
    but only the right one leaves the remaining dimensions looking like a room.
    """
    total = 0.0
    for i, (gap_i, weight, _) in enumerate(gaps):
        ceiling = gap_i * factor
        term = weight * ceiling_prior(ceiling)
        if term == 0.0:
            continue
        plans = [g * factor for j, (g, _, _) in enumerate(gaps) if j != i]
        for span in plans:
            term *= plan_prior(span)
        total += term * _aspect_factor(ceiling, plans)
    return total


def _aspect_factor(ceiling_m: float, plans_m: list[float]) -> float:
    """Penalty for an assignment that makes the room taller than it is wide.

    This is the single most useful piece of shape knowledge available before
    gravity is known, and it is what rescues large rooms from being read 2.54x
    too small. A 9.5 x 7.2 x 4.6 m hall measured in inches can be re-read as
    centimetres by calling the 7.2 m span the ceiling, which lands a very
    convincing 2.84 m -- and leaves the room 1.8 m wide with a 2.8 m ceiling.
    Rooms taller than they are wide exist (a stairwell, a lift lobby) so this is
    a soft penalty and not a veto, but it is a steep one.
    """
    if not plans_m or ceiling_m <= 0:
        return 1.0
    ratio = min(plans_m) / ceiling_m
    if ratio <= 0:
        return 0.0
    return 1.0 / (
        1.0 + math.exp(-(math.log(ratio) - math.log(_ASPECT_KNEE)) / _ASPECT_WIDTH)
    )


def _extent_diagonal(xyz: np.ndarray) -> float:
    """Diagonal of the oriented (PCA) bounding box, robustly clipped.

    The axis-aligned bounding box of a room that arrives rotated 41 degrees is
    much bigger than the room, which would drag the diagonal prior around for a
    reason that has nothing to do with units.
    """
    centred = xyz - xyz.mean(axis=0)
    _, _, basis = np.linalg.svd(centred, full_matrices=False)
    proj = centred @ basis.T
    lo = np.quantile(proj, 0.002, axis=0)
    hi = np.quantile(proj, 0.998, axis=0)
    return float(np.linalg.norm(hi - lo))


def _median_spacing(xyz: np.ndarray, seed: int = 0) -> float | None:
    """Median nearest-neighbour distance, corrected for subsampling.

    Building a KD-tree over twenty million points to answer a question about
    density is wasteful, but a random subsample has lower density than the
    original and would report a coarser scanner than we actually have. Scan
    points lie on surfaces, i.e. on a 2-manifold, so keeping a fraction f of
    them multiplies neighbour spacing by 1/sqrt(f); undoing that is exact enough
    for a prior that spans a factor of 20.
    """
    n = len(xyz)
    if n < 32:
        return None
    idx = _subsample(xyz, _NN_TREE_MAX, seed=seed)
    sample = xyz[idx]
    tree = cKDTree(sample)
    q_idx = _subsample(sample, _NN_QUERIES, seed=seed + 1)
    dist, _ = tree.query(sample[q_idx], k=2, workers=-1)
    d1 = dist[:, 1]
    d1 = d1[np.isfinite(d1) & (d1 > 0)]
    if len(d1) == 0:
        return None
    return float(np.median(d1)) * math.sqrt(len(sample) / n)


# ---------------------------------------------------------------------------
# the decision
# ---------------------------------------------------------------------------


def infer_unit_scale(
    cloud: PointCloud,
    *,
    planes: list[Plane] | None = None,
    hint: str | None = None,
    hint_confidence: float | None = None,
    hint_evidence: list[str] | None = None,
) -> ScaleEstimate:
    """Work out what to multiply source coordinates by to get metres.

    A unit written in the file header is believed outright: no prior beats a
    statement of fact, and a scan of a genuinely unusual space is exactly the
    case where the priors are weakest and the header is most needed. Without a
    header, every candidate unit is scored on the geometry it implies and the
    winner must clear an absolute plausibility floor as well as beat the others,
    so an input that is not a room at all comes back "unknown" rather than
    silently metres.

    `hint_confidence` exists for the one caller whose unit is certain but whose
    *scale* is not. A video reconstruction is converted to metres before it ever
    reaches here, so the unit is genuinely "m" and re-running the priors over it
    would be wrong -- but the conversion came from a monocular depth prior with
    a real error bar, and reporting that as the 1.0 of a declared header would
    launder an estimate into a measurement. The caller therefore supplies both
    the unit and how much to trust it.
    """
    evidence: list[str] = []

    if hint:
        key = _UNIT_ALIASES.get(str(hint).strip().lower())
        if key is not None:
            return ScaleEstimate(
                factor=CANDIDATE_UNITS[key],
                unit=key,
                confidence=1.0 if hint_confidence is None else float(hint_confidence),
                evidence=list(hint_evidence)
                if hint_evidence
                else [f"unit {key!r} declared by the source file header"],
            )
        evidence.append(f"ignored unrecognised unit hint {hint!r}")

    xyz = cloud.xyz
    if len(xyz) < 32:
        evidence.append(f"only {len(xyz)} points; nothing to measure")
        return ScaleEstimate(1.0, "unknown", 0.0, evidence)

    idx = _subsample(xyz, _SUBSAMPLE, seed=0)
    sample = xyz[idx]

    # 1. the strong signal: the facing-surface separations, in source units
    gaps: list[tuple[float, float, str]] = []
    if planes:
        gaps = _parallel_plane_gaps(planes)
    if not gaps:
        normals = cloud.normals
        if normals is not None:
            normals = normals[idx]
        else:
            normals = estimate_normals(PointCloud(sample))
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.where(norms == 0, 1.0, norms)
        gaps = _axis_slab_gaps(sample, normals)
    gaps = _mixture_weights(sorted(gaps, key=lambda t: -t[1])[:3])
    if not gaps:
        gaps = []
        evidence.append(
            "no pair of facing surfaces found; falling back to extent and "
            "spacing only, which cannot separate a room from a scale model of it"
        )

    # 2. the corroborators
    diagonal = _extent_diagonal(sample)
    spacing = _median_spacing(xyz, seed=0)

    scores: dict[str, float] = {}
    detail: dict[str, tuple[float, float, float]] = {}
    for unit, factor in CANDIDATE_UNITS.items():
        p_ceiling = _dimension_term(gaps, factor) if gaps else 1.0
        p_diag = diagonal_prior(diagonal * factor)
        p_space = spacing_prior(spacing * factor) if spacing is not None else 1.0
        detail[unit] = (p_ceiling, p_diag, p_space)
        w_ceiling = _W_CEILING if gaps else 0.0
        w_space = _W_SPACING if spacing is not None else 0.0
        total_w = w_ceiling + _W_DIAG + w_space
        log_score = (
            w_ceiling * math.log(max(p_ceiling, 1e-300))
            + _W_DIAG * math.log(max(p_diag, 1e-300))
            + w_space * math.log(max(p_space, 1e-300))
        )
        scores[unit] = log_score / max(total_w, 1e-9)

    best_unit = max(scores, key=lambda u: scores[u])
    plausibility = math.exp(scores[best_unit])
    ranked = sorted(scores.values(), reverse=True)
    share = 1.0
    if len(ranked) > 1:
        # softmax over the weighted-geometric-mean log scores; the runner-up is
        # what decides whether this was a close call, not the absolute value
        rel = np.array([s - ranked[0] for s in scores.values()])
        share = float(1.0 / np.exp(rel).sum())

    for g, w, label in sorted(gaps, key=lambda t: -t[1]):
        evidence.append(
            f"facing surfaces {g:.4g} source units apart ({label}, weight {w:.2f})"
        )
    evidence.append(f"oriented-bbox diagonal {diagonal:.4g} source units")
    if spacing is not None:
        evidence.append(f"median point spacing {spacing:.4g} source units")
    else:
        evidence.append("point spacing unavailable")

    if plausibility < _PLAUSIBILITY_FLOOR:
        evidence.append(
            f"best candidate {best_unit!r} scores {plausibility:.3g}, below the "
            f"{_PLAUSIBILITY_FLOOR} plausibility floor -- refusing to guess a unit"
        )
        for unit in CANDIDATE_UNITS:
            evidence.append(_explain(unit, gaps, diagonal, spacing, detail[unit]))
        return ScaleEstimate(1.0, "unknown", 0.0, evidence)

    confidence = share * float(np.clip(plausibility / _PLAUSIBILITY_FULL, 0.0, 1.0))
    if not gaps:
        # without a floor/ceiling pair the two remaining priors are broad enough
        # that several units stay arguable; saying so is the whole job
        confidence = min(confidence, _NO_GAP_MAX_CONFIDENCE)
    evidence.append(_explain(best_unit, gaps, diagonal, spacing, detail[best_unit]))
    runner = sorted(scores, key=lambda u: -scores[u])[1] if len(scores) > 1 else None
    if runner is not None:
        margin = scores[best_unit] - scores[runner]
        evidence.append(
            f"chose {best_unit!r} (x{CANDIDATE_UNITS[best_unit]:g}) over {runner!r} "
            f"by a log-odds margin of {margin:.1f}"
        )
    return ScaleEstimate(
        factor=CANDIDATE_UNITS[best_unit],
        unit=best_unit,
        confidence=confidence,
        evidence=evidence,
    )


def _explain(
    unit: str,
    gaps: list[tuple[float, float, str]],
    diagonal: float,
    spacing: float | None,
    priors: tuple[float, float, float],
) -> str:
    """One line per candidate saying what the room would be if it were true."""
    factor = CANDIDATE_UNITS[unit]
    parts = []
    if gaps:
        spans = ", ".join(f"{g * factor:.2f}" for g, _, _ in sorted(gaps, key=lambda t: -t[1]))
        parts.append(f"facing surfaces [{spans}] m (p={priors[0]:.3g})")
    parts.append(f"diagonal {diagonal * factor:.3f} m (p={priors[1]:.3g})")
    if spacing is not None:
        parts.append(f"spacing {spacing * factor * 1000:.1f} mm (p={priors[2]:.3g})")
    return f"as {unit}: " + ", ".join(parts)
