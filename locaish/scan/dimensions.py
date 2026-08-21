"""Measuring a room the way a person with a tape measure would.

There is a tempting wrong answer here: take the point cloud's bounding box and
call its extents the room's dimensions. It is wrong because a bounding box is
decided by the two most extreme points in the cloud, and the most extreme point
is by construction the worst outlier. With 6 mm of range noise and a hundred
thousand points per surface, the furthest sample sits about four standard
deviations out, so a 5.200 m room measures 5.246 m -- a 46 mm error that grows
with point count and never shrinks with better averaging.

The right answer is that a wall's position is the plane fitted to it, which
averages the noise away, and the room's width is the perpendicular distance
between two opposite fitted walls. That is also what the number means: nobody
measuring a room measures to the tip of the roughest bit of plaster.

Where opposite walls cannot be paired -- a round room, a single wall scanned, a
corridor open at one end -- we fall back to a high-percentile extent and say so,
because a dimension derived from a percentile is an estimate and must not be
reported with the same confidence as one derived from two planes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import PointCloud, Structure


#: A vertical plane must carry at least this share of the best wall's inliers
#: before it is allowed to define one of the room's spans. Set from observation:
#: real walls in a scanned room come back with 40-60% of the largest plane's
#: support, while furniture faces sit at 1-5%, so the gap is wide and the exact
#: cut is not delicate.
MIN_WALL_SUPPORT = 0.15


@dataclass
class Dimension:
    """One measured span, with how it was obtained.

    `method` is "planes" when it came from two fitted opposite surfaces and
    "percentile" when it is an extent estimate. The distinction is the whole
    point of this module, so it travels with the number rather than being
    dropped at the call site.
    """

    length: float
    method: str
    axis: str
    uncertainty: float = 0.0

    def __str__(self) -> str:
        return f"{self.length:.3f} m ({self.method})"


@dataclass
class RoomDimensions:
    x: Dimension
    y: Dimension
    z: Dimension

    @property
    def long_side(self) -> float:
        return max(self.x.length, self.y.length)

    @property
    def short_side(self) -> float:
        return min(self.x.length, self.y.length)

    @property
    def plan(self) -> tuple[float, float]:
        """The two horizontal spans, longest first. Ordering them removes the
        dependence on which wall the canonicaliser happened to call +X, so two
        twins of the same room compare equal even if one was scanned starting
        from a different corner."""
        return (self.long_side, self.short_side)

    def as_dict(self) -> dict[str, object]:
        return {
            axis: {"length_m": d.length, "method": d.method, "uncertainty_m": d.uncertainty}
            for axis, d in (("x", self.x), ("y", self.y), ("z", self.z))
        }


def _percentile_span(values: np.ndarray, low: float = 0.2, high: float = 99.8) -> tuple[float, float]:
    """Trimmed extent plus the width of the trim, used as an uncertainty.

    The trim discards the sensor's tail without discarding real geometry: at
    0.2% of a hundred thousand points that is two hundred samples per end,
    which a genuine wall corner will always have and a noise spike never does.
    """
    if len(values) == 0:
        return 0.0, 0.0
    lo, hi = np.percentile(values, [low, high])
    raw = float(values.max() - values.min())
    return float(hi - lo), max(0.0, raw - float(hi - lo)) / 2.0


def _opposite_pairs(structure: Structure, axis: int) -> list[tuple[float, float]]:
    """Distances between wall planes whose normals oppose along `axis`.

    Plane normals point into the room, so two facing walls have opposite
    normals and their perpendicular separation is -(o1 + o2). Deriving it from
    the offsets rather than from any point avoids reintroducing the outlier
    problem this module exists to solve.
    """
    walls = [p for p in structure.planes if p.kind == "wall"]
    if not walls:
        return []

    # A room's walls are the large planes. Plane detection also returns the
    # front of a wardrobe and the side of a desk, correctly labelled walls
    # because they are vertical, and one of those was observed sitting nearly
    # five degrees off axis with 3% of the support of a real wall. Measuring
    # the room between two pieces of furniture is a silent, plausible, wrong
    # answer, so anything without real support is not eligible to define a span.
    # Only consider walls that genuinely face along this axis; a wall 30 degrees
    # off would give a foreshortened separation.
    facing = [p for p in walls if abs(p.normal[axis]) >= 0.94]
    if not facing:
        return []

    # Support is judged against the other walls facing THIS axis, not against
    # the largest wall in the room. In an 11 x 2.2 m corridor the end walls
    # carry a fifth of the side walls' points purely because the room is long,
    # and measuring them against the side walls discards the only two surfaces
    # that define the corridor's length -- which sent the measurement back to a
    # percentile of the point cloud and cost 39 mm. Within one axis the
    # comparison is fair, because a real wall and a wardrobe standing against it
    # face the same way and differ only in how much of the room they are.
    strongest = max(p.inlier_count for p in facing)
    supported = [
        p for p in facing if p.inlier_count >= max(MIN_WALL_SUPPORT * strongest, 200)
    ] or facing

    positive, negative = [], []
    for p in supported:
        (positive if p.normal[axis] > 0 else negative).append(p)

    pairs = []
    for a in positive:
        for b in negative:
            # both facing walls have offset -h for a room of half-width h, so
            # their sum is the negated span regardless of which is which
            separation = abs(a.offset + b.offset)
            if separation > 0.5:
                weight = min(a.inlier_count, b.inlier_count)
                pairs.append((separation, float(weight)))
    return pairs


def measure_room(
    points: PointCloud, structure: Structure, *, prefer_planes: bool = True
) -> RoomDimensions:
    """Measure a canonical twin's X, Y and Z spans.

    Z comes from the floor and ceiling planes when both exist, since those are
    the two surfaces most reliably fitted in any interior scan. X and Y come
    from opposite wall pairs where available. Anything unpaired falls back to a
    trimmed extent and is labelled as such.
    """
    dims = []
    for axis, name in ((0, "x"), (1, "y")):
        chosen = None
        if prefer_planes:
            pairs = _opposite_pairs(structure, axis)
            if pairs:
                # the widest well-supported pair is the room; a narrower pair is
                # usually an alcove or a partition standing inside it
                best = max(pairs, key=lambda sw: (sw[0], sw[1]))
                chosen = Dimension(best[0], "planes", name, uncertainty=0.005)
        if chosen is None:
            span, unc = _percentile_span(points.xyz[:, axis])
            chosen = Dimension(span, "percentile", name, uncertainty=unc)
        dims.append(chosen)

    if structure.ceiling_z is not None:
        z = Dimension(
            float(structure.ceiling_z - structure.floor_z), "planes", "z", uncertainty=0.005
        )
    else:
        span, unc = _percentile_span(points.xyz[:, 2])
        z = Dimension(span, "percentile", "z", uncertainty=unc)

    return RoomDimensions(x=dims[0], y=dims[1], z=z)
