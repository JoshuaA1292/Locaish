"""Will the room ring, and can dialogue be recorded in it.

The sound recordist goes on the recce to answer one question that no photograph
can: how long a handclap hangs in the air. A room that looks perfect and rings
for a second and a half will cost a day of ADR, and the number that predicts it
is reverberation time, which follows from the room's volume and how absorbent
its surfaces are.

Sabine's equation is the standard estimate: `RT60 = 0.161 V / A`, where V is the
volume in cubic metres and A is the total absorption, the sum of every surface's
area times its absorption coefficient. It is an approximation -- it assumes a
diffuse field and it overestimates in very dead rooms, where Eyring's variant is
better -- and for the purpose of "is this location a problem" it is entirely
adequate.

**The twin cannot know what the surfaces are made of.** It measures geometry,
not material, and a plastered wall and an upholstered one are the same shape.
Absorption differs between them by a factor of ten, so reporting a single RT60
would be inventing the answer. Instead this reports a *range* across plausible
finishes, and the useful output is usually the shape of that range: a room that
is fine even at its most reflective needs no thought, one that is a problem even
at its softest needs a different location, and one that straddles is a question
to ask the location owner.

Coefficients are at 500 Hz, the middle of the speech band, and are the
conventional textbook figures for each class of surface rather than a
measurement of this room.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..types import Twin

SABINE_CONSTANT = 0.161          # metric form, seconds per metre

# Speech intelligibility thresholds. ANSI S12.60 puts the limit for learning
# spaces at 0.6 s; word recognition falls below about 85% past 1.0 s. Production
# sound is stricter than either, because a boom cannot un-ring a room.
RT60_CLEAN_DIALOGUE_S = 0.45
RT60_WORKABLE_S = 0.70
RT60_PROBLEM_S = 1.00


@dataclass(frozen=True)
class Surface:
    """An absorption coefficient at 500 Hz for one class of finish."""

    name: str
    alpha: float


# Ordered soft to hard within each family. Textbook 500 Hz values.
FLOORS = (
    Surface("heavy carpet on underlay", 0.37),
    Surface("carpet on concrete", 0.14),
    Surface("wood or vinyl on joists", 0.07),
    Surface("concrete, tile or stone", 0.02),
)
WALLS = (
    Surface("heavy drapes over the walls", 0.55),
    Surface("bookshelves and soft furnishing", 0.20),
    Surface("plasterboard on studs", 0.05),
    Surface("plaster on masonry, or glass", 0.02),
)
CEILINGS = (
    Surface("acoustic tile", 0.70),
    Surface("plasterboard", 0.05),
    Surface("plaster on concrete", 0.02),
)


@dataclass
class Acoustics:
    """A reverberation estimate for a room whose materials are unknown."""

    volume_m3: float
    floor_area_m2: float
    ceiling_area_m2: float
    wall_area_m2: float
    opening_area_m2: float
    rt60_softest_s: float
    rt60_hardest_s: float
    rt60_typical_s: float
    open_to_outside: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """What a sound recordist would say, judged on the hard end.

        The hard end rather than the typical one because that is the risk: a
        room assumed carpeted and found tiled is a problem discovered on the
        day, and the whole point of a recce is to not discover things then.
        """
        if self.rt60_hardest_s <= RT60_CLEAN_DIALOGUE_S:
            return "clean"
        if self.rt60_softest_s >= RT60_PROBLEM_S:
            return "problem"
        if self.rt60_softest_s >= RT60_WORKABLE_S:
            return "difficult"
        return "depends on the finishes"

    def to_dict(self) -> dict:
        return {
            "volume_m3": round(self.volume_m3, 1),
            "rt60_s": {
                "softest": round(self.rt60_softest_s, 2),
                "typical": round(self.rt60_typical_s, 2),
                "hardest": round(self.rt60_hardest_s, 2),
            },
            "verdict": self.verdict,
            "surfaces_m2": {
                "floor": round(self.floor_area_m2, 1),
                "ceiling": round(self.ceiling_area_m2, 1),
                "wall": round(self.wall_area_m2, 1),
                "openings": round(self.opening_area_m2, 1),
            },
            "warnings": self.warnings,
        }


def _rt60(volume_m3: float, absorption: float) -> float:
    if absorption <= 0:
        return float("inf")
    return float(SABINE_CONSTANT * volume_m3 / absorption)


def estimate(twin: Twin) -> Acoustics:
    """Reverberation range for a twin, across plausible finishes.

    Openings are counted as fully absorbent -- an open window returns nothing --
    which is right for a door standing open and pessimistic for a glazed one.
    Since glass is nearly as reflective as plaster, a room whose openings are
    all glazed sits near the hard end of the range anyway.
    """
    structure = twin.structure
    warnings: list[str] = []

    height = structure.ceiling_height
    if height is None or height <= 0:
        span = float(np.percentile(twin.points.xyz[:, 2], 99.5) - structure.floor_z)
        height = max(span, 0.1)
        warnings.append(
            f"no ceiling was captured, so the volume assumes the room stops at "
            f"{height:.2f} m, the top of the points; every figure below scales "
            "directly with that guess"
        )

    floor_area = float(structure.floor_area)
    if floor_area <= 0:
        raise ValueError("twin has no floor area to work from")

    perimeter = _perimeter(structure.footprint)
    wall_area = float(perimeter * height)
    opening_area = float(sum(o.area for o in structure.openings))
    if opening_area > 0.6 * wall_area:
        opening_area = 0.6 * wall_area
        warnings.append(
            "detected openings exceeded half the wall area, which is more likely "
            "a detection error than a room made mostly of doorway; the figure was "
            "capped"
        )
    wall_solid = max(wall_area - opening_area, 0.0)
    volume = floor_area * height

    def total(floor: Surface, wall: Surface, ceiling: Surface) -> float:
        # An opening absorbs everything that reaches it: alpha = 1.
        return (
            floor_area * floor.alpha
            + wall_solid * wall.alpha
            + floor_area * ceiling.alpha
            + opening_area * 1.0
        )

    softest = _rt60(volume, total(FLOORS[0], WALLS[0], CEILINGS[0]))
    hardest = _rt60(volume, total(FLOORS[-1], WALLS[-1], CEILINGS[-1]))
    typical = _rt60(volume, total(FLOORS[1], WALLS[2], CEILINGS[1]))

    return Acoustics(
        volume_m3=volume,
        floor_area_m2=floor_area,
        ceiling_area_m2=floor_area,
        wall_area_m2=wall_area,
        opening_area_m2=opening_area,
        rt60_softest_s=softest,
        rt60_hardest_s=hardest,
        rt60_typical_s=typical,
        open_to_outside=structure.ceiling_z is None,
        warnings=warnings,
    )


def _perimeter(footprint) -> float:
    if footprint is None or len(footprint) < 3:
        return 0.0
    p = np.asarray(footprint, dtype=np.float64)
    q = np.roll(p, -1, axis=0)
    return float(np.linalg.norm(q - p, axis=1).sum())
