"""What the grip truck actually holds, with the dimensions it actually has.

A scout report is only useful if "the dolly fits" means a specific dolly with a
specific width against a measured doorway. So this is a catalogue of real gear
with real numbers, and every entry carries where its numbers came from.

The `verified` flag is load-bearing rather than decorative. Manufacturers
publish these figures in scanned PDFs and rental houses paraphrase them, so
some dimensions here are confirmed against a published source and some are
representative figures for the class of item. Anything unverified must be
reported as an estimate -- a scout who drives out because a twin said the dolly
fits, and finds it does not, has been failed by exactly this distinction.
"""

from __future__ import annotations

from dataclasses import dataclass

INCH = 0.0254
FOOT = 0.3048
POUND = 0.45359237


@dataclass(frozen=True)
class Gear:
    """One piece of equipment, sized in metres and kilograms.

    `footprint_m` is (width, length) of the space it stands in, `height_m` its
    parked height. `clearance_m` is the working space it needs *around* that
    footprint for crew to operate it -- a dolly you cannot walk beside is a
    dolly you cannot push.
    """

    key: str
    name: str
    kind: str                      # dolly | track | support | lighting | crane
    footprint_m: tuple[float, float]
    height_m: float
    weight_kg: float = 0.0
    load_kg: float = 0.0
    min_lens_height_m: float | None = None
    max_lens_height_m: float | None = None
    clearance_m: float = 0.6
    power_w: float = 0.0
    verified: bool = False
    source: str = ""

    @property
    def area_m2(self) -> float:
        return float(self.footprint_m[0] * self.footprint_m[1])

    @property
    def swept_footprint_m(self) -> tuple[float, float]:
        """Footprint plus the working clearance, which is what must actually fit."""
        w, l = self.footprint_m
        return (w + 2 * self.clearance_m, l + 2 * self.clearance_m)


CATALOGUE: dict[str, Gear] = {
    # -- dollies ----------------------------------------------------------
    "super-peewee": Gear(
        key="super-peewee",
        name="Chapman Super PeeWee",
        kind="dolly",
        footprint_m=(27 * INCH, 43 * INCH),
        height_m=36 * INCH,
        clearance_m=0.6,
        verified=True,
        source="Chapman/Leonard published dimensions: L 43 in x W 27 in x H 36 in",
    ),
    "fisher-11": Gear(
        key="fisher-11",
        name="J.L. Fisher Model 11",
        kind="dolly",
        # Width is published; length is a representative figure for a dolly of
        # this class and is NOT from the spec sheet, which is why this entry is
        # unverified despite the width being solid.
        footprint_m=(21 * INCH, 48 * INCH),
        height_m=0.95,
        load_kg=900 * POUND,
        clearance_m=0.6,
        verified=False,
        source=(
            "J.L. Fisher: minimum width just under 21 in, load capacity 900 lb. "
            "Length and height are class-typical estimates, not published figures."
        ),
    ),
    "doorway-dolly": Gear(
        key="doorway-dolly",
        name="Doorway dolly",
        kind="dolly",
        footprint_m=(0.61, 1.22),
        height_m=0.35,
        clearance_m=0.5,
        verified=False,
        source="Class-typical: built to pass a standard 2 ft 6 in doorway.",
    ),
    # -- track ------------------------------------------------------------
    "track-24.5": Gear(
        key="track-24.5",
        name='Tubular dolly track, 24.5 in gauge',
        kind="track",
        footprint_m=(24.5 * INCH, 2.44),
        height_m=0.10,
        clearance_m=0.5,
        verified=True,
        source="Chapman: Super PeeWee runs on 24 1/2 in tubular track.",
    ),
    "track-880": Gear(
        key="track-880",
        name="Tubular dolly track, 880 mm gauge",
        kind="track",
        footprint_m=(0.88, 2.44),
        height_m=0.10,
        clearance_m=0.5,
        verified=True,
        source="Chapman: Super PeeWee also runs on 880 mm track.",
    ),
    # -- camera support ---------------------------------------------------
    "sticks-standard": Gear(
        key="sticks-standard",
        name="Standard tripod (Mitchell, spread legs)",
        kind="support",
        footprint_m=(1.20, 1.20),
        height_m=1.50,
        min_lens_height_m=0.70,
        max_lens_height_m=1.75,
        clearance_m=0.4,
        verified=False,
        source="Class-typical spread for a fluid-head tripod at working height.",
    ),
    "baby-legs": Gear(
        key="baby-legs",
        name="Baby legs",
        kind="support",
        footprint_m=(0.90, 0.90),
        height_m=0.60,
        min_lens_height_m=0.30,
        max_lens_height_m=0.75,
        clearance_m=0.3,
        verified=False,
        source="Class-typical.",
    ),
    "handheld": Gear(
        key="handheld",
        name="Handheld / shoulder",
        kind="support",
        footprint_m=(0.60, 0.60),
        height_m=1.70,
        min_lens_height_m=0.40,
        max_lens_height_m=1.90,
        clearance_m=0.3,
        verified=True,
        source="An operator's own footprint; a person occupies about this.",
    ),
    # -- lighting ---------------------------------------------------------
    "led-1x1": Gear(
        key="led-1x1",
        name="1x1 LED panel on a stand",
        kind="lighting",
        footprint_m=(1.00, 1.00),
        height_m=2.00,
        power_w=100.0,
        clearance_m=0.3,
        verified=False,
        source="Class-typical: 1x1 panel on a standard combo stand.",
    ),
    "m18": Gear(
        key="m18",
        name="1.8 kW HMI on a combo stand",
        kind="lighting",
        footprint_m=(1.20, 1.20),
        height_m=2.40,
        power_w=1800.0,
        clearance_m=0.5,
        verified=False,
        source="Class-typical footprint and draw for an M18-size HMI.",
    ),
}


def get(key: str) -> Gear:
    if key not in CATALOGUE:
        raise KeyError(f"unknown equipment {key!r}; have {sorted(CATALOGUE)}")
    return CATALOGUE[key]


def of_kind(kind: str) -> list[Gear]:
    return [g for g in CATALOGUE.values() if g.kind == kind]


def unverified() -> list[Gear]:
    """Everything whose dimensions are class-typical rather than published.

    A report that quotes these has to say so, which means something has to be
    able to ask.
    """
    return [g for g in CATALOGUE.values() if not g.verified]
