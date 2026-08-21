"""What a lens sees, and how big a person is in the frame when it sees them.

Everything a director asks for on a recce is a statement about framing -- "can
we get a clean two-shot from the corner", "do we have room for a wide" -- and
every one of those reduces to the same piece of similar-triangle geometry: a
subject of height `h` at distance `d` projects onto the sensor at `h * f / d`.
Turn that around and the whole scout question becomes answerable, because the
room fixes `d` and the shot fixes what fraction of the frame `h` has to fill.

The convention used throughout is **framed height**: the vertical extent of the
world, at the subject's distance, that exactly fills the frame. It is the
honest quantity to compute with, because it is a length in metres that can be
compared against a room, and every named shot size is just a band of it. A
"medium shot" is not a focal length or a distance -- it is a statement that
about 1.2 m of standing human fills the frame, and it stays that statement on
any sensor with any lens.

Sensor dimensions are the *recorded* area, which is what determines the field
of view. Numbers here are the manufacturers' published active areas for formats
that are effectively standards; anything approximate is flagged as such, because
a focal length recommendation that is quietly 10% off is worse than no
recommendation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# sensors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sensor:
    """A recording format's active area in millimetres."""

    name: str
    width_mm: float
    height_mm: float
    circle_of_confusion_mm: float
    note: str = ""

    @property
    def aspect(self) -> float:
        return self.width_mm / self.height_mm

    @property
    def diagonal_mm(self) -> float:
        return math.hypot(self.width_mm, self.height_mm)


# The circle of confusion is the diameter of the largest blur spot still read as
# a point. There is no single correct value -- it depends on how big the image
# is shown and how close the viewer sits -- so these are the conventional
# figures used in cinema depth-of-field tables: roughly the frame diagonal over
# 1500, which is stricter than the stills convention of d/1730 because a cinema
# screen is examined harder than a print.
SENSORS: dict[str, Sensor] = {
    "super35": Sensor(
        "Super 35 (3-perf, 16:9)", 24.89, 14.00, 0.020,
        "the common digital-cinema Super 35 recording area",
    ),
    "super35-4perf": Sensor(
        "Super 35 (4-perf, 1.33)", 24.89, 18.66, 0.025,
        "full-height Super 35, as on 4-perf film",
    ),
    "fullframe": Sensor(
        "Full frame (VistaVision-ish)", 36.00, 24.00, 0.029,
        "the 36x24 mm stills format, now common in cinema cameras",
    ),
    "m43": Sensor(
        "Micro Four Thirds", 17.30, 13.00, 0.015,
        "",
    ),
    "phone-main": Sensor(
        "Phone main camera (approximate)", 9.80, 7.35, 0.008,
        "APPROXIMATE. Recent flagship main sensors are near this; a phone's "
        "recorded area varies by mode and by stabilisation crop, so treat any "
        "focal length derived from it as indicative rather than as a spec.",
    ),
}

DEFAULT_SENSOR = "super35"


# ---------------------------------------------------------------------------
# shot sizes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShotSize:
    """A named framing, expressed as how much of the world fills the frame.

    `framed_height_m` is the vertical extent at the subject that exactly fills
    the frame -- so a close-up is not "50mm at two metres", it is "about 0.6 m
    of person fills the picture", which is true on every camera ever made.
    """

    key: str
    name: str
    framed_height_m: float
    covers: str


# Framed heights for a standing adult of about 1.75 m. These are the ordinary
# working definitions of the shot names -- what part of the body the frame
# holds -- rather than anything measured; a "medium shot" is a convention, and
# different crews cut it a few centimetres differently.
SHOT_SIZES: tuple[ShotSize, ...] = (
    ShotSize("ecu", "Extreme close-up", 0.20, "eyes and mouth"),
    ShotSize("bcu", "Big close-up", 0.35, "the whole face"),
    ShotSize("cu", "Close-up", 0.60, "head and shoulders"),
    ShotSize("mcu", "Medium close-up", 0.90, "chest up"),
    ShotSize("ms", "Medium shot", 1.20, "waist up"),
    ShotSize("mls", "Medium long shot", 1.55, "knees up, the cowboy"),
    ShotSize("ls", "Long shot", 2.10, "the whole figure with headroom"),
    ShotSize("els", "Extreme long shot", 4.50, "the figure small in its setting"),
)

SHOT_BY_KEY = {s.key: s for s in SHOT_SIZES}


def classify_framing(framed_height_m: float) -> ShotSize:
    """Name the shot whose framing is closest, in log space.

    Log space because framing is multiplicative: 0.6 m and 0.9 m are as far
    apart perceptually as 2.1 m and 3.2 m, and a linear nearest-neighbour would
    call almost everything an extreme long shot.
    """
    h = max(float(framed_height_m), 1e-6)
    return min(SHOT_SIZES, key=lambda s: abs(math.log(h / s.framed_height_m)))


# ---------------------------------------------------------------------------
# the geometry
# ---------------------------------------------------------------------------


def field_of_view_deg(sensor: Sensor, focal_mm: float) -> tuple[float, float]:
    """(horizontal, vertical) angular field of view in degrees."""
    if focal_mm <= 0:
        raise ValueError("focal length must be positive")
    h = 2.0 * math.degrees(math.atan(sensor.width_mm / (2.0 * focal_mm)))
    v = 2.0 * math.degrees(math.atan(sensor.height_mm / (2.0 * focal_mm)))
    return h, v


def framed_height_m(sensor: Sensor, focal_mm: float, distance_m: float) -> float:
    """How much vertical world the frame holds at `distance_m`.

    The whole scout calculation in one line. Note it is linear in distance and
    inverse in focal length, which is why a small room is a wide-lens problem
    and no amount of stepping back fixes it once the wall is behind you.
    """
    if focal_mm <= 0 or distance_m <= 0:
        raise ValueError("focal length and distance must be positive")
    return float(distance_m * sensor.height_mm / focal_mm)


def framed_width_m(sensor: Sensor, focal_mm: float, distance_m: float) -> float:
    if focal_mm <= 0 or distance_m <= 0:
        raise ValueError("focal length and distance must be positive")
    return float(distance_m * sensor.width_mm / focal_mm)


def distance_for_framing(sensor: Sensor, focal_mm: float, framed_height_m: float) -> float:
    """How far back the camera must be to frame that much world."""
    if focal_mm <= 0 or framed_height_m <= 0:
        raise ValueError("focal length and framing must be positive")
    return float(framed_height_m * focal_mm / sensor.height_mm)


def focal_for_framing(sensor: Sensor, distance_m: float, framed_height_m: float) -> float:
    """The lens that frames that much world from where the camera can actually stand.

    This is the question a room answers and a storyboard does not: the wall is
    where it is, so the distance is given and the lens is the free variable.
    """
    if distance_m <= 0 or framed_height_m <= 0:
        raise ValueError("distance and framing must be positive")
    return float(sensor.height_mm * distance_m / framed_height_m)


def depth_of_field(
    sensor: Sensor, focal_mm: float, aperture_f: float, distance_m: float
) -> tuple[float, float, float]:
    """(near, far, hyperfocal) in metres; `far` is inf beyond the hyperfocal.

    Standard thin-lens depth of field. It matters on a recce because a room can
    be geometrically fine and optically impossible: a wide lens close to an
    actor in a narrow hallway will hold the far wall in focus whether the
    director wants it or not.
    """
    if focal_mm <= 0 or aperture_f <= 0 or distance_m <= 0:
        raise ValueError("focal length, aperture and distance must be positive")
    f = focal_mm
    c = sensor.circle_of_confusion_mm
    hyperfocal_mm = (f * f) / (aperture_f * c) + f
    d = distance_m * 1000.0

    near_mm = d * (hyperfocal_mm - f) / (hyperfocal_mm + d - 2.0 * f)
    if d >= hyperfocal_mm:
        far_mm = float("inf")
    else:
        far_mm = d * (hyperfocal_mm - f) / (hyperfocal_mm - d)
    return near_mm / 1000.0, far_mm / 1000.0, hyperfocal_mm / 1000.0


def subject_height_in_frame(
    sensor: Sensor, focal_mm: float, distance_m: float, subject_height_m: float
) -> float:
    """Fraction of frame height a subject occupies. 1.0 exactly fills it."""
    return float(subject_height_m / framed_height_m(sensor, focal_mm, distance_m))


# Focal lengths a camera department actually carries. Used when recommending a
# lens for a shot: proposing 43.7 mm is not useful to anyone, and the answer a
# scout wants is which of the primes in the case will do the job.
PRIME_SET_MM: tuple[float, ...] = (12, 14, 16, 18, 21, 25, 27, 32, 35, 40, 50, 65, 75, 100, 135)


def nearest_prime(focal_mm: float, primes: tuple[float, ...] = PRIME_SET_MM) -> float:
    """The prime closest in log space -- lenses are spaced multiplicatively."""
    f = max(float(focal_mm), 1e-6)
    return min(primes, key=lambda p: abs(math.log(f / p)))
