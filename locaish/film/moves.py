"""Simulating a camera move, and the people it is pointed at, before anyone drives out.

A shot is not a position. It is a camera going somewhere while pointed at
someone who is also going somewhere, and almost everything that kills a shot on
the day is a fact about that whole span rather than about either end of it: the
dolly clears the sofa at the start and catches it at the third metre, the actor
is clean until the doorframe crosses them, the lens holds the two-shot until
the move brings the far wall into the back of frame.

So the unit here is a *move* sampled into beats, and every beat carries what a
scout would have written on a clipboard: where the camera is, whether it fits
there, whether the floor under it is level enough to lay track, what the lens is
holding, how big the subject is in frame, and whether anything is between the
two. A move is feasible when every beat is, and when it is not the report says
which beat and why -- which is the difference between "that shot won't work" and
"the shot works until 2.4 m along, where the track leaves the levelled floor".

Nothing here is a rendering. It is geometry against the twin, so every number
traces to a measurement or to a stated assumption, and a shot that this passes
is one whose *spatial* claims hold. Whether it is a good shot is a different
question and belongs to a human.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import optics
from .equipment import Gear
from .space import FloorMaps, visible

# Dolly track is levelled on wedges, and a floor that wanders more than this
# under the run needs cribbing -- which is time, and which the grip department
# would rather hear about on the recce than on the day.
TRACK_LEVEL_TOLERANCE_M = 0.02

# How finely a move is sampled. Twenty centimetres is well inside the scale of
# anything that goes wrong: furniture, doorways and the width of a dolly are all
# larger, so nothing gets stepped over.
BEAT_SPACING_M = 0.20
MIN_BEATS = 3


@dataclass
class Camera:
    """A camera somewhere, pointed at something, on a particular lens."""

    position: np.ndarray
    aim: np.ndarray
    focal_mm: float = 32.0
    aperture_f: float = 2.8
    sensor: str = optics.DEFAULT_SENSOR

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.aim = np.asarray(self.aim, dtype=np.float64).reshape(3)

    @property
    def spec(self) -> optics.Sensor:
        return optics.SENSORS[self.sensor]

    @property
    def forward(self) -> np.ndarray:
        d = self.aim - self.position
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])

    @property
    def pan_deg(self) -> float:
        """Bearing of the lens axis about +Z, degrees, 0 along +X."""
        f = self.forward
        return float(math.degrees(math.atan2(f[1], f[0])))

    @property
    def tilt_deg(self) -> float:
        """Elevation of the lens axis, degrees, positive upward."""
        f = self.forward
        return float(math.degrees(math.asin(np.clip(f[2], -1.0, 1.0))))

    def frames(self, point) -> tuple[bool, float, float]:
        """Whether a world point is in shot, and where. Returns (inside, u, v).

        `u` and `v` run -1 to 1 across the frame, so the centre is (0, 0) and
        the corners are the unit square. Points behind the camera come back
        outside regardless of where they project.
        """
        p = np.asarray(point, dtype=np.float64).reshape(3)
        rel = p - self.position
        fwd = self.forward
        depth = float(np.dot(rel, fwd))
        if depth <= 1e-6:
            return False, 0.0, 0.0

        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, world_up)
        n = float(np.linalg.norm(right))
        if n < 1e-9:                       # straight up or down: pick any horizontal
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / n
        up = np.cross(right, fwd)

        half_w = 0.5 * optics.framed_width_m(self.spec, self.focal_mm, depth)
        half_h = 0.5 * optics.framed_height_m(self.spec, self.focal_mm, depth)
        u = float(np.dot(rel, right)) / half_w
        v = float(np.dot(rel, up)) / half_h
        return bool(abs(u) <= 1.0 and abs(v) <= 1.0), u, v


@dataclass
class Subject:
    """A person, standing or walking, with a height."""

    position: np.ndarray
    height_m: float = 1.75

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)

    @property
    def eyeline(self) -> np.ndarray:
        """Where the eyes are: the point a camera is actually pointed at.

        Eye height is taken as 0.94 of stature, the usual anthropometric ratio.
        It matters because framing and sightlines are both about the head, not
        about the middle of a person.
        """
        p = self.position.copy()
        p[2] = p[2] + 0.94 * self.height_m
        return p

    @property
    def centre(self) -> np.ndarray:
        p = self.position.copy()
        p[2] = p[2] + 0.5 * self.height_m
        return p


@dataclass
class Beat:
    """One sampled instant of a move, with everything a scout would note."""

    t: float
    camera: Camera
    subject: Subject | None
    distance_m: float = 0.0
    framed_height_m: float = 0.0
    shot: str = ""
    subject_in_frame: bool = False
    frame_uv: tuple[float, float] = (0.0, 0.0)
    clear_sightline: bool = True
    clearance_m: float = 0.0
    headroom_m: float = 0.0
    floor_rise_m: float = float("nan")
    surveyed: bool = True
    gear_fits: bool = True
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass
class MoveReport:
    """A simulated move: every beat, and the verdict over all of them."""

    name: str
    beats: list[Beat]
    length_m: float = 0.0
    track_level_range_m: float = float("nan")
    notes: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return bool(self.beats) and all(b.ok for b in self.beats)

    @property
    def first_problem(self) -> Beat | None:
        return next((b for b in self.beats if b.problems), None)

    def summary(self) -> dict:
        shots = [b.shot for b in self.beats if b.shot]
        bad = self.first_problem
        return {
            "name": self.name,
            "feasible": self.feasible,
            "beats": len(self.beats),
            "length_m": round(self.length_m, 2),
            "shot_range": (
                f"{shots[0]} to {shots[-1]}" if shots and shots[0] != shots[-1]
                else (shots[0] if shots else "")
            ),
            "distance_m": (
                [round(min(b.distance_m for b in self.beats), 2),
                 round(max(b.distance_m for b in self.beats), 2)]
                if self.beats else []
            ),
            "track_level_range_m": (
                None if not np.isfinite(self.track_level_range_m)
                else round(self.track_level_range_m, 3)
            ),
            "fails_at_m": None if bad is None else round(bad.t * self.length_m, 2),
            "problems": [] if bad is None else bad.problems,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def straight(start, end, *, spacing_m: float = BEAT_SPACING_M) -> np.ndarray:
    """Sample a straight line -- a dolly move on laid track."""
    a = np.asarray(start, dtype=np.float64).reshape(3)
    b = np.asarray(end, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(b - a))
    n = max(MIN_BEATS, int(np.ceil(length / max(spacing_m, 1e-3))) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return a[None, :] + t * (b - a)[None, :]


def arc(centre, radius_m: float, start_deg: float, end_deg: float, height_m: float,
        *, spacing_m: float = BEAT_SPACING_M) -> np.ndarray:
    """Sample a circular arc about a centre -- a curved track or a jib swing."""
    c = np.asarray(centre, dtype=np.float64).reshape(3)
    sweep = math.radians(abs(end_deg - start_deg))
    length = radius_m * sweep
    n = max(MIN_BEATS, int(np.ceil(length / max(spacing_m, 1e-3))) + 1)
    ang = np.radians(np.linspace(start_deg, end_deg, n))
    return np.column_stack([
        c[0] + radius_m * np.cos(ang),
        c[1] + radius_m * np.sin(ang),
        np.full(n, height_m),
    ])


def through(waypoints, *, spacing_m: float = BEAT_SPACING_M) -> np.ndarray:
    """Sample a polyline -- a handheld walk, or an actor's blocking."""
    pts = np.asarray(waypoints, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 2:
        return pts
    out = [straight(pts[i], pts[i + 1], spacing_m=spacing_m)[:-1] for i in range(len(pts) - 1)]
    out.append(pts[-1:][None, 0][None, :].reshape(1, 3))
    return np.concatenate(out)


def _resample(path: np.ndarray, n: int) -> np.ndarray:
    """Resample a path to exactly `n` points, evenly in arc length.

    Needed because a camera path and a subject path are sampled independently
    and then have to be walked in step -- beat `i` of one must be the same
    instant as beat `i` of the other, or the simulation is comparing the camera
    at the start of the move against the actor at the end of the walk.
    """
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return np.repeat(pts[:1], n, axis=0)
    want = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(want, s, pts[:, k]) for k in range(3)])


# ---------------------------------------------------------------------------
# the simulation
# ---------------------------------------------------------------------------


def simulate(
    maps: FloorMaps,
    occupancy: tuple[np.ndarray, np.ndarray, float],
    camera_path: np.ndarray,
    *,
    name: str = "move",
    subject_path: np.ndarray | None = None,
    subject_height_m: float = 1.75,
    focal_mm: float = 32.0,
    aperture_f: float = 2.8,
    sensor: str = optics.DEFAULT_SENSOR,
    gear: Gear | None = None,
    on_track: bool = False,
    aim_at: np.ndarray | None = None,
) -> MoveReport:
    """Walk a camera along a path and report what it can and cannot do.

    `subject_path` may be a single point (a stationary actor), a path of its own
    (an actor walking, resampled to match the camera's beats), or None for a
    move with no subject, where only the physical questions are asked.
    """
    grid, origin, cell = occupancy
    cam_pts = np.asarray(camera_path, dtype=np.float64).reshape(-1, 3)
    if len(cam_pts) < 2:
        raise ValueError("a move needs at least two positions")
    n = len(cam_pts)

    subj_pts = None
    if subject_path is not None:
        subj_pts = _resample(np.asarray(subject_path, dtype=np.float64).reshape(-1, 3), n)

    seg = np.linalg.norm(np.diff(cam_pts, axis=0), axis=1)
    length = float(seg.sum())

    beats: list[Beat] = []
    notes: list[str] = []
    for i, pos in enumerate(cam_pts):
        subject = None
        if subj_pts is not None:
            subject = Subject(position=subj_pts[i], height_m=subject_height_m)
        target = (
            np.asarray(aim_at, dtype=np.float64).reshape(3) if aim_at is not None
            else (subject.eyeline if subject is not None else pos + np.array([1.0, 0.0, 0.0]))
        )
        cam = Camera(position=pos, aim=target, focal_mm=focal_mm,
                     aperture_f=aperture_f, sensor=sensor)
        beats.append(_beat(maps, grid, origin, cell, i / (n - 1), cam, subject, gear))

    report = MoveReport(name=name, beats=beats, length_m=length, notes=notes)

    if on_track:
        rise = np.array([b.floor_rise_m for b in beats], dtype=np.float64)
        seen = rise[np.isfinite(rise)]
        if len(seen) >= 2:
            report.track_level_range_m = float(seen.max() - seen.min())
            if report.track_level_range_m > TRACK_LEVEL_TOLERANCE_M:
                beats[int(np.nanargmax(np.abs(rise - np.nanmedian(rise))))].problems.append(
                    f"the floor under this run varies by "
                    f"{report.track_level_range_m * 1000:.0f} mm, past the "
                    f"{TRACK_LEVEL_TOLERANCE_M * 1000:.0f} mm a dolly can be wedged "
                    "level over; this needs cribbing"
                )
        else:
            notes.append(
                "the floor along this run was never properly seen by the sweep, so "
                "whether track can be levelled on it is unknown"
            )

    if not any(b.surveyed for b in beats):
        notes.append(
            "no part of this move is inside the surveyed area; it rests entirely "
            "on reconstructed geometry"
        )
    return report


def _beat(maps, grid, origin, cell, t, cam: Camera, subject, gear) -> Beat:
    xy = cam.position[:2]
    clearance = float(maps.sample(maps.clearance_m, xy)[0])
    headroom = float(maps.sample(maps.headroom_m, xy)[0])
    rise = float(maps.sample(maps.floor_rise_m, xy)[0])
    surveyed = bool(maps.sample(maps.surveyed, xy)[0])
    inside = bool(maps.sample(maps.inside, xy)[0])

    beat = Beat(
        t=float(t), camera=cam, subject=subject,
        clearance_m=clearance, headroom_m=headroom,
        floor_rise_m=rise, surveyed=surveyed,
    )

    if not inside:
        beat.problems.append("this position is outside the room's floor outline")

    if gear is not None:
        radius = 0.5 * float(np.hypot(*gear.footprint_m))
        beat.gear_fits = clearance >= radius
        if not beat.gear_fits:
            beat.problems.append(
                f"{gear.name} needs {radius * 100:.0f} cm of clear radius and has "
                f"{clearance * 100:.0f} cm here"
            )
        if gear.min_lens_height_m is not None and gear.max_lens_height_m is not None:
            lens_h = float(cam.position[2] - maps.floor_z)
            if not (gear.min_lens_height_m <= lens_h <= gear.max_lens_height_m):
                beat.problems.append(
                    f"lens height {lens_h:.2f} m is outside {gear.name}'s range of "
                    f"{gear.min_lens_height_m:.2f}-{gear.max_lens_height_m:.2f} m"
                )
        if headroom < gear.height_m:
            beat.problems.append(
                f"{gear.name} stands {gear.height_m:.2f} m and there is "
                f"{headroom:.2f} m of headroom here"
            )

    if subject is not None:
        target = subject.eyeline
        beat.distance_m = float(np.linalg.norm(target - cam.position))
        if beat.distance_m > 1e-6:
            beat.framed_height_m = optics.framed_height_m(cam.spec, cam.focal_mm, beat.distance_m)
            beat.shot = optics.classify_framing(beat.framed_height_m).name
        in_frame, u, v = cam.frames(target)
        beat.subject_in_frame = in_frame
        beat.frame_uv = (round(u, 3), round(v, 3))
        if not in_frame:
            beat.problems.append("the subject's eyeline falls outside the frame")
        beat.clear_sightline = visible(grid, origin, cell, cam.position, target)
        if not beat.clear_sightline:
            beat.problems.append("something is between the camera and the subject")

    return beat
