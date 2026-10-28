"""A camera simulation for the twin: keyframes, rigs, and the room answering back.

Blender will key a camera anywhere and never say a word. A room says plenty:
the dolly does not fit between the sofa and the wall, the jib's counterweight
goes through the table, the operator is walking faster than people walk, the
ceiling is 2.4 m and the crane key is at 2.6. This module is the keyframed
timeline (camera, actors, moved set pieces) evaluated frame by frame against
the measured twin, with every physical rig in the catalogue imposing its own
constraint on where the camera can be and how fast it can get there.

Three operations and one bridge:

- `solve` applies the rig's constraint to the camera keys and reports every
  key it had to move ("key 3 moved 0.42 m onto the track"). Pure.
- `evaluate` samples the solved timeline at the frame rate and asks the room
  about every frame: fit, headroom, sightline, framing, speed, collisions.
  Problems come back per frame and merged into segments for a timeline strip.
- `sample_time` is `evaluate` at one instant, for a live preview.
- `blender_script` writes a Blender 4.x script that rebuilds the whole thing
  (room, camera, actors, track, problems) so the offline render and the
  in-viewer preview describe the same move.

Coordinates are twin metres, Z up. Every `pos` is **absolute twin z**: the
floor is at `twin.structure.floor_z` (0 by construction), a lens 1.5 m off the
floor is at `z = floor_z + 1.5`. Nothing here is relative to the floor except
where a rig's limit is stated as a lens height, and those convert explicitly.

Nothing is a measurement that did not come from the twin, the catalogue or
`optics`. Framing arithmetic is `moves`' and `optics`'; this module keys it.
"""

from __future__ import annotations

import copy
import json
import math
import uuid
import zlib
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import equipment, moves, optics, space
from .moves import _beat as _move_beat  # the per-position checks the move simulator applies
from .render import EYELINE_RATIO, SUBJECT_HEIGHT_M

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

INTERPS = ("linear", "catmull", "ease")
EASES = ("inout", "linear")
DEFAULT_FPS = 24
DEFAULT_FOCAL_MM = 35.0
DEFAULT_APERTURE_F = 2.8
DEFAULT_SENSOR = optics.DEFAULT_SENSOR

# Short names the UI and the contract example use for sensors; canonical keys
# are `optics.SENSORS`.
SENSOR_ALIASES = {
    "ff": "fullframe", "full-frame": "fullframe", "full_frame": "fullframe", "vv": "fullframe",
    "s35": "super35", "super-35": "super35", "super_35": "super35",
    "s35-4perf": "super35-4perf", "mft": "m43", "phone": "phone-main",
}

# A lens needs air over it. The margin is the matte box and the operator's
# hands, not the ceiling's measurement error.
HEADROOM_MARGIN_M = 0.25

# How far a jib key may sit off its sphere before the snap is worth a note.
SNAP_REPORT_M = 0.01

# Handheld micro-shake: the angular amplitude (degrees, rms) at shake = 1.
SHAKE_FULL_DEG = 1.5
SHAKE_SMOOTH_FRAMES = 5

# Steadicam: the aim is low-passed over this many seconds of the past, so the
# operator's correction lags the subject the way it does on the day.
STEADICAM_LAG_S = 0.4

SEVERITIES = ("ok", "warn", "fail")

# Default jib base when the catalogue has no entry for it: a class-typical
# footprint for a mid-size jib's tripod base, stated as an estimate.
JIB_BASE_FOOTPRINT_M = (1.2, 1.2)
SLIDER_RAIL_WIDTH_M = 0.4


def new_id() -> str:
    return uuid.uuid4().hex[:8]


def canonical_sensor(key: str | None) -> str:
    """The `optics.SENSORS` key for a sensor name or alias; unknown names come back as given."""
    k = str(key or DEFAULT_SENSOR).strip().lower()
    return SENSOR_ALIASES.get(k, k)


def _f(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float(default)
    return x if math.isfinite(x) else float(default)


def _xyz(v) -> list[float]:
    a = [_f(x) for x in (list(v) + [0.0, 0.0, 0.0])[:3]]
    return a


def _r(v, nd: int = 4):
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_r(x, nd) for x in v]
    if isinstance(v, (float, np.floating)):
        x = float(v)
        return round(x, nd) if math.isfinite(x) else None
    if isinstance(v, np.generic):
        return v.item()
    return v


# ---------------------------------------------------------------------------
# the timeline
# ---------------------------------------------------------------------------


@dataclass
class CameraKey:
    """One camera keyframe. `pos` is absolute twin xyz; `aim` a point or None;
    `look_at` an actor name that wins over `aim`; `ease` shapes the segment
    this key starts (used by the "ease" and "catmull" interpolations)."""

    t: float
    pos: list[float]
    aim: list[float] | None = None
    look_at: str | None = None
    focal_mm: float = DEFAULT_FOCAL_MM
    roll_deg: float = 0.0
    ease: str = "inout"
    snapped_from: list[float] | None = None

    def to_dict(self) -> dict:
        d = {
            "t": round(float(self.t), 4),
            "pos": _r(self.pos),
            "aim": None if self.aim is None else _r(self.aim),
            "look_at": self.look_at,
            "focal_mm": round(float(self.focal_mm), 3),
            "roll_deg": round(float(self.roll_deg), 3),
            "ease": self.ease,
        }
        if self.snapped_from is not None:
            d["snapped_from"] = _r(self.snapped_from)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CameraKey":
        aim = d.get("aim")
        return cls(
            t=_f(d.get("t"), 0.0),
            pos=_xyz(d.get("pos") or [0.0, 0.0, 1.5]),
            aim=None if aim is None else _xyz(aim),
            look_at=(str(d["look_at"]) if d.get("look_at") else None),
            focal_mm=_f(d.get("focal_mm"), DEFAULT_FOCAL_MM),
            roll_deg=_f(d.get("roll_deg"), 0.0),
            ease=str(d.get("ease") or "inout"),
            snapped_from=None if d.get("snapped_from") is None else _xyz(d["snapped_from"]),
        )


@dataclass
class ActorKey:
    t: float
    pos: list[float]            # [x, y], feet on the floor
    facing_deg: float = 0.0

    def to_dict(self) -> dict:
        return {"t": round(float(self.t), 4), "pos": _r(self.pos[:2]), "facing_deg": round(float(self.facing_deg), 2)}

    @classmethod
    def from_dict(cls, d: dict) -> "ActorKey":
        p = list(d.get("pos") or [0.0, 0.0])
        return cls(t=_f(d.get("t")), pos=[_f(p[0]), _f(p[1]) if len(p) > 1 else 0.0], facing_deg=_f(d.get("facing_deg")))


@dataclass
class ActorTrack:
    height_m: float = SUBJECT_HEIGHT_M
    keys: list[ActorKey] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"height_m": round(float(self.height_m), 3), "keys": [k.to_dict() for k in self.keys]}

    @classmethod
    def from_dict(cls, d: dict) -> "ActorTrack":
        keys = sorted((ActorKey.from_dict(k) for k in d.get("keys") or []), key=lambda k: k.t)
        return cls(height_m=_f(d.get("height_m"), SUBJECT_HEIGHT_M), keys=keys)


@dataclass
class PieceKey:
    t: float
    dx: float = 0.0
    dy: float = 0.0
    dyaw_deg: float = 0.0

    def to_dict(self) -> dict:
        return {"t": round(float(self.t), 4), "dx": round(float(self.dx), 4),
                "dy": round(float(self.dy), 4), "dyaw_deg": round(float(self.dyaw_deg), 3)}

    @classmethod
    def from_dict(cls, d: dict) -> "PieceKey":
        return cls(t=_f(d.get("t")), dx=_f(d.get("dx")), dy=_f(d.get("dy")), dyaw_deg=_f(d.get("dyaw_deg")))


@dataclass
class PieceTrack:
    id: str
    keys: list[PieceKey] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "keys": [k.to_dict() for k in self.keys]}

    @classmethod
    def from_dict(cls, d: dict) -> "PieceTrack":
        keys = sorted((PieceKey.from_dict(k) for k in d.get("keys") or []), key=lambda k: k.t)
        return cls(id=str(d.get("id") or new_id()), keys=keys)


@dataclass
class Rig:
    kind: str = "tripod"
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        # Underscore params are solve-time caches (track arc lengths, the
        # pivot in 3D); `solve` rebuilds them, so they stay out of the JSON.
        return {"kind": self.kind, "params": _jsonable({k: v for k, v in self.params.items() if not str(k).startswith("_")})}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Rig":
        d = d or {}
        return cls(kind=str(d.get("kind") or "tripod"), params=dict(d.get("params") or {}))


@dataclass
class Timeline:
    """The contract object. Keys are kept sorted by `t`; the first camera key
    may start after 0 (the camera holds before it) and the last may end before
    `duration_s` (it holds after). Two camera keys at the same `t` are a cut."""

    id: str = field(default_factory=new_id)
    title: str = "Untitled"
    fps: int = DEFAULT_FPS
    duration_s: float = 8.0
    sensor: str = DEFAULT_SENSOR
    aperture_f: float = DEFAULT_APERTURE_F
    rig: Rig = field(default_factory=Rig)
    interp: str = "catmull"
    camera_keys: list[CameraKey] = field(default_factory=list)
    actors: dict[str, ActorTrack] = field(default_factory=dict)
    pieces: list[PieceTrack] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.sensor = canonical_sensor(self.sensor)
        self.sort()

    def sort(self) -> None:
        self.camera_keys.sort(key=lambda k: k.t)
        for tr in self.actors.values():
            tr.keys.sort(key=lambda k: k.t)
        for p in self.pieces:
            p.keys.sort(key=lambda k: k.t)

    # -- serialisation ---------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "fps": int(self.fps),
            "duration_s": round(float(self.duration_s), 4),
            "sensor": self.sensor,
            "aperture_f": round(float(self.aperture_f), 3),
            "rig": self.rig.to_dict(),
            "camera": {"interp": self.interp, "keys": [k.to_dict() for k in self.camera_keys]},
            "actors": {name: tr.to_dict() for name, tr in self.actors.items()},
            "pieces": [p.to_dict() for p in self.pieces],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Timeline":
        cam = d.get("camera") or {}
        return cls(
            id=str(d.get("id") or new_id()),
            title=str(d.get("title") or "Untitled"),
            fps=max(1, int(_f(d.get("fps"), DEFAULT_FPS))),
            duration_s=_f(d.get("duration_s"), 8.0),
            sensor=str(d.get("sensor") or DEFAULT_SENSOR),
            aperture_f=_f(d.get("aperture_f"), DEFAULT_APERTURE_F),
            rig=Rig.from_dict(d.get("rig")),
            interp=str(cam.get("interp") or "catmull"),
            camera_keys=[CameraKey.from_dict(k) for k in cam.get("keys") or []],
            actors={str(n): ActorTrack.from_dict(a) for n, a in (d.get("actors") or {}).items()},
            pieces=[PieceTrack.from_dict(p) for p in d.get("pieces") or []],
            notes=list(d.get("notes") or []),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "Timeline":
        return cls.from_dict(json.loads(s))

    # -- validation ------------------------------------------------------------

    def hard_problems(self) -> list[str]:
        """Problems that make the timeline impossible to evaluate at all."""
        out: list[str] = []
        if self.rig.kind not in RIGS:
            out.append(f"unknown rig {self.rig.kind!r}; have {sorted(RIGS)}")
        if self.sensor not in optics.SENSORS:
            out.append(f"unknown sensor {self.sensor!r}; have {sorted(optics.SENSORS)} "
                       f"(aliases {sorted(SENSOR_ALIASES)})")
        if self.fps <= 0:
            out.append(f"fps must be positive, got {self.fps}")
        if not (self.duration_s > 0):
            out.append(f"duration_s must be positive, got {self.duration_s}")
        if not self.camera_keys:
            out.append("the camera has no keys")
        if self.interp not in INTERPS:
            out.append(f"unknown interpolation {self.interp!r}; have {INTERPS}")
        if not (self.aperture_f > 0):
            out.append(f"aperture_f must be positive, got {self.aperture_f}")
        return out

    def validate(self, twin=None, *, maps: space.FloorMaps | None = None) -> list[str]:
        """Everything wrong with the timeline, soft issues included. Never raises."""
        out = self.hard_problems()
        for i, k in enumerate(self.camera_keys):
            if not (k.focal_mm > 0):
                out.append(f"key {i}: focal_mm must be positive, got {k.focal_mm}")
            if k.ease not in EASES:
                out.append(f"key {i}: unknown ease {k.ease!r}; have {EASES}")
            if k.t < 0 or k.t > self.duration_s + 1e-9:
                out.append(f"key {i}: t={k.t:g} s lies outside 0..{self.duration_s:g} s")
            if k.look_at and k.look_at not in self.actors:
                out.append(f"key {i}: look_at names {k.look_at!r}, who is not in the cast")
            if k.aim is None and not k.look_at:
                out.append(f"key {i}: neither aim nor look_at; the camera will look where the previous key did")
            if k.aim is not None and float(np.linalg.norm(np.subtract(k.aim, k.pos))) < 1e-6:
                out.append(f"key {i}: aim coincides with the camera position")
        for name, tr in self.actors.items():
            if not (1.0 <= tr.height_m <= 2.3):
                out.append(f"{name}: height {tr.height_m} m is not a human stature in metres")
            if not tr.keys:
                out.append(f"{name} has no keys and is never on the floor")
        if self.rig.kind in RIGS:
            out.extend(RIGS[self.rig.kind].validate_params(self.rig.params))
        if twin is not None and self.rig.kind in RIGS and self.camera_keys:
            try:
                m = maps if maps is not None else space.floor_maps(twin)
                for i, k in enumerate(self.camera_keys):
                    if not bool(m.sample(m.inside, k.pos[:2])[0]):
                        out.append(f"key {i}: the camera at ({k.pos[0]:.2f}, {k.pos[1]:.2f}) is outside the room's floor outline")
            except ValueError as e:
                out.append(f"the twin could not be rastered: {e}")
        return out


def _jsonable(v):
    if v is None:
        return None
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, np.generic):
        return _jsonable(v.item())
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    return v


# ---------------------------------------------------------------------------
# interpolation
# ---------------------------------------------------------------------------


def _smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _segment(ts: list[float], t: float) -> tuple[int, int, float]:
    """(i, j, u): the value at `t` lies between keys i and j at fraction u.
    Before the first key or after the last, i == j (a hold). Keys sharing a
    `t` form a cut: at that instant the later key wins."""
    n = len(ts)
    if n == 0:
        raise ValueError("no keys")
    if t <= ts[0]:
        return 0, 0, 0.0
    if t >= ts[-1]:
        return n - 1, n - 1, 0.0
    i = bisect_right(ts, t) - 1
    j = i + 1
    dt = ts[j] - ts[i]
    return i, j, (t - ts[i]) / dt if dt > 1e-9 else 1.0


def _catmull(P0, P1, P2, P3, u: float, alpha: float = 0.5) -> np.ndarray:
    """Centripetal Catmull-Rom (Barry-Goldman form) between P1 and P2."""
    P0, P1, P2, P3 = (np.asarray(p, dtype=np.float64) for p in (P0, P1, P2, P3))

    def knot(a, b):
        return max(float(np.linalg.norm(b - a)) ** alpha, 1e-6)

    t0 = 0.0
    t1 = t0 + knot(P0, P1)
    t2 = t1 + knot(P1, P2)
    t3 = t2 + knot(P2, P3)
    t = t1 + u * (t2 - t1)
    A1 = (t1 - t) / (t1 - t0) * P0 + (t - t0) / (t1 - t0) * P1
    A2 = (t2 - t) / (t2 - t1) * P1 + (t - t1) / (t2 - t1) * P2
    A3 = (t3 - t) / (t3 - t2) * P2 + (t - t2) / (t3 - t2) * P3
    B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
    B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
    return (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2


def _interp(ts: list[float], values, t: float, mode: str, eases: list[str] | None = None) -> np.ndarray:
    """Interpolate a series of vectors keyed at `ts`.

    "linear" ignores the keys' eases; "ease" applies each segment's start-key
    ease on top of a linear blend; "catmull" runs a centripetal Catmull-Rom
    through the keys with the same per-segment ease on the parameter. A cut
    (two keys at one t) is a jump in every mode.
    """
    V = np.asarray(values, dtype=np.float64)
    if V.ndim == 1:
        V = V[:, None]
    i, j, u = _segment(ts, t)
    if i == j:
        return V[i]
    if ts[j] - ts[i] <= 1e-9:
        return V[j]
    if mode != "linear" and eases is not None and eases[i] == "inout":
        u = _smoothstep(u)
    if mode == "catmull" and len(V) >= 3:
        P0 = V[i - 1] if i > 0 else V[i]
        P3 = V[j + 1] if j + 1 < len(V) else V[j]
        return _catmull(P0, V[i], V[j], P3, u)
    return V[i] + u * (V[j] - V[i])


def _lerp_angle_deg(a: float, b: float, u: float) -> float:
    d = (b - a + 180.0) % 360.0 - 180.0
    return a + u * d


# ---------------------------------------------------------------------------
# actors and pieces at a time
# ---------------------------------------------------------------------------


def actor_state(track: ActorTrack, t: float) -> tuple[float, float, float] | None:
    """(x, y, facing_deg) of an actor at `t`; None when the track has no keys.
    Actors always move linearly between keys: a walk is a polyline."""
    if not track.keys:
        return None
    ts = [k.t for k in track.keys]
    i, j, u = _segment(ts, t)
    a, b = track.keys[i], track.keys[j]
    if i == j or ts[j] - ts[i] <= 1e-9:
        k = b if i != j else a
        return float(k.pos[0]), float(k.pos[1]), float(k.facing_deg)
    x = a.pos[0] + u * (b.pos[0] - a.pos[0])
    y = a.pos[1] + u * (b.pos[1] - a.pos[1])
    return float(x), float(y), float(_lerp_angle_deg(a.facing_deg, b.facing_deg, u))


def actors_at(timeline: Timeline, t: float) -> dict[str, tuple[float, float, float]]:
    out = {}
    for name, tr in timeline.actors.items():
        s = actor_state(tr, t)
        if s is not None:
            out[name] = s
    return out


def eyeline(timeline: Timeline, name: str, t: float, floor_z: float) -> np.ndarray | None:
    s = actor_state(timeline.actors[name], t) if name in timeline.actors else None
    if s is None:
        return None
    return np.array([s[0], s[1], floor_z + EYELINE_RATIO * float(timeline.actors[name].height_m)])


def pieces_at(timeline: Timeline, t: float) -> dict[str, list[float]]:
    out = {}
    for p in timeline.pieces:
        if not p.keys:
            continue
        ts = [k.t for k in p.keys]
        v = _interp(ts, [[k.dx, k.dy, k.dyaw_deg] for k in p.keys], t, "linear")
        out[p.id] = [float(v[0]), float(v[1]), float(v[2])]
    return out


# ---------------------------------------------------------------------------
# problems
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    text: str
    severity: str = "fail"


# Which of `moves._beat`'s sentences are failures and which are warnings. The
# move simulator reports strings; the strip wants a colour.
_BEAT_WARN_MARKERS = ("lens height", "falls outside the frame")


def _from_beat(beat: moves.Beat) -> list[Problem]:
    out = []
    for p in beat.problems:
        sev = "warn" if any(m in p for m in _BEAT_WARN_MARKERS) else "fail"
        out.append(Problem(p, sev))
    return out


# ---------------------------------------------------------------------------
# rigs
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    """Everything an evaluation needs once, built in `evaluate`."""

    timeline: Timeline
    twin: Any
    maps: space.FloorMaps | None
    occupancy: tuple[np.ndarray, np.ndarray, float] | None
    floor_z: float
    rig: "_RigSpec"
    params: dict
    gear: equipment.Gear | None
    n_frames: int
    shake: np.ndarray | None = None       # (n_frames, 2) angular offsets in degrees


class _RigSpec:
    """A physical camera rig: how it constrains the keys, how its camera moves
    between them, and what it checks per frame."""

    kind = "rig"
    name = "rig"
    dof: tuple[str, ...] = ()
    speed_cap_mps: float | None = None
    accel_cap_mps2: float | None = None
    param_doc: dict[str, dict] = {}
    gear_kinds: tuple[str, ...] = ()
    checks: tuple[str, ...] = ("outside the footprint", "headroom", "sightline", "subject in frame")

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "degrees_of_freedom": list(self.dof),
            "limits": {
                "speed_mps": self.speed_cap_mps,
                "accel_mps2": self.accel_cap_mps2,
            },
            "params": copy.deepcopy(self.param_doc),
            "gear_keys": [g.key for g in equipment.CATALOGUE.values() if g.kind in self.gear_kinds],
            "checks": list(self.checks),
        }

    def validate_params(self, params: dict) -> list[str]:
        out = []
        gk = params.get("gear_key")
        if gk and gk not in equipment.CATALOGUE:
            out.append(f"unknown gear_key {gk!r}; have {sorted(equipment.CATALOGUE)}")
        return out

    def gear(self, params: dict) -> equipment.Gear | None:
        gk = params.get("gear_key")
        return equipment.CATALOGUE.get(gk) if gk else None

    # -- the constraint --------------------------------------------------------

    def prepare(self, keys: list[CameraKey], params: dict, floor_z: float, maps, twin) -> tuple[dict, list[str]]:
        """Resolve defaults from the keys and the room. Returns (params, notes)."""
        return dict(params), []

    def snap(self, pos: np.ndarray, params: dict, floor_z: float) -> np.ndarray:
        return np.asarray(pos, dtype=np.float64)

    def to_param(self, pos: np.ndarray, params: dict, floor_z: float) -> np.ndarray:
        return np.asarray(pos, dtype=np.float64)

    def from_param(self, q: np.ndarray, params: dict, floor_z: float) -> np.ndarray:
        return np.asarray(q, dtype=np.float64)

    def unwrap(self, qs: list[np.ndarray]) -> list[np.ndarray]:
        return qs

    # -- the checks --------------------------------------------------------------

    def frame_problems(self, ctx: _Ctx, cam: moves.Camera, speed: float, accel: float, cut: bool) -> list[Problem]:
        out: list[Problem] = []
        if not cut:
            if self.speed_cap_mps is not None and speed > self.speed_cap_mps + 1e-6:
                out.append(Problem(
                    f"the camera moves at {speed:.2f} m/s, over the {self.speed_cap_mps:.1f} m/s a {self.name} can do"))
            if self.accel_cap_mps2 is not None and abs(accel) > self.accel_cap_mps2 + 1e-6:
                out.append(Problem(
                    f"the camera accelerates at {abs(accel):.2f} m/s², over the {self.accel_cap_mps2:.1f} m/s² "
                    f"a {self.name} can take smoothly", "warn"))
        return out

    def path_problems(self, ctx: _Ctx, frames: list["FrameSample"]) -> None:
        """Problems about the path as a whole, attached to the frames they touch."""
        return None


class _Tripod(_RigSpec):
    kind = "tripod"
    name = "tripod"
    dof = ("pan", "tilt", "lens")
    gear_kinds = ("support",)
    param_doc = {"gear_key": {"default": "sticks-standard", "doc": "catalogue support the camera sits on"}}
    checks = _RigSpec.checks + ("gear footprint", "lens height within the legs' range", "tilt within ±80°")
    TILT_LIMIT_DEG = 80.0

    def gear(self, params):
        return equipment.CATALOGUE.get(params.get("gear_key") or "sticks-standard")

    def prepare(self, keys, params, floor_z, maps, twin):
        p = dict(params)
        p.setdefault("gear_key", "sticks-standard")
        p["anchor"] = [float(v) for v in keys[0].pos]
        return p, []

    def snap(self, pos, params, floor_z):
        return np.asarray(params["anchor"], dtype=np.float64)

    def frame_problems(self, ctx, cam, speed, accel, cut):
        out = super().frame_problems(ctx, cam, speed, accel, cut)
        if abs(cam.tilt_deg) > self.TILT_LIMIT_DEG:
            out.append(Problem(f"tilt of {cam.tilt_deg:.0f}° is past the ±{self.TILT_LIMIT_DEG:.0f}° a fluid head reaches", "warn"))
        return out


class _Slider(_RigSpec):
    kind = "slider"
    name = "slider"
    dof = ("slide", "pan", "tilt", "lens")
    gear_kinds = ("support",)
    speed_cap_mps = 1.0
    param_doc = {
        "length_m": {"default": 1.0, "range": [1.0, 2.0], "doc": "rail length; clamped"},
        "gear_key": {"default": None, "doc": "catalogue support the rail stands on"},
    }
    checks = _RigSpec.checks + ("rail footprint",)

    def validate_params(self, params):
        out = super().validate_params(params)
        L = params.get("length_m")
        if L is not None and not (1.0 <= _f(L, 1.0) <= 2.0):
            out.append(f"slider length_m={L} is outside 1.0..2.0 m and will be clamped")
        return out

    def gear(self, params):
        g = super().gear(params)
        if g is not None:
            return g
        L = float(params.get("length_m", 1.0))
        return equipment.Gear(key="slider", name=f"{L:.1f} m slider", kind="support",
                              footprint_m=(L, SLIDER_RAIL_WIDTH_M), height_m=1.5,
                              min_lens_height_m=0.3, max_lens_height_m=1.9, clearance_m=0.4,
                              verified=False, source="Class-typical rail on sticks; not a published spec.")

    def prepare(self, keys, params, floor_z, maps, twin):
        p = dict(params)
        notes = []
        L = _f(p.get("length_m"), 1.0)
        Lc = min(max(L, 1.0), 2.0)
        if abs(Lc - L) > 1e-9:
            notes.append(f"slider length {L:g} m clamped to {Lc:g} m")
        p["length_m"] = Lc
        a = np.asarray(keys[0].pos, dtype=np.float64)
        b = np.asarray(keys[-1].pos, dtype=np.float64)
        d = b[:2] - a[:2]
        n = float(np.linalg.norm(d))
        d = d / n if n > 1e-6 else np.array([1.0, 0.0])
        c = 0.5 * (a[:2] + b[:2])
        p["segment"] = [(c - 0.5 * Lc * d).tolist(), (c + 0.5 * Lc * d).tolist()]
        p["height_z"] = float(a[2])
        if n > Lc + 1e-6:
            notes.append(f"the keys span {n:.2f} m and the slider is {Lc:g} m; the ends are pulled in")
        return p, notes

    def snap(self, pos, params, floor_z):
        s0, s1 = (np.asarray(v, dtype=np.float64) for v in params["segment"])
        d = s1 - s0
        L2 = float(d @ d)
        u = float(np.clip(((pos[:2] - s0) @ d) / L2, 0.0, 1.0)) if L2 > 1e-12 else 0.0
        xy = s0 + u * d
        return np.array([xy[0], xy[1], params["height_z"]])


class _Dolly(_RigSpec):
    kind = "dolly"
    name = "dolly"
    dof = ("track", "column", "pan", "tilt", "lens")
    gear_kinds = ("dolly",)
    speed_cap_mps = 1.5
    accel_cap_mps2 = 1.0
    param_doc = {
        "track": {"default": None, "doc": "[[x, y], ...] polyline; derived from the keys when null"},
        "rail_height_m": {"default": 0.0, "doc": "track height above the floor"},
        "gear_key": {"default": "doorway-dolly", "doc": "catalogue dolly; its column sets the lens height range"},
    }
    checks = _RigSpec.checks + ("gear footprint along the whole track", "floor level along the track (moves.simulate)",
                                "lens height within the column", "speed 1.5 m/s", "accel 1.0 m/s²")
    DEFAULT_COLUMN_M = (0.6, 1.8)

    def gear(self, params):
        return equipment.CATALOGUE.get(params.get("gear_key") or "doorway-dolly")

    def prepare(self, keys, params, floor_z, maps, twin):
        p = dict(params)
        notes = []
        p.setdefault("gear_key", "doorway-dolly")
        p["rail_height_m"] = _f(p.get("rail_height_m"), 0.0)
        g = self.gear(p)
        if g is not None and g.min_lens_height_m is not None and g.max_lens_height_m is not None:
            zmin, zmax = g.min_lens_height_m, g.max_lens_height_m
        else:
            zmin, zmax = self.DEFAULT_COLUMN_M
            notes.append(f"{g.name if g else 'the dolly'} publishes no column range; using {zmin:.1f}-{zmax:.1f} m")
        p["z_range"] = [floor_z + p["rail_height_m"] + zmin, floor_z + p["rail_height_m"] + zmax]
        track = p.get("track")
        if not track:
            pts = []
            for k in keys:
                xy = [float(k.pos[0]), float(k.pos[1])]
                if not pts or math.dist(pts[-1], xy) > 0.01:
                    pts.append(xy)
            if len(pts) == 1:
                pts = [[pts[0][0] - 0.5, pts[0][1]], [pts[0][0] + 0.5, pts[0][1]]]
                notes.append("one camera position: laid 1 m of track through it along +X")
            else:
                notes.append("track derived from the keys: a " + ("straight run" if len(pts) == 2 else f"polyline through {len(pts)} points"))
            track = pts
        track = [[float(a), float(b)] for a, b in track]
        if len(track) < 2:
            track = [track[0], [track[0][0] + 1.0, track[0][1]]]
        p["track"] = track
        T = np.asarray(track, dtype=np.float64)
        seg = np.linalg.norm(np.diff(T, axis=0), axis=1)
        p["_cum"] = np.concatenate([[0.0], np.cumsum(seg)]).tolist()
        p["track_length_m"] = float(p["_cum"][-1])
        return p, notes

    def _project(self, xy: np.ndarray, params: dict) -> tuple[float, np.ndarray]:
        T = np.asarray(params["track"], dtype=np.float64)
        cum = params["_cum"]
        best = (float("inf"), 0.0, T[0])
        for i in range(len(T) - 1):
            a, b = T[i], T[i + 1]
            d = b - a
            L2 = float(d @ d)
            u = float(np.clip(((xy - a) @ d) / L2, 0.0, 1.0)) if L2 > 1e-12 else 0.0
            pnt = a + u * d
            dist = float(np.linalg.norm(xy - pnt))
            if dist < best[0]:
                best = (dist, cum[i] + u * math.sqrt(L2), pnt)
        return best[1], best[2]

    def _at(self, s: float, params: dict) -> np.ndarray:
        T = np.asarray(params["track"], dtype=np.float64)
        cum = np.asarray(params["_cum"], dtype=np.float64)
        s = float(np.clip(s, 0.0, cum[-1]))
        i = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(T) - 2))
        L = cum[i + 1] - cum[i]
        u = (s - cum[i]) / L if L > 1e-12 else 0.0
        return T[i] + u * (T[i + 1] - T[i])

    def snap(self, pos, params, floor_z):
        s, xy = self._project(np.asarray(pos[:2], dtype=np.float64), params)
        z = float(np.clip(pos[2], params["z_range"][0], params["z_range"][1]))
        return np.array([xy[0], xy[1], z])

    def to_param(self, pos, params, floor_z):
        s, _ = self._project(np.asarray(pos[:2], dtype=np.float64), params)
        return np.array([s, float(pos[2])])

    def from_param(self, q, params, floor_z):
        xy = self._at(float(q[0]), params)
        return np.array([xy[0], xy[1], float(q[1])])

    def path_problems(self, ctx, frames):
        if ctx.maps is None or ctx.occupancy is None or not frames:
            return
        p = ctx.params
        T = np.asarray(p["track"], dtype=np.float64)
        z = float(np.mean([f.cam[2] for f in frames]))
        path = moves.through(np.column_stack([T, np.full(len(T), z)]))
        if len(path) < 2:
            return
        report = moves.simulate(ctx.maps, ctx.occupancy, path, name="dolly track", gear=ctx.gear, on_track=True)
        length = report.length_m
        if length <= 1e-9:
            return
        # Each frame's arc-length along the track; a beat's problems land on
        # the frames nearest its station, or on the nearest frame when the
        # camera never reaches that stretch (the track is still laid there).
        fs = np.array([self.to_param(np.asarray(f.cam), p, ctx.floor_z)[0] for f in frames])
        for beat in report.beats:
            if not beat.problems:
                continue
            s = beat.t * length
            near = np.flatnonzero(np.abs(fs - s) <= 0.5 * moves.BEAT_SPACING_M + 1e-9)
            if not len(near):
                near = [int(np.argmin(np.abs(fs - s)))]
            for prob in _from_beat(beat):
                # The per-frame checks already said this where the camera
                # stood; the track pass adds what they could not see: stretches
                # the camera never reaches, and the floor level over the run.
                targets = [k for k in near if all(q.text != prob.text for q in frames[k].problems)]
                text = f"{s:.1f} m along the track: {prob.text}"
                for k in targets:
                    frames[k].add(Problem(text, prob.severity))
        for n in report.notes:
            ctx.timeline.notes.append(n) if n not in ctx.timeline.notes else None


class _Jib(_RigSpec):
    kind = "jib"
    name = "jib"
    dof = ("swing", "boom", "pan", "tilt", "lens")
    gear_kinds = ("support",)
    speed_cap_mps = 2.0
    param_doc = {
        "pivot": {"default": None, "doc": "[x, y] of the base; derived from the keys when null"},
        "base_height_m": {"default": 1.0, "doc": "height of the pivot above the floor"},
        "arm_m": {"default": 2.5, "range": [2.5, 6.0], "doc": "arm length from pivot to camera; clamped"},
        "gear_key": {"default": None, "doc": "catalogue support for the base footprint; a 1.2 x 1.2 m estimate otherwise"},
    }
    checks = _RigSpec.checks + ("arm clear of the room", "counterweight clear of the room and the floor", "base footprint at the pivot")

    def validate_params(self, params):
        out = super().validate_params(params)
        a = params.get("arm_m")
        if a is not None and not (2.5 <= _f(a, 2.5) <= 6.0):
            out.append(f"jib arm_m={a} is outside 2.5..6.0 m and will be clamped")
        return out

    def prepare(self, keys, params, floor_z, maps, twin):
        p = dict(params)
        notes = []
        arm = _f(p.get("arm_m"), 2.5)
        armc = min(max(arm, 2.5), 6.0)
        if abs(armc - arm) > 1e-9:
            notes.append(f"jib arm {arm:g} m clamped to {armc:g} m")
        p["arm_m"] = armc
        p["base_height_m"] = _f(p.get("base_height_m"), 1.0)
        if not p.get("pivot"):
            xy = np.array([[k.pos[0], k.pos[1]] for k in keys], dtype=np.float64).mean(axis=0)
            centre = np.zeros(2)
            if maps is not None:
                centre = maps.origin + 0.5 * np.array(maps.shape) * maps.cell
            d = centre - xy
            n = float(np.linalg.norm(d))
            d = d / n if n > 1e-6 else np.array([-1.0, 0.0])
            pivot = xy + 0.8 * armc * d
            p["pivot"] = [float(pivot[0]), float(pivot[1])]
            notes.append(f"jib pivot placed at ({pivot[0]:.2f}, {pivot[1]:.2f}), {0.8 * armc:.1f} m from the keys toward the room's middle")
        p["pivot"] = [float(p["pivot"][0]), float(p["pivot"][1])]
        p["_pivot3"] = [p["pivot"][0], p["pivot"][1], floor_z + p["base_height_m"]]
        return p, notes

    def gear(self, params):
        g = super().gear(params)
        if g is not None:
            return g
        return equipment.Gear(key="jib-base", name="jib base", kind="support", footprint_m=JIB_BASE_FOOTPRINT_M,
                              height_m=1.6, clearance_m=0.5, verified=False,
                              source="Class-typical tripod base for a mid-size jib; not a published spec.")

    def to_param(self, pos, params, floor_z):
        v = np.asarray(pos, dtype=np.float64) - np.asarray(params["_pivot3"])
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return np.array([0.0, 0.0])
        az = math.atan2(v[1], v[0])
        el = math.asin(float(np.clip(v[2] / n, -1.0, 1.0)))
        return np.array([az, el])

    def from_param(self, q, params, floor_z):
        az, el = float(q[0]), float(np.clip(q[1], -math.pi / 2, math.pi / 2))
        a = params["arm_m"]
        return np.asarray(params["_pivot3"]) + a * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])

    def snap(self, pos, params, floor_z):
        return self.from_param(self.to_param(pos, params, floor_z), params, floor_z)

    def unwrap(self, qs):
        out = [np.array(qs[0], dtype=np.float64)]
        for q in qs[1:]:
            q = np.array(q, dtype=np.float64)
            prev = out[-1][0]
            while q[0] - prev > math.pi:
                q[0] -= 2 * math.pi
            while q[0] - prev < -math.pi:
                q[0] += 2 * math.pi
            out.append(q)
        return out

    def frame_problems(self, ctx, cam, speed, accel, cut):
        out = super().frame_problems(ctx, cam, speed, accel, cut)
        pivot = np.asarray(ctx.params["_pivot3"], dtype=np.float64)
        end = cam.position
        cw = pivot - (end - pivot)
        if ctx.occupancy is not None:
            grid, origin, cell = ctx.occupancy
            if not space.visible(grid, origin, cell, pivot, end, slack=0.3):
                out.append(Problem("the jib arm passes through something between the pivot and the camera"))
            if not space.visible(grid, origin, cell, pivot, cw, slack=0.3):
                out.append(Problem("the counterweight arm swings through something"))
        if cw[2] < ctx.floor_z + 0.05:
            out.append(Problem(f"the counterweight end would be {ctx.floor_z - cw[2]:.2f} m below the floor at this boom angle"))
        if ctx.maps is not None and ctx.gear is not None:
            if not space.fits(ctx.maps, ctx.gear.footprint_m, pivot[:2]):
                out.append(Problem(f"the {ctx.gear.name} does not fit at the pivot ({pivot[0]:.2f}, {pivot[1]:.2f})"))
            if not bool(ctx.maps.sample(ctx.maps.inside, pivot[:2])[0]):
                out.append(Problem("the jib pivot is outside the room's floor outline"))
        return out


class _Handheld(_RigSpec):
    kind = "handheld"
    name = "handheld camera"
    dof = ("walk", "height", "pan", "tilt", "roll", "lens")
    gear_kinds = ()
    speed_cap_mps = 1.6
    param_doc = {
        "operator_height_m": {"default": 1.7, "doc": "the operator's stature"},
        "shake": {"default": 0.3, "range": [0.0, 1.0], "doc": "micro-shake amplitude; deterministic from the timeline id"},
    }
    checks = _RigSpec.checks + ("on the standable floor", "lens 0.9-2.0 m", "walking speed 1.6 m/s")
    Z_RANGE_M = (0.9, 2.0)
    shakes = True

    def gear(self, params):
        return equipment.CATALOGUE["handheld"]

    def prepare(self, keys, params, floor_z, maps, twin):
        p = dict(params)
        p["operator_height_m"] = _f(p.get("operator_height_m"), 1.7)
        if self.shakes:
            p["shake"] = min(max(_f(p.get("shake"), 0.3), 0.0), 1.0)
        p["z_range"] = [floor_z + self.Z_RANGE_M[0], floor_z + self.Z_RANGE_M[1]]
        return p, []

    def snap(self, pos, params, floor_z):
        out = np.asarray(pos, dtype=np.float64).copy()
        out[2] = float(np.clip(out[2], params["z_range"][0], params["z_range"][1]))
        return out

    def frame_problems(self, ctx, cam, speed, accel, cut):
        out = super().frame_problems(ctx, cam, speed, accel, cut)
        if ctx.maps is not None and not bool(ctx.maps.sample(ctx.maps.standable(), cam.position[:2])[0]):
            out.append(Problem("the operator is off the standable floor here"))
        return out


class _Steadicam(_Handheld):
    kind = "steadicam"
    name = "Steadicam"
    param_doc = {"operator_height_m": {"default": 1.7, "doc": "the operator's stature"}}
    checks = _Handheld.checks + ("aim low-passed over 0.4 s",)
    shakes = False


class _Crane(_RigSpec):
    kind = "crane"
    name = "crane"
    dof = ("x", "y", "z", "pan", "tilt", "roll", "lens")
    gear_kinds = ()
    speed_cap_mps = 2.0
    param_doc = {"max_height_m": {"default": "ceiling height, else 6.0", "doc": "lens height ceiling above the floor"}}
    checks = _RigSpec.checks + ("lens under max_height_m", "speed 2.0 m/s")

    def prepare(self, keys, params, floor_z, maps, twin):
        p = dict(params)
        notes = []
        mh = p.get("max_height_m")
        if mh is None:
            ceiling = None
            if twin is not None and getattr(twin, "structure", None) is not None and twin.structure.ceiling_z is not None:
                ceiling = float(twin.structure.ceiling_z) - floor_z
            elif maps is not None and maps.ceiling_z is not None:
                ceiling = float(maps.ceiling_z) - floor_z
            mh = ceiling if ceiling is not None else 6.0
            notes.append(f"crane max height taken as {'the ceiling, ' if ceiling is not None else ''}{mh:.2f} m")
        p["max_height_m"] = _f(mh, 6.0)
        p["z_range"] = [floor_z + 0.3, floor_z + p["max_height_m"]]
        return p, notes

    def snap(self, pos, params, floor_z):
        out = np.asarray(pos, dtype=np.float64).copy()
        out[2] = float(np.clip(out[2], params["z_range"][0], params["z_range"][1]))
        return out


class _Drone(_RigSpec):
    kind = "drone"
    name = "drone"
    dof = ("x", "y", "z", "pan", "tilt", "roll", "lens")
    gear_kinds = ()
    speed_cap_mps = 3.0
    param_doc = {"min_clearance_m": {"default": 0.5, "doc": "distance kept from every measured point"}}
    checks = _RigSpec.checks + ("0.5 m clear of every point", "speed 3.0 m/s")

    def prepare(self, keys, params, floor_z, maps, twin):
        p = dict(params)
        p["min_clearance_m"] = _f(p.get("min_clearance_m"), 0.5)
        return p, []

    def frame_problems(self, ctx, cam, speed, accel, cut):
        out = super().frame_problems(ctx, cam, speed, accel, cut)
        if ctx.occupancy is not None:
            d = _clearance_to_occupied(ctx.occupancy, cam.position, ctx.params["min_clearance_m"])
            if d is not None:
                out.append(Problem(f"the drone is {d:.2f} m from something, inside its {ctx.params['min_clearance_m']:.1f} m clearance"))
        return out


def _clearance_to_occupied(occupancy, pos, radius: float) -> float | None:
    """Distance to the nearest occupied voxel within `radius`, else None."""
    grid, origin, cell = occupancy
    p = np.asarray(pos, dtype=np.float64)
    lo = np.floor((p - radius - origin) / cell).astype(int)
    hi = np.floor((p + radius - origin) / cell).astype(int)
    lo = np.clip(lo, 0, np.array(grid.shape) - 1)
    hi = np.clip(hi, 0, np.array(grid.shape) - 1)
    sub = grid[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    if not sub.any():
        return None
    idx = np.argwhere(sub) + lo
    centres = origin + (idx + 0.5) * cell
    d = np.linalg.norm(centres - p, axis=1)
    m = float(d.min())
    return m if m <= radius else None


RIGS: dict[str, _RigSpec] = {r.kind: r for r in (_Tripod(), _Slider(), _Dolly(), _Jib(), _Handheld(), _Steadicam(), _Crane(), _Drone())}


def describe_rigs() -> list[dict]:
    """What the editor shows in its rig picker."""
    return [RIGS[k].describe() for k in RIGS]


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------


def _floor_z(twin, maps) -> float:
    if maps is not None:
        return float(maps.floor_z)
    if twin is not None and getattr(twin, "structure", None) is not None:
        return float(twin.structure.floor_z)
    return 0.0


def solve(timeline: Timeline, twin=None, *, maps: space.FloorMaps | None = None) -> Timeline:
    """Apply the rig's constraint to the camera keys. Returns a new timeline
    whose keys sit where the rig can put them, with resolved rig params and a
    note per moved key. The input is not touched."""
    hard = timeline.hard_problems()
    if hard:
        raise ValueError("; ".join(hard))
    tl = copy.deepcopy(timeline)
    tl.sort()
    rig = RIGS[tl.rig.kind]
    floor_z = _floor_z(twin, maps)
    params, notes = rig.prepare(tl.camera_keys, tl.rig.params, floor_z, maps, twin)
    for i, k in enumerate(tl.camera_keys):
        pos = np.asarray(k.pos, dtype=np.float64)
        new = rig.snap(pos, params, floor_z)
        moved = float(np.linalg.norm(new - pos))
        if moved > SNAP_REPORT_M:
            k.snapped_from = [float(v) for v in pos]
            what = {"dolly": "onto the track", "slider": "onto the rail", "tripod": "to the tripod's position",
                    "jib": "onto the arm's reach", "handheld": "into the operator's reach",
                    "steadicam": "into the operator's reach", "crane": "under the height limit"}.get(rig.kind, "")
            notes.append(f"key {i} moved {moved:.2f} m {what}".rstrip())
        else:
            k.snapped_from = None
        k.pos = [float(v) for v in new]
    tl.rig.params = {k: v for k, v in params.items()}
    # Re-solving a solved timeline replaces its snap notes rather than piling them up.
    tl.notes = [n for n in tl.notes if not (n.startswith("key ") and " moved " in n)]
    tl.notes.extend(n for n in notes if n not in tl.notes)
    return tl


# ---------------------------------------------------------------------------
# the camera at a time
# ---------------------------------------------------------------------------


@dataclass
class _CamState:
    pos: np.ndarray
    aim: np.ndarray
    focal_mm: float
    roll_deg: float
    look_at: str | None
    cut: bool


def _key_aim(tl: Timeline, k: CameraKey, t: float, floor_z: float, fallback: np.ndarray) -> np.ndarray:
    if k.look_at:
        e = eyeline(tl, k.look_at, t, floor_z)
        if e is not None:
            return e
    if k.aim is not None:
        return np.asarray(k.aim, dtype=np.float64)
    return fallback


def _camera_raw(tl: Timeline, rig: _RigSpec, params: dict, floor_z: float, t: float) -> _CamState:
    """Position, aim, focal and roll at `t` from the solved keys, before any
    rig-specific treatment of the aim (shake, low-pass)."""
    keys = tl.camera_keys
    ts = [k.t for k in keys]
    eases = [k.ease for k in keys]
    qs = rig.unwrap([rig.to_param(np.asarray(k.pos, dtype=np.float64), params, floor_z) for k in keys])
    q = _interp(ts, np.array(qs), t, tl.interp, eases)
    pos = rig.from_param(q, params, floor_z)
    i, j, u = _segment(ts, t)
    cut = i != j and ts[j] - ts[i] <= 1e-9

    # Aim: every key's target resolved at *this* instant, then blended, so
    # two keys on the same actor track them exactly and a key on a point
    # blends toward the actor as the segment runs.
    aims = []
    last = np.asarray(keys[0].pos, dtype=np.float64) + np.array([1.0, 0.0, 0.0])
    for k in keys:
        a = _key_aim(tl, k, t, floor_z, last)
        aims.append(a)
        last = a
    aim = _interp(ts, np.array(aims), t, tl.interp, eases)
    focal = float(math.exp(_interp(ts, np.log([max(k.focal_mm, 1e-3) for k in keys]), t, tl.interp, eases)[0]))
    roll = float(_interp(ts, [k.roll_deg for k in keys], t, tl.interp, eases)[0])
    look = keys[j].look_at if i == j or ts[j] - ts[i] <= 1e-9 else (keys[i].look_at or keys[j].look_at)
    if float(np.linalg.norm(aim - pos)) < 1e-6:
        aim = pos + np.array([1.0, 0.0, 0.0])
    return _CamState(pos=pos, aim=aim, focal_mm=focal, roll_deg=roll, look_at=look, cut=cut)


def _shake_table(seed_text: str, n: int) -> np.ndarray:
    """(n, 2) degrees of aim jitter, unit rms, smoothed to a hand's tremor.
    Seeded from the timeline id so the preview and the render agree."""
    rng = np.random.default_rng(zlib.crc32(seed_text.encode("utf-8")))
    raw = rng.normal(size=(n + 2 * SHAKE_SMOOTH_FRAMES, 2))
    k = np.ones(SHAKE_SMOOTH_FRAMES) / SHAKE_SMOOTH_FRAMES
    sm = np.column_stack([np.convolve(raw[:, c], k, mode="same") for c in range(2)])
    sm = sm[SHAKE_SMOOTH_FRAMES:SHAKE_SMOOTH_FRAMES + n]
    rms = float(np.sqrt(np.mean(sm ** 2))) if n else 1.0
    return sm / rms if rms > 1e-9 else sm


def _frame_index(tl: Timeline, t: float, n: int) -> int:
    return int(min(max(round(t * tl.fps), 0), n - 1))


def _camera_at(ctx: _Ctx, t: float) -> _CamState:
    tl, rig, p, fz = ctx.timeline, ctx.rig, ctx.params, ctx.floor_z
    st = _camera_raw(tl, rig, p, fz, t)
    if rig.kind == "steadicam":
        # A causal low-pass: the operator's correction lags the subject.
        n_lag = max(1, int(round(STEADICAM_LAG_S * tl.fps)))
        w = np.exp(-np.arange(n_lag + 1) / (n_lag / 3.0))
        acc = np.zeros(3)
        for j, wj in enumerate(w):
            tj = max(0.0, t - j / tl.fps)
            acc += wj * _camera_raw(tl, rig, p, fz, tj).aim
        st.aim = acc / w.sum()
    if ctx.shake is not None and p.get("shake", 0.0) > 0:
        k = _frame_index(tl, t, ctx.n_frames)
        ax, ay = np.radians(ctx.shake[k] * SHAKE_FULL_DEG * p["shake"])
        fwd = st.aim - st.pos
        d = float(np.linalg.norm(fwd))
        fwd = fwd / d
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        rn = float(np.linalg.norm(right))
        right = right / rn if rn > 1e-9 else np.array([1.0, 0.0, 0.0])
        up = np.cross(right, fwd)
        st.aim = st.pos + d * (fwd + math.tan(ax) * right + math.tan(ay) * up)
    return st


# ---------------------------------------------------------------------------
# frames and the evaluation
# ---------------------------------------------------------------------------


@dataclass
class FrameSample:
    t: float
    cam: list[float]
    aim: list[float]
    focal_mm: float
    pan_deg: float
    tilt_deg: float
    roll_deg: float
    speed_mps: float
    accel_mps2: float
    cut: bool
    actors: dict[str, list[float]]
    pieces: dict[str, list[float]]
    subject: str | None
    distance_m: float | None
    framed_height_m: float | None
    subject_fill: float | None
    shot_size: str | None
    shot_name: str | None
    in_frame: bool | None
    focus_m: float
    dof: dict
    fov_h_deg: float
    fov_v_deg: float
    problems: list[Problem] = field(default_factory=list)

    def add(self, p: Problem) -> None:
        if all(q.text != p.text for q in self.problems):
            self.problems.append(p)

    @property
    def severity(self) -> str:
        if any(p.severity == "fail" for p in self.problems):
            return "fail"
        return "warn" if self.problems else "ok"

    def to_dict(self) -> dict:
        return {
            "t": round(self.t, 4),
            "cam": _r(self.cam),
            "aim": _r(self.aim),
            "focal_mm": round(self.focal_mm, 3),
            "pan_deg": round(self.pan_deg, 2),
            "tilt_deg": round(self.tilt_deg, 2),
            "roll_deg": round(self.roll_deg, 2),
            "speed_mps": round(self.speed_mps, 3),
            "accel_mps2": round(self.accel_mps2, 3),
            "cut": bool(self.cut),
            "actors": {k: _r(v, 3) for k, v in self.actors.items()},
            "pieces": {k: _r(v, 4) for k, v in self.pieces.items()},
            "subject": self.subject,
            "distance_m": _r(self.distance_m, 3),
            "framed_height_m": _r(self.framed_height_m, 3),
            "subject_fill": _r(self.subject_fill, 3),
            "shot_size": self.shot_size,
            "shot_name": self.shot_name,
            "in_frame": self.in_frame,
            "focus_m": round(self.focus_m, 3),
            "dof": dict(self.dof),
            "fov_h_deg": round(self.fov_h_deg, 2),
            "fov_v_deg": round(self.fov_v_deg, 2),
            "problems": [p.text for p in self.problems],
            "severity": self.severity,
        }


@dataclass
class Evaluation:
    timeline: Timeline                  # the solved timeline
    frames: list[FrameSample]
    segments: list[dict]
    summary: dict

    def to_dict(self) -> dict:
        return {
            "fps": int(self.timeline.fps),
            "duration_s": round(float(self.timeline.duration_s), 4),
            "timeline": self.timeline.to_dict(),
            "frames": [f.to_dict() for f in self.frames],
            "segments": [dict(s) for s in self.segments],
            "summary": _jsonable(self.summary),
        }


def _build_ctx(timeline: Timeline, twin, maps, occupancy, pieces) -> _Ctx:
    if twin is not None and pieces:
        try:
            from ..product import setpieces as _sp
            objs = [p if not isinstance(p, dict) else _sp.SetPiece.from_dict(p) for p in pieces]
            twin = _sp.apply(twin, objs)
            if maps is None:
                maps = space.floor_maps(twin)
            if occupancy is None:
                occupancy = space.occupancy(twin)
        except Exception as e:  # noqa: BLE001 - the dressing is optional; the checks run on the bare twin
            timeline.notes.append(f"set pieces could not be applied to the twin ({e}); checks use the bare twin")
    if twin is not None:
        if maps is None:
            maps = space.floor_maps(twin)
        if occupancy is None:
            occupancy = space.occupancy(twin)
    solved = solve(timeline, twin, maps=maps)
    rig = RIGS[solved.rig.kind]
    floor_z = _floor_z(twin, maps)
    n = int(round(solved.duration_s * solved.fps)) + 1
    shake = None
    if getattr(rig, "shakes", False) and solved.rig.params.get("shake", 0.0) > 0:
        shake = _shake_table(solved.id, n)
    return _Ctx(timeline=solved, twin=twin, maps=maps, occupancy=occupancy, floor_z=floor_z, rig=rig,
                params=solved.rig.params, gear=rig.gear(solved.rig.params), n_frames=n, shake=shake)


def _frame(ctx: _Ctx, t: float, prev: FrameSample | None, prev2: FrameSample | None) -> FrameSample:
    tl = ctx.timeline
    st = _camera_at(ctx, t)
    sensor = optics.SENSORS[tl.sensor]
    cam = moves.Camera(position=st.pos, aim=st.aim, focal_mm=st.focal_mm, aperture_f=tl.aperture_f, sensor=tl.sensor)

    # Kinematics from the previous frames; a cut is a jump, not a speed.
    dt = 1.0 / tl.fps
    cut = st.cut or (prev is not None and _crosses_cut(tl, prev.t, t))
    speed = 0.0 if prev is None or cut else float(np.linalg.norm(st.pos - np.asarray(prev.cam))) / dt
    accel = 0.0 if prev is None or cut or prev.cut else (speed - prev.speed_mps) / dt

    actors = actors_at(tl, t)
    pieces = pieces_at(tl, t)

    # The subject: who the key says, else the nearest actor in frame, else the nearest actor.
    subject = st.look_at if st.look_at in actors else None
    if subject is None and actors:
        best = None
        for name, (x, y, _) in actors.items():
            eye = np.array([x, y, ctx.floor_z + EYELINE_RATIO * tl.actors[name].height_m])
            inside, _, _ = cam.frames(eye)
            d = float(np.linalg.norm(eye - st.pos))
            key = (0 if inside else 1, d)
            if best is None or key < best[0]:
                best = (key, name)
        subject = best[1]

    subj_obj = None
    if subject is not None:
        x, y, _ = actors[subject]
        subj_obj = moves.Subject(position=np.array([x, y, ctx.floor_z]), height_m=tl.actors[subject].height_m)

    problems: list[Problem] = []
    distance = framed_h = fill = None
    shot_key = shot_name = None
    in_frame = None
    if ctx.maps is not None and ctx.occupancy is not None:
        grid, origin, cell = ctx.occupancy
        gear = ctx.gear if ctx.rig.kind in ("tripod", "slider", "dolly", "handheld", "steadicam") else None
        beat = _move_beat(ctx.maps, grid, origin, cell, 0.0, cam, subj_obj, gear)
        problems.extend(_from_beat(beat))
        if subj_obj is not None:
            distance = beat.distance_m
            framed_h = beat.framed_height_m
            in_frame = beat.subject_in_frame
        # Headroom for the lens itself, with a margin for the matte box.
        lens_h = float(st.pos[2] - ctx.floor_z)
        headroom = float(ctx.maps.sample(ctx.maps.headroom_m, st.pos[:2])[0])
        if lens_h > headroom:
            problems.append(Problem(f"the lens at {lens_h:.2f} m is above the {headroom:.2f} m of headroom here"))
        elif lens_h + HEADROOM_MARGIN_M > headroom:
            problems.append(Problem(
                f"the lens at {lens_h:.2f} m leaves {headroom - lens_h:.2f} m under the {headroom:.2f} m of headroom, "
                f"less than the {HEADROOM_MARGIN_M:.2f} m margin", "warn"))
    elif subj_obj is not None:
        eye = subj_obj.eyeline
        distance = float(np.linalg.norm(eye - st.pos))
        framed_h = optics.framed_height_m(sensor, st.focal_mm, max(distance, 1e-3))
        in_frame = cam.frames(eye)[0]
        if not in_frame:
            problems.append(Problem("the subject's eyeline falls outside the frame", "warn"))

    if framed_h is not None and framed_h > 0:
        size = optics.classify_framing(framed_h)
        shot_key, shot_name = size.key, size.name
        fill = float(tl.actors[subject].height_m / framed_h)

    problems.extend(ctx.rig.frame_problems(ctx, cam, speed, accel, cut))

    focus = float(distance) if distance else float(np.linalg.norm(st.aim - st.pos))
    focus = max(focus, 0.05)
    near, far, hyper = optics.depth_of_field(sensor, st.focal_mm, tl.aperture_f, focus)
    dof = {"near_m": round(near, 3), "far_m": None if not math.isfinite(far) else round(far, 3),
           "infinite": not math.isfinite(far), "hyperfocal_m": round(hyper, 3)}
    fov_h, fov_v = optics.field_of_view_deg(sensor, st.focal_mm)

    fr = FrameSample(
        t=float(t), cam=[float(v) for v in st.pos], aim=[float(v) for v in st.aim], focal_mm=float(st.focal_mm),
        pan_deg=cam.pan_deg, tilt_deg=cam.tilt_deg, roll_deg=float(st.roll_deg),
        speed_mps=float(speed), accel_mps2=float(accel), cut=bool(cut),
        actors={k: [v[0], v[1], v[2]] for k, v in actors.items()}, pieces=pieces,
        subject=subject, distance_m=distance, framed_height_m=framed_h, subject_fill=fill,
        shot_size=shot_key, shot_name=shot_name, in_frame=in_frame, focus_m=focus, dof=dof,
        fov_h_deg=float(fov_h), fov_v_deg=float(fov_v),
    )
    for p in problems:
        fr.add(p)
    return fr


def _crosses_cut(tl: Timeline, t0: float, t1: float) -> bool:
    ts = [k.t for k in tl.camera_keys]
    for a, b in zip(ts, ts[1:]):
        if b - a <= 1e-9 and t0 < a <= t1:
            return True
    return False


def _segments(frames: list[FrameSample], dt: float) -> list[dict]:
    """Per-frame problems merged into runs of consecutive frames."""
    out: list[dict] = []
    open_runs: dict[str, dict] = {}
    for k, fr in enumerate(frames):
        texts = {p.text: p.severity for p in fr.problems}
        for text in list(open_runs):
            if text not in texts:
                out.append(open_runs.pop(text))
        for text, sev in texts.items():
            if text in open_runs:
                open_runs[text]["t_end"] = round(fr.t, 4)
                open_runs[text]["frames"] += 1
            else:
                open_runs[text] = {"t_start": round(fr.t, 4), "t_end": round(fr.t, 4), "text": text,
                                   "severity": sev, "frames": 1}
    out.extend(open_runs.values())
    for s in out:
        s["t_end"] = round(max(s["t_end"], s["t_start"] + dt * 0.999), 4) if s["frames"] == 1 else s["t_end"]
    out.sort(key=lambda s: (s["t_start"], 0 if s["severity"] == "fail" else 1, s["text"]))
    return out


def _summary(tl: Timeline, frames: list[FrameSample], segments: list[dict]) -> dict:
    travel = 0.0
    for a, b in zip(frames, frames[1:]):
        if not b.cut:
            travel += float(np.linalg.norm(np.subtract(b.cam, a.cam)))
    actor_travel: dict[str, float] = {}
    for a, b in zip(frames, frames[1:]):
        for name, pb in b.actors.items():
            pa = a.actors.get(name)
            if pa is not None:
                actor_travel[name] = actor_travel.get(name, 0.0) + math.dist(pa[:2], pb[:2])
    sized = [f for f in frames if f.framed_height_m]
    size_range = None
    if sized:
        tight = min(sized, key=lambda f: f.framed_height_m)
        wide = max(sized, key=lambda f: f.framed_height_m)
        size_range = [tight.shot_size, wide.shot_size]
    dists = [f.distance_m for f in frames if f.distance_m]
    fails = [s for s in segments if s["severity"] == "fail"]
    warns = [s for s in segments if s["severity"] == "warn"]
    worst = None
    if fails:
        worst = max(fails, key=lambda s: s["frames"])
    elif warns:
        worst = max(warns, key=lambda s: s["frames"])
    return {
        "title": tl.title,
        "rig": tl.rig.kind,
        "duration_s": round(tl.duration_s, 3),
        "fps": tl.fps,
        "frames": len(frames),
        "camera_travel_m": round(travel, 3),
        "actor_travel_m": {k: round(v, 3) for k, v in actor_travel.items()},
        "size_range": size_range,
        "distance_range_m": None if not dists else [round(min(dists), 3), round(max(dists), 3)],
        "focal_range_mm": [round(min(f.focal_mm for f in frames), 2), round(max(f.focal_mm for f in frames), 2)],
        "problem_frames": sum(1 for f in frames if f.problems),
        "fail_frames": sum(1 for f in frames if f.severity == "fail"),
        "worst": None if worst is None else worst["text"],
        "worst_severity": "ok" if worst is None else worst["severity"],
        "feasible": not fails,
        "cuts": sum(1 for f in frames if f.cut),
        "notes": list(tl.notes),
    }


def evaluate(timeline: Timeline, twin=None, *, maps: space.FloorMaps | None = None,
             occupancy=None, pieces=None) -> Evaluation:
    """Solve, then sample at the frame rate and ask the room about every frame.

    `twin` may be None for a purely kinematic evaluation (no fit, headroom or
    sightline checks; framing and depth of field still computed). `pieces` is
    an optional list of `product.setpieces.SetPiece` (or their dicts); when
    given with a twin, the room is dressed with them before the maps are
    built. Piece keys on the timeline are carried per frame for the viewer;
    the collision checks use the pieces where they stand at the dressing, not
    where a key moves them mid-shot.
    """
    ctx = _build_ctx(copy.deepcopy(timeline), twin, maps, occupancy, pieces)
    tl = ctx.timeline
    if any(p.keys and any(abs(k.dx) + abs(k.dy) + abs(k.dyaw_deg) > 1e-9 for k in p.keys) for p in tl.pieces):
        note = "set pieces are keyed; the checks use them where they stand in the dressing, not mid-move"
        if note not in tl.notes:
            tl.notes.append(note)
    frames: list[FrameSample] = []
    prev = prev2 = None
    for k in range(ctx.n_frames):
        t = min(k / tl.fps, tl.duration_s)
        fr = _frame(ctx, t, prev, prev2)
        frames.append(fr)
        prev2, prev = prev, fr
    ctx.rig.path_problems(ctx, frames)
    segments = _segments(frames, 1.0 / tl.fps)
    return Evaluation(timeline=tl, frames=frames, segments=segments, summary=_summary(tl, frames, segments))


def sample_time(timeline: Timeline, t: float, twin=None, *, maps: space.FloorMaps | None = None,
                occupancy=None) -> dict:
    """One frame's worth of answers at time `t` (the preview's scrub). With a
    twin the room's checks run; without, the sample is kinematic only. Agrees
    with `evaluate` at frame times, shake included."""
    ctx = _build_ctx(copy.deepcopy(timeline), twin, maps, occupancy, None)
    tl = ctx.timeline
    t = float(min(max(t, 0.0), tl.duration_s))
    dt = 1.0 / tl.fps
    prev2 = _frame(ctx, max(0.0, t - 2 * dt), None, None) if t >= 2 * dt - 1e-9 else None
    prev = _frame(ctx, max(0.0, t - dt), prev2, None) if t >= dt - 1e-9 else None
    fr = _frame(ctx, t, prev, prev2)
    d = fr.to_dict()
    d["checked_against_twin"] = twin is not None
    return d


# ---------------------------------------------------------------------------
# from a rehearsal
# ---------------------------------------------------------------------------


def from_rehearsal(rehearsal_dict: dict, *, title: str | None = None) -> Timeline:
    """A `Rehearsal.to_dict()` becomes a timeline: a camera key per shot
    (start, dolly end, and a hold to the cut), actor keys at every beat
    boundary and route waypoint, linear interpolation. One rig per timeline:
    a single dolly shot keeps the dolly; static setups at one spot are a
    tripod; anything else is a free handheld camera with the rehearsal's cuts
    kept as same-time keys, and the choice is written into the notes."""
    d = rehearsal_dict
    scene = d.get("scene") or {}
    frames = d.get("frames") or []
    shots = d.get("shots") or []
    beats = d.get("beats") or []
    duration = _f(d.get("duration_s"), 0.0)
    dt = _f(d.get("dt_s"), 0.25)
    notes: list[str] = []

    def frame_at(t: float) -> dict | None:
        if not frames:
            return None
        k = int(round(t / dt)) if dt > 0 else 0
        return frames[min(max(k, 0), len(frames) - 1)]

    def shot_frames(number: int) -> list[dict]:
        return [f for f in frames if f.get("camera") and f["camera"].get("shot") == number]

    cam_keys: list[CameraKey] = []
    for idx, st in enumerate(shots):
        setup = st.get("setup") or {}
        own = shot_frames(st["number"])
        if not own:
            continue
        subject = setup.get("subject")
        focal = _f(setup.get("focal_mm"), DEFAULT_FOCAL_MM)
        first, last = own[0], own[-1]

        z = _xyz(first["camera"]["pos"])[2]
        start_xy = [_f(setup.get("cam_x"), first["camera"]["pos"][0]), _f(setup.get("cam_y"), first["camera"]["pos"][1])]
        end_xy = [float(v) for v in setup["dolly_to"]] if setup.get("dolly_to") else start_xy

        def key(t, xy, f):
            # Positions come from the setup itself (exact); the aim from the
            # nearest rehearsal frame, which only matters when the subject is
            # not in the cast and look_at cannot take over.
            return CameraKey(t=float(t), pos=[xy[0], xy[1], z], aim=_xyz(f["camera"]["aim"]),
                             look_at=subject if subject in (scene.get("cast") or {}) else None,
                             focal_mm=focal, ease="linear")

        cam_keys.append(key(st["t_start"], start_xy, first))
        if setup.get("dolly_to"):
            end_t = _f(st.get("beat_end"), st["t_start"])
            parked = frame_at(end_t) or last
            if end_t > st["t_start"] + 1e-9:
                cam_keys.append(key(end_t, end_xy, parked))
        if idx < len(shots) - 1 and st["t_end"] > cam_keys[-1].t + 1e-9:
            cam_keys.append(key(st["t_end"], end_xy, last))

    cast = scene.get("cast") or {}
    actors: dict[str, ActorTrack] = {}
    for name, h in cast.items():
        actors[name] = ActorTrack(height_m=_f(h, SUBJECT_HEIGHT_M))
    seen_keys: dict[str, set] = {n: set() for n in actors}

    def add_actor_key(name: str, t: float, xyf) -> None:
        if name not in actors:
            actors[name] = ActorTrack(height_m=SUBJECT_HEIGHT_M)
            seen_keys[name] = set()
        tag = round(t, 4)
        if tag in seen_keys[name]:
            return
        seen_keys[name].add(tag)
        actors[name].keys.append(ActorKey(t=float(t), pos=[float(xyf[0]), float(xyf[1])], facing_deg=float(xyf[2])))

    for bt in beats:
        for t in (bt.get("t_start"), bt.get("t_end")):
            if t is None:
                continue
            f = frame_at(float(t))
            if f is None:
                continue
            for name, xyf in (f.get("actors") or {}).items():
                add_actor_key(name, float(t), xyf)
        route = bt.get("route")
        if route and len(route) > 2 and bt.get("t_end", 0) > bt.get("t_start", 0):
            pts = np.asarray(route, dtype=np.float64)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            if cum[-1] > 1e-9:
                span = bt["t_end"] - bt["t_start"]
                for i in range(1, len(pts) - 1):
                    t = bt["t_start"] + span * cum[i] / cum[-1]
                    head = math.degrees(math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0]))
                    add_actor_key(bt["actor"], float(t), (pts[i][0], pts[i][1], head))
    for tr in actors.values():
        tr.keys.sort(key=lambda k: k.t)

    # One rig for the whole timeline.
    dolly_shots = [s for s in shots if (s.get("setup") or {}).get("dolly_to")]
    rig = Rig("handheld", {})
    if len(shots) == 1 and dolly_shots:
        gk = (dolly_shots[0].get("setup") or {}).get("gear_key") or "doorway-dolly"
        rig = Rig("dolly", {"gear_key": gk})
        notes.append(f"rig: dolly ({gk}), the rehearsal's one shot rides a dolly")
    elif shots and not dolly_shots:
        positions = {tuple(round(v, 3) for v in k.pos) for k in cam_keys}
        if len(positions) == 1:
            rig = Rig("tripod", {})
            notes.append("rig: tripod, every setup stands in one place")
        else:
            notes.append(f"rig: handheld, the rehearsal cuts between {len(positions)} static setups and one rig cannot; "
                         "the cuts are kept as same-time keys")
    elif shots:
        notes.append("rig: handheld, the rehearsal mixes dolly and static setups; the dolly's own check lives in the rehearsal")
    else:
        notes.append("the rehearsal names no camera; a single key stands at the room's origin")
        cam_keys.append(CameraKey(t=0.0, pos=[0.0, 0.0, 1.5], aim=[1.0, 0.0, 1.5], ease="linear"))

    exits = [bt for bt in beats if bt.get("kind") == "exit"]
    for bt in exits:
        notes.append(f"{bt.get('actor')} exits at {_f(bt.get('t_end')):.1f} s; the timeline holds them at the exit mark "
                     "(actor keys carry no presence flag)")

    tl = Timeline(
        title=title or str(scene.get("title") or "Rehearsal"),
        fps=DEFAULT_FPS,
        duration_s=max(duration, dt, 1.0 / DEFAULT_FPS),
        rig=rig,
        interp="linear",
        camera_keys=cam_keys,
        actors=actors,
        notes=notes,
    )
    return tl


# ---------------------------------------------------------------------------
# Blender
# ---------------------------------------------------------------------------


def look_rotation(cam_pos, aim, roll_deg: float = 0.0) -> np.ndarray:
    """World-from-camera rotation for a Blender camera at `cam_pos` looking at
    `aim`, world Z up. Blender cameras look down their local -Z with local +Y
    up, so the columns are (right, up, -forward). Positive roll dips the right
    side of the frame (a clockwise Dutch seen from behind the camera)."""
    cam = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    a = np.asarray(aim, dtype=np.float64).reshape(3)
    fwd = a - cam
    n = float(np.linalg.norm(fwd))
    fwd = fwd / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    rn = float(np.linalg.norm(right))
    right = right / rn if rn > 1e-9 else np.array([1.0, 0.0, 0.0])
    up = np.cross(right, fwd)
    r = math.radians(roll_deg)
    right_r = math.cos(r) * right - math.sin(r) * up
    up_r = math.sin(r) * right + math.cos(r) * up
    return np.column_stack([right_r, up_r, -fwd])


def euler_xyz(R: np.ndarray) -> tuple[float, float, float]:
    """Blender's default XYZ Euler (R = Rz · Ry · Rx), radians, from a rotation matrix."""
    R = np.asarray(R, dtype=np.float64)
    sy = -float(R[2, 0])
    sy = max(-1.0, min(1.0, sy))
    b = math.asin(sy)
    if abs(math.cos(b)) > 1e-7:
        a = math.atan2(R[2, 1], R[2, 2])
        c = math.atan2(R[1, 0], R[0, 0])
    else:
        a = math.atan2(-R[1, 2], R[1, 1])
        c = 0.0
    return a, b, c


def rotation_from_euler_xyz(a: float, b: float, c: float) -> np.ndarray:
    """Rz(c) · Ry(b) · Rx(a): what Blender builds from `rotation_euler` in XYZ mode."""
    ca, sa, cb, sb, cc, sc = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=np.float64)
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=np.float64)
    Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _blender_frames(ev: Evaluation) -> list[dict]:
    out = []
    for k, f in enumerate(ev.frames):
        a, b, c = euler_xyz(look_rotation(f.cam, f.aim, f.roll_deg))
        out.append({"f": k + 1, "loc": _r(f.cam, 5), "rot": [round(a, 6), round(b, 6), round(c, 6)],
                    "lens": round(f.focal_mm, 3), "focus": round(f.focus_m, 4)})
    return out


def _blender_actors(ev: Evaluation, floor_z: float) -> dict[str, dict]:
    out = {}
    for name, tr in ev.timeline.actors.items():
        keys = []
        for k, f in enumerate(ev.frames):
            s = f.actors.get(name)
            if s is None:
                continue
            # The capsule's canonical facing is +Y (cast.figure); the yaw that
            # turns it to `facing_deg` is facing - 90.
            keys.append({"f": k + 1, "loc": [round(s[0], 5), round(s[1], 5), round(floor_z, 5)],
                         "yaw": round(math.radians(s[2] - 90.0), 6)})
        out[name] = {"height_m": round(float(tr.height_m), 4), "keys": keys}
    return out


def blender_summary(timeline: Timeline, evaluation: Evaluation | None = None) -> list[str]:
    """What `blender_script` creates, in the order it creates it."""
    tl = evaluation.timeline if evaluation is not None else timeline
    n = len(evaluation.frames) if evaluation is not None else int(round(tl.duration_s * tl.fps)) + 1
    sensor = optics.SENSORS.get(tl.sensor)
    out = [
        "imports the room GLB under an empty 'Locaish Twin' rotated by whichever axis correction lands the import on the twin's bounding box (Blender reads glTF as Y-up)",
        f"sets the scene to {tl.fps} fps, frames 1-{n} ({tl.duration_s:g} s), metric units",
        f"a camera 'Locaish Camera' with a {sensor.width_mm:g} x {sensor.height_mm:g} mm sensor ({sensor.name}), "
        f"lens, location, rotation_euler and focus_distance keyed on every frame, DoF at f/{tl.aperture_f:g}"
        if sensor else "a camera keyed on every frame",
    ]
    for name, tr in tl.actors.items():
        out.append(f"a {tr.height_m:g} m capsule stand-in for {name}, location and yaw keyed per frame")
    if tl.rig.kind == "dolly" and tl.rig.params.get("track"):
        out.append(f"a poly curve 'Dolly Track' with {len(tl.rig.params['track'])} points at rail height")
    if tl.rig.kind == "jib" and tl.rig.params.get("pivot"):
        out.append("an empty 'Jib Pivot' at the pivot")
    if evaluation is not None:
        probs = [s for s in evaluation.segments]
        out.append(f"a text object 'Locaish Problems' listing {len(probs)} problem segment{'s' if len(probs) != 1 else ''}")
    else:
        out.append("a text object 'Locaish Problems'")
    return out


def blender_script(timeline: Timeline, evaluation: Evaluation, *, glb_path: str, twin=None) -> str:
    """A self-contained Blender 4.x Python script rebuilding the evaluated move.

    Units are metres and the twin's frame is used as is. The GLB written by
    `locaish export` carries the twin's Z-up vertices unconverted into a
    format Blender reads as Y-up, so the import is parented to an empty whose
    rotation is *measured*: of nine axis rotations the one that lands the
    imported bounding box on the twin's own (`twin.structure` footprint and
    floor/ceiling) is applied, and the script warns if the heights still
    disagree. The camera, actors and track are placed in twin coordinates
    directly. In background mode the script saves a .blend beside itself and
    ends with a "LOCAISH SUMMARY:" line. Verified against Blender 5.2.3
    (slotted actions; F-curves reached through the channelbag).
    """
    tl = evaluation.timeline
    sensor = optics.SENSORS[tl.sensor]
    floor_z = _floor_z(twin, None)
    frames = _blender_frames(evaluation)
    actors = _blender_actors(evaluation, floor_z)
    track = tl.rig.params.get("track") if tl.rig.kind == "dolly" else None
    rail_z = floor_z + _f(tl.rig.params.get("rail_height_m"), 0.0) if track else floor_z
    pivot = None
    if tl.rig.kind == "jib" and tl.rig.params.get("pivot"):
        px, py = tl.rig.params["pivot"]
        pivot = [float(px), float(py), floor_z + _f(tl.rig.params.get("base_height_m"), 1.0)]
    problems = [f"{s['t_start']:.2f}-{s['t_end']:.2f} s  [{s['severity']}]  {s['text']}" for s in evaluation.segments]
    bbox = None
    if twin is not None and getattr(twin, "structure", None) is not None:
        st = twin.structure
        if st.footprint is not None and len(st.footprint) >= 3:
            fp = np.asarray(st.footprint, dtype=np.float64)
            top = float(st.ceiling_z) if st.ceiling_z is not None else float(st.floor_z) + 2.7
            bbox = [[float(fp[:, 0].min()), float(fp[:, 1].min()), float(st.floor_z)],
                    [float(fp[:, 0].max()), float(fp[:, 1].max()), top]]
    res_x = 1920
    res_y = int(round(res_x / sensor.aspect))

    data = {
        "title": tl.title, "fps": tl.fps, "n_frames": len(frames), "glb": str(glb_path),
        "sensor_width_mm": sensor.width_mm, "sensor_height_mm": sensor.height_mm,
        "aperture_f": tl.aperture_f, "res": [res_x, res_y],
        "frames": frames, "actors": actors, "track": track, "rail_z": rail_z, "pivot": pivot,
        "problems": problems, "bbox": bbox, "floor_z": floor_z,
        "rig": tl.rig.kind,
    }
    payload = json.dumps(_jsonable(data), indent=None)
    return _BLENDER_TEMPLATE.replace("__DATA__", payload)


_BLENDER_TEMPLATE = r'''# Locaish camera simulation -> Blender 4.x
# Generated by locaish.film.sim.blender_script. Run inside Blender:
#   blender --python this_script.py
# Units: metres, Z up (the twin's frame). Every number here traces to the
# evaluated timeline; the problems text lists what the room objected to.
import json
import math

import os

import bpy
from mathutils import Euler, Vector

DATA = json.loads(r"""__DATA__""")

scene = bpy.context.scene
scene.unit_settings.system = "METRIC"
scene.unit_settings.scale_length = 1.0
scene.render.fps = int(DATA["fps"])
scene.render.fps_base = 1.0
scene.frame_start = 1
scene.frame_end = int(DATA["n_frames"])
scene.render.resolution_x, scene.render.resolution_y = DATA["res"]

coll = bpy.data.collections.new("Locaish")
scene.collection.children.link(coll)


def link(obj):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)
    return obj


# -- the room ---------------------------------------------------------------
# The GLB holds the twin's Z-up vertices unconverted, and Blender's importer
# treats glTF as Y-up and rotates it. Rather than trust either convention,
# the import is measured: the correction applied to the parent empty is the
# axis rotation that lands the room's bounding box on the twin's own.
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=DATA["glb"])
imported = [o for o in bpy.data.objects if o not in before]
room = bpy.data.objects.new("Locaish Twin", None)
link(room)
bpy.context.view_layer.update()
corners = [o.matrix_world @ Vector(c) for o in imported if o.type == "MESH" for c in o.bound_box]


def bbox_of(points):
    lo = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    hi = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return lo, hi


CANDIDATES = [("none", (0.0, 0.0, 0.0)),
              ("X -90", (-math.pi / 2, 0.0, 0.0)), ("X +90", (math.pi / 2, 0.0, 0.0)), ("X 180", (math.pi, 0.0, 0.0)),
              ("Y -90", (0.0, -math.pi / 2, 0.0)), ("Y +90", (0.0, math.pi / 2, 0.0)),
              ("Z -90", (0.0, 0.0, -math.pi / 2)), ("Z +90", (0.0, 0.0, math.pi / 2)), ("Z 180", (0.0, 0.0, math.pi))]
correction = ("X -90", (-math.pi / 2, 0.0, 0.0))
room_lo = room_hi = None
if corners:
    if DATA.get("bbox"):
        want_lo, want_hi = Vector(DATA["bbox"][0]), Vector(DATA["bbox"][1])
        best = None
        for name, eul in CANDIDATES:
            R = Euler(eul, "XYZ").to_matrix()
            lo, hi = bbox_of([R @ p for p in corners])
            score = (lo - want_lo).length + (hi - want_hi).length
            if best is None or score < best[0]:
                best = (score, name, eul)
        correction = (best[1], best[2])
        print("Locaish: room import corrected by rotation %s (bbox mismatch %.2f m)" % (best[1], best[0]))
    room.rotation_euler = correction[1]
    for o in imported:
        if o.parent is None:
            o.parent = room
    bpy.context.view_layer.update()
    room_lo, room_hi = bbox_of([o.matrix_world @ Vector(c) for o in imported if o.type == "MESH" for c in o.bound_box])
    if DATA.get("bbox"):
        want_h = DATA["bbox"][1][2] - DATA["bbox"][0][2]
        got_h = room_hi.z - room_lo.z
        if abs(got_h - want_h) > 0.25 * max(want_h, 0.1):
            print("Locaish: WARNING the imported room is %.2f m tall, the twin %.2f m; check the GLB" % (got_h, want_h))
else:
    print("Locaish: WARNING the GLB imported no mesh")

# -- the camera -------------------------------------------------------------
cam_data = bpy.data.cameras.new("Locaish Camera")
cam_data.sensor_fit = "HORIZONTAL"
cam_data.sensor_width = DATA["sensor_width_mm"]
cam_data.sensor_height = DATA["sensor_height_mm"]
cam_data.dof.use_dof = True
cam_data.dof.aperture_fstop = DATA["aperture_f"]
cam = bpy.data.objects.new("Locaish Camera", cam_data)
link(cam)
scene.camera = cam
cam.rotation_mode = "XYZ"
for fr in DATA["frames"]:
    f = fr["f"]
    cam.location = fr["loc"]
    cam.rotation_euler = fr["rot"]
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_euler", frame=f)
    cam_data.lens = fr["lens"]
    cam_data.keyframe_insert("lens", frame=f)
    cam_data.dof.focus_distance = fr["focus"]
    cam_data.dof.keyframe_insert("focus_distance", frame=f)


def fcurves_of(obj):
    """An object's animation F-curves across Blender versions: slotted
    actions (4.4+) keep them in a channelbag; older actions on the action."""
    ad = obj.animation_data
    if ad is None or ad.action is None:
        return []
    act = ad.action
    if hasattr(act, "fcurves"):
        return list(act.fcurves)
    try:
        return list(act.layers[0].strips[0].channelbag(ad.action_slot, ensure=True).fcurves)
    except Exception as e:  # noqa: BLE001
        print("Locaish: could not reach the F-curves (%s)" % e)
        return []


camera_keyframes = 0
for fc in fcurves_of(cam) + fcurves_of(cam_data):
    for kp in fc.keyframe_points:
        kp.interpolation = "LINEAR"
    if fc.data_path == "location" and fc.array_index == 0:
        camera_keyframes = len(fc.keyframe_points)

# -- the cast ---------------------------------------------------------------
for name, actor in DATA["actors"].items():
    h = float(actor["height_m"])
    root = bpy.data.objects.new("Actor " + name, None)
    link(root)
    root.empty_display_type = "PLAIN_AXES"
    root.rotation_mode = "XYZ"
    r = 0.11 * h / 1.75
    body_h = h - 2.0 * r
    bpy.ops.mesh.primitive_cylinder_add(radius=0.17 * h / 1.75, depth=body_h, location=(0, 0, body_h / 2.0))
    body = bpy.context.active_object
    body.name = name + " body"
    link(body)
    body.parent = root
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(0, 0, h - r))
    head = bpy.context.active_object
    head.name = name + " head"
    link(head)
    head.parent = root
    for k in actor["keys"]:
        root.location = k["loc"]
        root.rotation_euler = (0.0, 0.0, k["yaw"])
        root.keyframe_insert("location", frame=k["f"])
        root.keyframe_insert("rotation_euler", frame=k["f"])

# -- the track and the pivot ------------------------------------------------
if DATA.get("track"):
    curve = bpy.data.curves.new("Dolly Track", "CURVE")
    curve.dimensions = "3D"
    spline = curve.splines.new("POLY")
    pts = DATA["track"]
    spline.points.add(len(pts) - 1)
    for p, (x, y) in zip(spline.points, pts):
        p.co = (x, y, DATA["rail_z"], 1.0)
    curve.bevel_depth = 0.02
    track = bpy.data.objects.new("Dolly Track", curve)
    link(track)
if DATA.get("pivot"):
    piv = bpy.data.objects.new("Jib Pivot", None)
    link(piv)
    piv.empty_display_type = "SPHERE"
    piv.empty_display_size = 0.15
    piv.location = DATA["pivot"]

# -- what the room said -----------------------------------------------------
text = bpy.data.curves.new("Locaish Problems", "FONT")
lines = [DATA["title"] + "  (" + DATA["rig"] + ")"] + (DATA["problems"] or ["no problems: every frame clears the room"])
text.body = "\n".join(lines)
text.size = 0.12
label = bpy.data.objects.new("Locaish Problems", text)
link(label)
label.location = (0.0, 0.0, DATA["floor_z"] + 0.02)

# -- done -------------------------------------------------------------------
created = [o.name for o in coll.objects]
extents = ("%.2f x %.2f x %.2f m" % tuple(room_hi[i] - room_lo[i] for i in range(3))) if room_lo is not None else "no room"
print("LOCAISH SUMMARY: %d objects created (%s); camera keyframes %d of %d frames; room %s; correction %s" % (
    len(created), ", ".join(created[:6]) + (", ..." if len(created) > 6 else ""),
    camera_keyframes, len(DATA["frames"]), extents, correction[0]))
if bpy.app.background:
    try:
        base = os.path.splitext(os.path.abspath(__file__))[0] if "__file__" in globals() else os.path.join(os.getcwd(), "locaish_sim")
        out = base + ".blend"
        bpy.ops.wm.save_as_mainfile(filepath=out)
        print("Locaish: saved %s" % out)
    except Exception as e:  # noqa: BLE001
        print("Locaish: could not save the .blend (%s)" % e)
'''
