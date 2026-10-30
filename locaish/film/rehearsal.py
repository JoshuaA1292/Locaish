"""Running a scene inside the measured room: bodies, time, the camera and the sun.

A twin is a thing you look at. A rehearsal is the twin *run*: the cast walks
the standable floor around the real furniture at walking pace, lines take the
time they take, the camera holds a setup or rides a dolly that physically
fits, the sun is where the ephemeris puts it at the shoot hour, and the
shoot-day schedule falls out of the timing. Everything here is deterministic
and model-free. A model (Muse Wonder, in the hackathon build) produces the
`Scene` -- beats, marks, cast, shot specs -- and this engine answers from
geometry, a solar ephemeris and the equipment catalogue. It cannot invent a
number, which is the house rule.

The pieces, in order:

**Blocking** -- a move is an A* path on the walkable grid (`space.floor_maps`
cells with 30 cm of clearance), smoothed by string-pulling against the
occupancy volume. If no path exists the beat is marked blocked, the actor
jumps to the mark so the timeline continues, and the rehearsal says so.

**Timing** -- walking is 1.2 m/s along the path; a line lasts
`0.35 + words / 2.7` seconds, which is an ordinary conversational rate with a
breath at the front. The timeline is sampled every quarter second into
`Frame`s carrying every actor's position and facing, the camera, the framing
it is holding and the sun.

**Camera** -- a beat's `ShotSpec` puts the camera somewhere, on a lens, aimed
at a named actor's eyeline; `dolly_to` moves it over the beat on a dolly from
the catalogue, and `moves.simulate` checks the run exactly as the coverage
planner would. Shot size per frame comes from the same framed-height geometry
the sweep uses, so "the 50 mm holds a medium until she reaches A, then it is a
close-up" is a measured statement.

**Sun** -- with a georeference and a shoot date and hour, each frame is tagged
with whether any window is passing direct sun and which. Times follow
`daylight`'s convention: local solar time, solar noon at 12:00.

**Schedule** -- shots whose look depends on a window (key or backlight) are
placed inside that window's direct-sun interval on the shoot date; the rest
fill the gaps. Each setup costs its setup minutes plus takes.
"""

from __future__ import annotations

import heapq
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import date as _date
from datetime import datetime, time as _time, timedelta
from pathlib import Path
from typing import Callable

import numpy as np

from ..types import Twin
from . import daylight, equipment, moves, optics, render, space
from .coverage import _esc, floor_plan_svg
from .render import EYELINE_RATIO

# How fast an actor crosses a room when the direction says "walks". Ordinary
# indoor walking pace; a hurry or a shuffle is a direction, not a default.
WALK_SPEED_MPS = 1.2

# A spoken line: a breath, then about 160 words a minute.
LINE_LEAD_S = 0.35
WORDS_PER_SECOND = 2.7

# Entering and leaving are a step through a doorway, not an instant.
ENTER_SECONDS = 0.5
DEFAULT_HOLD_S = 1.0

# Timeline sampling.
DT_S = 0.25

# A body needs this much clear radius to walk a cell; a little less than the
# standing radius, because people turn their shoulders through gaps.
WALK_CLEARANCE_M = 0.30

# A mark that falls on an unwalkable cell is snapped to the nearest walkable
# one within this distance -- a mark "at the door" is in the doorway, where
# the frame's returns make the cell read as occupied.
SNAP_RADIUS_M = 0.6

# Height at which a walking path is checked for things in the way.
BODY_CHECK_HEIGHT_M = 1.2

# Another actor is an occluder too. A standing body is taken as a vertical
# cylinder of this radius from the floor to the actor's height; one whose
# cylinder cuts the camera-to-eyeline segment, nearer the lens than the
# subject, blocks the shot. One closer than LENS_CLEARANCE_M to the lens fills
# the frame whatever the lens is pointed at.
OCCLUDER_RADIUS_M = 0.30
LENS_CLEARANCE_M = 0.6

# A grip pushing a loaded dolly on track; faster than this is a stunt.
MAX_DOLLY_SPEED_MPS = 1.5

# Scheduling.
DEFAULT_SETUP_MINUTES = 25
MIN_TAKES_MINUTES = 5
TAKES_PER_BEAT = 3
STRIKE_MINUTES = 10

# A window counts as the shot's light when, seen from the subject, it sits
# within this many degrees of the camera axis (key light) or of the axis
# behind the subject (backlight, in the background of the frame).
KEY_ARC_DEG = 60.0
BACKLIGHT_ARC_DEG = 60.0

BEAT_KINDS = ("line", "move", "hold", "enter", "exit")


# ---------------------------------------------------------------------------
# the scene: input
# ---------------------------------------------------------------------------


@dataclass
class ShotSpec:
    """Where the camera is for a beat, and what it does."""

    cam_xy: tuple[float, float]
    height_m: float
    focal_mm: float
    subject: str
    size: str | None = None
    dolly_to: tuple[float, float] | None = None
    gear_key: str | None = None

    def to_dict(self) -> dict:
        return {
            "cam_xy": [float(self.cam_xy[0]), float(self.cam_xy[1])],
            "height_m": float(self.height_m),
            "focal_mm": float(self.focal_mm),
            "subject": self.subject,
            "size": self.size,
            "dolly_to": None if self.dolly_to is None else [float(self.dolly_to[0]), float(self.dolly_to[1])],
            "gear_key": self.gear_key,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ShotSpec":
        dolly = d.get("dolly_to")
        return cls(
            cam_xy=(float(d["cam_xy"][0]), float(d["cam_xy"][1])),
            height_m=float(d.get("height_m", 1.5)),
            focal_mm=float(d.get("focal_mm", 35.0)),
            subject=str(d["subject"]),
            size=d.get("size") or None,
            dolly_to=None if dolly is None else (float(dolly[0]), float(dolly[1])),
            gear_key=d.get("gear_key") or None,
        )


@dataclass
class Beat:
    """One unit of the scene: someone says, walks, waits, arrives or leaves."""

    number: int
    kind: str
    actor: str
    text: str | None = None
    to_mark: str | None = None
    seconds: float | None = None
    shot: ShotSpec | None = None

    def __post_init__(self) -> None:
        if self.kind not in BEAT_KINDS:
            raise ValueError(f"beat {self.number}: unknown kind {self.kind!r}; have {BEAT_KINDS}")

    def to_dict(self) -> dict:
        return {
            "number": int(self.number),
            "kind": self.kind,
            "actor": self.actor,
            "text": self.text,
            "to_mark": self.to_mark,
            "seconds": self.seconds,
            "shot": None if self.shot is None else self.shot.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Beat":
        shot = d.get("shot")
        return cls(
            number=int(d["number"]),
            kind=str(d["kind"]),
            actor=str(d.get("actor") or ""),
            text=d.get("text"),
            to_mark=d.get("to_mark"),
            seconds=None if d.get("seconds") is None else float(d["seconds"]),
            shot=None if shot is None else ShotSpec.from_dict(shot),
        )


@dataclass
class Scene:
    """What the director wants, typed: beats on marks, a cast with heights,
    and when it shoots. `strikes` lists set pieces to remove before a shot:
    `[{"before_shot": 2, "piece": "sofa"}]`."""

    title: str
    beats: list[Beat]
    marks: dict[str, tuple[float, float]]
    cast: dict[str, float]
    shoot_date: _date | None = None
    shoot_start: _time | None = None
    strikes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "beats": [b.to_dict() for b in self.beats],
            "marks": {k: [float(v[0]), float(v[1])] for k, v in self.marks.items()},
            "cast": {k: float(v) for k, v in self.cast.items()},
            "shoot_date": None if self.shoot_date is None else self.shoot_date.isoformat(),
            "shoot_start": None if self.shoot_start is None else self.shoot_start.strftime("%H:%M"),
            "strikes": [dict(s) for s in self.strikes],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        sd = d.get("shoot_date")
        st = d.get("shoot_start")
        return cls(
            title=str(d.get("title") or "Scene"),
            beats=[Beat.from_dict(b) for b in d.get("beats") or []],
            marks={str(k): (float(v[0]), float(v[1])) for k, v in (d.get("marks") or {}).items()},
            cast={str(k): float(v) for k, v in (d.get("cast") or {}).items()},
            shoot_date=None if not sd else (sd if isinstance(sd, _date) else _date.fromisoformat(str(sd))),
            shoot_start=None if not st else (st if isinstance(st, _time) else _parse_hhmm(str(st))),
            strikes=[dict(s) for s in d.get("strikes") or []],
        )


def _parse_hhmm(s: str) -> _time:
    h, m = s.strip().split(":")[:2]
    return _time(int(h), int(m))


# ---------------------------------------------------------------------------
# the typed scene format
# ---------------------------------------------------------------------------

_XY = r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
_MARK_RE = re.compile(r"([A-Za-z_][\w-]*)\s*=\s*" + _XY)
_CAST_RE = re.compile(r"([A-Za-z_][\w'-]*)\s*=\s*(\d+(?:\.\d+)?)")
_BEAT_RE = re.compile(r"^\s*(\d+)\s*[.)]\s*(.*?)\s*$")
_SHOT_RE = re.compile(r"\[\s*cam\b(.*?)\]\s*$", re.IGNORECASE)
_LINE_RE = re.compile(r'^([A-Za-z_][\w\'-]*)\s*:\s*["“](.*?)["”]\s*$')
_LINE_BARE_RE = re.compile(r"^([A-Za-z_][\w'-]*)\s*:\s*(.+?)\s*$")
_ENTER_RE = re.compile(r"^([A-Za-z_][\w'-]*)\s+enters(?:\s+(?:at|from|through)\s+([A-Za-z_][\w-]*))?\s*$", re.IGNORECASE)
_EXIT_RE = re.compile(r"^([A-Za-z_][\w'-]*)\s+(?:exits|leaves)(?:\s+(?:at|to|through|by)\s+([A-Za-z_][\w-]*))?\s*$", re.IGNORECASE)
_MOVE_RE = re.compile(r"^([A-Za-z_][\w'-]*)\s+(?:moves|walks|crosses|goes|runs)\s+(?:to|towards|toward)\s+([A-Za-z_][\w-]*)\s*$", re.IGNORECASE)
_HOLD_RE = re.compile(r"^(?:([A-Za-z_][\w'-]*)\s+)?(?:hold|holds|beat|pause|pauses|waits?)\s*(\d+(?:\.\d+)?)?\s*s?(?:ec(?:onds?)?)?\s*$", re.IGNORECASE)
_STRIKE_RE = re.compile(r"^(.+?)\s+before\s+(?:shot\s+|setup\s+)?(\d+)\s*$", re.IGNORECASE)


def parse_scene(text: str) -> Scene:
    """Read the typed scene format.

    ```
    TITLE: The kitchen, morning
    MARKS: A=(0.5,1.0) B=(-1.2,0.8) DOOR=(2.0,-1.5)
    CAST: Maya=1.68 Theo=1.84
    DATE: 2026-11-14
    START: 10:40
    STRIKE: sofa before 2
    1. Maya enters at DOOR
    2. Maya moves to A
    3. Theo: "You came back."   [cam (0.2,-2.0) h1.5 50mm on Theo]
    4. Maya: "I never left."    [cam (-0.8,-2.2) h1.4 35mm on Maya dolly to (0.6,-2.2)]
    5. hold 2s
    6. Maya exits to DOOR
    ```

    The bracket after a beat is the shot: `cam (x,y)`, `h<metres>`, `<n>mm`,
    `on <actor>`, optionally a size key (`cu`, `ms`...), `dolly to (x,y)` and
    `gear <catalogue-key>`. A beat without a bracket keeps the previous camera.
    """
    title = "Scene"
    marks: dict[str, tuple[float, float]] = {}
    cast: dict[str, float] = {}
    shoot_date = None
    shoot_start = None
    strikes: list[dict] = []
    beats: list[Beat] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        head, _, rest = line.partition(":")
        key = head.strip().upper()
        if key == "TITLE" and _:
            title = rest.strip() or title
            continue
        if key == "MARKS" and _:
            for name, x, y in _MARK_RE.findall(rest):
                marks[name] = (float(x), float(y))
            continue
        if key == "CAST" and _:
            for name, h in _CAST_RE.findall(rest):
                cast[name] = float(h)
            continue
        if key == "DATE" and _:
            shoot_date = _date.fromisoformat(rest.strip())
            continue
        if key == "START" and _:
            shoot_start = _parse_hhmm(rest.strip())
            continue
        if key == "STRIKE" and _:
            m = _STRIKE_RE.match(rest.strip())
            if m:
                strikes.append({"before_shot": int(m.group(2)), "piece": m.group(1).strip()})
            continue
        m = _BEAT_RE.match(line)
        if not m:
            continue
        number = int(m.group(1))
        body = m.group(2)
        shot = None
        sm = _SHOT_RE.search(body)
        if sm:
            shot = _parse_shot(sm.group(1))
            body = body[: sm.start()].strip()
        beats.append(_parse_beat(number, body, shot))

    return Scene(title=title, beats=beats, marks=marks, cast=cast,
                 shoot_date=shoot_date, shoot_start=shoot_start, strikes=strikes)


def _parse_beat(number: int, body: str, shot: ShotSpec | None) -> Beat:
    m = _LINE_RE.match(body) or None
    if m:
        return Beat(number, "line", m.group(1), text=m.group(2), shot=shot)
    m = _ENTER_RE.match(body)
    if m:
        return Beat(number, "enter", m.group(1), to_mark=m.group(2), shot=shot)
    m = _EXIT_RE.match(body)
    if m:
        return Beat(number, "exit", m.group(1), to_mark=m.group(2), shot=shot)
    m = _MOVE_RE.match(body)
    if m:
        return Beat(number, "move", m.group(1), to_mark=m.group(2), shot=shot)
    m = _HOLD_RE.match(body)
    if m:
        secs = float(m.group(2)) if m.group(2) else DEFAULT_HOLD_S
        return Beat(number, "hold", m.group(1) or "", seconds=secs, shot=shot)
    m = _LINE_BARE_RE.match(body)
    if m:
        return Beat(number, "line", m.group(1), text=m.group(2).strip('"“”'), shot=shot)
    raise ValueError(f"beat {number}: could not read {body!r}")


def _parse_shot(spec: str) -> ShotSpec:
    s = spec.strip()
    xy = re.search(_XY, s)
    if not xy:
        raise ValueError(f"shot spec {spec!r} has no camera position")
    cam_xy = (float(xy.group(1)), float(xy.group(2)))
    tail = s[xy.end():]
    h = re.search(r"\bh\s*(\d+(?:\.\d+)?)", tail)
    f = re.search(r"(\d+(?:\.\d+)?)\s*mm\b", tail)
    on = re.search(r"\bon\s+([A-Za-z_][\w'-]*)", tail)
    dolly = re.search(r"\bdolly\s+to\s+" + _XY, tail)
    gear = re.search(r"\bgear\s+([\w-]+)", tail)
    size = None
    for tok in re.findall(r"\b([a-z]{2,3})\b", tail.lower()):
        if tok in optics.SHOT_BY_KEY:
            size = tok
            break
    if on is None:
        raise ValueError(f"shot spec {spec!r} names no subject (`on <actor>`)")
    return ShotSpec(
        cam_xy=cam_xy,
        height_m=float(h.group(1)) if h else 1.5,
        focal_mm=float(f.group(1)) if f else 35.0,
        subject=on.group(1),
        size=size,
        dolly_to=(float(dolly.group(1)), float(dolly.group(2))) if dolly else None,
        gear_key=gear.group(1) if gear else None,
    )


def format_scene(scene: Scene) -> str:
    """The typed format back out, so a scene round-trips through text."""
    out = [f"TITLE: {scene.title}"]
    if scene.marks:
        out.append("MARKS: " + " ".join(f"{k}=({v[0]:g},{v[1]:g})" for k, v in scene.marks.items()))
    if scene.cast:
        out.append("CAST: " + " ".join(f"{k}={v:g}" for k, v in scene.cast.items()))
    if scene.shoot_date:
        out.append(f"DATE: {scene.shoot_date.isoformat()}")
    if scene.shoot_start:
        out.append(f"START: {scene.shoot_start.strftime('%H:%M')}")
    for s in scene.strikes:
        out.append(f"STRIKE: {s['piece']} before {s['before_shot']}")
    for b in scene.beats:
        if b.kind == "line":
            body = f'{b.actor}: "{b.text}"'
        elif b.kind == "move":
            body = f"{b.actor} moves to {b.to_mark}"
        elif b.kind == "enter":
            body = f"{b.actor} enters" + (f" at {b.to_mark}" if b.to_mark else "")
        elif b.kind == "exit":
            body = f"{b.actor} exits" + (f" to {b.to_mark}" if b.to_mark else "")
        else:
            body = (f"{b.actor} " if b.actor else "") + f"hold {b.seconds if b.seconds is not None else DEFAULT_HOLD_S:g}s"
        if b.shot is not None:
            sp = b.shot
            spec = f"cam ({sp.cam_xy[0]:g},{sp.cam_xy[1]:g}) h{sp.height_m:g} {sp.focal_mm:g}mm on {sp.subject}"
            if sp.size:
                spec += f" {sp.size}"
            if sp.dolly_to is not None:
                spec += f" dolly to ({sp.dolly_to[0]:g},{sp.dolly_to[1]:g})"
            if sp.gear_key:
                spec += f" gear {sp.gear_key}"
            body = f"{body}   [{spec}]"
        out.append(f"{b.number}. {body}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# blocking: paths on the walkable grid
# ---------------------------------------------------------------------------


def walkable(maps: space.FloorMaps) -> np.ndarray:
    """Cells an actor can walk through: inside, headroom, 30 cm of clearance."""
    return maps.standable(radius_m=WALK_CLEARANCE_M)


def nearest_walkable(maps: space.FloorMaps, mask: np.ndarray, xy, *, radius_m: float = SNAP_RADIUS_M):
    """The cell at `xy`, or the nearest walkable one within `radius_m`; None if none."""
    ij = maps.index_of(xy)[0]
    nx, ny = maps.shape
    i0, j0 = int(np.clip(ij[0], 0, nx - 1)), int(np.clip(ij[1], 0, ny - 1))
    if 0 <= ij[0] < nx and 0 <= ij[1] < ny and mask[i0, j0]:
        return (i0, j0)
    r = int(math.ceil(radius_m / maps.cell))
    best = None
    best_d = float("inf")
    for i in range(max(0, i0 - r), min(nx, i0 + r + 1)):
        for j in range(max(0, j0 - r), min(ny, j0 + r + 1)):
            if not mask[i, j]:
                continue
            c = maps.world_of((i, j))[0]
            d = float(np.hypot(c[0] - xy[0], c[1] - xy[1]))
            if d < best_d and d <= radius_m:
                best, best_d = (i, j), d
    return best


_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def astar(mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """Shortest 8-connected path through `mask`; corners are not cut.

    Cost is Euclidean cell distance; a diagonal step is allowed only when both
    orthogonal neighbours are free, so a path never squeezes between two
    obstructions that touch at a corner. Returns the cell list, or None.
    """
    nx, ny = mask.shape
    if not (mask[start] and mask[goal]):
        return None
    if start == goal:
        return [start]

    def h(c):
        return math.hypot(c[0] - goal[0], c[1] - goal[1])

    g = {start: 0.0}
    came: dict[tuple[int, int], tuple[int, int]] = {}
    open_heap = [(h(start), 0.0, start)]
    closed = set()
    while open_heap:
        _, gc, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path
        closed.add(cur)
        ci, cj = cur
        for di, dj in _NEIGHBOURS:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < nx and 0 <= nj < ny) or not mask[ni, nj]:
                continue
            if di and dj and not (mask[ci + di, cj] and mask[ci, cj + dj]):
                continue
            step = math.sqrt(2.0) if (di and dj) else 1.0
            ng = gc + step
            nxt = (ni, nj)
            if ng < g.get(nxt, float("inf")):
                g[nxt] = ng
                came[nxt] = cur
                heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))
    return None


def _segment_walkable(maps: space.FloorMaps, mask: np.ndarray, a, b) -> bool:
    """Every cell under the segment a->b is walkable (sampled at half a cell)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    length = float(np.linalg.norm(b - a))
    n = max(2, int(math.ceil(length / (maps.cell * 0.5))) + 1)
    pts = a[None, :] + np.linspace(0.0, 1.0, n)[:, None] * (b - a)[None, :]
    ij = maps.index_of(pts)
    nx, ny = maps.shape
    inb = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
    if not inb.all():
        return False
    return bool(mask[ij[:, 0], ij[:, 1]].all())


def string_pull(maps: space.FloorMaps, mask: np.ndarray, occupancy, waypoints: np.ndarray) -> np.ndarray:
    """Straighten a grid path: skip ahead to the farthest waypoint reachable in
    a straight, walkable, unobstructed line. The usual string-pulling pass,
    with `space.visible` at body height as the obstruction test."""
    pts = np.asarray(waypoints, dtype=np.float64).reshape(-1, 2)
    if len(pts) <= 2:
        return pts
    grid, origin, cell = occupancy
    z = maps.floor_z + BODY_CHECK_HEIGHT_M
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1:
            a3 = (pts[i][0], pts[i][1], z)
            b3 = (pts[j][0], pts[j][1], z)
            if _segment_walkable(maps, mask, pts[i], pts[j]) and space.visible(grid, origin, cell, a3, b3, slack=0.05):
                break
            j -= 1
        out.append(pts[j])
        i = j
    return np.asarray(out)


@dataclass
class Route:
    """An actor's walk from one point to another, as the floor allows it."""

    points: np.ndarray                 # (K, 2) world XY, start to end
    length_m: float
    blocked: bool = False
    reason: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def cumulative(self) -> np.ndarray:
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1) if len(self.points) > 1 else np.zeros(0)
        return np.concatenate([[0.0], np.cumsum(seg)])

    def at(self, s: float) -> tuple[np.ndarray, float]:
        """Position and heading (degrees, 0 along +X) at arc length `s`."""
        pts = self.points
        if len(pts) == 1:
            return pts[0].copy(), 0.0
        cum = self.cumulative
        s = float(np.clip(s, 0.0, cum[-1]))
        k = int(np.searchsorted(cum, s, side="right") - 1)
        k = min(max(k, 0), len(pts) - 2)
        seg = pts[k + 1] - pts[k]
        seg_len = float(np.linalg.norm(seg))
        f = 0.0 if seg_len < 1e-9 else (s - cum[k]) / seg_len
        # Heading from the last segment with length, so the end of a route
        # still faces the way it arrived.
        kk = k
        while seg_len < 1e-9 and kk > 0:
            kk -= 1
            seg = pts[kk + 1] - pts[kk]
            seg_len = float(np.linalg.norm(seg))
        heading = math.degrees(math.atan2(seg[1], seg[0])) if seg_len > 1e-9 else 0.0
        return pts[k] + f * (pts[k + 1] - pts[k]), heading


def route(maps: space.FloorMaps, occupancy, start_xy, goal_xy, *, mask: np.ndarray | None = None) -> Route:
    """Plan a walk on the standable floor. Blocked routes still carry the
    straight line so the timeline has somewhere to put the actor."""
    mask = walkable(maps) if mask is None else mask
    a = np.asarray(start_xy, dtype=np.float64).reshape(2)
    b = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    straight = np.array([a, b])
    straight_len = float(np.linalg.norm(b - a))
    notes: list[str] = []

    sa = nearest_walkable(maps, mask, a)
    sb = nearest_walkable(maps, mask, b)
    if sa is None or sb is None:
        which = "start" if sa is None else "destination"
        return Route(straight, straight_len, blocked=True,
                     reason=f"the {which} ({(a if sa is None else b)[0]:.2f}, {(a if sa is None else b)[1]:.2f}) "
                            f"is not on standable floor and nothing walkable lies within {SNAP_RADIUS_M:.1f} m")
    for name, cellij, p in (("start", sa, a), ("destination", sb, b)):
        c = maps.world_of(cellij)[0]
        if float(np.hypot(*(c - p))) > maps.cell * 0.75:
            notes.append(f"the {name} mark ({p[0]:.2f}, {p[1]:.2f}) is not standable; the walk uses "
                         f"({c[0]:.2f}, {c[1]:.2f}), {float(np.hypot(*(c - p))):.2f} m away")

    cells = astar(mask, sa, sb)
    if cells is None:
        return Route(straight, straight_len, blocked=True,
                     reason=f"no walkable path from ({a[0]:.2f}, {a[1]:.2f}) to ({b[0]:.2f}, {b[1]:.2f}); "
                            "something solid separates them on the standable floor", notes=notes)
    way = maps.world_of(np.array(cells)) if len(cells) > 1 else maps.world_of(cells[0])
    way = np.asarray(way).reshape(-1, 2)
    # The route begins and ends on the marks themselves, not on cell centres.
    pts = np.vstack([a[None, :], way[1:-1], b[None, :]]) if len(way) > 2 else np.array([a, b])
    pts = string_pull(maps, mask, occupancy, pts)
    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0
    return Route(pts, length, notes=notes)


# ---------------------------------------------------------------------------
# the timeline
# ---------------------------------------------------------------------------


def line_seconds(text: str | None) -> float:
    """How long a line takes to say."""
    words = len((text or "").split())
    return LINE_LEAD_S + words / WORDS_PER_SECOND


@dataclass
class BeatTiming:
    """When a beat happens and what the floor said about it."""

    number: int
    kind: str
    actor: str
    t_start: float
    t_end: float
    route: list[list[float]] | None = None
    blocked: bool = False
    reason: str = ""

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start


@dataclass
class Frame:
    """One sampled instant: everyone's position and facing, the camera, the sun."""

    t: float
    clock: datetime | None
    actors: dict[str, tuple[float, float, float]]
    camera: tuple[tuple[float, float, float], tuple[float, float, float], float, int] | None
    sun: dict | None
    framing: dict | None = None
    beat: int | None = None

    def to_dict(self) -> dict:
        return {
            "t": round(self.t, 3),
            "clock": None if self.clock is None else self.clock.strftime("%H:%M:%S"),
            "beat": self.beat,
            "actors": {k: [round(v[0], 3), round(v[1], 3), round(v[2], 1)] for k, v in self.actors.items()},
            "camera": None if self.camera is None else {
                "pos": [round(c, 3) for c in self.camera[0]],
                "aim": [round(c, 3) for c in self.camera[1]],
                "focal_mm": self.camera[2],
                "shot": self.camera[3],
            },
            "sun": self.sun,
            "framing": self.framing,
        }


@dataclass
class ShotTiming:
    """A camera setup and the span of the scene it covers."""

    number: int
    t_start: float
    t_end: float                                           # the camera is held until here
    setup: dict
    beat_end: float = 0.0                                  # the beat that placed it ends here
    problems: list[str] = field(default_factory=list)
    sizes: list[dict] = field(default_factory=list)      # [{size, name, from, to}]
    move: dict | None = None                               # MoveReport.summary() for a dolly
    notes: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start

    @property
    def beat_duration_s(self) -> float:
        return self.beat_end - self.t_start

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "t_start": round(self.t_start, 3),
            "t_end": round(self.t_end, 3),
            "beat_end": round(self.beat_end, 3),
            "setup": _jsonable(self.setup),
            "problems": list(self.problems),
            "sizes": list(self.sizes),
            "move": _jsonable(self.move),
            "notes": list(self.notes),
        }


@dataclass
class Rehearsal:
    scene: Scene
    frames: list[Frame]
    shots: list[ShotTiming]
    beats: list[BeatTiming]
    duration_s: float
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scene": self.scene.to_dict(),
            "duration_s": round(self.duration_s, 3),
            "dt_s": DT_S,
            "beats": [asdict(b) for b in self.beats],
            "shots": [s.to_dict() for s in self.shots],
            "frames": [f.to_dict() for f in self.frames],
            "problems": list(self.problems),
            "notes": list(self.notes),
        }

    def frame_at(self, t: float) -> Frame:
        k = int(round(t / DT_S))
        return self.frames[min(max(k, 0), len(self.frames) - 1)]


def _jsonable(v):
    if v is None:
        return None
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    return v


@dataclass
class _ActorState:
    xy: np.ndarray
    facing: float
    present: bool


class _Sun:
    """Per-minute sun lookups for one twin and shoot date."""

    def __init__(self, twin: Twin, day: _date):
        self.twin = twin
        self.day = day
        self.schedule = daylight.sun_schedule(twin, day)
        openings = getattr(twin.structure, "openings", None) or []
        self.glass_idx = [j for j, o in enumerate(openings) if getattr(o, "kind", "window") != "door"]
        self._cache: dict[int, dict] = {}

    def at(self, clock: datetime) -> dict:
        minute = clock.hour * 60 + clock.minute
        hit = self._cache.get(minute)
        if hit is not None:
            return hit
        lit = []
        for w, j in zip(self.schedule.get("windows", []), self.glass_idx):
            for run in w["direct_sun"]:
                if _hhmm_to_min(run["from"]) <= minute <= _hhmm_to_min(run["to"]):
                    lit.append(j)
                    break
        from ..product import sunlight
        s = sunlight.sun_at(self.twin, self.day, minute / 60.0)
        out = {
            "direct_sun": bool(lit),
            "windows_lit": lit,
            "azimuth_deg": s["azimuth_deg"],
            "elevation_deg": s["elevation_deg"],
            "compass": s["compass"],
        }
        self._cache[minute] = out
        return out


def _hhmm_to_min(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def rehearse(
    twin: Twin,
    scene: Scene,
    *,
    maps: space.FloorMaps | None = None,
    occupancy=None,
) -> Rehearsal:
    """Run the scene in the room: block every move, time every beat, hold or
    ride every camera, tag the sun, and sample it all into frames."""
    maps = space.floor_maps(twin) if maps is None else maps
    occupancy = space.occupancy(twin) if occupancy is None else occupancy
    mask = walkable(maps)
    floor_z = float(maps.floor_z)
    problems: list[str] = []
    notes: list[str] = []

    marks = {k: np.asarray(v, dtype=np.float64) for k, v in scene.marks.items()}

    def mark_xy(name: str | None, beat: Beat) -> np.ndarray:
        if name is None or name not in marks:
            raise ValueError(f"beat {beat.number}: unknown mark {name!r}; the scene has {sorted(marks)}")
        return marks[name]

    def stature(name: str) -> float:
        return float(scene.cast.get(name, render.SUBJECT_HEIGHT_M))

    for name in scene.cast:
        if not (1.0 <= scene.cast[name] <= 2.3):
            raise ValueError(f"{name}'s height {scene.cast[name]} m is not a human stature in metres")

    # -- pass 1: time the beats and block the walks --------------------------
    states: dict[str, _ActorState] = {}
    beat_timings: list[BeatTiming] = []
    routes: dict[int, Route] = {}
    starts: list[dict[str, _ActorState]] = []   # actor states at each beat's start
    t = 0.0
    for b in scene.beats:
        starts.append({k: _ActorState(v.xy.copy(), v.facing, v.present) for k, v in states.items()})
        dur = 0.0
        rt = None
        blocked = False
        reason = ""
        if b.kind == "enter":
            xy = mark_xy(b.to_mark, b)
            dur = ENTER_SECONDS
            facing = states[b.actor].facing if b.actor in states else _facing_into_room(maps, xy)
            states[b.actor] = _ActorState(xy.copy(), facing, True)
            # The actor is on the mark from the beat's first instant.
            starts[-1][b.actor] = _ActorState(xy.copy(), facing, True)
        elif b.kind == "line":
            dur = line_seconds(b.text)
            if b.actor not in states or not states[b.actor].present:
                notes.append(f"beat {b.number}: {b.actor} speaks without having entered; placed at the room's centre")
                states[b.actor] = _ActorState(_room_centre(maps, mask), 0.0, True)
                starts[-1][b.actor] = _ActorState(states[b.actor].xy.copy(), 0.0, True)
        elif b.kind == "hold":
            dur = float(b.seconds if b.seconds is not None else DEFAULT_HOLD_S)
        elif b.kind in ("move", "exit"):
            if b.actor not in states or not states[b.actor].present:
                notes.append(f"beat {b.number}: {b.actor} moves without having entered; placed at the room's centre")
                states[b.actor] = _ActorState(_room_centre(maps, mask), 0.0, True)
                starts[-1][b.actor] = _ActorState(states[b.actor].xy.copy(), 0.0, True)
            if b.to_mark is None and b.kind == "exit":
                dur = ENTER_SECONDS
                states[b.actor].present = False
            else:
                goal = mark_xy(b.to_mark, b)
                rt = route(maps, occupancy, states[b.actor].xy, goal, mask=mask)
                routes[b.number] = rt
                notes.extend(f"beat {b.number}: {n}" for n in rt.notes)
                if rt.blocked:
                    blocked = True
                    reason = rt.reason
                    dur = 0.0
                    problems.append(
                        f"Beat {b.number}: {b.actor} cannot walk to {b.to_mark}: {rt.reason}. "
                        f"The rehearsal places {b.actor} on the mark so the scene continues."
                    )
                else:
                    dur = rt.length_m / WALK_SPEED_MPS
                end_xy, heading = rt.at(rt.length_m)
                states[b.actor].xy = goal.copy()
                states[b.actor].facing = heading
                if b.kind == "exit":
                    states[b.actor].present = False
        beat_timings.append(BeatTiming(
            number=b.number, kind=b.kind, actor=b.actor, t_start=t, t_end=t + dur,
            route=None if rt is None else [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in rt.points],
            blocked=blocked, reason=reason,
        ))
        t += dur
    duration = t

    # -- pass 2: the camera per beat ------------------------------------------
    shots: list[ShotTiming] = []
    beat_shot: list[ShotTiming | None] = []
    current: ShotTiming | None = None
    camera_specs: dict[int, ShotSpec] = {}
    for b, bt in zip(scene.beats, beat_timings):
        if b.shot is not None:
            sp = b.shot
            if sp.subject not in scene.cast and sp.subject not in {bb.actor for bb in scene.beats}:
                raise ValueError(f"beat {b.number}: shot subject {sp.subject!r} is not in the scene")
            if current is not None:
                current.t_end = bt.t_start
            setup = _setup_dict(sp, floor_z, stature(sp.subject))
            current = ShotTiming(number=b.number, t_start=bt.t_start, t_end=duration, setup=setup,
                                 beat_end=bt.t_end)
            camera_specs[b.number] = sp
            shots.append(current)
        beat_shot.append(current)
    if not shots:
        notes.append("the scene names no camera; the timeline has bodies and no frame")

    # -- pass 3: sample ---------------------------------------------------------
    sun = None
    clock0 = None
    if scene.shoot_date is not None and scene.shoot_start is not None:
        if twin.georeference is None:
            notes.append("the twin has no georeference, so the sun cannot be placed; frames carry sun=None")
        else:
            sun = _Sun(twin, scene.shoot_date)
            clock0 = datetime.combine(scene.shoot_date, scene.shoot_start)
            notes.append(f"clock times are {sun.schedule['times_are']}")
            if sun.schedule.get("heading_caveat"):
                notes.append(sun.schedule["heading_caveat"])
    else:
        notes.append("no shoot date and start hour given, so frames carry sun=None")

    n_frames = int(round(duration / DT_S)) + 1
    frames: list[Frame] = []
    sensor = optics.SENSORS[optics.DEFAULT_SENSOR]
    beat_index = 0
    for k in range(n_frames):
        tk = min(k * DT_S, duration)
        # The beat in progress: the first whose end lies past this instant.
        # Zero-length beats (a blocked move) are stepped over; the next beat's
        # start state already carries their effect.
        while beat_index < len(beat_timings) - 1 and tk >= beat_timings[beat_index].t_end:
            beat_index += 1
        bt = beat_timings[beat_index] if beat_timings else None
        b = scene.beats[beat_index] if scene.beats else None
        actors = _actors_at(tk, b, bt, starts[beat_index] if starts else {}, routes)
        camera = None
        framing = None
        st = beat_shot[beat_index] if beat_shot else None
        if st is not None:
            sp = camera_specs[st.number]
            cam_xy = _camera_xy_at(sp, st, tk)
            cam_pos = (float(cam_xy[0]), float(cam_xy[1]), floor_z + float(sp.height_m))
            subj = actors.get(sp.subject)
            if subj is not None:
                aim = (subj[0], subj[1], floor_z + EYELINE_RATIO * stature(sp.subject))
                dist = float(math.dist(cam_pos, aim))
                fh = optics.framed_height_m(sensor, sp.focal_mm, max(dist, 1e-3))
                size = optics.classify_framing(fh)
                framing = {
                    "subject": sp.subject,
                    "distance_m": round(dist, 3),
                    "framed_height_m": round(fh, 3),
                    "subject_fill": round(stature(sp.subject) / fh, 3),
                    "shot_size": size.key,
                    "size_name": size.name,
                }
            else:
                aim = _aim_from_setup(st.setup, floor_z, stature(sp.subject))
                framing = {"subject": sp.subject, "absent": True}
            camera = (cam_pos, tuple(float(a) for a in aim), float(sp.focal_mm), st.number)
        clock = None if clock0 is None else clock0 + timedelta(seconds=tk)
        sun_tag = None if sun is None or clock is None else sun.at(clock)
        frames.append(Frame(t=tk, clock=clock, actors=actors, camera=camera, sun=sun_tag,
                            framing=framing, beat=None if bt is None else bt.number))

    # -- sizes seen per shot, and the dolly checks -----------------------------
    for st in shots:
        sp = camera_specs[st.number]
        st.sizes = _sizes_seen(frames, st)
        _check_shot(st, sp, scene, maps, occupancy, frames, floor_z, stature(sp.subject))
        problems.extend(f"Shot {st.number}: {p}" for p in st.problems)
        notes.extend(_framing_sentences(st, sp))

    return Rehearsal(scene=scene, frames=frames, shots=shots, beats=beat_timings,
                     duration_s=duration, problems=problems, notes=notes)


def _room_centre(maps: space.FloorMaps, mask: np.ndarray) -> np.ndarray:
    idx = np.argwhere(mask)
    if not len(idx):
        return maps.world_of((maps.shape[0] // 2, maps.shape[1] // 2))[0]
    c = idx.mean(axis=0)
    best = idx[np.argmin(np.linalg.norm(idx - c, axis=1))]
    return maps.world_of(best)[0]


def _facing_into_room(maps: space.FloorMaps, xy: np.ndarray) -> float:
    """Someone entering faces away from the nearest wall: toward the room."""
    lo = maps.origin
    hi = maps.origin + np.array(maps.shape) * maps.cell
    c = 0.5 * (lo + hi)
    d = c - xy
    return math.degrees(math.atan2(d[1], d[0])) if np.linalg.norm(d) > 1e-6 else 0.0


def _actors_at(tk, b: Beat | None, bt: BeatTiming | None, start_states: dict[str, _ActorState],
               routes: dict[int, Route]) -> dict[str, tuple[float, float, float]]:
    """Every present actor's (x, y, facing) at time `tk` within beat `b`.

    `start_states` is where everyone stood when the beat began (an entering
    actor is already on the mark). The walking actor is carried along the
    beat's route at walking pace; everyone else stands and turns: toward the
    speaker, toward the nearest other actor when speaking, toward whoever is
    walking, or else as they were.
    """
    st = {k: _ActorState(v.xy.copy(), v.facing, v.present) for k, v in start_states.items()}
    if b is not None and bt is not None:
        walking = b.kind in ("move", "exit") and bt.number in routes and b.actor in st
        if walking:
            rt = routes[bt.number]
            s = WALK_SPEED_MPS * max(0.0, tk - bt.t_start)
            xy, heading = rt.at(min(s, rt.length_m))
            st[b.actor].xy = xy
            st[b.actor].facing = heading
        speaker = b.actor if b.kind == "line" else None
        others = [k for k, v in st.items() if v.present and k != speaker]
        for name, a in st.items():
            if not a.present or (walking and name == b.actor):
                continue
            target = None
            if speaker is not None and name != speaker and speaker in st and st[speaker].present:
                target = st[speaker].xy
            elif speaker == name and others:
                target = min((st[o].xy for o in others), key=lambda p: float(np.linalg.norm(p - a.xy)))
            elif b.kind in ("move", "exit") and name != b.actor and b.actor in st and st[b.actor].present:
                target = st[b.actor].xy
            if target is not None:
                d = target - a.xy
                if float(np.linalg.norm(d)) > 1e-6:
                    a.facing = math.degrees(math.atan2(d[1], d[0]))
    return {k: (float(v.xy[0]), float(v.xy[1]), float(v.facing)) for k, v in st.items() if v.present}


def _setup_dict(sp: ShotSpec, floor_z: float, stature_m: float) -> dict:
    cam = np.array([sp.cam_xy[0], sp.cam_xy[1], floor_z + sp.height_m])
    d = {
        "cam_x": float(sp.cam_xy[0]), "cam_y": float(sp.cam_xy[1]), "cam_z": float(sp.height_m),
        "focal_mm": float(sp.focal_mm), "sensor": optics.DEFAULT_SENSOR,
        "subject": sp.subject, "subject_height_m": float(stature_m),
        "shot_size": sp.size, "dolly_to": None if sp.dolly_to is None else [float(sp.dolly_to[0]), float(sp.dolly_to[1])],
        "gear_key": sp.gear_key or ("doorway-dolly" if sp.dolly_to is not None else None),
        "fov_h_deg": round(optics.field_of_view_deg(optics.SENSORS[optics.DEFAULT_SENSOR], sp.focal_mm)[0], 1),
    }
    return d


def _aim_from_setup(setup: dict, floor_z: float, stature_m: float) -> tuple[float, float, float]:
    return (float(setup.get("subj_x", setup["cam_x"] + 1.0)), float(setup.get("subj_y", setup["cam_y"])),
            floor_z + EYELINE_RATIO * stature_m)


def _camera_xy_at(sp: ShotSpec, st: ShotTiming, tk: float) -> np.ndarray:
    a = np.asarray(sp.cam_xy, dtype=np.float64)
    if sp.dolly_to is None:
        return a
    b = np.asarray(sp.dolly_to, dtype=np.float64)
    # The dolly rides over its own beat. The shot's `t_end` may run on into
    # later beats that keep the camera, and there it holds at the end mark.
    span = st.beat_duration_s
    f = 1.0 if span <= 1e-9 else float(np.clip((tk - st.t_start) / span, 0.0, 1.0))
    return a + f * (b - a)


def _sizes_seen(frames: list[Frame], st: ShotTiming) -> list[dict]:
    out: list[dict] = []
    for fr in frames:
        if fr.camera is None or fr.camera[3] != st.number or not fr.framing or "shot_size" not in fr.framing:
            continue
        key = fr.framing["shot_size"]
        if out and out[-1]["size"] == key:
            out[-1]["to"] = round(fr.t, 3)
        else:
            out.append({"size": key, "name": fr.framing["size_name"], "from": round(fr.t, 3), "to": round(fr.t, 3)})
    return out


def _framing_sentences(st: ShotTiming, sp: ShotSpec) -> list[str]:
    if not st.sizes:
        return []
    lens = f"{sp.focal_mm:g} mm"
    if len(st.sizes) == 1:
        s = st.sizes[0]
        return [f"Shot {st.number}: the {lens} holds {_a(s['name'])} on {sp.subject} "
                f"from {s['from']:.1f} s to {s['to']:.1f} s."]
    parts = [f"{_a(st.sizes[0]['name'])} until {st.sizes[1]['from']:.1f} s"]
    for i in range(1, len(st.sizes)):
        s = st.sizes[i]
        nxt = f" until {st.sizes[i + 1]['from']:.1f} s" if i + 1 < len(st.sizes) else ""
        parts.append(f"then {_a(s['name'])}{nxt}")
    return [f"Shot {st.number}: the {lens} on {sp.subject} holds " + ", ".join(parts) + "."]


def _a(name: str) -> str:
    low = name.lower()
    return ("an " if low[:1] in "aeiou" else "a ") + low


# ---------------------------------------------------------------------------
# other actors as occluders
# ---------------------------------------------------------------------------


def _xy(p) -> np.ndarray:
    return np.asarray(p, dtype=np.float64).reshape(-1)[:2]


def actor_blocks(
    cam_xyz,
    subject_xyz,
    others: dict[str, tuple[float, float]],
    *,
    radius_m: float = OCCLUDER_RADIUS_M,
    heights: dict[str, float] | None = None,
    floor_z: float = 0.0,
) -> list[str]:
    """Names of the other actors whose bodies cut the camera -> subject eyeline.

    `cam_xyz` is the lens, `subject_xyz` the point it is pointed at (the
    subject's eyeline); `others` maps every other actor to where they stand
    (x, y, and anything after is ignored). A body is a vertical cylinder of
    `radius_m` from `floor_z` up to the actor's height in `heights` (an actor
    not listed there, or no `heights` at all, is a column the eyeline can
    never pass over). An actor blocks when the cylinder meets the segment and
    their centre is nearer the lens than the subject by at least a radius: one
    standing at the subject's shoulder or behind them is in the frame, not in
    the way. Pure geometry; the order is nearest-to-the-lens first.
    """
    cam = np.asarray(cam_xyz, dtype=np.float64).reshape(-1)
    subj = np.asarray(subject_xyz, dtype=np.float64).reshape(-1)
    c2, s2 = cam[:2], subj[:2]
    axis = s2 - c2
    length = float(np.linalg.norm(axis))
    if length <= 1e-9:
        return []
    axis = axis / length
    cz = float(cam[2]) if cam.size > 2 else None
    sz = float(subj[2]) if subj.size > 2 else None
    hits: list[tuple[float, str]] = []
    for name, pos in others.items():
        d = _xy(pos) - c2
        along = float(d @ axis)
        if along < 0.0 or along > length - radius_m:
            continue
        across = abs(float(d[0] * axis[1] - d[1] * axis[0]))
        if across > radius_m:
            continue
        if heights is not None and cz is not None and sz is not None:
            top = floor_z + float(heights.get(name, render.SUBJECT_HEIGHT_M))
            z_here = cz + (along / length) * (sz - cz)
            if z_here > top:
                continue
        hits.append((along, name))
    return [name for _, name in sorted(hits)]


def actor_too_close(cam_xy, others: dict[str, tuple[float, float]], *, min_m: float = LENS_CLEARANCE_M) -> list[str]:
    """Other actors within `min_m` of the lens horizontally: they fill the
    frame whatever it is pointed at. Nearest first."""
    c2 = _xy(cam_xy)
    near = []
    for name, pos in others.items():
        d = float(np.linalg.norm(_xy(pos) - c2))
        if d < min_m:
            near.append((d, name))
    return [name for _, name in sorted(near)]


def _actor_occlusion_problems(st: ShotTiming, sp: ShotSpec, scene: Scene, subj_frames: list[Frame], floor_z: float) -> list[str]:
    """Per frame of the shot: who walks through the eyeline, who is on the lens."""
    heights = {n: float(h) for n, h in scene.cast.items()}
    crossing: dict[str, tuple[float, float]] = {}       # name -> (first t, last t)
    close: dict[str, tuple[float, float]] = {}          # name -> (nearest d, its t)
    for fr in subj_frames:
        others = {n: (x, y) for n, (x, y, _f) in fr.actors.items() if n != sp.subject}
        if not others:
            continue
        cam = fr.camera[0]
        for name in actor_blocks(cam, fr.camera[1], others, heights=heights, floor_z=floor_z):
            t0, _ = crossing.get(name, (fr.t, fr.t))
            crossing[name] = (t0, fr.t)
        for name in actor_too_close(cam, others):
            d = float(math.dist(cam[:2], others[name]))
            if name not in close or d < close[name][0]:
                close[name] = (d, fr.t)
    out = []
    for name, (t0, t1) in sorted(crossing.items(), key=lambda kv: kv[1]):
        s = f"at {t0:.1f} s {name} crosses between the camera and {sp.subject}"
        if t1 > t0 + DT_S / 2:
            s += f" and stays there until {t1:.1f} s"
        out.append(s)
    for name, (d, t) in sorted(close.items(), key=lambda kv: kv[1][1]):
        out.append(f"at {t:.1f} s {name} is {d:.1f} m from the lens and fills the frame")
    return out


def _check_shot(st: ShotTiming, sp: ShotSpec, scene: Scene, maps, occupancy, frames, floor_z, stature_m) -> None:
    """The physical questions, answered by the same code the planner uses."""
    grid, origin, cell = occupancy
    own = [fr for fr in frames if fr.camera is not None and fr.camera[3] == st.number]
    if not own:
        st.notes.append("no frame falls inside this shot's span")
        return
    first = own[0]
    cam0 = np.array(first.camera[0])
    inside = bool(maps.sample(maps.inside, cam0[:2])[0])
    if not inside:
        st.problems.append(f"the camera at ({cam0[0]:.2f}, {cam0[1]:.2f}) is outside the room's floor outline")
    if not bool(maps.sample(maps.surveyed, cam0[:2])[0]):
        st.notes.append("the camera stands outside the surveyed area, on reconstructed geometry")
    subj_frames = [fr for fr in own if fr.framing and "distance_m" in fr.framing]
    if not subj_frames:
        st.problems.append(f"{sp.subject} is not in the room during this shot")
        return
    # Sightline at the first and last instants of the shot, as the sweep would test.
    for fr in (subj_frames[0], subj_frames[-1]):
        if not space.visible(grid, origin, cell, fr.camera[0], fr.camera[1]):
            st.problems.append(f"at {fr.t:.1f} s something is between the camera and {sp.subject}")
            break
    # The other actors, every frame: the room is not the only thing that can
    # stand between the lens and the subject.
    st.problems.extend(_actor_occlusion_problems(st, sp, scene, subj_frames, floor_z))
    # Record the subject's start position for the plan's columns.
    sx, sy, _ = subj_frames[0].actors[sp.subject]
    st.setup.update({"subj_x": round(sx, 3), "subj_y": round(sy, 3),
                     "distance_m": subj_frames[0].framing["distance_m"],
                     "yaw_deg": round(math.degrees(math.atan2(sy - cam0[1], sx - cam0[0])), 1)})
    if not st.setup.get("shot_size") and st.sizes:
        st.setup["shot_size"] = st.sizes[0]["size"]
    if sp.size and st.sizes and st.sizes[0]["size"] != sp.size:
        st.notes.append(f"the brief asked for a {optics.SHOT_BY_KEY[sp.size].name.lower()} and the {sp.focal_mm:g} mm "
                        f"from here gives a {st.sizes[0]['name'].lower()} at {subj_frames[0].framing['distance_m']:.2f} m")

    if sp.dolly_to is not None:
        gear = equipment.get(sp.gear_key or "doorway-dolly")
        a = np.array([sp.cam_xy[0], sp.cam_xy[1], floor_z + sp.height_m])
        b = np.array([sp.dolly_to[0], sp.dolly_to[1], floor_z + sp.height_m])
        # The subject as seen over the ride itself, not over later beats that
        # keep the camera parked at the end of the track.
        ride = [fr for fr in subj_frames if fr.t <= st.beat_end + 1e-9] or subj_frames[:1]
        subj_path = np.array([[fr.actors[sp.subject][0], fr.actors[sp.subject][1], floor_z] for fr in ride])
        report = moves.simulate(
            maps, occupancy, moves.straight(a, b), name=f"shot {st.number} dolly",
            subject_path=subj_path, subject_height_m=stature_m, focal_mm=sp.focal_mm,
            gear=gear, on_track=True,
        )
        st.move = report.summary()
        bad = report.first_problem
        if bad is not None:
            for p in bad.problems:
                st.problems.append(f"the {gear.name} run fails {bad.t * report.length_m:.1f} m along: {p}")
        st.notes.extend(report.notes)
        if not gear.verified:
            st.notes.append(f"{gear.name}'s dimensions are class-typical, not a published spec")
        speed = report.length_m / max(st.beat_duration_s, 1e-6)
        if speed > MAX_DOLLY_SPEED_MPS:
            st.problems.append(f"the dolly would have to travel {report.length_m:.1f} m in {st.beat_duration_s:.1f} s "
                               f"({speed:.1f} m/s), faster than a grip pushes")


# ---------------------------------------------------------------------------
# the schedule
# ---------------------------------------------------------------------------


@dataclass
class Schedule:
    entries: list[dict]
    sentences: list[str]
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entries": [
                {**e, "start": e["start"].strftime("%H:%M"), "end": e["end"].strftime("%H:%M")}
                for e in self.entries
            ],
            "sentences": list(self.sentences),
            "problems": list(self.problems),
            "notes": list(self.notes),
        }


def window_dependence(twin: Twin, st: ShotTiming) -> dict | None:
    """Which window the shot's look depends on, and how.

    Seen from the subject, a window within `KEY_ARC_DEG` of the camera is the
    key light on the face; one within `BACKLIGHT_ARC_DEG` of the axis behind
    the subject is in the background of the frame. The largest apparent
    window in either arc wins. None when no window matters to the shot.
    """
    openings = getattr(twin.structure, "openings", None) or []
    setup = st.setup
    if "subj_x" not in setup:
        return None
    subj = np.array([setup["subj_x"], setup["subj_y"]], dtype=np.float64)
    cam = np.array([setup["cam_x"], setup["cam_y"]], dtype=np.float64)
    to_cam = cam - subj
    n = float(np.linalg.norm(to_cam))
    if n < 1e-9:
        return None
    to_cam /= n
    best = None
    glass_no = 0
    for j, o in enumerate(openings):
        if getattr(o, "kind", "window") == "door":
            continue
        glass_no += 1
        c = np.asarray(o.center, dtype=np.float64)[:2]
        rel = c - subj
        d = max(float(np.linalg.norm(rel)), 0.3)
        size = float(o.width * o.height) / (d * d)
        ang = math.degrees(math.acos(float(np.clip(np.dot(rel / d, to_cam), -1.0, 1.0))))
        role = "key" if ang <= KEY_ARC_DEG else ("backlight" if ang >= 180.0 - BACKLIGHT_ARC_DEG else None)
        if role is None:
            continue
        if best is None or size > best["apparent_size"]:
            best = {"opening": j, "window": glass_no, "role": role, "angle_deg": round(ang, 1), "apparent_size": size}
    return best


def schedule(
    rehearsal: Rehearsal,
    twin: Twin,
    *,
    setup_minutes: int = DEFAULT_SETUP_MINUTES,
    start: datetime,
) -> Schedule:
    """Lay the shots out on the shoot day so every window-dependent shot sits
    inside its window's direct sun. Times are local solar time, as
    `daylight.sun_schedule` reports them."""
    scene = rehearsal.scene
    day = scene.shoot_date or start.date()
    sentences: list[str] = []
    problems: list[str] = []
    notes: list[str] = []

    sun = None
    if twin.georeference is None:
        notes.append("no georeference: the shots are ordered as written, with no sun constraint")
    else:
        sun = daylight.sun_schedule(twin, day)
        notes.append(f"times are {sun['times_are']}")
        if sun.get("heading_caveat"):
            notes.append(sun["heading_caveat"])

    def block_minutes(st: ShotTiming) -> int:
        takes = max(MIN_TAKES_MINUTES, int(math.ceil(TAKES_PER_BEAT * st.duration_s / 60.0)))
        return int(setup_minutes) + takes

    def interval_for(st: ShotTiming):
        if sun is None:
            return None, None
        dep = window_dependence(twin, st)
        if dep is None:
            return None, None
        glass = sun.get("windows", [])
        k = dep["window"] - 1
        if k >= len(glass) or not glass[k]["direct_sun"]:
            return dep, []
        return dep, [(_hhmm_to_min(r["from"]), _hhmm_to_min(r["to"])) for r in glass[k]["direct_sun"]]

    day0 = datetime.combine(day, _time(0, 0))
    cursor = int((start - datetime.combine(start.date(), _time(0, 0))).total_seconds() // 60)

    constrained = []
    free = []
    for st in rehearsal.shots:
        dep, runs = interval_for(st)
        if runs:
            constrained.append((st, dep, runs))
        else:
            free.append((st, dep))
            if dep is not None and runs == []:
                notes.append(f"Shot {st.number} looks toward window {dep['window']} ({dep['role']}), which passes "
                             f"no direct sun on {day.isoformat()}; it can shoot at any hour")

    # Each constrained shot wants the first run that can hold it from the cursor.
    def earliest(runs, need, after):
        for lo, hi in runs:
            s = max(lo, after)
            if s + need <= hi:
                return s, (lo, hi)
        return None, runs[0]

    strikes_before: dict[int, list[str]] = {}
    for s in scene.strikes:
        try:
            strikes_before.setdefault(int(s["before_shot"]), []).append(str(s["piece"]))
        except (KeyError, TypeError, ValueError):
            problems.append(f"a strike entry could not be read: {s!r}")

    entries: list[dict] = []
    placed: set[int] = set()

    def place(st: ShotTiming, reason: str, at: int) -> int:
        nonlocal cursor
        for piece in strikes_before.get(st.number, []):
            entries.append({"shot": None, "start": day0 + timedelta(minutes=at), "end": day0 + timedelta(minutes=at + STRIKE_MINUTES),
                            "reason": f"strike the {piece} before setup {st.number}"})
            sentences.append(f"Strike the {piece} before setup {st.number} ({(day0 + timedelta(minutes=at)).strftime('%H:%M')}).")
            at += STRIKE_MINUTES
        need = block_minutes(st)
        entries.append({"shot": st.number, "start": day0 + timedelta(minutes=at), "end": day0 + timedelta(minutes=at + need), "reason": reason})
        placed.add(st.number)
        cursor = at + need
        return cursor

    # Constrained shots in order of their earliest feasible start from the
    # day's start; free shots fill the gaps before each one and the remainder
    # follows. Greedy, which is how a 1st AD lays a day out too.
    def need_for(st: ShotTiming) -> int:
        return block_minutes(st) + STRIKE_MINUTES * len(strikes_before.get(st.number, []))

    def sort_key(item):
        st, _dep, runs = item
        s, run = earliest(runs, need_for(st), cursor)
        return (s if s is not None else run[0], st.number)

    constrained.sort(key=sort_key)

    free_queue = list(free)
    for st, dep, runs in constrained:
        need = need_for(st)
        s, run = earliest(runs, need, cursor)
        if s is None:
            s = max(cursor, run[0])
            problems.append(
                f"Shot {st.number} needs window {dep['window']} lit ({dep['role']}) and takes {need} min, but the "
                f"direct sun runs {', '.join(f'{_mm(a)} to {_mm(b)}' for a, b in runs)}; it is placed at {_mm(s)} "
                "and will lose the light"
            )
        # Fill the gap with free shots that fit before `s`.
        while free_queue:
            fst, fdep = free_queue[0]
            fneed = block_minutes(fst) + STRIKE_MINUTES * len(strikes_before.get(fst.number, []))
            if cursor + fneed <= s:
                free_queue.pop(0)
                place(fst, _free_reason(fdep), cursor)
            else:
                break
        s = max(s, cursor)
        place(st, f"window {dep['window']} {dep['role']}: direct sun {_mm(run[0])} to {_mm(run[1])}", s)
        sentences.append(
            f"Shot {st.number} window coverage holds {_mm(run[0])} to {_mm(run[1])}; "
            f"set at {entries[-1]['start'].strftime('%H:%M')}, wrap by {entries[-1]['end'].strftime('%H:%M')}."
        )
    for fst, fdep in free_queue:
        place(fst, _free_reason(fdep) if sun is not None else "as written; no georeference, so no sun constraint", cursor)

    n_setups = len([e for e in entries if e["shot"] is not None])
    if entries:
        sentences.append(
            f"{scene.title}: {n_setups} setup{'s' if n_setups != 1 else ''} at {setup_minutes} minutes each, "
            f"{entries[0]['start'].strftime('%H:%M')} to {entries[-1]['end'].strftime('%H:%M')}."
        )
    return Schedule(entries=entries, sentences=sentences, problems=problems, notes=notes)


def _mm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _free_reason(dep: dict | None) -> str:
    if dep is None:
        return "no window in the shot's light; any hour"
    return f"window {dep['window']} is the {dep['role']} but passes no direct sun on the day; any hour"


def render_text(rehearsal: Rehearsal, sched: Schedule | None = None) -> str:
    """The call sheet, typed."""
    sc = rehearsal.scene
    lines = [sc.title.upper(), "=" * len(sc.title)]
    if sc.shoot_date:
        lines.append(f"Shoot date {sc.shoot_date.isoformat()}" + (f", first frame {sc.shoot_start.strftime('%H:%M')}" if sc.shoot_start else ""))
    lines.append(f"Running time {rehearsal.duration_s:.1f} s over {len(sc.beats)} beats; {len(rehearsal.shots)} setups.")
    if sc.cast:
        lines.append("Cast: " + ", ".join(f"{k} ({v:.2f} m)" for k, v in sc.cast.items()))
    lines.append("")
    lines.append("BEATS")
    for b, bt in zip(sc.beats, rehearsal.beats):
        what = {
            "line": f'{b.actor}: "{b.text}"',
            "move": f"{b.actor} walks to {b.to_mark}",
            "hold": "hold",
            "enter": f"{b.actor} enters" + (f" at {b.to_mark}" if b.to_mark else ""),
            "exit": f"{b.actor} exits" + (f" to {b.to_mark}" if b.to_mark else ""),
        }[b.kind]
        flag = "  BLOCKED: " + bt.reason if bt.blocked else ""
        lines.append(f"  {b.number:>2}. {bt.t_start:6.2f}-{bt.t_end:6.2f} s  {what}{flag}")
    lines.append("")
    lines.append("SETUPS")
    for st in rehearsal.shots:
        s = st.setup
        lens = f"{s['focal_mm']:g} mm at h{s['cam_z']:.2f}"
        where = f"({s['cam_x']:.2f}, {s['cam_y']:.2f})"
        if s.get("dolly_to"):
            where += f" dolly to ({s['dolly_to'][0]:.2f}, {s['dolly_to'][1]:.2f}) on the {equipment.get(s['gear_key']).name.lower()}"
        sizes = " -> ".join(x["name"].lower() for x in st.sizes) if st.sizes else "no framing"
        lines.append(f"  Shot {st.number}: {lens} from {where} on {s['subject']}; {st.t_start:.1f}-{st.t_end:.1f} s; {sizes}")
        for p in st.problems:
            lines.append(f"      ! {p}")
        for n in st.notes:
            lines.append(f"      - {n}")
    if sched is not None:
        lines.append("")
        lines.append("SCHEDULE")
        for e in sched.entries:
            label = f"Shot {e['shot']}" if e["shot"] is not None else "Strike"
            lines.append(f"  {e['start'].strftime('%H:%M')}-{e['end'].strftime('%H:%M')}  {label:<8} {e['reason']}")
        for s in sched.sentences:
            lines.append(f"  {s}")
        for p in sched.problems:
            lines.append(f"  ! {p}")
        for n in sched.notes:
            lines.append(f"  - {n}")
    if rehearsal.problems:
        lines.append("")
        lines.append("PROBLEMS")
        lines.extend(f"  ! {p}" for p in rehearsal.problems)
    if rehearsal.notes:
        lines.append("")
        lines.append("NOTES")
        lines.extend(f"  - {n}" for n in rehearsal.notes)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the animatic
# ---------------------------------------------------------------------------

CAPTION_PX = 22


def animatic(
    twin: Twin,
    rehearsal: Rehearsal,
    out_dir: str | Path,
    *,
    every_s: float = 1.0,
    figures_for: Callable[[Frame], list[render.Figure]] | None = None,
    width_px: int = 640,
) -> list[Path]:
    """One PNG per sampled second from the current camera, with a caption
    strip, plus `animatic.json` with each frame's metadata.

    `figures_for(frame)` returns the bodies to draw (the cast module's job);
    without it the renderer's stand-in figure marks the subject.
    """
    from PIL import Image, ImageDraw, ImageFont

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    paths: list[Path] = []
    meta: list[dict] = []
    floor_z = float(twin.structure.floor_z)
    n = int(math.floor(rehearsal.duration_s / every_s + 1e-9)) + 1
    for k in range(n):
        t = min(k * every_s, rehearsal.duration_s)
        fr = rehearsal.frame_at(t)
        if fr.camera is None:
            meta.append({**fr.to_dict(), "file": None, "note": "no camera yet"})
            continue
        cam_pos, aim, focal, shot_no = fr.camera
        subject = fr.framing["subject"] if fr.framing else None
        height = float(rehearsal.scene.cast.get(subject, render.SUBJECT_HEIGHT_M)) if subject else render.SUBJECT_HEIGHT_M
        subj_xy = (aim[0], aim[1])
        figs = figures_for(fr) if figures_for is not None else None
        others = [(a[0], a[1]) for name, a in fr.actors.items() if name != subject]
        im = render.render_shot(
            twin, cam_pos, subj_xy, focal, width_px=width_px,
            figures=figs, subject_height_m=height,
            draw_subject=figures_for is None and subject is not None and subject in fr.actors,
            subject_marks=others if figures_for is None else None,
        )
        strip = Image.new("RGB", (im.width, CAPTION_PX), (14, 16, 19))
        d = ImageDraw.Draw(strip)
        size = fr.framing.get("size_name", "") if fr.framing else ""
        clock = fr.clock.strftime("%H:%M:%S") if fr.clock else "--:--:--"
        sun = ""
        if fr.sun is not None:
            sun = "  sun " + ("on window " + ",".join(str(j + 1) for j in fr.sun["windows_lit"]) if fr.sun["direct_sun"] else "off the glass")
        d.text((6, 5), f"t={fr.t:5.2f}s  {clock}  shot {shot_no}  {focal:g}mm  {size}{sun}", fill=(232, 234, 237), font=font)
        canvas = Image.new("RGB", (im.width, im.height + CAPTION_PX), (14, 16, 19))
        canvas.paste(im, (0, 0))
        canvas.paste(strip, (0, im.height))
        path = out_dir / f"frame_{k:04d}.png"
        canvas.save(path)
        paths.append(path)
        meta.append({**fr.to_dict(), "file": path.name})
    (out_dir / "animatic.json").write_text(json.dumps({
        "title": rehearsal.scene.title,
        "every_s": every_s,
        "duration_s": round(rehearsal.duration_s, 3),
        "frames": meta,
        "problems": rehearsal.problems,
        "notes": rehearsal.notes,
    }, indent=1))
    return paths


# ---------------------------------------------------------------------------
# the overhead blocking diagram
# ---------------------------------------------------------------------------

_ACTOR_COLOURS = ("#e5c07b", "#57d38c", "#e0707a", "#c678dd", "#56b6c2", "#d19a66")


def blocking_svg(twin: Twin, rehearsal: Rehearsal) -> str:
    """The blocking diagram: footprint, marks, every actor's path with arrows
    and timecodes, and the camera for each shot. Drawn on the coverage
    planner's floor plan so the two diagrams read as one set."""
    scene = rehearsal.scene
    colours = {name: _ACTOR_COLOURS[i % len(_ACTOR_COLOURS)] for i, name in enumerate(
        dict.fromkeys([b.actor for b in scene.beats if b.actor] + list(scene.cast)))}

    def overlay(X, Y):
        parts: list[str] = []
        # marks
        for name, (mx, my) in scene.marks.items():
            parts.append(f'<circle cx="{X(mx):.1f}" cy="{Y(my):.1f}" r="8" fill="none" stroke="#e5c07b" stroke-width="1.5" stroke-dasharray="2 2"/>')
            parts.append(f'<text x="{X(mx):.1f}" y="{Y(my) - 11:.1f}" fill="#e5c07b" font-size="10" text-anchor="middle">{_esc(name)}</text>')
        # paths with arrows and timecodes
        for b, bt in zip(scene.beats, rehearsal.beats):
            if not bt.route or len(bt.route) < 2:
                continue
            col = colours.get(b.actor, "#e8eaed")
            pts = " ".join(f"{X(p[0]):.1f},{Y(p[1]):.1f}" for p in bt.route)
            dash = ' stroke-dasharray="5 4"' if bt.blocked else ""
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" stroke-opacity="0.9"{dash}/>')
            (x0, y0), (x1, y1) = bt.route[-2], bt.route[-1]
            ang = math.atan2(Y(y1) - Y(y0), X(x1) - X(x0))
            tip = (X(x1), Y(y1))
            l = (tip[0] - 9 * math.cos(ang - 0.4), tip[1] - 9 * math.sin(ang - 0.4))
            r = (tip[0] - 9 * math.cos(ang + 0.4), tip[1] - 9 * math.sin(ang + 0.4))
            parts.append(f'<polygon points="{tip[0]:.1f},{tip[1]:.1f} {l[0]:.1f},{l[1]:.1f} {r[0]:.1f},{r[1]:.1f}" fill="{col}"/>')
            sx, sy = bt.route[0]
            parts.append(f'<text x="{X(sx) + 4:.1f}" y="{Y(sy) - 4:.1f}" fill="{col}" font-size="9">{bt.t_start:.1f}s</text>')
            parts.append(f'<text x="{X(x1) + 4:.1f}" y="{Y(y1) + 12:.1f}" fill="{col}" font-size="9">{bt.t_end:.1f}s {_esc(b.actor)}</text>')
            if bt.blocked:
                mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
                parts.append(f'<text x="{X(mx):.1f}" y="{Y(my):.1f}" fill="#e0707a" font-size="10" text-anchor="middle">blocked</text>')
        # the cast, where the scene leaves them
        last = rehearsal.frames[-1] if rehearsal.frames else None
        if last is not None:
            for name, (ax, ay, facing) in last.actors.items():
                col = colours.get(name, "#e8eaed")
                parts.append(f'<circle cx="{X(ax):.1f}" cy="{Y(ay):.1f}" r="7" fill="{col}" fill-opacity="0.9"/>')
                fx = ax + 0.35 * math.cos(math.radians(facing))
                fy = ay + 0.35 * math.sin(math.radians(facing))
                parts.append(f'<line x1="{X(ax):.1f}" y1="{Y(ay):.1f}" x2="{X(fx):.1f}" y2="{Y(fy):.1f}" stroke="{col}" stroke-width="2"/>')
                parts.append(f'<text x="{X(ax):.1f}" y="{Y(ay) + 18:.1f}" fill="{col}" font-size="10" text-anchor="middle">{_esc(name)}</text>')
        # cameras per shot: mark, wedge, sightline, dolly run
        for st in rehearsal.shots:
            s = st.setup
            cx, cy = float(s["cam_x"]), float(s["cam_y"])
            if s.get("dolly_to"):
                dx, dy = s["dolly_to"]
                parts.append(f'<line x1="{X(cx):.1f}" y1="{Y(cy):.1f}" x2="{X(dx):.1f}" y2="{Y(dy):.1f}" stroke="#6ea8fe" stroke-width="4" stroke-opacity="0.5"/>')
            if "subj_x" in s:
                yaw = math.radians(float(s["yaw_deg"]))
                half = math.radians(float(s["fov_h_deg"]) / 2)
                reach = min(float(s["distance_m"]) * 1.15, 4.0)
                l = (cx + reach * math.cos(yaw - half), cy + reach * math.sin(yaw - half))
                r = (cx + reach * math.cos(yaw + half), cy + reach * math.sin(yaw + half))
                parts.append(f'<path d="M{X(cx):.1f},{Y(cy):.1f} L{X(l[0]):.1f},{Y(l[1]):.1f} L{X(r[0]):.1f},{Y(r[1]):.1f} Z" '
                             f'fill="#6ea8fe" fill-opacity="0.10" stroke="#6ea8fe" stroke-opacity="0.5" stroke-width="1"/>')
                parts.append(f'<line x1="{X(cx):.1f}" y1="{Y(cy):.1f}" x2="{X(s["subj_x"]):.1f}" y2="{Y(s["subj_y"]):.1f}" '
                             f'stroke="#6ea8fe" stroke-width="1" stroke-dasharray="3 3" stroke-opacity="0.8"/>')
            parts.append(f'<circle cx="{X(cx):.1f}" cy="{Y(cy):.1f}" r="11" fill="#0e1013" stroke="#6ea8fe" stroke-width="2.5"/>')
            parts.append(f'<text x="{X(cx):.1f}" y="{Y(cy) + 4:.1f}" fill="#e8eaed" font-size="11" font-weight="700" text-anchor="middle">{st.number}</text>')
            parts.append(f'<text x="{X(cx) + 14:.1f}" y="{Y(cy) - 7:.1f}" fill="#8b95a1" font-size="10">{s["focal_mm"]:g}mm · h{s["cam_z"]:.2f} · {st.t_start:.1f}-{st.t_end:.1f}s</text>')
        return parts

    caption = f"{scene.title} — blocking, {rehearsal.duration_s:.1f} s, {len(rehearsal.shots)} setups"
    return floor_plan_svg(twin, None, overlay=overlay, caption=caption)
