"""Coverage: the shot list for a scene, planned against a real room.

On a recce the director of photography walks the room with a viewfinder and,
for every shot the scene needs -- the master, each single, the over-the-
shoulders, the insert -- works out where the camera stands, at what height,
on which lens, whether the sightline is clean, whether there is room to back
up, and whether a window ends up behind the actor. Then someone draws the
overhead camera diagram and types the shot list. It is a day of work per
scene per location, and it is the day this module does.

The division of labour matches the rest of Locaish. A *shot* is a statement
of intent (a medium close-up of MAYA, on a long lens, no window behind her);
the shot table in ClickHouse holds every setup the room physically allows;
finding the setup is a filter over that table, and the filter is compiled
here from the shot's fields. The same predicate list drives two backends --
SQL against ClickHouse, and numpy against the in-memory sweep -- so the
studio degrades to a working planner when the warehouse is unreachable, and
the tests run without one. Whichever backend answers, the row it returns is
rendered through `render.py` and drawn on the floor plan, so every line of
the shot list can be checked by eye.

Nothing here decides what a scene *needs*; that is the breakdown, and it is
Gemini's job (`locaish/agent/coverage.py`). This module takes a shot list and
answers it.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..types import Twin
from . import optics
from . import sweep as sweepmod

# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------

SIZE_KEYS = tuple(s.key for s in optics.SHOT_SIZES)

# How a shot list talks about framing, mapped to the sweep's shot sizes. The
# aliases are the ordinary set-floor vocabulary; anything not here is left to
# the breakdown to translate.
SIZE_ALIASES: dict[str, str] = {
    "ecu": "ecu", "extreme close-up": "ecu", "extreme close up": "ecu", "insert": "ecu",
    "detail": "ecu", "macro": "ecu",
    "bcu": "bcu", "big close-up": "bcu", "big close up": "bcu", "choker": "bcu",
    "cu": "cu", "close-up": "cu", "close up": "cu", "closeup": "cu", "single": "cu",
    "mcu": "mcu", "medium close-up": "mcu", "medium close up": "mcu",
    "over the shoulder": "mcu", "over-the-shoulder": "mcu", "ots": "mcu",
    "ms": "ms", "medium": "ms", "medium shot": "ms", "mid": "ms", "mid shot": "ms",
    "two-shot": "ms", "two shot": "ms", "2-shot": "ms", "2 shot": "ms",
    "mls": "mls", "medium long": "mls", "medium long shot": "mls", "cowboy": "mls",
    "american": "mls", "three-quarter": "mls",
    "ls": "ls", "long shot": "ls", "full": "ls", "full shot": "ls", "wide": "ls",
    "wide shot": "ls", "master": "ls", "establishing": "ls",
    "els": "els", "extreme long shot": "els", "extreme wide": "els", "very wide": "els",
}

# Aliases that describe who is in the frame rather than how big they are.
_COMPOSITION_TERMS = {"two-shot", "two shot", "2-shot", "2 shot", "single", "ots",
                      "over the shoulder", "over-the-shoulder"}

HEIGHTS = ("low", "mid", "eye", "high")

# The sweep's working heights (see sweep.CAMERA_HEIGHTS_M) as the bands a
# shot list asks for. "high" has no sweep row above eye level -- a real
# high angle is a ladder or a jib -- so it resolves to the highest swept
# position and says so.
_HEIGHT_BANDS: dict[str, tuple[float, float]] = {
    "low": (0.0, 0.8),
    "mid": (0.8, 1.3),
    "eye": (1.3, 1.8),
    "high": (1.3, 9.0),
}

MOVEMENTS = ("static", "push-in", "pull-out", "dolly", "handheld", "pan")

# Two actors closer than this share a mark in all but name.
MIN_MARK_SEPARATION_M = 0.6


def normalise_size(text: str | None) -> str | None:
    if not text:
        return None
    t = text.strip().lower()
    if t in SIZE_ALIASES:
        return SIZE_ALIASES[t]
    # A size word beats a composition word: "wide two-shot" is a wide that
    # happens to hold two people. Among size words the longest match wins.
    hits = [(len(k), v) for k, v in SIZE_ALIASES.items() if k in t and k not in _COMPOSITION_TERMS]
    if hits:
        return max(hits)[1]
    hits = [(len(k), v) for k, v in SIZE_ALIASES.items() if k in t]
    if hits:
        return max(hits)[1]
    return None


def neighbour_sizes(key: str) -> list[str]:
    """The named sizes either side of `key`, nearest first."""
    i = SIZE_KEYS.index(key)
    out = []
    for step in (1, -1, 2, -2):
        j = i + step
        if 0 <= j < len(SIZE_KEYS):
            out.append(SIZE_KEYS[j])
    return out


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


@dataclass
class Shot:
    """One line of a shot list: what the director wants, before geometry."""

    number: int
    description: str
    size: str                          # one of SIZE_KEYS
    subject: str = "A"                 # name of the mark the shot frames
    second_subject: str | None = None  # a two-shot or OTS: must also be in frame
    lens_mm: float | None = None       # a preferred prime, or None for any
    height: str | None = None          # low | mid | eye | high | None
    movement: str = "static"
    no_window_behind: bool = False
    window_in_frame: bool | None = None
    notes: str = ""
    ots: bool = False                  # over the shoulder of second_subject, onto subject

    @property
    def slug(self) -> str:
        return f"{self.number}"

    @property
    def size_name(self) -> str:
        return optics.SHOT_BY_KEY[self.size].name

    def normalised(self) -> "Shot":
        size = normalise_size(self.size) or self.size
        if size not in optics.SHOT_BY_KEY:
            raise ValueError(f"shot {self.number}: unknown size {self.size!r}")
        height = (self.height or "").strip().lower() or None
        if height and height not in HEIGHTS:
            height = {"eye level": "eye", "eyeline": "eye", "waist": "mid", "hip": "mid",
                      "floor": "low", "ground": "low", "table": "mid"}.get(height)
        lens = float(self.lens_mm) if self.lens_mm else None
        if lens is not None:
            lens = optics.nearest_prime(lens, sweepmod.SWEEP_PRIMES_MM)
        movement = (self.movement or "static").strip().lower()
        if movement not in MOVEMENTS:
            movement = "static"
        return Shot(
            number=int(self.number),
            description=self.description.strip(),
            size=size,
            subject=(self.subject or "A").strip() or "A",
            second_subject=(self.second_subject or None) and self.second_subject.strip(),
            lens_mm=lens,
            height=height,
            movement=movement,
            no_window_behind=bool(self.no_window_behind),
            window_in_frame=self.window_in_frame,
            notes=(self.notes or "").strip(),
            ots=bool(self.ots) and bool(self.second_subject),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "Shot":
        return cls(
            number=int(d.get("number") or d.get("no") or 0),
            description=str(d.get("description") or d.get("desc") or ""),
            size=str(d.get("size") or d.get("shot_size") or "ms"),
            subject=str(d.get("subject") or "A"),
            second_subject=d.get("second_subject") or None,
            lens_mm=d.get("lens_mm") or d.get("lens") or None,
            height=d.get("height") or None,
            movement=str(d.get("movement") or "static"),
            no_window_behind=bool(d.get("no_window_behind") or False),
            window_in_frame=d.get("window_in_frame"),
            notes=str(d.get("notes") or ""),
            ots=bool(d.get("ots") or False),
        ).normalised()


# The columns of one shot_setups row, as the planner keeps them.
SETUP_COLUMNS = (
    "setup_id", "cam_x", "cam_y", "cam_z", "subj_x", "subj_y", "distance_m",
    "yaw_deg", "pitch_deg", "focal_mm", "sensor", "shot_size", "size_fit",
    "framed_height_m", "subject_fill", "fov_h_deg", "dof_near_m", "dof_far_m",
    "dof_infinite", "visible", "surveyed", "clearance_m", "headroom_m",
    "window_in_frame", "window_behind_subject", "background_depth_m", "backup_room_m",
    "key_angle_deg", "key_quality", "axis_wall_angle_deg", "portrait_ok", "score",
)


@dataclass
class Review:
    """What Gemini said when it looked at the frame."""

    score: float                 # 0-10
    verdict: str                 # "keep" | "adjust" | "reject"
    notes: str
    suggestion: dict = field(default_factory=dict)   # e.g. {"height": "low", "lens_mm": 50}
    model: str = ""


@dataclass
class PlannedShot:
    shot: Shot
    setup: dict | None            # a shot_setups row, or None when nothing fits
    candidates: int = 0           # rows that satisfied the final filters
    sql: str = ""                 # the query that found it (documentation of the claim)
    relaxed: list[str] = field(default_factory=list)
    frame: str | None = None      # rendered frame filename, under <plan>/frames/
    review: Review | None = None
    attempts: int = 1
    second_mark: tuple[float, float] | None = None
    why: str = ""                 # the DP's reasons, from the row's own numbers

    @property
    def ok(self) -> bool:
        return self.setup is not None

    def to_dict(self) -> dict:
        d = {
            "shot": asdict(self.shot),
            "setup": _jsonable(self.setup),
            "candidates": self.candidates,
            "sql": self.sql,
            "relaxed": list(self.relaxed),
            "frame": self.frame,
            "review": asdict(self.review) if self.review else None,
            "attempts": self.attempts,
            "second_mark": list(self.second_mark) if self.second_mark else None,
            "why": self.why,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PlannedShot":
        rv = d.get("review")
        return cls(
            shot=Shot.from_dict(d["shot"]),
            setup=d.get("setup"),
            candidates=int(d.get("candidates") or 0),
            sql=d.get("sql") or "",
            relaxed=list(d.get("relaxed") or []),
            frame=d.get("frame"),
            review=Review(**rv) if rv else None,
            attempts=int(d.get("attempts") or 1),
            second_mark=tuple(d["second_mark"]) if d.get("second_mark") else None,
            why=d.get("why") or "",
        )


@dataclass
class CoveragePlan:
    """A scene's coverage, planned: the shots, the marks, the diagram."""

    plan_id: str
    location: str
    title: str
    brief: str
    shots: list[PlannedShot]
    marks: dict[str, tuple[float, float]]
    planner: str = "deterministic"      # or "gemini"
    warnings: list[str] = field(default_factory=list)
    floor_plan_svg: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    trace: list[dict] = field(default_factory=list)

    @property
    def planned(self) -> int:
        return sum(1 for s in self.shots if s.ok)

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "location": self.location,
            "title": self.title,
            "brief": self.brief,
            "planner": self.planner,
            "created_at": self.created_at,
            "marks": {k: [float(v[0]), float(v[1])] for k, v in self.marks.items()},
            "shots": [s.to_dict() for s in self.shots],
            "planned": self.planned,
            "warnings": list(self.warnings),
            "floor_plan_svg": self.floor_plan_svg,
            "trace": self.trace,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CoveragePlan":
        return cls(
            plan_id=d["plan_id"],
            location=d["location"],
            title=d.get("title") or "",
            brief=d.get("brief") or "",
            shots=[PlannedShot.from_dict(s) for s in d.get("shots", [])],
            marks={k: (float(v[0]), float(v[1])) for k, v in (d.get("marks") or {}).items()},
            planner=d.get("planner") or "deterministic",
            warnings=list(d.get("warnings") or []),
            floor_plan_svg=d.get("floor_plan_svg") or "",
            created_at=d.get("created_at") or "",
            trace=list(d.get("trace") or []),
        )

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        out = directory / "plan.json"
        out.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        (directory / "shotlist.txt").write_text(render_text(self))
        if self.floor_plan_svg:
            (directory / "floorplan.svg").write_text(self.floor_plan_svg)
        return out

    @classmethod
    def load(cls, directory: str | Path) -> "CoveragePlan":
        return cls.from_dict(json.loads((Path(directory) / "plan.json").read_text()))


def new_plan_id() -> str:
    return uuid.uuid4().hex[:10]


# ---------------------------------------------------------------------------
# predicates: one definition, two backends
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Predicate:
    """A filter a shot imposes on the table, as SQL and as numpy.

    `name` is what the relaxation log reports when this filter has to be
    dropped to find anything; `hard` filters are never dropped.
    """

    name: str
    sql: str
    mask: Callable[[dict[str, np.ndarray]], np.ndarray]
    hard: bool = False


@dataclass
class PlanContext:
    """What earlier shots decided that later shots must respect.

    The line of action is the line through two characters' marks; the first
    shot that holds both picks a side, and every later shot on either of
    them stays on it (the 180-degree rule). A placed single is remembered so
    the reverse on the other character can match its lens and distance.
    """

    line: tuple[str, str] | None = None            # the two names
    line_side: int = 0                             # +1 / -1, 0 = not yet chosen
    singles: dict[tuple[str, str], dict] = field(default_factory=dict)   # (subject, size) -> setup

    def side_of(self, marks, cam_xy) -> int:
        if not self.line:
            return 0
        a, b = marks[self.line[0]], marks[self.line[1]]
        cross = (b[0] - a[0]) * (cam_xy[1] - a[1]) - (b[1] - a[1]) * (cam_xy[0] - a[0])
        return 1 if cross > 0 else -1 if cross < 0 else 0

    def learn(self, shot: "Shot", setup: dict, marks) -> None:
        names = [n for n in (shot.subject, shot.second_subject) if n]
        if len(names) == 2 and self.line is None:
            self.line = (names[0], names[1])
        if self.line and self.line_side == 0 and set(names) & set(self.line):
            # The first shot on the line fixes the side. A single sets it too:
            # once the audience has seen A looking left, B must look right.
            self.line_side = self.side_of(marks, (setup["cam_x"], setup["cam_y"]))
        if len(names) == 1:
            self.singles[(shot.subject, shot.size)] = setup

    def reverse_of(self, shot: "Shot") -> dict | None:
        """A placed single of the other character at this size, if any."""
        if not self.line or shot.second_subject:
            return None
        other = [n for n in self.line if n != shot.subject]
        if len(other) != 1 or shot.subject not in self.line:
            return None
        return self.singles.get((other[0], shot.size))


def _q(s: str) -> str:
    return "'" + str(s).replace("\\", "\\\\").replace("'", "\\'") + "'"


def predicates(
    shot: Shot,
    marks: dict[str, tuple[float, float]],
    *,
    location: str,
    context: PlanContext | None = None,
) -> list[Predicate]:
    """The filters this shot imposes, hard ones first, in relaxation order.

    Relaxation drops from the *end* of the soft list, so the order here is
    the order of what a DP gives up first: the exact lens before the height,
    the height before the backlight rule, and the framing only last.
    """
    subj = marks.get(shot.subject)
    if subj is None:
        raise ValueError(f"shot {shot.number}: no mark named {shot.subject!r}; have {sorted(marks)}")
    sx, sy = float(subj[0]), float(subj[1])
    preds: list[Predicate] = [
        Predicate("location", f"location = {_q(location)}",
                  lambda c: np.ones(len(c["setup_id"]), dtype=bool), hard=True),
        Predicate("visible", "visible = 1", lambda c: c["visible"] > 0, hard=True),
        # The mark is a swept subject position; match it to the sweep's own
        # precision rather than exactly, so a mark quoted to two decimals
        # still finds its rows.
        Predicate("subject mark",
                  f"abs(subj_x - {sx:.3f}) < 0.05 AND abs(subj_y - {sy:.3f}) < 0.05",
                  lambda c: (np.abs(c["subj_x"] - sx) < 0.05) & (np.abs(c["subj_y"] - sy) < 0.05),
                  hard=True),
    ]
    if shot.second_subject and shot.second_subject in marks:
        bx, by = (float(v) for v in marks[shot.second_subject])
        # The other actor is in frame when the bearing to their mark lies
        # within half the horizontal field of view of the lens axis. Pure
        # angles, and ClickHouse has the trigonometry. An over-the-shoulder
        # puts them at the very edge, so its tolerance is the full half-FOV.
        tol = 1.0 if shot.ots else 0.85
        bearing = f"(degrees(atan2({by:.3f} - cam_y, {bx:.3f} - cam_x)) - yaw_deg)"
        wrapped = f"abs({bearing} - 360 * floor(({bearing} + 180) / 360))"
        preds.append(Predicate(
            "second subject in frame",
            f"{wrapped} <= fov_h_deg / 2 * {tol}",
            lambda c, bx=bx, by=by, tol=tol: (
                np.abs(_wrap_deg(np.degrees(np.arctan2(by - c["cam_y"], bx - c["cam_x"])) - c["yaw_deg"]))
                <= c["fov_h_deg"] / 2 * tol
            ),
            hard=True,
        ))
        if shot.ots:
            # Over the shoulder: the camera stands just behind and beside
            # the foreground actor, so their mark is nearer the lens than
            # the subject's and close to the axis.
            dist_b = f"sqrt(pow(cam_x - {bx:.3f}, 2) + pow(cam_y - {by:.3f}, 2))"
            preds.append(Predicate(
                "over the shoulder geometry",
                f"{dist_b} BETWEEN 0.45 AND 1.6 AND {dist_b} < distance_m",
                lambda c, bx=bx, by=by: (
                    (np.hypot(c["cam_x"] - bx, c["cam_y"] - by) >= 0.45)
                    & (np.hypot(c["cam_x"] - bx, c["cam_y"] - by) <= 1.6)
                    & (np.hypot(c["cam_x"] - bx, c["cam_y"] - by) < c["distance_m"])
                ),
                hard=True,
            ))
    if context is not None and context.line and context.line_side and shot.subject in context.line:
        # The 180-degree rule as arithmetic: the camera stays on the side of
        # the line through the two marks that the first shot chose.
        ax, ay = (float(v) for v in marks[context.line[0]])
        bx2, by2 = (float(v) for v in marks[context.line[1]])
        side = int(context.line_side)
        cross_sql = f"(({bx2:.3f} - {ax:.3f}) * (cam_y - {ay:.3f}) - ({by2:.3f} - {ay:.3f}) * (cam_x - {ax:.3f}))"
        preds.append(Predicate(
            "same side of the line",
            f"sign({cross_sql}) = {side}",
            lambda c, ax=ax, ay=ay, bx2=bx2, by2=by2, side=side: (
                np.sign((bx2 - ax) * (c["cam_y"] - ay) - (by2 - ay) * (c["cam_x"] - ax)) == side
            ),
            hard=True,
        ))
    # -- soft, in the order they are given up ---------------------------
    reverse = context.reverse_of(shot) if context is not None else None
    if reverse is not None:
        # A reverse cuts cleanly against its partner when lens and distance
        # match; these are the first preferences to go if the room refuses.
        f = float(reverse["focal_mm"])
        d = float(reverse["distance_m"])
        preds.append(Predicate(
            f"match reverse distance {d:.2f} m",
            f"abs(distance_m - {d:.3f}) <= {0.2 * d:.3f}",
            lambda c, d=d: np.abs(c["distance_m"] - d) <= 0.2 * d,
        ))
        preds.append(Predicate(
            f"match reverse lens {f:g} mm",
            f"focal_mm = {f:g}",
            lambda c, f=f: np.abs(c["focal_mm"] - f) < 0.01,
        ))
    preds.append(Predicate(
        f"framing = {shot.size}",
        f"shot_size = {_q(shot.size)}",
        lambda c, k=shot.size: c["shot_size"].astype(str) == k,
    ))
    if shot.no_window_behind:
        preds.append(Predicate("no window behind subject", "window_behind_subject = 0",
                               lambda c: c["window_behind_subject"] == 0))
    if shot.window_in_frame is True:
        preds.append(Predicate("window in frame", "window_in_frame = 1",
                               lambda c: c["window_in_frame"] > 0))
    elif shot.window_in_frame is False:
        preds.append(Predicate("window out of frame", "window_in_frame = 0",
                               lambda c: c["window_in_frame"] == 0))
    if shot.height:
        lo, hi = _HEIGHT_BANDS[shot.height]
        preds.append(Predicate(
            f"height {shot.height}",
            f"cam_z >= {lo:.2f} AND cam_z < {hi:.2f}",
            lambda c, lo=lo, hi=hi: (c["cam_z"] >= lo) & (c["cam_z"] < hi),
        ))
    if shot.lens_mm:
        preds.append(Predicate(
            f"lens {shot.lens_mm:g} mm",
            f"focal_mm = {float(shot.lens_mm):g}",
            lambda c, f=float(shot.lens_mm): np.abs(c["focal_mm"] - f) < 0.01,
        ))
    return preds


def _wrap_deg(a: np.ndarray) -> np.ndarray:
    return (a + 180.0) % 360.0 - 180.0


# Ordering is where the craft lives. The sweep's score is a physical
# tie-breaker (level camera, room to work, measured ground, framing on a
# named size); on top of it each kind of shot weighs what a DP weighs:
#
#   wide      depth behind the subject, an axis into a corner rather than
#             square onto a wall, room to back the camera up
#   tight     a window keying from three-quarter or side rather than from
#             behind the camera (flat) or behind the subject (silhouette),
#             the camera far enough back that the face is not stretched
#   medium    a little of each
#
# The same expression is emitted as SQL and evaluated in numpy, so the
# ranking the agent sees in ClickHouse is the ranking the local planner uses.
KEY_PREF = {"three-quarter": 1.0, "side": 0.8, "rim": 0.6, "none": 0.5, "front": 0.3, "back": 0.0}
WIDE_SIZES = ("mls", "ls", "els")


def cine_terms(shot: Shot) -> list[tuple[float, str, Callable[[dict], np.ndarray], str]]:
    """(weight, sql expression, numpy expression, label) for a shot's kind."""
    key_sql = "multiIf(" + ", ".join(f"key_quality = '{k}', {v}" for k, v in KEY_PREF.items()) + ", 0.5)"
    wants_backlight = shot.window_in_frame is True or (
        not shot.no_window_behind and "backlit" in shot.description.lower()
    )
    if wants_backlight:
        # The brief asked for the window behind them: the silhouette is the shot.
        pref = {"back": 1.0, "rim": 0.8}
        key_sql = "multiIf(key_quality = 'back', 1.0, key_quality = 'rim', 0.8, 0.4)"
        key_np = lambda c, pref=pref: np.array([pref.get(str(k), 0.4) for k in c["key_quality"]])
    else:
        key_np = lambda c: np.array([KEY_PREF.get(str(k), 0.5) for k in c["key_quality"]])
    depth_sql = "least(background_depth_m, 4) / 4"
    depth_np = lambda c: np.minimum(c["background_depth_m"], 4.0) / 4.0
    corner_sql = "greatest(axis_wall_angle_deg, 0) / 45"
    corner_np = lambda c: np.maximum(c["axis_wall_angle_deg"], 0.0) / 45.0
    backup_sql = "least(backup_room_m, 2) / 2"
    backup_np = lambda c: np.minimum(c["backup_room_m"], 2.0) / 2.0
    portrait_sql = "portrait_ok"
    portrait_np = lambda c: c["portrait_ok"].astype(np.float64)
    if shot.size in WIDE_SIZES:
        return [(0.5, depth_sql, depth_np, "depth behind the subject"),
                (0.3, corner_sql, corner_np, "axis into a corner"),
                (0.2, backup_sql, backup_np, "room to back up")]
    if shot.size in ("ecu", "bcu", "cu", "mcu"):
        return [(0.5, key_sql, key_np, "window as key"),
                (0.3, portrait_sql, portrait_np, "distance that flatters a face"),
                (0.2, depth_sql, depth_np, "depth behind the subject")]
    return [(0.4, key_sql, key_np, "window as key"),
            (0.3, depth_sql, depth_np, "depth behind the subject"),
            (0.3, corner_sql, corner_np, "axis into a corner")]


def order_sql(shot: Shot) -> str:
    terms = " + ".join(f"{w} * ({sql})" for w, sql, _, _ in cine_terms(shot))
    return f"ORDER BY (score / 100 + {terms}) DESC, size_fit DESC, distance_m ASC"


def order_values(shot: Shot, c: dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
    sub = {k: v[idx] for k, v in c.items()}
    total = sub["score"].astype(np.float64) / 100.0
    for w, _, fn, _ in cine_terms(shot):
        total = total + w * np.asarray(fn(sub), dtype=np.float64)
    return total


def _order_key(shot: Shot, c: dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
    return np.lexsort((c["distance_m"][idx], -c["size_fit"][idx], -order_values(shot, c, idx)))


def compile_sql(preds: list[Predicate], *, db: str, table: str, limit: int = 5, shot: Shot | None = None) -> str:
    where = " AND ".join(p.sql for p in preds)
    cols = ", ".join(SETUP_COLUMNS)
    order = order_sql(shot) if shot is not None else "ORDER BY score DESC, size_fit DESC, distance_m ASC"
    return f"SELECT {cols} FROM {db}.{table} WHERE {where} {order} LIMIT {limit}"


def compile_count_sql(preds: list[Predicate], *, db: str, table: str) -> str:
    where = " AND ".join(p.sql for p in preds)
    return f"SELECT count() FROM {db}.{table} WHERE {where}"


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------


class SetupSource:
    """Where candidate setups come from. Two implementations, one contract."""

    kind = "abstract"

    def search(self, preds: list[Predicate], *, limit: int = 5, shot: Shot | None = None) -> tuple[list[dict], int, str]:
        """(best rows, count matching, the query as text)."""
        raise NotImplementedError

    def marks(self, location: str) -> list[tuple[float, float]]:
        raise NotImplementedError

    def by_id(self, location: str, setup_id: int) -> dict | None:
        """One row by its setup_id, or None."""
        raise NotImplementedError


class LocalSetups(SetupSource):
    """The in-memory sweep. What the studio uses when ClickHouse is offline."""

    kind = "local"

    def __init__(self, sweep: sweepmod.ShotSweep):
        self.sweep = sweep
        self.c = sweep.columns

    def search(self, preds, *, limit: int = 5, shot: Shot | None = None):
        c = self.c
        mask = np.ones(len(self.sweep), dtype=bool)
        for p in preds:
            mask &= p.mask(c)
        idx = np.flatnonzero(mask)
        count = int(len(idx))
        if count:
            if shot is not None:
                idx = idx[_order_key(shot, c, idx)][:limit]
            else:
                idx = idx[np.lexsort((c["distance_m"][idx], -c["size_fit"][idx], -c["score"][idx]))][:limit]
        rows = [{k: _scalar(c[k][i]) for k in SETUP_COLUMNS} for i in idx]
        sql = compile_sql(preds, db="(local)", table="shot_setups", limit=limit, shot=shot)
        return rows, count, sql

    def marks(self, location: str):
        return [(float(x), float(y)) for x, y in np.asarray(self.sweep.subject_marks)]

    def by_id(self, location: str, setup_id: int):
        idx = np.flatnonzero(self.c["setup_id"] == int(setup_id))
        if not len(idx):
            return None
        i = int(idx[0])
        return {k: _scalar(self.c[k][i]) for k in SETUP_COLUMNS}


class ClickHouseSetups(SetupSource):
    """The warehouse. Read here over clickhouse-connect for the studio's own
    planner; the agent reads the same table through the MCP server."""

    kind = "clickhouse"

    def __init__(self, client=None):
        from .. import warehouse

        self.ch = client or warehouse.client()
        self.db = warehouse.database()
        self.table = warehouse.TABLE

    def search(self, preds, *, limit: int = 5, shot: Shot | None = None):
        sql = compile_sql(preds, db=self.db, table=self.table, limit=limit, shot=shot)
        res = self.ch.query(sql)
        rows = [dict(zip(res.column_names, (_scalar(v) for v in r))) for r in res.result_rows]
        count = int(self.ch.query(compile_count_sql(preds, db=self.db, table=self.table)).result_rows[0][0])
        return rows, count, sql

    def marks(self, location: str):
        res = self.ch.query(
            f"SELECT DISTINCT subj_x, subj_y FROM {self.db}.{self.table} "
            f"WHERE location = {_q(location)} ORDER BY subj_x, subj_y"
        )
        return [(float(r[0]), float(r[1])) for r in res.result_rows]

    def by_id(self, location: str, setup_id: int):
        cols = ", ".join(SETUP_COLUMNS)
        res = self.ch.query(
            f"SELECT {cols} FROM {self.db}.{self.table} WHERE location = {_q(location)} "
            f"AND setup_id = {int(setup_id)} LIMIT 1"
        )
        if not res.result_rows:
            return None
        return dict(zip(res.column_names, (_scalar(v) for v in res.result_rows[0])))


def _scalar(v):
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, bytes):
        return v.decode()
    return v


def _jsonable(d):
    if d is None:
        return None
    return {k: _scalar(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# finding a setup for one shot
# ---------------------------------------------------------------------------


def find_setup(
    source: SetupSource,
    shot: Shot,
    marks: dict[str, tuple[float, float]],
    *,
    location: str,
    limit: int = 5,
    context: PlanContext | None = None,
) -> tuple[list[dict], int, str, list[str]]:
    """Best rows for a shot, relaxing soft constraints until something fits.

    Returns (rows, count, sql, relaxed). Relaxation is one filter at a time,
    last soft filter first, then a framing step to each neighbouring size;
    then, only then, the line of action -- crossing it is reported in so many
    words. A shot that survives none of it comes back empty rather than wrong.
    """
    preds = predicates(shot, marks, location=location, context=context)
    hard = [p for p in preds if p.hard and p.name != "same side of the line"]
    line = [p for p in preds if p.name == "same side of the line"]
    soft = [p for p in preds if not p.hard]
    relaxed: list[str] = []

    def run(ps):
        return source.search(ps, limit=limit, shot=shot)

    for attempt in (0, 1):
        base = hard + line if attempt == 0 else hard
        if attempt == 1:
            if not line:
                break
            relaxed.append("crossed the line of action (nothing on the chosen side)")
            soft = [p for p in preds if not p.hard]
        rows, count, sql = run(base + soft)
        while not rows and soft:
            # Framing is relaxed by stepping to a neighbour, not by dropping.
            dropped = soft.pop()
            if dropped.name.startswith("framing"):
                for nb in neighbour_sizes(shot.size):
                    alt = Predicate(f"framing = {nb}", f"shot_size = {_q(nb)}",
                                    lambda c, k=nb: c["shot_size"].astype(str) == k)
                    rows, count, sql = run(base + [alt] + soft)
                    if rows:
                        relaxed.append(f"framing {shot.size} -> {nb}")
                        return rows, count, sql, relaxed
                relaxed.append(f"framing {shot.size} (nothing at any neighbouring size)")
                rows, count, sql = run(base + soft)
            else:
                relaxed.append(dropped.name)
                rows, count, sql = run(base + soft)
        if rows:
            break
    # The second pass re-walks the same soft filters; say each thing once.
    seen: set[str] = set()
    relaxed = [r for r in relaxed if not (r in seen or seen.add(r))]
    return rows, count, sql, relaxed


# ---------------------------------------------------------------------------
# marks: where the actors can be
# ---------------------------------------------------------------------------


def describe_marks(twin: Twin, marks: list[tuple[float, float]]) -> list[dict]:
    """Each swept subject mark with the plain-language cues a breakdown needs.

    A breakdown says "MAYA at the window"; a mark is a coordinate. The bridge
    is distance to the things the twin knows the names of -- openings, walls
    by compass side, the room's centre -- so the assignment can be made
    from words and checked from numbers.
    """
    s = twin.structure
    fp = np.asarray(s.footprint, dtype=np.float64) if s.footprint is not None else None
    out = []
    centre = fp.mean(axis=0) if fp is not None and len(fp) else np.zeros(2)
    for i, (x, y) in enumerate(marks):
        d: dict[str, Any] = {"name": f"M{i + 1}", "x": round(float(x), 2), "y": round(float(y), 2)}
        cues = []
        for j, op in enumerate(s.openings):
            c = np.asarray(op.center, dtype=np.float64)
            dist = float(math.hypot(c[0] - x, c[1] - y))
            d[f"to_{op.kind}_{j + 1}_m"] = round(dist, 2)
            if dist < 1.2:
                cues.append(f"{dist:.1f} m from {op.kind} {j + 1}")
        if fp is not None and len(fp) >= 3:
            wall_d = _distance_to_polygon(fp, (x, y))
            d["to_nearest_wall_m"] = round(wall_d, 2)
            if wall_d < 0.6:
                cues.append("against a wall")
        dc = float(math.hypot(x - centre[0], y - centre[1]))
        d["to_room_centre_m"] = round(dc, 2)
        if dc < 0.7:
            cues.append("near the middle of the room")
        side = []
        if fp is not None and len(fp):
            if x > centre[0] + 0.4:
                side.append("east")
            elif x < centre[0] - 0.4:
                side.append("west")
            if y > centre[1] + 0.4:
                side.append("north")
            elif y < centre[1] - 0.4:
                side.append("south")
        if side:
            cues.append(f"{'-'.join(side)} side of the room")
        d["cues"] = cues
        out.append(d)
    return out


def _distance_to_polygon(poly: np.ndarray, p) -> float:
    px, py = float(p[0]), float(p[1])
    best = float("inf")
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        ab = b - a
        t = float(np.clip(((px - a[0]) * ab[0] + (py - a[1]) * ab[1]) / max(float(ab @ ab), 1e-9), 0, 1))
        q = a + t * ab
        best = min(best, float(math.hypot(px - q[0], py - q[1])))
    return best


def auto_marks(
    source: SetupSource, location: str, names: list[str]
) -> dict[str, tuple[float, float]]:
    """Assign character names to marks without a breakdown: most coverable first.

    The primary goes on the mark that the most visible setups frame; each
    further character on the best remaining mark at least a metre away, so
    a two-shot has two people rather than one standing in the other.
    """
    marks = source.marks(location)
    if not marks:
        raise ValueError("this location has no swept subject marks")
    scored = []
    for m in marks:
        preds = [
            Predicate("location", f"location = {_q(location)}", lambda c: np.ones(len(c["setup_id"]), bool), True),
            Predicate("visible", "visible = 1", lambda c: c["visible"] > 0, True),
            Predicate("mark", f"abs(subj_x - {m[0]:.3f}) < 0.05 AND abs(subj_y - {m[1]:.3f}) < 0.05",
                      lambda c, m=m: (np.abs(c["subj_x"] - m[0]) < 0.05) & (np.abs(c["subj_y"] - m[1]) < 0.05), True),
        ]
        _, count, _ = source.search(preds, limit=1)
        scored.append((count, m))
    scored.sort(key=lambda t: -t[0])
    out: dict[str, tuple[float, float]] = {}
    for name in names:
        pick = None
        # A metre apart when the room allows it, then whatever separation it
        # does allow; two names on one mark is the last resort.
        for sep in (1.0, MIN_MARK_SEPARATION_M, 0.0):
            for count, m in scored:
                if any(math.hypot(m[0] - o[0], m[1] - o[1]) < sep for o in out.values()):
                    continue
                pick = m
                break
            if pick is not None:
                break
        out[name] = (float(pick[0]), float(pick[1]))
    return out


# ---------------------------------------------------------------------------
# the deterministic planner
# ---------------------------------------------------------------------------


def plan(
    twin: Twin,
    shots: list[Shot],
    source: SetupSource,
    *,
    marks: dict[str, tuple[float, float]] | None = None,
    title: str = "",
    brief: str = "",
    workdir: str | Path | None = None,
    render: bool = True,
    progress: Callable[[str], None] | None = None,
    reviewer: Callable[["PlannedShot", Path], Review | None] | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> CoveragePlan:
    """Answer a shot list against the room. No model in the loop.

    This is the studio's planner when Gemini is unavailable, and the engine
    the agent's tools call when it is: the agent decides the shots and the
    marks, and re-decides after looking; the geometry is answered here.
    """
    location = twin.name
    shots = [s.normalised() for s in shots]
    names: list[str] = []
    for s in shots:
        for n in (s.subject, s.second_subject):
            if n and n not in names:
                names.append(n)
    if marks is None:
        marks = auto_marks(source, location, names)
    else:
        missing = [n for n in names if n not in marks]
        if missing:
            extra = auto_marks(source, location, missing)
            marks = {**marks, **extra}

    plan_id = new_plan_id()
    out = CoveragePlan(
        plan_id=plan_id, location=location, title=title or "Untitled scene",
        brief=brief, shots=[], marks=dict(marks),
    )
    frames_dir = Path(workdir) / "frames" if workdir else None
    context = PlanContext()

    def emit(kind: str, **extra) -> None:
        if on_event:
            try:
                on_event({"kind": kind, **extra})
            except Exception:  # noqa: BLE001 - a listener must never break the plan
                pass

    for shot in shots:
        if progress:
            progress(f"shot {shot.number}: {shot.size_name.lower()} of {shot.subject}")
        try:
            rows, count, sql, relaxed = find_setup(source, shot, marks, location=location, context=context)
        except ValueError as exc:
            out.warnings.append(f"shot {shot.number}: {exc}")
            out.shots.append(PlannedShot(shot=shot, setup=None))
            continue
        emit("candidates", shot=shot.number, rows=[row_brief(r) for r in rows], matched=count, sql=sql)
        ps = PlannedShot(shot=shot, setup=rows[0] if rows else None, candidates=count,
                         sql=sql, relaxed=relaxed)
        if shot.second_subject and shot.second_subject in marks:
            ps.second_mark = marks[shot.second_subject]
        if ps.setup:
            ps.why = explain(shot, ps.setup, marks, context)
            context.learn(shot, ps.setup, marks)
        if ps.setup and render and frames_dir is not None:
            try:
                ps.frame = render_frame(twin, ps, frames_dir)
            except Exception as exc:  # noqa: BLE001 - a frame is evidence, not the answer
                ps.frame = None
                out.warnings.append(f"shot {shot.number}: the frame could not be rendered: {exc}")
            if reviewer is not None and ps.frame:
                try:
                    ps.review = reviewer(ps, frames_dir / ps.frame)
                except Exception as exc:  # noqa: BLE001 - a review is advisory
                    out.warnings.append(f"shot {shot.number}: review failed: {exc}")
        if not ps.setup:
            out.warnings.append(
                f"shot {shot.number}: no setup in this room frames {shot.subject} as a "
                f"{shot.size_name.lower()} with a clear sightline"
            )
        out.shots.append(ps)
        emit("shot", shot=ps.to_dict())

    if progress:
        progress("drawing the camera plan")
    out.floor_plan_svg = floor_plan_svg(twin, out)
    if workdir:
        out.save(workdir)
    return out


def row_brief(r: dict) -> dict:
    """The dozen fields a person (or a viewfinder) needs from a row."""
    return {
        "setup_id": int(r["setup_id"]),
        "cam": [round(float(r["cam_x"]), 2), round(float(r["cam_y"]), 2), round(float(r["cam_z"]), 2)],
        "subject": [round(float(r["subj_x"]), 2), round(float(r["subj_y"]), 2)],
        "focal_mm": float(r["focal_mm"]), "shot_size": str(r["shot_size"]),
        "distance_m": round(float(r["distance_m"]), 2), "size_fit": round(float(r["size_fit"]), 2),
        "dof_m": ["inf" if int(r["dof_infinite"]) else round(float(r["dof_far_m"]), 2), round(float(r["dof_near_m"]), 2)],
        "window_behind_subject": int(r["window_behind_subject"]),
        "window_in_frame": int(r["window_in_frame"]),
        "key_quality": str(r.get("key_quality", "none")),
        "background_depth_m": round(float(r.get("background_depth_m", 0.0)), 2),
        "backup_room_m": round(float(r.get("backup_room_m", 0.0)), 2),
        "axis_wall_angle_deg": round(float(r.get("axis_wall_angle_deg", -1.0)), 0),
        "portrait_ok": int(r.get("portrait_ok", 1)),
        "clearance_m": round(float(r["clearance_m"]), 2), "score": round(float(r["score"]), 1),
    }


def explain(shot: Shot, st: dict, marks, context: PlanContext | None = None) -> str:
    """The reasons behind a placement, in the DP's words, from the row's numbers.

    Every clause is a column; nothing is asserted that the table does not
    hold. This is what goes on the shot card under the measurements.
    """
    bits = []
    f = int(st["focal_mm"])
    d = float(st["distance_m"])
    bits.append(f"{f} mm from {d:.1f} m at {float(st['cam_z']):.2f} m")
    kq = str(st.get("key_quality", "none"))
    ka = float(st.get("key_angle_deg", -1))
    if kq == "three-quarter":
        bits.append(f"the window keys from three-quarter ({ka:.0f}° off the lens axis)")
    elif kq == "side":
        bits.append(f"side light from the window ({ka:.0f}°)")
    elif kq == "rim":
        bits.append(f"the window rims from behind ({ka:.0f}°)")
    elif kq == "back":
        bits.append("the window is behind the subject — expect a silhouette or a flag")
    elif kq == "front":
        bits.append("the window is behind the camera, so the light is flat")
    depth = float(st.get("background_depth_m", 0.0))
    if depth >= 12.0 - 1e-6:
        bits.append("the background runs out past the capture")
    elif depth > 0:
        bits.append(f"{depth:.1f} m of depth behind the subject")
    wa = float(st.get("axis_wall_angle_deg", -1))
    if shot.size in WIDE_SIZES and wa >= 30:
        bits.append("the axis runs into a corner")
    elif shot.size in WIDE_SIZES and 0 <= wa < 15:
        bits.append("square onto a wall — flat background")
    if shot.size in ("ecu", "bcu", "cu", "mcu"):
        if int(st.get("portrait_ok", 1)):
            bits.append("far enough back that the face is not stretched")
        else:
            bits.append(f"inside {d:.1f} m the lens will widen the face; a longer lens from farther back would be kinder")
    if int(st["window_behind_subject"]):
        bits.append("window in frame behind the subject")
    if context is not None and context.line and context.line_side and shot.subject in context.line:
        bits.append(f"on the {'+' if context.line_side > 0 else '-'} side of the {context.line[0]}–{context.line[1]} line")
    if context is not None:
        rv = context.reverse_of(shot)
        if rv is not None:
            same_lens = abs(float(rv["focal_mm"]) - f) < 0.01
            close = abs(float(rv["distance_m"]) - d) <= 0.2 * float(rv["distance_m"])
            if same_lens and close:
                bits.append("matches the reverse's lens and distance")
            elif same_lens:
                bits.append("matches the reverse's lens")
    bu = float(st.get("backup_room_m", 0.0))
    if bu < 0.35:
        bits.append("no room to back up")
    return "; ".join(bits) + "."


def render_frame(twin: Twin, ps: PlannedShot, frames_dir: Path) -> str:
    from .render import render_shot

    st = ps.setup
    fname = f"shot_{ps.shot.number:02d}_{int(st['focal_mm'])}mm_{ps.attempts}.png"
    cam = (st["cam_x"], st["cam_y"], float(twin.structure.floor_z) + st["cam_z"])
    render_shot(
        twin, cam, (st["subj_x"], st["subj_y"]), float(st["focal_mm"]),
        sensor_key=str(st.get("sensor") or optics.DEFAULT_SENSOR),
        out=frames_dir / fname,
        subject_marks=[ps.second_mark] if ps.second_mark else None,
    )
    return fname


# ---------------------------------------------------------------------------
# the camera plan: an overhead diagram
# ---------------------------------------------------------------------------


_SVG_W = 720
_PAD_M = 0.6


def floor_plan_svg(twin: Twin, plan: CoveragePlan) -> str:
    """The overhead camera diagram a 1st AD tapes to the wall.

    Footprint, openings, the ground the capture actually surveyed, then every
    planned camera as a numbered mark with its field-of-view wedge and a
    line to the subject it frames. Metres throughout; a scale bar says so.
    """
    s = twin.structure
    xyz = np.asarray(twin.points.xyz)
    fp = np.asarray(s.footprint, dtype=np.float64) if s.footprint is not None and len(s.footprint) >= 3 else None
    lo = xyz[:, :2].min(axis=0) - _PAD_M
    hi = xyz[:, :2].max(axis=0) + _PAD_M
    if fp is not None:
        lo = np.minimum(lo, fp.min(axis=0) - _PAD_M)
        hi = np.maximum(hi, fp.max(axis=0) + _PAD_M)
    ext = np.maximum(hi - lo, 1e-3)
    scale = _SVG_W / ext[0]
    H = int(round(ext[1] * scale))

    def X(x):
        return (float(x) - lo[0]) * scale

    def Y(y):
        return (hi[1] - float(y)) * scale   # north up

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_SVG_W} {H}" width="{_SVG_W}" height="{H}" '
        f'font-family="ui-sans-serif, system-ui, sans-serif" font-size="12">',
        f'<rect width="{_SVG_W}" height="{H}" fill="#0e1013"/>',
    ]
    # grid every metre
    gx = math.floor(lo[0])
    while gx <= hi[0]:
        parts.append(f'<line x1="{X(gx):.1f}" y1="0" x2="{X(gx):.1f}" y2="{H}" stroke="#1c2128" stroke-width="1"/>')
        gx += 1
    gy = math.floor(lo[1])
    while gy <= hi[1]:
        parts.append(f'<line x1="0" y1="{Y(gy):.1f}" x2="{_SVG_W}" y2="{Y(gy):.1f}" stroke="#1c2128" stroke-width="1"/>')
        gy += 1

    # surveyed ground
    cb = twin.capture_bounds
    if cb is not None and cb.hull_xy is not None and len(cb.hull_xy) >= 3:
        pts = " ".join(f"{X(p[0]):.1f},{Y(p[1]):.1f}" for p in np.asarray(cb.hull_xy))
        parts.append(f'<polygon points="{pts}" fill="#57d38c" fill-opacity="0.06" stroke="#57d38c" '
                     f'stroke-opacity="0.35" stroke-width="1" stroke-dasharray="4 3"/>')
    # footprint
    if fp is not None:
        pts = " ".join(f"{X(p[0]):.1f},{Y(p[1]):.1f}" for p in fp)
        parts.append(f'<polygon points="{pts}" fill="#15181d" fill-opacity="0.6" stroke="#e8eaed" stroke-width="2.5"/>')
    # openings as ticks along their wall
    for j, op in enumerate(s.openings):
        c = np.asarray(op.center, dtype=np.float64)
        n = np.asarray(op.normal, dtype=np.float64)[:2]
        t = np.array([-n[1], n[0]])
        t = t / max(float(np.linalg.norm(t)), 1e-9)
        a = c[:2] - t * op.width / 2
        b = c[:2] + t * op.width / 2
        colour = "#6ea8fe" if op.kind == "window" else "#e5c07b"
        parts.append(f'<line x1="{X(a[0]):.1f}" y1="{Y(a[1]):.1f}" x2="{X(b[0]):.1f}" y2="{Y(b[1]):.1f}" '
                     f'stroke="{colour}" stroke-width="6" stroke-linecap="butt"/>')
        lx, ly = c[:2] + n * 0.25
        parts.append(f'<text x="{X(lx):.1f}" y="{Y(ly):.1f}" fill="{colour}" font-size="11" '
                     f'text-anchor="middle">{op.kind} {j + 1}</text>')
    # fixtures
    for fx in s.fixtures:
        poly = np.asarray(fx.footprint, dtype=np.float64)
        if len(poly) >= 3:
            pts = " ".join(f"{X(p[0]):.1f},{Y(p[1]):.1f}" for p in poly)
            parts.append(f'<polygon points="{pts}" fill="#8b95a1" fill-opacity="0.25" stroke="#8b95a1" stroke-width="1"/>')

    # cameras: wedge, sightline, numbered mark
    for ps in plan.shots:
        st = ps.setup
        if not st:
            continue
        cx, cy = float(st["cam_x"]), float(st["cam_y"])
        yaw = math.radians(float(st["yaw_deg"]))
        half = math.radians(float(st["fov_h_deg"]) / 2)
        reach = min(float(st["distance_m"]) * 1.15, 4.0)
        l = (cx + reach * math.cos(yaw - half), cy + reach * math.sin(yaw - half))
        r = (cx + reach * math.cos(yaw + half), cy + reach * math.sin(yaw + half))
        parts.append(f'<path d="M{X(cx):.1f},{Y(cy):.1f} L{X(l[0]):.1f},{Y(l[1]):.1f} A{reach * scale:.1f},{reach * scale:.1f} 0 0 0 '
                     f'{X(r[0]):.1f},{Y(r[1]):.1f} Z" fill="#6ea8fe" fill-opacity="0.10" stroke="#6ea8fe" stroke-opacity="0.5" stroke-width="1"/>')
        parts.append(f'<line x1="{X(cx):.1f}" y1="{Y(cy):.1f}" x2="{X(st["subj_x"]):.1f}" y2="{Y(st["subj_y"]):.1f}" '
                     f'stroke="#6ea8fe" stroke-width="1" stroke-dasharray="3 3" stroke-opacity="0.8"/>')
    # Shots that share a camera position share one mark, labelled with all
    # their numbers: two setups on the same spot are the norm (the master and
    # the two-shot from the corner), and stacked circles hide each other.
    groups: dict[tuple[int, int], list] = {}
    for ps in plan.shots:
        st = ps.setup
        if not st:
            continue
        key = (int(round(float(st["cam_x"]) / 0.15)), int(round(float(st["cam_y"]) / 0.15)))
        groups.setdefault(key, []).append(ps)
    for members in groups.values():
        st = members[0].setup
        cx, cy = float(st["cam_x"]), float(st["cam_y"])
        nums = ",".join(str(m.shot.number) for m in members)
        r = 11 if len(members) == 1 else 13 + 3 * min(len(members) - 1, 3)
        parts.append(f'<circle cx="{X(cx):.1f}" cy="{Y(cy):.1f}" r="{r}" fill="#0e1013" stroke="#6ea8fe" stroke-width="2.5"/>')
        parts.append(f'<text x="{X(cx):.1f}" y="{Y(cy) + 4:.1f}" fill="#e8eaed" font-size="{11 if len(nums) < 4 else 9}" '
                     f'font-weight="700" text-anchor="middle">{nums}</text>')
        lenses = sorted({int(m.setup["focal_mm"]) for m in members})
        heights = sorted({round(float(m.setup["cam_z"]), 2) for m in members})
        label = "/".join(str(l) for l in lenses) + "mm · h" + "/".join(f"{h:.2f}" for h in heights)
        parts.append(f'<text x="{X(cx) + r + 3:.1f}" y="{Y(cy) - r + 4:.1f}" fill="#8b95a1" font-size="10">{label}</text>')

    # subject marks, on top: an actor's mark must never hide under a camera
    for name, (mx, my) in plan.marks.items():
        parts.append(f'<circle cx="{X(mx):.1f}" cy="{Y(my):.1f}" r="9" fill="#e5c07b" fill-opacity="0.95"/>')
        parts.append(f'<text x="{X(mx):.1f}" y="{Y(my) + 4:.1f}" fill="#0e1013" font-size="10" font-weight="700" '
                     f'text-anchor="middle">{_esc(name[:3])}</text>')
        parts.append(f'<text x="{X(mx):.1f}" y="{Y(my) - 12:.1f}" fill="#e5c07b" font-size="10" '
                     f'text-anchor="middle">{_esc(name)}</text>')

    # scale bar and legend
    parts.append(f'<line x1="16" y1="{H - 18}" x2="{16 + scale:.1f}" y2="{H - 18}" stroke="#e8eaed" stroke-width="2"/>')
    parts.append(f'<text x="{20 + scale:.1f}" y="{H - 14}" fill="#8b95a1" font-size="11">1 m</text>')
    if twin.georeference is not None:
        parts.append(f'<text x="{_SVG_W - 16}" y="20" fill="#8b95a1" font-size="11" text-anchor="end">N ↑ '
                     f'({twin.georeference.heading_source} heading)</text>')
    else:
        parts.append(f'<text x="{_SVG_W - 16}" y="20" fill="#8b95a1" font-size="11" text-anchor="end">+y ↑ · heading unknown</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# the shot list, as text
# ---------------------------------------------------------------------------


def render_text(plan: CoveragePlan) -> str:
    lines = [
        f"COVERAGE PLAN: {plan.title}",
        f"  location {plan.location}   planned {plan.planned}/{len(plan.shots)} shots   "
        f"planner: {plan.planner}   {plan.created_at}",
        "",
        "MARKS",
    ]
    for name, (x, y) in plan.marks.items():
        lines.append(f"  {name:<10} x {x:+.2f}  y {y:+.2f}  (twin frame, metres)")
    lines += ["", "SHOTS"]
    hdr = f"  {'#':<3} {'SIZE':<5} {'LENS':<6} {'HEIGHT':<7} {'CAMERA (x, y)':<16} {'SUBJ':<6} {'DIST':<6} {'DOF':<13} {'LIGHT':<10} {'FIT':<5} DESCRIPTION"
    lines.append(hdr)
    for ps in plan.shots:
        sh = ps.shot
        if not ps.setup:
            lines.append(f"  {sh.number:<3} {sh.size:<5} {'-':<6} {'-':<7} {'NO SETUP':<16} {sh.subject[:6]:<6} "
                         f"{'':<6} {'':<13} {'':<10} {'':<5} {sh.description}")
            continue
        st = ps.setup
        dof = "to inf" if st["dof_infinite"] else f"{st['dof_near_m']:.2f}-{st['dof_far_m']:.2f} m"
        light = "backlit!" if st["window_behind_subject"] else ("window in" if st["window_in_frame"] else "clean")
        lines.append(
            f"  {sh.number:<3} {st['shot_size']:<5} {int(st['focal_mm']):>3} mm {st['cam_z']:.2f} m  "
            f"({st['cam_x']:+.2f}, {st['cam_y']:+.2f})  {sh.subject[:6]:<6} {st['distance_m']:.2f} m "
            f"{dof:<13} {light:<10} {st['size_fit']:.2f}  {sh.description}"
        )
        extras = []
        if ps.why:
            extras.append(ps.why)
        if ps.candidates:
            extras.append(f"{ps.candidates} setups matched")
        if ps.relaxed:
            extras.append("relaxed: " + "; ".join(ps.relaxed))
        if ps.review:
            extras.append(f"Gemini {ps.review.score:.0f}/10 {ps.review.verdict}: {ps.review.notes}")
        if sh.notes:
            extras.append(sh.notes)
        for e in extras:
            lines.append(f"      · {e}")
    if plan.warnings:
        lines += ["", "READ THIS BEFORE SHOOTING"]
        for w in plan.warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# parsing a structured brief without a model
# ---------------------------------------------------------------------------

# Capitalised words that are shot-list jargon rather than characters.
_NOT_NAMES = {"CU", "MCU", "MS", "MLS", "LS", "ELS", "ECU", "BCU", "OTS", "INT", "EXT",
              "POV", "VFX", "SFX", "MOS", "DAY", "NIGHT", "CONT", "CONTD", "AND", "WITH",
              "OVER", "ON", "THE", "OF", "TO", "AT", "IN"}

_LINE_RE = re.compile(
    r"^\s*(?P<num>\d+)?[\.\)]?\s*(?P<body>.+?)\s*$"
)


def parse_shot_lines(text: str) -> list[Shot]:
    """A shot list typed one shot per line, read without a model.

    Recognises the size word, an optional lens ("on a 50", "50mm"), a height
    word (low/eye/high), a second name after "and"/"with"/"over", and the
    phrases "no window behind" / "window in frame". Names are the words in
    capitals. Anything else is the description. This is the path for a
    studio without Gemini; the breakdown agent does the same job from prose.
    """
    shots: list[Shot] = []
    n = 0
    last_subject = "A"
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        body = m.group("body") if m else line
        n = int(m.group("num")) if m and m.group("num") else n + 1
        low = body.lower()
        size = normalise_size(low) or "ms"
        lens = None
        lm = re.search(r"(\d{2,3})\s*mm|on an?\s+(\d{2,3})\b", low)
        if lm:
            lens = float(lm.group(1) or lm.group(2))
        height = None
        for h in ("low", "high", "eye"):
            if re.search(rf"\b{h}(?:[ -]angle| level|line)?\b", low):
                height = h
                break
        if height is None:
            # Mood words carry a height: the camera drops to make someone
            # loom and rises to make them small.
            if re.search(r"\b(power|powerful|dominant|dominates|menacing|looming|imposing|threat)", low):
                height = "low"
            elif re.search(r"\b(vulnerable|small|weak|diminished|cornered|helpless|trapped)", low):
                height = "high"
        ots = bool(re.search(r"\b(ots|over[- ]the[- ]shoulder|over \w+'?s? shoulder)", low))
        names = [w for w in re.findall(r"\b([A-Z][A-Z]{1,15})\b", body)
                 if w not in _NOT_NAMES]
        subject = names[0] if names else last_subject
        second = names[1] if len(names) > 1 else None
        if ots and second and re.search(r"over\s+" + re.escape(names[0]), body, re.I):
            # "over JON's shoulder on MAYA": the shot is of MAYA.
            subject, second = second, subject
        last_subject = subject
        movement = "static"
        for mv in ("push-in", "push in", "pull-out", "pull out", "dolly", "handheld", "pan"):
            if mv in low:
                movement = mv.replace(" ", "-")
                break
        shots.append(Shot(
            number=n, description=body, size=size, subject=subject, second_subject=second,
            lens_mm=lens, height=height, movement=movement,
            no_window_behind=("no window behind" in low or "not backlit" in low or "no backlight" in low),
            window_in_frame=True if ("window in frame" in low or "window in shot" in low) else None,
            ots=ots,
        ).normalised())
    return shots
