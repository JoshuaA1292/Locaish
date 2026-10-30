"""The rehearsal, run by a model that may only ask for things the room can measure.

A director pastes a scene. Three kinds of thinking turn it into a rehearsal
in the real room, and only the first is the model's:

**Breakdown.** Muse Spark reads the scene with the room in front of it -- the
marks the sweep can frame an actor on, each with its plain-language cues, the
openings, what is in the room by name, the cast with their heights -- and
answers with beats: who enters where, who walks to which mark, who says what,
and for the beats that want a camera, what the shot *is* (a close-up of Maya,
low, pushing in) rather than where the camera goes. It never writes a
coordinate the room did not offer it.

**Placement.** Every shot request goes to the shot table through the same
`find_setup` the coverage planner uses: a position, a height and a lens that
the geometry already scored, on the right side of the line of action. A push-in
becomes a short dolly along the camera axis, which the engine will check beat
by beat against the floor and the furniture.

**The run.** `film.rehearsal.rehearse` walks the actors, times the lines, holds
or rides the camera, tags the sun; `schedule` lays the shoot day out; the
animatic is rendered from the twin with the cast drawn in at their heights;
Muse Spark may write one paragraph of notes at the end, from the numbers.

Everything the model decides is a choice among measured options. Everything
in the output traces to a row, a route on the standable floor, an ephemeris,
or a catalogue. When no model is configured the same engine runs from a typed
scene (`film.rehearsal.parse_scene`), which is what the tests do.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import date as _date
from datetime import datetime, time as _time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..film import coverage as cov
from ..film import rehearsal as reh
from ..types import Twin

# How far a push-in or pull-out travels when the breakdown asks for one and
# says nothing more. A metre is a real move on a 50 mm and still fits most
# rooms; the engine shortens nothing -- it reports the beat where it fails.
DEFAULT_DOLLY_M = 1.0
MIN_CAMERA_SUBJECT_M = 1.2
# How many scored rows placement looks through for one whose eyeline no other
# actor stands in. The table's best row is still the best only if nobody is
# standing in front of the lens.
CANDIDATE_ROWS = 12

_BREAKDOWN_SYSTEM = """You are a first assistant director and a DP breaking down a scene for a rehearsal inside a measured 3D twin of a real room.
You answer with JSON only. You may place actors ONLY on the listed marks (by their names, M1, M2, ...), and you describe shots by what they are, never by coordinates: the measurement engine places the camera.
Shot sizes: ecu, cu, mcu, ms, mls, ws (extreme close-up to wide). Heights: low, mid, eye, high. Movements: static, push-in, pull-out, dolly."""

_BREAKDOWN_PROMPT = """THE ROOM
{facts}
{contents}
THE CAST (name: height in metres)
{cast}

THE SCENE
{scene}

Return this JSON object and nothing else:
{{
  "title": "<short title>",
  "beats": [
    {{"kind": "enter|move|line|hold|exit", "actor": "<cast name>", "to_mark": "<M# for enter/move, else null>",
      "text": "<the line for kind line, else null>", "seconds": <number for hold, else null>,
      "shot": null | {{"size": "cu|mcu|ms|mls|ws|ecu", "subject": "<cast name>", "second_subject": "<cast name or null>",
                       "height": "low|mid|eye|high|null", "movement": "static|push-in|pull-out|dolly",
                       "no_window_behind": true|false, "description": "<what the shot is for>"}}
    }}
  ],
  "strikes": [{{"before_shot": <beat number>, "piece": "<a thing in the room to remove>"}}]
}}
Rules: every actor must enter (kind enter with a mark) before doing anything else; a move needs a different mark; keep lines verbatim from the scene; cover the scene with 3 to 8 shots, each attached to the beat where it starts; one shot per beat at most; prefer marks whose cues fit the scene's geography (a window, a door, the middle of the room).
"""

_NOTES_PROMPT = """You are a DP writing one short paragraph (60 to 110 words) for the director about this rehearsal of {title} in a measured room.
Facts (all measured, none estimated):
{facts}
Say what the room gives the scene, what it costs (blocked moves, a dolly that does not fit, a window shot with a narrow sun window), and the one thing to decide before the shoot day. Plain prose, no bullet points, no numbers you were not given."""


class RehearsalUnavailable(RuntimeError):
    """The scene could not be rehearsed: no model for the breakdown, or a scene the room cannot hold."""


# ---------------------------------------------------------------------------
# breakdown: scene prose -> beats on named marks, shot requests
# ---------------------------------------------------------------------------


def _marks_catalogue(twin: Twin, source: cov.SetupSource) -> tuple[dict[str, tuple[float, float]], list[dict]]:
    described = cov.describe_marks(twin, source.marks(twin.name))
    marks = {d["name"]: (float(d["x"]), float(d["y"])) for d in described}
    return marks, described


def room_facts(twin: Twin, source: cov.SetupSource) -> str:
    """The room in words, with the marks the breakdown may use."""
    from .coverage import room_facts as _facts

    return _facts(twin, source)


def breakdown(
    scene_text: str,
    twin: Twin,
    source: cov.SetupSource,
    cast: dict[str, float],
    *,
    client,
    model: str | None = None,
    contents: list[dict] | None = None,
) -> dict:
    """One structured call: the scene becomes beats on marks and shot requests.

    Returns the model's dict, validated and normalised: unknown marks are
    snapped to the nearest listed one by name similarity or dropped with a
    note, unknown actors are dropped, beat kinds are forced into the engine's
    vocabulary, shot sizes and heights into the coverage planner's.
    """
    marks, described = _marks_catalogue(twin, source)
    if not marks:
        raise RehearsalUnavailable("the sweep found no mark an actor could stand on")
    cast_lines = "\n".join(f"{n}: {h:.2f}" for n, h in cast.items()) or "(no cast given: use the names in the scene, 1.75 m)"
    contents_text = ""
    if contents:
        names = ", ".join(f"{c.get('label')} at x {float(c['xyz'][0]):+.1f}, y {float(c['xyz'][1]):+.1f}"
                          for c in contents if c.get("label") and c.get("xyz"))
        if names:
            contents_text = f"IN THE ROOM\n{names}\n"
    prompt = _BREAKDOWN_PROMPT.format(facts=room_facts(twin, source), contents=contents_text,
                                      cast=cast_lines, scene=scene_text.strip())
    data = client.chat_json(prompt, system=_BREAKDOWN_SYSTEM, model=model, max_tokens=4000)
    if not isinstance(data, dict):
        raise RehearsalUnavailable("the breakdown did not come back as an object")
    return normalise_breakdown(data, marks, cast, scene_text)


def normalise_breakdown(data: dict, marks: dict[str, tuple[float, float]], cast: dict[str, float],
                        scene_text: str = "") -> dict:
    names = {n.lower(): n for n in cast} if cast else {}
    notes: list[str] = []

    def actor_of(v: Any) -> str | None:
        s = str(v or "").strip()
        if not s:
            return None
        if not names:
            return s
        if s.lower() in names:
            return names[s.lower()]
        for low, real in names.items():
            if low in s.lower() or s.lower() in low:
                return real
        notes.append(f"the breakdown named {s!r}, who is not in the cast; the beat was dropped")
        return None

    def mark_of(v: Any) -> str | None:
        s = str(v or "").strip().upper()
        if s in marks:
            return s
        digits = "".join(ch for ch in s if ch.isdigit())
        if digits and f"M{int(digits)}" in marks:
            return f"M{int(digits)}"
        return None

    beats: list[dict] = []
    for raw in data.get("beats") or []:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind") or "").strip().lower()
        kind = {"enters": "enter", "walk": "move", "walks": "move", "moves": "move", "says": "line",
                "speak": "line", "dialogue": "line", "wait": "hold", "pause": "hold", "exits": "exit",
                "leave": "exit"}.get(kind, kind)
        if kind not in reh.BEAT_KINDS:
            continue
        actor = actor_of(raw.get("actor"))
        if actor is None:
            continue
        to_mark = mark_of(raw.get("to_mark")) if kind in ("enter", "move") else None
        if kind in ("enter", "move") and to_mark is None:
            if marks:
                to_mark = next(iter(marks))
                notes.append(f"{actor}'s {kind} named no listed mark; put on {to_mark}")
        text = raw.get("text") if kind == "line" else None
        seconds = raw.get("seconds") if kind == "hold" else None
        try:
            seconds = None if seconds is None else max(0.5, float(seconds))
        except (TypeError, ValueError):
            seconds = 1.5
        shot = raw.get("shot")
        req = None
        if isinstance(shot, dict) and shot:
            size = cov.normalise_size(str(shot.get("size") or "ms")) or "ms"
            subj = actor_of(shot.get("subject")) or actor
            second = actor_of(shot.get("second_subject")) if shot.get("second_subject") else None
            height = str(shot.get("height") or "").strip().lower() or None
            if height not in cov.HEIGHTS:
                height = None
            movement = str(shot.get("movement") or "static").strip().lower()
            if movement not in cov.MOVEMENTS:
                movement = "static"
            req = {"size": size, "subject": subj, "second_subject": second if second != subj else None,
                   "height": height, "movement": movement,
                   "no_window_behind": bool(shot.get("no_window_behind")),
                   "description": str(shot.get("description") or "")}
        beats.append({"kind": kind, "actor": actor, "to_mark": to_mark, "text": text,
                      "seconds": seconds, "shot": req})
    strikes = []
    for s in data.get("strikes") or []:
        if isinstance(s, dict) and s.get("piece"):
            try:
                strikes.append({"before_shot": int(s.get("before_shot") or 1), "piece": str(s["piece"])})
            except (TypeError, ValueError):
                continue
    return {"title": str(data.get("title") or "Scene"), "beats": beats, "strikes": strikes, "notes": notes}


# ---------------------------------------------------------------------------
# placement: shot requests -> setups from the table
# ---------------------------------------------------------------------------


def _positions_through(beats: list[dict], marks: dict[str, tuple[float, float]]) -> list[dict[str, str]]:
    """For each beat, where every actor present stands (mark name) when it starts."""
    where: dict[str, str] = {}
    out = []
    for b in beats:
        out.append(dict(where))
        if b["kind"] in ("enter", "move") and b.get("to_mark"):
            where[b["actor"]] = b["to_mark"]
        elif b["kind"] == "exit":
            where.pop(b["actor"], None)
    return out


def _occluder_positions(where: dict[str, str], standing: dict[str, str], marks: dict[str, tuple[float, float]],
                        exempt: set[str]) -> dict[str, tuple[float, float]]:
    """Where every other actor is during the beat: at its start and after its
    own move, as separate keys when the two differ."""
    out: dict[str, tuple[float, float]] = {}
    for name in set(where) | set(standing):
        if name in exempt:
            continue
        spots = []
        for src in (where, standing):
            m = src.get(name)
            if m in marks and marks[m] not in spots:
                spots.append(marks[m])
        for k, xy in enumerate(spots):
            out[name if k == 0 else f"{name}@{k}"] = (float(xy[0]), float(xy[1]))
    return out


def _behind_on_line(cam_xy, subject_xy, others: dict[str, tuple[float, float]],
                    radius_m: float = reh.OCCLUDER_RADIUS_M) -> list[str]:
    """Others standing on the eyeline beyond the subject: not in the way, but
    stacked straight behind them in the frame."""
    cam = np.asarray(cam_xy, dtype=np.float64)[:2]
    subj = np.asarray(subject_xy, dtype=np.float64)[:2]
    axis = subj - cam
    length = float(np.linalg.norm(axis))
    if length <= 1e-9:
        return []
    axis /= length
    out = []
    for name, pos in others.items():
        d = np.asarray(pos, dtype=np.float64)[:2] - cam
        along = float(d @ axis)
        across = abs(float(d[0] * axis[1] - d[1] * axis[0]))
        if along > length - radius_m and across <= radius_m:
            out.append(name)
    return out


def _clean_name(key: str) -> str:
    return key.split("@", 1)[0]


def _choose_row(rows: list[dict], shot: cov.Shot, subject_xy, others: dict[str, tuple[float, float]],
                heights: dict[str, float] | None) -> tuple[dict, dict]:
    """The best-scored row whose eyeline no other actor cuts and whose lens no
    other actor crowds; among those, one with nobody stacked behind the subject.
    Falls back to the best row, saying who is in the way."""
    subj_h = float((heights or {}).get(shot.subject, 1.75))
    subject_xyz = (float(subject_xy[0]), float(subject_xy[1]), reh.EYELINE_RATIO * subj_h)
    hts = None if heights is None else {k: float(heights.get(_clean_name(k), 1.75)) for k in others}
    judged = []
    for k, st in enumerate(rows):
        cam = (float(st["cam_x"]), float(st["cam_y"]), float(st["cam_z"]))
        blocked = sorted({_clean_name(n) for n in reh.actor_blocks(cam, subject_xyz, others, heights=hts)})
        close = sorted({_clean_name(n) for n in reh.actor_too_close(cam, others)})
        behind = sorted({_clean_name(n) for n in _behind_on_line(cam, subject_xy, others)})
        # A lens inside the subject's own body (the sweep's straight-up and
        # straight-down rows) has no eyeline for anyone to cross; it is not a
        # clear shot of them either, and never a replacement for one.
        inside = math.dist(cam[:2], subject_xy[:2]) < reh.OCCLUDER_RADIUS_M
        judged.append((k, st, blocked, close, behind, inside))
    clean = [j for j in judged if not j[2] and not j[3] and not j[5]]
    if clean:
        k, st, _b, _c, behind, _i = min(clean, key=lambda j: (len(j[4]), j[0]))
        return st, {"blocked_by": [], "too_close": [], "behind_subject": behind, "skipped_rows": k}
    k, st, blocked, close, behind, _i = judged[0]
    return st, {"blocked_by": blocked, "too_close": close, "behind_subject": behind, "skipped_rows": 0,
                "occluded": True}


def place_shots(
    beats: list[dict],
    marks: dict[str, tuple[float, float]],
    twin: Twin,
    source: cov.SetupSource,
    *,
    dolly_m: float = DEFAULT_DOLLY_M,
    cast: dict[str, float] | None = None,
    limit: int = CANDIDATE_ROWS,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[reh.Beat], list[dict]]:
    """Every shot request becomes a `ShotSpec` from a scored row, or a note.

    The row is the best-scored one that no other actor present at the beat
    stands in front of: the room is not the only occluder. When every
    candidate is blocked the best row is taken and the placement says who is
    in the way (`blocked_by`, `too_close`). `cast` gives heights, so a seated
    or short actor the eyeline clears is not counted.
    """
    positions = _positions_through(beats, marks)
    ctx = cov.PlanContext()
    out: list[reh.Beat] = []
    placements: list[dict] = []
    shot_no = 0
    for i, (b, where) in enumerate(zip(beats, positions), 1):
        spec = None
        req = b.get("shot")
        if req:
            shot_no += 1
            # Where the subject stands *after* this beat's own move.
            standing = dict(where)
            if b["kind"] in ("enter", "move") and b.get("to_mark"):
                standing[b["actor"]] = b["to_mark"]
            subj_mark = standing.get(req["subject"])
            if subj_mark is None:
                placements.append({"beat": i, "shot": shot_no, "placed": False,
                                   "why": f"{req['subject']} is not in the room at beat {i}"})
            else:
                char_marks = {name: marks[m] for name, m in standing.items() if m in marks}
                shot = cov.Shot(
                    number=shot_no, description=req.get("description") or "", size=req["size"],
                    subject=req["subject"], second_subject=req.get("second_subject"),
                    height=req.get("height"), movement=req.get("movement") or "static",
                    no_window_behind=bool(req.get("no_window_behind")),
                ).normalised()
                if shot.second_subject and shot.second_subject not in char_marks:
                    shot = cov.Shot(**{**shot.__dict__, "second_subject": None, "ots": False})
                if progress:
                    progress(f"placing shot {shot_no}: {shot.size_name} of {shot.subject}")
                rows, count, sql, relaxed = cov.find_setup(source, shot, char_marks, location=twin.name,
                                                           limit=limit, context=ctx)
                if rows:
                    exempt = {shot.subject}
                    if shot.second_subject:
                        exempt.add(shot.second_subject)
                    others = _occluder_positions(where, standing, marks, exempt)
                    st, verdict = _choose_row(rows, shot, char_marks[shot.subject], others, cast)
                    ctx.learn(shot, st, char_marks)
                    spec = _spec_from_row(st, shot, char_marks[shot.subject], dolly_m)
                    placement = {"beat": i, "shot": shot_no, "placed": True, "setup": st,
                                 "relaxed": relaxed, "candidates": count, "sql": sql,
                                 "request": req, "best_setup_id": rows[0].get("setup_id"), **verdict}
                    if verdict.get("occluded"):
                        who = " and ".join(verdict["blocked_by"] or verdict["too_close"])
                        placement["why"] = (f"{who} stands between every candidate camera and {shot.subject} "
                                            f"at beat {i}; the best row is used and the rehearsal will show it")
                        if progress:
                            progress(f"shot {shot_no}: {placement['why']}")
                    elif verdict["skipped_rows"]:
                        placement["why"] = (f"the top {verdict['skipped_rows']} row(s) had another actor in the "
                                            f"eyeline or on the lens; row {verdict['skipped_rows'] + 1} is clear")
                    placements.append(placement)
                else:
                    placements.append({"beat": i, "shot": shot_no, "placed": False, "request": req,
                                       "why": "no setup in this room holds this shot with a clear sightline",
                                       "relaxed": relaxed, "sql": sql})
        out.append(reh.Beat(number=i, kind=b["kind"], actor=b["actor"], text=b.get("text"),
                            to_mark=b.get("to_mark"), seconds=b.get("seconds"), shot=spec))
    return out, placements


def _spec_from_row(st: dict, shot: cov.Shot, subject_xy, dolly_m: float) -> reh.ShotSpec:
    cam = np.array([float(st["cam_x"]), float(st["cam_y"])])
    subj = np.array([float(subject_xy[0]), float(subject_xy[1])])
    dolly_to = None
    gear = None
    if shot.movement in ("push-in", "pull-out", "dolly"):
        axis = subj - cam
        dist = float(np.linalg.norm(axis))
        if dist > 1e-6:
            axis = axis / dist
            if shot.movement == "push-in":
                travel = min(dolly_m, max(0.0, dist - MIN_CAMERA_SUBJECT_M))
                end = cam + axis * travel
            elif shot.movement == "pull-out":
                end = cam - axis * dolly_m
            else:
                side = np.array([-axis[1], axis[0]])
                end = cam + side * dolly_m
            if float(np.linalg.norm(end - cam)) > 0.1:
                dolly_to = (float(end[0]), float(end[1]))
                gear = "doorway-dolly"
    return reh.ShotSpec(cam_xy=(float(cam[0]), float(cam[1])), height_m=float(st["cam_z"]),
                        focal_mm=float(st["focal_mm"]), subject=shot.subject,
                        size=str(st.get("shot_size") or shot.size), dolly_to=dolly_to, gear_key=gear)


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


@dataclass
class RehearsalResult:
    scene: reh.Scene
    rehearsal: reh.Rehearsal
    schedule: reh.Schedule | None
    placements: list[dict]
    animatic: list[str]
    blocking_svg: str
    callsheet: str
    notes: str = ""
    provider: str = "typed"
    breakdown_notes: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "scene": self.scene.to_dict(),
            "rehearsal": self.rehearsal.to_dict(),
            "schedule": None if self.schedule is None else self.schedule.to_dict(),
            "placements": [_jsonable(p) for p in self.placements],
            "animatic": list(self.animatic),
            "blocking_svg": self.blocking_svg,
            "callsheet": self.callsheet,
            "notes": self.notes,
            "provider": self.provider,
            "breakdown_notes": list(self.breakdown_notes),
            "seconds": round(self.seconds, 1),
        }

    def save(self, workdir: Path) -> Path:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "rehearsal.json").write_text(json.dumps(self.to_dict(), indent=1))
        (workdir / "callsheet.txt").write_text(self.callsheet)
        (workdir / "blocking.svg").write_text(self.blocking_svg)
        return workdir / "rehearsal.json"


def _jsonable(v):
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def run_rehearsal(
    twin: Twin,
    source: cov.SetupSource,
    *,
    workdir: Path,
    scene_text: str | None = None,
    scene: reh.Scene | None = None,
    cast: dict[str, float] | None = None,
    actors: list | None = None,
    pieces: list | None = None,
    contents: list[dict] | None = None,
    shoot_date: _date | str | None = None,
    shoot_start: _time | str | None = None,
    setup_minutes: int = reh.DEFAULT_SETUP_MINUTES,
    client=None,
    model: str | None = None,
    every_s: float = 1.0,
    progress: Callable[[str], None] | None = None,
) -> RehearsalResult:
    """Scene in, rehearsal out: files on disk and a result the page can read.

    Give `scene` (typed) or `scene_text` (prose, needs `client`). `actors` are
    `product.cast.Actor`s for the figures; `cast` maps names to heights and
    defaults to the actors' own. `pieces` are set pieces; moved ones change the
    room the actors walk and are drawn where they now stand.
    """
    t0 = time.perf_counter()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda m: None)

    cast = dict(cast or {})
    for a in actors or []:
        cast.setdefault(a.name, float(a.height_m))

    # The room the scene plays in: the twin with every piece where it was put.
    room = twin
    piece_figures: list = []
    if pieces:
        from ..product import setpieces as spm

        moved = [p for p in pieces if any(abs(float(v)) > 1e-6 for v in (p.pose.get("dx", 0), p.pose.get("dy", 0), p.pose.get("dyaw_deg", 0)))] \
            if hasattr(pieces[0], "pose") and isinstance(getattr(pieces[0], "pose"), dict) else []
        try:
            room = spm.apply(twin, pieces)
            piece_figures = [spm.figure(p, twin, workdir.parent) for p in moved]
        except Exception as exc:  # noqa: BLE001 - a bad piece must not stop the rehearsal
            say(f"set pieces could not be applied ({exc}); rehearsing the room as scanned")
            room = twin
            piece_figures = []

    provider = "typed"
    breakdown_notes: list[str] = []
    placements: list[dict] = []
    if scene is None:
        if not scene_text or not scene_text.strip():
            raise RehearsalUnavailable("give a scene: typed beats or prose")
        looks_typed = any(line.strip().upper().startswith(("MARKS:", "CAST:")) for line in scene_text.splitlines())
        if looks_typed:
            say("reading the typed scene")
            scene = reh.parse_scene(scene_text)
        else:
            if client is None:
                raise RehearsalUnavailable(
                    "a prose scene needs a model for the breakdown (set META_API_KEY), or type the "
                    "scene as beats: MARKS:, CAST:, then numbered beats"
                )
            say("Muse Spark is breaking the scene down")
            bd = breakdown(scene_text, room, source, cast, client=client, model=model, contents=contents)
            breakdown_notes = bd["notes"]
            marks, _ = _marks_catalogue(room, source)
            used = {b["to_mark"] for b in bd["beats"] if b.get("to_mark")}
            say(f"{len(bd['beats'])} beats on {len(used)} marks; placing the shots")
            beats, placements = place_shots(bd["beats"], marks, room, source, cast=cast or None, progress=say)
            if not cast:
                cast = {b.actor: 1.75 for b in beats}
                breakdown_notes.append("no heights were given; every actor is 1.75 m until cast")
            scene = reh.Scene(title=bd["title"], beats=beats,
                              marks={m: marks[m] for m in used if m in marks},
                              cast={n: cast.get(n, 1.75) for n in dict.fromkeys(b.actor for b in beats)},
                              strikes=bd["strikes"])
            provider = "meta"
    if shoot_date:
        scene.shoot_date = shoot_date if isinstance(shoot_date, _date) else _date.fromisoformat(str(shoot_date))
    if shoot_start:
        scene.shoot_start = shoot_start if isinstance(shoot_start, _time) else reh._parse_hhmm(str(shoot_start))

    say("rehearsing: walking the floor, timing the lines, riding the camera")
    rehearsal = reh.rehearse(room, scene)

    sched = None
    if scene.shoot_date is not None and room.georeference is not None:
        say("laying out the shoot day against the sun")
        start = datetime.combine(scene.shoot_date, scene.shoot_start or _time(8, 0))
        try:
            sched = reh.schedule(rehearsal, room, setup_minutes=setup_minutes, start=start)
        except Exception as exc:  # noqa: BLE001 - the rehearsal stands without a schedule
            rehearsal.notes.append(f"no schedule: {exc}")

    # Figures: the cast at their marks, and the pieces where they were moved to.
    figures_for = None
    by_name = {a.name: a for a in (actors or [])}
    # Every scene draws its cast as bodies, named or not: a scene with no
    # cast still has people in it, at the heights the breakdown gave them.
    if True:
        from ..product import cast as castmod
        from ..product import figures as figmod

        floor_z = float(room.structure.floor_z)
        frames = rehearsal.frames
        colour_order: dict[str, int] = {}

        def speed_of(name: str, frame) -> float:
            k = min(range(len(frames)), key=lambda i: abs(frames[i].t - frame.t)) if frames else 0
            j = k + 1 if k + 1 < len(frames) else k - 1
            if j < 0 or j == k:
                return 0.0
            a, b = frames[k].actors.get(name), frames[j].actors.get(name)
            if not a or not b:
                return 0.0
            dt = abs(frames[j].t - frames[k].t) or 1.0
            return float(math.hypot(b[0] - a[0], b[1] - a[1]) / dt)

        def figures_for(frame):
            figs = list(piece_figures)
            for name, (x, y, facing) in frame.actors.items():
                a = by_name.get(name)
                if a is None:
                    idx = colour_order.setdefault(name, len(colour_order))
                    a = castmod.Actor(name=name, height_m=float(scene.cast.get(name, 1.75)),
                                      colour=castmod.PALETTE[idx % len(castmod.PALETTE)])
                if getattr(a, "body_path", None):
                    figs.append(castmod.figure(a, (x, y), facing, floor_z, workdir=workdir.parent))
                    continue
                v = speed_of(name, frame)
                pose = "walking" if v > 0.15 else "standing"
                figs.append(figmod.figure_for_actor(a, (x, y), facing, floor_z, pose=pose,
                                                    phase=figmod.walk_phase(frame.t, v) if pose == "walking" else 0.0))
            return figs

    say("rendering the animatic")
    frames = reh.animatic(room, rehearsal, workdir / "animatic", every_s=every_s, figures_for=figures_for)
    svg = reh.blocking_svg(room, rehearsal)
    callsheet = reh.render_text(rehearsal, sched)

    notes = ""
    if client is not None:
        try:
            say("Muse Spark is writing the notes")
            notes = write_notes(rehearsal, sched, placements, client=client, model=model)
        except Exception as exc:  # noqa: BLE001
            rehearsal.notes.append(f"no notes from the model: {exc}")

    result = RehearsalResult(
        scene=scene, rehearsal=rehearsal, schedule=sched, placements=placements,
        animatic=[p.name for p in frames], blocking_svg=svg, callsheet=callsheet, notes=notes,
        provider=provider, breakdown_notes=breakdown_notes, seconds=time.perf_counter() - t0,
    )
    result.save(workdir)
    return result


def write_notes(rehearsal: reh.Rehearsal, sched: reh.Schedule | None, placements: list[dict], *,
                client, model: str | None = None) -> str:
    facts = [f"running time {rehearsal.duration_s:.0f} s over {len(rehearsal.scene.beats)} beats, {len(rehearsal.shots)} setups"]
    for s in rehearsal.shots:
        st = s.setup
        facts.append(f"shot {s.number}: {st.get('shot_size') or ''} {st.get('focal_mm', 0):.0f} mm at {st.get('cam_z', 0):.2f} m on {st.get('subject', '')}"
                     + (f"; problems: {'; '.join(s.problems)}" if s.problems else ""))
    for p in placements:
        if not p.get("placed"):
            facts.append(f"shot {p.get('shot')} could not be placed: {p.get('why')}")
        elif p.get("blocked_by") or p.get("too_close"):
            facts.append(f"shot {p.get('shot')}: {p.get('why')}")
    facts += [f"problem: {p}" for p in rehearsal.problems]
    if sched is not None:
        facts += sched.sentences
    text = _NOTES_PROMPT.format(title=rehearsal.scene.title, facts="\n".join(facts))
    from ..meta.client import model_for

    res = client.chat([{"role": "user", "content": text}], model=model or model_for("breakdown"), max_tokens=700)
    return (res.text or "").strip()
