"""The coverage planner as an agent workflow: break down, then find-look-decide.

A scout who plans coverage does three different kinds of thinking, and the
workflow here is built from three different kinds of agent because of it.

**Breakdown** is reading: a page of scene turns into a list of shots and a
cast list with each character on a mark. That is a single Gemini call with
a schema on its output, because the value is the judgement (what does this
scene need covered) and the geometry has not entered yet.

**Placement** is a loop, and it is the reason this is an agent rather than a
script. For each shot the planner asks the shot table for candidates -- the
compiled filter from `film.coverage`, or its own SQL through the ClickHouse
MCP server when it wants something the compiler does not offer -- places the
best one, and then *looks at the frame*: the setup is rendered from the
twin's gaussian field and shown to Gemini, which scores it as a DP would
(headroom, background, whether the second actor is really in the frame) and
may send the planner back to the table with a different height or lens.
Two attempts per shot, then the better one stands. The placement agent is
invoked once per shot with a fresh context -- the loop is in Python, on
purpose: an agent that carries six shots of tool traffic in one
conversation re-reads all of it on every call, and the bill for a scout
quadruples for no better placements.

**Notes** is writing: one short paragraph of what the room gives the scene
and what it costs, from the placed shots and the reviews.

Every number in the result comes from a row or a render; the models decide
what to ask for and whether what came back is good. The same engine
(`film.coverage.plan`) answers a shot list with no model at all, which is
what the studio does when Gemini is not configured, and what the tests do.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from google.adk.tools import ToolContext

from .. import warehouse
from ..film import coverage as cov
from ..types import Twin
from .core import DEFAULT_MODEL, AgentUnavailable

APP_NAME = "locaish-coverage"

MAX_ATTEMPTS_PER_SHOT = 2

# Spend guard: the most model calls one plan may make, across all its
# agents, before the rest is placed without Gemini. A normal scout is
# twenty to forty; this is the ceiling, not the expectation.
MAX_MODEL_CALLS = int(os.environ.get("LOCAISH_MAX_MODEL_CALLS", "80"))

# Frames are downscaled before review: a 720 px frame reads the same to the
# model as a 960 px one and costs fewer image tokens.
REVIEW_IMAGE_WIDTH = 720

# Each agent gets its own model so the run is spread across separate rate
# limit buckets -- the AI Studio free tier is five requests a minute *per
# model*, and a plan is a couple of dozen calls. All overridable; on Vertex
# or a paid key the same names work and the spread costs nothing.
AGENT_MODELS = {
    "breakdown": os.environ.get("LOCAISH_BREAKDOWN_MODEL", "gemini-3.5-flash"),
    "planner": os.environ.get("LOCAISH_PLANNER_MODEL", DEFAULT_MODEL),
    "reviewer": os.environ.get("LOCAISH_REVIEW_MODEL", "gemini-3.7-flash"),
    "notes": os.environ.get("LOCAISH_NOTES_MODEL", "gemini-3.5-flash"),
}


def _retry_options():
    from google.genai import types

    # 429 carries a retry-after of up to a minute on the free tier; the
    # backoff here reaches it by the fourth attempt.
    return types.HttpRetryOptions(
        attempts=6, initial_delay=4.0, max_delay=65.0, exp_base=2.0, jitter=0.2,
        http_status_codes=[429, 500, 502, 503, 504],
    )


def _model(name: str):
    """An ADK Gemini model with retries, or a BaseLlm double passed straight through."""
    if not isinstance(name, str):
        return name
    from google.adk.models.google_llm import Gemini

    return Gemini(model=name, retry_options=_retry_options())


def agent_configured() -> bool:
    """Whether Gemini can be reached at all, by either credential route."""
    return bool(
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
    )


# ---------------------------------------------------------------------------
# a planning run: the state the tools work on
# ---------------------------------------------------------------------------


class BudgetExhausted(RuntimeError):
    """The plan hit its spend guard; the rest is placed without Gemini."""


@dataclass
class _Run:
    plan_id: str
    twin: Twin
    workdir: Path
    source: cov.SetupSource
    brief: str
    title: str
    reviewer: Callable[[Path, cov.Shot, dict], cov.Review | None] | None
    on_event: Callable[[dict], None] | None
    shots: list[cov.Shot] = field(default_factory=list)
    marks: dict[str, tuple[float, float]] = field(default_factory=dict)
    planned: dict[int, cov.PlannedShot] = field(default_factory=dict)
    accepted: set[int] = field(default_factory=set)
    skipped: dict[int, str] = field(default_factory=dict)
    attempts: dict[int, list[cov.PlannedShot]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    breakdown: dict | None = None
    dp_notes: str = ""
    context: cov.PlanContext = field(default_factory=cov.PlanContext)
    model_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def spend(self) -> None:
        """One more model call; raises when the plan's budget is spent."""
        self.model_calls += 1
        if self.model_calls > MAX_MODEL_CALLS:
            raise BudgetExhausted(f"model-call budget of {MAX_MODEL_CALLS} spent")

    @property
    def location(self) -> str:
        return self.twin.name

    @property
    def frames_dir(self) -> Path:
        return self.workdir / "frames"

    def emit(self, kind: str, **extra) -> None:
        if self.on_event:
            try:
                self.on_event({"kind": kind, **extra})
            except Exception:  # noqa: BLE001 - a listener must never break the plan
                pass

    def pending(self) -> list[cov.Shot]:
        return [s for s in self.shots if s.number not in self.accepted and s.number not in self.skipped]


_RUNS: dict[str, _Run] = {}
_RUNS_LOCK = threading.Lock()


def _run(tool_context) -> _Run:
    pid = tool_context.state.get("plan_id") if tool_context is not None else None
    with _RUNS_LOCK:
        run = _RUNS.get(pid or "")
    if run is None:
        raise ValueError("no active coverage run in this session")
    return run


# ---------------------------------------------------------------------------
# the room, as the breakdown agent reads it
# ---------------------------------------------------------------------------


def room_facts(twin: Twin, source: cov.SetupSource) -> str:
    """The room in words and numbers: what the breakdown has to work with.

    Marks are the sweep's own subject positions with plain-language cues, so
    a character can be put "at the window" by name and the choice is still a
    coordinate the table knows.
    """
    s = twin.structure
    lines = [f"Location: {twin.name}"]
    if s.floor_area:
        lines.append(f"Floor area {float(s.floor_area):.1f} m2.")
    if s.ceiling_z is not None and s.floor_z is not None:
        lines.append(f"Ceiling {float(s.ceiling_z - s.floor_z):.2f} m.")
    else:
        lines.append("Ceiling not captured.")
    for j, op in enumerate(s.openings):
        c = op.center
        lines.append(
            f"{op.kind} {j + 1}: {op.width:.2f} x {op.height:.2f} m at x {float(c[0]):+.2f}, y {float(c[1]):+.2f}, "
            f"sill {op.sill_height:.2f} m."
        )
    qa = twin.qa
    lines.append(f"Twin QA verdict: {qa.verdict}.")
    lines.append("Subject marks (where an actor can stand and be framed):")
    for m in cov.describe_marks(twin, source.marks(twin.name)):
        cues = "; ".join(m.get("cues") or []) or "open floor"
        lines.append(f"  {m['name']}: x {m['x']:+.2f}, y {m['y']:+.2f} -- {cues}")
    lines.append(
        "Lenses swept: " + ", ".join(f"{int(f)} mm" for f in cov.sweepmod.SWEEP_PRIMES_MM)
        + ". Camera heights swept: " + ", ".join(f"{h:.2f} m" for h in cov.sweepmod.CAMERA_HEIGHTS_M) + "."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# tools for the placement loop
# ---------------------------------------------------------------------------


def next_shot(tool_context: ToolContext = None) -> dict:
    """The next shot still to be placed, with its marks and the compiled query.

    Returns {"done": true} when every shot is accepted or skipped -- call
    exit_loop then. Otherwise returns the shot's spec, the marks it uses,
    the number of attempts already made, and `suggested_sql`: the filter the
    planner compiles from the spec, which find_setups runs for you.
    """
    run = _run(tool_context)
    pending = run.pending()
    if not pending:
        return {"done": True, "accepted": sorted(run.accepted),
                "skipped": {str(k): v for k, v in run.skipped.items()}}
    shot = pending[0]
    preds = cov.predicates(shot, run.marks, location=run.location, context=run.context)
    db = warehouse.database() if warehouse.configured() else "(local)"
    continuity = []
    ctx = run.context
    if ctx.line and ctx.line_side and shot.subject in ctx.line:
        continuity.append(f"stay on the {'+' if ctx.line_side > 0 else '-'} side of the "
                          f"{ctx.line[0]}-{ctx.line[1]} line (the 180-degree rule; find_setups enforces it)")
    rv = ctx.reverse_of(shot)
    if rv is not None:
        continuity.append(f"this is the reverse of a placed single: match {int(rv['focal_mm'])} mm "
                          f"at about {float(rv['distance_m']):.2f} m so the cut is clean")
    # The ranked candidates come with the shot: one turn less per shot, and
    # the planner's first look is at real rows rather than at a template.
    rows, count, sql, relaxed = cov.find_setup(run.source, shot, run.marks, location=run.location,
                                               limit=5, context=run.context)
    briefs = [cov.row_brief(r) for r in rows]
    run.emit("candidates", shot=shot.number, rows=briefs, matched=count, sql=sql)
    return {
        "done": False,
        "shot": _shot_dict(shot),
        "marks": {k: [round(v[0], 2), round(v[1], 2)] for k, v in run.marks.items()
                  if k in (shot.subject, shot.second_subject)},
        "attempts_so_far": len(run.attempts.get(shot.number, [])),
        "max_attempts": MAX_ATTEMPTS_PER_SHOT,
        "remaining_after_this": len(pending) - 1,
        "continuity": continuity,
        "candidates": briefs,
        "matched": count,
        "relaxed": relaxed,
        "sql": sql,
    }


def find_setups(shot_number: int, drop: str = "", limit: int = 5, tool_context: ToolContext = None) -> dict:
    """Candidate setups for a shot from the shot table, best first.

    Applies the shot's own constraints (framing, lens, height, backlight,
    second subject in frame) and returns up to `limit` rows plus how many
    matched in total. If nothing matches, the planner relaxes constraints one
    at a time and reports which in `relaxed`. To loosen deliberately, name
    constraints in `drop`, comma-separated, from: lens, height, backlight,
    window, framing.

    Args:
        shot_number: The shot to search for.
        drop: Constraints to ignore, e.g. "lens,height". Empty for none.
        limit: Rows to return (1-10).
    """
    run = _run(tool_context)
    shot = _shot(run, shot_number)
    dropped = {d.strip().lower() for d in drop.split(",") if d.strip()}
    if dropped:
        shot = cov.Shot(**{**shot.__dict__,
                           "lens_mm": None if "lens" in dropped else shot.lens_mm,
                           "height": None if "height" in dropped else shot.height,
                           "no_window_behind": False if "backlight" in dropped else shot.no_window_behind,
                           "window_in_frame": None if "window" in dropped else shot.window_in_frame})
    rows, count, sql, relaxed = cov.find_setup(run.source, shot, run.marks, location=run.location,
                                               limit=max(1, min(int(limit), 10)), context=run.context)
    if "framing" in dropped and not rows:
        relaxed.append("framing dropped on request")
    briefs = [cov.row_brief(r) for r in rows]
    run.emit("candidates", shot=shot.number, rows=briefs, matched=count, sql=sql)
    return {
        "status": "success",
        "matched": count,
        "relaxed": relaxed + [f"dropped on request: {d}" for d in sorted(dropped)],
        "sql": sql,
        "rows": briefs,
    }


def place_shot(shot_number: int, setup_id: int, reasoning: str = "", tool_context: ToolContext = None) -> dict:
    """Place a shot on a setup, render the frame, and have it reviewed.

    Renders exactly this row through the twin (image_url in the result) and
    returns the reviewer's verdict: a 0-10 score, keep/adjust/reject, notes,
    and a suggestion (a different height, lens or framing) when it says
    adjust. After the second attempt the better-scoring frame stands; call
    accept_shot to close the shot, or find_setups again to try the suggestion.

    Args:
        shot_number: The shot being placed.
        setup_id: The setup_id of the chosen row from find_setups or run_query.
        reasoning: One line on why this setup, for the shot list.
    """
    run = _run(tool_context)
    shot = _shot(run, shot_number)
    attempts = run.attempts.setdefault(shot.number, [])
    if len(attempts) >= MAX_ATTEMPTS_PER_SHOT and shot.number in run.planned:
        best = run.planned[shot.number]
        return {"status": "closed", "note": "attempts exhausted; the best frame stands",
                "kept_setup_id": best.setup["setup_id"], "review": _review_dict(best.review)}
    row = run.source.by_id(run.location, int(setup_id))
    if row is None:
        return {"status": "error", "error": f"no setup {setup_id} at {run.location}"}
    if int(row.get("visible", 1)) == 0:
        return {"status": "error", "error": f"setup {setup_id} has no sightline to the subject; choose a visible row"}

    # The previous attempt's filters are what this row was found under; keep
    # the count and the SQL of the search that produced it when we have them.
    preds = cov.predicates(shot, run.marks, location=run.location, context=run.context)
    sql = cov.compile_sql(preds, db=warehouse.database() if warehouse.configured() else "(local)",
                          table=warehouse.TABLE, shot=shot)
    ps = cov.PlannedShot(shot=shot, setup=row, candidates=0, sql=sql, attempts=len(attempts) + 1,
                         why=cov.explain(shot, row, run.marks, run.context))
    if shot.second_subject and shot.second_subject in run.marks:
        ps.second_mark = run.marks[shot.second_subject]
    if reasoning:
        ps.shot = cov.Shot(**{**shot.__dict__, "notes": reasoning.strip()})
    run.emit("stage", text=f"rendering shot {shot.number} ({int(row['focal_mm'])} mm)")
    try:
        ps.frame = cov.render_frame(run.twin, ps, run.frames_dir)
    except Exception as exc:  # noqa: BLE001 - the setup stands even when its picture does not
        ps.frame = None
        run.warnings.append(f"shot {shot.number}: the frame could not be rendered: {exc}")
    review = None
    if run.reviewer is not None and ps.frame:
        run.emit("stage", text=f"Gemini is looking at shot {shot.number}")
        try:
            run.spend()
            _LAST_REVIEW_USAGE[0] = (0, 0)
            review = run.reviewer(run.frames_dir / ps.frame, ps.shot, row)
            run.tokens_in += _LAST_REVIEW_USAGE[0][0]
            run.tokens_out += _LAST_REVIEW_USAGE[0][1]
        except BudgetExhausted:
            raise
        except Exception as exc:  # noqa: BLE001 - advisory
            run.warnings.append(f"shot {shot.number}: frame review failed: {exc}")
    ps.review = review
    attempts.append(ps)

    # The best attempt so far is the one that stands unless the planner
    # replaces it; "best" is the review's score, or the sweep's when there
    # is no review.
    def _key(p: cov.PlannedShot) -> float:
        return p.review.score if p.review else float(p.setup["score"]) / 10.0

    run.planned[shot.number] = max(attempts, key=_key)
    run.emit("shot", shot=run.planned[shot.number].to_dict(), attempt=len(attempts))
    out = {
        "status": "success",
        "attempt": len(attempts),
        "attempts_left": max(0, MAX_ATTEMPTS_PER_SHOT - len(attempts)),
        "image_url": _frame_url(run, ps.frame) if ps.frame else None,
        "setup": cov.row_brief(row),
        "review": _review_dict(review),
    }
    # A frame the reviewer would shoot, or the last allowed attempt, closes
    # the shot here: the planner's next turn is the next shot, not a formality.
    verdict = review.verdict if review else "keep"
    if verdict == "keep" or len(attempts) >= MAX_ATTEMPTS_PER_SHOT:
        run.accepted.add(shot.number)
        run.context.learn(shot, run.planned[shot.number].setup, run.marks)
        out["accepted"] = True
        out["note"] = ("accepted; call next_shot" if verdict == "keep"
                       else "no attempts left; the best-reviewed frame stands, accepted; call next_shot")
    else:
        out["accepted"] = False
        out["note"] = "the reviewer says adjust: place_shot another candidate honouring the suggestion, or accept_shot to keep this one"
    return out


def accept_shot(shot_number: int, tool_context: ToolContext = None) -> dict:
    """Close a shot on its current best placement and move on.

    Args:
        shot_number: The shot to accept.
    """
    run = _run(tool_context)
    shot = _shot(run, shot_number)
    if shot.number not in run.planned:
        return {"status": "error", "error": "nothing placed yet for this shot; call place_shot first"}
    run.accepted.add(shot.number)
    run.context.learn(shot, run.planned[shot.number].setup, run.marks)
    return {"status": "success", "accepted": sorted(run.accepted), "remaining": len(run.pending())}


def skip_shot(shot_number: int, reason: str, tool_context: ToolContext = None) -> dict:
    """Give up on a shot the room cannot hold, with the reason for the shot list.

    Args:
        shot_number: The shot to skip.
        reason: Why, in one line -- it is printed on the shot list.
    """
    run = _run(tool_context)
    shot = _shot(run, shot_number)
    run.skipped[shot.number] = reason.strip() or "no setup fits"
    run.warnings.append(f"shot {shot.number}: {run.skipped[shot.number]}")
    if shot.number not in run.planned:
        run.planned[shot.number] = cov.PlannedShot(shot=shot, setup=None)
        run.emit("shot", shot=run.planned[shot.number].to_dict(), attempt=0)
    return {"status": "success", "skipped": {str(k): v for k, v in run.skipped.items()},
            "remaining": len(run.pending())}


def _shot(run: _Run, number: int) -> cov.Shot:
    for s in run.shots:
        if s.number == int(number):
            return s
    raise ValueError(f"no shot {number}; have {[s.number for s in run.shots]}")


def _shot_dict(s: cov.Shot) -> dict:
    return {
        "number": s.number, "description": s.description, "size": s.size,
        "size_name": s.size_name, "subject": s.subject, "second_subject": s.second_subject,
        "lens_mm": s.lens_mm, "height": s.height, "movement": s.movement,
        "no_window_behind": s.no_window_behind, "window_in_frame": s.window_in_frame,
        "ots": s.ots, "notes": s.notes,
    }


def _review_dict(rv: cov.Review | None) -> dict | None:
    if rv is None:
        return None
    return {"score": rv.score, "verdict": rv.verdict, "notes": rv.notes, "suggestion": rv.suggestion}


def _frame_url(run: _Run, frame: str) -> str:
    job_id = run.workdir.parent.parent.name
    return f"/plan-image/{job_id}/{run.plan_id}/{frame}"


# ---------------------------------------------------------------------------
# Gemini looks at the frame
# ---------------------------------------------------------------------------

REVIEW_PROMPT = """You are a director of photography on a tech scout, looking at a
frame rendered from a measured 3D twin of a real room. The blue stick figure
is a stand-in for the subject at a standard 1.75 m stature; an amber figure,
if present, is the second actor. Blurred or dark areas are parts of the room
the scan did not see -- judge the composition, not the image quality.

The shot brief: {brief}
The setup: {focal} mm on Super 35 from {distance} m, lens height {height} m,
framing classified as {size} ({size_name}). Backlight: {backlight}.
This room's swept lenses are {lenses} and its camera heights {heights}; do
not suggest glass or heights outside that set.

Judge it as a DP would: headroom and look room, whether the framing actually
reads as the asked size, what is behind the subject (clutter, a doorway, a
bright window, a blank wall), whether the second actor is genuinely in frame,
and whether the angle serves the beat described. Score 0-10. Verdict "keep"
if you would shoot it, "adjust" if a different height, lens or framing would
be clearly better (say which in suggestion), "reject" only if it cannot serve
the brief. Notes: two sentences, concrete, no preamble."""


_LAST_REVIEW_USAGE = [(0, 0)]


def _review_bytes(image_path: Path) -> bytes:
    import io

    from PIL import Image

    im = Image.open(image_path)
    if im.width > REVIEW_IMAGE_WIDTH:
        im = im.resize((REVIEW_IMAGE_WIDTH, int(im.height * REVIEW_IMAGE_WIDTH / im.width)))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def gemini_reviewer(model: str | None = None):
    """A reviewer that shows the rendered frame to Gemini and parses its verdict."""
    from google import genai
    from google.genai import types

    model = model or AGENT_MODELS["reviewer"]
    client = genai.Client(http_options=types.HttpOptions(retry_options=_retry_options()))

    schema = {
        "type": "OBJECT",
        "properties": {
            "score": {"type": "NUMBER"},
            "verdict": {"type": "STRING", "enum": ["keep", "adjust", "reject"]},
            "notes": {"type": "STRING"},
            "suggestion": {
                "type": "OBJECT",
                "properties": {
                    "height": {"type": "STRING", "enum": ["low", "mid", "eye", "high", "none"]},
                    "lens_mm": {"type": "NUMBER"},
                    "size": {"type": "STRING"},
                    "no_window_behind": {"type": "BOOLEAN"},
                },
            },
        },
        "required": ["score", "verdict", "notes"],
    }

    def review(image_path: Path, shot: cov.Shot, row: dict) -> cov.Review:
        prompt = REVIEW_PROMPT.format(
            brief=shot.description + (f" [{shot.notes}]" if shot.notes else ""),
            focal=int(row["focal_mm"]), distance=f"{float(row['distance_m']):.2f}",
            height=f"{float(row['cam_z']):.2f}", size=row["shot_size"],
            size_name=cov.optics.SHOT_BY_KEY[str(row["shot_size"])].name,
            backlight="window behind the subject" if int(row["window_behind_subject"]) else "none flagged",
            lenses=", ".join(f"{int(f)} mm" for f in cov.sweepmod.SWEEP_PRIMES_MM),
            heights=", ".join(f"{h:.2f} m" for h in cov.sweepmod.CAMERA_HEIGHTS_M),
        )
        img = types.Part.from_bytes(data=_review_bytes(image_path), mime_type="image/png")
        resp = client.models.generate_content(
            model=model,
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=schema, temperature=0.2,
            ),
        )
        data = json.loads(resp.text or "{}")
        u = getattr(resp, "usage_metadata", None)
        if u is not None:
            _LAST_REVIEW_USAGE[0] = (int(u.prompt_token_count or 0), int(u.candidates_token_count or 0))
        sug = {k: v for k, v in (data.get("suggestion") or {}).items() if v not in ("", "none", None, 0, False)}
        return cov.Review(
            score=float(data.get("score", 0.0)), verdict=str(data.get("verdict", "keep")),
            notes=str(data.get("notes", "")).strip(), suggestion=sug, model=model,
        )

    return review


# ---------------------------------------------------------------------------
# the agents
# ---------------------------------------------------------------------------


def _breakdown_instruction(ctx) -> str:
    facts = ctx.state.get("room_facts", "")
    brief = ctx.state.get("brief", "")
    return f"""You are a first assistant director breaking down a scene for a tech scout.

THE ROOM (measured; every mark is a real position the shot table knows):
{facts}

THE SCENE OR SHOT LIST:
{brief}

Produce the coverage this scene needs, as JSON matching the schema. Rules:
- If the text is already a numbered shot list, keep its shots and numbering;
  translate each line into the fields. If it is prose, design conventional
  coverage: a master, singles on each speaking character at the sizes the
  drama wants, a two-shot or over-the-shoulders where they share the frame,
  and inserts for anything the text singles out. Six to ten shots.
- Put each character on one of the marks by name (M1, M2, ...) in `marks`
  as character/mark pairs, using the cues: someone "at the window" goes on
  the mark nearest a window; two characters who face each other go on
  different marks. Every character in any shot must appear in marks.
- size is one of: ecu, bcu, cu, mcu, ms, mls, ls, els. lens_mm only if the
  text names one (one of 16, 25, 35, 50, 75, 100), else null.
- height: eye for neutral, low when a character should loom or dominate the
  beat, high when they are meant to look small or cornered; null if the
  text gives no reason. mid for a seated eyeline.
- Standard dialogue coverage is a master, a pair of over-the-shoulders and a
  pair of singles, and the singles on the two characters should be the
  same size so they cut as reverses. Mark an over-the-shoulder with
  ots: true, subject = the face we see, second_subject = the shoulder.
- no_window_behind true unless the text wants a silhouette or the window in
  shot; windows are the room's key light and read best from the side.
- second_subject only for two-shots and OTS.
- description is one line a camera crew would read. title is a short scene name."""


def _planner_instruction(ctx) -> str:
    db = warehouse.database() if warehouse.configured() else "(local)"
    mcp = (
        f"You also have run_query (the ClickHouse MCP server) for SQL of your own over "
        f"{db}.{warehouse.TABLE} -- use it when you want something find_setups does not "
        f"offer: a count per lens, a setup farther back, the second-best mark. Always filter "
        f"location = '{ctx.state.get('location', '')}' AND visible = 1, select only the columns you need, "
        f"and LIMIT 10: results are cut to {MAX_QUERY_ROWS} rows."
        if warehouse.configured() else
        "ClickHouse is not configured in this environment; find_setups searches the local sweep."
    )
    return f"""You are the director of photography placing ONE shot from a coverage plan,
in a measured room. Work strictly through the tools; never invent a setup.

1. Call next_shot. If it returns done, reply "done" and stop. Otherwise it
   returns the shot AND its ranked candidates (the same rows find_setups
   would give); only call find_setups when you want different constraints.
2. Read the candidates. Rows come back ranked by the craft --
   for wides: background_depth_m and an axis into a corner
   (axis_wall_angle_deg near 45) over square onto a wall (near 0), plus
   backup_room_m; for tight shots: key_quality (three-quarter and side
   beat front, which is flat, and back, which is a silhouette) and
   portrait_ok (the camera far enough back that the face is not stretched).
   You choose, and you may disagree with the ranking for a stated reason.
   Be decisive: the candidates already carry every column; at most one
   run_query per shot, and only for something they do not answer.
   Continuity is enforced for you: the 180-degree line once a side is
   chosen, and next_shot tells you when a reverse should match a placed
   single's lens and distance. {mcp}
3. Call place_shot with the setup_id and one line of reasoning. It renders
   the frame and returns the DP review. A keep is accepted for you (the
   result says accepted: true) -- go straight to next_shot. If the verdict
   is adjust and attempts remain, choose another candidate honouring the
   suggestion (or find_setups with `drop`) and place_shot once more; the
   better-reviewed frame stands. If the room cannot hold the shot at all
   (no candidates even after relaxing), call skip_shot with the reason.
4. Never place the same setup_id twice for one shot. Do not narrate; the
   tools are the record. When the shot is accepted or skipped, reply with
   one line of what you chose and why, and stop -- the next shot is a new
   conversation."""


def _notes_instruction(ctx) -> str:
    pid = ctx.state.get("plan_id")
    with _RUNS_LOCK:
        run = _RUNS.get(pid or "")
    summary = _placement_summary(run) if run is not None else ctx.state.get("placement_summary", "")
    if not summary.strip():
        summary = "(nothing was placed)"
    return f"""You are the DP writing the note at the top of the shot list, for the director
and the 1st AD, after planning coverage in a measured room. Here is what was
placed, in order, with the frame reviews:

{summary}

Write four to six sentences, plain and specific: what the room gives this
scene, which shots were compromised and how (a shorter lens than asked, a
neighbouring framing, a backlit window), and the one thing to decide before
the shoot day. No headings, no bullet points. Only facts that appear above:
no objects, lenses, distances or people that are not listed. If the list is
thin, write less."""


# The placer's context is resent on every turn, so a fat tool result is paid
# for again on each later call. run_query answers are cut to this many rows
# and characters; the planner can always query again, narrower.
MAX_QUERY_ROWS = 8
MAX_TOOL_CHARS = 3500


def _trim_tool_result(tool, args, tool_context, tool_response):
    """Shrink run_query payloads before they enter the conversation."""
    try:
        name = getattr(tool, "name", "")
        if name != "run_query" or not isinstance(tool_response, dict):
            return None
        content = tool_response.get("content")
        if not isinstance(content, list):
            return None
        out = []
        for part in content:
            text = part.get("text") if isinstance(part, dict) else None
            if not isinstance(text, str):
                out.append(part)
                continue
            try:
                data = json.loads(text)
            except ValueError:
                out.append({**part, "text": text[:MAX_TOOL_CHARS]})
                continue
            rows = data.get("rows") if isinstance(data, dict) else None
            if isinstance(rows, list) and len(rows) > MAX_QUERY_ROWS:
                data = {**data, "rows": rows[:MAX_QUERY_ROWS],
                        "note": f"{len(rows)} rows matched; showing the first {MAX_QUERY_ROWS} -- add LIMIT or narrow the WHERE"}
            text = json.dumps(data, separators=(",", ":"))
            out.append({**part, "text": text[:MAX_TOOL_CHARS]})
        return {**tool_response, "content": out}
    except Exception:  # noqa: BLE001 - trimming is an economy, never a failure
        return None


# A placer that looks at the table a dozen times per shot is not placing
# better, it is paying more. After this many tool calls in one shot's
# conversation the tools stop answering and tell it to decide.
MAX_TOOL_CALLS_PER_SHOT = int(os.environ.get("LOCAISH_MAX_TOOL_CALLS_PER_SHOT", "8"))


def _tool_budget(tool, args, tool_context):
    """Per-conversation tool budget; closing tools are always allowed."""
    name = getattr(tool, "name", "")
    if name in ("accept_shot", "skip_shot"):
        return None
    used = int(tool_context.state.get("tool_calls", 0)) + 1
    tool_context.state["tool_calls"] = used
    if used > MAX_TOOL_CALLS_PER_SHOT:
        return {"status": "budget", "error": (
            f"this shot's tool budget ({MAX_TOOL_CALLS_PER_SHOT} calls) is spent: "
            "call accept_shot to keep the best attempt, or skip_shot with a reason")}
    return None


def _spend_guard(callback_context, llm_request):
    """Counts every model call a plan makes; the counter lives on the run."""
    pid = callback_context.state.get("plan_id")
    with _RUNS_LOCK:
        run = _RUNS.get(pid or "")
    if run is not None:
        run.spend()
    return None


def build_workflow(*, models: dict[str, Any] | None = None) -> dict[str, Any]:
    """The three agents: breakdown, placer, notes.

    `models` may override the model per agent ("breakdown", "planner",
    "notes") with a model name or a BaseLlm instance, which is how the tests
    run the whole workflow against a scripted model.
    """
    from google.adk.agents import LlmAgent

    models = models or {}

    breakdown = LlmAgent(
        name="breakdown",
        before_model_callback=_spend_guard,
        model=_model(models.get("breakdown", AGENT_MODELS["breakdown"])),
        description="Turns a scene or shot list into structured coverage with characters on marks.",
        instruction=_breakdown_instruction,
        output_schema=_breakdown_schema(),
        output_key="breakdown",
        include_contents="none",
    )

    tools: list[Any] = [next_shot, find_setups, place_shot, accept_shot, skip_shot]
    if warehouse.configured():
        from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
        from mcp import StdioServerParameters

        tools.append(
            McpToolset(
                connection_params=StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command=sys.executable,
                        args=["-m", "mcp_clickhouse.main"],
                        env={**os.environ, **warehouse.connection_env()},
                    ),
                    timeout=20.0,
                ),
                tool_filter=["run_query"],
            )
        )

    placer = LlmAgent(
        name="placer",
        before_model_callback=_spend_guard,
        before_tool_callback=_tool_budget,
        after_tool_callback=_trim_tool_result,
        model=_model(models.get("planner", AGENT_MODELS["planner"])),
        description="Places one shot: searches the shot table, renders, reads the review.",
        instruction=_planner_instruction,
        tools=tools,
    )

    notes = LlmAgent(
        name="dp_notes",
        before_model_callback=_spend_guard,
        model=_model(models.get("notes", AGENT_MODELS["notes"])),
        description="Writes the DP's note for the shot list.",
        instruction=_notes_instruction,
        output_key="dp_notes",
        include_contents="none",
    )
    return {"breakdown": breakdown, "placer": placer, "notes": notes}


def _breakdown_schema():
    from pydantic import BaseModel, Field

    class ShotSpec(BaseModel):
        number: int
        description: str
        size: str
        subject: str
        second_subject: str | None = None
        lens_mm: float | None = None
        height: str | None = None
        movement: str = "static"
        no_window_behind: bool = False
        window_in_frame: bool | None = None
        notes: str = ""

    class MarkAssignment(BaseModel):
        character: str
        mark: str = Field(description="a mark name from the room, e.g. M2")

    class Breakdown(BaseModel):
        title: str
        marks: list[MarkAssignment]
        shots: list[ShotSpec]

    return Breakdown


# ---------------------------------------------------------------------------
# running it
# ---------------------------------------------------------------------------


class _PlannerService:
    """A persistent event loop for the planner runner, one per process.

    The reason is the same as `AgentService`: ADK pools MCP sessions per
    loop, and a loop per plan would spawn a ClickHouse MCP subprocess for
    every request.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._error: str | None = None
        threading.Thread(target=self._serve, daemon=True, name="locaish-coverage").start()

    def _serve(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            self._ready.set()
            return
        self._ready.set()
        self._loop.run_forever()

    def submit(self, coro, timeout_s: float):
        self._ready.wait(timeout=10.0)
        if self._loop is None or self._error:
            raise AgentUnavailable(self._error or "planner loop failed to start")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=timeout_s)
        except TimeoutError as exc:
            fut.cancel()
            raise AgentUnavailable(f"the planner did not finish within {timeout_s:.0f}s") from exc


_SERVICE: _PlannerService | None = None
_SERVICE_LOCK = threading.Lock()


def _service() -> _PlannerService:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = _PlannerService()
        return _SERVICE


def plan_coverage(
    twin: Twin,
    *,
    workdir: str | Path,
    source: cov.SetupSource,
    brief: str,
    title: str = "",
    on_event: Callable[[dict], None] | None = None,
    models: dict[str, Any] | None = None,
    reviewer: Callable | None = None,
    timeout_s: float = 900.0,
) -> cov.CoveragePlan:
    """Plan a scene's coverage with the agent workflow. Returns the saved plan.

    Raises AgentUnavailable when Gemini is not configured. `models` and
    `reviewer` exist so the workflow can be driven by a scripted model in
    tests; in the studio both default to Gemini.
    """
    if models is None and not agent_configured():
        raise AgentUnavailable(
            "Gemini is not configured. Set GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION and "
            "GOOGLE_GENAI_USE_VERTEXAI=TRUE with application-default credentials, or "
            "GOOGLE_API_KEY for AI Studio."
        )
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    plan_id = workdir.name
    if reviewer is None and models is None:
        reviewer = gemini_reviewer()
    run = _Run(
        plan_id=plan_id, twin=twin, workdir=workdir, source=source, brief=brief,
        title=title, reviewer=reviewer, on_event=on_event,
    )
    with _RUNS_LOCK:
        _RUNS[plan_id] = run
    try:
        return _service().submit(_plan(run, models), timeout_s)
    finally:
        with _RUNS_LOCK:
            _RUNS.pop(plan_id, None)


async def _plan(run: _Run, models: dict[str, Any] | None) -> cov.CoveragePlan:
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    t0 = time.perf_counter()
    facts = room_facts(run.twin, run.source)
    agents = build_workflow(models=models)
    trace: list[dict] = []
    base_state = {"plan_id": run.plan_id, "location": run.location, "room_facts": facts,
                  "brief": run.brief, "placement_summary": ""}

    async def invoke(agent, session_id: str, text: str, state: dict) -> str:
        """One agent, one fresh session, one message; returns its final text."""
        runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
        await runner.session_service.create_session(
            app_name=APP_NAME, user_id="studio", session_id=session_id, state=state)
        message = types.Content(role="user", parts=[types.Part(text=text)])
        final = ""
        async for event in runner.run_async(user_id="studio", session_id=session_id, new_message=message):
            author = getattr(event, "author", "") or ""
            usage = getattr(event, "usage_metadata", None)
            if usage is not None:
                run.tokens_in += int(getattr(usage, "prompt_token_count", 0) or 0)
                run.tokens_out += int(getattr(usage, "candidates_token_count", 0) or 0)
            for call in event.get_function_calls():
                args = dict(call.args or {})
                trace.append({"kind": "call", "agent": author, "tool": call.name, "args": args})
                run.emit("call", agent=author, tool=call.name, args=args)
            for resp in event.get_function_responses():
                summary = _summarise(resp.response)
                trace.append({"kind": "result", "agent": author, "tool": resp.name, "summary": summary})
                run.emit("result", agent=author, tool=resp.name, summary=summary)
            if event.content and event.content.parts:
                t = "".join(p.text or "" for p in event.content.parts if getattr(p, "text", None)).strip()
                if t:
                    final = t
        try:
            await runner.close()
        except Exception:  # noqa: BLE001
            pass
        return final

    degraded: str | None = None
    breakdown_seen = False
    try:
        run.emit("stage", text="Gemini is breaking down the scene")
        text = await invoke(agents["breakdown"], f"{run.plan_id}-breakdown",
                            f"Plan coverage for this scene:\n{run.brief}", dict(base_state))
        try:
            _apply_breakdown(run, json.loads(text))
        except Exception as exc:  # noqa: BLE001
            raise AgentUnavailable(f"the breakdown was not usable: {exc}") from exc
        breakdown_seen = True
        run.emit("agent", agent="breakdown",
                 text=f"{len(run.shots)} shots, {len(run.marks)} characters on marks")

        # One conversation per shot. The tools carry the plan's state, so a
        # fresh context loses nothing but the previous shots' tool traffic.
        for shot in list(run.shots):
            if shot.number in run.accepted or shot.number in run.skipped:
                continue
            run.emit("stage", text=f"placing shot {shot.number}: {shot.size_name.lower()} of {shot.subject}")
            reply = await invoke(
                agents["placer"], f"{run.plan_id}-shot{shot.number}",
                f"Place shot {shot.number} ({shot.size_name.lower()} of {shot.subject}): {shot.description}",
                dict(base_state),
            )
            if reply:
                run.emit("agent", agent="placer", text=reply[:400])
            if shot.number not in run.accepted and shot.number not in run.skipped:
                # The agent stopped without closing the shot: keep the best
                # attempt if there is one, else it is the planner's.
                if shot.number in run.planned and run.planned[shot.number].setup:
                    run.accepted.add(shot.number)
                    run.context.learn(shot, run.planned[shot.number].setup, run.marks)
                else:
                    _place_without_model(run, shot)

        run.emit("stage", text="writing the DP's note")
        notes = await invoke(agents["notes"], f"{run.plan_id}-notes",
                             "Write the note for this shot list.", dict(base_state))
        run.dp_notes = notes.strip()
        if run.dp_notes.lower().startswith("dp notes:"):
            run.dp_notes = run.dp_notes[len("dp notes:"):].strip()
        if run.dp_notes:
            run.emit("agent", agent="dp_notes", text=run.dp_notes)
    except AgentUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - quota, network, a model outage
        if not breakdown_seen or not run.shots:
            raise AgentUnavailable(f"Gemini could not break down the scene: {_short(exc)}") from exc
        # The scene is broken down and some shots are placed; the rest are
        # placed by the planner without a reviewer, and the plan says so.
        # A demo that dies at shot four because of a rate limit is worse
        # than one that finishes and tells the truth about how.
        degraded = _short(exc)
        run.emit("note", text=f"Gemini stopped answering ({degraded}); placing the remaining shots without review")
        for shot in run.pending():
            _place_without_model(run, shot)

    # Assemble the plan from what was placed, in shot order.
    shots = [run.planned.get(s.number, cov.PlannedShot(shot=s, setup=None)) for s in run.shots]
    plan = cov.CoveragePlan(
        plan_id=run.plan_id, location=run.location, title=run.title or (run.breakdown or {}).get("title") or "Untitled scene",
        brief=run.brief, shots=shots, marks=dict(run.marks), planner="gemini",
        warnings=list(run.warnings), trace=trace,
    )
    plan.trace.append({"kind": "usage", "model_calls": run.model_calls,
                       "tokens_in": run.tokens_in, "tokens_out": run.tokens_out})
    run.emit("note", text=f"Gemini usage: {run.model_calls} calls, {run.tokens_in:,} tokens in, {run.tokens_out:,} out")
    if degraded:
        plan.warnings.insert(0, f"Gemini stopped answering part-way ({degraded}); the shots after that were "
                                "placed by the planner from the table without a frame review")
    if run.dp_notes:
        plan.warnings.insert(0, "DP notes: " + run.dp_notes)
    for ps in plan.shots:
        if ps.setup is None and ps.shot.number not in run.skipped:
            plan.warnings.append(f"shot {ps.shot.number}: the planner never placed it")
    run.emit("stage", text="drawing the camera plan")
    plan.floor_plan_svg = cov.floor_plan_svg(run.twin, plan)
    plan.save(run.workdir)
    run.emit("stage", text=f"planned {plan.planned}/{len(plan.shots)} shots in {time.perf_counter() - t0:.0f}s")
    return plan


def _place_without_model(run: _Run, shot: cov.Shot) -> None:
    """The deterministic placement, for a shot the agent did not close."""
    try:
        rows, count, sql, relaxed = cov.find_setup(
            run.source, shot, run.marks, location=run.location, context=run.context)
    except ValueError as exc:
        run.warnings.append(f"shot {shot.number}: {exc}")
        run.skipped[shot.number] = str(exc)
        return
    run.emit("candidates", shot=shot.number, rows=[cov.row_brief(r) for r in rows], matched=count, sql=sql)
    ps = cov.PlannedShot(shot=shot, setup=rows[0] if rows else None, candidates=count, sql=sql, relaxed=relaxed)
    if shot.second_subject and shot.second_subject in run.marks:
        ps.second_mark = run.marks[shot.second_subject]
    if ps.setup:
        ps.why = cov.explain(shot, ps.setup, run.marks, run.context)
        run.context.learn(shot, ps.setup, run.marks)
        try:
            ps.frame = cov.render_frame(run.twin, ps, run.frames_dir)
        except Exception as exc:  # noqa: BLE001
            run.warnings.append(f"shot {shot.number}: the frame could not be rendered: {exc}")
    else:
        run.skipped[shot.number] = "no setup fits"
    run.planned[shot.number] = ps
    run.accepted.add(shot.number)
    run.emit("shot", shot=ps.to_dict(), attempt=1)


async def _session_state(runner, session_id: str):
    try:
        sess = await runner.session_service.get_session(app_name=APP_NAME, user_id="studio", session_id=session_id)
        return sess.state if sess is not None else None
    except Exception:  # noqa: BLE001
        return None


def _apply_breakdown(run: _Run, data: dict) -> None:
    run.breakdown = data
    available = {m["name"]: (m["x"], m["y"]) for m in cov.describe_marks(run.twin, run.source.marks(run.location))}
    marks: dict[str, tuple[float, float]] = {}
    raw = data.get("marks") or {}
    # A list of {character, mark} from the schema, or a plain mapping.
    pairs = raw.items() if isinstance(raw, dict) else (
        (m.get("character"), m.get("mark")) for m in raw if isinstance(m, dict)
    )
    for name, mark in pairs:
        key = str(mark or "").strip().upper()
        if name and key in available:
            marks[str(name).strip()] = available[key]
    shots = [cov.Shot.from_dict(s) for s in data.get("shots") or []]
    if not shots:
        raise ValueError("no shots in the breakdown")
    for i, s in enumerate(shots):
        if not s.number:
            s.number = i + 1
    names: list[str] = []
    for s in shots:
        for n in (s.subject, s.second_subject):
            if n and n not in names:
                names.append(n)
    missing = [n for n in names if n not in marks]
    if missing:
        marks.update(cov.auto_marks(run.source, run.location, missing))
        run.warnings.append(f"marks assigned automatically for: {', '.join(missing)}")
    run.shots = shots
    run.marks = marks
    if not run.title:
        run.title = str(data.get("title") or "")


def _placement_summary(run: _Run) -> str:
    lines = []
    for s in run.shots:
        ps = run.planned.get(s.number)
        if ps is None:
            continue
        if ps.setup is None:
            lines.append(f"shot {s.number} ({s.size_name.lower()} of {s.subject}): NOT PLACED -- {run.skipped.get(s.number, '')}")
            continue
        st = ps.setup
        rv = ps.review
        lines.append(
            f"shot {s.number} ({s.size_name.lower()} of {s.subject}): {int(st['focal_mm'])} mm from "
            f"{float(st['distance_m']):.2f} m at {float(st['cam_z']):.2f} m, framing {st['shot_size']}"
            + (", window behind subject" if int(st["window_behind_subject"]) else "")
            + (f"; review {rv.score:.0f}/10 {rv.verdict}: {rv.notes}" if rv else "")
            + (f"; attempts {ps.attempts}" if ps.attempts > 1 else "")
        )
    return "\n".join(lines)


def _short(exc: BaseException) -> str:
    if isinstance(exc, BudgetExhausted):
        return str(exc)
    text = str(exc)
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return "rate limit / quota exhausted (429)"
    if "UNAVAILABLE" in text or "503" in text:
        return "model temporarily unavailable (503)"
    return (text.splitlines() or ["error"])[0][:120]


def _summarise(response: Any) -> str:
    if not isinstance(response, dict):
        return str(response)[:200]
    if "image_url" in response:
        rv = response.get("review") or {}
        return f"{response['image_url']} review {rv.get('score', '-')}/10 {rv.get('verdict', '')}".strip()
    if "rows" in response and isinstance(response["rows"], list):
        return f"{len(response['rows'])} rows of {response.get('matched', '?')} matched"
    if response.get("done") is True:
        return "all shots placed"
    if "shot" in response and isinstance(response["shot"], dict):
        s = response["shot"]
        return f"shot {s.get('number')}: {s.get('size')} of {s.get('subject')}"
    text = json.dumps(response, default=str)
    return text[:200] + ("…" if len(text) > 200 else "")
