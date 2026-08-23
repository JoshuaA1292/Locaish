"""The agent's wiring: tools over the twin, SQL over the sweep, Gemini on top.

Three design decisions carry this module.

**The model reasons; the tools measure.** Every tool here returns measurements
from the twin or rows from ClickHouse, and the instruction tells the model that
a number it did not get from a tool is a number it may not state. That is the
difference between an agent and an autocomplete: the value of the answer is
that it is *checkable*, and it is checkable because each claim names the tool
result it came from.

**One event loop, for the life of the process.** ADK pools MCP sessions per
event loop, so a loop per request would spawn a fresh `mcp-clickhouse`
subprocess every time someone asks a question. A single background loop owns
the runner, the sessions and the MCP subprocess; request threads submit
coroutines to it and wait.

**ClickHouse is read through MCP, written around it.** The sweep is bulk-loaded
by `locaish.warehouse` over clickhouse-connect; the agent's own access is the
official `mcp-clickhouse` server, read-only as it ships. The agent could not
corrupt the table if it tried, and the judges can see the partner service on
the wire rather than in a README.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import warehouse
from ..types import Twin

APP_NAME = "locaish"
DEFAULT_MODEL = os.environ.get("LOCAISH_GEMINI_MODEL", "gemini-3.5-flash")


class AgentUnavailable(RuntimeError):
    """The agent cannot run: missing credentials, missing config, or a dead loop."""


# ---------------------------------------------------------------------------
# the locations the agent may scout
# ---------------------------------------------------------------------------


@dataclass
class _Location:
    twin: Twin
    workdir: Path
    maps: Any = None
    occupancy: Any = None
    shots_rendered: int = 0


_LOCATIONS: dict[str, _Location] = {}
_LOCATIONS_LOCK = threading.Lock()


def register_location(name: str, twin: Twin, workdir: Path) -> None:
    """Make a finished twin scoutable. Called by the studio after ingest."""
    with _LOCATIONS_LOCK:
        _LOCATIONS[name] = _Location(twin=twin, workdir=Path(workdir))


def _location(tool_context) -> tuple[str, _Location]:
    name = None
    if tool_context is not None:
        name = tool_context.state.get("location")
    if not name or name not in _LOCATIONS:
        raise ValueError(
            f"no active location named {name!r}; scan a room first, or ask about "
            f"one of: {sorted(_LOCATIONS)}"
        )
    return name, _LOCATIONS[name]


def _maps_and_grid(loc: _Location):
    from ..film import space as spacemod

    if loc.maps is None:
        loc.maps = spacemod.floor_maps(loc.twin)
    if loc.occupancy is None:
        loc.occupancy = spacemod.occupancy(loc.twin)
    return loc.maps, loc.occupancy


# ---------------------------------------------------------------------------
# function tools
# ---------------------------------------------------------------------------


def scout_report(tool_context=None) -> dict:
    """Full technical scout report for the active location, by department.

    Covers trust (QA verdict, scale confidence), space (dimensions, standable
    area), camera (longest sightlines, widest framings available), grip (which
    dollies/sliders/jibs physically fit and where), and sound (RT60 estimate,
    the acoustic character of the room). Call this before answering general
    questions about the location.
    """
    from ..film import report as reportmod

    _, loc = _location(tool_context)
    built = reportmod.build(loc.twin)
    return {"status": "success", "report": built.to_dict()}


def measure(
    x1: float, y1: float, z1: float, x2: float, y2: float, z2: float,
    tool_context=None,
) -> dict:
    """Measure the straight-line metric distance between two points in the room.

    Coordinates are in the twin's frame: metres, floor at z=0, origin at the
    middle of the floor. Also reports whether the segment is physically clear
    (line of sight) or passes through geometry.

    Args:
        x1: X of the first point in metres.
        y1: Y of the first point in metres.
        z1: Z (height above floor) of the first point in metres.
        x2: X of the second point in metres.
        y2: Y of the second point in metres.
        z2: Z of the second point in metres.
    """
    from ..film import space as spacemod

    _, loc = _location(tool_context)
    _, occ = _maps_and_grid(loc)
    a = np.array([x1, y1, z1], dtype=np.float64)
    b = np.array([x2, y2, z2], dtype=np.float64)
    grid, origin, cell = occ
    return {
        "status": "success",
        "distance_m": round(float(np.linalg.norm(b - a)), 3),
        "clear_line_of_sight": bool(spacemod.visible(grid, origin, cell, a, b)),
    }


def render_frame(
    cam_x: float, cam_y: float, cam_z: float,
    subj_x: float, subj_y: float,
    focal_mm: float,
    tool_context=None,
) -> dict:
    """Render the actual frame a camera setup would capture, from the twin's points.

    Use this to show the user what a setup from the shot table looks like: pass
    the row's cam_x/cam_y/cam_z, subj_x/subj_y and focal_mm. A stand-in figure
    of standard 1.75 m stature is drawn at the subject mark. Returns an
    image_url; include it in your answer as markdown: ![shot](image_url).

    Args:
        cam_x: Camera X in metres (twin frame).
        cam_y: Camera Y in metres.
        cam_z: Camera height above the floor in metres.
        subj_x: Subject mark X in metres.
        subj_y: Subject mark Y in metres.
        focal_mm: Lens focal length in millimetres.
    """
    from ..film.render import render_shot

    name, loc = _location(tool_context)
    loc.shots_rendered += 1
    fname = f"shot_{loc.shots_rendered:03d}_{int(focal_mm)}mm.png"
    out = loc.workdir / "shots" / fname
    render_shot(
        loc.twin,
        (cam_x, cam_y, float(loc.twin.structure.floor_z) + cam_z),
        (subj_x, subj_y),
        focal_mm,
        out=out,
    )
    return {
        "status": "success",
        "image_url": f"/shot-image/{tool_context.state.get('job_id', name)}/{fname}",
        "note": "the frame is rendered from the twin's own measured points; "
        "dark regions are parts of the room the capture never saw",
    }


def check_dolly_move(
    start_x: float, start_y: float, end_x: float, end_y: float,
    height_m: float, focal_mm: float,
    subj_x: float, subj_y: float,
    tool_context=None,
) -> dict:
    """Simulate a straight dolly move and report whether it physically works.

    Checks every beat along the track for gear fit, headroom, floor levelness
    (a dolly needs the floor flat to ~20 mm), and sightline to the subject, and
    reports how the framing evolves over the move. Use for "can we dolly from
    A to B" and "does the push-in hold focus" questions.

    Args:
        start_x: Track start X in metres.
        start_y: Track start Y in metres.
        end_x: Track end X in metres.
        end_y: Track end Y in metres.
        height_m: Lens height above the floor in metres.
        focal_mm: Lens focal length in millimetres.
        subj_x: Subject mark X in metres.
        subj_y: Subject mark Y in metres.
    """
    from ..film import equipment, moves

    _, loc = _location(tool_context)
    maps, occ = _maps_and_grid(loc)
    floor_z = float(loc.twin.structure.floor_z)
    path = moves.straight(
        (start_x, start_y, floor_z + height_m), (end_x, end_y, floor_z + height_m)
    )
    subject = np.array([[subj_x, subj_y, floor_z]])
    report = moves.simulate(
        maps, occ, path,
        name="dolly",
        subject_path=subject,
        focal_mm=focal_mm,
        gear=equipment.get("dolly-doorway"),
        on_track=True,
    )
    return {"status": "success", "move": report.summary()}


def sun_schedule(date: str = "", tool_context=None) -> dict:
    """The day's natural light at this location, computed from solar ephemeris.

    Sunrise, sunset, golden hours, the sun's peak elevation, and -- window by
    window -- when direct sun actually comes through each pane, with the
    compass bearing it faces. Times are local solar time (solar noon = 12:00).
    Needs the twin to be georeferenced; the error says how to add that if not.

    Args:
        date: ISO date like 2026-09-09. Empty means today.
    """
    from ..film import daylight

    _, loc = _location(tool_context)
    schedule = daylight.sun_schedule(loc.twin, date or None)
    return {"status": "success", "schedule": schedule}


def list_locations(tool_context=None) -> dict:
    """List every scanned location available in this session, with QA verdicts."""
    out = {}
    with _LOCATIONS_LOCK:
        for name, loc in _LOCATIONS.items():
            qa = loc.twin.qa
            out[name] = {
                "qa_verdict": qa.verdict,
                "points": int(len(loc.twin.points)),
            }
    return {"status": "success", "locations": out}


# ---------------------------------------------------------------------------
# the agent
# ---------------------------------------------------------------------------


INSTRUCTION = f"""You are Locaish's location scout: a virtual tech scout who has
already surveyed the room, working for a film crew who has not.

The active location is named in a bracketed line at the top of each user
message, e.g. [active location: kitchen]. All tools operate on that location.

## Ground rules
- Every number you state must come from a tool result in this conversation.
  If you have not measured it, say so and measure it.
- The twin carries its own honesty: relay QA verdicts, scale confidence and
  surveyed-fraction caveats when they matter to the answer. A twin from video
  has a measured shape and an *inferred* size.
- Answer like a scout talking to a department head: concrete, in metres and
  millimetres, brief. Recommend, then show the evidence.

## The shot table (ClickHouse, via run_query)
Every physically-possible camera setup in the room has been swept and scored
into `{warehouse.database()}.{warehouse.TABLE}`. One row = one setup: a camera
position and height, a subject mark, and a lens. Columns:

- location (String) -- ALWAYS filter `location = '<active location>'`
- cam_x, cam_y (m, twin frame), cam_z (m above floor)
- subj_x, subj_y (m) -- the subject mark this row frames
- distance_m, yaw_deg, pitch_deg
- focal_mm (one of 16, 25, 35, 50, 75, 100), sensor (super35)
- shot_size: ecu | bcu | cu | mcu | ms | mls | ls | els
- size_fit (0-1, how cleanly the framing lands on that named size)
- framed_height_m, subject_fill (fraction of frame height a 1.75 m subject fills)
- fov_h_deg, dof_near_m, dof_far_m, dof_infinite (at T2.8)
- visible (0/1 -- sightline to the subject is clear; filter visible = 1)
- surveyed (0/1 -- camera stands on ground the capture actually saw)
- clearance_m (room around the camera), headroom_m
- window_in_frame, window_behind_subject (0/1 -- backlight risk)
- score (0-100 tie-breaker; prefer ORDER BY score DESC, then your own criteria)

Typical query: SELECT cam_x, cam_y, cam_z, subj_x, subj_y, focal_mm, distance_m,
score FROM {warehouse.database()}.{warehouse.TABLE} WHERE location = '...' AND
shot_size = 'cu' AND visible = 1 AND window_behind_subject = 0 ORDER BY score
DESC LIMIT 5. Always LIMIT.

## Natural light
sun_schedule gives sunrise/sunset, golden hours and per-window direct-sun
intervals for any date, from real solar ephemeris. The shot table's
window_behind_subject flag says *geometry*; sun_schedule says *when* that
glass is actually hot. Combine them for briefs that mention light or time of
day. It requires a georeferenced twin and will say so if there is none.

## Workflow for a shot brief
1. Translate the brief into filters (shot size, lens preference, light, moves).
2. Query the table; if empty, relax one constraint and say which and why.
3. Sanity-check the winner with tools (measure, check_dolly_move) if the brief
   involves movement or tight geometry.
4. render_frame the winning setup and include the image in your answer.
5. State the physical reasoning: why this position, this height, this lens.
"""


def _build_agent():
    from google.adk.agents import LlmAgent

    tools: list[Any] = [
        scout_report, measure, render_frame, check_dolly_move, sun_schedule,
        list_locations,
    ]
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
                tool_filter=["run_query", "list_tables"],
            )
        )
        instruction = INSTRUCTION
    else:
        instruction = INSTRUCTION + (
            "\n## Degraded mode\nClickHouse is not configured in this "
            "environment, so the shot table is offline. Say so when a brief "
            "needs it, and answer what the other tools can measure."
        )

    return LlmAgent(
        model=DEFAULT_MODEL,
        name="locaish_scout",
        description="Virtual location scout over a measured digital twin.",
        instruction=instruction,
        tools=tools,
    )


@dataclass
class AgentTurn:
    """One answered question: the reply, and the tool calls that earned it."""

    reply: str
    trace: list[dict] = field(default_factory=list)
    seconds: float = 0.0


class AgentService:
    """The runner on its persistent loop. One per process; thread-safe ask()."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner = None
        self._ready = threading.Event()
        self._error: str | None = None
        self._sessions: set[str] = set()
        self._lock = threading.Lock()
        threading.Thread(target=self._serve_loop, daemon=True, name="locaish-agent").start()

    def _serve_loop(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            self._ready.set()
            return
        self._ready.set()
        self._loop.run_forever()

    def _ensure_runner(self):
        if self._runner is None:
            if not (
                os.environ.get("GOOGLE_API_KEY")
                or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
                or os.environ.get("GOOGLE_GENAI_USE_ENTERPRISE")
                or os.environ.get("GOOGLE_CLOUD_PROJECT")
            ):
                raise AgentUnavailable(
                    "Gemini is not configured. Either set GOOGLE_CLOUD_PROJECT, "
                    "GOOGLE_CLOUD_LOCATION and GOOGLE_GENAI_USE_VERTEXAI=TRUE "
                    "with gcloud application-default credentials (the hackathon "
                    "configuration), or set GOOGLE_API_KEY for AI Studio."
                )
            from google.adk.runners import InMemoryRunner

            self._runner = InMemoryRunner(agent=_build_agent(), app_name=APP_NAME)
        return self._runner

    def ask(self, session_id: str, location: str, text: str, *, timeout_s: float = 240.0) -> AgentTurn:
        """Answer one user message inside a persistent per-job session."""
        self._ready.wait(timeout=10.0)
        if self._loop is None or self._error:
            raise AgentUnavailable(self._error or "agent loop failed to start")
        future = asyncio.run_coroutine_threadsafe(
            self._ask(session_id, location, text), self._loop
        )
        try:
            return future.result(timeout=timeout_s)
        except AgentUnavailable:
            raise
        except TimeoutError as exc:
            future.cancel()
            raise AgentUnavailable(f"the agent did not answer within {timeout_s:.0f}s") from exc

    async def _ask(self, session_id: str, location: str, text: str) -> AgentTurn:
        from google.genai import types

        runner = self._ensure_runner()
        t0 = time.perf_counter()

        with self._lock:
            fresh = session_id not in self._sessions
        if fresh:
            try:
                await runner.session_service.create_session(
                    app_name=APP_NAME,
                    user_id="studio",
                    session_id=session_id,
                    state={"location": location, "job_id": session_id},
                )
            except Exception:  # noqa: BLE001 - already exists is fine
                pass
            with self._lock:
                self._sessions.add(session_id)

        message = types.Content(
            role="user",
            parts=[types.Part(text=f"[active location: {location}]\n{text}")],
        )
        reply_parts: list[str] = []
        trace: list[dict] = []
        async for event in runner.run_async(
            user_id="studio", session_id=session_id, new_message=message
        ):
            for call in event.get_function_calls():
                trace.append({"kind": "call", "tool": call.name, "args": dict(call.args or {})})
            for resp in event.get_function_responses():
                trace.append({"kind": "result", "tool": resp.name,
                              "summary": _summarise_result(resp.response)})
            if event.is_final_response() and event.content and event.content.parts:
                reply_parts.extend(p.text or "" for p in event.content.parts)

        return AgentTurn(
            reply="".join(reply_parts).strip(),
            trace=trace,
            seconds=round(time.perf_counter() - t0, 1),
        )


def _summarise_result(response: Any) -> str:
    """A one-line account of a tool result, for the UI's activity feed."""
    if not isinstance(response, dict):
        return str(response)[:200]
    if "image_url" in response:
        return response["image_url"]
    if "rows" in response and isinstance(response["rows"], list):
        return f"{len(response['rows'])} rows"
    text = str(response)
    return text[:200] + ("…" if len(text) > 200 else "")
