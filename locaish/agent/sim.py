"""The camera operator: Muse Wonder drives the simulation, Muse Spark watches the frames.

A director says what the camera should do -- "push in on Maya over four
seconds on the 50, low, as she sits" -- and the operator builds the move in
the measured room. The operator is a model with tools, and every tool is a
thing the engine (`film.sim`) does: name a rig, set the keys, move an actor,
evaluate the whole move against the room, look at a frame. It cannot place
the camera anywhere the engine will not check, and the engine's verdict on
each frame (gear fit, headroom, sightline, speed, subject in frame) comes
back as text the operator has to answer for.

Looking is a second model's job. `look(t)` renders the frame the engine
says the camera sees at time t -- from the twin's gaussian field, with the
cast drawn in -- and shows it to Muse Spark, which reads it as a DP would:
headroom, what is behind the subject, whether the move reads. Muse Spark's
words come back to Muse Wonder as the tool's result. One model decides, one
model looks, the geometry answers; nothing in the timeline came from a
model's sense of where a wall is.

The loop is bounded (a move is a dozen model calls, not a conversation),
every call is traced for the page, and the result is a timeline the viewer
can play plus the evaluation it was judged on. Without a Meta key the
engine still evaluates a typed timeline; this module simply does not run.
"""

from __future__ import annotations

import copy
import io
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..meta.client import ChatResult, model_for, tool_schema, user_message
from ..types import Twin

MAX_MODEL_CALLS = 24
MAX_LOOKS = 5
MAX_WATCHES = 2
FILMSTRIP_FRAMES = 8
REVIEW_WIDTH_PX = 640

_SYSTEM = """You are a camera operator and DP building one camera move inside a measured 3D twin of a real room, for a director who told you what the shot is.
You work only through the tools. The room, the rigs and the physics are the engine's; your job is taste: where the camera starts and ends, the lens, the height, the rig that gives that move, and how the actors sit in frame.
Method: read the room (describe_room), choose a rig (set_rig), write the camera keys (set_camera_keys) and the actors' marks (set_actor_keys) in twin metres, then evaluate. Read every problem the evaluation reports and fix the move until no frame fails; prefer fewer, better keys; then watch the whole move (watch) and look at one or two frames (look), and act on what the DP says before you finish. A push-in should end on the size the director named, never with the subject overflowing the frame; keep headroom and look room; a non-subject actor must not stand between the lens and the subject. Keep the move to the duration the director asked for. When the move holds, call finish with one paragraph for the director: what the move is, what it costs, what to watch on the day.
Rules: keys use absolute twin coordinates as the engine reports them (z is height above the floor plus floor_z, the engine tells you floor_z); subjects are actors by name; never claim a frame holds that the engine said fails; never invent a rig the engine does not list."""

_WATCH_PROMPT = """You are a DP watching a planned camera move in a real room, shown as a filmstrip of {n} frames left to right, top to bottom, each with its time, lens and shot size written under it. The room is rendered from its scan; the actors are coloured stand-in figures at their real heights.
The move the director asked for: {move}. Rig: {rig}. Duration {duration:.1f} s. Engine problems: {problems}.
Judge the MOVE, not one frame: does the framing develop the way the director wants, is the speed right for the duration, does the subject stay well placed through it, what ends up behind them, where does it get too tight or too loose. Answer JSON: {{"score": 1-10, "verdict": "keep|adjust|reject", "notes": "<three sentences on the move; name a concrete change to the keys, the lens or the duration if adjust>"}}"""

_VIDEO_PROMPT = """You are a DP watching a previsualisation of a camera move in a real room, rendered from a 3D scan with the actors as coloured stand-in figures. The move: {move}. Rig: {rig}, {duration:.1f} s, {focal} mm.
Describe the move in two sentences (what the camera does, how the framing changes), then judge it: speed and smoothness, how the subject sits in frame through the move, what is behind them, and one concrete change. JSON: {{"description": str, "score": 1-10, "verdict": "keep|adjust|reject", "notes": str}}"""

_DP_PROMPT = """You are a DP looking at one frame of a planned camera move in a real room. The frame is rendered from the room's scan with the actors drawn as coloured figures at their real heights.
The move: {move}. This frame: t = {t:.1f} s, {focal:.0f} mm, {size}, subject {subject} at {distance:.1f} m. Engine problems on this frame: {problems}.
Answer JSON: {{"score": 1-10, "verdict": "keep|adjust|reject", "notes": "<two sentences: headroom and look room, what is behind the subject, whether the lens and height suit the move; name a concrete change if adjust>"}}"""


@dataclass
class DirectResult:
    timeline: dict
    evaluation: dict
    notes: str
    trace: list[dict] = field(default_factory=list)
    model_calls: int = 0
    looks: int = 0
    seconds: float = 0.0
    finished: bool = False

    def to_dict(self) -> dict:
        return {
            "timeline": self.timeline, "evaluation": self.evaluation, "notes": self.notes,
            "trace": self.trace, "model_calls": self.model_calls, "looks": self.looks,
            "seconds": round(self.seconds, 1), "finished": self.finished, "provider": "meta",
        }


class _Operator:
    """One directing session: the tools' shared state."""

    def __init__(self, twin: Twin, timeline: dict, *, client, pieces=None, actors=None, workdir: Path | None = None,
                 on_event: Callable[[dict], None] | None = None, instruction: str = ""):
        from ..film import sim

        self.sim = sim
        self.twin = twin
        self.client = client
        self.pieces = pieces or []
        self.actors = {a.name: a for a in (actors or [])}
        self.workdir = workdir
        self.on_event = on_event or (lambda e: None)
        self.instruction = instruction
        self.timeline = copy.deepcopy(timeline)
        self.evaluation: dict | None = None
        self.looks = 0
        self.watches = 0
        self.notes = ""
        self.finished = False
        self._maps = None
        self._occ = None

    # -- tools ----------------------------------------------------------------

    def describe_room(self) -> dict:
        from ..film import coverage as cov
        from ..film import space as spacemod

        s = self.twin.structure
        maps = self._floor_maps()
        standable = maps.standable()
        ij = np.argwhere(standable)
        spots = []
        if len(ij):
            pick = ij[np.linspace(0, len(ij) - 1, min(12, len(ij))).astype(int)]
            spots = [[round(float(v), 2) for v in row] for row in np.asarray(maps.world_of(pick), dtype=float).reshape(-1, 2)]
        rigs = {k: r.describe() for k, r in self.sim.RIGS.items()}
        out = {
            "location": self.twin.name,
            "floor_z": round(float(s.floor_z), 3),
            "ceiling_height_m": None if s.ceiling_height is None else round(float(s.ceiling_height), 2),
            "floor_area_m2": round(float(s.floor_area), 1),
            "footprint_xy": None if s.footprint is None else [[round(float(x), 2), round(float(y), 2)] for x, y in np.asarray(s.footprint)[:12]],
            "openings": [{"kind": o.kind, "centre": [round(float(v), 2) for v in o.center], "width_m": round(float(o.width), 2)} for o in s.openings],
            "standable_spots_sample": spots,
            "actors": {n: {"height_m": a.height_m} for n, a in self.actors.items()},
            "timeline_actors": {n: v.get("height_m") for n, v in (self.timeline.get("actors") or {}).items()},
            "rigs": rigs,
            "sensor": self.timeline.get("sensor"),
            "current_timeline": self.timeline,
        }
        try:
            marks = cov.describe_marks(self.twin, [tuple(p) for p in spots])
            out["spot_cues"] = [{"xy": [m["x"], m["y"]], "cues": m["cues"]} for m in marks]
        except Exception:  # noqa: BLE001
            pass
        return out

    def set_rig(self, kind: str, params: dict | None = None) -> dict:
        if kind not in self.sim.RIGS:
            return {"error": f"no rig {kind!r}; rigs are {sorted(self.sim.RIGS)}"}
        self.timeline["rig"] = {"kind": kind, "params": dict(params or {})}
        return {"ok": True, "rig": self.timeline["rig"], "describe": self.sim.RIGS[kind].describe()}

    def set_camera_keys(self, keys: list[dict], interp: str | None = None, duration_s: float | None = None) -> dict:
        clean = []
        for k in keys or []:
            try:
                pos = [float(v) for v in k["pos"]]
                if len(pos) != 3:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                return {"error": "every key needs pos [x, y, z] in twin metres"}
            rec = {"t": float(k.get("t", 0.0)), "pos": pos, "focal_mm": float(k.get("focal_mm", 35.0)),
                   "roll_deg": float(k.get("roll_deg", 0.0)), "ease": str(k.get("ease") or "inout")}
            if k.get("look_at"):
                rec["look_at"] = str(k["look_at"])
                rec["aim"] = None
            elif k.get("aim") is not None:
                rec["aim"] = [float(v) for v in k["aim"]]
                rec["look_at"] = None
            else:
                rec["aim"] = None
                rec["look_at"] = None
            clean.append(rec)
        clean.sort(key=lambda r: r["t"])
        self.timeline.setdefault("camera", {})["keys"] = clean
        if interp:
            self.timeline["camera"]["interp"] = interp
        if duration_s:
            self.timeline["duration_s"] = float(duration_s)
        return {"ok": True, "keys": len(clean)}

    def set_actor_keys(self, name: str, keys: list[dict], height_m: float | None = None) -> dict:
        clean = []
        for k in keys or []:
            try:
                clean.append({"t": float(k.get("t", 0.0)), "pos": [float(k["pos"][0]), float(k["pos"][1])],
                              "facing_deg": float(k.get("facing_deg", 90.0))})
            except (KeyError, TypeError, ValueError, IndexError):
                return {"error": "every actor key needs pos [x, y]"}
        clean.sort(key=lambda r: r["t"])
        actors = self.timeline.setdefault("actors", {})
        h = height_m or (self.actors[name].height_m if name in self.actors else None) or (actors.get(name) or {}).get("height_m") or 1.75
        actors[name] = {"height_m": float(h), "keys": clean}
        return {"ok": True, "actor": name, "keys": len(clean), "height_m": float(h)}

    def evaluate(self) -> dict:
        try:
            tl = self.sim.Timeline.from_dict(self.timeline)
            ev = self.sim.evaluate(tl, self.twin, maps=self._floor_maps(), occupancy=self._occupancy(), pieces=self.pieces or None)
        except Exception as exc:  # noqa: BLE001 - the model must see why
            return {"error": f"{type(exc).__name__}: {exc}"}
        d = ev.to_dict()
        self.evaluation = d
        self.timeline = d.get("timeline") or self.timeline
        frames = d.get("frames") or []
        fails = [s for s in d.get("segments") or [] if s.get("severity") == "fail"]
        return {
            "frames": len(frames),
            "duration_s": self.timeline.get("duration_s"),
            "summary": d.get("summary"),
            "segments": (d.get("segments") or [])[:20],
            "fail_segments": len(fails),
            "notes": (self.timeline.get("notes") or [])[-6:],
            "sizes": sorted({str(f.get("shot_size")) for f in frames if f.get("shot_size")}),
            "sample_frames": [
                {"t": f["t"], "cam": f["cam"], "focal_mm": f["focal_mm"], "shot_size": f.get("shot_size"),
                 "distance_m": f.get("distance_m"), "problems": f.get("problems")}
                for f in frames[:: max(1, len(frames) // 6)][:7]
            ],
        }

    def look(self, t: float) -> dict:
        if self.evaluation is None:
            return {"error": "evaluate first"}
        if self.looks >= MAX_LOOKS:
            return {"error": f"you have looked {MAX_LOOKS} times; finish or fix the move"}
        frames = self.evaluation.get("frames") or []
        if not frames:
            return {"error": "no frames"}
        fr = min(frames, key=lambda f: abs(float(f["t"]) - float(t)))
        try:
            png = self.render_frame(fr)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"could not render: {type(exc).__name__}: {exc}"}
        self.looks += 1
        if self.workdir is not None:
            p = Path(self.workdir) / "looks" / f"look_{self.looks:02d}_t{float(fr['t']):05.2f}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(png)
            self.on_event({"kind": "look", "t": fr["t"], "file": p.name})
        prompt = _DP_PROMPT.format(
            move=self.instruction or self.timeline.get("title") or "the move", t=float(fr["t"]),
            focal=float(fr.get("focal_mm") or 0), size=fr.get("shot_size") or "?",
            subject=_subject_name(fr, self.timeline), distance=float(fr.get("distance_m") or 0),
            problems="; ".join(fr.get("problems") or []) or "none",
        )
        try:
            data = self.client.chat_json(prompt, images=[png], model=model_for("vision"))
        except Exception as exc:  # noqa: BLE001
            return {"t": fr["t"], "error": f"the DP could not look: {exc}"}
        if not isinstance(data, dict):
            data = {"notes": str(data)}
        out = {"t": fr["t"], "score": data.get("score"), "verdict": data.get("verdict"), "notes": data.get("notes"),
               "engine_problems": fr.get("problems") or []}
        self.on_event({"kind": "review", "t": fr["t"], "score": out["score"], "verdict": out["verdict"], "text": out["notes"]})
        return out

    def watch(self) -> dict:
        """The DP watches the whole move as a filmstrip: eight frames across
        the duration tiled in one picture with their times and lenses, read
        for how the framing develops rather than how one frame sits."""
        if self.evaluation is None:
            return {"error": "evaluate first"}
        if self.watches >= MAX_WATCHES:
            return {"error": f"you have watched {MAX_WATCHES} times; fix the move or finish"}
        frames = self.evaluation.get("frames") or []
        if len(frames) < 2:
            return {"error": "no frames"}
        try:
            sheet, picked = self.filmstrip(frames)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"could not render the filmstrip: {type(exc).__name__}: {exc}"}
        self.watches += 1
        if self.workdir is not None:
            p = Path(self.workdir) / "looks" / f"watch_{self.watches:02d}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(sheet)
            self.on_event({"kind": "watch", "file": p.name, "frames": [f["t"] for f in picked]})
        segs = self.evaluation.get("segments") or []
        prompt = _WATCH_PROMPT.format(
            n=len(picked), move=self.instruction or self.timeline.get("title") or "the move",
            rig=(self.timeline.get("rig") or {}).get("kind", "?"), duration=float(self.timeline.get("duration_s") or 0),
            problems="; ".join(f"{g['t_start']:.1f}-{g['t_end']:.1f}s {g['text']}" for g in segs[:6]) or "none",
        )
        try:
            data = self.client.chat_json(prompt, images=[sheet], model=model_for("vision"))
        except Exception as exc:  # noqa: BLE001
            return {"error": f"the DP could not watch: {exc}"}
        if not isinstance(data, dict):
            data = {"notes": str(data)}
        out = {"score": data.get("score"), "verdict": data.get("verdict"), "notes": data.get("notes"),
               "frames_shown": [round(float(f["t"]), 2) for f in picked]}
        self.on_event({"kind": "review", "t": None, "score": out["score"], "verdict": out["verdict"],
                       "text": out["notes"], "scope": "move"})
        return out

    def filmstrip(self, frames: list[dict], *, n: int = FILMSTRIP_FRAMES, width_px: int = 480) -> tuple[bytes, list[dict]]:
        """Eight frames across the move, tiled 4 x 2 with a caption each."""
        from PIL import Image, ImageDraw

        idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int)
        picked = [frames[int(i)] for i in idx]
        tiles = []
        for fr in picked:
            png = self.render_frame(fr, width_px=width_px)
            im = Image.open(io.BytesIO(png)).convert("RGB")
            cap = Image.new("RGB", (im.width, 22), (12, 14, 18))
            d = ImageDraw.Draw(cap)
            d.text((6, 5), f"t {float(fr['t']):.1f}s  {float(fr.get('focal_mm') or 0):.0f}mm  {fr.get('shot_size') or ''}  "
                           f"{float(fr.get('distance_m') or 0):.1f}m", fill=(230, 230, 230))
            tile = Image.new("RGB", (im.width, im.height + cap.height))
            tile.paste(im, (0, 0)); tile.paste(cap, (0, im.height))
            tiles.append(tile)
        cols = 4
        rows = (len(tiles) + cols - 1) // cols
        w, h = tiles[0].size
        sheet = Image.new("RGB", (cols * w, rows * h), (10, 12, 15))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * w, (i // cols) * h))
        buf = io.BytesIO()
        sheet.save(buf, format="PNG")
        return buf.getvalue(), picked

    def finish(self, notes: str) -> dict:
        self.notes = str(notes or "").strip()
        self.finished = True
        return {"ok": True}

    # -- rendering ------------------------------------------------------------------

    def render_frame(self, fr: dict, *, width_px: int = REVIEW_WIDTH_PX) -> bytes:
        from ..film import render as R
        from ..product import cast as castmod

        if hasattr(self.sim, "render_frame"):
            img = self.sim.render_frame(self.twin, fr, figures=self.figures_at(fr), width_px=width_px)
            if hasattr(img, "save"):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()
            if isinstance(img, (bytes, bytearray)):
                return bytes(img)
        cam = R.camera_at(fr["cam"], fr["aim"], float(fr["focal_mm"]),
                          sensor_key=str(self.timeline.get("sensor") or R.optics.DEFAULT_SENSOR), width_px=width_px)
        figs = self.figures_at(fr)
        field = R.gaussian_field(self.twin)
        if field is not None:
            if figs:
                field = R.merge_fields([field] + [f.as_gaussians() for f in figs if len(f.xyz)])
            arr = R._render_gaussians(field, cam)
        else:
            extra = [(f.xyz, (f.rgb * 255.0 + 0.5).astype(np.uint8)) for f in figs if len(f.xyz)]
            arr = R._render_points(self.twin, cam, extra=extra)
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    def figures_at(self, fr: dict) -> list:
        from ..product import cast as castmod
        from ..product import setpieces as spm

        floor_z = float(self.twin.structure.floor_z)
        figs = []
        try:
            from ..product import figures as figmod
        except ImportError:  # the articulated figures are optional until they land
            figmod = None
        for i, (name, v) in enumerate((fr.get("actors") or {}).items()):
            a = self.actors.get(name)
            if a is None:
                h = float(((self.timeline.get("actors") or {}).get(name) or {}).get("height_m") or 1.75)
                a = castmod.Actor(name=name, height_m=h, colour=castmod.PALETTE[i % len(castmod.PALETTE)])
            if figmod is not None and not getattr(a, "body_path", None):
                speed = self._actor_speed(name, fr)
                pose = "walking" if speed > 0.15 else "standing"
                phase = figmod.walk_phase(float(fr.get("t") or 0.0), speed) if pose == "walking" else 0.0
                figs.append(figmod.figure_for_actor(a, (float(v[0]), float(v[1])), float(v[2]), floor_z,
                                                    pose=pose, phase=phase))
                continue
            figs.append(castmod.figure(a, (float(v[0]), float(v[1])), float(v[2]), floor_z,
                                       workdir=self.workdir.parent if self.workdir else None))
        for p in self.pieces:
            if getattr(p, "moved", False):
                try:
                    figs.append(spm.figure(p, self.twin, self.workdir.parent if self.workdir else None))
                except Exception:  # noqa: BLE001
                    continue
        return figs

    def _actor_speed(self, name: str, fr: dict) -> float:
        """An actor's walking speed at this frame, from the evaluation's neighbours."""
        frames = (self.evaluation or {}).get("frames") or []
        if len(frames) < 2:
            return 0.0
        t = float(fr.get("t") or 0.0)
        k = min(range(len(frames)), key=lambda i: abs(float(frames[i]["t"]) - t))
        j = k + 1 if k + 1 < len(frames) else k - 1
        a, b = frames[k].get("actors", {}).get(name), frames[j].get("actors", {}).get(name)
        if not a or not b:
            return 0.0
        dt = abs(float(frames[j]["t"]) - float(frames[k]["t"])) or 1.0
        return float(math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])) / dt)

    # -- caches --------------------------------------------------------------------

    def _floor_maps(self):
        from ..film import space as spacemod

        if self._maps is None:
            self._maps = spacemod.floor_maps(self.twin)
        return self._maps

    def _occupancy(self):
        from ..film import space as spacemod

        if self._occ is None:
            self._occ = spacemod.occupancy(self.twin)
        return self._occ


def _subject_name(fr: dict, timeline: dict) -> str:
    s = fr.get("subject")
    if isinstance(s, dict):
        return str(s.get("name") or "?")
    if isinstance(s, str) and s:
        return s
    return "the aim point"


# ---------------------------------------------------------------------------
# tool schemas
# ---------------------------------------------------------------------------


def _tools() -> list[dict]:
    key = {"type": "object", "properties": {
        "t": {"type": "number"}, "pos": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        "aim": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        "look_at": {"type": "string"}, "focal_mm": {"type": "number"}, "roll_deg": {"type": "number"},
        "ease": {"type": "string", "enum": ["inout", "linear"]}}, "required": ["t", "pos"]}
    akey = {"type": "object", "properties": {
        "t": {"type": "number"}, "pos": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
        "facing_deg": {"type": "number"}}, "required": ["t", "pos"]}
    return [
        tool_schema("describe_room", "The room: floor_z, ceiling, footprint, openings, a sample of standable spots with cues, the cast, the rigs the engine offers, and the current timeline. Call first.", {"type": "object", "properties": {}}),
        tool_schema("set_rig", "Choose the rig for the move.", {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["tripod", "slider", "dolly", "jib", "handheld", "steadicam", "crane", "drone"]},
            "params": {"type": "object"}}, "required": ["kind"]}),
        tool_schema("set_camera_keys", "Replace the camera keyframes. Positions in twin metres (z absolute: floor_z plus height). Use look_at with an actor's name to track their eyeline, else aim.", {"type": "object", "properties": {
            "keys": {"type": "array", "items": key}, "interp": {"type": "string", "enum": ["linear", "catmull", "ease"]},
            "duration_s": {"type": "number"}}, "required": ["keys"]}),
        tool_schema("set_actor_keys", "Replace one actor's marks over time (x, y on the floor, facing in degrees).", {"type": "object", "properties": {
            "name": {"type": "string"}, "keys": {"type": "array", "items": akey}, "height_m": {"type": "number"}}, "required": ["name", "keys"]}),
        tool_schema("evaluate", "Solve the rig constraints and check every frame against the room: fit, headroom, sightline, speed, subject in frame. Returns problems by time segment.", {"type": "object", "properties": {}}),
        tool_schema("look", "Render the frame at time t and have a DP read it (headroom, background, whether the move reads). At most four looks.", {"type": "object", "properties": {"t": {"type": "number"}}, "required": ["t"]}),
        tool_schema("watch", "Have a DP watch the whole move as a filmstrip of eight frames across its duration: how the framing develops, the speed, where it gets too tight. Use after evaluate, before finish. At most two watches.", {"type": "object", "properties": {}}),
        tool_schema("finish", "Done: the move holds. One paragraph for the director.", {"type": "object", "properties": {"notes": {"type": "string"}}, "required": ["notes"]}),
    ]


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------


def direct(
    twin: Twin,
    timeline: dict,
    instruction: str,
    *,
    client,
    pieces=None,
    actors=None,
    workdir: Path | None = None,
    model: str | None = None,
    on_event: Callable[[dict], None] | None = None,
    max_calls: int = MAX_MODEL_CALLS,
) -> DirectResult:
    """Build the move the director asked for, with Muse Wonder on the tools."""
    t0 = time.perf_counter()
    say = on_event or (lambda e: None)
    op = _Operator(twin, timeline, client=client, pieces=pieces, actors=actors, workdir=workdir,
                   on_event=say, instruction=instruction)
    tools = _tools()
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM},
        user_message(f"The director says: {instruction.strip()}\n\nCurrent timeline (JSON): {json.dumps(timeline)[:4000]}"),
    ]
    trace: list[dict] = []
    calls = 0
    text_answers = 0
    while calls < max_calls and not op.finished:
        calls += 1
        res: ChatResult = client.chat(messages, model=model or model_for("agent"), tools=tools, max_tokens=2000)
        if res.tool_calls:
            messages.append({"role": "assistant", "content": res.text or None,
                             "tool_calls": [{"id": tc.id or f"c{calls}_{i}", "type": "function",
                                             "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                                            for i, tc in enumerate(res.tool_calls)]})
            for i, tc in enumerate(res.tool_calls):
                out = _dispatch(op, tc.name, tc.arguments)
                summary = _summarise(tc.name, out)
                trace.append({"kind": "call", "tool": tc.name, "args": tc.arguments, "summary": summary})
                say({"kind": "call", "tool": tc.name, "args": _short(tc.arguments), "summary": summary})
                messages.append({"role": "tool", "tool_call_id": tc.id or f"c{calls}_{i}",
                                 "content": json.dumps(out)[:12000]})
        else:
            text_answers += 1
            trace.append({"kind": "text", "text": res.text})
            say({"kind": "agent", "agent": "operator", "text": (res.text or "")[:400]})
            if op.evaluation is not None and text_answers >= 2 and not op.finished:
                # A model that stops calling tools after an evaluation is done, whatever it says.
                op.finish(res.text or "")
                break
            messages.append({"role": "assistant", "content": res.text or ""})
            messages.append(user_message("Continue with the tools: evaluate, fix any failing segment, look, then finish."))
    if op.evaluation is None:
        # The operator never evaluated: do it so the page has something honest to show.
        op.evaluate()
    if not op.finished:
        op.notes = op.notes or "The operator ran out of calls before finishing; the last evaluated move is shown with its problems."
    return DirectResult(timeline=op.timeline, evaluation=op.evaluation or {}, notes=op.notes, trace=trace,
                        model_calls=calls, looks=op.looks, seconds=time.perf_counter() - t0, finished=op.finished)


def _dispatch(op: _Operator, name: str, args: dict) -> dict:
    try:
        if name == "describe_room":
            return op.describe_room()
        if name == "set_rig":
            return op.set_rig(str(args.get("kind") or ""), args.get("params") or {})
        if name == "set_camera_keys":
            return op.set_camera_keys(args.get("keys") or [], args.get("interp"), args.get("duration_s"))
        if name == "set_actor_keys":
            return op.set_actor_keys(str(args.get("name") or ""), args.get("keys") or [], args.get("height_m"))
        if name == "evaluate":
            return op.evaluate()
        if name == "look":
            return op.look(float(args.get("t", 0.0)))
        if name == "watch":
            return op.watch()
        if name == "finish":
            return op.finish(str(args.get("notes") or ""))
    except Exception as exc:  # noqa: BLE001 - the model must see why, the run must not die
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"error": f"unknown tool {name!r}"}


def _summarise(name: str, out: dict) -> str:
    if "error" in out:
        return f"{name}: {out['error']}"[:200]
    if name == "evaluate":
        return f"evaluate: {out.get('frames')} frames, {out.get('fail_segments')} failing segments, sizes {out.get('sizes')}"
    if name == "look":
        return f"look at {out.get('t')} s: {out.get('score')}/10 {out.get('verdict')}"
    if name == "watch":
        return f"watched the move: {out.get('score')}/10 {out.get('verdict')}"
    if name == "set_camera_keys":
        return f"{out.get('keys')} camera keys"
    if name == "set_actor_keys":
        return f"{out.get('actor')}: {out.get('keys')} marks"
    if name == "set_rig":
        return f"rig {out.get('rig', {}).get('kind')}"
    if name == "describe_room":
        return "read the room"
    if name == "finish":
        return "finished"
    return name


def _short(args: dict) -> dict:
    s = json.dumps(args)
    return args if len(s) < 600 else {"_": s[:600] + "…"}


def review_move_video(mp4: bytes, *, client, instruction: str = "", timeline: dict | None = None, model: str | None = None) -> dict:
    """Muse Spark watches the rendered move as video and answers as a DP.

    The rendered mp4 is the measured previs: the room from its scan, the
    actors as figures. The answer is a description of what the camera does,
    a score, a verdict and one concrete change, for the page to show beside
    the video. Nothing in it changes the timeline; the director decides.
    """
    import base64

    tl = timeline or {}
    keys = (tl.get("camera") or {}).get("keys") or []
    focal = "/".join(f"{float(k.get('focal_mm') or 0):.0f}" for k in keys[:4]) or "?"
    prompt = _VIDEO_PROMPT.format(move=instruction or tl.get("title") or "the move",
                                  rig=(tl.get("rig") or {}).get("kind", "?"),
                                  duration=float(tl.get("duration_s") or 0), focal=focal)
    url = "data:video/mp4;base64," + base64.b64encode(mp4).decode("ascii")
    content = [{"type": "video_url", "video_url": {"url": url}}, {"type": "text", "text": prompt}]
    res = client.chat([{"role": "user", "content": content}], model=model or model_for("vision"),
                      response_format={"type": "json_object"}, max_tokens=600)
    from ..meta.client import parse_json_text

    data = parse_json_text(res.text)
    if not isinstance(data, dict):
        data = {"notes": str(data)}
    return {"description": data.get("description"), "score": data.get("score"), "verdict": data.get("verdict"),
            "notes": data.get("notes"), "model": res.model or model or model_for("vision"), "provider": "meta"}
