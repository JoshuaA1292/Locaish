"""The stage: a finished twin's cast, set pieces and rehearsals, as the studio serves them.

The studio's HTTP handler stays a router; everything a request needs to know
about a job's stage lives here, so the same calls answer the page, the agent
and the tests. One `Stage` per studio process. It owns:

- **the cast** of each job (`product.cast`): actors with real heights, their
  headshots, and -- when the SAM 3D Body endpoint is deployed -- their body
  meshes lifted from the headshot and scaled to the typed height;
- **the set pieces** (`product.setpieces`): labelled things lifted into
  movable solids, with the room re-swept when one moves so the shot table and
  the rehearsal see the room as it will be dressed;
- **the rehearsals** (`agent.rehearsal`): background runs with an event
  stream, like coverage plans, each leaving a directory of frames, a call
  sheet and a blocking diagram;
- **finished frames** (`meta.finish`): one animatic frame at a time, on
  request, stamped as a render;
- **camera simulations** (`film.sim`): keyed camera moves on a rig, solved
  and evaluated against the measured room on every edit, each kept as a
  directory with its timeline, its evaluation, its rendered move and the
  Blender hand-off. The engine is imported lazily so the studio serves the
  rest of the stage when it is not there.

What is configured decides what is offered: no Meta key, no prose breakdown
and no finished frames (typed scenes still rehearse); no SAM 3D endpoint, no
lifted bodies or splats (capsules and the scan's own points stand in).
"""

from __future__ import annotations

import copy
import json
import queue
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import meta as metamod
from ..film import coverage as cov


@dataclass
class RehearsalRun:
    """One rehearsal request against a finished job, and how it went."""

    id: str
    job_id: str
    workdir: Path
    scene_text: str
    events: queue.Queue = field(default_factory=queue.Queue)
    state: str = "queued"
    error: str | None = None
    result: dict | None = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def emit(self, kind: str, text: str = "", **extra) -> None:
        self.events.put({"kind": kind, "text": text, **extra})

    def listing(self) -> dict:
        r = self.result or {}
        sc = r.get("scene") or {}
        return {
            "rehearsal_id": self.id,
            "title": sc.get("title") or "Rehearsal",
            "state": self.state,
            "created_at": self.created_at,
            "duration_s": (r.get("rehearsal") or {}).get("duration_s"),
            "shots": len((r.get("rehearsal") or {}).get("shots") or []),
            "frames": len(r.get("animatic") or []),
            "provider": r.get("provider"),
            "problems": len((r.get("rehearsal") or {}).get("problems") or []),
        }


@dataclass
class SimRender:
    """One render of a simulated camera move: every frame as a PNG, then the mp4."""

    id: str
    job_id: str
    sim_id: str
    workdir: Path                      # <sim>/render
    events: queue.Queue = field(default_factory=queue.Queue)
    state: str = "queued"
    error: str | None = None
    result: dict | None = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def emit(self, kind: str, text: str = "", **extra) -> None:
        self.events.put({"kind": kind, "text": text, **extra})

    def done_item(self) -> dict:
        return {"kind": "done", "run": self.id, **(self.result or {})}


@dataclass
class SimDirect:
    """One directing session: Muse Wonder on the engine's tools, Muse Spark on the frames."""

    id: str
    job_id: str
    sim_id: str
    workdir: Path                      # <sim>/direct
    instruction: str
    events: queue.Queue = field(default_factory=queue.Queue)
    state: str = "queued"
    error: str | None = None
    result: dict | None = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def emit(self, kind: str, text: str = "", **extra) -> None:
        self.events.put({"kind": kind, "text": text, **extra})

    def done_item(self) -> dict:
        return {"kind": "done", "run": self.id, **(self.result or {})}


_SIM_ID = re.compile(r"[A-Za-z0-9_-]{1,40}")
_LOOK_NAME = re.compile(r"look_\d{2}_t[\d.]+\.png")
_SIM_FRAME_NAME = re.compile(r"frame_\d{5}\.png")
SIM_CAPTION_PX = 40
SIM_FIGURE_CAP = 3000


def _engine():
    """The simulation engine, imported when first asked for, so the stage
    serves cast, set pieces and rehearsals on a studio that has no engine."""
    try:
        from ..film import sim
    except ImportError as exc:
        raise RuntimeError(f"the camera simulation engine (locaish/film/sim.py) is not on this studio: {exc}") from exc
    return sim


class Stage:
    def __init__(self, studio) -> None:
        self.studio = studio
        self.runs: dict[str, dict[str, RehearsalRun]] = {}
        self._lock = threading.Lock()
        self._dressed: dict[str, tuple[str, Any, Any]] = {}   # job id -> (pose fingerprint, twin, sweep)
        self.sim_renders: dict[str, SimRender] = {}           # "<job>/<sim>/<run>" -> render
        self.sim_directs: dict[str, SimDirect] = {}
        self._sim_envs: dict[str, tuple[str, Any, Any]] = {}  # job id -> (pose fingerprint, floor maps, occupancy)
        self._sim_evals: dict[str, tuple[Any, Any]] = {}      # "<job>/<sim>" -> (Timeline, Evaluation) objects

    # -- what is on -------------------------------------------------------

    def status(self) -> dict:
        from ..meta import sam3d, vision

        return {
            "meta": metamod.configured(),
            "vision": vision.available(),
            "vision_provider": vision.provider(),
            "sam3d": sam3d.configured(),
            "finish": metamod.configured(),
            "breakdown": metamod.configured(),
        }

    def _client(self):
        return metamod.MetaClient() if metamod.configured() else None

    # -- cast ---------------------------------------------------------------

    def cast(self, job) -> list[dict]:
        from . import cast as castmod

        return castmod.describe(castmod.load_cast(job.workdir), workdir=job.workdir)

    def add_actor(self, job, body: dict) -> dict:
        from . import cast as castmod

        name = str(body.get("name") or "").strip()
        try:
            height = float(body.get("height_m"))
        except (TypeError, ValueError):
            raise ValueError("height_m is needed, in metres (1.75, not 175)")
        if height > 3.0:
            height = height / 100.0
        eye = body.get("eye_height_m")
        actor = castmod.add_actor(job.workdir, name, height, eye_height_m=float(eye) if eye else None)
        return actor.to_json()

    def remove_actor(self, job, actor_id: str) -> bool:
        from . import cast as castmod

        return castmod.remove_actor(job.workdir, actor_id)

    def headshot(self, job, actor_id: str, image: bytes, *, lift: bool = True) -> dict:
        """Save the headshot; lift a body when the endpoint is there."""
        from ..meta import sam3d
        from . import cast as castmod

        actor = castmod.get_actor(job.workdir, actor_id)
        if actor is None:
            raise KeyError("no such actor")
        ext = "png" if image[:8] == b"\x89PNG\r\n\x1a\n" else "jpg"
        rel = Path("headshots") / f"{actor.id}.{ext}"
        (job.workdir / rel).parent.mkdir(parents=True, exist_ok=True)
        (job.workdir / rel).write_bytes(image)
        actor.headshot = str(rel)
        cast = castmod.load_cast(job.workdir)
        castmod.save_cast(job.workdir, [actor if a.id == actor.id else a for a in cast])
        out = {"actor": actor.to_json(), "body": None, "note": ""}
        if lift and sam3d.configured():
            try:
                lifted = sam3d.lift_body(image)
                npz = job.workdir / "bodies" / f"{actor.id}.npz"
                sam3d.to_body_npz(lifted, npz)
                castmod.attach_body(actor, npz, workdir=job.workdir)
                out["body"] = {"path": str(npz.relative_to(job.workdir)), "model_height_units": lifted.height_units,
                               "vertices": int(len(lifted.vertices))}
                out["actor"] = actor.to_json()
            except Exception as exc:  # noqa: BLE001 - the capsule stands in
                out["note"] = f"the body could not be lifted ({type(exc).__name__}: {exc}); a capsule stands in"
        elif lift:
            out["note"] = "SAM 3D Body is not deployed on this studio (LOCAISH_SAM3D_URL); a capsule at the typed height stands in"
        return out

    def headshot_bytes(self, job, actor_id: str) -> tuple[bytes, str] | None:
        from . import cast as castmod

        actor = castmod.get_actor(job.workdir, actor_id)
        if actor is None or not actor.headshot:
            return None
        p = job.workdir / actor.headshot
        if not p.exists():
            return None
        return p.read_bytes(), "image/png" if p.suffix == ".png" else "image/jpeg"

    # -- set pieces -----------------------------------------------------------

    def pieces(self, job) -> list[dict]:
        from . import setpieces as spm

        twin = self.studio.twin_of(job)
        return spm.describe(spm.load_pieces(job.workdir), twin)

    def lift_piece(self, job, label_id: str | None = None, *, xyz=None, label: str | None = None) -> dict:
        from . import labels as labelsmod
        from . import setpieces as spm

        twin = self.studio.twin_of(job)
        rec = None
        if label_id:
            rec = next((l for l in labelsmod.load_labels(job.workdir) if l.get("id") == label_id), None)
            if rec is None:
                raise KeyError("no such label")
        elif xyz is not None:
            rec = {"id": None, "label": label or "object", "kind": "other", "xyz": [float(v) for v in xyz]}
        else:
            raise ValueError("give a label id or a point")
        existing = [p for p in spm.load_pieces(job.workdir) if p.label_id and p.label_id == rec.get("id")]
        if existing:
            return spm.describe(existing, twin)[0]
        piece = spm.lift_from_label(twin, rec, job.workdir)
        self._dressed.pop(job.id, None)
        return spm.describe([piece], twin)[0]

    def move_piece(self, job, piece_id: str, body: dict) -> dict:
        from . import setpieces as spm

        twin = self.studio.twin_of(job)
        pieces = spm.load_pieces(job.workdir)
        piece = next((p for p in pieces if p.id == piece_id), None)
        if piece is None:
            raise KeyError("no such piece")
        before = self.dressed_sweep(job)
        if body.get("reset"):
            spm.reset(piece)
        else:
            spm.move(piece, float(body.get("dx") or 0.0), float(body.get("dy") or 0.0),
                     float(body.get("dyaw_deg") or 0.0), relative=bool(body.get("relative")))
        spm.upsert_piece(job.workdir, piece)
        self._dressed.pop(job.id, None)
        after = self.dressed_sweep(job)
        diff = spm.sweep_diff(before, after) if before is not None and after is not None else None
        return {"piece": spm.describe([piece], twin)[0], "diff": diff}

    def remove_piece(self, job, piece_id: str) -> bool:
        from . import setpieces as spm

        ok = spm.remove_piece(job.workdir, piece_id)
        self._dressed.pop(job.id, None)
        return ok

    def lift_piece_3d(self, job, piece_id: str) -> dict:
        """SAM 3.1 masks the thing in the frame that saw it best; SAM 3D Objects
        lifts it to a textured splat; the splat is fitted to the measured
        footprint and attached. Needs both the Meta API and the Modal endpoint."""
        from ..meta import sam3d
        from . import setpieces as spm

        if not metamod.configured():
            raise RuntimeError("the Meta Model API is not configured (META_API_KEY), so SAM 3.1 cannot mask the object")
        if not sam3d.configured():
            raise RuntimeError("SAM 3D Objects is not deployed on this studio (LOCAISH_SAM3D_URL)")
        twin = self.studio.twin_of(job)
        pieces = spm.load_pieces(job.workdir)
        piece = next((p for p in pieces if p.id == piece_id), None)
        if piece is None:
            raise KeyError("no such piece")
        frames = self.studio.frames_of(job)
        if frames is None:
            raise RuntimeError("this twin has no posed frames to lift from")
        views = frames.best_views(piece.anchor, self.studio.tree_of(job), k=3)
        if not views:
            raise RuntimeError("no frame has a clear view of this piece")
        client = self._client()
        chosen = None
        for v in views:
            image = frames.paths[int(v["index"])].read_bytes()
            dets = client.segment(piece.label, image)
            # The detection whose box holds the pixel the anchor projects to.
            u, vv = float(v["u"]) / frames.size[0], float(v["v"]) / frames.size[1]
            hit = [d for d in dets if d.mask_png is not None and d.box[0] <= u <= d.box[2] and d.box[1] <= vv <= d.box[3]]
            if hit:
                chosen = (image, max(hit, key=lambda d: d.score))
                break
        if chosen is None:
            raise RuntimeError(f"SAM 3.1 found no {piece.label!r} where the twin says it is")
        image, det = chosen
        lifted = sam3d.lift_object(image, det.mask_png)
        if lifted.gaussians is None:
            raise RuntimeError("SAM 3D Objects returned no splat")
        fp = piece.footprint
        M = sam3d.fit_to_box(lifted.gaussians, (fp.width_m, fp.depth_m, fp.height_m),
                             (fp.cx, fp.cy, fp.z_base), fp.yaw_deg)
        npz = job.workdir / "setpieces" / f"{piece.id}_lift.npz"
        sam3d.to_field_npz(lifted, npz, transform=M)
        spm.attach_gaussians(piece, job.workdir, npz)
        spm.upsert_piece(job.workdir, piece)
        return {"piece": spm.describe([piece], twin)[0], "gaussians": int(len(lifted.gaussians["xyz"]))}

    def dressed(self, job):
        """The twin with every piece where it was put, and its sweep; cached on
        the pieces' poses. The room as scanned when nothing has moved."""
        from ..film import sweep as sweepmod
        from . import setpieces as spm

        pieces = spm.load_pieces(job.workdir)
        twin = self.studio.twin_of(job)
        moved = [p for p in pieces if p.moved and p.has_points]
        if not moved:
            return twin, self.studio.sweep_of(job), pieces
        key = json.dumps([(p.id, p.pose) for p in moved], sort_keys=True)
        with self._lock:
            hit = self._dressed.get(job.id)
            if hit and hit[0] == key:
                return hit[1], hit[2], pieces
        room = spm.apply(twin, pieces)
        sw = sweepmod.sweep(room)
        with self._lock:
            self._dressed[job.id] = (key, room, sw)
        return room, sw, pieces

    def dressed_sweep(self, job):
        try:
            return self.dressed(job)[1]
        except Exception:  # noqa: BLE001
            return None

    def plan_svg(self, job) -> str:
        """The floor plan with every set piece's footprint where it stands now."""
        from . import setpieces as spm

        twin = self.studio.twin_of(job)
        pieces = spm.load_pieces(job.workdir)

        def overlay(X, Y):
            parts = []
            for p in pieces:
                fp = p.footprint_now()
                c, s = np.cos(np.radians(fp.yaw_deg)), np.sin(np.radians(fp.yaw_deg))
                hw, hd = fp.width_m / 2, fp.depth_m / 2
                corners = [(fp.cx + c * dx - s * dy, fp.cy + s * dx + c * dy)
                           for dx, dy in ((-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd))]
                pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in corners)
                col = "#e5c07b" if p.moved else "#8ab4f8"
                parts.append(f'<polygon points="{pts}" fill="{col}" fill-opacity="0.18" stroke="{col}" stroke-width="1.5"/>')
                parts.append(f'<text x="{X(fp.cx):.1f}" y="{Y(fp.cy):.1f}" fill="{col}" font-size="11" text-anchor="middle">{_esc(p.label)}</text>')
                if p.moved:
                    o = p.footprint
                    parts.append(f'<line x1="{X(o.cx):.1f}" y1="{Y(o.cy):.1f}" x2="{X(fp.cx):.1f}" y2="{Y(fp.cy):.1f}" stroke="{col}" stroke-dasharray="3 3"/>')
            return parts

        return cov.floor_plan_svg(twin, None, overlay=overlay, caption="set pieces: blue as scanned, amber moved")

    # -- rehearsals --------------------------------------------------------------

    def rehearsals(self, job) -> list[dict]:
        runs = self.runs.get(job.id)
        if runs is None:
            runs = self._restore(job)
        out = [r.listing() for r in runs.values()]
        out.sort(key=lambda r: r["created_at"])
        return out

    def _restore(self, job) -> dict[str, RehearsalRun]:
        runs: dict[str, RehearsalRun] = {}
        root = job.workdir / "rehearsals"
        if root.is_dir():
            for d in sorted(root.iterdir()):
                f = d / "rehearsal.json"
                if f.exists():
                    try:
                        res = json.loads(f.read_text())
                    except ValueError:
                        continue
                    run = RehearsalRun(id=d.name, job_id=job.id, workdir=d, scene_text="", state="done", result=res)
                    run.created_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(f.stat().st_mtime))
                    runs[d.name] = run
        self.runs[job.id] = runs
        return runs

    def run(self, job, run_id: str) -> RehearsalRun | None:
        runs = self.runs.get(job.id)
        if runs is None:
            runs = self._restore(job)
        return runs.get(run_id)

    def start_rehearsal(self, job, body: dict) -> RehearsalRun:
        scene = str(body.get("scene") or "").strip()
        if not scene:
            raise ValueError("paste a scene")
        rid = uuid.uuid4().hex[:8]
        run = RehearsalRun(id=rid, job_id=job.id, workdir=job.workdir / "rehearsals" / rid, scene_text=scene)
        if self.runs.get(job.id) is None:
            self._restore(job)
        self.runs[job.id][rid] = run
        threading.Thread(target=self._run, args=(job, run, body), daemon=True).start()
        return run

    def _run(self, job, run: RehearsalRun, body: dict) -> None:
        from ..agent import rehearsal as ra
        from . import cast as castmod
        from . import labels as labelsmod

        run.state = "running"
        try:
            run.emit("stage", "loading the twin")
            room, sw, pieces = self.dressed(job)
            source = cov.LocalSetups(sw) if pieces and any(p.moved for p in pieces) else self.studio.setups_for(job)
            actors = castmod.load_cast(job.workdir)
            contents = labelsmod.describe(labelsmod.load_labels(job.workdir))
            client = self._client()
            looks_typed = any(l.strip().upper().startswith(("MARKS:", "CAST:")) for l in run.scene_text.splitlines())
            if client is None and not looks_typed:
                raise ra.RehearsalUnavailable(
                    "a prose scene needs Muse Spark for the breakdown (set META_API_KEY on this studio); "
                    "or type the scene as beats (MARKS:, CAST:, numbered beats)"
                )
            result = ra.run_rehearsal(
                self.studio.twin_of(job), source, workdir=run.workdir, scene_text=run.scene_text,
                actors=actors, pieces=pieces, contents=contents,
                shoot_date=body.get("shoot_date") or None, shoot_start=body.get("shoot_start") or None,
                setup_minutes=int(body.get("setup_minutes") or 25), client=client,
                every_s=float(body.get("every_s") or 1.0),
                progress=lambda m: run.emit("stage", m),
            )
            run.result = result.to_dict()
            for p in result.rehearsal.problems:
                run.emit("note", p)
            run.state = "done"
            run.events.put({"kind": "done", "rehearsal_id": run.id, "listing": run.listing()})
        except Exception as exc:  # noqa: BLE001 - the browser has to see why
            run.state = "failed"
            run.error = f"{type(exc).__name__}: {exc}"
            run.events.put({"kind": "error", "text": run.error, "detail": traceback.format_exc()[-2000:]})
        finally:
            run.events.put(None)

    # -- camera simulations --------------------------------------------------------

    def _sims_root(self, job) -> Path:
        return job.workdir / "sims"

    def _sim_dir(self, job, sim_id: str, *, must_exist: bool = True) -> Path:
        if not _SIM_ID.fullmatch(str(sim_id or "")):
            raise KeyError("no such simulation")
        d = self._sims_root(job) / sim_id
        if must_exist and not (d / "timeline.json").exists():
            raise KeyError("no such simulation")
        return d

    def sims(self, job) -> list[dict]:
        """Every timeline kept for this room, oldest first."""
        out = []
        root = self._sims_root(job)
        if not root.is_dir():
            return out
        for d in sorted(root.iterdir()):
            f = d / "timeline.json"
            if not f.exists():
                continue
            try:
                tl = json.loads(f.read_text())
            except ValueError:
                continue
            ev = None
            if (d / "evaluation.json").exists():
                try:
                    ev = json.loads((d / "evaluation.json").read_text())
                except ValueError:
                    ev = None
            out.append(self._sim_listing(d.name, tl, ev, f.stat().st_mtime))
        out.sort(key=lambda r: r["updated_at"])
        return out

    @staticmethod
    def _sim_listing(sim_id: str, tl: dict, ev: dict | None, mtime: float) -> dict:
        rig = (tl.get("rig") or {}).get("kind") or "tripod"
        problems = 0
        if ev:
            segs = ev.get("segments") or []
            problems = len(segs) if segs else sum(1 for f in ev.get("frames") or [] if f.get("problems"))
        return {
            "sim_id": sim_id,
            "title": tl.get("title") or "Camera move",
            "rig": rig,
            "duration_s": tl.get("duration_s"),
            "fps": tl.get("fps"),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mtime)),
            "problems": int(problems),
        }

    def _sim_env(self, job):
        """The room as dressed, with its floor maps and occupancy grid, cached
        on the set pieces' poses: an edit from the viewer must not pay for a
        raster of the floor every time a key moves."""
        from ..film import space

        room, _sw, pieces = self.dressed(job)
        moved = [p for p in pieces if p.moved and p.has_points]
        key = json.dumps([(p.id, p.pose) for p in moved], sort_keys=True)
        with self._lock:
            hit = self._sim_envs.get(job.id)
        if hit is not None and hit[0] == key:
            return room, hit[1], hit[2], pieces
        maps = space.floor_maps(room)
        occ = space.occupancy(room)
        with self._lock:
            self._sim_envs[job.id] = (key, maps, occ)
        return room, maps, occ, pieces

    def _run_engine(self, job, sim_id: str, timeline: dict) -> tuple[dict, dict]:
        """Solve the timeline against the dressed room and evaluate it; keep
        the engine's own objects for the Blender script."""
        sim = _engine()
        timeline = dict(timeline)
        timeline["id"] = sim_id
        tl = sim.Timeline.from_dict(timeline)
        room, maps, occ, pieces = self._sim_env(job)
        tl = sim.solve(tl, room)
        ev = sim.evaluate(tl, room, maps=maps, occupancy=occ, pieces=pieces)
        with self._lock:
            self._sim_evals[f"{job.id}/{sim_id}"] = (tl, ev)
        tl_d = tl.to_dict()
        tl_d["id"] = sim_id
        return tl_d, ev.to_dict()

    def _write_sim(self, job, sim_id: str, tl: dict, ev: dict) -> None:
        d = self._sim_dir(job, sim_id, must_exist=False)
        d.mkdir(parents=True, exist_ok=True)
        (d / "timeline.json").write_text(json.dumps(tl, indent=1))
        (d / "evaluation.json").write_text(json.dumps(ev))
        shutil.rmtree(d / "preview", ignore_errors=True)   # frames at the playhead are stale now

    def _answer(self, tl: dict, ev: dict) -> dict:
        info = self._rig_info(tl)
        return {"timeline": tl, "evaluation": ev, "rig_text": _rig_sentence(info), "rig_info": info if isinstance(info, dict) else None}

    @staticmethod
    def _rig_info(tl: dict):
        """What the engine says about the rig the timeline rides: a dict or a sentence."""
        try:
            sim = _engine()
            rig = sim.RIGS.get((tl.get("rig") or {}).get("kind") or "tripod")
            return None if rig is None else rig.describe()
        except Exception:  # noqa: BLE001 - a missing description is not an error
            return None

    def _load_sim(self, job, sim_id: str) -> tuple[dict, dict]:
        d = self._sim_dir(job, sim_id)
        tl = json.loads((d / "timeline.json").read_text())
        evf = d / "evaluation.json"
        if evf.exists():
            try:
                return tl, json.loads(evf.read_text())
            except ValueError:
                pass
        tl, ev = self._run_engine(job, sim_id, tl)
        self._write_sim(job, sim_id, tl, ev)
        return tl, ev

    def default_timeline(self, job, sim_id: str) -> dict:
        """Six seconds on a tripod at a spot the sweep found standable with a
        clean view, looking at the room's centre at a seated-to-standing eye."""
        twin = self.studio.twin_of(job)
        floor_z = float(twin.structure.floor_z)
        xyz = np.asarray(twin.points.xyz, dtype=np.float64)
        if twin.structure.footprint is not None and len(twin.structure.footprint):
            fp = np.asarray(twin.structure.footprint, dtype=np.float64).reshape(-1, 2)
            centre = [float(fp[:, 0].mean()), float(fp[:, 1].mean())]
        else:
            centre = [float(np.median(xyz[:, 0])), float(np.median(xyz[:, 1]))]
        pos = None
        try:
            sw = self.dressed(job)[1]
            c = sw.columns
            vis = np.asarray(c["visible"], dtype=bool)
            if vis.any():
                score = np.where(vis, np.asarray(c["score"], dtype=np.float64), -np.inf)
                i = int(np.argmax(score))
                pos = [float(c["cam_x"][i]), float(c["cam_y"][i]), floor_z + float(c["cam_z"][i])]
        except Exception:  # noqa: BLE001 - the sweep is a convenience here
            pos = None
        if pos is None:
            lo, hi = xyz.min(axis=0), xyz.max(axis=0)
            pos = [float(lo[0] + 0.25 * (hi[0] - lo[0])), float(lo[1] + 0.25 * (hi[1] - lo[1])), floor_z + 1.5]
        aim = [centre[0], centre[1], floor_z + 1.4]
        if abs(pos[0] - aim[0]) + abs(pos[1] - aim[1]) < 0.3:
            aim[0] += 1.0
        return {
            "id": sim_id, "title": "Camera move", "fps": 24, "duration_s": 6.0,
            "sensor": "ff", "aperture_f": 2.8,
            "rig": {"kind": "tripod", "params": {}},
            "camera": {"interp": "catmull", "keys": [
                {"t": 0.0, "pos": [round(v, 3) for v in pos], "aim": [round(v, 3) for v in aim], "look_at": None,
                 "focal_mm": 35, "roll_deg": 0, "ease": "inout"}]},
            "actors": {}, "pieces": [], "notes": [],
        }

    def create_sim(self, job, body: dict | None) -> dict:
        sim = _engine()
        sim_id = str(sim.new_id())
        if not _SIM_ID.fullmatch(sim_id):
            sim_id = uuid.uuid4().hex[:8]
        given = (body or {}).get("timeline")
        timeline = dict(given) if isinstance(given, dict) and given else self.default_timeline(job, sim_id)
        if (body or {}).get("title"):
            timeline["title"] = str(body["title"])
        tl, ev = self._run_engine(job, sim_id, timeline)
        self._write_sim(job, sim_id, tl, ev)
        return self._answer(tl, ev)

    def sim(self, job, sim_id: str) -> dict:
        tl, ev = self._load_sim(job, sim_id)
        return self._answer(tl, ev)

    def save_sim(self, job, sim_id: str, body: dict) -> dict:
        """Every edit from the viewer lands here: save, solve, evaluate, answer."""
        self._sim_dir(job, sim_id)
        timeline = body.get("timeline") if isinstance(body, dict) else None
        if not isinstance(timeline, dict) or not timeline:
            raise ValueError("send {timeline: {...}}")
        tl, ev = self._run_engine(job, sim_id, timeline)
        self._write_sim(job, sim_id, tl, ev)
        return self._answer(tl, ev)

    def delete_sim(self, job, sim_id: str) -> bool:
        try:
            d = self._sim_dir(job, sim_id)
        except KeyError:
            return False
        shutil.rmtree(d, ignore_errors=True)
        with self._lock:
            self._sim_evals.pop(f"{job.id}/{sim_id}", None)
            for k in [k for k in self.sim_renders if k.startswith(f"{job.id}/{sim_id}/")]:
                self.sim_renders.pop(k, None)
        return True

    def sim_from_rehearsal(self, job, sim_id: str, body: dict) -> dict:
        """A finished rehearsal's camera, cast and beats become an editable timeline."""
        sim = _engine()
        rid = str((body or {}).get("rehearsal_id") or "")
        run = self.run(job, rid) if rid else None
        if run is None:
            raise KeyError("no such rehearsal")
        if run.state != "done" or run.result is None:
            raise ValueError(f"that rehearsal is {run.state}")
        rehearsal = run.result
        f = run.workdir / "rehearsal.json"
        if f.exists():
            try:
                rehearsal = json.loads(f.read_text())
            except ValueError:
                pass
        # rehearsal.json is the agent's whole result; the engine reads the Rehearsal inside it.
        inner = rehearsal.get("rehearsal")
        if isinstance(inner, dict) and "frames" in inner:
            if not inner.get("scene") and isinstance(rehearsal.get("scene"), dict):
                inner = {**inner, "scene": rehearsal["scene"]}
            rehearsal = inner
        tl = sim.from_rehearsal(rehearsal, title=(body or {}).get("title") or None)
        timeline = tl.to_dict()
        timeline["id"] = sim_id
        tl_d, ev = self._run_engine(job, sim_id, timeline)
        self._write_sim(job, sim_id, tl_d, ev)
        return self._answer(tl_d, ev)

    # the cast and the moved pieces at a moment

    @staticmethod
    def _frame_at(ev: dict, t: float) -> dict | None:
        frames = ev.get("frames") or []
        if not frames:
            return None
        fps = _ev_fps(ev)
        i = int(round(float(t) * fps))
        if 0 <= i < len(frames) and abs(float(frames[i].get("t", i / fps)) - float(t)) <= 0.5 / fps + 1e-6:
            return frames[i]
        return min(frames, key=lambda f: abs(float(f.get("t", 0.0)) - float(t)))

    @staticmethod
    def _piece_offset_at(keys: list[dict], t: float) -> tuple[float, float, float]:
        """The piece's keyed offset at t: held before the first key, linear between, held after."""
        ks = sorted((k for k in keys or [] if "t" in k), key=lambda k: float(k["t"]))
        if not ks:
            return 0.0, 0.0, 0.0

        def val(k):
            return float(k.get("dx") or 0.0), float(k.get("dy") or 0.0), float(k.get("dyaw_deg") or 0.0)

        if t <= float(ks[0]["t"]):
            return val(ks[0])
        for a, b in zip(ks, ks[1:]):
            ta, tb = float(a["t"]), float(b["t"])
            if ta <= t <= tb:
                u = 0.0 if tb <= ta else (t - ta) / (tb - ta)
                va, vb = val(a), val(b)
                return tuple(va[i] + u * (vb[i] - va[i]) for i in range(3))
        return val(ks[-1])

    def _sim_figures(self, job, tl: dict, ev: dict, t: float):
        """The render.Figure objects (and the set pieces they stand for) at t."""
        from . import cast as castmod
        from . import setpieces as spm

        twin = self.studio.twin_of(job)
        floor_z = float(twin.structure.floor_z)
        frame = self._frame_at(ev, t)
        figures: list = []
        cast = {a.name: a for a in castmod.load_cast(job.workdir)}
        for name, where in ((frame or {}).get("actors") or {}).items():
            try:
                x, y = float(where[0]), float(where[1])
                facing = float(where[2]) if len(where) > 2 else 90.0
            except (TypeError, ValueError, IndexError):
                continue
            actor = cast.get(name)
            if actor is None:
                h = float(((tl.get("actors") or {}).get(name) or {}).get("height_m") or 1.75)
                actor = castmod.Actor(name=name, height_m=min(max(h, 1.2), 2.2),
                                      colour=castmod.next_colour(list(cast.values())))
            figures.append(castmod.figure(actor, (x, y), facing, floor_z, workdir=job.workdir))
        pieces = spm.load_pieces(job.workdir)
        keyed = {str(p.get("id")): p.get("keys") or [] for p in tl.get("pieces") or [] if p.get("id")}
        drawn: list = []
        for piece in pieces:
            if piece.id in keyed:
                dx, dy, dyaw = self._piece_offset_at(keyed[piece.id], t)
                shown = copy.deepcopy(piece)
                spm.move(shown, dx, dy, dyaw, relative=True)
            elif piece.moved:
                shown = piece
            else:
                continue
            figures.append(spm.figure(shown, twin, job.workdir))
            drawn.append(piece)
        return figures, drawn

    def sim_figures(self, job, sim_id: str, t: float) -> dict:
        """Point sets the viewer can draw: the cast where the evaluation puts
        them at t, and the pieces the timeline or the stage moved."""
        tl, ev = self._load_sim(job, sim_id)
        figures, _drawn = self._sim_figures(job, tl, ev, t)
        return {"t": float(t), "figures": [_figure_json(f) for f in figures]}

    # frames

    def _render_sim_frame(self, job, tl: dict, frame: dict, figures: list, drawn: list, width_px: int):
        """One frame of the move as a PIL image: the room through the frame's
        camera, the figures composited, a caption strip under it."""
        from PIL import Image, ImageDraw, ImageFont

        from ..film import render
        from . import setpieces as spm

        sim = _engine()
        twin = self.studio.twin_of(job)
        im = None
        if hasattr(sim, "render_frame"):
            try:
                got = sim.render_frame(twin, frame, figures, width_px)
                im = got if hasattr(got, "save") else Image.fromarray(np.asarray(got))
            except Exception:  # noqa: BLE001 - the local rasteriser stands in
                im = None
        if im is None:
            cam = _sim_camera(frame, tl, width_px)
            field = render.gaussian_field(twin)
            if field is not None:
                figs = [f.as_gaussians() for f in figures if len(f.xyz)]
                if figs:
                    field = render.merge_fields([field] + figs)
                img = render._render_gaussians(field, cam)
            else:
                extra = [(f.xyz, (f.rgb * 255.0 + 0.5).astype(np.uint8)) for f in figures if len(f.xyz)]
                exclude = spm.exclusion_mask(twin, drawn, moved_only=False) if drawn else None
                img = render._render_points(twin, cam, extra=extra, exclude=exclude)
            im = Image.fromarray(img)
        # libx264 with yuv420p wants even dimensions.
        W, H = im.width + (im.width % 2), im.height + (im.height % 2)
        canvas = Image.new("RGB", (W, H + SIM_CAPTION_PX), (14, 16, 19))
        canvas.paste(im, (0, 0))
        d = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        fps = _ev_fps({}, tl)
        t = float(frame.get("t") or 0.0)
        size = str(frame.get("shot_size") or "")
        dist = frame.get("distance_m")
        line = (f"t={t:5.2f}s  f{int(round(t * fps)):04d}  {float(frame.get('focal_mm') or 0):g}mm  {size}"
                + (f"  {float(dist):.1f} m" if dist is not None else "")
                + f"  dof {_dof_text(frame.get('dof'))}"
                + f"  pan {float(frame.get('pan_deg') or 0):.0f}  tilt {float(frame.get('tilt_deg') or 0):.0f}"
                + (f"  {float(frame.get('speed_mps') or 0):.2f} m/s" if frame.get("speed_mps") else ""))
        d.text((6, H + 4), line, fill=(232, 234, 237), font=font)
        probs = [str(p) for p in frame.get("problems") or []]
        if probs:
            d.text((6, H + 22), "; ".join(probs)[:160], fill=(224, 112, 122), font=font)
        return canvas

    def sim_frame_png(self, job, sim_id: str, t: float, *, width_px: int = 640) -> bytes:
        """The frame at the playhead, cached by frame index until the timeline changes."""
        import io

        tl, ev = self._load_sim(job, sim_id)
        frame = self._frame_at(ev, t)
        if frame is None:
            raise ValueError("the evaluation has no frames")
        fps = _ev_fps(ev, tl)
        idx = int(round(float(frame.get("t") or 0.0) * fps))
        width_px = int(min(max(width_px, 160), 1920))
        cache = self._sim_dir(job, sim_id) / "preview" / f"f_{idx:05d}_{width_px}.png"
        if cache.exists():
            return cache.read_bytes()
        figures, drawn = self._sim_figures(job, tl, ev, float(frame.get("t") or 0.0))
        im = self._render_sim_frame(job, tl, frame, figures, drawn, width_px)
        cache.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        data = buf.getvalue()
        try:
            cache.write_bytes(data)
        except OSError:
            pass
        return data

    # the rendered move

    def sim_render(self, job, sim_id: str, run_id: str) -> SimRender | None:
        return self.sim_renders.get(f"{job.id}/{sim_id}/{run_id}")

    def start_sim_render(self, job, sim_id: str, body: dict | None) -> SimRender:
        d = self._sim_dir(job, sim_id)
        body = body or {}
        run = SimRender(id=uuid.uuid4().hex[:8], job_id=job.id, sim_id=sim_id, workdir=d / "render")
        with self._lock:
            self.sim_renders[f"{job.id}/{sim_id}/{run.id}"] = run
        threading.Thread(target=self._render_move, args=(job, sim_id, run, body), daemon=True).start()
        return run

    def _render_move(self, job, sim_id: str, run: SimRender, body: dict) -> None:
        run.state = "running"
        t0 = time.time()
        try:
            tl, ev = self._load_sim(job, sim_id)
            frames = ev.get("frames") or []
            if not frames:
                raise ValueError("the evaluation has no frames to render")
            every = max(1, int(body.get("every") or 1))
            width_px = int(min(max(int(body.get("width_px") or 640), 160), 1920))
            fps = float(body.get("fps") or _ev_fps(ev, tl)) / every
            chosen = frames[::every]
            if run.workdir.exists():
                for old in run.workdir.glob("frame_*.png"):
                    old.unlink()
            run.workdir.mkdir(parents=True, exist_ok=True)
            run.emit("stage", f"rendering {len(chosen)} frames at {width_px} px", total=len(chosen))
            step = max(1, len(chosen) // 20)
            for i, fr in enumerate(chosen):
                figures, drawn = self._sim_figures(job, tl, ev, float(fr.get("t") or 0.0))
                im = self._render_sim_frame(job, tl, fr, figures, drawn, width_px)
                im.save(run.workdir / f"frame_{i:05d}.png")
                if (i + 1) % step == 0 or i + 1 == len(chosen):
                    run.emit("progress", f"frame {i + 1} of {len(chosen)}", done=i + 1, total=len(chosen),
                             frame=f"/sim-file/{job.id}/{sim_id}/frame_{i:05d}.png")
            mp4 = (run.workdir.parent / "move.mp4").resolve()
            note = ""
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                run.emit("stage", "encoding move.mp4")
                if mp4.exists():
                    mp4.unlink()
                proc = subprocess.run(
                    [ffmpeg, "-y", "-loglevel", "error", "-framerate", f"{fps:g}", "-i", "frame_%05d.png",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)],
                    cwd=run.workdir, capture_output=True, text=True, timeout=600,
                )
                if proc.returncode != 0 or not mp4.exists():
                    note = f"ffmpeg failed ({(proc.stderr or '').strip()[-300:]}); the frames are in render/"
            else:
                note = "ffmpeg is not on this studio's PATH, so no mp4: the frames are in render/"
            run.result = {
                "mp4": f"/sim-file/{job.id}/{sim_id}/move.mp4" if mp4.exists() else None,
                "frames": len(chosen), "fps": fps, "width_px": width_px,
                "seconds": round(time.time() - t0, 1), "note": note,
                "ffmpeg": bool(ffmpeg),
            }
            run.state = "done"
            run.events.put(run.done_item())
        except Exception as exc:  # noqa: BLE001 - the browser has to see why
            run.state = "failed"
            run.error = f"{type(exc).__name__}: {exc}"
            run.events.put({"kind": "error", "text": run.error, "detail": traceback.format_exc()[-2000:]})
        finally:
            run.events.put(None)

    # the files a move leaves behind

    # -- directing: a model on the engine's tools ----------------------------------

    def sim_direct(self, job, sim_id: str, run_id: str) -> SimDirect | None:
        return self.sim_directs.get(f"{job.id}/{sim_id}/{run_id}")

    def start_direct(self, job, sim_id: str, body: dict | None) -> SimDirect:
        """The director's sentence becomes a move: Muse Wonder builds it through
        the engine, Muse Spark looks at the frames, the result is saved as the
        timeline and evaluated like any edit."""
        if not metamod.configured():
            raise RuntimeError("directing needs the Meta Model API (META_API_KEY): Muse Wonder operates the camera")
        d = self._sim_dir(job, sim_id)
        instruction = str((body or {}).get("instruction") or "").strip()
        if not instruction:
            raise ValueError("say what the camera should do")
        run = SimDirect(id=uuid.uuid4().hex[:8], job_id=job.id, sim_id=sim_id, workdir=d / "direct",
                        instruction=instruction)
        with self._lock:
            self.sim_directs[f"{job.id}/{sim_id}/{run.id}"] = run
        threading.Thread(target=self._direct, args=(job, sim_id, run), daemon=True).start()
        return run

    def _direct(self, job, sim_id: str, run: SimDirect) -> None:
        from ..agent import sim as simagent
        from . import cast as castmod

        run.state = "running"
        try:
            run.emit("stage", "reading the room")
            tl, _ev = self._load_sim(job, sim_id)
            room, maps, occ, pieces = self._sim_env(job)
            actors = castmod.load_cast(job.workdir)
            run.workdir.mkdir(parents=True, exist_ok=True)
            run.emit("stage", "Muse Wonder is building the move")
            res = simagent.direct(
                room, tl, run.instruction, client=self._client(), pieces=pieces, actors=actors,
                workdir=run.workdir, on_event=run.events.put,
            )
            run.emit("stage", "evaluating the operator's move")
            answer = self.save_sim(job, sim_id, {"timeline": res.timeline})
            run.result = {**answer, "notes": res.notes, "trace": res.trace, "model_calls": res.model_calls,
                          "looks": res.looks, "seconds": round(res.seconds, 1), "finished": res.finished,
                          "looks_files": sorted(p.name for p in (run.workdir / "looks").glob("look_*.png"))
                          if (run.workdir / "looks").is_dir() else []}
            run.state = "done"
            run.events.put(run.done_item())
        except Exception as exc:  # noqa: BLE001 - the page has to see why
            run.state = "failed"
            run.error = f"{type(exc).__name__}: {exc}"
            run.events.put({"kind": "error", "text": run.error, "detail": traceback.format_exc()[-2000:]})
        finally:
            run.events.put(None)

    def sim_file(self, job, sim_id: str, name: str) -> tuple[bytes, str]:
        """`move.mp4`, `blender.py`, `room.glb`, `timeline.json`, `evaluation.json`,
        a rendered `frame_00000.png` or an operator's `look_01_t01.50.png`; (bytes, content type)."""
        d = self._sim_dir(job, sim_id)
        name = Path(name).name
        if _LOOK_NAME.fullmatch(name):
            f = d / "direct" / "looks" / name
            if not f.exists():
                raise KeyError("no such look")
            return f.read_bytes(), "image/png"
        if name == "timeline.json" or name == "evaluation.json":
            tl, ev = self._load_sim(job, sim_id)
            return json.dumps(tl if name == "timeline.json" else ev, indent=1).encode(), "application/json"
        if name == "move.mp4":
            f = d / "move.mp4"
            if not f.exists():
                raise KeyError("the move has not been rendered")
            return f.read_bytes(), "video/mp4"
        if _SIM_FRAME_NAME.fullmatch(name):
            f = d / "render" / name
            if not f.exists():
                raise KeyError("no such frame")
            return f.read_bytes(), "image/png"
        if name == "room.glb":
            return self._room_glb(job, d), "model/gltf-binary"
        if name == "blender.py":
            return self.blender_script(job, sim_id).encode(), "text/x-python; charset=utf-8"
        raise KeyError("not found")

    def blender_script(self, job, sim_id: str) -> str:
        sim = _engine()
        with self._lock:
            objs = self._sim_evals.get(f"{job.id}/{sim_id}")
        if objs is None:
            tl_d, _ev = self._load_sim(job, sim_id)
            self._run_engine(job, sim_id, tl_d)
            with self._lock:
                objs = self._sim_evals[f"{job.id}/{sim_id}"]
        tl, ev = objs
        try:
            return sim.blender_script(tl, ev, glb_path="room.glb", twin=self.studio.twin_of(job))
        except Exception:  # noqa: BLE001 - the twin only adds the room's extent; the move still exports
            return sim.blender_script(tl, ev, glb_path="room.glb", twin=None)

    def _room_glb(self, job, sim_dir: Path, *, max_points: int = 400_000) -> bytes:
        """The twin as GLB for Blender: its mesh when it has one, else its
        points as a coloured point primitive. Cached in the sim directory."""
        cache = sim_dir / "room.glb"
        if cache.exists():
            return cache.read_bytes()
        import trimesh

        twin = self.studio.twin_of(job)
        mesh = getattr(twin, "mesh", None)
        if mesh is not None and getattr(mesh, "faces", None) is not None and len(mesh.faces):
            tm = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces),
                                 vertex_colors=getattr(mesh, "vertex_colors", None), process=False)
            data = trimesh.Scene(tm).export(file_type="glb")
        else:
            xyz = np.asarray(twin.points.xyz, dtype=np.float64)
            rgb = twin.points.rgb
            if len(xyz) > max_points:
                idx = np.linspace(0, len(xyz) - 1, max_points).astype(np.int64)
                xyz = xyz[idx]
                rgb = rgb[idx] if rgb is not None else None
            if rgb is None:
                rgb = np.full((len(xyz), 3), 170, dtype=np.uint8)
            colours = np.concatenate([np.asarray(rgb, dtype=np.uint8).reshape(-1, 3),
                                      np.full((len(xyz), 1), 255, dtype=np.uint8)], axis=1)
            data = trimesh.Scene(trimesh.PointCloud(xyz, colors=colours)).export(file_type="glb")
        data = bytes(data)
        try:
            cache.write_bytes(data)
        except OSError:
            pass
        return data

    # -- finished frames -----------------------------------------------------------

    def finish(self, job, run: RehearsalRun, frame_name: str, body: dict) -> dict:
        from ..meta import finish as finmod
        from . import cast as castmod

        client = self._client()
        if client is None:
            raise RuntimeError("Muse Image needs the Meta Model API (META_API_KEY)")
        src = run.workdir / "animatic" / Path(frame_name).name
        if not src.exists():
            raise KeyError("no such frame")
        meta = {}
        try:
            info = json.loads((run.workdir / "animatic" / "animatic.json").read_text())
            frames = info.get("frames") if isinstance(info, dict) else info
            for fr in frames or []:
                if str(fr.get("file") or fr.get("png") or fr.get("name") or "") == src.name:
                    meta = fr
                    break
        except Exception:  # noqa: BLE001
            meta = {}
        focal = float(meta.get("focal_mm") or (meta.get("camera") or {}).get("focal_mm") or body.get("focal_mm") or 35.0)
        names = list((meta.get("actors") or {}).keys()) if isinstance(meta.get("actors"), dict) else []
        heads = []
        for a in castmod.load_cast(job.workdir):
            if (not names or a.name in names) and a.headshot and (job.workdir / a.headshot).exists():
                heads.append((job.workdir / a.headshot).read_bytes())
        out = finmod.finish_frame(
            src.read_bytes(), client=client, focal_mm=focal, headshots=heads, actors=names or None,
            description=str(body.get("description") or meta.get("caption") or ""),
            wardrobe=body.get("wardrobe") or None, light=body.get("light") or None, mood=str(body.get("mood") or ""),
        )
        dest = run.workdir / "finish" / f"finish_{src.stem}.png"
        out.save(dest)
        return {"file": dest.name, "prompt": out.prompt, "model": out.model, "note": out.note,
                "headshots": len(heads)}


def _ev_fps(ev: dict, tl: dict | None = None) -> float:
    """The evaluation's frame rate: top level, in its summary, in its timeline, or the timeline's own."""
    for v in (ev.get("fps"), (ev.get("summary") or {}).get("fps"), (ev.get("timeline") or {}).get("fps"),
              (tl or {}).get("fps")):
        try:
            if v is not None and float(v) > 0:
                return float(v)
        except (TypeError, ValueError):
            continue
    return 24.0


def _figure_json(fig) -> dict:
    """A render.Figure as the viewer draws it: flat xyz, one colour, a radius."""
    xyz = np.asarray(fig.xyz, dtype=np.float64).reshape(-1, 3)
    if len(xyz) > SIM_FIGURE_CAP:
        xyz = xyz[np.linspace(0, len(xyz) - 1, SIM_FIGURE_CAP).astype(np.int64)]
    rgb = np.asarray(fig.rgb, dtype=np.float32).reshape(-1, 3)
    colour = [int(round(float(v) * 255)) for v in (rgb.mean(axis=0) if len(rgb) else np.array([0.5, 0.5, 0.5]))]
    return {"name": fig.label or "", "colour": colour, "xyz": [round(float(v), 3) for v in xyz.ravel()],
            "radius_m": round(float(fig.radius_m), 4), "points": int(len(xyz))}


def _rig_sentence(info) -> str:
    """A rig's description as one readable line, whatever shape the engine gives it."""
    if not info:
        return ""
    if not isinstance(info, dict):
        return str(info)
    bits = []
    name = info.get("name") or info.get("kind") or "rig"
    dof = info.get("degrees_of_freedom") or info.get("dof")
    if dof:
        bits.append(f"{name}: moves in {', '.join(str(d) for d in dof)}")
    else:
        bits.append(str(name))
    lim = info.get("limits") or {}
    caps = []
    if lim.get("speed_mps") is not None:
        caps.append(f"{float(lim['speed_mps']):g} m/s")
    if lim.get("accel_mps2") is not None:
        caps.append(f"{float(lim['accel_mps2']):g} m/s² accel")
    if caps:
        bits.append("limits " + ", ".join(caps))
    checks = info.get("checks")
    if checks:
        bits.append("checks " + ", ".join(str(c) for c in checks))
    gear = info.get("gear_keys")
    if gear:
        bits.append("gear " + ", ".join(str(g) for g in gear))
    return "; ".join(bits) + "."


def _dof_text(dof) -> str:
    if not isinstance(dof, dict) or not dof:
        return "-"
    for near, far in (("near_m", "far_m"), ("near", "far")):
        if near in dof and far in dof:
            try:
                n = float(dof[near])
                f = dof[far]
                f = "inf" if f is None or (isinstance(f, (int, float)) and not np.isfinite(f)) else f"{float(f):.1f}"
                return f"{n:.1f}-{f} m"
            except (TypeError, ValueError):
                break
    return " ".join(f"{k}={v}" for k, v in list(dof.items())[:3])


def _sensor_key(sensor) -> str:
    from ..film import optics

    s = str(sensor or "").strip().lower()
    alias = {"ff": "fullframe", "full-frame": "fullframe", "full_frame": "fullframe", "35": "fullframe",
             "s35": "super35", "super-35": "super35", "mft": "m43", "phone": "phone-main"}
    s = alias.get(s, s)
    return s if s in optics.SENSORS else optics.DEFAULT_SENSOR


def _sim_camera(frame: dict, tl: dict, width_px: int):
    """The frame's pinhole, from its explicit aim point, with its roll."""
    from ..film import render

    cam = render.camera_at(frame["cam"], frame["aim"], float(frame.get("focal_mm") or 35.0),
                           sensor_key=_sensor_key(tl.get("sensor")), width_px=width_px)
    roll = float(frame.get("roll_deg") or 0.0)
    if abs(roll) < 1e-6:
        return cam
    c, s = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    right = c * cam.right + s * cam.down
    down = -s * cam.right + c * cam.down
    return render.Camera(cam.pos, cam.fwd, right, down, cam.fx, cam.fy, cam.width_px, cam.height_px)


def _esc(t: str) -> str:
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))
