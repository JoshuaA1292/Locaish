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
  request, stamped as a render.

What is configured decides what is offered: no Meta key, no prose breakdown
and no finished frames (typed scenes still rehearse); no SAM 3D endpoint, no
lifted bodies or splats (capsules and the scan's own points stand in).
"""

from __future__ import annotations

import json
import queue
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


class Stage:
    def __init__(self, studio) -> None:
        self.studio = studio
        self.runs: dict[str, dict[str, RehearsalRun]] = {}
        self._lock = threading.Lock()
        self._dressed: dict[str, tuple[str, Any, Any]] = {}   # job id -> (pose fingerprint, twin, sweep)

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


def _esc(t: str) -> str:
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))
