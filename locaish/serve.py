"""The studio: drop a video, get a twin, then talk to the scout about it.

One page, three states. A drop target; a quiet progress line while the
reconstruction runs; and the finished location -- the 3D twin embedded beside
the numbers that matter, with a chat to the Gemini scout agent underneath.
The whole product is this page: no accounts, no database of its own, no queue,
no build step.

By default it binds to loopback: the upload endpoint writes whatever bytes it
is given to disk and runs a reconstruction on them, which is exactly the kind
of endpoint that should not face the open internet casually. The hosted
deployment (Cloud Run) sets LOCAISH_HOST=0.0.0.0 and PORT explicitly -- an
opt-in with a platform in front of it, not a default.

The progress stream is server-sent events rather than a websocket because the
traffic is one-directional and a few dozen lines long; SSE is a `text/plain`
response that happens to be flushed early, and it needs neither a dependency
nor a handshake.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Uploads are streamed to disk in chunks rather than buffered: a room sweep off
# a modern phone is routinely several hundred megabytes, and reading that into
# a list of bytes to join at the end costs twice its size in RAM for no reason.
CHUNK = 1 << 20
MAX_UPLOAD_BYTES = 4 << 30


# What an uploaded capture can be, for recognising the source file of a job
# restored from disk. Video plus the scan formats the readers accept.
_MEDIA_SUFFIXES = {
    ".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm",
    ".ply", ".e57", ".las", ".laz", ".obj", ".glb", ".gltf", ".xyz", ".pts",
    ".twin",
}


@dataclass
class Job:
    """One uploaded capture, and everything that happened to it since."""

    id: str
    name: str
    source: Path
    workdir: Path
    events: queue.Queue = field(default_factory=queue.Queue)
    state: str = "queued"
    error: str | None = None
    twin_path: Path | None = None
    viewer_path: Path | None = None
    scout_path: Path | None = None
    location: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    summary: dict = field(default_factory=dict)
    # Cached after the first use: the loaded twin and its shot sweep. A
    # finished job is files on disk; these are the two that cost seconds to
    # rebuild and that every coverage request needs.
    sweep: object = None
    twin_obj: object = None
    plans: dict = field(default_factory=dict)

    def emit(self, kind: str, text: str, **extra) -> None:
        self.events.put({"kind": kind, "text": text, **extra})


@dataclass
class PlanRun:
    """One coverage request against a finished job, and how it went."""

    id: str
    job_id: str
    title: str
    brief: str
    mode: str
    workdir: Path
    events: queue.Queue = field(default_factory=queue.Queue)
    state: str = "queued"
    error: str | None = None
    plan_dict: dict | None = None
    floor_z: float = 0.0

    def emit(self, kind: str, text: str = "", **extra) -> None:
        self.events.put({"kind": kind, "text": text, **extra})

    def listing(self) -> dict:
        p = self.plan_dict or {}
        return {
            "plan_id": self.id,
            "title": p.get("title") or self.title,
            "created_at": p.get("created_at"),
            "planned": p.get("planned", 0),
            "shots": len(p.get("shots") or []),
            "planner": p.get("planner"),
            "state": self.state,
        }


class Studio:
    """The job registry and the work loop. Deliberately not a queue: one at a time.

    A reconstruction saturates the machine, so running two at once makes both
    slower. Serialising them behind a lock is both simpler and faster than any
    scheduling would be.
    """

    def __init__(self, root: Path, *, max_points: int):
        self.root = root
        self.max_points = max_points
        self.showcase = False
        self._thumb_lock = threading.Lock()
        self.gallery_root = Path(os.environ.get("LOCAISH_GALLERY") or root.parent / "showcase")
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._agent = None
        self._agent_lock = threading.Lock()
        self._stage = None
        # The product layer: projects, location details, share links, leads.
        from .product.store import Gate, Store

        self.store = Store(root)
        self.gate = Gate(os.environ.get("LOCAISH_PASSCODE"))
        self._reports: dict[str, object] = {}
        self.restored = self._restore()

    def _restore(self) -> int:
        """Re-register finished jobs from disk, so links survive a restart.

        The registry is in-memory because a running job is mostly queues and
        threads, but a *finished* job is just files -- and a viewer URL that
        someone bookmarked, or left open in a tab, must not die because the
        server was restarted. Anything with a rendered view.html in the root
        is a job that once completed; that file is the whole contract.
        """
        count = 0
        if not self.root.exists():
            return 0
        for d in sorted(self.root.iterdir()):
            viewer = d / "view.html"
            if not d.is_dir() or not viewer.exists():
                continue
            twins = sorted(d.glob("*.twin"))
            # splat.ply is the baked view layer, not the capture; naming a
            # room after it would be like crediting the projector.
            source = next(
                (p for p in sorted(d.iterdir())
                 if p.suffix.lower() in _MEDIA_SUFFIXES and p.name != "splat.ply"),
                d,
            )
            summary: dict = {"restored": True}
            if twins:
                try:
                    import zipfile

                    with zipfile.ZipFile(twins[-1]) as zf:
                        man = json.loads(zf.read("manifest.json"))
                    qa = man.get("qa") or {}
                    st = man.get("structure") or {}
                    metrics = qa.get("metrics") or {}
                    summary["verdict"] = qa.get("verdict", "?")
                    summary["points"] = int(metrics.get("point_count") or 0)
                    fa = metrics.get("floor_area_m2")
                    summary["floor_area_m2"] = None if fa is None or fa != fa else round(float(fa), 2)
                    cz, fz = st.get("ceiling_z"), st.get("floor_z")
                    summary["ceiling_height_m"] = (
                        None if cz is None or fz is None else round(float(cz) - float(fz), 3)
                    )
                    summary["openings"] = len(st.get("openings") or [])
                    summary["checks"] = {
                        "fail": [c["name"] for c in qa.get("checks", []) if c.get("status") == "fail"],
                        "warn": [c["name"] for c in qa.get("checks", []) if c.get("status") == "warn"],
                    }
                except Exception:
                    pass
            scout = d / "scout.txt"
            # A deployed gallery bakes the twin but not the multi-hundred-MB
            # source video, so fall back to the twin's own name before the
            # opaque job id.
            job = Job(
                id=d.name,
                name=(source.name if source != d
                      else twins[-1].stem if twins else d.name),
                source=source,
                workdir=d,
                state="done",
                twin_path=twins[-1] if twins else None,
                viewer_path=viewer,
                scout_path=scout if scout.exists() else None,
                summary=summary,
            )
            self.jobs[d.name] = job
            self._restore_plans(job)
            count += 1
        return count

    def _restore_plans(self, job: Job) -> None:
        """Finished coverage plans are directories with a plan.json; keep them."""
        pdir = job.workdir / "plans"
        if not pdir.is_dir():
            return
        for d in sorted(pdir.iterdir()):
            pj = d / "plan.json"
            if not d.is_dir() or not pj.exists():
                continue
            try:
                plan = json.loads(pj.read_text())
            except Exception:
                continue
            run = PlanRun(
                id=d.name, job_id=job.id, title=plan.get("title") or "",
                brief=plan.get("brief") or "", mode=plan.get("planner") or "",
                workdir=d, state="done", plan_dict=plan,
            )
            job.plans[d.name] = run

    def refresh_viewer(self, job: Job) -> None:
        """Re-render a restored job's viewer when the template has moved on.

        A viewer page is the template with the twin inlined, so a job
        finished before a template change serves the old page forever
        unless someone notices. Rendering costs seconds on a big twin, so it
        happens once, on the first request after the template changed.
        """
        from .viewer import build as buildmod

        try:
            template_mtime = Path(buildmod._TEMPLATE_PATH).stat().st_mtime
            if job.viewer_path.stat().st_mtime >= template_mtime or not job.twin_path:
                return
        except OSError:
            return
        with self._lock:
            try:
                if job.viewer_path.stat().st_mtime >= template_mtime:
                    return
                buildmod.render_html(self.twin_of(job), job.viewer_path, max_points=self.max_points)
            except Exception:  # noqa: BLE001 - serve the old page rather than nothing
                pass

    # -- the twin and its setups, lazily ---------------------------------
    def twin_of(self, job: Job):
        if job.twin_obj is None:
            if not job.twin_path or not job.twin_path.exists():
                raise FileNotFoundError("no twin on this job")
            from .types import Twin

            job.twin_obj = Twin.load(job.twin_path)
            self.apply_details(job)
        return job.twin_obj

    def apply_details(self, job: Job) -> None:
        """The scout's compass heading and address override the capture's.

        A twin georeferenced from the browser knows where it is but not
        which way it faces; the heading the scout types in (checked against
        a compass on the window) is what makes the sun schedule a
        measurement. Applied to the loaded twin only -- the .twin file on
        disk stays the record of what the capture itself established.
        """
        if job.twin_obj is None:
            return
        from .types import Georeference

        d = self.store.details(job.workdir)
        twin = job.twin_obj
        geo = twin.georeference
        lat = d.get("latitude", geo.latitude if geo else None)
        lon = d.get("longitude", geo.longitude if geo else None)
        if lat is None or lon is None:
            return
        heading = d.get("heading_deg")
        if heading is None and geo is not None:
            heading, source = geo.heading_deg, geo.heading_source
        elif heading is None:
            heading, source = 0.0, "assumed"
        else:
            source = "user"
        twin.georeference = Georeference(latitude=float(lat), longitude=float(lon),
                                         heading_deg=float(heading), heading_source=source,
                                         elevation_m=geo.elevation_m if geo else 0.0)
        self._reports.pop(job.id, None)

    def report_of(self, job: Job):
        """The scout report, surveyed once per loaded twin."""
        if job.id not in self._reports:
            from .film import report as reportmod

            self._reports[job.id] = reportmod.build(self.twin_of(job))
        return self._reports[job.id]

    def plans_of(self, job: Job) -> list[dict]:
        return [r.plan_dict for r in job.plans.values() if r.state == "done" and r.plan_dict]

    def display_name(self, job: Job) -> str:
        d = self.store.details(job.workdir)
        if d.get("display_name"):
            return str(d["display_name"])
        return Path(job.name).stem.replace("_", " ")

    def room_listing(self) -> list[dict]:
        """Every finished room, as the project page and the compare need it."""
        out = []
        for j in self.jobs.values():
            if j.state != "done":
                continue
            s = j.summary or {}
            meta = []
            if s.get("floor_area_m2") is not None:
                meta.append(f"{s['floor_area_m2']} m²")
            if s.get("openings"):
                meta.append(f"{s['openings']} window{'s' if s['openings'] != 1 else ''}")
            out.append({"id": j.id, "name": self.display_name(j), "meta": " · ".join(meta)})
        return out

    # -- what is in the room ---------------------------------------------
    def frames_of(self, job: Job):
        """The job's posed frames, loaded once; None for scan-file twins."""
        if not hasattr(job, "_frames"):
            from .product import labels as labelsmod

            try:
                job._frames = labelsmod.load_frames(job.workdir, self.twin_of(job))
            except Exception:  # noqa: BLE001
                job._frames = None
        return job._frames

    def tree_of(self, job: Job):
        if not hasattr(job, "_tree"):
            from .product import labels as labelsmod

            job._tree = labelsmod.surface_tree(self.twin_of(job))
        return job._tree

    def labels_of(self, job: Job) -> list[dict]:
        from .product import labels as labelsmod

        return labelsmod.load_labels(job.workdir)

    def detect_labels(self, job: Job, *, max_frames: int | None = None, progress=None) -> dict:
        from .product import labels as labelsmod

        twin = self.twin_of(job)
        frames = self.frames_of(job)
        if frames is None:
            raise ValueError("this twin has no posed frames to look at (scan-file twins carry no pictures)")
        return labelsmod.detect(job.workdir, twin, frames=frames, tree=self.tree_of(job),
                                max_frames=max_frames or labelsmod.DETECT_FRAMES, progress=progress)

    def build_package(self, job: Job, *, shoot_date: str | None, hours=None, image_base: str = "",
                      viewer_url: str = "", package_url: str = "") -> dict:
        from .product import package as pkgmod
        from .product import sunlight

        twin = self.twin_of(job)
        pkg = pkgmod.build(
            twin, job_id=job.id, name=self.display_name(job),
            details=self.store.details(job.workdir), shoot_date=shoot_date,
            hours=tuple(hours) if hours else sunlight.DEFAULT_HOURS,
            plans=self.plans_of(job), report=self.report_of(job),
            viewer_url=viewer_url, package_url=package_url,
            labels=self.labels_of(job),
        )
        # A caveat the sound section already states does not need saying twice.
        seen = set()
        pkg["caveats"] = [c for c in pkg["caveats"] if not (c in seen or seen.add(c))]
        return pkg

    def compare(self, project: dict) -> dict:
        """Rank the project's rooms against its brief. Synchronous: a room
        costs a second or two once its sweep is cached."""
        import time as _time

        from .product import fit as fitmod

        t0 = _time.time()
        shots = fitmod.shots_from_brief(project.get("brief") or "")
        date = project.get("shoot_date") or None
        rows = []
        for jid in project.get("locations") or []:
            job = self.jobs.get(jid)
            if not job or job.state != "done":
                continue
            try:
                twin = self.twin_of(job)
                row = fitmod.fit_location(twin, self.setups_for(job), shots, shoot_date=date,
                                          report=self.report_of(job))
            except Exception as exc:  # noqa: BLE001 - one bad room must not sink the table
                row = {"location": jid, "shots": [], "held": 0, "relaxed": 0, "missed": len(shots),
                       "score": 0.0, "total": len(shots), "warnings": [f"{type(exc).__name__}: {exc}"]}
            row["job_id"] = jid
            row["display_name"] = self.display_name(job)
            row["summary"] = fitmod.summary_line(row)
            rows.append(row)
        rows = fitmod.rank(rows)
        return {
            "shots": [{"number": s.number, "size": s.size, "subject": s.subject,
                       "description": s.description} for s in shots],
            "rows": rows,
            "shoot_date": date,
            "brief": project.get("brief") or "",
            "locations": list(project.get("locations") or []),
            "seconds": round(_time.time() - t0, 1),
            "ran_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        }

    def sweep_of(self, job: Job):
        if job.sweep is None:
            from .film import sweep as sweepmod

            job.sweep = sweepmod.sweep(self.twin_of(job))
        return job.sweep

    def floor_z_of(self, job: Job) -> float:
        return float(self.twin_of(job).structure.floor_z)

    def is_approved(self, job: Job) -> bool:
        """Approved = linked into the gallery root the showcase serves."""
        p = self.gallery_root / job.id
        return p.is_symlink() or p.exists()

    def set_approved(self, job: Job, flag: bool) -> None:
        p = self.gallery_root / job.id
        if flag:
            self.gallery_root.mkdir(parents=True, exist_ok=True)
            if not (p.is_symlink() or p.exists()):
                p.symlink_to(Path(os.path.relpath(job.workdir.resolve(), self.gallery_root.resolve())))
        elif p.is_symlink() or p.exists():
            p.unlink()

    _SIZE_RANK = {"els": 0, "ls": 1, "mls": 2, "ms": 3, "mcu": 4, "cu": 5, "bcu": 6, "ecu": 7}

    def render_thumb(self, job: Job, out: Path) -> None:
        """A still of the room for the location library.

        The widest frame any finished plan already rendered is free; failing
        that, render the best clear wide the sweep found. Cached on disk
        beside the job, so each room pays once."""
        import shutil

        with self._thumb_lock:
            if out.exists():
                return
            try:
                import numpy as np

                from .film.render import render_shot

                twin = self.twin_of(job)
                st = twin.structure
                # The twin's origin is the footprint centroid; stand in the
                # farthest corner, pull in a step, and look across the room.
                if st.footprint is not None and len(st.footprint) >= 3:
                    corners = np.asarray(st.footprint, dtype=np.float64)
                else:
                    b = twin.bounds
                    corners = np.array([[b[0, 0], b[0, 1]], [b[0, 0], b[1, 1]],
                                        [b[1, 0], b[0, 1]], [b[1, 0], b[1, 1]]])
                # The longest sightline in a room runs corner to corner
                # (docs/CINEMATOGRAPHY.md #7) -- shoot down it.
                d = ((corners[:, None, :] - corners[None, :, :]) ** 2).sum(axis=2)
                i, j = np.unravel_index(int(np.argmax(d)), d.shape)
                a, b2 = corners[i], corners[j]
                cam_xy = a + 0.12 * (b2 - a)
                subj_xy = a + 0.72 * (b2 - a)
                cam = (float(cam_xy[0]), float(cam_xy[1]), float(st.floor_z) + 1.45)
                render_shot(twin, cam, (float(subj_xy[0]), float(subj_xy[1])), 20.0,
                            width_px=720, out=out, draw_subject=False)
                return
            except Exception:  # noqa: BLE001 - fall back to a frame a plan rendered
                pass
            best = None
            for run in job.plans.values():
                for ps in (run.plan_dict or {}).get("shots") or []:
                    st, fr = ps.get("setup"), ps.get("frame")
                    if not st or not fr:
                        continue
                    fp = run.workdir / "frames" / fr
                    if not fp.exists():
                        continue
                    r = self._SIZE_RANK.get(st.get("shot_size"), 9)
                    if best is None or r < best[0]:
                        best = (r, fp)
            if best is None:
                raise ValueError("nothing to render a thumbnail from")
            shutil.copyfile(best[1], out)

    def ensure_location(self, job: Job) -> str | None:
        """A restored job never ran _run's agent registration; do it on demand.

        The twin is on disk and the chat is the first thing that needs it, so
        the first question to a restored room pays the twin load once and
        registers the location with the agent's tool surface."""
        if job.location:
            return job.location
        try:
            twin = self.twin_of(job)
        except Exception:  # noqa: BLE001 - no twin, no location
            return None
        try:
            from .agent import register_location

            register_location(twin.name, twin, job.workdir)
        except Exception:  # noqa: BLE001 - agent().ask reports its own absence
            pass
        job.location = twin.name
        return job.location

    def setups_for(self, job: Job):
        """Where this job's candidate setups come from: the warehouse when it
        holds the location, the in-memory sweep otherwise."""
        from . import warehouse
        from .film import coverage as cov

        twin = self.twin_of(job)
        if warehouse.configured():
            try:
                counts = warehouse.location_counts()
                if not counts.get(twin.name):
                    warehouse.load_sweep(self.sweep_of(job))
                return cov.ClickHouseSetups()
            except Exception:  # noqa: BLE001 - degrade to the local sweep
                pass
        return cov.LocalSetups(self.sweep_of(job))

    # -- coverage --------------------------------------------------------
    def start_plan(self, job: Job, brief: str, title: str, mode: str) -> PlanRun:
        from .film import coverage as cov

        pid = cov.new_plan_id()
        run = PlanRun(id=pid, job_id=job.id, title=title, brief=brief, mode=mode,
                      workdir=job.workdir / "plans" / pid)
        job.plans[pid] = run
        threading.Thread(target=self._run_plan, args=(job, run), daemon=True).start()
        return run

    def _run_plan(self, job: Job, run: PlanRun) -> None:
        from . import warehouse
        from .film import coverage as cov

        run.state = "running"
        try:
            run.emit("stage", "loading the twin")
            twin = self.twin_of(job)
            run.floor_z = float(twin.structure.floor_z)
            source = self.setups_for(job)
            run.emit("stage", f"setups from {'ClickHouse' if source.kind == 'clickhouse' else 'the local sweep'}")
            run.workdir.mkdir(parents=True, exist_ok=True)

            if run.mode == "agent":
                from .agent.coverage import plan_coverage

                plan = plan_coverage(
                    twin, workdir=run.workdir, source=source, brief=run.brief,
                    title=run.title, on_event=run.events.put,
                )
            else:
                shots = cov.parse_shot_lines(run.brief)
                if not shots:
                    raise ValueError("no shots found in the brief -- one shot per line")
                run.emit("stage", f"{len(shots)} shots read from the list")
                plan = cov.plan(
                    twin, shots, source, title=run.title, brief=run.brief,
                    workdir=run.workdir,
                    progress=lambda m: run.emit("stage", m),
                    on_event=run.events.put,
                )
            # The plan's own id is the run's: links on the page, rows in the
            # warehouse and the directory on disk must all say the same thing.
            plan.plan_id = run.id
            plan.save(run.workdir)
            for w in plan.warnings:
                run.emit("note", w)
            if warehouse.configured():
                try:
                    n = warehouse.load_plan(plan)
                    run.emit("note", f"{n} planned shots written to ClickHouse shot_plans")
                except Exception as exc:  # noqa: BLE001 - the plan stands without it
                    run.emit("note", f"could not write the plan to ClickHouse: {exc}")
            run.plan_dict = plan.to_dict()
            run.state = "done"
            run.events.put({"kind": "done", "plan": run.plan_dict, "floor_z": run.floor_z})
        except Exception as exc:  # noqa: BLE001 - the browser has to see why
            run.state = "failed"
            run.error = f"{type(exc).__name__}: {exc}"
            run.events.put({"kind": "error", "text": run.error,
                            "detail": traceback.format_exc()[-2000:]})
        finally:
            run.events.put(None)

    @property
    def stage(self):
        """Cast, set pieces and rehearsals: locaish/product/stage.py."""
        if self._stage is None:
            from .product.stage import Stage

            self._stage = Stage(self)
        return self._stage

    def agent(self):
        """The scout agent, built on first use so a scan-only session never pays for it."""
        with self._agent_lock:
            if self._agent is None:
                from .agent import AgentService

                self._agent = AgentService()
            return self._agent

    def create(self, name: str) -> Job:
        jid = uuid.uuid4().hex[:12]
        workdir = self.root / jid
        workdir.mkdir(parents=True, exist_ok=True)
        job = Job(id=jid, name=name, source=workdir / _safe_name(name), workdir=workdir)
        self.jobs[jid] = job
        return job

    def start(self, job: Job) -> None:
        threading.Thread(target=self._run, args=(job,), daemon=True).start()

    def _run(self, job: Job) -> None:
        from .scan.ingest import IngestOptions, ingest
        from .viewer.build import render_html

        # One reconstruction at a time. Saying so is the point of the note:
        # without it a capture dropped while another is running sits behind
        # the lock emitting nothing, which looks like a dead page.
        if not self._lock.acquire(blocking=False):
            job.emit("note", "queued behind the reconstruction already running")
            self._lock.acquire()
        try:
            job.state = "running"
            job.emit("stage", "starting")
            try:
                if job.source.suffix.lower() == ".twin":
                    # A finished twin needs no reconstruction: load it and go
                    # straight to the parts a judge wants to see. This is how
                    # the committed example room comes alive on a fresh clone.
                    from .types import Twin

                    job.emit("stage", "reading the twin")
                    twin = Twin.load(job.source)
                    job.twin_path = job.source
                    s, qa = twin.structure, twin.qa
                    summary = {
                        "name": twin.name,
                        "verdict": qa.verdict,
                        "points": len(twin.points),
                        "floor_area_m2": None if s.floor_area is None else round(float(s.floor_area), 2),
                        "ceiling_height_m": (
                            None if s.ceiling_z is None or s.floor_z is None
                            else round(float(s.ceiling_z - s.floor_z), 3)
                        ),
                        "openings": len(s.openings),
                        "coverage_agent": _coverage_agent_configured(),
                        "checks": {
                            "fail": [c["name"] for c in qa.checks if c.get("status") == "fail"],
                            "warn": [c["name"] for c in qa.checks if c.get("status") == "warn"],
                        },
                    }
                else:
                    opts = IngestOptions(
                        name=Path(job.name).stem,
                        video_workdir=job.workdir / "recon",
                        max_points=self.max_points,
                        latitude=job.latitude,
                        longitude=job.longitude,
                        progress=lambda m: job.emit("stage", m),
                    )
                    result = ingest(job.source, opts)
                    twin = result.twin
                    job.twin_path = twin.save(job.workdir / f"{twin.name}.twin")
                    summary = _summarise(twin, result)
                job.emit("stage", "rendering viewer")
                # max_points must flow through: render_html's own default is a
                # conservative 900k, which would silently thin the cloud the
                # reconstruction just paid for.
                job.viewer_path = render_html(
                    twin, job.workdir / "view.html", max_points=self.max_points
                )

                # The recce, written up. Cheap next to the reconstruction, and
                # it is the thing the twin exists to produce.
                job.emit("stage", "scouting the location")
                try:
                    from .film import report as reportmod

                    built = reportmod.build(twin)
                    job.scout_path = job.workdir / "scout.txt"
                    job.scout_path.write_text(reportmod.render_text(built))
                    (job.workdir / "scout.json").write_text(
                        json.dumps(built.to_dict(), indent=2, default=str)
                    )
                except Exception as exc:  # noqa: BLE001 - a twin can be too thin to survey
                    job.emit("note", f"the location could not be surveyed: {exc}")

                job.summary = summary
                job.summary["scout"] = job.scout_path is not None
                job.location = twin.name

                # The semantic pass: what is in the room, by name, placed
                # from the capture's own frames. Advisory and provisional;
                # a person confirms or corrects it in the studio.
                try:
                    from .product import labels as labelsmod

                    if labelsmod.vision_available() and self.frames_of(job) is not None:
                        job.emit("stage", "identifying what is in the room")
                        found = self.detect_labels(job, progress=lambda m: job.emit("stage", m))
                        job.emit("note", f"{found['objects']} objects named from {found['frames_looked_at']} frames "
                                         "(provisional until someone confirms them)")
                except Exception as exc:  # noqa: BLE001 - names are a bonus, never a blocker
                    job.emit("note", f"could not name the room's contents: {exc}")

                # The sweep: every physically-possible setup, scored, and --
                # when ClickHouse is configured -- loaded where the agent can
                # search it.
                self._sweep(job, twin)

                # Register the twin with the agent's tool surface whether or
                # not ClickHouse is up: the scout report, tape measure and
                # renderer work on the twin alone.
                try:
                    from .agent import register_location

                    register_location(twin.name, twin, job.workdir)
                    job.summary["agent"] = True
                except Exception as exc:  # noqa: BLE001
                    job.emit("note", f"agent tools unavailable: {exc}")
                    job.summary["agent"] = False

                if job.source.suffix.lower() != ".twin":
                    for w in result.warnings:
                        job.emit("note", w)
                job.state = "done"
                job.emit("done", "complete", summary=job.summary)
            except Exception as exc:  # noqa: BLE001 - the browser has to see why
                job.state = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.emit("error", job.error, detail=traceback.format_exc()[-2000:])
            finally:
                job.events.put(None)
        finally:
            self._lock.release()

    def _sweep(self, job: Job, twin) -> None:
        from . import warehouse
        from .film import sweep as sweepmod

        job.emit("stage", "sweeping camera setups")
        try:
            sw = sweepmod.sweep(twin, progress=lambda m: job.emit("stage", m))
        except Exception as exc:  # noqa: BLE001 - a thin twin may not be sweepable
            job.emit("note", f"the shot sweep declined: {exc}")
            job.summary["setups"] = 0
            job.summary["clickhouse"] = False
            return
        job.sweep = sw
        job.summary["setups"] = len(sw)
        for w in sw.warnings:
            job.emit("note", w)

        if not warehouse.configured():
            job.summary["clickhouse"] = False
            job.emit(
                "note",
                "ClickHouse is not configured (set CLICKHOUSE_HOST), so the "
                "shot table stays local and the agent cannot search it",
            )
            return
        job.emit("stage", "loading ClickHouse")
        try:
            n = warehouse.load_sweep(sw, progress=lambda m: job.emit("stage", m))
            job.summary["clickhouse"] = True
            job.emit("note", f"{n:,} setups searchable in ClickHouse")
        except Exception as exc:  # noqa: BLE001
            job.summary["clickhouse"] = False
            job.emit("note", f"ClickHouse load failed: {exc}")


def _notify_lead(rec: dict) -> None:
    """Forward a pilot request to a webhook (Slack, Zapier, anything that
    takes JSON) when LOCAISH_LEAD_WEBHOOK is set. Best effort: the lead is
    already on disk."""
    url = os.environ.get("LOCAISH_LEAD_WEBHOOK")
    if not url:
        return
    try:
        import urllib.request

        text = (f"New Locaish lead: {rec.get('name')} <{rec.get('email')}> at {rec.get('company')} "
                f"({rec.get('role')}), plan {rec.get('plan')}: {rec.get('message')}")
        data = json.dumps({"text": text, "lead": rec}).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()
    except Exception:  # noqa: BLE001
        pass


def _coverage_agent_configured() -> bool:
    try:
        from .agent.coverage import agent_configured

        return bool(agent_configured())
    except Exception:  # noqa: BLE001 - the module may be absent or unconfigured
        return False


def _summarise(twin, result) -> dict:
    s = twin.structure
    qa = twin.qa
    video = (result.steps or {}).get("video") or {}
    scale = video.get("scale") or {}
    return {
        "name": twin.name,
        "verdict": qa.verdict,
        "points": len(twin.points),
        "floor_area_m2": None if s.floor_area is None else round(float(s.floor_area), 2),
        "ceiling_height_m": (
            None
            if s.ceiling_z is None or s.floor_z is None
            else round(float(s.ceiling_z - s.floor_z), 3)
        ),
        "openings": len(s.openings),
        "scale_relative_error": scale.get("relative_error"),
        "scale_confidence": scale.get("confidence"),
        "frames": (video.get("frames") or {}).get("used"),
        "seconds": round(float(result.timings.get("total", sum(result.timings.values()))), 1),
        "coverage_agent": _coverage_agent_configured(),
        "checks": {
            "fail": [c["name"] for c in qa.checks if c.get("status") == "fail"],
            "warn": [c["name"] for c in qa.checks if c.get("status") == "warn"],
        },
    }


def _safe_name(name: str) -> str:
    """A filename that cannot escape the job directory or surprise a shell."""
    stem = Path(name).name.replace("\\", "_")
    keep = "".join(c for c in stem if c.isalnum() or c in "._- ")
    return keep.strip() or "upload.mov"


# ---------------------------------------------------------------------------
# http
# ---------------------------------------------------------------------------


_STAGE_GET = ("stage.js", "stage-status", "stage-plan", "cast", "headshot", "setpieces", "rehearsals",
              "rehearsal", "rehearsal-events", "rehearsal-image", "rehearsal-finish-events",
              "sims", "sim", "sim-events", "sim-file", "sim-frame", "sim-direct-events", "sim-finish-events",
              "worlds", "world-events", "world-file")
_STAGE_POST = ("cast", "setpieces", "rehearse", "rehearsal", "sims", "sim", "world")


class _Handler(BaseHTTPRequestHandler):
    studio: Studio
    server_version = "locaish"

    def log_message(self, fmt, *args):  # quieter than the default access log
        pass

    # -- the gate ----------------------------------------------------------
    # Paths anyone may open: share links, the public pages, the assets.
    _PUBLIC = ("s", "login", "logo.svg", "favicon.svg", "pricing", "capture", "lead", "health")

    def _allowed(self, parts: list[str]) -> bool:
        gate = self.studio.gate
        if not gate.enabled:
            return True
        if parts and parts[0] in self._PUBLIC:
            return True
        return gate.valid(gate.from_cookie_header(self.headers.get("Cookie")))

    def _deny(self) -> None:
        wants_html = "text/html" in (self.headers.get("Accept") or "")
        if wants_html:
            self.send_response(302)
            self.send_header("Location", "/login?next=" + self.path)
            self.end_headers()
            return
        self._json({"error": "this studio needs its passcode"}, 401)

    # -- GET ---------------------------------------------------------------
    def do_GET(self) -> None:
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        if not self._allowed(parts):
            return self._deny()

        if not parts:
            return self._html(PAGE.replace("__SHOWCASE__", "1" if self.studio.showcase else "0"))
        if parts[0] in ("logo.svg", "favicon.svg"):
            return self._svg(LOGO_SVG)
        if parts[0] == "health":
            return self._json({"ok": True, "jobs": len(self.studio.jobs)})
        if parts[0] in ("projects", "projects.json", "project", "package", "floorplan", "sun",
                        "details", "s", "pricing", "capture", "login", "share", "leads",
                        "labels", "label-views", "label-crop"):
            return self._product_get(parts, parse_qs(url.query))
        if parts[0] == "thumb" and len(parts) == 2:
            return self._thumb(parts[1])
        if parts[0] in _STAGE_GET:
            return self._stage_get(parts, parse_qs(url.query))
        if parts[0] == "events" and len(parts) == 2:
            return self._events(parts[1])
        if parts[0] == "view" and len(parts) == 2:
            job = self.studio.jobs.get(parts[1])
            if not job or not job.viewer_path or not job.viewer_path.exists():
                return self._json({"error": "no viewer for that job"}, 404)
            return self._serve_viewer(job)
        self._route_rest(parts, url)

    def _serve_viewer(self, job: Job) -> None:
        self.studio.refresh_viewer(job)
        if job.viewer_path is not None:
            # A deployed gallery bakes a pre-gzipped viewer beside the page:
            # hosts like Cloud Run cap responses (32 MiB) and the twin inlined
            # as HTML is bigger than that raw. Serve the sibling only while it
            # is at least as fresh as the page it compresses.
            gz = job.viewer_path.with_suffix(".html.gz")
            try:
                if (gz.exists()
                        and gz.stat().st_mtime >= job.viewer_path.stat().st_mtime
                        and "gzip" in (self.headers.get("Accept-Encoding") or "")):
                    data = gz.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Encoding", "gzip")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
            except OSError:
                pass
            return self._file(job.viewer_path, "text/html; charset=utf-8")

    def _route_rest(self, parts: list[str], url) -> None:
        if parts[0] == "scout" and len(parts) == 2:
            job = self.studio.jobs.get(parts[1])
            if not job or not job.scout_path or not job.scout_path.exists():
                return self._json({"error": "no scout report for that job"}, 404)
            return self._file(job.scout_path, "text/plain; charset=utf-8")
        if parts[0] == "twin" and len(parts) == 2:
            job = self.studio.jobs.get(parts[1])
            if not job or not job.twin_path or not job.twin_path.exists():
                return self._json({"error": "no twin for that job"}, 404)
            return self._file(
                job.twin_path, "application/octet-stream", download=job.twin_path.name
            )
        if parts[0] == "shot-image" and len(parts) == 3:
            job = self.studio.jobs.get(parts[1])
            if not job:
                return self._json({"error": "unknown job"}, 404)
            img = job.workdir / "shots" / Path(parts[2]).name
            if not img.exists():
                return self._json({"error": "no such frame"}, 404)
            return self._file(img, "image/png")
        if parts[0] == "jobs":
            return self._json(
                {
                    "jobs": [
                        {"id": j.id, "name": j.name, "state": j.state,
                         "approved": self.studio.is_approved(j),
                         "summary": self._job_summary(j)}
                        for j in self.studio.jobs.values()
                    ]
                }
            )
        if parts[0] == "plan-events" and len(parts) == 3:
            return self._plan_events(parts[1], parts[2])
        if parts[0] == "plans" and len(parts) == 2:
            job = self.studio.jobs.get(parts[1])
            if not job:
                return self._json({"error": "unknown job"}, 404)
            runs = [r.listing() for r in job.plans.values() if r.state == "done"]
            runs.sort(key=lambda r: r.get("created_at") or "")
            return self._json({"plans": runs})
        if parts[0] == "plan" and len(parts) in (3, 4):
            job = self.studio.jobs.get(parts[1])
            run = job.plans.get(parts[2]) if job else None
            if not run:
                return self._json({"error": "no such plan"}, 404)
            if len(parts) == 3:
                if run.state != "done" or run.plan_dict is None:
                    return self._json({"error": f"plan is {run.state}"}, 409)
                return self._json({"plan": run.plan_dict, "floor_z": self._floor_z(job)})
            name = parts[3]
            if name == "shotlist.txt":
                f = run.workdir / "shotlist.txt"
                if not f.exists():
                    return self._json({"error": "no shot list"}, 404)
                return self._file(f, "text/plain; charset=utf-8")
            if name == "floorplan.svg":
                f = run.workdir / "floorplan.svg"
                if not f.exists():
                    return self._json({"error": "no camera plan"}, 404)
                return self._file(f, "image/svg+xml")
            if name == "plan.json":
                f = run.workdir / "plan.json"
                if not f.exists():
                    return self._json({"error": "no plan"}, 404)
                return self._file(f, "application/json", download=f"coverage-{run.id}.json")
            return self._json({"error": "not found"}, 404)
        if parts[0] == "plan-image" and len(parts) == 4:
            job = self.studio.jobs.get(parts[1])
            run = job.plans.get(parts[2]) if job else None
            if not run:
                return self._json({"error": "no such plan"}, 404)
            img = run.workdir / "frames" / Path(parts[3]).name
            if not img.exists():
                return self._json({"error": "no such frame"}, 404)
            return self._file(img, "image/png")
        if parts[0] == "capacity" and len(parts) == 2:
            return self._capacity(parts[1])
        if parts[0] == "setup-near" and len(parts) == 2:
            return self._setup_near(parts[1], parse_qs(url.query))
        if parts[0] == "twin-info" and len(parts) == 2:
            return self._twin_info(parts[1])
        self._json({"error": "not found"}, 404)

    def _job_summary(self, job: Job) -> dict:
        s = dict(job.summary)
        if "coverage_agent" not in s:
            s["coverage_agent"] = _coverage_agent_configured()
        if "clickhouse" not in s and job.state == "done":
            # A restored job never loaded anything this process; report
            # whether the warehouse holds its location right now.
            s["clickhouse"] = self._clickhouse_has(job)
        return s

    def _clickhouse_has(self, job: Job) -> bool:
        from . import warehouse

        if not warehouse.configured():
            return False
        try:
            name = job.location or self.studio.twin_of(job).name
            return bool(warehouse.location_counts().get(name))
        except Exception:  # noqa: BLE001
            return False

    def _floor_z(self, job: Job) -> float:
        try:
            return self.studio.floor_z_of(job)
        except Exception:  # noqa: BLE001
            return 0.0

    def _capacity(self, job_id: str) -> None:
        from . import warehouse

        job = self.studio.jobs.get(job_id)
        if not job:
            return self._json({"error": "unknown job"}, 404)
        if not warehouse.configured():
            return self._json({"available": False, "reason": "ClickHouse is not configured"})
        try:
            name = job.location or self.studio.twin_of(job).name
            out = warehouse.capacity(name)
            out["available"] = True
            return self._json(out)
        except Exception as exc:  # noqa: BLE001
            return self._json({"available": False, "reason": f"{type(exc).__name__}: {exc}"})

    def _setup_near(self, job_id: str, params: dict) -> None:
        from . import warehouse

        job = self.studio.jobs.get(job_id)
        if not job:
            return self._json({"error": "unknown job"}, 404)
        try:
            x = float(params["x"][0]); y = float(params["y"][0]); z = float(params["z"][0])
            focal = float(params["focal"][0])
        except (KeyError, ValueError, IndexError):
            return self._json({"error": "need x, y, z (height above floor) and focal"}, 400)
        try:
            name = job.location or self.studio.twin_of(job).name
            if warehouse.configured() and self._clickhouse_has(job):
                row = warehouse.nearest_setup(name, x, y, z, focal)
                return self._json({"setup": row, "source": "clickhouse"})
            sw = self.studio.sweep_of(job)
            c = sw.columns
            import numpy as np

            sel = np.flatnonzero(np.abs(c["focal_mm"] - focal) < 0.01)
            if not len(sel):
                return self._json({"setup": None, "source": "local"})
            d = np.sqrt((c["cam_x"][sel] - x) ** 2 + (c["cam_y"][sel] - y) ** 2 + (c["cam_z"][sel] - z) ** 2)
            i = int(sel[int(np.argmin(d))])
            row = {k: (v[i].item() if hasattr(v[i], "item") else v[i]) for k, v in c.items()}
            row["away_m"] = round(float(d.min()), 3)
            return self._json({"setup": row, "source": "local"})
        except Exception as exc:  # noqa: BLE001
            return self._json({"setup": None, "error": f"{type(exc).__name__}: {exc}"})

    def _twin_info(self, job_id: str) -> None:
        from .film import coverage as cov

        job = self.studio.jobs.get(job_id)
        if not job:
            return self._json({"error": "unknown job"}, 404)
        try:
            twin = self.studio.twin_of(job)
            source = self.studio.setups_for(job)
            marks = cov.describe_marks(twin, source.marks(twin.name))
            return self._json({"floor_z": float(twin.structure.floor_z), "name": twin.name,
                               "marks": marks, "setups": source.kind})
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def _thumb(self, job_id: str) -> None:
        job = self.studio.jobs.get(job_id)
        if not job:
            return self._json({"error": "unknown job"}, 404)
        out = job.workdir / "thumb.png"
        try:
            if not out.exists():
                self.studio.render_thumb(job, out)
            return self._file(out, "image/png")
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    # -- POST --------------------------------------------------------------
    def do_POST(self) -> None:
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        if not self._allowed(parts):
            return self._deny()
        if parts and parts[0] in ("projects", "project", "details", "share", "lead", "login", "labels"):
            return self._product_post(parts, parse_qs(url.query))
        if parts and parts[0] == "chat" and len(parts) == 2:
            return self._chat(parts[1])
        if parts and parts[0] in _STAGE_POST:
            return self._stage_post(parts)
        if parts and parts[0] == "plan" and len(parts) == 2:
            return self._plan(parts[1])
        if parts and parts[0] == "approve" and len(parts) == 2:
            if self.studio.showcase:
                return self._json({"error": "the gallery is read-only"}, 403)
            job = self.studio.jobs.get(parts[1])
            if not job or job.state != "done":
                return self._json({"error": "unknown or unfinished job"}, 404)
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                flag = bool(body.get("approved"))
            except Exception:  # noqa: BLE001
                return self._json({"error": "bad body"}, 400)
            try:
                self.studio.set_approved(job, flag)
            except OSError as exc:
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            return self._json({"approved": flag})
        if url.path == "/upload" and self.studio.showcase:
            return self._json(
                {"error": "this studio is a gallery of scanned locations; scanning runs locally"}, 403)
        if url.path != "/upload":
            return self._json({"error": "not found"}, 404)

        params = parse_qs(url.query)
        name = (params.get("name") or ["upload.mov"])[0]
        lat = lon = None
        try:
            if params.get("lat") and params.get("lon"):
                lat, lon = float(params["lat"][0]), float(params["lon"][0])
        except ValueError:
            lat = lon = None
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._json({"error": "bad Content-Length"}, 400)
        if length <= 0:
            return self._json({"error": "empty upload"}, 400)
        if length > MAX_UPLOAD_BYTES:
            return self._json({"error": "upload too large"}, 413)

        job = self.studio.create(name)
        job.latitude, job.longitude = lat, lon
        remaining = length
        try:
            with open(job.source, "wb") as fh:
                while remaining > 0:
                    block = self.rfile.read(min(CHUNK, remaining))
                    if not block:
                        break
                    fh.write(block)
                    remaining -= len(block)
        except OSError as exc:
            return self._json({"error": f"could not save upload: {exc}"}, 500)
        if remaining:
            return self._json({"error": "upload truncated"}, 400)

        self.studio.start(job)
        self._json({"id": job.id, "name": job.name, "bytes": length})

    def _chat(self, job_id: str) -> None:
        job = self.studio.jobs.get(job_id)
        if not job or job.state != "done" or not self.studio.ensure_location(job):
            return self._json({"error": "no finished twin on that job yet"}, 404)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            message = str(body.get("message") or "").strip()
        except (ValueError, json.JSONDecodeError):
            return self._json({"error": "bad request body"}, 400)
        if not message:
            return self._json({"error": "empty message"}, 400)

        from .agent import AgentUnavailable

        try:
            turn = self.studio.agent().ask(job.id, job.location, message)
        except AgentUnavailable as exc:
            return self._json({"error": str(exc)}, 503)
        except Exception as exc:  # noqa: BLE001 - the browser has to see why
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
        return self._json(
            {"reply": turn.reply, "trace": turn.trace, "seconds": turn.seconds}
        )

    def _plan(self, job_id: str) -> None:
        job = self.studio.jobs.get(job_id)
        if not job or job.state != "done" or not job.twin_path:
            return self._json({"error": "no finished twin on that job yet"}, 404)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            brief = str(body.get("brief") or "").strip()
            title = str(body.get("title") or "").strip()
            mode = str(body.get("mode") or "auto").strip().lower()
        except (ValueError, json.JSONDecodeError):
            return self._json({"error": "bad request body"}, 400)
        if not brief:
            return self._json({"error": "empty brief"}, 400)
        if mode not in ("auto", "agent", "structured"):
            return self._json({"error": "mode must be auto, agent or structured"}, 400)
        agent_ok = _coverage_agent_configured()
        if mode == "agent" and not agent_ok:
            return self._json(
                {"error": "Gemini is not configured on this studio, so the coverage "
                          "agent cannot run; use the structured list, or set up "
                          "Vertex AI credentials (or GOOGLE_API_KEY) and restart"},
                503,
            )
        if mode == "auto":
            mode = "agent" if agent_ok else "structured"
        run = self.studio.start_plan(job, brief, title, mode)
        return self._json({"plan_id": run.id, "mode": mode})

    def _plan_events(self, job_id: str, plan_id: str) -> None:
        job = self.studio.jobs.get(job_id)
        run = job.plans.get(plan_id) if job else None
        if not run:
            return self._json({"error": "no such plan"}, 404)
        self._stream(
            run.events, run.state,
            done_item=lambda: {"kind": "done", "plan": run.plan_dict, "floor_z": run.floor_z},
            error_item=lambda: {"kind": "error", "text": run.error or "failed"},
        )

    # -- helpers -----------------------------------------------------------
    def _events(self, job_id: str) -> None:
        job = self.studio.jobs.get(job_id)
        if not job:
            return self._json({"error": "unknown job"}, 404)
        self._stream(
            job.events, job.state,
            done_item=lambda: {"kind": "done", "text": "complete", "summary": job.summary},
            error_item=lambda: {"kind": "error", "text": job.error or "failed"},
        )

    def _stream(self, events: queue.Queue, state: str, *, done_item, error_item) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            # A browser that reconnects after the job finished must not hang
            # on an empty queue: the terminal event is re-sent from the job's
            # state, which outlives the queue that once carried it.
            if state in ("done", "failed"):
                item = done_item() if state == "done" else error_item()
                self.wfile.write(f"data: {json.dumps(item)}\n\n".encode())
                self.wfile.flush()
                return
            while True:
                try:
                    item = events.get(timeout=15)
                except queue.Empty:
                    # SSE comment as keepalive: a reconstruction can sit in
                    # one stage for minutes, and an idle TCP stream is what
                    # browsers and proxies silently kill. It also makes a
                    # dead client raise here instead of never.
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                if item is None:
                    break
                self.wfile.write(f"data: {json.dumps(item)}\n\n".encode())
                self.wfile.flush()
                if item.get("kind") in ("done", "error"):
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_PUT(self) -> None:
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        if not self._allowed(parts):
            return self._deny()
        if len(parts) == 3 and parts[0] == "sim":
            return self._stage_post(parts)
        self._json({"error": "not found"}, 404)

    def do_DELETE(self) -> None:
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        if not self._allowed(parts):
            return self._deny()
        store = self.studio.store
        if len(parts) == 3 and parts[0] == "sim":
            job = self._stage_job(parts[1])
            if job is None:
                return self._json({"error": "no finished twin on that job"}, 404)
            return self._json({"removed": bool(self.studio.stage.delete_sim(job, parts[2]))})
        if len(parts) == 2 and parts[0] == "share":
            return self._json({"revoked": store.revoke_share(parts[1])})
        if len(parts) == 2 and parts[0] == "project":
            return self._json({"deleted": store.delete_project(parts[1])})
        if len(parts) == 3 and parts[0] == "labels":
            from .product import labels as labelsmod

            job = self.studio.jobs.get(parts[1])
            if not job:
                return self._json({"error": "unknown job"}, 404)
            return self._json({"removed": labelsmod.remove_label(job.workdir, parts[2])})
        if len(parts) == 3 and parts[0] in ("cast", "setpieces"):
            job = self.studio.jobs.get(parts[1])
            if not job:
                return self._json({"error": "unknown job"}, 404)
            stage = self.studio.stage
            removed = stage.remove_actor(job, parts[2]) if parts[0] == "cast" else stage.remove_piece(job, parts[2])
            return self._json({"removed": bool(removed)})
        self._json({"error": "not found"}, 404)

    # -- the stage: cast, set pieces, rehearsals ------------------------------
    def _stage_job(self, job_id: str):
        job = self.studio.jobs.get(job_id)
        if not job or job.state != "done" or not job.twin_path:
            return None
        return job

    def _stage_get(self, parts: list[str], params: dict) -> None:
        head = parts[0]
        if head == "stage.js":
            return self._file(Path(__file__).parent / "viewer" / "stage.js", "application/javascript; charset=utf-8")
        if len(parts) < 2:
            return self._json({"error": "not found"}, 404)
        job = self._stage_job(parts[1])
        if job is None:
            return self._json({"error": "no finished twin on that job"}, 404)
        stage = self.studio.stage
        try:
            if head == "stage-status":
                return self._json(stage.status())
            if head == "cast" and len(parts) == 2:
                return self._json({"cast": stage.cast(job)})
            if head == "headshot" and len(parts) == 3:
                got = stage.headshot_bytes(job, parts[2])
                if got is None:
                    return self._json({"error": "no headshot"}, 404)
                data, ctype = got
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "private, max-age=60")
                self.end_headers()
                self.wfile.write(data)
                return
            if head == "setpieces" and len(parts) == 2:
                return self._json({"pieces": stage.pieces(job)})
            if head == "stage-plan":
                return self._svg(stage.plan_svg(job))
            if head == "rehearsals":
                return self._json({"rehearsals": stage.rehearsals(job)})
            if head == "rehearsal-events" and len(parts) == 3:
                run = stage.run(job, parts[2])
                if run is None:
                    return self._json({"error": "no such rehearsal"}, 404)
                return self._stream(
                    run.events, run.state,
                    done_item=lambda: {"kind": "done", "rehearsal_id": run.id, "listing": run.listing()},
                    error_item=lambda: {"kind": "error", "text": run.error or "failed"},
                )
            if head == "rehearsal-image" and len(parts) == 4:
                run = stage.run(job, parts[2])
                name = Path(parts[3]).name
                if run is None:
                    return self._json({"error": "no such rehearsal"}, 404)
                sub = "finish" if name.startswith("finish_") else "animatic"
                img = run.workdir / sub / name
                if not img.exists():
                    return self._json({"error": "no such frame"}, 404)
                return self._file(img, "image/png")
            if head == "rehearsal" and len(parts) in (3, 4):
                run = stage.run(job, parts[2])
                if run is None:
                    return self._json({"error": "no such rehearsal"}, 404)
                if len(parts) == 3:
                    if run.state != "done" or run.result is None:
                        return self._json({"error": f"rehearsal is {run.state}", "detail": run.error}, 409)
                    return self._json(run.result)
                name = parts[3]
                if name == "finished":
                    return self._json({"files": stage.finished_files(job, run)})
                files = {"callsheet.txt": ("callsheet.txt", "text/plain; charset=utf-8"),
                         "blocking.svg": ("blocking.svg", "image/svg+xml"),
                         "rehearsal.json": ("rehearsal.json", "application/json"),
                         "animatic.json": ("animatic/animatic.json", "application/json")}
                if name not in files:
                    return self._json({"error": "not found"}, 404)
                f = run.workdir / files[name][0]
                if not f.exists():
                    return self._json({"error": "not yet"}, 404)
                return self._file(f, files[name][1])
            # the camera simulation
            if head == "sims" and len(parts) == 2:
                return self._json({"sims": stage.sims(job)})
            if head == "sim" and len(parts) == 3:
                return self._json(stage.sim(job, parts[2]))
            if head == "sim" and len(parts) == 4 and parts[3] == "figures":
                t = float((params.get("t") or ["0"])[0])
                return self._json(stage.sim_figures(job, parts[2], t))
            if head == "sim-frame" and len(parts) == 3:
                t = float((params.get("t") or ["0"])[0])
                w = int((params.get("w") or params.get("width_px") or ["640"])[0])
                data = stage.sim_frame_png(job, parts[2], t, width_px=w)
                return self._bytes(data, "image/png", cache="private, max-age=30")
            if head == "sim-file" and len(parts) == 4:
                data, ctype = stage.sim_file(job, parts[2], parts[3])
                download = parts[3] if parts[3] in ("blender.py", "room.glb", "timeline.json") else None
                return self._bytes(data, ctype, download=download)
            if head == "sim-events" and len(parts) == 4:
                run = stage.sim_render(job, parts[2], parts[3])
                if run is None:
                    return self._json({"error": "no such render"}, 404)
                return self._stream(
                    run.events, run.state,
                    done_item=run.done_item,
                    error_item=lambda: {"kind": "error", "text": run.error or "failed"},
                )
            if head == "sim-direct-events" and len(parts) == 4:
                run = stage.sim_direct(job, parts[2], parts[3])
                if run is None:
                    return self._json({"error": "no such directing run"}, 404)
                return self._stream(
                    run.events, run.state,
                    done_item=run.done_item,
                    error_item=lambda: {"kind": "error", "text": run.error or "failed"},
                )
            if head in ("rehearsal-finish-events", "sim-finish-events") and len(parts) == 4:
                run = stage.finish_run(job, "rehearsal" if head.startswith("rehearsal") else "sim", parts[2], parts[3])
                if run is None:
                    return self._json({"error": "no such finishing run"}, 404)
                return self._stream(
                    run.events, run.state,
                    done_item=run.done_item,
                    error_item=lambda: {"kind": "error", "text": run.error or "failed"},
                )
            # the world: the room as it will be on the day
            if head == "worlds" and len(parts) == 2:
                return self._json({"worlds": stage.worlds(job)})
            if head == "world-events" and len(parts) == 3:
                run = stage.world(job, parts[2])
                if run is None:
                    return self._json({"error": "no such world"}, 404)
                return self._stream(
                    run.events, run.state,
                    done_item=run.done_item,
                    error_item=lambda: {"kind": "error", "text": run.error or "failed"},
                )
            if head == "world-file" and len(parts) == 4:
                data, ctype = stage.world_file(job, parts[2], parts[3])
                download = parts[3] if parts[3] == "world.twin" else None
                cache = "private, max-age=300" if parts[3].endswith(".png") else None
                return self._bytes(data, ctype, download=download, cache=cache)
        except KeyError as exc:
            return self._json({"error": str(exc).strip("'")}, 404)
        except ValueError as exc:
            return self._json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001 - the page shows the reason
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
        return self._json({"error": "not found"}, 404)

    def _stage_post(self, parts: list[str]) -> None:
        head = parts[0]
        if len(parts) < 2:
            return self._json({"error": "not found"}, 404)
        job = self._stage_job(parts[1])
        if job is None:
            return self._json({"error": "no finished twin on that job"}, 404)
        stage = self.studio.stage
        try:
            if head == "cast" and len(parts) == 4 and parts[3] == "headshot":
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0 or length > 25_000_000:
                    return self._json({"error": "send the image as the request body"}, 400)
                data = self.rfile.read(length)
                return self._json(stage.headshot(job, parts[2], data))
            body = self._body()
            if head == "cast" and len(parts) == 2:
                return self._json({"actor": stage.add_actor(job, body)})
            if head == "cast" and len(parts) == 3:
                return self._json({"actor": stage.update_actor(job, parts[2], body)})
            if head == "setpieces" and len(parts) == 3 and parts[2] == "lift":
                return self._json({"piece": stage.lift_piece(job, body.get("label_id"), xyz=body.get("xyz"), label=body.get("label"))})
            if head == "setpieces" and len(parts) == 4 and parts[3] == "move":
                return self._json(stage.move_piece(job, parts[2], body))
            if head == "setpieces" and len(parts) == 4 and parts[3] == "lift3d":
                return self._json(stage.lift_piece_3d(job, parts[2]))
            if head == "rehearse" and len(parts) == 2:
                run = stage.start_rehearsal(job, body)
                return self._json({"rehearsal_id": run.id})
            if head == "rehearsal" and len(parts) == 5 and parts[3] == "finish":
                run = stage.run(job, parts[2])
                if run is None or run.state != "done":
                    return self._json({"error": "no finished rehearsal"}, 404)
                return self._json(stage.finish(job, run, parts[4], body))
            if head == "rehearsal" and len(parts) == 4 and parts[3] == "finish-all":
                run = stage.run(job, parts[2])
                if run is None:
                    return self._json({"error": "no such rehearsal"}, 404)
                frun = stage.start_finish_all(job, run, body)
                return self._json({"run": frun.id, "events": f"/rehearsal-finish-events/{job.id}/{run.id}/{frun.id}",
                                   "error": frun.error})
            if head == "sim" and len(parts) == 4 and parts[3] == "finish":
                frun = stage.start_sim_finish(job, parts[2], body)
                return self._json({"run": frun.id, "events": f"/sim-finish-events/{job.id}/{parts[2]}/{frun.id}",
                                   "error": frun.error})
            # the camera simulation
            if head == "sims" and len(parts) == 2:
                return self._json(stage.create_sim(job, body))
            if head == "sim" and len(parts) == 3:
                return self._json(stage.save_sim(job, parts[2], body))
            if head == "sim" and len(parts) == 4 and parts[3] == "from-rehearsal":
                return self._json(stage.sim_from_rehearsal(job, parts[2], body))
            if head == "sim" and len(parts) == 4 and parts[3] == "render":
                run = stage.start_sim_render(job, parts[2], body)
                return self._json({"run": run.id, "events": f"/sim-events/{job.id}/{parts[2]}/{run.id}"})
            if head == "sim" and len(parts) == 4 and parts[3] == "direct":
                run = stage.start_direct(job, parts[2], body)
                return self._json({"run": run.id, "events": f"/sim-direct-events/{job.id}/{parts[2]}/{run.id}"})
            # the world
            if head == "world" and len(parts) == 2:
                run = stage.start_world(job, body)
                return self._json({"run": run.id, "events": f"/world-events/{job.id}/{run.id}", "error": run.error})
        except KeyError as exc:
            return self._json({"error": str(exc).strip("'")}, 404)
        except ValueError as exc:
            return self._json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
        return self._json({"error": "not found"}, 404)

    def _body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
        except ValueError:
            return {}
        ctype = self.headers.get("Content-Type") or ""
        if "json" in ctype or raw[:1] in (b"{", b"["):
            try:
                out = json.loads(raw or b"{}")
                return out if isinstance(out, dict) else {}
            except ValueError:
                return {}
        return {k: v[0] for k, v in parse_qs(raw.decode("utf-8", "replace")).items()}

    def _base_url(self) -> str:
        host = self.headers.get("X-Forwarded-Host") or self.headers.get("Host") or ""
        proto = self.headers.get("X-Forwarded-Proto") or ("https" if host and not host.startswith(("127.", "localhost")) else "http")
        return f"{proto}://{host}" if host else ""

    def _product_get(self, parts: list[str], params: dict) -> None:
        from .product import pages

        studio, store = self.studio, self.studio.store
        head = parts[0]
        if head == "pricing":
            return self._html(pages.pricing_page(
                stripe_scout=os.environ.get("LOCAISH_STRIPE_SCOUT_URL", ""),
                stripe_production=os.environ.get("LOCAISH_STRIPE_PRODUCTION_URL", ""),
                contact_email=os.environ.get("LOCAISH_CONTACT_EMAIL", "")))
        if head == "capture":
            return self._html(pages.capture_page(showcase=studio.showcase))
        if head == "login":
            return self._html(pages.login_page(next_url=(params.get("next") or ["/"])[0]))
        if head == "projects.json" or (head == "projects" and "json" in (self.headers.get("Accept") or "")):
            return self._json({"projects": store.projects()})
        if head == "projects":
            return self._html(pages.projects_page(store.projects(), studio.room_listing()))
        if head == "leads":
            return self._json({"leads": store.leads()})
        if head == "project" and len(parts) >= 2:
            pid = parts[1]
            as_json = pid.endswith(".json")
            proj = store.project(pid[:-5] if as_json else pid)
            if not proj:
                return self._json({"error": "no such project"}, 404)
            if as_json or len(parts) == 3 and parts[2] == "compare":
                return self._json(proj if as_json else (proj.get("fit") or {"rows": []}))
            return self._html(pages.project_page(proj, studio.room_listing(), share_base=self._base_url()))
        if head == "details" and len(parts) == 2:
            job = studio.jobs.get(parts[1])
            if not job:
                return self._json({"error": "unknown job"}, 404)
            d = store.details(job.workdir)
            geo = None
            try:
                g = studio.twin_of(job).georeference
                if g is not None:
                    geo = g.to_dict()
            except Exception:  # noqa: BLE001
                pass
            return self._json({"details": d, "georeference": geo, "display_name": studio.display_name(job),
                               "shares": store.shares_for(job.id)})
        if head == "share" and len(parts) == 2:
            return self._json({"shares": store.shares_for(parts[1])})
        if head == "sun" and len(parts) == 2:
            job = studio.jobs.get(parts[1])
            if not job:
                return self._json({"error": "unknown job"}, 404)
            from .film import daylight

            try:
                return self._json(daylight.sun_schedule(studio.twin_of(job), (params.get("date") or [None])[0]))
            except ValueError as exc:
                return self._json({"error": str(exc)}, 400)
        if head == "floorplan" and len(parts) == 2:
            return self._floorplan(parts[1], params)
        if head == "package" and len(parts) == 2:
            jid = parts[1]
            as_json = jid.endswith(".json")
            return self._package(jid[:-5] if as_json else jid, params, as_json=as_json)
        if head == "s" and len(parts) >= 2:
            return self._shared(parts[1:], params)
        if head == "labels" and len(parts) == 2:
            job = studio.jobs.get(parts[1])
            if not job:
                return self._json({"error": "unknown job"}, 404)
            from .meta import vision as visionmod
            from .product import labels as labelsmod

            # "gemini" is the historical key the page reads for "a vision
            # model is available"; it now answers for whichever provider is chosen.
            return self._json({"labels": studio.labels_of(job), "frames": studio.frames_of(job) is not None,
                               "gemini": labelsmod.vision_available(),
                               "vision_provider": visionmod.provider(),
                               "vision_name": visionmod.display_name()})
        if head == "label-views" and len(parts) == 2:
            return self._label_views(parts[1], params)
        if head == "label-crop" and len(parts) == 2:
            return self._label_crop(parts[1], params)
        self._json({"error": "not found"}, 404)

    def _label_views(self, job_id: str, params: dict) -> None:
        """The frames that looked at a clicked point, with the point marked."""
        job = self.studio.jobs.get(job_id)
        if not job:
            return self._json({"error": "unknown job"}, 404)
        try:
            p = [float(params[k][0]) for k in ("x", "y", "z")]
        except (KeyError, ValueError, IndexError):
            return self._json({"error": "need x, y, z in twin metres"}, 400)
        frames = self.studio.frames_of(job)
        if frames is None:
            return self._json({"views": [], "reason": "this twin has no posed frames"})
        try:
            views = frames.best_views(p, self.studio.tree_of(job), k=4)
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
        for v in views:
            v["url"] = f"/label-crop/{job.id}?i={v['index']}&u={v['u']:.1f}&v={v['v']:.1f}"
        return self._json({"views": views, "point": p})

    def _label_crop(self, job_id: str, params: dict) -> None:
        job = self.studio.jobs.get(job_id)
        frames = self.studio.frames_of(job) if job else None
        if not job or frames is None:
            return self._json({"error": "no frames"}, 404)
        try:
            i = int(params["i"][0]); u = float(params["u"][0]); v = float(params["v"][0])
            if not 0 <= i < len(frames):
                raise ValueError
        except (KeyError, ValueError, IndexError):
            return self._json({"error": "need i, u, v"}, 400)
        data = frames.crop(i, u, v)
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "private, max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    def _floorplan(self, job_id: str, params: dict, *, token: str | None = None) -> None:
        from .film import coverage as cov
        from .product import sunlight

        job = self.studio.jobs.get(job_id)
        if not job:
            return self._json({"error": "unknown job"}, 404)
        try:
            twin = self.studio.twin_of(job)
            date = (params.get("date") or [None])[0]
            hour = params.get("hour")
            overlay = None
            caption = ""
            if hour and twin.georeference is not None:
                h = float(hour[0])
                from datetime import date as _date

                patches = sunlight.sun_patches(twin, date or _date.today().isoformat(), h)
                overlay = lambda X, Y, p=patches: sunlight.patches_svg(twin, p, X, Y)  # noqa: E731
                caption = f"{int(h):02d}:{int(round((h % 1) * 60)):02d} solar · sun {patches['compass']} {patches['elevation_deg']:.0f}° up"
            return self._svg(cov.floor_plan_svg(twin, None, overlay=overlay, caption=caption))
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def _package(self, job_id: str, params: dict, *, as_json: bool = False, token: str | None = None) -> None:
        from .product import package as pkgmod

        job = self.studio.jobs.get(job_id)
        if not job or job.state != "done" or not job.twin_path:
            return self._json({"error": "no finished twin on that job"}, 404)
        date = (params.get("date") or [None])[0]
        hours = None
        if params.get("hours"):
            try:
                hours = [float(x) for x in params["hours"][0].split(",") if x.strip()]
            except ValueError:
                hours = None
        base = f"/s/{token}" if token else ""
        try:
            pkg = self.studio.build_package(
                job, shoot_date=date, hours=hours, image_base=base,
                viewer_url=f"{base}/view/{job.id}", package_url=f"{base}/package/{job.id}")
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
        if as_json:
            return self._json(pkg)
        return self._html(pkgmod.render_html(pkg, image_base=base))

    def _shared(self, parts: list[str], params: dict) -> None:
        """Read-only access under a share token. The token decides which job
        or project is visible; everything else is a 404, not a 403, so a
        guessed URL learns nothing."""
        from .product import pages

        store = self.studio.store
        token = parts[0]
        rec = store.resolve_share(token, count=len(parts) == 1)
        if not rec:
            return self._html(pages.shell("Link not found", "<h1>This link is no longer available</h1><p class='lead'>Ask whoever sent it for a fresh one.</p>"), 404)
        kind, target = rec["kind"], rec["target"]
        allowed_jobs: set[str] = set()
        if kind == "project":
            proj = store.project(target)
            if not proj:
                return self._json({"error": "not found"}, 404)
            allowed_jobs = set(proj.get("locations") or [])
        else:
            allowed_jobs = {target}
        if len(parts) == 1:
            if kind == "project":
                rooms = {j: {"name": self.studio.display_name(self.studio.jobs[j])}
                         for j in allowed_jobs if j in self.studio.jobs}
                return self._html(pages.shared_project_page(proj, proj.get("fit"), rooms, token))
            if kind == "package":
                return self._package(target, params, token=token)
            return self._shared_view(target)
        sub = parts[1]
        jid = parts[2] if len(parts) > 2 else None
        if jid not in allowed_jobs:
            return self._json({"error": "not found"}, 404)
        if sub == "package":
            return self._package(jid, params, token=token)
        if sub == "view":
            return self._shared_view(jid)
        if sub == "thumb":
            return self._thumb(jid)
        if sub == "labels":
            return self._json({"labels": self.studio.labels_of(self.studio.jobs[jid])})
        if sub == "floorplan":
            return self._floorplan(jid, params, token=token)
        if sub == "plan-image" and len(parts) == 5:
            job = self.studio.jobs.get(jid)
            run = job.plans.get(parts[3]) if job else None
            img = run.workdir / "frames" / Path(parts[4]).name if run else None
            if not img or not img.exists():
                return self._json({"error": "not found"}, 404)
            return self._file(img, "image/png")
        self._json({"error": "not found"}, 404)

    def _shared_view(self, job_id: str) -> None:
        job = self.studio.jobs.get(job_id)
        if not job or not job.viewer_path or not job.viewer_path.exists():
            return self._json({"error": "not found"}, 404)
        # The ordinary viewer route; the gate was satisfied by the token.
        return self._serve_viewer(job)

    def _product_post(self, parts: list[str], params: dict) -> None:
        studio, store = self.studio, self.studio.store
        head = parts[0]
        body = self._body()
        if head == "login":
            gate = studio.gate
            nxt = str(body.get("next") or "/")
            if not nxt.startswith("/") or nxt.startswith("//"):
                nxt = "/"
            if gate.check(str(body.get("passcode") or "")):
                self.send_response(302)
                self.send_header("Set-Cookie", f"{gate.COOKIE}={gate.token()}; Path=/; Max-Age={gate.TTL_S}; HttpOnly; SameSite=Lax")
                self.send_header("Location", nxt)
                self.end_headers()
                return
            from .product import pages

            return self._html(pages.login_page(error="That passcode is not right.", next_url=nxt))
        if head == "lead":
            if not str(body.get("email") or "").strip():
                return self._json({"error": "an email address is needed"}, 400)
            rec = store.add_lead(body)
            _notify_lead(rec)
            return self._json({"ok": True, "id": rec["id"]})
        if studio.showcase and head != "share":
            return self._json({"error": "the gallery is read-only"}, 403)
        if head == "projects":
            proj = store.create_project(str(body.get("title") or ""), client=str(body.get("client") or ""),
                                        brief=str(body.get("brief") or ""), shoot_date=str(body.get("shoot_date") or ""),
                                        locations=[str(x) for x in (body.get("locations") or []) if str(x) in studio.jobs])
            return self._json(proj)
        if head == "project" and len(parts) == 2:
            proj = store.update_project(parts[1], body)
            if not proj:
                return self._json({"error": "no such project"}, 404)
            return self._json(proj)
        if head == "project" and len(parts) == 3 and parts[2] == "compare":
            proj = store.project(parts[1])
            if not proj:
                return self._json({"error": "no such project"}, 404)
            if not proj.get("locations"):
                return self._json({"error": "add at least one room to compare"}, 400)
            try:
                fit = studio.compare(proj)
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            store.update_project(parts[1], {"fit": fit})
            return self._json(fit)
        if head == "details" and len(parts) == 2:
            job = studio.jobs.get(parts[1])
            if not job:
                return self._json({"error": "unknown job"}, 404)
            d = store.update_details(job.workdir, body)
            studio.apply_details(job)
            return self._json({"details": d, "display_name": studio.display_name(job)})
        if head == "labels" and len(parts) >= 2:
            from .product import labels as labelsmod

            job = studio.jobs.get(parts[1])
            if not job or job.state != "done":
                return self._json({"error": "unknown or unfinished job"}, 404)
            if len(parts) == 3 and parts[2] == "detect":
                try:
                    found = studio.detect_labels(job, max_frames=int(body.get("frames") or 0) or None)
                except ValueError as exc:
                    return self._json({"error": str(exc)}, 400)
                except Exception as exc:  # noqa: BLE001
                    return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
                return self._json(found)
            if len(parts) == 4 and parts[3] == "confirm":
                rec = labelsmod.confirm_label(job.workdir, parts[2], label=body.get("label"), kind=body.get("kind"))
                if not rec:
                    return self._json({"error": "no such label"}, 404)
                return self._json({"label": rec, "labels": studio.labels_of(job)})
            if len(parts) == 2:
                try:
                    xyz = [float(v) for v in body.get("xyz") or []]
                    if len(xyz) != 3:
                        raise ValueError
                    rec = labelsmod.add_label(job.workdir, label=str(body.get("label") or ""), xyz=xyz,
                                              kind=str(body.get("kind") or "other"), frame=body.get("frame"),
                                              uv=body.get("uv"), note=str(body.get("note") or ""))
                except ValueError as exc:
                    return self._json({"error": str(exc) or "need label and xyz"}, 400)
                return self._json({"label": rec, "labels": studio.labels_of(job)})
            return self._json({"error": "not found"}, 404)
        if head == "share":
            kind = str(body.get("kind") or "package")
            target = str(body.get("target") or "")
            if kind == "project" and not store.project(target):
                return self._json({"error": "no such project"}, 404)
            if kind in ("package", "view") and target not in studio.jobs:
                return self._json({"error": "unknown job"}, 404)
            try:
                rec = store.create_share(kind, target, label=str(body.get("label") or ""))
            except ValueError as exc:
                return self._json({"error": str(exc)}, 400)
            rec["url"] = f"{self._base_url()}/s/{rec['token']}"
            return self._json(rec)
        self._json({"error": "not found"}, 404)

    def _html(self, body: str, status: int = 200) -> None:
        payload = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _svg(self, body: str) -> None:
        payload = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "image/svg+xml")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, obj: dict, status: int = 200) -> None:
        payload = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _bytes(self, data: bytes, content_type: str, *, download: str | None = None, cache: str | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if cache:
            self.send_header("Cache-Control", cache)
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: Path, content_type: str, download: str | None = None) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(data)


def serve(
    root: Path,
    *,
    port: int = 8765,
    host: str | None = None,
    max_points: int = 6_000_000,
    open_browser: bool = True,
    showcase: bool = False,
) -> str:
    """Run the studio until interrupted. Returns the URL it bound to."""
    import webbrowser

    host = host or os.environ.get("LOCAISH_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", port))
    root.mkdir(parents=True, exist_ok=True)
    studio = Studio(root, max_points=max_points)
    studio.showcase = bool(showcase) or os.environ.get("LOCAISH_SHOWCASE", "").lower() in ("1", "true", "yes")
    handler = type("Handler", (_Handler,), {"studio": studio})
    httpd = ThreadingHTTPServer((host, port), handler)
    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{shown}:{httpd.server_address[1]}"

    print(f"locaish studio on {url}")
    print(f"  jobs land in {root}")
    if studio.showcase:
        print("  showcase: uploads are off; the scanned locations are the product")
    if studio.restored:
        print(f"  {studio.restored} finished job(s) restored from disk")
    print("  ctrl-c to stop")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return url


LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32">
  <defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#6ea8fe"/><stop offset="1" stop-color="#a78bfa"/>
  </linearGradient></defs>
  <path d="M11 4H8a4 4 0 0 0-4 4v3M21 4h3a4 4 0 0 1 4 4v3M28 21v3a4 4 0 0 1-4 4h-3M11 28H8a4 4 0 0 1-4-4v-3" fill="none" stroke="url(#lg)" stroke-width="2.6" stroke-linecap="round"/>
  <rect x="11" y="11" width="10" height="10" rx="2.6" transform="rotate(45 16 16)" fill="url(#lg)"/>
</svg>
"""

PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Locaish</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<style>
  /* The system: black, white, two greys, one hairline, one warm tone that
     only ever marks a render. One sans for words, one mono for numbers. */
  :root {
    --bg: #000000; --surface: #0a0a0a; --surface2: #111111; --line: #1f1f1f; --line2: #2a2a2a;
    --ink: #ffffff; --dim: #9a9a9a; --mute: #5a5a5a;
    --accent: #ffffff; --accent-ink: #000000; --render: #d9a86c;
    --ok: #ffffff; --warn: #9a9a9a; --bad: #ffffff;
    --r: 12px; --r2: 8px; --r3: 6px; --gap: 64px; --pad: 24px;
    --sans: -apple-system, "Helvetica Neue", Inter, sans-serif;
    --mono: ui-monospace, "SF Mono", Menlo, monospace;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.5 var(--sans); -webkit-font-smoothing: antialiased; }
  a { color: inherit; }
  .hidden { display: none !important; }
  main { max-width: 1200px; margin: 0 auto; padding: 0 32px 120px; }
  button, input, select, textarea { font: inherit; color: inherit; }
  h1, h2, h3, h4 { margin: 0; font-weight: 600; }
  .dim { color: var(--dim); font-size: 13px; }
  .num, .mono { font-family: var(--mono); font-variant-numeric: tabular-nums; }

  /* -- controls: one primary, one secondary, one small control -- */
  .primary, #st_run, #planbtn, #pkg .acts .primary, .lprow .primary, .sm_acts .primary, #drop .btn {
    display: inline-flex; align-items: center; justify-content: center; gap: 6px; background: var(--ink); color: var(--bg);
    border: 1px solid var(--ink); border-radius: var(--r2); padding: 8px 16px; font-weight: 600; font-size: 13px;
    cursor: pointer; text-decoration: none; white-space: nowrap; }
  .primary:hover, #st_run:hover, #planbtn:hover, #drop .btn:hover { background: #e6e6e6; border-color: #e6e6e6; }
  .primary:disabled, #st_run:disabled, #planbtn:disabled { opacity: .4; cursor: default; }
  .lensbtn, #send, #vfbar .exit, #pkg .acts a, #pkg .acts button, #planlinks a, #planlinks button, .lprow button, .st_runrow .lensbtn {
    display: inline-flex; align-items: center; gap: 6px; background: transparent; color: var(--ink); border: 1px solid var(--ink);
    border-radius: var(--r2); padding: 7px 14px; font-size: 13px; cursor: pointer; text-decoration: none; white-space: nowrap; }
  .lensbtn:hover, #send:hover, #pkg .acts a:hover, #pkg .acts button:hover, #planlinks a:hover, #planlinks button:hover, .lprow button:hover { background: var(--surface2); }
  .lensbtn:disabled, #send:disabled { opacity: .4; cursor: default; }
  .lensbtn[aria-pressed="true"] { background: var(--ink); color: var(--bg); }
  .ctl, .st_x, .st_nudge button, .chip button, #vfbar .lens, #pkg .hours button, .room .gal button, #galbtn, #chips button, .st_toggle button {
    background: transparent; color: var(--dim); border: 1px solid var(--line2); border-radius: var(--r3); padding: 3px 9px;
    font-size: 12px; cursor: pointer; white-space: nowrap; line-height: 1.4; }
  .st_x:hover, .st_nudge button:hover, .chip button:hover, #vfbar .lens:hover, #pkg .hours button:hover, #galbtn:hover, #chips button:hover { color: var(--ink); border-color: var(--dim); }
  input, select, textarea { background: var(--bg); color: var(--ink); border: 1px solid var(--line2); border-radius: var(--r2); padding: 8px 10px; outline: none; font-size: 13px; }
  input:focus, select:focus, textarea:focus { border-color: var(--ink); }
  input::placeholder, textarea::placeholder { color: var(--mute); }
  input[type=range] { padding: 0; border: 0; accent-color: #fff; }
  input[type=checkbox] { accent-color: #fff; padding: 0; }
  label.dim { display: inline-flex; align-items: center; gap: 6px; }
  .pill { font: 11px/1.4 var(--mono); color: var(--dim); border: 1px solid var(--line2); border-radius: var(--r3); padding: 3px 8px; white-space: nowrap; letter-spacing: .02em; }
  .pill.on { color: var(--ink); }
  .st_tag { display: inline-block; font: 10.5px/1.4 var(--sans); letter-spacing: .08em; text-transform: uppercase; color: var(--render); margin-left: 6px; }
  .st_tag.dim { color: var(--dim); }
  .g { display: inline-block; font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase; color: var(--render); margin-right: 8px; font-weight: 500; }

  /* -- top bar -- */
  #top { position: sticky; top: 0; z-index: 5; display: flex; align-items: center; gap: 20px; height: 56px; margin: 0 -32px 32px; padding: 0 32px; background: var(--bg); border-bottom: 1px solid var(--line); }
  #top .brand { font-weight: 600; font-size: 15px; letter-spacing: -0.01em; text-decoration: none; display: inline-flex; align-items: center; gap: 8px; }
  #top .brand .mark { flex: none; }
  #top .loc { display: flex; align-items: baseline; gap: 10px; min-width: 0; }
  #top .loc b { font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  #top .loc span { color: var(--dim); font: 12px/1.4 var(--mono); white-space: nowrap; }
  #top .right { margin-left: auto; display: flex; align-items: center; gap: 10px; }
  #top .topnav { display: flex; gap: 18px; margin-left: 4px; }
  #top .topnav a { color: var(--dim); text-decoration: none; font-size: 13px; }
  #top .topnav a:hover { color: var(--ink); }
  #top .right a.quiet { color: var(--dim); font-size: 13px; text-decoration: none; }
  #top .right a.quiet:hover { color: var(--ink); }
  #galbtn.on { color: var(--ink); }
  @media (max-width: 720px) { #top .topnav { display: none; } }

  /* -- state 1: landing -- */
  #landing { min-height: calc(100vh - 88px); display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding-bottom: 6vh; }
  #landing h1 { font-size: 40px; letter-spacing: -0.03em; margin: 0 0 10px; }
  #landing p.lead { color: var(--dim); font-size: 15px; margin: 0 0 36px; max-width: 520px; }
  #drop { width: min(680px, 100%); border: 1px solid var(--line2); border-radius: var(--r); padding: 56px 24px 48px; cursor: pointer; background: var(--surface); }
  #drop.hot { border-color: var(--ink); }
  #drop b { display: block; font-size: 17px; margin-bottom: 8px; font-weight: 600; }
  #drop span { color: var(--dim); font-size: 13px; display: block; }
  #drop .btn { margin-top: 24px; padding: 10px 20px; }
  #geo { display: flex; gap: 8px; align-items: center; margin-top: 20px; color: var(--dim); font-size: 13px; cursor: pointer; }
  #landing .steps { display: flex; gap: 40px; margin-top: 64px; color: var(--mute); font-size: 13px; }
  #landing .steps b { display: block; color: var(--dim); font-weight: 500; margin-bottom: 2px; }
  #library { width: min(1020px, 100%); margin-top: 64px; text-align: left; }
  .libhead { color: var(--dim); font-size: 13px; letter-spacing: .08em; text-transform: uppercase; margin: 0 0 16px; font-weight: 500; }
  #rooms { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }
  .room { display: block; position: relative; text-decoration: none; background: var(--surface); border: 1px solid var(--line); border-radius: var(--r); overflow: hidden; }
  .room:hover { border-color: var(--dim); }
  .room .thumb { display: block; aspect-ratio: 16 / 9; background: var(--bg); }
  .room .thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .room .rbody { display: block; padding: 12px 16px 14px; }
  .room .rbody b { display: block; font-size: 14px; font-weight: 500; margin-bottom: 2px; }
  .room .rmeta { color: var(--dim); font: 12px/1.4 var(--mono); }
  .libfoot { color: var(--mute); font-size: 13px; margin: 20px 0 0; max-width: 720px; }
  .room .gal { position: absolute; top: 10px; right: 10px; display: flex; gap: 6px; }
  .room .gal button { background: rgba(0,0,0,.8); }
  .room .gal .in { color: var(--ink); cursor: default; }

  /* -- state 2: working -- */
  #working { max-width: 680px; margin: 14vh auto 0; }
  #stageline { display: flex; align-items: center; gap: 12px; font-size: 15px; min-height: 24px; }
  #stageline .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ink); animation: pulse 1.2s ease-in-out infinite; flex: none; }
  @keyframes pulse { 50% { opacity: .25; } }
  .bar { height: 2px; background: var(--line); margin: 16px 0 12px; overflow: hidden; }
  .bar i { display: block; height: 100%; width: 0; background: var(--ink); transition: width .5s; }
  details#logbox { margin-top: 12px; }
  details#logbox summary { color: var(--mute); font-size: 13px; cursor: pointer; }
  #log { margin-top: 10px; background: var(--surface); border: 1px solid var(--line); border-radius: var(--r2); padding: 14px 16px; max-height: 260px; overflow-y: auto; font: 12px/1.7 var(--mono); }
  #log div { display: flex; gap: 10px; }
  #log .t { color: var(--mute); min-width: 40px; text-align: right; flex: none; }
  .note { color: var(--dim); }
  .error { color: var(--ink); }

  /* -- state 3: the location -- */
  #hero { border: 1px solid var(--line); border-radius: var(--r); overflow: hidden; background: var(--bg); }
  #viewerwrap { height: min(66vh, 760px); min-height: 420px; background: var(--bg); }
  #viewerwrap iframe { width: 100%; height: 100%; border: 0; display: block; }
  #scoutcap { padding: 12px 20px; border-top: 1px solid var(--line); font-size: 13px; display: flex; gap: 12px; align-items: baseline; }
  #scoutcap b { font-weight: 600; white-space: nowrap; }
  #scoutcap span { color: var(--dim); }
  #vfbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; padding: 10px 20px; border-top: 1px solid var(--line); font-size: 13px; color: var(--dim); }
  #vfbar b { color: var(--ink); font-weight: 600; margin-right: 4px; }
  #vfbar > span:not(.near) { font-family: var(--mono); font-size: 12px; margin-right: 8px; }
  #vfbar .lens[aria-pressed="true"] { color: var(--ink); border-color: var(--ink); }
  #vfbar .near { margin-left: auto; font: 12px/1.5 var(--mono); }
  #heroactions { display: flex; gap: 20px; justify-content: flex-end; padding: 10px 4px 0; font-size: 12px; }
  #heroactions a { color: var(--mute); text-decoration: none; }
  #heroactions a:hover { color: var(--ink); }

  section { margin-top: var(--gap); }
  .sechead { display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; margin-bottom: 20px; }
  .sechead h2 { font-size: 22px; font-weight: 600; letter-spacing: -0.015em; line-height: 1.25; }
  .sechead p { margin: 6px 0 0; color: var(--dim); font-size: 14px; max-width: 720px; }
  .sechead .sub { display: flex; flex-direction: column; min-width: 0; }
  .sechead .acts { display: flex; gap: 8px; align-items: center; flex: none; }
  .panelbox { background: var(--surface); border: 1px solid var(--line); border-radius: var(--r); padding: var(--pad); min-width: 0; }
  .panelbox h4, .ptitle { font-size: 13px; font-weight: 500; letter-spacing: .08em; text-transform: uppercase; color: var(--dim); margin: 0 0 14px; }
  .panelbox h4 + h4, .panelbox .mt { margin-top: 20px; }
  .panelbox p { margin: 6px 0 0; }
  .st_log { font: 12px/1.7 var(--mono); color: var(--dim); max-height: 160px; overflow-y: auto; margin-top: 10px; }
  .st_log:empty { display: none; }
  .st_row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 12px; }
  .st_row:first-child { margin-top: 0; }
  .st_links { display: flex; gap: 20px; margin-top: 16px; font-size: 12px; }
  .st_links a { color: var(--mute); }
  .st_links a:hover { color: var(--ink); }
  .st_hint { font-size: 12px; margin-top: 10px; color: var(--mute); }
  .st_hint:empty { display: none; }
  .st_toggle { display: inline-flex; border: 1px solid var(--line2); border-radius: var(--r3); overflow: hidden; }
  .st_toggle button { border: 0; border-radius: 0; padding: 4px 12px; }
  .st_toggle button + button { border-left: 1px solid var(--line2); }
  .st_toggle button.on { background: var(--ink); color: var(--bg); }
  .st_toggle button:disabled { opacity: .4; cursor: default; }

  /* -- the world: the room as it will be on the day -- */
  .wd_controls label input[type=range] { width: 160px; vertical-align: middle; margin-left: 4px; }
  .wd_controls #wd_hourout { font: 13px/1 var(--mono); color: var(--ink); margin-left: 6px; }
  .wd_controls .wd_solar { font-size: 11px; color: var(--mute); }
  .wd_controls .wd_dressing { flex: 1; min-width: 240px; }
  .wd_controls .primary { margin-left: auto; }
  .wd_progress { color: var(--dim); font-size: 13px; margin-top: 12px; min-height: 1.4em; }
  .wd_progress:empty { display: none; }
  .wd_progress.error { color: var(--ink); }
  .wd_meter { height: 2px; background: var(--line); margin-top: 8px; overflow: hidden; }
  .wd_meter i { display: block; height: 100%; width: 0; background: var(--ink); transition: width .3s; }
  .wd_prev { margin-top: 10px; font-size: 12px; }
  .wd_prev a { color: var(--mute); margin-right: 14px; text-decoration: none; }
  .wd_prev a:hover, .wd_prev a.active { color: var(--ink); }
  .wd_bar { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin: 20px 0 10px; font-size: 13px; }
  .wd_bar .st_tag { margin: 0; }
  .wd_bar a { margin-left: auto; color: var(--mute); font-size: 12px; text-decoration: none; }
  .wd_bar a:hover { color: var(--ink); }
  #wd_wrap { height: min(66vh, 760px); min-height: 420px; background: var(--bg); border: 1px solid var(--line); border-radius: var(--r); overflow: hidden; }
  #wd_wrap iframe { width: 100%; height: 100%; border: 0; display: block; }
  .wd_strip img { opacity: .8; }

  /* -- the scene in this room -- */
  #st_status { margin: -8px 0 16px; font-size: 12px; color: var(--mute); }
  .st_grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
  .st_grid2 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-top: 24px; }
  @media (max-width: 1000px) { .st_grid, .st_grid2 { grid-template-columns: 1fr; } }
  .st_list { display: flex; flex-direction: column; gap: 8px; margin-bottom: 12px; }
  .st_list:empty { display: none; }
  .st_row input, .st_row select { font-size: 13px; }
  #st_name { flex: 1; min-width: 100px; }
  #st_height { width: 110px; }
  .st_runrow { margin-top: 12px; }
  .st_runrow a { color: var(--mute); font-size: 12px; }
  .st_runrow a:hover { color: var(--ink); }
  .st_runrow .primary { margin-left: auto; }
  .st_actor, .st_piece { display: flex; gap: 12px; align-items: flex-start; border: 1px solid var(--line); border-radius: var(--r2); padding: 12px; background: var(--bg); font-size: 13px; }
  .st_piece { justify-content: space-between; }
  .st_piece > div:first-child { flex: 1; min-width: 0; }
  .st_piece b { font-weight: 500; }
  .st_head { width: 44px; height: 44px; border-radius: var(--r3); object-fit: cover; flex: none; display: block; }
  .st_blank { opacity: .35; }
  .st_actorbody { flex: 1; min-width: 0; }
  .st_actorbody b { font-weight: 500; }
  .st_acts { display: flex; gap: 8px; align-items: center; margin-top: 10px; flex-wrap: wrap; }
  .st_file { font-size: 12px; color: var(--dim); border: 1px solid var(--line2); border-radius: var(--r3); padding: 3px 9px; cursor: pointer; }
  .st_file:hover { color: var(--ink); border-color: var(--dim); }
  .st_file input { display: none; }
  .st_nudge { display: grid; grid-template-columns: repeat(4, 30px); gap: 4px; align-content: start; }
  .st_nudge button { padding: 3px 0; text-align: center; }
  .st_nudge [data-reset], .st_nudge .lensbtn, .st_nudge .st_x { grid-column: span 2; }
  .st_nudge .lensbtn { padding: 3px 0; justify-content: center; font-size: 12px; border-color: var(--line2); color: var(--dim); }
  .st_moved { color: var(--ink); font-size: 12px; }
  .st_planbox svg { width: 100%; height: auto; display: block; }
  .st_words { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 8px; }
  .st_words input { width: 100%; font-size: 12px; padding: 4px 8px; }
  .st_scenebox { margin-top: 16px; }
  #st_scene { width: 100%; background: var(--bg); color: var(--ink); border: 1px solid var(--line2); border-radius: var(--r2); padding: 12px 14px; font: 13px/1.6 var(--mono); outline: none; resize: vertical; min-height: 150px; }
  #st_scene:focus { border-color: var(--ink); }
  #st_prev { margin-top: 10px; font-size: 12px; }
  #st_prev a { color: var(--mute); margin-right: 12px; text-decoration: none; }
  #st_prev a:hover { color: var(--ink); }
  #st_out { margin-top: 40px; }
  .st_head2 { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; margin: 0 0 12px; font-size: 16px; }
  .st_head2 b { font-weight: 600; }
  .st_head2 span { color: var(--dim); font-size: 13px; }
  .st_finishbar { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin: 0 0 12px; font-size: 13px; }
  .st_finishbar label { color: var(--dim); font-size: 12px; display: inline-flex; align-items: center; gap: 6px; }
  .st_finishbar .st_progress { color: var(--dim); font-size: 12px; margin-left: auto; font-family: var(--mono); }
  .st_finishbar .st_progress.error { color: var(--ink); }
  .st_framerow { display: grid; grid-template-columns: 1fr; gap: 16px; align-items: start; }
  .st_framerow.has_fin { grid-template-columns: minmax(0, 1fr) 300px; }
  @media (max-width: 1000px) { .st_framerow.has_fin { grid-template-columns: 1fr; } }
  .st_framerow img { width: 100%; border-radius: var(--r2); display: block; background: var(--surface); }
  .st_framerow .dim { margin-top: 8px; font-size: 12px; }
  .st_frame_finished { position: relative; }
  .st_frame_finished .st_stamp { position: absolute; top: 10px; left: 10px; font: 10.5px/1.4 var(--sans); letter-spacing: .08em; text-transform: uppercase; color: var(--render); background: rgba(0,0,0,.7); border-radius: 4px; padding: 3px 8px; }
  #st_frame .st_acts { margin-top: 14px; }
  #st_play { flex: none; }
  #st_scrub { flex: 1; }
  .st_strip { display: flex; gap: 6px; overflow-x: auto; padding: 8px 0 4px; }
  .st_strip:empty { display: none; }
  .st_strip img { height: 64px; border-radius: 4px; cursor: pointer; opacity: .55; border: 2px solid transparent; flex: none; }
  .st_strip img.finished { opacity: .85; }
  .st_strip img.active { opacity: 1; border-color: var(--ink); }
  .st_shot { border-top: 1px solid var(--line); padding: 10px 0; font-size: 13px; }
  .st_shot:first-child { border-top: 0; padding-top: 0; }
  .st_shot b { font-weight: 500; }
  .relax { color: var(--dim); font-size: 12px; margin-top: 4px; }
  #st_sched div, #st_notes p { font-size: 13px; }
  #st_sched b { font-family: var(--mono); font-weight: 500; font-size: 12px; margin-right: 6px; }
  #st_problems h4 { margin-top: 20px; }
  #st_blocking svg { width: 100%; height: auto; display: block; }

  /* -- simulate the camera -- */
  .sm_list { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .sm_list:empty { display: none; }
  .sm_item { display: inline-flex; gap: 10px; align-items: center; border: 1px solid var(--line2); border-radius: var(--r2); padding: 7px 10px 7px 12px; font-size: 13px; cursor: pointer; }
  .sm_item:hover { border-color: var(--dim); }
  .sm_item.active { border-color: var(--ink); }
  .sm_item b { font-weight: 500; }
  .sm_item .sm_probs { color: var(--dim); font-size: 12px; }
  .sm_item .st_x { padding: 0 6px; border: 0; }
  #sm_fromreh { max-width: 320px; }
  #sm_open { margin-top: 16px; }
  .sm_direct .st_row { margin-top: 0; }
  .sm_direct input { flex: 1; min-width: 260px; }
  .sm_looks { display: flex; gap: 10px; overflow-x: auto; margin-top: 12px; }
  .sm_looks:empty { display: none; }
  .sm_looks figure { margin: 0; flex: none; width: 220px; font-size: 11.5px; color: var(--dim); }
  .sm_looks img { width: 220px; border-radius: var(--r3); display: block; }
  .sm_looks b { color: var(--ink); font-weight: 500; }
  .sm_grid { display: grid; grid-template-columns: 1.3fr 1fr; gap: 16px; margin-top: 16px; }
  @media (max-width: 1000px) { .sm_grid { grid-template-columns: 1fr; } }
  .sm_frame img { width: 100%; border-radius: var(--r2); display: block; background: var(--bg); min-height: 120px; }
  .sm_frame .dim { margin-top: 8px; font-size: 12px; }
  .sm_rig { font-size: 13px; }
  .sm_rig b { font-weight: 500; }
  .sm_rig p { margin: 8px 0 0; color: var(--dim); font-size: 13px; }
  .sm_seg { display: flex; gap: 10px; align-items: baseline; font-size: 12.5px; padding: 5px 0; border-top: 1px solid var(--line); color: var(--dim); }
  .sm_seg:first-child { border-top: 0; padding-top: 0; }
  .sm_seg b { font-family: var(--mono); font-weight: 500; white-space: nowrap; font-size: 12px; }
  .sm_seg.red { color: var(--ink); }
  .sm_seg.amber { color: var(--dim); }
  .sm_summary { font-size: 13px; color: var(--dim); margin-top: 10px; }
  .sm_summary b { color: var(--ink); font-weight: 500; font-family: var(--mono); font-size: 12.5px; }
  .sm_acts { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 14px; }
  .sm_acts:empty { display: none; }
  .sm_video video { width: 100%; border-radius: var(--r2); display: block; margin-top: 12px; background: var(--bg); }
  .sm_video .sm_pair { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .sm_video .sm_pair.single { grid-template-columns: 1fr; }
  .sm_video figure { margin: 0; }
  .sm_video figcaption { color: var(--dim); font-size: 12px; margin-top: 6px; }
  .sm_howto { font-size: 12px; color: var(--mute); margin-top: 10px; }

  /* -- what is in the room -- */
  .labelchips { display: flex; flex-wrap: wrap; gap: 8px; }
  .chip { display: inline-flex; align-items: center; gap: 6px; border: 1px solid var(--line2); border-radius: var(--r3); padding: 4px 6px 4px 12px; font-size: 13px; cursor: default; }
  .chip.model { color: var(--dim); }
  .chip .k { color: var(--mute); font-size: 11px; }
  .chip button { border: 0; padding: 1px 5px; font-size: 12px; }
  .chip button.ok:hover { color: var(--ink); }
  #labelpick { margin-top: 20px; border-top: 1px solid var(--line); padding-top: 20px; }
  .crops { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 12px 0 14px; }
  @media (max-width: 900px) { .crops { grid-template-columns: 1fr 1fr; } }
  .crops img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: var(--r2); border: 2px solid transparent; cursor: pointer; background: var(--bg); }
  .crops img.on { border-color: var(--ink); }
  .crops .cap { color: var(--mute); font: 11px/1.4 var(--mono); margin-top: 4px; }
  .lprow { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .lprow input { flex: 1; min-width: 220px; }
  #lstatus { margin-top: 12px; }
  #lstatus:empty { display: none; }

  /* -- the scouting tools, folded -- */
  .moretools { margin-top: var(--gap); border-top: 1px solid var(--line); padding-top: 24px; }
  .moretools > summary { cursor: pointer; list-style: none; display: flex; align-items: baseline; gap: 12px; font-size: 13px; letter-spacing: .08em; text-transform: uppercase; color: var(--dim); font-weight: 500; }
  .moretools > summary::-webkit-details-marker { display: none; }
  .moretools > summary::before { content: '\\25B8'; color: var(--mute); font-size: 11px; }
  .moretools[open] > summary::before { content: '\\25BE'; }
  .moretools > summary span { text-transform: none; letter-spacing: 0; color: var(--mute); font-weight: 400; }
  .moretools > summary:hover { color: var(--ink); }
  .moretools > section:first-of-type { margin-top: 40px; }

  /* -- the location package -- */
  #pkg .pkgrow { display: grid; grid-template-columns: 1.1fr 1fr; gap: 16px; }
  @media (max-width: 900px) { #pkg .pkgrow { grid-template-columns: 1fr; } }
  #pkg .field { margin-bottom: 12px; }
  #pkg label { display: block; color: var(--dim); font-size: 12px; margin-bottom: 4px; }
  #pkg input, #pkg textarea { width: 100%; }
  #pkg .two { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  #pkg .three { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
  #pkg .acts { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 8px; }
  #pkg .acts .st { color: var(--mute); font-size: 12px; }
  #pkg .sunprev { background: var(--surface); border: 1px solid var(--line); border-radius: var(--r); overflow: hidden; }
  #pkg .sunprev img { width: 100%; display: block; }
  #pkg .hours { display: flex; gap: 6px; padding: 12px 16px; border-top: 1px solid var(--line); align-items: center; flex-wrap: wrap; }
  #pkg .hours button { font-family: var(--mono); font-size: 11.5px; }
  #pkg .hours button[aria-pressed="true"] { color: var(--bg); background: var(--ink); border-color: var(--ink); }
  #pkg .hours .sunline { margin-left: auto; color: var(--dim); font-size: 12px; }
  #pkg .sharebox { margin-top: 12px; color: var(--dim); font-size: 12px; word-break: break-all; }
  #pkg .sharebox a { color: var(--ink); }

  /* -- coverage -- */
  #briefwrap { display: grid; grid-template-columns: 1fr; gap: 12px; background: var(--surface); border: 1px solid var(--line); border-radius: var(--r); padding: var(--pad); }
  #brief { width: 100%; padding: 12px 14px; font: 14px/1.6 var(--sans); resize: vertical; min-height: 124px; }
  #planrow { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
  #modenote { color: var(--mute); font-size: 12px; flex: 1; min-width: 200px; }
  #example { color: var(--dim); font-size: 12px; text-decoration: none; border-bottom: 1px solid var(--line2); }
  #example:hover { color: var(--ink); }
  #prevplans { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; color: var(--mute); font-size: 12px; }
  #prevplans a { color: var(--dim); text-decoration: none; }
  #prevplans a:hover { color: var(--ink); }
  #trace { margin-top: 20px; color: var(--mute); font-size: 12px; }
  #trace summary { cursor: pointer; }
  #tracebody { margin-top: 10px; background: var(--surface); border: 1px solid var(--line); border-radius: var(--r2); padding: 12px 16px; max-height: 420px; overflow-y: auto; font: 12px/1.7 var(--mono); }
  #tracebody div { display: flex; gap: 10px; }
  #tracebody .t { color: var(--mute); min-width: 40px; text-align: right; flex: none; }
  #tracebody .ag { color: var(--ink); }
  #tracebody pre { white-space: pre-wrap; word-break: break-word; margin: 2px 0 8px 50px; color: var(--dim); background: var(--surface2); border-radius: var(--r3); padding: 8px 10px; font: 11px/1.5 var(--mono); }
  #planout { margin-top: 32px; }
  #planhead { display: flex; align-items: baseline; justify-content: space-between; margin: 0 0 14px; gap: 12px; flex-wrap: wrap; }
  #planhead b { font-size: 16px; font-weight: 600; }
  #planhead span { color: var(--dim); font-size: 13px; }
  #shots { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
  @media (max-width: 1100px) { #shots { grid-template-columns: 1fr 1fr; } }
  @media (max-width: 760px) { #shots { grid-template-columns: 1fr; } }
  .shot { background: var(--surface); border: 1px solid var(--line); border-radius: var(--r); overflow: hidden; }
  .shot.active { border-color: var(--ink); }
  .shot.none { padding: 16px; color: var(--dim); }
  .shot .frame { position: relative; aspect-ratio: 16 / 9; background: var(--bg); cursor: pointer; display: block; }
  .shot .frame img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .shot .frame .look { position: absolute; right: 10px; bottom: 10px; background: rgba(0,0,0,.75); color: var(--ink); border: 1px solid var(--line2); border-radius: var(--r3); padding: 4px 9px; font-size: 11.5px; }
  .shot .body { padding: 14px 16px 16px; }
  .shot h3 { margin: 0 0 4px; font-size: 14px; font-weight: 600; }
  .shot .meas { font: 12px/1.5 var(--mono); color: var(--dim); }
  .shot .meas .warnc, .warnc { color: var(--dim); }
  .shot .desc { margin-top: 8px; font-size: 13.5px; }
  .shot .why { color: var(--dim); font-size: 12.5px; margin-top: 6px; line-height: 1.5; }
  .shot .dim { color: var(--mute); font-size: 12px; margin-top: 4px; }
  .shot .relax { color: var(--dim); font-size: 12px; margin-top: 2px; }
  .shot .review { margin-top: 10px; padding: 8px 10px; border-radius: var(--r3); background: var(--surface2); font-size: 12.5px; }
  .shot details { margin-top: 8px; font-size: 12px; color: var(--mute); }
  .shot details pre { white-space: pre-wrap; word-break: break-word; font: 11px/1.5 var(--mono); background: var(--surface2); border-radius: var(--r3); padding: 8px 10px; margin: 6px 0 0; color: var(--dim); }
  #planbelow { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }
  @media (max-width: 900px) { #planbelow { grid-template-columns: 1fr; } }
  #floorplan svg { width: 100%; height: auto; border-radius: var(--r2); display: block; }
  #capacity table { border-collapse: separate; border-spacing: 3px; width: 100%; font: 12px/1.4 var(--mono); }
  #capacity th { color: var(--dim); font-weight: 500; text-align: center; font-size: 11px; }
  #capacity th.row { text-align: left; }
  #capacity td { text-align: center; border-radius: 4px; padding: 6px 2px; color: var(--ink); min-width: 34px; }
  #capacity .foot { color: var(--mute); font-size: 12px; margin-top: 10px; font-family: var(--sans); }
  #capacity .lead { font-size: 13px; margin-bottom: 10px; color: var(--dim); }
  #capacity .lead b { color: var(--ink); font-family: var(--mono); font-weight: 500; }
  #capacity ol { margin: 6px 0 0 18px; padding: 0; color: var(--dim); font-size: 12px; }
  #planlinks { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 16px; }

  /* -- chat -- */
  #chat .ptitle { margin-bottom: 16px; }
  #thread { display: flex; flex-direction: column; gap: 12px; margin-bottom: 16px; }
  #thread:empty { display: none; }
  .msg { max-width: 78%; padding: 10px 14px; border-radius: var(--r); white-space: pre-wrap; border: 1px solid var(--line); font-size: 13.5px; }
  .msg.user { align-self: flex-end; background: var(--surface2); }
  .msg.agent { align-self: flex-start; background: var(--surface); }
  .msg.agent img { max-width: 100%; border-radius: var(--r2); margin-top: 8px; display: block; }
  .msg .tools { color: var(--mute); font: 11.5px/1.7 var(--mono); border-top: 1px solid var(--line); margin-top: 8px; padding-top: 6px; }
  .msg.err { border-color: var(--line2); color: var(--dim); }
  .thinking { color: var(--dim); font-size: 13px; }
  #chips { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
  #askrow { display: flex; gap: 8px; }
  #ask { flex: 1; padding: 10px 14px; }
</style>
<main>
  <div id="top">
    <a class="brand" href="/"><svg class="mark" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><rect x="1" y="1" width="14" height="14" rx="3.5" fill="none" stroke="currentColor" stroke-width="1.5"/><circle cx="8" cy="8" r="2.6" fill="currentColor"/></svg>Locaish</a>
    <div class="loc hidden" id="toploc"><b id="locname"></b><span id="locmeta"></span></div>
    <nav class="topnav"><a href="/projects">Projects</a><a href="/capture">Capture</a><a href="/pricing">Pricing</a></nav>
    <div class="right">
      <button class="hidden" id="galbtn"></button>
      <span class="pill hidden" id="covpill">local sweep</span>
      <a class="quiet hidden" id="topagain" href="/">Scan another room</a>
    </div>
  </div>

  <div id="landing">
    <h1>Locaish</h1>
    <p class="lead" id="lead">Scan a room with your phone. Plan the scene before anyone drives out.</p>
    <div id="drop">
      <b>Drop a walkthrough video here</b>
      <span>or a scan export &mdash; ply, glb, obj &mdash; or a saved .twin. Sixty seconds of walking is plenty.</span>
      <span class="btn">Choose a file</span>
      <input id="file" type="file" accept="video/*,.ply,.obj,.glb,.gltf,.stl,.mov,.mp4,.mkv,.twin" class="hidden">
    </div>
    <label id="geo"><input type="checkbox" id="usegeo" checked>
      use my location so the sun schedule is real</label>
    <div id="library" class="hidden"></div>
    <div class="steps" id="steps">
      <div><b>1 &middot; Scan</b>a metric, gravity-aligned twin</div>
      <div><b>2 &middot; Plan</b>every shot placed, rendered, reviewed</div>
      <div><b>3 &middot; Look</b>through the lens, from the spot</div>
    </div>
  </div>

  <div id="working" class="hidden">
    <div id="stageline"><span class="dot"></span><span id="stage">starting</span></div>
    <div class="bar"><i id="bar"></i></div>
    <details id="logbox"><summary>everything the pipeline said</summary>
      <div id="log"></div>
    </details>
  </div>

  <div id="result" class="hidden">
    <div id="hero">
      <div id="viewerwrap"><iframe id="viewer" title="digital twin"></iframe></div>
      <div id="scoutcap" class="hidden"></div>
      <div id="vfbar" class="hidden"></div>
    </div>
    <div id="heroactions"></div>

    <section id="world">
      <div class="sechead">
        <div class="sub"><h2>Walk into it as it will be on the day</h2>
          <p>Pick the shoot date and the hour. The measured room is rendered as it will be, with the sun where the ephemeris puts it, beside the one you scanned.</p></div>
      </div>
      <div class="panelbox">
        <div class="st_row wd_controls">
          <label class="dim">shoot date <input id="wd_date" type="date"></label>
          <label class="dim">hour <input id="wd_hour" type="range" min="6" max="20" step="0.5" value="15"><b id="wd_hourout">15:00</b><span class="wd_solar">solar</span></label>
          <input id="wd_dressing" class="wd_dressing" placeholder="dressing: the table laid for dinner, candles lit, white lilies in the window">
          <button id="wd_build" class="primary">Build the world</button>
        </div>
        <div class="dim st_hint" id="wd_hint"></div>
        <div id="wd_progress" class="wd_progress"></div>
        <div id="wd_log" class="st_log"></div>
        <div id="wd_prev" class="hidden dim wd_prev"></div>
      </div>
      <div id="wd_out" class="hidden">
        <div class="wd_bar">
          <span class="st_toggle"><button id="wd_scanned" data-mode="scan">as scanned</button><button id="wd_asis" class="on" data-mode="world">as it will be</button></span>
          <span class="st_tag" id="wd_tag">render</span>
          <span class="dim" id="wd_meta"></span>
          <a id="wd_full" href="#" target="_blank">Full screen</a>
        </div>
        <div id="wd_wrap"><iframe id="wd_frame" title="the room as it will be on the day"></iframe></div>
        <div id="wd_strip" class="st_strip wd_strip"></div>
      </div>
    </section>

    <section id="stage">
      <div class="sechead">
        <div class="sub"><h2>The scene in this room</h2>
          <p>Cast it, dress it, paste the scene. Every frame is checked against the measured room, then finished as a photograph.</p></div>
      </div>
      <div class="dim" id="st_status"></div>
      <div class="st_grid">
        <div class="panelbox"><h4>Cast</h4>
          <div id="st_cast" class="st_list"></div>
          <div class="st_row"><input id="st_name" placeholder="name"><input id="st_height" type="number" step="0.01" min="1.2" max="2.2" placeholder="height, m"><button id="st_add" class="lensbtn">Add</button></div>
          <div class="dim st_hint">Height is the measurement. Looks and wardrobe are what the finish dresses the stand-in in.</div>
          <div class="dim" id="st_caststatus"></div>
        </div>
        <div class="panelbox"><h4>Set pieces</h4>
          <div class="st_row"><select id="st_liftsel"></select><button id="st_lift" class="lensbtn">Lift into a set piece</button></div>
          <div id="st_pieces" class="st_list"></div>
          <div class="dim" id="st_piecestatus"></div>
        </div>
        <div class="panelbox st_planbox"><h4>The room as dressed</h4><div id="st_plan"></div></div>
      </div>
      <div class="panelbox st_scenebox">
        <h4>The scene</h4>
        <textarea id="st_scene" rows="7" placeholder="Paste the scene, or type beats: MARKS: A=(0.4,0.8) ... CAST: Maya=1.68 ... 1. Maya enters at A ..."></textarea>
        <div class="st_row st_runrow">
          <label class="dim">shoot date <input id="st_date" type="date"></label>
          <label class="dim">first frame <input id="st_start" type="time" value="09:00"></label>
          <label class="dim">minutes per setup <input id="st_setup" type="number" value="25" min="5" max="120" style="width:64px"></label>
          <a id="st_example_typed" href="#">typed example</a><a id="st_example_prose" href="#">prose example</a>
          <button id="st_run" class="primary">Rehearse the scene</button>
        </div>
        <div id="st_prev" class="hidden dim"></div>
        <div id="st_log" class="st_log"></div>
      </div>
      <div id="st_out" class="hidden">
        <div id="st_head" class="st_head2"></div>
        <div id="st_finishbar" class="st_finishbar"></div>
        <div id="st_frame"></div>
        <div class="st_row"><button id="st_play" class="lensbtn">Play</button><input id="st_scrub" type="range" min="0" max="0" value="0"></div>
        <div id="st_strip" class="st_strip"></div>
        <div class="st_grid2">
          <div class="panelbox"><h4>Setups</h4><div id="st_shots"></div><div id="st_problems"></div></div>
          <div class="panelbox"><div id="st_sched"></div><div id="st_notes"></div></div>
          <div class="panelbox"><h4>Blocking</h4><div id="st_blocking"></div></div>
        </div>
        <div id="st_links" class="st_links"></div>
      </div>
    </section>
    <section id="sim">
      <div class="sechead">
        <div class="sub"><h2>Simulate the camera</h2>
          <p>Direct the camera, or key the move yourself in the twin. The engine runs it through the measured room on a real rig, frame by frame.</p></div>
      </div>
      <div class="sm_list" id="sm_list"></div>
      <div class="st_row"><button id="sm_new" class="lensbtn">New move</button><select id="sm_fromreh"><option value="">from rehearsal</option></select><span class="dim" id="sm_status"></span></div>
      <div id="sm_open" class="hidden">
        <div class="panelbox sm_direct"><h4>Direct the camera</h4>
          <div class="st_row"><input id="sm_instruction" placeholder="push in on Maya over four seconds on the 50, low, as she sits"><button id="sm_directbtn" class="primary">Direct</button></div>
          <div class="dim st_hint" id="sm_directnote"></div>
          <div class="st_log" id="sm_directlog"></div>
          <div class="sm_looks" id="sm_looks"></div>
          <div id="sm_directnotes"></div>
        </div>
        <div class="sm_grid">
          <div class="panelbox sm_frame"><h4>Frame at the playhead</h4><img id="sm_frameimg" alt="frame at the playhead"><div class="dim" id="sm_frametext"></div>
            <div class="sm_acts"><button id="sm_twin" class="primary">Open in the twin</button><button id="sm_render" class="lensbtn">Render the move</button></div>
            <div class="st_log" id="sm_log"></div>
            <div class="sm_video" id="sm_video"></div>
            <div class="sm_acts" id="sm_finishacts"></div>
            <div class="dim" id="sm_finishline"></div>
          </div>
          <div class="panelbox"><h4>Rig</h4><div class="sm_rig" id="sm_rig"></div>
            <h4 class="mt">What does not work</h4><div id="sm_problems"></div>
            <div class="sm_summary" id="sm_summary"></div>
            <h4 class="mt">Hand-off</h4>
            <div class="st_links" id="sm_links"></div>
            <div class="sm_howto">The timeline and its evaluation, the video, the room as a mesh, and the camera as a script another package can read.</div>
          </div>
        </div>
      </div>
    </section>
    <section id="contents">
      <div class="sechead">
        <div class="sub"><h2>What is in the room</h2>
          <p>Named from the capture's own frames by the vision model, confirmed or corrected by you.</p></div>
        <div class="acts"><button class="lensbtn" id="detectbtn">Name things with the vision model</button><button class="lensbtn" id="labelbtn">Label an object</button></div>
      </div>
      <div class="panelbox">
        <div id="chips" class="labelchips"><span class="dim">nothing named yet</span></div>
        <div id="labelpick" class="hidden">
          <div class="dim" id="lpmeta"></div>
          <div class="crops" id="crops"></div>
          <div class="lprow">
            <input id="lpname" list="lpnames" placeholder="what is it? e.g. fridge, radiator, piano" autocomplete="off">
            <datalist id="lpnames"><option value="sofa"><option value="armchair"><option value="dining table"><option value="coffee table"><option value="bed"><option value="desk"><option value="bookshelf"><option value="cabinet"><option value="kitchen island"><option value="counter"><option value="sink"><option value="fridge"><option value="oven"><option value="radiator"><option value="fireplace"><option value="column"><option value="beam"><option value="window"><option value="door"><option value="curtain"><option value="pendant lamp"><option value="floor lamp"><option value="ceiling fan"><option value="plant"><option value="mirror"><option value="artwork"><option value="television"><option value="piano"><option value="electrical panel"><option value="socket"></datalist>
            <select id="lpkind"><option value="furniture">furniture</option><option value="appliance">appliance</option><option value="fixture">fixture</option><option value="window">window</option><option value="door">door</option><option value="lighting">lighting</option><option value="plant">plant</option><option value="artwork">artwork</option><option value="equipment">equipment</option><option value="structure">structure</option><option value="other" selected>other</option></select>
            <button id="lpsave" class="primary">Save label</button>
            <button id="lpcancel">Cancel</button>
          </div>
        </div>
        <div class="dim" id="lstatus"></div>
      </div>
    </section>
    <details id="more" class="moretools"><summary>Scouting tools <span>the location package, the coverage planner, the scout</span></summary>
    <section id="pkg">
      <div class="sechead">
        <div class="sub"><h2>Location package</h2>
          <p>Dimensions, what fits, and the sun through every window on the shoot date, on one page a scout can send.</p></div>
      </div>
      <div class="pkgrow">
        <div class="panelbox">
          <div class="two">
            <div class="field"><label>Location name</label><input id="d_display_name" placeholder="The Fairfax kitchen"></div>
            <div class="field"><label>Shoot date</label><input id="d_shoot_date" type="date"></div>
          </div>
          <div class="field"><label>Address</label><input id="d_address" placeholder="street, city"></div>
          <div class="three">
            <div class="field"><label>Compass heading of +X (°)</label><input id="d_heading_deg" type="number" min="0" max="359" step="1" placeholder="e.g. 212"></div>
            <div class="field"><label>Latitude</label><input id="d_latitude" type="number" step="0.000001"></div>
            <div class="field"><label>Longitude</label><input id="d_longitude" type="number" step="0.000001"></div>
          </div>
          <div class="two">
            <div class="field"><label>Contact</label><input id="d_contact" placeholder="owner / manager, phone"></div>
            <div class="field"><label>Fee</label><input id="d_fee" placeholder="$ / day"></div>
          </div>
          <div class="two">
            <div class="field"><label>Power</label><input id="d_power" placeholder="2 × 20 A, tie-in possible"></div>
            <div class="field"><label>Parking &amp; load-in</label><input id="d_parking" placeholder="driveway for 2 trucks, 30 m push"></div>
          </div>
          <div class="two">
            <div class="field"><label>Access</label><input id="d_access" placeholder="ground floor, double doors"></div>
            <div class="field"><label>Restrictions</label><input id="d_restrictions" placeholder="no shoots after 22:00"></div>
          </div>
          <div class="field"><label>Notes</label><textarea id="d_notes" rows="2"></textarea></div>
          <div class="acts">
            <button id="d_save">Save details</button>
            <a id="pkgopen" class="primary" href="#" target="_blank">Open the package</a>
            <button id="pkgshare">Share link</button>
            <span class="st" id="d_status"></span>
          </div>
          <div class="sharebox hidden" id="sharebox"></div>
        </div>
        <div class="sunprev">
          <img id="sunimg" alt="floor plan with the sun's patch">
          <div class="hours" id="hours"></div>
        </div>
      </div>
    </section>
    <section id="coverage">
      <div class="sechead">
        <div class="sub"><h2>Describe the scene</h2>
          <p>Prose, or a shot list one per line. The scout places every shot in this room and draws the camera plan.</p></div>
      </div>
      <div id="briefwrap">
        <textarea id="brief" rows="5" placeholder="INT. KITCHEN &ndash; DAY. MAYA stands at the counter, back to the window, when JON comes in&hellip;"></textarea>
        <div id="planrow">
          <span id="modenote"></span>
          <a id="example" href="#">example</a>
          <button id="planbtn">Scout the scene</button>
        </div>
        <div id="prevplans" class="hidden"></div>
      </div>
      <div id="planout" class="hidden">
        <div id="planhead"></div>
        <div id="shots"></div>
        <div id="planbelow">
          <div class="panelbox"><h4>Camera plan</h4><div id="floorplan"></div></div>
          <div class="panelbox"><h4>What this room holds</h4><div id="capacity"></div></div>
        </div>
        <div id="planlinks"></div>
        <details id="trace"><summary>How it decided</summary><div id="tracebody"></div></details>
      </div>
    </section>
    <section id="chat">
      <div class="ptitle">Ask the scout</div>
      <div id="thread"></div>
      <div id="askrow">
        <input id="ask" placeholder="Ask the scout anything about this room&hellip;" autocomplete="off">
        <button id="send">Ask</button>
      </div>
    </section>
    </details>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
const SHOWCASE = '__SHOWCASE__' === '1';
if (SHOWCASE) $('topagain').textContent = 'All locations';
const drop = $('drop'), file = $('file'), working = $('working'), result = $('result');
const landing = $('landing');
const bar = $('bar'), stage = $('stage'), log = $('log');
let t0 = 0, jobId = null;

const STAGES = ['frames','decode','score','reconstruct','colmap features','colmap matching',
  'colmap mapping','stereo','scale','clean','refine','subsample','write','read','normals',
  'planes_canonical','planes','canonicalize','grid','structure','capture_bounds','bounds',
  'mesh','qa','plane_fill','semantic','rendering viewer',
  'scouting the location','sweeping camera setups','loading ClickHouse'];

drop.onclick = () => file.click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add('hot'); };
drop.ondragleave = () => drop.classList.remove('hot');
drop.ondrop = e => {
  e.preventDefault(); e.stopPropagation(); drop.classList.remove('hot');
  if (e.dataTransfer.files[0]) send(e.dataTransfer.files[0]);
};
file.onchange = () => { if (file.files[0]) send(file.files[0]); };

// A drop that lands a few pixels off the target is otherwise handled by the
// browser, which navigates away to play the file: from the far side of the
// screen that is indistinguishable from the page ignoring the capture.
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop', e => {
  e.preventDefault();
  if (SHOWCASE) return;
  if (!e.dataTransfer.files[0]) return;
  // Once a capture is on the page the drop target is gone, so a second one
  // lands on the body. Say what to do about it rather than swallowing it.
  if (!working.classList.contains('hidden')) {
    line('this page is already holding a capture -- reload it to start another', 'note');
    $('logbox').open = true;
    return;
  }
  send(e.dataTransfer.files[0]);
});

function line(text, cls) {
  const el = document.createElement('div');
  const dt = t0 ? ((Date.now() - t0) / 1000).toFixed(0) + 's' : '';
  el.innerHTML = '<span class="t">' + dt + '</span><span class="' + (cls||'') + '"></span>';
  el.lastChild.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
}

function send(f) {
  // The panel switches before anything that can block runs. Asking for a
  // position first meant an unanswered permission prompt left the page
  // looking exactly as it had before the drop.
  t0 = Date.now();
  landing.classList.add('hidden');
  working.classList.remove('hidden');
  log.innerHTML = '';
  bar.style.width = '1%';
  stage.textContent = 'reading ' + f.name;
  stage.className = '';
  line(stage.textContent);

  if (!($('usegeo').checked && navigator.geolocation)) { upload(f, null, null); return; }

  // getCurrentPosition's own timeout does not start until the prompt is
  // answered, so a prompt the browser never shows -- location services off
  // for it at the system level -- hangs forever. This one is wall clock:
  // when it fires the capture goes up without coordinates.
  let sent = false;
  const go = (lat, lon) => { if (!sent) { sent = true; upload(f, lat, lon); } };
  line('asking your browser where you are', 'note');
  const fallback = setTimeout(() => {
    if (!sent) line('no location yet; continuing without a sun schedule', 'note');
    go(null, null);
  }, 6000);
  navigator.geolocation.getCurrentPosition(
    p => { clearTimeout(fallback); go(p.coords.latitude, p.coords.longitude); },
    () => {
      clearTimeout(fallback);
      if (!sent) line('location declined; continuing without a sun schedule', 'note');
      go(null, null);
    },
    {timeout: 6000, maximumAge: 600000});
}

function upload(f, lat, lon) {
  bar.style.width = '2%';
  stage.textContent = 'uploading ' + f.name + ' (' + (f.size/1e6).toFixed(0) + ' MB)';
  line(stage.textContent);
  if (lat != null) line('georeferenced to your position; sun schedule unlocked');

  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/upload?name=' + encodeURIComponent(f.name)
    + (lat != null ? '&lat=' + lat + '&lon=' + lon : ''));
  xhr.setRequestHeader('Content-Type', 'application/octet-stream');
  xhr.upload.onprogress = e => {
    if (e.lengthComputable) bar.style.width = (2 + 8 * e.loaded / e.total) + '%';
  };
  xhr.onload = () => {
    let r; try { r = JSON.parse(xhr.responseText); } catch (_) { r = {}; }
    if (!r.id) { fail(r.error || 'upload failed'); return; }
    listen(r.id);
  };
  xhr.onerror = () => fail('upload failed');
  xhr.send(f);
}

function fail(text) {
  stage.textContent = text;
  stage.className = 'error';
  document.querySelector('#stageline .dot').style.animation = 'none';
  line(text, 'error');
  $('logbox').open = true;
}

function listen(id) {
  jobId = id;
  const es = new EventSource('/events/' + id);
  let settled = false;
  es.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.kind === 'stage') {
      stage.textContent = m.text;
      line(m.text);
      const i = STAGES.findIndex(s => m.text.startsWith(s));
      if (i >= 0) bar.style.width = (10 + 88 * (i + 1) / STAGES.length) + '%';
    } else if (m.kind === 'note') {
      line(m.text, 'note');
    } else if (m.kind === 'error') {
      settled = true; es.close(); fail(m.text);
    } else if (m.kind === 'done') {
      settled = true; bar.style.width = '100%'; es.close(); show(id, m.summary);
    }
  };
  // A dropped stream is not a dead job: the reconstruction runs for minutes
  // and any proxy or laptop sleep can cut the pipe. Reconnect; the server
  // replays the terminal event if the job finished while we were gone.
  es.onerror = () => {
    es.close();
    if (!settled) {
      line('progress stream dropped; reconnecting', 'note');
      setTimeout(() => { if (!settled && jobId === id) listen(id); }, 2000);
    }
  };
}

function show(id, s) {
  jobId = id;
  working.classList.add('hidden');
  landing.classList.add('hidden');
  result.classList.remove('hidden');
  $('viewer').src = '/view/' + id + '?embed=1';
  $('topagain').classList.remove('hidden');
  $('covpill').classList.remove('hidden');
  if (!SHOWCASE) initGalBtn(id);

  // The top bar carries the room in one quiet line; the twin speaks for itself.
  const loc = $('toploc');
  loc.classList.remove('hidden');
  $('locname').textContent = s.name || '';
  const meta = [];
  if (s.floor_area_m2 != null) meta.push(s.floor_area_m2 + ' m²');
  if (s.ceiling_height_m != null) meta.push(s.ceiling_height_m.toFixed(2) + ' m ceiling');
  if (s.openings) meta.push(s.openings + (s.openings === 1 ? ' window' : ' windows'));
  if (s.setups) meta.push(s.setups.toLocaleString() + ' setups swept');
  $('locmeta').textContent = meta.join(' · ');
  fetch('/twin-info/' + id).then(r => r.json()).then(r => {
    if (r.name && !s.name) $('locname').textContent = r.name;
  }).catch(() => {});
  if (!s.setups) fetch('/capacity/' + id).then(r => r.json()).then(c => {
    if (c && c.available && c.total) { meta.push(c.total.toLocaleString() + ' setups swept'); $('locmeta').textContent = meta.join(' · '); }
  }).catch(() => {});

  $('heroactions').innerHTML = '<a href="/view/' + id + '" target="_blank">Full screen</a>'
    + (SHOWCASE ? '' : '<a href="/twin/' + id + '">.twin</a>');

  // QA findings stay in the pipeline log, where a crew member who wants them looks.
  const checks = s.checks || {};
  const bad = (checks.fail || []).concat(checks.warn || []);
  if (bad.length) line('QA flagged: ' + bad.join(', '), 'note');

  if (!s.clickhouse) {
    addAgent('The shot table is offline (ClickHouse is not configured), so I can '
      + 'answer from the twin itself — measurements, the scout report, dolly '
      + 'checks — but not sweep-search for setups.', [], true);
  }
  initCoverage(id, s);
  initPackage(id);
  initLabels(id);
  if (typeof initStage === 'function') initStage(id);
  $('send').onclick = ask;
  $('ask').onkeydown = e => { if (e.key === 'Enter') ask(); };
}

function addUser(text) {
  const el = document.createElement('div');
  el.className = 'msg user';
  el.textContent = text;
  $('thread').appendChild(el);
  el.scrollIntoView({behavior: 'smooth', block: 'end'});
}

function esc(t) {
  const d = document.createElement('div'); d.textContent = t; return d.innerHTML;
}

function addAgent(text, trace, plain) {
  const el = document.createElement('div');
  el.className = 'msg agent' + (plain === 'err' ? ' err' : '');
  // Minimal, safe rendering: escape everything, then re-introduce the two
  // shapes the agent produces -- images and bold.
  let html = esc(text)
    .replace(/!\\[[^\\]]*\\]\\(([^)\\s]+)\\)/g, '<img src="$1" loading="lazy">')
    .replace(/\\*\\*([^*]+)\\*\\*/g, '<b>$1</b>');
  el.innerHTML = html;
  if (trace && trace.length) {
    const t = document.createElement('div');
    t.className = 'tools';
    t.innerHTML = trace.map(x => x.kind === 'call'
      ? '&rarr; ' + esc(x.tool) + '(' + esc(JSON.stringify(x.args)).slice(0, 160) + ')'
      : '&larr; ' + esc(x.summary || '')).join('<br>');
    el.appendChild(t);
    // Render any frames the tools produced even if the prose forgot them.
    trace.filter(x => x.kind === 'result' && /^\\/shot-image\\//.test(x.summary || ''))
      .forEach(x => {
        if (html.indexOf(x.summary) === -1) {
          const img = document.createElement('img');
          img.src = x.summary; img.loading = 'lazy';
          el.insertBefore(img, t);
        }
      });
  }
  $('thread').appendChild(el);
  el.scrollIntoView({behavior: 'smooth', block: 'end'});
}

let busy = false;
function ask() {
  const box = $('ask');
  const text = box.value.trim();
  if (!text || busy || !jobId) return;
  busy = true; $('send').disabled = true; box.value = '';
  addUser(text);
  const th = document.createElement('div');
  th.className = 'thinking';
  th.textContent = 'the scout is working…';
  $('thread').appendChild(th);

  fetch('/chat/' + jobId, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: text}),
  }).then(r => r.json()).then(r => {
    th.remove();
    if (r.error) addAgent(r.error, [], 'err');
    else addAgent(r.reply, r.trace);
  }).catch(e => { th.remove(); addAgent(String(e), [], 'err'); })
    .finally(() => { busy = false; $('send').disabled = false; $('ask').focus(); });
}

/* ---------------------------------------------------------------------- *
 * coverage: the scene, scouted in the room
 * ---------------------------------------------------------------------- */

const BRIEFS = [
  'INT. KITCHEN – DAY. MAYA stands at the counter, back to the window, when JON comes in through the door. They argue; she turns away to the sink. End on her hands gripping the counter edge.',
  ['1. Master wide: MAYA at the counter, JON enters', '2. Over JON\\'s shoulder onto MAYA, medium close-up', '3. Close-up MAYA, no window behind', '4. Close-up JON, no window behind', '5. Low angle: JON looms over her, medium', '6. Two-shot MAYA and JON, medium'].join(String.fromCharCode(10)),
];
const LENSES = [16, 25, 35, 50, 75, 100];
const SIZE_NAMES = {ecu: 'Extreme close-up', bcu: 'Big close-up', cu: 'Close-up', mcu: 'Medium close-up',
                    ms: 'Medium shot', mls: 'Medium long shot', ls: 'Long shot', els: 'Extreme long shot'};
const KEY_WORDS = {'three-quarter': 'three-quarter key', side: 'side key', rim: 'rim from the window',
                   back: 'window behind', front: 'flat front light'};
let covJob = null, covSummary = null, floorZ = 0, planT0 = 0, currentPlan = null, currentPlanId = null;
let planBusy = false, activeShot = null, exampleIdx = 0, lastPlanSeconds = null;

function heightWord(z) { z = Number(z); return (z < 0.8 ? 'low' : z < 1.3 ? 'mid' : 'eye level') + ' (' + z.toFixed(2) + ' m)'; }
function titleOf(brief) {
  const first = (brief.split(/\\r?\\n/)[0] || '').replace(/^\\s*\\d+[\\.\\)]?\\s*/, '').trim();
  return first.length > 60 ? first.slice(0, 57).trim() + '…' : first || 'Untitled scene';
}

function initCoverage(id, s) {
  covJob = id; covSummary = s || {};
  const pill = $('covpill');
  const agent = !!covSummary.coverage_agent, ch = !!covSummary.clickhouse;
  pill.textContent = agent && ch ? 'Gemini · ClickHouse' : ch ? 'ClickHouse' : 'local sweep';
  pill.className = 'pill' + (agent && ch ? ' on' : '');
  $('modenote').textContent = agent
    ? 'Gemini breakdown + frame review · ' + (ch ? 'ClickHouse shot table' : 'local sweep')
    : 'Gemini not connected — planning from a shot list, one shot per line' + (ch ? '' : ' · local sweep');
  $('example').onclick = e => { e.preventDefault(); $('brief').value = BRIEFS[exampleIdx % BRIEFS.length]; exampleIdx++; };
  $('planbtn').onclick = startPlan;
  fetch('/twin-info/' + id).then(r => r.json()).then(r => { if (r.floor_z != null) floorZ = r.floor_z; }).catch(() => {});
  fetch('/plans/' + id).then(r => r.json()).then(r => {
    const plans = (r.plans || []);
    const row = $('prevplans');
    row.innerHTML = '';
    if (!plans.length) { row.classList.add('hidden'); return; }
    row.classList.remove('hidden');
    row.appendChild(document.createTextNode('Recent:'));
    plans.slice().reverse().slice(0, 5).forEach(p => {
      const a = document.createElement('a');
      a.href = '#';
      a.textContent = (p.title || 'untitled') + ' · ' + p.planned + '/' + p.shots;
      a.onclick = e => { e.preventDefault(); loadPlan(p.plan_id, true); };
      row.appendChild(a);
    });
    if (!new URLSearchParams(location.search).get('demo')) loadPlan(plans[plans.length - 1].plan_id);
  }).catch(() => {});
}

/* --- the trace: everything the planner and the agent did ----------------- */

function traceLine(text, cls, sql) {
  const body = $('tracebody');
  const el = document.createElement('div');
  const dt = planT0 ? ((Date.now() - planT0) / 1000).toFixed(0) + 's' : '';
  const t = document.createElement('span'); t.className = 't'; t.textContent = dt;
  const b = document.createElement('span'); b.className = cls || ''; b.textContent = text;
  el.appendChild(t); el.appendChild(b);
  body.appendChild(el);
  if (sql) { const pre = document.createElement('pre'); pre.textContent = sql; body.appendChild(pre); }
  body.scrollTop = body.scrollHeight;
}

/* --- the caption over the twin: what the scout is doing right now --------- */

// Which model is reading the frames, as /labels reports it ('meta' or
// 'gemini'); the badges print its name rather than a hard-coded one.
let visionProvider = '';
const PROVIDER_NAMES = {meta: 'Muse Spark', gemini: 'Gemini'};
function providerName(p) { return PROVIDER_NAMES[p] || (p ? String(p) : 'the vision model'); }

function caption(head, text, model) {
  const cap = $('scoutcap');
  if (head == null) { cap.classList.add('hidden'); return; }
  cap.classList.remove('hidden');
  const badge = model ? '<span class="g">' + esc(providerName(model === true ? visionProvider : model)) + '</span>' : '';
  cap.innerHTML = '<b>' + esc(head) + '</b><span>' + badge + esc(text || '') + '</span>';
}

/* --- the walk: the viewer flies through what the scout is looking at ------- */

// Events arrive faster than a human can watch a camera move, so the walk is
// a queue: each step holds the view for a beat, and once the plan is done
// the remaining steps drain quickly rather than being skipped.
let walk = [], walking = false, walkFast = false;
function enqueue(fn, ms) { walk.push({fn, ms}); if (!walking) pump(); }
function pump() {
  const step = walk.shift();
  if (!step) { walking = false; return; }
  walking = true;
  try { step.fn(); } catch (_) {}
  setTimeout(pump, walkFast ? 120 : step.ms);
}
function lookAt(row, label, marks) {
  postViewer({
    type: 'locaish:viewfinder',
    cam: [row.cam[0], row.cam[1], floorZ + row.cam[2]],
    subj: [row.subject[0], row.subject[1]],
    focal_mm: row.focal_mm, sensor: 'super35', label, marks: marks || [],
  });
}

function startPlan() {
  const brief = $('brief').value.trim();
  if (!brief || planBusy || !covJob) return;
  planBusy = true; $('planbtn').disabled = true;
  planT0 = Date.now(); lastPlanSeconds = null;
  walk = []; walkFast = false;
  $('tracebody').innerHTML = '';
  $('trace').open = false;
  $('planout').classList.add('hidden');
  $('shots').innerHTML = '';
  $('planhead').innerHTML = '';
  currentPlan = null;
  caption('Scouting', 'reading the scene…');
  $('viewerwrap').scrollIntoView({behavior: 'smooth', block: 'start'});
  traceLine('scouting ' + titleOf(brief));
  fetch('/plan/' + covJob, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({brief, title: titleOf(brief), mode: 'auto'}),
  }).then(r => r.json()).then(r => {
    if (!r.plan_id) { planFail(r.error || 'could not start the plan'); return; }
    currentPlanId = r.plan_id;
    traceLine('planner: ' + (r.mode === 'agent' ? 'Gemini agent over ClickHouse' : 'shot list'));
    listenPlan(r.plan_id);
  }).catch(e => planFail(String(e)));
}

function planFail(text) {
  traceLine(text, 'error');
  caption('Could not scout', text);
  $('trace').open = true;
  $('planout').classList.remove('hidden');
  planBusy = false; $('planbtn').disabled = false;
}

function listenPlan(pid) {
  const es = new EventSource('/plan-events/' + covJob + '/' + pid);
  let settled = false;
  const shotLabel = n => 'Shot ' + n;
  es.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.kind === 'stage') { traceLine(m.text); enqueue(() => caption('Scouting', m.text), 300); }
    else if (m.kind === 'note') traceLine(m.text, 'note');
    else if (m.kind === 'agent') { traceLine((m.agent || 'agent') + ': ' + (m.text || ''), 'ag'); enqueue(() => caption(m.agent || 'agent', m.text || ''), 900); }
    else if (m.kind === 'call') {
      const isSql = m.tool === 'run_query' && m.args && m.args.query;
      traceLine('→ ' + m.tool + (isSql ? '' : '(' + JSON.stringify(m.args || {}).slice(0, 200) + ')'), 'sql', isSql ? m.args.query : null);
      if (isSql) enqueue(() => caption('Scouting', 'asking ClickHouse: ' + m.args.query.slice(0, 120) + '…'), 700);
    }
    else if (m.kind === 'result') {
      traceLine('← ' + (m.summary || m.tool || ''));
      const mm = /review (\\S+)\\/10 (\\w+)/.exec(m.summary || '');
      if (m.tool === 'place_shot' && mm) enqueue(() => caption('Looking at the frame', mm[1] + '/10 · ' + mm[2], true), 1400);
    }
    else if (m.kind === 'candidates') {
      const rows = m.rows || [];
      traceLine('shot ' + m.shot + ': ' + rows.length + ' candidates of ' + (m.matched || 0) + ' matched', '', m.sql || null);
      const src = covSummary.clickhouse ? 'ClickHouse' : 'the sweep';
      enqueue(() => caption(shotLabel(m.shot), 'looking at ' + (m.matched || 0) + ' setups from ' + src + '…'), 300);
      rows.slice(0, 4).forEach((row, i) => enqueue(() => {
        lookAt(row, shotLabel(m.shot) + ' · trying ' + Math.round(row.focal_mm) + ' mm');
        caption(shotLabel(m.shot), 'trying ' + Math.round(row.focal_mm) + ' mm from ' + row.distance_m.toFixed(1) + ' m'
          + (row.key_quality && KEY_WORDS[row.key_quality] ? ' · ' + KEY_WORDS[row.key_quality] : '')
          + (row.background_depth_m ? ' · ' + row.background_depth_m.toFixed(1) + ' m behind' : ''));
      }, 700));
    }
    else if (m.kind === 'shot') {
      const ps = m.shot, st = ps.setup;
      enqueue(() => {
        $('planout').classList.remove('hidden');
        if (!$('planhead').innerHTML) $('planhead').innerHTML = '<b>' + esc(titleOf($('brief').value)) + '</b><span>scouting…</span>';
        shotCard(ps, {plan_id: pid});
        if (st) {
          lookAt({cam: [st.cam_x, st.cam_y, st.cam_z], subject: [st.subj_x, st.subj_y], focal_mm: st.focal_mm},
                 shotLabel(ps.shot.number) + ' · ' + (SIZE_NAMES[st.shot_size] || st.shot_size) + ' · ' + Math.round(st.focal_mm) + ' mm',
                 ps.second_mark ? [ps.second_mark] : []);
          caption(shotLabel(ps.shot.number) + ' · ' + (SIZE_NAMES[st.shot_size] || st.shot_size), ps.why || '');
        } else {
          caption(shotLabel(ps.shot.number), 'no setup in this room holds this shot');
        }
      }, st ? 2200 : 900);
    }
    else if (m.kind === 'error') { settled = true; es.close(); planFail(m.text); }
    else if (m.kind === 'done') {
      settled = true; es.close();
      if (m.floor_z != null) floorZ = m.floor_z;
      lastPlanSeconds = Math.round((Date.now() - planT0) / 1000);
      traceLine('done: ' + m.plan.planned + ' of ' + m.plan.shots.length + ' shots placed in ' + lastPlanSeconds + ' s');
      planBusy = false; $('planbtn').disabled = false;
      currentPlanId = pid;
      walkFast = true;
      enqueue(() => { walkFast = false; renderPlan(m.plan, floorZ); caption('Scouted', m.plan.planned + ' of ' + m.plan.shots.length + ' shots placed'); }, 0);
    }
  };
  es.onerror = () => {
    es.close();
    if (!settled) { traceLine('plan stream dropped; reconnecting', 'note'); setTimeout(() => { if (!settled) listenPlan(pid); }, 2000); }
  };
}

function loadPlan(pid, scroll) {
  fetch('/plan/' + covJob + '/' + pid).then(r => r.json()).then(r => {
    if (!r.plan) { traceLine(r.error || 'could not load that plan', 'error'); return; }
    currentPlanId = pid;
    lastPlanSeconds = null;
    if (r.floor_z != null) floorZ = r.floor_z;
    $('shots').innerHTML = '';
    $('tracebody').innerHTML = '';
    (r.plan.trace || []).forEach(x => {
      if (x.kind === 'call') traceLine('→ ' + x.tool + ((x.tool === 'run_query' && x.args && x.args.query) ? '' : '(' + JSON.stringify(x.args || {}).slice(0, 200) + ')'), 'sql', (x.tool === 'run_query' && x.args && x.args.query) ? x.args.query : null);
      else if (x.kind === 'result') traceLine('← ' + (x.summary || ''));
    });
    r.plan.shots.forEach(ps => { if (ps.sql) traceLine('shot ' + ps.shot.number + ': ' + (ps.candidates || 0) + ' setups matched', '', ps.sql); });
    renderPlan(r.plan, floorZ);
    if (scroll) $('planout').scrollIntoView({behavior: 'smooth', block: 'start'});
  }).catch(e => traceLine(String(e), 'error'));
}

function fmtm(v, d) { return v == null ? '—' : Number(v).toFixed(d == null ? 2 : d) + ' m'; }

function shotCard(ps, plan) {
  const sh = ps.shot, st = ps.setup;
  const grid = $('shots');
  let card = grid.querySelector('[data-shot="' + sh.number + '"]');
  if (!card) {
    card = document.createElement('div');
    card.dataset.shot = sh.number;
    grid.appendChild(card);
  }
  card.className = 'shot' + (st ? '' : ' none') + (activeShot === sh.number ? ' active' : '');
  if (!st) {
    card.innerHTML = '<b>#' + sh.number + ' · ' + esc(SIZE_NAMES[sh.size] || sh.size) + ' of ' + esc(sh.subject)
      + '</b><div class="desc">' + esc(sh.description) + '</div>'
      + '<div class="why">No setup in this room holds this shot with a clear sightline.</div>';
    return card;
  }
  const dof = st.dof_infinite ? 'DoF to infinity' : 'DoF ' + Number(st.dof_near_m).toFixed(2) + '–' + Number(st.dof_far_m).toFixed(2) + ' m';
  const bits = [fmtm(st.distance_m, 1) + ' to ' + esc(sh.subject) + (sh.second_subject ? ' + ' + esc(sh.second_subject) : ''), dof];
  if (st.window_behind_subject) bits.push('<span class="warnc">window behind</span>');
  else if (st.key_quality && KEY_WORDS[st.key_quality]) bits.push(KEY_WORDS[st.key_quality]);
  if (st.background_depth_m != null && st.background_depth_m < 11.9) bits.push(Number(st.background_depth_m).toFixed(1) + ' m behind');
  const img = ps.frame ? '<img src="/plan-image/' + covJob + '/' + (plan.plan_id || currentPlanId) + '/' + encodeURIComponent(ps.frame) + '" loading="lazy" alt="shot ' + sh.number + '">' : '';
  let html = '<a class="frame" title="look through this setup in the viewer">' + img + '<span class="look">Look through it</span></a>'
    + '<div class="body"><h3>#' + sh.number + ' · ' + esc(SIZE_NAMES[st.shot_size] || st.shot_size) + ' · ' + Math.round(st.focal_mm) + ' mm · ' + heightWord(st.cam_z) + '</h3>'
    + '<div class="meas">' + bits.join(' · ') + '</div>'
    + '<div class="desc">' + esc(sh.description) + '</div>'
    + (ps.why ? '<div class="why">' + esc(ps.why) + '</div>' : '');
  if (ps.relaxed && ps.relaxed.length) html += '<div class="relax">relaxed: ' + esc(ps.relaxed.join('; ')) + '</div>';
  if (ps.review) {
    html += '<div class="review"><span class="g">Gemini</span>' + Number(ps.review.score).toFixed(0) + '/10 · ' + esc(ps.review.verdict || '') + ' — ' + esc(ps.review.notes || '') + '</div>';
  }
  html += '</div>';
  card.innerHTML = html;
  card.querySelector('.frame').onclick = e => { e.preventDefault(); lookThrough(ps); };
  return card;
}

function renderPlan(plan, fz) {
  currentPlan = plan;
  $('planout').classList.remove('hidden');
  $('planhead').innerHTML = '<b>' + esc(plan.title || 'Untitled scene') + '</b><span>'
    + plan.planned + ' of ' + plan.shots.length + ' shots'
    + (lastPlanSeconds != null ? ' · ' + lastPlanSeconds + ' s' : '')
    + (plan.planner && plan.planner !== 'deterministic' ? ' · ' + esc(providerName(plan.planner)) : '') + '</span>';
  plan.shots.forEach(ps => shotCard(ps, plan));
  $('floorplan').innerHTML = plan.floor_plan_svg || '<div class="dim">no camera plan</div>';
  const base = '/plan/' + covJob + '/' + plan.plan_id;
  $('planlinks').innerHTML = '<a href="' + base + '/shotlist.txt" target="_blank">Shot list (.txt)</a>'
    + '<a href="' + base + '/floorplan.svg" target="_blank">Camera plan (.svg)</a>'
    + '<a href="' + base + '/plan.json">Plan (.json)</a>';
  loadCapacity();
}

function loadCapacity() {
  fetch('/capacity/' + covJob).then(r => r.json()).then(renderCapacity).catch(() => {
    $('capacity').innerHTML = '<div class="foot">the shot table is local to this studio</div>';
  });
}

function renderCapacity(cap) {
  const box = $('capacity');
  if (!cap || !cap.available) {
    box.innerHTML = '<div class="foot">the shot table is local to this studio' + (cap && cap.reason ? ' — ' + esc(cap.reason) : '') + '</div>';
    return;
  }
  const sizes = cap.sizes && cap.sizes.length ? cap.sizes : Object.keys(SIZE_NAMES);
  const lenses = cap.lenses && cap.lenses.length ? cap.lenses : LENSES;
  const cells = {};
  let max = 0;
  (cap.cells || []).forEach(c => { cells[c.shot_size + '|' + Math.round(c.focal_mm)] = c; max = Math.max(max, c.clean || 0); });
  let html = '<div class="lead"><b>' + (cap.total || 0).toLocaleString() + '</b> setups swept · <b>'
    + (cap.clean || 0).toLocaleString() + '</b> clean</div>';
  html += '<table><tr><th class="row"></th>' + lenses.map(l => '<th>' + Math.round(l) + 'mm</th>').join('') + '</tr>';
  sizes.forEach(s => {
    html += '<tr><th class="row" title="' + esc(SIZE_NAMES[s] || s) + '">' + esc(s.toUpperCase()) + '</th>';
    lenses.forEach(l => {
      const c = cells[s + '|' + Math.round(l)];
      const v = c ? (c.clean || 0) : 0;
      const a = max ? v / max : 0;
      html += '<td style="background: rgba(255,255,255,' + (a * 0.85).toFixed(3) + ')" title="'
        + (c ? c.total + ' setups, ' + c.clean + ' clean, ' + c.backlit + ' backlit' : 'none') + '">' + (v ? v : '·') + '</td>';
    });
    html += '</tr>';
  });
  html += '</table><div class="foot">clean = clear sightline, no window behind the subject · from ClickHouse';
  if (cap.locations && cap.locations.length > 1) {
    html += '<ol>' + cap.locations.slice().sort((a, b) => (b.clean || 0) - (a.clean || 0)).map(l =>
      '<li>' + esc(l.location) + ' — ' + (l.clean || 0).toLocaleString() + ' clean of ' + (l.setups || 0).toLocaleString() + '</li>').join('') + '</ol>';
  }
  html += '</div>';
  box.innerHTML = html;
}

/* --- the viewfinder: look through a planned setup ------------------------ */

// The viewer is a large self-contained page; a message posted before its
// script has run is simply dropped, so anything sent early waits for load.
let viewerReady = false, viewerQueue = [];
$('viewer').addEventListener('load', () => { viewerReady = true; viewerQueue.splice(0).forEach(m => postViewer(m)); });
function viewerWin() { const f = $('viewer'); return f && f.contentWindow; }
function postViewer(m) {
  const w = viewerWin();
  if (!viewerReady || !w) { viewerQueue.push(m); return; }
  w.postMessage(m, '*');
}

function lookThrough(ps) {
  const st = ps.setup;
  if (!st) return;
  activeShot = ps.shot.number;
  document.querySelectorAll('#shots .shot').forEach(c => c.classList.toggle('active', c.dataset.shot == ps.shot.number));
  postViewer({
    type: 'locaish:viewfinder',
    cam: [Number(st.cam_x), Number(st.cam_y), floorZ + Number(st.cam_z)],
    subj: [Number(st.subj_x), Number(st.subj_y)],
    focal_mm: Number(st.focal_mm), sensor: st.sensor || 'super35',
    label: 'Shot ' + ps.shot.number + ' · ' + (SIZE_NAMES[st.shot_size] || st.shot_size) + ' · ' + Math.round(st.focal_mm) + ' mm',
    marks: ps.second_mark ? [ps.second_mark] : [],
  });
  caption('Shot ' + ps.shot.number + ' · ' + (SIZE_NAMES[st.shot_size] || st.shot_size), ps.why || '');
  $('viewerwrap').scrollIntoView({behavior: 'smooth', block: 'start'});
}

let vfState = null, vfTimer = null, vfNear = '';
window.addEventListener('message', ev => {
  const m = ev.data;
  if (!m || m.type !== 'locaish:viewfinder-state') return;
  vfState = m;
  renderVfBar();
  if (m.on) {
    clearTimeout(vfTimer);
    vfTimer = setTimeout(nearestSetup, 300);
  }
});

function renderVfBar() {
  const bar = $('vfbar');
  if (!vfState || !vfState.on) { bar.classList.add('hidden'); vfNear = ''; if (!planBusy) caption(null); return; }
  bar.classList.remove('hidden');
  let html = '<b>Viewfinder</b><span>' + Math.round(vfState.focal_mm) + ' mm · ' + fmtm(vfState.distance_m) + ' to subject · h ' + fmtm(vfState.height_m) + '</span>';
  html += LENSES.map(l => '<button class="lens" data-lens="' + l + '" aria-pressed="' + (Math.round(vfState.focal_mm) === l) + '">' + l + '</button>').join('');
  html += '<span class="near">' + esc(vfNear || 'nearest swept setup: …') + '</span>';
  html += '<button class="exit">Exit</button>';
  bar.innerHTML = html;
  bar.querySelectorAll('.lens').forEach(b => b.onclick = () => postViewer({type: 'locaish:lens', focal_mm: Number(b.dataset.lens)}));
  bar.querySelector('.exit').onclick = () => {
    postViewer({type: 'locaish:viewfinder', off: true});
    activeShot = null;
    document.querySelectorAll('#shots .shot').forEach(c => c.classList.remove('active'));
  };
}

function nearestSetup() {
  if (!vfState || !vfState.on || !covJob) return;
  const c = vfState.cam;
  const q = '?x=' + c[0].toFixed(3) + '&y=' + c[1].toFixed(3) + '&z=' + (c[2] - floorZ).toFixed(3) + '&focal=' + Math.round(vfState.focal_mm);
  fetch('/setup-near/' + covJob + q).then(r => r.json()).then(r => {
    const s = r.setup;
    if (!s) { vfNear = 'off the swept grid'; }
    else {
      const away = s.away_m != null ? s.away_m : Math.hypot(s.cam_x - c[0], s.cam_y - c[1], s.cam_z - (c[2] - floorZ));
      vfNear = 'nearest swept setup: ' + String(s.shot_size).toUpperCase() + ' · score ' + Math.round(s.score) + ' · ' + Number(away).toFixed(2) + ' m away'
        + (s.window_behind_subject ? ' · backlit' : '');
    }
    const el = document.querySelector('#vfbar .near');
    if (el) el.textContent = vfNear;
  }).catch(() => {});
}

/* --- what is in the room: model labels, click-to-name ---------------------- */

let lblJob = null, lblList = [], lblPick = null, lblView = null, lblMode = false;
function initLabels(id) {
  lblJob = id;
  loadLabels();
  $('detectbtn').onclick = detectLabels;
  $('labelbtn').onclick = () => setLabelMode(!lblMode);
  $('lpsave').onclick = saveLabel;
  $('lpcancel').onclick = () => { lblPick = null; $('labelpick').classList.add('hidden'); setLabelMode(false); };
  $('lpname').onkeydown = e => { if (e.key === 'Enter') saveLabel(); };
}
function setLabelMode(on) {
  lblMode = on;
  $('labelbtn').setAttribute('aria-pressed', String(on));
  $('labelbtn').textContent = on ? 'Click the thing in the twin…' : 'Label an object';
  postViewer({type: 'locaish:labelmode', on});
  if (on) $('viewerwrap').scrollIntoView({behavior: 'smooth', block: 'start'});
}
function loadLabels() {
  fetch('/labels/' + lblJob).then(r => r.json()).then(r => {
    lblList = r.labels || [];
    paintChips();
    postViewer({type: 'locaish:labels', labels: lblList});
    if (r.vision_provider) visionProvider = r.vision_provider;
    if (r.gemini) $('detectbtn').textContent = 'Name things with ' + providerName(visionProvider);
    if (r.frames === false) { $('detectbtn').disabled = true; $('detectbtn').title = 'this twin came from a scan file, not a video: no frames to look at'; $('labelbtn').disabled = true; }
    else if (!r.gemini) { $('detectbtn').disabled = true; $('detectbtn').title = 'no vision model configured'; }
  }).catch(() => {});
}
function paintChips() {
  const box = $('chips');
  if (!lblList.length) { box.innerHTML = '<span class="dim">nothing named yet</span>'; return; }
  box.innerHTML = '';
  lblList.forEach(l => {
    const c = document.createElement('span');
    c.className = 'chip' + (l.source === 'user' ? '' : ' model');
    c.title = (l.source === 'user' ? 'named by a person' : 'named by the model in ' + (l.confidence || 1) + ' frame(s)') + ' · ' + l.xyz.map(v => Number(v).toFixed(2)).join(', ') + ' m';
    c.innerHTML = esc(l.label) + (l.source === 'user' ? '' : '<span class="k">?</span>')
      + (l.source === 'user' ? '' : '<button class="ok" data-ok="' + l.id + '" title="confirm">✓</button>')
      + '<button data-look="' + l.id + '" title="show the photograph">◉</button>'
      + '<button data-rm="' + l.id + '" title="remove">×</button>';
    box.appendChild(c);
  });
  box.querySelectorAll('[data-ok]').forEach(b => b.onclick = () => fetch('/labels/' + lblJob + '/' + b.dataset.ok + '/confirm', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'}).then(r => r.json()).then(r => { if (r.labels) { lblList = r.labels; paintChips(); postViewer({type: 'locaish:labels', labels: lblList}); } }));
  box.querySelectorAll('[data-rm]').forEach(b => b.onclick = () => fetch('/labels/' + lblJob + '/' + b.dataset.rm, {method: 'DELETE'}).then(() => loadLabels()));
  box.querySelectorAll('[data-look]').forEach(b => b.onclick = () => { const l = lblList.find(x => x.id === b.dataset.look); if (l) showPick(l.xyz, l); });
}
function detectLabels() {
  $('detectbtn').disabled = true; $('lstatus').textContent = providerName(visionProvider) + ' is looking at eight frames of the capture…';
  fetch('/labels/' + lblJob + '/detect', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'})
    .then(r => r.json()).then(r => {
      $('detectbtn').disabled = false;
      if (r.error) { $('lstatus').textContent = r.error; return; }
      $('lstatus').textContent = r.objects + ' objects named from ' + r.frames_looked_at + ' frames (' + r.detections + ' sightings). Confirm the right ones, remove the wrong ones.';
      loadLabels();
    }).catch(e => { $('detectbtn').disabled = false; $('lstatus').textContent = String(e); });
}
window.addEventListener('message', ev => {
  const m = ev.data;
  if (!m || typeof m !== 'object') return;
  if (m.type === 'locaish:pick' && m.p) showPick(m.p, null);
  else if (m.type === 'locaish:labelmode') { lblMode = !!m.on; $('labelbtn').setAttribute('aria-pressed', String(lblMode)); $('labelbtn').textContent = lblMode ? 'Click the thing in the twin…' : 'Label an object'; }
});
function showPick(p, existing) {
  lblPick = p; lblView = null;
  const box = $('labelpick');
  box.classList.remove('hidden');
  $('lpmeta').textContent = 'point ' + p.map(v => Number(v).toFixed(2)).join(', ') + ' m · finding the frames that looked here…';
  $('crops').innerHTML = '';
  $('lpname').value = existing ? existing.label : '';
  if (existing && existing.kind) $('lpkind').value = existing.kind;
  box.scrollIntoView({behavior: 'smooth', block: 'center'});
  fetch('/label-views/' + lblJob + '?x=' + p[0] + '&y=' + p[1] + '&z=' + p[2]).then(r => r.json()).then(r => {
    const views = r.views || [];
    if (!views.length) { $('lpmeta').textContent = 'no frame of the capture had a clear view of that spot — name it anyway, or pick a point the camera saw'; return; }
    $('lpmeta').textContent = 'the capture saw this spot from ' + views.length + ' angle' + (views.length === 1 ? '' : 's') + ' — pick the clearest photograph, then name it';
    $('crops').innerHTML = views.map((v, i) => '<div><img src="' + v.url + '" data-i="' + i + '" class="' + (i === 0 ? 'on' : '') + '" alt=""><div class="cap">' + esc(v.frame) + ' · ' + v.depth_m + ' m away</div></div>').join('');
    lblView = views[0];
    $('crops').querySelectorAll('img').forEach(im => im.onclick = () => { $('crops').querySelectorAll('img').forEach(x => x.classList.remove('on')); im.classList.add('on'); lblView = views[Number(im.dataset.i)]; });
    if (!existing) $('lpname').focus();
  }).catch(() => { $('lpmeta').textContent = 'could not load the frames'; });
}
function saveLabel() {
  const name = $('lpname').value.trim();
  if (!name || !lblPick) return;
  const body = {label: name, kind: $('lpkind').value, xyz: lblPick, frame: lblView ? lblView.frame : null, uv: lblView ? [lblView.u, lblView.v] : null};
  fetch('/labels/' + lblJob, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})
    .then(r => r.json()).then(r => {
      if (r.error) { $('lstatus').textContent = r.error; return; }
      lblList = r.labels || lblList; paintChips(); postViewer({type: 'locaish:labels', labels: lblList});
      $('lstatus').textContent = '"' + name + '" saved at ' + lblPick.map(v => Number(v).toFixed(2)).join(', ') + ' m';
      lblPick = null; $('labelpick').classList.add('hidden'); setLabelMode(false);
    }).catch(e => $('lstatus').textContent = String(e));
}

/* --- the location package: details, the sun preview, the share link -------- */

const DETAIL_KEYS = ['display_name','address','contact','power','parking','access','restrictions','fee','notes','heading_deg','latitude','longitude'];
let pkgJob = null, pkgHour = 12, pkgHasGeo = false;
function pkgDate() { return $('d_shoot_date').value || new Date().toISOString().slice(0, 10); }
function initPackage(id) {
  pkgJob = id;
  try { const d = localStorage.getItem('locaish:shoot_date:' + id); if (d) $('d_shoot_date').value = d; } catch (_) {}
  fetch('/details/' + id).then(r => r.json()).then(r => {
    const d = r.details || {};
    DETAIL_KEYS.forEach(k => { const el = $('d_' + k); if (el && d[k] != null) el.value = d[k]; });
    if (r.georeference) {
      pkgHasGeo = true;
      if (!$('d_latitude').value) $('d_latitude').value = Number(r.georeference.latitude).toFixed(6);
      if (!$('d_longitude').value) $('d_longitude').value = Number(r.georeference.longitude).toFixed(6);
      if (!$('d_heading_deg').value && r.georeference.heading_source === 'user') $('d_heading_deg').value = Math.round(r.georeference.heading_deg);
    }
    if ((r.shares || []).length) showShare(r.shares[0]);
    paintSun();
  }).catch(() => paintSun());
  $('d_save').onclick = saveDetails;
  $('d_shoot_date').onchange = () => { try { localStorage.setItem('locaish:shoot_date:' + id, $('d_shoot_date').value); } catch (_) {} paintSun(); };
  $('pkgshare').onclick = () => fetch('/share', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({kind: 'package', target: id})}).then(r => r.json()).then(r => { if (!r.error) showShare(r); }).catch(() => {});
  paintSun();
}
function showShare(rec) {
  const url = location.origin + '/s/' + rec.token;
  const box = $('sharebox');
  box.classList.remove('hidden');
  box.innerHTML = 'Share link (read-only, no account needed): <a href="' + url + '" target="_blank">' + url + '</a>'
    + ' · opened ' + (rec.opens || 0) + ' time' + (rec.opens === 1 ? '' : 's') + ' · <a href="#" id="revoke">revoke</a>';
  navigator.clipboard && navigator.clipboard.writeText(url).catch(() => {});
  $('revoke').onclick = e => { e.preventDefault(); fetch('/share/' + rec.token, {method: 'DELETE'}).then(() => { box.textContent = 'Link revoked.'; }); };
}
function saveDetails() {
  const body = {};
  DETAIL_KEYS.forEach(k => { const el = $('d_' + k); if (el) body[k] = el.value; });
  $('d_status').textContent = 'saving…';
  fetch('/details/' + pkgJob, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})
    .then(r => r.json()).then(r => {
      if (r.error) { $('d_status').textContent = r.error; return; }
      $('d_status').textContent = 'saved';
      if (r.display_name) $('locname').textContent = r.display_name;
      setTimeout(() => $('d_status').textContent = '', 1500);
      pkgHasGeo = !!(body.latitude && body.longitude) || pkgHasGeo;
      paintSun();
    }).catch(e => $('d_status').textContent = String(e));
}
function paintSun() {
  if (!pkgJob) return;
  const date = pkgDate();
  $('pkgopen').href = '/package/' + pkgJob + '?date=' + date;
  const hrs = $('hours');
  hrs.innerHTML = [8, 10, 12, 14, 16, 18].map(h => '<button data-h="' + h + '" aria-pressed="' + (h === pkgHour) + '">' + h + ':00</button>').join('')
    + '<span class="sunline" id="sunline"></span>';
  hrs.querySelectorAll('button').forEach(b => b.onclick = () => { pkgHour = Number(b.dataset.h); paintSun(); });
  $('sunimg').src = '/floorplan/' + pkgJob + '?date=' + date + '&hour=' + pkgHour + '&t=' + Date.now();
  fetch('/sun/' + pkgJob + '?date=' + date).then(r => r.json()).then(sun => {
    const el = $('sunline');
    if (!el) return;
    if (sun.error) { el.textContent = 'set latitude, longitude and heading for the sun'; return; }
    const up = (sun.sun_up || []).map(g => g.from + '–' + g.to).join(', ') || 'never up';
    const lit = (sun.windows || []).filter(w => w.direct_sun.length).length;
    el.textContent = date + ': sun ' + up + ' solar · ' + lit + ' window' + (lit === 1 ? '' : 's') + ' get direct sun' + (sun.heading_caveat ? ' · heading assumed' : '');
  }).catch(() => {});
}

/* --- gallery approval: only the best scans go up ------------------------- */
let galApproved = false;
function paintGalBtn() {
  const b = $('galbtn');
  b.className = galApproved ? 'on' : '';
  b.textContent = galApproved ? '\u2713 In the gallery \u2014 remove' : 'Approve for the gallery';
}
function setApproved(id, flag) {
  return fetch('/approve/' + id, {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({approved: flag})}).then(r => r.json());
}
function initGalBtn(id) {
  fetch('/jobs').then(r => r.json()).then(r => {
    const j = (r.jobs || []).find(x => x.id === id);
    galApproved = !!(j && j.approved);
    paintGalBtn();
  }).catch(() => {});
  $('galbtn').onclick = () => setApproved(id, !galApproved)
    .then(r => { if (!r.error) { galApproved = !!r.approved; paintGalBtn(); } }).catch(() => {});
}

/* --- the landing library: every scanned location ------------------------- */
(function () {
  if (new URLSearchParams(location.search).get('job')) return;
  if (SHOWCASE) {
    $('drop').classList.add('hidden');
    $('geo').classList.add('hidden');
    $('lead').textContent = 'A location scout that has measured the room. '
      + 'Each location below was scanned from a sixty-second phone walkthrough. '
      + 'Open one, describe your scene, and look through the lenses.';
    $('steps').innerHTML = '<div><b>1 · Scanned</b>a phone walkthrough became a metric twin</div>'
      + '<div><b>2 · Swept</b>every camera position and lens, into ClickHouse</div>'
      + '<div><b>3 · Scouted</b>Gemini places, frames and reviews every shot</div>';
  }
  fetch('/jobs').then(r => r.json()).then(r => {
    const done = (r.jobs || []).filter(j => j.state === 'done');
    if (!done.length) return;
    const lib = $('library');
    lib.classList.remove('hidden');
    lib.innerHTML = '<div class="libhead">' + (SHOWCASE ? 'Locations' : 'Scanned locations')
      + '</div><div id="rooms"></div>'
      + (SHOWCASE ? '<p class="libfoot">Scanned with the Locaish pipeline — classical reconstruction, '
        + 'no neural models — then swept into ClickHouse and scouted by Gemini. '
        + 'Scanning runs in the local studio; this gallery is the result.</p>' : '');
    const rooms = document.getElementById('rooms');
    done.forEach(j => {
      const s = j.summary || {};
      const name = String(j.name || j.id).replace(/\\.[a-z0-9]+$/i, '').replace(/[_]+/g, ' ');
      const meta = [];
      if (s.floor_area_m2 != null) meta.push(s.floor_area_m2 + ' m²');
      if (s.ceiling_height_m != null) meta.push(Number(s.ceiling_height_m).toFixed(1) + ' m ceiling');
      if (s.openings) meta.push(s.openings + (s.openings === 1 ? ' window' : ' windows'));
      const a = document.createElement('a');
      a.className = 'room'; a.href = '/?job=' + j.id;
      a.innerHTML = '<span class="thumb"><img loading="lazy" alt="" src="/thumb/' + j.id + '"></span>'
        + '<span class="rbody"><b>' + esc(name) + '</b><span class="rmeta">'
        + esc(meta.join(' · ') || 'scanned location') + '</span></span>';
      const im = a.querySelector('img');
      im.onerror = () => im.remove();
      if (!SHOWCASE) {
        const g = document.createElement('span');
        g.className = 'gal';
        const paint = ok => {
          g.innerHTML = ok
            ? '<button class="in">\u2713 In gallery</button><button data-off="1">Remove</button>'
            : '<button data-on="1">Add to gallery</button>';
          g.querySelectorAll('button').forEach(b => b.onclick = e => {
            e.preventDefault(); e.stopPropagation();
            if (b.className === 'in') return;
            setApproved(j.id, !!b.dataset.on)
              .then(x => { if (!x.error) paint(!!x.approved); }).catch(() => {});
          });
        };
        paint(!!j.approved);
        a.appendChild(g);
      }
      rooms.appendChild(a);
      fetch('/capacity/' + j.id).then(x => x.json()).then(c => {
        if (c && c.available && c.total) {
          const el = a.querySelector('.rmeta');
          el.textContent = (meta.length ? meta.join(' · ') + ' · ' : '')
            + c.total.toLocaleString() + ' setups';
        }
      }).catch(() => {});
    });
  }).catch(() => {});
})();

/* --- open a finished job by URL: /?job=<id>[&look=<shot>][&demo=1] --------- */
(function () {
  const qp = new URLSearchParams(location.search);
  const want = qp.get('job');
  if (!want) return;
  fetch('/jobs').then(r => r.json()).then(r => {
    const j = (r.jobs || []).find(x => x.id === want);
    if (!j || j.state !== 'done') { line('no finished job ' + want, 'error'); return; }
    landing.classList.add('hidden');
    show(j.id, j.summary);
    if (qp.get('demo')) {
      $('brief').value = BRIEFS[1];
      setTimeout(startPlan, 800);
      return;
    }
    const look = parseInt(qp.get('look') || '0', 10);
    if (look) {
      let tries = 0;
      const tick = setInterval(() => {
        tries++;
        const ps = currentPlan && currentPlan.shots.find(s => s.shot.number === look && s.setup);
        if (ps) { clearInterval(tick); lookThrough(ps); }
        else if (tries > 60) clearInterval(tick);
      }, 500);
    }
  });
})();
</script>
<script src="/stage.js"></script>
"""
