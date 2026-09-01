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
            source = next(
                (p for p in sorted(d.iterdir()) if p.suffix.lower() in _MEDIA_SUFFIXES),
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
            job = Job(
                id=d.name,
                name=source.name if source != d else d.name,
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
        return job.twin_obj

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

                job.summary = _summarise(twin, result)
                job.summary["scout"] = job.scout_path is not None
                job.location = twin.name

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


class _Handler(BaseHTTPRequestHandler):
    studio: Studio
    server_version = "locaish"

    def log_message(self, fmt, *args):  # quieter than the default access log
        pass

    # -- GET ---------------------------------------------------------------
    def do_GET(self) -> None:
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]

        if not parts:
            return self._html(PAGE.replace("__SHOWCASE__", "1" if self.studio.showcase else "0"))
        if parts[0] in ("logo.svg", "favicon.svg"):
            return self._svg(LOGO_SVG)
        if parts[0] == "thumb" and len(parts) == 2:
            return self._thumb(parts[1])
        if parts[0] == "events" and len(parts) == 2:
            return self._events(parts[1])
        if parts[0] == "view" and len(parts) == 2:
            job = self.studio.jobs.get(parts[1])
            if not job or not job.viewer_path or not job.viewer_path.exists():
                return self._json({"error": "no viewer for that job"}, 404)
            self.studio.refresh_viewer(job)
            return self._file(job.viewer_path, "text/html; charset=utf-8")
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
        if parts and parts[0] == "chat" and len(parts) == 2:
            return self._chat(parts[1])
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
        if not job or job.state != "done" or not job.location:
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

    def _html(self, body: str) -> None:
        payload = body.encode()
        self.send_response(200)
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
  :root {
    --bg: #0b0d10; --surface: #12151a; --surface2: #171b21; --line: #232830; --line2: #2c323b;
    --ink: #eef1f5; --dim: #8a94a3; --mute: #5c6672; --accent: #5b9dff; --accent-ink: #081120;
    --ok: #4fcf87; --warn: #e2b96a; --bad: #e0707a;
    --r: 12px; --shadow: 0 1px 0 rgba(255,255,255,.03) inset, 0 8px 30px rgba(0,0,0,.35);
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 14.5px/1.55 "Inter", ui-sans-serif, -apple-system, "SF Pro Text", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  a { color: inherit; }
  .hidden { display: none !important; }
  main { max-width: 1240px; margin: 0 auto; padding: 0 28px 96px; }

  /* -- top bar -- */
  #top { position: sticky; top: 0; z-index: 5; display: flex; align-items: center; gap: 18px;
         height: 58px; margin: 0 -28px 22px; padding: 0 28px; background: rgba(11,13,16,.85);
         backdrop-filter: blur(12px); border-bottom: 1px solid var(--line); }
  #top .brand { font-weight: 700; letter-spacing: -0.02em; font-size: 17px; text-decoration: none;
                display: inline-flex; align-items: center; gap: 9px; }
  #top .brand .mark { flex: none; }
  #top .loc { display: flex; align-items: baseline; gap: 10px; min-width: 0; }
  #top .loc b { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  #top .loc span { color: var(--dim); font-size: 13px; white-space: nowrap; }
  #top .right { margin-left: auto; display: flex; align-items: center; gap: 8px; }
  .pill { font-size: 11.5px; color: var(--mute); border: 1px solid var(--line); border-radius: 999px;
          padding: 3px 10px; white-space: nowrap; }
  .pill.on { color: var(--ok); border-color: rgba(79,207,135,.35); }
  #top .right a.quiet { color: var(--dim); font-size: 13px; text-decoration: none; margin-left: 8px; }
  #top .right a.quiet:hover { color: var(--ink); }

  /* -- state 1: landing -- */
  #landing { min-height: calc(100vh - 80px); display: flex; flex-direction: column; align-items: center;
             justify-content: center; text-align: center; padding-bottom: 6vh; }
  #landing h1 { font-size: 40px; letter-spacing: -0.03em; margin: 0 0 8px; font-weight: 700;
                display: flex; align-items: center; justify-content: center; gap: 14px; }
  #landing p.lead { color: var(--dim); font-size: 16px; margin: 0 0 34px; max-width: 560px; }
  #drop {
    width: min(720px, 100%); border: 1.5px dashed var(--line2); border-radius: 18px; padding: 56px 24px 46px;
    cursor: pointer; transition: .15s; background: var(--surface); box-shadow: var(--shadow);
  }
  #drop.hot { border-color: var(--accent); background: #131a26; }
  #drop b { display: block; font-size: 18px; margin-bottom: 8px; font-weight: 600; }
  #drop span { color: var(--dim); font-size: 13.5px; display: block; }
  #drop .btn { display: inline-block; margin-top: 22px; background: var(--accent); color: var(--accent-ink);
               border-radius: 10px; padding: 10px 18px; font-weight: 600; font-size: 14px; }
  #geo { display: flex; gap: 8px; align-items: center; margin-top: 18px; color: var(--mute);
         font-size: 13px; cursor: pointer; }
  #geo input { accent-color: var(--accent); }
  #landing .steps { display: flex; gap: 36px; margin-top: 56px; color: var(--mute); font-size: 13px; }
  #landing .steps b { display: block; color: var(--dim); font-weight: 600; margin-bottom: 2px; }

  #library { width: min(1020px, 100%); margin-top: 44px; text-align: left; }
  .libhead { color: var(--mute); font-size: 12px; letter-spacing: .08em; text-transform: uppercase;
             margin: 0 0 12px 2px; }
  #rooms { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }
  .room { display: block; position: relative; text-decoration: none; background: var(--surface); border: 1px solid var(--line);
          border-radius: 14px; overflow: hidden; box-shadow: var(--shadow);
          transition: transform .18s, border-color .18s, box-shadow .18s; }
  .room:hover { transform: translateY(-3px); border-color: var(--line2); box-shadow: 0 14px 44px rgba(0,0,0,.5); }
  .room .thumb { display: block; aspect-ratio: 16 / 9; background: #0a0c0f; }
  .room .thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .room .rbody { display: block; padding: 11px 14px 12px; }
  .room .rbody b { display: block; font-size: 14.5px; font-weight: 600; margin-bottom: 2px; color: var(--ink); }
  .room .rmeta { color: var(--dim); font-size: 12.5px; }
  .libfoot { color: var(--mute); font-size: 12.5px; margin: 18px 2px 0; max-width: 760px; }
  .room .gal { position: absolute; top: 8px; right: 8px; display: flex; gap: 6px; }
  .room .gal button { background: rgba(11,13,16,.85); border: 1px solid var(--line2); color: var(--dim);
                      border-radius: 8px; padding: 3px 9px; font: inherit; font-size: 11px; cursor: pointer; }
  .room .gal button:hover { color: var(--ink); }
  .room .gal .in { color: var(--ok); border-color: rgba(79,207,135,.4); cursor: default; }
  #galbtn { background: none; border: 1px solid var(--line); color: var(--dim); border-radius: 999px;
            padding: 3px 12px; font: inherit; font-size: 11.5px; cursor: pointer; white-space: nowrap; }
  #galbtn:hover { color: var(--ink); border-color: var(--line2); }
  #galbtn.on { color: var(--ok); border-color: rgba(79,207,135,.35); }

  /* -- state 2: working -- */
  #working { max-width: 720px; margin: 12vh auto 0; }
  #stageline { display: flex; align-items: center; gap: 12px; color: var(--ink);
               font-size: 15px; min-height: 24px; }
  #stageline .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent);
                    animation: pulse 1.2s ease-in-out infinite; flex: none; }
  @keyframes pulse { 50% { opacity: .3; } }
  .bar { height: 3px; background: var(--line); border-radius: 2px; margin: 14px 0 12px; overflow: hidden; }
  .bar i { display: block; height: 100%; width: 0; background: var(--accent); transition: width .5s; }
  details#logbox { margin-top: 12px; }
  details#logbox summary { color: var(--mute); font-size: 12.5px; cursor: pointer; }
  #log {
    margin-top: 8px; background: var(--surface); border: 1px solid var(--line);
    border-radius: 10px; padding: 12px 14px; max-height: 260px; overflow-y: auto;
    font: 12px/1.7 ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  #log div { display: flex; gap: 10px; }
  #log .t { color: var(--mute); min-width: 40px; text-align: right; flex: none; }
  .note { color: var(--warn); }
  .error { color: var(--bad); }

  /* -- state 3: the location -- */
  #hero { border: 1px solid var(--line); border-radius: 16px; overflow: hidden; background: var(--surface);
          box-shadow: var(--shadow); }
  #viewerwrap { height: min(68vh, 780px); min-height: 420px; background: #0a0c0f; }
  #viewerwrap iframe { width: 100%; height: 100%; border: 0; display: block; }
  #vfbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
           padding: 8px 14px; background: var(--surface); border-top: 1px solid var(--line);
           font-size: 12.5px; color: var(--dim); }
  #vfbar b { color: var(--ink); font-weight: 600; }
  #vfbar .lens { background: var(--surface); color: var(--dim); border: 1px solid var(--line);
                 border-radius: 8px; padding: 3px 9px; font: inherit; font-size: 12px; cursor: pointer; }
  #vfbar .lens[aria-pressed="true"] { color: var(--ink); border-color: var(--accent); }
  #vfbar .near { margin-left: auto; font: 12px/1.5 ui-monospace, Menlo, monospace; }
  #vfbar .exit { background: none; border: 1px solid var(--line); color: var(--dim);
                 border-radius: 8px; padding: 3px 9px; font: inherit; font-size: 12px; cursor: pointer; }
  #heroactions { display: flex; gap: 18px; justify-content: flex-end; padding: 8px 4px 0; font-size: 12px; }
  #heroactions a { color: var(--mute); text-decoration: none; }
  #heroactions a:hover { color: var(--ink); }

  section { margin-top: 44px; }
  .sechead { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 14px; }
  .sechead h2 { font-size: 19px; font-weight: 600; letter-spacing: -0.01em; margin: 0; }
  .sechead p { margin: 0; color: var(--dim); font-size: 13.5px; }
  .sechead .sub { display: flex; flex-direction: column; gap: 2px; }

  /* -- coverage -- */
  #briefwrap { display: grid; grid-template-columns: 1fr; gap: 10px; background: var(--surface);
               border: 1px solid var(--line); border-radius: 14px; padding: 14px; box-shadow: var(--shadow); }
  #brief { width: 100%; background: var(--bg); color: var(--ink); border: 1px solid var(--line);
           border-radius: 10px; padding: 12px 14px; font: 14px/1.6 ui-sans-serif, system-ui, sans-serif;
           outline: none; resize: vertical; min-height: 124px; }
  #brief:focus { border-color: var(--accent); }
  #planrow { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
  #modenote { color: var(--mute); font-size: 12.5px; flex: 1; min-width: 200px; }
  #example { color: var(--dim); font-size: 12.5px; text-decoration: none; border-bottom: 1px dotted var(--line2); }
  #example:hover { color: var(--ink); }
  #planbtn { background: var(--accent); color: var(--accent-ink); border: 0; border-radius: 10px; padding: 10px 22px;
             font: inherit; font-weight: 600; cursor: pointer; }
  #planbtn:hover { filter: brightness(1.08); }
  #planbtn:disabled { opacity: .5; cursor: default; }
  #prevplans { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; color: var(--mute); font-size: 12.5px; }
  #prevplans a { color: var(--dim); text-decoration: none; border-bottom: 1px dotted var(--line2); }
  #prevplans a:hover { color: var(--ink); }
  #scoutcap { padding: 11px 16px; background: var(--surface2); border-top: 1px solid var(--line);
              font-size: 13.5px; color: var(--ink); display: flex; gap: 12px; align-items: baseline; }
  #scoutcap b { color: var(--accent); font-weight: 600; white-space: nowrap; }
  #scoutcap span { color: var(--dim); }
  #scoutcap .g { display: inline-block; font-size: 10.5px; letter-spacing: .06em; text-transform: uppercase;
                 color: #0b1220; background: linear-gradient(90deg, #6ea8fe, #a78bfa); border-radius: 6px;
                 padding: 1px 6px; margin-right: 6px; font-weight: 700; }
  #trace { margin-top: 18px; color: var(--mute); font-size: 12.5px; }
  #trace summary { cursor: pointer; }
  #tracebody { margin-top: 8px; background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
               padding: 10px 14px; max-height: 420px; overflow-y: auto;
               font: 12px/1.7 ui-monospace, SFMono-Regular, Menlo, monospace; }
  #tracebody div { display: flex; gap: 10px; }
  #tracebody .t { color: var(--mute); min-width: 40px; text-align: right; flex: none; }
  #tracebody .ag { color: var(--accent); }
  #tracebody pre { white-space: pre-wrap; word-break: break-word; margin: 2px 0 8px 50px; color: #b8c4d3;
                   background: var(--surface2); border-radius: 8px; padding: 8px 10px; font: 11px/1.5 ui-monospace, Menlo, monospace; }
  #planhead { display: flex; align-items: baseline; justify-content: space-between; margin: 26px 0 12px; gap: 12px; flex-wrap: wrap; }
  #planhead b { font-size: 17px; font-weight: 600; }
  #planhead span { color: var(--dim); font-size: 13px; }
  #shots { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
  @media (max-width: 1100px) { #shots { grid-template-columns: 1fr 1fr; } }
  @media (max-width: 760px) { #shots { grid-template-columns: 1fr; } }
  .shot { background: var(--surface); border: 1px solid var(--line); border-radius: var(--r); overflow: hidden;
          box-shadow: var(--shadow); }
  .shot.active { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
  .shot.none { border-style: dashed; padding: 14px; color: var(--dim); box-shadow: none; }
  .shot .frame { position: relative; aspect-ratio: 16 / 9; background: #0a0c0f; cursor: pointer; display: block; }
  .shot .frame img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .shot .frame .look { position: absolute; right: 8px; bottom: 8px; background: rgba(11,13,16,.8);
                       color: var(--ink); border: 1px solid var(--line2); border-radius: 8px; padding: 4px 9px;
                       font-size: 11.5px; }
  .shot .body { padding: 11px 13px 12px; }
  .shot h3 { margin: 0 0 3px; font-size: 14px; font-weight: 600; }
  .shot .meas { font-size: 12.5px; color: var(--dim); }
  .shot .meas .warnc { color: var(--warn); }
  .shot .desc { margin-top: 6px; font-size: 13.5px; }
  .shot .why { color: var(--dim); font-size: 12.5px; margin-top: 6px; line-height: 1.5; }
  .shot .dim { color: var(--mute); font-size: 12px; margin-top: 4px; }
  .shot .relax { color: var(--warn); font-size: 12px; margin-top: 2px; }
  .shot .review { margin-top: 8px; padding: 8px 10px; border-radius: 8px; background: var(--surface2);
                  border: 1px solid var(--line); font-size: 12.5px; }
  .shot .review .g { display: inline-block; font-size: 10.5px; letter-spacing: .06em; text-transform: uppercase;
                     color: #0b1220; background: linear-gradient(90deg, #6ea8fe, #a78bfa); border-radius: 6px;
                     padding: 1px 6px; margin-right: 6px; font-weight: 700; }
  .shot details { margin-top: 8px; font-size: 12px; color: var(--mute); }
  .shot details pre { white-space: pre-wrap; word-break: break-word; font: 11px/1.5 ui-monospace, Menlo, monospace;
                      background: var(--surface2); border-radius: 8px; padding: 8px 10px; margin: 6px 0 0; color: #b8c4d3; }
  #planbelow { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px; }
  @media (max-width: 900px) { #planbelow { grid-template-columns: 1fr; } }
  .panelbox { background: var(--surface); border: 1px solid var(--line); border-radius: var(--r); padding: 14px 16px;
              box-shadow: var(--shadow); }
  .panelbox h4 { margin: 0 0 10px; font-size: 13.5px; font-weight: 600; }
  #floorplan svg { width: 100%; height: auto; border-radius: 8px; display: block; }
  #capacity table { border-collapse: separate; border-spacing: 3px; width: 100%; font-size: 12px; }
  #capacity th { color: var(--dim); font-weight: 500; text-align: center; font-size: 11px; }
  #capacity th.row { text-align: left; }
  #capacity td { text-align: center; border-radius: 5px; padding: 6px 2px; color: var(--ink); min-width: 34px; }
  #capacity .foot { color: var(--mute); font-size: 12px; margin-top: 8px; }
  #capacity .lead { font-size: 13px; margin-bottom: 8px; color: var(--dim); }
  #capacity .lead b { color: var(--ink); }
  #capacity ol { margin: 6px 0 0 18px; padding: 0; color: var(--dim); font-size: 12px; }
  #planlinks { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
  #planlinks a, #planlinks button { text-align: center; text-decoration: none; padding: 8px 12px;
             border-radius: 10px; border: 1px solid var(--line); color: var(--ink);
             background: var(--surface); font: inherit; font-size: 13px; cursor: pointer; }
  #planlinks a:hover, #planlinks button:hover { border-color: var(--line2); }

  /* -- chat -- */
  #chat { margin-top: 48px; }
  #thread { display: flex; flex-direction: column; gap: 12px; margin-bottom: 14px; }
  .msg { max-width: 78%; padding: 10px 14px; border-radius: 14px; white-space: pre-wrap; }
  .msg.user { align-self: flex-end; background: #16233a; border: 1px solid #22355a; }
  .msg.agent { align-self: flex-start; background: var(--surface); border: 1px solid var(--line); }
  .msg.agent img { max-width: 100%; border-radius: 10px; margin-top: 8px; display: block; }
  .msg .tools { color: var(--mute); font: 11.5px/1.7 ui-monospace, Menlo, monospace;
                border-top: 1px solid var(--line); margin-top: 8px; padding-top: 6px; }
  .msg.err { border-color: #5c2b30; color: var(--bad); }
  .thinking { color: var(--dim); font-size: 13px; }
  #chips { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
  #chips button {
    background: transparent; color: var(--dim); border: 1px solid var(--line);
    border-radius: 999px; padding: 5px 12px; font-size: 12.5px; cursor: pointer;
  }
  #chips button:hover { color: var(--ink); border-color: var(--line2); }
  #askrow { display: flex; gap: 8px; }
  #ask { flex: 1; background: var(--surface); color: var(--ink); border: 1px solid var(--line);
         border-radius: 10px; padding: 11px 14px; font: inherit; outline: none; }
  #ask:focus { border-color: var(--accent); }
  #send { background: var(--surface2); color: var(--ink); border: 1px solid var(--line); border-radius: 10px;
          padding: 0 18px; font: inherit; font-weight: 600; cursor: pointer; }
  #send:hover { border-color: var(--line2); }
  #send:disabled { opacity: .5; cursor: default; }
</style>
<main>
  <div id="top">
    <a class="brand" href="/"><svg class="mark" viewBox="0 0 32 32" width="18" height="18" aria-hidden="true"><defs><linearGradient id="lga" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6ea8fe"/><stop offset="1" stop-color="#a78bfa"/></linearGradient></defs><path d="M11 4H8a4 4 0 0 0-4 4v3M21 4h3a4 4 0 0 1 4 4v3M28 21v3a4 4 0 0 1-4 4h-3M11 28H8a4 4 0 0 1-4-4v-3" fill="none" stroke="url(#lga)" stroke-width="2.6" stroke-linecap="round"/><rect x="11" y="11" width="10" height="10" rx="2.6" transform="rotate(45 16 16)" fill="url(#lga)"/></svg>Locaish</a>
    <div class="loc hidden" id="toploc"><b id="locname"></b><span id="locmeta"></span></div>
    <div class="right">
      <button class="hidden" id="galbtn"></button>
      <span class="pill hidden" id="covpill">local sweep</span>
      <a class="quiet hidden" id="topagain" href="/">Scan another room</a>
    </div>
  </div>

  <div id="landing">
    <h1><svg class="mark" viewBox="0 0 32 32" width="40" height="40" aria-hidden="true"><defs><linearGradient id="lgb" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6ea8fe"/><stop offset="1" stop-color="#a78bfa"/></linearGradient></defs><path d="M11 4H8a4 4 0 0 0-4 4v3M21 4h3a4 4 0 0 1 4 4v3M28 21v3a4 4 0 0 1-4 4h-3M11 28H8a4 4 0 0 1-4-4v-3" fill="none" stroke="url(#lgb)" stroke-width="2.6" stroke-linecap="round"/><rect x="11" y="11" width="10" height="10" rx="2.6" transform="rotate(45 16 16)" fill="url(#lgb)"/></svg>Locaish</h1>
    <p class="lead" id="lead">Scan a room with your phone. Plan the scene before anyone drives out.</p>
    <div id="drop">
      <b>Drop a walkthrough video here</b>
      <span>or a scan export &mdash; ply, glb, obj. Sixty seconds of walking is plenty.</span>
      <span class="btn">Choose a file</span>
      <input id="file" type="file" accept="video/*,.ply,.obj,.glb,.gltf,.stl,.mov,.mp4,.mkv" class="hidden">
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

    <section id="coverage">
      <div class="sechead">
        <div class="sub"><h2>Describe the scene</h2>
          <p>Prose, or a shot list one per line. The scout places every shot in this room, looks through it, and draws the camera plan.</p></div>
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
      <div id="thread"></div>
      <div id="askrow">
        <input id="ask" placeholder="Ask the scout anything about this room&hellip;" autocomplete="off">
        <button id="send">Ask</button>
      </div>
    </section>
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
    + '<a href="/twin/' + id + '">.twin</a>';

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

function caption(head, text, gemini) {
  const cap = $('scoutcap');
  if (head == null) { cap.classList.add('hidden'); return; }
  cap.classList.remove('hidden');
  cap.innerHTML = '<b>' + esc(head) + '</b><span>' + (gemini ? '<span class="g">Gemini</span>' : '') + esc(text || '') + '</span>';
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
    + (plan.planner === 'gemini' ? ' · Gemini' : '') + '</span>';
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
      html += '<td style="background: rgba(110,168,254,' + (a * 0.85).toFixed(3) + ')" title="'
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
"""
