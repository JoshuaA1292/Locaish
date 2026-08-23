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
    summary: dict = field(default_factory=dict)

    def emit(self, kind: str, text: str, **extra) -> None:
        self.events.put({"kind": kind, "text": text, **extra})


class Studio:
    """The job registry and the work loop. Deliberately not a queue: one at a time.

    A reconstruction saturates the machine, so running two at once makes both
    slower. Serialising them behind a lock is both simpler and faster than any
    scheduling would be.
    """

    def __init__(self, root: Path, *, max_points: int):
        self.root = root
        self.max_points = max_points
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._agent = None
        self._agent_lock = threading.Lock()

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

        with self._lock:
            job.state = "running"
            job.emit("stage", "starting")
            try:
                opts = IngestOptions(
                    name=Path(job.name).stem,
                    video_workdir=job.workdir / "recon",
                    max_points=self.max_points,
                    progress=lambda m: job.emit("stage", m),
                )
                result = ingest(job.source, opts)
                twin = result.twin

                job.twin_path = twin.save(job.workdir / f"{twin.name}.twin")
                job.emit("stage", "rendering viewer")
                job.viewer_path = render_html(twin, job.workdir / "view.html")

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
            return self._html(PAGE)
        if parts[0] == "events" and len(parts) == 2:
            return self._events(parts[1])
        if parts[0] == "view" and len(parts) == 2:
            job = self.studio.jobs.get(parts[1])
            if not job or not job.viewer_path or not job.viewer_path.exists():
                return self._json({"error": "no viewer for that job"}, 404)
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
                        {"id": j.id, "name": j.name, "state": j.state, "summary": j.summary}
                        for j in self.studio.jobs.values()
                    ]
                }
            )
        self._json({"error": "not found"}, 404)

    # -- POST --------------------------------------------------------------
    def do_POST(self) -> None:
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        if parts and parts[0] == "chat" and len(parts) == 2:
            return self._chat(parts[1])
        if url.path != "/upload":
            return self._json({"error": "not found"}, 404)

        params = parse_qs(url.query)
        name = (params.get("name") or ["upload.mov"])[0]
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._json({"error": "bad Content-Length"}, 400)
        if length <= 0:
            return self._json({"error": "empty upload"}, 400)
        if length > MAX_UPLOAD_BYTES:
            return self._json({"error": "upload too large"}, 413)

        job = self.studio.create(name)
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

    # -- helpers -----------------------------------------------------------
    def _events(self, job_id: str) -> None:
        job = self.studio.jobs.get(job_id)
        if not job:
            return self._json({"error": "unknown job"}, 404)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                item = job.events.get()
                if item is None:
                    break
                self.wfile.write(f"data: {json.dumps(item)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _html(self, body: str) -> None:
        payload = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
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
    max_points: int = 1_500_000,
    open_browser: bool = True,
) -> str:
    """Run the studio until interrupted. Returns the URL it bound to."""
    import webbrowser

    host = host or os.environ.get("LOCAISH_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", port))
    root.mkdir(parents=True, exist_ok=True)
    studio = Studio(root, max_points=max_points)
    handler = type("Handler", (_Handler,), {"studio": studio})
    httpd = ThreadingHTTPServer((host, port), handler)
    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{shown}:{httpd.server_address[1]}"

    print(f"locaish studio on {url}")
    print(f"  jobs land in {root}")
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


PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Locaish</title>
<style>
  :root {
    --bg: #0e1013; --panel: #15181d; --panel2: #191d23; --line: #262b33;
    --ink: #e8eaed; --dim: #8b95a1; --accent: #6ea8fe;
    --ok: #57d38c; --warn: #e5c07b; --bad: #e06c75;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.55 ui-sans-serif, -apple-system, "SF Pro Text", system-ui, sans-serif;
  }
  main { max-width: 1060px; margin: 0 auto; padding: 40px 22px 80px; }
  header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 30px; }
  header h1 { font-size: 20px; margin: 0; letter-spacing: -0.01em; }
  header span { color: var(--dim); font-size: 13px; }

  /* -- state 1: drop -- */
  #drop {
    border: 1.5px dashed var(--line); border-radius: 16px; padding: 72px 24px;
    text-align: center; cursor: pointer; transition: .15s; background: var(--panel);
  }
  #drop.hot { border-color: var(--accent); background: #1b2432; }
  #drop b { display: block; font-size: 17px; margin-bottom: 6px; font-weight: 600; }
  #drop span { color: var(--dim); font-size: 13px; }
  .hidden { display: none !important; }

  /* -- state 2: working -- */
  #working { margin-top: 8px; }
  #stageline { display: flex; align-items: center; gap: 12px; color: var(--dim);
               font-size: 14px; min-height: 24px; }
  #stageline .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent);
                    animation: pulse 1.2s ease-in-out infinite; flex: none; }
  @keyframes pulse { 50% { opacity: .3; } }
  .bar { height: 3px; background: var(--line); border-radius: 2px; margin: 14px 0 10px; overflow: hidden; }
  .bar i { display: block; height: 100%; width: 0; background: var(--accent); transition: width .5s; }
  details#logbox { margin-top: 10px; }
  details#logbox summary { color: var(--dim); font-size: 12.5px; cursor: pointer; }
  #log {
    margin-top: 8px; background: var(--panel); border: 1px solid var(--line);
    border-radius: 10px; padding: 12px 14px; max-height: 240px; overflow-y: auto;
    font: 12px/1.7 ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  #log div { display: flex; gap: 10px; }
  #log .t { color: #5b636d; min-width: 40px; text-align: right; flex: none; }
  .note { color: var(--warn); }
  .error { color: var(--bad); }

  /* -- state 3: the location -- */
  #result { margin-top: 4px; }
  .split { display: grid; grid-template-columns: 1fr 280px; gap: 14px; }
  @media (max-width: 860px) { .split { grid-template-columns: 1fr; } }
  #viewerwrap { border: 1px solid var(--line); border-radius: 14px; overflow: hidden;
                background: var(--panel); aspect-ratio: 16 / 10; }
  #viewerwrap iframe { width: 100%; height: 100%; border: 0; display: block; }
  .facts { display: flex; flex-direction: column; gap: 10px; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
          padding: 11px 14px; }
  .card em { display: block; font-style: normal; color: var(--dim); font-size: 11px;
             text-transform: uppercase; letter-spacing: .07em; margin-bottom: 2px; }
  .card b { font-size: 18px; font-weight: 600; }
  .verdict-pass b { color: var(--ok); } .verdict-warn b { color: var(--warn); }
  .verdict-fail b { color: var(--bad); }
  .links { display: flex; gap: 8px; flex-wrap: wrap; }
  .links a { flex: 1; text-align: center; text-decoration: none; padding: 9px 10px;
             border-radius: 10px; border: 1px solid var(--line); color: var(--ink);
             background: var(--panel); font-size: 13px; white-space: nowrap; }
  .links a:hover { border-color: var(--accent); }

  /* -- chat -- */
  #chat { margin-top: 22px; }
  #chat h2 { font-size: 14px; color: var(--dim); font-weight: 500; margin: 0 0 10px;
             text-transform: uppercase; letter-spacing: .07em; }
  #thread { display: flex; flex-direction: column; gap: 12px; margin-bottom: 14px; }
  .msg { max-width: 85%; padding: 10px 14px; border-radius: 14px; white-space: pre-wrap; }
  .msg.user { align-self: flex-end; background: #1b2a44; border: 1px solid #26395c; }
  .msg.agent { align-self: flex-start; background: var(--panel); border: 1px solid var(--line); }
  .msg.agent img { max-width: 100%; border-radius: 10px; margin-top: 8px; display: block; }
  .msg .tools { color: var(--dim); font: 11.5px/1.7 ui-monospace, Menlo, monospace;
                border-top: 1px solid var(--line); margin-top: 8px; padding-top: 6px; }
  .msg.err { border-color: #5c2b30; color: var(--bad); }
  .thinking { color: var(--dim); font-size: 13px; }
  #chips { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
  #chips button {
    background: var(--panel); color: var(--dim); border: 1px solid var(--line);
    border-radius: 999px; padding: 6px 13px; font-size: 12.5px; cursor: pointer;
  }
  #chips button:hover { color: var(--ink); border-color: var(--accent); }
  #askrow { display: flex; gap: 8px; }
  #ask { flex: 1; background: var(--panel); color: var(--ink); border: 1px solid var(--line);
         border-radius: 12px; padding: 12px 14px; font: inherit; outline: none; }
  #ask:focus { border-color: var(--accent); }
  #send { background: var(--accent); color: #0b1220; border: 0; border-radius: 12px;
          padding: 0 20px; font: inherit; font-weight: 600; cursor: pointer; }
  #send:disabled { opacity: .5; cursor: default; }
  #again { margin-top: 26px; }
  #again a { color: var(--dim); font-size: 13px; }
</style>
<main>
  <header><h1>Locaish</h1><span>scan a room, then ask the scout</span></header>

  <div id="drop">
    <b>Drop a room here</b>
    <span>a video walkthrough, or a scan export (ply&thinsp;/&thinsp;glb&thinsp;/&thinsp;obj)
      &middot; sixty seconds of walking is plenty</span>
    <input id="file" type="file" accept="video/*,.ply,.obj,.glb,.gltf,.stl,.mov,.mp4,.mkv"
           class="hidden">
  </div>

  <div id="working" class="hidden">
    <div class="bar"><i id="bar"></i></div>
    <div id="stageline"><span class="dot"></span><span id="stage">starting</span></div>
    <details id="logbox"><summary>everything the pipeline said</summary>
      <div id="log"></div>
    </details>
  </div>

  <div id="result" class="hidden">
    <div class="split">
      <div id="viewerwrap"><iframe id="viewer" title="digital twin"></iframe></div>
      <div class="facts" id="facts"></div>
    </div>
    <div id="chat">
      <h2>Ask the scout</h2>
      <div id="thread"></div>
      <div id="chips"></div>
      <div id="askrow">
        <input id="ask" placeholder="Describe the shot you need&hellip;" autocomplete="off">
        <button id="send">Ask</button>
      </div>
    </div>
    <p id="again"><a href="/">scan another room</a></p>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
const drop = $('drop'), file = $('file'), working = $('working'), result = $('result');
const bar = $('bar'), stage = $('stage'), log = $('log');
let t0 = 0, jobId = null;

const STAGES = ['frames','decode','score','reconstruct','colmap features','colmap matching',
  'colmap mapping','stereo','scale','clean','subsample','write','read','normals','planes',
  'canonicalize','grid','mesh','bounds','structure','qa','rendering viewer',
  'scouting the location','sweeping camera setups','loading ClickHouse'];

const CHIPS = [
  'Find the cleanest close-up in this room',
  'Where can a doorway dolly actually track?',
  'A 75mm medium shot with no window behind the subject',
  'How live is this room for dialogue?',
];

drop.onclick = () => file.click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add('hot'); };
drop.ondragleave = () => drop.classList.remove('hot');
drop.ondrop = e => {
  e.preventDefault(); drop.classList.remove('hot');
  if (e.dataTransfer.files[0]) send(e.dataTransfer.files[0]);
};
file.onchange = () => { if (file.files[0]) send(file.files[0]); };

function line(text, cls) {
  const el = document.createElement('div');
  const dt = t0 ? ((Date.now() - t0) / 1000).toFixed(0) + 's' : '';
  el.innerHTML = '<span class="t">' + dt + '</span><span class="' + (cls||'') + '"></span>';
  el.lastChild.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
}

function send(f) {
  t0 = Date.now();
  drop.classList.add('hidden');
  working.classList.remove('hidden');
  log.innerHTML = '';
  bar.style.width = '2%';
  stage.textContent = 'uploading ' + f.name + ' (' + (f.size/1e6).toFixed(0) + ' MB)';
  line(stage.textContent);

  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/upload?name=' + encodeURIComponent(f.name));
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
      es.close(); fail(m.text);
    } else if (m.kind === 'done') {
      bar.style.width = '100%'; es.close(); show(id, m.summary);
    }
  };
  es.onerror = () => es.close();
}

function card(label, value, cls) {
  return '<div class="card ' + (cls||'') + '"><em>' + label + '</em><b>' + value + '</b></div>';
}

function show(id, s) {
  working.classList.add('hidden');
  result.classList.remove('hidden');
  $('viewer').src = '/view/' + id;

  const pct = s.scale_relative_error == null ? null
            : '&plusmn;' + (s.scale_relative_error * 100).toFixed(0) + '%';
  let facts = card('verdict', s.verdict, 'verdict-' + s.verdict)
    + card('floor area', s.floor_area_m2 == null ? '&mdash;' : s.floor_area_m2 + ' m&sup2;')
    + card('ceiling', s.ceiling_height_m == null ? 'not seen' : s.ceiling_height_m.toFixed(2) + ' m');
  if (pct) facts += card('size known to', pct);
  if (s.setups) facts += card('camera setups swept', s.setups.toLocaleString());
  facts += '<div class="links">'
    + '<a href="/view/' + id + '" target="_blank">Full screen</a>'
    + (s.scout ? '<a href="/scout/' + id + '" target="_blank">Scout report</a>' : '')
    + '<a href="/twin/' + id + '">.twin</a>'
    + '</div>';
  $('facts').innerHTML = facts;

  const bad = (s.checks.fail || []).concat(s.checks.warn || []);
  if (bad.length) line('QA flagged: ' + bad.join(', '), 'note');

  CHIPS.forEach(c => {
    const b = document.createElement('button');
    b.textContent = c;
    b.onclick = () => { $('ask').value = c; ask(); };
    $('chips').appendChild(b);
  });
  if (!s.clickhouse) {
    addAgent('The shot table is offline (ClickHouse is not configured), so I can '
      + 'answer from the twin itself — measurements, the scout report, dolly '
      + 'checks — but not sweep-search for setups.', [], true);
  }
  $('send').onclick = ask;
  $('ask').onkeydown = e => { if (e.key === 'Enter') ask(); };
  $('ask').focus();
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
</script>
"""
