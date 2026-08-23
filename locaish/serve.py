"""A local drop target: drag a video onto a page, get a twin back.

The CLI is the honest interface to this pipeline and will stay that way. But
"ingest a video" is a two-minute operation with six named stages, and the thing
a person actually wants to do with it -- hand it a file, watch it work, look at
the result -- is a worse fit for a terminal than for a page. So this is the
whole CLI ingest path with an upload form in front of it and the viewer behind
it, and nothing else: no accounts, no database, no queue, no build step.

It binds to loopback by design. The upload endpoint writes whatever bytes it is
given to disk and then runs a reconstruction on them, which is exactly the kind
of endpoint that should never be reachable from another machine, and `--host`
is deliberately not a flag.

The progress stream is server-sent events rather than a websocket because the
traffic is one-directional and a few dozen lines long; SSE is a `text/plain`
response that happens to be flushed early, and it needs neither a dependency nor
a handshake.
"""

from __future__ import annotations

import json
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
    """One uploaded video, and everything that happened to it since."""

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
    summary: dict = field(default_factory=dict)

    def emit(self, kind: str, text: str, **extra) -> None:
        self.events.put({"kind": kind, "text": text, **extra})


class Studio:
    """The job registry and the work loop. Deliberately not a queue: one at a time.

    A reconstruction saturates the GPU, so running two at once makes both slower
    and can exhaust unified memory outright. Serialising them behind a lock is
    both simpler and faster than any scheduling would be.
    """

    def __init__(self, root: Path, *, max_points: int):
        self.root = root
        self.max_points = max_points
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

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
    max_points: int = 1_500_000,
    open_browser: bool = True,
) -> str:
    """Run the studio until interrupted. Returns the URL it bound to."""
    import webbrowser

    root.mkdir(parents=True, exist_ok=True)
    studio = Studio(root, max_points=max_points)
    handler = type("Handler", (_Handler,), {"studio": studio})
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{httpd.server_address[1]}"

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
<title>Locaish — drop a room</title>
<style>
  :root {
    --bg: #0e1013; --panel: #171a1f; --line: #262b33;
    --ink: #e8eaed; --dim: #9aa3ad; --accent: #6ea8fe;
    --ok: #57d38c; --warn: #e5c07b; --bad: #e06c75;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.55 ui-sans-serif, -apple-system, "SF Pro Text", system-ui, sans-serif;
    display: flex; justify-content: center; padding: 48px 20px;
  }
  main { width: 100%; max-width: 780px; }
  h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }
  .sub { color: var(--dim); margin: 0 0 28px; }
  #drop {
    border: 1.5px dashed var(--line); border-radius: 14px; padding: 54px 24px;
    text-align: center; cursor: pointer; transition: .15s; background: var(--panel);
  }
  #drop.hot { border-color: var(--accent); background: #1b2432; }
  #drop b { display: block; font-size: 17px; margin-bottom: 6px; }
  #drop span { color: var(--dim); font-size: 13px; }
  .hidden { display: none; }
  #log {
    margin-top: 22px; background: var(--panel); border: 1px solid var(--line);
    border-radius: 12px; padding: 16px 18px; max-height: 320px; overflow-y: auto;
    font: 12.5px/1.7 ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  #log div { display: flex; gap: 10px; }
  #log .t { color: #5b636d; min-width: 44px; text-align: right; }
  .note { color: var(--warn); }
  .error { color: var(--bad); }
  .bar { height: 3px; background: var(--line); border-radius: 2px; margin-top: 14px; overflow: hidden; }
  .bar i { display: block; height: 100%; width: 0; background: var(--accent); transition: width .4s; }
  #result { margin-top: 22px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; }
  .card em { display: block; font-style: normal; color: var(--dim); font-size: 11.5px;
             text-transform: uppercase; letter-spacing: .06em; margin-bottom: 3px; }
  .card b { font-size: 19px; font-weight: 600; }
  .verdict-pass b { color: var(--ok); } .verdict-warn b { color: var(--warn); }
  .verdict-fail b { color: var(--bad); }
  .actions { display: flex; gap: 10px; margin-top: 16px; }
  .actions a {
    flex: 1; text-align: center; text-decoration: none; padding: 11px 14px;
    border-radius: 10px; border: 1px solid var(--line); color: var(--ink); background: var(--panel);
  }
  .actions a.primary { background: var(--accent); color: #0b1220; border-color: transparent; font-weight: 600; }
</style>
<main>
  <h1>Drop a room</h1>
  <p class="sub">A video of a walk through the space. Sixty seconds is plenty —
     walk, don't pan.</p>

  <div id="drop">
    <b>Drop a video here</b>
    <span>or click to choose &middot; mov, mp4, mkv</span>
    <input id="file" type="file" accept="video/*" class="hidden">
  </div>
  <div class="bar hidden" id="barwrap"><i id="bar"></i></div>
  <div id="log" class="hidden"></div>
  <div id="result"></div>
</main>
<script>
const drop = document.getElementById('drop');
const file = document.getElementById('file');
const log = document.getElementById('log');
const bar = document.getElementById('bar');
const barwrap = document.getElementById('barwrap');
const result = document.getElementById('result');
let t0 = 0;

// The stages the pipeline announces, in order, so a progress bar can mean
// something instead of animating for its own sake.
const STAGES = ['frames','decode','score','reconstruct','colmap','stereo','scale','clean',
                'subsample','write','read','normals','planes','canonicalize','grid',
                'mesh','bounds','structure','qa','rendering viewer','scouting the location'];

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
  log.innerHTML = ''; result.innerHTML = '';
  log.classList.remove('hidden'); barwrap.classList.remove('hidden');
  bar.style.width = '2%';
  line('uploading ' + f.name + ' (' + (f.size/1e6).toFixed(0) + ' MB)');

  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/upload?name=' + encodeURIComponent(f.name));
  xhr.setRequestHeader('Content-Type', 'application/octet-stream');
  xhr.upload.onprogress = e => {
    if (e.lengthComputable) bar.style.width = (2 + 8 * e.loaded / e.total) + '%';
  };
  xhr.onload = () => {
    let r; try { r = JSON.parse(xhr.responseText); } catch (_) { r = {}; }
    if (!r.id) { line(r.error || 'upload failed', 'error'); return; }
    line('uploaded, reconstructing');
    listen(r.id);
  };
  xhr.onerror = () => line('upload failed', 'error');
  xhr.send(f);
}

function listen(id) {
  const es = new EventSource('/events/' + id);
  es.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.kind === 'stage') {
      line(m.text);
      const i = STAGES.indexOf(m.text.split(' ')[0]);
      if (i >= 0) bar.style.width = (10 + 85 * (i + 1) / STAGES.length) + '%';
    } else if (m.kind === 'note') {
      line(m.text, 'note');
    } else if (m.kind === 'error') {
      line(m.text, 'error'); bar.style.width = '100%'; es.close();
    } else if (m.kind === 'done') {
      bar.style.width = '100%'; es.close(); show(id, m.summary);
    }
  };
  es.onerror = () => es.close();
}

function show(id, s) {
  const pct = s.scale_relative_error == null ? null
            : '±' + (s.scale_relative_error * 100).toFixed(0) + '%';
  const cards = [
    ['verdict', s.verdict, 'verdict-' + s.verdict],
    ['floor area', s.floor_area_m2 == null ? '—' : s.floor_area_m2 + ' m²'],
    ['ceiling', s.ceiling_height_m == null ? 'open' : s.ceiling_height_m.toFixed(2) + ' m'],
    ['openings', s.openings],
    ['points', (s.points/1e6).toFixed(2) + ' M'],
    ['scale', pct || '—'],
  ];
  result.innerHTML =
    '<div class="grid">' + cards.map(c =>
      '<div class="card ' + (c[2]||'') + '"><em>' + c[0] + '</em><b>' + c[1] + '</b></div>').join('') +
    '</div><div class="actions">' +
      '<a class="primary" href="/view/' + id + '" target="_blank">Open the twin</a>' +
      (s.scout ? '<a href="/scout/' + id + '" target="_blank">Scout report</a>' : '') +
      '<a href="/twin/' + id + '">Download .twin</a>' +
    '</div>';
  const bad = (s.checks.fail || []).concat(s.checks.warn || []);
  if (bad.length) line('QA flagged: ' + bad.join(', '), 'note');
  line('done in ' + s.seconds + 's');
}
</script>
"""
