"""Record a browser page to video through Chrome's DevTools protocol.

    python docs/demo/record.py --url URL --out DIR --actions actions.json [--headed]

The actions file is a list of steps run against the page at the given
second:  {"at": 1.5, "js": "walkSet(true)"}  evaluates JavaScript;
         {"at": 2.0, "hold": "w", "for": 2.5}  holds a key in the viewer's
         `keys` set (the viewer polls it every frame);
         {"at": 5.0, "look": [400, 0], "over": 1.5}  turns the walker's head
         by that many mouse pixels over that many seconds;
         {"at": 8.0, "click": "#btn-walk"}  clicks an element;
         {"at": 9.0, "type": ["#st_scene", "text"]}  sets a field and fires input;
         {"at": 9.5, "scroll": 1200, "over": 1.0}  scrolls the page smoothly.
Frames arrive from Page.startScreencast with timestamps and are written as
JPEGs; `frames.txt` is an ffmpeg concat list with real durations, and
`clip.mp4` is the 30 fps result. Chrome is launched here with its own
profile; `--headed` uses the real GPU (a window appears), otherwise
SwiftShader renders WebGL in headless mode.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import websockets

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


class CDP:
    def __init__(self, ws):
        self.ws = ws
        self.n = 0
        self.pending: dict[int, asyncio.Future] = {}
        self.events: asyncio.Queue = asyncio.Queue()

    async def pump(self):
        async for raw in self.ws:
            m = json.loads(raw)
            if "id" in m and m["id"] in self.pending:
                self.pending.pop(m["id"]).set_result(m)
            elif "method" in m:
                await self.events.put(m)

    async def call(self, method: str, **params):
        self.n += 1
        fut = asyncio.get_event_loop().create_future()
        self.pending[self.n] = fut
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        m = await fut
        if "error" in m:
            raise RuntimeError(f"{method}: {m['error']}")
        return m.get("result", {})

    async def js(self, expr: str):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        ex = r.get("exceptionDetails")
        if ex:
            raise RuntimeError(f"js: {ex.get('text')} {ex.get('exception', {}).get('description', '')[:300]}")
        return r.get("result", {}).get("value")


def launch(port: int, width: int, height: int, headed: bool, profile: Path) -> subprocess.Popen:
    args = [CHROME, f"--remote-debugging-port={port}", "--remote-allow-origins=*", f"--user-data-dir={profile}",
            f"--window-size={width},{height}", "--no-first-run", "--no-default-browser-check", "--hide-scrollbars",
            "--disable-infobars", "--autoplay-policy=no-user-gesture-required", "--force-device-scale-factor=1"]
    if not headed:
        args += ["--headless=new", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
    else:
        args += ["--window-position=0,0", "--app=about:blank"]
    return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ws_url(port: int) -> str:
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2) as r:
                for t in json.load(r):
                    if t.get("type") == "page":
                        return t["webSocketDebuggerUrl"]
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    raise RuntimeError("chrome did not expose a page")


async def run(url: str, out: Path, actions: list[dict], seconds: float, width: int, height: int,
              headed: bool, port: int, settle: float, quality: int, every_nth: int = 1,
              poll: float = 0.0) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("f_*.jpg"):
        f.unlink()
    profile = out / "profile"
    proc = launch(port, width, height, headed, profile)
    try:
        ws = ws_url(port)
        async with websockets.connect(ws, max_size=64 * 1024 * 1024) as sock:
            cdp = CDP(sock)
            pump = asyncio.create_task(cdp.pump())
            await cdp.call("Page.enable")
            await cdp.call("Runtime.enable")
            await cdp.call("Emulation.setDeviceMetricsOverride", width=width, height=height,
                           deviceScaleFactor=1, mobile=False)
            await cdp.call("Page.navigate", url=url)
            await asyncio.sleep(settle)
            frames: list[tuple[float, Path]] = []
            # Two ways to get frames: the screencast pushes one per compositor
            # repaint (right for motion); a poll takes a screenshot every
            # `poll` seconds (right for a time-lapse of a page that barely
            # repaints, where the screencast goes quiet for minutes).
            if not poll:
                await cdp.call("Page.startScreencast", format="jpeg", quality=quality, maxWidth=width,
                               maxHeight=height, everyNthFrame=max(1, int(every_nth)))
            next_poll = 0.0
            t0 = time.monotonic()
            todo = sorted(actions, key=lambda a: a["at"])
            holds: list[tuple[float, str]] = []           # (release at, key)
            tasks: list[asyncio.Task] = []

            async def act(a: dict):
                if "js" in a:
                    await cdp.js(a["js"])
                if "hold" in a:
                    k = a["hold"]
                    await cdp.js(f"keys.add({json.dumps(k)}); true")
                    holds.append((a["at"] + float(a.get("for", 1.0)), k))
                if "look" in a:
                    dx, dy = a["look"]
                    steps = max(1, int(float(a.get("over", 1.0)) * 30))
                    for _ in range(steps):
                        await cdp.js(f"walkLook({dx / steps}, {dy / steps}); true")
                        await asyncio.sleep(float(a.get("over", 1.0)) / steps)
                if "click" in a:
                    await cdp.js(f"(function(){{const e=document.querySelector({json.dumps(a['click'])}); if(e) e.click(); return !!e}})()")
                if "type" in a:
                    sel, text = a["type"]
                    await cdp.js(f"(function(){{const e=document.querySelector({json.dumps(sel)}); if(!e) return false; "
                                 f"e.value={json.dumps(text)}; e.dispatchEvent(new Event('input',{{bubbles:true}})); "
                                 f"e.dispatchEvent(new Event('change',{{bubbles:true}})); return true}})()")
                if "scroll" in a:
                    await cdp.js(f"window.scrollBy({{top:{float(a['scroll'])}, behavior:'smooth'}}); true")

            closed = False
            while True:
                now = time.monotonic() - t0
                if pump.done():                      # Chrome went away: keep what was captured
                    print("chrome closed the connection early; keeping captured frames", file=sys.stderr)
                    closed = True
                    break
                while todo and todo[0]["at"] <= now:
                    a = todo.pop(0)
                    tasks.append(asyncio.create_task(act(a)))
                tasks = [t for t in tasks if not t.done() or (t.exception() is None)]
                for rel, k in list(holds):
                    if now >= rel:
                        holds.remove((rel, k))
                        tasks.append(asyncio.create_task(cdp.js(f"keys.delete({json.dumps(k)}); true")))
                if poll and now >= next_poll:
                    next_poll = now + poll
                    try:
                        shot = await cdp.call("Page.captureScreenshot", format="jpeg", quality=quality)
                        path = out / f"f_{len(frames):06d}.jpg"
                        path.write_bytes(base64.b64decode(shot["data"]))
                        frames.append((time.monotonic() - t0, path))
                    except Exception as e:  # noqa: BLE001
                        print("screenshot failed:", e, file=sys.stderr)
                try:
                    ev = await asyncio.wait_for(cdp.events.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    ev = None
                if ev and ev.get("method") == "Page.screencastFrame":
                    p = ev["params"]
                    path = out / f"f_{len(frames):06d}.jpg"
                    path.write_bytes(base64.b64decode(p["data"]))
                    frames.append((time.monotonic() - t0, path))
                    await cdp.call("Page.screencastFrameAck", sessionId=p["sessionId"])
                if now >= seconds:
                    break
            if not closed and not poll:
                try:
                    await cdp.call("Page.stopScreencast")
                except Exception:  # noqa: BLE001
                    pass
            for t in tasks:
                try:
                    await t
                except Exception as e:  # noqa: BLE001
                    print("action failed:", e, file=sys.stderr)
            pump.cancel()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            proc.kill()
    if not frames:
        raise RuntimeError("no frames captured")
    # Concat list with real durations, then a constant 30 fps clip.
    lines = []
    for i, (t, p) in enumerate(frames):
        nxt = frames[i + 1][0] if i + 1 < len(frames) else seconds
        lines.append(f"file '{p.name}'\nduration {max(nxt - t, 1 / 60):.4f}")
    lines.append(f"file '{frames[-1][1].name}'")
    (out / "frames.txt").write_text("\n".join(lines) + "\n")
    clip = out / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(out / "frames.txt"),
                    "-vf", f"fps=30,scale={width}:{height}:flags=lanczos,format=yuv420p", "-c:v", "libx264",
                    "-crf", "17", "-preset", "medium", str(clip)], check=True)
    print(f"{len(frames)} frames over {seconds:.1f} s -> {clip}")
    return clip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--actions", default=None, help="json list of steps")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--port", type=int, default=9333)
    ap.add_argument("--settle", type=float, default=4.0, help="seconds to wait after navigation before recording")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--every-nth", type=int, default=1, help="keep every nth repaint (time-lapses)")
    ap.add_argument("--poll", type=float, default=0.0, help="screenshot every N seconds instead of the screencast")
    a = ap.parse_args()
    actions = json.loads(Path(a.actions).read_text()) if a.actions else []
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is needed")
    asyncio.run(run(a.url, Path(a.out), actions, a.seconds, a.width, a.height, a.headed, a.port, a.settle, a.quality,
                    a.every_nth, a.poll))


if __name__ == "__main__":
    main()
