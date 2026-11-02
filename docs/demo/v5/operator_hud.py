"""The operator's console: the director's sentence, then Muse Wonder's tool
calls through the engine and Muse Spark's DP verdicts on the frames it
looked at, streamed as they happened (four minutes compressed to twelve
seconds), with the frames it looked at beside the log.

    python docs/demo/v5/operator_hud.py JOB SID EVENTS_JSON OUT_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hud  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SECONDS = 14.0
INSTRUCTION = ("A slow handheld push-in on Maya at the window from just inside the door, 16 mm, ending on a "
               "medium close-up as she turns; keep Theo in the edge of frame at the start, never cross the bunk bed.")


def schedule(events: list[dict]) -> list[tuple[float, dict]]:
    """Compress the run onto the clip: event order kept, gaps squeezed."""
    keep = [e for e in events if e["kind"] in ("call", "review", "look", "watch", "stage", "agent")]
    t0, t1 = 2.2, SECONDS - 1.0
    out = []
    last = t0
    span = max(1.0, keep[-1]["_t"] - keep[0]["_t"])
    for e in keep:
        at = t0 + (e["_t"] - keep[0]["_t"]) / span * (t1 - t0)
        at = max(at, last + 0.22)
        out.append((at, e)); last = at
    return out


def line_for(e: dict) -> tuple[str, tuple, str]:
    k = e["kind"]
    if k == "call":
        tool = e.get("tool"); s = e.get("summary") or ""
        if tool == "set_camera_keys":
            keys = (e.get("args") or {}).get("keys") or []
            s = f"camera keys  " + "  →  ".join(f"[{kk['pos'][0]:.2f}, {kk['pos'][1]:.2f}, {kk['pos'][2]:.2f}] {kk['focal_mm']:.0f} mm" for kk in keys[:2])
        return (f"MUSE WONDER  ›  {tool}", hud.ICE, s)
    if k == "review":
        tt = e.get("t")
        where = f"frame at {tt:.2f} s" if tt is not None else "the whole move"
        return (f"MUSE SPARK · DP  ›  {where}  ·  {e.get('score')}/10  ·  {str(e.get('verdict')).upper()}",
                hud.GREEN if e.get("verdict") == "ok" else hud.AMBER, str(e.get("text") or ""))
    if k == "look":
        return ("MUSE SPARK  ›  looks at the frame", hud.DIM, f"{e.get('file')}  ·  t = {e.get('t'):.2f} s")
    if k == "watch":
        return ("MUSE SPARK  ›  watches the move", hud.DIM, f"{len(e.get('frames') or [])} frames · {e.get('file')}")
    if k == "agent":
        return ("MUSE WONDER  ›  rebuilding the move", hud.ICE, "")
    return (str(e.get("text") or ""), hud.DIM, "")


def wrap(s: str, width: int) -> list[str]:
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


def main() -> None:
    job, sid, evp, outp = sys.argv[1:5]
    only = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else None
    events = json.load(open(evp))
    sched = schedule(events)
    looks_dir = ROOT / "twins/studio" / job / "sims" / sid / "direct/looks"
    imgs = {p.name: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in looks_dir.glob("*.png")}
    out = Path(outp); out.mkdir(parents=True, exist_ok=True)
    for i in (only or range(int(SECONDS * hud.FPS))):
        t = i / hud.FPS
        fr = hud.blank()
        ov = hud.Overlay()
        x0 = 70
        ov.text((x0, 52), "LOCAISH  ·  THE OPERATOR", kind="mono", size=18, color=hud.DIM)
        ov.text((x0, 86), "DIRECTOR", kind="mono", size=16, color=hud.ICE)
        typed = hud.typed(INSTRUCTION, t - 0.3, cps=110)
        y = 110
        for ln in wrap(typed, 78):
            ov.text((x0, y), ln, kind="body", size=24, color=hud.INK); y += 32
        # the log
        ly = 230
        ov.line([(x0, ly - 12), (x0 + 1000, ly - 12)], hud.DIM, 1, 110)
        shown = [(at, e) for at, e in sched if at <= t]
        # keep the last N that fit
        rows = []
        for at, e in shown:
            head, col, body = line_for(e)
            body_lines = wrap(body, 96)[:3] if e["kind"] == "review" else wrap(body, 110)[:1]
            rows.append((at, e, head, col, body_lines))
        total_h = sum(26 + 22 * len(b) + 10 for _, _, _, _, b in rows)
        max_h = hud.H - ly - 80
        while rows and total_h > max_h:
            r = rows.pop(0); total_h -= 26 + 22 * len(r[4]) + 10
        yy = ly
        for at, e, head, col, body_lines in rows:
            a = hud.ease((t - at) / 0.25)
            ov.text((x0, yy), f"{e['_t']:6.1f} s", kind="mono", size=15, color=hud.DIM, alpha=int(255 * a))
            ov.text((x0 + 90, yy), head, kind="monob", size=16, color=col, alpha=int(255 * a))
            yy += 26
            for bl in body_lines:
                ov.text((x0 + 90, yy), bl, kind="mono", size=16, color=hud.INK if e["kind"] != "review" else (200, 206, 214), alpha=int(255 * a))
                yy += 22
            yy += 10
        # the frames Muse Spark looked at, on the right
        rx, ry = 1210, 86
        ov.text((rx, 52), "WHAT MUSE SPARK LOOKED AT", kind="mono", size=16, color=hud.ICE)
        looked = [e for at, e in shown if e["kind"] in ("look", "watch")]
        tiles = looked[-4:]
        for n, e in enumerate(tiles):
            im = imgs.get(e.get("file"))
            if im is None:
                continue
            if e["kind"] == "watch":
                w, h = 640, 195
            else:
                w, h = 640, 360
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
            if ry + h > hud.H - 40:
                break
            fr[ry:ry + h, rx:rx + w] = im
            ov.bracket((rx - 4, ry - 4, rx + w + 4, ry + h + 4), hud.ICE, 14, 1, 200)
            rev = next((r for a2, r in shown if r["kind"] == "review" and r.get("t") == e.get("t") and a2 > 0 and r["_t"] > e["_t"]), None)
            cap = f"{e.get('file')}" + (f"   ·   {rev.get('score')}/10 {str(rev.get('verdict')).upper()}" if rev else "")
            ov.tag((rx + 8, ry + h - 34), cap, size=15, color=(hud.AMBER if rev and rev.get("verdict") != "ok" else hud.INK), bg_alpha=170)
            ry += h + 22
        fr = ov.composite(fr)
        fr = hud.vignette(fr, 0.25)
        if t < 0.5:
            fr = hud.fade(fr, t / 0.5)
        hud.save(fr, out / f"f_{i:05d}.jpg")
    if not only:
        hud.encode(out, out / "clip.mp4")
        print("->", out / "clip.mp4")


if __name__ == "__main__":
    main()
