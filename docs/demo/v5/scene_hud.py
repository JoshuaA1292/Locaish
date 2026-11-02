"""The scene beat: the script types in, Muse Spark's breakdown lands beside
it, the shot table comes up with the room's own verdicts (the dolly that
does not fit, in amber), each shot with its measured frame.

    python docs/demo/v5/scene_hud.py JOB RID OUT_DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hud  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SECONDS = 22.0
SIZE_WORDS = {"bcu": "BIG CLOSE-UP", "cu": "CLOSE-UP", "mcu": "MEDIUM CLOSE-UP", "ms": "MEDIUM SHOT",
              "mls": "MEDIUM LONG SHOT", "ls": "LONG SHOT", "els": "EXTREME LONG SHOT"}

SCENE = """INT. DORM ROOM - LATE AFTERNOON

Maya stands at the window with her back to the room,
looking down at the street. Theo comes in fast and
stops just inside the door, out of breath.

THEO: They know.

Maya turns from the window.

MAYA: How long have they known?

THEO: Long enough to be outside.

Maya crosses to the wardrobe and pulls the suitcase
down. Theo stays by the door, watching the corridor.

MAYA: Then we go now."""


def load(job: str, rid: str):
    d = ROOT / "twins/studio" / job / "rehearsals" / rid
    r = json.loads((d / "rehearsal.json").read_text())
    beats = r["scene"]["beats"]
    shots = r["rehearsal"]["shots"]
    ani = json.loads((d / "animatic/animatic.json").read_text())
    frames = ani["frames"] if isinstance(ani, dict) else ani
    thumbs = {}
    for s in shots:
        for f in frames:
            if (f.get("camera") or {}).get("shot") == s["number"]:
                im = cv2.imread(str(d / "animatic" / f["file"]))
                if im is not None:
                    thumbs[s["number"]] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    break
    return beats, shots, thumbs, r


def beat_line(b: dict) -> str:
    kind = b["kind"].upper()
    who = b["actor"]
    if b["kind"] == "line":
        return f'{kind:<5} {who:<5} "{b["text"]}"'
    biz = (b.get("business") or "").strip()
    mark = f" → {b['to_mark']}" if b.get("to_mark") else ""
    return f"{kind:<5} {who:<5} {b.get('pose') or ''}{mark} · {biz}"


def render(i: int, beats, shots, thumbs, reh):
    t = i / hud.FPS
    frame = hud.blank()
    ov = hud.Overlay()
    # left: the script typing in
    x0, y0 = 70, 110
    ov.text((x0, 52), "LOCAISH  ·  THE SCENE", kind="mono", size=18, color=hud.DIM)
    shown = hud.typed(SCENE, t - 0.3, cps=95)
    ov.text((x0, y0), shown, kind="mono", size=22, color=hud.INK, spacing=9)
    if t < 6.5 and int(t * 3) % 2 == 0:
        # a caret
        lines = shown.split("\n")
        cw = ov.width(lines[-1], "mono", 22)
        ov.rect([x0 + cw + 2, y0 + (len(lines) - 1) * 31, x0 + cw + 12, y0 + (len(lines) - 1) * 31 + 24], fill=hud.ICE)
    # right: the breakdown
    rx = 900
    a = hud.ease((t - 4.0) / 0.6)
    if a > 0:
        ov.text((rx, 52), "MUSE SPARK  ·  BREAKDOWN", kind="mono", size=18, color=hud.ICE, alpha=int(255 * a))
        ov.text((rx + 420, 52), f"{len(beats)} beats · 3 marks · {len(shots)} setups", kind="mono", size=18, color=hud.DIM, alpha=int(255 * a))
        for n, b in enumerate(beats):
            ab = hud.ease_out((t - 4.4 - n * 0.35) / 0.4)
            if ab <= 0:
                continue
            col = hud.RED if b["actor"] == "Maya" else hud.BLUE
            yy = 100 + n * 34
            ov.rect([rx, yy + 4, rx + 6, yy + 22], fill=col, alpha=int(255 * ab))
            line = f"{b['number']}  " + beat_line(b)
            if len(line) > 82:
                line = line[:80].rstrip() + "…"
            ov.text((rx + 18 + int(20 * (1 - ab)), yy), line, kind="mono", size=19, color=hud.INK, alpha=int(255 * ab))
    # the shot table with the room's verdicts
    ts = 8.0
    a2 = hud.ease((t - ts) / 0.6)
    if a2 > 0:
        ty = 345
        ov.text((rx, ty), "THE ROOM CHECKS EVERY SHOT", kind="mono", size=18, color=hud.ICE, alpha=int(255 * a2))
        ov.line([(rx, ty + 30), (rx + 950, ty + 30)], hud.DIM, 1, int(120 * a2))
        for n, s in enumerate(shots):
            ash = hud.ease_out((t - ts - 0.5 - n * 0.9) / 0.5)
            if ash <= 0:
                continue
            st = s["setup"]
            yy = ty + 48 + n * 134
            # thumbnail
            th = thumbs.get(s["number"])
            tw, thh = 200, 120
            if th is not None:
                im = cv2.resize(th, (tw, thh), interpolation=cv2.INTER_AREA)
                frame[yy:yy + thh, rx:rx + tw] = (im * ash).astype(np.uint8)
                ov.bracket((rx - 4, yy - 4, rx + tw + 4, yy + thh + 4), hud.ICE, 12, 1, int(200 * ash))
            cx = rx + tw + 28
            size = SIZE_WORDS.get(st.get("shot_size"), str(st.get("shot_size")).upper())
            ov.text((cx, yy), f"SHOT {s['number']}", kind="head", size=34, color=hud.INK, alpha=int(255 * ash))
            ov.text((cx + 150, yy + 8), f"{size}  ·  {st['focal_mm']:.0f} mm  ·  {st['subject'].upper()}", kind="mono", size=20, color=hud.INK, alpha=int(255 * ash))
            ov.text((cx, yy + 46), f"camera  x {st['cam_x']:+.2f}  y {st['cam_y']:+.2f}  h {st['cam_z']:.2f} m   ·   {st['distance_m']:.2f} m to subject   ·   fov {st['fov_h_deg']:.0f}°",
                    kind="mono", size=16, color=hud.DIM, alpha=int(255 * ash))
            probs = s.get("problems") or []
            if probs:
                ap = hud.ease((t - ts - 0.5 - n * 0.9 - 0.6) / 0.4)
                txt = str(probs[0])
                txt = txt.replace("the Doorway dolly run fails 0.0 m along: ", "")
                ov.rect([cx - 10, yy + 72, cx + 660, yy + 118], fill=(60, 36, 6), alpha=int(230 * ap))
                ov.rect([cx - 10, yy + 72, cx - 6, yy + 118], fill=hud.AMBER, alpha=int(255 * ap))
                ov.text((cx + 4, yy + 77), "DOLLY MOVE FAILS", kind="monob", size=16, color=hud.AMBER, alpha=int(255 * ap))
                ov.text((cx + 4, yy + 97), txt[0].upper() + txt[1:], kind="mono", size=15, color=hud.INK, alpha=int(255 * ap))
            else:
                ov.text((cx, yy + 78), "sightline clear  ·  placed", kind="mono", size=16, color=hud.GREEN, alpha=int(255 * ash))
    # the schedule line at the end
    a3 = hud.ease((t - 16.5) / 0.6)
    if a3 > 0:
        sched = (reh.get("schedule") or {}).get("sentences") or []
        yb = 700
        ov.text((x0, yb), "THE SHOOT DAY  ·  14 NOVEMBER  ·  LAID OUT AGAINST THE SUN", kind="mono", size=18, color=hud.ICE, alpha=int(255 * a3))
        ov.line([(x0, yb + 30), (x0 + 760, yb + 30)], hud.DIM, 1, int(120 * a3))
        for n, line in enumerate(sched[:4]):
            al = hud.ease((t - 17.0 - n * 0.3) / 0.4)
            ov.text((x0, yb + 44 + n * 28), line.replace("window coverage holds", "sun through the window"), kind="mono", size=17, color=hud.INK, alpha=int(255 * al))
    frame = ov.composite(frame)
    frame = hud.vignette(frame, 0.3)
    if t < 0.5:
        frame = hud.fade(frame, t / 0.5)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job"); ap.add_argument("rid"); ap.add_argument("out")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    beats, shots, thumbs, reh = load(a.job, a.rid)
    idx = [int(x) for x in a.only.split(",")] if a.only else range(int(SECONDS * hud.FPS))
    for i in idx:
        hud.save(render(i, beats, shots, thumbs, reh), out / f"f_{i:05d}.jpg")
    if not a.only:
        hud.encode(out, out / "clip.mp4")
        print("->", out / "clip.mp4")


if __name__ == "__main__":
    main()
