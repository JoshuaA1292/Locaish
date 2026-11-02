"""The finished film: the rehearsal's frames in shot order, each held with a
slow push, a slate for every shot, and the plan's own words under it.

    python docs/demo/film.py JOB RID OUT_DIR [--hold 2.2]

Reads the rehearsal (shots, setups) and the animatic (per-frame camera and
staging) from the running studio, takes the finished frames, and writes
OUT_DIR/clip.mp4 at 1920x1080, 30 fps. A frame that was never finished is
left out, so the film is only what Muse Image made.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
SERIF = "/System/Library/Fonts/NewYork.ttf"
SANS = "/System/Library/Fonts/SFNS.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"


def font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except Exception:  # noqa: BLE001
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia.ttf", size)


def get(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=60) as r:
        return r.read()


SIZE_WORDS = {"bcu": "Big close-up", "cu": "Close-up", "mcu": "Medium close-up", "ms": "Medium shot",
              "mls": "Medium long shot", "ls": "Long shot", "wide": "Wide"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("rid")
    ap.add_argument("out")
    ap.add_argument("--studio", default="http://127.0.0.1:8765")
    ap.add_argument("--hold", type=float, default=2.2, help="seconds per frame")
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()
    base = a.studio.rstrip("/")
    reh = json.loads(get(base, f"/rehearsal/{a.job}/{a.rid}/rehearsal.json"))
    ani = json.loads(get(base, f"/rehearsal/{a.job}/{a.rid}/animatic.json"))
    finished = set(json.loads(get(base, f"/rehearsal/{a.job}/{a.rid}/finished")).get("files") or [])
    shots = {int(s["number"]): s for s in (reh.get("rehearsal") or {}).get("shots") or []}
    out = Path(a.out)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    big = font(SERIF, 54)
    small = font(SANS, 28)
    mono = font(MONO, 24)
    n = 0
    hold = int(a.hold * a.fps)
    fade = int(0.3 * a.fps)
    prev = None
    seen_shots: set[int] = set()
    for f in ani.get("frames") or []:
        if not f.get("file"):
            continue
        fin = "finish_" + f["file"]
        if fin not in finished:
            continue
        img = Image.open(__import__("io").BytesIO(get(base, f"/rehearsal-image/{a.job}/{a.rid}/{fin}"))).convert("RGB")
        cam = f.get("camera") or {}
        shot_no = cam.get("shot")
        shot = shots.get(int(shot_no)) if shot_no is not None else None
        st = (shot or {}).get("setup") or {}
        focal = cam.get("focal_mm") or st.get("focal_mm") or 0
        subject = st.get("subject") or (f.get("composition") or {}).get("subject") or ""
        sizes = (shot or {}).get("sizes") or []
        size = ""
        for z in sizes:
            if float(z.get("from", 0)) <= float(f["t"]) <= float(z.get("to", 1e9)):
                size = SIZE_WORDS.get(str(z.get("size") or z.get("name") or ""), str(z.get("name") or z.get("size") or ""))
        line = f.get("line")
        speaker = f.get("speaker")
        clock = f.get("clock") or ""
        # fit the frame on the canvas with a slow push over the hold
        for k in range(hold):
            z = 1.0 + 0.045 * (k / max(1, hold - 1))
            iw, ih = img.size
            s = max(W / iw, H / ih) * z
            im = img.resize((int(iw * s) + 1, int(ih * s) + 1), Image.BILINEAR)
            x0 = (im.width - W) // 2
            y0 = (im.height - H) // 2
            canvas = im.crop((x0, y0, x0 + W, y0 + H))
            d = ImageDraw.Draw(canvas, "RGBA")
            # the slate: shot, lens, size, subject; the clock; the line
            slate = f"Shot {shot_no}" if shot_no is not None else "Shot"
            parts = [p for p in (f"{float(focal):.0f} mm" if focal else "", size, subject, clock) if p]
            d.rounded_rectangle([48, 44, 48 + 22 + d.textlength(slate, font=big) + 22, 44 + 78], radius=8, fill=(0, 0, 0, 170))
            d.text((70, 52), slate, font=big, fill=(255, 255, 255))
            sub = "  ·  ".join(parts)
            if sub:
                tw = d.textlength(sub, font=small)
                d.rounded_rectangle([48, 134, 48 + 22 + tw + 22, 134 + 48], radius=6, fill=(0, 0, 0, 150))
                d.text((70, 142), sub, font=small, fill=(230, 230, 230))
            if line and speaker:
                txt = f"{speaker.upper()}:  {line}"
                tw = d.textlength(txt, font=small)
                d.rounded_rectangle([W / 2 - tw / 2 - 24, H - 110, W / 2 + tw / 2 + 24, H - 54], radius=6, fill=(0, 0, 0, 170))
                d.text((W / 2 - tw / 2, H - 100), txt, font=small, fill=(255, 255, 255))
            d.text((W - 150, H - 40), "A RENDER", font=mono, fill=(232, 160, 72))
            # a crossfade from the previous frame over the first `fade` frames
            if prev is not None and k < fade:
                canvas = Image.blend(prev, canvas, (k + 1) / fade)
            canvas.save(frames_dir / f"f_{n:05d}.jpg", quality=93)
            n += 1
        prev = canvas
        if shot_no is not None:
            seen_shots.add(int(shot_no))
    if not n:
        raise SystemExit("no finished frames")
    clip = out / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(a.fps), "-i", str(frames_dir / "f_%05d.jpg"),
                    "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "17", str(clip)], check=True)
    print(f"{n} frames, {n / a.fps:.1f} s, shots {sorted(seen_shots)} -> {clip}")


if __name__ == "__main__":
    main()
