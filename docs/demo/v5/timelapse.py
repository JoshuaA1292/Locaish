"""A real run of the studio page as a time-lapse: the polled screenshots
(one every N seconds) become one frame each at 30 fps, the static tail
after the last change is cut, and a clock in the corner says how much real
time has passed.

    python docs/demo/v5/timelapse.py RUN_DIR OUT_DIR [--poll 2.0] [--tail 1.5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hud  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("out")
    ap.add_argument("--poll", type=float, default=2.0, help="seconds between the recording's screenshots")
    ap.add_argument("--tail", type=float, default=1.5, help="seconds of hold after the last change")
    ap.add_argument("--skip", type=int, default=3, help="screenshots to drop at the start (the blank page)")
    ap.add_argument("--step", type=int, default=1, help="use every Nth screenshot")
    ap.add_argument("--hold", type=int, default=1, help="output frames per screenshot")
    a = ap.parse_args()
    run = Path(a.run)
    files = sorted(run.glob("f_*.jpg"))[a.skip:]
    if not files:
        raise SystemExit("no screenshots")
    # the last screenshot that differs from the one before it
    last = 0
    prev = None
    small = []
    for k, p in enumerate(files):
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (160, 90), interpolation=cv2.INTER_AREA)
        if prev is not None and np.abs(im.astype(np.int16) - prev.astype(np.int16)).mean() > 1.0:
            last = k
        prev = im
    end = min(len(files), last + 1 + int(a.tail * hud.FPS))
    files = files[:end][::max(1, a.step)]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    speed = a.poll * hud.FPS * max(1, a.step) / max(1, a.hold)
    n_out = 0
    for k, p in enumerate(files):
        im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        ih, iw = im.shape[:2]
        s = min(hud.W / iw, hud.H / ih)
        im = cv2.resize(im, (int(iw * s), int(ih * s)), interpolation=cv2.INTER_AREA)
        fr = hud.blank((0, 0, 0))
        y0 = (hud.H - im.shape[0]) // 2; x0 = (hud.W - im.shape[1]) // 2
        fr[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
        real = (k * max(1, a.step) + a.skip) * a.poll
        ov = hud.Overlay()
        ov.tag((hud.W - 60, 60), f"REAL TIME  {int(real // 60):02d}:{int(real % 60):02d}   ·   {speed:.0f}×", size=22, color=hud.ICE, anchor="ra", bg_alpha=190)
        ov.tag((60, 60), "ONE RUN  ·  the studio, unedited", size=18, color=hud.DIM, bg_alpha=190)
        fr = ov.composite(fr)
        if k < 10:
            fr = hud.fade(fr, k / 10)
        for _ in range(max(1, a.hold)):
            hud.save(fr, out / f"f_{n_out:05d}.jpg"); n_out += 1
    hud.encode(out, out / "clip.mp4")
    print(f"{n_out} frames, {n_out / hud.FPS:.1f} s, {files[-1].name} last -> {out / 'clip.mp4'}")


if __name__ == "__main__":
    main()
