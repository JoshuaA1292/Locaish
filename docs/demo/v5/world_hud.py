"""The shoot day: sixty measured views of the room wipe into Muse Image's
finished photographs on a grid, then one of them fills the frame.

    python docs/demo/v5/world_hud.py JOB RUN OUT_DIR
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hud  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SECONDS = 14.0
COLS, ROWS = 10, 6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job"); ap.add_argument("run"); ap.add_argument("out")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    d = ROOT / "twins/studio" / a.job / "worlds" / a.run
    info = json.loads((d / "world.json").read_text())
    date, hour = info.get("date"), float(info.get("hour") or 0)
    hh, mm = int(hour), int(round((hour % 1) * 60))
    views = sorted(glob.glob(str(d / "views/view_*.png")))
    fins = sorted(glob.glob(str(d / "finished/*.png")))
    n = min(len(views), len(fins), COLS * ROWS)
    gw, gh = 184, 104
    gx0 = (hud.W - COLS * (gw + 6)) // 2
    gy0 = 170
    V = [cv2.cvtColor(cv2.resize(cv2.imread(p), (gw, gh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for p in views[:n]]
    F = [cv2.cvtColor(cv2.resize(cv2.imread(p), (gw, gh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for p in fins[:n]]
    bright = [float(cv2.imread(p).mean()) for p in fins[:n]]
    hero_i = int(np.argsort(bright)[-4])     # one of the sunlit views, not the brightest wall
    hero = cv2.cvtColor(cv2.imread(fins[hero_i]), cv2.COLOR_BGR2RGB)
    hero_m = cv2.cvtColor(cv2.imread(views[hero_i]), cv2.COLOR_BGR2RGB)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    idx = [int(x) for x in a.only.split(",")] if a.only else range(int(SECONDS * hud.FPS))
    order = np.random.RandomState(3).permutation(n)
    for i in idx:
        t = i / hud.FPS
        fr = hud.blank()
        ov = hud.Overlay()
        ov.text((gx0, 60), f"LOCAISH  ·  THE WORLD  ·  {date}  ·  {hh:02d}:{mm:02d}", kind="mono", size=18, color=hud.DIM)
        ov.text((gx0, 90), f"{n} views of the measured room  ·  finished by Muse Image with the sun where it will be  ·  painted back onto the scan", kind="mono", size=18, color=hud.INK)
        done = 0
        for k in range(n):
            c, r = k % COLS, k // COLS
            x, y = gx0 + c * (gw + 6), gy0 + r * (gh + 6)
            ta = 0.6 + order[k] * (6.5 / n)          # when this tile starts its wipe
            u = hud.ease((t - ta) / 0.5)
            im = V[k].copy()
            if u > 0:
                xx = int(u * gw)
                im[:, :xx] = F[k][:, :xx]
                if u < 1:
                    cv2.line(im, (xx, 0), (xx, gh), hud.ICE, 2)
                else:
                    done += 1
            else:
                im = (im * 0.55).astype(np.uint8)
            fr[y:y + gh, x:x + gw] = im
        ov.text((gx0 + COLS * (gw + 6) - 6, 90), f"FINISHED  {done:>2} / {n}", kind="mono", size=18, color=hud.ICE, anchor="ra")
        ov.text((gx0, gy0 + ROWS * (gh + 6) + 20), "measured view  →  finished photograph   ·   the walls stay where they were measured; only the light is painted",
                kind="mono", size=17, color=hud.DIM)
        fr = ov.composite(fr)
        # one view grows to fill the frame
        a2 = hud.ease((t - 9.0) / 1.2)
        if a2 > 0:
            c, r = hero_i % COLS, hero_i // COLS
            x0, y0 = gx0 + c * (gw + 6), gy0 + r * (gh + 6)
            w = int(gw + (hud.W - gw) * a2); h = int(gh + (hud.H - gh) * a2)
            x = int(x0 * (1 - a2)); y = int(y0 * (1 - a2))
            src = hero if a2 >= 0.5 else hero_m
            big = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA if w < hero.shape[1] else cv2.INTER_CUBIC)
            fr[y:y + h, x:x + w] = big
            if a2 >= 1.0:
                ov2 = hud.Overlay()
                ov2.text((hud.W - 60, hud.H - 44), "A RENDER", kind="mono", size=20, color=hud.AMBER, anchor="ra")
                ov2.tag((60, hud.H - 70), f"view {hero_i:02d}  ·  {date}  {hh:02d}:{mm:02d}  ·  the sun through the west window", size=20)
                fr = ov2.composite(fr)
                z = 1.0 + 0.03 * hud.ease((t - 10.2) / 3.8)
                if z > 1.0:
                    ih, iw = fr.shape[:2]
                    big = cv2.resize(fr, (int(iw * z) + 1, int(ih * z) + 1))
                    xx = (big.shape[1] - iw) // 2; yy = (big.shape[0] - ih) // 2
                    fr = big[yy:yy + ih, xx:xx + iw]
        fr = hud.vignette(fr, 0.3)
        if t < 0.5:
            fr = hud.fade(fr, t / 0.5)
        hud.save(fr, out / f"f_{i:05d}.jpg")
    if not a.only:
        hud.encode(out, out / "clip.mp4")
        print("->", out / "clip.mp4")


if __name__ == "__main__":
    main()
