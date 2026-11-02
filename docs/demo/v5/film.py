"""The finished scene, shot by shot: at every new shot the measured render
holds for a beat, then a scan wipe reveals what Muse Image made of it; the
slate says the shot, the lens, the size, the subject and the clock.

    python docs/demo/v5/film.py JOB RID OUT_DIR [--hold 1.3]
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
SIZE_WORDS = {"bcu": "BIG CLOSE-UP", "cu": "CLOSE-UP", "mcu": "MEDIUM CLOSE-UP", "ms": "MEDIUM SHOT",
              "mls": "MEDIUM LONG SHOT", "ls": "LONG SHOT", "wide": "WIDE"}


def fit(im: np.ndarray, z: float = 1.0) -> np.ndarray:
    ih, iw = im.shape[:2]
    s = max(hud.W / iw, hud.H / ih) * z
    im = cv2.resize(im, (int(iw * s) + 1, int(ih * s) + 1), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    x0 = (im.shape[1] - hud.W) // 2; y0 = (im.shape[0] - hud.H) // 2
    return im[y0:y0 + hud.H, x0:x0 + hud.W]


def measured_look(im: np.ndarray) -> np.ndarray:
    """The measured render as the engine's own view: cooler, scanlined, crisp edges."""
    f = im.astype(np.float32)
    grey = f.mean(axis=2, keepdims=True)
    f = f * 0.55 + grey * 0.45
    f = f * np.array([0.82, 0.95, 1.08], np.float32)
    f = np.clip(f * 0.9, 0, 255).astype(np.uint8)
    return hud.scanlines(f, 3, 0.86)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job"); ap.add_argument("rid"); ap.add_argument("out")
    ap.add_argument("--hold", type=float, default=1.3)
    a = ap.parse_args()
    d = ROOT / "twins/studio" / a.job / "rehearsals" / a.rid
    reh = json.loads((d / "rehearsal.json").read_text())
    shots = {int(s["number"]): s for s in reh["rehearsal"]["shots"]}
    ani = json.loads((d / "animatic/animatic.json").read_text())
    frames = ani["frames"] if isinstance(ani, dict) else ani
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n = 0
    prev_shot = None
    fps = hud.FPS
    hold = int(a.hold * fps)
    prev = None

    def slate(ov, f, shot_no, st, size, alpha=255):
        ov.rect([48, 44, 48 + 230, 44 + 72], fill=(0, 0, 0), alpha=int(170 * alpha / 255))
        ov.text((64, 50), f"SHOT {shot_no}", kind="head", size=52, color=hud.INK, alpha=alpha)
        parts = [f"{float(st.get('focal_mm') or 0):.0f} mm", size, str(st.get("subject") or "").upper(), f.get("clock") or ""]
        ov.tag((64, 130), "  ·  ".join(p for p in parts if p), size=20, color=hud.INK, bg_alpha=int(150 * alpha / 255))
        if f.get("line") and f.get("speaker"):
            txt = f"{f['speaker'].upper()}:  {f['line']}"
            ov.tag((hud.W / 2, hud.H - 96), txt, kind="body", size=28, color=hud.INK, anchor="ma", bg_alpha=int(170 * alpha / 255), pad=12)

    for f in frames:
        name = f.get("file")
        if not name:
            continue
        fin_p = d / "finish" / f"finish_{name}"
        mea_p = d / "animatic" / name
        if not fin_p.exists() or not mea_p.exists():
            continue
        fin = cv2.cvtColor(cv2.imread(str(fin_p)), cv2.COLOR_BGR2RGB)
        mea = cv2.cvtColor(cv2.imread(str(mea_p)), cv2.COLOR_BGR2RGB)
        cam = f.get("camera") or {}
        shot_no = cam.get("shot")
        shot = shots.get(int(shot_no)) if shot_no is not None else {}
        st = shot.get("setup") or {}
        size = ""
        for z in shot.get("sizes") or []:
            if float(z.get("from", 0)) <= float(f["t"]) <= float(z.get("to", 1e9)):
                size = SIZE_WORDS.get(str(z.get("size") or ""), str(z.get("name") or "").upper())
        new_shot = shot_no != prev_shot
        prev_shot = shot_no
        fin_fit = fit(fin)
        if new_shot:
            mea_fit = measured_look(fit(mea))
            # the measured hold
            for k in range(int(0.55 * fps)):
                fr = mea_fit.copy()
                ov = hud.Overlay()
                ov.bracket((40, 40, hud.W - 40, hud.H - 40), hud.ICE, 36, 2, 200)
                ov.text((hud.W - 64, 56), "MEASURED FRAME", kind="mono", size=20, color=hud.ICE, anchor="ra")
                ov.text((hud.W - 64, 84), f"camera  x {st.get('cam_x', 0):+.2f}  y {st.get('cam_y', 0):+.2f}  h {st.get('cam_z', 0):.2f} m  ·  {st.get('distance_m', 0):.2f} m to {str(st.get('subject') or '')}",
                        kind="mono", size=16, color=hud.DIM, anchor="ra")
                slate(ov, f, shot_no, st, size)
                fr = ov.composite(fr)
                if prev is not None and k < int(0.25 * fps):
                    fr = cv2.addWeighted(prev, 1 - (k + 1) / (0.25 * fps), fr, (k + 1) / (0.25 * fps), 0)
                hud.save(fr, out / f"f_{n:05d}.jpg"); n += 1
            # the wipe
            wn = int(0.5 * fps)
            for k in range(wn):
                u = hud.ease(k / (wn - 1))
                x = int(u * hud.W)
                fr = mea_fit.copy()
                fr[:, :x] = fin_fit[:, :x]
                cv2.line(fr, (x, 0), (x, hud.H), hud.ICE, 3)
                fr = hud.glow(fr, 0.25, 31) if k % 2 == 0 else fr
                ov = hud.Overlay()
                ov.bracket((40, 40, hud.W - 40, hud.H - 40), hud.ICE, 36, 2, int(200 * (1 - u)))
                ov.text((hud.W - 64, 56), "MUSE IMAGE  ·  FINISHING", kind="mono", size=20, color=hud.ICE, anchor="ra")
                slate(ov, f, shot_no, st, size)
                ov.text((hud.W - 60, hud.H - 44), "A RENDER", kind="mono", size=20, color=hud.AMBER, anchor="ra", alpha=int(255 * u))
                hud.save(ov.composite(fr), out / f"f_{n:05d}.jpg"); n += 1
        # the finished hold with a slow push
        for k in range(hold):
            z = 1.0 + 0.04 * k / max(1, hold - 1)
            fr = fit(fin, z) if z > 1.0 else fin_fit.copy()
            ov = hud.Overlay()
            slate(ov, f, shot_no, st, size)
            ov.text((hud.W - 60, hud.H - 44), "A RENDER", kind="mono", size=20, color=hud.AMBER, anchor="ra")
            fr = ov.composite(fr)
            if not new_shot and prev is not None and k < int(0.3 * fps):
                w = (k + 1) / (0.3 * fps)
                fr = cv2.addWeighted(prev, 1 - w, fr, w, 0)
            hud.save(fr, out / f"f_{n:05d}.jpg"); n += 1
            if k == hold - 1:
                prev = fr
    if not n:
        raise SystemExit("no finished frames")
    hud.encode(out, out / "clip.mp4")
    print(f"{n} frames, {n / fps:.1f} s -> {out / 'clip.mp4'}")


if __name__ == "__main__":
    main()
