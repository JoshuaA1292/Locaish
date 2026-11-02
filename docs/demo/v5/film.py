"""The finished scene, shot by shot: for every new setup the measured
render (the bodies) stands beside the scan's own photograph with the
stand-ins placed in it, the photograph wipes into the finished still, and
the still fills the frame; later frames of the setup hold with a slow push.

    python docs/demo/v5/film.py JOB RID OUT_DIR [--hold 1.3] [--skip-shots 6]
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


def fit(im: np.ndarray, w: int = hud.W, h: int = hud.H, z: float = 1.0) -> np.ndarray:
    ih, iw = im.shape[:2]
    s = max(w / iw, h / ih) * z
    im = cv2.resize(im, (int(iw * s) + 1, int(ih * s) + 1), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    x0 = (im.shape[1] - w) // 2; y0 = (im.shape[0] - h) // 2
    return im[y0:y0 + h, x0:x0 + w]


def measured_look(im: np.ndarray) -> np.ndarray:
    f = im.astype(np.float32)
    grey = f.mean(axis=2, keepdims=True)
    f = f * 0.55 + grey * 0.45
    f = f * np.array([0.82, 0.95, 1.08], np.float32)
    f = np.clip(f * 0.9, 0, 255).astype(np.uint8)
    return hud.scanlines(f, 3, 0.86)


def read(p: Path, strip: bool = False):
    if strip:
        from locaish.meta.finish import strip_caption
        buf = np.frombuffer(strip_caption(p.read_bytes()), np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        im = cv2.imread(str(p))
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job"); ap.add_argument("rid"); ap.add_argument("out")
    ap.add_argument("--hold", type=float, default=1.3)
    ap.add_argument("--skip-shots", default="")
    ap.add_argument("--max-per-shot", type=int, default=0, help="frames held per setup (0 = all)")
    ap.add_argument("--render-based", default="skip", choices=["skip", "keep"],
                    help="a frame finished from the render rather than a photograph: skip it when the setup has photograph-based frames")
    a = ap.parse_args()
    skip = {int(x) for x in a.skip_shots.split(",") if x.strip()}
    d = ROOT / "twins/studio" / a.job / "rehearsals" / a.rid
    reh = json.loads((d / "rehearsal.json").read_text())
    shots = {int(s["number"]): s for s in reh["rehearsal"]["shots"]}
    ani = json.loads((d / "animatic/animatic.json").read_text())
    frames = ani["frames"] if isinstance(ani, dict) else ani
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fps = hud.FPS
    hold = int(a.hold * fps)
    n = 0
    prev = None
    prev_shot = None
    per_shot = 0
    half = hud.W // 2

    def slate(ov, f, shot_no, st, size, base_info=None):
        ov.rect([48, 44, 48 + 230, 44 + 72], fill=(0, 0, 0), alpha=170)
        ov.text((64, 50), f"SHOT {shot_no}", kind="head", size=52, color=hud.INK)
        parts = [f"{float(st.get('focal_mm') or 0):.0f} mm", size, str(st.get("subject") or "").upper(), f.get("clock") or ""]
        ov.tag((64, 130), "  ·  ".join(p for p in parts if p), size=20, color=hud.INK, bg_alpha=150)
        if base_info:
            ov.tag((64, 172), base_info, size=16, color=hud.ICE, bg_alpha=150)
        if f.get("line") and f.get("speaker"):
            ov.tag((hud.W / 2, hud.H - 96), f"{f['speaker'].upper()}:  {f['line']}", kind="body", size=28, anchor="ma", bg_alpha=170, pad=12)

    def save(fr):
        nonlocal n
        hud.save(fr, out / f"f_{n:05d}.jpg"); n += 1

    for f in frames:
        name = f.get("file")
        if not name:
            continue
        fin_p = d / "finish" / f"finish_{name}"
        mea_p = d / "animatic" / name
        base_p = d / "finish" / f"finish_{Path(name).stem}_base.png"
        if not fin_p.exists() or not mea_p.exists():
            continue
        cam = f.get("camera") or {}
        shot_no = cam.get("shot")
        if shot_no is not None and int(shot_no) in skip:
            continue
        shot = shots.get(int(shot_no)) if shot_no is not None else {}
        st = shot.get("setup") or {}
        size = ""
        for z in shot.get("sizes") or []:
            if float(z.get("from", 0)) <= float(f["t"]) <= float(z.get("to", 1e9)):
                size = SIZE_WORDS.get(str(z.get("size") or ""), str(z.get("name") or "").upper())
        info = {}
        jp = d / "finish" / f"finish_{Path(name).stem}.json"
        if jp.exists():
            try:
                info = json.loads(jp.read_text()).get("base") or {}
            except ValueError:
                info = {}
        base_info = (f"the scan's own {str(info.get('frame') or '').replace('.jpg', '')}  ·  {float(info.get('distance_m') or 0):.2f} m from the planned camera"
                     if info.get("kind") == "photo" else "finished from the measured render")
        if a.render_based == "skip" and info.get("kind") != "photo":
            continue
        fin = read(fin_p); mea = read(mea_p, strip=True); base = read(base_p) if base_p.exists() else None
        fin_fit = fit(fin)
        new_shot = shot_no != prev_shot
        if not new_shot:
            per_shot += 1
            if a.max_per_shot and per_shot >= a.max_per_shot:
                continue
        else:
            per_shot = 0
        prev_shot = shot_no
        if new_shot:
            left = measured_look(fit(mea, half, hud.H))
            right_src = base if base is not None else mea
            right = fit(right_src, half, hud.H)
            fin_half = fit(fin, half, hud.H)
            # 1. the bodies beside the photograph with the stand-ins (0.9 s)
            for k in range(int(0.9 * fps)):
                fr = np.concatenate([left, right], axis=1)
                ov = hud.Overlay()
                ov.bracket((28, 40, half - 28, hud.H - 40), hud.ICE, 30, 2, 200)
                ov.tag((half / 2, hud.H - 150), "MEASURED  ·  the bodies on their marks", size=18, color=hud.ICE, anchor="ma")
                ov.bracket((half + 28, 40, hud.W - 28, hud.H - 40), hud.AMBER, 30, 2, 200)
                ov.tag((half + half / 2, hud.H - 150), "THE SCAN'S OWN PHOTOGRAPH  ·  the stand-ins placed by the geometry" if base is not None else "THE MEASURED RENDER", size=18, color=hud.AMBER, anchor="ma")
                slate(ov, f, shot_no, st, size, base_info)
                fr = ov.composite(fr)
                if prev is not None and k < int(0.25 * fps):
                    fr = cv2.addWeighted(prev, 1 - (k + 1) / (0.25 * fps), fr, (k + 1) / (0.25 * fps), 0)
                save(fr)
            # 2. the right half wipes top-to-bottom into the people (0.6 s)
            wn = int(0.6 * fps)
            for k in range(wn):
                u = hud.ease(k / (wn - 1))
                y = int(u * hud.H)
                r = right.copy(); r[:y] = fin_half[:y]
                cv2.line(r, (0, y), (half, y), hud.ICE, 3)
                fr = np.concatenate([left, r], axis=1)
                ov = hud.Overlay()
                ov.bracket((28, 40, half - 28, hud.H - 40), hud.ICE, 30, 2, 200)
                ov.tag((half / 2, hud.H - 150), "MEASURED  ·  the bodies on their marks", size=18, color=hud.ICE, anchor="ma")
                ov.tag((half + half / 2, hud.H - 150), "MUSE IMAGE  ·  the people, nothing else", size=18, color=hud.ICE, anchor="ma")
                slate(ov, f, shot_no, st, size, base_info)
                ov.text((hud.W - 60, hud.H - 44), "A RENDER", kind="mono", size=20, color=hud.AMBER, anchor="ra")
                save(ov.composite(fr))
            # 3. the finished still grows to fill the frame (0.5 s)
            gn = int(0.5 * fps)
            for k in range(gn):
                u = hud.ease(k / (gn - 1))
                w = int(half + (hud.W - half) * u)
                x0 = int(half * (1 - u))
                fr = np.concatenate([left, fin_half], axis=1)
                fr[:, x0:x0 + w] = cv2.resize(fin_fit, (w, hud.H), interpolation=cv2.INTER_AREA)
                ov = hud.Overlay()
                slate(ov, f, shot_no, st, size, base_info)
                ov.text((hud.W - 60, hud.H - 44), "A RENDER", kind="mono", size=20, color=hud.AMBER, anchor="ra")
                save(ov.composite(fr))
        for k in range(hold):
            z = 1.0 + 0.04 * k / max(1, hold - 1)
            fr = fit(fin, z=z) if z > 1.0 else fin_fit.copy()
            ov = hud.Overlay()
            slate(ov, f, shot_no, st, size, base_info)
            ov.text((hud.W - 60, hud.H - 44), "A RENDER", kind="mono", size=20, color=hud.AMBER, anchor="ra")
            fr = ov.composite(fr)
            if not new_shot and prev is not None and k < int(0.3 * fps):
                w = (k + 1) / (0.3 * fps)
                fr = cv2.addWeighted(prev, 1 - w, fr, w, 0)
            save(fr)
            if k == hold - 1:
                prev = fr
    if not n:
        raise SystemExit("no finished frames")
    hud.encode(out, out / "clip.mp4")
    print(f"{n} frames, {n / fps:.1f} s -> {out / 'clip.mp4'}")


if __name__ == "__main__":
    main()
