"""A 14-second, wordless teaser cut from the v5 renders: five steps, each a
big caption over its own footage, a hard title at the end.

    python docs/demo/v5/teaser.py OUT_DIR [--music build/v5/bed.wav --music-in 1.4]

Writes OUT_DIR/teaser.mp4 (with the music bed) and OUT_DIR/teaser_silent.mp4.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hud  # noqa: E402

BUILD = Path(__file__).resolve().parents[1] / "build"
FPS = hud.FPS
XF = 5  # crossfade frames

# (clip, seconds in, seconds out, speed, step, caption)
STEPS = [
    ("phone_6x.mp4", 0.0, 1.6, 1.6, "01", "Film a room on your phone"),
    ("v5/cloud/clip.mp4", 2.0, 3.0, 6.0, "02", "It becomes a measured 3D twin"),
    ("v5/sweep/clip.mp4", 1.0, 2.0, 3.0, "03", "Every camera position, tested"),
    ("v5/bodies_r/clip.mp4", 1.0, 2.0, 2.2, "04", "The scene, blocked with the cast"),
    ("v5/film/clip.mp4", 10.4, 4.1, 1.0, "05", "The actors, painted into the real room"),
]
END_SECONDS = 1.6


def frames_of(path: Path, start: float, seconds: float, speed: float) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    want = [int(round(start * src_fps + k * speed * src_fps / FPS)) for k in range(int(round(seconds * FPS)))]
    cap.set(cv2.CAP_PROP_POS_FRAMES, want[0])
    i, out, last = want[0], [], None
    for w in want:
        while i <= w:
            ok, fr = cap.read()
            if not ok:
                break
            last = fr
            i += 1
        out.append(cv2.cvtColor(cv2.resize(last, (hud.W, hud.H)), cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def caption(frame: np.ndarray, step: str, text: str, a: float) -> np.ndarray:
    # a dark band at the bottom so the words read over anything
    f = frame.astype(np.float32)
    ramp = np.clip((np.arange(hud.H) - (hud.H - 330)) / 330.0, 0, 1)[:, None, None] ** 1.4
    f = f * (1 - 0.82 * ramp * a)
    fr = f.astype(np.uint8)
    ov = hud.Overlay()
    x, y = 96, hud.H - 190
    dx = int(30 * (1 - a))
    ov.text((x + dx, y), step, kind="monob", size=34, color=hud.ICE, alpha=int(255 * a))
    ov.line([(x + dx, y + 48), (x + dx + 64, y + 48)], hud.ICE, 3, int(255 * a))
    ov.text((x + dx, y + 64), text, kind="display", size=86, color=hud.INK, alpha=int(255 * a))
    return ov.composite(fr)


def end_card(k: int, n: int) -> np.ndarray:
    t = k / max(1, n - 1)
    fr = hud.blank((0, 0, 0))
    ov = hud.Overlay()
    a = hud.ease_out(min(1.0, t * 2.5))
    ov.text((hud.W / 2, hud.H / 2 - 40), "LOCAISH", kind="display", size=200, color=hud.INK, anchor="mm", alpha=int(255 * a))
    a2 = hud.ease(min(1.0, max(0.0, t * 2.5 - 0.4)))
    ov.text((hud.W / 2, hud.H / 2 + 100), "Shoot the scene before anyone drives there.", kind="body", size=40,
            color=hud.ICE, anchor="mm", alpha=int(255 * a2))
    return hud.glow(ov.composite(fr), 0.25, 41)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--music", default=str(BUILD / "v5" / "bed.wav"))
    ap.add_argument("--music-in", type=float, default=1.4, help="seconds into the bed: its hit lands on the last step")
    a = ap.parse_args()
    out = Path(a.out); fdir = out / "frames"; fdir.mkdir(parents=True, exist_ok=True)
    for p in fdir.glob("*.jpg"):
        p.unlink()
    seq: list[np.ndarray] = []
    for clip, start, secs, speed, step, text in STEPS:
        frs = frames_of(BUILD / clip, start, secs, speed)
        n = len(frs)
        for k, fr in enumerate(frs):
            ca = hud.ease_out(min(1.0, k / 8))
            frs[k] = caption(fr, step, text, ca)
        if seq:
            # crossfade the seam
            for j in range(XF):
                w = (j + 1) / (XF + 1)
                seq[-XF + j] = cv2.addWeighted(seq[-XF + j], 1 - w, frs[j], w, 0)
            frs = frs[XF:]
        seq.extend(frs)
    n_end = int(END_SECONDS * FPS)
    seq.extend(end_card(k, n_end) for k in range(n_end))
    seq[:6] = [hud.fade(f, (j + 1) / 6) for j, f in enumerate(seq[:6])]
    for i, fr in enumerate(seq):
        hud.save(fr, fdir / f"f_{i:05d}.jpg")
    total = len(seq) / FPS
    silent = out / "teaser_silent.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS), "-i", str(fdir / "f_%05d.jpg"),
                    "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "16", "-preset", "slow",
                    "-movflags", "+faststart", str(silent)], check=True)
    with_music = out / "teaser.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(silent), "-ss", f"{a.music_in:.2f}", "-i", a.music,
                    "-filter_complex", f"[1:a]atrim=duration={total:.2f},afade=t=in:d=0.4,"
                    f"afade=t=out:st={total - 1.2:.2f}:d=1.2,volume=2dB[a]",
                    "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart", str(with_music)], check=True)
    print(f"{len(seq)} frames, {total:.1f} s -> {with_music} and {silent}")


if __name__ == "__main__":
    main()
