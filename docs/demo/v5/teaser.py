"""A 14-second, wordless teaser cut from the v5 renders and the recorded
run's own finished frames.

    python docs/demo/v5/teaser.py OUT_DIR [--music build/v5/bed.wav --music-in 1.4]

1. the phone walk; 2. the room assembling around the tracked camera; 3. the
shot table filling in as the room checks every shot (one dolly move fails);
4. the whole scene made: four shots, each the scan's own photograph with
flat stand-ins, wiping into the actors; 5. the last shot held full frame;
6. the title. Captions sit over the footage for 1-2 and above it for 3-4,
so no face is covered.

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

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "docs/demo/build"
FIN = ROOT / "twins/studio/511430a1306a/rehearsals/c03fdb72/finish"
FPS = hud.FPS
XF = 6                      # crossfade frames
TOP = 150                   # the caption band above the content in steps 3-4

# the four shots of the recorded run, in scene order: (file stem, slate)
SHOTS = [
    ("finish_frame_0000", "SHOT 1  ·  25 mm  ·  MAYA"),
    ("finish_frame_0001", "SHOT 2  ·  25 mm  ·  THEO"),
    ("finish_frame_0003", "SHOT 4  ·  25 mm  ·  MAYA"),
    ("finish_frame_0010", "SHOT 7  ·  16 mm  ·  MAYA, THEO"),
]


def read_clip(path: Path, start: float, seconds: float, speed: float) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    src = cap.get(cv2.CAP_PROP_FPS) or FPS
    want = [int(round(start * src + k * speed * src / FPS)) for k in range(int(round(seconds * FPS)))]
    cap.set(cv2.CAP_PROP_POS_FRAMES, want[0])
    i, out, last = want[0], [], None
    for w in want:
        while i <= w:
            ok, fr = cap.read()
            if not ok:
                break
            last = fr
            i += 1
        out.append(cv2.cvtColor(last, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def img(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def fit(im: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = im.shape[:2]
    s = max(w / iw, h / ih)
    im = cv2.resize(im, (int(iw * s) + 1, int(ih * s) + 1), interpolation=cv2.INTER_AREA)
    x0 = (im.shape[1] - w) // 2; y0 = (im.shape[0] - h) // 2
    return im[y0:y0 + h, x0:x0 + w]


def caption_over(frame: np.ndarray, step: str, text: str, a: float) -> np.ndarray:
    """Steps 1-2: the caption over the footage, on a dark ramp at the bottom."""
    ramp = np.clip((np.arange(hud.H) - (hud.H - 330)) / 330.0, 0, 1)[:, None, None] ** 1.4
    fr = (frame.astype(np.float32) * (1 - 0.82 * ramp * a)).astype(np.uint8)
    ov = hud.Overlay()
    x, y, dx = 96, hud.H - 190, int(30 * (1 - a))
    ov.text((x + dx, y), step, kind="monob", size=34, color=hud.ICE, alpha=int(255 * a))
    ov.line([(x + dx, y + 48), (x + dx + 64, y + 48)], hud.ICE, 3, int(255 * a))
    ov.text((x + dx, y + 64), text, kind="display", size=86, color=hud.INK, alpha=int(255 * a))
    return ov.composite(fr)


def caption_top(ov: hud.Overlay, step: str, text: str, a: float) -> None:
    """Steps 3-4: the caption in the band above the content."""
    dx = int(30 * (1 - a))
    ov.text((96 + dx, 34), step, kind="monob", size=30, color=hud.ICE, alpha=int(255 * a))
    ov.text((176 + dx, 22), text, kind="display", size=78, color=hud.INK, alpha=int(255 * a))


def step_plans(seconds: float) -> list[np.ndarray]:
    """The shot table from the scene beat, cropped and enlarged, filling in."""
    src = read_clip(BUILD / "v5/scene/clip.mp4", 8.3, seconds, 1.0)
    x0, y0, x1, y1 = 884, 360, 1860, 800          # the header, shots 1, 2 and 4 and the dolly that fails
    w = hud.W - 160
    h = int((y1 - y0) * w / (x1 - x0))
    out = []
    for k, fr in enumerate(src):
        crop = cv2.resize(fr[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_CUBIC)
        canvas = hud.blank()
        cx = (hud.W - w) // 2
        cy = TOP + (hud.H - TOP - h) // 2
        canvas[cy:cy + h, cx:cx + w] = crop
        ov = hud.Overlay()
        caption_top(ov, "03", "Every shot, checked against the real room", hud.ease_out(min(1.0, k / 8)))
        out.append(hud.vignette(ov.composite(canvas), 0.25))
    return out


def step_scene(seconds: float) -> list[np.ndarray]:
    """Four shots in a grid; each is the photograph with stand-ins, then wipes into the actors."""
    gap = 14
    th = (hud.H - TOP - 20 - gap) // 2
    tw = int(th * 1920 / 1104)
    gx = (hud.W - (2 * tw + gap)) // 2
    tiles = []
    for stem, slate in SHOTS:
        base = FIN / f"{stem}_base.png"
        before = fit(img(base if base.exists() else FIN / f"{stem}.png"), tw, th)
        after = fit(img(FIN / f"{stem}.png"), tw, th)
        tiles.append((before, after, slate))
    n = int(round(seconds * FPS))
    starts = [0.35, 0.95, 1.55, 2.15]     # seconds: when each tile's wipe begins
    wipe = 0.55
    out = []
    for k in range(n):
        t = k / FPS
        canvas = hud.blank()
        ov = hud.Overlay()
        for j, (before, after, slate) in enumerate(tiles):
            r, c = divmod(j, 2)
            x = gx + c * (tw + gap); y = TOP + r * (th + gap)
            u = hud.ease((t - starts[j]) / wipe)
            tile = before.copy()
            if u > 0:
                xx = int(u * tw)
                tile[:, :xx] = after[:, :xx]
                if u < 1:
                    cv2.line(tile, (xx, 0), (xx, th), hud.ICE, 4)
            canvas[y:y + th, x:x + tw] = tile
            ov.bracket((x - 3, y - 3, x + tw + 3, y + th + 3), hud.ICE if u >= 1 else hud.AMBER, 22, 2, 230)
            ov.tag((x + 14, y + th - 40), slate, size=18, color=hud.INK, bg_alpha=170)
            if 0 < u < 1:
                ov.tag((x + tw - 14, y + 16), "MUSE IMAGE", size=16, color=hud.ICE, anchor="ra", bg_alpha=170)
        caption_top(ov, "04", "The whole scene, shot with the cast", hud.ease_out(min(1.0, k / 8)))
        out.append(ov.composite(canvas))
    return out


def step_hold(seconds: float) -> list[np.ndarray]:
    """The last shot full frame, a slow push, its line."""
    im = img(FIN / f"{SHOTS[-1][0]}.png")
    n = int(round(seconds * FPS))
    out = []
    for k in range(n):
        z = 1.0 + 0.035 * k / max(1, n - 1)
        ih, iw = im.shape[:2]
        s = max(hud.W / iw, hud.H / ih) * z
        big = cv2.resize(im, (int(iw * s) + 1, int(ih * s) + 1), interpolation=cv2.INTER_AREA)
        x0 = (big.shape[1] - hud.W) // 2; y0 = (big.shape[0] - hud.H) // 2
        fr = big[y0:y0 + hud.H, x0:x0 + hud.W]
        ov = hud.Overlay()
        ov.tag((hud.W / 2, hud.H - 96), "MAYA:  Then we go now.", kind="body", size=30, anchor="ma", bg_alpha=170, pad=12)
        out.append(ov.composite(fr))
    return out


def end_card(seconds: float) -> list[np.ndarray]:
    n = int(round(seconds * FPS))
    out = []
    for k in range(n):
        t = k / max(1, n - 1)
        ov = hud.Overlay()
        a = hud.ease_out(min(1.0, t * 2.5))
        ov.text((hud.W / 2, hud.H / 2 - 40), "LOCAISH", kind="display", size=200, color=hud.INK, anchor="mm", alpha=int(255 * a))
        a2 = hud.ease(min(1.0, max(0.0, t * 2.5 - 0.4)))
        ov.text((hud.W / 2, hud.H / 2 + 100), "Shoot the scene before anyone drives there.", kind="body", size=40,
                color=hud.ICE, anchor="mm", alpha=int(255 * a2))
        out.append(hud.glow(ov.composite(hud.blank((0, 0, 0))), 0.25, 41))
    return out


def join(seq: list[np.ndarray], frs: list[np.ndarray]) -> None:
    if seq:
        for j in range(XF):
            w = (j + 1) / (XF + 1)
            seq[-XF + j] = cv2.addWeighted(seq[-XF + j], 1 - w, frs[j], w, 0)
        frs = frs[XF:]
    seq.extend(frs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--music", default=str(BUILD / "v5" / "bed.wav"))
    ap.add_argument("--music-in", type=float, default=2.6, help="seconds into the bed: its hit lands as the scene is made")
    a = ap.parse_args()
    out = Path(a.out); fdir = out / "frames"; fdir.mkdir(parents=True, exist_ok=True)
    for p in fdir.glob("*.jpg"):
        p.unlink()
    seq: list[np.ndarray] = []
    phone = read_clip(BUILD / "phone_6x.mp4", 0.0, 1.4, 1.6)
    join(seq, [caption_over(f, "01", "Film a room on your phone", hud.ease_out(min(1.0, k / 8))) for k, f in enumerate(phone)])
    twin = read_clip(BUILD / "v5/cloud/clip.mp4", 2.0, 2.8, 6.0)
    join(seq, [caption_over(f, "02", "It becomes a measured 3D twin", hud.ease_out(min(1.0, k / 8))) for k, f in enumerate(twin)])
    join(seq, step_plans(3.6))
    join(seq, step_scene(3.4))
    join(seq, step_hold(1.9))
    join(seq, end_card(1.4))
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
