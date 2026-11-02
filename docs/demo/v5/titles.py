"""Title cards for the cut: the hook (typed lines on black), the title, the
section slugs and the end card. Each card is one clip.

    python docs/demo/v5/titles.py OUT_DIR

Writes OUT_DIR/<name>/clip.mp4 for every card in CARDS.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hud  # noqa: E402


def hook(i: int, lines: list[tuple[float, str]], seconds: float) -> np.ndarray:
    """Typed mono lines, each starting at its time; the last one holds."""
    t = i / hud.FPS
    fr = hud.blank((0, 0, 0))
    ov = hud.Overlay()
    y = hud.H / 2 - 34 * len(lines) / 2
    for n, (at, s) in enumerate(lines):
        if t < at:
            break
        shown = hud.typed(s, t - at, cps=34)
        ov.text((hud.W / 2, y + n * 56), shown, kind="mono", size=30, color=hud.INK if n == len(lines) - 1 else hud.DIM, anchor="ma")
        if shown != s and int(t * 4) % 2 == 0:
            w = ov.width(shown, "mono", 30)
            ov.rect([hud.W / 2 + w / 2 + 4, y + n * 56 + 2, hud.W / 2 + w / 2 + 16, y + n * 56 + 30], fill=hud.ICE)
    fr = ov.composite(fr)
    if t > seconds - 0.4:
        fr = hud.fade(fr, (seconds - t) / 0.4)
    return fr


def title(i: int, seconds: float) -> np.ndarray:
    """LOCAISH lands on the hit at frame 0: a flash, then the word, then the line."""
    t = i / hud.FPS
    fr = hud.blank((0, 0, 0))
    flash = max(0.0, 1 - t / 0.25)
    fr[:] = (int(40 * flash), int(60 * flash), int(80 * flash))
    ov = hud.Overlay()
    a = hud.ease_out(t / 0.35)
    size = int(260 + 40 * (1 - a))
    ov.text((hud.W / 2, hud.H / 2 - 40), "LOCAISH", kind="display", size=size, color=hud.INK, anchor="mm", alpha=int(255 * a))
    a2 = hud.ease((t - 0.5) / 0.5)
    ov.text((hud.W / 2, hud.H / 2 + 120), "THE LOCATION, BEFORE ANYONE DRIVES THERE", kind="mono", size=26, color=hud.ICE, anchor="mm", alpha=int(255 * a2))
    a3 = hud.ease((t - 1.0) / 0.5)
    ov.text((hud.W / 2, hud.H / 2 + 170), "Meta's models inside a measured room", kind="body", size=26, color=hud.DIM, anchor="mm", alpha=int(255 * a3))
    ov.bracket((hud.W / 2 - 520, hud.H / 2 - 200, hud.W / 2 + 520, hud.H / 2 + 210), hud.ICE, 40, 2, int(200 * a2))
    fr = ov.composite(fr)
    fr = hud.glow(fr, 0.3, 41)
    if t > seconds - 0.4:
        fr = hud.fade(fr, (seconds - t) / 0.4)
    return fr


def slug(i: int, number: str, head: str, sub: str, seconds: float) -> np.ndarray:
    """A section slug: a number, a headline sliding in, one line under it."""
    t = i / hud.FPS
    fr = hud.blank((0, 0, 0))
    ov = hud.Overlay()
    a = hud.ease_out(t / 0.3)
    ov.text((120 + int(60 * (1 - a)), hud.H / 2 - 150), number, kind="mono", size=26, color=hud.ICE, alpha=int(255 * a))
    ov.text((120 + int(60 * (1 - a)), hud.H / 2 - 110), head, kind="display", size=150, color=hud.INK, alpha=int(255 * a))
    a2 = hud.ease((t - 0.25) / 0.4)
    ov.text((126, hud.H / 2 + 70), sub, kind="body", size=30, color=hud.DIM, alpha=int(255 * a2))
    ov.line([(120, hud.H / 2 + 50), (120 + int(900 * a2), hud.H / 2 + 50)], hud.ICE, 2, int(255 * a2))
    fr = ov.composite(fr)
    if t > seconds - 0.3:
        fr = hud.fade(fr, (seconds - t) / 0.3)
    return fr


def end(i: int, seconds: float) -> np.ndarray:
    t = i / hud.FPS
    fr = hud.blank((0, 0, 0))
    ov = hud.Overlay()
    a = hud.ease(t / 0.6)
    ov.text((hud.W / 2, 330), "LOCAISH", kind="display", size=220, color=hud.INK, anchor="mm", alpha=int(255 * a))
    ov.text((hud.W / 2, 470), "NOTHING A MODEL SAYS BECOMES A MEASUREMENT", kind="mono", size=24, color=hud.ICE, anchor="mm", alpha=int(255 * hud.ease((t - 0.5) / 0.5)))
    rows = [("MUSE SPARK", "reads the room · breaks the scene · reviews every frame"),
            ("MUSE WONDER", "operates the camera through the engine"),
            ("MUSE IMAGE", "finishes the frames and the world"),
            ("SAM 3.1 · SAM 3D", "masks, lifts and stands things up")]
    for n, (k, v) in enumerate(rows):
        al = hud.ease((t - 1.0 - n * 0.2) / 0.4)
        y = 580 + n * 48
        ov.text((hud.W / 2 - 40, y), k, kind="monob", size=22, color=hud.INK, anchor="ra", alpha=int(255 * al))
        ov.text((hud.W / 2 + 0, y), v, kind="mono", size=22, color=hud.DIM, anchor="la", alpha=int(255 * al))
    a4 = hud.ease((t - 2.2) / 0.5)
    ov.text((hud.W / 2, 860), "github.com/JoshuaA1292/Locaish   ·   Meta Global AI Developer Hackathon 2026", kind="mono", size=22, color=hud.DIM, anchor="mm", alpha=int(255 * a4))
    fr = ov.composite(fr)
    if t > seconds - 1.0:
        fr = hud.fade(fr, (seconds - t) / 1.0)
    return fr


CARDS = {
    "hook": (10.0, lambda i: hook(i, [
        (0.2, "A film crew picks a location from photographs."),
        (2.6, "Then someone flies out with a tape measure."),
        (5.2, "This room was scanned with a phone, in 50 seconds."),
    ], 10.0)),
    "title": (4.0, lambda i: title(i, 4.0)),
    "slug_sweep": (2.4, lambda i: slug(i, "02", "THE SWEEP", "every camera position in the room, tested before anyone writes a shot", 2.4)),
    "slug_scene": (2.4, lambda i: slug(i, "03", "THE SCENE", "paste a scene · Muse Spark breaks it down · the room checks every shot", 2.4)),
    "slug_camera": (2.4, lambda i: slug(i, "04", "THE CAMERA", "Muse Wonder operates · Muse Spark reviews as a DP", 2.4)),
    "slug_film": (2.4, lambda i: slug(i, "05", "THE FILM", "Muse Image finishes every frame from the measured render and the scan's own photographs", 2.4)),
    "slug_dress": (2.4, lambda i: slug(i, "06", "THE SET", "SAM 3D lifts the furniture out of the scan · move it, and the room is swept again", 2.4)),
    "slug_world": (2.4, lambda i: slug(i, "07", "THE DAY", "the shoot date and the hour · the sun where it will be", 2.4)),
    "end": (8.0, lambda i: end(i, 8.0)),
}


def main() -> None:
    out = Path(sys.argv[1])
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else list(CARDS)
    for name in names:
        secs, fn = CARDS[name]
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(int(secs * hud.FPS)):
            hud.save(fn(i), d / f"f_{i:05d}.jpg")
        hud.encode(d, d / "clip.mp4")
        print("->", d / "clip.mp4")


if __name__ == "__main__":
    main()
