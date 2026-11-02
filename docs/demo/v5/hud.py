"""The demo's look, shared by every renderer: fonts, colours, HUD text,
brackets, counters, glow. Frames are RGB uint8 numpy arrays, 1920x1080.

Every renderer imports this so the cut reads as one film: black ground,
ice-blue data, amber for anything a model painted, the cast's red and blue.
"""
from __future__ import annotations

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 30

# colours (RGB)
BG = (5, 7, 10)
INK = (236, 240, 244)
DIM = (120, 132, 146)
ICE = (120, 220, 255)
ICE_DEEP = (40, 140, 210)
AMBER = (255, 170, 60)
RED = (230, 80, 60)
BLUE = (60, 130, 230)
GREEN = (90, 220, 140)

_AVENIR_C = "/System/Library/Fonts/Avenir Next Condensed.ttc"   # 8 heavy, 2 demi, 5 medium
_AVENIR = "/System/Library/Fonts/Avenir Next.ttc"
_MONO = "/System/Library/Fonts/SFNSMono.ttf"
_MENLO = "/System/Library/Fonts/Menlo.ttc"

_cache: dict = {}


def font(kind: str, size: int):
    """kind: 'display' (condensed heavy), 'head' (condensed demi), 'body'
    (Avenir medium), 'mono' (SF Mono), 'monob' (Menlo bold)."""
    key = (kind, size)
    if key in _cache:
        return _cache[key]
    if kind == "display":
        f = ImageFont.truetype(_AVENIR_C, size, index=8)
    elif kind == "head":
        f = ImageFont.truetype(_AVENIR_C, size, index=2)
    elif kind == "body":
        f = ImageFont.truetype(_AVENIR, size, index=5)
    elif kind == "monob":
        f = ImageFont.truetype(_MENLO, size, index=1)
    else:
        f = ImageFont.truetype(_MONO, size)
    _cache[key] = f
    return f


def blank(color=BG) -> np.ndarray:
    im = np.empty((H, W, 3), np.uint8)
    im[:] = color
    return im


class Overlay:
    """Text and vector drawing on an RGBA layer, composited once per frame."""

    def __init__(self):
        self.im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def text(self, xy, s, kind="mono", size=22, color=INK, alpha=255, anchor="la", spacing=4):
        self.d.text(xy, s, font=font(kind, size), fill=tuple(color) + (int(alpha),), anchor=anchor, spacing=spacing)

    def width(self, s, kind="mono", size=22) -> float:
        return self.d.textlength(s, font=font(kind, size))

    def rect(self, box, fill=None, outline=None, width=1, alpha=255):
        f = None if fill is None else tuple(fill) + (int(alpha),)
        o = None if outline is None else tuple(outline) + (int(alpha),)
        self.d.rectangle(box, fill=f, outline=o, width=width)

    def line(self, pts, color=ICE, width=1, alpha=255):
        self.d.line(pts, fill=tuple(color) + (int(alpha),), width=width)

    def bracket(self, box, color=ICE, arm=18, width=2, alpha=255):
        """Viewfinder corners around a box."""
        x0, y0, x1, y1 = box
        c = tuple(color) + (int(alpha),)
        for (x, y, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            self.d.line([(x, y), (x + sx * arm, y)], fill=c, width=width)
            self.d.line([(x, y), (x, y + sy * arm)], fill=c, width=width)

    def tag(self, xy, s, kind="mono", size=20, color=INK, bg=(0, 0, 0), bg_alpha=170, pad=8, anchor="la"):
        """Text on a dark pill. xy is the text anchor (default: left/top)."""
        f = font(kind, size)
        w = self.d.textlength(s, font=f)
        x, y = xy
        if anchor[0] == "r":
            x -= w
        if anchor[0] == "m":
            x -= w / 2
        self.d.rectangle([x - pad, y - pad + 2, x + w + pad, y + size + pad - 2], fill=tuple(bg) + (bg_alpha,))
        self.d.text((x, y), s, font=f, fill=tuple(color) + (255,))
        return (x - pad, y - pad + 2, x + w + pad, y + size + pad - 2)

    def composite(self, frame: np.ndarray) -> np.ndarray:
        base = Image.fromarray(frame).convert("RGBA")
        out = Image.alpha_composite(base, self.im).convert("RGB")
        return np.array(out)


def glow(frame: np.ndarray, mask_color_gain: float = 0.6, radius: int = 25) -> np.ndarray:
    """A soft bloom: blur the bright parts and add them back."""
    k = radius | 1
    bright = cv2.GaussianBlur(frame, (k, k), 0)
    return np.clip(frame.astype(np.int16) + (bright.astype(np.int16) * mask_color_gain), 0, 255).astype(np.uint8)


def vignette(frame: np.ndarray, strength: float = 0.45) -> np.ndarray:
    key = ("vig", strength)
    if key not in _cache:
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        _cache[key] = np.clip(1.0 - strength * np.clip(r - 0.55, 0, 1) ** 1.5 * 1.6, 0, 1)[:, :, None].astype(np.float32)
    return (frame.astype(np.float32) * _cache[key]).astype(np.uint8)


def scanlines(frame: np.ndarray, every: int = 3, dim: float = 0.88) -> np.ndarray:
    out = frame.copy()
    out[::every] = (out[::every].astype(np.float32) * dim).astype(np.uint8)
    return out


def ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ease_out(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def typed(s: str, t: float, cps: float = 40.0) -> str:
    """The first characters of s as typed by time t (seconds)."""
    n = int(t * cps)
    return s[:max(0, n)]


def fmt_int(n: int) -> str:
    return f"{int(n):,}"


def fade(frame: np.ndarray, a: float) -> np.ndarray:
    a = min(max(a, 0.0), 1.0)
    return (frame.astype(np.float32) * a).astype(np.uint8)


def save(frame: np.ndarray, path) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 94])


def encode(frames_dir, out, fps: int = FPS, pattern: str = "f_%05d.jpg") -> None:
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i", f"{frames_dir}/{pattern}",
                    "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "16", "-preset", "medium", str(out)], check=True)
