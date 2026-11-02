"""Render a lifted set piece (its gaussians) as stills on a plain ground.

    python docs/demo/render_piece.py PIECE_gaussians.npz OUT_DIR [--frames 48] [--size 1920x1080]

Each frame turns the piece a little further about its vertical axis; the
gaussians are drawn as soft discs, back to front, lit by a fixed key so the
turn reads as a solid. Nothing here is a measurement: it is the demo's
turntable for a thing SAM 3D Objects lifted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

BG = (10, 10, 11)


def render(xyz, rgb, opacity, cov, yaw, w, h, pitch=0.32):
    c = xyz.mean(axis=0)
    p = xyz - c
    cy, sy = np.cos(yaw), np.sin(yaw)
    x = cy * p[:, 0] - sy * p[:, 1]
    y = sy * p[:, 0] + cy * p[:, 1]
    z = p[:, 2]
    cp, sp = np.cos(pitch), np.sin(pitch)
    depth = cp * y + sp * z          # toward the camera
    up = -sp * y + cp * z
    ext = float(np.percentile(np.abs(np.concatenate([x, up])), 99.5)) * 1.25 + 1e-6
    scale = 0.5 * h / ext
    sx = (w / 2 + x * scale).astype(np.int32)
    sy_ = (h / 2 - up * scale).astype(np.int32)
    size = np.sqrt(np.clip(np.einsum("nii->n", cov), 1e-8, None) / 3.0) * scale
    size = np.clip(size * 1.6, 0.6, 14.0)
    # a cheap key light from the upper left, through the depth gradient
    shade = 0.78 + 0.22 * np.clip((x / ext) * -0.5 + (up / ext) * 0.6 + 0.5, 0, 1)
    col = np.clip(rgb * shade[:, None], 0, 1)
    order = np.argsort(depth)          # far first
    img = np.full((h, w, 3), BG, dtype=np.float32)
    acc = np.zeros((h, w), dtype=np.float32)
    # bucket by disc size so each pass is one vectorised stamp
    for lo, hi in ((0, 1.2), (1.2, 2.5), (2.5, 5), (5, 14.1)):
        sel = order[(size[order] >= lo) & (size[order] < hi)]
        if not len(sel):
            continue
        r = int(np.ceil(hi))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        for i in sel:
            s = size[i]
            d2 = (xx * xx + yy * yy) / (s * s)
            a = np.exp(-1.5 * d2) * float(np.clip(opacity[i], 0, 1)) * 0.9
            x0, y0 = sx[i] - r, sy_[i] - r
            if x0 < 0 or y0 < 0 or x0 + 2 * r + 1 > w or y0 + 2 * r + 1 > h:
                continue
            tile = img[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1]
            tile *= (1 - a)[..., None]
            tile += a[..., None] * (col[i] * 255)
            acc[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1] += a
    out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    # a soft contact shadow under the piece
    mask = Image.fromarray((np.clip(acc, 0, 1) * 255).astype(np.uint8))
    shadow = mask.resize((w, max(1, h // 8))).filter(ImageFilter.GaussianBlur(28))
    sh = np.asarray(shadow, dtype=np.float32) / 255.0 * 0.55
    base_y = int(np.clip(sy_.max() + 8, 0, h - h // 8))
    band = np.asarray(out, dtype=np.float32)
    strip = band[base_y:base_y + sh.shape[0]]
    strip *= (1 - sh[: strip.shape[0]])[..., None]
    return Image.fromarray(np.clip(band, 0, 255).astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--max", type=int, default=400000, help="subsample to this many gaussians")
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.split("x"))
    z = np.load(a.npz)
    xyz, rgb, op, cov = z["xyz"], z["rgb"], z["opacity"], z["cov"]
    if len(xyz) > a.max:
        keep = np.random.default_rng(0).choice(len(xyz), a.max, replace=False)
        xyz, rgb, op, cov = xyz[keep], rgb[keep], op[keep], cov[keep]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(a.frames):
        yaw = 2 * np.pi * i / a.frames + 0.6
        render(xyz, rgb, op, cov, yaw, w, h).save(out / f"turn_{i:03d}.png")
        print(f"\r{i + 1}/{a.frames}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)


if __name__ == "__main__":
    main()
