"""The set: the bunk bed's own points, found by SAM 3.1 in the frame that
saw it best and lifted by SAM 3D, rise out of the measured cloud, move half
a metre, and settle; the shot table's answer follows.

    python docs/demo/v5/lift_hud.py JOB PIECE_ID OUT_DIR

Rides on cloud.py's scene; the mask is the set piece's own (setpieces/<id>.npy).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hud  # noqa: E402
import cloud  # noqa: E402

cv2.setNumThreads(0)
ROOT = Path(__file__).resolve().parents[3]
SECONDS = 14.0
BASE, AFTER, GAIN = 8414, 8590, 176           # the studio's own re-sweep for this move (0.5 m north)
B: dict = {}


def load(job: str, piece: str, stride: int):
    cloud.load_scene(job, stride)
    jdir = ROOT / "twins/studio" / job
    mask = np.load(jdir / "setpieces" / f"{piece}.npy")
    bed = mask[cloud.G["orig"]]
    pieces = json.loads((jdir / "setpieces.json").read_text())["pieces"]
    info = next(p for p in pieces if p["id"] == piece)
    B.update(bed=bed, info=info, n_returns=int(mask.sum()))
    print("bed points in view", int(bed.sum()), "of", int(mask.sum()))


def phase(t: float):
    """lift (m), move (m), glow (0-1), counter progress (0-1), phase name."""
    if t < 3.0:
        return 0.0, 0.0, hud.ease((t - 0.8) / 0.8), 0.0, "found"
    if t < 5.5:
        u = hud.ease((t - 3.0) / 2.0)
        return 0.9 * u, 0.0, 1.0, 0.0, "lifted"
    if t < 8.0:
        u = hud.ease((t - 5.5) / 2.0)
        return 0.9, 0.5 * u, 1.0, 0.0, "moved"
    if t < 10.0:
        u = hud.ease((t - 8.0) / 1.6)
        return 0.9 * (1 - u), 0.5, 1.0 - 0.6 * u, 0.0, "set down"
    return 0.0, 0.5, 0.4, hud.ease((t - 10.0) / 1.0), "swept again"


def render(i: int):
    t = i / hud.FPS
    G = cloud.G
    lift, move, glow, prog, name = phase(t)
    c = (G["lo"] + G["hi"]) / 2; c[2] = G["floor"] + 0.8
    span = max(G["hi"][0] - G["lo"][0], G["hi"][1] - G["lo"][1])
    az = np.deg2rad(-95 + 30 * hud.ease(t / SECONDS))
    el = np.deg2rad(36)
    eye = c + span * 1.1 * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    Rv, eye = cloud.view_matrix(eye, c)
    fpx = hud.W / (2 * np.tan(np.deg2rad(50) / 2))
    frame = hud.blank()
    lo, hi, fz = G["lo"], G["hi"], G["floor"]
    for x in np.arange(np.floor(lo[0]), np.ceil(hi[0]) + 0.01, 1.0):
        cloud.draw_poly(frame, [[x, lo[1], fz], [x, hi[1], fz]], Rv, eye, fpx, (18, 26, 36))
    for y in np.arange(np.floor(lo[1]), np.ceil(hi[1]) + 0.01, 1.0):
        cloud.draw_poly(frame, [[lo[0], y, fz], [hi[0], y, fz]], Rv, eye, fpx, (18, 26, 36))
    bed = B["bed"]
    P = G["xyz"].copy()
    C = (G["rgb"].astype(np.float32) * 0.8)
    P[bed, 2] += lift
    P[bed, 1] += move
    amber = np.array(hud.AMBER, np.float32)
    C[bed] = C[bed] * (1 - 0.75 * glow) + amber * (0.75 * glow)
    C = C.astype(np.uint8)
    u, v, z = cloud.project(P, Rv, eye, fpx)
    ok = (z > 0.3) & (u >= 1) & (u < hud.W - 2) & (v >= 1) & (v < hud.H - 2)
    u, v, z, C, bb = u[ok], v[ok], z[ok], C[ok], bed[ok]
    order = np.argsort(-z, kind="stable")
    u = u[order].astype(np.int32); v = v[order].astype(np.int32); z = z[order]; C = C[order]; bb = bb[order]
    dim = np.clip(1.15 - 0.06 * z, 0.55, 1.0)[:, None]
    col = (C.astype(np.float32) * dim).astype(np.uint8)
    frame[v, u] = col; frame[v, u + 1] = col; frame[v + 1, u] = col; frame[v + 1, u + 1] = col
    if glow > 0.5:
        for dx, dy in ((-1, 0), (0, -1), (2, 0), (0, 2)):
            frame[np.clip(v[bb] + dy, 0, hud.H - 1), np.clip(u[bb] + dx, 0, hud.W - 1)] = col[bb]
    # the footprint on the floor: where it was, and where it goes
    fp = B["info"]["footprint"]
    cs, sn = np.cos(np.deg2rad(fp["yaw_deg"])), np.sin(np.deg2rad(fp["yaw_deg"]))
    hw, hd = fp["width_m"] / 2, fp["depth_m"] / 2
    for dy_, colr, th in (((0.0), (120, 90, 40), 1), ((move), hud.AMBER, 2)):
        pts = [[fp["cx"] + cs * sx * hw - sn * sy * hd, fp["cy"] + dy_ + sn * sx * hw + cs * sy * hd, fz + 0.01]
               for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        if dy_ == 0.0 or move > 0.01:
            cloud.draw_poly(frame, pts, Rv, eye, fpx, colr, th, closed=True)
    frame = hud.glow(frame, 0.3, 21)
    frame = hud.vignette(frame, 0.45)
    ov = hud.Overlay()
    ov.rect([36, 36, 720, 230], fill=(0, 0, 0), alpha=120)
    ov.text((60, 52), "LOCAISH  ·  THE SET", kind="mono", size=18, color=hud.DIM)
    ov.text((60, 86), "BUNK BED", kind="head", size=44, color=hud.INK)
    ov.text((60, 140), f"{fp['width_m']:.2f} × {fp['depth_m']:.2f} × {fp['height_m']:.2f} m measured  ·  {B['n_returns']:,} returns in the scan",
            kind="mono", size=18, color=hud.INK)
    ov.text((60, 168), "SAM 3.1 masked it in the frame that saw it best  ·  SAM 3D Objects lifted it into a solid", kind="mono", size=17, color=hud.DIM)
    ov.text((60, 194), {"found": "its own points, found in the cloud", "lifted": "lifted out of the scan",
                        "moved": "moved 0.50 m north", "set down": "set down where the director wants it",
                        "swept again": "the room swept again, as dressed"}[name], kind="mono", size=20, color=hud.AMBER)
    if name == "swept again" or prog > 0:
        shown = int(round(BASE + (AFTER - BASE) * prog))
        ov.text((60, 260), f"clear setups   {BASE:,}  →  {shown:,}", kind="mono", size=26, color=hud.INK, alpha=int(255 * min(1, prog * 2)))
        if prog >= 1.0:
            ov.text((60, 300), f"+{GAIN}", kind="display", size=96, color=hud.GREEN)
            ov.text((60, 410), "every frame downstream follows it", kind="mono", size=18, color=hud.DIM)
    ov.text((hud.W - 60, hud.H - 44), "the measured room  ·  the piece's own points", kind="mono", size=16, color=hud.DIM, anchor="ra")
    frame = ov.composite(frame)
    if t < 0.5:
        frame = hud.fade(frame, t / 0.5)
    return frame


def work(args):
    i, out = args
    hud.save(render(i), Path(out) / f"f_{i:05d}.jpg")
    return i


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job"); ap.add_argument("piece"); ap.add_argument("out")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    load(a.job, a.piece, a.stride)
    if a.only:
        for i in [int(x) for x in a.only.split(",")]:
            work((i, out))
        return
    n = int(SECONDS * hud.FPS)
    with mp.get_context("fork").Pool(a.workers) as pool:
        for j, _ in enumerate(pool.imap_unordered(work, [(i, out) for i in range(n)], chunksize=4)):
            if j % 60 == 0:
                print(f"  {j}/{n}", flush=True)
    hud.encode(out, out / "clip.mp4")
    print("->", out / "clip.mp4")


if __name__ == "__main__":
    main()
