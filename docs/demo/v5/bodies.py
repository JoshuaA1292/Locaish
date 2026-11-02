"""The simulation: the cast as bodies at their real heights on their marks,
moving through the scene inside the measured room, with every shot's camera
drawn where the sweep placed it and the live shot's measured frame inset.

    python docs/demo/v5/bodies.py JOB RID OUT_DIR

Rides on cloud.py's scene (the same crisp cloud, the same orbit maths); the
bodies are point-sampled capsules so the room occludes them properly.
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

cv2.setNumThreads(0)   # cv2's thread pool does not survive a fork

ROOT = Path(__file__).resolve().parents[3]
SECONDS = 12.0
SCENE_PLAY = (1.0, 10.5)      # clip seconds over which the animatic's 0..9 s plays
B: dict = {}


def body_points(h: float, rng: np.random.RandomState, n: int = 60000) -> np.ndarray:
    """Points on a standing figure of height h at the origin, facing +x."""
    pts = []
    # legs: two cylinders
    for sx in (-0.09, 0.09):
        k = n // 6
        a = rng.uniform(0, 2 * np.pi, k); z = rng.uniform(0.0, 0.47 * h, k)
        pts.append(np.stack([0.07 * np.cos(a), sx + 0.07 * np.sin(a), z], 1))
    # torso: an ellipse cylinder
    k = n // 3
    a = rng.uniform(0, 2 * np.pi, k); z = rng.uniform(0.47 * h, 0.84 * h, k)
    pts.append(np.stack([0.12 * np.cos(a), 0.19 * np.sin(a), z], 1))
    # shoulders / arms
    for sx in (-0.24, 0.24):
        k = n // 10
        a = rng.uniform(0, 2 * np.pi, k); z = rng.uniform(0.47 * h, 0.82 * h, k)
        pts.append(np.stack([0.05 * np.cos(a), sx + 0.05 * np.sin(a), z], 1))
    # head: a sphere
    k = n // 6
    u = rng.normal(size=(k, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    pts.append(u * 0.105 + np.array([0, 0, 0.93 * h]))
    return np.concatenate(pts).astype(np.float32)


def load(job: str, rid: str, stride: int):
    cloud.load_scene(job, stride)
    d = ROOT / "twins/studio" / job / "rehearsals" / rid
    reh = json.loads((d / "rehearsal.json").read_text())
    ani = json.loads((d / "animatic/animatic.json").read_text())
    frames = ani["frames"] if isinstance(ani, dict) else ani
    cast = json.loads((ROOT / "twins/studio" / job / "cast.json").read_text())["actors"]
    rng = np.random.RandomState(1)
    bodies = {c["name"]: (body_points(float(c["height_m"]), rng), np.array(c["colour"], np.uint8), float(c["height_m"])) for c in cast}
    shots = {int(s["number"]): s for s in reh["rehearsal"]["shots"]}
    thumbs = {f["file"]: cv2.cvtColor(cv2.imread(str(d / "animatic" / f["file"])), cv2.COLOR_BGR2RGB) for f in frames if f.get("file")}
    B.update(frames=frames, bodies=bodies, shots=shots, thumbs=thumbs, floor=cloud.G["floor"],
             duration=float(frames[-1]["t"]) if frames else 9.0)


def actor_state(name: str, t: float):
    """(x, y, facing_deg) at scene time t, interpolated between animatic frames."""
    fr = B["frames"]
    prev = None
    for f in fr:
        a = (f.get("actors") or {}).get(name)
        if a is None:
            continue
        if float(f["t"]) >= t:
            if prev is None:
                return np.array(a[:2]), float(a[2])
            u = (t - prev[0]) / max(1e-6, float(f["t"]) - prev[0])
            u = hud.ease(u)
            p = np.array(prev[1][:2]) * (1 - u) + np.array(a[:2]) * u
            return p, float(a[2])
        prev = (float(f["t"]), a)
    return (np.array(prev[1][:2]), float(prev[1][2])) if prev else (None, 0.0)


def frame_at(t: float):
    fr = B["frames"]
    k = 0
    for i, f in enumerate(fr):
        if float(f["t"]) <= t:
            k = i
    return fr[k]


def render(i: int):
    t = i / hud.FPS
    G = cloud.G
    st = (t - SCENE_PLAY[0]) / (SCENE_PLAY[1] - SCENE_PLAY[0])
    scene_t = min(max(st, 0.0), 1.0) * B["duration"]
    f = frame_at(scene_t)
    cam = f.get("camera") or {}
    # the orbit: slow, high, from the door side
    c = (G["lo"] + G["hi"]) / 2; c[2] = G["floor"] + 0.9
    span = max(G["hi"][0] - G["lo"][0], G["hi"][1] - G["lo"][1])
    az = np.deg2rad(-120 + 40 * hud.ease(t / SECONDS))
    el = np.deg2rad(34)
    eye = c + span * 1.1 * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    Rv, eye = cloud.view_matrix(eye, c)
    fpx = hud.W / (2 * np.tan(np.deg2rad(50) / 2))
    frame = hud.blank()
    # the grid
    lo, hi, fz = G["lo"], G["hi"], G["floor"]
    for x in np.arange(np.floor(lo[0]), np.ceil(hi[0]) + 0.01, 1.0):
        cloud.draw_poly(frame, [[x, lo[1], fz], [x, hi[1], fz]], Rv, eye, fpx, (18, 26, 36))
    for y in np.arange(np.floor(lo[1]), np.ceil(hi[1]) + 0.01, 1.0):
        cloud.draw_poly(frame, [[lo[0], y, fz], [hi[0], y, fz]], Rv, eye, fpx, (18, 26, 36))
    # the room and the bodies, one z-sorted draw
    P = [G["xyz"]]; C = [(G["rgb"].astype(np.float32) * 0.8).astype(np.uint8)]
    for name, (pts, col, h) in B["bodies"].items():
        xy, facing = actor_state(name, scene_t)
        if xy is None:
            continue
        a = np.deg2rad(facing)
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], np.float32)
        p = pts @ R.T + np.array([xy[0], xy[1], fz], np.float32)
        P.append(p); C.append(np.tile(col, (len(p), 1)))
    P = np.concatenate(P); C = np.concatenate(C)
    u, v, z = cloud.project(P, Rv, eye, fpx)
    ok = (z > 0.3) & (u >= 1) & (u < hud.W - 2) & (v >= 1) & (v < hud.H - 2)
    u, v, z, C = u[ok], v[ok], z[ok], C[ok]
    order = np.argsort(-z, kind="stable")
    u = u[order].astype(np.int32); v = v[order].astype(np.int32); z = z[order]; C = C[order]
    dim = np.clip(1.15 - 0.06 * z, 0.55, 1.0)[:, None]
    col = (C.astype(np.float32) * dim).astype(np.uint8)
    frame[v, u] = col; frame[v, u + 1] = col; frame[v + 1, u] = col; frame[v + 1, u + 1] = col
    # every shot's camera, the live one bright
    live = cam.get("shot")
    for no, s in B["shots"].items():
        su = s["setup"]
        pos = np.array([su["cam_x"], su["cam_y"], su["cam_z"]])
        aim = np.array([su["subj_x"], su["subj_y"], fz + 1.5])
        fwd = aim - pos; fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
        hx = np.tan(np.deg2rad(su["fov_h_deg"] / 2)); hy = hx * 9 / 16
        d = 0.55
        corners = [pos + d * (fwd + sx * hx * right + sy * hy * up) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        colr = hud.ICE if no == live else (70, 110, 140)
        th = 2 if no == live else 1
        cloud.draw_poly(frame, corners, Rv, eye, fpx, colr, th, closed=True)
        for cc in corners:
            cloud.draw_poly(frame, [pos, cc], Rv, eye, fpx, colr, th)
        if no == live:
            cloud.draw_poly(frame, [pos, aim], Rv, eye, fpx, (90, 170, 210), 1)
    frame = hud.glow(frame, 0.3, 21)
    frame = hud.vignette(frame, 0.45)
    ov = hud.Overlay()
    # tags on the cameras and the bodies
    for no, s in B["shots"].items():
        su = s["setup"]
        uu, vv, zz = cloud.project(np.array([[su["cam_x"], su["cam_y"], su["cam_z"] + 0.12]], np.float32), Rv, eye, fpx)
        if zz[0] > 0.3:
            ov.tag((uu[0] + 8, vv[0] - 30), f"SHOT {no} · {su['focal_mm']:.0f} mm", size=16, color=hud.ICE if no == live else hud.DIM, bg_alpha=150)
    for name, (pts, col, h) in B["bodies"].items():
        xy, facing = actor_state(name, scene_t)
        if xy is None:
            continue
        uu, vv, zz = cloud.project(np.array([[xy[0], xy[1], fz + h + 0.12]], np.float32), Rv, eye, fpx)
        if zz[0] > 0.3:
            ov.tag((uu[0], vv[0] - 28), f"{name.upper()}  {h:.2f} m", size=17, color=tuple(int(x) for x in col), bg_alpha=160, anchor="ma")
    ov.rect([36, 36, 700, 150], fill=(0, 0, 0), alpha=110)
    ov.text((60, 52), "LOCAISH  ·  THE SIMULATION", kind="mono", size=18, color=hud.DIM)
    ov.text((60, 84), f"SCENE   {scene_t:04.1f} s   ·   {f.get('clock') or ''}   ·   beat {f.get('beat') or ''}", kind="mono", size=22, color=hud.INK)
    stag = f.get("staging") or {}
    line = "  ·  ".join(f"{n}: {d.get('business')}" for n, d in stag.items() if isinstance(d, dict) and d.get("business"))
    ov.text((60, 116), (line[:92] + "…") if len(line) > 93 else line, kind="mono", size=16, color=hud.DIM)
    # the live measured frame, inset
    th = B["thumbs"].get(f.get("file"))
    if th is not None:
        iw, ih = 512, 306
        im = cv2.resize(th, (iw, ih), interpolation=cv2.INTER_AREA)
        x0, y0 = hud.W - iw - 60, 60
        frame[y0:y0 + ih, x0:x0 + iw] = im
        ov.bracket((x0 - 6, y0 - 6, x0 + iw + 6, y0 + ih + 6), hud.ICE, 20, 2, 230)
        fr_ = f.get("framing") or {}
        ov.text((x0, y0 + ih + 18), f"MEASURED FRAME  ·  shot {live}  ·  {cam.get('focal_mm', 0):.0f} mm  ·  {str(fr_.get('size_name') or '').upper()}  ·  {fr_.get('distance_m', 0):.2f} m",
                kind="mono", size=16, color=hud.ICE)
    if f.get("line") and f.get("speaker"):
        ov.tag((hud.W / 2, hud.H - 90), f"{f['speaker'].upper()}:  {f['line']}", kind="body", size=26, anchor="ma", bg_alpha=170, pad=12)
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
    ap.add_argument("job"); ap.add_argument("rid"); ap.add_argument("out")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    load(a.job, a.rid, a.stride)
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
