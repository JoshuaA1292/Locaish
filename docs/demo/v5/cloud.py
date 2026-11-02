"""The hero shot: the measured room assembling itself around the phone's
tracked path, then Muse Spark's names dropping onto the geometry.

    python docs/demo/v5/cloud.py JOB OUT_DIR [--stride 2] [--workers 6]

Every point is drawn the first time a registered phone frame saw it, so the
room grows exactly as the walk uncovered it; the frustum is the real pose
COLMAP solved for that frame, mapped through the twin's canonical transform.
Nothing here is animated by hand: the path, the timing and the geometry are
the reconstruction's own.
"""
from __future__ import annotations

import argparse
import json
import sys
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hud  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]

SCAN_START, SCAN_END = 1.0, 17.0        # clip seconds over which the 50 s walk plays
LABELS_START = 18.5
CLIP_SECONDS = 27.0
POINT_DIMS = {"bunk bed": "2.24 × 1.09 × 1.12 m", "desk hutch": "1.28 × 0.86 × 1.56 m",
              "small window": "1.93 × 1.18 m · sill 0.62 m"}

G: dict = {}   # the shared scene, filled before the pool forks


def quat_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_scene(job: str, stride: int) -> None:
    from locaish.types import Twin

    jdir = ROOT / "twins/studio" / job
    twin_path = next(jdir.glob("*.twin"))
    twin = Twin.load(twin_path)
    M = np.asarray(twin.canonical_transform, np.float64)
    MR = M[:3, :3]
    xyz = twin.points.xyz[::stride].astype(np.float32)
    rgb = twin.points.rgb[::stride].astype(np.uint8)
    # drop the stereo fuzz: a point whose 6 cm voxel holds fewer than a few
    # neighbours is noise around a surface, not the surface
    vox = np.floor(xyz / 0.06).astype(np.int32)
    key = (vox[:, 0].astype(np.int64) * 73856093) ^ (vox[:, 1].astype(np.int64) * 19349663) ^ (vox[:, 2].astype(np.int64) * 83492791)
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    dense = cnt[inv] >= max(2, 8 // stride)
    xyz, rgb = xyz[dense], rgb[dense]
    man = json.load(open(jdir / "recon/manifest.json"))
    stamps = np.array(man["frames"]["timestamps_s"], np.float64)
    duration = float(man["video"]["duration_s"])
    # COLMAP poses by frame name; centres through the twin's own chain
    names, qs, ts = [], [], []
    for line in open(jdir / "recon/colmap/sparse/0_txt/images.txt"):
        p = line.split()
        if len(p) == 10 and p[-1].endswith(".jpg"):
            names.append(p[-1]); qs.append([float(v) for v in p[1:5]]); ts.append([float(v) for v in p[5:8]])
    order = np.argsort(names)
    qs = np.array(qs)[order]; ts = np.array(ts)[order]; names = [names[i] for i in order]
    factor = float(((twin.provenance.get("steps") or {}).get("video") or {}).get("scale", {}).get("factor"))
    Rwc = np.array([quat_rot(q) for q in qs])                     # world->cam (colmap frame)
    C = np.array([-R.T @ t for R, t in zip(Rwc, ts)]) * factor    # centres, metric colmap frame
    cams = C @ MR.T + M[:3, 3]                                   # canonical
    Rcw = np.array([MR @ R.T for R in Rwc])                       # cam->world, canonical
    # intrinsics of the registered frames
    cam_line = [l.split() for l in open(jdir / "recon/colmap/sparse/0_txt/cameras.txt") if not l.startswith("#")][0]
    fw, fh, f = float(cam_line[2]), float(cam_line[3]), float(cam_line[4])
    hx, hy = fw / 2 / f, fh / 2 / f
    # first frame that saw each point (in-frustum, within 4.5 m), cached
    cache = jdir / "recon" / f"first_seen_s{stride}_d.npy"
    if cache.exists():
        first = np.load(cache)
    else:
        first = np.full(len(xyz), 10_000, np.int32)
        for k in range(len(cams)):
            d = xyz - cams[k].astype(np.float32)
            pc = d @ Rcw[k].astype(np.float32)               # = Rcw^T @ d
            z = pc[:, 2]
            ok = (z > 0.25) & (z < 4.5)
            ok &= np.abs(pc[:, 0]) < hx * z
            ok &= np.abs(pc[:, 1]) < hy * z
            ok &= first == 10_000
            first[ok] = k
        np.save(cache, first)
    keep = first < 10_000
    xyz, rgb, first = xyz[keep], rgb[keep], first[keep]
    labels = json.load(open(jdir / "labels.json"))["labels"]
    seen, lab = set(), []
    wanted = ("bunk bed", "study desk", "desk hutch", "wooden wardrobe", "tower fan", "wall poster",
              "stacked suitcase", "window curtain", "desk lamp", "small window", "lower mattress")
    for l in sorted(labels, key=lambda l: wanted.index(l["label"]) if l["label"] in wanted else 99):
        if l["label"] in seen or l["label"] not in wanted:
            continue
        seen.add(l["label"]); lab.append((l["label"], np.array(l["xyz"], np.float64)))
    floor = float(twin.structure.floor_z)
    lo, hi = twin.points.xyz.min(0), twin.points.xyz.max(0)
    G.update(xyz=xyz, rgb=rgb, first=first, cams=cams, Rcw=Rcw, hx=hx, hy=hy, stamps=stamps,
             duration=duration, labels=lab, floor=floor, lo=lo, hi=hi, names=names,
             frames_dir=jdir / "recon/frames", n_points_total=int(len(twin.points.xyz)), stride=stride)


def view_matrix(eye, target):
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0, 0, 1.0])); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd]).astype(np.float32), eye.astype(np.float32)


def project(P, Rv, eye, fpx):
    pc = (P - eye) @ Rv.T
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = hud.W / 2 + fpx * pc[:, 0] / z
        v = hud.H / 2 + fpx * pc[:, 1] / z
    return u, v, z


def orbit(t: float):
    c = (G["lo"] + G["hi"]) / 2; c[2] = G["floor"] + 1.0
    span = max(G["hi"][0] - G["lo"][0], G["hi"][1] - G["lo"][1])
    az = np.deg2rad(-150 + 80 * hud.ease(t / CLIP_SECONDS))
    el = np.deg2rad(38 + 5 * np.sin(t / CLIP_SECONDS * np.pi))
    r = span * 1.28
    eye = c + r * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return view_matrix(eye, c)


def draw_poly(img, pts3, Rv, eye, fpx, color, thick=1, closed=False):
    u, v, z = project(np.asarray(pts3, np.float32), Rv, eye, fpx)
    pts = [(int(a), int(b)) for a, b, zz in zip(u, v, z) if zz > 0.05]
    if len(pts) < 2:
        return
    cv2.polylines(img, [np.array(pts, np.int32)], closed, color, thick, cv2.LINE_AA)


def render(i: int):
    t = i / hud.FPS
    frame = hud.blank()
    Rv, eye = orbit(t)
    fpx = hud.W / (2 * np.tan(np.deg2rad(52) / 2))
    # the scan clock
    s = hud.ease_out((t - SCAN_START) / (SCAN_END - SCAN_START)) if t >= SCAN_START else 0.0
    vt = s * G["duration"]
    k = int(np.searchsorted(G["stamps"], vt, side="right") - 1) if t >= SCAN_START else -1
    k = min(k, len(G["cams"]) - 1)
    # the floor grid
    g = (18, 26, 36)
    lo, hi, fz = G["lo"], G["hi"], G["floor"]
    for x in np.arange(np.floor(lo[0]), np.ceil(hi[0]) + 0.01, 0.5):
        draw_poly(frame, [[x, lo[1], fz], [x, hi[1], fz]], Rv, eye, fpx, g if abs(x - round(x)) < 0.01 else (10, 14, 20))
    for y in np.arange(np.floor(lo[1]), np.ceil(hi[1]) + 0.01, 0.5):
        draw_poly(frame, [[lo[0], y, fz], [hi[0], y, fz]], Rv, eye, fpx, g if abs(y - round(y)) < 0.01 else (10, 14, 20))
    # the points seen so far, far to near
    shown = 0
    if k >= 0:
        first = G["first"]
        vis = first <= k
        P = G["xyz"][vis]; Cc = G["rgb"][vis]; age = (k - first[vis])
        shown = int(len(P))
        u, v, z = project(P, Rv, eye, fpx)
        ok = (z > 0.3) & (u >= 1) & (u < hud.W - 1) & (v >= 1) & (v < hud.H - 1)
        u, v, z, Cc, age = u[ok], v[ok], z[ok], Cc[ok], age[ok]
        order = np.argsort(-z, kind="stable")
        u = u[order].astype(np.int32); v = v[order].astype(np.int32); z = z[order]; Cc = Cc[order]; age = age[order]
        depth_dim = np.clip(1.15 - 0.06 * z, 0.55, 1.0)[:, None]
        col = (Cc.astype(np.float32) * depth_dim)
        fresh = age < 6
        a = (1.0 - age[fresh] / 6.0)[:, None]
        col[fresh] = col[fresh] * (1 - a) + np.array([200, 245, 255], np.float32) * a
        col = col.astype(np.uint8)
        big = z < 4.0
        frame[v, u] = col
        frame[v[big], u[big] + 1] = col[big]
        frame[v[big] + 1, u[big]] = col[big]
        frame[v[big] + 1, u[big] + 1] = col[big]
        fb = fresh & (z < 7)
        for dx, dy in ((-1, 0), (0, -1), (1, 1), (2, 0), (0, 2)):
            frame[np.clip(v[fb] + dy, 0, hud.H - 1), np.clip(u[fb] + dx, 0, hud.W - 1)] = col[fb]
    # the camera path and the live frustum
    if k >= 0:
        cams = G["cams"]
        trail = cams[: k + 1]
        if len(trail) > 1:
            draw_poly(frame, trail, Rv, eye, fpx, hud.ICE_DEEP, 2)
        for j in range(0, k + 1, 10):
            u, v, z = project(cams[j: j + 1], Rv, eye, fpx)
            if z[0] > 0.1:
                cv2.circle(frame, (int(u[0]), int(v[0])), 3, hud.ICE_DEEP, -1, cv2.LINE_AA)
        C = cams[k]; R = G["Rcw"][k]; d = 0.42
        corners = [C + R @ np.array([sx * G["hx"] * d, sy * G["hy"] * d, d]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        draw_poly(frame, corners, Rv, eye, fpx, hud.ICE, 2, closed=True)
        for c in corners:
            draw_poly(frame, [C, c], Rv, eye, fpx, hud.ICE, 2)
        draw_poly(frame, [C, C + R @ np.array([0, 0, 1.6])], Rv, eye, fpx, (90, 170, 210), 1)
        u, v, z = project(cams[k: k + 1], Rv, eye, fpx)
        cv2.circle(frame, (int(u[0]), int(v[0])), 6, hud.ICE, -1, cv2.LINE_AA)
    frame = hud.glow(frame, 0.35, 21)
    frame = hud.vignette(frame, 0.5)
    ov = hud.Overlay()
    ov.rect([36, 36, 640, 204], fill=(0, 0, 0), alpha=110)
    # labels after the scan
    if t >= LABELS_START:
        for n, (name, p) in enumerate(G["labels"]):
            a = hud.ease_out((t - LABELS_START - n * 0.28) / 0.5)
            if a <= 0:
                continue
            u, v, z = project(np.array([p, p + [0, 0, 0.32]], np.float32), Rv, eye, fpx)
            if z[0] <= 0.3:
                continue
            x0, y0, x1, y1 = float(u[0]), float(v[0]), float(u[1]), float(v[1])
            if not (40 < x0 < hud.W - 40 and 40 < y0 < hud.H - 40):
                continue
            x1 = min(x1, hud.W - 320)
            y1 = y0 + (y1 - y0) * a
            ov.line([(x0, y0), (x1, y1)], hud.ICE, 1, int(220 * a))
            ov.rect([x0 - 3, y0 - 3, x0 + 3, y0 + 3], fill=hud.ICE, alpha=int(255 * a))
            if a > 0.6:
                txt = name
                dims = POINT_DIMS.get(name)
                box = ov.tag((x1 + 8, y1 - 26), txt, size=22, color=hud.INK, bg_alpha=int(190 * a))
                if dims:
                    ov.text((box[0] + 8, box[3] + 4), dims, size=16, color=hud.ICE, alpha=int(255 * a))
    # the HUD
    hud_a = int(255 * hud.ease((t - 0.3) / 0.8))
    ov.text((60, 52), "LOCAISH  ·  RECONSTRUCTION", kind="mono", size=18, color=hud.DIM, alpha=hud_a)
    kk = max(k, 0) + (1 if k >= 0 else 0)
    ov.text((60, 86), f"FRAMES    {kk:>3} / {len(G['cams'])}", kind="mono", size=24, color=hud.INK, alpha=hud_a)
    ov.text((60, 118), f"POINTS    {hud.fmt_int(shown * G['stride']):>10}", kind="mono", size=24, color=hud.INK, alpha=hud_a)
    ov.text((60, 150), "SCALE     0.384 m / unit  ·  metric anchor MoGe-2", kind="mono", size=18, color=hud.DIM, alpha=hud_a)
    ov.text((60, 176), "POSES     COLMAP · ALIKED · GLOMAP  ·  dense OpenMVS", kind="mono", size=18, color=hud.DIM, alpha=hud_a)
    # the walk's timeline
    bx0, by, bw = 60, hud.H - 70, 520
    ov.rect([bx0, by, bx0 + bw, by + 3], fill=(40, 50, 62), alpha=hud_a)
    ov.rect([bx0, by, bx0 + bw * s, by + 3], fill=hud.ICE, alpha=hud_a)
    ov.text((bx0, by - 30), f"PHONE WALK   {vt:05.1f} s / {G['duration']:.1f} s", kind="mono", size=18, color=hud.DIM, alpha=hud_a)
    # the phone's frame, inset
    if k >= 0:
        im = cv2.imread(str(G["frames_dir"] / G["names"][k]))
        if im is not None:
            iw, ih = 512, 288
            im = cv2.cvtColor(cv2.resize(im, (iw, ih), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            x0, y0 = hud.W - iw - 60, 60
            frame[y0:y0 + ih, x0:x0 + iw] = im
            ov.bracket((x0 - 6, y0 - 6, x0 + iw + 6, y0 + ih + 6), hud.ICE, 20, 2, 230)
            ov.text((x0, y0 + ih + 18), f"PHONE  ·  {G['names'][k].replace('.jpg', '')}  ·  {G['stamps'][k]:05.2f} s",
                    kind="mono", size=16, color=hud.ICE)
    frame = ov.composite(frame)
    if t < 0.8:
        frame = hud.fade(frame, t / 0.8)
    return frame


def work(args):
    i, out = args
    hud.save(render(i), Path(out) / f"f_{i:05d}.jpg")
    return i


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job"); ap.add_argument("out")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", type=str, default=None, help="comma list of frame indices to render (a test)")
    ap.add_argument("--seconds", type=float, default=CLIP_SECONDS)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    load_scene(a.job, a.stride)
    print("scene", len(G["xyz"]), "points,", len(G["cams"]), "poses,", len(G["labels"]), "labels", flush=True)
    if a.only:
        for i in [int(x) for x in a.only.split(",")]:
            work((i, out)); print("frame", i, flush=True)
        return
    n = int(a.seconds * hud.FPS)
    with mp.get_context("fork").Pool(a.workers) as pool:
        for j, _ in enumerate(pool.imap_unordered(work, [(i, out) for i in range(n)], chunksize=4)):
            if j % 60 == 0:
                print(f"  {j}/{n}", flush=True)
    hud.encode(out, out / "clip.mp4")
    print("->", out / "clip.mp4")


if __name__ == "__main__":
    main()
