"""The shot sweep as a tactical plan: the measured room from above, every
tested camera setup lighting up in sweep order, then a few of them answering
a question.

    python docs/demo/v5/sweep.py [--frames 0,120,270,420] [--workers 6]

Every number on screen is the sweep's own: the setup count, the clear and
blocked tallies, the lenses, the close-up candidates and the dolly fits are
read from the columns `locaish.film.sweep.sweep(twin)` returns. The plan is a
density image of the twin's own points between knee and shoulder height.
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

ROOT = Path(__file__).resolve().parents[3]
JOB = ROOT / "twins/studio/511430a1306a"
OUT = ROOT / "docs/demo/build/v5/sweep"

CLIP_SECONDS = 16.0
N_FRAMES = int(CLIP_SECONDS * hud.FPS)

# timing (seconds)
T_PLAN_IN = (0.0, 1.0)
T_SWEEP = (1.0, 7.0)
T_ASK = 7.0
T_FANS = (7.6, 0.55)           # first fan, stagger between fans
T_DOLLY = 11.0

DOLLY_RADIUS_M = 0.68          # a Doorway dolly's clear radius
CLOSE_SIZES = ("cu", "mcu")

# plan placement: centred slightly right, ~70 % of frame height
S0 = 215.0                     # px per metre
PLAN_CX, PLAN_CY = 1150.0, 555.0
COL_X = 72                     # the left readout column

BLOCKED_RGB = np.array([150, 62, 58], np.float32)
G: dict = {}                   # the shared scene, filled before the pool forks


# ---------------------------------------------------------------------------
# data


def lens_ramp(focals: np.ndarray) -> np.ndarray:
    """ice-blue for the widest prime, white for the longest."""
    u = np.unique(focals)
    t = np.searchsorted(u, focals) / max(len(u) - 1, 1)
    a = np.array(hud.ICE_DEEP, np.float32)
    b = np.array(hud.ICE, np.float32)
    c = np.array(hud.INK, np.float32)
    t = t[:, None]
    lo = a + (b - a) * np.clip(t * 2, 0, 1)
    hi = b + (c - b) * np.clip(t * 2 - 1, 0, 1)
    return np.where(t < 0.5, lo, hi).astype(np.float32)


def load_scene() -> None:
    from locaish.types import Twin
    from locaish.film import sweep as sweepmod

    OUT.mkdir(parents=True, exist_ok=True)
    twin = Twin.load(next(JOB.glob("*.twin")))
    floor = float(twin.structure.floor_z)
    cache = OUT / "sweep_columns.npz"
    if cache.exists():
        d = np.load(cache)
        c = {k: d[k] for k in d.files}
        marks = c.pop("subject_marks")
    else:
        sw = sweepmod.sweep(twin)
        print(len(sw)); print(sw.summary())
        c = {k: (v.astype(str) if v.dtype == object else v) for k, v in sw.columns.items()}
        marks = sw.subject_marks
        np.savez_compressed(cache, subject_marks=marks, **c)
    print("setups", len(c["setup_id"]), "lenses", np.unique(c["focal_mm"]).tolist(),
          "sizes", np.unique(c["shot_size"]).tolist())

    # the plan: points between knee and shoulder height, binned at 1 cm
    pcache = OUT / "plan_density.npz"
    if pcache.exists():
        d = np.load(pcache)
        dens, lo, res = d["dens"], d["lo"], float(d["res"])
    else:
        xyz = twin.points.xyz
        band = xyz[(xyz[:, 2] > floor + 0.15) & (xyz[:, 2] < floor + 1.6)][:, :2]
        res = 0.01
        lo = band.min(0)
        ij = ((band - lo) / res).astype(np.int32)
        dens = np.zeros(tuple((ij.max(0) + 1)[::-1]), np.float32)
        np.add.at(dens, (ij[:, 1], ij[:, 0]), 1)
        np.savez_compressed(pcache, dens=dens, lo=lo, res=res, floor=floor)
    # tone: log density, clipped at the 99th percentile of occupied cells,
    # masked to the measured footprint plus a hand's width
    lo_t, hi_t = np.log1p(5.0), np.log1p(np.percentile(dens[dens > 0], 99))
    tone = np.clip((np.log1p(dens) - lo_t) / (hi_t - lo_t), 0, 1).astype(np.float32) ** 1.5
    fp = np.asarray(twin.structure.footprint, np.float64)
    mask = np.zeros_like(tone, np.uint8)
    poly = ((fp - lo) / res).astype(np.int32)
    cv2.fillPoly(mask, [poly], 1)
    mask = cv2.dilate(mask, np.ones((41, 41), np.uint8))
    tone *= mask
    tone = cv2.GaussianBlur(tone, (3, 3), 0)

    fx0, fy0 = fp.min(0)
    fx1, fy1 = fp.max(0)
    room_w, room_d = fx1 - fx0, fy1 - fy0
    world_c = np.array([(fx0 + fx1) / 2, (fy0 + fy1) / 2])

    # each setup gets a slot inside its 0.3 m camera cell so 9,666 rows read
    # as 9,666 dots, not 91; the slot order is the setup order
    cam = np.stack([c["cam_x"], c["cam_y"]], 1).astype(np.float64)
    cells, inv, counts = np.unique(cam, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    rank = np.empty(len(inv), np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rank[order] = np.arange(len(inv)) - starts[inv[order]]
    k = np.ceil(np.sqrt(counts[inv])).astype(np.int64)
    pitch = 0.30 * 0.86
    off = np.stack([((rank % k) + 0.5) / k - 0.5, ((rank // k) + 0.5) / k - 0.5], 1) * pitch
    pos = cam + off

    # the sweep frontier: a line that moves through the cell columns as the
    # count climbs, so the field reads as being scanned
    xs = np.unique(c["cam_x"])
    col_counts = np.array([(c["cam_x"] == x).sum() for x in xs])
    cum = np.concatenate([[0], np.cumsum(col_counts)]).astype(np.float64)
    x_edges = np.concatenate([[xs[0] - 0.15], xs + 0.15]).astype(np.float64)

    vis = c["visible"] > 0
    close = np.isin(c["shot_size"], CLOSE_SIZES)
    n_close_clear = int((vis & close).sum())
    fit = vis & (c["clearance_m"] >= DOLLY_RADIUS_M)
    n_fit, n_clear = int(fit.sum()), int(vis.sum())

    # the fans: for a handful of marks, the three best clear close-ups on the
    # longest lens that frames one
    fans = []
    for m in marks:
        sel = vis & close & np.isclose(c["subj_x"], m[0], atol=1e-3) & np.isclose(c["subj_y"], m[1], atol=1e-3)
        if not sel.any():
            continue
        fmax = c["focal_mm"][sel].max()
        idx = np.where(sel & (c["focal_mm"] == fmax))[0]
        idx = idx[np.argsort(-c["score"][idx])][:3]
        fans.append((m, idx))
    # four marks spread across the room, by x
    fans.sort(key=lambda f: f[0][0])
    if len(fans) > 4:
        pick = np.linspace(0, len(fans) - 1, 4).round().astype(int)
        fans = [fans[i] for i in pick]

    labels = json.load(open(JOB / "labels.json"))["labels"]
    seen, lab = set(), []
    for l in labels:
        xy = np.array(l["xyz"][:2], np.float64)
        inside = (fx0 - 0.2 <= xy[0] <= fx1 + 0.2) and (fy0 - 0.2 <= xy[1] <= fy1 + 0.2)
        if l["label"] in seen or l["kind"] == "structure" or not inside:
            continue
        seen.add(l["label"]); lab.append([l["label"], xy, 0.0])
    # nudge labels that would sit on each other (in metres, at the base scale)
    lab.sort(key=lambda l: (-l[1][1], l[1][0]))
    for i in range(len(lab)):
        for j in range(i):
            dx = (lab[i][1][0] - lab[j][1][0]) * S0
            dy = ((lab[j][1][1] - lab[j][2]) - (lab[i][1][1] - lab[i][2])) * S0
            if abs(dx) < 150 and 0 <= dy < 20:
                lab[i][2] += (20 - dy) / S0

    print("clear", n_clear, "blocked", int((~vis).sum()), "close-ups clear", n_close_clear,
          "dolly fits", n_fit, "of", n_clear, "room", round(room_w, 2), round(room_d, 2),
          "fans", [(tuple(np.round(m, 2)), [(c["shot_size"][i], float(c["focal_mm"][i]), round(float(c["distance_m"][i]), 2))
                                             for i in idx]) for m, idx in fans])
    G.update(c=c, marks=marks, pos=pos, cam=cam, cells=cells, tone=tone, lo=lo, res=res, fp=fp,
             world_c=world_c, room_w=room_w, room_d=room_d, cum=cum, x_edges=x_edges,
             vis=vis, fit=fit, n_fit=n_fit, n_clear=n_clear, n_close_clear=n_close_clear,
             lens_rgb=lens_ramp(c["focal_mm"]), lenses=np.unique(c["focal_mm"]),
             heights=np.unique(c["cam_z"]), fans=fans, labels=lab, n=len(c["setup_id"]),
             twin_name=next(JOB.glob("*.twin")).stem)


# ---------------------------------------------------------------------------
# geometry


def view(t: float):
    """scale (px/m) and screen centre of the plan at time t: a slow push-in
    over the dolly beat."""
    s = S0 * (1.0 + 0.06 * hud.ease((t - T_DOLLY) / (CLIP_SECONDS - T_DOLLY)))
    return s, PLAN_CX, PLAN_CY


def to_screen(xy, s, cx, cy):
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    u = cx + (xy[:, 0] - G["world_c"][0]) * s
    v = cy - (xy[:, 1] - G["world_c"][1]) * s
    return u, v


def plan_layer(s, cx, cy, alpha: float) -> np.ndarray:
    """The density plan and the metre grid, as an RGB float layer."""
    lo, res = G["lo"], G["res"]
    a = res * s
    tx = cx + (lo[0] - G["world_c"][0]) * s
    ty = cy - (lo[1] - G["world_c"][1]) * s
    M = np.array([[a, 0, tx], [0, -a, ty]], np.float64)
    tone = cv2.warpAffine(G["tone"], M, (hud.W, hud.H), flags=cv2.INTER_LINEAR, borderValue=0)
    layer = tone[:, :, None] * np.array([44, 88, 116], np.float32) * alpha
    # the metre grid, over the footprint's extent plus a margin
    fp = G["fp"]
    x0, y0 = fp.min(0) - 0.35
    x1, y1 = fp.max(0) + 0.35
    u0, v1 = to_screen([[x0, y0]], s, cx, cy)
    u1, v0 = to_screen([[x1, y1]], s, cx, cy)
    u0, u1, v0, v1 = int(u0[0]), int(u1[0]), int(v0[0]), int(v1[0])
    grid = np.array([16, 26, 34], np.float32) * alpha
    for gx in np.arange(np.ceil(x0), x1, 1.0):
        u = int(round(cx + (gx - G["world_c"][0]) * s))
        layer[v0:v1, u] += grid
    for gy in np.arange(np.ceil(y0), y1, 1.0):
        v = int(round(cy - (gy - G["world_c"][1]) * s))
        layer[v, u0:u1] += grid
    return layer, (u0, v0, u1, v1)


def draw_dots(layer: np.ndarray, idx: np.ndarray, rgb: np.ndarray, s, cx, cy) -> None:
    """Every setup in idx as a 3x3 dot at its slot; rgb is (len(idx), 3)."""
    if len(idx) == 0:
        return
    u, v = to_screen(G["pos"][idx], s, cx, cy)
    u = np.round(u).astype(np.int64); v = np.round(v).astype(np.int64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            w = 1.0 if (dx == 0 and dy == 0) else (0.55 if (dx == 0 or dy == 0) else 0.22)
            layer[v + dy, u + dx] = np.maximum(layer[v + dy, u + dx], rgb * w)


def sweep_line(layer: np.ndarray, x_world: float, box, s, cx, cy, alpha: float) -> None:
    u = int(round(cx + (x_world - G["world_c"][0]) * s))
    u0, v0, u1, v1 = box
    width = 46
    ramp = np.linspace(0, 1, width, dtype=np.float32) ** 2.2
    a, b = max(u - width, 0), u
    seg = ramp[width - (b - a):][None, :, None]
    layer[v0:v1, a:b] += seg * np.array([70, 150, 190], np.float32) * alpha
    layer[v0:v1, u:u + 2] += np.array([180, 235, 255], np.float32) * alpha


# ---------------------------------------------------------------------------
# the frame


def render(i: int) -> np.ndarray:
    t = i / hud.FPS
    c = G["c"]
    n = G["n"]
    s, cx, cy = view(t)
    plan_a = hud.ease((t - T_PLAN_IN[0]) / (T_PLAN_IN[1] - T_PLAN_IN[0]))
    layer, box = plan_layer(s, cx, cy, plan_a)

    # how many setups have been tested by now
    p = np.clip((t - T_SWEEP[0]) / (T_SWEEP[1] - T_SWEEP[0]), 0, 1)
    p = 1 - (1 - p) ** 1.15
    shown = int(round(n * p))
    idx = np.arange(shown)
    vis = G["vis"][idx]
    rgb = np.where(vis[:, None], G["lens_rgb"][idx], BLOCKED_RGB[None, :])
    if shown:
        # the newest dots flare
        age = (shown - 1 - idx).astype(np.float32)
        flare = 1.0 + 1.1 * np.exp(-age / 160.0)
        rgb = rgb * flare[:, None]
        rgb = np.where(vis[:, None], rgb, rgb * 0.55)
    # the dolly beat recolours the field
    d = hud.ease((t - T_DOLLY) / 0.8)
    if d > 0:
        fit = G["fit"][idx]
        amber = np.array(hud.AMBER, np.float32)
        dolly_rgb = np.where(fit[:, None], amber[None, :], np.where(vis[:, None], G["lens_rgb"][idx] * 0.28, BLOCKED_RGB[None, :] * 0.35))
        rgb = rgb * (1 - d) + dolly_rgb * d
    draw_dots(layer, idx, np.clip(rgb, 0, 255), s, cx, cy)

    if 0 < shown < n:
        xw = float(np.interp(shown, G["cum"], G["x_edges"]))
        sweep_line(layer, xw, box, s, cx, cy, 1.0)
    elif shown == n and t < T_SWEEP[1] + 0.5:
        sweep_line(layer, float(G["x_edges"][-1]), box, s, cx, cy, 1 - (t - T_SWEEP[1]) / 0.5)

    frame = np.clip(layer + np.array(hud.BG, np.float32), 0, 255).astype(np.uint8)
    frame = hud.glow(frame, 0.3, 21)
    frame = hud.vignette(frame)

    ov = hud.Overlay()
    text_a = int(255 * hud.ease((t - 0.3) / 0.6))

    # the fans
    fan_a = 1.0 if t < T_DOLLY else 1 - hud.ease((t - T_DOLLY) / 0.5)
    placed: list = []
    tags_todo: list = []
    if fan_a > 0:
        for k, (m, idx_k) in enumerate(G["fans"]):
            t0 = T_FANS[0] + k * T_FANS[1]
            g = hud.ease_out((t - t0) / 0.55)
            if g <= 0:
                continue
            settle = hud.ease((t - t0 - 1.0) / 0.6)      # the fan dims once the next one has had its turn
            a_k = int(255 * g * fan_a * (1 - 0.35 * settle))
            best = idx_k[0]
            for j, r in enumerate(idx_k):
                camx, camy = float(c["cam_x"][r]), float(c["cam_y"][r])
                sx, sy = float(c["subj_x"][r]), float(c["subj_y"][r])
                ang = np.arctan2(sy - camy, sx - camx)
                half = np.deg2rad(float(c["fov_h_deg"][r])) / 2
                L = float(c["distance_m"][r]) * g
                pts = [(camx, camy)]
                for a in np.linspace(ang - half, ang + half, 9):
                    pts.append((camx + L * np.cos(a), camy + L * np.sin(a)))
                pu, pv = to_screen(np.array(pts), s, cx, cy)
                poly = list(zip(pu.tolist(), pv.tolist()))
                strong = j == 0
                fa = (38 if strong else 16) * (1 - 0.5 * settle)
                oa = (220 if strong else 100) * (1 - (0.3 if strong else 0.6) * settle)
                ov.d.polygon(poly, fill=hud.ICE + (int(fa * g * fan_a),), outline=hud.ICE + (int(oa * g * fan_a),))
                cu_, cv_ = to_screen([[camx, camy]], s, cx, cy)
                ov.d.ellipse([cu_[0] - 5, cv_[0] - 5, cu_[0] + 5, cv_[0] + 5], outline=hud.INK + (a_k,), width=2)
            # the tag by the best camera
            r = best
            cu_, cv_ = to_screen([[float(c["cam_x"][r]), float(c["cam_y"][r])]], s, cx, cy)
            tag = f"{str(c['shot_size'][r]).upper()} · {int(c['focal_mm'][r])} mm · {float(c['distance_m'][r]):.2f} m · clear"
            # place the tag away from the subject
            sx_, _ = to_screen([[float(c["subj_x"][r]), float(c["subj_y"][r])]], s, cx, cy)
            if g > 0.6:
                ta = hud.ease((g - 0.6) / 0.4) * fan_a
                anchor = "ra" if sx_[0] > cu_[0] else "la"
                w = ov.width(tag, size=15)
                if anchor == "la" and cu_[0] + 14 + w > hud.W - 60:
                    anchor = "ra"
                tx = cu_[0] - 14 if anchor == "ra" else cu_[0] + 14
                ty = cv_[0] - 10
                x0 = tx - w if anchor == "ra" else tx
                for _ in range(8):                     # step down until the pill sits clear of the others
                    if not any(x0 < bx1 and x0 + w > bx0 and ty < by1 and ty + 22 > by0 for bx0, by0, bx1, by1 in placed):
                        break
                    ty += 26
                placed.append((x0, ty, x0 + w, ty + 22))
                col = tuple(int(v * ta) for v in hud.INK)
                tags_todo.append(((tx, ty), tag, col, int(190 * ta), anchor))


    # the footprint and the subject marks, on the plan
    fu, fv = to_screen(G["fp"], s, cx, cy)
    ov.line([(float(a), float(b)) for a, b in zip(fu, fv)] + [(float(fu[0]), float(fv[0]))],
            color=hud.ICE, width=1, alpha=int(90 * plan_a))
    mu, mv = to_screen(G["marks"], s, cx, cy)
    for a, b in zip(mu, mv):
        ov.line([(a - 9, b), (a + 9, b)], color=hud.ICE, width=1, alpha=int(210 * plan_a))
        ov.line([(a, b - 9), (a, b + 9)], color=hud.ICE, width=1, alpha=int(210 * plan_a))
    # Muse Spark's names, small, where the object sits
    for name, xy, nudge in G["labels"]:
        lu, lv = to_screen(xy, s, cx, cy)
        ov.rect([lu[0] - 1.5, lv[0] - 1.5, lu[0] + 1.5, lv[0] + 1.5], fill=hud.DIM, alpha=int(200 * plan_a))
        ov.text((lu[0] + 6, lv[0] - 8 + nudge * s), name, size=14, color=hud.DIM, alpha=int(170 * plan_a))
    ov.bracket((box[0] - 10, box[1] - 10, box[2] + 10, box[3] + 10), color=hud.ICE, arm=22, width=2, alpha=int(200 * plan_a))
    # scale bar, 1 m
    bu = box[2] - 10 - s
    bv = box[3] + 30
    ov.line([(bu, bv), (bu + s, bv)], color=hud.DIM, width=1, alpha=int(220 * plan_a))
    ov.line([(bu, bv - 5), (bu, bv + 5)], color=hud.DIM, width=1, alpha=int(220 * plan_a))
    ov.line([(bu + s, bv - 5), (bu + s, bv + 5)], color=hud.DIM, width=1, alpha=int(220 * plan_a))
    ov.text((bu + s / 2, bv + 8), "1 m", size=14, color=hud.DIM, alpha=int(220 * plan_a), anchor="ma")
    ov.text((box[0], box[3] + 24), f"plan · {G['twin_name']} · {len(G['c']['setup_id']):,} rows", size=14,
            color=hud.DIM, alpha=int(190 * plan_a))

    # the left column
    y = 72
    ov.text((COL_X, y), "LOCAISH · SHOT SWEEP", size=20, color=hud.ICE, alpha=text_a)
    ov.text((COL_X, y + 30), "twins/studio/511430a1306a", size=15, color=hud.DIM, alpha=text_a)
    y = 150
    ov.text((COL_X, y), "ROOM", size=15, color=hud.DIM, alpha=text_a)
    ov.text((COL_X, y + 22), f"{G['room_w']:.2f} × {G['room_d']:.2f} m", size=24, color=hud.INK, alpha=text_a)
    ov.text((COL_X, y + 54), f"measured footprint · {len(G['cells'])} camera cells", size=15, color=hud.DIM, alpha=text_a)
    hs = " / ".join(f"{h:.2f}" for h in G["heights"])
    ov.text((COL_X, y + 76), f"{len(G['heights'])} lens heights {hs} m", size=15, color=hud.DIM, alpha=text_a)
    ov.text((COL_X, y + 98), f"{len(G['marks'])} subject marks · {len(G['lenses'])} primes", size=15, color=hud.DIM, alpha=text_a)

    # the counter
    ca = int(255 * hud.ease((t - 0.8) / 0.5))
    y = 290
    ov.text((COL_X, y), "SETUPS TESTED", size=15, color=hud.DIM, alpha=ca)
    num = f"{shown:,}"
    ov.text((COL_X - 2, y + 18), num, kind="display", size=64, color=hud.INK, alpha=ca)
    ov.text((COL_X + ov.width(num, "display", 64) + 12, y + 52), f"/ {n:,}", size=20, color=hud.DIM, alpha=ca)
    n_vis = int(vis.sum()); n_blk = shown - n_vis
    ov.text((COL_X, y + 100), f"clear {n_vis:,}  ·  blocked {n_blk:,}", size=18, color=hud.INK, alpha=ca)
    # the lens legend
    ly = y + 140
    ov.text((COL_X, ly), "PRIME", size=15, color=hud.DIM, alpha=ca)
    lx = COL_X
    uniq = G["lenses"]
    ramp = lens_ramp(uniq)
    for f, col in zip(uniq, ramp):
        ov.rect([lx, ly + 26, lx + 9, ly + 35], fill=tuple(int(v) for v in col), alpha=ca)
        ov.text((lx + 14, ly + 22), f"{int(f)}", size=15, color=hud.INK, alpha=ca)
        lx += 14 + ov.width(f"{int(f)}", size=15) + 16
    ov.text((lx, ly + 22), "mm", size=15, color=hud.DIM, alpha=ca)
    ov.rect([COL_X, ly + 54, COL_X + 9, ly + 63], fill=tuple(int(v) for v in BLOCKED_RGB * 0.7), alpha=ca)
    ov.text((COL_X + 14, ly + 50), "blocked sightline", size=15, color=hud.DIM, alpha=ca)

    # the question
    if t >= T_ASK:
        q = "ask: where can a close-up stand?"
        qy = 540
        shown_q = hud.typed(q, t - T_ASK, cps=38)
        ov.text((COL_X, qy), shown_q, size=20, color=hud.ICE)
        if t - T_ASK > len(q) / 38 + 0.3:
            aa = int(255 * hud.ease((t - T_ASK - len(q) / 38 - 0.3) / 0.5))
            ov.text((COL_X, qy + 34), f"{G['n_close_clear']:,} clear CU / MCU setups", size=22, color=hud.INK, alpha=aa)
            ov.text((COL_X, qy + 64), f"across {len(G['marks'])} marks · showing the 3 best", size=15, color=hud.DIM, alpha=aa)
            ov.text((COL_X, qy + 84), "on the longest lens at 4 marks", size=15, color=hud.DIM, alpha=aa)
    # the dolly
    if t >= T_DOLLY:
        da = int(255 * hud.ease((t - T_DOLLY - 0.4) / 0.6))
        dy_ = 760
        ov.text((COL_X, dy_), "DOORWAY DOLLY", size=20, color=hud.AMBER, alpha=da)
        ov.text((COL_X, dy_ + 30), f"needs {DOLLY_RADIUS_M:.2f} m clear radius", size=15, color=hud.DIM, alpha=da)
        ov.text((COL_X, dy_ + 56), f"fits on {G['n_fit']:,} of {G['n_clear']:,}", size=24, color=hud.INK, alpha=da)
        ov.text((COL_X, dy_ + 88), "clear setups", size=15, color=hud.DIM, alpha=da)
        ov.rect([COL_X, dy_ + 116, COL_X + 9, dy_ + 125], fill=hud.AMBER, alpha=da)
        ov.text((COL_X + 14, dy_ + 112), "dolly fits", size=15, color=hud.DIM, alpha=da)
        ov.rect([COL_X + 150, dy_ + 116, COL_X + 159, dy_ + 125], fill=tuple(int(v * 0.5) for v in hud.ICE), alpha=da)
        ov.text((COL_X + 164, dy_ + 112), "clear, too tight", size=15, color=hud.DIM, alpha=da)

    # time code, bottom left
    ov.text((COL_X, hud.H - 60), f"t+{t:05.2f} s", size=15, color=hud.DIM, alpha=text_a)
    for xy, tag, col, ba, anchor in tags_todo:
        ov.tag(xy, tag, size=15, color=col, bg=(0, 0, 0), bg_alpha=ba, anchor=anchor)

    return ov.composite(frame)


def work(args):
    i, out = args
    hud.save(render(i), Path(out) / f"f_{i:05d}.jpg")
    return i


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="", help="comma list of frame indices to render (test mode)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-encode", action="store_true")
    a = ap.parse_args()
    load_scene()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.frames:
        for i in [int(v) for v in a.frames.split(",")]:
            work((i, OUT))
            print("frame", i)
        return
    with mp.get_context("fork").Pool(a.workers) as pool:
        for j, _ in enumerate(pool.imap_unordered(work, [(i, OUT) for i in range(N_FRAMES)], chunksize=4)):
            if j % 60 == 0:
                print(f"{j}/{N_FRAMES}", flush=True)
    if not a.no_encode:
        hud.encode(OUT, OUT / "clip.mp4")
        print("wrote", OUT / "clip.mp4")


if __name__ == "__main__":
    main()
