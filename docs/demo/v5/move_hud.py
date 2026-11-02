"""The set: the bunk bed lifted out of the scan, moved on the measured plan,
and the shot table's answer each time the room is swept again as dressed.

    python docs/demo/v5/move_hud.py OUT_DIR

Rides on sweep.py's plan (same scene cache, same look).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hud  # noqa: E402
import sweep  # noqa: E402

SECONDS = 13.0
# the studio's own re-sweep sentences for this piece (job 511430a1306a, piece 2beb74c2)
BASE = 8414
NORTH = (0.5, 8590, "the room gained 176 clear setups overall (8414 to 8590)")
SOUTH = (-0.5, 5708, "the room lost 2,706 clear setups (8414 to 5708) · a 16 mm long shot lost 273")
PIECE = {"cx": 1.2537, "cy": 0.9682, "w": 2.2413, "d": 1.0851, "h": 1.1179, "yaw": 6.765, "returns": 275139}


def footprint(dy: float):
    c, s = np.cos(np.deg2rad(PIECE["yaw"])), np.sin(np.deg2rad(PIECE["yaw"]))
    hw, hd = PIECE["w"] / 2, PIECE["d"] / 2
    pts = np.array([[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]])
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([PIECE["cx"], PIECE["cy"] + dy])


def render(i: int, mask_img) -> np.ndarray:
    t = i / hud.FPS
    s, cx, cy = sweep.S0, sweep.PLAN_CX, sweep.PLAN_CY
    layer, box = sweep.plan_layer(s, cx, cy, 1.0)
    G = sweep.G
    idx = np.arange(G["n"])
    rgb = np.where(G["vis"][:, None], G["lens_rgb"] * 0.45, np.array([70, 40, 40], np.float32) * 0.5)
    sweep.draw_dots(layer, idx, rgb, s, cx, cy)
    frame = np.clip(layer, 0, 255).astype(np.uint8)
    # the move
    if t < 2.5:
        dy, phase = 0.0, "lifted"
    elif t < 4.0:
        dy, phase = NORTH[0] * hud.ease((t - 2.5) / 1.0), "north"
    elif t < 6.5:
        dy, phase = NORTH[0], "north"
    elif t < 8.0:
        dy, phase = NORTH[0] + (SOUTH[0] - NORTH[0]) * hud.ease((t - 6.5) / 1.2), "south"
    elif t < 10.5:
        dy, phase = SOUTH[0], "south"
    elif t < 11.5:
        dy, phase = SOUTH[0] * (1 - hud.ease((t - 10.5) / 0.8)), "reset"
    else:
        dy, phase = 0.0, "reset"
    # the footprint, filled amber, with its ghost at the measured place
    u0, v0 = sweep.to_screen(footprint(0.0), s, cx, cy)
    ghost = np.stack([u0, v0], 1).astype(np.int32)
    cv2.polylines(frame, [ghost], True, (120, 90, 40), 1, cv2.LINE_AA)
    u, v = sweep.to_screen(footprint(dy), s, cx, cy)
    poly = np.stack([u, v], 1).astype(np.int32)
    fill = frame.copy()
    cv2.fillPoly(fill, [poly], hud.AMBER)
    frame = cv2.addWeighted(frame, 0.78, fill, 0.22, 0)
    cv2.polylines(frame, [poly], True, hud.AMBER, 2, cv2.LINE_AA)
    frame = hud.glow(frame, 0.3, 21)
    frame = hud.vignette(frame, 0.35)
    ov = hud.Overlay()
    ov.text((60, 52), "LOCAISH  ·  THE SET", kind="mono", size=18, color=hud.DIM)
    ov.text((60, 90), "BUNK BED", kind="head", size=44, color=hud.INK)
    ov.text((60, 146), f"{PIECE['w']:.2f} × {PIECE['d']:.2f} × {PIECE['h']:.2f} m measured  ·  {PIECE['returns']:,} returns in the scan", kind="mono", size=18, color=hud.INK)
    ov.text((60, 174), "SAM 3.1 masks it in the frame that saw it best  ·  SAM 3D Objects lifts it into a solid", kind="mono", size=18, color=hud.DIM)
    ov.text((60, 198), "fitted to its measured footprint  ·  movable", kind="mono", size=18, color=hud.DIM)
    # the readout
    y = 300
    ov.text((60, y), "THE ROOM, SWEPT AGAIN AS DRESSED", kind="mono", size=18, color=hud.ICE)
    ov.line([(60, y + 30), (560, y + 30)], hud.DIM, 1, 120)
    if phase == "lifted":
        ov.text((60, y + 48), f"clear setups   {BASE:,}", kind="mono", size=26, color=hud.INK)
        ov.text((60, y + 90), "move it, and every number downstream follows", kind="mono", size=18, color=hud.DIM)
    else:
        d, n, sent = (NORTH if phase == "north" else SOUTH if phase == "south" else (0.0, BASE, "back where the scan found it"))
        a = hud.ease((t - {"north": 3.4, "south": 7.4, "reset": 11.2}[phase]) / 0.6)
        shown = int(round(BASE + (n - BASE) * a))
        ov.text((60, y + 48), f"moved  {abs(d):.2f} m {'north' if d > 0 else 'south' if d < 0 else ''}".strip(), kind="mono", size=22, color=hud.AMBER)
        ov.text((60, y + 86), f"clear setups   {BASE:,}  →  {shown:,}", kind="mono", size=26, color=hud.INK)
        delta = n - BASE
        if a >= 1.0 and delta:
            ov.text((60, y + 128), f"{'+' if delta > 0 else '−'}{abs(delta):,}", kind="display", size=96, color=hud.GREEN if delta > 0 else hud.AMBER)
            ov.text((60, y + 240), sent, kind="mono", size=17, color=hud.DIM)
        elif a >= 1.0:
            ov.text((60, y + 128), sent, kind="mono", size=18, color=hud.DIM)
    # the mask, inset
    if mask_img is not None:
        iw, ih = 512, 288
        im = cv2.resize(mask_img, (iw, ih), interpolation=cv2.INTER_AREA)
        x0, y0 = hud.W - iw - 60, hud.H - ih - 90
        frame[y0:y0 + ih, x0:x0 + iw] = im
        ov.bracket((x0 - 6, y0 - 6, x0 + iw + 6, y0 + ih + 6), hud.ICE, 20, 2, 230)
        ov.text((x0, y0 + ih + 18), "SAM 3.1  ·  the mask, in the frame that saw it best", kind="mono", size=16, color=hud.ICE)
    ov.text((hud.W - 60, hud.H - 44), "plan · the measured room", kind="mono", size=16, color=hud.DIM, anchor="ra")
    frame = ov.composite(frame)
    if t < 0.5:
        frame = hud.fade(frame, t / 0.5)
    return frame


def main() -> None:
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    only = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
    sweep.load_scene()
    mask = cv2.imread(str(sweep.OUT.parent.parent / "sam3d_5114" / "mask.png"))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) if mask is not None else None
    for i in (only or range(int(SECONDS * hud.FPS))):
        hud.save(render(i, mask), out / f"f_{i:05d}.jpg")
    if not only:
        hud.encode(out, out / "clip.mp4")
        print("->", out / "clip.mp4")


if __name__ == "__main__":
    main()
