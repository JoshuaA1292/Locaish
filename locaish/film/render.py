"""Render what a proposed camera would actually see, from the twin itself.

Not a visualisation of the answer -- the answer. A row in the shot table says
"50 mm from (2.1, 0.4) at 1.55 m reads as a medium close-up"; this module
points a pinhole camera with exactly those numbers at the twin's own points
and produces the frame, so the claim can be checked by looking at it. The
projection is the same thin-lens arithmetic the sweep scored with, which is
the point: if the render looks wrong, the table *is* wrong, and no amount of
prose in between can hide it.

A stand-in for the subject is drawn at the mark -- a simple figure of the
standard 1.75 m stature the sweep framed against -- because a frame of an
empty room does not show what "waist up" means.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..types import Twin
from . import optics

WIDTH_PX = 960
BACKGROUND = (14, 16, 19)
SUBJECT_HEIGHT_M = 1.75


def render_shot(
    twin: Twin,
    cam_pos,
    subject_xy,
    focal_mm: float,
    *,
    sensor_key: str = optics.DEFAULT_SENSOR,
    width_px: int = WIDTH_PX,
    out: str | Path | None = None,
):
    """Project the twin through a pinhole at the given setup. Returns a PIL image.

    The camera is aimed at the subject's eyeline, exactly as the sweep assumed
    when it scored the row being rendered.
    """
    from PIL import Image, ImageDraw

    sensor = optics.SENSORS[sensor_key]
    cam = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    subj = np.asarray(subject_xy, dtype=np.float64).reshape(2)
    floor_z = float(twin.structure.floor_z)
    eye = np.array([subj[0], subj[1], floor_z + 0.94 * SUBJECT_HEIGHT_M])

    # Camera basis: forward at the eyeline, x right, y down (image convention).
    fwd = eye - cam
    n = float(np.linalg.norm(fwd))
    if n < 1e-6:
        raise ValueError("camera and subject coincide")
    fwd = fwd / n
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    rn = float(np.linalg.norm(right))
    right = right / rn if rn > 1e-9 else np.array([1.0, 0.0, 0.0])
    down = np.cross(fwd, right)

    height_px = int(round(width_px * sensor.height_mm / sensor.width_mm))
    fx = focal_mm / sensor.width_mm * width_px
    fy = focal_mm / sensor.height_mm * height_px

    xyz = np.asarray(twin.points.xyz, dtype=np.float64)
    rgb = twin.points.rgb
    if rgb is None:
        rgb = np.full((len(xyz), 3), 170, dtype=np.uint8)

    rel = xyz - cam
    z = rel @ fwd
    infront = z > 0.05
    rel, z = rel[infront], z[infront]
    cols = rgb[infront]

    u = (rel @ right) / z * fx + width_px / 2.0
    v = (rel @ down) / z * fy + height_px / 2.0
    inframe = (u >= 0) & (u < width_px - 1) & (v >= 0) & (v < height_px - 1)
    u, v, z, cols = u[inframe], v[inframe], z[inframe], cols[inframe]

    img = np.full((height_px, width_px, 3), BACKGROUND, dtype=np.uint8)
    if len(z):
        # Painter's algorithm: far points first, near points overwrite them.
        order = np.argsort(-z)
        ui = u[order].astype(np.int32)
        vi = v[order].astype(np.int32)
        zo = z[order]
        cz = cols[order]
        # Splat size scales with closeness, like a real surface would: a point
        # a metre away covers pixels, a point across the room covers one.
        # Without this a sparse cloud reads as fog at every distance equally.
        # The base scale adapts to how densely this twin was sampled, so a
        # sparse fixture and a million-point capture both read as surfaces.
        density_scale = float(np.clip(2.2 * math.sqrt(600_000 / max(len(xyz), 1)), 1.6, 6.0))
        radius = np.clip(np.round(density_scale / zo).astype(np.int32), 1, 5)
        for r in np.unique(radius):
            sel = radius == r
            us, vs, cs = ui[sel], vi[sel], cz[sel]
            for du in range(-int(r) + 1, int(r)):
                for dv in range(-int(r) + 1, int(r)):
                    uu = np.clip(us + du, 0, width_px - 1)
                    vv = np.clip(vs + dv, 0, height_px - 1)
                    img[vv, uu] = cs

    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im, "RGBA")
    _draw_subject(draw, cam, right, down, fwd, fx, fy, width_px, height_px, subj, floor_z)

    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        im.save(out)
    return im


def _project(p, cam, right, down, fwd, fx, fy, w, h):
    rel = np.asarray(p, dtype=np.float64) - cam
    z = float(rel @ fwd)
    if z <= 0.05:
        return None
    return (float(rel @ right) / z * fx + w / 2.0, float(rel @ down) / z * fy + h / 2.0)


def _draw_subject(draw, cam, right, down, fwd, fx, fy, w, h, subj_xy, floor_z):
    """A stand-in figure at the mark: feet, stature line, head, shoulder bar."""
    x, y = float(subj_xy[0]), float(subj_xy[1])
    feet = _project((x, y, floor_z), cam, right, down, fwd, fx, fy, w, h)
    head = _project((x, y, floor_z + SUBJECT_HEIGHT_M), cam, right, down, fwd, fx, fy, w, h)
    if feet is None or head is None:
        return
    shoulder_z = floor_z + 0.82 * SUBJECT_HEIGHT_M
    # Shoulder endpoints perpendicular to the lens axis, so the bar faces camera.
    lat = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    n = float(np.linalg.norm(lat))
    lat = lat / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    half = 0.23
    s1 = _project((x - lat[0] * half, y - lat[1] * half, shoulder_z), cam, right, down, fwd, fx, fy, w, h)
    s2 = _project((x + lat[0] * half, y + lat[1] * half, shoulder_z), cam, right, down, fwd, fx, fy, w, h)

    colour = (110, 168, 254, 230)
    draw.line([feet, head], fill=colour, width=3)
    if s1 and s2:
        draw.line([s1, s2], fill=colour, width=3)
    r = max(3.0, abs(head[1] - feet[1]) * 0.045)
    draw.ellipse([head[0] - r, head[1] - r * 2, head[0] + r, head[1]], outline=colour, width=3)
