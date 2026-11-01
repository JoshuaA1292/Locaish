"""Walk into the location as it will be on the day.

A twin is the room as the phone found it: measured, but lit by whatever
light there was and dressed however the owner left it. The film will be
shot on a date, at an hour, with the set dressed and the cast on their
marks, and no one can stand in *that* room before the day -- unless it is
built. This module builds it.

The construction is a round trip through pictures. Sixty cameras are placed
over the standable floor so that between them they look at every wall and
every opening from at least three places; each is rendered from the twin,
measured geometry only, with the cast as solid figures; a finisher (Muse
Image, injected by the stage; the measured frames themselves, in tests)
paints every render photoreal; and a new gaussian splat is trained on the
finished frames with the poses they were rendered from. The poses are the
one thing the finisher cannot touch -- they are ours, exact, in the twin's
own metres -- so the trained world lands in the twin frame and the viewer
walks it with the identity transform.

Two rules hold the honesty. The world is a render: every finished frame and
the splat trained from them are labelled so, and nothing in them becomes a
measurement -- the derived twin carries the source twin's points, mesh and
structure untouched, and only its view layer is new. And a splat must not
learn a stamp: the frames handed to the trainer carry no caption and no
label, because anything burnt into sixty images becomes a feature of the
room. The page carries the label instead.
"""

from __future__ import annotations

import copy
import io
import json
import math
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..film import optics, render, space
from ..types import Twin
from ..video import splat as splatmod

# One view: a 24 mm on the sweep's default sensor, 960 px wide. Wide enough
# that sixty of them overlap generously, which is what a splat trainer needs
# -- every surface in several frames from several places -- and not so wide
# that the finisher is asked for a perspective it has not seen in a
# photograph.
VIEW_WIDTH_PX = 960
VIEW_FOCAL_MM = 24.0
EYE_HEIGHT_M = 1.5
LOW_HEIGHT_M = 1.0
HIGH_HEIGHT_M = 2.0
# A camera is never nearer a wall or a piece of furniture than this: a lens
# a hand's width from plaster sees a blur the finisher cannot help.
WALL_CLEARANCE_M = 0.35
# Coverage: every wall chunk and every opening centre, seen by this many
# views with a clear ray. Three is the least from which a trainer can
# triangulate a surface instead of painting a billboard.
MIN_VIEWS_PER_TARGET = 3
TARGET_HEIGHT_M = 1.2
WALL_CHUNK_M = 2.0
# Footprint vertices closer than this are raster jitter on one corner.
SHORT_EDGE_M = 0.25
FRUSTUM_MARGIN = 0.06
NEAREST_TARGET_M = 0.5
RAY_SLACK_M = 0.25
# The trainer's initial points: a subsample of what was measured.
POINTS3D_N = 50_000
# The label every artefact of the world carries.
RENDER_NOTE = "a render, not a measurement: the room as it will be, built from the measured twin"

ProgressFn = Callable[[str, int, int], None]


# ---------------------------------------------------------------------------
# a view
# ---------------------------------------------------------------------------


def frame_size(width_px: int = VIEW_WIDTH_PX, *, sensor_key: str = optics.DEFAULT_SENSOR) -> tuple[int, int]:
    """The exact pixel size `render.camera_at` gives a frame of this width."""
    sensor = optics.SENSORS[sensor_key]
    return int(width_px), int(round(width_px * sensor.height_mm / sensor.width_mm))


@dataclass
class View:
    """One camera the world is built from: where it stands, what it looks at,
    what lens, and the exact frame it produces. `role` says why it exists
    (`centre`, `left`, `right`, `corner`, `low`, `high`, `fix`) so the
    selection can be trimmed from the least load-bearing end."""

    name: str
    cam: tuple[float, float, float]
    aim: tuple[float, float, float]
    focal_mm: float = VIEW_FOCAL_MM
    width_px: int = VIEW_WIDTH_PX
    height_px: int = 0
    role: str = ""

    def __post_init__(self) -> None:
        self.cam = tuple(float(v) for v in np.asarray(self.cam, dtype=np.float64).reshape(3))
        self.aim = tuple(float(v) for v in np.asarray(self.aim, dtype=np.float64).reshape(3))
        if not self.height_px:
            self.height_px = frame_size(self.width_px)[1]

    def camera(self, *, sensor_key: str = optics.DEFAULT_SENSOR) -> render.Camera:
        return render.camera_at(self.cam, self.aim, self.focal_mm, sensor_key=sensor_key, width_px=self.width_px)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cam": [round(v, 4) for v in self.cam],
            "aim": [round(v, 4) for v in self.aim],
            "focal_mm": float(self.focal_mm),
            "width_px": int(self.width_px),
            "height_px": int(self.height_px),
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "View":
        return cls(
            name=str(d["name"]),
            cam=tuple(d["cam"]),
            aim=tuple(d["aim"]),
            focal_mm=float(d.get("focal_mm", VIEW_FOCAL_MM)),
            width_px=int(d.get("width_px", VIEW_WIDTH_PX)),
            height_px=int(d.get("height_px", 0)),
            role=str(d.get("role", "")),
        )


# ---------------------------------------------------------------------------
# plan geometry
# ---------------------------------------------------------------------------


def _simplify(poly: np.ndarray, tol: float = SHORT_EDGE_M) -> np.ndarray:
    """Merge footprint vertices closer than `tol`: a raster-traced corner
    comes out as two vertices a centimetre apart, and that centimetre is
    not a wall."""
    pts = [np.asarray(p, dtype=np.float64) for p in poly]
    changed = True
    while changed and len(pts) > 3:
        changed = False
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            if np.linalg.norm(pts[j] - pts[i]) < tol:
                pts[i] = 0.5 * (pts[i] + pts[j])
                del pts[j]
                changed = True
                break
    return np.asarray(pts).reshape(-1, 2)


def _ccw(poly: np.ndarray) -> np.ndarray:
    q = np.roll(poly, -1, axis=0)
    area2 = float(np.sum(poly[:, 0] * q[:, 1] - q[:, 0] * poly[:, 1]))
    return poly if area2 >= 0 else poly[::-1].copy()


def _ray_to_boundary(p: np.ndarray, d: np.ndarray, poly: np.ndarray | None) -> float:
    """Distance from `p` along unit `d` to the footprint boundary; inf if none."""
    if poly is None or len(poly) < 3:
        return float("inf")
    best = float("inf")
    for i in range(len(poly)):
        a = poly[i]
        e = poly[(i + 1) % len(poly)] - a
        den = d[0] * (-e[1]) - d[1] * (-e[0])
        if abs(den) < 1e-12:
            continue
        r = a - p
        t = (r[0] * (-e[1]) - r[1] * (-e[0])) / den
        s = (d[0] * r[1] - d[1] * r[0]) / den
        if t > 1e-9 and -1e-9 <= s <= 1 + 1e-9:
            best = min(best, float(t))
    return best


def _inside(poly: np.ndarray | None, xy) -> bool:
    if poly is None or len(poly) < 3:
        return True
    xy = np.asarray(xy, dtype=np.float64).reshape(2)
    return bool(space._points_in_polygon(np.array([xy[0]]), np.array([xy[1]]), poly)[0])


def _farthest_point_order(pts: np.ndarray, start: int, count: int) -> list[int]:
    """Indices of `pts` in farthest-point order from `start`: each pick is
    the point farthest from every point picked so far, which spreads
    cameras over a room the way a scout would walk it."""
    n = len(pts)
    count = min(count, n)
    if n == 0:
        return []
    chosen = [int(start)]
    dist = np.linalg.norm(pts - pts[start], axis=1)
    while len(chosen) < count:
        k = int(np.argmax(dist))
        if dist[k] <= 0 and len(chosen) > 0 and n > len(chosen):
            # Only duplicates remain.
            rest = [i for i in range(n) if i not in chosen]
            if not rest:
                break
            k = rest[0]
        chosen.append(k)
        dist = np.minimum(dist, np.linalg.norm(pts - pts[k], axis=1))
    return chosen


@dataclass
class _Target:
    name: str
    kind: str
    point: np.ndarray     # (3,)
    inward: np.ndarray    # (2,) unit, into the room


def _targets(twin: Twin, poly: np.ndarray | None) -> list[_Target]:
    floor_z = float(twin.structure.floor_z)
    ceiling = twin.structure.drawable_ceiling_z
    tz = floor_z + TARGET_HEIGHT_M
    if ceiling is not None:
        tz = min(tz, float(ceiling) - 0.3)
    out: list[_Target] = []
    if poly is not None and len(poly) >= 3:
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            e = b - a
            L = float(np.linalg.norm(e))
            if L < SHORT_EDGE_M:
                continue
            u = e / L
            inward = np.array([-u[1], u[0]])      # left normal of a CCW edge points in
            chunks = max(1, int(math.ceil(L / WALL_CHUNK_M)))
            for j in range(chunks):
                mid = a + u * L * (j + 0.5) / chunks
                out.append(_Target(f"wall_{i:02d}_{j}", "wall", np.array([mid[0], mid[1], tz]), inward))
    for k, o in enumerate(twin.structure.openings):
        nrm = np.asarray(o.normal[:2], dtype=np.float64)
        n = float(np.linalg.norm(nrm))
        nrm = nrm / n if n > 1e-9 else np.array([0.0, 0.0])
        out.append(_Target(f"{o.kind}_{k}", o.kind, np.asarray(o.center, dtype=np.float64).copy(), nrm))
    return out


def _sees(cam: render.Camera, grid, origin, cell, target: np.ndarray) -> bool:
    """Whether a camera sees a target: inside the frustum, not on top of it,
    and a clear ray through the measured occupancy."""
    uv = cam.project(target)
    if uv is None:
        return False
    W, H = cam.width_px, cam.height_px
    m = FRUSTUM_MARGIN
    if not (m * W <= uv[0] <= (1 - m) * W and m * H <= uv[1] <= (1 - m) * H):
        return False
    if float(np.linalg.norm(target - cam.pos)) < NEAREST_TARGET_M:
        return False
    return space.visible(grid, origin, cell, cam.pos, target, slack=RAY_SLACK_M)


def _seen_matrix(views: list[View], targets: list[_Target], grid, origin, cell) -> np.ndarray:
    seen = np.zeros((len(views), len(targets)), dtype=bool)
    for i, v in enumerate(views):
        cam = v.camera()
        for j, t in enumerate(targets):
            seen[i, j] = _sees(cam, grid, origin, cell, t.point)
    return seen


def coverage_report(targets: list[_Target], seen: np.ndarray, notes: list[str]) -> dict[str, Any]:
    counts = seen.sum(axis=0) if len(seen) else np.zeros(len(targets), dtype=int)
    rows = [
        {"name": t.name, "kind": t.kind, "point": [round(float(v), 3) for v in t.point], "views": int(c)}
        for t, c in zip(targets, counts)
    ]
    uncovered = [r["name"] for r in rows if r["views"] < MIN_VIEWS_PER_TARGET]
    return {
        "targets": rows,
        "walls": sum(1 for t in targets if t.kind == "wall"),
        "openings": sum(1 for t in targets if t.kind != "wall"),
        "min_views": MIN_VIEWS_PER_TARGET,
        "least_seen": int(counts.min()) if len(counts) else 0,
        "uncovered": uncovered,
        "complete": not uncovered,
        "notes": list(notes),
    }


# ---------------------------------------------------------------------------
# selecting the views
# ---------------------------------------------------------------------------


def select_views(
    twin: Twin,
    *,
    maps: space.FloorMaps | None = None,
    n: int = 60,
    height_m: float = EYE_HEIGHT_M,
    focal_mm: float = VIEW_FOCAL_MM,
    seed: int = 0,
    width_px: int = VIEW_WIDTH_PX,
) -> tuple[list[View], dict[str, Any]]:
    """Place `n` cameras over the room and prove they cover it.

    Positions come from farthest-point sampling over the cells a person can
    stand on, kept `WALL_CLEARANCE_M` from anything solid. Each position
    looks three ways -- at the room's centre and tangentially left and right
    -- so between them the sixty frames look around the room rather than
    all at its middle. A few more stand in the corners looking in, and a few
    sit low (1.0 m) and high (2.0 m) so the floor and the ceiling are seen
    from more than one pitch. Then the coverage check: every wall chunk of
    the footprint and every opening centre must sit inside the frustum of at
    least `MIN_VIEWS_PER_TARGET` views with a clear occupancy ray, and views
    aimed at whatever is short are added until it does. The result is
    exactly `n` views when the room allows it, fewer with a note when it
    does not, and the coverage dict that says what was and was not seen.
    """
    rng = np.random.default_rng(seed)
    maps = maps or space.floor_maps(twin)
    floor_z = float(twin.structure.floor_z)
    notes: list[str] = []

    stand = maps.standable() & (maps.clearance_m >= WALL_CLEARANCE_M)
    if not stand.any():
        stand = maps.inside & (maps.clearance_m >= 0.2) & (maps.headroom_m >= 1.6)
        notes.append("no cell clears 0.35 m from the walls at standing headroom; relaxed to 0.2 m")
    if not stand.any():
        raise ValueError("nowhere in this room to stand a camera")
    cells = maps.world_of(np.argwhere(stand))                      # (M, 2)

    poly = twin.structure.footprint
    if poly is not None and len(poly) >= 3:
        poly = _ccw(_simplify(np.asarray(poly, dtype=np.float64)))
    else:
        poly = None
        notes.append("the twin has no footprint; walls are not coverage targets")
    centroid = poly.mean(axis=0) if poly is not None else cells.mean(axis=0)
    centre = cells[int(np.argmin(np.linalg.norm(cells - centroid, axis=1)))]

    def z_at(xy, h):
        head = float(maps.sample(maps.headroom_m, xy)[0])
        return floor_z + min(float(h), max(head - 0.15, 0.8))

    def aim_xy(p, d):
        reach = _ray_to_boundary(p, d, poly)
        reach = min(3.0, 0.8 * reach) if math.isfinite(reach) else 2.5
        return p + d * max(reach, 0.3)

    def look(p, d, h, role):
        a = aim_xy(p, d)
        az = floor_z + (1.3 if h <= 1.6 else 1.2)
        return View("", (p[0], p[1], z_at(p, h)), (a[0], a[1], az), focal_mm, width_px, role=role)

    def towards_centre(p):
        d = centre - p
        L = float(np.linalg.norm(d))
        if L < 0.3:
            ang = rng.uniform(0, 2 * math.pi)
            return np.array([math.cos(ang), math.sin(ang)])
        return d / L

    n_corner = min(4, n // 12) if poly is not None else 0
    n_height = min(6, 2 * (n // 20))
    reserve = max(3, n // 10)
    k_pos = max(1, int(math.ceil((n - n_corner - n_height - reserve) / 3.0)))
    order = _farthest_point_order(cells, int(np.argmin(np.linalg.norm(cells - centre, axis=1))), len(cells))

    views: list[View] = []
    used = 0
    for idx in order[:k_pos]:
        p = cells[idx]
        d0 = towards_centre(p)
        views.append(look(p, d0, height_m, "centre"))
        views.append(look(p, np.array([-d0[1], d0[0]]), height_m, "left"))
        views.append(look(p, np.array([d0[1], -d0[0]]), height_m, "right"))
        used += 1

    # Corners: convex footprint vertices, the four farthest apart, each
    # looked in from the nearest standable cell.
    if n_corner and poly is not None:
        q = np.roll(poly, -1, axis=0)
        r = np.roll(poly, 1, axis=0)
        cross = (poly - r)[:, 0] * (q - poly)[:, 1] - (poly - r)[:, 1] * (q - poly)[:, 0]
        convex = poly[cross > 0]
        if len(convex):
            picks = _farthest_point_order(convex, 0, n_corner)
            for i in picks:
                c = convex[i]
                dist = np.linalg.norm(cells - c, axis=1)
                j = int(np.argmin(dist))
                if dist[j] > 1.6:
                    continue
                p = cells[j]
                d = towards_centre(p)
                a = aim_xy(p, d)
                views.append(View("", (p[0], p[1], z_at(p, height_m)), (a[0], a[1], floor_z + 1.3),
                                  focal_mm, width_px, role="corner"))

    # Low and high: the most spread positions, pitched at the centre.
    if n_height:
        for i, idx in enumerate(order[: n_height]):
            p = cells[idx]
            h = LOW_HEIGHT_M if i % 2 == 0 else HIGH_HEIGHT_M
            d = towards_centre(p)
            a = aim_xy(p, d)
            views.append(View("", (p[0], p[1], z_at(p, h)), (a[0], a[1], floor_z + 1.25),
                              focal_mm, width_px, role="low" if h < height_m else "high"))

    # Coverage.
    grid, origin, cell = space.occupancy(twin)
    targets = _targets(twin, poly)
    seen = _seen_matrix(views, targets, grid, origin, cell)
    # Fixes, as a greedy set cover: candidate views aimed at each short
    # target from a few comfortable standing places, and whichever candidate
    # retires the most shortfall is added first. One well-placed view often
    # covers two wall chunks and a window at once, which keeps the fixes
    # within the budget a small n leaves for them.
    def shortfall(seen_):
        counts = seen_.sum(axis=0) if len(seen_) else np.zeros(len(targets), dtype=int)
        return np.maximum(MIN_VIEWS_PER_TARGET - counts, 0)

    cam_xy = np.array([v.cam[:2] for v in views]).reshape(-1, 2)
    pool: list[tuple[View, np.ndarray]] = []
    for j, t in enumerate(targets):
        if shortfall(seen)[j] <= 0:
            continue
        rel = cells - t.point[:2]
        dist = np.linalg.norm(rel, axis=1)
        front = rel @ t.inward
        ok = (front > 0.5) & (dist >= 1.0) & (dist <= 4.5)
        if not ok.any():
            ok = (front > 0.3) & (dist >= 0.6)
        cand = np.flatnonzero(ok)
        if not len(cand):
            notes.append(f"{t.name}: no standable cell faces it")
            continue
        crowd = (np.linalg.norm(cells[cand][:, None, :] - cam_xy[None], axis=2).min(axis=1)
                 if len(cam_xy) else np.full(len(cand), 9.0))
        score = -np.abs(dist[cand] - 2.5) + np.minimum(crowd, 0.6) + rng.random(len(cand)) * 0.05
        made = 0
        for ci in cand[np.argsort(-score)]:
            if made >= 6:
                break
            p = cells[ci]
            a = t.point[:2] + t.inward * 0.1
            v = View("", (p[0], p[1], z_at(p, height_m)), (a[0], a[1], float(t.point[2])),
                     focal_mm, width_px, role="fix")
            row = np.array([_sees(v.camera(), grid, origin, cell, tt.point) for tt in targets])
            if row[j]:
                pool.append((v, row))
                made += 1
    while pool and shortfall(seen).sum() > 0:
        need = shortfall(seen) > 0
        gains = [int((row & need).sum()) for _, row in pool]
        k = int(np.argmax(gains))
        if gains[k] <= 0:
            break
        v, row = pool.pop(k)
        views.append(v)
        seen = np.vstack([seen, row[None]]) if len(seen) else row[None]
    for j, s in enumerate(shortfall(seen)):
        if s > 0:
            notes.append(f"{targets[j].name}: only {MIN_VIEWS_PER_TARGET - int(s)} views reach it with a clear ray")

    # Fill to n with further look-around positions; trim past n from the
    # least load-bearing views whose removal keeps every target covered.
    while len(views) < n and used < len(order):
        p = cells[order[used]]
        used += 1
        d0 = towards_centre(p)
        for d, role in ((d0, "centre"), (np.array([-d0[1], d0[0]]), "left"), (np.array([d0[1], -d0[0]]), "right")):
            if len(views) >= n:
                break
            v = look(p, d, height_m, role)
            views.append(v)
            row = np.array([_sees(v.camera(), grid, origin, cell, tt.point) for tt in targets])
            seen = np.vstack([seen, row[None]]) if len(seen) else row[None]
    if len(views) < n:
        notes.append(f"{len(views)} views: the room has only {len(cells)} standable cells")

    removable = {"right": 0, "left": 1, "centre": 2, "low": 3, "high": 3, "corner": 4, "fix": 9}
    forced = 0
    while len(views) > n:
        counts = seen.sum(axis=0) if len(targets) else np.zeros(0)
        cand = sorted(range(len(views)), key=lambda i: (removable.get(views[i].role, 5), -i))
        dropped = False
        for i in cand:
            if removable.get(views[i].role, 5) >= 9:
                continue
            if len(targets) and np.any(counts - seen[i] < np.minimum(counts, MIN_VIEWS_PER_TARGET)):
                continue
            del views[i]
            seen = np.delete(seen, i, axis=0)
            dropped = True
            break
        if not dropped:
            i = cand[0]
            forced += 1
            del views[i]
            seen = np.delete(seen, i, axis=0)
    if forced:
        notes.append(f"dropped {forced} covering views to meet n={n}; coverage thinned")

    for i, v in enumerate(views):
        v.name = f"view_{i:04d}"
    cov = coverage_report(targets, seen, notes)
    cov["requested"] = int(n)
    cov["returned"] = len(views)
    cov["roles"] = {r: sum(1 for v in views if v.role == r) for r in sorted({v.role for v in views})}
    return views, cov


# ---------------------------------------------------------------------------
# rendering the measured frames
# ---------------------------------------------------------------------------


def render_view(twin: Twin, view: View, figures: list[render.Figure] | None = None, *,
                exclude: np.ndarray | None = None):
    """The twin through one view, as a PIL image of exactly the view's size.

    The gaussian field when the twin has one, the measured points otherwise;
    splat figures merged into the field, mesh figures rasterised over it
    with a z-test against the measured room. No caption, no stamp: this
    frame is what the finisher is shown and what the trainer learns from.
    """
    from PIL import Image

    cam = view.camera()
    figs = list(figures or [])
    solids = [f for f in figs if f.mesh is not None]
    splats = [f for f in figs if f.mesh is None and len(f.xyz)]
    fld = render.gaussian_field(twin)
    if fld is not None:
        if splats:
            fld = render.merge_fields([fld] + [f.as_gaussians() for f in splats])
        img = render._render_gaussians(fld, cam)
    else:
        extra = [(f.xyz, (f.rgb * 255.0 + 0.5).astype(np.uint8)) for f in splats]
        img = render._render_points(twin, cam, extra=extra, exclude=exclude)
    if solids:
        img = render.draw_figures(img, twin, cam, solids, exclude=exclude)
    return Image.fromarray(np.asarray(img))


def _figures_of(figures_for, view: View) -> list[render.Figure]:
    if figures_for is None:
        return []
    if callable(figures_for):
        return list(figures_for(view) or [])
    return list(figures_for)


def _actors_of(figures: list[render.Figure], actors: dict | None) -> dict[str, list[float]]:
    """The cast for the finisher's meta: a name, a mark, a facing. Given
    explicitly when the stage knows them; read off the labelled figures'
    footprints otherwise, with the facing unknown (0)."""
    if actors:
        return {str(k): [float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else 0.0] for k, v in actors.items()}
    out: dict[str, list[float]] = {}
    for f in figures:
        if f.label and len(f.xyz):
            c = f.xyz[:, :2].mean(axis=0)
            out[f.label] = [round(float(c[0]), 3), round(float(c[1]), 3), 0.0]
    return out


def view_anchor(view: View, file: str) -> dict[str, Any]:
    """A finished view as the viewer pins it in the room: where the camera
    stood, what it looked at, its field in degrees both ways, the frame's
    size, and the file next to the page that holds the picture."""
    cam = view.camera()
    return {
        "name": view.name,
        "file": str(file),
        "cam": [round(float(v), 4) for v in view.cam],
        "aim": [round(float(v), 4) for v in view.aim],
        "focal_mm": float(view.focal_mm),
        "fov_h_deg": round(math.degrees(2.0 * math.atan(cam.width_px / (2.0 * cam.fx))), 3),
        "fov_v_deg": round(math.degrees(2.0 * math.atan(cam.height_px / (2.0 * cam.fy))), 3),
        "width_px": int(cam.width_px),
        "height_px": int(cam.height_px),
    }


def view_meta(view: View, *, actors: dict | None = None, light: dict | None = None,
              dressing: str = "") -> dict[str, Any]:
    return {
        "name": view.name,
        "camera": {"pos": list(view.cam), "aim": list(view.aim), "focal_mm": float(view.focal_mm)},
        "actors": dict(actors or {}),
        "light": light,
        "dressing": str(dressing or ""),
        "neighbours": [],
        "note": RENDER_NOTE,
    }


def render_views(
    twin: Twin,
    views: list[View],
    out_dir: str | Path,
    *,
    figures_for=None,
    exclude: np.ndarray | None = None,
    progress: ProgressFn | None = None,
    light: dict | None = None,
    dressing: str = "",
    actors: dict | None = None,
) -> list[tuple[str, Path, dict[str, Any]]]:
    """Render every view to `out_dir/view_NNNN.png` and write `views.json`.

    `figures_for` is a list of figures drawn in every frame, or a callable
    of the view returning the figures for that frame. Returns one
    (name, path, meta) per view, the meta in the finisher's shape.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[tuple[str, Path, dict[str, Any]]] = []
    for i, v in enumerate(views):
        if progress:
            progress(f"rendering {v.name}", i, len(views))
        figs = _figures_of(figures_for, v)
        im = render_view(twin, v, figs, exclude=exclude)
        path = out_dir / f"{v.name}.png"
        im.save(path)
        meta = view_meta(v, actors=_actors_of(figs, actors), light=light, dressing=dressing)
        # How much of the frame the twin could paint. A view into a part of
        # the room the scan never saw comes out as background, and the
        # finisher should know it is being asked to invent rather than finish.
        arr = np.asarray(im).astype(np.int32)
        bg = np.all(np.abs(arr - np.asarray(render.BACKGROUND, dtype=np.int32)) < 3, axis=-1)
        meta["painted"] = round(float(1.0 - bg.mean()), 3)
        items.append((v.name, path, meta))
    (out_dir / "views.json").write_text(json.dumps({
        "note": RENDER_NOTE,
        "views": [v.to_dict() for v in views],
        "items": [{"name": n, "path": p.name, "meta": m} for n, p, m in items],
    }, indent=1))
    if progress:
        progress("rendered", len(views), len(views))
    return items


# ---------------------------------------------------------------------------
# the COLMAP model the trainer reads
# ---------------------------------------------------------------------------


def _quat_from_rot(R: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of a rotation matrix, by the largest-diagonal branch."""
    R = np.asarray(R, dtype=np.float64)
    t = float(np.trace(R))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def world_to_camera(view: View) -> tuple[np.ndarray, np.ndarray]:
    """COLMAP's pose for a view: (R, t) with X_cam = R X_world + t, camera
    axes x right, y down, z forward. `render.camera_at` gives the camera's
    axes in world coordinates; those are the columns of camera-to-world, so
    world-to-camera is their transpose and t = -R p."""
    cam = view.camera()
    R = np.stack([cam.right, cam.down, cam.fwd])      # rows = camera axes in world = R_c2w^T
    t = -R @ cam.pos
    return R, t


def write_colmap_model(twin: Twin, views: list[View], image_size: tuple[int, int], out_dir: str | Path) -> Path:
    """Write cameras.txt, images.txt and points3D.txt for the views.

    One PINHOLE camera per distinct intrinsics (there is usually one), the
    pose of every view as a world-to-camera quaternion and translation in
    the twin's metres, and a 50k subsample of the measured points with their
    colours for the trainer to start from. Text format: a dozen lines a
    person can read is worth more here than a binary nobody can check.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H = int(image_size[0]), int(image_size[1])
    sensor = optics.SENSORS[optics.DEFAULT_SENSOR]

    cameras: dict[tuple, int] = {}
    cam_lines: list[str] = []
    img_lines: list[str] = []
    for i, v in enumerate(views):
        w, h = int(v.width_px), int(v.height_px)
        if (w, h) != (W, H):
            raise ValueError(f"{v.name} is {w}x{h}; the model is {W}x{H}")
        fx = float(v.focal_mm) / sensor.width_mm * w
        fy = float(v.focal_mm) / sensor.height_mm * h
        key = (w, h, round(fx, 6), round(fy, 6), w / 2.0, h / 2.0)
        if key not in cameras:
            cameras[key] = len(cameras) + 1
            cam_lines.append(f"{cameras[key]} PINHOLE {w} {h} {fx:.6f} {fy:.6f} {w / 2.0:.1f} {h / 2.0:.1f}")
        R, t = world_to_camera(v)
        q = _quat_from_rot(R)
        img_lines.append(
            f"{i + 1} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
            f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {cameras[key]} {v.name}.png"
        )
        img_lines.append("")

    (out_dir / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"# Number of cameras: {len(cameras)}\n" + "\n".join(cam_lines) + "\n"
    )
    (out_dir / "images.txt").write_text(
        "# Image list with two lines of data per image:\n"
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        f"# Number of images: {len(views)}, mean observations per image: 0\n" + "\n".join(img_lines) + "\n"
    )

    pts = twin.points.measured()
    xyz = np.asarray(pts.xyz, dtype=np.float64)
    rgb = pts.rgb
    if rgb is None:
        rgb = np.full((len(xyz), 3), 170, dtype=np.uint8)
    if len(xyz) > POINTS3D_N:
        pick = np.random.default_rng(0).choice(len(xyz), POINTS3D_N, replace=False)
        xyz, rgb = xyz[pick], rgb[pick]
    rows = [
        f"{k + 1} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[0])} {int(c[1])} {int(c[2])} 0.0"
        for k, (p, c) in enumerate(zip(xyz, np.asarray(rgb, dtype=np.int64)))
    ]
    (out_dir / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        f"# Number of points: {len(rows)}, mean track length: 0\n" + "\n".join(rows) + "\n"
    )
    (out_dir / "README.txt").write_text(
        "COLMAP text model for the world trainer, in the twin's frame (metres).\n"
        "Poses are exact: each image was rendered through this camera, not solved from it.\n"
        f"{len(views)} images, {len(cameras)} camera(s), {len(rows)} seed points from the measured cloud.\n"
        "The images are finished renders, not photographs; the splat trained on them is a render.\n"
    )
    return out_dir


# ---------------------------------------------------------------------------
# training, and the twin that carries the result
# ---------------------------------------------------------------------------


def train_world(
    images_dir: str | Path,
    model_dir: str | Path,
    out_dir: str | Path,
    *,
    steps: int = 6000,
    resolution: int = VIEW_WIDTH_PX,
    progress: ProgressFn | None = None,
) -> Path:
    """Train the world's splat with Brush on the finished frames at their
    exact poses; returns the PLY. Raises when Brush is not installed, since
    a world without a splat is not a world."""
    ply = splatmod.train_splat(
        images_dir, model_dir, out_dir, steps=int(steps), resolution=int(resolution),
        progress=(lambda msg: progress(msg, 0, 0)) if progress else None,
    )
    if ply is None:
        raise RuntimeError("Brush is not installed (see video/splat.brush_binary); the world cannot be trained")
    return Path(ply)


def bake_world(
    twin: Twin,
    views: list[View],
    finished: list[str | Path],
    out_ply: str | Path,
    *,
    exclude: np.ndarray | None = None,
    progress: ProgressFn | None = None,
    keep_unseen: bool = True,
    relight: bool = True,
) -> Path:
    """Paint the finished views onto the scan's own gaussians.

    Training a fresh splat on sixty generated pictures threw away the one
    thing the scan had, crisp geometry, and learned the views' disagreements
    as fog. So the world keeps every gaussian the scan trained, exactly
    where it is, and only its colour changes: for each finished view, every
    gaussian that projects into the frame and sits on the measured surface
    (its depth within a tolerance of the measured depth at that pixel) takes
    the finished colour at that pixel; the colour written is the mean over
    the views that saw it. Gaussians no view saw keep the scan's colour
    (`keep_unseen`). Deterministic, no training, minutes not hours, and the
    walls stay walls. What it cannot do is move geometry: a bed the finisher
    made keeps its scanned shape under new colours.

    With `relight` (the default) a view is not copied blindly: the finisher
    was shown a render of the scan, and its answer is compared with that
    render. Where the two agree on the structure (the same edges in the same
    places) the finished pixel is the room's own detail, relit and dressed,
    and is painted. Where they disagree the gaussian keeps the scan's colour
    and takes only the light: the ratio of the finished view to the render
    at low frequency, which is where the sun patches and the hour's colour
    live. Views are weighted by that agreement and by nearness, so one
    close, faithful view outvotes three far, invented ones, and the walls
    stop averaging into fog.

    The ply is written in the twin frame (identity transform, factor 1), so
    `world_twin` can hand it straight to the viewer and the renderer.
    """
    from PIL import Image

    from ..video.splat import read_gaussian_ply, write_gaussian_ply

    src = (twin.provenance or {}).get("splat_ply")
    if not src or not Path(src).exists():
        raise RuntimeError("this twin has no gaussian field to bake onto")
    steps = (twin.provenance or {}).get("steps", {})
    factor = float(((steps.get("video") or {}).get("scale") or {}).get("factor") or 1.0)
    g = read_gaussian_ply(src)
    M = np.asarray(twin.canonical_transform, dtype=np.float64)
    R, t = M[:3, :3], M[:3, 3]
    xyz = (g["xyz"] * factor) @ R.T + t
    scale = g["scale"] * factor
    quat = _compose_quat(R, g["quat"])
    n = len(xyz)
    acc = np.zeros((n, 3), dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)
    own = np.clip(np.asarray(g["rgb01"], dtype=np.float64), 0.0, 1.0)
    tick = progress or (lambda *_: None)
    for i, (view, fin) in enumerate(zip(views, finished)):
        tick(f"baking {view.name}", i, len(views))
        cam = view.camera()
        try:
            im = np.asarray(Image.open(fin).convert("RGB").resize((cam.width_px, cam.height_px), Image.BILINEAR),
                            dtype=np.float64) / 255.0
        except Exception:  # noqa: BLE001 - a missing finished view bakes nothing
            continue
        basis = np.stack([cam.right, cam.down, cam.fwd])
        pc = (xyz - cam.pos) @ basis.T
        z = pc[:, 2]
        front = z > 0.05
        u = np.full(n, -1.0)
        v = np.full(n, -1.0)
        u[front] = pc[front, 0] / z[front] * cam.fx + cam.width_px / 2.0
        v[front] = pc[front, 1] / z[front] * cam.fy + cam.height_px / 2.0
        inside = front & (u >= 0) & (u < cam.width_px - 1) & (v >= 0) & (v < cam.height_px - 1)
        if not inside.any():
            continue
        depth = render.twin_depth_map(twin, cam, exclude=exclude)
        ui = np.clip(np.round(u[inside]).astype(np.int64), 0, cam.width_px - 1)
        vi = np.clip(np.round(v[inside]).astype(np.int64), 0, cam.height_px - 1)
        d = depth[vi, ui]
        zz = z[inside]
        visible = ~np.isfinite(d) | (zz <= d + 0.06 + 0.04 * zz)
        idx = np.flatnonzero(inside)[visible]
        px, py, pz = ui[visible], vi[visible], zz[visible]
        if relight:
            # What the finisher was shown, and how its answer relates to it.
            shown = np.asarray(render_view(twin, view, exclude=exclude), dtype=np.float64) / 255.0
            light, agree = _light_and_agreement(shown, im)
            # Where model and measurement agree on the structure, the model's
            # pixel is the room's own detail, relit and dressed: paint it.
            # Where they disagree (a bed the model moved, a wall it invented)
            # the scan keeps its own colour and takes only the light.
            a = agree[py, px][:, None]
            relit = np.clip(own[idx] * light[py, px], 0.0, 1.0)
            sample = a * im[py, px] + (1.0 - a) * relit
            w = (0.25 + agree[py, px]) / (1.0 + pz)
        else:
            sample = im[py, px]
            w = 1.0 / (1.0 + pz)
        acc[idx] += sample * w[:, None]
        wsum[idx] += w
    seen = wsum > 0
    rgb = g["rgb01"].copy()
    rgb[seen] = acc[seen] / wsum[seen, None]
    if not keep_unseen:
        rgb[~seen] = 0.0
    out = write_gaussian_ply(out_ply, {"xyz": xyz, "rgb01": rgb, "opacity": g["opacity"], "scale": scale, "quat": quat})
    tick(f"baked {int(seen.sum()):,} of {n:,} gaussians", len(views), len(views))
    return out


def _light_and_agreement(shown: np.ndarray, fin: np.ndarray, *, radius: int | None = None):
    """Two maps from a render and the finished view of the same camera.

    `light` (H, W, 3): the finished view over the render, both low-passed,
    clamped to [0, 4]: a per-pixel, per-channel multiplier that carries the
    hour's light and colour and none of the detail.
    `agree` (H, W) in [0, 1]: the local correlation of the two luminances in
    a window, so an edge the model kept where it was scores one and a wall
    it repainted into something else scores zero. Flat regions (no edge in
    either) score zero: there is nothing there to agree on, so the light
    carries them.
    """
    from scipy import ndimage

    H, W = shown.shape[:2]
    r = int(radius or max(8, W // 24))
    lo_s = ndimage.uniform_filter(shown, size=(r, r, 1), mode="nearest")
    lo_f = ndimage.uniform_filter(fin, size=(r, r, 1), mode="nearest")
    light = np.clip(lo_f / np.maximum(lo_s, 0.04), 0.0, 4.0)
    ls = shown @ np.array([0.299, 0.587, 0.114])
    lf = fin @ np.array([0.299, 0.587, 0.114])
    k = max(5, r // 2)
    m_s = ndimage.uniform_filter(ls, k, mode="nearest")
    m_f = ndimage.uniform_filter(lf, k, mode="nearest")
    v_s = ndimage.uniform_filter(ls * ls, k, mode="nearest") - m_s * m_s
    v_f = ndimage.uniform_filter(lf * lf, k, mode="nearest") - m_f * m_f
    cov = ndimage.uniform_filter(ls * lf, k, mode="nearest") - m_s * m_f
    eps = 1e-4
    ncc = cov / np.sqrt(np.maximum(v_s, eps) * np.maximum(v_f, eps))
    # Flat in either picture: no structure to agree on.
    flat = (v_s < eps) | (v_f < eps)
    agree = np.clip(ncc, 0.0, 1.0)
    agree[flat] = 0.0
    return light, agree


def _compose_quat(R: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """wxyz quaternions rotated by the twin's canonical rotation R."""
    from ..video.splat import _quat_to_rot

    Rq = _quat_to_rot(np.asarray(quat, dtype=np.float64))
    Rn = R[None] @ Rq
    m00, m11, m22 = Rn[:, 0, 0], Rn[:, 1, 1], Rn[:, 2, 2]
    tr = m00 + m11 + m22
    q = np.zeros((len(Rn), 4))
    a = tr > 0
    s = np.sqrt(np.maximum(tr[a] + 1.0, 1e-12)) * 2
    q[a, 0] = 0.25 * s
    q[a, 1] = (Rn[a, 2, 1] - Rn[a, 1, 2]) / s
    q[a, 2] = (Rn[a, 0, 2] - Rn[a, 2, 0]) / s
    q[a, 3] = (Rn[a, 1, 0] - Rn[a, 0, 1]) / s
    b = ~a
    if b.any():
        Rb = Rn[b]
        i = np.argmax(np.stack([Rb[:, 0, 0], Rb[:, 1, 1], Rb[:, 2, 2]], axis=1), axis=1)
        qb = np.zeros((len(Rb), 4))
        for k in range(3):
            sel = i == k
            if not sel.any():
                continue
            Rk = Rb[sel]
            j, l = (k + 1) % 3, (k + 2) % 3
            sk = np.sqrt(np.maximum(1.0 + Rk[:, k, k] - Rk[:, j, j] - Rk[:, l, l], 1e-12)) * 2
            qk = np.zeros((len(Rk), 4))
            qk[:, 0] = (Rk[:, l, j] - Rk[:, j, l]) / sk
            qk[:, 1 + k] = 0.25 * sk
            qk[:, 1 + j] = (Rk[:, j, k] + Rk[:, k, j]) / sk
            qk[:, 1 + l] = (Rk[:, l, k] + Rk[:, k, l]) / sk
            qb[sel] = qk
        q[b] = qb
    return q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)


def world_twin(twin: Twin, ply_path: str | Path, *, world_meta: dict[str, Any]) -> Twin:
    """The twin that carries the world: the source twin's points, mesh and
    structure as they are, with the new splat as its view layer in the twin
    frame -- identity transform, scale factor 1.0 -- and `provenance.world`
    saying which day and hour it shows. Nothing measured has changed, and
    the provenance says so."""
    prov = copy.deepcopy(twin.provenance or {})
    steps = prov.setdefault("steps", {})
    if not isinstance(steps, dict):
        steps = prov["steps"] = {}
    video = steps.get("video")
    if not isinstance(video, dict):
        video = steps["video"] = {}
    scale = video.get("scale")
    if not isinstance(scale, dict):
        scale = {}
    scale = dict(scale)
    scale["factor"] = 1.0
    scale["note"] = "the world splat is trained in the twin frame; no scale applies"
    video["scale"] = scale
    prov["source_splat_ply"] = (twin.provenance or {}).get("splat_ply")
    prov["splat_ply"] = str(Path(ply_path).resolve())
    world = dict(world_meta or {})
    world.setdefault("built_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    world.setdefault("note", RENDER_NOTE)
    world["source_twin"] = twin.name
    prov["world"] = world
    return Twin(
        name=f"{twin.name}-world",
        points=twin.points,
        mesh=twin.mesh,
        structure=twin.structure,
        georeference=twin.georeference,
        capture_bounds=twin.capture_bounds,
        qa=twin.qa,
        canonical_transform=np.eye(4),
        provenance=prov,
        shell_texture=twin.shell_texture,
        shell_texture_format=twin.shell_texture_format,
        shell_texture_layout=twin.shell_texture_layout,
    )


# ---------------------------------------------------------------------------
# the finisher's answers
# ---------------------------------------------------------------------------


def measured_finisher(items: list[tuple[str, bytes, dict]]) -> list[Any]:
    """The finisher that paints nothing: the measured renders come back as
    they went in. The world it yields is the twin re-splatted from its own
    renders -- a check of the whole path that costs no model calls, and the
    fallback when no finisher is available."""
    from ..meta.finish import FinishedFrame

    return [FinishedFrame(png=png, prompt="", model="none", provider="measured", name=name,
                          raw=png, meta=dict(meta)) for name, png, meta in items]


def _finished_bytes(finished: Any, items: list[tuple[str, bytes, dict]]) -> dict[str, bytes]:
    """Name -> PNG bytes from whatever shape a finisher answered in:
    FinishedFrame objects (unstamped `raw` preferred, `png` otherwise),
    (name, bytes[, meta]) tuples, dicts, or bare bytes aligned with items."""
    out: dict[str, bytes] = {}
    names = [n for n, _, _ in items]
    for k, f in enumerate(finished or []):
        if f is None:
            continue
        if isinstance(f, (bytes, bytearray)):
            if k < len(names):
                out[names[k]] = bytes(f)
            continue
        if isinstance(f, dict):
            name = f.get("name") or (names[k] if k < len(names) else None)
            data = f.get("raw") or f.get("png")
        elif isinstance(f, (tuple, list)):
            name, data = f[0], f[1]
        else:
            name = getattr(f, "name", "") or (names[k] if k < len(names) else None)
            data = getattr(f, "raw", None) or getattr(f, "png", None)
            if data is None and getattr(f, "path", None):
                data = Path(f.path).read_bytes()
        if name and data:
            out[str(name)] = bytes(data)
    return out


def _exact_png(data: bytes, size: tuple[int, int]) -> bytes:
    """The bytes as an RGB PNG of exactly `size`, resampled if the finisher
    changed the frame -- a trainer given a 1024-wide answer for a 960-wide
    pose would learn a camera that never existed."""
    from PIL import Image

    im = Image.open(io.BytesIO(data))
    im.load()
    if im.mode != "RGB":
        im = im.convert("RGB")
    if im.size != tuple(size):
        im = im.resize(tuple(size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# the whole thing
# ---------------------------------------------------------------------------


@dataclass
class WorldResult:
    views: list[View]
    rendered: list[Path]
    finished: list[Path]
    ply: Path | None
    html: Path | None
    coverage: dict[str, Any]
    seconds: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    twin_path: Path | None = None
    flythrough: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "views": [v.to_dict() for v in self.views],
            "rendered": [str(p) for p in self.rendered],
            "finished": [str(p) for p in self.finished],
            "ply": None if self.ply is None else str(self.ply),
            "html": None if self.html is None else str(self.html),
            "twin": None if self.twin_path is None else str(self.twin_path),
            "flythrough": None if self.flythrough is None else str(self.flythrough),
            "coverage": self.coverage,
            "seconds": {k: round(float(v), 2) for k, v in self.seconds.items()},
            "notes": list(self.notes),
            "note": RENDER_NOTE,
        }


def build_world(
    twin: Twin,
    workdir: str | Path,
    *,
    finish_fn: Callable[[list[tuple[str, bytes, dict]]], Any],
    figures_for=None,
    exclude: np.ndarray | None = None,
    n_views: int = 60,
    steps: int = 6000,
    method: str = "bake",
    light: dict | None = None,
    dressing: str = "",
    progress: ProgressFn | None = None,
    actors: dict | None = None,
    world_meta: dict[str, Any] | None = None,
    width_px: int = VIEW_WIDTH_PX,
    flythrough_s: float = 0.0,
) -> WorldResult:
    """Select, render, finish, train, and serve: the room as it will be.

    `finish_fn` takes the list of (name, png_bytes, meta) and returns the
    finished frames (the stage hands in Muse Image; `measured_finisher`
    hands the renders back). Everything lands under `workdir`: views/,
    finished/, colmap/, splat/, world.html, world.twin, coverage.json and
    world.json; `flythrough.mp4` too when `flythrough_s` > 0 and ffmpeg is
    on the path. `light` is the ephemeris answer for the hour, passed
    through to the finisher untouched -- this module computes no sun.
    `progress(what, done, total)` is called as stages advance.
    """
    t_start = time.time()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    seconds: dict[str, float] = {}
    notes: list[str] = []
    meta = dict(world_meta or {})
    meta.setdefault("dressing", dressing)
    if light:
        meta.setdefault("date", light.get("date"))
        meta.setdefault("hour", light.get("clock"))

    def tick(what, done=0, total=0):
        if progress:
            progress(what, int(done), int(total))

    tick("selecting views", 0, n_views)
    t0 = time.time()
    views, coverage = select_views(twin, n=n_views, focal_mm=VIEW_FOCAL_MM, width_px=width_px)
    seconds["select"] = time.time() - t0
    notes.extend(coverage.get("notes", []))
    if not coverage.get("complete", False):
        notes.append("coverage incomplete: " + ", ".join(coverage.get("uncovered", [])))
    (workdir / "coverage.json").write_text(json.dumps(coverage, indent=1))
    meta["views"] = len(views)

    t0 = time.time()
    items = render_views(twin, views, workdir / "views", figures_for=figures_for, exclude=exclude,
                         progress=progress, light=light, dressing=dressing, actors=actors)
    seconds["render"] = time.time() - t0
    rendered = [p for _, p, _ in items]

    tick("finishing views", 0, len(items))
    t0 = time.time()
    payload = [(n, p.read_bytes(), m) for n, p, m in items]
    answered = _finished_bytes(finish_fn(payload), payload)
    size = frame_size(width_px)
    fin_dir = workdir / "finished"
    fin_dir.mkdir(parents=True, exist_ok=True)
    img_dir = workdir / "colmap" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    finished: list[Path] = []
    missing = 0
    for n, png, _ in payload:
        data = answered.get(n)
        if data is None:
            missing += 1
            data = png
        exact = _exact_png(data, size)
        out = fin_dir / f"finish_{n}.png"
        out.write_bytes(exact)
        (img_dir / f"{n}.png").write_bytes(exact)
        finished.append(out)
    if missing:
        notes.append(f"{missing} views came back unfinished; the measured render stands in for them")
    seconds["finish"] = time.time() - t0
    tick("finished views", len(items), len(items))

    tick("writing the camera model", 0, 0)
    t0 = time.time()
    model_dir = write_colmap_model(twin, views, size, workdir / "colmap" / "sparse" / "0")
    seconds["model"] = time.time() - t0

    has_field = bool((twin.provenance or {}).get("splat_ply")) and Path(str((twin.provenance or {}).get("splat_ply"))).exists()
    if method == "bake" and has_field:
        tick("painting the finished views onto the scan's gaussians", 0, 0)
        t0 = time.time()
        ply = bake_world(twin, views, finished, workdir / "splat" / "world_baked.ply", exclude=exclude, progress=progress)
        seconds["bake"] = time.time() - t0
    else:
        if method == "bake":
            notes.append("this twin has no gaussian field to bake onto, so the world was trained from the views instead")
            method = "train"
        tick("training the world", 0, 0)
        t0 = time.time()
        ply = train_world(img_dir, model_dir, workdir / "splat", steps=steps, resolution=width_px, progress=progress)
        seconds["train"] = time.time() - t0

    tick("building the page", 0, 0)
    t0 = time.time()
    meta["steps"] = int(steps)
    meta["method"] = method
    # The finished pictures, pinned where they were taken: the viewer fades
    # one in when the walker stands on its spot and looks its way.
    meta["anchors"] = [view_anchor(v, f"finish_{v.name}.png") for v in views]
    wt = world_twin(twin, ply, world_meta=meta)
    from ..viewer.build import render_html

    html = render_html(wt, workdir / "world.html", max_points=600_000)
    twin_path = wt.save(workdir / "world.twin")
    seconds["html"] = time.time() - t0

    fly = None
    if flythrough_s and flythrough_s > 0:
        tick("rendering the flythrough", 0, 0)
        t0 = time.time()
        fly = flythrough(wt, views, workdir / "flythrough.mp4", seconds=float(flythrough_s), progress=progress)
        if fly is None:
            notes.append("no flythrough: ffmpeg is not on the path")
        seconds["flythrough"] = time.time() - t0

    seconds["total"] = time.time() - t_start
    result = WorldResult(views, rendered, finished, Path(ply), html, coverage, seconds, notes, twin_path, fly)
    (workdir / "world.json").write_text(json.dumps(result.to_dict(), indent=1))
    tick("world built", 1, 1)
    return result


# ---------------------------------------------------------------------------
# a walk through it
# ---------------------------------------------------------------------------


def _catmull_rom(pts: np.ndarray, count: int) -> np.ndarray:
    """A smooth open path through `pts`, sampled `count` times."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 1:
        return np.repeat(pts, count, axis=0)
    P = np.vstack([pts[:1], pts, pts[-1:]])
    segs = len(pts) - 1
    out = np.empty((count, pts.shape[1]))
    for k in range(count):
        s = (k / max(count - 1, 1)) * segs
        i = min(int(math.floor(s)), segs - 1)
        u = s - i
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        out[k] = 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                        + (-p0 + 3 * p1 - 3 * p2 + p3) * u * u * u)
    return out


def flythrough(
    world: Twin,
    views: list[View],
    out_mp4: str | Path,
    *,
    fps: int = 24,
    seconds: float = 8.0,
    width_px: int = VIEW_WIDTH_PX,
    progress: ProgressFn | None = None,
    focal_mm: float = VIEW_FOCAL_MM,
) -> Path | None:
    """A smooth camera path through six well-spread views, rendered from the
    new splat and encoded with ffmpeg. Returns the mp4, or None when ffmpeg
    is not installed -- the page still has the world; this is the trailer."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or not views:
        return None
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cams = np.array([v.cam for v in views], dtype=np.float64)
    aims = np.array([v.aim for v in views], dtype=np.float64)
    eye = [i for i, v in enumerate(views) if v.role not in ("low", "high")] or list(range(len(views)))
    start = eye[int(np.argmin(np.linalg.norm(cams[eye][:, :2] - cams[eye][:, :2].mean(axis=0), axis=1)))]
    picks = _farthest_point_order(cams[eye], eye.index(start), min(6, len(eye)))
    picks = [eye[i] for i in picks]
    # A short tour: greedy nearest-neighbour from the first pick.
    tour = [picks[0]]
    rest = picks[1:]
    while rest:
        last = cams[tour[-1]]
        j = min(range(len(rest)), key=lambda k: np.linalg.norm(cams[rest[k]] - last))
        tour.append(rest.pop(j))
    count = max(2, int(round(fps * seconds)))
    path_c = _catmull_rom(cams[tour], count)
    path_a = _catmull_rom(aims[tour], count)

    with tempfile.TemporaryDirectory() as td:
        from PIL import Image

        for k in range(count):
            if progress:
                progress("flythrough frame", k, count)
            v = View(f"fly_{k:04d}", tuple(path_c[k]), tuple(path_a[k]), focal_mm, width_px)
            im = render_view(world, v)
            W, H = im.width + (im.width % 2), im.height + (im.height % 2)
            if (W, H) != im.size:
                canvas = Image.new("RGB", (W, H), render.BACKGROUND)
                canvas.paste(im, (0, 0))
                im = canvas
            im.save(Path(td) / f"f_{k:05d}.png")
        proc = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(int(fps)),
             "-i", str(Path(td) / "f_%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(out_mp4)],
            capture_output=True, text=True, timeout=600,
        )
    if proc.returncode != 0 or not out_mp4.exists():
        return None
    return out_mp4


__all__ = [
    "View",
    "WorldResult",
    "frame_size",
    "select_views",
    "render_view",
    "render_views",
    "view_meta",
    "view_anchor",
    "world_to_camera",
    "write_colmap_model",
    "train_world",
    "world_twin",
    "measured_finisher",
    "build_world",
    "flythrough",
    "RENDER_NOTE",
]
