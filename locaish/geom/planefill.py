"""Completing walls the capture proved exist, on planes it already measured.

Textureless indoor walls are where photometric stereo goes to die: there is
nothing to match on blank paint, so the dense stage returns walls full of
holes, and every viewer shows a room whose boundaries are made of confetti.
The research consensus on this failure (the ACMP/ACMMP line of work) is that
low-texture indoor regions are overwhelmingly *planar*, so a planar prior
recovers what photometric consistency cannot. This module is that prior,
applied after the fact and only where it can be defended.

The pipeline has already done the hard part twice over. Plane detection fitted
each wall with normals and an inlier set; the occupancy grid knows which cells
of the wall hold measured returns and which are void; and the camera path says
which parts of the room were actually looked at. So for each wall plane, the
void cells inside the wall's convex support are candidates, and a candidate is
filled only when the capture's own geometry proves the wall is the surface
there:

- the space immediately *in front* of the candidate, into the room, must have
  been carved free by camera rays -- the camera looked at this patch of wall
  and nothing nearer intercepted the view. A void that was never observed
  stays a void; the twin does not get to invent wall behind the sofa.

- the space immediately *behind* the candidate must NOT have been carved free
  -- a camera that saw through this cell saw a doorway or a window, not paint.
  Detected openings are additionally masked out by their fitted rectangles,
  but the ray test is the structural guarantee: nothing this module writes can
  seal an aperture the sweep looked through.

Every point written here is tagged `inferred = 1.0` and desaturated, exactly
as `Mesh.filled` treats completed mesh surface: the viewer shows a complete
wall, the label says which parts of it are a statement of the plane model
rather than a measurement, and everything that quotes a number excludes them
(`PointCloud.measured`). A complete-looking wall with an auditable honesty
boundary, not a better-looking lie.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..types import PointCloud, Structure
from .infill import MIN_CARVED_FRACTION, carve_free_space, observed_volume
from .planes import plane_frame

# Band around the plane whose points count as wall material, matching the
# structure module's own reading of "on the wall".
PLANE_BAND_M = 0.06

# Resolution of the wall raster the voids are found in. Coarser than the
# occupancy grid buys nothing; finer than it asks the carve for answers it
# does not have.
RASTER_RES_M = 0.05

# Spacing of the resampled points inside an accepted void cell. Half the
# raster, so filled wall reads as surface rather than as a grid of beacons.
SAMPLE_SPACING_M = 0.025

# Walls shorter than this along either axis are furniture flanks, and holes in
# furniture are not the pipeline's to fill.
MIN_WALL_SPAN_M = 1.2

# How far in front of / behind the plane the observation tests probe, in
# multiples of the grid voxel. 1.5 puts the probe safely in the neighbouring
# cell without reaching past it.
PROBE_VOXELS = 1.5

# Margin added around a detected opening's rectangle before masking it out of
# the fill, absorbing the raster quantisation of the opening's own edges.
OPENING_MARGIN_M = 0.05

# Backstop against a degenerate scene filling without bound.
MAX_FILL_POINTS = 500_000


def fill_wall_planes(
    cloud: PointCloud,
    structure: Structure,
    grid: Any,
    cameras: np.ndarray | None,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Resample points onto wall planes where the capture proved wall.

    Returns `(xyz, rgb, normals, stats)` where `xyz` is (N, 3) of new inferred
    points, `rgb` their desaturated colours, `normals` the owning plane's
    normal per point, and `stats` the receipts. N is 0 -- with the reason
    recorded -- whenever the evidence is not there: no camera poses, no carve,
    no walls.
    """
    stats: dict = {"filled": False, "reason": None, "walls": []}
    empty = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), np.zeros((0, 3))
    if cameras is None or len(np.atleast_2d(cameras)) < 2:
        stats["reason"] = "no camera poses, so observation of the voids cannot be proven"
        return (*empty, stats)
    walls = [p for p in structure.walls() if _span_ok(p)]
    if not walls:
        stats["reason"] = "no wall planes large enough to fill"
        return (*empty, stats)

    cams = np.asarray(cameras, dtype=np.float64).reshape(-1, 3)
    occupied = grid.occupied
    free = carve_free_space(
        grid.shape, grid.origin, grid.voxel, cloud.xyz, cams,
        solid=occupied, seed=seed,
    )
    free &= ~occupied
    if float(free.sum()) / float(free.size) < MIN_CARVED_FRACTION:
        stats["reason"] = "too little of the volume could be shown empty to test the voids"
        return (*empty, stats)
    # Two readings of the carve, one per test. The raw carve is speckled --
    # rays are a sample, and most voxels in genuinely observed space have no
    # ray through them by luck -- so the *front* test (was this patch of wall
    # observed?) reads the morphologically closed volume, which is the same
    # repair `complete_shell` applies before trusting the carve. The *behind*
    # test (did the camera see through here?) keeps the raw carve: closing
    # only ever adds free space, and an added voxel behind a wall would veto
    # fill the capture never argued against. Asymmetric on purpose -- each
    # test errs toward filling less. `shadow_close_m=0` matters: the default
    # occlusion-shadow bridge exists to put completed floor under furniture,
    # and here it would claim the wall *behind* the furniture was observed,
    # which is the one promise this module must not break.
    observed = observed_volume(free, voxel=grid.voxel, shadow_close_m=0.0)

    xyz = cloud.xyz
    rgb = cloud.rgb
    out_pts: list[np.ndarray] = []
    out_rgb: list[np.ndarray] = []
    out_nrm: list[np.ndarray] = []
    total = 0
    for plane in walls:
        got = _fill_one_wall(xyz, rgb, plane, structure, grid, observed, free)
        if got is None:
            continue
        pts, colour, wall_stats = got
        stats["walls"].append(wall_stats)
        if len(pts) == 0:
            continue
        if total + len(pts) > MAX_FILL_POINTS:
            pts = pts[: MAX_FILL_POINTS - total]
        out_pts.append(pts)
        out_rgb.append(np.tile(colour, (len(pts), 1)))
        out_nrm.append(np.tile(plane.normal, (len(pts), 1)))
        total += len(pts)
        if total >= MAX_FILL_POINTS:
            break

    if not out_pts:
        stats["reason"] = stats["reason"] or "no void passed the observation tests"
        return (*empty, stats)
    stats.update(filled=True, points_added=total)
    return (
        np.concatenate(out_pts),
        np.concatenate(out_rgb).astype(np.uint8),
        np.concatenate(out_nrm),
        stats,
    )


def _span_ok(plane) -> bool:
    if plane.extent_2d is None:
        return True  # measured against the inliers in _fill_one_wall instead
    umin, vmin, umax, vmax = plane.extent_2d
    return (umax - umin) >= MIN_WALL_SPAN_M and (vmax - vmin) >= MIN_WALL_SPAN_M


def _fill_one_wall(xyz, rgb, plane, structure, grid, observed, free):
    """The fill for a single wall plane, or None when it has nothing to say."""
    u_ax, v_ax = plane_frame(plane)
    s = xyz @ plane.normal - plane.offset
    on = np.abs(s) <= PLANE_BAND_M
    if int(on.sum()) < 500:
        return None
    wall = xyz[on]
    wu = wall @ u_ax
    wv = wall @ v_ax

    # Trim the straggler inliers every plane fit picks up from the walls that
    # meet this one at the corners, then check the span the raster will cover.
    u0, u1 = (float(x) for x in np.percentile(wu, [0.3, 99.7]))
    v0, v1 = (float(x) for x in np.percentile(wv, [0.3, 99.7]))
    if u1 - u0 < MIN_WALL_SPAN_M or v1 - v0 < MIN_WALL_SPAN_M:
        return None

    res = RASTER_RES_M
    nu = max(3, int(np.ceil((u1 - u0) / res)))
    nv = max(3, int(np.ceil((v1 - v0) / res)))
    iu = np.clip(((wu - u0) / res).astype(np.int64), 0, nu - 1)
    iv = np.clip(((wv - v0) / res).astype(np.int64), 0, nv - 1)
    material = np.zeros((nu, nv), dtype=bool)
    material[iu, iv] = True

    # The wall's convex support: voids are only voids inside the region the
    # wall demonstrably spans. skimage's convex hull of the material cells is
    # exactly that region on the raster.
    from skimage.morphology import convex_hull_image

    hull = convex_hull_image(material)
    void = hull & ~material
    if not void.any():
        return None

    # Detected openings are masked out with a margin. The ray test below would
    # catch them too -- carved space behind the plane -- but the fitted
    # rectangle is sharper than the carve's voxelisation at the edges, and a
    # window half-sealed by fill is worse than either state.
    for opening in structure.openings:
        if abs(float(opening.normal @ plane.normal)) < 0.85:
            continue
        oc_u = float(opening.center @ u_ax)
        oc_v = float(opening.center @ v_ax)
        half_w = opening.width / 2.0 + OPENING_MARGIN_M
        half_h = opening.height / 2.0 + OPENING_MARGIN_M
        lo_i = max(0, int((oc_u - half_w - u0) / res))
        hi_i = min(nu, int(np.ceil((oc_u + half_w - u0) / res)))
        lo_j = max(0, int((oc_v - half_h - v0) / res))
        hi_j = min(nv, int(np.ceil((oc_v + half_h - v0) / res)))
        void[lo_i:hi_i, lo_j:hi_j] = False
    if not void.any():
        return None

    # World positions of the void cell centres, on the plane.
    ii, jj = np.nonzero(void)
    cu = u0 + (ii + 0.5) * res
    cv = v0 + (jj + 0.5) * res
    base = plane.offset * plane.normal
    centres = base + cu[:, None] * u_ax + cv[:, None] * v_ax

    # Observation tests against the carve: observed in front (the camera saw
    # up to the wall here, nothing closer intercepted -- read off the closed
    # volume, since the raw carve is ray-sampled speckle), not seen *through*
    # (read off the raw carve, where a free voxel is a proof and not a
    # morphological artefact).
    delta = PROBE_VOXELS * float(np.min(grid.voxel))
    front_free = _free_at(observed, grid, centres + delta * plane.normal)
    behind_free = _free_at(free, grid, centres - delta * plane.normal)
    ok = front_free & ~behind_free
    if not ok.any():
        return _nothing(plane, void, 0)

    # Resample the accepted cells at sub-cell spacing so filled wall reads as
    # surface. A fixed sub-grid per cell, no randomness to seed.
    k = max(1, int(round(res / SAMPLE_SPACING_M)))
    offs = (np.arange(k) + 0.5) / k * res
    du, dv = np.meshgrid(offs, offs, indexing="ij")
    du, dv = du.ravel(), dv.ravel()
    su = (u0 + ii[ok] * res)[:, None] + du[None, :]
    sv = (v0 + jj[ok] * res)[:, None] + dv[None, :]
    pts = base + su.ravel()[:, None] * u_ax + sv.ravel()[:, None] * v_ax

    # One colour per wall: the median of its measured points, pulled hard
    # toward grey exactly as the mesher mutes filled vertices. The uniformity
    # is deliberate -- invented surface should look like a label, not like
    # plaster the camera photographed.
    if rgb is not None:
        base_col = np.median(rgb[on].astype(np.float64), axis=0)
    else:
        base_col = np.array([200.0, 200.0, 200.0])
    colour = np.clip(base_col * 0.25 + 128.0 * 0.75, 0, 255).astype(np.uint8)

    wall_stats = {
        "kind": plane.kind,
        "void_cells": int(void.sum()),
        "filled_cells": int(ok.sum()),
        "filled_area_m2": float(ok.sum()) * res * res,
    }
    return pts, colour, wall_stats


def _nothing(plane, void, filled):
    return (
        np.zeros((0, 3)),
        np.zeros(3, dtype=np.uint8),
        {
            "kind": plane.kind,
            "void_cells": int(void.sum()),
            "filled_cells": int(filled),
            "filled_area_m2": 0.0,
        },
    )


def _free_at(free: np.ndarray, grid, points: np.ndarray) -> np.ndarray:
    """Whether each world point falls in a carved-free voxel. Outside is False.

    False outside the grid on purpose, and asymmetrically: for the front test
    it means an unproven view stays unfilled, and for the behind test it means
    space beyond the grid -- which the carve never cleared -- reads as
    unobserved rather than as an aperture.
    """
    idx = np.floor((points - grid.origin) / grid.voxel).astype(np.int64)
    dims = np.array(free.shape, dtype=np.int64)
    inside = np.all((idx >= 0) & (idx < dims), axis=1)
    out = np.zeros(len(points), dtype=bool)
    if inside.any():
        sel = idx[inside]
        out[inside] = free[sel[:, 0], sel[:, 1], sel[:, 2]]
    return out
