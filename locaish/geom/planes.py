"""Extraction of the planar surfaces a room is actually made of.

A room is mostly planes, and almost everything Locaish claims about a twin --
that the floor is level, that the ceiling is 2.74 m up, that this rectangle of
missing points is a window and not a shadow -- is a statement about one of
them. So this module has to find them without inventing any.

Two decisions here are load-bearing.

    The fit is total least squares, never `z = ax + by + c`. An ordinary least
    squares fit minimises error along one chosen axis, which is fine for a
    floor and catastrophic for a wall, where the surface is vertical and the
    "height above the plane" the fit is minimising runs along the wall instead
    of across it. `fit_plane` minimises perpendicular distance, so it does not
    care how the surface is oriented.

    RANSAC is normal-aware. Distance-only RANSAC on a furnished room is happy
    to report a plane that slices diagonally through a sofa, a table edge and
    half a wall, because those points really are coplanar -- they just are not
    a surface. Requiring each inlier's own normal to agree with the candidate
    within a dozen degrees removes that family of ghosts entirely, and it also
    stops the floor plane swallowing the bottom few centimetres of every
    vertical face in the room.
"""

from __future__ import annotations

import numpy as np

from ..types import Plane, PointCloud, chunked
from .normals import estimate_normals, orient_normals

__all__ = [
    "fit_plane",
    "detect_planes",
    "plane_inliers",
    "plane_frame",
    "planarity_rms",
]

# Cap on the number of points used to *propose* and *score* hypotheses. The
# winning hypothesis is always re-fitted and re-scored against the full cloud,
# so this trades nothing but a little seeding luck for a large constant factor.
_SAMPLE_CAP = 50_000
# Hypotheses tried per plane. Each is a point plus its own normal rather than a
# random triple, so a single sample on any real surface is already a near-miss
# and a couple of hundred of them saturate long before the classic 3-point
# RANSAC iteration count would.
_HYPOTHESES = 256
_REFIT_ROUNDS = 3
# |normal . Z| past which a plane counts as horizontal for framing purposes,
# i.e. within about 8 degrees of level. See `plane_frame`.
_HORIZONTAL_COS = 0.99


# ---------------------------------------------------------------------------
# single-plane primitives
# ---------------------------------------------------------------------------


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Total-least-squares plane through `points`, as (unit normal, offset).

    The convention is `dot(normal, p) = offset`, so a point's signed distance
    is `dot(normal, p) - offset`. The normal is the eigenvector of the centred
    covariance with the smallest eigenvalue, which is the direction of least
    spread and therefore the one perpendicular to the surface.

    The sign of the normal is not determined by the geometry and this function
    does not pretend otherwise; `detect_planes` fixes it afterwards from the
    inliers' own oriented normals.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 3:
        raise ValueError(f"a plane needs at least 3 points, got {len(pts)}")
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    cov = centred.T @ centred / float(len(pts))
    _, vectors = np.linalg.eigh(cov)
    normal = vectors[:, 0]
    length = float(np.linalg.norm(normal))
    if length < 1e-12:  # only reachable if the covariance is not finite
        raise ValueError("degenerate point set: no plane direction")
    normal = normal / length
    return normal, float(normal @ centroid)


def plane_frame(plane: Plane) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal in-plane basis (u, v), right-handed so that `cross(u, v)` is
    the plane normal.

    Stability matters more here than elegance. Every consumer of `extent_2d`
    compares u against v -- a window is wider than it is tall, a door reaches
    the floor -- so the basis must not swing by 90 degrees because a wall
    normal moved by a thousandth of a degree, and v must mean "up" on anything
    vertical.

    The reference axis is the world axis least aligned with the normal, but the
    choice is made by a threshold rather than by a bare `argmin`, because on a
    real fitted wall two axes are tied to within the fit's own noise and an
    argmin over that noise is a coin toss: a wall whose normal happens to carry
    a 1e-4 Z component and a 1e-6 Y component would take Y as reference and
    come back with v pointing sideways. So any plane more than about 8 degrees
    off level takes Z as its reference and gets v = the in-plane steepest
    ascent, which for a wall is exactly +Z. Only near-horizontal planes, where
    "up" has no meaning inside the plane and the projection of Z would be pure
    noise, fall back to the world Y axis, giving a floor u = +X and v = +Y.
    """
    n = np.asarray(plane.normal, dtype=np.float64).reshape(3)
    n = n / np.linalg.norm(n)

    reference = (
        np.array([0.0, 1.0, 0.0])
        if abs(n[2]) > _HORIZONTAL_COS
        else np.array([0.0, 0.0, 1.0])
    )

    v = reference - (reference @ n) * n
    v /= np.linalg.norm(v)
    u = np.cross(v, n)
    u /= np.linalg.norm(u)
    return u, v


def plane_inliers(
    cloud: PointCloud | np.ndarray,
    plane: Plane,
    *,
    distance_thresh: float = 0.03,
    normals: np.ndarray | None = None,
    normal_thresh_deg: float | None = None,
) -> np.ndarray:
    """Boolean mask of the points that belong to `plane`.

    `distance_thresh` should be a few times the scanner's range noise: 30 mm
    covers the ~6 mm per-point noise of a phone sweep plus the millimetres of
    real relief in plaster and paint, without reaching across a skirting board.

    Passing `normal_thresh_deg` adds the agreement test that makes this a
    surface query rather than a slab query. The comparison is on the absolute
    dot product, so it is indifferent to whether the plane normal and the point
    normals happen to have been oriented the same way. Asking for that test
    without supplying normals raises rather than quietly returning the weaker
    distance-only answer, because the two differ by thousands of points and a
    caller that thinks it got the strict test must not silently get the loose
    one.
    """
    xyz = cloud.xyz if isinstance(cloud, PointCloud) else np.asarray(cloud, np.float64).reshape(-1, 3)
    if normals is None and isinstance(cloud, PointCloud):
        normals = cloud.normals

    if normal_thresh_deg is not None and normals is None:
        raise ValueError(
            "normal_thresh_deg was given but no normals are available; "
            "pass normals= or set them on the cloud"
        )

    mask = np.abs(xyz @ plane.normal - plane.offset) <= float(distance_thresh)
    if normal_thresh_deg is not None:
        nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        limit = float(np.cos(np.radians(normal_thresh_deg)))
        mask &= np.abs(nrm @ plane.normal) >= limit
    return mask


def planarity_rms(
    cloud: PointCloud | np.ndarray,
    plane: Plane,
    mask: np.ndarray | None = None,
) -> float:
    """RMS perpendicular distance of the masked points from the plane, in metres.

    This is the number QA uses to tell a wall from a drifting reconstruction of
    a wall: sub-centimetre is a real flat surface seen through range noise, and
    anything past that means the surface is bowed, the sweep drifted, or the
    "plane" is really two planes at a shallow angle.
    """
    xyz = cloud.xyz if isinstance(cloud, PointCloud) else np.asarray(cloud, np.float64).reshape(-1, 3)
    if mask is not None:
        mask = np.asarray(mask)
        xyz = xyz[mask] if mask.dtype == bool else xyz[mask.astype(np.int64)]
    if len(xyz) == 0:
        return float("nan")
    d = xyz @ plane.normal - plane.offset
    return float(np.sqrt(np.mean(d * d)))


# ---------------------------------------------------------------------------
# extent and area of a set of inliers
# ---------------------------------------------------------------------------


def _extent_and_area(
    points: np.ndarray, plane: Plane
) -> tuple[tuple[float, float, float, float], float, float]:
    """In-plane bounding box, occupied area, and the occupancy fraction.

    The bounding rectangle is a bad estimate of a wall's area and an actively
    misleading one for the shapes rooms are full of: an L-shaped wall, a floor
    with a chimney breast cut out of it, a wall with a two-square-metre window
    in it. So we bin the inliers onto a regular grid in the plane's own frame
    and count occupied cells. That charges nothing for the window and nothing
    for the missing corner of the L.

    The cell size is derived from the observed point density, targeting a few
    points per occupied cell. Fixing it instead would make the estimate a
    measure of scan density rather than of area -- too fine and a sparse
    capture reads as a sieve, too coarse and a window disappears into its own
    surround.

    Counting occupied cells still under-reads, because a covered cell that
    happens to catch no sample looks like a hole. With samples landing roughly
    Poisson at rate lambda per cell, an occupied count O over N points implies
    C covered cells through O = C (1 - exp(-N / C)), which a few fixed-point
    passes invert. Without that correction the same wall measures several
    percent smaller in a sparse scan than in a dense one, which is precisely
    the scan-density artefact the adaptive cell size is there to avoid.
    """
    u, v = plane_frame(plane)
    uu = points @ u
    vv = points @ v
    umin, umax = float(uu.min()), float(uu.max())
    vmin, vmax = float(vv.min()), float(vv.max())
    extent = (umin, vmin, umax, vmax)

    span_u = max(umax - umin, 1e-9)
    span_v = max(vmax - vmin, 1e-9)
    rect = span_u * span_v

    density = len(points) / rect                      # points per m^2
    cell = float(np.clip(np.sqrt(4.0 / max(density, 1e-9)), 0.04, 0.40))

    nu = int(span_u / cell) + 1
    nv = int(span_v / cell) + 1
    iu = np.minimum(((uu - umin) / cell).astype(np.int64), nu - 1)
    iv = np.minimum(((vv - vmin) / cell).astype(np.int64), nv - 1)
    occupied = float(np.unique(iu * nv + iv).size)

    covered = occupied
    for _ in range(8):
        empty_fraction = 1.0 - np.exp(-len(points) / max(covered, 1e-9))
        if empty_fraction < 1e-6:  # far too few points per cell to correct
            break
        covered = min(occupied / empty_fraction, float(nu * nv))

    # The grid pads the rectangle out to whole cells, so cap the answer at the
    # rectangle it came from rather than reporting more area than exists.
    area = min(covered * cell * cell, rect)
    return extent, float(area), float(area / rect)


# ---------------------------------------------------------------------------
# iterative RANSAC
# ---------------------------------------------------------------------------


def _score_hypotheses(
    xyz: np.ndarray,
    normals: np.ndarray,
    cand_n: np.ndarray,
    cand_d: np.ndarray,
    distance_thresh: float,
    cos_limit: float,
) -> np.ndarray:
    """Inlier count of every candidate plane against every point, in one pass.

    Chunked over the points because the natural (points, candidates) score
    matrix is hundreds of megabytes at the sizes this runs on, and we only ever
    need its column sums.
    """
    counts = np.zeros(len(cand_n), dtype=np.int64)
    for sl in chunked(len(xyz), 20_000):
        dist = xyz[sl] @ cand_n.T - cand_d[None, :]
        agree = normals[sl] @ cand_n.T
        counts += (
            (np.abs(dist) <= distance_thresh) & (np.abs(agree) >= cos_limit)
        ).sum(axis=0)
    return counts


def detect_planes(
    cloud: PointCloud,
    *,
    normals: np.ndarray | None = None,
    min_inliers: int = 500,
    max_planes: int = 24,
    distance_thresh: float = 0.03,
    normal_thresh_deg: float = 12.0,
    seed: int = 0,
) -> list[Plane]:
    """Find the dominant planar surfaces, largest first.

    The loop is: propose candidates, score them, take the best, refit it on its
    own inliers, remove those inliers, repeat. The refit is the step that
    earns the accuracy -- a hypothesis anchored on one noisy point carries that
    point's error, while a least-squares fit over tens of thousands of inliers
    averages the range noise away and lands the floor normal within hundredths
    of a degree of the truth.

    Candidates are seeded as "a point and its own normal" rather than as random
    triples. Three random points from a room are almost never coplanar with a
    real surface, so classical RANSAC needs thousands of iterations to get
    lucky; a point plus its estimated normal is already a plane hypothesis that
    is only wrong by that normal's noise, which the refit then removes.

    The cloud must already be in metres. `distance_thresh` is an absolute
    length, so handing this a centimetre-unit export asks it to fit planes to a
    0.3 mm tolerance and it will honestly find nothing; the unit inference
    stage runs before this one for exactly that reason.

    `normals` may be supplied to avoid recomputing them. When they are not, we
    estimate and orient them here, using the cloud centroid as the viewpoint --
    correct for a room seen from inside, which is what a scan is.

    Planes come back with `kind` left as "other" on purpose. Deciding which one
    is the floor is an architectural judgement about the whole set, not a
    property of any single fit, and it belongs to the structure stage.
    """
    xyz = np.ascontiguousarray(cloud.xyz, dtype=np.float64)
    n_points = len(xyz)
    if n_points < max(3, min_inliers):
        return []

    if normals is None:
        normals = cloud.normals
    if normals is None:
        normals = orient_normals(estimate_normals(cloud), xyz)
    normals = np.ascontiguousarray(np.asarray(normals, dtype=np.float64).reshape(-1, 3))
    if len(normals) != n_points:
        raise ValueError(f"{len(normals)} normals for {n_points} points")

    rng = np.random.default_rng(seed)
    cos_limit = float(np.cos(np.radians(normal_thresh_deg)))
    active = np.ones(n_points, dtype=bool)

    planes: list[Plane] = []
    misses = 0
    while len(planes) < max_planes and misses < 2:
        idx_active = np.flatnonzero(active)
        if len(idx_active) < min_inliers:
            break

        # Score on a bounded subsample; the winner is re-scored on everything.
        if len(idx_active) > _SAMPLE_CAP:
            idx_sub = idx_active[
                rng.choice(len(idx_active), size=_SAMPLE_CAP, replace=False)
            ]
        else:
            idx_sub = idx_active
        sub_xyz = xyz[idx_sub]
        sub_nrm = normals[idx_sub]

        seeds = rng.choice(len(idx_sub), size=min(_HYPOTHESES, len(idx_sub)), replace=False)
        cand_n = sub_nrm[seeds]
        cand_d = np.einsum("ij,ij->i", cand_n, sub_xyz[seeds])
        counts = _score_hypotheses(
            sub_xyz, sub_nrm, cand_n, cand_d, distance_thresh, cos_limit
        )
        best = int(np.argmax(counts))
        normal, offset = cand_n[best], float(cand_d[best])

        # Refit on the full remaining cloud. Each round widens the inlier set
        # slightly as the fit centres itself on the surface, and it converges
        # in two or three passes; more would only chase noise.
        mask = np.zeros(0, dtype=bool)
        for _ in range(_REFIT_ROUNDS):
            dist = np.abs(xyz @ normal - offset) <= distance_thresh
            agree = np.abs(normals @ normal) >= cos_limit
            mask = active & dist & agree
            if int(mask.sum()) < 3:
                break
            normal, offset = fit_plane(xyz[mask])

        count = int(mask.sum())
        if count < min_inliers:
            misses += 1
            continue
        misses = 0

        inlier_xyz = xyz[mask]
        # PCA gave an arbitrary sign; the inliers' own oriented normals know
        # which side the scanner was on, so let them vote.
        if float(np.einsum("ij,j->i", normals[mask], normal).sum()) < 0.0:
            normal, offset = -normal, -offset

        plane = Plane(normal=normal, offset=offset, kind="other", inlier_count=count)
        plane.extent_2d, plane.area, _ = _extent_and_area(inlier_xyz, plane)
        planes.append(plane)
        active &= ~mask

    planes.sort(key=lambda p: p.inlier_count, reverse=True)
    return planes
