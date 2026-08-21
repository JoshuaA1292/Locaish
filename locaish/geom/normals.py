"""Per-point surface normals by local principal component analysis.

A normal is the first thing in the pipeline that turns a bag of coordinates
into a surface, and everything after it -- plane detection, gravity alignment,
opening detection, the QA planarity checks -- is only as good as this. So the
estimator is deliberately plain: take the k nearest neighbours, take the
eigenvector of their covariance with the smallest eigenvalue, and be honest
about the fact that the sign of that eigenvector is meaningless until something
external fixes it.

The two failure modes this module is built around:

    Scale. A LiDAR sweep of a flat is one to twenty million points, and the
    obvious `xyz[idx] - mean` intermediate for all of them at once is tens of
    gigabytes. Everything here works in chunks over the kd-tree query and uses
    a batched `eigh` on a (chunk, 3, 3) stack rather than a Python loop.

    Sign. PCA cannot tell the front of a wall from its back. Getting that wrong
    inverts every clearance query downstream, so `orient_normals` insists on
    some outside evidence -- the camera path when the export gave us poses, and
    otherwise the cloud centroid, which is the correct assumption for a room
    photographed from the inside and the wrong one for an object photographed
    from the outside.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..types import PointCloud, chunked

__all__ = ["estimate_normals", "orient_normals", "local_curvature"]


# ---------------------------------------------------------------------------
# shared machinery
# ---------------------------------------------------------------------------


def _as_xyz(cloud: PointCloud | np.ndarray) -> np.ndarray:
    """Accept either a PointCloud or a raw (N, 3) array.

    The pipeline passes clouds, but callers doing a one-off check on a slice of
    points should not have to wrap it in a dataclass just to ask a question.
    """
    if isinstance(cloud, PointCloud):
        return cloud.xyz
    return np.asarray(cloud, dtype=np.float64).reshape(-1, 3)


def _local_pca(
    xyz: np.ndarray,
    k: int,
    chunk: int,
    *,
    want_vectors: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Eigenvalues (and optionally eigenvectors) of every point's k-neighbourhood.

    Returns `(eigenvalues, eigenvectors)` with eigenvalues (N, 3) ascending and
    eigenvectors (N, 3, 3) whose *columns* are the axes, matching `eigh`. The
    covariance is divided by k rather than k-1 because the absolute scale never
    matters here -- only ratios of eigenvalues and the direction of the first
    one -- and the unbiased form buys nothing while inviting a divide by zero
    when a neighbourhood collapses to a single point.
    """
    n = len(xyz)
    k = int(max(3, min(k, n)))
    tree = cKDTree(xyz)

    values = np.empty((n, 3), dtype=np.float64)
    vectors = np.empty((n, 3, 3), dtype=np.float64) if want_vectors else None

    for sl in chunked(n, max(1, int(chunk))):
        # workers=-1 because the query is the wall-clock cost of this function
        # on a real sweep and it is embarrassingly parallel.
        _, idx = tree.query(xyz[sl], k=k, workers=-1)
        idx = np.atleast_2d(idx)
        nb = xyz[idx]                                   # (m, k, 3)
        nb -= nb.mean(axis=1, keepdims=True)
        cov = np.matmul(nb.transpose(0, 2, 1), nb) / float(k)
        if want_vectors:
            w, v = np.linalg.eigh(cov)
            values[sl] = w
            vectors[sl] = v
        else:
            values[sl] = np.linalg.eigvalsh(cov)

    return values, vectors


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def estimate_normals(
    cloud: PointCloud | np.ndarray,
    *,
    k: int = 24,
    chunk: int = 200_000,
) -> np.ndarray:
    """(N, 3) unit normals from local PCA over the k nearest neighbours.

    k is a compromise. Too few and range noise rotates the normal by degrees;
    too many and the neighbourhood spills across a corner, so the normal at the
    floor-wall junction ends up at 45 degrees and belongs to neither surface.
    24 at the ~900 points/m^2 a handheld sweep yields covers a patch of roughly
    9 cm across, which averages out millimetre noise while staying well inside
    a skirting board.

    The returned signs are arbitrary; pass the result through `orient_normals`
    before anything reads them as "which way does this surface face".
    """
    xyz = _as_xyz(cloud)
    if len(xyz) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if len(xyz) < 3:
        # Not enough points to define a plane at all. Claiming +Z would be a
        # silent guess, but a zero vector propagates as a NaN later, so we
        # return +Z and rely on the caller's inlier counts to reject it.
        return np.tile(np.array([0.0, 0.0, 1.0]), (len(xyz), 1))

    _, vectors = _local_pca(xyz, k=k, chunk=chunk, want_vectors=True)
    normals = np.ascontiguousarray(vectors[:, :, 0])

    # A neighbourhood of coincident points gives a zero covariance and an
    # arbitrary (but still unit) eigenbasis; renormalising is cheap insurance
    # against any degenerate case that does slip through as a short vector.
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(length < 1e-12, 1.0, length)


def orient_normals(
    normals: np.ndarray,
    xyz: np.ndarray | PointCloud,
    *,
    viewpoints: np.ndarray | None = None,
    centroid_fallback: bool = True,
) -> np.ndarray:
    """Resolve the sign of PCA normals so they face where the scanner was.

    With camera positions this is a measurement rather than a prior: the
    scanner only ever saw the front of a surface, so the normal of a point must
    have a positive component toward the camera that observed it. We use the
    nearest recorded viewpoint as a stand-in for the observing one, which is
    correct except for points seen only from far across the room at a grazing
    angle -- and those are exactly the points whose PCA normal is unreliable
    anyway.

    Without poses we flip toward the cloud centroid. For a room captured from
    the inside that gives floor normals +Z, ceiling normals -Z and wall normals
    pointing at the interior, which is the convention `Plane` documents. It is
    the wrong assumption for an object scanned from the outside, so it is a
    fallback we name rather than hide; set `centroid_fallback=False` to get the
    raw signs back untouched when the caller has a better idea.
    """
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    pts = _as_xyz(xyz)
    if len(normals) != len(pts):
        raise ValueError(f"{len(normals)} normals for {len(pts)} points")
    if len(pts) == 0:
        return normals.copy()

    if viewpoints is not None:
        vp = np.asarray(viewpoints, dtype=np.float64).reshape(-1, 3)
        if len(vp) == 0:
            viewpoints = None

    if viewpoints is not None:
        tree = cKDTree(vp)
        toward = np.empty_like(pts)
        for sl in chunked(len(pts), 200_000):
            _, idx = tree.query(pts[sl], k=1, workers=-1)
            toward[sl] = vp[np.asarray(idx).reshape(-1)] - pts[sl]
    elif centroid_fallback:
        toward = pts.mean(axis=0) - pts
    else:
        return normals.copy()

    flip = np.einsum("ij,ij->i", normals, toward) < 0.0
    out = normals.copy()
    out[flip] *= -1.0
    return out


def local_curvature(
    cloud: PointCloud | np.ndarray,
    *,
    k: int = 24,
    chunk: int = 200_000,
) -> np.ndarray:
    """(N,) surface variation, the smallest eigenvalue over the sum of the three.

    This is bounded in [0, 1/3] by construction: 0 when the neighbourhood is
    perfectly flat, 1/3 when it is isotropic noise with no surface in it at
    all. It is not curvature in the differential-geometry sense and does not
    pretend to be -- what it actually measures is how much of the local point
    spread fails to lie in a plane, which is the number plane seeding and QA
    want. A painted wall sits near 0.001, a potted plant near 0.2.
    """
    xyz = _as_xyz(cloud)
    if len(xyz) < 3:
        return np.zeros(len(xyz), dtype=np.float64)
    values, _ = _local_pca(xyz, k=k, chunk=chunk, want_vectors=False)
    total = values.sum(axis=1)
    return values[:, 0] / np.where(total < 1e-30, 1.0, total)
