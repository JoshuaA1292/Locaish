"""Geometry: the numerical layer that turns points into surfaces.

Only the surface-fitting primitives live here so far. Other stages of the
pipeline add their own modules alongside these and re-export from this file.
"""

from __future__ import annotations

from .normals import estimate_normals, local_curvature, orient_normals
from .planes import (
    detect_planes,
    fit_plane,
    plane_frame,
    plane_inliers,
    planarity_rms,
)

__all__ = [
    "estimate_normals",
    "orient_normals",
    "local_curvature",
    "fit_plane",
    "detect_planes",
    "plane_inliers",
    "plane_frame",
    "planarity_rms",
]
