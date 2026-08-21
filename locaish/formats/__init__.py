"""Readers that turn a consumer scanner's export into a `ScanImport`.

This is the pipeline's front door, and its job is to be the only place that
knows what a file format is. Everything downstream sees points, an optional
mesh, and a list of things that were odd about the file -- never a byte offset,
never a property name, never an exporter's quirk.

Two decisions live here rather than in the individual readers because they are
about the twin, not about the format:

Subsampling is done in space, not in file order. A LiDAR export is written in
scan order, so truncating a 20M-point PLY to its first 2M points keeps only the
corner of the room the user happened to start in and throws the rest away --
the resulting twin is not a coarser room, it is a smaller one. `read_scan`
instead keeps one point per voxel at whatever voxel size hits the budget, which
preserves the extent and the wall geometry that every measurement depends on.

A mesh-only import gets its cloud by sampling the surface, because the rest of
Phase 1 -- normals, plane fitting, occupancy, structure -- is written against
points. A mesh whose vertices are far too sparse to carry the geometry (a
RoomPlan export can describe a whole flat in 200 triangles) is treated the same
way, since a 200-point "cloud" would defeat plane fitting entirely.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..types import Mesh, PointCloud
from .detect import EXTENSION_MAP, READABLE, sniff

__all__ = [
    "ScanImport",
    "read_scan",
    "sniff",
    "write_ply",
    "voxel_subsample_indices",
]

# How many points to sample per unit of *shape-normalised* area, where the
# normalised area is surface area over the squared bounding-box diagonal (about
# 1.8 for a closed room). Normalising is what makes this unit-blind: the reader
# runs before `scale.infer_unit_scale`, so a room may still be measured in
# centimetres here, and a points-per-square-metre rule would ask for ten
# thousand times as many samples for the same room.
MESH_SAMPLE_SCALE = 50_000.0
MESH_SAMPLE_MIN = 50_000
MESH_SAMPLE_MAX = 3_000_000

# A cloud with fewer points than this cannot support plane fitting or normal
# estimation at all, so when a mesh is available its surface is sampled
# instead. The test is an absolute count rather than a density because density
# depends on the unit, which nothing has inferred yet, and because a mesh whose
# triangles are pathological would otherwise drag the threshold with it.
MIN_CLOUD_FOR_PLANES = 20_000

# Sampling is deterministic; the seed is fixed so two runs of the same import
# produce byte-identical twins.
SAMPLE_SEED = 0


@dataclass
class ScanImport:
    """One scan file, decoded, with everything we noticed on the way in.

    `points` is always present, even if it had to be sampled off a mesh.
    `mesh` is None when the file had no faces -- the mesher builds one later,
    and inventing an empty mesh here would hide that from it.

    `up_hint` is a unit vector in source coordinates that the source claims
    points away from the floor -- from a device that recorded its own
    orientation, or from reconstructed camera poses. Like the rest, it is a
    declaration and may be wrong; gravity recovery weighs it against the room's
    own geometry rather than obeying it.

    `software`, `unit_hint`, `up_hint` and `camera_positions` are
    *declarations*, not measurements: each is None unless the file actually said so, because
    downstream has to be able to distinguish "the file says metres" from "we
    assumed metres". `warnings` carries every lossy or ambiguous decision made
    while reading, and `raw_header` keeps the original header fields so a
    surprising twin can be traced back to the bytes that produced it.
    """

    points: PointCloud
    mesh: Mesh | None = None
    source_path: Path = Path(".")
    source_format: str = "unknown"
    software: str | None = None
    camera_positions: np.ndarray | None = None
    up_hint: np.ndarray | None = None
    unit_hint: str | None = None
    warnings: list[str] = field(default_factory=list)
    raw_header: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source_path = Path(self.source_path)
        if self.camera_positions is not None:
            self.camera_positions = np.asarray(
                self.camera_positions, dtype=np.float64
            ).reshape(-1, 3)
        if self.up_hint is not None:
            up = np.asarray(self.up_hint, dtype=np.float64).reshape(3)
            norm = float(np.linalg.norm(up))
            self.up_hint = up / norm if norm > 1e-9 else None

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.source_path),
            "format": self.source_format,
            "software": self.software,
            "points": len(self.points),
            "faces": 0 if self.mesh is None else len(self.mesh),
            "has_color": self.points.rgb is not None,
            "has_normals": self.points.normals is not None,
            "has_texture": self.mesh is not None and self.mesh.texture is not None,
            "camera_positions": (
                0 if self.camera_positions is None else len(self.camera_positions)
            ),
            "unit_hint": self.unit_hint,
            "warnings": len(self.warnings),
        }


def read_scan(path: str | Path, *, max_points: int | None = None) -> ScanImport:
    """Read any supported scan export into a `ScanImport`.

    The format is sniffed from the bytes, not the extension. `max_points` caps
    the cloud by voxel subsampling -- uniform in space and deterministic -- and
    the reduction is recorded in `warnings` so no downstream density statistic
    is quoted as if it came from the original file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such scan file: {path}")

    fmt = sniff(path)
    scan = _dispatch(path, fmt)

    if scan.mesh is not None:
        target = _sampling_target(scan.mesh)
        if len(scan.points) == 0:
            _sample_mesh_into(scan, target, max_points, "the import had a mesh but no points")
        elif len(scan.points) < MIN_CLOUD_FOR_PLANES:
            _sample_mesh_into(
                scan,
                target,
                max_points,
                f"the mesh carried only {len(scan.points)} vertices, below the "
                f"{MIN_CLOUD_FOR_PLANES} a plane fit needs",
            )

    if max_points is not None and len(scan.points) > max_points:
        before = len(scan.points)
        keep = voxel_subsample_indices(scan.points.xyz, max_points)
        scan.points = scan.points.subset(keep)
        scan.warnings.append(
            f"cloud subsampled from {before} to {len(scan.points)} points by voxel "
            f"stride to honour max_points={max_points}; density statistics no longer "
            "reflect the source file"
        )
        scan.raw_header["subsampled_from"] = before
    return scan


def _dispatch(path: Path, fmt: str) -> ScanImport:
    """Route to the reader for `fmt`, falling back to trimesh for a broken PLY.

    The fallback exists because a PLY that our parser rejects is usually one
    with a vendor extension we have not seen; getting geometry out of it with a
    warning beats refusing the import, as long as the warning says which reader
    actually ran.
    """
    if fmt == "ply":
        from . import ply

        try:
            return ply.read_ply(path)
        except (ValueError, KeyError, IndexError, struct.error) as exc:
            from . import gltf

            try:
                scan = gltf.read_with_trimesh(path, "ply")
            except Exception as fallback_exc:  # noqa: BLE001 - trimesh raises freely
                raise ValueError(
                    f"{path.name}: not a readable PLY. Our parser said {exc}; "
                    f"trimesh said {fallback_exc}"
                ) from exc
            scan.warnings.insert(
                0, f"the first-party PLY reader failed ({exc}); fell back to trimesh"
            )
            return scan
    if fmt == "obj":
        from . import obj

        return obj.read_obj(path)
    if fmt in ("glb", "gltf", "stl", "dae", "off"):
        from . import gltf

        return gltf.read_with_trimesh(path, fmt)
    if fmt == "xyz":
        return _read_xyz(path)
    if fmt in EXTENSION_MAP.values() or fmt == "unknown":
        raise ValueError(
            f"{path.name}: format {fmt!r} is recognised but not readable; "
            f"supported formats are {sorted(READABLE)}"
        )
    raise ValueError(f"{path.name}: unsupported format {fmt!r}")


# ---------------------------------------------------------------------------
# plain text point lists
# ---------------------------------------------------------------------------


def _read_xyz(path: Path) -> ScanImport:
    """Read a bare XYZ/PTS/ASC/CSV point list.

    These files have no header to speak of, so the column layout is inferred
    from the width: the widths that occur in the wild are 3 (xyz), 4 (xyz plus
    intensity), 6 (xyz rgb), 7 (the PTS layout, xyz intensity rgb) and 9 (xyz
    rgb normals). Anything else keeps only the first three columns and says so.
    """
    warnings: list[str] = []
    text = path.read_bytes().decode("utf-8", errors="replace")
    lines = [
        ln for ln in text.splitlines()
        if ln.strip() and not ln.lstrip().startswith(("#", "//", ";"))
    ]
    if not lines:
        raise ValueError(f"{path.name}: no data rows")
    if len(lines[0].split()) == 1:
        # PTS starts with a bare point count
        lines = lines[1:]
    delimiter = "," if lines[0].count(",") >= 2 else None
    try:
        float(lines[0].replace(",", " ").split()[0])
    except ValueError:
        warnings.append("first row is not numeric; treated as a column header")
        lines = lines[1:]
    table = np.loadtxt(
        io.StringIO("\n".join(lines)), delimiter=delimiter, dtype=np.float64, ndmin=2
    )
    width = table.shape[1]
    xyz = table[:, :3]
    rgb = None
    normals = None
    if width == 6:
        rgb = table[:, 3:6]
    elif width == 7:
        rgb = table[:, 4:7]
    elif width == 9:
        rgb = table[:, 3:6]
        normals = table[:, 6:9]
    elif width not in (3, 4):
        warnings.append(f"{width} columns in the point list; kept the first three as xyz")
    if rgb is not None:
        peak = float(np.nanmax(np.abs(rgb))) if rgb.size else 0.0
        if peak <= 1.0 + 1e-6:
            warnings.append("point colours are floats in 0..1; scaled to 0..255")
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return ScanImport(
        points=PointCloud(xyz=xyz, rgb=rgb, normals=normals),
        mesh=None,
        source_path=path,
        source_format="xyz",
        software=None,
        camera_positions=None,
        unit_hint=None,
        warnings=warnings,
        raw_header={"columns": int(width), "rows": int(len(table))},
    )


# ---------------------------------------------------------------------------
# mesh sampling and subsampling
# ---------------------------------------------------------------------------


def _sampling_target(mesh: Mesh) -> int:
    """How many surface samples this mesh's shape deserves, independent of unit.

    Area divided by the squared bounding-box diagonal is dimensionless, so a
    room exported in centimetres asks for exactly as many samples as the same
    room in metres -- which matters because nothing has inferred the unit yet.
    """
    extent = mesh.bounds[1] - mesh.bounds[0]
    diagonal = float(np.linalg.norm(extent))
    area = mesh.area
    if diagonal <= 0 or area <= 0:
        return 0
    normalised = area / (diagonal * diagonal)
    return int(np.clip(MESH_SAMPLE_SCALE * normalised, MESH_SAMPLE_MIN, MESH_SAMPLE_MAX))


def _sample_mesh_into(
    scan: ScanImport, target: int, max_points: int | None, reason: str
) -> None:
    assert scan.mesh is not None
    count = target if max_points is None else min(target, max_points)
    if count <= 0:
        scan.warnings.append("mesh has no area to sample; the cloud stays as it is")
        return
    scan.points = scan.mesh.sample_surface(count, seed=SAMPLE_SEED)
    scan.warnings.append(
        f"{reason}; sampled {count} points off the surface instead "
        f"(seed {SAMPLE_SEED}, so the result is reproducible)"
    )
    scan.raw_header["points_sampled_from_mesh"] = count


def voxel_subsample_indices(
    xyz: np.ndarray, max_points: int, *, probe_limit: int = 400_000
) -> np.ndarray:
    """Indices of one point per voxel, at the coarsest voxel that fits the budget.

    Deterministic: the surviving point in each voxel is the one that appeared
    first in the file, and the returned indices are in file order, so the same
    export always yields the same cloud.

    The voxel size is found by bisection on a strided probe of the cloud rather
    than the whole thing. At the coarse voxel sizes this search operates in,
    occupancy is saturated -- a tenth of the points already touch nearly all of
    the occupied voxels -- so the probe gives the right answer for a twentieth
    of the work, and the final pass over the full cloud is done once.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    n = len(xyz)
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if n <= max_points:
        return np.arange(n)

    lo = xyz.min(axis=0)
    extent = xyz.max(axis=0) - lo
    diagonal = float(np.linalg.norm(extent))
    if diagonal <= 0:
        return np.arange(max_points)

    probe = xyz[:: max(1, n // probe_limit)]
    v_fine, v_coarse = diagonal / 8192.0, diagonal
    best = v_coarse
    if _occupancy(probe, lo, v_fine) <= max_points:
        best = v_fine
    else:
        for _ in range(24):
            mid = float(np.sqrt(v_fine * v_coarse))
            if _occupancy(probe, lo, mid) <= max_points:
                best, v_coarse = mid, mid
            else:
                v_fine = mid
            if v_coarse / v_fine < 1.02:
                break

    keys = _voxel_keys(xyz, lo, best)
    _, first = np.unique(keys, return_index=True)
    keep = np.sort(first)
    if len(keep) > max_points:
        # the probe under-counted; trim the overshoot without re-voxelising the
        # whole cloud, evenly over the kept set so the room stays whole
        pick = np.unique(np.linspace(0, len(keep) - 1, max_points).round().astype(np.int64))
        keep = keep[pick]
    return keep


def _voxel_keys(xyz: np.ndarray, lo: np.ndarray, voxel: float) -> np.ndarray:
    idx = np.floor((xyz - lo) / voxel).astype(np.int64)
    dims = idx.max(axis=0) - idx.min(axis=0) + 1
    idx -= idx.min(axis=0)
    return (idx[:, 0] * int(dims[1]) + idx[:, 1]) * int(dims[2]) + idx[:, 2]


def _occupancy(xyz: np.ndarray, lo: np.ndarray, voxel: float) -> int:
    return int(len(np.unique(_voxel_keys(xyz, lo, voxel))))


# Re-exported so that callers (and the round-trip tests) can write a cloud back
# out without reaching into the reader modules.
def write_ply(*args: Any, **kwargs: Any) -> Path:
    """See `locaish.formats.ply.write_ply`."""
    from .ply import write_ply as _write_ply

    return _write_ply(*args, **kwargs)
