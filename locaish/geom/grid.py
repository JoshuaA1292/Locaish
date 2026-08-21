"""Voxel occupancy over a canonical point cloud.

A point cloud answers "is there a surface near here?" only by way of a nearest
neighbour query, which is fine once and ruinous a million times. Almost every
downstream question in Locaish -- can a tripod stand here, is that wall solid or
is it a doorway, how high is the ceiling above this square metre of floor -- is
really a query against a regular volume, so we build the volume once and index
into it afterwards.

The voxel size is deliberately anisotropic. Horizontal resolution is set by how
finely we need to resolve a footprint; vertical resolution is set by the thin
things that decide camera height -- a parapet, a ledge, the head of a door. An
isotropic voxel small enough for the latter costs cubes of memory in the former
direction, so the two are separate parameters and 5 cm is only a default.

The grid is stored dense. That is the right call for a room: an interior at 5 cm
is a few million voxels, well under the point count, and dense arrays make every
downstream query a slice rather than a hash lookup. It is the wrong call for a
city block, which is why `build_grid` refuses absurd allocations loudly instead
of discovering them via the OOM killer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import PointCloud, chunked

# The default interior resolution. 5 cm is below the thickness of anything
# architectural we care about and above the noise floor of a consumer LiDAR
# sweep, so a voxel that is occupied is occupied because of a surface and not
# because of a stray return.
ROOM_VOXEL = 0.05

# Refuse to allocate beyond this many voxels. At one byte for `occupied` and
# four for `counts` this ceiling is already two gigabytes, which is as far as we
# are willing to go before telling the caller to coarsen the voxel instead.
MAX_VOXELS = 400_000_000

# Points are binned in slabs rather than all at once so that a 20M-point sweep
# never materialises more than a few million int64 indices at a time.
_BIN_CHUNK = 4_000_000


@dataclass
class OccupancyGrid:
    """A dense axis-aligned occupancy volume in canonical twin space.

    `occupied` and `counts` are both (nx, ny, nz) and share an indexing
    convention: voxel (i, j, k) covers the half-open box from
    `origin + (i, j, k) * voxel` to `origin + (i+1, j+1, k+1) * voxel`. `origin`
    is the *corner* of voxel (0, 0, 0), not its centre, because corner
    arithmetic is what makes `world_to_index` a single floor division with no
    half-voxel fudge to get wrong.

    `counts` is kept alongside `occupied` because "one stray return" and "four
    hundred returns" are the same bit of occupancy and very different evidence.
    Anything that has to decide whether a voxel is real geometry or scanner
    noise needs the count, not the bit.
    """

    occupied: np.ndarray
    counts: np.ndarray
    origin: np.ndarray
    voxel: np.ndarray

    def __post_init__(self) -> None:
        self.occupied = np.asarray(self.occupied, dtype=bool)
        self.counts = np.asarray(self.counts, dtype=np.int32)
        if self.occupied.shape != self.counts.shape or self.occupied.ndim != 3:
            raise ValueError(
                f"occupied {self.occupied.shape} and counts {self.counts.shape} "
                "must be the same 3D shape"
            )
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(3)
        self.voxel = np.asarray(self.voxel, dtype=np.float64).reshape(3)
        if np.any(self.voxel <= 0):
            raise ValueError(f"voxel sizes must be positive, got {self.voxel.tolist()}")

    # -- shape -------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.occupied.shape[0]), int(self.occupied.shape[1]), int(self.occupied.shape[2]))

    @property
    def dims(self) -> np.ndarray:
        """(3,) int array of voxel counts per axis."""
        return np.array(self.occupied.shape, dtype=np.int64)

    @property
    def bounds(self) -> np.ndarray:
        """(2, 3) world-space [min, max] of the volume the grid covers."""
        return np.stack([self.origin, self.origin + self.dims * self.voxel])

    @property
    def voxel_volume(self) -> float:
        return float(np.prod(self.voxel))

    @property
    def occupied_count(self) -> int:
        return int(self.occupied.sum())

    @property
    def memory_bytes(self) -> int:
        """Bytes actually held by the two volumes.

        Exposed because the honest way to choose a voxel size is to ask what it
        would cost, and callers that are about to build several grids need to be
        able to add that up before they do it.
        """
        return int(self.occupied.nbytes + self.counts.nbytes)

    # -- coordinates -------------------------------------------------------

    def world_to_index(self, xyz: np.ndarray) -> np.ndarray:
        """(N, 3) integer voxel indices. Deliberately unclipped.

        Clipping here would silently map a point outside the volume onto a real
        voxel on the boundary, which is the sort of quiet lie that turns into a
        tripod standing inside a wall. Out-of-range indices come back negative
        or past the end and `contains` is how you find out.
        """
        pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        return np.floor((pts - self.origin) / self.voxel).astype(np.int64)

    def index_to_world(self, ijk: np.ndarray) -> np.ndarray:
        """(N, 3) world coordinates of voxel *centres*.

        Centres rather than corners because every consumer of this -- a height
        map, a clearance query, a marching-cubes seed -- wants the
        representative point of the voxel, and the corner is representative of
        nothing.
        """
        idx = np.asarray(ijk, dtype=np.float64).reshape(-1, 3)
        return self.origin + (idx + 0.5) * self.voxel

    def contains(self, xyz: np.ndarray) -> np.ndarray:
        """(N,) mask of which world points fall inside the grid's extent."""
        idx = self.world_to_index(xyz)
        return self._index_in_range(idx)

    def _index_in_range(self, idx: np.ndarray) -> np.ndarray:
        dims = self.dims
        return np.all((idx >= 0) & (idx < dims), axis=1)

    def occupied_at(self, xyz: np.ndarray) -> np.ndarray:
        """(N,) mask: is the voxel containing each point occupied?

        Points outside the volume are reported unoccupied rather than raising,
        because "there is nothing there" is the truthful answer for a query
        beyond the scanned region and the caller already has `contains` if it
        needs to distinguish empty from unknown.
        """
        idx = self.world_to_index(xyz)
        good = self._index_in_range(idx)
        out = np.zeros(len(idx), dtype=bool)
        if good.any():
            sel = idx[good]
            out[good] = self.occupied[sel[:, 0], sel[:, 1], sel[:, 2]]
        return out

    def column(self, xy: np.ndarray) -> np.ndarray:
        """(nz,) occupancy of the vertical column through one XY position.

        A column is the unit of almost every architectural question: floor to
        ceiling, what is overhead, where the free height is. Off-grid columns
        return all-empty rather than raising so a caller sweeping a rectangle
        that overhangs the scan does not have to guard every sample.
        """
        pt = np.asarray(xy, dtype=np.float64).reshape(2)
        ij = np.floor((pt - self.origin[:2]) / self.voxel[:2]).astype(np.int64)
        nx, ny, nz = self.shape
        if not (0 <= ij[0] < nx and 0 <= ij[1] < ny):
            return np.zeros(nz, dtype=bool)
        return self.occupied[ij[0], ij[1]]

    # -- 2.5D projections --------------------------------------------------

    def height_map(self) -> np.ndarray:
        """(nx, ny) world Z of the topmost occupied voxel centre, NaN if empty.

        NaN rather than zero for an empty column, because zero is a legal
        height -- it is the floor -- and a map that cannot tell "the floor" from
        "we never saw anything here" is worse than no map.
        """
        return self._extreme_map(top=True)

    def floor_map(self) -> np.ndarray:
        """(nx, ny) world Z of the lowest occupied voxel centre, NaN if empty."""
        return self._extreme_map(top=False)

    def _extreme_map(self, *, top: bool) -> np.ndarray:
        nz = self.shape[2]
        any_hit = self.occupied.any(axis=2)
        if top:
            # argmax on the reversed axis finds the first True from the top;
            # argmax over a bool array short-circuits, so this is one pass.
            k = nz - 1 - self.occupied[:, :, ::-1].argmax(axis=2)
        else:
            k = self.occupied.argmax(axis=2)
        z = self.origin[2] + (k.astype(np.float64) + 0.5) * self.voxel[2]
        return np.where(any_hit, z, np.nan)

    def occupied_points(self) -> np.ndarray:
        """(M, 3) centres of every occupied voxel.

        The cheap way to hand a decimated, uniformly spaced version of the cloud
        to something that does not want twenty million points.
        """
        idx = np.argwhere(self.occupied)
        if len(idx) == 0:
            return np.zeros((0, 3))
        return self.index_to_world(idx)


def build_grid(
    cloud: PointCloud,
    *,
    voxel_xy: float = ROOM_VOXEL,
    voxel_z: float = ROOM_VOXEL,
    bounds: np.ndarray | None = None,
    pad: float = 0.1,
) -> OccupancyGrid:
    """Bin a cloud into a dense occupancy volume.

    `bounds` is an optional (2, 3) [min, max] box; when given it is used
    verbatim and `pad` still widens it, which is how callers keep several grids
    of the same room mutually indexable. Points outside the box are dropped
    rather than clamped, so a grid built with explicit bounds never
    accidentally piles the whole outside world onto its boundary voxels.

    `pad` exists mainly for the mesher: an isosurface extracted from a volume
    whose solid touches the array edge comes out with an open boundary there.
    A margin of empty voxels costs nothing and closes it.

    Binning is a `bincount` over flattened indices. `np.add.at` on a 3D array is
    the obvious spelling and is roughly two orders of magnitude slower, which on
    a 20M-point sweep is the difference between a second and a coffee break.
    """
    if voxel_xy <= 0 or voxel_z <= 0:
        raise ValueError(f"voxel sizes must be positive, got xy={voxel_xy}, z={voxel_z}")

    voxel = np.array([voxel_xy, voxel_xy, voxel_z], dtype=np.float64)
    xyz = cloud.xyz

    if bounds is None:
        box = cloud.bounds
    else:
        box = np.asarray(bounds, dtype=np.float64).reshape(2, 3)
        if np.any(box[1] < box[0]):
            raise ValueError("bounds max is below bounds min")
    lo = box[0] - pad
    hi = box[1] + pad

    # A degenerate axis (a perfectly flat cloud, or an empty one) still has to
    # produce a usable grid rather than a zero-length axis that breaks indexing.
    span = np.maximum(hi - lo, voxel)
    dims = np.maximum(np.ceil(span / voxel).astype(np.int64), 1)

    total = int(np.prod(dims))
    if total > MAX_VOXELS:
        raise ValueError(
            f"occupancy grid would be {dims[0]}x{dims[1]}x{dims[2]} = {total:,} voxels "
            f"(limit {MAX_VOXELS:,}) for an extent of "
            f"{span[0]:.2f}x{span[1]:.2f}x{span[2]:.2f} m at voxel "
            f"{voxel_xy:g}x{voxel_xy:g}x{voxel_z:g} m; coarsen the voxel or "
            f"crop the cloud"
        )

    flat_counts = np.zeros(total, dtype=np.int64)
    strides = np.array([dims[1] * dims[2], dims[2], 1], dtype=np.int64)

    for sl in chunked(len(xyz), _BIN_CHUNK):
        idx = np.floor((xyz[sl] - lo) / voxel).astype(np.int64)
        keep = np.all((idx >= 0) & (idx < dims), axis=1)
        if not keep.any():
            continue
        flat = idx[keep] @ strides
        flat_counts += np.bincount(flat, minlength=total)

    counts = np.minimum(flat_counts, np.iinfo(np.int32).max).astype(np.int32)
    counts = counts.reshape(tuple(int(d) for d in dims))
    return OccupancyGrid(
        occupied=counts > 0,
        counts=counts,
        origin=lo,
        voxel=voxel,
    )
