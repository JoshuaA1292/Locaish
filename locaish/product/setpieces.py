"""The room becomes a set: a labelled object lifts into a solid, movable piece.

A location is scouted as it stands; a scene is shot in a room the art
department has rearranged. The question a producer asks of a twin --
"if the sofa goes under the window, does the 50 mm wide hold?" -- is a
question about the room with the sofa moved, and every map, sweep and shot
table this product makes is derived from the point cloud. So moving a piece
of furniture is moving its points, and everything downstream recomputes from
the moved cloud as if the scan had found the room that way.

What a set piece *is* keeps the twin honest: its geometry is exactly the
measured points the scan put there, grown from the label's position outward
through neighbouring returns and stopped at the fitted architecture (the
floor, the ceiling, the walls), because the sofa ends where the wall begins.
Nothing is modelled, fitted or invented; a piece with too few returns to be
solid falls back to a catalogue box for its kind, says so in `source`, and is
drawn but never enters a measurement. When SAM 3D Objects later supplies a
textured gaussian splat for the piece, that splat is attached as a *view*
layer and rides the same pose; the structure still comes from the points.

Store layout, beside `labels.json` in the job workdir:

    setpieces.json                 -- the pieces, pose and footprint; no arrays
    setpieces/<id>.npy             -- bool mask over twin.points (which points)
    setpieces/<id>_gaussians.npz   -- optional: xyz (N,3), rgb (N,3) float 0-1,
                                      opacity (N,), cov (N,3,3); twin frame,
                                      at the UNMOVED position
"""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..types import Opening, PointCloud, Twin
from ..film import render
from ..film import optics

#: Region growing starts with every candidate return this close to the label.
SEED_RADIUS_M = 0.35

#: Two returns closer than this many median nearest-neighbour spacings are
#: the same surface. Not 2.5: for a uniformly sampled surface that radius
#: gives each return about 4.9 neighbours, which is the continuum
#: percolation threshold (4.5), and a face then falls into fragments at
#: random. Four spacings gives about 12.6 and a face stays in one piece.
GROW_SPACING_MULT = 4.0
GROW_RADIUS_MIN_M = 0.03
GROW_RADIUS_MAX_M = 0.12

#: A return this close to a fitted architectural plane is the architecture.
PLANE_TOL_M = 0.06

#: A return this close to the footprint outline is wall, fitted plane or not.
OUTLINE_TOL_M = 0.08

#: An opening's reveal -- jambs, head and sill -- is a strip this wide
#: running from the wall plane into the room as far as the returns stay
#: contiguous with the wall, up to this depth. Furniture is pushed up
#: against door frames all the time and must not take the frame with it,
#: and the frame must not take the furniture's face with it either.
REVEAL_STRIP_M = 0.05
REVEAL_DEPTH_MAX_M = 0.30
REVEAL_DEPTH_MIN_M = 0.04
REVEAL_BIN_M = 0.01
#: Range noise smears the frame's end over a couple of centimetres; the
#: reveal is carried that much farther so the smear does not grow with the
#: furniture. A face closer than that to a door frame is touching it.
REVEAL_TAIL_M = 0.02

#: Furniture stops here; above it is the ceiling, light fittings and beams.
MAX_HEIGHT_M = 2.2

#: No single piece of furniture is bigger than this; a cluster that is has
#: run into its neighbours and is clipped around the click.
MAX_EXTENT_M = 3.5

#: Fewer returns than this is not a solid thing to move; the catalogue speaks.
MIN_POINTS = 200

#: A cluster whose lowest return sits this close to the floor stands on it,
#: and its height is measured from the floor rather than from its lowest
#: unexcluded return (the bottom slice is lost to the floor plane).
ON_FLOOR_M = 0.15

#: Catalogue footprints (width, depth, height) in metres, by kind or label.
CATALOGUE: dict[str, tuple[float, float, float]] = {
    "sofa": (2.0, 0.9, 0.85),
    "couch": (2.0, 0.9, 0.85),
    "table": (1.4, 0.8, 0.75),
    "desk": (1.4, 0.8, 0.75),
    "chair": (0.5, 0.5, 0.9),
    "stool": (0.5, 0.5, 0.9),
    "fridge": (0.7, 0.7, 1.8),
    "refrigerator": (0.7, 0.7, 1.8),
    "bed": (2.0, 1.5, 0.6),
}
CATALOGUE_DEFAULT = (0.8, 0.8, 0.9)

_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# the piece
# ---------------------------------------------------------------------------


@dataclass
class Footprint:
    """The oriented bounding box of the piece's returns, seen from above.

    `yaw_deg` is the heading of the box's long axis from +X; `width_m` runs
    along that axis and `depth_m` across it. `z_base` is where the piece's
    lowest return sits (floor_z for anything standing on the floor).
    """

    cx: float
    cy: float
    width_m: float
    depth_m: float
    height_m: float
    yaw_deg: float = 0.0
    z_base: float = 0.0

    @property
    def centre(self) -> np.ndarray:
        return np.array([self.cx, self.cy], dtype=np.float64)

    def dims_sorted(self) -> tuple[float, float, float]:
        return tuple(sorted((float(self.width_m), float(self.depth_m), float(self.height_m))))

    def to_dict(self) -> dict[str, float]:
        # Unrounded: the footprint centre is the pivot every move turns about,
        # and a reloaded piece must land its points exactly where the live
        # one did.
        return {k: float(v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Footprint":
        return cls(**{k: float(d.get(k, 0.0)) for k in
                      ("cx", "cy", "width_m", "depth_m", "height_m", "yaw_deg", "z_base")})


@dataclass
class SetPiece:
    id: str
    label: str
    kind: str
    anchor: np.ndarray                      # the label's xyz
    footprint: Footprint
    point_mask: np.ndarray | None = None    # bool over twin.points; None for a catalogue default
    pose: dict[str, float] = field(default_factory=lambda: {"dx": 0.0, "dy": 0.0, "dyaw_deg": 0.0})
    source: str = "scan"                    # scan | sam3d | catalogue-default
    gaussians_path: str | None = None       # relative to the workdir
    label_id: str | None = None
    point_count: int = 0
    notes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        self.anchor = np.asarray(self.anchor, dtype=np.float64).reshape(3)
        if self.point_mask is not None:
            self.point_mask = np.asarray(self.point_mask, dtype=bool).reshape(-1)
            self.point_count = int(self.point_mask.sum())

    @property
    def moved(self) -> bool:
        p = self.pose
        return bool(abs(p["dx"]) > 1e-9 or abs(p["dy"]) > 1e-9 or abs(p["dyaw_deg"]) > 1e-9)

    @property
    def has_points(self) -> bool:
        return self.point_mask is not None and self.point_count > 0

    def mask_name(self) -> str:
        return f"setpieces/{self.id}.npy"

    def transform(self) -> np.ndarray:
        """The 4x4 that carries the piece from the scan to its pose: rotation
        about Z through the footprint centre, then the translation."""
        yaw = math.radians(float(self.pose["dyaw_deg"]))
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        centre = np.array([self.footprint.cx, self.footprint.cy, 0.0])
        t = centre - R @ centre + np.array([self.pose["dx"], self.pose["dy"], 0.0])
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = t
        return M

    def footprint_now(self) -> Footprint:
        """The footprint where the piece stands after its pose."""
        fp = self.footprint
        return replace(
            fp,
            cx=fp.cx + float(self.pose["dx"]),
            cy=fp.cy + float(self.pose["dy"]),
            yaw_deg=_wrap_axis_deg(fp.yaw_deg + float(self.pose["dyaw_deg"])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "label_id": self.label_id,
            "anchor": [round(float(v), 6) for v in self.anchor],
            "footprint": self.footprint.to_dict(),
            "pose": {k: round(float(self.pose[k]), 6) for k in ("dx", "dy", "dyaw_deg")},
            "source": self.source,
            "mask_path": self.mask_name() if self.point_mask is not None else None,
            "point_count": int(self.point_count),
            "gaussians_path": self.gaussians_path,
            "notes": list(self.notes),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], point_mask: np.ndarray | None = None) -> "SetPiece":
        pose = d.get("pose") or {}
        piece = cls(
            id=str(d["id"]),
            label=str(d.get("label") or ""),
            kind=str(d.get("kind") or "other"),
            anchor=np.asarray(d.get("anchor") or [0.0, 0.0, 0.0], dtype=np.float64),
            footprint=Footprint.from_dict(d.get("footprint") or {}),
            point_mask=point_mask,
            pose={"dx": float(pose.get("dx", 0.0)), "dy": float(pose.get("dy", 0.0)),
                  "dyaw_deg": float(pose.get("dyaw_deg", 0.0))},
            source=str(d.get("source") or "scan"),
            gaussians_path=d.get("gaussians_path"),
            label_id=d.get("label_id"),
            notes=list(d.get("notes") or []),
            created_at=str(d.get("created_at") or _now()),
        )
        if point_mask is None:
            piece.point_count = int(d.get("point_count") or 0)
        return piece


def _wrap_deg(a: float) -> float:
    return float((a + 180.0) % 360.0 - 180.0)


def _wrap_axis_deg(a: float) -> float:
    """A heading of a box's long axis is the same heading turned 180."""
    return float((a + 90.0) % 180.0 - 90.0)


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------


def pieces_path(workdir: Path) -> Path:
    return Path(workdir) / "setpieces.json"


def load_pieces(workdir: Path) -> list[SetPiece]:
    workdir = Path(workdir)
    try:
        d = json.loads(pieces_path(workdir).read_text())
    except (OSError, ValueError):
        return []
    out: list[SetPiece] = []
    for rec in d.get("pieces") or []:
        mask = None
        mp = rec.get("mask_path")
        if mp:
            try:
                mask = np.load(workdir / mp).astype(bool)
            except (OSError, ValueError):
                mask = None
        out.append(SetPiece.from_dict(rec, mask))
    return out


def save_pieces(workdir: Path, pieces: list[SetPiece]) -> None:
    workdir = Path(workdir)
    (workdir / "setpieces").mkdir(parents=True, exist_ok=True)
    for p in pieces:
        if p.point_mask is not None:
            np.save(workdir / p.mask_name(), p.point_mask)
    path = pieces_path(workdir)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"pieces": [p.to_dict() for p in pieces], "updated_at": _now()}, indent=2))
    os.replace(tmp, path)


def upsert_piece(workdir: Path, piece: SetPiece) -> list[SetPiece]:
    with _LOCK:
        pieces = [p for p in load_pieces(workdir) if p.id != piece.id]
        pieces.append(piece)
        save_pieces(workdir, pieces)
    return pieces


def remove_piece(workdir: Path, piece_id: str) -> bool:
    workdir = Path(workdir)
    with _LOCK:
        pieces = load_pieces(workdir)
        kept = [p for p in pieces if p.id != piece_id]
        if len(kept) == len(pieces):
            return False
        save_pieces(workdir, kept)
        for name in (f"setpieces/{piece_id}.npy", f"setpieces/{piece_id}_gaussians.npz"):
            try:
                (workdir / name).unlink()
            except OSError:
                pass
        return True


def attach_gaussians(piece: SetPiece, workdir: Path, npz_path: str | Path) -> SetPiece:
    """Attach a SAM 3D Objects splat (already in the twin frame at the
    unmoved position) as the piece's view layer. Copies it into the store."""
    workdir = Path(workdir)
    src = Path(npz_path)
    with np.load(src) as z:
        xyz = np.asarray(z["xyz"], dtype=np.float64).reshape(-1, 3)
        rgb = np.asarray(z["rgb"], dtype=np.float32).reshape(-1, 3)
        opacity = np.asarray(z["opacity"], dtype=np.float32).reshape(-1)
        cov = np.asarray(z["cov"], dtype=np.float32).reshape(-1, 3, 3)
    n = len(xyz)
    if not (len(rgb) == len(opacity) == len(cov) == n):
        raise ValueError("gaussians npz: xyz, rgb, opacity and cov disagree on N")
    rel = f"setpieces/{piece.id}_gaussians.npz"
    (workdir / "setpieces").mkdir(parents=True, exist_ok=True)
    np.savez(workdir / rel, xyz=xyz, rgb=rgb, opacity=opacity, cov=cov)
    piece.gaussians_path = rel
    if piece.source != "catalogue-default":
        piece.source = "sam3d"
    return piece


# ---------------------------------------------------------------------------
# lifting a label into a piece
# ---------------------------------------------------------------------------


def median_spacing(xyz: np.ndarray, *, sample: int = 20_000, seed: int = 0) -> float:
    """Median nearest-neighbour distance, from a sample of the cloud."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(xyz) < 2:
        return 0.02
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(xyz), size=min(sample, len(xyz)), replace=False)
    d, _ = cKDTree(xyz).query(xyz[pick], k=2)
    return float(np.median(d[:, 1]))


def architecture_mask(twin: Twin, xyz: np.ndarray | None = None, *,
                      plane_tol: float = PLANE_TOL_M, outline_tol: float = OUTLINE_TOL_M) -> np.ndarray:
    """Which returns belong to the room rather than to its contents.

    A return within `plane_tol` of the floor, the ceiling or a wall plane that
    carries the room outline is architecture, and so is the reveal of every
    opening (its jambs, head and sill). The room solve fits planes to
    furniture faces too (a wardrobe front is as planar as a wall), so a wall
    plane only counts when a footprint edge lies in it; without a footprint,
    when it is big enough to be a wall. Returns hugging the footprint outline
    count as wall whether or not a plane was fitted there.
    """
    pts = np.asarray(twin.points.xyz if xyz is None else xyz, dtype=np.float64).reshape(-1, 3)
    out = np.zeros(len(pts), dtype=bool)
    st = twin.structure
    for plane in st.planes:
        if plane.kind in ("floor", "ceiling") or _is_room_wall(plane, st):
            out |= np.abs(plane.signed_distance(pts)) < plane_tol
    poly = st.footprint
    if poly is not None and len(poly) >= 3:
        out |= _outline_distance(pts[:, :2], poly) < outline_tol
    for o in st.openings:
        # The reveal's depth is read off the whole cloud, so that asking about
        # a subset of returns gets the same answer as asking about all of them.
        depth = _reveal_depth(o, np.asarray(twin.points.xyz, dtype=np.float64))
        out |= _reveal_mask(o, pts, depth)
    return out


def _reveal_frame(o: Opening, pts: np.ndarray):
    """(depth into the room, lateral offset, rise) of returns about an
    opening's centre, and the strip mask of its jambs, head and sill. None
    when the opening has no usable normal."""
    n = o.normal
    if not np.isfinite(n).all() or np.linalg.norm(n) < 0.5:
        return None
    lat = np.cross(n, [0.0, 0.0, 1.0])
    if np.linalg.norm(lat) < 1e-6:
        return None
    lat = lat / np.linalg.norm(lat)
    rel = pts - o.center[None, :]
    depth = rel @ n
    side = rel @ lat
    up = rel[:, 2]
    span_w = np.abs(side) <= o.width / 2.0 + REVEAL_STRIP_M
    span_h = np.abs(up) <= o.height / 2.0 + REVEAL_STRIP_M
    jamb = (np.abs(np.abs(side) - o.width / 2.0) < REVEAL_STRIP_M) & span_h
    head_sill = (np.abs(np.abs(up) - o.height / 2.0) < REVEAL_STRIP_M) & span_w
    return depth, jamb | head_sill


def _reveal_depth(o: Opening, pts: np.ndarray) -> float:
    """How far the opening's frame runs into the room: walk out from the wall
    in bins while the strip keeps returning; the first empty bin is where the
    frame ends, and whatever stands beyond it is contents."""
    fr = _reveal_frame(o, pts)
    if fr is None:
        return REVEAL_DEPTH_MIN_M
    depth, strip = fr
    d = depth[strip & (depth > -REVEAL_STRIP_M) & (depth < REVEAL_DEPTH_MAX_M)]
    reveal = REVEAL_DEPTH_MIN_M
    if len(d):
        edges = np.arange(0.0, REVEAL_DEPTH_MAX_M + REVEAL_BIN_M, REVEAL_BIN_M)
        counts, _ = np.histogram(d, bins=edges)
        # The frame's own return density, read off its first few centimetres,
        # sets the bar: a bin that falls well below it is the frame's end,
        # whatever stands a little farther out (a sofa's top crossing the
        # strip returns a trickle, not a frame).
        bar = max(2.0, 0.25 * float(np.mean(counts[:4])))
        for i, c in enumerate(counts):
            if c < bar:
                break
            reveal = max(reveal, float(edges[i + 1]))
    return reveal + REVEAL_TAIL_M


def _reveal_mask(o: Opening, pts: np.ndarray, reveal: float) -> np.ndarray:
    """Returns on the jambs, head and sill of an opening, out to `reveal`."""
    fr = _reveal_frame(o, pts)
    if fr is None:
        return np.zeros(len(pts), dtype=bool)
    depth, strip = fr
    return strip & (depth > -REVEAL_STRIP_M) & (depth < reveal)


def _is_room_wall(plane, structure) -> bool:
    if plane.kind != "wall":
        return False
    poly = structure.footprint
    if poly is None or len(poly) < 3:
        return float(plane.area) >= 3.0
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if np.linalg.norm(b - a) < 0.5:
            continue
        ends = np.array([[a[0], a[1], structure.floor_z + 1.0], [b[0], b[1], structure.floor_z + 1.0]])
        if np.all(np.abs(plane.signed_distance(ends)) < 0.2):
            return True
    return False


def _outline_distance(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly, dtype=np.float64)
    best = np.full(len(xy), np.inf)
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            d = np.linalg.norm(xy - a, axis=1)
        else:
            t = np.clip(((xy - a) @ ab) / L2, 0.0, 1.0)
            d = np.linalg.norm(xy - (a + t[:, None] * ab), axis=1)
        best = np.minimum(best, d)
    return best


def grow_from(twin: Twin, anchor, *, seed_radius: float = SEED_RADIUS_M,
              grow_radius: float | None = None) -> tuple[np.ndarray, list[str]]:
    """Region-grow the returns of one object outward from `anchor`.

    Candidates are measured returns below `MAX_HEIGHT_M`, within `MAX_EXTENT_M`
    of the anchor and not architecture. Two candidates closer than the grow
    radius are connected; the piece is every candidate connected to one that
    lies within `seed_radius` of the anchor. Returns (mask over twin.points,
    notes).
    """
    anchor = np.asarray(anchor, dtype=np.float64).reshape(3)
    xyz = np.asarray(twin.points.xyz, dtype=np.float64)
    N = len(xyz)
    notes: list[str] = []
    mask = np.zeros(N, dtype=bool)
    if N == 0:
        return mask, ["the twin has no points"]
    floor_z = float(twin.structure.floor_z)

    cand = np.ones(N, dtype=bool)
    if twin.points.inferred is not None:
        cand &= twin.points.inferred <= 0.5
    cand &= (xyz[:, 2] >= floor_z - 0.05) & (xyz[:, 2] <= floor_z + MAX_HEIGHT_M)
    cand &= np.linalg.norm(xyz[:, :2] - anchor[None, :2], axis=1) <= MAX_EXTENT_M
    idx = np.flatnonzero(cand)
    if not len(idx):
        return mask, ["no measured returns near the label"]
    idx = idx[~architecture_mask(twin, xyz[idx])]
    if not len(idx):
        return mask, ["every return near the label belongs to the floor, ceiling or walls"]

    pts = xyz[idx]
    if grow_radius is None:
        grow_radius = float(np.clip(GROW_SPACING_MULT * median_spacing(xyz), GROW_RADIUS_MIN_M, GROW_RADIUS_MAX_M))

    seeds = np.flatnonzero(np.linalg.norm(pts - anchor, axis=1) <= seed_radius)
    if not len(seeds):
        return mask, [f"no object return within {seed_radius:.2f} m of the label"]

    labels = _components(pts, grow_radius)
    chosen = np.isin(labels, np.unique(labels[seeds]))
    picked = idx[chosen]

    # A cluster bigger than any piece of furniture has bridged into its
    # neighbours (a sofa touching a side table touching a lamp). Clip it to a
    # box around the click rather than hand back half the room.
    ext = xyz[picked].max(axis=0) - xyz[picked].min(axis=0)
    if ext[0] > MAX_EXTENT_M or ext[1] > MAX_EXTENT_M:
        near = np.all(np.abs(xyz[picked, :2] - anchor[None, :2]) <= MAX_EXTENT_M / 2.0, axis=1)
        picked = picked[near]
        notes.append(f"cluster ran past {MAX_EXTENT_M:.1f} m and was clipped around the label; "
                     "it was touching its neighbours")
    mask[picked] = True
    return mask, notes


def _components(pts: np.ndarray, radius: float, *, cap: int = 1_500_000) -> np.ndarray:
    """Connected-component labels of a point set under a distance threshold."""
    n = len(pts)
    if n <= cap:
        tree = cKDTree(pts)
        pairs = tree.query_pairs(radius, output_type="ndarray")
        if not len(pairs):
            return np.arange(n)
        g = sparse.coo_matrix((np.ones(len(pairs), dtype=np.uint8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, labels = connected_components(g, directed=False)
        return labels
    # Too many returns for an exact graph: solve on a sample, then give every
    # dropped return the label of its nearest sampled neighbour within reach.
    # Thinning the cloud widens its spacing by sqrt(n / cap), so the radius
    # widens with it, or the sample falls below the percolation threshold
    # that the multiplier was chosen to clear.
    rng = np.random.default_rng(0)
    keep = np.sort(rng.choice(n, size=cap, replace=False))
    r_sample = radius * math.sqrt(n / cap)
    sub = _components(pts[keep], r_sample, cap=cap)
    tree = cKDTree(pts[keep])
    d, j = tree.query(pts, k=1, distance_upper_bound=r_sample)
    labels = np.full(n, -1, dtype=np.int64)
    ok = np.isfinite(d)
    labels[ok] = sub[j[ok]]
    labels[~ok] = sub.max() + 1 + np.arange(int((~ok).sum()))
    return labels


def oriented_footprint(xyz: np.ndarray, floor_z: float) -> Footprint:
    """Oriented bounding box of a cluster seen from above: the minimum-area
    rectangle around its XY convex hull (rotating calipers), so a square
    object reads as a square and not as a diamond's diagonal."""
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    xy = pts[:, :2]
    yaw, u, v = _min_area_axes(xy)
    pu, pv = xy @ u, xy @ v
    width = float(pu.max() - pu.min())
    depth = float(pv.max() - pv.min())
    if depth > width:
        # the long axis names the heading
        width, depth = depth, width
        u, v = v, -u
        pu, pv = xy @ u, xy @ v
        yaw = yaw + 90.0
    yaw = _wrap_axis_deg(yaw)
    centre = u * (pu.max() + pu.min()) / 2.0 + v * (pv.max() + pv.min()) / 2.0
    zmin, zmax = float(pts[:, 2].min()), float(pts[:, 2].max())
    if zmin - floor_z < ON_FLOOR_M:
        z_base, height = float(floor_z), zmax - float(floor_z)
    else:
        z_base, height = zmin, zmax - zmin
    return Footprint(cx=float(centre[0]), cy=float(centre[1]), width_m=width, depth_m=depth,
                     height_m=float(height), yaw_deg=float(yaw), z_base=z_base)


def _min_area_axes(xy: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Heading (deg) and unit axes (u along it, v across) of the smallest
    rectangle enclosing the points. Falls back to the axes when the hull
    degenerates (a line of points)."""
    from scipy.spatial import ConvexHull, QhullError

    best = (float("inf"), 0.0)
    try:
        hull = ConvexHull(xy)
        h = xy[hull.vertices]
    except (QhullError, ValueError):
        h = None
    if h is not None and len(h) >= 3:
        edges = np.roll(h, -1, axis=0) - h
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        for ang in np.unique(np.round(angles, 6)):
            c, s = math.cos(ang), math.sin(ang)
            u = np.array([c, s])
            v = np.array([-s, c])
            pu, pv = h @ u, h @ v
            area = (pu.max() - pu.min()) * (pv.max() - pv.min())
            if area < best[0] - 1e-12:
                best = (area, float(ang))
    ang = best[1] if np.isfinite(best[0]) else 0.0
    yaw = _wrap_deg(math.degrees(ang))
    rad = math.radians(yaw)
    return yaw, np.array([math.cos(rad), math.sin(rad)]), np.array([-math.sin(rad), math.cos(rad)])


def catalogue_size(label: str, kind: str) -> tuple[float, float, float]:
    words = set((label or "").lower().split())
    for key, dims in CATALOGUE.items():
        if key in words:
            return dims
    return CATALOGUE_DEFAULT


def lift_from_label(twin: Twin, label: dict, workdir: Path | None = None, *, save: bool = True) -> SetPiece:
    """A label becomes a set piece: its returns grown from the label's spot.

    When fewer than `MIN_POINTS` returns grow, the piece is a catalogue box
    for its kind, centred on the label, with `source="catalogue-default"`;
    it is drawn and door-checked but has no points to move, so it never
    changes a map or a sweep.
    """
    anchor = np.asarray(label["xyz"], dtype=np.float64).reshape(3)
    name = str(label.get("label") or "object")
    kind = str(label.get("kind") or "other")
    floor_z = float(twin.structure.floor_z)
    mask, notes = grow_from(twin, anchor)
    count = int(mask.sum())
    if count >= MIN_POINTS:
        fp = oriented_footprint(twin.points.xyz[mask], floor_z)
        piece = SetPiece(id=uuid.uuid4().hex[:8], label=name, kind=kind, anchor=anchor,
                         footprint=fp, point_mask=mask, source="scan",
                         label_id=label.get("id"), notes=notes)
    else:
        w, d, h = catalogue_size(name, kind)
        fp = Footprint(cx=float(anchor[0]), cy=float(anchor[1]), width_m=w, depth_m=d,
                       height_m=h, yaw_deg=0.0, z_base=floor_z)
        notes = notes + [f"only {count} returns grew from the label; size is the catalogue "
                         f"default for a {name}, not a measurement"]
        piece = SetPiece(id=uuid.uuid4().hex[:8], label=name, kind=kind, anchor=anchor,
                         footprint=fp, point_mask=None, source="catalogue-default",
                         label_id=label.get("id"), notes=notes)
    if save and workdir is not None:
        upsert_piece(workdir, piece)
    return piece


# ---------------------------------------------------------------------------
# moving
# ---------------------------------------------------------------------------


def move(piece: SetPiece, dx: float, dy: float, dyaw_deg: float = 0.0, *, relative: bool = False) -> SetPiece:
    """Set the piece's pose: an offset from where the scan found it, in the
    twin frame, plus a turn about its own footprint centre. `relative` adds
    to the current pose instead."""
    if relative:
        piece.pose = {"dx": piece.pose["dx"] + float(dx), "dy": piece.pose["dy"] + float(dy),
                      "dyaw_deg": _wrap_deg(piece.pose["dyaw_deg"] + float(dyaw_deg))}
    else:
        piece.pose = {"dx": float(dx), "dy": float(dy), "dyaw_deg": _wrap_deg(float(dyaw_deg))}
    return piece


def reset(piece: SetPiece) -> SetPiece:
    piece.pose = {"dx": 0.0, "dy": 0.0, "dyaw_deg": 0.0}
    return piece


def apply(twin: Twin, pieces: list[SetPiece]) -> Twin:
    """The twin as the room would scan with every piece where its pose puts it.

    A new Twin: the cloud is copied with each moved piece's returns carried
    by its transform; mesh, structure, bounds and provenance are the same
    objects. Floor maps, occupancy and the sweep are then ordinary calls on
    the result. Pieces without points (catalogue defaults) change nothing.
    """
    src = twin.points
    xyz = np.array(src.xyz, dtype=np.float64, copy=True)
    normals = None if src.normals is None else np.array(src.normals, dtype=np.float64, copy=True)
    n = len(xyz)
    for p in pieces:
        if not p.moved or not p.has_points:
            continue
        if len(p.point_mask) != n:
            raise ValueError(f"set piece {p.id}: mask has {len(p.point_mask)} entries, twin has {n} points")
        M = p.transform()
        sel = p.point_mask
        xyz[sel] = xyz[sel] @ M[:3, :3].T + M[:3, 3]
        if normals is not None:
            normals[sel] = normals[sel] @ M[:3, :3].T
    cloud = PointCloud(xyz=xyz, rgb=None if src.rgb is None else src.rgb.copy(),
                       normals=normals, inferred=None if src.inferred is None else src.inferred.copy())
    return replace(twin, points=cloud)


def exclusion_mask(twin: Twin, pieces: list[SetPiece], *, moved_only: bool = True) -> np.ndarray:
    """Points of the original cloud the renderer should leave out: a moved
    piece's returns where the scan found them (the figure draws them at the
    pose). With `moved_only=False` every lifted piece is excluded, for a
    frame that composes all pieces from `figure`."""
    out = np.zeros(len(twin.points), dtype=bool)
    for p in pieces:
        if not p.has_points or len(p.point_mask) != len(out):
            continue
        if p.moved or not moved_only:
            out |= p.point_mask
    return out


def figure(piece: SetPiece, twin: Twin, workdir: Path | None = None) -> render.Figure:
    """The piece as the renderer draws it, at its pose.

    The attached gaussians when there are any (positions and covariances
    carried by the pose), else the piece's own returns as small opaque
    gaussians, else -- for a catalogue default -- a grey sampled box so the
    frame shows the stand-in and the label says what it is.
    """
    M = piece.transform()
    R, t = M[:3, :3], M[:3, 3]
    if piece.gaussians_path and workdir is not None and (Path(workdir) / piece.gaussians_path).exists():
        with np.load(Path(workdir) / piece.gaussians_path) as z:
            xyz = np.asarray(z["xyz"], dtype=np.float64).reshape(-1, 3) @ R.T + t
            rgb = np.asarray(z["rgb"], dtype=np.float32).reshape(-1, 3)
            opacity = np.asarray(z["opacity"], dtype=np.float32).reshape(-1)
            cov = np.asarray(z["cov"], dtype=np.float32).reshape(-1, 3, 3)
        cov = np.einsum("ij,njk,lk->nil", R, cov, R).astype(np.float32)
        return render.Figure(xyz, rgb, cov=cov, opacity=opacity, label=f"{piece.label} (sam3d)")
    if piece.has_points and len(piece.point_mask) == len(twin.points):
        xyz = np.asarray(twin.points.xyz, dtype=np.float64)[piece.point_mask] @ R.T + t
        rgb = (twin.points.rgb[piece.point_mask] if twin.points.rgb is not None
               else np.full((len(xyz), 3), 170, dtype=np.uint8))
        radius = float(np.clip(median_spacing(twin.points.xyz, sample=5000), 0.006, 0.03))
        return render.Figure(xyz, rgb, radius_m=radius, label=piece.label)
    fp = piece.footprint_now()
    xyz = _box_points(fp)
    return render.Figure(xyz, np.array([150, 150, 150], dtype=np.uint8), radius_m=0.02,
                         label=f"{piece.label} (catalogue default)")


def _box_points(fp: Footprint, step: float = 0.05) -> np.ndarray:
    """Surface samples of a footprint box, for drawing a catalogue stand-in."""
    w, d, h = fp.width_m, fp.depth_m, fp.height_m
    us = np.arange(-w / 2.0, w / 2.0 + 1e-9, step)
    vs = np.arange(-d / 2.0, d / 2.0 + 1e-9, step)
    zs = np.arange(0.0, h + 1e-9, step)
    faces = []
    U, V = np.meshgrid(us, vs, indexing="ij")
    faces.append(np.stack([U.ravel(), V.ravel(), np.full(U.size, h)], axis=1))
    U, Z = np.meshgrid(us, zs, indexing="ij")
    for vv in (-d / 2.0, d / 2.0):
        faces.append(np.stack([U.ravel(), np.full(U.size, vv), Z.ravel()], axis=1))
    V, Z = np.meshgrid(vs, zs, indexing="ij")
    for uu in (-w / 2.0, w / 2.0):
        faces.append(np.stack([np.full(V.size, uu), V.ravel(), Z.ravel()], axis=1))
    local = np.concatenate(faces)
    rad = math.radians(fp.yaw_deg)
    c, s = math.cos(rad), math.sin(rad)
    x = fp.cx + local[:, 0] * c - local[:, 1] * s
    y = fp.cy + local[:, 0] * s + local[:, 1] * c
    return np.stack([x, y, fp.z_base + local[:, 2]], axis=1)


# ---------------------------------------------------------------------------
# questions about the piece
# ---------------------------------------------------------------------------


def door_fit(piece: SetPiece, twin: Twin, *, margin_m: float = 0.0) -> dict:
    """Can the piece come through the room's doors? The load-in question.

    A piece passes a door when its two smallest box dimensions fit inside the
    door's width and height, either way up. The clearance is the smaller of
    the two slacks in the better orientation; negative means it does not go.
    """
    dims = piece.footprint.dims_sorted()
    a, b = dims[0], dims[1]
    doors = [o for o in twin.structure.openings if o.kind == "door"]
    verdicts = []
    for i, o in enumerate(doors):
        w, h = float(o.width) - margin_m, float(o.height) - margin_m
        upright = min(w - a, h - b)
        on_side = min(w - b, h - a)
        best = max(upright, on_side)
        verdicts.append({
            "door": i,
            "center": [round(float(v), 3) for v in o.center],
            "width_m": round(float(o.width), 3),
            "height_m": round(float(o.height), 3),
            "fits": bool(best >= 0.0),
            "clearance_m": round(float(best), 3),
            "orientation": "upright" if upright >= on_side else "on its side",
            "confidence": round(float(o.confidence), 2),
        })
    tightest = min((v["clearance_m"] for v in verdicts), default=None)
    return {
        "piece": piece.id,
        "label": piece.label,
        "passes_m": [round(a, 3), round(b, 3)],
        "doors": verdicts,
        "fits_all": None if not verdicts else all(v["fits"] for v in verdicts),
        "fits_any": None if not verdicts else any(v["fits"] for v in verdicts),
        "tightest_clearance_m": tightest,
        "note": None if verdicts else "no door opening in the twin to check against",
    }


def sweep_diff(before, after) -> dict:
    """What a move did to the shot table: clear (visible) setups by shot size
    and by lens, before, after and the change, with sentences for the agent.

    Accepts `ShotSweep` objects or their `columns` dicts; needs `visible`,
    `shot_size` and `focal_mm`.
    """
    b = _columns(before)
    a = _columns(after)
    by_size = _count_diff(b, a, "shot_size")
    by_focal = _count_diff(b, a, "focal_mm")
    pairs = _count_diff(b, a, ("focal_mm", "shot_size"))
    vis_b = int(np.asarray(b["visible"]).astype(bool).sum())
    vis_a = int(np.asarray(a["visible"]).astype(bool).sum())

    sentences: list[str] = []
    ranked = sorted(pairs.items(), key=lambda kv: -abs(kv[1]["delta"]))
    for (focal, size), c in ranked:
        name = _size_name(size)
        lens = f"{float(focal):g} mm"
        if c["before"] == 0 and c["after"] > 0:
            sentences.append(f"a {lens} {name} that had no clear setup now has {c['after']}")
        elif c["after"] == 0 and c["before"] > 0:
            sentences.append(f"a {lens} {name} lost all {c['before']} of its clear setups")
        elif c["before"] and abs(c["delta"]) >= max(5, 0.25 * c["before"]):
            verb = "gained" if c["delta"] > 0 else "lost"
            sentences.append(f"a {lens} {name} {verb} {abs(c['delta'])} clear setups "
                             f"({c['before']} to {c['after']})")
        if len(sentences) >= 6:
            break
    if vis_a == vis_b and not any(c["delta"] for c in pairs.values()):
        sentences.append("the move changed no clear setup count")
    else:
        verb = "gained" if vis_a >= vis_b else "lost"
        sentences.insert(0, f"the room {verb} {abs(vis_a - vis_b)} clear setups overall ({vis_b} to {vis_a})")
        if len(sentences) == 1 and ranked and ranked[0][1]["delta"]:
            (focal, size), c = ranked[0]
            sentences.append(f"the biggest change: a {float(focal):g} mm {_size_name(size)} went from "
                             f"{c['before']} clear setups to {c['after']}")

    return {
        "visible_before": vis_b,
        "visible_after": vis_a,
        "delta": vis_a - vis_b,
        "by_shot_size": {str(k): v for k, v in by_size.items()},
        "by_focal_mm": {f"{float(k):g}": v for k, v in by_focal.items()},
        "by_focal_and_size": [{"focal_mm": float(k[0]), "shot_size": str(k[1]), **v} for k, v in pairs.items()],
        "sentences": sentences,
    }


def _columns(sw) -> dict:
    cols = sw.columns if hasattr(sw, "columns") else sw
    for k in ("visible", "shot_size", "focal_mm"):
        if k not in cols:
            raise KeyError(f"sweep columns lack {k!r}")
    return cols


def _count_diff(b: dict, a: dict, key) -> dict:
    keys = (key,) if isinstance(key, str) else tuple(key)

    def counts(c):
        vis = np.asarray(c["visible"]).astype(bool)
        arrs = [np.asarray(c[k]) for k in keys]
        out: dict = {}
        if not vis.any():
            return out
        if len(arrs) == 1:
            u, n = np.unique(arrs[0][vis], return_counts=True)
            for k, v in zip(u.tolist(), n.tolist()):
                out[_key(k)] = int(v)
        else:
            stacked = np.array(list(zip(*[x[vis].tolist() for x in arrs])), dtype=object)
            seen: dict = {}
            for row in stacked:
                kk = tuple(_key(v) for v in row)
                seen[kk] = seen.get(kk, 0) + 1
            out = seen
        return out

    cb, ca = counts(b), counts(a)
    out = {}
    for k in sorted(set(cb) | set(ca), key=lambda x: str(x)):
        nb, na = cb.get(k, 0), ca.get(k, 0)
        out[k] = {"before": nb, "after": na, "delta": na - nb}
    return out


def _key(v):
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode()
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return str(v)


def _size_name(key) -> str:
    s = optics.SHOT_BY_KEY.get(str(key))
    if s is None:
        return str(key)
    return s.name.lower().replace("extreme ", "extreme-")


def describe(pieces: list[SetPiece], twin: Twin | None = None) -> list[dict]:
    """Pieces as the agent and the package read them: name, measured size,
    where they stand, how far they were moved, and whether they come through
    the door when a twin is given."""
    out = []
    for p in pieces:
        fp, now = p.footprint, p.footprint_now()
        rec: dict[str, Any] = {
            "id": p.id,
            "label": p.label,
            "kind": p.kind,
            "source": p.source,
            "size": f"{fp.width_m:.2f} x {fp.depth_m:.2f} x {fp.height_m:.2f} m",
            "width_m": round(fp.width_m, 3),
            "depth_m": round(fp.depth_m, 3),
            "height_m": round(fp.height_m, 3),
            "measured": p.source != "catalogue-default",
            "points": int(p.point_count),
            "scan_position_m": [round(fp.cx, 3), round(fp.cy, 3)],
            "position_m": [round(now.cx, 3), round(now.cy, 3)],
            "yaw_deg": round(now.yaw_deg, 1),
            "moved": p.moved,
            "moved_by": {"dx_m": round(p.pose["dx"], 3), "dy_m": round(p.pose["dy"], 3),
                         "dyaw_deg": round(p.pose["dyaw_deg"], 1)},
            "moved_text": _moved_text(p),
            "notes": list(p.notes),
        }
        if twin is not None:
            rec["door_fit"] = door_fit(p, twin)
        out.append(rec)
    return out


def _moved_text(p: SetPiece) -> str:
    if not p.moved:
        return "where the scan found it"
    dist = math.hypot(p.pose["dx"], p.pose["dy"])
    parts = []
    if dist > 1e-6:
        parts.append(f"moved {dist:.2f} m")
    if abs(p.pose["dyaw_deg"]) > 1e-6:
        parts.append(f"turned {p.pose['dyaw_deg']:+.0f} deg")
    return ", ".join(parts)


__all__ = [
    "SetPiece", "Footprint", "CATALOGUE", "load_pieces", "save_pieces", "upsert_piece",
    "remove_piece", "attach_gaussians", "lift_from_label", "grow_from", "architecture_mask",
    "oriented_footprint", "move", "reset", "apply", "exclusion_mask", "figure", "door_fit",
    "sweep_diff", "describe", "median_spacing",
]
