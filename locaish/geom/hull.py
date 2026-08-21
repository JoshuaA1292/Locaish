"""Footprints, and the honest limit of what a scan is entitled to claim.

Two related jobs live here. The first is turning a smear of XY points into a
polygon that follows the shape they actually make -- an L-shaped room is not its
convex hull, and a shot search that believes it is will happily propose a tripod
standing in the neighbour's flat. The second is `capture_bounds`, which answers
the question the rest of the system keeps needing: where was the scanner, and
therefore which part of this twin is measurement rather than extrapolation.

`CaptureBounds` is a safety rail, not a decoration. Outside it the geometry is a
guess dressed up as a surface, so everything here errs deliberately toward the
smaller region. Claiming too little costs a few square metres of usable floor.
Claiming too much puts a camera somewhere the scanner never looked and quietly
turns the twin's central promise into a lie.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import ConvexHull, Delaunay, QhullError, cKDTree
from skimage import measure

from ..types import CaptureBounds, PointCloud

# How far past its own footprint a handheld scanner sees. A phone at arm's
# length with a 60-70 degree LiDAR field of view resolves the floor a good way
# in front of the operator; 0.6 m is deliberately conservative next to that.
POSE_DILATION_M = 0.6

# Thickness of the slab counted as "floor" when there are no poses to go on.
FLOOR_BAND_M = 0.25

# Alpha, when not given, is this multiple of the median Delaunay edge length.
# Below about 1.5 the shape erodes into the point set and starts inventing
# concavities out of sampling noise; far above it the alpha shape converges to
# the convex hull and stops being worth computing.
_ALPHA_EDGE_FACTOR = 2.5

# A fragment of the alpha shape smaller than this is dropped. The threshold is
# absolute rather than a share of the total on purpose: a share turns into a
# cliff, where a fragment one per cent either side of it changes the answer by
# tens of square metres, and where the same capture in a bigger room gets a
# different verdict. A quarter of a square metre is smaller than the footprint
# of a tripod, so nothing the shot search could act on is ever discarded, while
# the 0.000-0.004 m2 specks that occlusion shadows strand behind furniture go.
_MIN_PART_AREA_M2 = 0.25

# An interior void smaller than this is filled in rather than preserved. Voids
# are the opposite direction of error from fragments -- filling one claims floor
# nobody scanned -- so the bar is lower: 0.09 m2 is a 30 cm square, too small to
# stand anything in and far more likely to be a gap between Delaunay triangles
# than a place the scanner could not see.
_MIN_VOID_AREA_M2 = 0.09

# Working resolution for the raster used to dilate a polygon, and the cell
# budget that resolution is allowed to blow before it is coarsened.
_RASTER_CELL_M = 0.03
_RASTER_MAX_CELLS = 6_000_000

# Points handed to Delaunay are first collapsed onto this XY lattice. A 20M
# point sweep has no more boundary information in it than its 2 cm footprint
# does, and triangulating all of it is minutes of work for an identical answer.
_HULL_LATTICE_M = 0.02


# ---------------------------------------------------------------------------
# polygons
# ---------------------------------------------------------------------------


def polygon_area(poly_xy: np.ndarray) -> float:
    """Unsigned shoelace area in square metres. Degenerate rings give zero."""
    p = np.asarray(poly_xy, dtype=np.float64).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    q = np.roll(p, -1, axis=0)
    return float(abs(np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1])) / 2.0)


def signed_area(poly_xy: np.ndarray) -> float:
    """Signed shoelace area: positive for counter-clockwise, negative for
    clockwise. The sign is how we tell an outer boundary from a hole."""
    p = np.asarray(poly_xy, dtype=np.float64).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    q = np.roll(p, -1, axis=0)
    return float(np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]) / 2.0)


def simplify_polygon(poly_xy: np.ndarray, tolerance: float = 0.05) -> np.ndarray:
    """Douglas-Peucker on a closed ring, preserving orientation.

    A raster-derived or alpha-shape-derived boundary has a vertex every few
    centimetres, nearly all of which say nothing: a straight wall does not need
    two hundred points to be a straight wall. The tolerance is a real distance,
    so the guarantee is the one that matters -- no point of the original ring
    ends up more than `tolerance` metres from the simplified one.

    The ring is cut at its two most distant vertices before simplifying rather
    than at an arbitrary index, because Douglas-Peucker pins the endpoints of
    whatever chain it is given and pinning two adjacent noise vertices leaves a
    visible spike in an otherwise clean wall.
    """
    p = np.asarray(poly_xy, dtype=np.float64).reshape(-1, 2)
    if len(p) < 4 or tolerance <= 0:
        return p

    anchor = int(np.argmax(np.linalg.norm(p - p.mean(axis=0), axis=1)))
    p = np.roll(p, -anchor, axis=0)
    far = int(np.argmax(np.linalg.norm(p - p[0], axis=1)))
    if far == 0:
        return p

    first = _douglas_peucker(p[: far + 1], tolerance)
    second = _douglas_peucker(np.concatenate([p[far:], p[:1]]), tolerance)
    out = np.concatenate([first[:-1], second[:-1]])
    if len(out) < 3:
        return p
    return out


def _douglas_peucker(chain: np.ndarray, tolerance: float) -> np.ndarray:
    """Simplify an open polyline, keeping both endpoints.

    Iterative with an explicit stack rather than recursive: a raster contour can
    be tens of thousands of points long and Python's recursion limit is not a
    geometry parameter.
    """
    n = len(chain)
    if n < 3:
        return chain
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        seg = chain[j] - chain[i]
        length = float(np.hypot(seg[0], seg[1]))
        rel = chain[i + 1 : j] - chain[i]
        if length < 1e-12:
            dist = np.linalg.norm(rel, axis=1)
        else:
            dist = np.abs(rel[:, 0] * seg[1] - rel[:, 1] * seg[0]) / length
        k = int(np.argmax(dist))
        if dist[k] > tolerance:
            k += i + 1
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return chain[keep]


# ---------------------------------------------------------------------------
# concave hull
# ---------------------------------------------------------------------------


def concave_hull_xy(points_xy: np.ndarray, *, alpha: float | None = None) -> np.ndarray:
    """Alpha shape of a 2D point set, returned as a single CCW ring.

    A Delaunay triangulation of a point set covers its convex hull. The alpha
    shape is what is left after deleting every triangle whose circumcircle is
    bigger than `alpha`, which is exactly the triangles that span a void the
    points never filled: the notch of an L-shaped room, the gap between two
    separate parts of a capture, the inside of a walked loop.

    `alpha` defaults to a multiple of the median Delaunay edge length so that
    the same call works on a 2 cm-spaced floor slab and a 30 cm-spaced camera
    path without the caller having to know which it has.

    This entry point answers with the outer boundary of the largest piece, which
    is what a caller wanting one simple ring means. Interior voids are filled
    and detached pieces are dropped, both of which are the conservative
    direction for a camera path -- a hole you walked all the way around is an
    artefact of the sampling, and a piece left behind is floor we then decline
    to claim. The one exception is a piece pinched to a single point, whose
    lobes are all kept and joined by a doubled bridge edge, because they are the
    same walked region and dropping one would be arbitrary. Callers that need
    the voids kept or every piece represented want `_concave_hull`, which
    returns them along with a note saying what it did.

    The convex hull is returned only when there is no alpha shape to be had at
    all: Qhull fails, every triangle is bigger than alpha, or the boundary walk
    does not close. It is strictly larger than the truth, so it is a fallback of
    last resort rather than a tidy answer to fragmentation.
    """
    ring, _ = _concave_hull(points_xy, alpha=alpha)
    return ring


def _concave_hull(
    points_xy: np.ndarray,
    *,
    alpha: float | None = None,
    keep_parts: bool = False,
    keep_voids: bool = False,
    simplify_tolerance: float = 0.0,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """The alpha shape, plus a note of anything the single ring had to give up.

    `keep_parts` and `keep_voids` exist because the two callers of this want
    genuinely different things from the same computation. A camera path is one
    walk, and both a second fragment and an interior hole in it are sampling
    artefacts; a floor slab is a record of what was looked at, and there a
    second fragment is a second room and a hole is a patch of floor nobody
    scanned. Getting that backwards is how `contains` ends up returning True in
    the middle of a void, which is the single failure `CaptureBounds` exists to
    prevent.

    Multiple pieces and interior voids both come back inside one vertex array,
    stitched together with doubled bridge edges. That works because `contains`
    is an even-odd crossing test: a bridge traversed once in each direction
    contributes its crossing twice and cancels exactly, so a piece adds its
    interior and a void removes its own. The shoelace area cancels on the
    bridges for the same reason, so `CaptureBounds.area` stays exact.

    The returned notes name every place the answer is not simply "the alpha
    shape of these points": a convex hull fallback, dropped pieces, kept voids.
    They exist so the caller can put them where a human will see them rather
    than leaving an inflated number looking like a measurement.
    """
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    pts = _lattice_unique(pts, _HULL_LATTICE_M)
    if len(pts) < 3:
        return pts, ()

    try:
        tri = Delaunay(pts)
    except (QhullError, ValueError):
        return _convex_ring(pts), ("convex-hull-fallback",)

    simplices = tri.simplices
    if len(simplices) == 0:
        return _convex_ring(pts), ("convex-hull-fallback",)

    # Orient every triangle counter-clockwise so that its directed edges carry a
    # consistent sense; the boundary walk depends on it.
    a, b, c = pts[simplices[:, 0]], pts[simplices[:, 1]], pts[simplices[:, 2]]
    twice_area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
        c[:, 0] - a[:, 0]
    )
    flip = twice_area < 0
    simplices = np.where(flip[:, None], simplices[:, ::-1], simplices)
    twice_area = np.abs(twice_area)

    if alpha is None:
        alpha = _auto_alpha(pts, simplices)

    radius = _circumradius(pts, simplices, twice_area)
    keep = np.isfinite(radius) & (radius <= alpha)
    if not keep.any():
        return _convex_ring(pts), ("convex-hull-fallback",)

    parts = _substantial_parts(len(pts), simplices[keep], twice_area[keep] / 2.0)
    if not parts:
        return _convex_ring(pts), ("convex-hull-fallback",)

    notes: list[str] = []
    if not keep_parts and len(parts) > 1:
        parts = parts[:1]
        notes.append("largest-part")

    rings: list[np.ndarray] = []
    voids = 0
    for part in parts:
        outer, holes = _part_rings(pts, part)
        if outer is None:
            continue
        rings.extend(outer)
        if keep_voids:
            holes = [h for h in holes if polygon_area(h) >= _MIN_VOID_AREA_M2]
            voids += len(holes)
            rings.extend(holes)
    if not rings:
        return _convex_ring(pts), ("convex-hull-fallback",)

    if simplify_tolerance > 0:
        # Each ring is simplified on its own and stitched afterwards: running
        # Douglas-Peucker across a bridge would drop one of the two copies of a
        # bridge vertex and leave a ring that no longer closes on itself.
        rings = [simplify_polygon(r, tolerance=simplify_tolerance) for r in rings]

    if len(parts) > 1:
        notes.append("multipart")
    if voids:
        notes.append("voids")
    return _stitch_rings(rings), tuple(notes)


def _substantial_parts(
    n_points: int, simplices: np.ndarray, areas: np.ndarray
) -> list[np.ndarray]:
    """Split the alpha shape into pieces worth describing, largest first.

    The version this replaces asked whether the largest piece held 98 per cent
    of the area and fell back to the convex hull of everything if it did not.
    That is a cliff: on a two-room capture measured here it turned 60 m2 of
    scanned floor into an 83 m2 claim, 23 m2 of which is corridor nobody walked,
    and a 6 per cent alcove turned 31.8 m2 into 37.0 m2. Both land on the wrong
    side of a threshold that the fixtures, whose fragments are 0.004 m2 specks,
    never approached.

    So the fraction is gone. Pieces are judged on their own absolute area, every
    piece that could hold a camera survives, and the caller decides whether it
    can represent more than one. Erring toward the smaller region is the
    module's stated principle and dropping a speck obeys it; enlarging the
    claim to the convex hull to be rid of one never did.
    """
    if len(simplices) == 0:
        return []
    rows = np.concatenate([simplices[:, 0], simplices[:, 1], simplices[:, 2]])
    cols = np.concatenate([simplices[:, 1], simplices[:, 2], simplices[:, 0]])
    graph = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_points, n_points)
    ).tocsr()
    ncomp, labels = csgraph.connected_components(graph, directed=False)
    per_triangle = labels[simplices[:, 0]]
    by_component = np.bincount(per_triangle, weights=areas, minlength=ncomp)

    # Descending area, ties broken by component label, so the answer does not
    # depend on how scipy happened to number the pieces.
    order = np.lexsort((np.arange(ncomp), -by_component))
    order = order[by_component[order] > 0]
    if len(order) == 0:
        return []

    big = order[by_component[order] >= _MIN_PART_AREA_M2]
    if len(big) == 0:
        # Everything is smaller than a tripod. The largest piece is still the
        # honest answer -- a small capture is small, not a failure.
        big = order[:1]
    return [simplices[per_triangle == k] for k in big]


def _part_rings(
    pts: np.ndarray, simplices: np.ndarray
) -> tuple[list[np.ndarray] | None, list[np.ndarray]]:
    """Split one piece's boundary cycles into outer rings and interior voids.

    Sign does the work: the triangles are wound counter-clockwise, so a cycle
    that encloses kept region has positive area and a cycle that encloses a void
    has negative area. A piece pinched to a point can present two positive cycles,
    and those are all returned rather than resolved, because the even-odd test
    downstream does not care which void belongs to which ring -- it only counts
    crossings.

    Voids are deliberately left wound clockwise. The even-odd test does not care
    either way, but the shoelace does: with the void negative, the stitched
    polygon's signed area is the region's area less the void's, which is what
    `CaptureBounds.area` has to report. Flipping them to counter-clockwise for
    tidiness would make a void *add* its area to the claim.
    """
    cycles = _boundary_rings(pts, simplices)
    if not cycles:
        return None, []
    outer = [r for r in cycles if signed_area(r) > 0]
    voids = [r for r in cycles if signed_area(r) < 0]
    if not outer:
        return None, []
    outer.sort(key=polygon_area, reverse=True)
    voids.sort(key=polygon_area, reverse=True)
    return outer, voids


def _stitch_rings(rings: list[np.ndarray]) -> np.ndarray:
    """Join several rings into one vertex array using doubled bridge edges.

    `CaptureBounds.hull_xy` is a single (M, 2) array and the dataclass is
    frozen, so a region with two pieces or a void in it has to be expressed
    within that. A bridge -- walk out to the next ring along a segment, around
    it, and back along the same segment -- does exactly that under the even-odd
    crossing rule `contains` implements, because both traversals of the bridge
    return the same crossing answer and cancel. The same cancellation happens in
    the shoelace sum, so the area comes out as the pieces' areas less the voids'
    with no correction term.

    The bridge is drawn between the closest pair of vertices, which keeps it
    inside the region it spans in the ordinary case and makes the polygon
    legible if anyone draws it. Nothing depends on that: the cancellation is
    exact whatever the bridge crosses.
    """
    if not rings:
        return np.zeros((0, 2))
    ring = rings[0]
    for other in rings[1:]:
        if len(other) < 3:
            continue
        d, i = cKDTree(ring).query(other, k=1)
        j = int(np.argmin(d))
        anchor = int(i[j])
        ring = np.concatenate(
            [
                ring[: anchor + 1],
                np.roll(other, -j, axis=0),
                other[j : j + 1],
                ring[anchor:],
            ]
        )
    return ring


def _auto_alpha(pts: np.ndarray, simplices: np.ndarray) -> float:
    """Alpha from the data: a multiple of the median triangulation edge."""
    e = np.concatenate(
        [simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]]
    )
    lengths = np.linalg.norm(pts[e[:, 0]] - pts[e[:, 1]], axis=1)
    median = float(np.median(lengths))
    if not np.isfinite(median) or median <= 0:
        return float("inf")
    return _ALPHA_EDGE_FACTOR * median


def _circumradius(
    pts: np.ndarray, simplices: np.ndarray, twice_area: np.ndarray
) -> np.ndarray:
    """R = abc / (4A) for every triangle, vectorised. Slivers give inf."""
    a = np.linalg.norm(pts[simplices[:, 1]] - pts[simplices[:, 0]], axis=1)
    b = np.linalg.norm(pts[simplices[:, 2]] - pts[simplices[:, 1]], axis=1)
    c = np.linalg.norm(pts[simplices[:, 0]] - pts[simplices[:, 2]], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(twice_area > 1e-18, a * b * c / (2.0 * twice_area), np.inf)


def _boundary_rings(pts: np.ndarray, simplices: np.ndarray) -> list[np.ndarray]:
    """Walk the kept triangles' boundary and return every closed cycle.

    Working with *directed* edges rather than undirected ones is what makes this
    robust. Every triangle is wound counter-clockwise, so an interior edge
    appears once in each direction and a boundary edge appears once only. The
    surviving directed edges therefore already chain head-to-tail into closed
    cycles with the outer boundary counter-clockwise and any hole clockwise, and
    orientation falls out for free instead of having to be repaired afterwards.

    Every cycle is returned, not just the biggest. Which of them matter is not a
    question this routine can answer: a hole in a walked camera loop is an
    artefact and a hole in a floor slab is a place nothing looked at, and only
    the caller knows which of those it is holding. Discarding them here is what
    let `capture_bounds` claim floor in the middle of an unscanned void.
    """
    n = len(pts)
    starts = np.concatenate([simplices[:, 0], simplices[:, 1], simplices[:, 2]])
    ends = np.concatenate([simplices[:, 1], simplices[:, 2], simplices[:, 0]])
    key = starts.astype(np.int64) * n + ends.astype(np.int64)
    reverse = ends.astype(np.int64) * n + starts.astype(np.int64)
    boundary = ~np.isin(reverse, key, assume_unique=False)
    if not boundary.any():
        return []

    bs, be = starts[boundary], ends[boundary]
    order = np.argsort(bs, kind="stable")
    bs, be = bs[order], be[order]
    group_start = np.searchsorted(bs, np.arange(n), side="left")
    group_end = np.searchsorted(bs, np.arange(n), side="right")

    used = np.zeros(len(bs), dtype=bool)
    rings: list[np.ndarray] = []

    for seed in range(len(bs)):
        if used[seed]:
            continue
        cycle = [int(bs[seed])]
        used[seed] = True
        v = int(be[seed])
        ok = True
        while v != cycle[0]:
            lo, hi = group_start[v], group_end[v]
            nxt = -1
            for e in range(lo, hi):
                if not used[e]:
                    nxt = e
                    break
            if nxt < 0:
                ok = False
                break
            used[nxt] = True
            cycle.append(v)
            v = int(be[nxt])
            if len(cycle) > len(bs) + 1:
                ok = False
                break
        if not ok or len(cycle) < 3:
            continue
        rings.append(pts[np.array(cycle, dtype=np.int64)])

    return rings


def _convex_ring(pts: np.ndarray) -> np.ndarray:
    """Convex hull as a CCW ring; the fallback when the alpha shape gives up."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return pts
    try:
        hull = ConvexHull(pts)
    except (QhullError, ValueError):
        return pts
    ring = pts[hull.vertices]
    if signed_area(ring) < 0:
        ring = ring[::-1]
    return ring


def _lattice_unique(pts: np.ndarray, cell: float) -> np.ndarray:
    """Collapse points onto an XY lattice, keeping one representative per cell.

    Deterministic where random subsampling would not be, and it makes the
    automatic alpha stable: after this the median Delaunay edge is set by the
    lattice rather than by whatever density the scanner happened to achieve on
    that particular wall.
    """
    if len(pts) == 0 or cell <= 0:
        return pts
    keys = np.floor(pts / cell).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(first)]


# ---------------------------------------------------------------------------
# dilation, via a raster
# ---------------------------------------------------------------------------


def dilate_polygon(
    poly_xy: np.ndarray,
    radius: float,
    *,
    seed_points: np.ndarray | None = None,
    cell: float = _RASTER_CELL_M,
) -> np.ndarray:
    """Grow a region outward by `radius` metres and return its outer ring.

    Offsetting each edge along its normal is the textbook answer and it is a
    trap: at a concave vertex the offset edges cross, and a region derived from
    a walked loop is nothing but concave vertices, so the naive version produces
    a self-intersecting mess whose area is meaningless. Rasterising, taking a
    distance transform and re-extracting the contour cannot self-intersect by
    construction.

    Three things seed the raster -- the polygon's interior, its edges, and
    `seed_points` -- and the union is taken because a "polygon" derived from a
    one-dimensional path can legitimately enclose almost no area at all. A
    camera track is a curve, and the alpha shape of a curve is a thin sliver
    whose interior is empty; dilating only that interior would grow nothing.
    Seeding from the raw positions as well means the dilation is anchored to
    where the operator provably stood no matter what shape the hull came out.

    Holes are filled after dilating. A void enclosed by the dilated region is
    floor the operator walked all the way around, which is scanned floor by any
    reasonable reading, and leaving it as a hole would carve the middle out of
    every room captured by walking its perimeter.

    The cost is a boundary quantised to `cell`, which is why the result gets
    simplified afterwards and why the default cell is well under any dilation
    radius we use.
    """
    poly = np.asarray(poly_xy, dtype=np.float64).reshape(-1, 2)
    pts = (
        np.zeros((0, 2))
        if seed_points is None
        else np.asarray(seed_points, dtype=np.float64).reshape(-1, 2)
    )
    everything = np.concatenate([poly, pts]) if len(pts) else poly
    if len(everything) == 0 or radius <= 0:
        return poly

    lo = everything.min(axis=0) - radius - 4 * cell
    hi = everything.max(axis=0) + radius + 4 * cell
    span = hi - lo
    # Keep the raster affordable for a warehouse-sized capture by coarsening
    # rather than by allocating; a 3 cm boundary on a 60 m building is a lie of
    # precision anyway.
    est = (span[0] / cell + 1) * (span[1] / cell + 1)
    if est > _RASTER_MAX_CELLS:
        cell *= float(np.sqrt(est / _RASTER_MAX_CELLS))
    nx = int(np.ceil(span[0] / cell)) + 1
    ny = int(np.ceil(span[1] / cell)) + 1

    seed = np.zeros((nx, ny), dtype=bool)
    if len(poly) >= 3:
        gx = lo[0] + np.arange(nx) * cell
        gy = lo[1] + np.arange(ny) * cell
        centres = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1).reshape(-1, 2)
        # Reuse the frozen point-in-polygon test rather than writing a second
        # one that can disagree with it about edge cases.
        seed |= CaptureBounds(hull_xy=poly).contains(centres).reshape(nx, ny)
    _mark(seed, _densify(poly, cell * 0.5), lo, cell)
    _mark(seed, pts, lo, cell)
    if not seed.any():
        return poly

    distance = ndimage.distance_transform_edt(~seed, sampling=(cell, cell))
    mask = ndimage.binary_fill_holes(seed | (distance <= radius))

    contours = measure.find_contours(mask.astype(np.float64), 0.5)
    if not contours:
        return poly
    rings = [
        np.stack([lo[0] + c[:, 0] * cell, lo[1] + c[:, 1] * cell], axis=1)
        for c in contours
    ]
    ring = max(rings, key=polygon_area)
    if signed_area(ring) < 0:
        ring = ring[::-1]
    return ring


def _densify(poly: np.ndarray, step: float) -> np.ndarray:
    """Sample along a ring's edges so rasterising it cannot leak through gaps."""
    if len(poly) < 2 or step <= 0:
        return poly
    a = poly
    b = np.roll(poly, -1, axis=0)
    counts = np.maximum(np.ceil(np.linalg.norm(b - a, axis=1) / step).astype(int), 1)
    out = [
        a[i] + np.linspace(0.0, 1.0, counts[i], endpoint=False)[:, None] * (b[i] - a[i])
        for i in range(len(a))
    ]
    return np.concatenate(out)


def _mark(mask: np.ndarray, pts: np.ndarray, lo: np.ndarray, cell: float) -> None:
    """Set the raster cell containing each point. In place, vectorised."""
    if len(pts) == 0:
        return
    idx = np.round((pts - lo) / cell).astype(np.int64)
    ok = (
        (idx[:, 0] >= 0)
        & (idx[:, 0] < mask.shape[0])
        & (idx[:, 1] >= 0)
        & (idx[:, 1] < mask.shape[1])
    )
    idx = idx[ok]
    mask[idx[:, 0], idx[:, 1]] = True


# ---------------------------------------------------------------------------
# capture bounds
# ---------------------------------------------------------------------------


def capture_bounds(
    cloud: PointCloud,
    *,
    camera_positions: np.ndarray | None = None,
    floor_z: float = 0.0,
    alpha: float | None = None,
) -> CaptureBounds:
    """The region of floor this twin is allowed to make claims about.

    With poses, the answer is measured: the concave hull of the camera path,
    grown by `POSE_DILATION_M` because the scanner saw somewhat past where the
    operator's feet were. `source` comes back as "poses" and `z_range` is the
    real height range of the path.

    Without poses, there is no measurement to be had, and the honest proxy is
    the floor we actually reconstructed -- points within `FLOOR_BAND_M` of
    `floor_z`, hulled. That region is a genuine lower bound on where the scanner
    could see, since floor does not appear in a scan unless something looked at
    it. `source` comes back as "inferred" and `z_range` collapses to the floor,
    because with no poses we know nothing whatsoever about how high the camera
    was and a plausible-looking range would be a fabrication. Callers that need
    a camera height must supply their own when `source != "poses"`.

    The two branches read the same geometry differently and have to. A hole in
    the camera path is floor the operator walked around and it gets filled; a
    hole in the floor slab is floor that no beam ever reached and it is kept, so
    that `contains` says False there. Skipping that distinction is what let this
    function hand the shot search a tripod position in the middle of an
    unscanned void.

    Both branches are deliberately conservative. The consumer of this is the
    shot search, and the failure it exists to prevent is a tripod proposed in a
    corner of the room that was never scanned -- geometry there is interpolation
    between the walls either side of it, and it will not match the real place.

    Anything the boundary could not represent honestly is appended to `source`
    after a "+": "convex-hull-fallback" when the alpha shape failed and the
    region returned is the convex hull and therefore larger than what was
    scanned, "largest-part" when a disjoint piece of the capture was dropped,
    "multipart" when the region is more than one piece, "voids" when it has
    interior holes. A caller with a `Twin` in hand should put that string into
    `provenance`; the number on its own cannot say whether it is a measurement
    or a fallback, and PHASE1_SPEC rule 3 does not allow it to stay silent.
    """
    cams = (
        None
        if camera_positions is None
        else np.asarray(camera_positions, dtype=np.float64).reshape(-1, 3)
    )
    if cams is not None and len(cams):
        # One walk, one ring: a second fragment of a camera path is a tracking
        # glitch rather than a second room, and the dilation below would bridge
        # the two with a corridor of floor nobody walked if both were kept.
        path, notes = _concave_hull(cams[:, :2], alpha=alpha)
        hull = dilate_polygon(path, POSE_DILATION_M, seed_points=cams[:, :2])
        hull = simplify_polygon(hull, tolerance=0.05)
        return CaptureBounds(
            hull_xy=hull,
            z_range=(float(cams[:, 2].min()), float(cams[:, 2].max())),
            camera_positions=cams,
            source=_source("poses", notes),
        )

    xyz = cloud.xyz
    band = np.abs(xyz[:, 2] - floor_z) <= FLOOR_BAND_M
    if band.sum() < 3:
        # No reconstructed floor at all. Claiming the whole cloud's footprint
        # here would be exactly the overreach this function exists to stop, so
        # claim nothing and let the caller see an empty hull.
        return CaptureBounds(
            hull_xy=np.zeros((0, 2)),
            z_range=(float(floor_z), float(floor_z)),
            camera_positions=None,
            source="inferred",
        )

    # Voids and separate pieces are both kept here: with no poses, the only
    # evidence of coverage is where floor came back, so an absence of floor is
    # the evidence of an absence of coverage. Simplification happens per ring
    # inside, because the stitched ring cannot survive a pass afterwards.
    hull, notes = _concave_hull(
        xyz[band][:, :2],
        alpha=alpha,
        keep_parts=True,
        keep_voids=True,
        simplify_tolerance=0.05,
    )
    return CaptureBounds(
        hull_xy=hull,
        z_range=(float(floor_z), float(floor_z)),
        camera_positions=None,
        source=_source("inferred", notes),
    )


def _source(base: str, notes: tuple[str, ...]) -> str:
    """Fold the hull's caveats into the `source` string.

    `CaptureBounds` is frozen and there is nowhere else to put this, but the
    caveats are exactly the difference between a measured boundary and a
    fallback that is knowingly too generous, and rule 3 says a downstream reader
    has to be able to tell those apart. The base word stays first and unchanged,
    so a reader that only wants to know where the region came from can take
    `source.split("+")[0]` and get exactly the old two values back.
    """
    return base if not notes else base + "+" + "+".join(notes)
