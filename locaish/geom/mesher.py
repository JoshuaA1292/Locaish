"""Surface reconstruction for imports that arrived as points and nothing else.

LAS, XYZ, E57 and most Gaussian-splat exports carry no faces. The viewer needs a
surface, and so does anything that wants to talk about occlusion, so when the
reader hands us a bare cloud we build one here.

The method is volumetric rather than point-based on purpose. Ball pivoting and
Poisson both want reliable oriented normals, and a handheld sweep of a room --
with its occlusion shadows, its density that varies by an order of magnitude
between the wall you stood in front of and the one you glanced at, and its
duplicate returns from revisiting the same corner -- does not reliably provide
them. An occupancy volume does not care about normals at all: it cares whether
anything was ever seen in a 4 cm box, which is a question a noisy cloud can
answer honestly.

What comes out is a *shell*, not a solid. A room scanned from the inside is a
surface with two sides and finite thickness -- the isosurface wraps the thin
crust of occupied voxels that the walls occupy, so it is closed but it encloses
the wall material, not the room. Nothing downstream may treat it as a watertight
solid, ask it for a volume, or use inside/outside tests against it.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree
from skimage import measure

from ..types import Mesh, PointCloud
from .grid import build_grid

# Smoothing sigma for the distance field, in voxels. This is a hard upper
# bound, not a taste setting: a scanned wall is a crust roughly one voxel thick,
# so the field's negative lobe is one sample wide, and a sigma much past 0.5
# blurs that lobe away entirely and marching cubes returns an empty mesh. 0.5
# leaves the crust at about -0.57 voxels while still rounding off the staircase.
_FIELD_SIGMA_VOXELS = 0.5

# The isosurface is taken slightly *inside* the field's own zero. A crust one
# voxel thick puts the zero level about 0.42 voxels beyond the outermost
# occupied voxel centre on each side, so the room comes out about a voxel too
# big in every dimension -- measured on the clean fixture, the bounding box at
# level 0 is 12/25/29 mm wider than the source cloud's. Pulling the level to
# -0.25 voxels moves the crossing to about 0.24 voxels and takes that to
# 0/12/11 mm, still without shrinking below the cloud. It is a bias correction
# with a known cause and a measured size, not a fudge factor; the value is well
# clear of the -0.57 where the crust would vanish and the mesh with it.
_LEVEL_VOXELS = -0.25

# Empty margin around the volume. Marching cubes emits no faces on the array
# boundary, so a solid that touches the edge produces a surface with a hole in
# it; three voxels of air guarantees it never does.
_MARGIN_VOXELS = 3


def reconstruct_mesh(
    cloud: PointCloud,
    *,
    voxel: float = 0.04,
    smooth_iterations: int = 8,
    fill_holes: bool = False,
    camera_positions: np.ndarray | None = None,
    fill_radius_m: float | None = None,
    seed: int = 0,
) -> Mesh:
    """Build a triangle shell from a bare point cloud.

    The pipeline is: occupancy at `voxel`, a signed distance field made by
    differencing the Euclidean distance transforms of the empty and the occupied
    sides, a light Gaussian blur, marching cubes, largest-component extraction,
    an outward orientation pass, Taubin smoothing, and colour transfer from the
    nearest source point.

    Differencing two transforms rather than computing a true signed distance is
    the point: a real SDF needs a consistent inside, and a hollow scan of a room
    has no consistent inside. The difference of the two unsigned transforms is
    well defined for any occupancy pattern at all, including one voxel floating
    on its own, and its zero level is exactly the boundary of the occupied set.

    **The result is a shell, not a watertight solid.** It encloses the crust of
    voxels the surfaces live in, so it has an inner and an outer face and its
    enclosed volume is the volume of the walls rather than of the room.

    With `fill_holes` and camera poses, the crust is first closed against the
    volume the capture swept, so a partial capture comes back closed instead of
    perforated -- see `geom.infill` for why that is carving rather than
    interpolation.
    The returned mesh then carries a per-vertex `filled` weight, and callers
    that quote distances off it are expected to consult that rather than treat
    every triangle alike. Filling is off by default: a scan complete enough not
    to need it should not be quietly modified.

    `seed` is threaded through so that the whole reconstruction is reproducible.
    The only randomness is the ray subsample inside the free-space carve, and it
    is seeded from here.
    """
    if voxel <= 0:
        raise ValueError(f"voxel must be positive, got {voxel}")
    if len(cloud) == 0:
        return Mesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int32))

    grid = build_grid(
        cloud,
        voxel_xy=voxel,
        voxel_z=voxel,
        pad=_MARGIN_VOXELS * voxel,
    )
    solid = grid.occupied
    if not solid.any():
        return Mesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int32))

    solid = _close_pinholes(solid)

    inferred = None
    if fill_holes:
        from . import infill

        solid, inferred, fill_stats = infill.complete_shell(
            solid,
            origin=grid.origin,
            voxel=grid.voxel,
            points=cloud.xyz,
            cameras=camera_positions,
            shadow_close_m=(
                infill.SHADOW_CLOSE_M if fill_radius_m is None else fill_radius_m
            ),
            seed=seed,
        )
        if not inferred.any():
            inferred = None

    spacing = tuple(float(v) for v in grid.voxel)
    # Distances come out in metres, so the smoothing and the level stay
    # meaningful if a caller ever hands this an anisotropic grid.
    outside = ndimage.distance_transform_edt(~solid, sampling=spacing)
    inside = ndimage.distance_transform_edt(solid, sampling=spacing)
    field = outside - inside
    if _FIELD_SIGMA_VOXELS > 0:
        field = ndimage.gaussian_filter(field, sigma=_FIELD_SIGMA_VOXELS, mode="nearest")

    level = _LEVEL_VOXELS * voxel
    if field.min() >= level or field.max() <= level:
        # Nothing crosses the level: a cloud too thin or too sparse to enclose
        # anything at this voxel size. Returning empty is the honest answer;
        # inventing a surface here would be inventing geometry.
        return Mesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int32))

    verts, faces, _, _ = measure.marching_cubes(field, level=level, spacing=spacing)
    # Marching cubes places sample (i, j, k) at i*spacing; the grid places that
    # sample at the voxel *centre*, so the half-voxel has to be put back or the
    # whole mesh sits one half voxel low on every axis.
    verts = verts + grid.origin + 0.5 * grid.voxel

    verts, faces = _largest_component(verts, faces)
    if len(faces) == 0:
        return Mesh(vertices=verts, faces=np.zeros((0, 3), dtype=np.int32))

    faces = _orient_outward(verts, faces)

    mesh = Mesh(vertices=verts, faces=faces)
    if smooth_iterations > 0:
        mesh = taubin_smooth(mesh, iterations=smooth_iterations)

    mesh.vertex_colors = _transfer_colors(cloud, mesh.vertices)

    if inferred is not None:
        from . import infill

        # Sampled after smoothing rather than carried through it: Taubin moves
        # vertices by a fraction of a voxel, so reading the mask at the final
        # positions is both simpler and more accurate than reindexing a label
        # through the component filter and the smoother.
        mesh.filled = infill.sample_mask(
            inferred,
            mesh.vertices,
            origin=grid.origin,
            voxel=grid.voxel,
        )
        mesh.vertex_colors = _mute_filled(mesh.vertex_colors, mesh.filled)
    return mesh


# ---------------------------------------------------------------------------
# post-processing steps
# ---------------------------------------------------------------------------


def _mute_filled(colors: np.ndarray | None, filled: np.ndarray) -> np.ndarray | None:
    """Desaturate invented surface so it does not pass for photographed surface.

    Colour on a filled vertex comes from `_transfer_colors`, which finds the
    nearest source point -- and across a bridged hole the nearest point can be a
    metre away, on a different surface, in different light. Left alone it paints
    a convincing wall out of a smear of the sofa next to it.

    This is a fallback for viewers that do not read the `filled` attribute. It
    is not the label; the label is the array.
    """
    if colors is None or len(colors) != len(filled):
        return colors
    w = np.clip(filled, 0.0, 1.0)[:, None]
    grey = np.full_like(colors, 128, dtype=np.float64)
    muted = colors.astype(np.float64) * (1.0 - 0.75 * w) + grey * (0.75 * w)
    return np.clip(muted, 0, 255).astype(np.uint8)


def _close_pinholes(solid: np.ndarray) -> np.ndarray:
    """Morphologically close the occupied crust before differencing distances.

    Point sampling is a Poisson process, so at any voxel size fine enough to be
    interesting a good fraction of the surface cells are empty purely by luck --
    at 4 cm and 900 points per square metre roughly one cell in four contains no
    return at all. Meshing that directly gives a sponge: every missed cell
    becomes a hole the isosurface has to wrap around, the surface fragments, and
    the largest-component filter then throws most of the room away.

    Closing is dilation followed by erosion, so it is extensive and cannot move
    the outer boundary of the crust: the extreme voxel on each axis survives the
    erosion because all of its neighbours are in the dilated set. That is the
    property that matters here -- it fills the statistical holes without
    inflating a single one of the room's dimensions. Genuine holes, the ones an
    occluding body left, are far wider than the structuring element and stay
    holes, which is correct: we did not see behind the person.
    """
    structure = ndimage.generate_binary_structure(3, 3)
    return ndimage.binary_closing(solid, structure=structure, border_value=0)


def _largest_component(
    verts: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the biggest connected surface.

    Occlusion shadows and stray returns leave marching cubes with a scatter of
    detached blobs -- a few triangles around a noise voxel three metres outside
    the wall. They are indistinguishable from real geometry to anything
    downstream and they wreck the bounding box, which is the number the whole
    1:1 claim is measured on. Room furniture is not at risk here: it stands on
    the floor, so its occupied voxels touch the floor's and it is part of the
    same component.

    Size is counted in faces rather than vertices because a long thin sliver of
    noise can carry a surprising number of vertices for very little surface.
    """
    n = len(verts)
    e0 = faces[:, [0, 1, 2]].ravel()
    e1 = faces[:, [1, 2, 0]].ravel()
    graph = sparse.coo_matrix(
        (np.ones(len(e0), dtype=np.int8), (e0, e1)), shape=(n, n)
    ).tocsr()
    ncomp, labels = csgraph.connected_components(graph, directed=False)
    if ncomp <= 1:
        return verts, faces

    face_label = labels[faces[:, 0]]
    biggest = np.argmax(np.bincount(face_label, minlength=ncomp))
    keep_faces = faces[face_label == biggest]

    used = np.zeros(n, dtype=bool)
    used[keep_faces.ravel()] = True
    remap = np.full(n, -1, dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    return verts[used], remap[keep_faces].astype(np.int32)


def _orient_outward(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Make the winding globally consistent, then point the whole thing outward.

    The obvious implementation -- sample the field gradient at each face centre
    and flip whichever faces disagree with it -- is a trap, and an expensive one.
    The field here is the difference of two Euclidean distance transforms across
    a crust one voxel thick, so it is very nearly symmetric about that crust and
    the central-difference gradient sampled at the crust is identically zero:
    measured on the clean fixture, 60.0 per cent of face centroids land on a
    voxel whose gradient magnitude is exactly zero, at which point the sign test
    is a coin flip. It flipped 33392 of 257244 faces, 33351 of which were
    already correct, taking the mesh from 99.98 per cent outward-facing to 87.0
    and tearing 13692 edges' winding apart in the process. The one place the
    gradient is not degenerate is the outermost boundary, which is exactly where
    a spot check would look, so the failure is invisible to casual inspection.

    Marching cubes already emits a consistently wound surface, so the honest job
    here is not to re-derive each face's orientation but to *verify* the global
    sense and fix it once. Consistency is propagated across the face adjacency
    graph -- two triangles sharing an edge must traverse it in opposite
    directions -- which is a topological statement that no amount of field
    degeneracy can corrupt. The remaining binary choice per connected piece is
    then settled by that piece's signed volume, which for a closed surface is
    positive exactly when the winding is outward and which integrates over the
    whole component rather than trusting any single sample. A component whose
    signed volume is degenerate (an open sheet, where the quantity means
    nothing) falls back to its most extreme face along its longest axis, whose
    outward direction is unambiguous because the entire component lies behind
    it.
    """
    if len(faces) == 0:
        return faces
    faces = _propagate_winding(faces, len(verts))

    # Work about the centroid: the signed volume of a closed surface is
    # origin-independent in exact arithmetic, and centring it keeps the cubed
    # coordinates from losing precision on a twin that sits 50 m from origin.
    centre = verts.mean(axis=0)
    v = verts[faces] - centre
    vol6 = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2]))

    comp, ncomp = _face_components(faces, len(verts))
    volume = np.bincount(comp, weights=vol6, minlength=ncomp) / 6.0

    span = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    degenerate = np.abs(volume) <= 1e-6 * max(span, 1e-9) ** 3
    flip_comp = volume < 0.0
    if degenerate.any():
        flip_comp[degenerate] = _extreme_face_says_flip(v, faces, comp, ncomp)[degenerate]

    flip = flip_comp[comp]
    return np.where(flip[:, None], faces[:, ::-1], faces).astype(np.int32)


def _propagate_winding(faces: np.ndarray, n_verts: int) -> np.ndarray:
    """Flip faces until every shared edge is traversed in both directions.

    The relative orientation of two edge-adjacent triangles is a hard fact, so
    the only freedom left is one sign per connected piece. Rather than walking a
    spanning tree in Python, the parity is solved by connected components on the
    bipartite double cover: each face gets two nodes, one per orientation, and a
    shared edge wires the pair together either straight or crossed depending on
    whether the two triangles currently agree. Faces that end up in the same
    component as the seed keep their winding and the rest are flipped, which is
    one sparse graph traversal for the whole mesh instead of a per-face loop.

    A non-orientable component -- impossible from marching cubes, but cheap to
    survive -- collapses the double cover into one component, and every face
    then compares equal to the seed and is left exactly as it arrived.
    """
    fa, fb, parity = _edge_pairs(faces, n_verts)
    n = len(faces)
    if len(fa) == 0:
        return faces

    partner = np.where(parity, fb + n, fb)
    rows = np.concatenate([fa, fa + n])
    cols = np.concatenate([partner, np.where(parity, fb, fb + n)])
    cover = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(2 * n, 2 * n)
    ).tocsr()
    _, cover_label = csgraph.connected_components(cover, directed=False)

    comp, ncomp = _face_components(faces, n_verts, pairs=(fa, fb))
    # Lowest face index in each component is the seed, so the answer does not
    # depend on how scipy happened to number the components.
    seed = np.zeros(ncomp, dtype=np.int64)
    backwards = np.arange(n - 1, -1, -1)
    seed[comp[backwards]] = backwards

    flip = cover_label[: n] != cover_label[seed[comp]]
    return np.where(flip[:, None], faces[:, ::-1], faces)


def _edge_pairs(faces: np.ndarray, n_verts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Manifold edge pairs and whether the two faces disagree about them.

    Edges shared by more than two triangles are dropped rather than repaired:
    they carry no unambiguous relative orientation, and letting one decide the
    winding of a whole component would put the coin flip back.
    """
    starts = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]]).astype(np.int64)
    ends = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]]).astype(np.int64)
    owner = np.tile(np.arange(len(faces), dtype=np.int64), 3)
    real = starts != ends
    starts, ends, owner = starts[real], ends[real], owner[real]

    lo = np.minimum(starts, ends)
    hi = np.maximum(starts, ends)
    forward = starts < ends
    key = lo * np.int64(n_verts) + hi

    order = np.argsort(key, kind="stable")
    key, owner, forward = key[order], owner[order], forward[order]
    _, first, counts = np.unique(key, return_index=True, return_counts=True)
    manifold = first[counts == 2]
    fa, fb = owner[manifold], owner[manifold + 1]
    # Consistent means one triangle walks the edge low-to-high and the other
    # high-to-low, so equal senses are exactly the pairs needing a relative flip.
    parity = forward[manifold] == forward[manifold + 1]
    return fa, fb, parity


def _face_components(
    faces: np.ndarray,
    n_verts: int,
    pairs: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, int]:
    """Label faces by edge connectivity, which is what orientation travels along.

    Vertex connectivity is not enough: two triangles meeting at a single point
    are one blob to `_largest_component` but have no shared edge, so nothing
    forces their orientations to agree and each needs its own global decision.
    """
    if pairs is None:
        fa, fb, _ = _edge_pairs(faces, n_verts)
    else:
        fa, fb = pairs
    n = len(faces)
    if len(fa) == 0:
        return np.arange(n, dtype=np.int64), n
    graph = sparse.coo_matrix(
        (np.ones(len(fa), dtype=np.int8), (fa, fb)), shape=(n, n)
    ).tocsr()
    ncomp, labels = csgraph.connected_components(graph, directed=False)
    return labels.astype(np.int64), int(ncomp)


def _extreme_face_says_flip(
    v: np.ndarray, faces: np.ndarray, comp: np.ndarray, ncomp: int
) -> np.ndarray:
    """Per-component fallback: does the outermost face point the wrong way?

    Only reached for a component whose signed volume is meaningless because the
    component is an open sheet rather than a closed surface, which the shell this
    module builds never is. The best evidence left is the face whose centroid is
    furthest along the axis the mesh is longest in: the rest of the component
    lies behind it, so a normal pointing back into the component is wrong. It is
    one sample rather than an integral and it can be defeated by a surface that
    curls over its own extreme face, which is why it is the fallback and the
    volume is the rule.
    """
    centroids = v.mean(axis=1)
    normals = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    axis = int(np.argmax(v.reshape(-1, 3).max(axis=0) - v.reshape(-1, 3).min(axis=0)))

    order = np.lexsort((np.arange(len(faces)), -centroids[:, axis]))
    chosen = np.full(ncomp, -1, dtype=np.int64)
    # Last write wins, so walking the worst candidate first leaves the best.
    chosen[comp[order[::-1]]] = order[::-1]
    return normals[chosen, axis] < 0.0


def _transfer_colors(cloud: PointCloud, verts: np.ndarray) -> np.ndarray | None:
    """Nearest-source-point colour for each vertex.

    Nearest neighbour rather than an interpolated average: the interesting
    colour edges in a room are material boundaries -- skirting, a door frame,
    the line where the wall meets the floor -- and averaging across them smears
    exactly the features that make the reconstruction legible.
    """
    if cloud.rgb is None or len(cloud) == 0 or len(verts) == 0:
        return None
    tree = cKDTree(cloud.xyz)
    _, nn = tree.query(verts, k=1, workers=-1)
    return cloud.rgb[nn]


# ---------------------------------------------------------------------------
# smoothing and decimation
# ---------------------------------------------------------------------------


def taubin_smooth(
    mesh: Mesh,
    iterations: int = 8,
    lam: float = 0.5,
    mu: float = -0.53,
) -> Mesh:
    """Smooth without shrinking, via alternating Laplacian passes.

    Plain Laplacian smoothing is a low-pass filter whose gain is below one at
    every non-zero frequency, so it does not merely round off the stair steps --
    it contracts the whole surface toward its centroid. Run it eight times on a
    room and the room is centimetres smaller, which destroys the only property
    this project actually sells. Taubin follows each shrinking pass (`lam > 0`)
    with a slightly larger inflating pass (`mu < -lam`); the composite transfer
    function is near unity in the low frequencies that carry the room's
    dimensions and still attenuates the high ones that carry the voxel
    staircase.

    The umbrella operator is built once as a sparse matrix and each pass is a
    single matrix product, because rebuilding neighbour lists per iteration is
    where the naive implementation spends all its time.
    """
    if iterations <= 0 or len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        return mesh

    n = len(mesh.vertices)
    f = mesh.faces
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    adj = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n, n)
    ).tocsr()
    adj.data[:] = 1.0  # duplicate edges from adjacent triangles collapse to one
    degree = np.asarray(adj.sum(axis=1)).ravel()
    inv_deg = np.where(degree > 0, 1.0 / np.where(degree > 0, degree, 1.0), 0.0)
    weights = sparse.diags(inv_deg) @ adj

    verts = mesh.vertices.copy()
    for _ in range(iterations):
        verts += lam * (weights @ verts - verts)
        verts += mu * (weights @ verts - verts)

    return Mesh(
        vertices=verts,
        faces=mesh.faces,
        vertex_colors=mesh.vertex_colors,
        uv=mesh.uv,
        texture=mesh.texture,
        texture_format=mesh.texture_format,
    )


def decimate(mesh: Mesh, target_faces: int, *, seed: int = 0) -> Mesh:
    """Reduce the face count by clustering vertices onto a regular grid.

    Quadric edge collapse gives prettier silhouettes, but it moves vertices to
    minimise an error metric and it is fiddly to make deterministic. Vertex
    clustering snaps vertices to cell centroids, which bounds every vertex's
    displacement by the cell diagonal and therefore bounds the change in the
    room's overall dimensions by a number we can state. For a twin that is sold
    on being 1:1, a stated dimensional bound beats a nicer outline.

    The cell size is found by bisection on the resulting face count rather than
    from a closed-form guess, because the relationship between cell size and
    surviving faces depends on how the surface folds and no formula gets it
    right for both a flat wall and a chair.

    `seed` is accepted for interface symmetry with the rest of the module; the
    algorithm is fully deterministic and does not use it.
    """
    if target_faces <= 0:
        raise ValueError(f"target_faces must be positive, got {target_faces}")
    if len(mesh.faces) <= target_faces:
        return mesh

    lo_corner = mesh.vertices.min(axis=0)
    span = mesh.vertices.max(axis=0) - lo_corner
    diag = float(np.linalg.norm(span))
    if diag <= 0:
        return mesh

    # Bracket: a cell far below the finest edge keeps everything, a cell the
    # size of the whole model collapses it to nothing.
    low, high = diag * 1e-4, diag
    best = _cluster(mesh, high)
    for _ in range(40):
        mid = 0.5 * (low + high)
        candidate = _cluster(mesh, mid)
        count = len(candidate.faces)
        if count > target_faces:
            low = mid
        else:
            high = mid
            best = candidate
            # Close enough. Bisecting to the last face is pure cost: the caller
            # asked for a budget, not for an exact count.
            if count >= 0.95 * target_faces:
                break
        if high - low < diag * 1e-5:
            break
    return best


def _cluster(mesh: Mesh, cell: float) -> Mesh:
    """One vertex-clustering pass at a given cell size."""
    lo = mesh.vertices.min(axis=0)
    keys = np.floor((mesh.vertices - lo) / cell).astype(np.int64)
    _, cluster_of, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    cluster_of = cluster_of.ravel()
    ncluster = len(counts)

    sums = np.zeros((ncluster, 3), dtype=np.float64)
    np.add.at(sums, cluster_of, mesh.vertices)
    reps = sums / counts[:, None]

    faces = cluster_of[mesh.faces]
    degenerate = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    )
    faces = faces[~degenerate]
    if len(faces):
        # Two source triangles can land on the same cluster triple; keep one.
        _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
        faces = faces[np.sort(first)]

    # UVs and the texture are deliberately dropped: a cluster representative has
    # no defensible texture coordinate, and carrying the old ones forward would
    # smear the atlas across the model rather than admit the loss.
    colors = None
    if mesh.vertex_colors is not None:
        csum = np.zeros((ncluster, 3), dtype=np.float64)
        np.add.at(csum, cluster_of, mesh.vertex_colors.astype(np.float64))
        colors = np.clip(csum / counts[:, None], 0, 255).astype(np.uint8)

    return Mesh(
        vertices=reps,
        faces=faces.astype(np.int32),
        vertex_colors=colors,
        texture=None,
        texture_format=mesh.texture_format,
    )
