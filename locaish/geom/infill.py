"""Closing the holes a real capture leaves, and admitting which ones we closed.

A room filmed by someone who is not a surveyor comes back with gaps, and they
are not small: floor hidden under furniture, a wall glanced at once from an
angle, and -- almost always -- no ceiling at all, because nobody points a phone
at the ceiling. Meshing that directly produces a shell full of windows, which is
useless to look at and useless to reason about. Demanding a better capture is
not a fix; most captures will look like that one.

The completion here is deliberately *generic*. It knows nothing about rooms,
walls, or which way is up, and it works by asking a question the data can
actually answer: **where did the camera prove there was nothing?**

Every camera-to-point ray sweeps out a corridor of space that was definitely
empty, because light travelled along it. Union those corridors and you have the
volume the capture demonstrably saw into. That volume is a solid region, and a
solid region has a boundary -- closed, by construction, however full of holes
the point cloud was. Where the sweep looked at a wall, the boundary sits on the
wall. Where the sweep never looked, it sits at the frontier of what was seen.
There is no hole for it to have, because it is the edge of a volume rather than
a patchwork of observed fragments.

This is space carving, and choosing it over interpolation was not taste.
Bridging holes by interpolating the surface across them -- inpainting the
distance field, stretching a minimal surface -- collapses on exactly the case
that matters here. A missing ceiling is not a hole in a sheet; it is a
metre-tall void with almost no boundary to interpolate from, and an
interpolating filler given one either packs the room solid or hangs a membrane
in mid-air. Both were measured on a synthetic room before this module was
rewritten. Carving cannot do either, because it never invents matter: it only
ever reports the limit of what was observed.

Two properties keep it honest.

**It cannot seal an opening the camera saw through.** A doorway the sweep looked
through has carved space beyond it, so the boundary runs past the door rather
than across it. That is the most visible and most damaging thing a filler can
get wrong, and the method excludes it structurally rather than by a threshold.

**Every completed vertex is labelled.** Boundary that coincides with measured
returns is measured; boundary that does not is the frontier of the sweep, and
the mesh carries a per-vertex `filled` weight saying which is which. A twin is
allowed to contain inference. It is not allowed to let inference pass for
measurement -- so where this module guessed, `Mesh.filled` says so, the colours
are muted, and QA reports the fraction.

What it cannot do is tell you where the ceiling is. The boundary above a sweep
that never tilted up sits wherever the highest rays happened to reach, which is
a record of the capture rather than of the architecture. That surface is
labelled inferred, `ceiling_z` still comes back None, and no measurement is
taken from it.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# Rays cast to carve free space. Each is a camera-to-point segment. The number
# is set by area, not by taste: a room presents on the order of a hundred
# thousand voxel-sized boundary cells at 5 cm, and the carve has to put several
# rays through the neighbourhood of each one for the swept volume to come out
# solid rather than speckled.
MAX_CARVE_RAYS = 400_000

# Sample spacing along a ray, in voxels. One voxel rather than a half: the
# closing pass afterwards repairs the occasional skipped cell, and halving the
# sample count is what makes four hundred thousand rays affordable.
RAY_STEP_VOXELS = 1.0
MAX_RAY_SAMPLES = 384

# Carving stops this far short of the point it was cast at. It is deliberately
# under a voxel, and the surface is protected by masking the carve against the
# occupied crust instead -- a mask cannot be off by a fraction of a cell the way
# a distance margin can.
#
# The margin used to be 1.5 voxels, and that was a real bug rather than a
# conservative choice: it set the swept volume's boundary 7 cm inside the
# measured crust, so the two surfaces were disconnected, marching cubes returned
# them as separate components, and the largest-component filter kept the
# inferred one and discarded every measured triangle in the twin. Co-locating
# them makes the completed crust a single connected surface that is merely
# thicker where both agree.
CARVE_STOP_VOXELS = 0.5

# Morphological closing applied to the carved volume, in voxels. Ray sampling
# leaves the carved region speckled -- a voxel here and there that no ray
# happened to pass through -- and without closing, every speckle becomes a bump
# on the boundary. Two voxels removes the sampling noise and is far too small to
# bridge any real feature.
CLOSE_VOXELS = 2

# Radius, in metres, of the ball used to smooth the swept volume's frontier.
#
# Rays that graze past an occluder leave a notch: over a patch of floor hidden
# by a sofa there is no ray reaching the floor itself, only rays passing above
# it, so the frontier rides up into a ledge and the completed floor comes out
# tens of centimetres high. Closing the *volume* fills any concavity a ball of
# this radius cannot enter, which is exactly the shape an occlusion shadow
# makes, and leaves alone anything deeper -- a real alcove, or the void where a
# ceiling was never filmed.
#
# Closing free space can only ever *add* free space, so it can never seal an
# opening; that would take an erosion, and there is none here.
#
# This radius is the one real dial in the module, and it trades two things
# against each other. A gap up to twice the radius gets bridged, so raising it
# closes bigger holes -- but closing also fills the concavity *behind* a sofa,
# so raising it far enough smooths the furniture out of the room. 0.45 m bridges
# the metre-ish shadow a piece of furniture casts on the floor while leaving the
# furniture itself standing; `--fill-radius` exists because the right answer
# depends on whether the twin is wanted for its walls or its contents.
SHADOW_CLOSE_M = 0.45

# A carved volume smaller than this fraction of the grid is not a room that was
# swept; it is a handful of rays from a capture whose poses did not solve.
# Completion declines rather than drawing a boundary around noise.
MIN_CARVED_FRACTION = 0.01


def carve_free_space(
    shape: tuple[int, int, int],
    origin: np.ndarray,
    voxel: np.ndarray,
    points: np.ndarray,
    cameras: np.ndarray,
    *,
    solid: np.ndarray | None = None,
    max_rays: int = MAX_CARVE_RAYS,
    seed: int = 0,
) -> np.ndarray:
    """Mark voxels that a camera definitely saw through.

    Which camera each point is traced back to matters more than it looks. The
    obvious choice -- the nearest one -- is the worst available: it makes every
    ray as short as it can possibly be, so the carve collapses into a thin
    sheath around the camera path and clears almost none of the room. Measured
    on a synthetic room, nearest-camera assignment carved a single column of
    free space 15 cm tall where the correct answer was floor to ceiling.

    So the camera is chosen at random instead, which makes rays long and their
    directions diverse, and the ray is then *truncated at the first surface it
    meets*. Truncation is what makes the random choice safe: a ray to a point
    the chosen camera could not actually see is stopped at whatever stood in the
    way, so it never clears space on the far side of a wall. That is ordinary
    occlusion testing, and it turns a guess about visibility into a computation.

    The result stays deliberately *conservative*: it says where we know there
    was nothing, never where we merely suspect it. Everything downstream depends
    on that asymmetry. A voxel wrongly marked free is a hole that never gets
    closed; a voxel wrongly left unknown is only a boundary drawn a little
    tight.
    """
    free = np.zeros(shape, dtype=bool)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cameras = np.asarray(cameras, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0 or len(cameras) == 0:
        return free

    rng = np.random.default_rng(seed)
    if len(points) > max_rays:
        idx = rng.choice(len(points), max_rays, replace=False)
        points = points[idx]
    origins = cameras[rng.integers(0, len(cameras), size=len(points))]

    lengths = np.linalg.norm(points - origins, axis=1)
    longest = float(lengths.max()) if len(lengths) else 0.0
    if longest <= 0:
        return free

    step = float(np.min(voxel)) * RAY_STEP_VOXELS
    samples = int(min(MAX_RAY_SAMPLES, max(8, np.ceil(longest / max(step, 1e-6)))))
    t = np.linspace(0.0, 1.0, samples)[None, :, None]

    stop = CARVE_STOP_VOXELS * float(np.min(voxel))
    limit = np.clip(1.0 - stop / np.maximum(lengths, 1e-9), 0.0, 1.0)[:, None]

    hi = np.array(shape) - 1
    chunk = max(1, int(4_000_000 / max(samples, 1)))
    for lo_i in range(0, len(points), chunk):
        sl = slice(lo_i, lo_i + chunk)
        o, p, lim = origins[sl], points[sl], limit[sl]
        pos = o[:, None, :] + t * (p - o)[:, None, :]
        ijk = np.floor((pos - origin) / voxel).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk <= hi), axis=-1)
        clipped = np.clip(ijk, 0, hi)
        ok = (t[..., 0] <= lim) & inside

        if solid is not None:
            hit = solid[clipped[..., 0], clipped[..., 1], clipped[..., 2]] & inside
            # Everything at or past the first surface the ray meets is behind
            # it, and unobserved by this camera whatever the endpoint claimed.
            ok &= np.cumsum(hit, axis=1) < 1

        sel = clipped[ok]
        if len(sel):
            free[sel[:, 0], sel[:, 1], sel[:, 2]] = True
    return free


# Visibility-based outlier deletion. A voxel is condemned when at least this
# many rays were *blocked* by it on their way to a surface well beyond it.
FLOATER_MIN_RAYS = 4
# ... where "well beyond" means the ray's endpoint lies at least this many
# voxels past the blocking voxel. Rays that stop at or just behind a surface
# are that surface being seen, not seen through -- the margin also absorbs the
# centimetre depth noise a stereo point carries along its own ray.
FLOATER_BEYOND_VOXELS = 4.0
# The peel iterates because condemned fog shields the fog behind it: deleting
# a shell exposes the next one. Real captures converge in two or three rounds;
# the cap is a backstop, not a target.
FLOATER_MAX_ITERATIONS = 6
# Grid ceiling for the vote volume. Coarsening the voxel beats refusing to
# filter: the fog this removes is decimetres across.
FLOATER_MAX_VOXELS = 200_000_000


def contradicted_points(
    points: np.ndarray,
    cameras: np.ndarray,
    *,
    voxel_m: float = 0.05,
    min_rays: int = FLOATER_MIN_RAYS,
    max_rays: int = MAX_CARVE_RAYS,
    seed: int = 0,
) -> np.ndarray:
    """Points sitting in space that rays from other views proved empty.

    `carve_free_space` uses camera-to-point rays to *add* surface where the
    capture's frontier ran out. This is the same evidence run in reverse to
    *delete* surface that should never have existed: a mismatched stereo patch
    triangulates a point into mid-air, and mid-air is exactly where the rays
    to the real surfaces behind it keep passing. This is the filter a
    k-nearest-neighbour trim cannot be -- floaters travel in clusters, one bad
    patch produces a puff of them, so they look densely neighboured to each
    other and sail through any statistical test on local spacing. They cannot
    fake visibility: rays end *on* a wall, they end *behind* a floater.

    The test is being seen through, not being crossed. Each ray is marched to
    the first occupied voxel it meets, and that voxel -- alone -- collects a
    vote if the ray's endpoint lies well beyond it, because a surface that
    blocks the view of another surface metres further on is contradicted by
    that very observation. Marching to the first hit rather than counting
    every crossing is what protects real geometry: a grazing ray to a far
    point on the same wall skims *inside* the wall's own voxel layer for its
    whole length, and counting those skim crossings condemns mid-wall points
    -- measured at 2.7% of a synthetic room's walls before this was changed,
    and 0.0% after, with the fog cluster still fully removed.

    Condemned voxels stop blocking on the next iteration, which peels the fog
    from the outside in -- a shell of deleted fog would otherwise shield the
    fog behind it. Iteration also re-deals the camera assignment, so a ray
    unlucky enough to be cast from the one camera that could not see past a
    floater gets another draw.

    Returns an (N,) bool mask, True where the point is condemned.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cams = np.asarray(cameras, dtype=np.float64).reshape(-1, 3)
    out = np.zeros(len(pts), dtype=bool)
    if len(pts) < 100 or len(cams) == 0:
        return out

    lo = pts.min(axis=0) - voxel_m
    hi = pts.max(axis=0) + voxel_m
    voxel = float(voxel_m)
    dims = np.maximum(np.ceil((hi - lo) / voxel).astype(np.int64), 1)
    while int(np.prod(dims)) > FLOATER_MAX_VOXELS:
        voxel *= 2.0
        dims = np.maximum(np.ceil((hi - lo) / voxel).astype(np.int64), 1)
    strides = np.array([dims[1] * dims[2], dims[2], 1], dtype=np.int64)
    total = int(np.prod(dims))

    idx = np.floor((pts - lo) / voxel).astype(np.int64)
    np.clip(idx, 0, dims - 1, out=idx)
    point_voxel = idx @ strides

    beyond_m = FLOATER_BEYOND_VOXELS * voxel
    condemned = np.zeros(total, dtype=bool)
    for iteration in range(FLOATER_MAX_ITERATIONS):
        solid = np.zeros(total, dtype=bool)
        live = ~condemned[point_voxel]
        solid[point_voxel[live]] = True

        rng = np.random.default_rng(seed + iteration)
        live_idx = np.flatnonzero(live)
        if len(live_idx) == 0:
            break
        # Two rays per point rather than one: a streak of stereo fog hanging
        # along a single camera's viewing ray is exactly the case where the one
        # randomly drawn camera can be the camera that produced it, and a ray
        # cast from there ends *on* the fog instead of being blocked by it.
        # A second independent draw halves the odds per iteration without
        # raising the ray budget.
        rays_per_point = 2 if len(cams) >= 2 else 1
        budget = max(1, max_rays // rays_per_point)
        if len(live_idx) > budget:
            live_idx = rng.choice(live_idx, budget, replace=False)
        ends = np.repeat(pts[live_idx], rays_per_point, axis=0)
        origins = cams[rng.integers(0, len(cams), size=len(ends))]

        lengths = np.linalg.norm(ends - origins, axis=1)
        longest = float(lengths.max()) if len(lengths) else 0.0
        if longest <= 0:
            break
        samples = int(min(MAX_RAY_SAMPLES, max(8, np.ceil(longest / voxel))))
        t = np.linspace(0.0, 1.0, samples)

        votes = np.zeros(total, dtype=np.int64)
        chunk = max(1, int(4_000_000 / max(samples, 1)))
        for lo_i in range(0, len(ends), chunk):
            sl = slice(lo_i, lo_i + chunk)
            o, p, length = origins[sl], ends[sl], lengths[sl]
            pos = o[:, None, :] + t[None, :, None] * (p - o)[:, None, :]
            ijk = np.floor((pos - lo) / voxel).astype(np.int64)
            inside = np.all((ijk >= 0) & (ijk < dims), axis=-1)
            flat = np.clip(ijk, 0, dims - 1) @ strides
            hit = solid[flat] & inside
            blocked = hit.any(axis=1)
            first = hit.argmax(axis=1)
            remaining = (1.0 - t[first]) * length
            ok = blocked & (remaining > beyond_m)
            sel = flat[np.arange(len(o)), first][ok]
            if len(sel):
                votes += np.bincount(sel, minlength=total)

        fresh = (votes >= min_rays) & ~condemned
        if not fresh.any():
            break
        condemned |= fresh

    return condemned[point_voxel]


def observed_volume(
    free: np.ndarray,
    *,
    voxel: np.ndarray | None = None,
    close_voxels: int = CLOSE_VOXELS,
    shadow_close_m: float = SHADOW_CLOSE_M,
) -> np.ndarray:
    """Turn the speckled union of ray corridors into one solid swept region.

    Three steps, each undoing a different artefact of sampling a volume with
    rays. Closing at `close_voxels` removes the speckle -- the odd cell no ray
    happened to cross. Closing again at `shadow_close_m` smooths the frontier's
    grazing notches, which is what puts the completed floor back down at floor
    level instead of on a ledge above the shadow that furniture cast. Filling
    holes then removes pockets sealed *inside* the swept volume, which are
    interior to the room by any sensible reading and would otherwise strand a
    closed bubble of surface in mid-air.
    """
    if not free.any():
        return free
    structure = ndimage.generate_binary_structure(3, 1)
    volume = _closing(free, structure, close_voxels)
    if voxel is not None and shadow_close_m > 0:
        volume = _closing(volume, structure, int(round(shadow_close_m / float(np.min(voxel)))))
    return ndimage.binary_fill_holes(volume)


def _closing(mask: np.ndarray, structure: np.ndarray, iterations: int) -> np.ndarray:
    """Morphological closing that does not eat the array's own edges.

    `ndimage.binary_closing` dilates and then erodes, and the erosion treats
    everything outside the array as empty. So when the dilation reaches the
    array boundary -- which it does whenever the radius exceeds the grid's pad,
    three voxels here -- the erosion cuts that many voxels back off the region,
    and a closing meant to smooth small notches quietly shrinks the whole room.
    That cost 10 cm off every wall of a test room before it was spotted.

    Padding by more than the radius keeps the dilation clear of the edge, and
    the crop puts the array back the size the caller handed over.
    """
    if iterations <= 0:
        return mask
    pad = iterations + 1
    wide = np.pad(mask, pad, mode="constant", constant_values=False)
    wide = ndimage.binary_closing(wide, structure=structure, iterations=iterations)
    return wide[pad:-pad, pad:-pad, pad:-pad]


def boundary_shell(volume: np.ndarray) -> np.ndarray:
    """The one-voxel skin of a solid region: what marching cubes will wrap."""
    if not volume.any():
        return volume
    structure = ndimage.generate_binary_structure(3, 1)
    return volume & ~ndimage.binary_erosion(volume, structure=structure, border_value=0)


def complete_shell(
    solid: np.ndarray,
    *,
    origin: np.ndarray,
    voxel: np.ndarray,
    points: np.ndarray,
    cameras: np.ndarray | None,
    close_voxels: int = CLOSE_VOXELS,
    shadow_close_m: float = SHADOW_CLOSE_M,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Close an incomplete crust against the volume the capture swept.

    Returns `(completed_solid, inferred_mask, stats)`. The completed crust is
    the measured one *unioned* with the boundary of the swept volume, so
    measured geometry is never moved or overwritten, only added to.
    `inferred_mask` marks what was added, and becomes `Mesh.filled`.

    Without camera poses this declines and returns its input unchanged. That is
    not a gap to be plugged later: with no record of where the camera went, an
    unobserved hole and an open doorway are the same picture, and a filler that
    cannot tell them apart will eventually wall one of them in.
    """
    stats: dict = {"carved": False, "reason": None}
    cams = None if cameras is None else np.asarray(cameras, dtype=np.float64).reshape(-1, 3)
    if cams is None or len(cams) < 2:
        stats["reason"] = "no camera poses, so no free space could be established"
        return solid, np.zeros_like(solid), stats

    free = carve_free_space(
        solid.shape, origin, voxel, points, cams, solid=solid, seed=seed
    )
    # Space the surfaces themselves occupy is not free, whatever a ray that
    # grazed past them suggests.
    free &= ~solid

    carved_fraction = float(free.sum()) / float(free.size)
    if carved_fraction < MIN_CARVED_FRACTION:
        stats["reason"] = (
            f"only {carved_fraction:.2%} of the volume could be shown empty, "
            "too little to describe a swept room"
        )
        return solid, np.zeros_like(solid), stats

    volume = observed_volume(
        free, voxel=voxel, close_voxels=close_voxels, shadow_close_m=shadow_close_m
    )

    # The measured crust is kept whole and the swept volume's frontier is added
    # to it. Two subtler alternatives were tried and both failed in ways worth
    # recording, because they look better on paper:
    #
    # Adding only the frontier that keeps clear of the crust leaves it a
    # disconnected island, and the largest-component filter deletes it -- the
    # completion silently does nothing.
    #
    # Replacing the crust with a one-voxel skin of the swept room produces a
    # single clean closed surface and then loses the floor: the mesher's
    # isosurface sits a quarter of a voxel *inside* the crust, and a sheet one
    # voxel thick barely has an inside for it to sit in, so after the field is
    # blurred the level no longer crosses and those triangles vanish.
    #
    # Union has neither problem. The crust ends up two voxels thick where the
    # sweep and the sensor agree, which only deepens the field's negative lobe
    # and makes the isosurface more robust, not less.
    shell = boundary_shell(volume)
    completed = solid | shell

    # "Inferred" means completed crust with no measured return beside it. The
    # dilation keeps the seam from being an artefact of which side of a voxel
    # boundary a point happened to land on.
    measured_near = ndimage.binary_dilation(
        solid, structure=ndimage.generate_binary_structure(3, 2), iterations=1
    )
    inferred = shell & ~measured_near

    voxel_volume = float(np.prod(voxel))
    stats.update(
        carved=True,
        free_voxels=int(free.sum()),
        swept_volume_m3=float(volume.sum()) * voxel_volume,
        occupied_voxels=int(solid.sum()),
        shell_voxels=int(shell.sum()),
        inferred_voxels=int(inferred.sum()),
        inferred_fraction=float(inferred.sum()) / max(int(shell.sum()), 1),
        fill_radius_m=shadow_close_m,
    )
    return completed, inferred, stats


def sample_mask(
    mask: np.ndarray,
    points: np.ndarray,
    *,
    origin: np.ndarray,
    voxel: np.ndarray,
) -> np.ndarray:
    """How inferred each point is, in [0, 1], read off the voxel mask.

    Trilinear rather than nearest, so the label degrades across the seam instead
    of flipping: a vertex on the boundary between measured and completed really
    is half of each, and a viewer fading between the two reads more honestly
    than one drawing a hard line the geometry does not have.

    Trilinear rather than *blurred*, which is what this did first and got wrong.
    The inferred region is a shell one voxel thick, so convolving it with a
    Gaussian of comparable width spreads its mass over the neighbours and caps
    the result near 0.4 -- every vertex of a wholly invented wall then reported
    as less than half inferred, and any threshold at a half saw nothing at all.
    Interpolation preserves the peak because it never moves mass.
    """
    coords = (np.asarray(points, dtype=np.float64) - origin) / voxel - 0.5
    values = ndimage.map_coordinates(
        mask.astype(np.float32), coords.T, order=1, mode="nearest"
    )
    return np.clip(values, 0.0, 1.0).astype(np.float32)
