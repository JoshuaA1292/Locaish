"""Surface completion: what it must close, and what it must never close.

Every test here builds a room whose truth is known by construction and then
checks a property of the completion rather than a golden number, because the
thing that makes a filler dangerous is not being slightly off -- it is being
confidently wrong about geometry nobody measured. So the assertions are of the
form "measured walls did not move", "the doorway is still a doorway", "what was
invented is labelled as invented".

The two failure modes worth naming, both of which happened during development
and both of which are pinned below:

*Silently doing nothing.* A completion whose output is disconnected from the
crust gets deleted by the largest-component filter, and the pipeline reports
success while filling nothing at all.

*Silently doing everything.* A completion that treats unobserved space as
matter packs the room solid, or moves every wall inward, and still produces a
plausible-looking closed mesh.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from locaish.geom import infill, mesher
from locaish.types import PointCloud


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


W, D, H = 4.0, 3.0, 2.6


def _panel(span, origin, count, seed):
    rng = np.random.default_rng(seed)
    return np.asarray(origin, float) + rng.random((count, 3)) * np.asarray(span, float)


def _room(*, hole=None, ceiling=True, n=40_000, seed=0):
    """A rectangular room, optionally missing a patch of floor or its ceiling."""
    floor = _panel([W, D, 0], [0, 0, 0], n, seed)
    if hole is not None:
        x0, x1, y0, y1 = hole
        keep = ~(
            (floor[:, 0] > x0) & (floor[:, 0] < x1)
            & (floor[:, 1] > y0) & (floor[:, 1] < y1)
        )
        floor = floor[keep]
    parts = [
        floor,
        _panel([W, 0, H], [0, 0, 0], n // 2, seed + 1),
        _panel([W, 0, H], [0, D, 0], n // 2, seed + 2),
        _panel([0, D, H], [0, 0, 0], n // 2, seed + 3),
        _panel([0, D, H], [W, 0, 0], n // 2, seed + 4),
    ]
    if ceiling:
        parts.append(_panel([W, D, 0], [0, 0, H], n, seed + 5))
    return np.concatenate(parts)


def _walk(height=1.5, count=20):
    """A camera path down the middle of the room at chest height."""
    return np.column_stack([
        np.linspace(0.6, W - 0.6, count),
        np.full(count, D / 2),
        np.full(count, height),
    ])


def _grid_for(points, voxel=0.05):
    from locaish.geom.grid import build_grid

    g = build_grid(PointCloud(points), voxel_xy=voxel, voxel_z=voxel, pad=3 * voxel)
    return g, mesher._close_pinholes(g.occupied)


# ---------------------------------------------------------------------------
# carving
# ---------------------------------------------------------------------------


def test_carving_clears_the_room_it_walked_through():
    """The interior must come back overwhelmingly free, or nothing else works."""
    pts = _room()
    g, solid = _grid_for(pts)
    free = infill.carve_free_space(g.shape, g.origin, g.voxel, pts, _walk(), solid=solid)

    ii, jj, kk = np.indices(g.shape)
    centre = g.origin + (np.stack([ii, jj, kk], -1) + 0.5) * g.voxel
    inside = (
        (centre[..., 0] > 0.3) & (centre[..., 0] < W - 0.3)
        & (centre[..., 1] > 0.3) & (centre[..., 1] < D - 0.3)
        & (centre[..., 2] > 0.3) & (centre[..., 2] < H - 0.3)
    )

    cleared = float((free & inside).sum()) / float(inside.sum())
    assert cleared > 0.9, f"only {cleared:.0%} of the interior was carved"


def test_carving_never_clears_a_voxel_that_holds_returns():
    pts = _room()
    g, solid = _grid_for(pts)
    free = infill.carve_free_space(g.shape, g.origin, g.voxel, pts, _walk(), solid=solid)
    assert not (free & solid).any(), "carving must not clear occupied space"


def test_carving_stops_at_the_first_surface_it_meets():
    """A ray aimed past a wall must not clear the space behind it.

    This is what makes it safe to pick the camera at random rather than knowing
    which view actually saw each point. Without truncation, a point behind a
    wall would be traced from a camera on the wrong side and carve a tunnel
    straight through the wall -- and a tunnel through a wall is a hole the
    completion would then dutifully leave open.
    """
    # A wall at x = 2, a camera at x = 0.5, and a target at x = 3.5 behind it.
    rng = np.random.default_rng(0)
    wall = np.column_stack([
        np.full(20_000, 2.0),
        rng.uniform(0, D, 20_000),
        rng.uniform(0, H, 20_000),
    ])
    behind = np.column_stack([
        np.full(2_000, 3.5),
        rng.uniform(1.0, 2.0, 2_000),
        rng.uniform(1.0, 2.0, 2_000),
    ])
    pts = np.concatenate([wall, behind])
    g, solid = _grid_for(pts)
    camera = np.array([[0.5, D / 2, 1.3], [0.55, D / 2, 1.3]])

    guarded = infill.carve_free_space(g.shape, g.origin, g.voxel, pts, camera, solid=solid)
    unguarded = infill.carve_free_space(g.shape, g.origin, g.voxel, pts, camera, solid=None)

    ii, jj, kk = np.indices(g.shape)
    centre = g.origin + (np.stack([ii, jj, kk], -1) + 0.5) * g.voxel
    beyond = centre[..., 0] > 2.3

    assert unguarded[beyond].sum() > 0, "the test is vacuous unless the ray would overshoot"
    assert guarded[beyond].sum() == 0, "carving punched through a solid wall"


# ---------------------------------------------------------------------------
# closing
# ---------------------------------------------------------------------------


def test_closing_does_not_shrink_a_region_near_the_array_edge():
    """scipy's closing erodes against the array border; ours must not.

    Left unguarded this took 10 cm off every wall of a room, because the
    dilation reached the edge of the volume and the erosion then treated
    everything outside it as empty. The regression is cheap to state: a slab
    with two voxels of margin must survive a closing far wider than that margin.
    """
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[2:-2, 2:-2, 2:-2] = True
    structure = ndimage.generate_binary_structure(3, 1)

    ours = infill._closing(mask, structure, 5)
    scipys = ndimage.binary_closing(mask, structure=structure, iterations=5)

    assert ours.sum() >= mask.sum(), "closing must be extensive"
    assert np.array_equal(ours, mask), "a convex slab must survive closing unchanged"
    assert scipys.sum() < mask.sum(), "the test is vacuous unless scipy's version shrinks"


def test_closing_fills_a_notch_it_can_span_but_not_a_void():
    """The radius is the whole dial: it must separate a shadow from a chasm.

    A shallow notch is what an occluder casts on the floor and has to close. A
    deep bite out of the same face is a part of the room nobody swept, and
    closing it would hang surface across open space. The only difference
    between them is depth against the ball's radius, which is exactly what the
    parameter is for.
    """
    structure = ndimage.generate_binary_structure(3, 1)

    shallow = np.zeros((30, 30, 30), dtype=bool)
    shallow[5:25, 5:25, 5:25] = True
    shallow[10:14, 10:14, 5:8] = False        # 3 voxels deep
    filled = infill._closing(shallow, structure, 4)
    notch = filled[10:14, 10:14, 5:8]
    # Not 100%: the layer flush with the outer face is not enclosed by anything,
    # so no amount of radius will close it. What matters is that the body of the
    # notch fills, which is what lifts a shadowed floor back to floor level.
    assert notch.mean() > 0.5, f"only {notch.mean():.0%} of a shallow notch closed"

    deep = np.zeros((30, 30, 30), dtype=bool)
    deep[5:25, 5:25, 5:25] = True
    deep[8:22, 8:22, 5:20] = False            # 15 voxels deep and wide
    still_open = infill._closing(deep, structure, 4)
    void = still_open[10:20, 10:20, 7:18]
    assert void.mean() < 0.25, f"{void.mean():.0%} of a deep void was filled in"


# ---------------------------------------------------------------------------
# completion
# ---------------------------------------------------------------------------


def test_completion_does_not_move_measured_walls():
    """The load-bearing safety property: inference may add, never displace.

    An earlier formulation replaced the crust with a skin of the swept volume
    and pulled every wall 10 cm inward -- a twin that still measured, still
    passed its own checks, and was wrong by a hand's width everywhere.
    """
    pts = _room(ceiling=False)
    plain = mesher.reconstruct_mesh(PointCloud(pts), voxel=0.05)
    filled = mesher.reconstruct_mesh(
        PointCloud(pts), voxel=0.05, fill_holes=True, camera_positions=_walk()
    )

    for axis, name in enumerate("xyz"):
        lo = abs(filled.vertices[:, axis].min() - plain.vertices[:, axis].min())
        hi = abs(filled.vertices[:, axis].max() - plain.vertices[:, axis].max())
        # The ceiling is the one bound allowed to move: it was never measured.
        if name == "z":
            assert lo < 0.06, f"floor moved {lo * 100:.0f} cm"
            continue
        assert lo < 0.06 and hi < 0.06, f"{name} bounds moved {lo:.3f}/{hi:.3f} m"


def test_completion_adds_surface_where_the_ceiling_was_never_filmed():
    pts = _room(ceiling=False)
    plain = mesher.reconstruct_mesh(PointCloud(pts), voxel=0.05)
    filled = mesher.reconstruct_mesh(
        PointCloud(pts), voxel=0.05, fill_holes=True, camera_positions=_walk()
    )

    def upper(mesh):
        v = mesh.vertices
        mid = (v[:, 0] > 0.5) & (v[:, 0] < W - 0.5) & (v[:, 1] > 0.5) & (v[:, 1] < D - 0.5)
        return int((mid & (v[:, 2] > 2.0)).sum())

    assert upper(plain) == 0, "the test is vacuous unless the ceiling really is missing"
    assert upper(filled) > 500, "nothing was added above the room"
    assert filled.filled is not None


def test_what_was_invented_is_labelled_and_what_was_measured_is_not():
    pts = _room(ceiling=False)
    m = mesher.reconstruct_mesh(
        PointCloud(pts), voxel=0.05, fill_holes=True, camera_positions=_walk()
    )
    v, f = m.vertices, m.filled

    on_floor = v[:, 2] < 0.15
    above = (v[:, 2] > 2.0) & (v[:, 0] > 0.5) & (v[:, 0] < W - 0.5)

    assert f[on_floor].mean() < 0.1, "measured floor must not be labelled inferred"
    assert f[above].mean() > 0.5, "invented ceiling must be labelled inferred"


def test_completion_cannot_seal_a_doorway_the_camera_saw_through():
    """The most damaging thing a filler can do, excluded by construction.

    The camera looks through a gap in a wall and sees a surface beyond it, so
    the space in the doorway is carved free and can never become matter. A
    filler working from surface proximity alone -- close the hole, it is only a
    metre wide -- would brick it up and produce a room with no exit.
    """
    rng = np.random.default_rng(1)
    n = 20_000
    # Far wall at x = W with a doorway punched through it, and a surface beyond.
    far = _panel([0, D, H], [W, 0, 0], n, 11)
    door = (far[:, 1] > 1.0) & (far[:, 1] < 1.9) & (far[:, 2] < 2.0)
    far = far[~door]
    beyond = np.column_stack([
        np.full(6_000, W + 1.2),
        rng.uniform(1.0, 1.9, 6_000),
        rng.uniform(0.0, 2.0, 6_000),
    ])
    pts = np.concatenate([
        _panel([W, D, 0], [0, 0, 0], n, 12),
        _panel([W, 0, H], [0, 0, 0], n // 2, 13),
        _panel([W, 0, H], [0, D, 0], n // 2, 14),
        _panel([0, D, H], [0, 0, 0], n // 2, 15),
        far,
        beyond,
    ])

    g, solid = _grid_for(pts)
    completed, inferred, stats = infill.complete_shell(
        solid, origin=g.origin, voxel=g.voxel, points=pts, cameras=_walk(), seed=0
    )

    assert stats["carved"], stats.get("reason")
    ii, jj, kk = np.indices(g.shape)
    centre = g.origin + (np.stack([ii, jj, kk], -1) + 0.5) * g.voxel
    opening = (
        (np.abs(centre[..., 0] - W) < 0.06)
        & (centre[..., 1] > 1.2) & (centre[..., 1] < 1.7)
        & (centre[..., 2] > 0.3) & (centre[..., 2] < 1.7)
    )

    assert opening.sum() > 0, "the doorway must exist in the grid for this to mean anything"
    sealed = float(completed[opening].mean())
    assert sealed < 0.2, f"{sealed:.0%} of the doorway was filled in"


def test_completion_declines_without_camera_poses():
    """No poses, no completion -- and it says so rather than guessing."""
    pts = _room(ceiling=False)
    g, solid = _grid_for(pts)

    completed, inferred, stats = infill.complete_shell(
        solid, origin=g.origin, voxel=g.voxel, points=pts, cameras=None
    )

    assert stats["carved"] is False
    assert "no camera poses" in stats["reason"]
    assert np.array_equal(completed, solid)
    assert not inferred.any()


def test_mesh_without_filling_carries_no_label():
    """None and all-zeros mean different things and must stay distinguishable."""
    pts = _room()
    m = mesher.reconstruct_mesh(PointCloud(pts), voxel=0.05)
    assert m.filled is None


def test_the_label_survives_a_save_and_reload(tmp_path):
    from locaish.types import Twin

    pts = _room(ceiling=False)
    m = mesher.reconstruct_mesh(
        PointCloud(pts), voxel=0.05, fill_holes=True, camera_positions=_walk()
    )
    twin = Twin(name="filled", points=PointCloud(pts), mesh=m)
    path = twin.save(tmp_path / "filled.twin")
    back = Twin.load(path)

    assert back.mesh.filled is not None
    assert back.mesh.filled.shape == m.filled.shape
    assert np.allclose(back.mesh.filled, m.filled, atol=1e-6)


def test_a_mismatched_label_is_rejected():
    from locaish.types import Mesh

    with pytest.raises(ValueError, match="filled has"):
        Mesh(
            vertices=np.zeros((4, 3)),
            faces=np.zeros((1, 3), dtype=np.int32),
            filled=np.zeros(3, dtype=np.float32),
        )
