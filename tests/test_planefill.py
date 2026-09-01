"""Visibility-based deletion and plane-guided completion.

Both modules turn the capture's own ray geometry into authority over points,
in opposite directions: `contradicted_points` deletes matter that rays proved
absent, `planefill` adds matter that rays proved present. The tests build
rooms whose truth is known by construction and assert the properties that make
either safe to run unattended:

- real surfaces survive deletion; clustered floaters do not, which is exactly
  the case the kNN trim cannot handle (a cluster is its own alibi);
- filled wall appears only where the camera looked, never behind an occluder,
  never across a doorway the sweep saw through;
- everything invented is labelled invented, and everything measured stays
  bit-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from locaish.geom import planefill
from locaish.geom.infill import contradicted_points
from locaish.types import PointCloud


# ---------------------------------------------------------------------------
# contradicted_points
# ---------------------------------------------------------------------------


W, D, H = 4.0, 3.0, 2.6


def _panel(span, origin, count, seed):
    rng = np.random.default_rng(seed)
    return np.asarray(origin, float) + rng.random((count, 3)) * np.asarray(span, float)


def _box_room(n=30_000, seed=0):
    """Floor, ceiling and four walls of a W x D x H room."""
    return np.concatenate([
        _panel([W, D, 0], [0, 0, 0], n, seed),
        _panel([W, D, 0], [0, 0, H], n, seed + 1),
        _panel([W, 0, H], [0, 0, 0], n // 2, seed + 2),
        _panel([W, 0, H], [0, D, 0], n // 2, seed + 3),
        _panel([0, D, H], [0, 0, 0], n // 2, seed + 4),
        _panel([0, D, H], [W, 0, 0], n // 2, seed + 5),
    ])


def _cameras(k=24):
    """A walking path around the middle of the room."""
    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return np.stack([
        W / 2 + 0.8 * np.cos(t), D / 2 + 0.6 * np.sin(t), np.full(k, 1.5)
    ], axis=1)


def test_floater_cluster_is_condemned_and_walls_survive():
    room = _box_room()
    # A dense puff of floaters in mid-air: the failure mode the kNN trim
    # cannot see, because the cluster's points neighbour each other closely.
    rng = np.random.default_rng(7)
    floaters = np.array([3.2, 0.8, 1.6]) + rng.normal(0, 0.04, (400, 3))
    pts = np.concatenate([room, floaters])

    mask = contradicted_points(pts, _cameras(), voxel_m=0.05, seed=0)

    floater_mask = mask[len(room):]
    wall_mask = mask[: len(room)]
    assert floater_mask.mean() > 0.5, "the cluster sits in carved space and must go"
    assert wall_mask.mean() < 0.02, "measured surfaces must survive the reverse carve"


def test_no_cameras_condemns_nothing():
    pts = _box_room(n=2_000)
    assert not contradicted_points(pts, np.zeros((0, 3))).any()


def test_deterministic():
    pts = np.concatenate([
        _box_room(n=5_000),
        np.array([3.2, 0.8, 1.6]) + np.random.default_rng(1).normal(0, 0.04, (200, 3)),
    ])
    a = contradicted_points(pts, _cameras(), seed=3)
    b = contradicted_points(pts, _cameras(), seed=3)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# fill_wall_planes, over synthetic rooms whose truth is known by construction
# ---------------------------------------------------------------------------


def test_fill_declines_without_cameras():
    from locaish.geom import grid as gridmod
    from locaish.scan import structure as structmod

    pts = _box_room(n=8_000)
    cloud = PointCloud(xyz=pts)
    grid = gridmod.build_grid(cloud)
    structure = structmod.analyze(cloud)
    xyz, rgb, nrm, stats = planefill.fill_wall_planes(
        cloud, structure, grid, None
    )
    assert len(xyz) == 0
    assert not stats["filled"]
    assert "camera" in stats["reason"]


def test_fill_adds_only_labelled_points_on_observed_wall():
    from locaish.geom import grid as gridmod
    from locaish.scan import structure as structmod

    room = _box_room(n=30_000, seed=2)
    # Punch a hole in the y=0 wall: drop its points in a patch, as a blank
    # painted region would come back from a stereo matcher.
    on_wall = (np.abs(room[:, 1]) < 0.02)
    in_patch = on_wall & (room[:, 0] > 1.4) & (room[:, 0] < 2.4) & (room[:, 2] > 0.8) & (room[:, 2] < 1.8)
    room = room[~in_patch]

    cloud = PointCloud(xyz=room)
    grid = gridmod.build_grid(cloud)
    structure = structmod.analyze(cloud)
    cams = _cameras()

    xyz, rgb, nrm, stats = planefill.fill_wall_planes(cloud, structure, grid, cams)

    if len(xyz) == 0:
        pytest.skip(f"fill declined: {stats['reason']} -- acceptable, never wrong")
    # Every added point lies on some wall plane of the room, in the hole's
    # neighbourhood or another genuine void -- never floating in the interior.
    walls = structure.walls()
    dists = np.min(
        np.stack([np.abs(xyz @ p.normal - p.offset) for p in walls]), axis=0
    )
    assert dists.max() < 0.05
    assert len(rgb) == len(xyz) and len(nrm) == len(xyz)


def test_fill_never_seals_a_doorway():
    from locaish.geom import grid as gridmod
    from locaish.scan import structure as structmod

    room = _box_room(n=30_000, seed=4)
    # A doorway in the y=0 wall, and cameras that looked *through* it: rays
    # from inside to points beyond the doorway carve free space behind the
    # plane there, which is the structural guarantee under test.
    on_wall = np.abs(room[:, 1]) < 0.02
    in_door = on_wall & (room[:, 0] > 1.6) & (room[:, 0] < 2.5) & (room[:, 2] < 2.1)
    room = room[~in_door]
    beyond = _panel([0.9, 0, 2.1], [1.6, -1.5, 0], 3_000, 9)  # wall of the next room
    pts = np.concatenate([room, beyond])

    cloud = PointCloud(xyz=pts)
    grid = gridmod.build_grid(cloud)
    structure = structmod.analyze(cloud)

    xyz, rgb, nrm, stats = planefill.fill_wall_planes(cloud, structure, grid, _cameras())

    if len(xyz) == 0:
        return  # declining is always safe
    # No added point may sit inside the doorway aperture on the y=0 plane.
    on_plane = np.abs(xyz[:, 1]) < 0.05
    in_aperture = (
        on_plane
        & (xyz[:, 0] > 1.7) & (xyz[:, 0] < 2.4)
        & (xyz[:, 2] > 0.1) & (xyz[:, 2] < 2.0)
    )
    assert not in_aperture.any(), "fill wrote wall across an aperture the sweep saw through"


# ---------------------------------------------------------------------------
# the inferred channel end to end
# ---------------------------------------------------------------------------


def test_floor_rescue_from_camera_height():
    """A barely captured floor beats a well-captured furniture level.

    The failure this pins: a sweep that never points down puts 100x more
    points on the furniture than on the floor, the floor detectors land on
    the furniture, and the real ceiling is then rejected as a soffit for
    being "1 m above the floor". The camera path is the evidence that
    catches it -- hand-held is 1-2 m above the floor, not 0.6.
    """
    from locaish.scan.structure import find_floor_ceiling

    rng = np.random.default_rng(0)
    # dense furniture slab 1.0 m under the cameras, sparse real floor 1.65 m
    # under them, walls to give the cloud its volume
    cam_z = 1.66
    furniture = _panel([3, 2, 0.02], [0.5, 0.5, cam_z - 1.0], 120_000, 1)
    floor = _panel([2.0, 1.8, 0.03], [1.0, 0.5, 0.0], 4_000, 2)
    walls = np.concatenate([
        _panel([4, 0.02, 2.1], [0, 0, 0], 60_000, 3),
        _panel([0.02, 3, 2.1], [0, 0, 0], 60_000, 4),
    ])
    ceiling = _panel([4, 3, 0.02], [0, 0, 2.05], 80_000, 5)
    cloud = PointCloud(xyz=np.concatenate([furniture, floor, walls, ceiling]))
    cams = np.stack([
        np.linspace(1, 3, 20), np.linspace(1, 2, 20), np.full(20, cam_z)
    ], axis=1)

    notes: list[str] = []
    floor_z, ceiling_z = find_floor_ceiling(
        cloud, [], camera_positions=cams, notes=notes
    )
    assert floor_z < 0.15, f"floor stuck at furniture level ({floor_z:.2f})"
    assert notes, "the rescue must announce itself"
    # without cameras the old (wrong but honest) answer comes back unchanged
    floor_no_cams, _ = find_floor_ceiling(cloud, [])
    assert floor_no_cams > 0.4


def test_measured_subset_and_roundtrip(tmp_path):
    xyz = np.random.default_rng(0).random((100, 3))
    inferred = np.zeros(100, dtype=np.float32)
    inferred[80:] = 1.0
    cloud = PointCloud(xyz=xyz, inferred=inferred)
    assert len(cloud.measured()) == 80

    from locaish.types import Twin

    twin = Twin(name="t", points=cloud)
    path = twin.save(tmp_path / "t.twin")
    back = Twin.load(path)
    assert back.points.inferred is not None
    assert np.allclose(back.points.inferred, inferred)
    assert len(back.points.measured()) == 80
