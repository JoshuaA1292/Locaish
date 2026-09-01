"""The room as a closed surface.

A shell that is not closed is worse than no shell: it is a picture of a room
with a hole in it, and every hole invites the reader to wonder whether the room
really has one. So the properties asserted here are structural -- every edge
shared by exactly two faces, every normal facing the room -- rather than
cosmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from locaish.geom import shell as shellmod
from locaish.types import Opening, Plane, Structure

SQUARE = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])
ELL = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [2.0, 2.0], [2.0, 3.0], [0.0, 3.0]])


def _area(poly: np.ndarray, tris: np.ndarray) -> float:
    total = 0.0
    for a, b, c in tris:
        pa, pb, pc = poly[a], poly[b], poly[c]
        total += abs(
            (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
        ) / 2.0
    return total


def test_triangulation_conserves_area():
    assert _area(SQUARE, shellmod.triangulate(SQUARE)) == pytest.approx(12.0)
    assert _area(ELL, shellmod.triangulate(ELL)) == pytest.approx(10.0)


def test_triangulation_does_not_fill_in_the_notch_of_an_l():
    """The failure a convex triangulator makes, pinned.

    Delaunay over the same six vertices returns the convex hull -- 11 m2 of
    floor in a 10 m2 room -- and the extra square metre is exactly the corner
    that is not part of the room.
    """
    tris = shellmod.triangulate(ELL)
    assert _area(ELL, tris) == pytest.approx(10.0, abs=1e-9)
    hull_area = 4.0 * 3.0 - 2.0
    assert _area(ELL, tris) < hull_area + 1e-9


def test_triangulation_survives_either_winding():
    cw = SQUARE[::-1]
    assert _area(cw, shellmod.triangulate(cw)) == pytest.approx(12.0)


def _shell(**kw):
    structure = Structure(floor_z=0.0, ceiling_z=2.5, footprint=SQUARE, **kw)
    return shellmod.build_shell(
        structure, wall_measured={i: True for i in range(len(SQUARE))}
    )


def test_the_shell_is_closed():
    sh = _shell()
    edges: dict[tuple[int, int], int] = {}
    verts = np.round(sh.vertices, 6)
    key = {}
    for i, v in enumerate(verts):
        key[i] = key.get(tuple(v), tuple(v))
    for tri in sh.faces:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            pa, pb = tuple(verts[tri[a]]), tuple(verts[tri[b]])
            edge = tuple(sorted([pa, pb]))
            edges[edge] = edges.get(edge, 0) + 1
    open_edges = [e for e, n in edges.items() if n != 2]
    assert not open_edges, f"{len(open_edges)} edges are not shared by two faces"


def test_every_face_looks_into_the_room():
    sh = _shell()
    v = sh.vertices
    a, b, c = v[sh.faces[:, 0]], v[sh.faces[:, 1]], v[sh.faces[:, 2]]
    normals = np.cross(b - a, c - a)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    centre = np.array([2.0, 1.5, 1.25])
    mid = (a + b + c) / 3.0
    assert np.all(np.sum((centre - mid) * normals, axis=1) > -1e-9)


def test_area_is_the_room_minus_its_openings():
    plain = _shell()
    walls = 2 * 4.0 * 2.5 + 2 * 3.0 * 2.5
    assert plain.area == pytest.approx(12.0 + 12.0 + walls)

    door = Opening(
        center=np.array([2.0, 0.0, 1.0]),
        width=0.9,
        height=2.0,
        normal=np.array([0.0, 1.0, 0.0]),
        sill_height=0.0,
        kind="door",
    )
    cut = _shell(openings=[door])
    assert cut.area == pytest.approx(plain.area - 0.9 * 2.0)


def test_an_inferred_ceiling_is_drawn_and_labelled():
    """Drawing it is right; letting it pass for a measurement is not."""
    structure = Structure(
        floor_z=0.0,
        ceiling_z=None,
        ceiling_z_inferred=2.42,
        ceiling_source="carve",
        footprint=SQUARE,
    )
    sh = shellmod.build_shell(structure)
    assert sh is not None
    assert sh.vertices[:, 2].max() == pytest.approx(2.42)
    ceiling = [i for i, part in enumerate(sh.parts) if part == "ceiling"]
    assert ceiling
    assert all(sh.inferred[sh.faces[i]].min() == 1.0 for i in ceiling)
    assert structure.ceiling_height is None


def test_provenance_follows_the_walls_that_have_returns():
    walls = [
        Plane(normal=np.array([0.0, 1.0, 0.0]), offset=0.0, kind="wall", inlier_count=5000),
        Plane(normal=np.array([-1.0, 0.0, 0.0]), offset=-4.0, kind="wall", inlier_count=0),
    ]
    got = shellmod.wall_provenance(SQUARE, walls)
    assert got[0] is True  # y = 0, located by returns
    assert got[1] is False  # x = 4, placed by the carve alone


def test_a_room_with_no_outline_has_no_shell():
    assert shellmod.build_shell(Structure(footprint=None)) is None
    assert shellmod.build_shell(Structure(footprint=SQUARE[:2])) is None
