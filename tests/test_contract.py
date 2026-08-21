"""The frozen data model, and the invariants every module agreed to obey.

These are cheap and have no dependency on the geometry modules, so they are the
first thing to run when something breaks: if the contract is intact the bug is
in an algorithm, and if it is not, nothing downstream can be trusted.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from locaish.types import (
    CaptureBounds,
    Georeference,
    Mesh,
    Opening,
    Plane,
    PointCloud,
    QAReport,
    Structure,
    Twin,
    homogeneous,
    rotation_between,
    rotation_z,
)


def test_pointcloud_rejects_mismatched_attributes():
    with pytest.raises(ValueError):
        PointCloud(xyz=np.zeros((10, 3)), rgb=np.zeros((7, 3), dtype=np.uint8))


def test_pointcloud_transform_round_trips():
    rng = np.random.default_rng(0)
    cloud = PointCloud(xyz=rng.normal(size=(500, 3)), normals=rng.normal(size=(500, 3)))
    m = homogeneous(rotation_z(0.7), np.array([1.0, -2.0, 3.0]))
    back = cloud.transformed(m).transformed(np.linalg.inv(m))
    assert np.allclose(back.xyz, cloud.xyz, atol=1e-9)
    assert np.allclose(np.linalg.norm(back.normals, axis=1), 1.0)


def test_rotation_between_handles_antiparallel():
    """The naive Rodrigues form divides by zero here, and the case is not
    exotic -- it is exactly what happens when a scanner exports a room upside
    down, which some Android exports do."""
    a = np.array([0.0, 0.0, 1.0])
    r = rotation_between(a, -a)
    assert np.allclose(r @ a, -a, atol=1e-12)
    assert np.isclose(np.linalg.det(r), 1.0)


def test_rotation_between_identity():
    a = np.array([0.3, -0.5, 0.81])
    a /= np.linalg.norm(a)
    assert np.allclose(rotation_between(a, a), np.eye(3), atol=1e-12)


def test_mesh_transform_preserves_outward_normals_under_mirroring():
    mesh = Mesh(
        vertices=np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2]]),
    )
    before = mesh.face_normals[0]
    mirror = np.eye(4)
    mirror[0, 0] = -1.0
    after = mesh.transformed(mirror).face_normals[0]
    # the mirrored triangle's normal should be the mirrored normal, not its
    # negation -- a flipped winding would give the wrong sign here
    assert np.allclose(after, np.array([-before[0], before[1], before[2]]), atol=1e-12)


def test_mesh_sample_surface_is_deterministic_and_on_the_surface():
    mesh = Mesh(
        vertices=np.array([[0.0, 0, 0], [2, 0, 0], [0, 3, 0], [2, 3, 0]]),
        faces=np.array([[0, 1, 2], [1, 3, 2]]),
    )
    a = mesh.sample_surface(1000, seed=1)
    b = mesh.sample_surface(1000, seed=1)
    assert np.array_equal(a.xyz, b.xyz)
    assert np.allclose(a.xyz[:, 2], 0.0, atol=1e-12)
    assert mesh.area == pytest.approx(6.0)


def test_plane_normal_is_normalised_and_distance_is_signed():
    p = Plane(normal=np.array([0.0, 0.0, 5.0]), offset=2.0, kind="floor")
    assert np.allclose(p.normal, [0, 0, 1])
    d = p.signed_distance(np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]]))
    assert d[0] == pytest.approx(1.0) and d[1] == pytest.approx(-1.0)


def test_structure_floor_area_uses_the_polygon_not_the_bounding_box():
    """An L-shaped room is the whole reason footprint is a polygon; if this
    ever silently becomes a bounding box the area inflates by 30% and every
    per-square-metre number downstream inflates with it."""
    l_shape = np.array([[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]], dtype=float)
    s = Structure(footprint=l_shape, ceiling_z=2.6)
    assert s.floor_area == pytest.approx(12.0)
    assert s.ceiling_height == pytest.approx(2.6)


def test_georeference_enu_axes():
    """+X bearing 90 degrees means +X points east, which must come back as the
    ENU vector (1, 0, 0). A sign error here silently mirrors every sun angle."""
    g = Georeference(latitude=37.77, longitude=-122.42, heading_deg=90.0)
    enu = g.enu_from_twin()
    assert np.allclose(enu[:, 0], [1, 0, 0], atol=1e-12)
    # +Y is 90 degrees counter-clockwise from +X seen from above, so with +X
    # east, +Y is north. Getting this backwards mirrors the room east-west and
    # every sun angle with it.
    assert np.allclose(enu[:, 1], [0, 1, 0], atol=1e-12)
    assert np.isclose(np.linalg.det(enu), 1.0)

    north = Georeference(latitude=0.0, longitude=0.0, heading_deg=0.0)
    assert np.allclose(north.enu_from_twin()[:, 0], [0, 1, 0], atol=1e-12)


def test_georeference_rejects_impossible_coordinates():
    with pytest.raises(ValueError):
        Georeference(latitude=91.0, longitude=0.0)
    with pytest.raises(ValueError):
        Georeference(latitude=0.0, longitude=181.0)


def test_georeference_solar_offset_follows_longitude():
    assert Georeference(0.0, -120.0).utc_offset_hours == pytest.approx(-8.0)


def test_capture_bounds_point_in_polygon():
    square = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
    cb = CaptureBounds(hull_xy=square)
    inside = cb.contains(np.array([[2.0, 2.0], [0.1, 3.9]]))
    outside = cb.contains(np.array([[-1.0, 2.0], [2.0, 9.0], [5.0, 5.0]]))
    assert inside.all()
    assert not outside.any()
    assert cb.area == pytest.approx(16.0)


def test_capture_bounds_handles_concave_polygon():
    """Ray crossing must get a concave hull right, since an inferred capture
    boundary around furniture is routinely concave."""
    c_shape = np.array([[0, 0], [4, 0], [4, 1], [1, 1], [1, 3], [4, 3], [4, 4], [0, 4]], dtype=float)
    cb = CaptureBounds(hull_xy=c_shape)
    assert cb.contains(np.array([[0.5, 2.0]]))[0]
    assert not cb.contains(np.array([[3.0, 2.0]]))[0]


def test_qareport_verdict_is_the_worst_status():
    r = QAReport()
    r.add("a", "pass", "fine")
    assert r.finalize().verdict == "pass"
    r.add("b", "warn", "hmm")
    assert r.finalize().verdict == "warn"
    r.add("c", "fail", "no")
    assert r.finalize().verdict == "fail"
    assert len(r.failures()) == 1
    with pytest.raises(ValueError):
        r.add("d", "catastrophe", "not a status")


def test_twin_round_trips_through_disk(tmp_path):
    rng = np.random.default_rng(3)
    n = 2500
    points = PointCloud(
        xyz=rng.normal(size=(n, 3)),
        rgb=rng.integers(0, 255, (n, 3), dtype=np.uint8),
        normals=rng.normal(size=(n, 3)),
    )
    mesh = Mesh(
        vertices=rng.normal(size=(40, 3)),
        faces=rng.integers(0, 40, (60, 3), dtype=np.int32),
        vertex_colors=rng.integers(0, 255, (40, 3), dtype=np.uint8),
        uv=rng.random((40, 2)).astype(np.float32),
        texture=b"\x89PNG\r\n\x1a\n-not-really",
    )
    qa = QAReport(metrics={"floor_area_m2": 21.3}, checks=[])
    qa.add("gravity", "pass", "the floor is level to 0.04 degrees")
    twin = Twin(
        name="round-trip",
        points=points,
        mesh=mesh,
        structure=Structure(
            floor_z=0.0,
            ceiling_z=2.71,
            planes=[Plane(np.array([0, 0, 1.0]), 0.0, kind="floor", inlier_count=900)],
            openings=[
                Opening(
                    center=np.array([1.0, 2.0, 1.4]),
                    width=1.2,
                    height=1.3,
                    normal=np.array([0, -1.0, 0]),
                    sill_height=0.9,
                    kind="window",
                    confidence=0.82,
                )
            ],
            footprint=np.array([[0, 0], [3, 0], [3, 4], [0, 4]], dtype=float),
        ),
        georeference=Georeference(51.5074, -0.1278, heading_deg=212.0, heading_source="user"),
        capture_bounds=CaptureBounds(
            hull_xy=np.array([[0, 0], [3, 0], [3, 4], [0, 4]], dtype=float),
            z_range=(1.2, 1.6),
            camera_positions=rng.normal(size=(30, 3)),
            source="poses",
        ),
        qa=qa.finalize(),
    )

    path = twin.save(tmp_path / "rt.twin")
    back = Twin.load(path)

    assert back.name == twin.name
    assert np.allclose(back.points.xyz, twin.points.xyz, atol=1e-6)
    assert np.array_equal(back.points.rgb, twin.points.rgb)
    assert np.array_equal(back.mesh.faces, twin.mesh.faces)
    assert back.mesh.texture == twin.mesh.texture
    assert back.structure.ceiling_z == pytest.approx(2.71)
    assert back.structure.openings[0].kind == "window"
    assert back.structure.openings[0].confidence == pytest.approx(0.82)
    assert back.georeference.heading_deg == pytest.approx(212.0)
    assert back.georeference.heading_source == "user"
    assert back.capture_bounds.source == "poses"
    assert back.capture_bounds.camera_positions.shape == (30, 3)
    assert back.qa.verdict == "pass"
    assert back.structure.floor_area == pytest.approx(12.0)


def test_twin_save_without_optional_parts(tmp_path):
    twin = Twin(name="bare", points=PointCloud(xyz=np.zeros((5, 3))))
    back = Twin.load(twin.save(tmp_path / "bare.twin"))
    assert back.mesh is None
    assert back.georeference is None
    assert back.capture_bounds is None
    assert len(back.points) == 5


def test_twin_refuses_a_newer_schema(tmp_path, monkeypatch):
    """A twin written by a future build must not be silently misread; the
    fields would load but mean something else."""
    import json
    import zipfile

    twin = Twin(name="future", points=PointCloud(xyz=np.zeros((3, 3))))
    path = twin.save(tmp_path / "future.twin")
    with zipfile.ZipFile(path) as zf:
        items = {n: zf.read(n) for n in zf.namelist()}
    manifest = json.loads(items["manifest.json"])
    manifest["schema_version"] = 999
    items["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as zf:
        for n, data in items.items():
            zf.writestr(n, data)
    with pytest.raises(ValueError, match="schema"):
        Twin.load(path)


def test_fixture_truth_matches_what_the_generator_built():
    """The fixtures are the measuring stick, so they get checked too: undo the
    applied transform and the room must be exactly the size it claims.

    Measured on a trimmed extent rather than min-to-max, for the reason set out
    in locaish/scan/dimensions.py -- the outermost sample of a noisy surface
    sits about four sigma proud of the wall, so a raw bounding box reports a
    5.200 m room as 5.246 m and would make this test assert the noise.
    """
    from locaish import fixtures

    fx = fixtures.build("tilted")
    inv = np.linalg.inv(fx.truth.applied_transform)
    local = fx.points.xyz @ inv[:3, :3].T + inv[:3, 3]
    lo, hi = np.percentile(local, [0.2, 99.8], axis=0)
    extent = hi - lo
    assert extent[0] == pytest.approx(fx.truth.width, abs=0.03)
    assert extent[1] == pytest.approx(fx.truth.depth, abs=0.03)
    assert extent[2] == pytest.approx(fx.truth.height, abs=0.03)

    raw = local.max(axis=0) - local.min(axis=0)
    assert raw[0] > extent[0], "the raw bounding box should overshoot; that is the point"
