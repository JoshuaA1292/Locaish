"""The room fitter, against rooms whose dimensions are known exactly.

These tests exercise the path the rest of the accuracy suite cannot reach. A
fixture ingested through a PLY file arrives without camera poses, so it takes
the plane-fitting route; the fitter here only runs when a capture came with
poses, which in production means every video reconstruction. That is precisely
the input the plane-fitting route was measured to fail on, so it needs its own
ground truth rather than a real capture and an opinion.

The fixtures are built without the export-side mangling -- no tilt, no yaw, no
unit scale -- because the canonicaliser's job is to undo that and it is tested
elsewhere. What is under test here is only whether the room comes back.
"""

from __future__ import annotations

import numpy as np
import pytest

from locaish import fixtures
from locaish.geom import room as roommod

# The same tolerance the rest of the accuracy suite holds the pipeline to.
DIMENSION_TOL_M = 0.015


def _fit(name: str, **overrides):
    entry = fixtures.catalogue()[name]
    kwargs = dict(entry["kwargs"])
    kwargs.update(tilt_deg=0.0, yaw_deg=0.0, translate=(0.0, 0.0, 0.0), unit_scale=1.0)
    kwargs.update(overrides)
    fx = fixtures.make_room(**kwargs)
    if entry.get("post") == "strip_ceiling":
        fx = fixtures._strip_ceiling(fx)
    fit = roommod.fit_room(fx.points, fx.camera_positions, floor_z=0.0, notes=[])
    return fit, fx


def _wall_distances(fit) -> dict[tuple[int, int], float]:
    """Distance from the room's origin to each axis-aligned wall, by inward normal."""
    out: dict[tuple[int, int], float] = {}
    for line in sorted(fit.lines, key=lambda l: -l.length):
        n = line.normal
        if max(abs(n[0]), abs(n[1])) < 0.98:
            continue
        key = (int(round(n[0])), int(round(n[1])))
        out.setdefault(key, -line.offset)
    return out


@pytest.mark.parametrize("room", ["clean", "sparse", "tall", "noceiling"])
def test_four_walls_at_their_true_distance(room):
    fit, fx = _fit(room)
    assert fit is not None, f"{room}: the fitter declined a room it was given poses for"
    got = _wall_distances(fit)
    want = {
        (1, 0): fx.truth.width / 2,
        (-1, 0): fx.truth.width / 2,
        (0, 1): fx.truth.depth / 2,
        (0, -1): fx.truth.depth / 2,
    }
    missing = sorted(set(want) - set(got))
    assert not missing, f"{room}: no wall fitted facing {missing}"
    for key, truth in want.items():
        assert got[key] == pytest.approx(truth, abs=DIMENSION_TOL_M), (
            f"{room}: wall facing {key} sits at {got[key]:.4f} m from centre, "
            f"truth {truth:.4f} m ({(got[key] - truth) * 1000:+.0f} mm)"
        )


@pytest.mark.parametrize("room", ["clean", "sparse", "tall"])
def test_ceiling_is_measured_when_it_was_scanned(room):
    fit, fx = _fit(room)
    assert fit.ceiling_source == "returns", (
        f"{room}: a fully scanned ceiling should be a measurement, "
        f"not {fit.ceiling_source!r}"
    )
    assert fit.ceiling_z == pytest.approx(fx.truth.height, abs=DIMENSION_TOL_M)


def test_an_open_capture_reports_no_measured_ceiling():
    """The one thing the carve must never do is roof a room nobody filmed.

    The top of a swept volume is a record of where the phone was pointed. If
    that were allowed to become `ceiling_z`, every truncated capture in the
    world would come back with a confident ceiling height that is really the
    height of the operator's attention.
    """
    fit, _ = _fit("noceiling")
    assert fit.ceiling_z is None
    assert fit.ceiling_source in ("none", "carve")


def test_furniture_does_not_become_architecture():
    """Every fixture is furnished, and none of it may be reported as a wall.

    The discriminator is not size -- a wardrobe is bigger than a doorway -- but
    what lies behind: step through a wall and you have left the room. The
    regression this pins is the one that made the fitter unusable on its first
    run, where the collinear bottom edges of two cupboards at opposite ends of
    a room merged into a single 5.2 m surface floating a metre inside the wall,
    and then shadowed the real wall behind it.
    """
    fit, fx = _fit("clean")
    assert len(fit.walls) == 4, (
        "expected exactly the four walls of a rectangular room, got "
        + ", ".join(f"{l.length:.2f} m at {-l.offset:.2f}" for l in fit.lines)
    )
    assert fit.diagnostics["surfaces_dropped_as_contents"] > 0

    half = np.array([fx.truth.width / 2, fx.truth.depth / 2])
    for line in fit.lines:
        axis = int(np.argmax(np.abs(line.normal)))
        assert abs(line.offset) >= half[axis] - 0.10, (
            f"a surface at {-line.offset:.2f} m was called a wall, but the room's "
            f"boundary on that axis is at {half[axis]:.2f} m"
        )


def test_walls_are_refit_to_the_returns_not_left_on_the_voxel():
    """The carve locates a wall to 5 cm; the returns locate it to millimetres.

    Without the refit every dimension in the twin would inherit the voxel, and
    the pipeline is held to 15 mm.
    """
    fit, _ = _fit("clean")
    assert all(l.source == "returns" for l in fit.lines)
    assert max(l.rms for l in fit.lines) < 0.05


def test_declines_when_there_are_no_poses_to_carve_with():
    fx = fixtures.make_room(seed=1)
    assert roommod.fit_room(fx.points, np.zeros((0, 3)), floor_z=0.0) is None
    assert roommod.fit_room(fx.points, fx.camera_positions[:3], floor_z=0.0) is None


def _shoelace(poly: np.ndarray) -> float:
    q = np.roll(poly, -1, axis=0)
    return float(np.sum(poly[:, 0] * q[:, 1] - q[:, 0] * poly[:, 1]) / 2.0)


@pytest.mark.parametrize("room", ["clean", "sparse", "tall", "noceiling"])
def test_footprint_is_the_room_and_not_a_blob(room):
    """The outline must come back as the walls, closed at the corners.

    This is the regression the cell footprint exists to pin. The old path
    rastered the occupied columns and traced the contour, which on a video
    capture produced a 200-vertex staircase around a rounded blob -- the
    'carved cylinder' twin. An outline read from the wall arrangement has one
    vertex per corner and every edge on a fitted wall, or it declines.
    """
    fit, fx = _fit(room)
    assert fit is not None and fit.footprint is not None, (
        f"{room}: no footprint came back from the cell solve"
    )
    poly = fit.footprint
    assert _shoelace(poly) > 0, f"{room}: footprint is not CCW"
    assert len(poly) <= 8, (
        f"{room}: {len(poly)} vertices for a rectangular room is a trace, not a fit"
    )
    assert _shoelace(poly) == pytest.approx(
        fx.truth.width * fx.truth.depth, rel=0.02
    ), f"{room}: footprint area is off"

    half = np.array([fx.truth.width / 2, fx.truth.depth / 2])
    for p in poly:
        assert np.all(np.abs(np.abs(p) - half) < 0.05), (
            f"{room}: vertex {p} is not a corner of the room"
        )

    assert len(fit.edge_sources) == len(poly)
    assert all(s in ("returns", "carve", "frontier") for s in fit.edge_sources)
    # a fully furnished but fully scanned fixture: every wall stands on returns
    assert fit.edge_sources.count("returns") >= 3, (
        f"{room}: edges came back as {fit.edge_sources}"
    )


def test_footprint_declines_rather_than_guessing():
    """With no cameras there is no interior evidence, and the answer is None,
    not a rectangle drawn around whatever points exist."""
    import locaish.geom.room as rm

    poly, sources, diag = rm.footprint_from_cells(
        [], np.zeros((10, 10), dtype=bool), np.zeros(3), np.full(3, 0.05), np.zeros((0, 2))
    )
    assert poly is None
    assert sources == []
    assert "footprint_declined" in diag


def test_straighten_puts_a_staircase_back_on_its_wall():
    """A rastered outline is a staircase; the walls behind it are not."""
    wall_x, wall_y = 2.0, 1.5
    lines = [
        roommod.WallLine(normal=np.array([-1.0, 0.0]), offset=-wall_x, t0=-wall_y, t1=wall_y),
        roommod.WallLine(normal=np.array([1.0, 0.0]), offset=-wall_x, t0=-wall_y, t1=wall_y),
        roommod.WallLine(normal=np.array([0.0, -1.0]), offset=-wall_y, t0=-wall_x, t1=wall_x),
        roommod.WallLine(normal=np.array([0.0, 1.0]), offset=-wall_y, t0=-wall_x, t1=wall_x),
    ]
    rng = np.random.default_rng(0)
    poly = []
    for t in np.linspace(-wall_x, wall_x, 24):
        poly.append([t, -wall_y + rng.uniform(0, 0.05)])
    for t in np.linspace(-wall_y, wall_y, 18):
        poly.append([wall_x - rng.uniform(0, 0.05), t])
    for t in np.linspace(wall_x, -wall_x, 24):
        poly.append([t, wall_y - rng.uniform(0, 0.05)])
    for t in np.linspace(wall_y, -wall_y, 18):
        poly.append([-wall_x + rng.uniform(0, 0.05), t])
    poly = np.array(poly)

    out, inferred = roommod.straighten_footprint(poly, lines)
    assert len(out) >= 4
    assert len(inferred) == len(out)
    for p in out:
        on_x = abs(abs(p[0]) - wall_x) < 1e-6
        on_y = abs(abs(p[1]) - wall_y) < 1e-6
        assert on_x or on_y, f"vertex {p} did not land on any wall"

    area = 0.5 * abs(
        np.sum(out[:, 0] * np.roll(out[:, 1], -1) - np.roll(out[:, 0], -1) * out[:, 1])
    )
    assert area == pytest.approx(4 * wall_x * wall_y, rel=0.02)
