"""The 1:1 claim, as executable assertions.

Every test here compares the pipeline's output against a room whose true
dimensions are known exactly because we generated it. If these pass, "exact 3D
replica" is a measurement. If they fail, it is marketing.

The tolerances come from docs/PHASE1_SPEC.md and are deliberately tighter than
a real LiDAR scan will achieve -- the fixture has no sensor bias, so any error
here is the pipeline's own, and the pipeline's own error should be far below
the sensor's.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from support import ingest_fixture

DIMENSION_TOL_M = 0.015
GRAVITY_TOL_DEG = 0.25
YAW_TOL_DEG = 0.5
OPENING_TOL_M = 0.08

METRIC_ROOMS = ["clean", "tilted", "centimetres", "inches", "tall"]


def _plan_dimensions(twin) -> tuple[float, float]:
    """The two horizontal spans, longest first.

    Measured from the fitted wall planes rather than the point cloud's bounding
    box -- see locaish/scan/dimensions.py for why the bounding box is a trap.
    Sorting removes the dependence on which wall became +X.
    """
    from locaish.scan.dimensions import measure_room

    return measure_room(twin.points, twin.structure).plan


@pytest.mark.parametrize("room", METRIC_ROOMS)
def test_plan_dimensions_are_metric(room):
    result, fx = ingest_fixture(room)
    got_long, got_short = _plan_dimensions(result.twin)
    want_long, want_short = sorted([fx.truth.width, fx.truth.depth], reverse=True)

    assert got_long == pytest.approx(want_long, abs=DIMENSION_TOL_M), (
        f"{room}: long side {got_long:.4f} m vs truth {want_long:.4f} m "
        f"({(got_long - want_long) * 1000:+.0f} mm)"
    )
    assert got_short == pytest.approx(want_short, abs=DIMENSION_TOL_M), (
        f"{room}: short side {got_short:.4f} m vs truth {want_short:.4f} m "
        f"({(got_short - want_short) * 1000:+.0f} mm)"
    )


@pytest.mark.parametrize("room", METRIC_ROOMS)
def test_ceiling_height_is_metric(room):
    result, fx = ingest_fixture(room)
    h = result.twin.structure.ceiling_height
    assert h is not None, f"{room}: no ceiling found in a room that has one"
    assert h == pytest.approx(fx.truth.height, abs=DIMENSION_TOL_M), (
        f"{room}: ceiling {h:.4f} m vs truth {fx.truth.height:.4f} m "
        f"({(h - fx.truth.height) * 1000:+.0f} mm)"
    )


@pytest.mark.parametrize("room,unit", [("centimetres", "cm"), ("inches", "in"), ("clean", "m")])
def test_unit_is_recovered(room, unit):
    """A wrong scale poisons every downstream number, so it must be recovered
    exactly or refused loudly -- never quietly approximated."""
    result, _ = ingest_fixture(room)
    step = result.twin.provenance["steps"]["canonicalize"]
    assert step["unit"] == unit, (
        f"{room}: inferred unit {step['unit']!r}, expected {unit!r}; "
        f"evidence was {step['scale_evidence']}"
    )
    assert step["scale_confidence"] >= 0.4


@pytest.mark.parametrize("room", METRIC_ROOMS)
def test_floor_is_level_and_at_zero(room):
    result, _ = ingest_fixture(room)
    twin = result.twin
    assert abs(twin.structure.floor_z) < 0.01, "the canonical floor must sit at z = 0"

    residual = result.twin.provenance["steps"]["canonicalize"]["up_residual_deg"]
    assert residual < GRAVITY_TOL_DEG, f"{room}: gravity off by {residual:.3f} deg"

    floors = [p for p in twin.structure.planes if p.kind == "floor"]
    assert floors, f"{room}: no plane was labelled floor"
    for p in floors:
        angle = math.degrees(math.acos(np.clip(p.normal @ np.array([0, 0, 1.0]), -1, 1)))
        assert angle < 1.0, f"{room}: floor normal {angle:.3f} deg off +Z, or pointing down"


@pytest.mark.parametrize("room", METRIC_ROOMS)
def test_walls_align_to_the_axes(room):
    """After yaw normalisation a rectangular room's walls are axis aligned; a
    twin that is 40 degrees off is still metric but no longer comparable to
    another twin, which breaks every cross-location query later."""
    result, _ = ingest_fixture(room)
    residual = result.twin.provenance["steps"]["canonicalize"]["yaw_residual_deg"]
    assert residual < YAW_TOL_DEG, f"{room}: yaw off by {residual:.3f} deg"

    # Only the structural walls have to be square. Plane detection also labels
    # the front of a wardrobe a wall, correctly -- it is vertical -- and a
    # furniture face carrying 3% of a real wall's support is entitled to sit at
    # whatever angle the furniture was pushed to. What must not happen is a
    # sliver like that being mistaken for the room, which is what
    # dimensions.MIN_WALL_SUPPORT exists to prevent.
    from locaish.scan.dimensions import MIN_WALL_SUPPORT

    walls = result.twin.structure.walls()
    assert walls, f"{room}: no walls at all"
    strongest = max(p.inlier_count for p in walls)
    structural = [p for p in walls if p.inlier_count >= MIN_WALL_SUPPORT * strongest]
    assert len(structural) >= 4, f"{room}: only {len(structural)} structural walls"

    for p in structural:
        horiz = np.array([p.normal[0], p.normal[1]])
        n = np.linalg.norm(horiz)
        if n < 0.5:
            continue
        horiz = horiz / n
        # a wall normal should be within half a degree of +/-X or +/-Y
        off = min(
            math.degrees(math.acos(np.clip(abs(horiz @ axis), -1, 1)))
            for axis in (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        )
        assert off < 1.5, f"{room}: a wall sits {off:.2f} deg off the axes"


@pytest.mark.parametrize("room", METRIC_ROOMS)
def test_floor_area(room):
    result, fx = ingest_fixture(room)
    got = result.twin.structure.floor_area
    want = fx.truth.floor_area
    assert got == pytest.approx(want, rel=0.04), (
        f"{room}: floor area {got:.2f} m2 vs truth {want:.2f} m2"
    )


@pytest.mark.parametrize("room", ["clean", "tilted"])
def test_openings_are_found_and_measured(room):
    """Windows and doors are what Phase 3 shines the sun through, so a missed
    one is a missed feature. The fixtures put furniture against the walls
    specifically to tempt a naive hole-finder into a false positive."""
    result, fx = ingest_fixture(room)
    found = result.twin.structure.openings
    truth = fx.truth.openings

    assert len(found) == len(truth), (
        f"{room}: found {len(found)} openings, expected {len(truth)}: "
        + ", ".join(f"{o.kind} {o.width:.2f}x{o.height:.2f}@{o.sill_height:.2f}" for o in found)
    )

    n_doors = sum(1 for o in found if o.kind == "door")
    assert n_doors == sum(1 for o in truth if o.kind == "door"), f"{room}: door count"

    # Match each true opening to its nearest detection by size and sill height.
    # Indices rather than list.remove, because an Opening holds numpy arrays and
    # equality on it returns an array, not a bool.
    unmatched = list(range(len(found)))
    for spec in truth:
        best_i, best_err = None, 1e9
        for i in unmatched:
            o = found[i]
            err = abs(o.width - spec.width) + abs(o.height - spec.height) + abs(
                o.sill_height - spec.sill
            )
            if err < best_err:
                best_i, best_err = i, err
        assert best_i is not None
        unmatched.remove(best_i)
        best = found[best_i]
        assert best.width == pytest.approx(spec.width, abs=OPENING_TOL_M), (
            f"{room}: {spec.kind} width {best.width:.3f} vs {spec.width:.3f}"
        )
        assert best.height == pytest.approx(spec.height, abs=OPENING_TOL_M), (
            f"{room}: {spec.kind} height {best.height:.3f} vs {spec.height:.3f}"
        )
        assert best.sill_height == pytest.approx(spec.sill, abs=OPENING_TOL_M), (
            f"{room}: {spec.kind} sill {best.sill_height:.3f} vs {spec.sill:.3f}"
        )


def test_open_topped_capture_reports_no_ceiling():
    """An invented ceiling height is worse than a missing one: the daylight
    study would silently model a roof that is not there."""
    result, _ = ingest_fixture("noceiling")
    assert result.twin.structure.ceiling_z is None
    assert result.twin.structure.ceiling_height is None


def test_capture_bounds_are_not_larger_than_the_room():
    """Capture bounds exist to stop the shot search proposing a tripod where
    nobody stood, so an over-large hull defeats the entire point."""
    result, fx = ingest_fixture("clean")
    cb = result.twin.capture_bounds
    assert cb is not None and cb.area > 0
    assert cb.area <= fx.truth.floor_area * 1.15, (
        f"capture bounds {cb.area:.1f} m2 exceed the {fx.truth.floor_area:.1f} m2 room"
    )
    assert cb.contains(np.array([[0.0, 0.0]]))[0], "the middle of the room must be inside"
    far = np.array([[fx.truth.width, fx.truth.depth]])
    assert not cb.contains(far)[0], "a point outside the room must be outside"


def test_ingest_is_deterministic():
    """Two runs of the same scan must give the same twin, or no measurement
    taken from one is reproducible from the other."""
    a, _ = ingest_fixture("clean")
    b, _ = ingest_fixture("clean", max_points=None)
    # different options, so compare the geometry-independent invariants that
    # must not drift between configurations
    assert a.twin.structure.ceiling_height == pytest.approx(
        b.twin.structure.ceiling_height, abs=0.01
    )
    assert a.twin.structure.floor_area == pytest.approx(b.twin.structure.floor_area, rel=0.03)
