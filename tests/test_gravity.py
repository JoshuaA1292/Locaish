"""Levelling on the floor, and knowing when the floor is not a floor.

Gravity in this pipeline is settled twice: an axis chosen from the great circle
of wall normals, then a refinement that levels it on the heaviest horizontal
surface. The refinement is refused when the two disagree by more than half a
degree, on the reasoning that a floor and a wall cannot honestly differ by more
than that.

That reasoning holds for a laser scan of a plastered room and fails for a video
reconstruction, where the walls bow by centimetres and three of them may be all
there is. These tests pin the behaviour that replaced it: the walls only get to
overrule the floor when the walls are actually precise enough to have an
opinion, measured from their own scatter rather than assumed.

The second half is about the independent check on the result. A residual that
cannot fail is worse than no residual -- but so is one that fails for the wrong
reason, and on a partial capture the column-bottom ground extraction fits a
plane through furniture and reports its height as tilt.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from locaish.geom import align
from locaish.types import Plane


def _wall(normal, offset, area=20.0):
    n = np.asarray(normal, float)
    return Plane(normal=n / np.linalg.norm(n), offset=offset, kind="wall", area=area,
                 inlier_count=int(area * 500))


def _floor(tilt_deg, area=20.0):
    t = math.radians(tilt_deg)
    return Plane(
        normal=np.array([math.sin(t), 0.0, math.cos(t)]),
        offset=0.0,
        kind="floor",
        area=area,
        inlier_count=int(area * 2000),
    )


# ---------------------------------------------------------------------------
# how precise are the walls?
# ---------------------------------------------------------------------------


def test_exact_walls_scatter_almost_nothing():
    axis = np.array([0.0, 0.0, 1.0])
    planes = [_wall([1, 0, 0], 0), _wall([0, 1, 0], 0), _wall([-1, 0, 0], -4), _wall([0, -1, 0], -3)]

    scatter, n = align._wall_scatter_deg(planes, axis)

    assert n == 4
    assert scatter < 0.01


def test_bowed_walls_scatter_measurably():
    """Each wall leaning a different way is what drift looks like from here."""
    axis = np.array([0.0, 0.0, 1.0])
    planes = [
        _wall([1, 0, 0.05], 0),
        _wall([0, 1, -0.04], 0),
        _wall([-1, 0, 0.06], -4),
        _wall([0, -1, -0.03], -3),
    ]

    scatter, n = align._wall_scatter_deg(planes, axis)

    assert n == 4
    assert 1.0 < scatter < 5.0, f"scatter {scatter:.2f} deg is not in the drifty range"


def test_scatter_is_weighted_by_wall_size():
    """A window reveal must not shout down a wall about which way is up."""
    axis = np.array([0.0, 0.0, 1.0])
    big = [_wall([1, 0, 0], 0, area=30.0), _wall([0, 1, 0], 0, area=30.0)]
    plus_scrap = big + [_wall([1, 0, 0.2], 0, area=0.2)]

    clean, _ = align._wall_scatter_deg(big, axis)
    with_scrap, _ = align._wall_scatter_deg(plus_scrap, axis)

    scrap_tilt = math.degrees(math.asin(0.2 / math.sqrt(1 + 0.2**2)))
    unweighted = math.sqrt(scrap_tilt**2 / 3)

    assert clean < 0.01
    assert with_scrap < unweighted / 5, (
        f"the scrap moved the estimate to {with_scrap:.2f} deg; unweighted it "
        f"would be {unweighted:.2f} deg, so weighting is barely doing anything"
    )


def test_scatter_is_undefined_with_fewer_than_two_walls():
    scatter, n = align._wall_scatter_deg([_wall([1, 0, 0], 0)], np.array([0.0, 0.0, 1.0]))
    assert math.isnan(scatter)
    assert n == 1


# ---------------------------------------------------------------------------
# who wins when the floor and the walls disagree
# ---------------------------------------------------------------------------


def test_sharp_walls_still_overrule_a_wildly_tilted_floor():
    """The original protection has to survive: a tilted scrap is not a floor.

    With exact walls there is no excuse for a two-degree floor, and levelling on
    it would tip an otherwise perfect twin.
    """
    axis = np.array([0.0, 0.0, 1.0])
    planes = [
        _wall([1, 0, 0], 0), _wall([0, 1, 0], 0),
        _wall([-1, 0, 0], -4), _wall([0, -1, 0], -3),
        _floor(2.0),
    ]
    warnings: list[str] = []

    refined, swing = align._refine_on_floor(axis, planes, warnings)

    assert swing == pytest.approx(2.0, abs=0.1)
    assert np.allclose(refined, axis), "gravity must not follow a tilted scrap"
    assert any("one of the two is wrong" in w for w in warnings)


def test_scattered_walls_step_aside_for_the_floor():
    """The new behaviour, and the reason the video path was a degree out.

    The floor sits 1.3 degrees off the walls' vertical -- but those walls
    disagree with each other by more than that, so they cannot resolve the
    dispute and the large, well-sampled surface wins.
    """
    axis = np.array([0.0, 0.0, 1.0])
    planes = [
        _wall([1, 0, 0.05], 0),
        _wall([0, 1, -0.04], 0),
        _wall([-1, 0, 0.06], -4),
        _wall([0, -1, -0.03], -3),
        _floor(1.3),
    ]
    warnings: list[str] = []

    refined, swing = align._refine_on_floor(axis, planes, warnings)

    assert swing == pytest.approx(1.3, abs=0.1)
    assert not np.allclose(refined, axis), "the floor should have won"
    assert refined[0] > 0, "the refined axis must lean the way the floor does"
    assert any("cannot resolve" in w for w in warnings)


def test_a_floor_beyond_all_tolerance_is_still_refused():
    """However bad the walls are, some disagreements are a fitting failure."""
    axis = np.array([0.0, 0.0, 1.0])
    planes = [
        _wall([1, 0, 0.1], 0), _wall([0, 1, -0.1], 0),
        _wall([-1, 0, 0.12], -4), _wall([0, -1, -0.09], -3),
        _floor(8.0),
    ]
    warnings: list[str] = []

    refined, _ = align._refine_on_floor(axis, planes, warnings)

    assert np.allclose(refined, axis)
    assert any("one of the two is wrong" in w for w in warnings)


def test_a_small_disagreement_is_taken_without_comment():
    axis = np.array([0.0, 0.0, 1.0])
    planes = [_wall([1, 0, 0], 0), _wall([0, 1, 0], 0), _floor(0.2)]
    warnings: list[str] = []

    refined, swing = align._refine_on_floor(axis, planes, warnings)

    assert swing == pytest.approx(0.2, abs=0.05)
    assert refined[0] > 0
    assert warnings == []


# ---------------------------------------------------------------------------
# the independent residual has to be able to say "I don't know"
# ---------------------------------------------------------------------------


def _flat_floor_cloud(tilt_deg=0.0, n=60_000, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 4, (n, 2))
    t = math.radians(tilt_deg)
    z = xy[:, 0] * math.tan(t) + rng.normal(0, 0.003, n)
    walls = np.column_stack([
        rng.choice([0.0, 4.0], n // 4),
        rng.uniform(0, 4, n // 4),
        rng.uniform(0.1, 2.5, n // 4),
    ])
    return np.concatenate([np.column_stack([xy, z]), walls])


def test_the_residual_reports_a_real_tilt():
    angle, cells = align.independent_up_residual(_flat_floor_cloud(tilt_deg=1.5))
    assert cells > 12
    assert angle == pytest.approx(1.5, abs=0.3)


def test_the_residual_reports_level_when_level():
    angle, _ = align.independent_up_residual(_flat_floor_cloud(tilt_deg=0.0))
    assert angle < 0.2


def test_the_residual_refuses_when_the_ground_cells_are_clutter():
    """A partial capture bottoms out on furniture, not floor.

    The fitted plane then scatters by tens of centimetres and the angle it
    reports is the height of a chair. Reporting that as tilt straddles a QA gate
    and contradicts the better estimator working on the same twin, so the check
    has to know it failed.
    """
    rng = np.random.default_rng(2)
    # Every 30 cm cell bottoms out on something at its own height -- a desk here,
    # a chair there, a cable tray -- and the real floor is never visible. Note
    # that uniform noise would *not* do: its low envelope is flat, so a cloud of
    # it has a perfectly good floor at the bottom.
    cells, per = 18, 220
    cols = []
    for cx in range(cells):
        for cy in range(cells):
            base = rng.uniform(0.0, 0.55)
            cols.append(np.column_stack([
                rng.uniform(cx * 0.3, (cx + 1) * 0.3, per),
                rng.uniform(cy * 0.3, (cy + 1) * 0.3, per),
                base + rng.uniform(0.0, 0.25, per),
            ]))
    cloud = np.concatenate(cols)

    angle, _ = align.independent_up_residual(cloud)

    assert math.isnan(angle), f"reported {angle:.2f} deg from a cloud with no floor"


def test_the_residual_refuses_on_too_few_points():
    assert math.isnan(align.independent_up_residual(np.zeros((10, 3)))[0])
