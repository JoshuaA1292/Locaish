"""The search phase: the sweep's physics, the schema it must match, the sun.

None of these tests needs ClickHouse or a network. The warehouse test pins the
*contract* between the sweep's columns and the table DDL, because that is the
break that would otherwise only surface as a failed insert at demo time.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import numpy as np
import pytest

from locaish import warehouse
from locaish.film import daylight, optics
from locaish.film import sweep as sweepmod

# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def swept(clean):
    result, _fx = clean
    return sweepmod.sweep(result.twin)


def test_sweep_columns_are_parallel_and_named_for_the_table(swept):
    """Every column the DDL declares, exactly once, all the same length.

    The insert is column-oriented and positional over `columns.keys()`, so a
    drifted name or a missing column is data corruption, not an error message.
    """
    ddl_names = re.findall(r"^\s{4}(\w+)\s", warehouse.DDL, re.MULTILINE)
    assert ddl_names, "the DDL regex found no columns; the test itself broke"
    assert list(swept.columns.keys()) == ddl_names

    n = len(swept)
    assert n > 1000
    for name, col in swept.columns.items():
        assert len(col) == n, f"column {name} is not parallel"


def test_sweep_setups_are_physically_coherent(swept):
    c = swept.columns
    dist = np.linalg.norm(
        np.stack([c["subj_x"] - c["cam_x"], c["subj_y"] - c["cam_y"]], axis=1), axis=1
    )
    # distance_m is 3D (camera height to eyeline) so it can only exceed plan distance.
    assert (c["distance_m"] >= dist - 1e-3).all()
    assert (c["distance_m"] >= sweepmod.MIN_DISTANCE_M - 1e-6).all()
    assert set(np.unique(c["focal_mm"])) <= set(sweepmod.SWEEP_PRIMES_MM)
    assert set(np.unique(c["shot_size"])) <= {s.key for s in optics.SHOT_SIZES}
    fill = c["subject_fill"]
    assert (fill >= sweepmod.MIN_SUBJECT_FILL).all() and (fill <= sweepmod.MAX_SUBJECT_FILL).all()


def test_framing_arithmetic_matches_the_optics_module(swept):
    """The sweep vectorises the thin-lens maths; optics.py is the reference."""
    c = swept.columns
    i = int(np.argmax(c["score"]))
    sensor = optics.SENSORS[str(c["sensor"][i])]
    expected = optics.framed_height_m(sensor, float(c["focal_mm"][i]), float(c["distance_m"][i]))
    assert float(c["framed_height_m"][i]) == pytest.approx(expected, rel=1e-5)

    near, far, _ = optics.depth_of_field(
        sensor, float(c["focal_mm"][i]), sweepmod.WORKING_APERTURE_F, float(c["distance_m"][i])
    )
    assert float(c["dof_near_m"][i]) == pytest.approx(near, rel=1e-4)
    if np.isfinite(far):
        assert float(c["dof_far_m"][i]) == pytest.approx(far, rel=1e-4)
    else:
        assert c["dof_infinite"][i] == 1


def test_blocked_sightlines_score_zero_but_stay_in_the_table(swept):
    c = swept.columns
    blocked = c["visible"] == 0
    if blocked.any():
        assert float(c["score"][blocked].max()) == 0.0
    assert float(c["score"][c["visible"] == 1].min()) >= 0.0
    assert float(c["score"].max()) <= 100.0


# ---------------------------------------------------------------------------
# daylight
# ---------------------------------------------------------------------------


def test_solar_position_matches_the_almanac():
    """Equinox noon in London: the sun is due south at 90 - latitude degrees."""
    az, el = daylight.solar_position(
        51.5074, 0.0, datetime(2026, 3, 20, 12, 2, tzinfo=timezone.utc)
    )
    assert el == pytest.approx(90.0 - 51.5074, abs=1.0)
    assert az == pytest.approx(180.0, abs=3.0)


def test_summer_sun_is_higher_than_winter_sun():
    _, summer = daylight.solar_position(40.0, 0.0, datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc))
    _, winter = daylight.solar_position(40.0, 0.0, datetime(2026, 12, 21, 12, 0, tzinfo=timezone.utc))
    assert summer - winter == pytest.approx(2 * 23.44, abs=1.0)


def test_sun_schedule_requires_a_georeference(clean):
    result, _fx = clean
    twin = result.twin
    assert twin.georeference is None
    with pytest.raises(ValueError, match="georeference"):
        daylight.sun_schedule(twin, "2026-09-09")


def test_sun_schedule_reads_the_twin_windows(clean):
    from locaish.types import Georeference

    result, _fx = clean
    twin = result.twin
    old = twin.georeference
    twin.georeference = Georeference(
        latitude=34.05, longitude=-118.24, heading_deg=90.0, heading_source="user"
    )
    try:
        s = daylight.sun_schedule(twin, "2026-09-09")
    finally:
        twin.georeference = old

    assert s["sun_up"], "the sun rises in Los Angeles in September"
    assert s["golden_hour"]
    assert 0 < s["max_elevation_deg"] < 90
    for w in s["windows"]:
        assert w["faces"] in daylight.COMPASS
        # A vertical pane cannot receive direct sun while the sun is down.
        for run in w["direct_sun"]:
            assert run["from"] >= s["sun_up"][0]["from"]
            assert run["to"] <= s["sun_up"][-1]["to"]


def test_intervals_finds_runs_and_edges():
    minutes = np.arange(0, 60, 10)
    mask = np.array([True, True, False, False, True, True])
    runs = daylight._intervals(minutes, mask)
    assert runs == [
        {"from": "00:00", "to": "00:10"},
        {"from": "00:40", "to": "00:50"},
    ]
    assert daylight._intervals(minutes, np.zeros(6, dtype=bool)) == []
