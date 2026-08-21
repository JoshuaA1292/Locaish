"""The QA report has to discriminate, not decorate.

A quality report that says "pass" on every scan is worse than no report: it
converts an unknown into a false assurance. These tests check that the bad
fixtures come out bad, that they come out bad *for the right reason*, and that
the good ones are not gratuitously failed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from support import ingest_fixture


def _statuses(twin) -> dict[str, str]:
    return {c["name"]: c["status"] for c in twin.qa.checks}


def _flagged(twin) -> set[str]:
    return {c["name"] for c in twin.qa.checks if c["status"] in {"warn", "fail"}}


@pytest.mark.parametrize("room", ["clean", "tilted", "tall"])
def test_a_good_scan_is_not_failed(room):
    result, _ = ingest_fixture(room)
    assert result.twin.qa.verdict in {"pass", "warn"}, (
        f"{room} was failed by: "
        + "; ".join(c["message"] for c in result.twin.qa.failures())
    )


def test_drift_is_detected():
    """SLAM drift bows a flat wall. It is the error mode that most convincingly
    fakes a good scan -- everything looks plausible, the room is just quietly
    the wrong shape -- so it must be named explicitly."""
    result, _ = ingest_fixture("drifty")
    assert result.twin.qa.verdict in {"warn", "fail"}
    flagged = _flagged(result.twin)
    assert flagged & {"drift", "wall_planarity"}, (
        f"drifty scan flagged nothing about its geometry; flagged {flagged}"
    )
    assert result.twin.qa.metrics.get("drift_estimate_m", 0.0) > 0.01


def test_thin_coverage_is_detected():
    result, _ = ingest_fixture("sparse")
    assert result.twin.qa.verdict in {"warn", "fail"}
    flagged = _flagged(result.twin)
    assert flagged & {"density", "coverage", "capture_bounds"}, (
        f"sparse scan flagged nothing about coverage; flagged {flagged}"
    )


def test_manhattan_never_fails_a_room_for_being_round():
    """A curved or angled room is architecture, not an error. If this check can
    fail, every non-rectangular location gets rejected by a pipeline that has
    no business having an opinion about the building's floor plan."""
    for room in ["clean", "tilted", "tall", "drifty", "sparse"]:
        result, _ = ingest_fixture(room)
        status = _statuses(result.twin).get("manhattan")
        assert status in {None, "info", "pass"}, f"{room}: manhattan status {status}"


def test_metrics_are_present_and_finite():
    result, _ = ingest_fixture("clean")
    m = result.twin.qa.metrics
    required = [
        "point_count",
        "density_per_m2",
        "median_spacing_m",
        "floor_area_m2",
        "floor_coverage",
        "gravity_residual_deg",
        "drift_estimate_m",
        "scale_confidence",
        "opening_count",
    ]
    missing = [k for k in required if k not in m]
    assert not missing, f"QA report is missing metrics: {missing}"
    bad = {k: v for k, v in m.items() if isinstance(v, float) and math.isinf(v)}
    assert not bad, f"non-finite metrics: {bad}"


def test_every_check_message_quotes_a_number():
    """A check that says only 'coverage is low' cannot be acted on. Each
    message has to carry the measured value so an operator knows how far off
    the scan was and whether a re-scan would help."""
    result, _ = ingest_fixture("sparse")
    for c in result.twin.qa.checks:
        assert any(ch.isdigit() for ch in c["message"]), (
            f"check {c['name']!r} has no number in it: {c['message']!r}"
        )
        assert len(c["message"]) > 25, f"check {c['name']!r} message is too terse"


def test_measurement_reports_honest_uncertainty():
    from locaish.scan.qa import verify_measurement

    result, fx = ingest_fixture("clean")
    twin = result.twin
    a = np.array([-1.0, 0.0, 1.0])
    b = np.array([1.0, 0.0, 1.0])
    r = verify_measurement(twin, a, b)
    assert r["distance_m"] == pytest.approx(2.0, abs=1e-6)
    assert 0.0 < r["uncertainty_m"] < 0.25, (
        f"uncertainty {r['uncertainty_m']} is either dishonestly zero or useless"
    )
    assert "within_capture_bounds" in r


def test_report_renders_without_colour_codes_when_asked():
    from locaish.scan.qa import format_report

    result, _ = ingest_fixture("clean")
    text = format_report(result.twin.qa, color=False)
    assert "\x1b[" not in text
    assert result.twin.qa.verdict in text.lower()
