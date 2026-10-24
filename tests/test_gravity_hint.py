"""A close vertical-axis vote is a coin toss only when nothing else voted.

The device's own orientation is an independent witness: the geometric vote
reads the room's shape, the hint reads how the phone was held. When the vote
was close and a coherently held phone picked the same axis, the twin must say
"near-tie broken by a measurement" rather than fail; when the hint was
incoherent, or disagreed, or was absent, the fail stands.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from locaish.scan.ingest import _check_gravity_decision
from locaish.types import QAReport


def _status(canon):
    report = QAReport()
    _check_gravity_decision(report, canon)
    return next(c for c in report.checks if c["name"] == "gravity_axis")


def _canon(margin, agreement=None, coherence=None):
    return SimpleNamespace(
        gravity_axis_margin=margin,
        gravity_sign_margin=0.9,
        gravity_hint_agreement=agreement,
        gravity_hint_coherence=coherence,
    )


def test_a_close_vote_with_no_hint_still_fails():
    assert _status(_canon(0.06))["status"] == "fail"


def test_a_coherent_agreeing_hint_turns_the_coin_toss_into_a_warning():
    check = _status(_canon(0.06, agreement=math.cos(math.radians(2.0)), coherence=0.97))
    assert check["status"] == "warn"
    assert "independently picks the same axis" in check["message"]


def test_an_incoherent_hint_does_not_rescue_the_vote():
    assert _status(_canon(0.06, agreement=0.999, coherence=0.80))["status"] == "fail"


def test_a_disagreeing_hint_does_not_rescue_the_vote():
    """A hint 20 degrees off the chosen axis is evidence of nothing."""
    assert _status(_canon(0.06, agreement=math.cos(math.radians(20.0)), coherence=0.98))["status"] == "fail"


def test_a_clear_vote_is_unaffected_by_the_hint():
    assert _status(_canon(0.4, agreement=0.999, coherence=0.98))["status"] == "pass"


def test_the_hint_is_recorded_in_the_metrics():
    report = QAReport()
    _check_gravity_decision(report, _canon(0.3, agreement=0.998, coherence=0.95))
    assert report.metrics["gravity_hint_agreement"] == 0.998
    assert report.metrics["gravity_hint_coherence"] == 0.95
