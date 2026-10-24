"""The stature anchor: people in the frames as a fourth, independent scale."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from locaish.meta import Detection, FakeMetaClient
from locaish.meta import stature as S
from locaish.product import labels as L
from locaish.video import metric as M
from test_labels import _frames


# ---------------------------------------------------------------------------
# scale_from_stature
# ---------------------------------------------------------------------------


def test_three_consistent_readings_give_the_population_mean_over_the_median():
    est = M.scale_from_stature([1.0, 1.02, 0.98])
    assert est is not None
    assert est.source == "stature"
    assert est.factor == pytest.approx(M.STATURE_MEAN_M / 1.0, rel=1e-9)
    assert est.frames_used == 3 and est.frames_rejected == 0
    assert est.prior_bias == M.STATURE_LOG_SIGMA
    # Close readings leave the bar near the population prior: the prior is the
    # floor, the per-reading noise and the small spread add a little on top.
    assert M.STATURE_LOG_SIGMA <= est.total_log_sigma < 0.10
    assert est.to_dict()["frames_used"] == 3


def test_disagreeing_readings_widen_the_sigma():
    tight = M.scale_from_stature([1.0, 1.01, 0.99, 1.0])
    wild = M.scale_from_stature([1.0, 1.6, 0.7, 1.3])
    assert tight is not None and wild is not None
    assert wild.total_log_sigma > 2 * tight.total_log_sigma
    assert wild.confidence < tight.confidence
    assert any("scatter" in w for w in wild.warnings)


def test_more_readings_shrink_the_measured_part_but_never_the_prior():
    few = M.scale_from_stature([1.0, 1.0])
    many = M.scale_from_stature([1.0] * 50)
    assert few.log_spread > many.log_spread
    assert many.total_log_sigma >= M.STATURE_LOG_SIGMA
    assert many.total_log_sigma == pytest.approx(math.hypot(0.08 / math.sqrt(50), M.STATURE_LOG_SIGMA))


def test_fewer_than_two_readings_is_not_an_anchor():
    assert M.scale_from_stature([]) is None
    assert M.scale_from_stature([1.0]) is None
    assert M.scale_from_stature([1.0, float("nan"), -2.0]) is None


def test_implausible_people_are_rejected_under_the_current_scale():
    # Under 0.5 m per unit, a 1.0-unit reading is a 0.5 m "person" (a child or
    # a seated torso) and a 6.0-unit one is a 3 m giant; both are dropped.
    est = M.scale_from_stature([3.4, 3.5, 1.0, 6.0, 3.3], current_factor=0.5)
    assert est is not None
    assert est.frames_used == 3 and est.frames_rejected == 2
    assert est.factor == pytest.approx(M.STATURE_MEAN_M / 3.4)
    assert any("2 of 5" in w for w in est.warnings)
    # Reject enough and nothing is left.
    assert M.scale_from_stature([1.0, 6.0], current_factor=0.5) is None
    # Without a current scale nothing is judged implausible.
    assert M.scale_from_stature([1.0, 6.0]).frames_rejected == 0


def test_combining_with_a_disagreeing_camera_height_inflates_the_uncertainty():
    camera = M.ScaleEstimate(factor=1.0, confidence=0.6, log_spread=0.0,
                             source="camera-height", prior_bias=M.CAMERA_HEIGHT_LOG_BIAS)
    people = M.scale_from_stature([1.0, 1.02, 0.98])          # says 1.70: 70% apart
    assert people is not None
    combined = M.combine_scales([camera, people])
    assert combined.source == "combined"
    assert combined.log_spread > min(camera.total_log_sigma, people.total_log_sigma)
    # Inverse-variance alone would have narrowed the bar below the tighter input;
    # the disagreement forces it wider than the *looser* input too.
    assert combined.log_spread > max(camera.total_log_sigma, people.total_log_sigma)
    assert any("disagree" in w for w in combined.warnings)
    assert {c["source"] for c in combined.components} == {"camera-height", "stature"}
    # The factor sits between the two: neither estimator won outright.
    assert camera.factor < combined.factor < people.factor


def test_combining_with_an_agreeing_camera_height_tightens_the_bar():
    camera = M.ScaleEstimate(factor=1.70, confidence=0.6, log_spread=0.0,
                             source="camera-height", prior_bias=M.CAMERA_HEIGHT_LOG_BIAS)
    people = M.scale_from_stature([1.0, 1.0, 1.0, 1.0])
    combined = M.combine_scales([camera, people])
    assert combined.log_spread < min(camera.total_log_sigma, people.total_log_sigma)
    assert combined.factor == pytest.approx(1.70)


# ---------------------------------------------------------------------------
# height_readings: the geometry
# ---------------------------------------------------------------------------


def _standing_camera(height_units: float, depth_units: float, *, pitch_deg: float = 0.0,
                     factor: float = 0.5, t=None, n: int = 1) -> L.PosedFrames:
    """One pinhole camera at (-depth, 0, height) in colmap units, looking along
    +x and pitched down by `pitch_deg`, in a twin scaled by `factor`."""
    K = np.array([[800.0, 0, 400.0], [0, 800.0, 300.0], [0, 0, 1.0]])
    c = np.array([-depth_units, 0.0, height_units])
    p = math.radians(pitch_deg)
    fwd = np.array([math.cos(p), 0.0, -math.sin(p)])
    up = np.array([0, 0, 1.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    Rc = np.stack([right, down, fwd])
    E = np.zeros((3, 4)); E[:, :3] = Rc; E[:, 3] = -Rc @ c
    t = np.array([1.0, -2.0, 0.3]) if t is None else np.asarray(t, dtype=np.float64)
    return L.PosedFrames(names=[f"f{k}.jpg" for k in range(n)], paths=[Path(f"f{k}.jpg") for k in range(n)],
                         extr=np.stack([E] * n), K=np.stack([K] * n), size=(800, 600),
                         R=np.eye(3), t=t, factor=factor)


def _person_box(fr: L.PosedFrames, i: int, x: float, y: float, stature_units: float,
                *, half_width_units: float = 0.2) -> tuple[float, float, float, float]:
    """The normalised xyxy box of a `stature_units`-tall person standing at
    colmap (x, y) on the colmap floor z=0, as frame `i` sees it."""
    w, h = fr.size
    feet = fr.to_twin(np.array([x, y, 0.0]))
    head = fr.to_twin(np.array([x, y, stature_units]))
    uf, vf, df = fr.project(feet)
    uh, vh, dh = fr.project(head)
    assert df[i] > 0 and dh[i] > 0
    half_px = half_width_units * fr.factor * 800.0 / df[i]
    return (float((uf[i] - half_px) / w), float(vh[i] / h), float((uf[i] + half_px) / w), float(vf[i] / h))


def test_height_readings_recover_a_standing_person_in_reconstruction_units():
    fr = _standing_camera(height_units=1.2, depth_units=5.0, factor=0.5)
    floor_z_twin = float(fr.to_twin(np.zeros(3))[2])          # colmap z=0 lands at twin z=0.3
    assert floor_z_twin == pytest.approx(0.3)
    box = _person_box(fr, 0, 0.0, 0.0, 1.75)
    assert 0.25 < box[3] - box[1] < 0.99, "the synthetic person fills a good part of the frame"
    dets = {0: [Detection("person", box, score=0.9)]}
    readings = S.height_readings(fr, dets, floor_z_twin)
    assert len(readings) == 1
    # The reading is in colmap units (1.75), not twin units (0.875).
    assert readings[0] == pytest.approx(1.75, rel=0.02)
    # Under the provisional 0.5 m per unit this person would be 0.875 m tall,
    # so the sanity window drops the reading; a provisional scale that is off
    # by less than the window's reach lets the anchor correct it to 1.70/1.75.
    assert M.scale_from_stature(readings * 2, current_factor=fr.factor) is None
    est = M.scale_from_stature(readings * 2, current_factor=0.8)
    assert est is not None and est.factor == pytest.approx(1.70 / 1.75, rel=0.02)


def test_height_readings_do_not_depend_on_the_provisional_factor():
    """The same scene posed with factor=1.0 and with factor=0.5 reads the same."""
    a = _standing_camera(1.3, 4.0, factor=1.0, t=np.zeros(3))
    b = _standing_camera(1.3, 4.0, factor=0.5, t=np.array([0.0, 0.0, 2.0]))
    box_a = _person_box(a, 0, 0.0, 0.3, 1.62)
    box_b = _person_box(b, 0, 0.0, 0.3, 1.62)
    assert np.allclose(box_a, box_b), "a similarity transform of the twin does not move pixels"
    ra = S.height_readings(a, {0: [Detection("person", box_a, 0.8)]}, 0.0)
    rb = S.height_readings(b, {0: [Detection("person", box_b, 0.8)]}, 2.0)
    assert ra[0] == pytest.approx(1.62, rel=0.02)
    assert rb[0] == pytest.approx(ra[0], rel=1e-6)


def test_height_readings_are_exact_under_pitch_and_off_centre():
    fr = _standing_camera(height_units=1.5, depth_units=4.5, pitch_deg=12.0, factor=0.7)
    floor_z = float(fr.to_twin(np.zeros(3))[2])
    boxes = [_person_box(fr, 0, 0.0, -0.8, 1.80), _person_box(fr, 0, 0.6, 0.9, 1.58)]
    readings = S.height_readings(fr, {0: [Detection("person", b, 0.95) for b in boxes]}, floor_z)
    assert len(readings) == 2
    assert readings[0] == pytest.approx(1.80, rel=0.02)
    assert readings[1] == pytest.approx(1.58, rel=0.02)


def test_height_readings_accept_a_general_floor_normal():
    # Tilt the whole twin by a rotation: the floor's normal and offset follow.
    ang = math.radians(25.0)
    Rt = np.array([[1, 0, 0], [0, math.cos(ang), -math.sin(ang)], [0, math.sin(ang), math.cos(ang)]])
    fr = _standing_camera(1.2, 5.0, factor=0.5)
    fr = L.PosedFrames(names=fr.names, paths=fr.paths, extr=fr.extr, K=fr.K, size=fr.size,
                       R=Rt, t=fr.t, factor=fr.factor)
    normal = Rt @ np.array([0, 0, 1.0])
    floor_offset = float(fr.to_twin(np.zeros(3)) @ normal)
    box = _person_box(fr, 0, 0.0, 0.0, 1.75)
    readings = S.height_readings(fr, {0: [Detection("person", box, 0.9)]}, floor_offset, floor_normal=normal)
    assert readings[0] == pytest.approx(1.75, rel=0.02)


def test_cut_off_small_unsure_and_non_person_boxes_are_skipped():
    fr = _standing_camera(1.2, 5.0, factor=0.5)
    floor_z = float(fr.to_twin(np.zeros(3))[2])
    good = _person_box(fr, 0, 0.0, 0.0, 1.75)
    x1, y1, x2, y2 = good
    cut_bottom = (x1, y1, x2, 1.0)
    cut_top = (x1, 0.0, x2, y2)
    small = (x1, y2 - 0.2, x2, y2)
    dets = {0: [
        Detection("person", cut_bottom, 0.9),
        Detection("person", cut_top, 0.9),
        Detection("person", small, 0.9),
        Detection("person", good, 0.3),
        Detection("chair", good, 0.9),
        Detection("person", good, 0.9),
    ]}
    readings = S.height_readings(fr, dets, floor_z)
    assert len(readings) == 1 and readings[0] == pytest.approx(1.75, rel=0.02)
    # A frame index the frames do not have is ignored, not an error.
    assert S.height_readings(fr, {7: [Detection("person", good, 0.9)]}, floor_z) == []


def test_a_box_whose_feet_ray_misses_the_floor_yields_nothing():
    # The camera looks up: the bottom pixel's ray never reaches the floor.
    fr = _standing_camera(1.2, 5.0, pitch_deg=-40.0, factor=0.5)
    floor_z = float(fr.to_twin(np.zeros(3))[2])
    assert S.height_readings(fr, {0: [Detection("person", (0.4, 0.3, 0.6, 0.7), 0.9)]}, floor_z) == []


# ---------------------------------------------------------------------------
# detect_people
# ---------------------------------------------------------------------------


def test_detect_people_maps_detections_to_frame_indices(tmp_path):
    fr = _frames(n=9)
    fr.paths = [tmp_path / name for name in fr.names]
    for p in fr.paths:
        p.write_bytes(b"\x89PNG\r\n\x1a\n not really a png")
    client = FakeMetaClient(detections=[Detection("person", (0.4, 0.2, 0.6, 0.8), 0.9),
                                        Detection("chair", (0.1, 0.5, 0.3, 0.9), 0.9)])
    log: list[str] = []
    found = S.detect_people(fr, client, max_frames=4, progress=log.append)
    assert sorted(found) == fr.spread(4)
    assert client.calls == 4
    assert all(c["kind"] == "segment" and c["prompt"] == "person" and c["image"] for c in client.log)
    for i, dets in found.items():
        assert len(dets) == 1 and dets[0].label == "person" and dets[0].frame == i
    assert log and "people found in 4 of 4 frames" in log[-1]


def test_detect_people_skips_unreadable_frames_and_people_free_frames(tmp_path):
    fr = _frames(n=3)
    fr.paths = [tmp_path / name for name in fr.names]
    fr.paths[0].write_bytes(b"x")
    fr.paths[2].write_bytes(b"x")            # f1.jpg is missing

    def answer(prompt):
        return [Detection("person", (0.4, 0.2, 0.6, 0.8), 0.9)] if answer.calls == 0 else []

    def counting(prompt):
        out = answer(prompt)
        answer.calls += 1
        return out

    answer.calls = 0
    client = FakeMetaClient(detections=counting)
    log: list[str] = []
    found = S.detect_people(fr, client, max_frames=3, progress=log.append)
    assert list(found) == [0]
    assert any("1 frames failed" in m for m in log)
