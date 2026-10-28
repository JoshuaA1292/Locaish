"""The camera simulation: a keyed timeline evaluated against the measured room.

Every test asks something with a geometric answer: where the rig puts a key
it cannot honour, which frame the room objects to, whether the preview and the
render would agree. The clean fixture room is about 5.2 x 4.1 m with a 2.85 m
ceiling; a sofa-sized box stands at x -2.45..-1.85, y -0.9..0.9, a low table at
x 0.6..2.1, y 1.2..1.9, and a 1.15 m unit at x 1.5..2.4, y -1.85..-0.95.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from locaish.film import optics, sim, space
from locaish.film import rehearsal as R
from locaish.film.render import EYELINE_RATIO


@pytest.fixture(scope="module")
def room(clean):
    twin = clean[0].twin
    return twin, space.floor_maps(twin), space.occupancy(twin)


def _tl(rig="handheld", params=None, keys=None, *, actors="walk", interp="linear", duration=4.0,
        sensor="ff", fps=24, tid="deadbeef"):
    if keys is None:
        keys = [{"t": 0.0, "pos": [-1.0, -0.8, 1.4], "look_at": "Maya", "focal_mm": 35.0},
                {"t": duration, "pos": [1.0, -0.8, 1.4], "look_at": "Maya", "focal_mm": 50.0}]
    if actors == "walk":
        actors = {"Maya": {"height_m": 1.68, "keys": [
            {"t": 0.0, "pos": [0.0, 0.8], "facing_deg": -90.0},
            {"t": duration, "pos": [-0.6, 1.2], "facing_deg": -90.0}]}}
    elif actors == "still":
        actors = {"Maya": {"height_m": 1.68, "keys": [{"t": 0.0, "pos": [0.0, 0.8], "facing_deg": -90.0}]}}
    elif actors is None:
        actors = {}
    return sim.Timeline.from_dict({
        "id": tid, "title": "Push-in on Maya", "fps": fps, "duration_s": duration, "sensor": sensor,
        "aperture_f": 2.8, "rig": {"kind": rig, "params": params or {}},
        "camera": {"interp": interp, "keys": keys}, "actors": actors, "pieces": [], "notes": [],
    })


def _eyeline(tl, name, t, floor_z):
    x, y, _ = sim.actor_state(tl.actors[name], t)
    return np.array([x, y, floor_z + EYELINE_RATIO * tl.actors[name].height_m])


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------


def test_json_round_trip_keeps_everything_and_canonicalises_the_sensor():
    tl = _tl("dolly", {"gear_key": "super-peewee"})
    tl.pieces.append(sim.PieceTrack("92aa1c88", [sim.PieceKey(0.0), sim.PieceKey(2.0, 0.4, -0.2, 15.0)]))
    d = tl.to_dict()
    back = sim.Timeline.from_dict(json.loads(json.dumps(d)))
    assert back.to_dict() == d
    assert d["sensor"] == "fullframe", "the contract's 'ff' is an alias for optics' 'fullframe'"
    assert d["id"] == "deadbeef" and d["camera"]["interp"] == "linear"
    assert d["rig"] == {"kind": "dolly", "params": {"gear_key": "super-peewee"}}
    assert d["pieces"][0]["keys"][1] == {"t": 2.0, "dx": 0.4, "dy": -0.2, "dyaw_deg": 15.0}
    assert set(d) == {"id", "title", "fps", "duration_s", "sensor", "aperture_f", "rig", "camera", "actors", "pieces", "notes"}


def test_keys_are_sorted_and_the_camera_holds_before_the_first_and_after_the_last():
    tl = _tl(keys=[{"t": 3.0, "pos": [1.0, 0.0, 1.5], "aim": [0.0, 0.0, 1.5]},
                   {"t": 1.0, "pos": [-1.0, 0.0, 1.5], "aim": [0.0, 0.0, 1.5]}], actors=None)
    assert [k.t for k in tl.camera_keys] == [1.0, 3.0]
    ev = sim.evaluate(tl)
    assert ev.frames[0].cam == pytest.approx([-1.0, 0.0, 1.5]), "holds the first key before it"
    assert ev.frames[-1].cam == pytest.approx([1.0, 0.0, 1.5]), "holds the last key after it"
    mid = sim.sample_time(tl, 2.0)
    assert mid["cam"] == pytest.approx([0.0, 0.0, 1.5])


def test_linear_catmull_and_ease_differ_where_they_should():
    keys = [{"t": 0.0, "pos": [0.0, 0.0, 1.5], "aim": [0.0, 5.0, 1.5]},
            {"t": 2.0, "pos": [2.0, 0.0, 1.5], "aim": [0.0, 5.0, 1.5]},
            {"t": 4.0, "pos": [2.0, 2.0, 1.5], "aim": [0.0, 5.0, 1.5]}]
    lin = sim.sample_time(_tl(keys=keys, actors=None, interp="linear"), 0.5)
    ease = sim.sample_time(_tl(keys=keys, actors=None, interp="ease"), 0.5)
    cat = sim.sample_time(_tl(keys=keys, actors=None, interp="catmull"), 1.0)
    assert lin["cam"] == pytest.approx([0.5, 0.0, 1.5])
    # smoothstep(0.25) = 0.15625 of the 2 m segment
    assert ease["cam"] == pytest.approx([0.3125, 0.0, 1.5])
    assert sim.sample_time(_tl(keys=keys, actors=None, interp="ease"), 1.0)["cam"] == pytest.approx([1.0, 0.0, 1.5])
    # The spline bows toward the third key instead of running along y = 0.
    assert cat["cam"][1] != pytest.approx(0.0, abs=1e-3)
    assert 0.0 < cat["cam"][0] < 2.0
    # It still passes through the keys.
    assert sim.sample_time(_tl(keys=keys, actors=None, interp="catmull"), 2.0)["cam"] == pytest.approx([2.0, 0.0, 1.5])


def test_focal_length_interpolates_in_log_space():
    keys = [{"t": 0.0, "pos": [0.0, 0.0, 1.5], "aim": [0.0, 5.0, 1.5], "focal_mm": 25.0},
            {"t": 2.0, "pos": [0.0, 0.0, 1.5], "aim": [0.0, 5.0, 1.5], "focal_mm": 100.0}]
    mid = sim.sample_time(_tl("tripod", keys=keys, actors=None), 1.0)
    assert mid["focal_mm"] == pytest.approx(50.0, abs=1e-6), "the geometric, not arithmetic, midpoint"


def test_look_at_tracks_a_moving_actor_eyeline_every_frame():
    tl = _tl("handheld", {"shake": 0.0})
    ev = sim.evaluate(tl)
    assert len(ev.frames) == 4 * 24 + 1
    for f in ev.frames:
        assert f.aim == pytest.approx(_eyeline(tl, "Maya", f.t, 0.0).tolist(), abs=1e-9)
        assert f.subject == "Maya"
    assert ev.frames[0].actors["Maya"][:2] == pytest.approx([0.0, 0.8])
    assert ev.frames[-1].actors["Maya"][:2] == pytest.approx([-0.6, 1.2])


# ---------------------------------------------------------------------------
# rigs
# ---------------------------------------------------------------------------


def test_tripod_snaps_every_key_to_the_first_position_and_says_so():
    tl = _tl("tripod")
    solved = sim.solve(tl)
    assert all(k.pos == solved.camera_keys[0].pos for k in solved.camera_keys)
    assert solved.camera_keys[1].snapped_from == pytest.approx([1.0, -0.8, 1.4])
    assert any(n.startswith("key 1 moved 2.00 m") for n in solved.notes)
    assert tl.camera_keys[1].pos == pytest.approx([1.0, -0.8, 1.4]), "solve is pure"


def test_slider_length_is_clamped_and_the_keys_pulled_onto_the_rail():
    tl = _tl("slider", {"length_m": 5.0}, keys=[
        {"t": 0.0, "pos": [-1.5, 0.0, 1.2], "aim": [0.0, 3.0, 1.5]},
        {"t": 4.0, "pos": [1.5, 0.0, 1.4], "aim": [0.0, 3.0, 1.5]}], actors=None)
    solved = sim.solve(tl)
    assert solved.rig.params["length_m"] == 2.0
    a, b = (np.array(k.pos) for k in solved.camera_keys)
    assert np.linalg.norm(b - a) == pytest.approx(2.0, abs=1e-6)
    assert a[2] == b[2] == 1.2, "the rail is level at the first key's height"
    assert any("clamped to 2" in n for n in solved.notes)
    assert any("slider length_m=5.0" in p for p in tl.validate())


def test_dolly_projects_keys_onto_a_straight_track_and_reports_the_one_that_moved(room):
    twin, maps, occ = room
    tl = _tl("dolly", keys=[
        {"t": 0.0, "pos": [-1.0, -0.6, 1.2], "look_at": "Maya"},
        {"t": 2.0, "pos": [0.0, -0.2, 1.2], "look_at": "Maya"},
        {"t": 4.0, "pos": [1.0, -0.6, 1.2], "look_at": "Maya"}], actors="still")
    solved = sim.solve(tl, twin, maps=maps)
    track = np.asarray(solved.rig.params["track"])
    assert len(track) == 3, "the track is the polyline through the keys"
    assert solved.camera_keys[1].snapped_from is None
    # Pin the track and the middle key is off it by 0.4 m.
    tl2 = _tl("dolly", {"track": [[-1.0, -0.6], [1.0, -0.6]]}, keys=tl.to_dict()["camera"]["keys"], actors="still")
    solved2 = sim.solve(tl2, twin, maps=maps)
    assert solved2.camera_keys[1].pos == pytest.approx([0.0, -0.6, 1.2])
    assert solved2.camera_keys[1].snapped_from == pytest.approx([0.0, -0.2, 1.2])
    assert any(n.startswith("key 1 moved 0.40 m onto the track") for n in solved2.notes)
    ev = sim.evaluate(tl2, twin, maps=maps, occupancy=occ)
    assert all(abs(f.cam[1] + 0.6) < 1e-6 for f in ev.frames), "the camera rides the track between keys"


def test_dolly_speed_cap_is_flagged_for_three_metres_in_a_second(room):
    twin, maps, occ = room
    tl = _tl("dolly", keys=[
        {"t": 1.0, "pos": [-1.5, -0.6, 1.2], "look_at": "Maya"},
        {"t": 2.0, "pos": [1.5, -0.6, 1.2], "look_at": "Maya"}], actors="still")
    ev = sim.evaluate(tl, twin, maps=maps, occupancy=occ)
    fast = [s for s in ev.segments if "m/s, over the 1.5 m/s" in s["text"]]
    assert fast and fast[0]["severity"] == "fail"
    assert 1.0 <= fast[0]["t_start"] < fast[0]["t_end"] <= 2.0
    assert ev.frames[0].speed_mps == 0.0 and max(f.speed_mps for f in ev.frames) > 1.5


def test_dolly_track_through_the_sofa_fails_gear_fit_along_the_whole_track(room):
    twin, maps, occ = room
    # The camera never leaves the start; the track runs on into the sofa.
    tl = _tl("dolly", {"track": [[-0.5, 0.0], [-2.3, 0.0]]}, keys=[
        {"t": 0.0, "pos": [-0.5, 0.0, 1.2], "look_at": "Maya"},
        {"t": 4.0, "pos": [-0.5, 0.0, 1.2], "look_at": "Maya"}], actors="still")
    ev = sim.evaluate(tl, twin, maps=maps, occupancy=occ)
    along = [s for s in ev.segments if "along the track" in s["text"] and "clear radius" in s["text"]]
    assert along, "moves.simulate's gear-fit failure propagates from the unreached stretch"
    assert all(s["severity"] == "fail" for s in along)
    assert not any("clear radius" in p.text and "along" not in p.text for p in ev.frames[0].problems), \
        "where the camera actually stands, the dolly fits"
    assert ev.summary["feasible"] is False


def test_jib_keys_land_on_the_arm_sphere_and_the_snap_is_reported(room):
    twin, maps, occ = room
    tl = _tl("jib", {"pivot": [0.0, 0.0], "base_height_m": 1.0, "arm_m": 3.0}, keys=[
        {"t": 0.0, "pos": [2.0, 0.0, 1.5], "look_at": "Maya"},
        {"t": 4.0, "pos": [0.0, -2.0, 2.0], "look_at": "Maya"}], actors="still")
    solved = sim.solve(tl, twin, maps=maps)
    pivot = np.array([0.0, 0.0, maps.floor_z + 1.0])
    for k in solved.camera_keys:
        assert np.linalg.norm(np.array(k.pos) - pivot) == pytest.approx(3.0, abs=1e-6)
        assert k.snapped_from is not None
    assert any("onto the arm's reach" in n for n in solved.notes)
    ev = sim.evaluate(tl, twin, maps=maps, occupancy=occ)
    for f in ev.frames:
        assert np.linalg.norm(np.array(f.cam) - pivot) == pytest.approx(3.0, abs=1e-6), "the swing stays on the sphere"


def test_jib_arm_through_a_furniture_box_is_flagged(room):
    twin, maps, occ = room
    # Pivot at the room's middle-south, arm booming down into the 1.15 m unit at x 1.5..2.4.
    tl = _tl("jib", {"pivot": [0.0, -1.4], "base_height_m": 1.0, "arm_m": 2.5}, keys=[
        {"t": 0.0, "pos": [2.4, -1.4, 0.4], "aim": [2.6, -1.4, 0.3]}], actors=None, duration=1.0)
    ev = sim.evaluate(tl, twin, maps=maps, occupancy=occ)
    assert any("jib arm passes through" in s["text"] for s in ev.segments)


def test_jib_counterweight_is_checked_against_the_floor_and_the_room(room):
    twin, maps, occ = room
    # Arm up at 25 degrees: the counterweight end would be 0.27 m under the floor.
    el = math.radians(25.0)
    cam = [3.0 * math.cos(el), 0.0, maps.floor_z + 1.0 + 3.0 * math.sin(el)]
    tl = _tl("jib", {"pivot": [0.0, 0.0], "base_height_m": 1.0, "arm_m": 3.0}, keys=[
        {"t": 0.0, "pos": cam, "aim": [2.0, 0.0, 1.0]}], actors=None, duration=1.0)
    ev = sim.evaluate(tl, twin, maps=maps, occupancy=occ)
    assert any("counterweight end would be 0.27 m below the floor" in s["text"] for s in ev.segments)
    # Level arm: no floor problem, and the counterweight swings into the sofa at x < -1.85.
    tl2 = _tl("jib", {"pivot": [-0.5, 0.0], "base_height_m": 0.5, "arm_m": 2.5}, keys=[
        {"t": 0.0, "pos": [2.0, 0.0, maps.floor_z + 0.5], "aim": [2.5, 0.0, 0.5]}], actors=None, duration=1.0)
    ev2 = sim.evaluate(tl2, twin, maps=maps, occupancy=occ)
    texts = [s["text"] for s in ev2.segments]
    assert not any("below the floor" in t for t in texts)
    assert any("counterweight arm swings through" in t for t in texts)


def test_handheld_shake_is_deterministic_and_zero_at_zero():
    a = sim.evaluate(_tl("handheld", {"shake": 0.5}))
    b = sim.evaluate(_tl("handheld", {"shake": 0.5}))
    assert [f.aim for f in a.frames] == [f.aim for f in b.frames]
    other = sim.evaluate(_tl("handheld", {"shake": 0.5}, tid="0badf00d"))
    assert [f.aim for f in a.frames] != [f.aim for f in other.frames], "a different timeline id shakes differently"
    still = sim.evaluate(_tl("handheld", {"shake": 0.0}))
    tl = still.timeline
    for f in still.frames:
        assert f.aim == pytest.approx(_eyeline(tl, "Maya", f.t, 0.0).tolist(), abs=1e-9)
    off = [np.linalg.norm(np.subtract(f.aim, _eyeline(tl, "Maya", f.t, 0.0))) for f in a.frames]
    assert max(off) > 0.005 and max(off) < 0.3, "a tremor, not a swing"
    # The preview and the render agree frame for frame.
    k = 17
    assert sim.sample_time(_tl("handheld", {"shake": 0.5}), k / 24)["aim"] == a.frames[k].to_dict()["aim"]


def test_steadicam_lags_the_subject_without_shaking():
    ev = sim.evaluate(_tl("steadicam"))
    tl = ev.timeline
    assert "shake" not in tl.rig.params
    lag = [np.linalg.norm(np.subtract(f.aim, _eyeline(tl, "Maya", f.t, 0.0))) for f in ev.frames[12:]]
    assert 0.0 < max(lag) < 0.2, "the operator trails a walking actor by centimetres, not metres"
    # The same input twice is the same output: no randomness anywhere.
    assert [f.aim for f in sim.evaluate(_tl("steadicam")).frames] == [f.aim for f in ev.frames]


def test_crane_is_held_under_the_ceiling_and_warned_about_headroom(room):
    twin, maps, occ = room
    high = _tl("crane", keys=[{"t": 0.0, "pos": [0.0, 0.0, 3.5], "aim": [1.0, 0.0, 1.0]}], actors=None, duration=1.0)
    ev = sim.evaluate(high, twin, maps=maps, occupancy=occ)
    ceiling = float(twin.structure.ceiling_z)
    assert ev.timeline.rig.params["max_height_m"] == pytest.approx(ceiling - maps.floor_z)
    assert ev.frames[0].cam[2] == pytest.approx(ceiling, abs=1e-6), "snapped to the ceiling"
    assert any("headroom" in s["text"] for s in ev.segments)
    tight = _tl("crane", keys=[{"t": 0.0, "pos": [0.0, 0.0, ceiling - 0.1], "aim": [1.0, 0.0, 1.0]}], actors=None, duration=1.0)
    ev2 = sim.evaluate(tight, twin, maps=maps, occupancy=occ)
    warn = [s for s in ev2.segments if "margin" in s["text"]]
    assert warn and warn[0]["severity"] == "warn"


def test_drone_clearance_to_the_ceiling_and_a_clean_hover(room):
    twin, maps, occ = room
    ceiling = float(twin.structure.ceiling_z)
    near = _tl("drone", keys=[{"t": 0.0, "pos": [0.0, 0.0, ceiling - 0.25], "aim": [1.0, 0.0, 1.0]}], actors=None, duration=1.0)
    ev = sim.evaluate(near, twin, maps=maps, occupancy=occ)
    assert any("inside its 0.5 m clearance" in s["text"] for s in ev.segments)
    clear = _tl("drone", keys=[{"t": 0.0, "pos": [0.0, 0.0, 1.6], "aim": [1.0, 0.0, 1.0]}], actors=None, duration=1.0)
    ev2 = sim.evaluate(clear, twin, maps=maps, occupancy=occ)
    assert not any("clearance" in s["text"] for s in ev2.segments)


# ---------------------------------------------------------------------------
# framing, depth of field, problems
# ---------------------------------------------------------------------------


def test_subject_out_of_frame_is_flagged_and_in_frame_is_sized(room):
    twin, maps, occ = room
    away = _tl("tripod", keys=[{"t": 0.0, "pos": [-1.0, -0.8, 1.4], "aim": [-1.0, -1.9, 1.4], "focal_mm": 50.0}],
               actors="still", duration=1.0)
    ev = sim.evaluate(away, twin, maps=maps, occupancy=occ)
    f = ev.frames[0]
    assert f.subject == "Maya" and f.in_frame is False
    assert any("falls outside the frame" in p.text for p in f.problems)
    toward = _tl("tripod", keys=[{"t": 0.0, "pos": [0.0, -1.2, 1.4], "look_at": "Maya", "focal_mm": 50.0}],
                 actors="still", duration=1.0)
    ev2 = sim.evaluate(toward, twin, maps=maps, occupancy=occ)
    f2 = ev2.frames[0]
    assert f2.in_frame is True and f2.distance_m == pytest.approx(math.hypot(2.0, 1.4 - EYELINE_RATIO * 1.68), abs=0.01)
    assert f2.shot_size == optics.classify_framing(f2.framed_height_m).key
    assert f2.subject_fill == pytest.approx(1.68 / f2.framed_height_m)


def test_depth_of_field_numbers_are_finite_and_infinite_when_they_should_be():
    ev = sim.evaluate(_tl("tripod", keys=[{"t": 0.0, "pos": [0.0, -1.2, 1.4], "look_at": "Maya", "focal_mm": 50.0}],
                          actors="still", duration=1.0))
    d = ev.frames[0].dof
    assert d["infinite"] is False and 0 < d["near_m"] < ev.frames[0].focus_m < d["far_m"] < 10
    assert math.isfinite(d["hyperfocal_m"])
    wide = sim.evaluate(_tl("tripod", keys=[{"t": 0.0, "pos": [0.0, -1.2, 1.4], "aim": [0.0, 3.0, 1.4], "focal_mm": 12.0}],
                            actors=None, duration=1.0))
    wide.timeline.aperture_f = 2.8
    far = sim.sample_time(sim.Timeline.from_dict({**wide.timeline.to_dict(), "aperture_f": 16.0}), 0.0)
    assert far["dof"]["infinite"] is True and far["dof"]["far_m"] is None
    assert far["fov_h_deg"] == pytest.approx(optics.field_of_view_deg(optics.SENSORS["fullframe"], 12.0)[0], abs=0.01)


def test_segments_merge_adjacent_frames_into_one_run(room):
    twin, maps, occ = room
    # A tripod 30 cm from the south wall never fits, so one problem spans the whole timeline.
    tl = _tl("tripod", keys=[{"t": 0.0, "pos": [0.0, -1.75, 1.4], "look_at": "Maya"}], actors="still", duration=2.0)
    ev = sim.evaluate(tl, twin, maps=maps, occupancy=occ)
    fit = [s for s in ev.segments if "clear radius" in s["text"]]
    assert len(fit) == 1
    assert fit[0]["t_start"] == 0.0 and fit[0]["t_end"] == pytest.approx(2.0) and fit[0]["frames"] == len(ev.frames)
    assert ev.summary["worst"] == fit[0]["text"] and ev.summary["worst_severity"] == "fail"
    assert ev.summary["problem_frames"] == len(ev.frames)


def test_a_cut_is_a_jump_not_a_speed():
    tl = _tl("handheld", {"shake": 0.0}, keys=[
        {"t": 0.0, "pos": [-1.0, -0.8, 1.4], "look_at": "Maya"},
        {"t": 2.0, "pos": [-1.0, -0.8, 1.4], "look_at": "Maya"},
        {"t": 2.0, "pos": [1.0, -0.8, 1.4], "look_at": "Maya"},
        {"t": 4.0, "pos": [1.0, -0.8, 1.4], "look_at": "Maya"}])
    ev = sim.evaluate(tl)
    cuts = [f for f in ev.frames if f.cut]
    assert len(cuts) == 1 and cuts[0].t == pytest.approx(2.0)
    assert cuts[0].cam == pytest.approx([1.0, -0.8, 1.4]) and cuts[0].speed_mps == 0.0
    assert not any("m/s" in p.text for f in ev.frames for p in f.problems)
    assert ev.summary["cuts"] == 1 and ev.summary["camera_travel_m"] == pytest.approx(0.0)


def test_evaluate_with_no_actors_still_answers(room):
    twin, maps, occ = room
    tl = _tl("handheld", keys=[{"t": 0.0, "pos": [-1.0, 0.0, 1.5], "aim": [1.0, 0.0, 1.5]},
                               {"t": 2.0, "pos": [-0.5, 0.0, 1.5], "aim": [1.0, 0.0, 1.5]}], actors=None, duration=2.0)
    ev = sim.evaluate(tl, twin, maps=maps, occupancy=occ)
    f = ev.frames[10]
    assert f.subject is None and f.distance_m is None and f.shot_size is None
    assert f.focus_m == pytest.approx(np.linalg.norm(np.subtract(f.aim, f.cam)))
    assert f.dof["near_m"] > 0 and f.actors == {}
    assert ev.summary["size_range"] is None and ev.summary["distance_range_m"] is None
    json.dumps(ev.to_dict())


def test_validate_reports_an_unknown_rig_a_bad_sensor_and_a_missing_actor():
    tl = _tl("hoverboard", sensor="imax")
    tl.camera_keys[0].look_at = "Theo"
    problems = tl.validate()
    assert any("unknown rig 'hoverboard'" in p for p in problems)
    assert any("unknown sensor 'imax'" in p for p in problems)
    assert any("look_at names 'Theo'" in p for p in problems)
    with pytest.raises(ValueError):
        sim.evaluate(tl)
    assert _tl().validate() == []


def test_validate_flags_a_camera_outside_the_room(room):
    twin, maps, occ = room
    tl = _tl("tripod", keys=[{"t": 0.0, "pos": [4.0, 0.0, 1.4], "look_at": "Maya"}], actors="still")
    assert any("outside the room's floor outline" in p for p in tl.validate(twin, maps=maps))


def test_describe_rigs_lists_every_rig_with_its_gear():
    rigs = {r["kind"]: r for r in sim.describe_rigs()}
    assert set(rigs) == {"tripod", "slider", "dolly", "jib", "handheld", "steadicam", "crane", "drone"}
    assert set(rigs["dolly"]["gear_keys"]) == {"super-peewee", "fisher-11", "doorway-dolly"}
    assert rigs["dolly"]["limits"] == {"speed_mps": 1.5, "accel_mps2": 1.0}
    assert rigs["handheld"]["params"]["shake"]["range"] == [0.0, 1.0]
    for r in rigs.values():
        assert r["degrees_of_freedom"] and r["checks"]


def test_evaluation_to_dict_shape_is_what_the_viewer_reads(room):
    twin, maps, occ = room
    d = sim.evaluate(_tl("dolly"), twin, maps=maps, occupancy=occ).to_dict()
    assert set(d) == {"fps", "duration_s", "timeline", "frames", "segments", "summary"}
    assert d["fps"] == 24 and d["duration_s"] == 4.0
    f = d["frames"][3]
    assert set(f) == {"t", "cam", "aim", "focal_mm", "pan_deg", "tilt_deg", "roll_deg", "speed_mps", "accel_mps2",
                      "cut", "actors", "pieces", "subject", "distance_m", "framed_height_m", "subject_fill",
                      "shot_size", "shot_name", "in_frame", "focus_m", "dof", "fov_h_deg", "fov_v_deg",
                      "problems", "severity"}
    assert set(f["dof"]) == {"near_m", "far_m", "infinite", "hyperfocal_m"}
    for s in d["segments"]:
        assert set(s) == {"t_start", "t_end", "text", "severity", "frames"} and s["severity"] in ("warn", "fail")
    assert {"rig", "duration_s", "fps", "frames", "camera_travel_m", "actor_travel_m", "size_range",
            "distance_range_m", "focal_range_mm", "problem_frames", "fail_frames", "worst", "worst_severity",
            "feasible", "cuts", "notes", "title"} == set(d["summary"])
    json.dumps(d)


# ---------------------------------------------------------------------------
# from a rehearsal
# ---------------------------------------------------------------------------


SCENE_TEXT = """
TITLE: The kitchen, morning
MARKS: A=(0.0,0.6) B=(-1.0,-0.6) DOOR=(-1.4,0.4) C=(0.0,-1.2)
CAST: Maya=1.68 Theo=1.84
1. Theo enters at B
2. Maya enters at DOOR
3. Maya moves to A
4. Theo: "You came back."   [cam (0.2,-1.8) h1.5 50mm on Theo]
5. Maya: "I never left."    [cam (-0.8,-1.9) h1.4 35mm on Maya dolly to (0.6,-1.9)]
6. hold 2s
7. Maya moves to C   [cam (0.2,-1.8) h1.5 50mm on Maya]
8. Maya exits to DOOR
"""


def test_from_rehearsal_keys_every_shot_and_every_beat(room):
    twin, maps, occ = room
    reh = R.rehearse(twin, R.parse_scene(SCENE_TEXT), maps=maps, occupancy=occ)
    tl = sim.from_rehearsal(reh.to_dict())
    assert tl.title == "The kitchen, morning" and tl.interp == "linear"
    assert tl.duration_s == pytest.approx(reh.duration_s, abs=1e-3)
    # One key at each shot's start, the dolly's parked end, and a hold to each cut.
    starts = {round(s.t_start, 3) for s in reh.shots}
    assert starts <= {round(k.t, 3) for k in tl.camera_keys}
    assert len(tl.camera_keys) >= len(reh.shots)
    dolly = next(s for s in reh.shots if s.setup["dolly_to"])
    parked = next(k for k in tl.camera_keys if abs(k.t - dolly.beat_end) < 1e-3)
    assert parked.pos[:2] == pytest.approx([0.6, -1.9], abs=1e-3)
    assert all(k.look_at in ("Maya", "Theo") for k in tl.camera_keys)
    assert set(tl.actors) == {"Maya", "Theo"} and tl.actors["Theo"].height_m == 1.84
    boundaries = {round(b.t_start, 3) for b in reh.beats} | {round(b.t_end, 3) for b in reh.beats}
    assert boundaries <= {round(k.t, 3) for k in tl.actors["Maya"].keys} | {round(k.t, 3) for k in tl.actors["Theo"].keys}
    assert len(tl.actors["Maya"].keys) > len(boundaries) // 2, "the walk's route waypoints are keyed too"
    assert tl.rig.kind == "handheld" and any(n.startswith("rig: handheld") for n in tl.notes)
    ev = sim.evaluate(tl, twin, maps=maps, occupancy=occ)
    assert ev.summary["cuts"] >= 2 and ev.summary["frames"] == int(round(tl.duration_s * 24)) + 1
    assert any(f.subject == "Theo" for f in ev.frames) and any(f.subject == "Maya" for f in ev.frames)


def test_from_rehearsal_picks_the_dolly_for_a_single_dolly_shot(room):
    twin, maps, occ = room
    scene = R.Scene(
        title="push", marks={"A": (0.0, 0.6)}, cast={"Maya": 1.68},
        beats=[R.Beat(1, "enter", "Maya", to_mark="A"),
               R.Beat(2, "line", "Maya", text="One two three four five six seven eight.",
                      shot=R.ShotSpec((-0.6, -1.0), 1.4, 35, "Maya", dolly_to=(0.8, -1.0)))])
    reh = R.rehearse(twin, scene, maps=maps, occupancy=occ)
    tl = sim.from_rehearsal(reh.to_dict(), title="Push")
    assert tl.rig.kind == "dolly" and tl.rig.params["gear_key"] == "doorway-dolly" and tl.title == "Push"
    ev = sim.evaluate(tl, twin, maps=maps, occupancy=occ)
    assert ev.summary["feasible"], [s["text"] for s in ev.segments]
    assert ev.timeline.rig.params["track"][0] == pytest.approx([-0.6, -1.0], abs=1e-3)


# ---------------------------------------------------------------------------
# Blender
# ---------------------------------------------------------------------------


def test_blender_look_rotation_points_the_camera_down_minus_z_at_the_aim():
    rng = np.random.default_rng(3)
    cases = [(rng.uniform(-3, 3, 3), rng.uniform(-3, 3, 3), rng.uniform(-30, 30)) for _ in range(40)]
    cases.append((np.array([0.0, 0.0, 1.5]), np.array([0.0, 0.0, 1.5 + 2.0 * math.tan(math.radians(89.0))]) + [2.0, 0, 0], 0.0))
    for cam, aim, roll in cases:
        if np.linalg.norm(aim - cam) < 1e-3:
            continue
        R_ = sim.look_rotation(cam, aim, roll)
        fwd = (aim - cam) / np.linalg.norm(aim - cam)
        assert R_ @ np.array([0.0, 0.0, -1.0]) == pytest.approx(fwd, abs=1e-9)
        assert R_.T @ R_ == pytest.approx(np.eye(3), abs=1e-9) and np.linalg.det(R_) == pytest.approx(1.0)
        a, b, c = sim.euler_xyz(R_)
        assert sim.rotation_from_euler_xyz(a, b, c) == pytest.approx(R_, abs=1e-7), "Blender's XYZ Euler rebuilds it"
    # No roll: the camera's up has no sideways lean (its X axis is horizontal).
    R0 = sim.look_rotation([0, 0, 1.5], [2, 1, 1.0], 0.0)
    assert R0[2, 0] == pytest.approx(0.0, abs=1e-12)
    # Roll dips the right side of the frame.
    Rr = sim.look_rotation([0, 0, 1.5], [2, 0, 1.5], 20.0)
    assert Rr[2, 0] < 0


def test_blender_script_carries_the_room_the_camera_every_frame_and_the_problems(room, tmp_path):
    twin, maps, occ = room
    tl = _tl("dolly", keys=[{"t": 0.0, "pos": [-1.0, -0.6, 1.2], "look_at": "Maya", "focal_mm": 35.0},
                            {"t": 3.0, "pos": [1.0, -0.6, 1.2], "look_at": "Maya", "focal_mm": 50.0}],
             actors="walk", duration=3.0)
    ev = sim.evaluate(tl, twin, maps=maps, occupancy=occ)
    glb = tmp_path / "room.glb"
    script = sim.blender_script(tl, ev, glb_path=str(glb), twin=twin)
    assert "import bpy" in script and "bpy.ops.import_scene.gltf(filepath=DATA[\"glb\"])" in script
    assert str(glb) in script and "Locaish Camera" in script and "primitive_cylinder_add" in script
    assert "Dolly Track" in script and "Locaish Problems" in script
    payload = script.split('json.loads(r"""', 1)[1].split('""")', 1)[0]
    data = json.loads(payload)
    assert data["n_frames"] == len(ev.frames) == 73 and len(data["frames"]) == 73
    assert data["fps"] == 24 and data["sensor_width_mm"] == 36.0 and data["aperture_f"] == 2.8
    assert data["frames"][0]["loc"] == pytest.approx(ev.frames[0].cam, abs=1e-4)
    assert data["frames"][-1]["lens"] == pytest.approx(50.0, abs=1e-3)
    assert len(data["actors"]["Maya"]["keys"]) == 73 and data["actors"]["Maya"]["height_m"] == 1.68
    assert data["track"] == ev.timeline.rig.params["track"]
    assert len(data["problems"]) == len(ev.segments)
    # Every keyed rotation looks from the frame's camera to its aim.
    for fr, f in zip(data["frames"][::12], ev.frames[::12]):
        Rm = sim.rotation_from_euler_xyz(*fr["rot"])
        fwd = np.subtract(f.aim, f.cam) / np.linalg.norm(np.subtract(f.aim, f.cam))
        assert Rm @ [0.0, 0.0, -1.0] == pytest.approx(fwd, abs=1e-5)
    compile(script, "locaish_blender.py", "exec")
    summary = sim.blender_summary(tl, ev)
    assert any("Dolly Track" in s for s in summary) and any("Maya" in s for s in summary)
    assert any("1-73" in s for s in summary)


BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def test_blender_script_runs_headless_in_a_real_blender_when_one_is_installed(room, tmp_path):
    """Compiles always; when the Blender binary is present the script is run
    against a GLB exported the way `locaish export` does it, and the summary
    line the script ends with is asserted (every camera frame keyed, the room
    the twin's size, a .blend saved beside the script)."""
    import os
    import re
    import subprocess

    twin, maps, occ = room
    tl = _tl("dolly", keys=[{"t": 0.0, "pos": [-1.0, -0.6, 1.2], "look_at": "Maya", "focal_mm": 35.0},
                            {"t": 2.0, "pos": [1.0, -0.6, 1.2], "look_at": "Maya", "focal_mm": 50.0}],
             actors="walk", duration=2.0)
    ev = sim.evaluate(tl, twin, maps=maps, occupancy=occ)
    glb = tmp_path / "room.glb"
    script = sim.blender_script(tl, ev, glb_path=str(glb), twin=twin)
    compile(script, "blender.py", "exec")
    assert "LOCAISH SUMMARY" in script and "channelbag" in script and "save_as_mainfile" in script
    if not (os.path.exists(BLENDER) and os.access(BLENDER, os.X_OK)) or twin.mesh is None:
        pytest.skip("no Blender binary at the known path (or the twin has no mesh)")
    import trimesh
    trimesh.Trimesh(vertices=twin.mesh.vertices, faces=twin.mesh.faces,
                    vertex_colors=twin.mesh.vertex_colors, process=False).export(glb)
    path = tmp_path / "blender.py"
    path.write_text(script)
    run = subprocess.run([BLENDER, "-b", "--python", str(path)], cwd=tmp_path, capture_output=True, text=True, timeout=240)
    out = run.stdout + run.stderr
    assert run.returncode == 0, out[-3000:]
    assert "Traceback" not in out, out[-3000:]
    summary = [line for line in out.splitlines() if line.startswith("LOCAISH SUMMARY:")]
    assert len(summary) == 1, out[-3000:]
    assert f"camera keyframes {len(ev.frames)} of {len(ev.frames)} frames" in summary[0]
    assert "Locaish Camera" in summary[0] and "Dolly Track" in summary[0]
    ceiling = float(twin.structure.ceiling_z) - float(twin.structure.floor_z)
    m = re.search(r"room ([\d.]+) x ([\d.]+) x ([\d.]+) m", summary[0])
    assert m and abs(float(m.group(3)) - ceiling) < 0.1, summary[0]
    assert "WARNING" not in out
    assert (tmp_path / "blender.blend").exists()


def test_walking_rigs_route_round_the_furniture(clean):
    """A handheld move whose straight line crosses the furniture gets routed
    keys inserted on the standable floor; the keys are derived, and a second
    solve drops and re-derives them rather than piling them up."""
    import numpy as np
    import pytest

    from locaish.film import rehearsal as reh
    from locaish.film import sim, space

    result, _fx = clean
    twin = result.twin
    fz = float(twin.structure.floor_z)
    maps = space.floor_maps(twin)
    mask = maps.standable(radius_m=sim.OPERATOR_RADIUS_M + float(maps.cell))
    cells = np.asarray(maps.world_of(np.argwhere(mask))).reshape(-1, 2)
    rng = np.random.default_rng(0)
    if len(cells) > 160:
        cells = cells[rng.choice(len(cells), 160, replace=False)]
    occ = space.occupancy(twin)
    # The longest pair whose straight line leaves the operator's floor but
    # which a route joins: the furniture box sits between them.
    best = None
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            L = float(np.linalg.norm(cells[j] - cells[i]))
            if best is not None and L <= best[0]:
                continue
            if reh._segment_walkable(maps, mask, cells[i], cells[j]):
                continue
            r = reh.route(maps, occ, cells[i], cells[j], mask=mask)
            if not r.blocked and len(r.points) >= 3:
                best = (L, cells[i], cells[j])
    if best is None:
        pytest.skip("the fixture has no pair of standable cells with furniture between them")
    _L, a, b = best
    tl = sim.Timeline.from_dict({
        "id": "walk", "title": "walk", "fps": 12, "duration_s": 4.0, "sensor": "fullframe", "aperture_f": 2.8,
        "rig": {"kind": "handheld", "params": {"shake": 0.0}},
        "camera": {"interp": "linear", "keys": [
            {"t": 0.0, "pos": [float(a[0]), float(a[1]), fz + 1.5], "aim": [float(b[0]), float(b[1]), fz + 1.2], "focal_mm": 24},
            {"t": 4.0, "pos": [float(b[0]), float(b[1]), fz + 1.5], "aim": [float(b[0]), float(b[1]), fz + 1.2], "focal_mm": 24}]},
        "actors": {}, "pieces": [], "notes": []})
    solved = sim.solve(tl, twin, maps=maps)
    routed = [k for k in solved.camera_keys if k.routed]
    assert routed, "the straight line through the furniture should have been routed round it"
    assert any(n.startswith("walked round") for n in solved.notes)
    assert not solved.camera_keys[0].routed and not solved.camera_keys[-1].routed
    ts = [k.t for k in solved.camera_keys]
    assert ts == sorted(ts) and 0.0 < min(k.t for k in routed) <= max(k.t for k in routed) < 4.0
    again = sim.solve(solved, twin, maps=maps)
    assert sum(k.routed for k in again.camera_keys) == len(routed)
    ev = sim.evaluate(solved, twin, maps=maps)
    off = [f for f in ev.frames if any("off the standable floor" in p for p in f.problems)]
    assert not off, [f.t for f in off][:5]
    d = solved.to_dict()
    assert any(k.get("routed") for k in d["camera"]["keys"])
    assert sim.Timeline.from_dict(d).camera_keys[1].routed is True
