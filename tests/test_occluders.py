"""Other actors as occluders: the room is not the only thing that can stand
between the lens and the subject.

The clean fixture room is about 5.2 x 4.1 m; (0, 0.6), (-1, -1), (1, -1) and
(0.5, -1.5) are standable floor, and (0.2, -1.8) is inside the outline.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from locaish.agent import rehearsal as ra
from locaish.film import coverage as cov
from locaish.film import rehearsal as R
from locaish.film import sim, space
from locaish.film import sweep as sweepmod
from locaish.film.render import EYELINE_RATIO


@pytest.fixture(scope="module")
def room(clean):
    twin = clean[0].twin
    return twin, space.floor_maps(twin), space.occupancy(twin)


@pytest.fixture(scope="module")
def table(clean):
    twin = clean[0].twin
    return twin, cov.LocalSetups(sweepmod.sweep(twin))


CAM = (0.0, 0.0, 1.5)
MAYA = (4.0, 0.0, EYELINE_RATIO * 1.68)


# ---------------------------------------------------------------------------
# the geometry
# ---------------------------------------------------------------------------


def test_a_body_on_the_line_blocks_and_beside_it_does_not():
    assert R.actor_blocks(CAM, MAYA, {"Theo": (2.0, 0.0)}) == ["Theo"]
    assert R.actor_blocks(CAM, MAYA, {"Theo": (2.0, 0.25)}) == ["Theo"], "a 30 cm body 25 cm off axis still cuts the line"
    assert R.actor_blocks(CAM, MAYA, {"Theo": (2.0, 0.45)}) == []
    assert R.actor_blocks(CAM, MAYA, {"Theo": (2.0, -0.45)}) == []


def test_behind_the_subject_or_behind_the_camera_does_not_block():
    assert R.actor_blocks(CAM, MAYA, {"Theo": (5.0, 0.0)}) == [], "behind Maya is in the background, not in the way"
    assert R.actor_blocks(CAM, MAYA, {"Theo": (3.9, 0.0)}) == [], "at Maya's shoulder is in the frame with her"
    assert R.actor_blocks(CAM, MAYA, {"Theo": (-1.0, 0.0)}) == [], "behind the lens is not in the frame at all"


def test_blockers_come_nearest_first_and_the_subject_is_never_listed():
    others = {"Theo": (3.0, 0.1), "Ana": (1.0, -0.1), "Far": (2.0, 2.0)}
    assert R.actor_blocks(CAM, MAYA, others) == ["Ana", "Theo"]


def test_a_short_actor_under_a_high_eyeline_does_not_block():
    high = (0.0, 0.0, 2.6)                      # a high angle looking down at Maya's eyes
    others = {"Kid": (1.0, 0.0)}
    assert R.actor_blocks(high, MAYA, others, heights={"Kid": 1.1}) == [], "the line passes 2.3 m up, over a 1.1 m head"
    assert R.actor_blocks(high, MAYA, others, heights={"Kid": 1.1}, floor_z=1.4) == ["Kid"], "the floor is where the twin says"
    assert R.actor_blocks(high, MAYA, others) == ["Kid"], "without heights a body is a column"
    assert R.actor_blocks(CAM, MAYA, others, heights={"Kid": 1.1}) == [], "an eye-level line passes 1.5 m up, over the same head"
    assert R.actor_blocks(CAM, MAYA, others, heights={"Kid": 1.6}) == ["Kid"], "a 1.6 m head reaches it"


def test_too_close_is_a_horizontal_radius_round_the_lens():
    others = {"Theo": (0.4, 0.1), "Ana": (0.0, 0.59), "Maya": (0.7, 0.0)}
    assert R.actor_too_close(CAM, others) == ["Theo", "Ana"]
    assert R.actor_too_close(CAM, others, min_m=0.45) == ["Theo"]
    assert R.actor_too_close(CAM, {}) == []


def test_degenerate_segment_blocks_nothing():
    assert R.actor_blocks((0.0, 0.0, 1.5), (0.0, 0.0, 1.5), {"Theo": (0.0, 0.0)}) == []


# ---------------------------------------------------------------------------
# placement
# ---------------------------------------------------------------------------


def _mcu_request(subject="Maya", second=None):
    return {"size": "mcu", "subject": subject, "second_subject": second, "height": "eye", "movement": "static",
            "no_window_behind": False, "description": "her face"}


def test_placement_passes_over_a_row_another_actor_stands_in_front_of(table):
    twin, source = table
    marks, _ = ra._marks_catalogue(twin, source)
    maya_mark = next(iter(marks))
    shot = cov.Shot(number=1, description="", size="mcu", subject="Maya", height="eye").normalised()
    rows, _, _, _ = cov.find_setup(source, shot, {"Maya": marks[maya_mark]}, location=twin.name, limit=ra.CANDIDATE_ROWS)
    assert rows, "the table holds a medium close-up of the first mark"
    best = rows[0]
    cam = np.array([best["cam_x"], best["cam_y"]])
    subj = np.array(marks[maya_mark])
    # Theo stands exactly on the eyeline, 60% of the way from the lens to Maya.
    theo_xy = cam + 0.6 * (subj - cam)
    marks2 = dict(marks)
    marks2["T"] = (float(theo_xy[0]), float(theo_xy[1]))
    beats = [
        {"kind": "enter", "actor": "Maya", "to_mark": maya_mark, "text": None, "seconds": None, "shot": None},
        {"kind": "enter", "actor": "Theo", "to_mark": "T", "text": None, "seconds": None, "shot": None},
        {"kind": "line", "actor": "Maya", "to_mark": None, "text": "You said you'd call.", "seconds": None,
         "shot": _mcu_request()},
    ]
    placed, placements = ra.place_shots(beats, marks2, twin, source, cast={"Maya": 1.68, "Theo": 1.84})
    p = placements[0]
    assert p["placed"] and p["best_setup_id"] == best["setup_id"]
    chosen = p["setup"]
    if p["blocked_by"]:
        assert p["blocked_by"] == ["Theo"] and chosen["setup_id"] == best["setup_id"] and "Theo" in p["why"]
    else:
        assert chosen["setup_id"] != best["setup_id"], "the best row has Theo's back in it"
        assert p["skipped_rows"] >= 1
        cam3 = (chosen["cam_x"], chosen["cam_y"], chosen["cam_z"])
        eye = (subj[0], subj[1], EYELINE_RATIO * 1.68)
        assert R.actor_blocks(cam3, eye, {"Theo": marks2["T"]}, heights={"Theo": 1.84}) == []
        assert R.actor_too_close(cam3, {"Theo": marks2["T"]}) == []
    spec = placed[2].shot
    assert isinstance(spec, R.ShotSpec) and spec.cam_xy == (pytest.approx(chosen["cam_x"]), pytest.approx(chosen["cam_y"]))


def test_a_two_shot_exempts_the_second_subject_from_the_occluder_test(table):
    twin, source = table
    marks, _ = ra._marks_catalogue(twin, source)
    maya_mark = next(iter(marks))
    shot = cov.Shot(number=1, description="", size="mcu", subject="Maya", height="eye").normalised()
    rows, _, _, _ = cov.find_setup(source, shot, {"Maya": marks[maya_mark]}, location=twin.name, limit=ra.CANDIDATE_ROWS)
    best = rows[0]
    cam = np.array([best["cam_x"], best["cam_y"]])
    subj = np.array(marks[maya_mark])
    theo_xy = cam + 0.6 * (subj - cam)
    marks2 = dict(marks)
    marks2["T"] = (float(theo_xy[0]), float(theo_xy[1]))
    beats = [
        {"kind": "enter", "actor": "Maya", "to_mark": maya_mark, "text": None, "seconds": None, "shot": None},
        {"kind": "enter", "actor": "Theo", "to_mark": "T", "text": None, "seconds": None, "shot": None},
        {"kind": "line", "actor": "Maya", "to_mark": None, "text": "You said you'd call.", "seconds": None,
         "shot": _mcu_request(second="Theo")},
    ]
    _, placements = ra.place_shots(beats, marks2, twin, source, cast={"Maya": 1.68, "Theo": 1.84})
    p = placements[0]
    assert p["placed"] and p["blocked_by"] == [] and p["too_close"] == []


def test_choose_row_skips_blocked_rows_and_a_lens_inside_the_subject():
    def row(i, x, y, z=1.5):
        return {"setup_id": i, "cam_x": x, "cam_y": y, "cam_z": z, "score": 90 - i}
    shot = cov.Shot(number=1, description="", size="mcu", subject="Theo").normalised()
    theo, maya = (0.0, 0.0), (2.0, 0.0)
    others = {"Maya": maya}
    rows = [row(1, 3.0, 0.0),            # Maya stands on this eyeline
            row(2, 0.0, 0.0, 0.45),      # the lens is at Theo's feet looking up: no eyeline at all
            row(3, 2.0, 0.3),            # Maya is 0.3 m from this lens
            row(4, 0.0, -2.0),           # clear
            row(5, -2.0, 0.0)]           # clear, but Maya is stacked straight behind Theo
    st, verdict = ra._choose_row(rows, shot, theo, others, {"Theo": 1.84, "Maya": 1.68})
    assert st["setup_id"] == 4 and verdict["skipped_rows"] == 3 and verdict["blocked_by"] == []
    st, verdict = ra._choose_row(rows[:3], shot, theo, others, None)
    assert st["setup_id"] == 1 and verdict["occluded"] and verdict["blocked_by"] == ["Maya"]
    st, verdict = ra._choose_row([rows[4], rows[3]], shot, theo, others, None)
    assert st["setup_id"] == 4, "nobody stacked behind the subject beats table order"
    assert ra._choose_row([rows[4]], shot, theo, others, None)[1]["behind_subject"] == ["Maya"]


def test_placement_with_nobody_else_in_the_room_keeps_the_best_row(table):
    twin, source = table
    marks, _ = ra._marks_catalogue(twin, source)
    maya_mark = next(iter(marks))
    beats = [
        {"kind": "enter", "actor": "Maya", "to_mark": maya_mark, "text": None, "seconds": None, "shot": _mcu_request()},
    ]
    _, placements = ra.place_shots(beats, marks, twin, source)
    p = placements[0]
    assert p["placed"] and p["skipped_rows"] == 0 and p["setup"]["setup_id"] == p["best_setup_id"]
    assert p["blocked_by"] == [] and "occluded" not in p


# ---------------------------------------------------------------------------
# the rehearsal
# ---------------------------------------------------------------------------


CROSSING_SCENE = """
TITLE: The crossing
MARKS: A=(0.0,0.6) B=(-1.0,-1.0) C=(1.0,-1.0)
CAST: Maya=1.68 Theo=1.84
1. Maya enters at A   [cam (0.2,-1.8) h1.5 50mm on Maya]
2. Theo enters at B
3. Maya: "Stay where you are."
4. Theo moves to C
5. hold 1s
"""


def test_an_actor_walking_through_the_eyeline_is_reported_at_the_time_he_crosses(room):
    twin, maps, occ = room
    scene = R.parse_scene(CROSSING_SCENE)
    rh = R.rehearse(twin, scene, maps=maps, occupancy=occ)
    st = rh.shots[0]
    hits = [p for p in st.problems if "Theo crosses between the camera and Maya" in p]
    assert len(hits) == 1, st.problems
    t = float(re.match(r"at ([\d.]+) s", hits[0]).group(1))
    walk = next(b for b in rh.beats if b.kind == "move" and b.actor == "Theo")
    assert walk.t_start <= t <= walk.t_end, "he is only in the way while he walks"
    # And at that instant he is, in fact, on the line.
    fr = rh.frame_at(t)
    others = {"Theo": fr.actors["Theo"][:2]}
    assert R.actor_blocks(fr.camera[0], fr.camera[1], others, heights=scene.cast, floor_z=maps.floor_z) == ["Theo"]
    assert any(p.startswith("Shot 1") and "Theo crosses" in p for p in rh.problems)
    assert not any("fills the frame" in p for p in st.problems)
    # Standing at B before the walk and at C after it he is clear.
    for fr in (rh.frame_at(walk.t_start - R.DT_S), rh.frames[-1]):
        assert R.actor_blocks(fr.camera[0], fr.camera[1], {"Theo": fr.actors["Theo"][:2]}) == []


def test_an_actor_on_the_lens_fills_the_frame(room):
    twin, maps, occ = room
    scene = R.parse_scene("""
MARKS: A=(0.0,0.6) D=(0.5,-1.5)
CAST: Maya=1.68 Theo=1.84
1. Maya enters at A   [cam (0.2,-1.8) h1.5 50mm on Maya]
2. Theo enters at D
3. hold 1s
""")
    rh = R.rehearse(twin, scene, maps=maps, occupancy=occ)
    st = rh.shots[0]
    hits = [p for p in st.problems if "Theo is 0.4 m from the lens and fills the frame" in p]
    assert len(hits) == 1, st.problems
    assert not any("crosses between" in p for p in st.problems), "0.4 m off axis beside the lens is not on the eyeline"


def test_the_subject_alone_raises_no_actor_problem(room):
    twin, maps, occ = room
    scene = R.parse_scene("""
MARKS: A=(0.0,0.6) C=(1.0,-1.0)
CAST: Maya=1.68
1. Maya enters at A   [cam (0.2,-1.8) h1.5 50mm on Maya]
2. Maya moves to C
""")
    rh = R.rehearse(twin, scene, maps=maps, occupancy=occ)
    assert not any("crosses between" in p or "from the lens" in p for p in rh.shots[0].problems)


# ---------------------------------------------------------------------------
# the simulator
# ---------------------------------------------------------------------------


def _timeline(theo_keys, *, duration=4.0):
    return sim.Timeline.from_dict({
        "id": "0cc1uder", "title": "Held on Maya", "fps": 24, "duration_s": duration, "sensor": "ff",
        "aperture_f": 2.8, "rig": {"kind": "handheld", "params": {}},
        "camera": {"interp": "linear", "keys": [
            {"t": 0.0, "pos": [0.2, -1.8, 1.4], "look_at": "Maya", "focal_mm": 50.0},
            {"t": duration, "pos": [0.2, -1.8, 1.4], "look_at": "Maya", "focal_mm": 50.0}]},
        "actors": {
            "Maya": {"height_m": 1.68, "keys": [{"t": 0.0, "pos": [0.0, 0.6], "facing_deg": -90.0}]},
            "Theo": {"height_m": 1.84, "keys": theo_keys},
        },
        "pieces": [], "notes": [],
    })


def test_sim_flags_the_frames_where_an_actor_walks_through_the_eyeline(room):
    twin, maps, occ = room
    tl = _timeline([{"t": 0.0, "pos": [-1.0, -1.0], "facing_deg": 0.0},
                    {"t": 4.0, "pos": [1.0, -1.0], "facing_deg": 0.0}])
    ev = sim.evaluate(tl, twin, maps=maps, occupancy=occ)
    seg = [s for s in ev.segments if s["text"] == "Theo crosses between the camera and the subject"]
    assert len(seg) == 1, ev.segments
    seg = seg[0]
    assert seg["severity"] == "fail" and seg["frames"] >= 2
    # He crosses x ~ 0.13 at y = -1: about 1.1 m into a 2 m walk over 4 s.
    assert 1.4 < seg["t_start"] < 2.6 and seg["t_end"] > seg["t_start"]
    for fr in (ev.frames[0], ev.frames[-1]):
        assert not any("Theo" in p.text for p in fr.problems), "at B and at C he is clear of the line"
    assert all(fr.subject == "Maya" for fr in ev.frames)
    assert ev.summary["feasible"] is False


def test_sim_warns_when_an_actor_stands_on_the_lens(room):
    twin, maps, occ = room
    tl = _timeline([{"t": 0.0, "pos": [0.5, -1.5], "facing_deg": 0.0}])
    ev = sim.evaluate(tl, twin, maps=maps, occupancy=occ)
    warn = [s for s in ev.segments if "Theo is within 0.6 m of the lens" in s["text"]]
    assert len(warn) == 1 and warn[0]["severity"] == "warn" and warn[0]["frames"] == len(ev.frames)
    assert not any("crosses between" in s["text"] for s in ev.segments)


def test_sim_actor_checks_run_without_a_twin():
    tl = _timeline([{"t": 0.0, "pos": [0.1, -0.6], "facing_deg": 0.0}])
    ev = sim.evaluate(tl)
    assert any(s["text"] == "Theo crosses between the camera and the subject" for s in ev.segments)
    d = sim.sample_time(tl, 1.0)
    assert any("Theo crosses" in p for p in d["problems"])
