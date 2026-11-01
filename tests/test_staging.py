"""Staging: scenes that look directed, agree across their frames, and are lit
for the hour.

The breakdown may now say how an actor holds themselves (pose), what their
hands are doing (business, naming a thing the labels placed) and what they
look at (eyeline). The engine turns the eyeline into a facing, sits a seated
actor on the labelled chair, composes the shot on a third with look room,
and the finisher holds every frame of a scene to its first finished frame at
one hour. Every test asks a question with a geometric or textual answer; the
one live test is marked slow and runs only when asked for.
"""

from __future__ import annotations

import io
import json
import math
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

from locaish.agent import rehearsal as ra
from locaish.film import rehearsal as R
from locaish.film import render, space
from locaish.meta import FakeMetaClient
from locaish.meta import finish as F
from locaish.product import cast as castmod
from locaish.product import figures as fm

RED, BLUE = (230, 80, 60), (60, 130, 230)


@pytest.fixture(scope="module")
def room(clean):
    twin = clean[0].twin
    return twin, space.floor_maps(twin), space.occupancy(twin)


# ---------------------------------------------------------------------------
# the breakdown carries staging, and bad values fall back
# ---------------------------------------------------------------------------


def test_normalise_breakdown_keeps_good_staging_and_drops_bad():
    marks = {"M1": (0.0, 0.0), "M2": (1.0, 0.0)}
    cast = {"Maya": 1.68, "Theo": 1.84}
    data = {"title": "T", "beats": [
        {"kind": "enter", "actor": "Maya", "to_mark": "M1", "pose": "Sitting", "business": "typing at the laptop",
         "eyeline": "the window"},
        {"kind": "enter", "actor": "Theo", "to_mark": "M2", "pose": "crouching", "business": {"not": "a phrase"},
         "eyeline": "Maya"},
        {"kind": "line", "actor": "Theo", "text": "Hi.", "pose": "leans", "business": "x" * 200, "eyeline": "Theo"},
        {"kind": "line", "actor": "Maya", "text": "Hi.", "pose": None, "business": "null", "eyeline": "the laptop"},
        {"kind": "line", "actor": "Maya", "text": "Hey.", "eyeline": "the ceiling fan"},
    ]}
    bd = ra.normalise_breakdown(data, marks, cast, labels=["laptop", "desk"])
    b = bd["beats"]
    assert (b[0]["pose"], b[0]["business"], b[0]["eyeline"]) == ("sitting", "typing at the laptop", "window")
    assert b[1]["pose"] is None and b[1]["business"] is None and b[1]["eyeline"] == "Maya"
    # A verb form is a pose; a wall of text is cut to a phrase; one's own name is no eyeline.
    assert b[2]["pose"] == "leaning" and len(b[2]["business"]) == ra.MAX_BUSINESS_CHARS and b[2]["eyeline"] is None
    # "null" is nothing; a labelled thing in the room is an eyeline.
    assert b[3]["business"] is None and b[3]["eyeline"] == "laptop"
    # A thing nobody labelled is not.
    assert b[4]["eyeline"] is None


def test_typed_scene_round_trips_staging_targets_and_composition():
    text = """TITLE: Staged
MARKS: A=(0,0.6) B=(-1,-1)
TARGETS: desk=(0.5,1.2)
CAST: Maya=1.68 Theo=1.84
1. Maya enters at A   [sitting, typing at the laptop, desk]
2. Theo enters at B   [standing, hands in pockets, Maya]   [cam (1,-1.5) h1.5 35mm on Theo thirds over Maya]
3. Maya: "You said you'd call."   [-, -, window]
4. Theo: "I said I'd try."
"""
    sc = R.parse_scene(text)
    assert sc.targets == {"desk": (0.5, 1.2)}
    assert (sc.beats[0].pose, sc.beats[0].business, sc.beats[0].eyeline) == ("sitting", "typing at the laptop", "desk")
    assert (sc.beats[2].pose, sc.beats[2].business, sc.beats[2].eyeline) == (None, None, "window")
    assert not sc.beats[3].staged and sc.beats[1].staged
    sp = sc.beats[1].shot
    assert sp.compose and sp.ots_over == "Maya" and sp.focal_mm == 35
    d = sc.to_dict()
    assert json.loads(json.dumps(d)) == d
    assert R.parse_scene(R.format_scene(sc)).to_dict() == d
    assert R.Scene.from_dict(d).to_dict() == d
    # A beat with no staging prints none, so an old scene reads back as it was.
    assert "[-, -, -]" not in R.format_scene(sc)
    with pytest.raises(ValueError):
        R.Beat(1, "line", "Maya", text="x", pose="crouching")


# ---------------------------------------------------------------------------
# the eyeline becomes a facing
# ---------------------------------------------------------------------------


def _heading_to(a, b) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _ang_close(a: float, b: float, tol: float = 2.0) -> bool:
    return abs((a - b + 180.0) % 360.0 - 180.0) <= tol


def test_facing_follows_the_eyeline_to_window_door_actor_and_camera(room):
    twin, maps, occ = room
    door = next(o for o in twin.structure.openings if o.kind == "door").center[:2]
    windows = [o.center[:2] for o in twin.structure.openings if o.kind != "door"]
    text = """MARKS: A=(0,0.6) B=(-1,-1)
CAST: Maya=1.68 Theo=1.84
1. Maya enters at A   [standing, looking out of the window, window]
2. Theo enters at B   [standing, hands in pockets, door]   [cam (1,-1.5) h1.5 35mm on Theo]
3. Theo: "You came back."   [-, -, Maya]
4. Maya: "I never left."   [-, -, camera]
5. hold 1s
"""
    sc = R.parse_scene(text)
    reh = R.rehearse(twin, sc, maps=maps, occupancy=occ)
    a = (0.0, 0.6)
    b = (-1.0, -1.0)
    nearest_window = min(windows, key=lambda w: math.hypot(w[0] - a[0], w[1] - a[1]))
    # Beat 1: Maya looks at the nearest window, not into the room's centre.
    f1 = reh.frame_at(0.1)
    assert _ang_close(f1.actors["Maya"][2], _heading_to(a, nearest_window))
    # Beat 2: Theo looks at the door; Maya keeps her window while he enters.
    f2 = reh.frame_at(reh.beats[1].t_start + 0.1)
    assert _ang_close(f2.actors["Theo"][2], _heading_to(b, door))
    assert _ang_close(f2.actors["Maya"][2], _heading_to(a, nearest_window))
    # Beat 3: Theo speaks to Maya and faces her; Maya's eyeline persists through his line.
    f3 = reh.frame_at(reh.beats[2].t_start + 0.1)
    assert _ang_close(f3.actors["Theo"][2], _heading_to(b, a))
    assert _ang_close(f3.actors["Maya"][2], _heading_to(a, nearest_window))
    # Beat 4: Maya looks into the lens.
    f4 = reh.frame_at(reh.beats[3].t_start + 0.1)
    assert _ang_close(f4.actors["Maya"][2], _heading_to(a, (1.0, -1.5)))
    assert f4.staging["Maya"] == {"pose": "standing", "business": "looking out of the window", "eyeline": "camera"}
    assert f4.speaker == "Maya" and f4.line == "I never left."
    # Beat 5 (a hold with no staging): everyone keeps their last eyeline.
    f5 = reh.frame_at(reh.beats[4].t_start + 0.2)
    assert _ang_close(f5.actors["Maya"][2], _heading_to(a, (1.0, -1.5)))
    assert _ang_close(f5.actors["Theo"][2], _heading_to(b, a))
    d = reh.to_dict()
    assert json.loads(json.dumps(d))["frames"][0]["staging"]["Maya"]["eyeline"] == "window"


def test_without_staging_the_old_rules_decide_the_facing(room):
    twin, maps, occ = room
    text = """MARKS: A=(0,0.6) B=(-1,-1)
CAST: Maya=1.68 Theo=1.84
1. Maya enters at A
2. Theo enters at B
3. Theo: "You came back."   [cam (1,-1.5) h1.5 35mm on Theo]
"""
    reh = R.rehearse(twin, R.parse_scene(text), maps=maps, occupancy=occ)
    f = reh.frame_at(reh.beats[2].t_start + 0.1)
    assert _ang_close(f.actors["Theo"][2], _heading_to((-1, -1), (0, 0.6)))
    assert _ang_close(f.actors["Maya"][2], _heading_to((0, 0.6), (-1, -1)))
    assert f.staging == {} and f.composition is None
    # The camera aims dead at the eyeline, as before.
    assert f.camera[1][:2] == pytest.approx(f.actors["Theo"][:2])


# ---------------------------------------------------------------------------
# a seated actor sits on the chair the labels found
# ---------------------------------------------------------------------------


def test_sitting_snaps_to_a_labelled_chair_within_reach_and_faces_the_desk():
    marks = {"M1": (0.0, 0.0), "M2": (3.0, 3.0), "M3": (-2.0, 0.0)}
    contents = [
        {"label": "chair", "xyz": [0.6, 0.3, 0.4]},
        {"label": "desk", "x_m": 0.9, "y_m": 1.1, "height_m": 0.7},
        {"label": "bunk bed", "xyz": [3.0, 4.5, 0.6]},     # 1.5 m from M2: too far
        {"label": "backpack", "xyz": [-2.1, 0.1, 0.2]},     # near M3 but not a seat
    ]
    beats = [
        {"kind": "enter", "actor": "Theo", "to_mark": "M1", "pose": "sitting", "business": "typing", "eyeline": None},
        {"kind": "move", "actor": "Maya", "to_mark": "M2", "pose": "sitting", "business": None, "eyeline": None},
        {"kind": "enter", "actor": "Ana", "to_mark": "M3", "pose": "sitting", "business": None, "eyeline": "Theo"},
        {"kind": "line", "actor": "Theo", "text": "Hi.", "pose": "sitting", "business": None, "eyeline": None},
    ]
    targets: dict = {}
    notes = ra.stage_seats(beats, marks, contents, targets)
    assert beats[0]["to_mark"] == "M1@chair" and marks["M1@chair"] == (0.6, 0.3)
    assert beats[0]["eyeline"] == "desk" and targets["desk"] == (0.9, 1.1)
    assert beats[1]["to_mark"] == "M2", "a bed 1.5 m away is not the seat the breakdown meant"
    assert beats[2]["to_mark"] == "M3" and beats[2]["eyeline"] == "Theo"
    assert beats[3].get("to_mark") is None
    assert any("sits on the labelled chair" in n for n in notes) and any("faces the desk" in n for n in notes)
    # No labels at all: nothing moves and nothing is said.
    assert ra.stage_seats([dict(beats[1])], dict(marks), None, {}) == []


def test_label_points_reads_both_label_shapes():
    pts = ra.label_points([{"label": "chair", "xyz": [1, 2, 0.4]}, {"label": "desk", "x_m": 3, "y_m": 4},
                           {"label": "nowhere"}, {"xyz": [0, 0, 0]}])
    assert pts == [("chair", (1.0, 2.0)), ("desk", (3.0, 4.0))]


# ---------------------------------------------------------------------------
# poses: a lean tilts the shoulders, a raised forearm reaches the surface
# ---------------------------------------------------------------------------


def test_leaning_tilts_the_shoulders_and_rests_a_forearm_at_a_metre():
    stand = fm.joints(1.75)
    lean = fm.joints(1.75, pose="leaning")
    assert stand["shoulder_l"][2] == pytest.approx(stand["shoulder_r"][2])
    assert abs(lean["shoulder_l"][2] - lean["shoulder_r"][2]) > 0.04, "the shoulders tilt"
    assert lean["wrist_r"][2] == pytest.approx(fm.REST_HEIGHT_M, abs=0.05) and lean["wrist_r"][1] > 0.15
    assert lean["wrist_l"][2] < 0.9, "the other arm hangs"
    # Weight on one leg: the free leg is set out to its side.
    assert abs(lean["ankle_l"][0]) > abs(lean["ankle_r"][0]) + 0.08
    Vs, Fs, _ = fm.figure_parts(1.75)
    Vl, Fl, _ = fm.figure_parts(1.75, pose="leaning")
    assert Vs.shape == Vl.shape and not np.allclose(Vs, Vl)
    assert Vl[:, 2].min() == pytest.approx(0.0) and Vl[:, 2].max() == pytest.approx(1.75, abs=0.02)


def test_arms_variants_move_the_hands_and_scale_with_stature():
    down = fm.joints(1.75)
    folded = fm.joints(1.75, arms="folded")
    pockets = fm.joints(1.75, arms="pockets")
    raised = fm.joints(1.75, arms="one_raised")
    # Folded: both wrists cross to the far side of the chest, above the waist.
    assert folded["wrist_r"][0] < 0.05 and folded["wrist_l"][0] > -0.05 and folded["wrist_r"][2] > 1.05
    # Pockets: hands at hip height, in front of the body, nearer the centre than hanging.
    assert 0.85 < pockets["wrist_r"][2] < 1.0 and pockets["wrist_r"][1] > 0.08 and abs(pockets["wrist_r"][0]) < abs(down["wrist_r"][0])
    # One raised: the right wrist at the surface, the left as it hangs.
    assert raised["wrist_r"][2] == pytest.approx(fm.REST_HEIGHT_M, abs=0.02) and raised["wrist_l"][2] == pytest.approx(down["wrist_l"][2])
    short = fm.joints(1.55, arms="one_raised")
    assert short["wrist_r"][2] == pytest.approx(fm.REST_HEIGHT_M * 1.55 / 1.75, abs=0.02)
    with pytest.raises(ValueError):
        fm.figure_parts(1.75, arms="akimbo")
    # The business picks the arms; the figure takes them.
    assert fm.arms_for_business("arms folded") == "folded" and fm.arms_for_business("hands in pockets") == "pockets"
    assert fm.arms_for_business("typing at the laptop") == "one_raised" and fm.arms_for_business("waiting") == "down"
    fig = fm.figure_for_actor(castmod.Actor(name="Maya", height_m=1.68, colour=RED), (0, 0), 90, 0.0, pose="leaning", arms="pockets")
    assert fig.mesh is not None and fig.label == "Maya"
    assert ra.pose_for({"pose": "sitting", "business": "typing at the laptop"}, 0.0) == ("sitting", "one_raised")
    assert ra.pose_for({"pose": "sitting"}, 1.0) == ("walking", "down")
    assert ra.pose_for({}, 0.0) == ("standing", "down")


# ---------------------------------------------------------------------------
# composition: a third, look room, headroom, never off the frame
# ---------------------------------------------------------------------------


def _project_x(frame: R.Frame, name: str, height_m: float) -> float:
    pos, aim, focal, _ = frame.camera
    cam = render.camera_at(pos, aim, focal)
    x, y, _ = frame.actors[name]
    uv = cam.project((x, y, R.EYELINE_RATIO * height_m))
    assert uv is not None
    return uv[0] / cam.width_px


def test_composed_shot_puts_the_subject_on_a_third_with_look_room(room):
    twin, maps, occ = room
    # Theo at B looks at Maya at A; camera at (1, -1.5) sees A to Theo's frame left.
    text = """MARKS: A=(0,0.6) B=(-1,-1)
CAST: Maya=1.68 Theo=1.84
1. Maya enters at A   [standing, arms folded, Theo]
2. Theo enters at B   [standing, hands in pockets, Maya]   [cam (1,-1.5) h1.5 35mm on Theo thirds]
3. Theo: "You came back."
4. Theo: "I did."   [cam (1,-1.5) h1.5 35mm on Theo thirds over Maya]
5. Theo: "Close."   [cam (-0.4,-1.3) h1.6 85mm on Theo thirds]
"""
    reh = R.rehearse(twin, R.parse_scene(text), maps=maps, occupancy=occ)
    f = reh.frame_at(reh.beats[1].t_start + 0.1)
    comp = f.composition
    assert comp["subject"] == "Theo" and comp["third"] in ("left", "right") and comp["look_room"]
    x = _project_x(f, "Theo", 1.84)
    # On a third: near 0.32 or 0.68 of the width, on the side the composition names.
    assert abs(x - (0.32 if comp["third"] == "left" else 0.68)) < 0.03
    # Look room on the side he looks toward: Maya's side of the frame is the open one.
    xm = _project_x(f, "Maya", 1.68)
    assert (xm > x) == (comp["third"] == "left")
    # Headroom: the aim sits under the eyeline.
    assert f.camera[1][2] == pytest.approx(R.EYELINE_RATIO * 1.84 + twin.structure.floor_z - R.HEADROOM_M, abs=0.01)
    # The framing's distance is to the eyeline, not the offset point.
    tx, ty, _ = f.actors["Theo"]
    assert f.framing["distance_m"] == pytest.approx(math.dist((1.0, -1.5, 1.5), (tx, ty, R.EYELINE_RATIO * 1.84)), abs=0.02)
    # The over-the-shoulder keeps Maya (the foreground) on the edge opposite Theo.
    f4 = reh.frame_at(reh.beats[3].t_start + 0.1)
    assert "over Maya" in f4.composition["why"]
    x4 = _project_x(f4, "Theo", 1.84)
    xm4 = _project_x(f4, "Maya", 1.68)
    assert (xm4 > x4) == (f4.composition["third"] == "left")
    # The 85 mm from 1.4 m cannot hold a third without cutting the body: centred, and nothing off the frame.
    f5 = reh.frame_at(reh.beats[4].t_start + 0.1)
    assert f5.composition["third"] == "centre" and not f5.composition["look_room"]
    assert abs(_project_x(f5, "Theo", 1.84) - 0.5) < 0.02
    # Every composed frame keeps the subject's shoulders inside the picture.
    for fr in reh.frames:
        if fr.composition is None or fr.composition["third"] == "centre":
            continue
        pos, aim, focal, _ = fr.camera
        cam = render.camera_at(pos, aim, focal)
        x, y, _ = fr.actors["Theo"]
        right = cam.right[:2] / max(np.linalg.norm(cam.right[:2]), 1e-9)
        for side in (-1, 1):
            uv = cam.project((x + side * R.BODY_HALF_WIDTH_M * right[0], y + side * R.BODY_HALF_WIDTH_M * right[1], 1.2))
            assert uv is not None and 0.0 <= uv[0] <= cam.width_px
    assert reh.to_dict()["frames"][4]["composition"]["third"] in ("left", "right")


def test_placement_composes_every_shot_and_names_the_ots_foreground(clean):
    from locaish.film import coverage as cov
    from locaish.film import sweep as sweepmod

    twin = clean[0].twin
    source = cov.LocalSetups(sweepmod.sweep(twin))
    marks, _ = ra._marks_catalogue(twin, source)
    names = list(marks)
    a, b = names[0], names[min(1, len(names) - 1)]
    beats = [
        {"kind": "enter", "actor": "Maya", "to_mark": a, "pose": "standing", "business": None, "eyeline": "Theo"},
        {"kind": "enter", "actor": "Theo", "to_mark": b, "pose": "leaning", "business": "hands in pockets", "eyeline": "Maya",
         "shot": {"size": "ms", "subject": "Theo", "second_subject": "Maya", "height": "eye", "movement": "static",
                  "no_window_behind": False, "description": "over Maya at Theo"}},
    ]
    placed, placements = ra.place_shots(beats, marks, twin, source)
    assert placed[1].pose == "leaning" and placed[1].business == "hands in pockets" and placed[1].eyeline == "Maya"
    sp = placed[1].shot
    if sp is not None:
        assert sp.compose is True
        assert sp.ots_over in ("Maya", None)


def test_a_walk_shot_is_not_placed_on_the_subjects_first_step():
    from locaish.film import coverage as cov

    shot = cov.Shot(number=1, description="", size="mls", subject="Maya").normalised()
    end = (3.0, 0.0)
    rows = [
        {"cam_x": 0.2, "cam_y": 0.1, "cam_z": 1.5, "focal_mm": 35.0, "setup_id": "on-the-start-mark"},
        {"cam_x": 1.5, "cam_y": 0.6, "cam_z": 1.5, "focal_mm": 35.0, "setup_id": "beside-the-walk"},
        {"cam_x": 1.5, "cam_y": -2.0, "cam_z": 1.5, "focal_mm": 35.0, "setup_id": "clear"},
    ]
    st, verdict = ra._choose_row(rows, shot, end, {}, {"Maya": 1.68}, subject_start=(0.0, 0.0))
    assert st["setup_id"] == "clear" and verdict["skipped_rows"] == 2
    assert ra._dist_to_segment((1.5, 0.6), (0, 0), (3, 0)) == pytest.approx(0.6)
    assert ra._dist_to_segment((-1.0, 0.0), (0, 0), (3, 0)) == pytest.approx(1.0)
    # Without a start (a static beat) the best row stands.
    st2, _ = ra._choose_row(rows, shot, end, {}, {"Maya": 1.68})
    assert st2["setup_id"] == "on-the-start-mark"


# ---------------------------------------------------------------------------
# the finisher: the anchor rides, the hour is said once, the staging is said
# ---------------------------------------------------------------------------


def _png(w=80, h=50, colour=(90, 120, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format="PNG")
    return buf.getvalue()


def _frame_png(w=160, h=90) -> bytes:
    im = Image.new("RGB", (w, h), (120, 118, 110))
    d = ImageDraw.Draw(im)
    d.rectangle([int(0.2 * w), int(0.3 * h), int(0.4 * w), int(0.95 * h)], fill=RED)
    d.rectangle([int(0.6 * w), int(0.1 * h), int(0.8 * w), int(0.95 * h)], fill=BLUE)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _cast():
    return [castmod.Actor(name="Maya", height_m=1.68, colour=RED, wardrobe="a dark green jumper", description="19, short brown hair"),
            castmod.Actor(name="Theo", height_m=1.84, colour=BLUE, wardrobe="grey t-shirt", description="20, dark curly hair")]


def _staged_meta(i: int, cam, aim, clock: str, speaker: str = "Maya") -> dict:
    return {
        "t": float(i), "clock": clock, "beat": 3,
        "actors": {"Maya": [-1.72, -1.22, 45.0], "Theo": [-1.02, -0.52, -135.0]},
        "camera": {"pos": list(cam), "aim": list(aim), "focal_mm": 35.0, "shot": 3},
        "sun": {"direct_sun": True, "windows_lit": [0], "azimuth_deg": 229.5 + 0.1 * i, "elevation_deg": 19.9, "compass": "SW"},
        "framing": {"subject": "Maya", "size_name": "Medium close-up", "distance_m": 2.3},
        "staging": {"Maya": {"pose": "standing", "business": "hands in pockets", "eyeline": "Theo"},
                    "Theo": {"pose": "sitting", "business": "typing at the laptop", "eyeline": "desk"}},
        "speaker": speaker, "line": "You said you'd call?" if speaker == "Maya" else "I said I'd try.",
        "composition": {"subject": "Maya", "third": "left", "look_room": True, "offset_m": 0.41, "headroom_m": 0.06},
    }


def test_sequence_prompt_says_pose_business_eyeline_expression_composition_hour_and_azimuth():
    meta = _staged_meta(0, (0.28, -0.12, 1.55), (-1.5, -1.4, 1.52), "15:00:04")
    p = F.build_sequence_prompt(meta=meta, cast=_cast(), headshot_index={}, light=F.frame_light(meta))
    assert "Maya" in p and "hands in pockets" in p and "looking at Theo, not at the camera" in p
    assert "SITTING down, typing at the laptop, looking at the desk" in p
    assert "asking a question" in p and "listening to Maya" in p
    assert "standing exactly where the figure is" not in p, "a staged person is not the passport sentence"
    assert "Maya sits on the left third of the frame" in p and "room in front of Maya on the right" in p
    assert "at 15:00" in p and "azimuth 230 degrees" in p and "from the SW" in p
    # Unstaged frames keep the old sentence, and the clock is to the minute.
    bare = dict(meta)
    bare.pop("staging")
    bare.pop("composition")
    p2 = F.build_sequence_prompt(meta=bare, cast=_cast(), headshot_index={})
    assert "standing exactly where the figure is" in p2 and "Composition" not in p2
    assert F.frame_light(meta)["clock"] == "15:00"


def test_anchor_rides_in_every_later_frame_with_the_nearest_neighbour_and_one_hour(tmp_path):
    fake = FakeMetaClient(image=_png())
    # Two setups: frames 0-2 from one camera, 3-5 from another; the clock runs on.
    frames = []
    for i in range(6):
        cam = (0.28, -0.12, 1.55) if i < 3 else (-1.2, 1.08, 1.55)
        aim = (-1.5, -1.4, 1.52) if i < 3 else (1.0, 0.2, 1.52)
        frames.append((f"frame_{i:04d}.png", _frame_png(), _staged_meta(i, cam, aim, f"15:00:{i:02d}", speaker="Maya" if i < 3 else "Theo")))
    out = F.finish_sequence(frames, client=fake, cast=_cast(), headshots={"Theo": _png(8, 8)}, out_dir=tmp_path / "fin",
                            workers=2, room_refs=lambda m: [b"\xff\xd8\xff1", b"\xff\xd8\xff2"])
    assert len(out) == 6 and fake.calls == 6
    first = fake.log[0]["prompt"]
    assert "first photograph of this scene" not in first and "previous finished frame" not in first
    assert fake.log[0]["references"] == 4   # frame, headshot, two room photographs
    # Every later frame says the hour and azimuth the scene's first frame has, whatever its own clock.
    for call in fake.log[1:]:
        assert "at 15:00" in call["prompt"] and "azimuth 230 degrees" in call["prompt"]
    # Frames 1-2 share the anchor's camera: the anchor rides once, as both references.
    for call in fake.log[1:3]:
        assert call["references"] == 5 and "Image 5 is the previous finished frame of this scene and its first photograph" in call["prompt"]
    # Frames 3-4 moved the camera, but every finished frame so far stands at the
    # anchor's camera, so the anchor is as near as any and rides once.
    for call in fake.log[3:5]:
        assert call["references"] == 5 and "and its first photograph" in call["prompt"]
    # Frame 5 (the chunk after 3 and 4) has a finished frame of its own setup:
    # the anchor rides AND the nearest finished frame (by camera) does too.
    call = fake.log[5]
    assert call["references"] == 6
    assert "Image 5 is the first photograph of this scene" in call["prompt"]
    assert "Image 6 is the previous finished frame of this scene from the camera nearest this one" in call["prompt"]
    assert "image 2 is Theo's headshot" in call["prompt"] and "Image 3 and image 4 are photographs of THIS EXACT ROOM" in call["prompt"]
    assert out[0].meta["anchor"] is False and all(f.meta["anchor"] for f in out[1:])
    # Staged people in every prompt.
    assert all("typing at the laptop" in c["prompt"] and "hands in pockets" in c["prompt"] for c in fake.log)
    assert all("Maya sits on the left third" in c["prompt"] for c in fake.log)


def test_anchor_survives_a_failed_first_frame(tmp_path):
    from locaish.meta import MetaError

    class Flaky(FakeMetaClient):
        def generate_image(self, prompt, *, references=None, model=None, size=None):
            if "15:00:00" in prompt or self.calls == 0:
                self.calls += 1
                raise MetaError("fell over", status=500)
            return super().generate_image(prompt, references=references, model=model, size=size)

    fake = Flaky(image=_png())
    frames = [(f"frame_{i:04d}.png", _frame_png(), _staged_meta(i, (0.28, -0.12, 1.55), (-1.5, -1.4, 1.52), f"15:00:{i:02d}"))
              for i in range(4)]
    out = F.finish_sequence(frames, client=fake, cast=_cast(), headshots={}, out_dir=tmp_path / "fin", workers=2)
    assert [f.name for f in out] == ["frame_0001.png", "frame_0002.png", "frame_0003.png"]
    # The failed call is not logged: log[0] is frame 1, which became the anchor.
    assert "first photograph" not in fake.log[0]["prompt"], "frame 1 became the anchor"
    assert all("first photograph" in c["prompt"] for c in fake.log[1:])


def test_nearest_finished_is_shared_with_the_world_finisher():
    pos = [(0, 0, 1.5), (0.1, 0, 1.5), (3, 0, 1.5), (0.1, 0, 1.5)]
    dirs = [(1, 0, 0)] * 4
    finished = {0: object(), 1: object(), 2: object(), 3: object()}
    assert F.nearest_finished(1, finished, pos, dirs, k=2) == [3, 0]
    assert F.nearest_finished(0, finished, pos, dirs, k=1, prefer_recent=True) == [3]
    assert F.nearest_finished(0, finished, pos, dirs, k=1) == [1]


# ---------------------------------------------------------------------------
# live, once, when asked: the staged scene on the real job
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("LOCAISH_LIVE_TESTS"), reason="set LOCAISH_LIVE_TESTS=1 to run against the Meta Model API")
def test_live_breakdown_stages_every_beat(tmp_path):
    from locaish.cli import _load_dotenv
    from locaish.film import coverage as cov
    from locaish.film import sweep as sweepmod
    from locaish.meta import MetaClient
    from locaish.types import Twin

    _load_dotenv()
    twin = Twin.load("twins/studio/d3752e88f78a/IMG_6191.twin")
    source = cov.LocalSetups(sweepmod.sweep(twin))
    labels = json.loads(open("twins/studio/d3752e88f78a/labels.json").read())["labels"]
    bd = ra.breakdown("Maya comes in. Theo is at the desk. Maya: You said you'd call. Theo: I said I'd try.",
                      twin, source, {"Maya": 1.68, "Theo": 1.84}, client=MetaClient(), contents=labels)
    assert bd["beats"] and all(b["pose"] for b in bd["beats"])
