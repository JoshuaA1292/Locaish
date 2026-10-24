"""The rehearsal: a scene run inside the measured room, deterministically.

Bodies walk the standable floor at walking pace, lines take the time they
take, the camera holds or rides, the sun is where the ephemeris says, and the
schedule falls out. Every test here asks a question with a geometric answer.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from datetime import date, datetime, time

import numpy as np
import pytest

from locaish.film import rehearsal as R
from locaish.film import space
from locaish.types import Georeference, Plane, PointCloud, QAReport, Structure, Twin

LA = dict(latitude=34.05, longitude=-118.24)

SCENE_TEXT = """
TITLE: The kitchen, morning
MARKS: A=(0.0,0.6) B=(-1.0,-0.6) DOOR=(-1.4,0.4) C=(0.0,-1.2)
CAST: Maya=1.68 Theo=1.84
DATE: 2026-11-14
START: 10:40
STRIKE: sofa before 4
1. Theo enters at B
2. Maya enters at DOOR
3. Maya moves to A
4. Theo: "You came back."   [cam (0.2,-1.8) h1.5 50mm on Theo]
5. Maya: "I never left."    [cam (-0.8,-1.9) h1.4 35mm on Maya dolly to (0.6,-1.9)]
6. hold 2s
7. Maya moves to C   [cam (0.2,-1.8) h1.5 50mm on Maya]
8. Maya exits to DOOR
"""


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _box_twin(w=6.0, d=4.0, h=2.7, *, obstacle=None, n=60_000):
    """A rectangular room, optionally with a solid block (x0, x1, y0, y1, z1) in it."""
    rng = np.random.default_rng(0)
    parts = [
        np.column_stack([rng.uniform(0, w, n), rng.uniform(0, d, n), np.zeros(n)]),
        np.column_stack([rng.uniform(0, w, n // 2), np.zeros(n // 2), rng.uniform(0, h, n // 2)]),
        np.column_stack([rng.uniform(0, w, n // 2), np.full(n // 2, d), rng.uniform(0, h, n // 2)]),
        np.column_stack([np.zeros(n // 2), rng.uniform(0, d, n // 2), rng.uniform(0, h, n // 2)]),
        np.column_stack([np.full(n // 2, w), rng.uniform(0, d, n // 2), rng.uniform(0, h, n // 2)]),
        np.column_stack([rng.uniform(0, w, n), rng.uniform(0, d, n), np.full(n, h)]),
    ]
    if obstacle is not None:
        x0, x1, y0, y1, z1 = obstacle
        k = 30_000
        parts.append(np.column_stack([rng.uniform(x0, x1, k), rng.uniform(y0, y1, k), rng.uniform(0, z1, k)]))
    xyz = np.concatenate(parts)
    return Twin(
        name="box",
        points=PointCloud(xyz),
        structure=Structure(
            floor_z=0.0, ceiling_z=h,
            footprint=np.array([[0, 0], [w, 0], [w, d], [0, d]], dtype=float),
            planes=[Plane(normal=[0, 0, 1], offset=0.0, kind="floor", area=w * d)],
        ),
        qa=QAReport(verdict="pass"),
    )


@pytest.fixture(scope="module")
def room(clean):
    """The clean fixture twin with its maps, computed once."""
    twin = clean[0].twin
    return twin, space.floor_maps(twin), space.occupancy(twin)


@pytest.fixture
def georeferenced(room):
    twin, maps, occ = room
    old = twin.georeference
    # +X points west: the windows on the +Y wall face south, and get LA sun all day.
    twin.georeference = Georeference(heading_deg=270.0, heading_source="user", **LA)
    try:
        yield twin, maps, occ
    finally:
        twin.georeference = old


@pytest.fixture(scope="module")
def rehearsed(room):
    twin, maps, occ = room
    scene = R.parse_scene(SCENE_TEXT)
    return R.rehearse(twin, scene, maps=maps, occupancy=occ)


# ---------------------------------------------------------------------------
# the scene format
# ---------------------------------------------------------------------------


def test_parse_scene_reads_every_part():
    sc = R.parse_scene(SCENE_TEXT)
    assert sc.title == "The kitchen, morning"
    assert sc.marks["DOOR"] == (-1.4, 0.4)
    assert sc.cast == {"Maya": 1.68, "Theo": 1.84}
    assert sc.shoot_date == date(2026, 11, 14) and sc.shoot_start == time(10, 40)
    assert sc.strikes == [{"before_shot": 4, "piece": "sofa"}]
    kinds = [b.kind for b in sc.beats]
    assert kinds == ["enter", "enter", "move", "line", "line", "hold", "move", "exit"]
    assert sc.beats[3].text == "You came back."
    assert sc.beats[5].seconds == 2.0
    sp = sc.beats[4].shot
    assert sp.cam_xy == (-0.8, -1.9) and sp.height_m == 1.4 and sp.focal_mm == 35
    assert sp.subject == "Maya" and sp.dolly_to == (0.6, -1.9) and sp.gear_key is None
    assert sc.beats[2].shot is None


def test_scene_round_trips_through_dict_json_and_text():
    sc = R.parse_scene(SCENE_TEXT)
    d = sc.to_dict()
    assert json.loads(json.dumps(d)) == d, "the contract is plain JSON"
    assert R.Scene.from_dict(json.loads(json.dumps(d))).to_dict() == d
    assert R.parse_scene(R.format_scene(sc)).to_dict() == d


def test_a_shot_spec_can_name_size_and_gear():
    sc = R.parse_scene('MARKS: A=(0,0)\nCAST: Z=1.7\n1. Z: "Hi"  [cam (1,1) h1.2 32mm on Z cu dolly to (2,1) gear super-peewee]')
    sp = sc.beats[0].shot
    assert sp.size == "cu" and sp.gear_key == "super-peewee" and sp.focal_mm == 32


def test_unknown_beat_kind_is_refused():
    with pytest.raises(ValueError):
        R.Beat(1, "dance", "Maya")


# ---------------------------------------------------------------------------
# blocking
# ---------------------------------------------------------------------------


def test_astar_finds_a_path_between_standable_cells(room):
    twin, maps, occ = room
    mask = R.walkable(maps)
    rt = R.route(maps, occ, (-1.4, 0.4), (0.0, -1.2), mask=mask)
    assert not rt.blocked
    assert rt.length_m >= math.dist((-1.4, 0.4), (0.0, -1.2)) - 1e-9
    # Every point on the route stands on walkable floor.
    for p in rt.points:
        i, j = maps.index_of(p)[0]
        assert mask[i, j]


def test_astar_goes_around_an_injected_block():
    twin = _box_twin(obstacle=(2.8, 3.2, 1.0, 3.0, 2.4))
    maps = space.floor_maps(twin)
    occ = space.occupancy(twin)
    rt = R.route(maps, occ, (1.0, 2.0), (5.0, 2.0))
    assert not rt.blocked
    assert rt.length_m > 4.0 + 0.5, "the straight line is through the block; the walk is longer"
    # The path keeps a body's clearance from the block the whole way.
    for a, b in zip(rt.points[:-1], rt.points[1:]):
        for f in np.linspace(0, 1, 50):
            x, y = a + f * (b - a)
            dx = max(2.8 - x, 0.0, x - 3.2)
            dy = max(1.0 - y, 0.0, y - 3.0)
            assert math.hypot(dx, dy) >= 0.25, (x, y)


def test_diagonals_do_not_cut_corners():
    # From (1,1) to (2,2) a diagonal step would squeeze between (2,1) and
    # (1,2). Block both and the path must go around.
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 1] = False
    mask[1, 2] = False
    path = R.astar(mask, (1, 1), (2, 2))
    assert path is not None
    assert len(path) > 2, "no corner cutting between two touching obstructions"


def test_walking_time_is_path_length_over_speed(rehearsed):
    walk = next(b for b in rehearsed.beats if b.kind == "move")
    pts = np.asarray(walk.route)
    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    assert walk.duration_s == pytest.approx(length / R.WALK_SPEED_MPS, abs=R.DT_S)
    # And the actor really is on the route midway through the beat.
    mid = rehearsed.frame_at(0.5 * (walk.t_start + walk.t_end))
    x, y, facing = mid.actors[walk.actor]
    s = R.WALK_SPEED_MPS * (mid.t - walk.t_start)
    assert 0 < s < length
    heading = math.degrees(math.atan2(pts[-1][1] - pts[0][1], pts[-1][0] - pts[0][0]))
    assert facing == pytest.approx(heading, abs=1.0), "facing is the direction of travel"


def test_line_timing(rehearsed):
    line = next(b for b in rehearsed.beats if b.kind == "line")
    assert R.line_seconds("You came back.") == pytest.approx(0.35 + 3 / 2.7)
    assert line.duration_s == pytest.approx(R.line_seconds("You came back."))
    # The listener turns to the speaker.
    fr = rehearsed.frame_at(line.t_start + 0.5)
    tx, ty, _ = fr.actors["Theo"]
    mx, my, m_facing = fr.actors["Maya"]
    assert m_facing == pytest.approx(math.degrees(math.atan2(ty - my, tx - mx)), abs=1.0)


def test_a_blocked_move_is_reported_not_crashed():
    twin = _box_twin(obstacle=(2.9, 3.1, 0.0, 4.0, 2.4))   # a wall across the room
    scene = R.Scene(
        title="wall", marks={"L": (1.0, 2.0), "Rt": (5.0, 2.0)}, cast={"Maya": 1.7},
        beats=[R.Beat(1, "enter", "Maya", to_mark="L"), R.Beat(2, "move", "Maya", to_mark="Rt"),
               R.Beat(3, "hold", "", seconds=1.0)],
    )
    reh = R.rehearse(twin, scene)
    walk = reh.beats[1]
    assert walk.blocked and "no walkable path" in walk.reason
    assert any("cannot walk" in p for p in reh.problems)
    # The timeline continues with the actor on the mark.
    x, y, _ = reh.frames[-1].actors["Maya"]
    assert (x, y) == pytest.approx((5.0, 2.0))


# ---------------------------------------------------------------------------
# the camera
# ---------------------------------------------------------------------------


def test_the_dolly_beat_yields_a_move_report_and_its_problems_surface(rehearsed):
    dolly = next(s for s in rehearsed.shots if s.setup["dolly_to"])
    assert dolly.move is not None and dolly.move["name"] == "shot 5 dolly"
    assert dolly.move["feasible"] is False
    assert any("clear radius" in p for p in dolly.problems), "a doorway dolly 15 cm from the wall does not fit"
    assert any(p.startswith("Shot 5") and "clear radius" in p for p in rehearsed.problems)
    assert dolly.setup["gear_key"] == "doorway-dolly"
    # The camera rides over its own beat and parks at the end for the hold.
    start = rehearsed.frame_at(dolly.t_start + R.DT_S)
    parked = rehearsed.frame_at(dolly.beat_end + 0.5)
    assert start.camera[0][0] < parked.camera[0][0]
    assert parked.camera[0][:2] == pytest.approx((0.6, -1.9))


def test_a_dolly_with_room_around_it_is_feasible(room):
    twin, maps, occ = room
    scene = R.Scene(
        title="push", marks={"A": (0.0, 0.6)}, cast={"Maya": 1.68},
        beats=[R.Beat(1, "enter", "Maya", to_mark="A"),
               R.Beat(2, "line", "Maya", text="One two three four five six seven eight.",
                      shot=R.ShotSpec((-0.6, -1.0), 1.4, 35, "Maya", dolly_to=(0.8, -1.0)))],
    )
    reh = R.rehearse(twin, scene, maps=maps, occupancy=occ)
    st = reh.shots[0]
    assert st.move is not None and st.move["feasible"], st.problems
    assert not st.problems


def test_shot_size_tightens_as_the_actor_walks_toward_the_camera(rehearsed):
    walk_in = next(s for s in rehearsed.shots if s.number == 7)
    keys = [s["size"] for s in walk_in.sizes]
    assert len(set(keys)) >= 2, "the lens cannot hold one size while the distance halves"
    own = [f for f in rehearsed.frames if f.camera and f.camera[3] == 7 and f.t <= walk_in.beat_end]
    assert own[0].framing["framed_height_m"] > own[-1].framing["framed_height_m"]
    order = [s.key for s in R.optics.SHOT_SIZES]
    assert order.index(own[-1].framing["shot_size"]) < order.index(own[0].framing["shot_size"])
    assert any("Shot 7" in n and "then" in n for n in rehearsed.notes)


def test_a_beat_without_a_shot_keeps_the_previous_camera(rehearsed):
    hold = next(b for b in rehearsed.beats if b.kind == "hold")
    fr = rehearsed.frame_at(hold.t_start + 1.0)
    assert fr.camera is not None and fr.camera[3] == 5
    before = rehearsed.frame_at(0.25)
    assert before.camera is None, "no camera until the first shot spec"


def test_camera_aims_at_the_subject_eyeline(rehearsed):
    st = next(s for s in rehearsed.shots if s.number == 4)
    fr = rehearsed.frame_at(st.t_start + 0.5)
    pos, aim, focal, no = fr.camera
    tx, ty, _ = fr.actors["Theo"]
    assert aim[:2] == pytest.approx((tx, ty))
    assert aim[2] == pytest.approx(R.EYELINE_RATIO * 1.84, abs=0.01)
    assert pos[2] == pytest.approx(1.5, abs=0.01) and focal == 50


# ---------------------------------------------------------------------------
# the sun
# ---------------------------------------------------------------------------


def test_sun_is_none_without_a_georeference(rehearsed):
    assert all(f.sun is None for f in rehearsed.frames)
    assert any("georeference" in n for n in rehearsed.notes)


def test_sun_tags_frames_when_georeferenced(georeferenced):
    twin, maps, occ = georeferenced
    scene = R.parse_scene(SCENE_TEXT)
    reh = R.rehearse(twin, scene, maps=maps, occupancy=occ)
    f0 = reh.frames[0]
    assert f0.clock == datetime(2026, 11, 14, 10, 40)
    assert isinstance(f0.sun["direct_sun"], bool)
    assert f0.sun["direct_sun"] is True, "south windows in LA at 10:40 solar time in November"
    assert f0.sun["windows_lit"], "and the frame says which"
    assert all(twin.structure.openings[j].kind == "window" for j in f0.sun["windows_lit"])

    night = R.parse_scene(SCENE_TEXT.replace("START: 10:40", "START: 22:00"))
    reh_n = R.rehearse(twin, night, maps=maps, occupancy=occ)
    assert reh_n.frames[0].sun["direct_sun"] is False and reh_n.frames[0].sun["windows_lit"] == []


# ---------------------------------------------------------------------------
# the schedule
# ---------------------------------------------------------------------------


def test_schedule_places_a_lit_window_shot_inside_the_sun_interval(georeferenced):
    twin, maps, occ = georeferenced
    scene = R.parse_scene(SCENE_TEXT)
    reh = R.rehearse(twin, scene, maps=maps, occupancy=occ)
    sch = R.schedule(reh, twin, start=datetime(2026, 11, 14, 5, 0))

    # Camera south of the subject, windows on the north wall of the twin: backlight.
    dep = R.window_dependence(twin, reh.shots[0])
    assert dep is not None and dep["role"] == "backlight"
    sun = R.daylight.sun_schedule(twin, date(2026, 11, 14))
    run = sun["windows"][dep["window"] - 1]["direct_sun"][0]
    lo = datetime.combine(date(2026, 11, 14), time.fromisoformat(run["from"]))
    hi = datetime.combine(date(2026, 11, 14), time.fromisoformat(run["to"]))

    shots = [e for e in sch.entries if e["shot"] is not None]
    assert [e["shot"] for e in shots] == [4, 5, 7]
    for e in shots:
        assert lo <= e["start"] and e["end"] <= hi, e
        assert (e["end"] - e["start"]).total_seconds() / 60 == R.DEFAULT_SETUP_MINUTES + R.MIN_TAKES_MINUTES
    assert shots[0]["start"] > datetime(2026, 11, 14, 5, 0), "nothing shoots before the sun is on the glass"
    assert any(s.startswith("Shot 4 window coverage holds") for s in sch.sentences)
    strike = next(e for e in sch.entries if e["shot"] is None)
    assert "sofa" in strike["reason"] and strike["end"] <= shots[0]["start"]
    assert any("Strike the sofa before setup 4" in s for s in sch.sentences)
    assert not sch.problems


def test_schedule_without_georeference_orders_as_written(rehearsed, room):
    twin, _, _ = room
    sch = R.schedule(rehearsed, twin, start=datetime(2026, 11, 14, 8, 0), setup_minutes=20)
    assert [e["shot"] for e in sch.entries if e["shot"] is not None] == [4, 5, 7]
    assert sch.entries[0]["start"] == datetime(2026, 11, 14, 8, 0)
    assert any("no georeference" in n for n in sch.notes)
    text = R.render_text(rehearsed, sch)
    assert "SCHEDULE" in text and "Shot 5" in text and "clear radius" in text


def test_schedule_names_a_shot_that_cannot_fit_its_light(georeferenced):
    twin, maps, occ = georeferenced
    scene = R.parse_scene(SCENE_TEXT)
    reh = R.rehearse(twin, scene, maps=maps, occupancy=occ)
    # Start after the sun has left the glass: the shot is placed and the loss named.
    sch = R.schedule(reh, twin, start=datetime(2026, 11, 14, 17, 30))
    assert sch.problems and "lose the light" in sch.problems[0]


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------


def test_rehearsal_to_dict_is_json(rehearsed):
    d = rehearsed.to_dict()
    s = json.dumps(d)
    back = json.loads(s)
    assert back["duration_s"] == pytest.approx(rehearsed.duration_s, abs=1e-3)
    assert len(back["frames"]) == len(rehearsed.frames) == int(round(rehearsed.duration_s / R.DT_S)) + 1
    assert back["shots"][1]["move"]["feasible"] is False


def test_animatic_writes_frames_and_json(rehearsed, room, tmp_path):
    twin, _, _ = room
    paths = R.animatic(twin, rehearsed, tmp_path / "anim", every_s=1.0, width_px=320)
    n_samples = int(math.floor(rehearsed.duration_s)) + 1
    with_camera = sum(1 for k in range(n_samples) if rehearsed.frame_at(float(k)).camera is not None)
    assert len(paths) == with_camera > 0
    assert all(p.exists() and p.suffix == ".png" for p in paths)
    meta = json.loads((tmp_path / "anim" / "animatic.json").read_text())
    assert len(meta["frames"]) == n_samples
    assert sum(1 for f in meta["frames"] if f["file"]) == len(paths)
    from PIL import Image
    im = Image.open(paths[0])
    assert im.width == 320 and im.height > 320 * 14 / 24.89


def test_animatic_accepts_plugged_in_figures(rehearsed, room, tmp_path):
    twin, _, _ = room
    seen = []

    def figures_for(frame):
        seen.append(frame.t)
        pts = [[x, y, 1.0] for (x, y, _f) in frame.actors.values()]
        return [R.render.Figure(np.array(pts), np.array([200, 200, 200]), radius_m=0.1)]

    paths = R.animatic(twin, rehearsed, tmp_path / "anim2", every_s=4.0, figures_for=figures_for, width_px=240)
    assert paths and len(seen) == len(paths)


def test_blocking_svg_is_xml_with_the_cast(rehearsed, room):
    twin, _, _ = room
    svg = R.blocking_svg(twin, rehearsed)
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "Maya" in svg and "Theo" in svg
    for mark in ("DOOR", "A", "B", "C"):
        assert f">{mark}<" in svg
    assert "blocking" in svg and "50mm" in svg


def test_render_text_is_a_call_sheet(rehearsed):
    text = R.render_text(rehearsed)
    assert text.startswith("THE KITCHEN, MORNING")
    assert "BEATS" in text and "SETUPS" in text and "PROBLEMS" in text
    assert '"You came back."' in text
    assert "dolly to (0.60, -1.90)" in text
