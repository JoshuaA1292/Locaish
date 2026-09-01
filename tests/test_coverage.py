"""Coverage planning: the shot list answered against a room, with and without a model.

The deterministic planner runs on the synthetic `clean` room through the
local sweep backend, so nothing here needs ClickHouse or a network. The
agent workflow is driven end to end by a scripted model double -- the ADK
plumbing (state, tools, the loop, exit_loop, the breakdown schema) is what
is under test, not Gemini's judgement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from locaish.film import coverage as cov
from locaish.film import optics
from locaish.film import sweep as sweepmod

BRIEF = """1. Master wide: MAYA at the counter, JON enters
2. Medium shot of MAYA at the counter, no window behind
3. Close-up MAYA on a 50, eye level
4. Two-shot MAYA and JON, medium
5. Low angle medium close-up of JON
6. Insert: hands on the counter"""


@pytest.fixture(scope="module")
def room(clean):
    result, _fx = clean
    twin = result.twin
    sw = sweepmod.sweep(twin)
    return twin, cov.LocalSetups(sw)


# ---------------------------------------------------------------------------
# reading a shot list without a model
# ---------------------------------------------------------------------------


def test_parse_shot_lines_reads_size_lens_height_and_names():
    shots = cov.parse_shot_lines(BRIEF)
    assert [s.number for s in shots] == [1, 2, 3, 4, 5, 6]
    assert shots[0].size == "ls" and shots[0].subject == "MAYA" and shots[0].second_subject == "JON"
    assert shots[1].no_window_behind is True and shots[1].size == "ms"
    assert shots[2].lens_mm == 50.0 and shots[2].height == "eye" and shots[2].size == "cu"
    assert shots[3].second_subject == "JON"
    assert shots[4].height == "low" and shots[4].size == "mcu" and shots[4].subject == "JON"
    # A line naming nobody keeps the previous subject; "insert" is an ECU.
    assert shots[5].subject == "JON" and shots[5].size == "ecu"


def test_size_vocabulary_maps_to_the_sweeps_sizes():
    assert cov.normalise_size("wide two-shot") == "ls"
    assert cov.normalise_size("OTS on JON") == "mcu"
    assert cov.normalise_size("nonsense") is None
    for key in cov.SIZE_KEYS:
        assert key in optics.SHOT_BY_KEY


# ---------------------------------------------------------------------------
# the predicates and the search
# ---------------------------------------------------------------------------


def test_predicates_compile_to_sql_and_to_the_same_numpy_mask(room):
    twin, local = room
    marks = cov.auto_marks(local, twin.name, ["MAYA", "JON"])
    shot = cov.Shot(number=1, description="two-shot", size="ms", subject="MAYA",
                    second_subject="JON", no_window_behind=True, height="eye").normalised()
    preds = cov.predicates(shot, marks, location=twin.name)
    sql = cov.compile_sql(preds, db="locaish", table="shot_setups")
    assert "shot_size = 'ms'" in sql and "window_behind_subject = 0" in sql
    assert "atan2" in sql and "fov_h_deg" in sql
    assert sql.rstrip().endswith("LIMIT 5")
    rows, count, _ = local.search(preds)
    assert count >= 0
    for r in rows:
        assert r["shot_size"] == "ms" and r["visible"] == 1 and r["window_behind_subject"] == 0
        assert 1.3 <= r["cam_z"] < 1.8


def test_relaxation_reports_what_it_gave_up(room):
    twin, local = room
    marks = cov.auto_marks(local, twin.name, ["A"])
    # A 100 mm extreme close-up at a low height is unlikely to exist as swept;
    # the search must come back with something and say what it dropped.
    shot = cov.Shot(number=1, description="x", size="ecu", subject="A", lens_mm=100, height="low").normalised()
    rows, count, sql, relaxed = cov.find_setup(local, shot, marks, location=twin.name)
    if rows:
        assert relaxed or rows[0]["shot_size"] == "ecu"
    else:
        assert relaxed, "an empty answer must explain itself"


def test_auto_marks_keeps_two_characters_apart(room):
    twin, local = room
    marks = cov.auto_marks(local, twin.name, ["MAYA", "JON"])
    a, b = marks["MAYA"], marks["JON"]
    assert float(np.hypot(a[0] - b[0], a[1] - b[1])) >= cov.MIN_MARK_SEPARATION_M - 1e-6


# ---------------------------------------------------------------------------
# the deterministic planner, end to end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def planned(room, tmp_path_factory):
    twin, local = room
    workdir = tmp_path_factory.mktemp("plan")
    plan = cov.plan(twin, cov.parse_shot_lines(BRIEF), local, title="Kitchen", brief=BRIEF,
                    workdir=workdir, render=True)
    return plan, workdir


def test_plan_places_shots_and_documents_each_one(planned):
    plan, workdir = planned
    assert plan.planned >= 4
    for ps in plan.shots:
        if ps.setup is None:
            continue
        assert ps.candidates >= 1
        assert ps.sql.startswith("SELECT")
        assert ps.frame and (workdir / "frames" / ps.frame).exists()
        # The setup really frames the mark the shot asked for.
        mx, my = plan.marks[ps.shot.subject]
        assert abs(ps.setup["subj_x"] - mx) < 0.05 and abs(ps.setup["subj_y"] - my) < 0.05
        assert ps.setup["visible"] == 1
    two = next(ps for ps in plan.shots if ps.shot.number == 4)
    assert two.second_mark is not None
    assert (workdir / "plan.json").exists() and (workdir / "shotlist.txt").exists()


def test_plan_round_trips_through_json(planned):
    plan, workdir = planned
    again = cov.CoveragePlan.load(workdir)
    assert again.plan_id == plan.plan_id
    assert [s.shot.number for s in again.shots] == [s.shot.number for s in plan.shots]
    assert again.marks == {k: tuple(v) for k, v in plan.marks.items()}
    assert again.shots[0].setup == plan.shots[0].setup


def test_floor_plan_draws_every_placed_camera_and_mark(planned):
    plan, _ = planned
    svg = plan.floor_plan_svg
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    for name in plan.marks:
        assert f">{name}<" in svg
    # every placed shot number appears in a camera label
    placed = [ps.shot.number for ps in plan.shots if ps.setup]
    for n in placed:
        assert f">{n}<" in svg or f",{n}<" in svg or f">{n},".replace(">", ">") in svg or f"{n}," in svg


def test_shot_list_text_carries_the_measurements(planned):
    plan, _ = planned
    text = cov.render_text(plan)
    assert "COVERAGE PLAN" in text and "MARKS" in text
    ps = next(ps for ps in plan.shots if ps.setup)
    assert f"{int(ps.setup['focal_mm']):>3} mm" in text
    assert f"({ps.setup['cam_x']:+.2f}, {ps.setup['cam_y']:+.2f})" in text


# ---------------------------------------------------------------------------
# the agent workflow, with a scripted model
# ---------------------------------------------------------------------------


def _scripted_models():
    """Model doubles for the three agents. The placer's policy is what a
    well-behaved Gemini would do: read the last tool result, act on it."""
    from google.adk.models import BaseLlm, LlmResponse
    from google.genai import types

    def call(name, **args):
        return LlmResponse(content=types.Content(role="model", parts=[
            types.Part.from_function_call(name=name, args=args)]))

    def text(t):
        return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=t)]))

    def last_function_response(req):
        for content in reversed(req.contents or []):
            for part in reversed(content.parts or []):
                fr = getattr(part, "function_response", None)
                if fr is not None:
                    return fr.name, dict(fr.response or {})
        return None, None

    class Breakdown(BaseLlm):
        model: str = "scripted-breakdown"

        async def generate_content_async(self, llm_request, stream=False):
            yield text(json.dumps({
                "title": "Kitchen argument",
                "marks": [{"character": "MAYA", "mark": "M1"}, {"character": "JON", "mark": "M2"}],
                "shots": [
                    {"number": 1, "description": "Master wide, MAYA at the counter, JON enters",
                     "size": "ls", "subject": "MAYA", "second_subject": "JON"},
                    {"number": 2, "description": "Close-up MAYA", "size": "cu", "subject": "MAYA",
                     "lens_mm": 50, "height": "eye", "no_window_behind": True},
                    {"number": 3, "description": "Impossible: a 100mm ECU from the floor", "size": "ecu",
                     "subject": "JON", "lens_mm": 100, "height": "low"},
                ],
            }))

    class Placer(BaseLlm):
        model: str = "scripted-placer"
        calls: list = []
        current: int = 0
        tried: dict = {}

        async def generate_content_async(self, llm_request, stream=False):
            name, resp = last_function_response(llm_request)
            self.calls.append(name)
            if name is None or name in ("accept_shot", "skip_shot"):
                yield call("next_shot")
            elif name == "next_shot":
                if resp.get("done"):
                    yield text("done")
                else:
                    self.current = int(resp["shot"]["number"])
                    rows = resp.get("candidates") or []
                    tried = self.tried.setdefault(self.current, [])
                    if not rows:
                        yield call("skip_shot", shot_number=self.current, reason="nothing the room can hold")
                    else:
                        tried.append(rows[0]["setup_id"])
                        yield call("place_shot", shot_number=self.current, setup_id=rows[0]["setup_id"],
                                   reasoning="top of the ranking")
            elif name == "find_setups":
                rows = resp.get("rows") or []
                tried = self.tried.setdefault(self.current, [])
                fresh = [r for r in rows if r["setup_id"] not in tried]
                if not fresh:
                    yield call("skip_shot", shot_number=self.current, reason="nothing the room can hold")
                else:
                    tried.append(fresh[0]["setup_id"])
                    yield call("place_shot", shot_number=self.current, setup_id=fresh[0]["setup_id"],
                               reasoning="best score with a clean sightline")
            elif name == "place_shot":
                rv = resp.get("review") or {}
                if resp.get("accepted"):
                    yield text("placed: " + str(resp.get("setup", {}).get("setup_id")))
                elif rv.get("verdict") == "adjust" and resp.get("attempts_left", 0) > 0:
                    yield call("find_setups", shot_number=self.current, drop="lens,height")
                else:
                    yield call("accept_shot", shot_number=self.current)
            else:
                yield text("done")

    class Notes(BaseLlm):
        model: str = "scripted-notes"

        async def generate_content_async(self, llm_request, stream=False):
            yield text("The room holds the scene on short glass; the close-up needed a wider lens than asked.")

    return {"breakdown": Breakdown(), "planner": Placer(), "notes": Notes()}


def test_agent_workflow_places_reviews_retries_and_assembles_a_plan(room, tmp_path):
    from locaish.agent import coverage as agentcov

    twin, local = room
    reviews: list[tuple[int, int]] = []

    def reviewer(image_path: Path, shot: cov.Shot, row: dict) -> cov.Review:
        assert image_path.exists()
        n = sum(1 for s, _ in reviews if s == shot.number) + 1
        reviews.append((shot.number, n))
        # First attempt on shot 2 is sent back for a different lens; everything else keeps.
        if shot.number == 2 and n == 1:
            return cov.Review(score=4.0, verdict="adjust", notes="too tight", suggestion={"lens_mm": 35})
        return cov.Review(score=8.0, verdict="keep", notes="clean frame", model="scripted")

    events: list[dict] = []
    models = _scripted_models()
    workdir = tmp_path / "plans" / "abc123"
    plan = agentcov.plan_coverage(
        twin, workdir=workdir, source=local, brief="INT. KITCHEN - DAY. MAYA and JON argue.",
        on_event=events.append, models=models, reviewer=reviewer, timeout_s=300,
    )

    assert plan.planner == "gemini"
    assert plan.title == "Kitchen argument"
    assert set(plan.marks) == {"MAYA", "JON"}
    assert [ps.shot.number for ps in plan.shots] == [1, 2, 3]
    # The breakdown's marks were honoured: MAYA on M1, JON on M2.
    described = {m["name"]: (m["x"], m["y"]) for m in cov.describe_marks(twin, local.marks(twin.name))}
    assert plan.marks["MAYA"] == described["M1"] and plan.marks["JON"] == described["M2"]

    one, two, three = plan.shots
    assert one.setup and one.review and one.review.verdict == "keep"
    # Shot 2 was placed twice and the better-reviewed attempt stands.
    assert two.setup and two.attempts == 2 and two.review.score == 8.0
    assert (2, 1) in reviews and (2, 2) in reviews
    # Every rendered attempt is on disk, and the plan is saved.
    assert len(list((workdir / "frames").glob("shot_*.png"))) >= 3 or three.setup is None
    assert (workdir / "plan.json").exists()
    assert plan.floor_plan_svg.startswith("<svg")
    assert any(w.startswith("DP notes:") for w in plan.warnings)

    kinds = [e["kind"] for e in events]
    assert "shot" in kinds and "call" in kinds and "result" in kinds
    tools_called = [e["tool"] for e in events if e["kind"] == "call"]
    assert tools_called[:2] == ["next_shot", "place_shot"]
    assert "find_setups" in tools_called          # the retry on shot 2 went back to the table
    # One placer conversation per shot: next_shot was asked once per shot.
    assert tools_called.count("next_shot") == 3
    usage = [e for e in plan.trace if e.get("kind") == "usage"]
    assert usage and usage[0]["model_calls"] >= 5


def test_agent_refuses_cleanly_without_credentials(room, tmp_path, monkeypatch):
    from locaish.agent import AgentUnavailable
    from locaish.agent import coverage as agentcov

    for var in ("GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    twin, local = room
    assert agentcov.agent_configured() is False
    with pytest.raises(AgentUnavailable):
        agentcov.plan_coverage(twin, workdir=tmp_path / "p", source=local, brief="x")


# ---------------------------------------------------------------------------
# the craft: what the sweep measures and what the planner enforces
# ---------------------------------------------------------------------------


def test_sweep_measures_depth_light_and_corners(room):
    twin, local = room
    c = local.c
    for name in ("background_depth_m", "backup_room_m", "key_angle_deg", "key_quality",
                 "axis_wall_angle_deg", "portrait_ok"):
        assert name in c and len(c[name]) == len(local.sweep)
    assert (c["background_depth_m"] > 0).all() and (c["background_depth_m"] <= sweepmod.MAX_DEPTH_M).all()
    assert (c["backup_room_m"] >= 0).all() and (c["backup_room_m"] <= sweepmod.MAX_BACKUP_M).all()
    ang = c["axis_wall_angle_deg"]
    assert ((ang == -1) | ((ang >= 0) & (ang <= 45.01))).all()
    assert set(np.unique(c["key_quality"].astype(str))) <= {"none", "front", "three-quarter", "side", "rim", "back"}
    # portrait_ok is exactly the distance rule on tight framings, and 1 otherwise.
    tight = np.isin(c["shot_size"].astype(str), list(sweepmod.TIGHT_SIZES))
    expect = ~tight | (c["distance_m"] >= sweepmod.PORTRAIT_MIN_DISTANCE_M)
    assert (c["portrait_ok"].astype(bool) == expect).all()


def test_key_quality_bands_follow_the_lighting_vocabulary():
    assert sweepmod.KEY_QUALITY(-1) == "none"
    assert sweepmod.KEY_QUALITY(10) == "front"
    assert sweepmod.KEY_QUALITY(45) == "three-quarter"
    assert sweepmod.KEY_QUALITY(90) == "side"
    assert sweepmod.KEY_QUALITY(130) == "rim"
    assert sweepmod.KEY_QUALITY(170) == "back"


def test_first_hit_marches_to_the_first_occupied_voxel():
    grid = np.zeros((40, 5, 5), dtype=bool)
    grid[30, 2, 2] = True                       # a wall 3.0 m along x at 0.1 m cells
    origin = np.zeros(3)
    start = np.array([[0.25, 0.25, 0.25]])
    fwd = np.array([[1.0, 0.0, 0.0]])
    hit = sweepmod._first_hit_batch(grid, origin, 0.1, start, fwd, 8.0)
    assert 2.6 < float(hit[0]) < 2.9
    back = sweepmod._first_hit_batch(grid, origin, 0.1, start, -fwd, 8.0)
    assert float(back[0]) == 8.0                 # nothing behind: open, capped


def test_line_of_action_is_enforced_after_the_first_shot(room):
    twin, local = room
    marks = cov.auto_marks(local, twin.name, ["MAYA", "JON"])
    ctx = cov.PlanContext()
    master = cov.Shot(number=1, description="master", size="ls", subject="MAYA", second_subject="JON").normalised()
    rows, count, sql, relaxed = cov.find_setup(local, master, marks, location=twin.name, context=ctx)
    assert rows
    ctx.learn(master, rows[0], marks)
    assert ctx.line == ("MAYA", "JON") and ctx.line_side in (1, -1)
    single = cov.Shot(number=2, description="cu", size="cu", subject="JON").normalised()
    preds = cov.predicates(single, marks, location=twin.name, context=ctx)
    names = [p.name for p in preds]
    assert "same side of the line" in names
    rows2, _, sql2, relaxed2 = cov.find_setup(local, single, marks, location=twin.name, context=ctx)
    assert "sign(" in sql2
    for r in rows2:
        if "crossed the line of action" in " ".join(relaxed2):
            break
        assert ctx.side_of(marks, (r["cam_x"], r["cam_y"])) == ctx.line_side


def test_reverse_single_asks_to_match_lens_and_distance(room):
    twin, local = room
    marks = cov.auto_marks(local, twin.name, ["MAYA", "JON"])
    ctx = cov.PlanContext()
    ctx.line = ("MAYA", "JON")
    a = cov.Shot(number=1, description="cu", size="cu", subject="MAYA").normalised()
    rows, *_ = cov.find_setup(local, a, marks, location=twin.name, context=ctx)
    assert rows
    ctx.learn(a, rows[0], marks)
    b = cov.Shot(number=2, description="cu", size="cu", subject="JON").normalised()
    preds = cov.predicates(b, marks, location=twin.name, context=ctx)
    names = [p.name for p in preds]
    assert any(n.startswith("match reverse lens") for n in names)
    assert any(n.startswith("match reverse distance") for n in names)
    why = cov.explain(b, rows[0], marks, ctx)
    assert "reverse" in why


def test_over_the_shoulder_geometry_and_parsing():
    shots = cov.parse_shot_lines("1. Over JON's shoulder onto MAYA, medium close-up\n2. JON looms over her, menacing")
    assert shots[0].ots and shots[0].subject == "MAYA" and shots[0].second_subject == "JON"
    assert shots[1].height == "low"
    marks = {"MAYA": (0.0, 0.0), "JON": (1.0, 0.0)}
    preds = cov.predicates(shots[0], marks, location="x")
    assert any(p.name == "over the shoulder geometry" for p in preds)
    c = {
        "setup_id": np.array([1, 2]), "visible": np.array([1, 1]),
        "subj_x": np.array([0.0, 0.0]), "subj_y": np.array([0.0, 0.0]),
        "cam_x": np.array([1.8, -1.5]), "cam_y": np.array([0.2, 0.0]),
        "yaw_deg": np.array([np.degrees(np.arctan2(-0.2, -1.8)), 0.0]),
        "fov_h_deg": np.array([60.0, 60.0]), "distance_m": np.array([1.9, 1.5]),
        "shot_size": np.array(["mcu", "mcu"], dtype=object),
    }
    ots = next(p for p in preds if p.name == "over the shoulder geometry")
    mask = ots.mask(c)
    assert bool(mask[0]) and not bool(mask[1])   # behind JON's shoulder yes; on the far side no


def test_ordering_weighs_the_craft_in_sql_and_numpy(room):
    twin, local = room
    wide = cov.Shot(number=1, description="master", size="ls", subject="A").normalised()
    tight = cov.Shot(number=2, description="cu", size="cu", subject="A").normalised()
    assert "background_depth_m" in cov.order_sql(wide) and "axis_wall_angle_deg" in cov.order_sql(wide)
    assert "key_quality" in cov.order_sql(tight) and "portrait_ok" in cov.order_sql(tight)
    idx = np.arange(min(50, len(local.sweep)))
    vals = cov.order_values(tight, local.c, idx)
    assert vals.shape == (len(idx),) and np.isfinite(vals).all()
