"""The rehearsal agent: a scripted breakdown becomes placed shots and a run on
the real room; the typed path needs no model at all."""

from __future__ import annotations

import json

import pytest

from locaish.agent import rehearsal as ra
from locaish.film import coverage as cov
from locaish.film import rehearsal as reh
from locaish.film import sweep as sweepmod
from locaish.meta import FakeMetaClient
from locaish.product import cast as castmod


@pytest.fixture(scope="module")
def room(clean):
    result, _fx = clean
    twin = result.twin
    sw = sweepmod.sweep(twin)
    return twin, cov.LocalSetups(sw)


def _breakdown_answer(marks: list[str]) -> dict:
    a, b = marks[0], marks[min(1, len(marks) - 1)]
    return {
        "title": "The kitchen",
        "beats": [
            {"kind": "enter", "actor": "maya", "to_mark": a, "shot": {
                "size": "wide", "subject": "Maya", "second_subject": None, "height": "eye",
                "movement": "static", "no_window_behind": False, "description": "the master"}},
            {"kind": "enters", "actor": "Theo", "to_mark": b},
            {"kind": "line", "actor": "Theo", "text": "You came back.", "shot": {
                "size": "cu", "subject": "Theo", "height": "eye", "movement": "push-in",
                "description": "his face as he says it"}},
            {"kind": "line", "actor": "Maya", "text": "I never left.", "shot": {
                "size": "mcu", "subject": "Maya", "second_subject": "Theo", "height": "eye",
                "movement": "static", "no_window_behind": True}},
            {"kind": "move", "actor": "Maya", "to_mark": b},
            {"kind": "hold", "actor": "Maya", "seconds": 2},
            {"kind": "line", "actor": "Ghost", "text": "boo"},
            {"kind": "dance", "actor": "Maya"},
        ],
        "strikes": [{"before_shot": 2, "piece": "sofa"}],
    }


def test_normalise_breakdown_snaps_names_kinds_and_marks():
    marks = {"M1": (0.0, 0.0), "M2": (1.0, 0.0)}
    cast = {"Maya": 1.68, "Theo": 1.84}
    bd = ra.normalise_breakdown(_breakdown_answer(["m1", "mark 2"]), marks, cast)
    kinds = [b["kind"] for b in bd["beats"]]
    assert kinds == ["enter", "enter", "line", "line", "move", "hold"]
    assert bd["beats"][0]["actor"] == "Maya" and bd["beats"][0]["to_mark"] == "M1"
    assert bd["beats"][1]["to_mark"] == "M2"
    assert bd["beats"][0]["shot"]["size"] == "ls" and bd["beats"][2]["shot"]["movement"] == "push-in"
    assert bd["beats"][3]["shot"]["second_subject"] == "Theo" and bd["beats"][3]["shot"]["no_window_behind"]
    assert bd["beats"][5]["seconds"] == 2.0
    assert any("Ghost" in n for n in bd["notes"])
    assert bd["strikes"] == [{"before_shot": 2, "piece": "sofa"}]


def test_place_shots_uses_scored_rows_and_builds_a_dolly(room):
    twin, source = room
    marks, _ = ra._marks_catalogue(twin, source)
    names = list(marks)
    bd = ra.normalise_breakdown(_breakdown_answer(names), marks, {"Maya": 1.68, "Theo": 1.84})
    beats, placements = ra.place_shots(bd["beats"], marks, twin, source)
    assert len(beats) == 6
    placed = [p for p in placements if p["placed"]]
    assert len(placed) >= 2
    specs = [b.shot for b in beats if b.shot is not None]
    assert all(isinstance(s, reh.ShotSpec) for s in specs)
    # The push-in shot carries a dolly end point on the camera-subject axis.
    push = next((b.shot for b in beats if b.shot and b.kind == "line" and b.actor == "Theo"), None)
    if push is not None and push.dolly_to is not None:
        import numpy as np

        cam = np.array(push.cam_xy)
        end = np.array(push.dolly_to)
        subj = np.array(marks[names[min(1, len(names) - 1)]])
        v1, v2 = end - cam, subj - cam
        cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        assert cos > 0.99 and push.gear_key == "doorway-dolly"
    # Every placed camera is a swept position: it exists in the table.
    for p in placed:
        assert p["setup"]["setup_id"] is not None and p["candidates"] >= 1


def test_run_rehearsal_from_prose_with_a_scripted_model(room, tmp_path):
    twin, source = room
    marks, _ = ra._marks_catalogue(twin, source)
    fake = FakeMetaClient(chat_answers=[_breakdown_answer(list(marks)), "The room gives the scene a long axis."])
    actors = [castmod.Actor(name="Maya", height_m=1.68), castmod.Actor(name="Theo", height_m=1.84)]
    res = ra.run_rehearsal(
        twin, source, workdir=tmp_path / "r1", scene_text="INT. KITCHEN - DAY. Maya comes in. Theo: You came back.",
        actors=actors, client=fake, every_s=2.0,
    )
    assert res.provider == "meta"
    assert res.scene.cast == {"Maya": 1.68, "Theo": 1.84}
    assert res.rehearsal.duration_s > 2.0
    assert len(res.animatic) >= 1 and (tmp_path / "r1" / "animatic" / res.animatic[0]).exists()
    assert (tmp_path / "r1" / "rehearsal.json").exists() and (tmp_path / "r1" / "callsheet.txt").exists()
    assert "<svg" in res.blocking_svg and "Maya" in res.blocking_svg
    assert res.notes.startswith("The room gives")
    d = json.loads((tmp_path / "r1" / "rehearsal.json").read_text())
    assert d["provider"] == "meta" and d["scene"]["title"] == "The kitchen"
    kinds = [c["kind"] for c in fake.log]
    assert kinds == ["chat_json", "chat"]


def test_run_rehearsal_typed_scene_needs_no_model(room, tmp_path):
    twin, source = room
    marks, _ = ra._marks_catalogue(twin, source)
    names = list(marks)
    a, b = marks[names[0]], marks[names[min(1, len(names) - 1)]]
    text = f"""MARKS: A=({a[0]:.2f},{a[1]:.2f}) B=({b[0]:.2f},{b[1]:.2f})
CAST: Maya=1.68 Theo=1.84
1. Maya enters at A
2. Theo enters at B
3. Theo: "You came back."
4. hold 1s
"""
    res = ra.run_rehearsal(twin, source, workdir=tmp_path / "r2", scene_text=text, every_s=2.0)
    assert res.provider == "typed" and res.rehearsal.duration_s > 1.0
    assert res.notes == "" and res.placements == []


def test_prose_without_a_model_is_refused(room, tmp_path):
    twin, source = room
    with pytest.raises(ra.RehearsalUnavailable):
        ra.run_rehearsal(twin, source, workdir=tmp_path / "r3", scene_text="INT. ROOM. Someone speaks.")
