"""Directing the camera: Muse Wonder on the engine's tools, scripted, on the clean room.

The model double answers with tool calls in a fixed order; what is under
test is the loop, the tools, the engine round trip, the look (rendered frame
shown to the vision double), the trace, and the route that runs it all and
saves the result as the timeline the viewer plays.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from locaish.agent import sim as simagent
from locaish.film import space as spacemod
from locaish.meta.client import ChatResult, FakeMetaClient, ToolCall
from locaish.product import cast as castmod


def _spots(twin, n=3):
    maps = spacemod.floor_maps(twin)
    st = maps.standable()
    import numpy as np

    ij = np.argwhere(st)
    pick = ij[np.linspace(0, len(ij) - 1, n).astype(int)]
    xy = np.asarray(maps.world_of(pick), dtype=float).reshape(-1, 2)
    return [tuple(float(v) for v in row) for row in xy], float(maps.floor_z)


def _script(twin, *, look=True):
    (a, b, c), fz = _spots(twin)
    calls = [
        [ToolCall("c1", "describe_room", {})],
        [ToolCall("c2", "set_rig", {"kind": "handheld", "params": {"shake": 0.0}})],
        [ToolCall("c3", "set_actor_keys", {"name": "Maya", "keys": [{"t": 0, "pos": [a[0], a[1]], "facing_deg": 90}]})],
        [ToolCall("c4", "set_camera_keys", {"keys": [
            {"t": 0.0, "pos": [b[0], b[1], fz + 1.5], "look_at": "Maya", "focal_mm": 35},
            {"t": 4.0, "pos": [c[0], c[1], fz + 1.4], "look_at": "Maya", "focal_mm": 50}],
            "interp": "catmull", "duration_s": 4.0})],
        [ToolCall("c5", "evaluate", {})],
    ]
    if look:
        calls.append([ToolCall("c6", "look", {"t": 2.0})])
    calls.append([ToolCall("c7", "finish", {"notes": "A slow walk-in on Maya; mind the sofa at the second metre."})])
    answers = []
    for tcs in calls:
        answers.append(ChatResult(text="", tool_calls=tcs, model="fake"))
    # The vision double answers the DP read; chat_json pops from the same list,
    # so the review answer sits where the look happens.
    if look:
        answers.insert(6, {"score": 7, "verdict": "keep", "notes": "Headroom is right; the far wall sits clean behind her."})
    return answers


def _timeline(twin):
    (a, b, c), fz = _spots(twin)
    return {"id": "t1", "title": "Walk-in", "fps": 12, "duration_s": 4.0, "sensor": "ff", "aperture_f": 2.8,
            "rig": {"kind": "tripod", "params": {}},
            "camera": {"interp": "linear", "keys": [{"t": 0.0, "pos": [b[0], b[1], fz + 1.5], "aim": [a[0], a[1], fz + 1.5],
                                                     "look_at": None, "focal_mm": 35, "roll_deg": 0, "ease": "inout"}]},
            "actors": {}, "pieces": [], "notes": []}


def test_direct_runs_the_tools_evaluates_looks_and_finishes(clean, tmp_path):
    twin = clean[0].twin
    fake = FakeMetaClient(chat_answers=_script(twin))
    events = []
    res = simagent.direct(twin, _timeline(twin), "walk in on Maya over four seconds, 35 to 50",
                          client=fake, actors=[castmod.Actor(name="Maya", height_m=1.68)],
                          workdir=tmp_path / "direct", on_event=events.append)
    assert res.finished and res.notes.startswith("A slow walk-in")
    assert res.timeline["rig"]["kind"] == "handheld"
    own = [k for k in res.timeline["camera"]["keys"] if not k.get("routed")]
    assert len(own) == 2 and res.timeline["actors"]["Maya"]["height_m"] == 1.68
    assert res.evaluation["frames"] and res.evaluation["frames"][0]["actors"]["Maya"]
    assert res.looks == 1 and (tmp_path / "direct" / "looks").is_dir()
    kinds = [e["kind"] for e in events]
    assert "look" in kinds and "review" in kinds
    review = next(e for e in events if e["kind"] == "review")
    assert review["score"] == 7 and review["verdict"] == "keep"
    tools = [t["tool"] for t in res.trace if t["kind"] == "call"]
    assert tools == ["describe_room", "set_rig", "set_actor_keys", "set_camera_keys", "evaluate", "look", "finish"]
    # The vision double was shown an image.
    assert any(e["kind"] == "chat_json" and e["images"] == 1 for e in fake.log)
    assert res.model_calls == 7


def test_direct_without_finish_still_evaluates_and_says_so(clean, tmp_path):
    twin = clean[0].twin
    answers = _script(twin, look=False)[:4]            # stops after setting keys
    answers += ["I think that is the move.", "Yes, done."]
    fake = FakeMetaClient(chat_answers=answers)
    res = simagent.direct(twin, _timeline(twin), "walk in", client=fake, workdir=tmp_path / "d2", max_calls=8)
    assert res.evaluation.get("frames")
    assert "ran out" in res.notes or res.finished


def test_tool_errors_are_returned_not_raised(clean, tmp_path):
    twin = clean[0].twin
    op = simagent._Operator(twin, _timeline(twin), client=FakeMetaClient())
    assert "error" in op.set_rig("zeppelin")
    assert "error" in op.set_camera_keys([{"t": 0, "pos": [1, 2]}])
    assert "error" in op.look(1.0)
    assert "error" in simagent._dispatch(op, "nope", {})
    room = op.describe_room()
    assert "rigs" in room and "floor_z" in room and room["standable_spots_sample"]


# ---------------------------------------------------------------------------
# the route
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def direct_url(clean, tmp_path_factory):
    from locaish import serve as servemod
    from locaish.viewer.build import render_html

    result, _fx = clean
    root = tmp_path_factory.mktemp("direct")
    job = root / "cleanroom"
    job.mkdir()
    result.twin.save(job / "test-clean.twin")
    render_html(result.twin, job / "view.html", max_points=20_000)
    studio = servemod.Studio(root, max_points=20_000)
    handler = type("H", (servemod._Handler,), {"studio": studio})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", studio
    finally:
        httpd.shutdown()


def _json(url, *, method="GET", body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def _sse(url):
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    out = []
    with urllib.request.urlopen(req, timeout=180) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data:"):
                m = json.loads(line[5:])
                out.append(m)
                if m.get("kind") in ("done", "error"):
                    break
    return out


def test_direct_route_needs_a_key_then_saves_the_operators_move(direct_url, monkeypatch):
    url, studio = direct_url
    twin = studio.twin_of(studio.jobs["cleanroom"])
    status, r = _json(f"{url}/sims/cleanroom", method="POST", body={"timeline": _timeline(twin)})
    assert status == 200, r
    sid = r["timeline"]["id"]
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    status, r = _json(f"{url}/sim/cleanroom/{sid}/direct", method="POST", body={"instruction": "push in"})
    assert status == 500 and "META_API_KEY" in r["error"]
    monkeypatch.setenv("META_API_KEY", "test")
    fake = FakeMetaClient(chat_answers=_script(twin))
    monkeypatch.setattr(studio.stage, "_client", lambda: fake)
    status, r = _json(f"{url}/sim/cleanroom/{sid}/direct", method="POST", body={"instruction": ""})
    assert status == 400
    status, r = _json(f"{url}/sim/cleanroom/{sid}/direct", method="POST", body={"instruction": "walk in on Maya"})
    assert status == 200 and r["run"]
    events = _sse(f"{url}{r['events']}")
    assert events[-1]["kind"] == "done", events[-1]
    done = events[-1]
    assert done["timeline"]["rig"]["kind"] == "handheld" and done["evaluation"]["frames"]
    assert done["notes"].startswith("A slow walk-in") and done["looks"] == 1 and done["looks_files"]
    assert any(e["kind"] == "review" for e in events)
    # Saved: the sim now plays the operator's move, and the look image is served.
    status, again = _json(f"{url}/sim/cleanroom/{sid}")
    assert status == 200 and again["timeline"]["rig"]["kind"] == "handheld"
    look = done["looks_files"][0]
    req = urllib.request.Request(f"{url}/sim-file/cleanroom/{sid}/{look}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        assert resp.status == 200 and resp.read()[:8] == b"\x89PNG\r\n\x1a\n"


def test_watch_tiles_a_filmstrip_and_the_dp_judges_the_move(clean, tmp_path):
    import io

    from PIL import Image

    twin = clean[0].twin
    answers = _script(twin, look=False)[:5]                       # through evaluate
    answers.append([ToolCall("w1", "watch", {})])
    answers.append({"score": 6, "verdict": "adjust", "notes": "Ends too tight; stop on a medium close-up."})
    answers.append([ToolCall("c7", "finish", {"notes": "Shortened the push."})])
    answers = [ChatResult(text="", tool_calls=a, model="fake") if isinstance(a, list) else a for a in answers]
    fake = FakeMetaClient(chat_answers=answers)
    events = []
    res = simagent.direct(twin, _timeline(twin), "push in on Maya", client=fake,
                          actors=[castmod.Actor(name="Maya", height_m=1.68)], workdir=tmp_path / "w", on_event=events.append)
    assert res.finished
    tools = [t["tool"] for t in res.trace if t["kind"] == "call"]
    assert "watch" in tools
    watch = next(e for e in events if e["kind"] == "watch")
    assert len(watch["frames"]) == simagent.FILMSTRIP_FRAMES
    sheet = Image.open(io.BytesIO((tmp_path / "w" / "looks" / watch["file"]).read_bytes()))
    assert sheet.width == 4 * 480 and sheet.height > 2 * 200
    review = next(e for e in events if e["kind"] == "review" and e.get("scope") == "move")
    assert review["score"] == 6 and review["verdict"] == "adjust"
    assert any(e["kind"] == "chat_json" and e["images"] == 1 for e in fake.log)


def test_review_move_video_sends_the_mp4_and_parses_the_verdict():
    fake = FakeMetaClient(chat_answers=[{"description": "A slow push.", "score": 7, "verdict": "keep", "notes": "Fine."}])
    out = simagent.review_move_video(b"\x00\x00\x00\x18ftypisom", client=fake, instruction="push in", timeline={"title": "t", "rig": {"kind": "dolly"}, "duration_s": 4.0})
    assert out["score"] == 7 and out["verdict"] == "keep" and out["description"] == "A slow push."
    msg = fake.log[-1]["messages"][0]["content"]
    assert msg[0]["type"] == "video_url" and msg[0]["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert "DP watching" in msg[1]["text"]
