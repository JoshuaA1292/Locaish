"""The camera simulation over HTTP: timelines kept under a finished job,
solved and evaluated on every save, the cast as point sets at a moment, the
frame at the playhead, the rendered move, and the Blender hand-off.

A real server on a free port with the clean room as a finished job, as in
`test_stage.py`. The routes import without the engine; when
`locaish/film/sim.py` is not there yet, a thin double that follows the shared
contract stands in so the stage's own work (storage, figures, frames, the
render run, the files) is exercised end to end.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
import sys
import threading
import types
import urllib.error
import urllib.request
import uuid
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from locaish.film import coverage as cov


# ---------------------------------------------------------------------------
# the engine double (used only while locaish/film/sim.py is absent)
# ---------------------------------------------------------------------------


def _install_engine_double() -> bool:
    try:
        import locaish.film.sim  # noqa: F401
        return False
    except ImportError:
        pass

    mod = types.ModuleType("locaish.film.sim")
    mod.__doc__ = "test double for the camera simulation engine (contract-shaped)"

    class _Rig:
        def __init__(self, kind):
            self.kind = kind

        def describe(self):
            return f"{self.kind}: a test double rig"

    class Timeline:
        def __init__(self, d):
            self.d = d

        @classmethod
        def from_dict(cls, d):
            d = copy.deepcopy(d)
            d.setdefault("id", uuid.uuid4().hex[:8])
            d.setdefault("title", "Camera move")
            d.setdefault("fps", 24)
            d.setdefault("duration_s", 6.0)
            d.setdefault("sensor", "ff")
            d.setdefault("aperture_f", 2.8)
            d.setdefault("rig", {"kind": "tripod", "params": {}})
            d.setdefault("camera", {"interp": "catmull", "keys": []})
            d.setdefault("actors", {})
            d.setdefault("pieces", [])
            d.setdefault("notes", [])
            return cls(d)

        def to_dict(self):
            return copy.deepcopy(self.d)

    class Evaluation:
        def __init__(self, d):
            self.d = d

        def to_dict(self):
            return copy.deepcopy(self.d)

    def _lerp_keys(keys, t, fields):
        ks = sorted(keys, key=lambda k: float(k["t"]))
        if not ks:
            return None
        if t <= float(ks[0]["t"]):
            return {f: ks[0].get(f) for f in fields}
        for a, b in zip(ks, ks[1:]):
            ta, tb = float(a["t"]), float(b["t"])
            if ta <= t <= tb:
                u = 0.0 if tb <= ta else (t - ta) / (tb - ta)
                out = {}
                for f in fields:
                    va, vb = a.get(f), b.get(f)
                    if isinstance(va, (list, tuple)) and isinstance(vb, (list, tuple)):
                        out[f] = [float(x) + u * (float(y) - float(x)) for x, y in zip(va, vb)]
                    elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                        out[f] = float(va) + u * (float(vb) - float(va))
                    else:
                        out[f] = va if va is not None else vb
                return out
        return {f: ks[-1].get(f) for f in fields}

    def _actor_at(d, name, t):
        a = d["actors"].get(name) or {}
        got = _lerp_keys(a.get("keys") or [], t, ("pos", "facing_deg"))
        if not got or got.get("pos") is None:
            return None
        return [float(got["pos"][0]), float(got["pos"][1]), float(got.get("facing_deg") or 90.0)]

    def solve(tl, twin):
        d = tl.to_dict()
        floor_z = float(twin.structure.floor_z)
        for k in d["camera"]["keys"]:
            if k.get("aim") is None and k.get("look_at"):
                who = _actor_at(d, k["look_at"], float(k["t"]))
                h = float((d["actors"].get(k["look_at"]) or {}).get("height_m") or 1.75)
                if who is not None:
                    k["aim"] = [who[0], who[1], floor_z + 0.94 * h]
            k.setdefault("focal_mm", 35)
            k.setdefault("roll_deg", 0)
            k.setdefault("ease", "inout")
        d["notes"] = list(d.get("notes") or []) + ["solved by the test double"]
        return Timeline(d)

    def evaluate(tl, twin, *, maps=None, occupancy=None, pieces=None):
        d = tl.to_dict()
        fps = float(d["fps"])
        n = int(round(float(d["duration_s"]) * fps)) + 1
        xyz = np.asarray(twin.points.xyz)
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        frames, prev = [], None
        for i in range(n):
            t = i / fps
            cam = _lerp_keys(d["camera"]["keys"], t, ("pos", "aim", "focal_mm", "roll_deg"))
            pos = [float(v) for v in cam["pos"]]
            aim = [float(v) for v in (cam["aim"] or [pos[0] + 1.0, pos[1], pos[2]])]
            focal = float(cam.get("focal_mm") or 35.0)
            dist = math.dist(pos, aim)
            fwd = np.array(aim) - np.array(pos)
            pan = math.degrees(math.atan2(fwd[1], fwd[0]))
            tilt = math.degrees(math.atan2(fwd[2], math.hypot(fwd[0], fwd[1])))
            speed = 0.0 if prev is None else math.dist(pos, prev) * fps
            prev = pos
            framed = 24.0 / focal * dist
            size = ("CU" if framed < 0.6 else "MCU" if framed < 1.0 else "MS" if framed < 1.6
                    else "MLS" if framed < 2.2 else "LS" if framed < 3.5 else "WS")
            problems = []
            if not (lo[0] <= pos[0] <= hi[0] and lo[1] <= pos[1] <= hi[1]):
                problems.append("camera outside the room")
            if speed > 2.0:
                problems.append(f"camera moving at {speed:.1f} m/s")
            actors = {name: a for name in d["actors"] if (a := _actor_at(d, name, t)) is not None}
            frames.append({
                "t": round(t, 4), "cam": [round(v, 4) for v in pos], "aim": [round(v, 4) for v in aim],
                "focal_mm": focal, "pan_deg": round(pan, 2), "tilt_deg": round(tilt, 2),
                "roll_deg": float(cam.get("roll_deg") or 0.0), "speed_mps": round(speed, 3),
                "actors": actors, "shot_size": size, "distance_m": round(dist, 3),
                "dof": {"near_m": round(dist * 0.85, 2), "far_m": round(dist * 1.2, 2)}, "problems": problems,
            })
        segments, open_seg = [], None
        for f in frames:
            text = "; ".join(f["problems"])
            if text and open_seg and open_seg["text"] == text:
                open_seg["t_end"] = f["t"]
            elif text:
                open_seg = {"t_start": f["t"], "t_end": f["t"], "text": text, "severity": "red"}
                segments.append(open_seg)
            else:
                open_seg = None
        travel = sum(math.dist(a["cam"], b["cam"]) for a, b in zip(frames, frames[1:]))
        sizes = [f["shot_size"] for f in frames]
        return Evaluation({
            "fps": fps, "frames": frames, "segments": segments,
            "summary": {"distance_m": round(travel, 2), "size_range": f"{sizes[0]} to {sizes[-1]}",
                        "worst_problem": segments[0]["text"] if segments else None},
            "timeline": d,
        })

    def from_rehearsal(rehearsal, title=None):
        reh = rehearsal.get("rehearsal") or rehearsal
        scene = reh.get("scene") or {}
        keys, actors = [], {}
        for name, h in (scene.get("cast") or {}).items():
            actors[name] = {"height_m": float(h), "keys": []}
        for fr in reh.get("frames") or []:
            cam = fr.get("camera")
            if cam:
                keys.append({"t": float(fr["t"]), "pos": list(cam["pos"]), "aim": list(cam["aim"]), "look_at": None,
                             "focal_mm": float(cam["focal_mm"]), "roll_deg": 0, "ease": "inout"})
            for name, a in (fr.get("actors") or {}).items():
                actors.setdefault(name, {"height_m": 1.75, "keys": []})
                actors[name]["keys"].append({"t": float(fr["t"]), "pos": [a[0], a[1]], "facing_deg": a[2]})
        return Timeline.from_dict({
            "title": title or scene.get("title") or "From the rehearsal", "fps": 24,
            "duration_s": float(reh.get("duration_s") or 1.0), "rig": {"kind": "dolly", "params": {"track_m": 3.0}},
            "camera": {"interp": "catmull", "keys": keys}, "actors": actors, "pieces": [], "notes": [],
        })

    def blender_script(tl, ev, glb_path="room.glb", twin=None):
        d = tl.to_dict()
        return ("import bpy\n" f"# {d['title']} -- test double\n" f"bpy.ops.import_scene.gltf(filepath={glb_path!r})\n"
                f"FRAMES = {json.dumps([f['cam'] for f in ev.to_dict()['frames']])}\n")

    mod.Timeline, mod.Evaluation = Timeline, Evaluation
    mod.RIGS = {k: _Rig(k) for k in ("tripod", "slider", "dolly", "jib", "handheld", "steadicam", "crane", "drone")}
    mod.new_id = lambda: uuid.uuid4().hex[:8]
    mod.solve, mod.evaluate, mod.from_rehearsal, mod.blender_script = solve, evaluate, from_rehearsal, blender_script
    sys.modules["locaish.film.sim"] = mod
    import locaish.film

    locaish.film.sim = mod
    return True


ENGINE_IS_DOUBLE = _install_engine_double()


# ---------------------------------------------------------------------------
# the server
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim_url(clean, tmp_path_factory):
    from locaish import serve as servemod
    from locaish.viewer.build import render_html

    result, fx = clean
    root = tmp_path_factory.mktemp("simstage")
    job = root / "cleanroom"
    job.mkdir()
    result.twin.save(job / "test-clean.twin")
    render_html(result.twin, job / "view.html", max_points=20_000)
    studio = servemod.Studio(root, max_points=20_000)
    handler = type("H", (servemod._Handler,), {"studio": studio})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", studio, fx
    finally:
        httpd.shutdown()


def _call(url, *, method="GET", body=None, ctype=None):
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode() if isinstance(body, dict) else body
        headers["Content-Type"] = ctype or ("application/json" if isinstance(body, dict) else "application/octet-stream")
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _json(url, **kw):
    status, raw = _call(url, **kw)
    return status, json.loads(raw)


def _sse_until_done(url: str, timeout: float = 300.0) -> list[dict]:
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    out = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data:"):
                m = json.loads(line[5:])
                out.append(m)
                if m.get("kind") in ("done", "error"):
                    break
    return out


def _standable(studio):
    """Two camera spots the sweep found standable, two marks, and the floor."""
    job = studio.jobs["cleanroom"]
    sw = studio.sweep_of(job)
    c = sw.columns
    vis = np.flatnonzero(np.asarray(c["visible"], dtype=bool))
    assert len(vis) >= 2
    order = vis[np.argsort(-np.asarray(c["score"], dtype=float)[vis])]
    i, j = int(order[0]), int(order[min(len(order) - 1, len(order) // 2)])
    floor_z = studio.floor_z_of(job)
    cams = [[float(c["cam_x"][k]), float(c["cam_y"][k]), floor_z + float(c["cam_z"][k])] for k in (i, j)]
    marks = cov.LocalSetups(sw).marks("test-clean")
    return cams, [list(marks[0]), list(marks[min(3, len(marks) - 1)])], floor_z


STATE: dict = {}


# ---------------------------------------------------------------------------
# the tests
# ---------------------------------------------------------------------------


def test_routes_answer_without_a_job_or_a_sim(sim_url):
    url, _, _ = sim_url
    assert _json(f"{url}/sims/nosuchjob")[0] == 404
    assert _json(f"{url}/sim/cleanroom/nope")[0] == 404
    assert _json(f"{url}/sim/cleanroom/../../etc")[0] in (400, 404)
    status, r = _json(f"{url}/sims/cleanroom")
    assert status == 200 and r["sims"] == []


def test_create_list_get(sim_url):
    url, studio, _ = sim_url
    status, r = _json(f"{url}/sims/cleanroom", method="POST", body={})
    assert status == 200, r
    tl, ev = r["timeline"], r["evaluation"]
    assert tl["id"] and tl["rig"]["kind"] == "tripod" and tl["duration_s"] == 6.0 and len(tl["camera"]["keys"]) == 1
    key = tl["camera"]["keys"][0]
    assert len(key["pos"]) == 3 and len(key["aim"]) == 3
    # The default spot is standable: inside the room, at a camera height.
    twin = studio.twin_of(studio.jobs["cleanroom"])
    lo, hi = twin.points.xyz.min(axis=0), twin.points.xyz.max(axis=0)
    assert lo[0] <= key["pos"][0] <= hi[0] and lo[1] <= key["pos"][1] <= hi[1]
    assert 0.3 < key["pos"][2] - studio.floor_z_of(studio.jobs["cleanroom"]) < 2.5
    assert ev["frames"] and "summary" in ev and "segments" in ev
    assert (ev.get("fps") or ev["summary"].get("fps") or ev["timeline"]["fps"]) == 24
    assert "rig_text" in r
    STATE["sim_id"] = tl["id"]
    status, r = _json(f"{url}/sims/cleanroom")
    assert status == 200 and [s["sim_id"] for s in r["sims"]] == [tl["id"]]
    item = r["sims"][0]
    assert item["rig"] == "tripod" and item["fps"] == 24 and item["updated_at"] and "problems" in item
    status, r = _json(f"{url}/sim/cleanroom/{tl['id']}")
    assert status == 200 and r["timeline"]["id"] == tl["id"] and len(r["evaluation"]["frames"]) == len(ev["frames"])
    d = studio.jobs["cleanroom"].workdir / "sims" / tl["id"]
    assert (d / "timeline.json").exists() and (d / "evaluation.json").exists()


def test_put_two_key_handheld_is_solved_and_evaluated(sim_url):
    url, studio, _ = sim_url
    cams, marks, floor_z = _standable(studio)
    sid = STATE["sim_id"]
    timeline = {
        "id": sid, "title": "Handheld to Maya", "fps": 24, "duration_s": 2.0, "sensor": "ff", "aperture_f": 2.8,
        "rig": {"kind": "handheld", "params": {"operator_height_m": 1.75}},
        "camera": {"interp": "catmull", "keys": [
            {"t": 0.0, "pos": cams[0], "aim": None, "look_at": "Maya", "focal_mm": 35, "roll_deg": 0, "ease": "inout"},
            {"t": 2.0, "pos": cams[1], "aim": None, "look_at": "Maya", "focal_mm": 50, "roll_deg": 0, "ease": "inout"},
        ]},
        "actors": {"Maya": {"height_m": 1.68, "keys": [
            {"t": 0.0, "pos": marks[0], "facing_deg": 90},
            {"t": 2.0, "pos": marks[1], "facing_deg": 180},
        ]}},
        "pieces": [], "notes": [],
    }
    status, r = _json(f"{url}/sim/cleanroom/{sid}", method="PUT", body={"timeline": timeline})
    assert status == 200, r
    tl, ev = r["timeline"], r["evaluation"]
    assert tl["id"] == sid and tl["rig"]["kind"] == "handheld" and len(tl["camera"]["keys"]) == 2
    assert all(k.get("aim") is not None or k.get("look_at") for k in tl["camera"]["keys"]), "a solved key knows where it looks"
    frames = ev["frames"]
    assert len(frames) >= 2.0 * 24 and len(frames[0]["cam"]) == 3 and len(frames[0]["aim"]) == 3
    assert "Maya" in frames[0]["actors"] and len(frames[0]["actors"]["Maya"]) == 3
    assert frames[-1]["t"] >= 1.9 and frames[0]["focal_mm"] == 35
    for f in (frames[0], frames[-1]):
        assert isinstance(f["problems"], list) and "shot_size" in f and "dof" in f
    # POST saves the same way PUT does, and the listing reflects the edit.
    status, r2 = _json(f"{url}/sim/cleanroom/{sid}", method="POST", body={"timeline": timeline})
    assert status == 200 and r2["timeline"]["title"] == "Handheld to Maya"
    _, lst = _json(f"{url}/sims/cleanroom")
    assert lst["sims"][0]["title"] == "Handheld to Maya" and lst["sims"][0]["rig"] == "handheld"
    assert _json(f"{url}/sim/cleanroom/{sid}", method="PUT", body={})[0] == 400
    STATE["marks"] = marks


def test_figures_at_t_returns_the_cast_as_bodies(sim_url):
    url, studio, _ = sim_url
    sid = STATE["sim_id"]
    # Maya is cast at her real height; the figure uses it.
    status, r = _json(f"{url}/cast/cleanroom", method="POST", body={"name": "Maya", "height_m": 1.68})
    assert status == 200
    status, r = _json(f"{url}/sim/cleanroom/{sid}/figures?t=0.0")
    assert status == 200, r
    figs = r["figures"]
    assert [f["name"] for f in figs] == ["Maya"]
    f = figs[0]
    assert len(f["colour"]) == 3 and all(0 <= c <= 255 for c in f["colour"])
    # A body is a mesh: flat vertices and triangle indices in twin metres.
    assert "mesh" in f and len(f["mesh"]["vertices"]) % 3 == 0 and len(f["mesh"]["faces"]) % 3 == 0
    xyz = np.asarray(f["mesh"]["vertices"]).reshape(-1, 3)
    faces = np.asarray(f["mesh"]["faces"]).reshape(-1, 3)
    assert 200 <= len(xyz) <= 5000 and len(faces) >= 400 and faces.max() < len(xyz)
    floor_z = studio.floor_z_of(studio.jobs["cleanroom"])
    mark = STATE["marks"][0]
    assert abs(float(xyz[:, 0].mean()) - mark[0]) < 0.3 and abs(float(xyz[:, 1].mean()) - mark[1]) < 0.3
    assert floor_z - 0.05 <= xyz[:, 2].min() and xyz[:, 2].max() <= floor_z + 1.68 + 0.05, "feet on the floor, crown at her height"
    assert all(round(v, 3) == v for v in f["mesh"]["vertices"][:30])
    # At the end of the move she stands on the second mark.
    status, r = _json(f"{url}/sim/cleanroom/{sid}/figures?t=2.0")
    xyz2 = np.asarray(r["figures"][0]["mesh"]["vertices"]).reshape(-1, 3)
    mark2 = STATE["marks"][1]
    assert abs(float(xyz2[:, 0].mean()) - mark2[0]) < 0.3 and abs(float(xyz2[:, 1].mean()) - mark2[1]) < 0.3


def test_sim_frame_is_a_png(sim_url):
    url, studio, _ = sim_url
    sid = STATE["sim_id"]
    status, raw = _call(f"{url}/sim-frame/cleanroom/{sid}?t=0&w=320")
    assert status == 200 and raw[:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image
    import io

    im = Image.open(io.BytesIO(raw))
    assert im.width == 320 and im.height > 200 and im.height % 2 == 0
    # The cache serves the same frame the second time; a save empties it.
    d = studio.jobs["cleanroom"].workdir / "sims" / sid
    assert list((d / "preview").glob("f_00000_320.png"))
    status, raw2 = _call(f"{url}/sim-frame/cleanroom/{sid}?t=0.01&w=320")
    assert status == 200 and raw2 == raw


def test_render_run_streams_to_done_and_writes_frames_and_the_mp4(sim_url):
    url, studio, _ = sim_url
    sid = STATE["sim_id"]
    status, r = _json(f"{url}/sim/cleanroom/{sid}/render", method="POST", body={"width_px": 320, "every": 6})
    assert status == 200 and r["run"] and r["events"].startswith(f"/sim-events/cleanroom/{sid}/")
    events = _sse_until_done(f"{url}{r['events']}")
    done = events[-1]
    assert done["kind"] == "done", done
    assert any(e["kind"] == "progress" for e in events)
    d = studio.jobs["cleanroom"].workdir / "sims" / sid
    pngs = sorted((d / "render").glob("frame_*.png"))
    assert len(pngs) == done["frames"] >= 8 and pngs[0].name == "frame_00000.png"
    assert isinstance(done["seconds"], (int, float))
    if shutil.which("ffmpeg"):
        assert done["mp4"] == f"/sim-file/cleanroom/{sid}/move.mp4", done
        assert (d / "move.mp4").exists()
        status, raw = _call(f"{url}{done['mp4']}")
        assert status == 200 and raw[4:8] == b"ftyp"
    else:
        assert done["mp4"] is None and "ffmpeg" in done["note"]
    status, raw = _call(f"{url}/sim-file/cleanroom/{sid}/frame_00000.png")
    assert status == 200 and raw[:8] == b"\x89PNG\r\n\x1a\n"
    # A reconnect after the run finished gets the terminal event again.
    again = _sse_until_done(f"{url}{r['events']}")
    assert again[-1]["kind"] == "done" and again[-1]["frames"] == done["frames"]


def test_sim_files_blender_glb_timeline(sim_url):
    url, _, _ = sim_url
    sid = STATE["sim_id"]
    status, raw = _call(f"{url}/sim-file/cleanroom/{sid}/blender.py")
    assert status == 200 and b"bpy" in raw and b"room.glb" in raw
    status, raw = _call(f"{url}/sim-file/cleanroom/{sid}/room.glb")
    assert status == 200 and raw[:4] == b"glTF" and len(raw) > 1000
    status, raw = _call(f"{url}/sim-file/cleanroom/{sid}/timeline.json")
    assert status == 200 and json.loads(raw)["id"] == sid
    status, raw = _call(f"{url}/sim-file/cleanroom/{sid}/evaluation.json")
    assert status == 200 and json.loads(raw)["frames"]
    assert _call(f"{url}/sim-file/cleanroom/{sid}/secrets.txt")[0] == 404
    assert _call(f"{url}/sim-file/cleanroom/{sid}/frame_99999.png")[0] == 404


def test_from_rehearsal_builds_a_timeline(sim_url, monkeypatch):
    url, studio, _ = sim_url
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    job = studio.jobs["cleanroom"]
    marks = cov.LocalSetups(studio.sweep_of(job)).marks("test-clean")
    a, b = marks[0], marks[min(3, len(marks) - 1)]
    scene = (f"TITLE: The clean room\nMARKS: A=({a[0]:.2f},{a[1]:.2f}) B=({b[0]:.2f},{b[1]:.2f})\nCAST: Theo=1.84 Maya=1.68\n"
             f"1. Theo enters at B\n2. Maya enters at A   [cam ({a[0]+0.1:.2f},{a[1]-1.6:.2f}) h1.5 35mm on Maya]\n"
             f"3. Theo: \"You came back.\"\n4. Maya: \"I never left.\"\n5. hold 1s\n")
    status, r = _json(f"{url}/rehearse/cleanroom", method="POST", body={"scene": scene, "every_s": 2.0})
    assert status == 200 and r["rehearsal_id"]
    rid = r["rehearsal_id"]
    events = _sse_until_done(f"{url}/rehearsal-events/cleanroom/{rid}")
    assert events[-1]["kind"] == "done", events[-1]
    # A fresh sim, then the rehearsal poured into it.
    status, r = _json(f"{url}/sims/cleanroom", method="POST", body={})
    assert status == 200
    sid2 = r["timeline"]["id"]
    status, r = _json(f"{url}/sim/cleanroom/{sid2}/from-rehearsal", method="POST", body={"rehearsal_id": rid})
    assert status == 200, r
    tl, ev = r["timeline"], r["evaluation"]
    assert tl["id"] == sid2 and tl["camera"]["keys"] and set(tl["actors"]) >= {"Theo", "Maya"}
    assert abs(tl["actors"]["Maya"]["height_m"] - 1.68) < 1e-6 and tl["duration_s"] > 1.0
    assert ev["frames"] and any(f["actors"] for f in ev["frames"])
    assert _json(f"{url}/sim/cleanroom/{sid2}/from-rehearsal", method="POST", body={"rehearsal_id": "nope"})[0] == 404
    _, lst = _json(f"{url}/sims/cleanroom")
    assert {s["sim_id"] for s in lst["sims"]} == {STATE["sim_id"], sid2}
    STATE["sim_id2"] = sid2


def test_delete(sim_url):
    url, studio, _ = sim_url
    for sid in (STATE["sim_id"], STATE["sim_id2"]):
        status, r = _json(f"{url}/sim/cleanroom/{sid}", method="DELETE")
        assert status == 200 and r["removed"] is True
        assert not (studio.jobs["cleanroom"].workdir / "sims" / sid).exists()
        assert _json(f"{url}/sim/cleanroom/{sid}")[0] == 404
    status, r = _json(f"{url}/sim/cleanroom/{STATE['sim_id']}", method="DELETE")
    assert status == 200 and r["removed"] is False
    _, lst = _json(f"{url}/sims/cleanroom")
    assert lst["sims"] == []
    assert _json(f"{url}/sim/nosuchjob/x", method="DELETE")[0] == 404
