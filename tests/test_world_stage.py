"""The world over HTTP: the room as it will be on the day, as the studio serves it.

A real server on a free port with the clean room as a finished job. The
builder (product.world) and the finisher (meta.finish) are stood in for by
small doubles that match their signatures and leave the files a real build
leaves, because the real ones cost a Muse Image call per view and a Brush
training run: what is tested here is the stage's own work -- refusing without
a key, the light from the ephemeris, the cast on their marks, the event
stream and its replay, the listing, and the files served back.
"""

from __future__ import annotations

import copy
import io
import json
import sys
import threading
import types
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image

LA = dict(latitude=34.05, longitude=-118.24)


@pytest.fixture(scope="module")
def world_url(clean, tmp_path_factory):
    from locaish import serve as servemod
    from locaish.viewer.build import render_html

    result, fx = clean
    root = tmp_path_factory.mktemp("world")
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
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def _json(url, **kw):
    status, raw, _ = _call(url, **kw)
    return status, json.loads(raw)


def _png(w=960, h=540, colour=(200, 150, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format="PNG")
    return buf.getvalue()


def _sse_until_done(url: str, timeout: float = 120.0) -> list[dict]:
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


# -- the doubles ---------------------------------------------------------------


class FakeMetaClient:
    """Stands where MetaClient would: the finisher double never calls it."""


@dataclass
class FakeFinished:
    name: str
    path: Path
    raw: bytes
    png: bytes


def _install_doubles(monkeypatch, seen: dict):
    """META_API_KEY set, the client a fake, the builder and the finisher
    doubles that leave what a real build leaves and record what they got."""
    from locaish.meta import finish as finmod
    from locaish.product import stage as stagemod

    monkeypatch.setenv("META_API_KEY", "test-key")
    monkeypatch.setattr(stagemod.Stage, "_client", lambda self: FakeMetaClient())

    def fake_finish_world_views(views, *, client, cast, headshots, room_refs, light, dressing, out_dir,
                                workers=4, force=False, progress=None, **_kw):
        assert isinstance(client, FakeMetaClient)
        seen["finish"] = {"n": len(views), "cast": [a.name for a in cast], "light": light, "dressing": dressing,
                          "workers": workers, "room_refs": room_refs is not None,
                          "metas": [m for _, _, m in views]}
        out = []
        for i, (name, png, meta) in enumerate(views):
            dest = Path(out_dir) / f"finish_{name}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = _png(96, 54, (90, 120, 200))
            dest.write_bytes(data)
            if progress:
                progress({"kind": "done", "index": i, "total": len(views), "name": name, "file": dest.name,
                          "text": f"finished {name}"})
            out.append(FakeFinished(name=name, path=dest, raw=data, png=data))
        return out

    monkeypatch.setattr(finmod, "finish_world_views", fake_finish_world_views, raising=False)

    class FakeWorldResult:
        def __init__(self, workdir, n_fin):
            self.views = [{"name": "view_0000"}]
            self.finished = [workdir / "finished" / "finish_view_0000.png"]
            self.ply = workdir / "splat" / "world.ply"
            self.html = workdir / "world.html"
            self.coverage = {"complete": True}
            self.notes = ["a test world"]
            self.twin_path = None
            self.flythrough = None
            self._n_fin = n_fin

        def to_dict(self):
            return {"views": [{"name": "view_0000"}], "finished": [str(p) for p in self.finished],
                    "ply": str(self.ply), "html": str(self.html), "coverage": self.coverage, "notes": self.notes,
                    "seconds": {"select": 0.0}}

    def fake_build_world(twin, workdir, *, finish_fn, figures_for, exclude, n_views, steps, light, dressing,
                         progress, **extra):
        workdir = Path(workdir)
        (workdir / "views").mkdir(parents=True, exist_ok=True)
        figs = figures_for(None)
        seen["build"] = {"n_views": n_views, "steps": steps, "light": light, "dressing": dressing,
                         "figures": len(figs), "exclude": exclude, "extra": extra,
                         "points": int(len(twin.points.xyz))}
        progress("selecting views", 0, n_views)
        png = _png(960, 540)
        (workdir / "views" / "view_0000.png").write_bytes(png)
        progress("rendering view_0000", 1, 1)
        meta = {"name": "view_0000", "camera": {"pos": [0.0, 0.0, 1.5], "aim": [1.0, 1.0, 1.4], "focal_mm": 24.0},
                "light": light, "dressing": dressing}
        finished = finish_fn([("view_0000", png, meta)])
        assert len(finished) == 1 and finished[0].path.exists()
        progress("training the world", 0, 0)
        (workdir / "splat").mkdir(exist_ok=True)
        (workdir / "splat" / "world.ply").write_bytes(b"ply\n")
        (workdir / "world.html").write_text("<!DOCTYPE html><html><body>a render of the room</body></html>")
        (workdir / "coverage.json").write_text(json.dumps({"complete": True}))
        return FakeWorldResult(workdir, len(finished))

    try:
        import locaish.product.world as worldmod
    except ImportError:
        worldmod = types.ModuleType("locaish.product.world")
        monkeypatch.setitem(sys.modules, "locaish.product.world", worldmod)
    monkeypatch.setattr(worldmod, "build_world", fake_build_world, raising=False)


# -- the tests -----------------------------------------------------------------


def test_without_a_key_the_world_refuses_with_the_reason(world_url, monkeypatch):
    url, _, _ = world_url
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    status, r = _json(f"{url}/world/cleanroom", method="POST", body={"date": "2026-11-14", "hour": 15, "dressing": "candles"})
    assert status == 200 and r["run"] and r["events"] == f"/world-events/cleanroom/{r['run']}"
    assert "META_API_KEY" in (r["error"] or "")
    events = _sse_until_done(url + r["events"])
    assert events[-1]["kind"] == "error" and "META_API_KEY" in events[-1]["text"]
    # The refusal is replayed to a browser that comes back.
    again = _sse_until_done(url + r["events"])
    assert again[-1]["kind"] == "error" and "META_API_KEY" in again[-1]["text"]
    # Bad inputs are refused before a run exists.
    assert _json(f"{url}/world/cleanroom", method="POST", body={"date": "yesterday"})[0] == 400
    assert _json(f"{url}/world/cleanroom", method="POST", body={"hour": 31})[0] == 400
    assert _json(f"{url}/world/nosuchjob", method="POST", body={})[0] == 404
    assert _json(f"{url}/world-events/cleanroom/nosuch")[0] == 404
    assert _json(f"{url}/world-file/cleanroom/nosuch/world.html")[0] == 404


def test_world_builds_streams_lists_and_serves_its_files(world_url, monkeypatch):
    url, studio, _ = world_url
    seen: dict = {}
    _install_doubles(monkeypatch, seen)
    status, r = _json(f"{url}/cast/cleanroom", method="POST", body={"name": "Maya", "height_m": 1.68})
    assert status == 200
    status, r = _json(f"{url}/cast/cleanroom", method="POST", body={"name": "Theo", "height_m": 1.84})
    assert status == 200

    status, r = _json(f"{url}/world/cleanroom", method="POST",
                      body={"date": "2026-11-14", "hour": 15, "dressing": "  the table laid   for dinner ", "steps": 500})
    assert status == 200 and r["run"] and not r.get("error"), r
    events = _sse_until_done(url + r["events"])
    done = events[-1]
    assert done["kind"] == "done", done
    wid = done["world_id"]
    assert done["world_html"] == f"/world-file/cleanroom/{wid}/world.html"
    assert done["views"] == 1 and done["finished"] == 1 and done["ply"].endswith("world.ply")
    assert done["seconds"] >= 0 and done["dressing"] == "the table laid for dinner"
    assert done["light"] is None, "the clean room has no georeference yet"
    assert any(e["kind"] == "note" and "georeference" in e["text"] for e in events), "no sun placed is said, not hidden"
    assert any(e["kind"] == "progress" and e["what"] == "finishing" and e["url"].endswith("finish_view_0000.png") for e in events)
    assert any(e["kind"] == "progress" and e["total"] == 1 and e["done"] == 1 for e in events), "the builder's counted steps arrive"
    assert any(e["kind"] == "stage" and "training the world" in e["text"] for e in events)

    # What the stage handed the builder and the finisher.
    assert seen["build"]["steps"] == 500 and seen["build"]["n_views"] == 60 and seen["build"]["light"] is None
    assert seen["build"]["figures"] == 0, "a world is the empty location by default; people are asked for"
    assert seen["build"]["extra"]["method"] == "bake", "the scan's own gaussians are painted, not retrained"
    assert seen["build"]["exclude"] is None, "nothing moved, nothing excluded"
    wm = seen["build"]["extra"]["world_meta"]
    assert wm["date"] == "2026-11-14" and wm["hour"] == 15.0 and wm["dressing"] == "the table laid for dinner" and wm["built_at"]
    assert set(seen["build"]["extra"]["actors"]) == {"Maya", "Theo"}, "the marks are kept for the viewer to draw bodies on"
    fin = seen["finish"]
    assert fin["n"] == 1 and fin["cast"] == [] and fin["workers"] == 4, "nobody is painted into the location"
    # The scan's photographs go along when the twin has posed frames; the
    # synthetic room has none, and the finisher is told so with None.
    assert fin["room_refs"] is (studio.frames_of(studio.jobs["cleanroom"]) is not None)
    assert fin["dressing"] == "the table laid for dinner"
    meta = fin["metas"][0]
    assert set(meta["actors"]) == {"Maya", "Theo"} and all(len(v) == 3 for v in meta["actors"].values())
    # The two stand on different marks, facing the room.
    (xa, ya, _), (xb, yb, _) = meta["actors"]["Maya"], meta["actors"]["Theo"]
    assert abs(xa - xb) + abs(ya - yb) > 0.3

    # A reconnect replays the terminal event.
    again = _sse_until_done(url + r["events"])
    assert again[-1]["kind"] == "done" and again[-1]["world_id"] == wid

    # The listing and the files.
    status, r = _json(f"{url}/worlds/cleanroom")
    assert status == 200
    mine = [w for w in r["worlds"] if w["world_id"] == wid]
    assert mine and mine[0]["state"] == "done" and mine[0]["world_html"] == done["world_html"]
    assert mine[0]["date"] == "2026-11-14" and mine[0]["hour"] == 15.0 and mine[0]["views"] == 1
    status, raw, headers = _call(f"{url}/world-file/cleanroom/{wid}/world.html")
    assert status == 200 and b"a render of the room" in raw and headers["Content-Type"].startswith("text/html")
    status, raw, _ = _call(f"{url}/world-file/cleanroom/{wid}/finish_view_0000.png")
    assert status == 200 and raw[:8] == b"\x89PNG\r\n\x1a\n"
    status, raw, _ = _call(f"{url}/world-file/cleanroom/{wid}/view_0000.png")
    assert status == 200 and raw[:8] == b"\x89PNG\r\n\x1a\n"
    status, cov = _json(f"{url}/world-file/cleanroom/{wid}/coverage.json")
    assert status == 200 and cov["complete"] is True
    assert _json(f"{url}/world-file/cleanroom/{wid}/finish_view_0007.png")[0] == 404
    assert _json(f"{url}/world-file/cleanroom/{wid}/flythrough.mp4")[0] == 404
    assert _json(f"{url}/world-file/cleanroom/{wid}/world.twin")[0] == 404
    assert _json(f"{url}/world-file/cleanroom/{wid}/..%2Fworld.json")[0] == 404

    # A second studio process over the same root restores the world from its world.json.
    job = studio.jobs["cleanroom"]
    assert (job.workdir / "worlds" / wid / "world.json").exists()
    studio.stage.world_runs.clear()
    status, r = _json(f"{url}/worlds/cleanroom")
    assert status == 200 and [w["world_id"] for w in r["worlds"]] == [wid]
    assert r["worlds"][0]["dressing"] == "the table laid for dinner" and r["worlds"][0]["finished"] == 1
    again = _sse_until_done(url + f"/world-events/cleanroom/{wid}")
    assert again[-1]["kind"] == "done" and again[-1]["world_html"] == done["world_html"]


def test_light_for_a_georeferenced_room(world_url, monkeypatch):
    from locaish.film import daylight

    url, studio, _ = world_url
    status, _ = _json(f"{url}/details/cleanroom", method="POST", body={"heading_deg": 270, **LA})
    assert status == 200
    job = studio.jobs["cleanroom"]
    twin = studio.twin_of(job)
    assert twin.georeference is not None

    light, note = studio.stage._world_light(twin, "2026-11-14", 15.0)
    assert light["clock"] == "15:00" and light["date"] == "2026-11-14"
    # Independent of the schedule: mid-afternoon in November in Los Angeles
    # the sun is up and in the south-west.
    assert 0 < light["elevation_deg"] < 45
    assert 180 < light["azimuth_deg"] < 270
    assert isinstance(light["windows_lit"], list) and all(isinstance(i, int) for i in light["windows_lit"])
    assert light["heading_source"] == "user" and note == "", "a measured heading needs no caveat"
    # 12:00 in the module's time (mean solar time, the longitude offset) puts
    # the sun within the equation of time of due south: at most ~16.5 minutes,
    # about 4 degrees of azimuth, so a 6 degree bound is astronomy, not code.
    noon, _ = studio.stage._world_light(twin, "2026-11-14", 12.0)
    assert abs(noon["azimuth_deg"] - 180.0) < 6.0
    # Midnight is below the horizon, and lights no window.
    night, _ = studio.stage._world_light(twin, "2026-11-14", 0.0)
    assert night["elevation_deg"] < 0 and night["windows_lit"] == []
    # Consistency with the schedule's own intervals, and with the geometry:
    # a window the schedule says passes direct sun at 15:00 is lit, and the
    # sun in the south-west cannot light a window facing north or east.
    sched = daylight.sun_schedule(twin, on="2026-11-14")
    expect = [i for i, w in enumerate(sched["windows"]) if any(iv["from"] <= "15:00" <= iv["to"] for iv in w["direct_sun"])]
    assert light["windows_lit"] == expect
    for i in light["windows_lit"]:
        assert sched["windows"][i]["faces"] not in ("N", "NNE", "NE", "ENE", "E", "NNW", "NW")
    assert [w["faces"] for w in light["windows"]] == [w["faces"] for w in sched["windows"]]
    # Half hours read as a clock.
    half, _ = studio.stage._world_light(twin, "2026-11-14", 15.5)
    assert half["clock"] == "15:30"

    # Without a georeference the sun is not placed, and the reason is given.
    bare = copy.copy(twin)
    bare.georeference = None
    none, why = studio.stage._world_light(bare, "2026-11-14", 15.0)
    assert none is None and "georeference" in why

    # Over HTTP the light reaches the finisher and the done event.
    seen: dict = {}
    _install_doubles(monkeypatch, seen)
    status, r = _json(f"{url}/world/cleanroom", method="POST", body={"date": "2026-11-14", "hour": 15.5, "dressing": ""})
    assert status == 200 and not r.get("error")
    events = _sse_until_done(url + r["events"])
    done = events[-1]
    assert done["kind"] == "done", done
    assert done["light"]["clock"] == "15:30" and done["light"]["azimuth_deg"] == seen["finish"]["light"]["azimuth_deg"]
    assert seen["finish"]["metas"][0]["light"]["elevation_deg"] == done["light"]["elevation_deg"]
    assert any(e["kind"] == "stage" and "azimuth" in e["text"] for e in events)
    assert seen["build"]["extra"]["world_meta"]["hour"] == 15.5
