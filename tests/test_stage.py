"""The stage over HTTP: cast, set pieces and a rehearsal on a restored job.

A real server on a free port with the clean room as a finished job, no model,
no SAM 3D endpoint: the typed scene rehearses, the capsule stands in for the
body, and the routes say plainly what is not configured.
"""

from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from locaish.film import coverage as cov


@pytest.fixture(scope="module")
def stage_url(clean, tmp_path_factory):
    from locaish import serve as servemod
    from locaish.viewer.build import render_html

    result, fx = clean
    root = tmp_path_factory.mktemp("stage")
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
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _json(url, **kw):
    status, raw = _call(url, **kw)
    return status, json.loads(raw)


def _png(w=64, h=80, colour=(200, 150, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format="PNG")
    return buf.getvalue()


def test_status_says_what_is_configured(stage_url, monkeypatch):
    url, _, _ = stage_url
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("LOCAISH_SAM3D_URL", raising=False)
    status, s = _json(f"{url}/stage-status/cleanroom")
    assert status == 200 and s["meta"] is False and s["sam3d"] is False and s["breakdown"] is False
    status, _ = _call(f"{url}/stage.js")
    assert status == 200
    assert _json(f"{url}/cast/nosuchjob")[0] == 404


def test_cast_round_trip_and_headshot_without_sam3d(stage_url, monkeypatch):
    url, studio, _ = stage_url
    monkeypatch.delenv("LOCAISH_SAM3D_URL", raising=False)
    status, r = _json(f"{url}/cast/cleanroom", method="POST", body={"name": "Maya", "height_m": 1.68})
    assert status == 200 and r["actor"]["name"] == "Maya" and r["actor"]["eye_height_m"] < 1.68
    aid = r["actor"]["id"]
    status, r = _json(f"{url}/cast/cleanroom", method="POST", body={"name": "Theo", "height_m": 184})
    assert status == 200 and r["actor"]["height_m"] == 1.84, "centimetres are forgiven"
    status, r = _json(f"{url}/cast/cleanroom", method="POST", body={"name": "", "height_m": 1.7})
    assert status == 400
    status, r = _json(f"{url}/cast/cleanroom/{aid}/headshot", method="POST", body=_png(), ctype="image/png")
    assert status == 200 and r["actor"]["headshot"] and r["body"] is None and "capsule" in r["note"]
    status, raw = _call(f"{url}/headshot/cleanroom/{aid}")
    assert status == 200 and raw[:8] == b"\x89PNG\r\n\x1a\n"
    status, r = _json(f"{url}/cast/cleanroom")
    assert status == 200 and [a["name"] for a in r["cast"]] == ["Maya", "Theo"]
    assert r["cast"][0]["headshot"]
    status, r = _json(f"{url}/cast/cleanroom/{aid}", method="DELETE")
    assert status == 200 and r["removed"] is True
    _, r = _json(f"{url}/cast/cleanroom")
    assert [a["name"] for a in r["cast"]] == ["Theo"]


def test_set_piece_lift_move_diff_and_plan(stage_url):
    url, studio, fx = stage_url
    lo, hi = fx.truth.boxes[0]
    anchor = [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), float(hi[2])]
    status, r = _json(f"{url}/labels/cleanroom", method="POST", body={"label": "sofa", "kind": "furniture", "xyz": anchor})
    assert status == 200
    label_id = r["label"]["id"]
    status, r = _json(f"{url}/setpieces/cleanroom/lift", method="POST", body={"label_id": label_id})
    assert status == 200, r
    piece = r["piece"]
    assert piece["measured"] is True and piece["label"] == "sofa"
    truth = sorted([float(hi[0] - lo[0]), float(hi[1] - lo[1])])
    got = sorted([piece["width_m"], piece["depth_m"]])
    assert abs(got[0] - truth[0]) < 0.08 and abs(got[1] - truth[1]) < 0.08
    # Lifting the same label again returns the same piece.
    _, again = _json(f"{url}/setpieces/cleanroom/lift", method="POST", body={"label_id": label_id})
    assert again["piece"]["id"] == piece["id"]
    status, r = _json(f"{url}/setpieces/cleanroom/{piece['id']}/move", method="POST",
                      body={"dx": 0.6, "dy": 0.0, "dyaw_deg": 0.0, "relative": True})
    assert status == 200, r
    assert r["piece"]["moved"] is True and abs(r["piece"]["moved_by"]["dx_m"] - 0.6) < 1e-6
    assert r["diff"] is not None and "visible_before" in r["diff"] and isinstance(r["diff"]["sentences"], list)
    status, raw = _call(f"{url}/stage-plan/cleanroom")
    assert status == 200 and b"<svg" in raw and b"sofa" in raw
    status, r = _json(f"{url}/setpieces/cleanroom")
    assert status == 200 and r["pieces"][0]["moved"] is True
    # The dressed room differs from the scan where the sofa was and now is.
    room, sw, pieces = studio.stage.dressed(studio.jobs["cleanroom"])
    assert len(sw) > 0 and not np.allclose(room.points.xyz, studio.twin_of(studio.jobs["cleanroom"]).points.xyz)
    status, r = _json(f"{url}/setpieces/cleanroom/{piece['id']}/move", method="POST", body={"reset": True})
    assert status == 200 and r["piece"]["moved"] is False
    status, r = _json(f"{url}/setpieces/cleanroom/{piece['id']}/lift3d", method="POST", body={})
    assert status == 500 and ("META_API_KEY" in r["error"] or "LOCAISH_SAM3D_URL" in r["error"])
    status, r = _json(f"{url}/setpieces/cleanroom/{piece['id']}", method="DELETE")
    assert status == 200 and r["removed"] is True


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


def test_typed_rehearsal_runs_streams_and_serves_its_files(stage_url, monkeypatch):
    url, studio, _ = stage_url
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
    assert any(e["kind"] == "stage" for e in events)
    status, res = _json(f"{url}/rehearsal/cleanroom/{rid}")
    assert status == 200 and res["provider"] == "typed" and res["scene"]["title"] == "The clean room"
    assert res["rehearsal"]["duration_s"] > 1.0 and len(res["rehearsal"]["shots"]) == 1
    assert "<svg" in res["blocking_svg"]
    status, anim = _json(f"{url}/rehearsal/cleanroom/{rid}/animatic.json")
    assert status == 200
    frames = [f for f in anim["frames"] if f.get("file")]
    assert frames, "the camera beat should yield frames"
    status, raw = _call(f"{url}/rehearsal-image/cleanroom/{rid}/{frames[0]['file']}")
    assert status == 200 and raw[:8] == b"\x89PNG\r\n\x1a\n"
    status, raw = _call(f"{url}/rehearsal/cleanroom/{rid}/callsheet.txt")
    assert status == 200 and b"THE CLEAN ROOM" in raw.upper()
    status, r = _json(f"{url}/rehearsals/cleanroom")
    assert status == 200 and r["rehearsals"][0]["rehearsal_id"] == rid and r["rehearsals"][0]["state"] == "done"
    # A second studio over the same root restores the finished rehearsal from disk.
    studio.stage.runs.pop(job.id, None)
    status, r = _json(f"{url}/rehearsals/cleanroom")
    assert status == 200 and r["rehearsals"][0]["rehearsal_id"] == rid
    # Finishing a frame needs Muse Image.
    status, r = _json(f"{url}/rehearsal/cleanroom/{rid}/finish/{frames[0]['file']}", method="POST", body={})
    assert status == 500 and "META_API_KEY" in r["error"]


def test_prose_without_a_model_fails_with_a_reason(stage_url, monkeypatch):
    url, _, _ = stage_url
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    status, r = _json(f"{url}/rehearse/cleanroom", method="POST", body={"scene": "INT. ROOM. Someone speaks."})
    assert status == 200
    events = _sse_until_done(f"{url}/rehearsal-events/cleanroom/{r['rehearsal_id']}")
    assert events[-1]["kind"] == "error" and "META_API_KEY" in events[-1]["text"]
    assert _json(f"{url}/rehearse/cleanroom", method="POST", body={"scene": ""})[0] == 400
