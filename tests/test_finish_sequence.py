"""Finishing a whole sequence with Muse Image, on the fake client.

The prompt names each stand-in by its colour as the actor it stands for,
with the actor's own words for looks and wardrobe; the first frame goes
alone and every later frame carries the last finished frame as the
continuity reference; a frame the model fails on leaves no file and is
reported; a finished file on disk is not redone. Then the routes: finish-all
on a restored rehearsal and the finish of a rendered move, each refusing
cleanly without the key and streaming to `done` with the fake.
"""

from __future__ import annotations

import io
import json
import shutil
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from PIL import Image, ImageDraw

from locaish.meta import FakeMetaClient, MetaError
from locaish.meta import finish as F
from locaish.product import cast as castmod

RED, BLUE = (230, 80, 60), (60, 130, 230)



def _images(client):
    """The image calls only: the review's vision calls sit between them in the log."""
    return [e for e in client.log if e["kind"] == "image"]


def _png(w=80, h=50, colour=(90, 120, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format="PNG")
    return buf.getvalue()


def _frame(w=160, h=90, figures=None, caption=12) -> bytes:
    """A grey room with coloured stand-ins at given boxes (0-1) and a caption bar under it."""
    im = Image.new("RGB", (w, h + caption), (14, 16, 19))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w - 1, h - 1], fill=(120, 118, 110))
    for colour, (x0, y0, x1, y1) in (figures or []):
        d.rectangle([int(x0 * w), int(y0 * h), int(x1 * w) - 1, int(y1 * h) - 1], fill=colour)
    d.text((4, h + 1), "t=1.0s shot 3", fill=(200, 200, 200))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _cast():
    return [
        castmod.Actor(name="Maya", height_m=1.68, colour=RED, wardrobe="a dark green jumper, black jeans",
                      description="19, short brown hair, slim"),
        castmod.Actor(name="Theo", height_m=1.84, colour=BLUE, wardrobe="grey t-shirt, white trousers, barefoot",
                      description="20, dark curly hair, tall and lean"),
    ]


def _meta(focal=35.0, both=True):
    # Camera at (0.28, -0.12); Maya at (-1.72, -1.22) facing 45 deg (towards the camera);
    # Theo at (-1.02, -0.52) facing -135 deg (back to the camera).
    actors = {"Maya": [-1.72, -1.22, 45.0]}
    if both:
        actors["Theo"] = [-1.02, -0.52, -135.0]
    return {"t": 1.0, "actors": actors, "camera": {"pos": [0.281, -0.115, 1.551], "aim": [-1.72, -1.22, 1.58], "focal_mm": focal, "shot": 3},
            "sun": {"direct_sun": True, "windows_lit": [0], "compass": "SW", "elevation_deg": 19.9},
            "framing": {"subject": "Maya", "size_name": "Medium close-up", "distance_m": 2.3}}


# ---------------------------------------------------------------------------
# the prompt
# ---------------------------------------------------------------------------


def test_prompt_names_each_stand_in_by_colour_with_looks_wardrobe_and_facing():
    boxes = {"Maya": (0.3, 0.32, 0.6, 0.94), "Theo": (0.42, 0.0, 1.0, 0.94)}
    p = F.build_sequence_prompt(meta=_meta(), cast=_cast(), headshot_index={"Theo": 2}, continuity_index=3, boxes=boxes)
    assert "The RED figure is Maya" in p and "short brown hair" in p and "dark green jumper" in p
    assert "The BLUE figure is Theo" in p and "dark curly hair" in p and "white trousers" in p
    assert "image 2 is Theo's headshot" in p and "Image 3 is the previous finished frame" in p
    assert "35 mm" in p and "EXACTLY 2 people" in p and "Maya and Theo" in p
    # The measured geometry, in words: Maya faces the camera, Theo has his back to it and fills the right.
    assert "Maya (19, short brown hair, slim, 1.68 m tall), a real person standing exactly where the figure is, the same height, FACING THE CAMERA" in p
    assert "BACK TO THE CAMERA" in p and "Theo is in the FOREGROUND" in p
    assert "on the right of the frame, 58% of the frame wide and 94% of the frame tall" in p
    assert "MEDIUM CLOSE-UP on Maya" in p and "direct sun through window 1 from the SW" in p
    assert "SAME CROP" in p and "No text" in p
    # Without a headshot or a previous frame, neither is mentioned.
    p1 = F.build_prompt(focal_mm=50, actors=["Maya"])
    assert "EXACTLY" in p1
    p2 = F.build_sequence_prompt(meta=_meta(both=False), cast=_cast(), headshot_index={})
    assert "Theo" not in p2 and "EXACTLY 1 person" in p2 and "previous finished frame" not in p2 and "headshot" not in p2


def test_prompt_with_nobody_in_frame_and_scene_light_override():
    meta = _meta()
    meta["actors"] = {}
    p = F.build_sequence_prompt(meta=meta, cast=_cast(), headshot_index={}, scene_light="is one practical lamp, warm, from the left", mood="late and tired")
    assert "no people in this frame" in p and "Maya" not in p
    assert "one practical lamp" in p and "late and tired." in p and "direct sun" not in p


def test_figure_boxes_from_the_stand_in_colours_and_the_caption_strip():
    png = _frame(figures=[(RED, (0.3, 0.3, 0.6, 0.9)), (BLUE, (0.5, 0.0, 1.0, 0.95))])
    im = Image.open(io.BytesIO(png))
    assert im.size == (160, 102)
    bare = F.strip_caption(png)
    assert Image.open(io.BytesIO(bare)).size == (160, 90)
    boxes = F.figure_boxes(bare, _cast())
    assert set(boxes) == {"Maya", "Theo"}
    mx0, my0, mx1, my1 = boxes["Maya"]
    assert abs(mx0 - 0.3) < 0.02 and abs(my0 - 0.3) < 0.02 and abs(mx1 - 0.5) < 0.02 and abs(my1 - 0.9) < 0.02
    tx0, ty0, tx1, ty1 = boxes["Theo"]
    assert abs(tx0 - 0.5) < 0.02 and ty0 == 0.0 and tx1 == 1.0 and abs(ty1 - 0.95) < 0.02
    assert "cut off by the top edge and the right edge" in F._place_words(boxes["Theo"])
    # A frame with no bar passes through untouched; a frame with nobody has no boxes.
    plain = _frame(caption=0)
    assert F.strip_caption(plain) == plain and F.figure_boxes(plain, _cast()) == {}
    assert F.colour_name((231, 79, 61)) == "red" and F.colour_name((60, 130, 230)) == "blue"


# ---------------------------------------------------------------------------
# the sequence
# ---------------------------------------------------------------------------


def _frames(n=5):
    out = []
    for i in range(n):
        png = _frame(figures=[(RED, (0.3, 0.3, 0.6, 0.9)), (BLUE, (0.5, 0.0, 1.0, 0.95))])
        out.append((f"frame_{i:04d}.png", png, _meta(focal=20.0 + i)))
    return out


def test_first_frame_goes_alone_then_continuity_rides_along_and_files_land(tmp_path):
    fake = FakeMetaClient(image=_png(80, 50, (90, 120, 200)))
    events = []
    out = F.finish_sequence(_frames(5), client=fake, cast=_cast(), headshots={"Theo": _png(8, 8)},
                            out_dir=tmp_path / "fin", workers=2, progress=events.append)
    assert [f.name for f in out] == [f"frame_{i:04d}.png" for i in range(5)]
    assert all((tmp_path / "fin" / f"finish_frame_{i:04d}.png").exists() for i in range(5))
    assert len(_images(fake)) == 5 and all(e["kind"] in ("start", "done") for e in events)
    assert sum(1 for e in events if e["kind"] == "done") == 5 and events[0]["total"] == 5
    first = _images(fake)[0]
    assert "20 mm" in first["prompt"], "the first frame is finished first, before any parallel chunk"
    assert "previous finished frame" not in first["prompt"] and first["references"] == 2  # the frame + Theo's headshot
    for call in _images(fake)[1:]:
        # the frame after the anchor carries it once, as both; a later frame of the
        # same setup carries the anchor and the setup\'s previous finished frame
        assert "previous finished frame" in call["prompt"] and call["references"] in (3, 4)
        assert "image 2 is Theo's headshot" in call["prompt"] and "Image 3 is the" in call["prompt"]
    assert out[0].references == 1 and out[1].references == 2 and out[1].meta["continuity"] is True and out[0].meta["continuity"] is False
    # Stamped as a render: the bottom-left corner is darkened.
    im = Image.open(io.BytesIO(out[0].png))
    # the answer is cropped to the render's aspect (16:9) before it is stamped
    assert im.size == (80, 45) and sum(im.getpixel((6, 28))) < sum((90, 120, 200))
    assert Image.open(io.BytesIO(out[0].raw)).size == (80, 45) and out[0].png != out[0].raw
    # Cache: a second run touches nothing; force redoes it all.
    events.clear()
    again = F.finish_sequence(_frames(5), client=fake, cast=_cast(), headshots={}, out_dir=tmp_path / "fin", progress=events.append)
    assert len(_images(fake)) == 5 and all(f.cached for f in again) and all(e["kind"] == "skip" for e in events)
    forced = F.finish_sequence(_frames(5), client=fake, cast=_cast(), headshots={}, out_dir=tmp_path / "fin", force=True)
    assert len(_images(fake)) == 10 and not any(f.cached for f in forced)
    assert F.finish_sequence([], client=fake, cast=_cast(), headshots={}, out_dir=tmp_path / "fin") == []


class _FlakyClient(FakeMetaClient):
    def generate_image(self, prompt, *, references=None, model=None, size=None):
        if "22 mm" in prompt:
            self.calls += 1
            raise MetaError("/images/edits returned 500: the model fell over", status=500)
        return super().generate_image(prompt, references=references, model=model, size=size)


def test_a_failing_frame_leaves_no_file_and_is_reported_and_the_run_goes_on(tmp_path):
    fake = _FlakyClient(image=_png())
    events = []
    out = F.finish_sequence(_frames(5), client=fake, cast=_cast(), headshots={}, out_dir=tmp_path / "fin", workers=1, progress=events.append)
    assert [f.name for f in out] == ["frame_0000.png", "frame_0001.png", "frame_0003.png", "frame_0004.png"]
    assert not (tmp_path / "fin" / "finish_frame_0002.png").exists()
    errs = [e for e in events if e["kind"] == "error"]
    assert len(errs) == 1 and errs[0]["name"] == "frame_0002.png" and "fell over" in errs[0]["text"]
    assert len(_images(fake)) == 4   # the failed call is not logged
    # The frame after the failure still has a continuity reference: the last one that finished.
    assert "previous finished frame" in _images(fake)[-1]["prompt"]


def test_no_continuity_means_no_extra_reference(tmp_path):
    fake = FakeMetaClient(image=_png())
    F.finish_sequence(_frames(3), client=fake, cast=_cast(), headshots={}, out_dir=tmp_path / "fin", continuity=False, workers=3)
    assert all(c["references"] == 1 and "previous finished frame" not in c["prompt"] for c in _images(fake))


# ---------------------------------------------------------------------------
# the routes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fin_url(clean, tmp_path_factory):
    from locaish import serve as servemod
    from locaish.viewer.build import render_html

    result, _fx = clean
    root = tmp_path_factory.mktemp("finseq")
    job = root / "cleanroom"
    job.mkdir()
    result.twin.save(job / "test-clean.twin")
    render_html(result.twin, job / "view.html", max_points=20_000)
    # A rehearsal on disk, as a finished run leaves it: the stage restores it.
    rd = job / "rehearsals" / "abc12345"
    (rd / "animatic").mkdir(parents=True)
    frames = []
    for i in range(4):
        png = _frame(figures=[(RED, (0.3, 0.3, 0.6, 0.9)), (BLUE, (0.5, 0.0, 1.0, 0.95))])
        (rd / "animatic" / f"frame_{i:04d}.png").write_bytes(png)
        m = _meta(focal=20.0 + i)
        m["file"] = f"frame_{i:04d}.png"
        frames.append(m)
    (rd / "animatic" / "animatic.json").write_text(json.dumps({"title": "T", "frames": frames}))
    (rd / "rehearsal.json").write_text(json.dumps({"scene": {"title": "T", "beats": []}, "rehearsal": {"shots": [], "problems": [], "duration_s": 4}, "animatic": frames, "provider": "typed"}))
    studio = servemod.Studio(root, max_points=20_000)
    handler = type("H", (servemod._Handler,), {"studio": studio})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", studio
    finally:
        httpd.shutdown()


def _call(url, *, method="GET", body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _json(url, **kw):
    status, raw = _call(url, **kw)
    return status, json.loads(raw or b"{}")


def _sse(url, timeout=300.0):
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


STATE: dict = {}


def test_cast_carries_wardrobe_and_description_and_they_edit_in_place(fin_url):
    url, _ = fin_url
    status, r = _json(f"{url}/cast/cleanroom", method="POST", body={"name": "Maya", "height_m": 1.68, "wardrobe": "a dark green jumper", "description": "19, short brown hair"})
    assert status == 200 and r["actor"]["wardrobe"] == "a dark green jumper" and r["actor"]["colour"] == list(RED)
    maya = r["actor"]["id"]
    status, r = _json(f"{url}/cast/cleanroom", method="POST", body={"name": "Theo", "height_m": 1.84})
    assert status == 200 and r["actor"]["wardrobe"] == "" and r["actor"]["colour"] == list(BLUE)
    theo = r["actor"]["id"]
    status, r = _json(f"{url}/cast/cleanroom/{theo}", method="POST", body={"wardrobe": "grey t-shirt, white trousers", "description": "20, curly hair"})
    assert status == 200 and r["actor"]["wardrobe"] == "grey t-shirt, white trousers" and r["actor"]["description"] == "20, curly hair"
    status, r = _json(f"{url}/cast/cleanroom")
    by = {a["name"]: a for a in r["cast"]}
    assert by["Theo"]["wardrobe"] == "grey t-shirt, white trousers" and by["Maya"]["description"] == "19, short brown hair"
    assert _json(f"{url}/cast/cleanroom/nobody", method="POST", body={"wardrobe": "x"})[0] == 404
    STATE["maya"], STATE["theo"] = maya, theo


def test_finish_all_refuses_without_a_key_then_finishes_the_rehearsal(fin_url, monkeypatch):
    url, studio = fin_url
    status, r = _json(f"{url}/rehearsal/cleanroom/abc12345/finished")
    assert status == 200 and r["files"] == []
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    status, r = _json(f"{url}/rehearsal/cleanroom/abc12345/finish-all", method="POST", body={})
    assert status == 200 and r["run"] and "META_API_KEY" in (r["error"] or "")
    ev = _sse(f"{url}{r['events']}")
    assert ev[-1]["kind"] == "error" and "META_API_KEY" in ev[-1]["text"]
    assert _json(f"{url}/rehearsal/cleanroom/nosuch/finish-all", method="POST", body={})[0] == 404
    monkeypatch.setenv("META_API_KEY", "test")
    fake = FakeMetaClient(image=_png(80, 50, (90, 120, 200)))
    monkeypatch.setattr(studio.stage, "_client", lambda: fake)
    status, r = _json(f"{url}/rehearsal/cleanroom/abc12345/finish-all", method="POST", body={"workers": 2})
    assert status == 200 and r["run"] and r["error"] is None and r["events"].startswith("/rehearsal-finish-events/cleanroom/abc12345/")
    first_events = r["events"]
    ev = _sse(f"{url}{first_events}")
    done = ev[-1]
    assert done["kind"] == "done", done
    assert done["files"] == [f"finish_frame_{i:04d}.png" for i in range(4)] and done["finished"] == 4 and done["failed"] == 0
    frames = [e for e in ev if e["kind"] == "frame"]
    assert len(frames) == 4 and frames[0]["url"] == "/rehearsal-image/cleanroom/abc12345/finish_frame_0000.png"
    assert "The RED figure is Maya" in done["prompt"] and "dark green jumper" in done["prompt"]
    assert len(_images(fake)) == 4 and _images(fake)[0]["references"] == 1 and _images(fake)[1]["references"] == 2
    status, raw = _call(f"{url}/rehearsal-image/cleanroom/abc12345/finish_frame_0002.png")
    assert status == 200 and raw[:8] == b"\x89PNG\r\n\x1a\n"
    status, r = _json(f"{url}/rehearsal/cleanroom/abc12345/finished")
    assert r["files"] == done["files"]
    # Again: everything is cached, the model is not called; a reconnect replays done.
    status, r2 = _json(f"{url}/rehearsal/cleanroom/abc12345/finish-all", method="POST", body={})
    ev2 = _sse(f"{url}{r2['events']}")
    assert ev2[-1]["kind"] == "done" and ev2[-1]["cached"] == 4 and len(_images(fake)) == 4
    assert _sse(f"{url}{first_events}")[-1]["kind"] == "done"
    assert _json(f"{url}/rehearsal-finish-events/cleanroom/abc12345/nope")[0] == 404
    # The one-frame button uses the same instruction and lands in the same place.
    status, r = _json(f"{url}/rehearsal/cleanroom/abc12345/finish/frame_0001.png", method="POST", body={})
    assert status == 200 and r["file"] == "finish_frame_0001.png" and "BLUE figure is Theo" in r["prompt"] and len(_images(fake)) == 5


def test_sim_finish_after_a_render_makes_the_held_video(fin_url, monkeypatch):
    url, studio = fin_url
    status, r = _json(f"{url}/sims/cleanroom", method="POST", body={})
    assert status == 200, r
    sid = r["timeline"]["id"]
    status, r = _json(f"{url}/sim/cleanroom/{sid}")
    assert status == 200 and r["files"] == {"move_mp4": False, "move_finished_mp4": False, "rendered": 0, "finished": 0}
    # Before a render there is nothing to finish.
    monkeypatch.setenv("META_API_KEY", "test")
    fake = FakeMetaClient(image=_png(96, 54, (90, 120, 200)))
    monkeypatch.setattr(studio.stage, "_client", lambda: fake)
    status, r = _json(f"{url}/sim/cleanroom/{sid}/finish", method="POST", body={"every": 8})
    assert status == 200 and "render the move first" in r["error"]
    assert _sse(f"{url}{r['events']}")[-1]["kind"] == "error"
    status, r = _json(f"{url}/sim/cleanroom/{sid}/render", method="POST", body={"width_px": 240, "every": 6})
    assert status == 200
    rend = _sse(f"{url}{r['events']}")[-1]
    assert rend["kind"] == "done" and rend["frames"] >= 8
    # Without the key the finish refuses on its own stream.
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setattr(studio.stage, "_client", lambda: None)
    status, r = _json(f"{url}/sim/cleanroom/{sid}/finish", method="POST", body={"every": 8})
    assert status == 200 and "META_API_KEY" in r["error"] and _sse(f"{url}{r['events']}")[-1]["kind"] == "error"
    monkeypatch.setenv("META_API_KEY", "test")
    monkeypatch.setattr(studio.stage, "_client", lambda: fake)
    status, r = _json(f"{url}/sim/cleanroom/{sid}/finish", method="POST", body={"every": 8})
    assert status == 200 and r["error"] is None and r["events"].startswith(f"/sim-finish-events/cleanroom/{sid}/")
    ev = _sse(f"{url}{r['events']}")
    done = ev[-1]
    assert done["kind"] == "done", done
    expect = list(range(0, rend["frames"], 8))
    if (rend["frames"] - 1) not in expect:
        expect.append(rend["frames"] - 1)
    assert done["files"] == [f"finish_frame_{i:05d}.png" for i in expect] and done["every"] == 8
    assert len(_images(fake)) == len(expect) and "SAME CROP" in done["prompt"]
    d = studio.jobs["cleanroom"].workdir / "sims" / sid
    assert (d / "finish" / "finish_frame_00000.png").exists()
    status, raw = _call(f"{url}/sim-file/cleanroom/{sid}/finish_frame_00000.png")
    assert status == 200 and raw[:8] == b"\x89PNG\r\n\x1a\n"
    if shutil.which("ffmpeg"):
        assert done["mp4"] == f"/sim-file/cleanroom/{sid}/move_finished.mp4", done
        assert (d / "move_finished.mp4").exists() and (d / "finish" / "frames.txt").exists()
        status, raw = _call(f"{url}{done['mp4']}")
        assert status == 200 and raw[4:8] == b"ftyp"
    else:
        assert done["mp4"] is None and "ffmpeg" in done["note"]
    status, r = _json(f"{url}/sim/cleanroom/{sid}")
    assert r["files"]["finished"] == len(expect) and r["files"]["move_finished_mp4"] == bool(shutil.which("ffmpeg"))
    assert _json(f"{url}/sim-finish-events/cleanroom/{sid}/nope")[0] == 404



def test_room_photographs_ride_as_references_and_the_prompt_says_what_they_fix(tmp_path):
    from PIL import Image

    from locaish.meta import FakeMetaClient
    from locaish.meta import finish as F
    from locaish.product import cast as castmod

    def png(w, h, colour):
        buf = io.BytesIO()
        Image.new("RGB", (w, h), colour).save(buf, format="PNG")
        return buf.getvalue()

    fake = FakeMetaClient(image=png(64, 36, (80, 90, 100)))
    frame = png(200, 100, (40, 40, 40))
    room1, room2 = b"\xff\xd8\xff" + b"r1", b"\xff\xd8\xff" + b"r2"
    cast = [castmod.Actor(name="Maya", height_m=1.68)]
    meta = {"camera": {"pos": [0, -2, 1.5], "aim": [0, 0, 1.5], "focal_mm": 35}, "actors": {"Maya": [0, 0, 270]}}
    seen = []

    def room_refs(m):
        seen.append(m)
        return [room1, room2]

    out = F.finish_sequence([("frame_0001.png", frame, meta)], client=fake, cast=cast, headshots={},
                            out_dir=tmp_path, room_refs=room_refs, continuity=False)
    assert seen and out and out[0].meta["room_photos"] == 2
    assert _images(fake)[-1]["references"] == 3                      # the frame plus two photographs
    assert "THIS EXACT ROOM" in out[0].prompt and "Image 2 and image 3 are" in out[0].prompt
    # A reference source that fails leaves the frame finished without photographs.
    fake2 = FakeMetaClient(image=png(64, 36, (80, 90, 100)))

    def broken(m):
        raise RuntimeError("no frames")

    out2 = F.finish_sequence([("frame_0002.png", frame, meta)], client=fake2, cast=cast, headshots={},
                             out_dir=tmp_path, room_refs=broken, continuity=False)
    assert out2 and out2[0].meta["room_photos"] == 0 and "THIS EXACT ROOM" not in out2[0].prompt
