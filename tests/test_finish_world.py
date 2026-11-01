"""Finishing a set of views so they agree, on the fake client.

The views are finished in a greedy nearest-neighbour order from view 0; the
first finished view is the anchor in every later call; references respect
the ten-image limit in the order frame, anchor, neighbour, room photographs,
headshots; the outputs are unstamped and exactly the measured view's size
whatever size the model answered at; the prompt carries the dressing, the
clock and azimuth, THIS EXACT ROOM when photographs ride, and the sentence
that makes the set agree; a view the model fails on is skipped and reported;
a finished file on disk is kept unless forced.
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

from locaish.meta import FakeMetaClient, MetaError
from locaish.meta import finish as F
from locaish.product import cast as castmod

RED, BLUE = (230, 80, 60), (60, 130, 230)
LIGHT = {"azimuth_deg": 225, "elevation_deg": 20, "clock": "16:40", "date": "2026-11-14", "windows_lit": [1]}


def _png(w=80, h=50, colour=(90, 120, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format="PNG")
    return buf.getvalue()


def _view(w=160, h=107, figures=None) -> bytes:
    """A grey room, no caption strip, with coloured stand-ins at boxes in 0-1."""
    im = Image.new("RGB", (w, h), (120, 118, 110))
    d = ImageDraw.Draw(im)
    for colour, (x0, y0, x1, y1) in (figures or []):
        d.rectangle([int(x0 * w), int(y0 * h), int(x1 * w) - 1, int(y1 * h) - 1], fill=colour)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _cast():
    return [
        castmod.Actor(name="Maya", height_m=1.68, colour=RED, wardrobe="a dark green jumper, black jeans",
                      description="19, short brown hair, slim"),
        castmod.Actor(name="Theo", height_m=1.84, colour=BLUE, wardrobe="grey t-shirt, white trousers",
                      description="20, dark curly hair, tall and lean"),
    ]


# Cameras on a ring, listed in a scrambled order so the nearest-neighbour
# order is visibly not the given order: 0 at the east, then its neighbours
# are 3 and 2 (not 1).
POS = {0: (2.0, 0.0, 1.5), 1: (-2.0, 0.0, 1.5), 2: (1.4, 1.4, 1.5), 3: (1.4, -1.4, 1.5),
       4: (0.0, 2.0, 1.5), 5: (0.0, -2.0, 1.5)}


def _views(n=6, figures=True, size=(160, 107)):
    out = []
    for i in range(n):
        figs = [(RED, (0.3, 0.3, 0.45, 0.9)), (BLUE, (0.6, 0.2, 0.75, 0.95))] if figures else None
        png = _view(size[0], size[1], figs)
        meta = {"name": f"view_{i:04d}", "camera": {"pos": list(POS[i]), "aim": [0.0, 0.0, 1.5], "focal_mm": 24.0},
                "actors": {"Maya": [0.3, 0.2, 0.0], "Theo": [-0.4, 0.1, 180.0]} if figures else {}}
        out.append((f"view_{i:04d}", png, meta))
    return out


# ---------------------------------------------------------------------------
# the words
# ---------------------------------------------------------------------------


def test_light_words_from_the_ephemeris_dict_and_the_animatic_sun_dict():
    w = F.light_words_for(LIGHT, rendered_sun=False)
    assert "at 16:40 on 2026-11-14" in w and "through window 2" in w and "from the SW (azimuth 225 degrees)" in w
    assert "20 degrees elevation" in w and "where that sun falls" in w and "already shows" not in w
    # The animatic's own sun dict reads exactly as it always has.
    sun = {"direct_sun": True, "windows_lit": [0], "compass": "SW", "elevation_deg": 19.9}
    assert F.light_words_for(sun) == F._light_words({"sun": sun}, "")
    assert F.light_words_for(sun).startswith("is real daylight: direct sun through window 1 from the SW at 20 degrees elevation")
    # Sun up but through no window: soft daylight, with the sun placed; sun down: after sunset; nothing: the render's light.
    soft = F.light_words_for({"azimuth_deg": 90, "elevation_deg": 30, "clock": "09:00", "windows_lit": []})
    assert soft.startswith("is soft daylight at 09:00") and "in the E at 30 degrees" in soft
    assert F.light_words_for({"azimuth_deg": 300, "elevation_deg": -5, "clock": "19:30", "windows_lit": []}).startswith("is after sunset at 19:30")
    assert F.light_words_for(None).startswith("is the room's own light")
    assert F._compass(0) == "N" and F._compass(225) == "SW" and F._compass(350) == "N" and F._compass(120) == "SE"


def test_world_prompt_says_set_agreement_dressing_light_room_and_people():
    _name, png, meta = _views()[2]
    boxes = F.figure_boxes(png, _cast())
    p = F.build_world_prompt(meta=meta, cast=_cast(), headshot_index={"Theo": 6}, anchor_index=2, neighbour_index=3,
                             room_index=[4, 5], light=LIGHT, dressing="as it is, the bed made", boxes=boxes)
    assert "ONE OF A SET of photographs of the same room at the same moment" in p
    assert "the set must agree on every surface, object, light and person" in p
    assert "Image 2 is the first finished photograph of this set" in p and "same people in the same clothes" in p
    assert "Image 3 is a finished photograph of this set taken from a camera near this one" in p
    assert "Image 4 and image 5 are photographs of THIS EXACT ROOM" in p
    assert "The room is dressed as: as it is, the bed made." in p
    assert "16:40" in p and "azimuth 225 degrees" in p and "through window 2" in p
    assert "The RED figure is Maya" in p and "dark green jumper" in p and "image 6 is Theo's headshot" in p
    assert "EXACTLY 2 people" in p and "24 mm lens" in p and "SAME CROP" in p
    assert "no stamp" in p and "no caption bar" in p and "with flat coloured stand-in figures" in p
    # Nobody, no anchor, no photographs, no dressing, no light: each sentence stays away.
    meta2 = dict(meta, actors={})
    p2 = F.build_world_prompt(meta=meta2, cast=_cast(), headshot_index={})
    assert "no people in this frame" in p2 and "Maya" not in p2 and "stand-in figures" not in p2
    assert "first finished photograph" not in p2 and "THIS EXACT ROOM" not in p2 and "dressed as" not in p2
    assert "room's own light" in p2
    # Light and dressing in the meta are used when none is given.
    meta3 = dict(meta, light=LIGHT, dressing="for a wake")
    p3 = F.build_world_prompt(meta=meta3, cast=_cast(), headshot_index={})
    assert "dressed as: for a wake." in p3 and "16:40" in p3


def test_resize_to_makes_the_exact_size():
    out = F.resize_to(_png(1024, 683), (160, 107))
    assert Image.open(io.BytesIO(out)).size == (160, 107) and out[:8] == b"\x89PNG\r\n\x1a\n"
    assert Image.open(io.BytesIO(F.resize_to(_png(160, 107), (160, 107)))).size == (160, 107)


# ---------------------------------------------------------------------------
# the order and the references
# ---------------------------------------------------------------------------


def test_world_order_is_greedy_nearest_neighbour_from_view_zero():
    assert F.world_order(_views()) == [0, 2, 4, 1, 5, 3] or F.world_order(_views()) == [0, 3, 5, 1, 4, 2]
    # A view with no camera comes last; an empty set is empty.
    v = _views(4)
    v[1][2]["camera"] = {}
    assert F.world_order(v)[-1] == 1 and F.world_order([]) == []


class _Recorder(FakeMetaClient):
    """Keeps the reference bytes, not just their count, so a test can say which picture rode where."""

    def generate_image(self, prompt, *, references=None, model=None, size=None):
        out = super().generate_image(prompt, references=references, model=model, size=size)
        self.log[-1]["refs"] = list(references or [])
        return out


def test_anchor_rides_in_every_later_call_and_neighbours_are_the_nearest_finished(tmp_path):
    # The model answers at a size of its own, with a picture nothing like the view.
    fake = _Recorder(image=_png(1024, 683, (200, 40, 40)))
    events = []
    views = _views()
    out = F.finish_world_views(views, client=fake, cast=_cast(), out_dir=tmp_path / "fin", workers=2,
                               light=LIGHT, dressing="as it is", progress=events.append)
    assert [f.name for f in out] == [f"view_{i:04d}" for i in range(6)]
    assert fake.calls == 6 and sum(1 for e in events if e["kind"] == "done") == 6
    order = [int(e["index"]) for e in events if e["kind"] == "start"]
    want = F.world_order(views)
    # The anchor alone, then chunks of two whose threads may start in either order.
    assert order[0] == want[0] == 0 and want[1] in (2, 3)
    assert all(set(order[k:k + 2]) == set(want[k:k + 2]) for k in range(1, 6, 2))
    anchor = out[0]
    first = fake.log[0]
    assert first["references"] == 1 and "first finished photograph" not in first["prompt"]  # the log counts the frame itself
    for call in fake.log[1:]:
        assert call["refs"][0] is not anchor.raw and call["refs"][1] == anchor.raw, "image 2 is always the anchor"
        assert "Image 2 is the first finished photograph of this set" in call["prompt"]
        assert "as it is" in call["prompt"] and "16:40" in call["prompt"] and "azimuth 225" in call["prompt"]
    # The second chunk has a finished neighbour besides the anchor; the first chunk has only the anchor.
    assert all(not f.meta["neighbour"] for f in out if f.name in (views[want[1]][0], views[want[2]][0]))
    later = [f for f in out if f.name in (views[want[3]][0], views[want[4]][0], views[want[5]][0])]
    assert later and all(f.meta["neighbour"] and f.references == 2 for f in later)
    # The neighbour is the nearest finished view that is not the anchor, by camera position.
    by_name = {f.name: f for f in out}
    idx = {v[0]: i for i, v in enumerate(views)}
    for f in later:
        i = idx[f.name]
        finished_before = [idx[n] for n in f.meta["neighbours"]]
        assert all(F._dist(POS[i], POS[j]) <= F._dist(POS[i], POS[k]) for j in finished_before[:1] for k in finished_before[1:])
        assert f.meta["neighbours"] == views[i][2]["neighbours"]
    # Outputs: unstamped, exactly the measured size, and raw is the same bytes (nothing to strip).
    for f in out:
        im = Image.open(io.BytesIO(f.png))
        assert im.size == (160, 107) and f.raw == f.png and f.path.exists() and f.path.name == f"finish_{f.name}.png"
        assert im.getpixel((6, 95)) == im.getpixel((80, 50)), "no stamp darkens the corner"
    assert by_name["view_0000"].meta["anchor"] is False and by_name["view_0001"].meta["anchor"] is True


def test_reference_priority_within_the_ten_image_limit(tmp_path):
    """Frame, anchor, neighbour, two room photographs, then headshots; the limit bites the headshots."""
    fake = _Recorder(image=_png(64, 43))
    room1, room2 = b"\xff\xd8\xff" + b"r1", b"\xff\xd8\xff" + b"r2"
    heads = {"Maya": _png(8, 8, (1, 1, 1)), "Theo": _png(8, 8, (2, 2, 2))}
    views = _views(4)
    out = F.finish_world_views(views, client=fake, cast=_cast(), headshots=heads, room_refs=lambda m: [room1, room2, b"x3"],
                               out_dir=tmp_path / "fin", workers=1)
    assert len(out) == 4
    first = fake.log[0]
    assert first["references"] == 5 and first["refs"][1:3] == [room1, room2] and first["refs"][3:] == [heads["Maya"], heads["Theo"]]
    assert "Image 2 and image 3 are photographs of THIS EXACT ROOM" in first["prompt"] and "image 4 is Maya's headshot" in first["prompt"]
    second = fake.log[1]
    assert second["references"] == 6 and second["refs"][1] == out[0].raw and second["refs"][2:4] == [room1, room2]
    assert "Image 2 is the first finished photograph" in second["prompt"] and "Image 3 and image 4 are photographs of THIS EXACT ROOM" in second["prompt"]
    assert "a camera near this one" not in second["prompt"], "nothing finished yet but the anchor"
    third = fake.log[2]
    assert third["references"] == 7 and third["refs"][1] == out[0].raw and third["refs"][2] in [f.raw for f in out[1:]]
    assert third["refs"][3:5] == [room1, room2] and "Image 3 is a finished photograph of this set taken from a camera near" in third["prompt"]
    assert "Image 4 and image 5 are photographs of THIS EXACT ROOM" in third["prompt"] and "image 6 is Maya's headshot" in third["prompt"]
    # Eight actors in frame: the limit drops the last headshots, never the anchor, neighbour or room.
    big_cast = [castmod.Actor(name=f"A{k}", height_m=1.7, colour=F.COLOUR_NAMES[k][0]) for k in range(8)]
    big_heads = {a.name: _png(8, 8, (k, k, k)) for k, a in enumerate(big_cast)}
    fake2 = _Recorder(image=_png(64, 43))
    views2 = _views(3)
    for _n, _p, m in views2:
        m["actors"] = {a.name: [0.0, 0.0, 0.0] for a in big_cast}
    F.finish_world_views(views2, client=fake2, cast=big_cast, headshots=big_heads, room_refs=lambda m: [room1, room2],
                         out_dir=tmp_path / "fin2", workers=1)
    last = fake2.log[-1]
    assert last["references"] == 10 and len(last["refs"]) == 10
    assert last["refs"][3:5] == [room1, room2] and last["refs"][5:] == [big_heads[f"A{k}"] for k in range(5)]
    assert "image 10 is A4's headshot" in last["prompt"] and "A5's headshot" not in last["prompt"]


# ---------------------------------------------------------------------------
# failure, cache, force
# ---------------------------------------------------------------------------


class _Flaky(FakeMetaClient):
    def __init__(self, fail_when: str, **kw):
        super().__init__(**kw)
        self.fail_when = fail_when

    def generate_image(self, prompt, *, references=None, model=None, size=None):
        self.log.append({"kind": "image", "prompt": prompt, "references": len(references or [])})
        self.calls += 1
        if self.fail_when == "first" and self.calls == 1:
            raise MetaError("/images/edits returned 500", status=500)
        if self.fail_when == "third" and self.calls == 3:
            raise MetaError("/images/edits returned 503", status=503)
        return self._image


def test_a_failing_view_is_skipped_and_reported_and_a_failing_anchor_passes_the_role_on(tmp_path):
    fake = _Flaky("third", image=_png(64, 43))
    events = []
    out = F.finish_world_views(_views(5), client=fake, cast=_cast(), out_dir=tmp_path / "fin", workers=1, progress=events.append)
    assert len(out) == 4 and fake.calls == 5
    errs = [e for e in events if e["kind"] == "error"]
    assert len(errs) == 1 and errs[0]["name"] == f"view_{errs[0]['index']:04d}" and "503" in errs[0]["text"]
    assert not (tmp_path / "fin" / f"finish_{errs[0]['name']}.png").exists()
    assert errs[0]["name"] not in [f.name for f in out]
    # View 0 fails: the next in the order finishes alone and becomes the anchor for the rest.
    fake2 = _Flaky("first", image=_png(64, 43))
    events.clear()
    out2 = F.finish_world_views(_views(4), client=fake2, cast=_cast(), out_dir=tmp_path / "fin2", workers=2, progress=events.append)
    names = [f.name for f in out2]
    assert "view_0000" not in names and len(out2) == 3
    assert fake2.log[0]["references"] == 1 and fake2.log[1]["references"] == 1, "two alone, the first failed"
    assert all(c["references"] >= 2 and "first finished photograph" in c["prompt"] for c in fake2.log[2:])
    assert [e["kind"] for e in events].count("error") == 1


def test_cache_keeps_finished_files_and_force_redoes_them(tmp_path):
    fake = FakeMetaClient(image=_png(64, 43))
    F.finish_world_views(_views(4), client=fake, cast=_cast(), out_dir=tmp_path / "fin", workers=2)
    assert fake.calls == 4
    events = []
    again = F.finish_world_views(_views(4), client=fake, cast=_cast(), out_dir=tmp_path / "fin", workers=2, progress=events.append)
    assert fake.calls == 4 and all(f.cached for f in again) and all(e["kind"] == "skip" for e in events)
    assert all(f.raw == f.png and Image.open(io.BytesIO(f.png)).size == (160, 107) for f in again)
    # One file missing: only that one is finished, with the cached anchor riding along.
    (tmp_path / "fin" / "finish_view_0002.png").unlink()
    part = F.finish_world_views(_views(4), client=fake, cast=_cast(), out_dir=tmp_path / "fin", workers=2)
    assert fake.calls == 5 and [f.name for f in part if not f.cached] == ["view_0002"]
    assert fake.log[-1]["references"] >= 2 and "first finished photograph" in fake.log[-1]["prompt"]
    forced = F.finish_world_views(_views(4), client=fake, cast=_cast(), out_dir=tmp_path / "fin", workers=2, force=True)
    assert fake.calls == 9 and not any(f.cached for f in forced)
    assert F.finish_world_views([], client=fake, cast=_cast(), out_dir=tmp_path / "fin") == []


def test_a_pan_from_one_spot_runs_in_aim_order_and_neighbours_are_the_adjacent_aims(tmp_path):
    """Four cameras on one spot, aims scrambled: position ties, so the aim decides."""
    import math

    views = []
    for i, deg in enumerate((0.0, 50.0, 25.0, 75.0)):
        aim = [-0.8 + 3 * math.cos(math.radians(deg)), -0.8 + 3 * math.sin(math.radians(deg)), 1.5]
        views.append((f"view_{i:04d}", _view(), {"camera": {"pos": [-0.8, -0.8, 1.5], "aim": aim, "focal_mm": 24.0}, "actors": {}}))
    assert F.world_order(views) == [0, 2, 1, 3]
    fake = FakeMetaClient(image=_png(64, 43))
    out = F.finish_world_views(views, client=fake, cast=[], out_dir=tmp_path / "fin", workers=1)
    by = {f.name: f for f in out}
    assert by["view_0003"].meta["neighbours"][0] == "view_0001", "75 deg's nearest finished neighbour is 50 deg, not the anchor at 0"
    assert by["view_0001"].meta["neighbours"][0] == "view_0002"
    # A world of far-apart cameras is still ordered by position first.
    assert F.world_order(_views()) == [0, 2, 4, 1, 5, 3]
