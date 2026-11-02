"""The finishing contract that stops the painter inventing a room: what the
frame's camera sees, what the photographs were shot under, who is out of
frame, what the reviewer is shown, and the two picture fixes (the caption
strip, the answer's aspect)."""
from __future__ import annotations

import io

from PIL import Image, ImageDraw

from locaish.meta import finish as F
from locaish.meta.client import FakeMetaClient


def _png(w=160, h=90, colour=(40, 40, 40), figures=None, bar=0):
    im = Image.new("RGB", (w, h + bar), colour)
    d = ImageDraw.Draw(im)
    for col, (x0, y0, x1, y1) in figures or []:
        d.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], fill=col)
    if bar:
        d.rectangle([0, h, w, h + bar], fill=(14, 16, 19))
        d.text((6, h + 2), "15:30:01 shot 3 35mm", fill=(200, 200, 200))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _cast():
    from locaish.product.cast import Actor

    return [Actor(name="Maya", height_m=1.68, colour=(230, 80, 60)), Actor(name="Theo", height_m=1.84, colour=(60, 130, 230))]


def _meta(**kw):
    m = {"camera": {"pos": [0.0, 0.0, 1.5], "aim": [3.0, 0.0, 1.4], "focal_mm": 16.0, "shot": 1},
         "actors": {"Maya": [2.0, 0.5, 180.0], "Theo": [2.0, -3.0, 90.0]}, "clock": "15:30:00",
         "sun": {"direct_sun": True, "windows_lit": [1], "azimuth_deg": 235.0, "elevation_deg": 15.0, "compass": "SW"},
         "staging": {"Maya": {"pose": "standing", "business": "at the window", "eyeline": "Theo"},
                     "Theo": {"pose": "standing", "business": "inside the door", "eyeline": "Maya"}}}
    m.update(kw)
    return m


class _Opening:
    def __init__(self, center, width, height, kind="window"):
        self.center, self.width, self.height, self.kind = center, width, height, kind


def test_layout_words_say_what_the_camera_sees_left_to_right_and_what_it_does_not():
    things = [("bunk bed", [2.5, 0.9, 0.6]), ("wardrobe", [2.5, -0.9, 1.0]), ("tower fan", [-1.0, 0.0, 0.8]),
              ("carpet floor", [1.0, 0.0, 0.0]), ("study desk", [3.0, 0.0, 0.7])]
    wins = [_Opening([3.2, 0.2, 1.2], 1.9, 1.2), _Opening([-2.0, -0.5, 1.2], 0.7, 1.3)]
    s = F.layout_words(_meta(), things, wins)
    assert s.startswith("In this frame, left to right: the bunk bed on the left, 2.5 m away")
    assert "the study desk in the centre, 3.0 m away, the wardrobe on the right, 2.5 m away" in s
    assert "window 1 (1.9 x 1.2 m) in the centre, 3.2 m away" in s and "The only window in this frame" in s
    assert "Not in this frame: tower fan." in s and "carpet" not in s, "floors are not things; the fan is behind the camera"
    assert "paint nothing that is not listed and no window that is not listed" in s and "doors are as the photographs" in s
    # the camera turned to the blank wall: no window, and where the windows are instead
    s2 = F.layout_words(_meta(camera={"pos": [0.0, 0.0, 1.5], "aim": [0.0, 3.0, 1.4], "focal_mm": 16.0}), things, wins)
    assert "No window is in this frame (window 1 is off frame to the right; window 2 is behind the camera)" in s2 \
        or "No window is in this frame (window 2 is behind the camera; window 1 is off frame to the right)" in s2
    assert F.layout_words({"camera": {}}, things, wins) == ""


def test_photo_light_words_tell_the_painter_the_photographs_are_not_the_hour():
    info = {"time": "night", "source": "artificial", "windows": "black"}
    s = F.photo_light_words(info, [3, 4])
    assert s.startswith("The photographs were taken at night under the room's own lamps and ceiling lights, with the windows black")
    assert "NOT their light" in s
    assert F.photo_light_words(info, None) == "" and F.photo_light_words(None, [3]) == ""
    fake = FakeMetaClient(image=_png(), chat_answers=[info])
    assert F.classify_photo_light(fake, [_png(), _png()]) == info
    assert fake.log[-1]["kind"] == "chat_json" and fake.log[-1]["images"] == 2
    assert F.classify_photo_light(FakeMetaClient(image=_png(), chat_answers=[{"nonsense": 1}]), [_png()]) is None


def test_the_daylight_sentence_makes_the_windows_show_the_day():
    s = F.light_words_for({"direct_sun": True, "windows_lit": [1], "azimuth_deg": 235, "elevation_deg": 15, "clock": "15:30"})
    assert "Every window in this frame shows a pale daylight sky" in s and "never black glass" in s
    assert "if the window that passes the sun is not in this frame, the sun comes in from outside the frame" in s
    assert "ceiling lights are off" in s
    assert "ceiling lights are off" not in F.light_words_for({"windows_lit": [], "azimuth_deg": 250, "elevation_deg": -4})


def test_the_prompt_carries_the_layout_the_photo_light_and_keeps_an_out_of_frame_actor_out():
    meta = _meta()
    boxes = {"Maya": (0.4, 0.2, 0.6, 0.9)}          # Theo has no stand-in: he is off frame
    p = F.build_sequence_prompt(meta=meta, cast=_cast(), headshot_index={}, boxes=boxes, room_index=[2, 3],
                                layout="In this frame, left to right: the bunk bed on the left, 2.5 m away.",
                                photo_light={"time": "night", "source": "artificial", "windows": "black"})
    assert "There are EXACTLY 1 person in this frame, Maya, and nobody else" in p
    assert "Theo is in the room but OUT OF FRAME: do not paint Theo or any part of them" in p
    assert "The RED figure is Maya" in p and "The BLUE figure is Theo" not in p
    assert "In this frame, left to right: the bunk bed on the left" in p
    # Maya is "at the window" but this camera sees none: the painter is told
    p_nowin = F.build_sequence_prompt(meta=meta, cast=_cast(), headshot_index={}, boxes=boxes, room_index=[2],
                                      layout="No window is in this frame (window 1 is behind the camera).")
    assert "The window Maya is at or looks toward is OUTSIDE this frame: paint no window for it." in p_nowin
    assert "The window Theo" not in p_nowin
    assert "The photographs were taken at night" in p and "from OTHER camera positions" in p
    # without boxes (a blank render in tests) the old path stands: everyone on the marks is described
    p0 = F.build_sequence_prompt(meta=meta, cast=_cast(), headshot_index={})
    assert "EXACTLY 2 people" in p0 and "OUT OF FRAME" not in p0 and "The photographs were taken" not in p0


def test_the_reviewer_sees_the_photographs_and_the_anchor_and_holds_the_frame_to_the_light_and_layout():
    fake = FakeMetaClient(image=_png(), chat_answers=[{"ok": False, "problems": ["make the window daylight"]}])
    ok, problems = F.review_finished(fake, _png(), _png(), _meta(), _cast(), photos=[_png(), _png()], anchor=_png(),
                                     boxes={"Maya": (0, 0, 1, 1)}, light_text="is real daylight at 15:30",
                                     layout="No window is in this frame.")
    assert not ok and problems == ["make the window daylight"]
    call = fake.log[-1]
    assert call["images"] == 5
    assert "Images 3 and 4 are photographs of the real room" in call["text"]
    assert "Image 5 is another finished frame of this same scene" in call["text"]
    assert "exactly 1 person in it (Maya)" in call["text"], "the count comes from the stand-ins in the render"
    assert 'the light should be: "is real daylight at 15:30"' in call["text"] and "black glass" in call["text"]
    assert 'the layout says: "No window is in this frame."' in call["text"]
    assert "Maya must be on their feet" in call["text"] and "Theo must" not in call["text"], "Theo is out of frame"


def test_the_caption_strip_is_cut_even_with_text_in_it_and_the_answer_is_cropped_to_the_render():
    png = _png(160, 90, bar=14)
    assert Image.open(io.BytesIO(png)).size == (160, 104)
    assert Image.open(io.BytesIO(F.strip_caption(png))).size == (160, 90)
    assert Image.open(io.BytesIO(F.strip_caption(_png(160, 90)))).size == (160, 90)
    wide = _png(400, 160)
    out = Image.open(io.BytesIO(F.crop_to_aspect(wide, (160, 90))))
    assert out.size == (284, 160) or out.size == (285, 160)
    assert F.crop_to_aspect(_png(320, 180), (160, 90)) == _png(320, 180)


def test_a_sequence_finishes_setup_by_setup_and_a_later_frame_keeps_its_setups_background(tmp_path):
    cast = _cast()
    figs = [((230, 80, 60), (0.3, 0.3, 0.6, 0.9))]
    frames = []
    for i, shot in enumerate([1, 1, 2, 2, 1]):
        cam = {"pos": [0.0, float(shot), 1.5], "aim": [3.0, float(shot), 1.4], "focal_mm": 16.0, "shot": shot}
        frames.append((f"frame_{i:04d}.png", _png(figures=figs), _meta(camera=cam)))
    fake = FakeMetaClient(image=_png())
    out = F.finish_sequence(frames, client=fake, cast=cast, out_dir=tmp_path, workers=2, review=False,
                            layout=lambda m: f"In this frame: shot {m['camera']['shot']}.",
                            photo_light={"time": "night", "source": "artificial", "windows": "black"},
                            room_refs=lambda m: [_png(), _png()])
    assert len(out) == 5
    prompts = [e["prompt"] for e in fake.log if e["kind"] == "image"]
    # the anchor (shot 1) first, then shot 2's first frame, then the rest
    assert "In this frame: shot 1." in prompts[0] and "In this frame: shot 2." in prompts[1]
    assert all("The photographs were taken at night" in p for p in prompts)
    same = [p for p in prompts if "THIS SAME SETUP" in p]
    assert len(same) == 3, "frames 1, 3 and 4 each carry their own setup's previous finished frame"
    assert (tmp_path / "finish_frame_0004.json").exists()
