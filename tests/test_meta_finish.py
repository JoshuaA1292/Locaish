"""Finishing a frame: the prompt forbids moving anything, the output is stamped."""

from __future__ import annotations

import io

from PIL import Image

from locaish.meta import FakeMetaClient
from locaish.meta import finish as F


def _png(w=120, h=60, colour=(40, 40, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format="PNG")
    return buf.getvalue()


def test_prompt_pins_camera_and_names_actors():
    p = F.build_prompt(focal_mm=50, actors=["Maya", "Theo"], description="Theo turns to her", wardrobe="a wet raincoat")
    assert "EXACTLY" in p and "50 mm" in p and "Maya, Theo" in p and "wet raincoat" in p
    assert "Theo turns to her" in p


def test_finish_frame_sends_frame_first_and_stamps_the_render():
    fake = FakeMetaClient(image=_png(200, 100, (90, 120, 200)))
    frame = _png()
    out = F.finish_frame(frame, client=fake, focal_mm=35, headshots=[_png(8, 8)], references=[_png(8, 8)] * 12,
                         actors=["Maya"])
    assert fake.log[0]["kind"] == "image" and fake.log[0]["references"] == 10
    assert out.references == 9 and out.model == "muse-image-1.0" and out.provider == "meta"
    im = Image.open(io.BytesIO(out.png))
    assert im.size == (200, 100)
    # The stamp darkens the bottom-left corner.
    px = im.getpixel((6, 85))
    assert sum(px) < sum((90, 120, 200))
    assert "not a measurement" in out.note


def test_stamp_is_optional():
    fake = FakeMetaClient(image=_png(50, 50, (10, 200, 10)))
    out = F.finish_frame(_png(), client=fake, focal_mm=35, stamp=False)
    assert out.png == fake._image


def _sitting_meta():
    return {"actors": {"Theo": [1.4, -1.1, 147.5]},
            "staging": {"Theo": {"pose": "sitting", "business": "typing at the study desk", "eyeline": "down"}}}


def test_a_seated_person_is_put_on_a_named_seat():
    posture, _expr = F._staging_words(_sitting_meta(), "Theo")
    assert "SITTING on the desk chair at the desk" in posture and "never floating" in posture
    assert F._seat_words("perched on the edge of the bed") == "the edge of the bed"
    assert F._seat_words("waiting") == "a chair"


def test_the_review_sends_a_floating_person_back_and_the_frame_is_finished_again(tmp_path):
    from locaish.product.cast import Actor

    fake = FakeMetaClient(image=_png(160, 90, (90, 120, 200)), chat_answers=[
        {"ok": False, "problems": ["Theo floats above the floor with no chair under him; seat him on the desk chair"]},
        {"ok": True, "problems": []},
    ])
    theo = Actor(name="Theo", height_m=1.84)
    out = F.finish_sequence([("frame_0001.png", _png(160, 90), _sitting_meta())], client=fake, cast=[theo],
                            out_dir=tmp_path, workers=1)
    images = [e for e in fake.log if e["kind"] == "image"]
    reviews = [e for e in fake.log if e["kind"] == "chat_json"]
    assert len(images) == 2 and len(reviews) == 2
    assert "CORRECTION" in images[1]["prompt"] and "no chair under him" in images[1]["prompt"]
    assert "CORRECTION" not in images[0]["prompt"]
    assert reviews[0]["images"] == 2 and "sitting on the desk chair" in reviews[0]["text"]
    assert out[0].meta["review"] == {"rejected": 1, "verdicts": [
        {"ok": False, "problems": ["Theo floats above the floor with no chair under him; seat him on the desk chair"]},
        {"ok": True, "problems": []}]}


def test_the_review_can_be_switched_off_and_gives_up_after_its_tries(tmp_path):
    from locaish.product.cast import Actor

    theo = Actor(name="Theo", height_m=1.84)
    quiet = FakeMetaClient(image=_png(160, 90))
    F.finish_sequence([("frame_0001.png", _png(160, 90), _sitting_meta())], client=quiet, cast=[theo],
                      out_dir=tmp_path / "a", workers=1, review=False)
    assert [e["kind"] for e in quiet.log] == ["image"]
    stubborn = FakeMetaClient(image=_png(160, 90), chat_answers=[{"ok": False, "problems": ["still floating"]}] * 5)
    out = F.finish_sequence([("frame_0001.png", _png(160, 90), _sitting_meta())], client=stubborn, cast=[theo],
                            out_dir=tmp_path / "b", workers=1, review_tries=2)
    # every attempt is judged, the last one too, and the best of them is kept
    assert [e["kind"] for e in stubborn.log] == ["image", "chat_json", "image", "chat_json", "image", "chat_json"]
    assert out[0].meta["review"]["rejected"] == 2 and out[0].path.exists()
    assert len(out[0].meta["review"]["verdicts"]) == 3 and out[0].path.with_suffix(".json").exists()
