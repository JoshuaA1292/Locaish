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
