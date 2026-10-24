"""Finish a frame photoreal with Muse Image, and say so on the picture.

The renderer's frame is the measurement: a pinhole at the planned setup,
through the planned lens, at the twin's own gaussian field, with the cast
drawn in at their heights. It is also grey bodies in a slightly soft room,
and a director will not fall in love with it. Muse Image takes that frame as
the picture to change, the actor's headshot and a wardrobe or mood reference
as the pictures to match, and an instruction that says exactly what may
move: nothing. Faces, clothes, light and texture; not the camera, not the
lens, not where anyone stands, not the room.

Whatever comes back is a render. It is stamped as one in the corner, kept
beside the measured frame rather than in its place, and nothing downstream
reads a number off it. That is the bargain the twin makes and the finished
frame does not get to break it.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .client import model_for

STAMP = "RENDER · Muse Image · not a measurement"

_INSTRUCTION = (
    "Photoreal cinematography still. Keep the camera position, lens, framing, "
    "perspective, room geometry, furniture positions and the position and "
    "height of every person EXACTLY as in the first image; do not move, add or "
    "remove anything. Replace the grey stand-in figures with the real people "
    "from the reference headshots (same faces, natural skin, hair and "
    "expression), wearing {wardrobe}. Render the room's surfaces as real "
    "materials with the light coming {light}. {mood} Natural film look, "
    "shallow depth of field appropriate to a {focal_mm:.0f} mm lens on a "
    "{sensor}. No text, no watermark, no extra people."
)


@dataclass
class FinishedFrame:
    png: bytes
    prompt: str
    model: str
    provider: str = "meta"
    references: int = 0
    note: str = "a render, not a measurement; the measured frame stands beside it"
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.png)
        return path


def build_prompt(
    *,
    focal_mm: float,
    sensor: str = "full-frame sensor",
    wardrobe: str = "the clothes described in the scene",
    light: str = "through the windows as it does at the planned hour",
    mood: str = "",
    actors: list[str] | None = None,
    description: str = "",
) -> str:
    text = _INSTRUCTION.format(wardrobe=wardrobe, light=light, mood=mood.strip(),
                               focal_mm=float(focal_mm), sensor=sensor)
    if actors:
        text += " The people, left to right as the headshots are given: " + ", ".join(actors) + "."
    if description:
        text += " The shot: " + description.strip()
    return " ".join(text.split())


def finish_frame(
    frame_png: bytes,
    *,
    client,
    focal_mm: float,
    headshots: list[bytes] | None = None,
    references: list[bytes] | None = None,
    actors: list[str] | None = None,
    description: str = "",
    wardrobe: str | None = None,
    light: str | None = None,
    mood: str = "",
    sensor: str = "full-frame sensor",
    model: str | None = None,
    stamp: bool = True,
) -> FinishedFrame:
    """One rendered frame in, one finished still out, stamped as a render.

    `headshots` are the cast in the order they appear left to right;
    `references` are wardrobe or mood pictures. The API takes up to ten
    pictures in all, the frame included; the rest are dropped, cast first.
    """
    kwargs: dict[str, Any] = {"focal_mm": focal_mm, "sensor": sensor, "mood": mood,
                              "actors": actors, "description": description}
    if wardrobe:
        kwargs["wardrobe"] = wardrobe
    if light:
        kwargs["light"] = light
    prompt = build_prompt(**kwargs)
    refs = [frame_png] + list(headshots or []) + list(references or [])
    refs = refs[:10]
    model = model or model_for("image")
    out = client.generate_image(prompt, references=refs, model=model)
    if stamp:
        out = stamp_render(out)
    return FinishedFrame(png=out, prompt=prompt, model=model, references=len(refs) - 1,
                         meta={"headshots": len(headshots or []), "references": len(references or [])})


def stamp_render(png: bytes, text: str = STAMP) -> bytes:
    """A small label in the bottom-left corner that survives a screenshot."""
    from PIL import Image, ImageDraw

    im = Image.open(io.BytesIO(png)).convert("RGB")
    draw = ImageDraw.Draw(im, "RGBA")
    pad = max(4, im.width // 160)
    try:
        box = draw.textbbox((0, 0), text)
    except Exception:  # noqa: BLE001 - very old PIL
        box = (0, 0, 7 * len(text), 11)
    w, h = box[2] - box[0], box[3] - box[1]
    x0, y0 = pad, im.height - h - 3 * pad
    draw.rectangle([x0 - pad, y0 - pad, x0 + w + pad, y0 + h + pad], fill=(0, 0, 0, 170))
    draw.text((x0, y0), text, fill=(255, 255, 255, 255))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()
