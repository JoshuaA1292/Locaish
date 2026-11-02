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

`finish_sequence` does it for a whole rehearsal or a rendered move: every
frame gets the same instruction, each stand-in figure is named by its colour
as the actor it stands for (with the actor's own words for how they look and
what they wear, their headshot when there is one, and when the beat staged
them, their pose, business, eyeline and an expression read off the line),
and from the second frame on the first finished frame of the scene rides
along as the ANCHOR every later frame is held to, with the nearest finished
frame by camera as a second reference, so the scene does not change clothes,
faces or light between setups. The light is said once for the scene, from
the rehearsal's own sun and clock. The first frame goes alone; the rest run
in order with up to `workers` in flight.

`finish_world_views` does it for a set of views that must agree rather than
follow: sixty cameras covering the room at one hour, finished UNSTAMPED and
at the measured view's exact size because a gaussian splat is trained on
them and must not learn a stamp. The first view is the anchor every later
view is held to; each later view also carries its nearest finished neighbour
and the scan's photographs of what it looks at. The viewer page that serves
the result carries the "render" label the pictures cannot.
"""

from __future__ import annotations

import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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
    name: str = ""                 # the measured frame's file name, in a sequence
    path: Path | None = None       # where the finished PNG was written
    cached: bool = False           # it was already on disk and not redone
    raw: bytes | None = None       # the model's answer before the stamp, for the next frame to match

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


# ---------------------------------------------------------------------------
# a whole sequence
# ---------------------------------------------------------------------------

# The palette's names, so a prompt can say "the RED figure" about (230, 80, 60).
# Nearest colour wins, so an actor on a colour off the palette still gets a word.
COLOUR_NAMES: tuple[tuple[tuple[int, int, int], str], ...] = (
    ((230, 80, 60), "red"),
    ((60, 130, 230), "blue"),
    ((240, 180, 40), "amber"),
    ((70, 190, 110), "green"),
    ((190, 90, 210), "violet"),
    ((240, 120, 200), "pink"),
    ((40, 190, 200), "teal"),
    ((250, 140, 50), "orange"),
)

_SEQUENCE_INSTRUCTION = (
    "Photoreal film still. {shot}Image 1 is a rough 3D render of a real room with flat "
    "coloured stand-in figures. Re-render image 1 as a photograph taken from EXACTLY "
    "the same camera: same position and height, same {focal_mm:.0f} mm lens, same framing "
    "and perspective, same picture edges; the walls, floor, ceiling, windows and every "
    "piece of furniture stay exactly where they are and the same size. Do not move, add "
    "or remove anything. The output is the SAME CROP as image 1: do not widen the shot, do "
    "not step back, do not show more of the room or more of anyone than image 1 shows; a "
    "person cut off by a frame edge in image 1 stays cut off the same way. Where image 1 is black, "
    "smeared or empty the room was not measured there, not dark: fill it with the real surface the "
    "photographs show at that place, never with a dark blur. {room}{layout}{count}{people}{nobody}"
    "Make every surface a real material: the real walls, floor, furniture and fabrics "
    "with texture and grain, sharp and clean, not blurred. {photo_light}The light {light}{mood} "
    "{continuity}Natural film look, correct skin, correct hands. No text, no caption bar, "
    "no watermark, no extra people, no border."
)

_PERSON = (
    "The {colour} figure is {name} ({about}){ref}, a real person standing exactly where "
    "the figure is, the same height, {facing}the same pose and the same "
    "distance from the camera, wearing {wardrobe}.{place} "
)

# A staged person: the beat said how they hold themselves, what their hands
# are doing and where they look, and the line says what their face is doing.
_PERSON_STAGED = (
    "The {colour} figure is {name} ({about}){ref}, a real person exactly where the figure "
    "is, the same height and the same distance from the camera, {facing}{posture}, "
    "{expression}, wearing {wardrobe}.{place} "
)

_COMPOSITION = (
    "Composition: {subject} is placed in the {third} third of the frame, {look}, {room}; keep "
    "that framing exactly, with the headroom image 1 has. "
)

_NOBODY = "There are no people in this frame; leave it empty of people. "

_ROOM = (
    "{images} {verb} photographs of THIS EXACT ROOM taken on site from OTHER camera positions: "
    "the output must show this real room and no other, with the same walls, floor, "
    "ceiling, window openings, furniture, bedding, objects, clutter, colours and materials as "
    "those photographs, seen from image 1's camera and at image 1's crop. They fix what the "
    "room is made of and NOT where the camera is: do not copy their framing, their viewpoint "
    "or how much of the room they show, and do not bring in anything they show that image 1's "
    "camera cannot see. Image 1 fixes the geometry and where the people stand. "
)

# What is in THIS frame, from the twin's own labels and openings projected
# through the frame's camera: without it the photographs (which show other
# walls) become the layout authority and a wardrobe turns into a window.
_LAYOUT = (
    "In this frame, left to right: {present}. {windows} Not in this frame: {absent}. Image 1 is "
    "blurred where these things are; this list is the authority: paint nothing that is not "
    "listed and no window that is not listed (the room's doors are as the photographs show them). "
)

# The scan's photographs and the frame can differ in time of day, and the
# painter has to be told which to believe about the light.
_PHOTO_LIGHT = (
    "The photographs were taken {when} under {source}, with the windows {windows}: take the "
    "room's layout, furniture, materials and colours from them and NOT their light; ignore "
    "their lamps, their colour cast and what their windows show. "
)

_SAME_SETUP_AND_ANCHOR = (
    "Image {n} is the first photograph of this scene and the previous finished frame of THIS SAME "
    "SETUP, from the SAME camera: its background is this frame's background; every wall, window, "
    "piece of furniture, object and patch of light is in exactly the same place and looks exactly "
    "the same, and its materials, light, faces, hair and wardrobe are matched EXACTLY; change only "
    "the people's pose and expression. "
)

_SAME_SETUP = (
    "Image {n} is the previous finished frame of THIS SAME SETUP, from the SAME camera: its "
    "background is this frame's background; every wall, window, piece of furniture, object and "
    "patch of light is in exactly the same place and looks exactly the same; change only the "
    "people's pose and expression. "
)

_CONTINUITY = (
    "Image {n} is the previous finished frame of this scene{near}: match its look, the room's "
    "materials and colours, the lighting, the furniture and objects in the same places in the "
    "room, and each person's face, hair and wardrobe EXACTLY. "
)

_SCENE_ANCHOR = (
    "Image {n} is the first photograph of this scene, from its own camera: match its materials, "
    "light, faces, hair and wardrobe EXACTLY, and keep the furniture and objects where that frame "
    "has them in the room, so this frame and image {n} cut together as one scene shot at one hour. "
    "Do not copy its composition or anything it shows that image 1's camera cannot see. "
)

_SCENE_ANCHOR_AND_PREVIOUS = (
    "Image {n} is the previous finished frame of this scene and its first photograph: match its "
    "materials, light, faces, hair and wardrobe EXACTLY, and keep the furniture and objects where "
    "it has them in the room, so this frame and image {n} cut together as one scene shot at one "
    "hour. Do not copy its composition or anything it shows that image 1's camera cannot see. "
)


def colour_name(rgb) -> str:
    r, g, b = (int(v) for v in rgb)
    best, best_d = "coloured", 10 ** 9
    for (cr, cg, cb), name in COLOUR_NAMES:
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < best_d:
            best, best_d = name, d
    return best


def figure_boxes(png: bytes, cast: list[Any], *, tolerance: int = 70) -> dict[str, tuple[float, float, float, float]]:
    """Where each stand-in sits in the measured frame, from its own colour: the
    actor's name to a box (x0, y0, x1, y1) in 0-1 of the picture. A pixel
    belongs to the actor whose palette colour it is nearest, if it is within
    `tolerance` (the render blurs the edges). Actors with no pixels are absent."""
    from PIL import Image
    import numpy as np

    if not cast:
        return {}
    im = Image.open(io.BytesIO(png)).convert("RGB")
    a = np.asarray(im, dtype=np.int16)
    cols = np.asarray([[int(v) for v in c.colour] for c in cast], dtype=np.int16)  # (n, 3)
    d = np.abs(a[:, :, None, :] - cols[None, None, :, :]).sum(axis=-1)           # (h, w, n) L1
    nearest = d.argmin(axis=-1)
    close = d.min(axis=-1) <= tolerance
    out: dict[str, tuple[float, float, float, float]] = {}
    h, w = a.shape[:2]
    for k, actor in enumerate(cast):
        m = close & (nearest == k)
        if m.sum() < max(20, (h * w) // 2000):
            continue
        ys, xs = np.nonzero(m)
        out[actor.name] = (float(xs.min() / w), float(ys.min() / h), float((xs.max() + 1) / w), float((ys.max() + 1) / h))
    return out


def _facing_words(meta: dict[str, Any], name: str) -> str:
    """How the actor stands to the camera, from the animatic's own numbers:
    the actor's (x, y, facing_deg) against the camera's position. Facing is
    degrees anticlockwise from +X in plan, as the cast module draws it."""
    import math

    actors = meta.get("actors")
    cam = (meta.get("camera") or {}).get("pos") or meta.get("cam")
    if not isinstance(actors, dict) or cam is None:
        return "facing the same way, "
    entry = actors.get(name)
    try:
        ax, ay, yaw = float(entry[0]), float(entry[1]), float(entry[2])
        cx, cy = float(cam[0]), float(cam[1])
    except (TypeError, ValueError, IndexError):
        return "facing the same way, "
    to_cam = (cx - ax, cy - ay)
    n = math.hypot(*to_cam)
    if n < 1e-6:
        return "facing the same way, "
    fx, fy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    dot = (fx * to_cam[0] + fy * to_cam[1]) / n
    cross = (to_cam[0] * fy - to_cam[1] * fx) / n   # >0: facing is anticlockwise of the camera direction
    if dot > 0.6:
        return "FACING THE CAMERA (face visible), "
    if dot < -0.6:
        return "with their BACK TO THE CAMERA (seen from behind, face not visible), "
    side = "left" if cross > 0 else "right"
    return f"in profile, facing frame {side}, "


def _staging_words(meta: dict[str, Any], name: str) -> tuple[str, str] | None:
    """The posture and the expression for one actor, from the frame's own
    staging and line: None when the beat staged nothing for them (the old
    "standing exactly where the figure is" sentence then applies)."""
    staged = (meta.get("staging") or {}).get(name) if isinstance(meta.get("staging"), dict) else None
    if not isinstance(staged, dict) or not any(staged.get(k) for k in ("pose", "business", "eyeline")):
        return None
    pose = str(staged.get("pose") or "standing")
    business = str(staged.get("business") or "").strip().rstrip(".")
    eyeline = str(staged.get("eyeline") or "").strip()
    bits = []
    if pose == "sitting":
        # The render shows a seated figure and nothing under it, and left to
        # itself the model paints a person squatting in mid-air: name the seat.
        bits.append(f"SITTING on {_seat_words(business)}, their full weight on it, the seat plainly "
                    "visible under them (a real chair or seat of this room, never floating, never a "
                    "crouch in mid-air)")
    else:
        bits.append({"leaning": "LEANING with their weight on one side, on their feet",
                     "walking": "mid-stride, walking, on their feet",
                     "standing": _STANDING}.get(pose, pose))
    if business:
        bits.append(business if business.lower().startswith(("with ", "hands", "arms", "leaning", "looking", "typing", "turning", "holding"))
                    else business)
    if eyeline:
        low = eyeline.lower()
        if low == "camera":
            bits.append("looking straight into the lens")
        elif low == "down":
            bits.append("eyes down, not at the camera")
        elif low in ("window", "door"):
            bits.append(f"looking at the {low}, not at the camera")
        else:
            bits.append(f"looking at {eyeline if eyeline[:1].isupper() else 'the ' + eyeline}, not at the camera")
    posture = ", ".join(bits)
    return posture, _expression(meta, name, eyeline)


# A waist-up or chest-up framing of a standing figure reads to the painter as a
# person perched on something, and it paints a crouch: say what the framing is.
_STANDING = ("STANDING upright on both feet at their full height (never sitting, crouching, kneeling, "
             "squatting or perched on anything; if the frame cuts them at the chest or waist, that is "
             "the framing of a standing person, not a seated one)")

_SEATS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("desk", "laptop", "typing", "keyboard", "writing"), "the desk chair at the desk"),
    (("bed", "mattress", "bunk"), "the edge of the bed"),
    (("sofa", "couch", "settee"), "the sofa"),
    (("armchair",), "the armchair"),
    (("stool",), "the stool"),
    (("bench",), "the bench"),
    (("sill", "window"), "the window sill"),
    (("floor", "carpet", "rug"), "the floor"),
    (("stairs", "step"), "the step"),
)


def _seat_words(business: str) -> str:
    """What a seated person sits on, read off the beat's business: the desk
    chair when they type at the desk, the edge of the bed, the sofa; a chair
    when the beat does not say."""
    low = business.lower()
    for keys, seat in _SEATS:
        if any(k in low for k in keys):
            return seat
    return "a chair"


# The review: Muse Spark looks at the finished frame beside the render and
# names what the model invented, so the frame can be finished again with the
# correction in the prompt. It fails a frame for what a producer would see
# at once and a diff of pixels never would: a person with no seat under
# them, the wrong head count, a hand too many, the furniture rearranged.
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}, "problems": {"type": "array", "items": {"type": "string"}}},
    "required": ["ok", "problems"],
}

_REVIEW = (
    "Image 1 is a finished photoreal film still. Image 2 is the rough 3D render it was made from, "
    "with flat coloured stand-in figures where the people go. {photos}{anchor}Judge image 1 and answer JSON "
    '{{"ok": true/false, "problems": [...]}}. It FAILS when any of these is true: there are not '
    "exactly {count} {people_word} in it{names}; a seated person has no chair or seat visibly under "
    "them, floats, or squats in mid-air; a person who should be on their feet is sitting, "
    "crouching, kneeling, squatting or perched on anything; anyone has an extra or missing limb, hand or face; a "
    "person stands where image 2 has no figure, or a figure's place has no person; the framing "
    "differs from image 2 (image 1 shows more or less of the room, the camera stepped back or in, "
    "or a person is clearly smaller or larger in the frame than their figure in image 2); the room's "
    "furniture is moved, added or removed compared with image 2{photo_fail}{layout_fail}{light_fail}; "
    "there is text, a caption bar or a watermark. {seated}Each problem is one short sentence saying "
    "what to change, as an instruction to the painter. Say nothing about the stand-in figures "
    "themselves and nothing about image quality or style. An empty list means ok."
)

_REVIEW_PHOTOS = (
    "Images {nums} are photographs of the real room taken on site from other positions: they say what "
    "the room contains and what it is made of. "
)
_REVIEW_ANCHOR = "Image {n} is another finished frame of this same scene, from its own camera. "
_REVIEW_PHOTO_FAIL = ("; a window, door, piece of furniture or object appears that the photographs do not have at "
                      "that place, or the room is plainly a different room from the photographs, or the room is "
                      "shown mirrored or from the opposite side")
_REVIEW_ANCHOR_FAIL = "; where image 1 and image {n} show the same part of the room, its furniture, walls or objects differ"
_REVIEW_LAYOUT_FAIL = "; the layout says: \"{layout}\": a thing it lists as not in frame appears, or a window it does not list appears"
_REVIEW_LIGHT_FAIL = ("; the light should be: \"{light}\": a window shows black glass or night while the room is in "
                      "daylight (or daylight while the frame is after sunset), sunlight enters from a wall with no "
                      "window, or a ceiling light burns in daylight")


def _small(png: bytes, side: int = 1024) -> bytes:
    """The picture no larger than `side` on its long edge, as JPEG, for a
    judge; a small picture passes through as it is."""
    from PIL import Image

    try:
        im = Image.open(io.BytesIO(png))
    except Exception:  # noqa: BLE001 - not a picture we can shrink: send as is
        return png
    if max(im.size) <= side and len(png) < 600_000:
        return png
    im = im.convert("RGB")
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def review_finished(client, finished_png: bytes, measured_png: bytes, meta: dict[str, Any], cast: list[Any],
                    *, model: str | None = None, photos: list[bytes] | None = None, anchor: bytes | None = None,
                    boxes: dict[str, Any] | None = None, light_text: str = "", layout: str = "") -> tuple[bool, list[str]]:
    """Muse Spark's verdict on a finished frame: (ok, problems). Problems are
    phrased as corrections for the next attempt. `photos` are the room's
    photographs the painter was given, `anchor` the scene's first finished
    frame, `boxes` the stand-ins found in the render (the people count comes
    from them when given), `light_text` and `layout` the sentences the prompt
    said, so the reviewer holds the frame to the same contract."""
    present = meta.get("actors")
    names = list(present.keys()) if isinstance(present, dict) else [a.name for a in cast]
    if boxes:
        names = [n for n in names if n in boxes]
    # the judge needs no more than 1024 px a side: five full-size stills is
    # a request the API refuses, and a refused review is no review
    images = [_small(finished_png), _small(measured_png)]
    photo_nums: list[int] = []
    for p in list(photos or [])[:2]:
        images.append(_small(p))
        photo_nums.append(len(images))
    anchor_n = None
    if anchor:
        images.append(_small(anchor))
        anchor_n = len(images)
    seated = []
    for name in names:
        staged = (meta.get("staging") or {}).get(name) if isinstance(meta.get("staging"), dict) else None
        if isinstance(staged, dict) and str(staged.get("pose") or "") == "sitting":
            seated.append(f"{name} must be sitting on {_seat_words(str(staged.get('business') or ''))}")
        elif isinstance(staged, dict) and str(staged.get("pose") or "standing") in ("standing", "walking", "leaning"):
            seated.append(f"{name} must be on their feet, standing at full height, not sitting, crouching or kneeling")
    nums = (f"{photo_nums[0]}" if len(photo_nums) == 1 else " and ".join(str(n) for n in photo_nums))
    text = _REVIEW.format(
        count=len(names), people_word="person" if len(names) == 1 else "people",
        names=(" (" + ", ".join(names) + ")") if names else "",
        photos=_REVIEW_PHOTOS.format(nums=nums).replace("Images " + nums + " are", "Image " + nums + " is") if len(photo_nums) == 1 else (_REVIEW_PHOTOS.format(nums=nums) if photo_nums else ""),
        anchor=_REVIEW_ANCHOR.format(n=anchor_n) if anchor_n else "",
        photo_fail=_REVIEW_PHOTO_FAIL if photo_nums else "",
        layout_fail=(_REVIEW_LAYOUT_FAIL.format(layout=layout.strip()) if layout and layout.strip() else "")
                    + (_REVIEW_ANCHOR_FAIL.format(n=anchor_n) if anchor_n else ""),
        light_fail=_REVIEW_LIGHT_FAIL.format(light=light_text.strip().rstrip(".")) if light_text and light_text.strip() else "",
        seated=(" ".join(s + "." for s in seated) + " ") if seated else "",
    )
    ans = client.chat_json(text, images, model=model, schema=REVIEW_SCHEMA, max_tokens=500)
    if not isinstance(ans, dict):
        return True, []
    problems = [str(p).strip() for p in (ans.get("problems") or []) if str(p).strip()]
    ok = bool(ans.get("ok", not problems)) and not problems
    return ok, problems


def correction_words(problems: list[str]) -> str:
    """The sentence a rejected frame's prompt gains for its second attempt."""
    fixes = " ".join(p.rstrip(".") + "." for p in problems)
    return (" CORRECTION, from a review of the previous attempt at this frame: " + fixes +
            " Fix exactly these and change nothing else.")


def _expression(meta: dict[str, Any], name: str, eyeline: str) -> str:
    """What the face is doing, read off the beat: the speaker speaks (asks,
    raises their voice), someone whose eyeline is on the speaker listens,
    someone looking elsewhere is turned away. Not a mood the model invents."""
    speaker = meta.get("speaker")
    line = str(meta.get("line") or "").strip()
    if speaker == name and line:
        if line.endswith("?"):
            return "mid-sentence, asking a question, mouth slightly open"
        if "!" in line:
            return "mid-sentence, raising their voice"
        return "mid-sentence, speaking, mouth slightly open"
    if speaker and speaker != name:
        if eyeline and eyeline.lower() in ("window", "door", "down"):
            return "turned away from the person speaking, not meeting their eyes"
        return f"listening to {speaker}, mouth closed"
    if eyeline and eyeline.lower() in ("window", "door", "down"):
        return "quiet, turned away"
    return "a natural, unposed expression, mouth closed"


def _composition_words(meta: dict[str, Any]) -> str:
    comp = meta.get("composition")
    if not isinstance(comp, dict) or not comp.get("subject"):
        return ""
    third = str(comp.get("third") or "centre")
    if third == "centre":
        return ""
    subject = comp["subject"]
    look = _facing_words(meta, subject).strip().rstrip(",")
    if "profile" in look:
        look = look.replace("in profile, ", "in profile ")
    room = (f"with room in front of {subject} on the {'right' if third == 'left' else 'left'}"
            if comp.get("look_room") else "with the other edge clear")
    return _COMPOSITION.format(subject=subject, third=third, look=look.lower() if look.isupper() else look, room=room)


def _shot_words(meta: dict[str, Any], boxes: dict[str, tuple[float, float, float, float]] | None) -> str:
    """The lead sentence: what size of shot this is and who is in the foreground."""
    framing = meta.get("framing") or {}
    present = meta.get("actors")
    bits = []
    in_frame = framing.get("subject") in present if isinstance(present, dict) else bool(framing.get("subject"))
    if framing.get("size_name") and in_frame:
        d = framing.get("distance_m")
        bits.append(f"This is a {str(framing['size_name']).upper()} on {framing['subject']}"
                    + (f", {float(d):.1f} m from the camera" if d else "") + ".")
    if boxes and len(boxes) > 1:
        tall = sorted(boxes.items(), key=lambda kv: -(kv[1][3] - kv[1][1]))
        (big, bb), (small, sb) = tall[0], tall[-1]
        if (bb[3] - bb[1]) > 1.3 * (sb[3] - sb[1]) and bb[1] <= 0.02:
            bits.append(f"{big} is in the FOREGROUND, much closer to the camera than {small}, large and partly "
                        f"out of frame, over {big}'s shoulder at {small}.")
    return (" ".join(bits) + " ") if bits else ""


def _place_words(box: tuple[float, float, float, float] | None) -> str:
    """A box in the frame, as the words a DP would use: where, how big, what the edges cut."""
    if box is None:
        return ""
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2
    wf, hf = x1 - x0, y1 - y0
    where = "on the left of the frame" if cx < 0.36 else "on the right of the frame" if cx > 0.64 else "in the middle of the frame"
    cut = []
    if y0 <= 0.02:
        cut.append("the top edge")
    if y1 >= 0.98:
        cut.append("the bottom edge")
    if x0 <= 0.02:
        cut.append("the left edge")
    if x1 >= 0.98:
        cut.append("the right edge")
    size = (f" The figure is {where}, {wf * 100:.0f}% of the frame wide and {hf * 100:.0f}% of the frame tall"
            + (f", cut off by {' and '.join(cut)}" if cut else ", seen in full"))
    if hf >= 0.9 and y0 <= 0.02 and wf >= 0.45:
        size += (": very close to the camera, their back, shoulder and head FILLING that half of the frame "
                 "from top to bottom, the top of the head cut off by the frame edge")
    elif hf >= 0.7:
        size += ": close to the camera"
    elif hf < 0.35:
        size += ": far from the camera, small in the frame"
    return size + "; keep them exactly that size and place in the picture."


def _light_words(meta: dict[str, Any], scene_light: str, light: dict[str, Any] | None = None) -> str:
    """The light sentence: the director's own words if given, else the
    scene's sun and clock (`light`, the same for every frame of a scene),
    else this frame's own sun."""
    if scene_light.strip():
        return scene_light.strip().rstrip(".") + "."
    return light_words_for(light if light is not None else frame_light(meta))


def frame_light(meta: dict[str, Any]) -> dict[str, Any] | None:
    """The animatic frame's sun with its clock (to the minute: the hour a
    scene plays at is one hour), as `light_words_for` reads it. None when
    the frame carries no sun."""
    sun = meta.get("sun")
    if not isinstance(sun, dict):
        return None
    out = dict(sun)
    clock = meta.get("clock")
    if clock and "clock" not in out:
        out["clock"] = str(clock)[:5]
    return out


def _compass(azimuth_deg: float) -> str:
    """An azimuth from true north, clockwise, as the eight-point word a DP uses."""
    names = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    return names[int(round(float(azimuth_deg) / 45.0)) % 8]


_DAY_WINDOWS = ("Every window in this frame shows a pale daylight sky and the real outside, bright against the "
                "room, never black glass or a reflection of a lit room; the ceiling lights are off.")


def light_words_for(light: dict[str, Any] | None, *, rendered_sun: bool = True) -> str:
    """The light, as the sentence that follows "The light" in a prompt, from
    the numbers the ephemeris gives and nothing the model said.

    Takes both shapes the product uses: the animatic's `sun` dict
    (`direct_sun`, `windows_lit`, `compass`, `elevation_deg`) and the world's
    `light` dict (`azimuth_deg`, `elevation_deg`, `clock`, `date`,
    `windows_lit`). Window indices are 0-based in and 1-based in the prose,
    as the daylight module numbers them. When `direct_sun` is not given,
    direct sun is read off `windows_lit`: a sun that passes no window is soft
    daylight. `rendered_sun` says whether the render already drew the sunlit
    patches (the animatic does; a bare view of the twin does not), so the
    model is told to keep them or to put them where that sun falls.
    """
    if not light:
        return "is the room's own light as the render shows it, with the bright and dark areas in the same places."
    wins = light.get("windows_lit") or []
    direct = bool(light.get("direct_sun")) if "direct_sun" in light else bool(wins)
    az = light.get("azimuth_deg")
    el = light.get("elevation_deg")
    compass = light.get("compass") or (_compass(az) if az is not None else "")
    when = ""
    if light.get("clock"):
        when += f" at {light['clock']}"
    if light.get("date"):
        when += f" on {light['date']}"
    if el is not None and float(el) <= 0 and not direct:
        return (f"is after sunset{when}: no sun and no daylight on the surfaces; the room is lit by its own "
                "lamps, and the windows show a dark blue sky.")
    if direct:
        where = f" through window {', '.join(str(int(w) + 1) for w in wins)}" if wins else " through the window"
        az_words = f" (azimuth {float(az):.0f} degrees)" if az is not None else ""
        patches = ("where the render already shows them; if the window that passes the sun is not in this "
                   "frame, the sun comes in from outside the frame and only its light shows"
                   if rendered_sun
                   else "where that sun falls through those windows into this view, and only if that window "
                        "or the surfaces it lights are in this frame")
        return (f"is real daylight{when}: direct sun{where} from the {compass}{az_words} at "
                f"{float(el or 0):.0f} degrees elevation, with warm sunlit patches on the "
                f"floor and walls {patches}, and soft daylight everywhere else. {_DAY_WINDOWS}").replace("the  at", "the at")
    sun_pos = ""
    if az is not None and el is not None:
        sun_pos = f" (the sun is in the {compass} at {float(el):.0f} degrees elevation but passes no window)"
    return f"is soft daylight{when} from the windows with no direct sun{sun_pos}, as in the render. {_DAY_WINDOWS}"


def build_sequence_prompt(
    *,
    meta: dict[str, Any],
    cast: list[Any],
    headshot_index: dict[str, int],
    continuity_index: int | None = None,
    scene_light: str = "",
    mood: str = "",
    boxes: dict[str, tuple[float, float, float, float]] | None = None,
    room_index: list[int] | None = None,
    anchor_index: int | None = None,
    light: dict[str, Any] | None = None,
    layout: str = "",
    photo_light: dict[str, Any] | None = None,
    same_setup: bool = False,
) -> str:
    """The instruction for one frame of a sequence. `meta` is the animatic or
    evaluation frame (focal_mm, actors, sun, framing, staging, composition);
    `cast` the Actor list; `headshot_index` maps an actor's name to the
    1-based image number of the headshot in the references; `continuity_index`
    the image number of the nearest finished frame, when one rides along;
    `anchor_index` that of the scene's first finished frame, when it is a
    different image (the same image is described once, as both); `light` the
    scene's sun and clock, said the same for every frame; `layout` the
    sentence `layout_words` made for this camera; `photo_light` what the
    scan's photographs were shot under (`classify_photo_light`); `same_setup`
    that the continuity frame is from this very camera."""
    cam = meta.get("camera") or {}
    focal = float(meta.get("focal_mm") or cam.get("focal_mm") or 35.0)
    count, people, nobody = _people_words(meta, cast, headshot_index, boxes)
    present = meta.get("actors")
    names = list(present.keys()) if isinstance(present, dict) else None
    if boxes and names is not None:
        names = [n for n in names if n in boxes]
    if anchor_index and continuity_index and anchor_index == continuity_index:
        cont = (_SAME_SETUP_AND_ANCHOR if same_setup else _SCENE_ANCHOR_AND_PREVIOUS).format(n=anchor_index)
    else:
        if continuity_index and same_setup:
            cont = _SAME_SETUP.format(n=continuity_index)
        else:
            cont = _CONTINUITY.format(n=continuity_index, near=" from the camera nearest this one" if anchor_index else "") if continuity_index else ""
        if anchor_index:
            cont = _SCENE_ANCHOR.format(n=anchor_index) + cont
    # The staging may say "at the window" about a window this camera does
    # not see: the painter is told so, or it paints one.
    if layout and "No window is in this frame" in layout and isinstance(meta.get("staging"), dict):
        for who, st in meta["staging"].items():
            words = " ".join(str(st.get(k) or "") for k in ("business", "eyeline")).lower() if isinstance(st, dict) else ""
            if "window" in words and (names is None or who in names):
                people += f"The window {who} is at or looks toward is OUTSIDE this frame: paint no window for it. "
    text = _SEQUENCE_INSTRUCTION.format(
        shot=_shot_words(meta, boxes) + _composition_words(meta), count=count, room=_room_words(room_index),
        layout=(layout.strip() + " ") if layout and layout.strip() else "",
        photo_light=photo_light_words(photo_light, room_index),
        focal_mm=focal, people=people, nobody=nobody, light=_light_words(meta, scene_light, light),
        mood=(" " + mood.strip().rstrip(".") + ".") if mood.strip() else "", continuity=cont,
    )
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# what is in the frame, and what the photographs were shot under
# ---------------------------------------------------------------------------

_SENSOR_MM = {"super35": (24.89, 14.0), "super35-4perf": (24.89, 18.66), "fullframe": (36.0, 24.0),
              "m43": (17.3, 13.0), "phone-main": (9.8, 7.35)}
_LAYOUT_SKIP = ("floor", "carpet", "wall", "ceiling", "rug", "corridor", "hallway")


def layout_words(meta: dict[str, Any], things: list[tuple[str, Any]], openings: list[Any] | None = None,
                 *, max_absent: int = 6) -> str:
    """The sentence that says what this camera sees, from the twin's own
    labels and window openings projected through the frame's camera (pos,
    aim, focal_mm, sensor): each thing in frame by its third and distance,
    left to right; which windows are in frame, or that none is and where the
    nearest one lies instead; and which labelled things are NOT in frame.
    `things` are (name, xyz); `openings` have `center`, `width`, `height` and
    `kind` (attributes or keys). Empty when the frame has no camera."""
    import math

    cam = meta.get("camera") or {}
    pos = cam.get("pos") or meta.get("cam")
    aim = cam.get("aim") or meta.get("aim")
    if pos is None or aim is None:
        return ""
    try:
        px, py, pz = (float(v) for v in pos)
        ax, ay, az = (float(v) for v in aim)
    except (TypeError, ValueError):
        return ""
    focal = float(meta.get("focal_mm") or cam.get("focal_mm") or 35.0)
    sw, sh = _SENSOR_MM.get(str(meta.get("sensor") or cam.get("sensor") or "super35"), _SENSOR_MM["super35"])
    fx, fy, fz = ax - px, ay - py, az - pz
    n = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
    fx, fy, fz = fx / n, fy / n, fz / n
    rx, ry, rz = fy, -fx, 0.0                      # fwd x up(z)
    rn = math.hypot(rx, ry) or 1.0
    rx, ry = rx / rn, ry / rn
    ux, uy, uz = ry * fz - rz * fy, rz * fx - rx * fz, rx * fy - ry * fx   # right x fwd

    def project(p) -> tuple[float, float, float]:
        dx, dy, dz = float(p[0]) - px, float(p[1]) - py, float(p[2]) - pz
        z = dx * fx + dy * fy + dz * fz
        if z < 0.25:
            return (0.0, 0.0, z)
        u = (dx * rx + dy * ry + dz * rz) / z * focal / (sw / 2)
        v = (dx * ux + dy * uy + dz * uz) / z * focal / (sh / 2)
        return (u, v, z)

    def get(o, key):
        return o.get(key) if isinstance(o, dict) else getattr(o, key, None)

    present: list[tuple[float, str]] = []
    present_names: set[str] = set()
    absent: list[str] = []
    for name, xyz in things:
        name = str(name).strip()
        low = name.lower()
        if not name or any(w in low for w in _LAYOUT_SKIP):
            continue
        u, v, z = project(xyz)
        if z >= 0.25 and abs(u) <= 1.05 and abs(v) <= 1.1:
            third = "on the left" if u < -1 / 3 else "on the right" if u > 1 / 3 else "in the centre"
            article = "another" if low in present_names else "the"
            present.append((u, f"{article} {name} {third}, {z:.1f} m away"))
            present_names.add(low)
        else:
            absent.append(name)
    # a name is absent only when no thing of that name is in frame (two desks)
    absent = [n for k, n in enumerate(absent) if n.lower() not in present_names and n.lower() not in {m.lower() for m in absent[:k]}]
    wins_in: list[tuple[float, str]] = []
    wins_out: list[tuple[float, str]] = []
    for k, o in enumerate(openings or []):
        if str(get(o, "kind") or "window") != "window":
            continue
        c = get(o, "center")
        if c is None:
            continue
        w, h = float(get(o, "width") or 0), float(get(o, "height") or 0)
        u, v, z = project(c)
        label = f"window {k + 1} ({w:.1f} x {h:.1f} m)"
        if z >= 0.25 and abs(u) <= 1.05 and abs(v) <= 1.1:
            third = "on the left" if u < -1 / 3 else "on the right" if u > 1 / 3 else "in the centre"
            wins_in.append((u, f"{label} {third}, {z:.1f} m away"))
        else:
            dx, dy = float(c[0]) - px, float(c[1]) - py
            side = (dx * rx + dy * ry)
            where = "behind the camera" if (dx * fx + dy * fy) < 0 else ("off frame to the left" if side < 0 else "off frame to the right")
            wins_out.append((abs(side), f"window {k + 1} is {where}"))
    present.sort(key=lambda t: t[0])
    wins_in.sort(key=lambda t: t[0])
    items = [t for _u, t in present] + [t for _u, t in wins_in]
    if not items and not absent and not wins_out:
        return ""
    present_words = ", ".join(items) if items else "no labelled furniture and no window; only the walls, floor and ceiling"
    if wins_in:
        windows = f"The only window{'s' if len(wins_in) > 1 else ''} in this frame: {', '.join(t for _u, t in wins_in)}."
    else:
        windows = "No window is in this frame" + (f" ({'; '.join(t for _s, t in sorted(wins_out)[:2])})" if wins_out else "") + "."
    absent_words = ", ".join(absent[:max_absent]) if absent else "nothing else that is labelled"
    return _LAYOUT.format(present=present_words, windows=windows, absent=absent_words).strip()


PHOTO_LIGHT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "time": {"type": "string", "enum": ["day", "night", "dusk"]},
        "source": {"type": "string", "enum": ["daylight", "artificial", "mixed"]},
        "windows": {"type": "string", "enum": ["bright", "black", "curtained", "none"]},
    },
    "required": ["time", "source", "windows"],
}

_PHOTO_LIGHT_ASK = (
    "These are photographs of one room, taken on site. Answer JSON "
    '{"time": "day"|"night"|"dusk", "source": "daylight"|"artificial"|"mixed", "windows": '
    '"bright"|"black"|"curtained"|"none"}: when they were taken, what lights the room in them, '
    "and what the window glass shows (bright = daylight outside, black = night or reflections)."
)


def classify_photo_light(client, photos: list[bytes], *, model: str | None = None) -> dict[str, Any] | None:
    """Muse Spark's one-time reading of what the scan's photographs were shot
    under, so the finish can say how this frame's hour differs from them.
    None when the client cannot chat or the answer is not the schema."""
    if not photos or not hasattr(client, "chat_json"):
        return None
    try:
        ans = client.chat_json(_PHOTO_LIGHT_ASK, list(photos)[:4], model=model, schema=PHOTO_LIGHT_SCHEMA, max_tokens=120)
    except Exception:  # noqa: BLE001 - no reading is no sentence, not a failed frame
        return None
    if not isinstance(ans, dict) or not all(k in ans for k in ("time", "source", "windows")):
        return None
    return {k: str(ans[k]) for k in ("time", "source", "windows")}


def photo_light_words(info: dict[str, Any] | None, room_index: list[int] | None = None) -> str:
    """The sentence that tells the painter the photographs' light is not the
    frame's: empty without a reading or without photographs in the references."""
    if not info or not room_index:
        return ""
    when = {"night": "at night", "day": "by day", "dusk": "at dusk"}.get(str(info.get("time")), "")
    source = {"artificial": "the room's own lamps and ceiling lights", "daylight": "daylight",
              "mixed": "daylight and the room's own lights"}.get(str(info.get("source")), "their own light")
    windows = {"black": "black", "bright": "bright with daylight", "curtained": "curtained",
               "none": "out of view"}.get(str(info.get("windows")), "as they were")
    if not when:
        return ""
    return _PHOTO_LIGHT.format(when=when, source=source, windows=windows)


def _people_words(meta: dict[str, Any], cast: list[Any], headshot_index: dict[str, int],
                  boxes: dict[str, tuple[float, float, float, float]] | None) -> tuple[str, str, str]:
    """The count sentence, the per-person sentences and the nobody sentence
    for a frame: each stand-in named by its colour as the actor it stands for,
    with looks, wardrobe, facing and place, exactly as the sequence prompt
    has always said them. Shared with the world prompt so a set of views
    describes its cast in one voice."""
    present = meta.get("actors")
    names = list(present.keys()) if isinstance(present, dict) else (list(present) if isinstance(present, list) else None)
    # The render is the truth about who is in frame: an actor on the marks
    # but with no stand-in in the picture is out of frame, and saying so is
    # the only thing that stops the painter inventing them.
    out_of_frame: list[str] = []
    if boxes and names is not None:
        out_of_frame = [n for n in names if n not in boxes]
        names = [n for n in names if n in boxes]
    people = []
    for a in cast:
        if names is not None and a.name not in names:
            continue
        about = []
        if getattr(a, "description", ""):
            about.append(a.description.strip().rstrip("."))
        about.append(f"{float(a.height_m):.2f} m tall")
        ref = f" (image {headshot_index[a.name]} is {a.name}'s headshot: the same face)" if a.name in headshot_index else ""
        wardrobe = (getattr(a, "wardrobe", "") or "").strip().rstrip(".") or "plain everyday clothes"
        staged = _staging_words(meta, a.name)
        if staged is not None:
            posture, expression = staged
            people.append(_PERSON_STAGED.format(colour=colour_name(a.colour).upper(), name=a.name, about=", ".join(about),
                                                ref=ref, wardrobe=wardrobe, place=_place_words((boxes or {}).get(a.name)),
                                                facing=_facing_words(meta, a.name), posture=posture, expression=expression))
        else:
            people.append(_PERSON.format(colour=colour_name(a.colour).upper(), name=a.name, about=", ".join(about),
                                         ref=ref, wardrobe=wardrobe, place=_place_words((boxes or {}).get(a.name)),
                                         facing=_facing_words(meta, a.name)))
    nobody = _NOBODY if (names is not None and not names) or (not people and not cast) else ""
    if nobody:
        people = []
    who = [a.name for a in cast if names is None or a.name in names]
    count = (f"There are EXACTLY {len(who)} {'person' if len(who) == 1 else 'people'} in this frame, "
             f"{' and '.join(who)}, and nobody else: no extra body, limb or foot anywhere. ") if who and not nobody else ""
    for n in out_of_frame:
        count += (f"{n} is in the room but OUT OF FRAME: do not paint {n} or any part of them, even if "
                  f"another reference image shows them. ")
    return count, "".join(people), nobody


def _room_words(room_index: list[int] | None, template: str = _ROOM) -> str:
    """The sentence that makes the output THIS room: which images are the
    scan's photographs and what they fix. The world's template says the
    photographs fix the materials and NOT the camera: the live probe showed
    that when the render is a smear the model takes the photograph's
    viewpoint instead of image 1's, and a splat with known poses cannot
    forgive that."""
    if not room_index:
        return ""
    nums = [f"image {n}" for n in room_index]
    images = (nums[0] if len(nums) == 1 else ", ".join(nums[:-1]) + " and " + nums[-1]).capitalize()
    room = template.format(images=images, verb="is a photograph" if len(nums) == 1 else "are")
    return room.replace("is a photograph photographs", "is a photograph")


def strip_caption(png: bytes) -> bytes:
    """Cut the renderer's caption bar (a near-black strip with text, appended
    under the picture) off the bottom before the model sees it, so it is not
    asked to keep a bar it should not paint. Frames without one pass through."""
    from PIL import Image
    import numpy as np

    im = Image.open(io.BytesIO(png)).convert("RGB")
    a = np.asarray(im, dtype=np.int16)
    bar = np.array([14, 16, 19], dtype=np.int16)
    h = a.shape[0]
    n = 0
    edge = max(2, a.shape[1] // 10)
    for y in range(h - 1, -1, -1):
        row = a[y]
        # the bar is its colour at the right edge (the text sits at the left)
        # and over most of the row; the picture above it is neither
        near = np.abs(row - bar).max(axis=1) <= 14
        if near[-edge:].all() and near.mean() >= 0.5:
            n += 1
        else:
            break
    if 8 <= n <= 80 and n < h // 3:
        buf = io.BytesIO()
        im.crop((0, 0, im.width, h - n)).save(buf, format="PNG")
        return buf.getvalue()
    return png


def finish_sequence(
    frames: list[tuple[str, bytes, dict[str, Any]]],
    *,
    client,
    cast: list[Any],
    headshots: dict[str, bytes] | None = None,
    out_dir: str | Path,
    workers: int = 4,
    continuity: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
    scene_light: str = "",
    mood: str = "",
    force: bool = False,
    stamp: bool = True,
    model: str | None = None,
    room_refs: Callable[[dict[str, Any]], list[bytes]] | None = None,
    review: bool = True,
    review_tries: int = 2,
    layout: Callable[[dict[str, Any]], str] | None = None,
    photo_light: dict[str, Any] | None = None,
) -> list[FinishedFrame]:
    """Finish every frame of a sequence; the finished frames, in order.

    `layout(meta)` returns the sentence that says what this frame's camera
    sees (`layout_words`); `photo_light` is what the scan's photographs were
    shot under (`classify_photo_light`): both go into the prompt and the
    review. Frames are finished setup by setup: the scene's anchor first,
    then the first frame of every other setup, then the rest, each later
    frame of a setup carrying that setup's first finished frame as the
    background it must keep.

    With `review`, Muse Spark judges each finished frame beside its render
    (`review_finished`) and a rejected frame is finished again, up to
    `review_tries` more times, with the review's corrections appended to the
    prompt; the last attempt stands either way, and the verdicts are kept in
    the frame's meta under "review".

    `frames` are (name, measured PNG, meta) in sequence order; each finished
    PNG lands at `out_dir/finish_<name>`. A frame whose finished file already
    exists is kept as it is unless `force`. A frame the model fails on is
    reported through `progress` (kind "error") and left out of the result:
    the measured frame stands in for it downstream. `progress` gets dicts
    with a `kind` of start, done, skip or error, the frame's index, name and
    file, and the total.

    `room_refs(meta)` returns up to two photographs of the real room from
    near this frame's camera (the scan's own frames): without them the model
    has only the render's smear to go on and paints a room of its own; with
    them the output is this room.

    Continuity is the world finisher's: the first frame the model finishes
    is the ANCHOR, and every later frame carries it ("the first photograph of
    this scene: match its materials, light, faces, hair and wardrobe") and
    its nearest already-finished frame by camera position and aim (the
    previous frame of the same setup, usually). When the anchor is itself as
    near as any (the camera has not moved since it), it rides once, as both.
    References, in order: the frame, headshots, room photographs, then the
    finished frames, within the API's ten; the finished frames are reserved
    first, so a crowded cast never pushes the anchor out.

    The light is said once for the scene: the first frame's sun and clock
    (`frame_light`) go into every frame's prompt, so every setup is finished
    at the same hour with the sun from the same azimuth; `scene_light` in
    words overrides it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    headshots = dict(headshots or {})
    model = model or model_for("image")
    total = len(frames)
    pos = _view_positions(frames)
    dirs = _view_dirs(frames, pos)
    lock = threading.Lock()
    done: dict[int, FinishedFrame] = {}
    scene_sun = next((frame_light(m) for _n, _p, m in frames if frame_light(m)), None)

    def say(**ev) -> None:
        if progress is not None:
            try:
                progress({"total": total, **ev})
            except Exception:  # noqa: BLE001 - a reporter must not stop the run
                pass

    def shot_of(i: int):
        m = frames[i][2]
        return (m.get("camera") or {}).get("shot") if isinstance(m.get("camera"), dict) else None

    def one(i: int, anchor: FinishedFrame | None, anchor_i: int | None, finished: dict[int, FinishedFrame]) -> FinishedFrame | None:
        name, png, meta = frames[i]
        dest = out_dir / f"finish_{Path(name).name}"
        if dest.suffix.lower() != ".png":
            dest = dest.with_suffix(".png")
        if dest.exists() and not force:
            raw = dest.read_bytes()
            ff = FinishedFrame(png=raw, raw=raw, prompt="", model=model, name=name, path=dest, cached=True)
            say(kind="skip", index=i, name=name, file=dest.name, text=f"{name}: already finished")
            return ff
        say(kind="start", index=i, name=name, file=dest.name, text=f"finishing {name}")
        try:
            present = meta.get("actors")
            names = list(present.keys()) if isinstance(present, dict) else None
            refs: list[bytes] = [strip_caption(png)]
            boxes = figure_boxes(refs[0], cast)
            # The finished frames this one is held to, decided first so their
            # slots are reserved: the nearest finished frame by camera (ties to
            # the most recent), and the anchor when it is a different frame
            # and the camera has moved since it.
            tail: list[tuple[str, FinishedFrame]] = []
            same_setup = False
            if continuity and anchor is not None and anchor_i != i:
                near = nearest_finished(i, finished, pos, dirs, k=1, prefer_recent=True)
                j = near[0] if near else anchor_i
                # the same setup: the same shot number from the same camera
                same_setup = (shot_of(j) is not None and shot_of(j) == shot_of(i)
                              and _nearness(i, j, pos, dirs) < 0.05)
                if j != anchor_i and shot_of(anchor_i) == shot_of(i) and _nearness(i, anchor_i, pos, dirs) < 0.05 \
                        and _nearness(i, anchor_i, pos, dirs) <= _nearness(i, j, pos, dirs) + 1e-9:
                    same_setup = True
                if j == anchor_i or (not same_setup and _nearness(i, anchor_i, pos, dirs) <= _nearness(i, j, pos, dirs) + 1e-9):
                    tail.append(("both", anchor))
                else:
                    tail.append(("anchor", anchor))
                    tail.append(("neighbour", finished[j]))
            budget = 10 - len(tail)
            index: dict[str, int] = {}
            for a in cast:
                if names is not None and a.name not in names:
                    continue
                if boxes and a.name not in boxes:
                    continue                      # out of frame: no headshot needed
                shot = headshots.get(a.name) or headshots.get(getattr(a, "id", ""))
                if shot and len(refs) < budget:
                    refs.append(shot)
                    index[a.name] = len(refs)
            room_index: list[int] = []
            photos: list[bytes] = []
            if room_refs is not None:
                try:
                    for shot in list(room_refs(meta) or [])[:2]:
                        if shot and len(refs) < budget:
                            refs.append(shot)
                            room_index.append(len(refs))
                            photos.append(shot)
                except Exception as exc:  # noqa: BLE001 - no room photos is a worse frame, not a failed one
                    say(kind="note", index=i, name=name, text=f"{name}: no room photographs ({exc})")
            anchor_index = cont_index = None
            for role, ff_ref in tail:
                refs.append(ff_ref.raw or ff_ref.png)
                if role == "both":
                    anchor_index = cont_index = len(refs)
                elif role == "anchor":
                    anchor_index = len(refs)
                else:
                    cont_index = len(refs)
            layout_text = ""
            if layout is not None:
                try:
                    layout_text = str(layout(meta) or "")
                except Exception as exc:  # noqa: BLE001 - no layout is a weaker prompt, not a failed frame
                    say(kind="note", index=i, name=name, text=f"{name}: no layout ({exc})")
            prompt = build_sequence_prompt(meta=meta, cast=cast, headshot_index=index, continuity_index=cont_index,
                                           scene_light=scene_light, mood=mood, boxes=boxes, room_index=room_index,
                                           anchor_index=anchor_index, light=scene_sun, layout=layout_text,
                                           photo_light=photo_light, same_setup=same_setup)
            light_text = _light_words(meta, scene_light, scene_sun)
            raw = crop_to_aspect(client.generate_image(prompt, references=refs, model=model), _image_size(refs[0]))
            verdicts: list[dict[str, Any]] = []
            attempts: list[tuple[int, bytes]] = []       # (problem count, picture), the best one is kept
            attempt = 0
            can_review = review and hasattr(client, "chat_json")
            while can_review:
                try:
                    ok, problems = review_finished(client, raw, refs[0], meta, cast, photos=photos,
                                                   anchor=(anchor.raw or anchor.png) if anchor is not None and anchor_i != i else None,
                                                   boxes=boxes, light_text=light_text, layout=layout_text)
                except Exception as exc:  # noqa: BLE001 - a review that fails is no verdict, not a failed frame
                    say(kind="note", index=i, name=name, text=f"{name}: the review did not answer ({exc})")
                    break
                verdicts.append({"ok": ok, "problems": problems})
                attempts.append((0 if ok else max(1, len(problems)), raw))
                if ok or attempt >= max(0, int(review_tries)):
                    break
                attempt += 1
                say(kind="note", index=i, name=name,
                    text=f"{name}: Muse Spark sent it back ({'; '.join(problems)[:160]}), finishing again")
                prompt = prompt + correction_words(problems)
                raw = crop_to_aspect(client.generate_image(prompt, references=refs, model=model), _image_size(refs[0]))
            if attempts:
                best = min(range(len(attempts)), key=lambda k: (attempts[k][0], -k))
                raw = attempts[best][1]
                if attempts[best][0] and len(attempts) > 1:
                    say(kind="note", index=i, name=name,
                        text=f"{name}: no attempt passed; keeping attempt {best + 1} of {len(attempts)} "
                             f"({attempts[best][0]} problem{'s' if attempts[best][0] != 1 else ''})")
            out = stamp_render(raw) if stamp else raw
            ff = FinishedFrame(png=out, prompt=prompt, model=model, references=len(refs) - 1, name=name, path=dest,
                               raw=raw, meta={"headshots": len(index), "continuity": cont_index is not None,
                                              "anchor": anchor_index is not None, "room_photos": len(room_index),
                                              "same_setup": same_setup, "layout": layout_text,
                                              "review": {"rejected": attempt, "verdicts": verdicts}})
            ff.save(dest)
            try:
                dest.with_suffix(".json").write_text(json.dumps({"prompt": prompt, "review": ff.meta["review"],
                                                                 "layout": layout_text, "light": light_text}, indent=1))
            except OSError:
                pass
            say(kind="done", index=i, name=name, file=dest.name, text=f"finished {name}")
            return ff
        except Exception as exc:  # noqa: BLE001 - one frame failing is not the run failing
            say(kind="error", index=i, name=name, file=dest.name, text=f"{name}: {type(exc).__name__}: {exc}")
            return None

    if not frames:
        return []
    # The anchor goes alone: the first frame the model finishes (a cached
    # file counts: it is the look the scene already has). Then chunks of
    # `workers` run in parallel, each frame carrying the anchor and its
    # nearest finished frame from before its chunk started.
    workers = max(1, int(workers))
    anchor: FinishedFrame | None = None
    anchor_i: int | None = None
    i = 0
    while i < total and anchor is None:
        r = one(i, None, None, dict(done))
        if r is not None:
            done[i] = r
            anchor, anchor_i = r, i
        i += 1
    # Setup by setup: the first frame of every other setup goes next (each
    # carrying the anchor), then the rest, each carrying its own setup's
    # first finished frame as the background it keeps.
    rest = [k for k in range(i, total)]
    firsts: list[int] = []
    seen_shots: set = set([shot_of(anchor_i)] if anchor_i is not None else [])
    for k in rest:
        sh = shot_of(k)
        if sh is None or sh in seen_shots:
            continue
        seen_shots.add(sh)
        firsts.append(k)
    others = [k for k in rest if k not in firsts]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for group in (firsts, others):
            g = 0
            while g < len(group):
                chunk = group[g: g + workers]
                snapshot = dict(done)
                results = list(pool.map(lambda k: one(k, anchor, anchor_i, snapshot), chunk))
                with lock:
                    for k, r in zip(chunk, results):
                        if r is not None:
                            done[k] = r
                g += len(chunk)
    return [done[k] for k in sorted(done)]


# ---------------------------------------------------------------------------
# a set of views that agree: the world
# ---------------------------------------------------------------------------

# The world prompt is the sequence prompt with one more demand: this picture
# is one of a set, and the set must agree. A splat trained on sixty finished
# views forgives nothing that differs between them; a wall that is cream in
# one view and grey in the next becomes a smear in the world. So every view
# after the first carries the first finished view as an anchor and its nearest
# finished neighbour, and the instruction says plainly what those are for.
_WORLD_INSTRUCTION = (
    "Photoreal photograph of a real room. Image 1 is a rough, blurred 3D render of the room{figures_note}: "
    "it fixes the camera and everything's place, and the output is image 1 resolved into a sharp "
    "photograph, not a different picture. This picture is ONE OF A SET of photographs of the same "
    "room at the same moment, taken from different positions: the set must agree on every surface, "
    "object, light and person, so what is painted here must be exactly what the other photographs "
    "of the set show, seen from this position. Re-render image 1 as a photograph taken from EXACTLY "
    "the same camera: same position and height, same {focal_mm:.0f} mm lens, same framing and "
    "perspective, same picture edges; the walls, floor, ceiling, windows and every piece of furniture "
    "stay exactly where they are in image 1 and the same size. Do not move, add or remove anything. "
    "The output is the SAME CROP as image 1: do not widen the shot, do not step back, do not show "
    "more of the room or more of anyone than image 1 shows; where image 1 is close up on a bed, a "
    "wall or a shelf, the output is just as close up on that same thing, filling the frame the same "
    "way. Where image 1 is black or empty the room was not measured there, not dark: fill it with "
    "the real surface the photographs show at that place. No new windows, doors, lamps, trees, "
    "posters, furniture or lit corridor that the photographs do not show. "
    "{anchor}{neighbour}{room}{layout}{dressing}{count}{people}{nobody}"
    "Make every surface a real material: the real walls, floor, furniture and fabrics with texture "
    "and grain, sharp and clean, not blurred. {photo_light}The light {light} "
    "Natural film look, correct skin, correct hands. No text, no caption bar, no stamp, no label, "
    "no watermark, no extra people, no border."
)

_ANCHOR = (
    "Image {n} is the first finished photograph of this set: another photograph of this same room "
    "at this same moment from its own, different camera, with the same materials, the same light "
    "and the same people in the same clothes. Use it only to match colours, materials, light and "
    "the hour. Do NOT reproduce its composition or anything it shows that image 1 does not: if "
    "image 1 shows a bed and image {n} a wardrobe, paint the bed. "
)

_NEIGHBOUR = (
    "Image {n} is a finished photograph of this set taken from a camera near this one, looking at "
    "much of the same part of the room: every surface, object and patch of light it shows must "
    "appear here identically where it is in view, from this camera, not its; what it shows that "
    "image 1's camera cannot see is not painted here. "
)

_WORLD_ROOM = (
    "{images} {verb} photographs of THIS EXACT ROOM taken on site from OTHER camera positions: they "
    "fix what the room is made of (the same walls, floor, ceiling, furniture, bedding, objects, "
    "clutter, colours and materials, and the same window openings in the same places, with this "
    "hour's sky in them) and NOT where the camera is. Do not copy their framing, their viewpoint "
    "or how much of the room they show; the output is image 1's camera and image 1's crop, with "
    "image 1's blurred surfaces resolved into those real materials. "
)

_DRESSING = "The room is dressed as: {dressing}. Every photograph of the set shows this same dressing. "


def resize_to(png: bytes, size: tuple[int, int]) -> bytes:
    """The picture at exactly `size` (width, height), as PNG. The model answers
    at a size of its own; a splat trained on the views is told their exact
    pixel size, so the finished picture must be the measured view's size and
    no other. A picture already that size passes through re-encoded."""
    from PIL import Image

    im = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = int(size[0]), int(size[1])
    if im.size != (w, h):
        im = im.resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def crop_to_aspect(png: bytes, size: tuple[int, int]) -> bytes:
    """The picture centre-cropped to the aspect of `size` (width, height),
    at its own resolution: the model answers at an aspect of its own, and a
    frame that is wider or taller than the render has shown more of the room
    than the render did. A picture already that aspect passes through."""
    from PIL import Image

    im = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = im.size
    want = float(size[0]) / float(size[1])
    have = w / h
    if abs(have - want) < 0.01:
        return png
    if have > want:
        nw = int(round(h * want))
        x0 = (w - nw) // 2
        im = im.crop((x0, 0, x0 + nw, h))
    else:
        nh = int(round(w / want))
        y0 = (h - nh) // 2
        im = im.crop((0, y0, w, y0 + nh))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _image_size(png: bytes) -> tuple[int, int]:
    from PIL import Image

    return Image.open(io.BytesIO(png)).size


def build_world_prompt(
    *,
    meta: dict[str, Any],
    cast: list[Any],
    headshot_index: dict[str, int],
    anchor_index: int | None = None,
    neighbour_index: int | None = None,
    room_index: list[int] | None = None,
    light: dict[str, Any] | None = None,
    dressing: str = "",
    boxes: dict[str, tuple[float, float, float, float]] | None = None,
    layout: str = "",
    photo_light: dict[str, Any] | None = None,
) -> str:
    """The instruction for one view of a world. `anchor_index` is the image
    number of the first finished view, `neighbour_index` that of the nearest
    finished neighbour; `light` is the ephemeris dict for the hour (or None
    for the room's own light) and `dressing` the dressing note."""
    cam = meta.get("camera") or {}
    focal = float(meta.get("focal_mm") or cam.get("focal_mm") or 24.0)
    count, people, nobody = _people_words(meta, cast, headshot_index, boxes)
    figures_note = " with flat coloured stand-in figures" if people else ""
    dressing = (dressing or meta.get("dressing") or "").strip().rstrip(".")
    text = _WORLD_INSTRUCTION.format(
        figures_note=figures_note, focal_mm=focal,
        anchor=_ANCHOR.format(n=anchor_index) if anchor_index else "",
        neighbour=_NEIGHBOUR.format(n=neighbour_index) if neighbour_index else "",
        room=_room_words(room_index, _WORLD_ROOM),
        layout=(layout.strip() + " ") if layout and layout.strip() else "",
        photo_light=photo_light_words(photo_light, room_index),
        dressing=_DRESSING.format(dressing=dressing) if dressing else "",
        count=count, people=people, nobody=nobody,
        light=light_words_for(light if light is not None else meta.get("light"), rendered_sun=False),
    )
    return " ".join(text.split())


def _view_positions(views: list[tuple[str, bytes, dict[str, Any]]]) -> list[tuple[float, float, float] | None]:
    out: list[tuple[float, float, float] | None] = []
    for _name, _png, meta in views:
        pos = (meta.get("camera") or {}).get("pos") or meta.get("cam")
        try:
            out.append((float(pos[0]), float(pos[1]), float(pos[2])))
        except (TypeError, ValueError, IndexError):
            out.append(None)
    return out


def _dist(a: tuple[float, float, float] | None, b: tuple[float, float, float] | None) -> float:
    if a is None or b is None:
        return float("inf")
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _view_dirs(views: list[tuple[str, bytes, dict[str, Any]]],
               pos: list[tuple[float, float, float] | None]) -> list[tuple[float, float, float] | None]:
    out: list[tuple[float, float, float] | None] = []
    for (_n, _p, meta), p in zip(views, pos):
        aim = (meta.get("camera") or {}).get("aim") or meta.get("aim")
        try:
            d = (float(aim[0]) - p[0], float(aim[1]) - p[1], float(aim[2]) - p[2])
            n = (d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 0.5
            out.append((d[0] / n, d[1] / n, d[2] / n) if n > 1e-9 else None)
        except (TypeError, ValueError, IndexError):
            out.append(None)
    return out


# A camera 0.3 m away looking the same way shares more of its picture than
# one 0.1 m away looking the other way; this is what a turn of the aim costs
# in metres of distance when two finished views are near the same spot. Small
# enough that position still decides, as the contract says, when it differs.
_AIM_TURN_M = 0.3


def _nearness(i: int, j: int, pos, dirs) -> float:
    d = _dist(pos[i], pos[j])
    if dirs[i] is not None and dirs[j] is not None:
        cos = dirs[i][0] * dirs[j][0] + dirs[i][1] * dirs[j][1] + dirs[i][2] * dirs[j][2]
        d += _AIM_TURN_M * (1.0 - cos)
    return d


def nearest_finished(i: int, finished: dict[int, Any], pos, dirs, *, k: int = 2, prefer_recent: bool = False) -> list[int]:
    """The indices of the `k` finished views nearest view `i` by camera
    position and aim (`_nearness`), nearest first. `prefer_recent` breaks
    ties toward the most recently finished (the highest index), which for a
    held camera is the previous frame. Views with no camera are never near."""
    cands = [j for j in finished if j != i and pos[j] is not None]
    if prefer_recent:
        cands.sort(key=lambda j: (_nearness(i, j, pos, dirs), -j))
    else:
        cands.sort(key=lambda j: _nearness(i, j, pos, dirs))
    return cands[:k]


def world_order(views: list[tuple[str, bytes, dict[str, Any]]]) -> list[int]:
    """The order to finish a set of views in: view 0 first, then always the
    unfinished view whose camera stands nearest the one just finished. Views
    with no camera position come last, in their own order. Greedy, so every
    view has a finished neighbour close by when its turn comes."""
    n = len(views)
    if n == 0:
        return []
    pos = _view_positions(views)
    dirs = _view_dirs(views, pos)
    order = [0]
    left = [i for i in range(1, n) if pos[i] is not None]
    cur = 0
    while left:
        k = min(left, key=lambda i: _nearness(cur, i, pos, dirs))
        order.append(k)
        left.remove(k)
        cur = k
    order.extend(i for i in range(1, n) if pos[i] is None)
    return order


def finish_world_views(
    views: list[tuple[str, bytes, dict[str, Any]]],
    *,
    client,
    cast: list[Any],
    headshots: dict[str, bytes] | None = None,
    room_refs: Callable[[dict[str, Any]], list[bytes]] | None = None,
    light: dict[str, Any] | None = None,
    dressing: str = "",
    out_dir: str | Path,
    workers: int = 4,
    force: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
    model: str | None = None,
    layout: Callable[[dict[str, Any]], str] | None = None,
    photo_light: dict[str, Any] | None = None,
    review: bool = False,
) -> list[FinishedFrame]:
    """Finish a set of views of one room so they agree; the finished views, in
    the order the views were given. `layout(meta)` and `photo_light` are the
    sequence finisher's (what this camera sees; what the photographs were
    shot under); with `review`, Muse Spark judges each view beside its render
    and the photographs and the view is painted once more on a rejection.

    `views` are (name, measured PNG, meta) with meta carrying `camera` (pos,
    aim, focal_mm), `actors`, and optionally `light` and `dressing`; `light`
    and `dressing` given here win over the meta's. Each finished PNG lands at
    `out_dir/finish_<name>.png`, UNSTAMPED and resized to exactly the measured
    view's size: these pictures train a splat, and a splat must not learn a
    stamp; the viewer page carries the "render" label instead. A finished file
    already on disk is kept unless `force`.

    The first view is finished alone and becomes the ANCHOR, carried into
    every later call as "another photograph of this same room at this same
    moment". The rest are finished in a greedy nearest-neighbour order from
    view 0 (`world_order`), in chunks of `workers`; each carries, besides the
    anchor, its nearest already-finished neighbour by camera position, then
    up to two of the scan's photographs of the view from `room_refs(meta)`,
    then headshots, within the API's ten-image limit in that priority. The
    finisher writes `meta["neighbours"]`, the names of the two nearest
    finished views, before it builds the prompt.

    A view the model fails on is reported through `progress` (kind "error")
    and left out of the result: the caller puts the measured view in its
    place, so the set stays complete for the splat. `progress` gets dicts
    with a `kind` of start, done, skip, note or error, the view's index, name
    and file, and the total, as `finish_sequence` does.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    headshots = dict(headshots or {})
    model = model or model_for("image")
    total = len(views)
    pos = _view_positions(views)
    dirs = _view_dirs(views, pos)
    lock = threading.Lock()
    done: dict[int, FinishedFrame] = {}

    def say(**ev) -> None:
        if progress is not None:
            try:
                progress({"total": total, **ev})
            except Exception:  # noqa: BLE001 - a reporter must not stop the run
                pass

    def dest_of(name: str) -> Path:
        dest = out_dir / f"finish_{Path(name).name}"
        return dest if dest.suffix.lower() == ".png" else dest.with_suffix(".png")

    def one(i: int, anchor: FinishedFrame | None, anchor_i: int | None, finished: dict[int, FinishedFrame]) -> FinishedFrame | None:
        name, png, meta = views[i]
        dest = dest_of(name)
        if dest.exists() and not force:
            raw = dest.read_bytes()
            ff = FinishedFrame(png=raw, raw=raw, prompt="", model=model, name=name, path=dest, cached=True)
            say(kind="skip", index=i, name=name, file=dest.name, text=f"{name}: already finished")
            return ff
        say(kind="start", index=i, name=name, file=dest.name, text=f"finishing {name}")
        try:
            size = _image_size(png)
            refs: list[bytes] = [png]
            boxes = figure_boxes(png, cast)
            near = nearest_finished(i, finished, pos, dirs, k=2)
            meta["neighbours"] = [views[j][0] for j in near]
            anchor_index = neighbour_index = None
            if anchor is not None and anchor_i != i:
                refs.append(anchor.raw or anchor.png)
                anchor_index = len(refs)
            for j in near:
                if j != anchor_i and finished.get(j) is not None and len(refs) < 10:
                    refs.append(finished[j].raw or finished[j].png)
                    neighbour_index = len(refs)
                    break
            room_index: list[int] = []
            if room_refs is not None:
                try:
                    for shot in list(room_refs(meta) or [])[:2]:
                        if shot and len(refs) < 10:
                            refs.append(shot)
                            room_index.append(len(refs))
                except Exception as exc:  # noqa: BLE001 - no room photos is a worse view, not a failed one
                    say(kind="note", index=i, name=name, text=f"{name}: no room photographs ({exc})")
            present = meta.get("actors")
            names = list(present.keys()) if isinstance(present, dict) else None
            index: dict[str, int] = {}
            for a in cast:
                if names is not None and a.name not in names:
                    continue
                if boxes and a.name not in boxes:
                    continue                      # out of frame: no headshot needed
                shot = headshots.get(a.name) or headshots.get(getattr(a, "id", ""))
                if shot and len(refs) < 10:
                    refs.append(shot)
                    index[a.name] = len(refs)
            layout_text = ""
            if layout is not None:
                try:
                    layout_text = str(layout(meta) or "")
                except Exception as exc:  # noqa: BLE001
                    say(kind="note", index=i, name=name, text=f"{name}: no layout ({exc})")
            prompt = build_world_prompt(meta=meta, cast=cast, headshot_index=index, anchor_index=anchor_index,
                                        neighbour_index=neighbour_index, room_index=room_index, light=light,
                                        dressing=dressing, boxes=boxes, layout=layout_text, photo_light=photo_light)
            answer = client.generate_image(prompt, references=refs, model=model)
            if review and hasattr(client, "chat_json"):
                try:
                    photos = [refs[n - 1] for n in room_index]
                    light_text = light_words_for(light if light is not None else meta.get("light"), rendered_sun=False)
                    ok, problems = review_finished(client, crop_to_aspect(answer, size), png, meta, cast, photos=photos,
                                                   anchor=(anchor.raw or anchor.png) if anchor is not None and anchor_i != i else None,
                                                   boxes=boxes, light_text=light_text, layout=layout_text)
                    if not ok:
                        say(kind="note", index=i, name=name,
                            text=f"{name}: Muse Spark sent it back ({'; '.join(problems)[:160]}), finishing again")
                        prompt = prompt + correction_words(problems)
                        answer = client.generate_image(prompt, references=refs, model=model)
                except Exception as exc:  # noqa: BLE001 - a review that fails is no verdict, not a failed view
                    say(kind="note", index=i, name=name, text=f"{name}: the review did not answer ({exc})")
            out = resize_to(answer, size)
            ff = FinishedFrame(png=out, prompt=prompt, model=model, references=len(refs) - 1, name=name, path=dest,
                               raw=out, meta={"headshots": len(index), "anchor": anchor_index is not None,
                                              "neighbour": neighbour_index is not None, "neighbours": list(meta["neighbours"]),
                                              "room_photos": len(room_index), "size": [size[0], size[1]]})
            ff.save(dest)
            say(kind="done", index=i, name=name, file=dest.name, text=f"finished {name}")
            return ff
        except Exception as exc:  # noqa: BLE001 - one view failing is not the run failing
            say(kind="error", index=i, name=name, file=dest.name, text=f"{name}: {type(exc).__name__}: {exc}")
            return None

    if not views:
        return []
    order = world_order(views)
    workers = max(1, int(workers))
    # The anchor goes alone: the first view in the order that the model
    # finishes. If view 0 fails, the next one to finish takes the role, so
    # the set still has one picture everything else is held to.
    anchor: FinishedFrame | None = None
    anchor_i: int | None = None
    k = 0
    while k < len(order) and anchor is None:
        i = order[k]
        k += 1
        r = one(i, None, None, dict(done))
        if r is not None:
            done[i] = r
            anchor, anchor_i = r, i
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while k < len(order):
            chunk = order[k:k + workers]
            snapshot = dict(done)
            results = list(pool.map(lambda i: one(i, anchor, anchor_i, snapshot), chunk))
            with lock:
                for i, r in zip(chunk, results):
                    if r is not None:
                        done[i] = r
            k += len(chunk)
    return [done[i] for i in sorted(done)]
