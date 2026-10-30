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
what they wear, and their headshot when there is one), and from the second
frame on the last finished frame rides along as a reference so the sequence
does not change clothes or light between frames. The first frame goes alone;
the rest run in order with up to `workers` in flight.
"""

from __future__ import annotations

import io
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
    "person cut off by a frame edge in image 1 stays cut off the same way. {room}{count}{people}{nobody}"
    "Make every surface a real material: the real walls, floor, furniture and fabrics "
    "with texture and grain, sharp and clean, not blurred. The light {light}{mood} "
    "{continuity}Natural film look, correct skin, correct hands. No text, no caption bar, "
    "no watermark, no extra people, no border."
)

_PERSON = (
    "The {colour} figure is {name} ({about}){ref}, a real person standing exactly where "
    "the figure is, the same height, {facing}the same pose and the same "
    "distance from the camera, wearing {wardrobe}.{place} "
)

_NOBODY = "There are no people in this frame; leave it empty of people. "

_ROOM = (
    "{images} {verb} photographs of THIS EXACT ROOM taken on site, from near this camera: "
    "the output must show this real room and no other, with the same walls, floor, "
    "ceiling, windows, furniture, bedding, objects, clutter, colours and materials as "
    "those photographs, seen from image 1's camera and at image 1's crop. Image 1 only "
    "fixes the geometry and where the people stand; the photographs fix what the room "
    "looks like. "
)

_CONTINUITY = (
    "Image {n} is the previous finished frame of this scene: match its look, the room's "
    "materials and colours, the lighting, and each person's face, hair and wardrobe EXACTLY. "
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


def _light_words(meta: dict[str, Any], scene_light: str) -> str:
    if scene_light.strip():
        return scene_light.strip().rstrip(".") + "."
    sun = meta.get("sun") or {}
    if sun.get("direct_sun"):
        wins = sun.get("windows_lit") or []
        where = f" through window {', '.join(str(int(w) + 1) for w in wins)}" if wins else " through the window"
        return (f"is real daylight: direct sun{where} from the {sun.get('compass', '')} at "
                f"{float(sun.get('elevation_deg') or 0):.0f} degrees elevation, with warm sunlit patches on the "
                "floor and walls where the render already shows them, and soft daylight everywhere else.").replace("the  at", "the at")
    if sun:
        return "is soft daylight from the windows with no direct sun, as in the render."
    return "is the room's own light as the render shows it, with the bright and dark areas in the same places."


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
) -> str:
    """The instruction for one frame of a sequence. `meta` is the animatic or
    evaluation frame (focal_mm, actors, sun, framing); `cast` the Actor list;
    `headshot_index` maps an actor's name to the 1-based image number of the
    headshot in the references; `continuity_index` the image number of the
    previous finished frame, when one rides along."""
    cam = meta.get("camera") or {}
    focal = float(meta.get("focal_mm") or cam.get("focal_mm") or 35.0)
    present = meta.get("actors")
    names = list(present.keys()) if isinstance(present, dict) else (list(present) if isinstance(present, list) else None)
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
        people.append(_PERSON.format(colour=colour_name(a.colour).upper(), name=a.name, about=", ".join(about),
                                     ref=ref, wardrobe=wardrobe, place=_place_words((boxes or {}).get(a.name)),
                                     facing=_facing_words(meta, a.name)))
    nobody = _NOBODY if (names is not None and not names) or (not people and not cast) else ""
    if nobody:
        people = []
    who = [a.name for a in cast if names is None or a.name in names]
    count = (f"There are EXACTLY {len(who)} {'person' if len(who) == 1 else 'people'} in this frame, "
             f"{' and '.join(who)}, and nobody else: no extra body, limb or foot anywhere. ") if who and not nobody else ""
    cont = _CONTINUITY.format(n=continuity_index) if continuity_index else ""
    room = ""
    if room_index:
        nums = [f"image {n}" for n in room_index]
        images = (nums[0] if len(nums) == 1 else ", ".join(nums[:-1]) + " and " + nums[-1]).capitalize()
        room = _ROOM.format(images=images, verb="is a photograph" if len(nums) == 1 else "are")
        room = room.replace("is a photograph photographs", "is a photograph")
    text = _SEQUENCE_INSTRUCTION.format(
        shot=_shot_words(meta, boxes), count=count, room=room,
        focal_mm=focal, people="".join(people), nobody=nobody, light=_light_words(meta, scene_light),
        mood=(" " + mood.strip().rstrip(".") + ".") if mood.strip() else "", continuity=cont,
    )
    return " ".join(text.split())


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
    edge = max(2, a.shape[1] // 40)
    n = 0
    for y in range(h - 1, -1, -1):
        row = a[y]
        left = np.abs(row[:edge] - bar).max()
        right = np.abs(row[-edge:] - bar).max()
        if left <= 14 and right <= 14:
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
) -> list[FinishedFrame]:
    """Finish every frame of a sequence; the finished frames, in order.

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
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    headshots = dict(headshots or {})
    model = model or model_for("image")
    total = len(frames)
    lock = threading.Lock()
    done: dict[int, FinishedFrame] = {}

    def say(**ev) -> None:
        if progress is not None:
            try:
                progress({"total": total, **ev})
            except Exception:  # noqa: BLE001 - a reporter must not stop the run
                pass

    def one(i: int, prev: FinishedFrame | None) -> FinishedFrame | None:
        name, png, meta = frames[i]
        dest = out_dir / f"finish_{Path(name).name}"
        if dest.suffix.lower() != ".png":
            dest = dest.with_suffix(".png")
        if dest.exists() and not force:
            ff = FinishedFrame(png=dest.read_bytes(), prompt="", model=model, name=name, path=dest, cached=True)
            say(kind="skip", index=i, name=name, file=dest.name, text=f"{name}: already finished")
            return ff
        say(kind="start", index=i, name=name, file=dest.name, text=f"finishing {name}")
        try:
            present = meta.get("actors")
            names = list(present.keys()) if isinstance(present, dict) else None
            refs: list[bytes] = [strip_caption(png)]
            boxes = figure_boxes(refs[0], cast)
            index: dict[str, int] = {}
            for a in cast:
                if names is not None and a.name not in names:
                    continue
                shot = headshots.get(a.name) or headshots.get(getattr(a, "id", ""))
                if shot and len(refs) < 9:
                    refs.append(shot)
                    index[a.name] = len(refs)
            room_index: list[int] = []
            if room_refs is not None:
                try:
                    for shot in list(room_refs(meta) or [])[:2]:
                        if shot and len(refs) < 9:
                            refs.append(shot)
                            room_index.append(len(refs))
                except Exception as exc:  # noqa: BLE001 - no room photos is a worse frame, not a failed one
                    say(kind="note", index=i, name=name, text=f"{name}: no room photographs ({exc})")
            cont_index = None
            if continuity and prev is not None and len(refs) < 10:
                refs.append(prev.raw or prev.png)
                cont_index = len(refs)
            prompt = build_sequence_prompt(meta=meta, cast=cast, headshot_index=index, continuity_index=cont_index,
                                           scene_light=scene_light, mood=mood, boxes=boxes, room_index=room_index)
            raw = client.generate_image(prompt, references=refs, model=model)
            out = stamp_render(raw) if stamp else raw
            ff = FinishedFrame(png=out, prompt=prompt, model=model, references=len(refs) - 1, name=name, path=dest,
                               raw=raw, meta={"headshots": len(index), "continuity": cont_index is not None,
                                              "room_photos": len(room_index)})
            ff.save(dest)
            say(kind="done", index=i, name=name, file=dest.name, text=f"finished {name}")
            return ff
        except Exception as exc:  # noqa: BLE001 - one frame failing is not the run failing
            say(kind="error", index=i, name=name, file=dest.name, text=f"{name}: {type(exc).__name__}: {exc}")
            return None

    if not frames:
        return []
    # The first frame sets the look; it goes alone. Then chunks of `workers`
    # run in parallel, each frame carrying the last finished frame before its
    # chunk started, so the look is handed forward in order.
    first = one(0, None)
    if first is not None:
        done[0] = first
    workers = max(1, int(workers))
    last: FinishedFrame | None = first
    i = 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while i < total:
            chunk = list(range(i, min(total, i + workers)))
            prev = last if continuity else None
            results = list(pool.map(lambda k: one(k, prev), chunk))
            with lock:
                for k, r in zip(chunk, results):
                    if r is not None:
                        done[k] = r
            for k in reversed(chunk):
                if done.get(k) is not None and not done[k].cached:
                    last = done[k]
                    break
            else:
                for k in reversed(chunk):
                    if done.get(k) is not None:
                        last = done[k]
                        break
            i = chunk[-1] + 1
    return [done[k] for k in sorted(done)]
