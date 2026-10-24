"""The people who walked through the scan, as a scale anchor.

A phone video has no size of its own, and `locaish.video.metric` recovers it
from anchors that share no evidence: how high the phone rode, how tall a door
is, what a depth network believes. This module supplies a fourth: adult
stature. Someone standing in a frame is a vertical segment of known expected
length (1.70 m, give or take 9 cm across the population), and nothing about
that prior is shared with the camera path, with carpentry or with a training
set of rooms. SAM 3.1 finds the people; this module turns each full-length
sighting into the person's apparent height in reconstruction units, and
`metric.scale_from_stature` turns those readings into a `ScaleEstimate`
that enters the same inverse-variance fusion as every other anchor.

**Frames and units.** `PosedFrames` casts rays in *twin space*, which is the
canonical frame under whatever provisional scale the frames were built with:
twin = R @ (factor * colmap) + t. So a length measured along those rays is
`factor` times its length in COLMAP's reconstruction units. `height_readings`
takes its floor in that same twin frame (`floor_z`, the plane's offset along
`floor_normal`), measures the person there, and then **divides by
`frames.factor`** so what it returns is in reconstruction units, independent
of the provisional scale. That is the unit `scale_from_stature` expects
(metres per unit = 1.70 / reading), and it means a `PosedFrames` built with
`factor=1.0` and `floor_z` in raw reconstruction units gives identical
readings to one built with the provisional metric factor and a metric floor.

**Why no intrinsics.** The height is not read off the pixel count with a
focal length. Instead two rays are cast through the box's bottom-centre and
top-centre pixels: the first meets the floor plane at the feet; the second is
followed to the point where it passes directly above the feet, and the rise
from the floor to that point is the height. That uses only what `ray()` gives
-- origin and direction in the twin -- and is exact for a pinhole camera with
no roll about its axis, whatever its pitch. With roll the box's centre column
is not the person's axis and the reading picks up a small error, which is
what the per-reading sigma in `scale_from_stature` is for.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

import numpy as np

from .client import Detection, MetaError

# A box that reaches the top or bottom edge of the frame is a person cut off
# by the frame, whose height is unknown. One percent of the frame is the
# tolerance for a box that was clipped to the image rather than drawn there.
EDGE_FRACTION = 0.01

# Below this share of the frame height a person is too far, or too small, for
# a mask edge to carry a height; a pixel of mask error at 15% of a 1080-pixel
# frame is already a centimetre of stature.
MIN_BOX_FRACTION = 0.25

# SAM's own confidence that the mask is a person at all.
MIN_SCORE = 0.5

# Labels a segmenter answers for the "person" prompt that mean a whole human.
PERSON_LABELS = ("person", "people", "human", "man", "woman", "adult", "pedestrian")

DEFAULT_MAX_FRAMES = 24


def is_person(label: str) -> bool:
    text = (label or "").strip().lower()
    return any(word in text for word in PERSON_LABELS)


def height_readings(
    frames,
    detections_by_frame: dict[int, list[Detection]],
    floor_z: float,
    *,
    floor_normal=(0.0, 0.0, 1.0),
) -> list[float]:
    """Apparent heights, in reconstruction units, of the people seen standing.

    `frames` is a `locaish.product.labels.PosedFrames`; `detections_by_frame`
    maps a frame index into it to the SAM detections for that frame, with
    boxes in normalised xyxy over the frame; `floor_z` is the floor plane's
    offset along `floor_normal` in the frames' twin space (for the default
    normal, simply the twin z of the floor). A detection counts only if it is
    a person, scores at least `MIN_SCORE`, touches neither the top nor the
    bottom edge of the frame, and stands at least `MIN_BOX_FRACTION` of the
    frame tall. Everything else is skipped silently: a missed sighting costs
    nothing, a wrong one costs the anchor.

    Each reading is divided by `frames.factor`, so it is in COLMAP's
    reconstruction units whatever provisional scale the frames carry; pass
    the list to `metric.scale_from_stature` with `current_factor=frames.factor`.
    """
    n = np.asarray(floor_normal, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(n))
    if norm < 1e-9:
        raise ValueError("floor_normal must be a non-zero vector")
    n = n / norm
    factor = float(getattr(frames, "factor", 1.0) or 1.0)
    width, height = frames.size

    readings: list[float] = []
    for i, dets in (detections_by_frame or {}).items():
        if i is None or not (0 <= int(i) < len(frames)):
            continue
        for det in dets or []:
            if not is_person(det.label) or float(det.score) < MIN_SCORE:
                continue
            x1, y1, x2, y2 = (float(v) for v in det.box)
            if y1 <= EDGE_FRACTION or y2 >= 1.0 - EDGE_FRACTION:
                continue
            if (y2 - y1) < MIN_BOX_FRACTION:
                continue
            u = 0.5 * (x1 + x2) * width
            h_units = _person_height(frames, int(i), u, y1 * height, y2 * height, floor_z, n)
            if h_units is None:
                continue
            readings.append(h_units / factor)
    return readings


def _person_height(frames, i: int, u: float, v_top: float, v_bot: float, floor_z: float, n: np.ndarray) -> float | None:
    """The rise, in twin units, from the floor hit of the bottom pixel's ray to
    the point on the top pixel's ray directly above it; None when the rays do
    not make a standing person."""
    o, d_bot = frames.ray(i, u, v_bot)
    _, d_top = frames.ray(i, u, v_top)

    # Feet: where the bottom ray meets the floor plane. It has to be heading
    # down and the floor has to be in front of the camera.
    down_rate = float(d_bot @ n)
    if down_rate > -1e-6:
        return None
    s_bot = (floor_z - float(o @ n)) / down_rate
    if s_bot <= 0:
        return None
    feet = o + s_bot * d_bot

    # Head: the point on the top ray whose horizontal position (components
    # orthogonal to the floor normal) matches the feet's.
    def flat(x):
        return x - n * float(x @ n)

    d_flat = flat(d_top)
    span = float(d_flat @ d_flat)
    if span < 1e-12:
        return None            # looking straight up or down; no horizontal reach
    s_top = float(flat(feet - o) @ d_flat) / span
    if s_top <= 0:
        return None
    head = o + s_top * d_top

    rise = float((head - feet) @ n)
    if not np.isfinite(rise) or rise <= 0:
        return None
    return rise


def detect_people(
    frames,
    client,
    *,
    max_frames: int = DEFAULT_MAX_FRAMES,
    progress: Callable[[str], None] | None = None,
) -> dict[int, list[Detection]]:
    """Ask SAM for the people in a spread of the sweep's frames.

    Returns frame index -> detections, each detection stamped with its frame
    index, for the frames where at least one person was found. One frame
    failing (an unreadable file, an API error) costs that frame and nothing
    else; the count is reported through `progress`.
    """
    picks = frames.spread(max_frames)
    found: dict[int, list[Detection]] = {}
    failed = 0
    for k, i in enumerate(picks):
        try:
            image = Path(frames.paths[i]).read_bytes()
            dets = client.segment("person", image)
        except (OSError, MetaError) as exc:
            failed += 1
            if progress:
                progress(f"stature: frame {frames.names[i]} skipped ({exc})")
            continue
        stamped = [dataclasses.replace(d, frame=int(i)) for d in dets if is_person(d.label)]
        if stamped:
            found[int(i)] = stamped
        if progress and (k + 1) % 8 == 0:
            progress(f"stature: {k + 1}/{len(picks)} frames segmented, people in {len(found)}")
    if progress:
        tail = f", {failed} frames failed" if failed else ""
        progress(f"stature: people found in {len(found)} of {len(picks)} frames{tail}")
    return found
