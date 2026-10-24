"""Scanning through a crowd: masks for the things that moved during the sweep.

Structure from motion assumes the world stood still while the camera moved.
A person walking through the shot breaks that assumption in the worst way:
their edges are sharp, textured and plentiful, so the feature detector
fills its budget with keypoints on the person, the matcher chains those
keypoints across frames, and bundle adjustment then explains a moving body
with a wrong camera. The poses bend, the walls bow, and nothing in the
solve's own residuals says why, because the matches were "correct".

The fix is to tell the detector where not to look. SAM 3.1 takes a noun
phrase ("person", "dog", "car") and returns a mask for every match; the
masks for every mover prompt are unioned, grown a little (the edges of a
moving body are blurred, and the halo around it is as unreliable as the body
itself), and written in the convention COLMAP's feature extractor reads:
one 8-bit PNG per frame named `<image name>.png`, white where features may
be found, black where they may not.

Two outcomes need a decision rather than a mask. A frame that is mostly
mover -- someone walked right in front of the lens -- has no room left to
match, and keeping it only costs a registration failure or, worse, a
registration against the person; such frames are reported as `dropped` for
the caller to remove from the set. And a frame where the API failed is
left all-white: a mask stage that fails must degrade to the solve the
pipeline ran yesterday, never block it.

The masks are a *prior on where not to measure*, nothing more. No point in
the twin comes from them, and no number in the QA report depends on them
except the count of frames they touched.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

# What moves through a room while someone films it. Overridable with
# `LOCAISH_MOVER_PROMPTS` (comma-separated) for a location with, say,
# pigeons or forklifts.
MOVER_PROMPTS: tuple[str, ...] = ("person", "dog", "cat", "car", "bicycle")

# Masked area, as a fraction of the frame after dilation, past which the
# frame is reported for dropping: with under 55% of the picture left, and
# that mostly at the edges, the solve has nothing to hold onto.
DROP_FRACTION = 0.45

# How far the mask grows, as a fraction of the frame's width. A walking
# body blurs over a few pixels at 30 fps, and the detector finds strong
# corners exactly on that blurred boundary.
DILATE_FRACTION = 0.02

# SAM 3.1 tracks through a clip; the request is chunked so one bad frame
# cannot cost the whole sweep, and so the payload stays well under what a
# single request can carry at these frame sizes.
VIDEO_CHUNK = 60

# What the written mask means to COLMAP.
USE = 255
IGNORE = 0

PROVIDER = "meta:sam-3.1"


def mover_prompts() -> tuple[str, ...]:
    """The prompts to mask, honouring `LOCAISH_MOVER_PROMPTS`."""
    raw = os.environ.get("LOCAISH_MOVER_PROMPTS")
    if not raw:
        return MOVER_PROMPTS
    got = tuple(p.strip() for p in raw.split(",") if p.strip())
    return got or MOVER_PROMPTS


@dataclass
class MaskReport:
    """What the mask stage did to each frame, and what it asks the caller to do."""

    provider: str = PROVIDER
    prompts: list[str] = field(default_factory=list)
    #: frame filename -> masked fraction of the frame (after dilation)
    masked_fraction: dict[str, float] = field(default_factory=dict)
    #: prompt -> number of detections across every frame
    counts: dict[str, int] = field(default_factory=dict)
    #: frames whose masked fraction exceeded DROP_FRACTION, in frame order
    dropped: list[str] = field(default_factory=list)
    #: frames whose API calls failed and were left all-white
    failed: list[str] = field(default_factory=list)
    mask_dir: str = ""
    #: frame filename -> the person detections seen there. Kept in memory
    #: only (not in to_dict): the stature scale anchor reads the people the
    #: mask stage already found rather than asking SAM a second time.
    people: dict[str, list[Any]] = field(default_factory=dict, repr=False)

    @property
    def frames_masked(self) -> int:
        """Frames where a mover was found and something was blacked out."""
        return sum(1 for f in self.masked_fraction.values() if f > 0.0)

    @property
    def median_masked_fraction(self) -> float:
        vals = [f for f in self.masked_fraction.values() if f > 0.0]
        return float(np.median(vals)) if vals else 0.0

    def fingerprint(self) -> str:
        """A short stable identity for *these* masks, for cache keys and stamps."""
        body = json.dumps(
            {
                "prompts": list(self.prompts),
                "fractions": {k: round(v, 4) for k, v in sorted(self.masked_fraction.items())},
                "dropped": list(self.dropped),
            },
            sort_keys=True,
        )
        return hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]

    def note(self) -> str | None:
        """One QA-report line, or None when no mover was found anywhere."""
        n = self.frames_masked
        if not n and not self.dropped:
            return None
        kinds = [p for p in self.prompts if self.counts.get(p)]
        what = _describe(kinds)
        line = (
            f"{n} frame{'s' if n != 1 else ''} had {what} masked out of the camera "
            f"solve (median {self.median_masked_fraction:.0%} of the frame)"
        )
        if self.dropped:
            k = len(self.dropped)
            line += f"; {k} frame{'s' if k != 1 else ''} {'were' if k != 1 else 'was'} dropped as mostly movers"
        if self.failed:
            line += f"; {len(self.failed)} frame(s) could not be masked and were used whole"
        return line

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "prompts": list(self.prompts),
            "frames": len(self.masked_fraction),
            "frames_masked": self.frames_masked,
            "median_masked_fraction": round(self.median_masked_fraction, 4),
            "masked_fraction": {k: round(float(v), 4) for k, v in self.masked_fraction.items()},
            "counts": dict(self.counts),
            "frames_dropped": list(self.dropped),
            "frames_failed": list(self.failed),
            "mask_dir": self.mask_dir,
            "fingerprint": self.fingerprint(),
        }

    def summary(self) -> dict[str, Any]:
        """The provenance entry the reconstruction records."""
        return {
            "provider": self.provider,
            "frames_masked": self.frames_masked,
            "median_masked_fraction": round(self.median_masked_fraction, 4),
            "frames_dropped": list(self.dropped),
            "prompts": list(self.prompts),
            "fingerprint": self.fingerprint(),
        }


def _describe(kinds: Sequence[str]) -> str:
    people = [k for k in kinds if k in ("person", "people", "man", "woman", "child")]
    animals = [k for k in kinds if k in ("dog", "cat", "pet", "bird", "horse")]
    vehicles = [k for k in kinds if k in ("car", "bicycle", "bus", "truck", "motorcycle", "van", "scooter")]
    parts = []
    if people:
        parts.append("people")
    if animals:
        parts.append("animals")
    if vehicles:
        parts.append("vehicles")
    other = [k for k in kinds if k not in people + animals + vehicles]
    parts += other
    if not parts:
        return "movers"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " or " + parts[-1]


# ---------------------------------------------------------------------------
# the two entry points
# ---------------------------------------------------------------------------


def mask_frames(
    frame_paths: Sequence[str | Path],
    out_dir: str | Path,
    *,
    client: Any,
    prompts: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> MaskReport:
    """One `segment` call per frame per prompt; masks written for every frame.

    The simplest path and the fallback for the clip path below. A frame whose
    call fails (any exception from the client) is logged through `progress`
    and written all-white, so the solve sees exactly what it would have seen
    without this stage.
    """
    paths = [Path(p) for p in frame_paths]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = tuple(prompts) if prompts is not None else mover_prompts()
    report = MaskReport(prompts=list(prompts), mask_dir=str(out_dir))
    for i, path in enumerate(paths):
        if progress and (i % 25 == 0):
            progress(f"masking movers ({i + 1}/{len(paths)})")
        dets, ok = _segment_image(client, path, prompts, progress)
        if not ok:
            report.failed.append(path.name)
        _finish_frame(path, dets, out_dir, report)
    return report


def mask_video_frames(
    frame_paths: Sequence[str | Path],
    out_dir: str | Path,
    *,
    client: Any,
    prompts: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
    chunk: int = VIDEO_CHUNK,
) -> MaskReport:
    """The clip path: `segment(prompt, frames=[...])` in chunks, tracked.

    Tracking through consecutive frames is what SAM 3.1 is for, and it costs
    a handful of calls per prompt instead of one per frame. The answer is
    only usable when every detection says which frame it belongs to; a chunk
    whose answer does not (or whose call fails) is redone frame by frame.
    """
    paths = [Path(p) for p in frame_paths]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = tuple(prompts) if prompts is not None else mover_prompts()
    report = MaskReport(prompts=list(prompts), mask_dir=str(out_dir))
    chunk = max(1, int(chunk))
    for start in range(0, len(paths), chunk):
        batch = paths[start : start + chunk]
        if progress:
            progress(f"masking movers ({start + 1}-{start + len(batch)}/{len(paths)})")
        per_frame: list[list[Any]] = [[] for _ in batch]
        ok = True
        frame_bytes = None
        for prompt in prompts:
            try:
                if frame_bytes is None:
                    frame_bytes = [p.read_bytes() for p in batch]
                dets = list(client.segment(prompt, frames=frame_bytes))
            except Exception as exc:  # noqa: BLE001 - one bad chunk, not a dead solve
                if progress:
                    progress(f"mover masks: clip call for {prompt!r} failed ({exc}); masking frame by frame")
                ok = False
                break
            if any(d.frame is None for d in dets):
                ok = False
                break
            for d in dets:
                if d.frame is not None and 0 <= int(d.frame) < len(batch):
                    per_frame[int(d.frame)].append(d)
        if not ok:
            for path in batch:
                dets, good = _segment_image(client, path, prompts, progress)
                if not good:
                    report.failed.append(path.name)
                _finish_frame(path, dets, out_dir, report)
            continue
        for path, dets in zip(batch, per_frame):
            _finish_frame(path, dets, out_dir, report)
    return report


def write_report(report: MaskReport, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(report.to_dict(), indent=2))
    return path


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _segment_image(client: Any, path: Path, prompts: Sequence[str], progress) -> tuple[list[Any], bool]:
    """Every detection for one frame across all prompts; False when a call failed."""
    out: list[Any] = []
    ok = True
    data = None
    for prompt in prompts:
        try:
            if data is None:
                data = path.read_bytes()
            out.extend(client.segment(prompt, data))
        except Exception as exc:  # noqa: BLE001 - one bad call, not a dead solve
            ok = False
            if progress:
                progress(f"mover masks: {path.name} {prompt!r} failed ({exc}); frame used whole")
    return out, ok


def _finish_frame(path: Path, dets: list[Any], out_dir: Path, report: MaskReport) -> None:
    """Union, dilate, write, and account for one frame's detections."""
    size = _image_size(path)
    mover = union_mask(dets, size)
    for d in dets:
        label = str(getattr(d, "label", "") or "")
        key = label if label in report.prompts else _match_prompt(label, report.prompts)
        report.counts[key] = report.counts.get(key, 0) + 1
        if "person" in label.lower() or "people" in label.lower():
            report.people.setdefault(path.name, []).append(d)
    if mover.any():
        mover = dilate(mover, max(1, int(round(DILATE_FRACTION * size[0]))))
    frac = float(mover.mean()) if mover.size else 0.0
    report.masked_fraction[path.name] = frac
    if frac > DROP_FRACTION:
        report.dropped.append(path.name)
    write_mask(mover, out_dir / f"{path.name}.png")


def _match_prompt(label: str, prompts: Sequence[str]) -> str:
    low = label.lower()
    for p in prompts:
        if p.lower() in low or low in p.lower():
            return p
    return label or "?"


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size  # (width, height)


def union_mask(dets: Sequence[Any], size: tuple[int, int]) -> np.ndarray:
    """Boolean (H, W): True where any detection says a mover is.

    A detection with a mask contributes the mask, resized to the frame if
    the API answered at another resolution. One with only a box contributes
    the box: a crude mask beats no mask, because the alternative is to let
    the detector loose on the thing we know is moving.
    """
    w, h = int(size[0]), int(size[1])
    out = np.zeros((h, w), dtype=bool)
    for d in dets:
        m = None
        png = getattr(d, "mask_png", None)
        if png:
            m = decode_mask(png, (w, h))
        if m is None:
            box = getattr(d, "box", None)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            c0, c1 = int(np.floor(x1 * w)), int(np.ceil(x2 * w))
            r0, r1 = int(np.floor(y1 * h)), int(np.ceil(y2 * h))
            c0, c1 = max(0, min(c0, w)), max(0, min(c1, w))
            r0, r1 = max(0, min(r0, h)), max(0, min(r1, h))
            if c1 > c0 and r1 > r0:
                out[r0:r1, c0:c1] = True
        else:
            out |= m
    return out


def decode_mask(png: bytes, size: tuple[int, int]) -> np.ndarray | None:
    """PNG bytes to a boolean (H, W) at the frame's size, or None if unreadable.

    Non-zero means object, in whichever channel carries it: a greyscale or
    binary mask by value, an alpha-carrying image by its alpha.
    """
    from PIL import Image

    try:
        with Image.open(io.BytesIO(png)) as im:
            im.load()
            if im.mode in ("LA", "RGBA", "PA", "La", "RGBa"):
                chan = im.getchannel("A")
            elif im.mode == "1":
                chan = im.convert("L")
            else:
                chan = im.convert("L")
            if chan.size != tuple(size):
                chan = chan.resize(tuple(size), Image.NEAREST)
            return np.asarray(chan) > 127
    except Exception:  # noqa: BLE001 - a broken mask is "no mask", the box fills in
        return None


def dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Grow a boolean mask by a square of the given radius (separable, fast)."""
    if radius_px <= 0 or not mask.any():
        return mask
    from scipy.ndimage import maximum_filter

    return maximum_filter(mask.astype(np.uint8), size=2 * int(radius_px) + 1) > 0


def write_mask(mover: np.ndarray, path: Path) -> Path:
    """COLMAP's convention: 8-bit, 255 = use for features, 0 = ignore."""
    from PIL import Image

    img = np.where(mover, IGNORE, USE).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="L").save(path, format="PNG")
    return path


def read_mask(path: str | Path) -> np.ndarray:
    """The written mask back as a boolean 'mover' array (True = ignored)."""
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("L")) < 128

