"""The check that runs before the eight minutes, not after them.

Both failures it exists to catch are synthesised here rather than sampled from
a real capture, because the point of each test is that the module reads the
*cause* correctly: a sequence built from pure rotation must be called rotation
whatever else is in it, and a blank wall must be called blank even when it is
perfectly sharp and perfectly exposed.
"""

from __future__ import annotations

import numpy as np
import pytest

from locaish.video import preflight

cv2 = pytest.importorskip("cv2")


def _texture(rng, w=640, h=360):
    """A frame with something in it: broadband noise, blurred to be matchable."""
    img = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
    return cv2.GaussianBlur(img, (5, 5), 0)


def _write(tmp_path, images, prefix="f"):
    paths = []
    for i, im in enumerate(images):
        p = tmp_path / f"{prefix}_{i:03d}.png"
        cv2.imwrite(str(p), im)
        paths.append(p)
    return paths


def test_a_pan_is_reported_as_a_pan(tmp_path):
    """Frames related by a homography carry no depth, however many there are.

    Warping one image by a succession of homographies is exactly what a camera
    rotating on the spot produces, so a fitter that cannot tell this from a
    walk would happily spend eight minutes on it.
    """
    rng = np.random.default_rng(0)
    base = _texture(rng, 900, 500)
    frames = []
    for i in range(12):
        # a rotation about the optical centre, as a homography
        a = np.radians(2.0 * i)
        h = np.array(
            [[np.cos(a), -np.sin(a), 8.0 * i], [np.sin(a), np.cos(a), 0.0], [0, 0, 1.0]]
        )
        frames.append(cv2.warpPerspective(base, h, (900, 500)))

    got = preflight.inspect(_write(tmp_path, frames))
    assert got.metrics["pairs_compared"] >= 4
    assert got.metrics["rotation_dominant_fraction"] >= preflight.ROTATION_FRACTION
    assert got.verdict == "unusable"
    assert any("turning rather than travelling" in n for n in got.notes)
    assert not got.ok


def test_a_walk_past_two_depths_is_not_reported_as_a_pan(tmp_path):
    """Two layers sliding at different rates is parallax, and must read as such."""
    rng = np.random.default_rng(1)
    far = _texture(rng, 900, 500)
    near = _texture(rng, 900, 500)
    frames = []
    for i in range(12):
        canvas = np.roll(far, 4 * i, axis=1).copy()
        # the near layer sweeps across four times faster: depth, not rotation
        patch = np.roll(near, 16 * i, axis=1)[:, :420]
        canvas[40:460, 240:660] = patch[40:460, :420]
        frames.append(canvas)

    got = preflight.inspect(_write(tmp_path, frames))
    assert got.metrics["pairs_compared"] >= 4
    assert got.metrics["rotation_dominant_fraction"] < preflight.ROTATION_FRACTION


def test_a_blank_room_is_reported_as_blank(tmp_path):
    """Sharp, steady and useless: the surfaces have nothing on them to match."""
    rng = np.random.default_rng(2)
    frames = []
    for i in range(10):
        canvas = np.full((500, 900), 226, dtype=np.uint8)
        # a little furniture in one corner, the way a real blank room has some
        canvas[380:480, 40:240] = _texture(rng, 200, 100)
        canvas = np.roll(canvas, 6 * i, axis=1)
        frames.append(canvas)

    got = preflight.inspect(_write(tmp_path, frames))
    assert got.metrics["blank_fraction"] >= preflight.TEXTURE_BLANK_WARN
    assert got.verdict in ("thin", "unusable")
    assert any("no detail in it" in n for n in got.notes)


def test_a_textured_walk_passes(tmp_path):
    rng = np.random.default_rng(3)
    far = _texture(rng, 900, 500)
    near = _texture(rng, 900, 500)
    frames = []
    for i in range(12):
        canvas = np.roll(far, 5 * i, axis=1).copy()
        patch = np.roll(near, 18 * i, axis=1)[:, :420]
        canvas[40:460, 240:660] = patch[40:460, :420]
        frames.append(canvas)

    got = preflight.inspect(_write(tmp_path, frames))
    assert got.verdict == "good", got.notes
    assert got.ok
    assert not got.notes


def test_too_few_frames_is_declined_rather_than_guessed(tmp_path):
    rng = np.random.default_rng(4)
    got = preflight.inspect(_write(tmp_path, [_texture(rng) for _ in range(2)]))
    assert got.verdict == "unknown"
    assert got.ok


def test_the_verdict_round_trips_to_json():
    got = preflight.Preflight(verdict="thin", notes=["a"], metrics={"x": 1.0})
    assert got.to_dict() == {"verdict": "thin", "notes": ["a"], "metrics": {"x": 1.0}}
