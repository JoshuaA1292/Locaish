"""Reading a sweep before paying to reconstruct it.

A room reconstruction costs minutes, and the two ways it most often comes back
useless are both visible in the first twenty seconds of arithmetic on a few
dozen frames. Finding them afterwards is the worst of both: the operator has
waited, the answer is a failure, and the video is on a phone that has since
been put away.

The two failures are these.

**A pan is not a sweep.** Structure from motion recovers depth from parallax,
and parallax comes from the camera *moving*, not from it turning. Someone who
stands in the middle of a room and rotates has taken a beautiful panorama and a
reconstruction of nothing: every frame pair is related by a homography, and a
homography has no depth in it. The measured capture that prompted this module
walked 1.6 m in 54 seconds inside a room 4 m across, and the reconstruction that
came out covered a third of the floor.

**Stereo cannot see a blank wall.** Dense matching needs texture, and a modern
interior is mostly the absence of it: white plaster, white cabinet doors, a
plain ceiling. A capture can be perfectly steady, perfectly exposed and
perfectly useless because three quarters of what it looked at has no detail to
match. This is measurable directly -- the fraction of the frame whose local
gradient is below what a matcher needs -- and it is worth saying out loud,
because the fix is a human one: film the room, but also film the *edges* of it,
where the surfaces meet and there is something to see.

Nothing here blocks a reconstruction. The judgement is the operator's and a
thin capture is still worth something; what is not acceptable is letting them
find out at the end. Every check reports what it measured, what it means, and
what to do differently, and the numbers are all classical image statistics --
gradients, feature matches, a homography and a fundamental matrix -- with no
model of any kind involved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# How many frames the check runs on. Enough to span the sweep and cheap enough
# that the whole pass is a fraction of the decode the reconstruction pays for
# anyway.
SAMPLE_FRAMES = 30

# Frames are compared to a partner this far ahead rather than to their
# neighbour: consecutive frames of a slow walk are separated by centimetres,
# which is too little baseline for the parallax test to say anything, and the
# question being asked is about the sweep rather than about one step of it.
PAIR_STRIDE = 3

# A pair whose motion a homography explains this well has no usable parallax in
# it -- the camera turned, or everything it saw was flat. The threshold is on
# the ratio of homography inliers to fundamental-matrix inliers, which is the
# standard way monocular SLAM decides the same question at initialisation.
ROTATION_RATIO = 0.92
# ... and the capture is called rotation-dominant when this share of its pairs
# are.
ROTATION_FRACTION = 0.6

# Local gradient below this (8-bit levels, per pixel, from a 3x3 Sobel) is a
# surface a dense matcher cannot get a fix on. Six is deliberately low -- it is
# a claim that there is *nothing* there rather than not much, so that the
# warning it drives stays rare enough to mean something.
TEXTURE_GRADIENT = 6.0
# A capture with more than this share of blank pixels will reconstruct its
# clutter and lose its walls. Measured on the kitchen that prompted this
# module: 85% blank at this threshold, and a plain ceiling frame that scores
# 100% on any threshold at all, because there is genuinely nothing in it.
TEXTURE_BLANK_WARN = 0.70

# Sharpness is scored against the best frame in the sample, because absolute
# Laplacian variance depends on the scene as much as on the focus.
BLUR_RATIO = 0.15
BLUR_FRACTION = 0.25

# Fewer matched features than this and the pair is not evidence of anything.
MIN_MATCHES = 40


@dataclass
class Preflight:
    """What a sweep looks like before anything expensive happens to it."""

    verdict: str = "unknown"  # good | thin | unusable | unknown
    notes: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.verdict in ("good", "unknown")

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "notes": list(self.notes), "metrics": dict(self.metrics)}


def inspect(
    frames: list[Path] | list[str],
    *,
    stride: int = PAIR_STRIDE,
    progress=None,
) -> Preflight:
    """Judge a sweep from frames already on disk.

    Takes decoded frames rather than a video path so that it can run on the
    candidates the reconstruction has already extracted -- the check is meant to
    cost a few seconds of arithmetic, not a second decode of a 360 MB file.
    """
    out = Preflight()
    try:
        import cv2
    except ImportError:  # pragma: no cover - the video extra is optional
        out.notes.append(
            "OpenCV is not installed, so the capture could not be checked "
            "before reconstructing it"
        )
        return out

    paths = [Path(p) for p in frames]
    if len(paths) < 4:
        return out
    if len(paths) > SAMPLE_FRAMES:
        idx = np.linspace(0, len(paths) - 1, SAMPLE_FRAMES).round().astype(int)
        paths = [paths[i] for i in dict.fromkeys(idx.tolist())]
        # Sampling across the whole sweep already puts seconds between
        # neighbours, which is the baseline the stride existed to create.
        # Keeping it as well leaves pairs so far apart that they share no view
        # and match nothing, and the parallax test then reports on the three
        # pairs that happened to survive.
        stride = 1

    if progress:
        progress("checking the sweep")

    images = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    if len(images) < 4:
        return out

    sharp = np.array([float(cv2.Laplacian(im, cv2.CV_64F).var()) for im in images])
    blank = np.array([_blank_fraction(cv2, im) for im in images])
    rotation, pairs = _rotation_fraction(cv2, images, stride)

    out.metrics = {
        "frames_checked": float(len(images)),
        "blank_fraction": float(np.mean(blank)),
        "blurred_fraction": float(np.mean(sharp < BLUR_RATIO * max(sharp.max(), 1e-9))),
        "rotation_dominant_fraction": float(rotation),
        "pairs_compared": float(pairs),
    }

    bad = False
    thin = False

    if pairs >= 4 and rotation >= ROTATION_FRACTION:
        bad = True
        out.notes.append(
            f"{rotation * 100:.0f}% of this sweep is the camera turning rather "
            "than travelling. Depth comes from moving between viewpoints, so a "
            "pan from one spot reconstructs almost nothing however long it runs "
            "-- walk the perimeter of the room instead, keeping the far wall in "
            "shot, and the same 50 seconds will carry the whole room"
        )
    elif pairs >= 4 and rotation >= ROTATION_FRACTION * 0.6:
        thin = True
        out.notes.append(
            f"{rotation * 100:.0f}% of this sweep is rotation with little "
            "travel; the parts of the room filmed while standing still will "
            "come back thinner than the rest"
        )

    if out.metrics["blank_fraction"] >= TEXTURE_BLANK_WARN:
        thin = True
        out.notes.append(
            f"{out.metrics['blank_fraction'] * 100:.0f}% of these frames is "
            "surface with no detail in it -- plain walls, plain doors, plain "
            "ceiling. Stereo matching has nothing to lock onto there, so those "
            "surfaces will be found from where the camera could see rather than "
            "from returns. Filming the corners and edges of the room, where "
            "surfaces meet, gives the solve something to hold"
        )

    if out.metrics["blurred_fraction"] >= BLUR_FRACTION:
        thin = True
        out.notes.append(
            f"{out.metrics['blurred_fraction'] * 100:.0f}% of the sampled "
            "frames are motion-blurred; sweeping more slowly is worth more than "
            "sweeping for longer"
        )

    out.verdict = "unusable" if bad else ("thin" if thin else "good")
    return out


def _blank_fraction(cv2, image: np.ndarray) -> float:
    """Share of the frame whose local gradient is below what a matcher needs."""
    small = cv2.resize(image, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # Sobel sums nine samples, so divide back to per-pixel levels before
    # comparing against a threshold expressed in them.
    return float(np.mean(mag / 8.0 < TEXTURE_GRADIENT))


def _rotation_fraction(cv2, images: list[np.ndarray], stride: int) -> tuple[float, int]:
    """Share of frame pairs whose motion a homography explains as well as depth does.

    Two models are fitted to the same matches. A homography maps one image onto
    another exactly when the camera only rotated, or when everything in view was
    a plane; a fundamental matrix is the general case and can also account for
    depth. Comparing how many matches each one keeps is therefore a direct
    reading of whether there is any parallax to reconstruct from -- and it is
    the same test a monocular SLAM system makes before it will initialise,
    for the same reason.
    """
    orb = cv2.ORB_create(nfeatures=1500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    feats = []
    for im in images:
        kp, des = orb.detectAndCompute(im, None)
        feats.append((kp, des))

    rotational = 0
    compared = 0
    for i in range(len(images) - stride):
        kp1, des1 = feats[i]
        kp2, des2 = feats[i + stride]
        if des1 is None or des2 is None or len(kp1) < MIN_MATCHES or len(kp2) < MIN_MATCHES:
            continue
        matches = matcher.match(des1, des2)
        if len(matches) < MIN_MATCHES:
            continue
        src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        _, h_mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        _, f_mask = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC, 3.0, 0.99)
        if h_mask is None or f_mask is None:
            continue
        h_in = float(h_mask.sum())
        f_in = float(f_mask.sum())
        if f_in < MIN_MATCHES:
            continue
        compared += 1
        if h_in / f_in >= ROTATION_RATIO:
            rotational += 1

    if compared == 0:
        return 0.0, 0
    return rotational / compared, compared


__all__ = ["Preflight", "inspect"]
