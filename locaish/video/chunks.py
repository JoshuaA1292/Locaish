"""Reconstructing more frames than the network can hold at once.

The network attends across every frame at once, which is what makes the frames
mutually consistent -- and what makes memory grow with the square of their
number. Twenty-four is the working limit on a 32 GB machine. A ninety-second
walk through a room offers several hundred usable frames, and the twenty-four
that fit see a fraction of the room; the holes people complain about in the
resulting twin are mostly not reconstruction failures at all, but parts of the
room no submitted frame ever looked at.

So the sweep is cut into overlapping windows, each reconstructed on its own, and
the pieces are put back together. Each window comes out in its own arbitrary
frame with its own arbitrary scale -- the network has no idea the windows are
of the same room -- so the join has to be computed, and the overlap is what
makes that possible: the frames a window shares with its predecessor have a
pose in both, and one rigid similarity carries one onto the other.

Two details decide whether this works or produces a room folded across itself.

**Rotation comes from the cameras' orientations, not their positions.** Fitting a
similarity to the shared camera *centres* is the textbook move and it degenerates
on the most common capture there is: someone walking in a straight line. Collinear
points leave the rotation about the walking axis completely undetermined, so the
room is free to barrel-roll between one window and the next. Each camera's
orientation is a full 3D rotation on its own, so averaging those pins all three
degrees of freedom no matter what path the operator walked.

**Scale is separate and comes from distances.** Two windows of the same room
disagree about how big it is, and that ratio is recovered from the spread of the
shared camera centres, which needs no more than two distinct positions.

The result is a single cloud in a single frame with more of the room in it. What
it is not is a bundle adjustment: errors accumulate window to window, and a long
enough sweep will drift. The drift is reported rather than hidden, and the QA
pass measures it downstream on the twin like any other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Frames per window. Matched to what the network holds comfortably; the whole
# module exists because this cannot simply be raised.
DEFAULT_CHUNK = 24

# Frames shared between neighbouring windows. Every one is a constraint on the
# join, and eight is comfortably past the point where adding more measurably
# improves it -- while still being a third of the window, so two thirds of each
# reconstruction is new ground rather than re-reconstructed old ground.
DEFAULT_OVERLAP = 8

# Below this the join is not trustworthy: too few shared views to average a
# rotation over, and the similarity would be fitted to noise.
MIN_OVERLAP = 4

# Rotation error accumulates along a chain of joins -- there is no bundle
# adjustment here to spread it -- and on a four-window sweep it left the twin a
# degree and a half off level where a single window was a quarter of a degree.
#
# The obvious fix was tried and removed, and it is worth recording why. Each
# window estimates gravity independently from how the phone was held during it,
# so after registration those estimates ought to coincide and any disagreement
# looks like drift that can simply be rotated back out. It is not. The estimate
# measures the operator's *posture*, and posture genuinely changes over a
# ninety-second walk -- people tilt down to watch the screen and up to see where
# they are going. Cross-window disagreement is therefore part drift and part
# real, nothing here can separate the two, and a correction that assumes it is
# all drift injects error as readily as it removes it.
#
# So the accumulated tilt stands, and is reported. Removing it honestly needs a
# global solve over all the windows at once, not a per-join patch.


@dataclass
class Similarity:
    """The transform `p_a = scale * R @ p_b + t` carrying window b onto window a."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    residual_m: float = 0.0
    shared: int = 0

    def apply(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return (self.scale * (self.rotation @ pts.T)).T + self.translation

    def apply_direction(self, vec: np.ndarray) -> np.ndarray:
        """Rotate a direction. No scale, no translation -- it is not a point."""
        return self.rotation @ np.asarray(vec, dtype=np.float64).reshape(3)

    def apply_extrinsics(self, extrinsics: np.ndarray) -> np.ndarray:
        """Re-express world-to-camera matrices in the target frame.

        A camera photographs the same image whichever frame the world is
        described in, so scaling the world by `s` scales everything the camera
        measures -- including the depth to each point, and therefore the
        extrinsic translation -- while leaving its orientation untouched.

        Writing that out: `R_b x_b + t_b = (1/s)(R_a x_a + t_a)` with
        `x_a = s R x_b + t` gives `R_a = R_b R^T` and `t_a = s t_b - R_b R^T t`.
        The scale on `t_b` is the part that is easy to drop, and dropping it
        leaves every camera in the window at a plausible orientation and the
        wrong distance away.
        """
        out = np.array(extrinsics, dtype=np.float64, copy=True)
        rt = self.rotation.T
        for i in range(len(out)):
            r, t = out[i, :, :3].copy(), out[i, :, 3].copy()
            out[i, :, :3] = r @ rt
            out[i, :, 3] = self.scale * t - (r @ rt @ self.translation)
        return out


def camera_centres(extrinsics: np.ndarray) -> np.ndarray:
    """World positions of cameras given world-to-camera `[R|t]` matrices."""
    e = np.asarray(extrinsics, dtype=np.float64)
    return np.einsum("nij,nj->ni", e[:, :, :3].transpose(0, 2, 1), -e[:, :, 3])


def register(extr_a: np.ndarray, extr_b: np.ndarray) -> Similarity:
    """Find the similarity carrying frame b onto frame a, from shared poses.

    `extr_a[i]` and `extr_b[i]` are the same physical camera as solved by two
    different windows. Rotation is averaged over the pairs and projected back
    onto SO(3); scale is the ratio of how far apart the centres are in each; the
    translation then follows from the centroids.

    The residual is reported because it is the only thing that will ever say the
    join went wrong: two windows that share eight views and disagree about where
    those views were by half a metre have not been registered, whatever transform
    comes back.
    """
    a = np.asarray(extr_a, dtype=np.float64)
    b = np.asarray(extr_b, dtype=np.float64)
    if len(a) != len(b) or len(a) < 2:
        raise ValueError(f"need matching pose pairs, got {len(a)} and {len(b)}")

    # -- rotation ---------------------------------------------------------
    # A camera's image does not depend on which frame the world is written in,
    # so `R_b = R_a @ Rot` for every shared view and `Rot = R_a^T @ R_b`.
    # Summing those individual estimates and re-orthogonalising is the chordal
    # mean on SO(3): cheap, and robust to one bad pose among eight.
    acc = np.zeros((3, 3))
    for i in range(len(a)):
        acc += a[i, :, :3].T @ b[i, :, :3]
    u, _, vt = np.linalg.svd(acc)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt

    # -- scale ------------------------------------------------------------
    ca, cb = camera_centres(a), camera_centres(b)
    span_a = ca - ca.mean(axis=0)
    span_b = cb - cb.mean(axis=0)
    denom = float((span_b**2).sum())
    scale = float(np.sqrt((span_a**2).sum() / denom)) if denom > 1e-18 else 1.0

    # -- translation ------------------------------------------------------
    translation = ca.mean(axis=0) - scale * (rot @ cb.mean(axis=0))

    sim = Similarity(scale=scale, rotation=rot, translation=translation, shared=len(a))
    moved = sim.apply(cb)
    sim.residual_m = float(np.sqrt(((moved - ca) ** 2).sum(axis=1)).mean())
    return sim


def windows(count: int, chunk: int = DEFAULT_CHUNK, overlap: int = DEFAULT_OVERLAP):
    """Index ranges covering `count` frames in overlapping windows.

    The last window is pulled back to end on the final frame rather than left
    short, so the end of the sweep is reconstructed with a full window's context
    instead of whatever remainder the arithmetic produced.
    """
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    if count <= chunk:
        return [(0, count)]
    overlap = int(np.clip(overlap, 0, chunk - 1))
    stride = chunk - overlap
    out: list[tuple[int, int]] = []
    start = 0
    while True:
        end = start + chunk
        if end >= count:
            out.append((max(0, count - chunk), count))
            break
        out.append((start, end))
        start += stride
    return out
