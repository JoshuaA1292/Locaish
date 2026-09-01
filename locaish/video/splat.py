"""Gaussian-splat densification: per-scene optimisation as a stereo backend.

Classical patch-match stereo cannot return surface from a blank wall -- there
is nothing for pixels to match -- and every pretrained depth model is banned
by the hackathon's rules. Between those two sits a technology that is
neither: 3D Gaussian Splatting, which fits a set of oriented Gaussian discs
to THIS scene's own frames by gradient descent. No pretrained weights, no
external data, no API -- the same mathematical family as the bundle
adjustment that solved the cameras, distributed as ordinary open-source
software (Brush, a Rust/Metal trainer). Where stereo sees nothing, the
optimiser must still explain every wall pixel of every frame, so it puts
view-consistent surface there. That is the exact gap in the classical
pipeline, filled per-scene.

The output is a cloud, on purpose. The splats are sampled back into a dense,
uniform, coloured point cloud (one point per ~8 mm of real surface, opacity-
gated, floaters dropped), and from there the twin pipeline neither knows nor
cares that the points came from an optimiser: the sparse-box clip, the
ray-contradiction fog filter, the kNN trim, the MLS surface refinement and
the room solve all run unchanged. A splat cloud is held to the same audit as
a stereo cloud, which is the project's whole bargain.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np

# f_dc -> linear colour: the degree-0 spherical-harmonic basis constant.
SH_C0 = 0.28209479177387814

# Sampling defaults, in METRES (converted with the provisional scale before
# use, because the splat trains in COLMAP units).
SAMPLE_PITCH_M = 0.004
# Ellipse cutoff in sigmas: ~94% of a Gaussian's 2D mass, dense enough that
# neighbouring splats overlap without stacking double-thickness seams.
SAMPLE_K = 1.7
# A Gaussian dimmer than this models haze, not surface. Nearly open on
# purpose: Brush builds an opaque wall as a STACK of translucent Gaussians,
# so each contributes points in proportion to its alpha and the stack
# integrates to full surface density. True dust is handled by the
# probabilistic counts (an alpha-0.03 patch almost never draws a point).
OPACITY_MIN = 0.02
# ... and one larger than this (major-axis sigma) is a floater or the sky.
S1_CAP_M = 0.20
CAP_PER_GAUSSIAN = 400

# Training defaults. ~12k steps at 1280 px is the quality/time knee on Apple
# silicon (~15 min); the env vars exist for captures that deserve more.
TRAIN_STEPS = 12_000
TRAIN_RESOLUTION = 1280


def brush_binary() -> str | None:
    """Locate the Brush trainer, or None."""
    env = os.environ.get("LOCAISH_BRUSH")
    if env and Path(env).exists():
        return env
    found = shutil.which("brush_app")
    if found:
        return found
    for cand in (
        Path.home() / "tools/brush/brush-app-aarch64-apple-darwin/brush_app",
        Path.home() / "tools/brush/brush_app",
    ):
        if cand.exists():
            return str(cand)
    return None


def read_gaussian_ply(path: str | Path) -> dict[str, np.ndarray]:
    """A standard 3DGS PLY, decoded. Property-order agnostic.

    Brush writes properties in alphabetical order, INRIA in semantic order;
    mapping by header names handles both. Decodes the stored activations:
    scales are logs of standard deviations (scene units), opacity is a
    logit, colour is the degree-0 SH coefficient, the quaternion is stored
    (w, x, y, z) and unnormalised.
    """
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            chunk = f.readline()
            if not chunk:
                raise ValueError(f"{path}: no end_header")
            header += chunk
        lines = header.decode("ascii", "replace").splitlines()
        if not any(l.strip() == "format binary_little_endian 1.0" for l in lines):
            raise ValueError(f"{path}: expected binary_little_endian 1.0")
        n = None
        names: list[str] = []
        types: list[str] = []
        TYPES = {
            "float": "<f4", "float32": "<f4", "double": "<f8",
            "uchar": "u1", "uint8": "u1", "int": "<i4", "uint": "<u4",
        }
        for l in lines:
            t = l.split()
            if t[:2] == ["element", "vertex"]:
                n = int(t[2])
            elif t and t[0] == "property" and n is not None:
                types.append(TYPES[t[1]])
                names.append(t[2])
            elif t and t[0] == "element" and n is not None and t[1] != "vertex":
                break
        if n is None:
            raise ValueError(f"{path}: no vertex element")
        dtype = np.dtype(list(zip(names, types)))
        v = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype, count=n)

    def cols(prefix: str, k: int) -> np.ndarray:
        return np.stack(
            [v[f"{prefix}_{i}"] for i in range(k)], axis=1
        ).astype(np.float64)

    q = cols("rot", 4)
    q /= np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    return {
        "xyz": np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64),
        "rgb01": np.clip(0.5 + SH_C0 * cols("f_dc", 3), 0.0, 1.0),
        "opacity": 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64))),
        "scale": np.exp(cols("scale", 3)),
        "quat": q,
    }


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(N, 4) normalised wxyz -> (N, 3, 3); columns are the Gaussian's axes."""
    w, x, y, z = q.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=1,
    )


def assert_colmap_frame(
    splat_xyz: np.ndarray, sparse_xyz: np.ndarray, centres: np.ndarray
) -> None:
    """Refuse a splat that is not in the COLMAP frame it claims to be.

    A trainer that silently recentres or rescales would poison every metric
    downstream; better no cloud than a confidently wrong one. The splat's
    core must overlap the sparse model's box, and every camera must sit
    within a plausible distance of the splat.
    """
    lo_s, hi_s = (
        np.percentile(splat_xyz, 2, axis=0),
        np.percentile(splat_xyz, 98, axis=0),
    )
    lo_m, hi_m = (
        np.percentile(sparse_xyz, 2, axis=0),
        np.percentile(sparse_xyz, 98, axis=0),
    )
    span = float(np.linalg.norm(hi_m - lo_m))
    if span <= 0:
        raise ValueError("sparse model has no extent")
    centre_off = float(
        np.linalg.norm(((lo_s + hi_s) - (lo_m + hi_m)) / 2.0)
    )
    if centre_off > 0.5 * span:
        raise ValueError(
            f"splat centre is {centre_off:.2f} units from the sparse model's "
            f"({span:.2f} across); the trainer appears to have re-normalised "
            "the scene"
        )
    if len(centres):
        mid = np.median(splat_xyz, axis=0)
        far = float(np.max(np.linalg.norm(centres - mid, axis=1)))
        if far > 3.0 * span:
            raise ValueError("cameras sit implausibly far from the splat")


def sample_surface(
    g: dict[str, np.ndarray],
    *,
    pitch: float,
    k: float = SAMPLE_K,
    opacity_min: float = OPACITY_MIN,
    s1_cap: float,
    cap_per_gaussian: int = CAP_PER_GAUSSIAN,
    max_total: int = 15_000_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sampled surface points from the Gaussians, ~uniform at `pitch`.

    A trained interior Gaussian is a flattened disc: its two major axes span
    a patch of real surface and its smallest sigma is thickness -- which is
    exactly the part not sampled, because thickness is the noise the plane
    solve must not see. Uniform sampling on each disc's k-sigma ellipse, one
    point per pitch^2 of area, scaled by opacity; dim and oversized
    Gaussians (haze, sky) are dropped. All lengths in the splat's own units.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(-g["scale"], axis=1)
    s_sorted = np.take_along_axis(g["scale"], order, axis=1)
    s1, s2 = s_sorted[:, 0], s_sorted[:, 1]
    alpha = g["opacity"]

    keep = (alpha >= opacity_min) & (s1 <= s1_cap)
    idx0 = np.flatnonzero(keep)
    if not len(idx0):
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    s1, s2, alpha = s1[idx0], s2[idx0], alpha[idx0]

    area = np.pi * (k * s1) * (k * s2)
    expected = float((area * alpha).sum()) / (pitch * pitch)
    if expected > max_total:
        pitch = pitch * float(np.sqrt(expected / max_total))
    # probabilistic rounding, not ceil: a ceiling would hand every dust
    # Gaussian one point ("at least one") and the dust outnumbers the
    # surface; stochastic rounding keeps the expectation exact instead
    raw = area * alpha / (pitch * pitch)
    counts = np.clip(
        np.floor(raw + rng.random(len(raw))), 0, cap_per_gaussian
    ).astype(np.int64)

    rep = np.repeat(np.arange(len(idx0)), counts)
    m = len(rep)
    R = _quat_to_rot(g["quat"][idx0])
    Rs = np.take_along_axis(R, order[idx0][:, None, :], axis=2)
    u1, u2 = Rs[:, :, 0], Rs[:, :, 1]

    r = np.sqrt(rng.random(m))
    th = 2.0 * np.pi * rng.random(m)
    a = (r * np.cos(th)) * (k * s1[rep])
    b = (r * np.sin(th)) * (k * s2[rep])
    xyz = g["xyz"][idx0][rep] + a[:, None] * u1[rep] + b[:, None] * u2[rep]
    rgb = (g["rgb01"][idx0][rep] * 255.0 + 0.5).astype(np.uint8)
    return xyz, rgb


def train_splat(
    image_dir: str | Path,
    model_dir: str | Path,
    out_dir: str | Path,
    *,
    steps: int | None = None,
    resolution: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path | None:
    """Train a per-scene splat with Brush; returns the exported PLY path.

    The dataset is assembled from symlinks (Brush wants images/ + sparse/0),
    training runs headless -- Brush prints nothing, so the exported file is
    the completion signal -- and the export is stamped with the settings so
    a cached splat is reused only when they match.
    """
    binary = brush_binary()
    if binary is None:
        return None
    steps = steps or int(os.environ.get("LOCAISH_SPLAT_STEPS", TRAIN_STEPS))
    resolution = resolution or int(
        os.environ.get("LOCAISH_SPLAT_RESOLUTION", TRAIN_RESOLUTION)
    )
    out_dir = Path(out_dir)
    ply = out_dir / f"splat_{steps}.ply"
    stamp = out_dir / "splat_stamp.txt"
    stamp_val = f"steps={steps} resolution={resolution}"
    if ply.exists() and stamp.exists() and stamp.read_text().strip() == stamp_val:
        if progress:
            progress("reusing trained splat")
        return ply

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = out_dir / "dataset"
    (ds / "sparse").mkdir(parents=True, exist_ok=True)
    img_link = ds / "images"
    sparse_link = ds / "sparse" / "0"
    for link, target in ((img_link, Path(image_dir)), (sparse_link, Path(model_dir))):
        if link.is_symlink() or link.exists():
            if link.is_symlink():
                link.unlink()
            else:
                shutil.rmtree(link)
        link.symlink_to(Path(target).resolve())

    if progress:
        progress(
            f"training gaussian splat ({steps:,} steps, ~"
            f"{max(5, steps // 900)} min on this machine)"
        )
    # Brush runs with the splat directory as its working directory, so every
    # path it is handed must be absolute: a studio launched with a relative
    # root would otherwise point it at a dataset that does not exist from
    # there, and it dies before training with an I/O error.
    proc = subprocess.run(
        [
            binary,
            str(ds.resolve()),
            "--total-steps", str(steps),
            "--max-resolution", str(resolution),
            "--export-every", str(steps),
            "--export-path", str(out_dir.resolve()),
            "--export-name", "splat_{iter}.ply",
        ],
        capture_output=True,
        text=True,
        cwd=str(out_dir.resolve()),
        timeout=3 * 3600,
    )
    (out_dir / "train.log").write_text(
        (proc.stdout or "") + "\n" + (proc.stderr or "")
    )
    if proc.returncode != 0 or not ply.exists():
        raise RuntimeError(
            f"splat training failed (exit {proc.returncode}); see "
            f"{out_dir / 'train.log'}"
        )
    stamp.write_text(stamp_val)
    return ply


def densify_splat(
    image_dir: str | Path,
    model_dir: str | Path,
    work_dir: str | Path,
    model: Any,
    up_hint: np.ndarray | None,
    *,
    max_points: int,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray | None:
    """The splat backend end to end: train (or reuse), audit, sample.

    Returns an (M, 6) xyz+rgb array in COLMAP units, like every other
    densifier, or None when Brush is not installed. Raises on a splat that
    fails the frame audit -- the caller treats that like any backend failure
    and falls through to stereo.
    """
    from . import metric as metricmod

    out_dir = Path(work_dir) / "splat"
    ply = train_splat(image_dir, model_dir, out_dir, progress=progress)
    if ply is None:
        return None
    densify_splat.last_ply = str(ply)

    if progress:
        progress("sampling splat surface")
    g = read_gaussian_ply(ply)
    centres = np.stack(
        [-(e[:, :3].T @ e[:, 3]) for e in model.extrinsics]
    ) if len(model.extrinsics) else np.zeros((0, 3))
    assert_colmap_frame(g["xyz"], model.points, centres)

    # pitch and the floater cap are metric; the splat is in COLMAP units, so
    # convert with the same camera-height estimator the scale solve trusts.
    # A provisional factor 20% off moves the sampling pitch 20%, which the
    # analytic budget and the final voxel subsample both absorb.
    f_prov = 1.0
    try:
        est = metricmod.scale_from_camera_height(model.points, centres, up_hint)
        if est is not None and getattr(est, "factor", None):
            f_prov = float(est.factor)
    except Exception:
        pass

    xyz, rgb = sample_surface(
        g,
        pitch=SAMPLE_PITCH_M / f_prov,
        s1_cap=S1_CAP_M / f_prov,
        max_total=4 * max_points,
    )
    # The optimiser also reconstructs whatever it saw through doorways and
    # windows -- rooms beyond the room. Those points are real, but rays to
    # them pass through this room's walls and poison every visibility test
    # downstream, so the sample is cut to a box around where the camera
    # actually walked: the room plus a doorway's depth, nothing further.
    if len(centres):
        margin_xy = 2.5 / f_prov
        lo = centres.min(axis=0) - margin_xy
        hi = centres.max(axis=0) + margin_xy
        span_z = 2.0 / f_prov
        lo[2] = centres[:, 2].min() - span_z
        hi[2] = centres[:, 2].max() + span_z
        inside = np.all((xyz >= lo) & (xyz <= hi), axis=1)
        xyz, rgb = xyz[inside], rgb[inside]
    if len(xyz) < 100_000:
        raise RuntimeError(
            f"splat sampling produced only {len(xyz):,} points; the trained "
            "splat is too thin to stand in for stereo"
        )
    return np.hstack([xyz, rgb.astype(np.float64)])


__all__ = [
    "brush_binary",
    "read_gaussian_ply",
    "sample_surface",
    "train_splat",
    "densify_splat",
]
