"""Reconstruction without a neural network: classical structure from motion.

This is the same job `backend.py` does -- frames in, a dense cloud and camera
poses out -- performed by matching hand-designed features between images and
bundle-adjusting the result. SIFT is from 1999 and bundle adjustment is older
than that. There is no model file, nothing was trained, and every number it
produces traces to a corner detected in an image and a least-squares solve over
reprojection error.

That property is the reason this module exists. The Agentic Cinema rules permit
only Google Cloud AI tools and prohibit "other AI models, agent frameworks, or
AI APIs ... regardless of vendor", while explicitly allowing open-source non-AI
software. A pretrained depth network is an AI model whoever wrote it; SIFT is
not. This path is compliant by construction rather than by argument.

**Classical SfM needs frames that chain.** That is the one thing it will not
forgive, and it is where a neural reconstruction is simply better. Matching two
views of a blank painted wall taken a second apart finds nothing, so a sweep
sampled sparsely in time fragments into disconnected pieces: measured on a real
capture, 72 frames produced four fragments whose largest held 28 of them, while
251 frames of the same clip registered 240 into a single model with a mean
reprojection error under a pixel. The frames must be dense enough in time to
carry correspondences from one to the next, which for a 60 fps phone clip is not
a hardship -- it is a decode setting.

**The dense stage wants a GPU.** COLMAP's PatchMatch stereo is CUDA-only, which
is the right answer on the hosted deployment and unavailable on a Mac, so there
is a second path here using OpenCV's semi-global block matching on the CPU.
Both are classical; they differ in quality and speed, not in what they are.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Frames per second pulled from the clip for the classical path. Far denser than
# the neural path needs, because correspondence is the binding constraint: see
# the module docstring for what happens when this is too low.
CLASSICAL_FPS = 8.0

# Long side, in pixels, that frames are decoded to for matching. SIFT wants
# resolution -- it is finding corners a few pixels across -- and 1600 keeps
# enough of them on a plain wall without making the match quadratically slow.
CLASSICAL_LONG_SIDE = 1600

# Views on either side that each frame is matched against. Sequential rather
# than exhaustive because a walk revisits its neighbours in time, and quadratic
# overlap adds the powers-of-two jumps that catch a loop closing.
SEQUENTIAL_OVERLAP = 20


class ColmapError(RuntimeError):
    """COLMAP is missing, or refused to reconstruct this sweep."""


@dataclass
class SparseModel:
    """A solved sparse reconstruction: who was where, and what they saw."""

    names: list[str]                 # image filename per registered view
    extrinsics: np.ndarray           # (N, 3, 4) world-to-camera
    intrinsics: np.ndarray           # (N, 3, 3)
    points: np.ndarray               # (M, 3)
    colors: np.ndarray               # (M, 3) uint8
    errors: np.ndarray               # (M,) mean reprojection error, px
    track_lengths: np.ndarray        # (M,) how many views saw each point
    image_size: tuple[int, int]      # (width, height) of the undistorted frames
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.names)

    def summary(self) -> dict:
        return {
            "registered": len(self.names),
            "points": int(len(self.points)),
            "mean_track_length": float(self.track_lengths.mean()) if len(self.track_lengths) else 0.0,
            "mean_reprojection_px": float(self.errors.mean()) if len(self.errors) else 0.0,
        }


def executable() -> str:
    """Locate the COLMAP binary, or explain how to get it."""
    exe = shutil.which("colmap")
    if not exe:
        raise ColmapError(
            "colmap is not on PATH. Install with `brew install colmap` (macOS) "
            "or `apt install colmap` (Debian). It is the classical "
            "structure-from-motion engine the non-neural video path is built on."
        )
    return exe


def supports_cuda() -> bool:
    """Whether this COLMAP build can run its dense stereo stage.

    COLMAP prints its CUDA status in the banner of every command, which is a
    more reliable test than looking for a GPU: a machine can have a GPU and a
    COLMAP compiled without it, and it is the build that decides.
    """
    try:
        out = subprocess.run(
            [executable(), "-h"], capture_output=True, text=True, timeout=30
        ).stdout
    except (ColmapError, OSError, subprocess.SubprocessError):
        return False
    return "without CUDA" not in out


def _run(args: list[str], log: Path, step: str) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(args, capture_output=True, text=True)
    log.write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        raise ColmapError(f"colmap {step} failed: {' / '.join(tail)[:400]}")


def run_sfm(
    image_dir: str | Path,
    work_dir: str | Path,
    *,
    overlap: int = SEQUENTIAL_OVERLAP,
    progress=None,
) -> Path:
    """Feature-extract, match and map. Returns the sparse model directory.

    Sequential matching, not exhaustive: a video's correspondences are between
    frames near each other in time, and matching all pairs of 250 frames costs
    thirty thousand comparisons to find what twenty neighbours already found.
    """
    exe = executable()
    image_dir, work_dir = Path(image_dir), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    database = work_dir / "database.db"
    sparse = work_dir / "sparse"
    if database.exists():
        database.unlink()
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)

    if progress:
        progress("colmap features")
    _run(
        [
            exe, "feature_extractor",
            "--database_path", str(database),
            "--image_path", str(image_dir),
            # One physical camera shot the whole clip, so its intrinsics are one
            # unknown solved from every frame at once rather than N unknowns
            # solved from one frame each.
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "SIMPLE_RADIAL",
        ],
        work_dir / "features.log",
        "feature_extractor",
    )

    if progress:
        progress("colmap matching")
    _run(
        [
            exe, "sequential_matcher",
            "--database_path", str(database),
            "--SequentialMatching.overlap", str(overlap),
            "--SequentialMatching.quadratic_overlap", "1",
        ],
        work_dir / "matching.log",
        "sequential_matcher",
    )

    if progress:
        progress("colmap mapping")
    _run(
        [
            exe, "mapper",
            "--database_path", str(database),
            "--image_path", str(image_dir),
            "--output_path", str(sparse),
        ],
        work_dir / "mapping.log",
        "mapper",
    )

    models = sorted(p for p in sparse.iterdir() if p.is_dir())
    if not models:
        raise ColmapError(
            "colmap registered no images at all. The sweep has too little "
            "texture or too little parallax for feature matching -- see "
            "CAPTURE.md on walking rather than panning."
        )
    return _largest_model(models)


def _largest_model(models: list[Path]) -> Path:
    """The sub-model with the most registered views.

    A fragmented reconstruction is normal and is not by itself a failure: a
    sweep that pauses on a blank wall breaks the chain, and COLMAP honestly
    reports two models rather than joining them on a guess. Taking the largest
    is right -- the fragments are in unrelated coordinate frames and cannot be
    merged without correspondences that, by definition, were not found.
    """
    best, best_n = models[0], -1
    for m in models:
        n = len(_read_images_text(m).get("ids", []))
        if n > best_n:
            best, best_n = m, n
    return best


def read_model(model_dir: str | Path, *, exe: str | None = None) -> SparseModel:
    """Load a COLMAP sparse model, converting to text first.

    Text rather than the binary format on purpose: `model_converter` is a
    supported entry point and parsing its output is twenty lines, where a binary
    reader is a hundred lines that has to track upstream struct changes to stay
    correct.
    """
    model_dir = Path(model_dir)
    exe = exe or executable()
    text_dir = model_dir.parent / f"{model_dir.name}_txt"
    text_dir.mkdir(parents=True, exist_ok=True)
    if not (text_dir / "images.txt").exists():
        _run(
            [
                exe, "model_converter",
                "--input_path", str(model_dir),
                "--output_path", str(text_dir),
                "--output_type", "TXT",
            ],
            text_dir / "convert.log",
            "model_converter",
        )

    cameras = _read_cameras_text(text_dir / "cameras.txt")
    images = _read_images_text(text_dir)
    points, colors, errors, tracks = _read_points_text(text_dir / "points3D.txt")

    order = np.argsort(images["names"])
    names = [images["names"][i] for i in order]
    quats = images["quats"][order]
    trans = images["trans"][order]
    cam_ids = [images["cam_ids"][i] for i in order]

    extr = np.zeros((len(names), 3, 4))
    intr = np.zeros((len(names), 3, 3))
    for i, (q, t, cid) in enumerate(zip(quats, trans, cam_ids)):
        extr[i, :, :3] = _quat_to_rotation(q)
        extr[i, :, 3] = t
        intr[i] = cameras[cid]["K"]

    size = (0, 0)
    if cam_ids:
        first = cameras[cam_ids[0]]
        size = (int(first["width"]), int(first["height"]))

    return SparseModel(
        names=names,
        extrinsics=extr,
        intrinsics=intr,
        points=points,
        colors=colors,
        errors=errors,
        track_lengths=tracks,
        image_size=size,
    )


# ---------------------------------------------------------------------------
# text-model parsing
# ---------------------------------------------------------------------------


def _lines(path: Path):
    if not path.exists():
        raise ColmapError(f"colmap model is missing {path.name}")
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line


def _read_cameras_text(path: Path) -> dict:
    """CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]."""
    out: dict[int, dict] = {}
    for line in _lines(path):
        parts = line.split()
        cid, model = int(parts[0]), parts[1]
        width, height = int(parts[2]), int(parts[3])
        p = [float(v) for v in parts[4:]]
        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            f, cx, cy = p[0], p[1], p[2]
            fx = fy = f
        elif model in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        else:
            raise ColmapError(f"unsupported colmap camera model {model!r}")
        out[cid] = {
            "width": width,
            "height": height,
            "K": np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
        }
    return out


def _read_images_text(text_dir: Path) -> dict:
    """IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME, then a points line."""
    path = Path(text_dir) / "images.txt"
    if not path.exists():
        return {"ids": [], "names": [], "quats": np.zeros((0, 4)), "trans": np.zeros((0, 3)), "cam_ids": []}
    ids, names, quats, trans, cam_ids = [], [], [], [], []
    for i, line in enumerate(_lines(path)):
        if i % 2:            # every second line lists the 2D observations
            continue
        p = line.split()
        ids.append(int(p[0]))
        quats.append([float(v) for v in p[1:5]])
        trans.append([float(v) for v in p[5:8]])
        cam_ids.append(int(p[8]))
        names.append(p[9])
    return {
        "ids": ids,
        "names": names,
        "quats": np.array(quats, dtype=np.float64).reshape(-1, 4),
        "trans": np.array(trans, dtype=np.float64).reshape(-1, 3),
        "cam_ids": cam_ids,
    }


def _read_points_text(path: Path):
    """POINT3D_ID X Y Z R G B ERROR TRACK[]."""
    xyz, rgb, err, track = [], [], [], []
    for line in _lines(path):
        p = line.split()
        xyz.append([float(v) for v in p[1:4]])
        rgb.append([int(v) for v in p[4:7]])
        err.append(float(p[7]))
        track.append((len(p) - 8) // 2)
    return (
        np.array(xyz, dtype=np.float64).reshape(-1, 3),
        np.array(rgb, dtype=np.uint8).reshape(-1, 3),
        np.array(err, dtype=np.float64).reshape(-1),
        np.array(track, dtype=np.int64).reshape(-1),
    )


def _quat_to_rotation(q: np.ndarray) -> np.ndarray:
    """COLMAP stores rotation as (qw, qx, qy, qz), world-to-camera."""
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
