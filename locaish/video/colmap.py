"""Structure from motion: frames in, camera poses and a sparse cloud out.

Features are detected in every frame, matched between frames near each other
in time, verified geometrically, and the whole set is bundle-adjusted into one
model. Every pose that comes out traces to pixel correspondences and a
least-squares solve over reprojection error, which is why the twin can audit
itself downstream: the geometry is the evidence, not a prior.

**Which features.** Two detectors are wired in and the build decides. ALIKED
(a small learned keypoint network, BSD licence, shipped inside COLMAP as an
ONNX model) is the default wherever the installed COLMAP offers it: on the
painted walls rooms are mostly made of it returns four to five times as many
geometrically verified matches per pair as SIFT, from a third as many
keypoints, and it holds a chain across a wider baseline -- which means fewer
frames to match and a solve that breaks less. SIFT remains as the fallback
(`LOCAISH_FEATURES=sift`), and because it is the detector that chains worst
across blank plaster, its frame budget is denser in time and larger in pixels.
Matching is COLMAP's own brute-force descriptor matcher for either detector;
LightGlue (`LOCAISH_MATCHER=lightglue`) is available for ALIKED and is more
selective still, but on a CPU it costs a second a pair, so it is opt-in and
meant for a GPU host.

**Loop closure.** A walk that circles the room must be told, at the matching
stage, that its last frames see what its first frames saw, or the solve
accumulates drift all the way round and the walls bow. COLMAP finds those
pairs with a vocabulary tree over the descriptors. The tree it distributes is
built on SIFT and is useless for ALIKED, so for ALIKED a small tree is built
from the sweep's own descriptors first -- a minute of work that buys a closed
loop.

**Which mapper.** The global mapper (GLOMAP, inside COLMAP 4.x) solves every
rotation jointly and is preferred; the incremental mapper is the fallback when
the global solve registers too little of the sweep.

The dense stage lives in `dense.py` and `learned.py`; this module ends at the
sparse model.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# -- the SIFT budget ----------------------------------------------------------
#
# Frames per second pulled from the clip. SIFT chains correspondences frame to
# frame and loses the chain on a blank wall, so it wants frames dense in time.
CLASSICAL_FPS = 8.0

# Long side, in pixels, that frames are decoded to for matching. SIFT wants
# resolution -- it is finding corners a few pixels across -- and on the blank
# interior walls that break the matching chain, the corners that exist are
# small: paint texture, outlet screws, shadow edges. 2400 finds the features
# 1600 missed, and an unbroken chain is worth the extra matching time, because
# every frame that falls out of the chain is a wall missing from the twin.
CLASSICAL_LONG_SIDE = 2400

# Features extracted per frame. The default 8192 is tuned for photo
# collections; a video frame of a low-texture room needs the budget raised so
# the weak corners on plain surfaces survive the ranking.
SIFT_MAX_FEATURES = 16384

# Views on either side that each frame is matched against. Sequential rather
# than exhaustive because a walk revisits its neighbours in time, and quadratic
# overlap adds the powers-of-two jumps that catch a loop closing.
SEQUENTIAL_OVERLAP = 30

# -- the ALIKED budget --------------------------------------------------------
#
# Measured on a real kitchen sweep (six 2400 px frames, five pairs each):
# SIFT 2682 keypoints and 117 verified matches per pair; ALIKED 908 keypoints
# and 302 verified matches; ALIKED at 2048 keypoints and 1600 px, 524 matches
# per pair with the brute-force matcher at 0.03 s a pair. Fewer frames are
# needed because the chain survives a wider baseline. The frames are still
# decoded at full matching resolution, because they are also the stereo
# stage's input: a first cut at 1600 px and 5 fps halved OpenMVS's working
# resolution and cut its views by a third, and the floor it returned covered
# 15% of the room against 24% from the SIFT budget. ALIKED costs about a
# second a frame at 2400 px on a laptop CPU, which is cheaper than the
# stereo it would otherwise starve.
ALIKED_FPS = 6.0
ALIKED_LONG_SIDE = 2400
ALIKED_MAX_FEATURES = 2048
ALIKED_MIN_SCORE = 0.15
ALIKED_OVERLAP = 20
# Loop-closure tree built from the sweep's own descriptors: visual words and
# the descriptor sample the k-means runs on. Small on purpose -- it only has to
# tell the frames of one room apart, not index the internet.
ALIKED_VOCAB_WORDS = 4096
ALIKED_VOCAB_MAX_DESCRIPTORS = 120_000
LOOP_DETECTION_NUM_IMAGES = 10

# A frame needs this many inliers to join the model. COLMAP's default of 30 is
# right for photo collections; on a sweep, a frame that fails here breaks the
# chain and everything after it lands in a separate fragment, so the threshold
# is lowered -- a marginal registration constrained by its neighbours beats an
# amputated half of the room. Only the incremental fallback mapper reads this.
MAPPER_MIN_INLIERS = 15

# The global mapper (GLOMAP, shipped inside COLMAP 4.x as `global_mapper`) is
# preferred over incremental mapping wherever the build offers it. Two reasons,
# both load-bearing for a room walk. Rotation averaging solves every camera's
# orientation *jointly*, so the loop-closure error of a walk that circles the
# room is distributed around the whole trajectory instead of accumulating to
# the last frame -- which is exactly the drift that bows the walls of a twin
# and shows up in QA's 0.5 m plane-residual tiles. And it is one to two orders
# of magnitude faster, because it never replays the sweep frame by frame.
# Below this fraction of frames registered, the global solve is judged to have
# fallen apart (weak parallax breaks rotation averaging before it breaks
# incremental chaining) and the incremental mapper is run instead.
GLOBAL_MAPPER_MIN_REGISTERED = 0.5

# Stamped into the workspace beside the frame fingerprint, and bumped whenever
# the *solve* itself changes behaviour. Frame identity alone is not enough to
# reuse solved poses: the poses of the same frames under a better solver are
# different poses, and a workspace stamped by the old solver must not be
# mistaken for this one's output. 2: view-graph focal calibration before the
# global mapper. 3: a full bundle adjustment after the global solve.
SOLVE_VERSION = 3

# Vocabulary tree for loop detection: when the walk comes back to where it
# started, this is what tells the matcher to close the loop instead of letting
# drift accumulate to the end of the chain. Downloaded once and cached; the
# pipeline works without it, just less well on captures that circle a room.
# The faiss-format index -- COLMAP moved off flann in 2025, and the legacy
# trees on demuc.de crash the matcher outright.
VOCAB_TREE_URL = (
    "https://github.com/colmap/colmap/releases/download/3.11.1/"
    "vocab_tree_faiss_flickr100K_words32K.bin"
)
VOCAB_TREE_CACHE = "~/.cache/locaish/vocab_tree_faiss_flickr100K_words32K.bin"


class ColmapError(RuntimeError):
    """COLMAP is missing, or refused to reconstruct this sweep."""


# Below this mean resultant length, the frames do not agree which way is up and
# the hint is withheld entirely: a confidently wrong up is worse for the
# pipeline than no up at all, because it would outvote the room's own geometry.
UP_COHERENCE_FLOOR = 0.75


def up_from_cameras(extrinsics: np.ndarray) -> tuple[np.ndarray | None, float, str | None]:
    """Recover which way is up from how the phone was held.

    Nothing in the reconstruction itself knows about gravity -- feature matching
    sees pixels, and a room filmed upside down reconstructs perfectly happily
    upside down. But the *poses* carry the answer for free: a person filming a
    room holds the phone roughly upright, so the camera's own down axis is
    roughly gravity, in every frame, and the frames were solved in a single
    shared world frame. Averaging the per-frame down axes therefore recovers
    gravity directly, without a single assumption about furniture, ceilings or
    what a room looks like.

    The mean resultant length of those directions is returned alongside, and it
    is the whole safety net: it is near 1 when the phone was held consistently
    and collapses toward 0 when it was not, which is exactly when this estimate
    deserves to be ignored.

    With OpenCV extrinsics `x_cam = R x_world + t`, a camera-frame direction d
    maps back to the world as `R^T d`; camera down is +Y, so world down is the
    second row of R.
    """
    if extrinsics is None or len(extrinsics) == 0:
        return None, 0.0, None
    downs = np.asarray(extrinsics, dtype=np.float64)[:, 1, :3]
    norms = np.linalg.norm(downs, axis=1, keepdims=True)
    good = norms[:, 0] > 1e-9
    if good.sum() < 2:
        return None, 0.0, None
    ups = -downs[good] / norms[good]
    mean = ups.mean(axis=0)
    coherence = float(np.linalg.norm(mean))
    if coherence < UP_COHERENCE_FLOOR:
        return (
            None,
            coherence,
            f"the camera was not held in a consistent orientation (frames agree "
            f"on which way is up only {coherence:.0%}), so gravity was left to be "
            "recovered from the room's own geometry",
        )
    return mean / coherence, coherence, None


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


def supports_global_mapper(exe: str | None = None) -> bool:
    """Whether this COLMAP build ships GLOMAP's global mapper.

    GLOMAP was upstreamed into COLMAP (the standalone repository is archived),
    so the test is simply whether `global_mapper` is a registered command.
    """
    try:
        out = subprocess.run(
            [exe or executable(), "help"], capture_output=True, text=True, timeout=30
        )
    except (ColmapError, OSError, subprocess.SubprocessError):
        return False
    return "global_mapper" in (out.stdout + out.stderr)


_ALIKED_SUPPORT: dict[str, bool] = {}


def supports_aliked(exe: str | None = None) -> bool:
    """Whether this COLMAP build carries the ALIKED detector (ONNX runtime).

    Read off the extractor's own help text, which lists the `AlikedExtraction`
    option group only when the build can honour it. Cached per binary: the
    answer is asked for on every solve and never changes.
    """
    try:
        exe = exe or executable()
    except ColmapError:
        return False
    if exe in _ALIKED_SUPPORT:
        return _ALIKED_SUPPORT[exe]
    try:
        out = subprocess.run(
            [exe, "feature_extractor", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        ok = "AlikedExtraction" in (out.stdout + out.stderr)
    except (OSError, subprocess.SubprocessError):
        ok = False
    _ALIKED_SUPPORT[exe] = ok
    return ok


def feature_config(features: str | None = None, matcher: str | None = None) -> dict:
    """Resolve the detector and matcher to use, and the frame budget they want.

    `features` is `auto` (ALIKED when the build has it, else SIFT), `aliked`
    or `sift`; `matcher` is `auto` (brute force), `bruteforce` or `lightglue`.
    Both default from the environment so a deployment can pin them without a
    code change. Asking for ALIKED on a build without it falls back to SIFT
    and says so in the config rather than failing the solve.
    """
    want = (features or os.environ.get("LOCAISH_FEATURES", "auto")).strip().lower()
    want_m = (matcher or os.environ.get("LOCAISH_MATCHER", "auto")).strip().lower()
    note = None
    if want not in ("auto", "aliked", "sift"):
        note = f"unknown LOCAISH_FEATURES={want!r}; using auto"
        want = "auto"
    if want == "auto":
        use = "aliked" if supports_aliked() else "sift"
    elif want == "aliked" and not supports_aliked():
        note = "ALIKED was requested but this COLMAP build has no ONNX support; using SIFT"
        use = "sift"
    else:
        use = want
    if use == "aliked":
        m = "lightglue" if want_m == "lightglue" else "bruteforce"
        return {
            "features": "aliked",
            "matcher": m,
            "fps": ALIKED_FPS,
            "long_side": ALIKED_LONG_SIDE,
            "overlap": ALIKED_OVERLAP,
            "note": note,
        }
    m = "lightglue" if want_m == "lightglue" else "bruteforce"
    return {
        "features": "sift",
        "matcher": m,
        "fps": CLASSICAL_FPS,
        "long_side": CLASSICAL_LONG_SIDE,
        "overlap": SEQUENTIAL_OVERLAP,
        "note": note,
    }


def extraction_args(config: dict) -> list[str]:
    """The feature_extractor options for a config -- one place, so a test can
    pin them and the solve cannot drift from what the docs claim."""
    if config["features"] == "aliked":
        return [
            "--FeatureExtraction.type", "ALIKED_N16ROT",
            "--AlikedExtraction.max_num_features", str(ALIKED_MAX_FEATURES),
            "--AlikedExtraction.min_score", str(ALIKED_MIN_SCORE),
        ]
    return [
        "--FeatureExtraction.type", "SIFT",
        "--SiftExtraction.max_num_features", str(SIFT_MAX_FEATURES),
    ]


def matching_type(config: dict) -> str:
    base = "ALIKED" if config["features"] == "aliked" else "SIFT"
    return f"{base}_LIGHTGLUE" if config["matcher"] == "lightglue" else f"{base}_BRUTEFORCE"


def feature_extractor_command(
    exe: str,
    database: str | Path,
    image_dir: str | Path,
    config: dict,
    *,
    mask_dir: str | Path | None = None,
) -> list[str]:
    """The whole `feature_extractor` invocation, as a list a test can read.

    `mask_dir` holds one `<image name>.png` per frame in COLMAP's convention
    (zero = ignore); with it, keypoints are never detected on the people and
    vehicles that moved through the sweep (see `locaish.meta.masks`).
    """
    cmd = [
        exe, "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(image_dir),
        # One physical camera shot the whole clip, so its intrinsics are one
        # unknown solved from every frame at once rather than N unknowns
        # solved from one frame each.
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
    ]
    if mask_dir is not None:
        cmd += ["--ImageReader.mask_path", str(mask_dir)]
    return cmd + extraction_args(config)


def build_vocab_tree(exe: str, database: Path, out: Path, progress=None) -> Path | None:
    """A loop-closure vocabulary from this sweep's own descriptors.

    COLMAP's distributed trees are built on SIFT descriptors and retrieve
    nothing useful for ALIKED's (measured: zero pairs). Clustering the sweep's
    own descriptors into a few thousand words is cheap and is exactly the
    index loop detection needs: it only has to recognise which earlier frame
    of *this* room a later frame is looking at. Best-effort -- None means
    "match without loop detection", never a failed solve.
    """
    try:
        if progress:
            progress("building loop-closure vocabulary from this sweep")
        _run(
            [
                exe, "vocab_tree_builder",
                "--database_path", str(database),
                "--vocab_tree_path", str(out),
                "--num_visual_words", str(ALIKED_VOCAB_WORDS),
                "--max_num_descriptors", str(ALIKED_VOCAB_MAX_DESCRIPTORS),
            ],
            out.parent / "vocab_tree.log",
            "vocab_tree_builder",
        )
    except ColmapError:
        return None
    return out if out.exists() and out.stat().st_size > 1000 else None


def mapper_used(work_dir: str | Path) -> str | None:
    """Which mapper solved the poses in this workspace, if it recorded one."""
    stamp = Path(work_dir) / "frames_used.json"
    if not stamp.exists():
        return None
    try:
        data = json.loads(stamp.read_text())
    except ValueError:
        return None
    return data.get("mapper") if isinstance(data, dict) else None


def vocab_tree(progress=None) -> Path | None:
    """The loop-detection vocabulary, downloaded once and cached. None if offline.

    150 MB, from the COLMAP author's own distribution. Failure here must never
    fail a reconstruction: loop closure is an upgrade, not a dependency.
    """
    path = Path(VOCAB_TREE_CACHE).expanduser()
    if path.exists() and path.stat().st_size > 5_000_000:
        return path
    try:
        import urllib.request

        if progress:
            progress("fetching loop-closure vocabulary (once)")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(VOCAB_TREE_URL, tmp)
        tmp.rename(path)
        return path
    except Exception:  # noqa: BLE001 - offline is a mode, not an error
        return None


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
    overlap: int | None = None,
    reuse: bool = True,
    config: dict | None = None,
    mask_dir: str | Path | None = None,
    progress=None,
) -> Path:
    """Feature-extract, match and map. Returns the sparse model directory.

    `mask_dir`, when given, is passed to the feature extractor as
    `--ImageReader.mask_path`: a `<frame>.png` per frame, black where the
    detector must not look (movers, see `locaish.meta.masks`). The set of
    masks rides in the reuse stamp beside the frames, so a workspace solved
    without masks is not handed back to a run that has them.

    Sequential matching, not exhaustive: a video's correspondences are between
    frames near each other in time, and matching all pairs of 250 frames costs
    thirty thousand comparisons to find what twenty neighbours already found.

    With `reuse` (the default), a workspace that already holds a solved model
    is returned as-is: the solve is minutes, it is deterministic for the same
    frames and the same detector, and a failure in the *dense* stage should
    not cost re-solving the cameras on retry. Callers who changed the frames
    pass `reuse=False`; a change of detector or matcher is caught by the stamp.
    """
    exe = executable()
    cfg = config or feature_config()
    overlap = int(overlap or cfg["overlap"])
    image_dir, work_dir = Path(image_dir), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    database = work_dir / "database.db"
    sparse = work_dir / "sparse"

    if reuse and database.exists() and sparse.exists():
        # Only a solve of *these exact frames* with *this detector*. Decoding
        # is deterministic for the same video and options, so name+size
        # identity is the right test; modification times are useless here
        # because retries re-extract the frames and stamp them with fresh
        # times.
        stamp = work_dir / "frames_used.json"
        models = sorted(p for p in sparse.iterdir()
                        if p.is_dir() and not p.name.endswith("_txt"))
        stamped = json.loads(stamp.read_text()) if stamp.exists() else None
        # The stamp grew a "mapper" key when the global mapper landed and a
        # "features" key when ALIKED did; a stamp without the latter is a SIFT
        # solve, which is still the right poses for a SIFT config.
        frames_match = (
            isinstance(stamped, dict)
            and stamped.get("frames") == _frames_fingerprint(image_dir)
            and stamped.get("solve_version") == SOLVE_VERSION
            and stamped.get("features", "sift") == cfg["features"]
            and stamped.get("matcher", "bruteforce") == cfg["matcher"]
            and stamped.get("masks") == _masks_fingerprint(mask_dir)
        )
        if frames_match and models and any(_count_registered(m) > 0 for m in models):
            if progress:
                progress("reusing solved camera poses")
            return _largest_model(models)

    if database.exists():
        database.unlink()
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)

    if progress:
        progress(f"colmap features ({cfg['features']})")
    _run(
        feature_extractor_command(exe, database, image_dir, cfg, mask_dir=mask_dir),
        work_dir / "features.log",
        "feature_extractor",
    )

    if progress:
        progress(f"colmap matching ({cfg['matcher']})")
    matching = [
        exe, "sequential_matcher",
        "--database_path", str(database),
        "--SequentialMatching.overlap", str(overlap),
        "--SequentialMatching.quadratic_overlap", "1",
        "--FeatureMatching.type", matching_type(cfg),
    ]
    if cfg["features"] == "aliked":
        vocab = build_vocab_tree(
            exe, database, work_dir / "vocab_tree_aliked.bin", progress=progress
        )
    else:
        vocab = vocab_tree(progress=progress)
    if vocab is not None:
        matching += [
            "--SequentialMatching.loop_detection", "1",
            "--SequentialMatching.loop_detection_num_images", str(LOOP_DETECTION_NUM_IMAGES),
            "--SequentialMatching.vocab_tree_path", str(vocab),
        ]
    _run(matching, work_dir / "matching.log", "sequential_matcher")

    n_frames = len(_frames_fingerprint(image_dir))
    mapper = "incremental"
    models: list[Path] = []
    if supports_global_mapper(exe):
        # Video frames carry no EXIF focal length, and the global mapper's own
        # log says it "depends on reasonably good focal length priors". The
        # view-graph calibrator solves the focals from the two-view geometries
        # already in the database -- cheap next to matching, and focal error
        # is a direct cause of walls bowing, since a wrong focal bends every
        # triangulation consistently outward. Best-effort: a build without
        # the command just maps from the uncalibrated priors as before.
        try:
            if progress:
                progress("colmap view-graph calibration")
            _run(
                [
                    exe, "view_graph_calibrator",
                    "--database_path", str(database),
                ],
                work_dir / "calibration.log",
                "view_graph_calibrator",
            )
        except ColmapError:
            pass
        if progress:
            progress("colmap global mapping (glomap)")
        try:
            _run(
                [
                    exe, "global_mapper",
                    "--database_path", str(database),
                    "--image_path", str(image_dir),
                    "--output_path", str(sparse),
                ],
                work_dir / "mapping_global.log",
                "global_mapper",
            )
        except ColmapError:
            pass  # the log stays on disk; the incremental mapper runs below
        else:
            models = _model_dirs(sparse)
            registered = max((_count_registered(m) for m in models), default=0)
            if registered >= GLOBAL_MAPPER_MIN_REGISTERED * max(n_frames, 1):
                mapper = "global"
                # The global mapper's own refinement is lighter than COLMAP's
                # full bundle adjuster, and a room walk's residual drift lives
                # in exactly the long-range consistency a final joint solve
                # of every pose, point and the shared intrinsics recovers.
                # Best-effort: a build or a model that refuses it keeps the
                # global solve as it was.
                best = _largest_model(models)
                try:
                    if progress:
                        progress("colmap bundle adjustment")
                    _run(
                        [
                            exe, "bundle_adjuster",
                            "--input_path", str(best),
                            "--output_path", str(best),
                            "--BundleAdjustment.refine_focal_length", "1",
                            "--BundleAdjustment.refine_extra_params", "1",
                            "--BundleAdjustment.refine_principal_point", "0",
                        ],
                        work_dir / "bundle.log",
                        "bundle_adjuster",
                    )
                except ColmapError:
                    pass
            else:
                # A global solve that lost half the sweep is not a model to
                # keep -- rotation averaging degrades all at once where the
                # incremental chain merely fragments. Start the sparse
                # directory over so the fallback cannot mix the two.
                models = []
                shutil.rmtree(sparse)
                sparse.mkdir(parents=True)

    if mapper != "global":
        if progress:
            progress("colmap mapping (incremental fallback)")
        _run(
            [
                exe, "mapper",
                "--database_path", str(database),
                "--image_path", str(image_dir),
                "--output_path", str(sparse),
                "--Mapper.abs_pose_min_num_inliers", str(MAPPER_MIN_INLIERS),
            ],
            work_dir / "mapping.log",
            "mapper",
        )
        models = _model_dirs(sparse)

    if not models:
        raise ColmapError(
            "colmap registered no images at all. The sweep has too little "
            "texture or too little parallax for feature matching -- see "
            "CAPTURE.md on walking rather than panning."
        )
    (work_dir / "frames_used.json").write_text(
        json.dumps({
            "frames": _frames_fingerprint(image_dir),
            "mapper": mapper,
            "solve_version": SOLVE_VERSION,
            "features": cfg["features"],
            "matcher": cfg["matcher"],
            "masks": _masks_fingerprint(mask_dir),
        })
    )
    return _largest_model(models)


def _masks_fingerprint(mask_dir: str | Path | None) -> list[list] | None:
    """Name and size of every mask PNG, or None for an unmasked solve."""
    if mask_dir is None:
        return None
    d = Path(mask_dir)
    if not d.is_dir():
        return None
    return sorted([f.name, f.stat().st_size] for f in d.iterdir() if f.suffix.lower() == ".png")


def _model_dirs(sparse: Path) -> list[Path]:
    return sorted(p for p in sparse.iterdir() if p.is_dir() and not p.name.endswith("_txt"))


def _frames_fingerprint(image_dir: Path) -> list[list]:
    """Name and byte size of every frame, sorted -- identity for a decode."""
    return sorted(
        [f.name, f.stat().st_size]
        for f in Path(image_dir).iterdir()
        if f.suffix.lower() in (".jpg", ".png")
    )


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
        n = _count_registered(m)
        if n > best_n:
            best, best_n = m, n
    return best


def _count_registered(model_dir: Path) -> int:
    """How many views a model holds, without converting it to text.

    A fresh mapper output is binary, and `images.bin` starts with the count as
    a little-endian uint64 -- reading eight bytes beats running a converter on
    every fragment just to pick the biggest one.
    """
    binary = model_dir / "images.bin"
    if binary.exists():
        import struct

        with open(binary, "rb") as fh:
            raw = fh.read(8)
        return struct.unpack("<Q", raw)[0] if len(raw) == 8 else 0
    text = model_dir / "images.txt"
    if text.exists():
        return len(_read_images_text(model_dir).get("ids", []))
    return 0


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
