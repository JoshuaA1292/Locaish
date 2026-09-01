"""Turning solved camera poses into a dense cloud, without a neural network.

Structure from motion gives back a few tens of thousands of points -- the
corners it could match -- and a room twin needs a few million. Filling in
between them is the dense stage, and it is pure photometric geometry: for every
pixel of one view, search along the corresponding line in another view for the
patch that looks the same, and the disparity where it matches gives the depth.

Several implementations, because the good one needs hardware that a laptop
does not have.

**PatchMatch stereo**, COLMAP's own, is the best of them by a wide margin: it
optimises depth *and* surface normal per pixel, propagates good hypotheses
between neighbours, and enforces consistency across many views at once. It is
also CUDA-only, which makes it the right answer on a GPU host, the reason the
`remote` module can ship this one stage to a GCP GPU instance, and no answer
at all on a bare Mac.

**OpenMVS** (`densify_openmvs`) is the same family of algorithm implemented
for the CPU, and the strongest local option on a machine without CUDA.

**Semi-global block matching**, from OpenCV, is the CPU floor. It rectifies
one pair at a time so the search is along image rows, matches blocks along those
rows, and regularises the result along eight directions through the image. It is
a 2005 algorithm, it runs anywhere, and it is markedly weaker on the textureless
walls that rooms are mostly made of -- there is nothing to match, and honest
block matching returns nothing rather than inventing a surface.

Neither is an AI model. Both are search and optimisation over pixel
similarities, with no trained parameters anywhere.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

# Disparity search settings for the CPU path. The window is large-ish because
# interior walls carry very low-contrast texture and a small window matches
# noise; the penalties are OpenCV's documented defaults scaled by window area,
# which is what makes the regularisation strength independent of that choice.
SGBM_WINDOW = 7
SGBM_MIN_DISPARITY = 0
SGBM_NUM_DISPARITIES = 128       # must be divisible by 16
SGBM_UNIQUENESS = 10
SGBM_SPECKLE_WINDOW = 128
SGBM_SPECKLE_RANGE = 2

# A stereo pair needs the cameras far enough apart to triangulate and close
# enough to still see the same surfaces. Expressed as a fraction of how far away
# the scene is, so it adapts to a cupboard and to a hall.
# Short on purpose: block matching assumes near-fronto-parallel views, and a
# baseline past a quarter of the depth warps perspective enough that the match
# itself fails. The price of short baselines -- centimetre depth noise -- is
# paid downstream by multi-pair fusion instead.
MIN_BASELINE_RATIO = 0.04
MAX_BASELINE_RATIO = 0.25

# Depths outside this multiple of the sparse model's own median depth are
# discarded: triangulation from a short baseline goes wild at long range, and
# those are the points that would otherwise become a fog around the room.
MAX_DEPTH_RATIO = 3.0

# A disparity is only kept where the image actually has something to match:
# below this Sobel gradient magnitude the pixel is blank wall, and whatever
# SGBM returned there is regularisation, not measurement.
TEXTURE_MIN_GRADIENT = 10.0

# A point survives fusion only if the voxel it falls in was produced by at
# least this many *different* stereo pairs. A real surface is seen again and
# again as the camera walks past; a mismatch is seen once. This is the same
# idea as COLMAP's multi-view geometric consistency, done at fusion time.
CONFIRM_PAIRS = 2
CONFIRM_VOXEL_DEPTH_RATIO = 0.02


def densify_patchmatch(
    image_dir: str | Path,
    model_dir: str | Path,
    work_dir: str | Path,
    *,
    max_image_size: int = 1600,
    progress=None,
) -> np.ndarray:
    """COLMAP's PatchMatch stereo. Requires a CUDA build. Returns (N, 6) xyz+rgb.

    Three commands in sequence, and the first is not optional: PatchMatch works
    on undistorted, pinhole images, and feeding it the radial model that SfM
    solved would put every depth slightly off in a way that grows toward the
    frame edges.
    """
    from .colmap import ColmapError, executable

    exe = executable()
    work_dir = Path(work_dir)
    dense = work_dir / "dense"
    if dense.exists():
        shutil.rmtree(dense)
    dense.mkdir(parents=True)

    def run(args, step):
        proc = subprocess.run(args, capture_output=True, text=True)
        (work_dir / f"{step}.log").write_text(proc.stdout + "\n" + proc.stderr)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            raise ColmapError(f"colmap {step} failed: {' / '.join(tail)[:400]}")

    if progress:
        progress("colmap undistort")
    run([
        exe, "image_undistorter",
        "--image_path", str(image_dir),
        "--input_path", str(model_dir),
        "--output_path", str(dense),
        "--output_type", "COLMAP",
        "--max_image_size", str(max_image_size),
    ], "undistort")

    if progress:
        progress("colmap patchmatch stereo")
    run([
        exe, "patch_match_stereo",
        "--workspace_path", str(dense),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
    ], "patchmatch")

    if progress:
        progress("colmap stereo fusion")
    fused = dense / "fused.ply"
    run([
        exe, "stereo_fusion",
        "--workspace_path", str(dense),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(fused),
    ], "fusion")

    from ..formats.ply import read_ply

    scan = read_ply(fused)
    xyz = scan.points.xyz
    rgb = scan.points.rgb
    if rgb is None:
        rgb = np.full((len(xyz), 3), 200, dtype=np.uint8)
    return np.hstack([xyz, rgb.astype(np.float64)])


def openmvs_binary() -> str | None:
    """Locate OpenMVS's DensifyPointCloud, or None if it is not installed.

    OpenMVS is classical patch-based multi-view stereo -- depth *and* normal
    per pixel, propagated between neighbours, checked across many views -- the
    same family of algorithm as COLMAP's CUDA PatchMatch, implemented for the
    CPU. No model file, nothing trained. Where it is available it replaces the
    block-matching fallback outright, because it is better at everything the
    fallback is weak at.
    """
    found = shutil.which("DensifyPointCloud")
    if found:
        return found
    for cand in (
        Path.home() / "tools/openMVS/out/bin/DensifyPointCloud",
        Path.home() / "tools/openMVS/build/bin/DensifyPointCloud",
        Path("/usr/local/bin/OpenMVS/DensifyPointCloud"),
        Path("/usr/local/bin/DensifyPointCloud"),
    ):
        if cand.exists():
            return str(cand)
    return None


def densify_openmvs(
    image_dir: str | Path,
    model_dir: str | Path,
    work_dir: str | Path,
    *,
    binary: str | None = None,
    resolution_level: int | None = None,
    progress=None,
) -> np.ndarray:
    """OpenMVS patch-match stereo over the COLMAP model. Returns (N, 6) xyz+rgb.

    Four steps: undistort to a pinhole COLMAP workspace (OpenMVS, like every
    MVS, assumes no radial distortion), convert the workspace to an .mvs scene,
    densify, then filter the dense cloud on visibility.

    The settings are tuned for *reliability on a hand-held indoor sweep*, not
    for detail, because the frames do not carry reliable detail: they are
    low-light, motion-blurred, and mostly pointed at painted walls. That
    reverses two earlier choices. `resolution_level` defaults to 1 (half
    resolution) rather than 0 -- matching blurry 2400 px frames at full
    resolution buys correlated noise, not surfaces, and costs double the
    runtime. And fusion demands three agreeing views per point rather than
    OpenMVS's permissive default, plus three geometric-consistency iterations,
    because a mismatched patch is seen once and a wall is seen thirty times --
    the extra agreement is what removes the speckle fog that a 2-view fusion
    lets through. `LOCAISH_MVS_LEVEL=0` restores full-resolution matching for
    a tripod-quality capture that can actually use it.
    """
    from .colmap import ColmapError, executable

    binary = binary or openmvs_binary()
    if binary is None:
        raise ColmapError("OpenMVS's DensifyPointCloud is not installed")
    # Absolute, unconditionally: the OpenMVS steps run with the dense folder as
    # their working directory (they write log and depth-map files beside the
    # scene), so a relative input path would resolve against the wrong base.
    image_dir = Path(image_dir).resolve()
    model_dir = Path(model_dir).resolve()
    work_dir = Path(work_dir).resolve()
    if resolution_level is None:
        # Each level halves the matching resolution and quarters the work.
        # Level 1 (1200 px from 2400 px frames) is the default: on hand-held
        # low-light footage, level 0 matches blur against blur and returns
        # noise dressed as density, at twice the runtime. LOCAISH_MVS_LEVEL=0
        # is for captures sharp enough to deserve it.
        resolution_level = int(os.environ.get("LOCAISH_MVS_LEVEL", "1"))
    # How many neighbour views weigh in on each depth map. OpenMVS's default
    # is 5; 8 buys measurably more floor and ceiling -- the surfaces a walking
    # sweep only ever sees at grazing angles, where any single pair is weak
    # and the extra baselines are what push a match over the line.
    number_views = int(os.environ.get("LOCAISH_MVS_VIEWS", "8"))
    # Depth estimates that survive fusion must agree across this many views.
    # This is the single most effective fog killer in the stage: a hallucinated
    # patch match agrees with nothing, a wall agrees with every frame that
    # walked past it.
    number_views_fuse = int(os.environ.get("LOCAISH_MVS_VIEWS_FUSE", "3"))
    # Extra rounds of geometric-consistency refinement between depth maps.
    geometric_iters = int(os.environ.get("LOCAISH_MVS_GEOMETRIC_ITERS", "3"))
    interface = str(Path(binary).parent / "InterfaceCOLMAP")

    exe = executable()
    work_dir = Path(work_dir)
    dense = work_dir / "dense"
    # A finished OpenMVS workspace is minutes of depth estimation, and the
    # depth maps in it feed the mining stage downstream; a cache miss in the
    # *post-processing* must not throw it away. The stamp records the knobs
    # that shape the stereo itself -- re-run only when one of those changed.
    stamp_path = dense / "densify_stamp.json"
    stamp = {
        "resolution_level": resolution_level,
        "number_views": number_views,
        "number_views_fuse": number_views_fuse,
        "geometric_iters": geometric_iters,
    }
    if stamp_path.exists():
        for cand in ("scene_dense_filtered.ply", "scene_dense.ply"):
            cached = dense / cand
            if cached.exists():
                try:
                    if json.loads(stamp_path.read_text()) == stamp:
                        if progress:
                            progress("reusing dense stereo")
                        return _read_fused(cached)
                except (ValueError, OSError):
                    pass
                break
    if dense.exists():
        shutil.rmtree(dense)
    dense.mkdir(parents=True)

    def run(args, step, cwd=None):
        proc = subprocess.run(args, capture_output=True, text=True, cwd=cwd)
        (work_dir / f"{step}.log").write_text(proc.stdout + "\n" + proc.stderr)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            raise ColmapError(f"{step} failed: {' / '.join(tail)[:400]}")

    if progress:
        progress("undistort")
    run([
        exe, "image_undistorter",
        "--image_path", str(image_dir),
        "--input_path", str(model_dir),
        "--output_path", str(dense),
        "--output_type", "COLMAP",
    ], "undistort")

    if progress:
        progress("openmvs interface")
    run([
        interface, "-i", str(dense), "-o", str(dense / "scene.mvs"),
        "--image-folder", str(dense / "images"),
    ], "interface", cwd=str(dense))

    if progress:
        progress("openmvs densify")
    densify_args = [
        binary, str(dense / "scene.mvs"),
        "-o", str(dense / "scene_dense.mvs"),
        "--resolution-level", str(resolution_level),
        "--number-views", str(number_views),
        "--number-views-fuse", str(number_views_fuse),
        "--geometric-iters", str(geometric_iters),
        "-w", str(dense),
    ]
    try:
        run(densify_args, "densify", cwd=str(dense))
    except ColmapError as exc:
        # An older OpenMVS build rejects the whole command line over one flag
        # it has never heard of; the geometric refinement is the only recent
        # one here, and losing it beats losing the stage.
        if "geometric-iters" not in str(exc):
            raise
        densify_args.remove("--geometric-iters")
        densify_args.remove(str(geometric_iters))
        run(densify_args, "densify", cwd=str(dense))

    fused = dense / "scene_dense.ply"
    if not fused.exists():
        raise ColmapError("OpenMVS produced no dense cloud")

    # Second pass: OpenMVS's own visibility filter, which deletes points that
    # the depth maps of the views that should see them contradict. Best-effort
    # -- a filter that fails leaves the unfiltered cloud, never no cloud.
    if progress:
        progress("openmvs filter")
    try:
        run([
            binary, str(dense / "scene_dense.mvs"),
            "--filter-point-cloud", "1",
            "-o", str(dense / "scene_dense_filtered.mvs"),
            "-w", str(dense),
        ], "filter", cwd=str(dense))
        filtered = dense / "scene_dense_filtered.ply"
        if filtered.exists():
            fused = filtered
    except ColmapError:
        pass

    stamp_path.write_text(json.dumps(stamp))
    return _read_fused(fused)


def _read_fused(fused: Path) -> np.ndarray:
    """The fused cloud as the (N, 6) xyz+rgb array the caller merges."""
    from ..formats.ply import read_ply

    scan = read_ply(fused)
    xyz = scan.points.xyz
    rgb = scan.points.rgb
    if rgb is None:
        rgb = np.full((len(xyz), 3), 200, dtype=np.uint8)
    return np.hstack([xyz, rgb.astype(np.float64)])


def densify_sgbm(
    model,
    image_dir: str | Path,
    *,
    max_pairs: int | None = None,
    scale: float = 0.5,
    progress=None,
) -> tuple[np.ndarray, list[str]]:
    """Semi-global block matching over consecutive pairs. CPU, works anywhere.

    Each registered view is matched against a later one chosen for baseline --
    far enough to triangulate, near enough to overlap -- and the resulting
    disparity becomes a depth map, which becomes world points.

    Returns `(points_rgb, warnings)` where `points_rgb` is (N, 6).
    """
    import cv2

    image_dir = Path(image_dir)
    warnings: list[str] = []
    centres = _camera_centres(model.extrinsics)
    depths = _median_scene_depth(model)
    if not np.isfinite(depths) or depths <= 0:
        return np.zeros((0, 6)), ["the sparse model has no depth to work from"]

    pairs = _choose_pairs(centres, depths)
    if not pairs:
        return np.zeros((0, 6)), [
            "no pair of views had enough baseline to triangulate; the sweep "
            "rotated rather than moved"
        ]
    if max_pairs:
        pairs = pairs[:max_pairs]

    matcher = cv2.StereoSGBM_create(
        minDisparity=SGBM_MIN_DISPARITY,
        numDisparities=SGBM_NUM_DISPARITIES,
        blockSize=SGBM_WINDOW,
        P1=8 * 3 * SGBM_WINDOW**2,
        P2=32 * 3 * SGBM_WINDOW**2,
        disp12MaxDiff=1,
        uniquenessRatio=SGBM_UNIQUENESS,
        speckleWindowSize=SGBM_SPECKLE_WINDOW,
        speckleRange=SGBM_SPECKLE_RANGE,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    out: list[np.ndarray] = []
    pair_of: list[np.ndarray] = []
    for n, (i, j) in enumerate(pairs):
        if progress and n % 10 == 0:
            progress(f"stereo {n + 1}/{len(pairs)}")
        got = _pair_points(
            cv2, matcher, model, image_dir, i, j, scale=scale, max_depth=depths * MAX_DEPTH_RATIO
        )
        if got is not None and len(got):
            out.append(got)
            pair_of.append(np.full(len(got), n, dtype=np.int32))

    if not out:
        return np.zeros((0, 6)), warnings + [
            "block matching found no consistent depth in any pair; the surfaces "
            "in this sweep carry too little texture for photometric matching"
        ]

    pts = np.concatenate(out)
    fused, kept = _fuse_across_pairs(
        pts, np.concatenate(pair_of), voxel=depths * CONFIRM_VOXEL_DEPTH_RATIO
    )
    warnings.append(
        f"{len(pts) - kept:,} of {len(pts):,} stereo samples were seen by only "
        "one pair and discarded; the rest were fused to one point per voxel, "
        "because a block matcher's depth is noisy at the centimetre scale and "
        "averaging its repeated sightings of a surface is what recovers it"
    )
    return fused, warnings


def _fuse_across_pairs(pts: np.ndarray, pair_id: np.ndarray, *, voxel: float):
    """Fuse (N, 6) xyzrgb samples into per-voxel centroids, multi-pair confirmed.

    Two jobs in one pass, both classical. *Confirmation*: a voxel survives only
    if points from >= CONFIRM_PAIRS distinct stereo pairs landed in it -- a real
    surface is seen again and again as the camera walks past, a mismatch once.
    *Fusion*: the survivors are averaged per voxel, which divides the matcher's
    depth noise by the root of the sample count and leaves a cloud whose point
    spacing and noise are the same scale -- the property every downstream
    plane-fitting threshold quietly assumes.

    Returns (fused_points, raw_samples_kept).
    """
    if voxel <= 0 or len(pts) == 0:
        return pts, len(pts)
    key = np.floor(pts[:, :3] / voxel).astype(np.int64)
    vox_all, inverse = np.unique(key, axis=0, return_inverse=True)
    vp = np.unique(np.column_stack([inverse, pair_id]), axis=0)
    pairs_per_voxel = np.bincount(vp[:, 0], minlength=len(vox_all))
    good = pairs_per_voxel >= CONFIRM_PAIRS

    keep = good[inverse]
    inv = inverse[keep]
    sel = pts[keep]
    counts = np.bincount(inv, minlength=len(vox_all)).astype(np.float64)
    counts[counts == 0] = 1.0
    fused = np.empty((len(vox_all), 6), dtype=np.float64)
    for c in range(6):
        fused[:, c] = np.bincount(inv, weights=sel[:, c], minlength=len(vox_all)) / counts
    return fused[good], int(keep.sum())


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _camera_centres(extrinsics: np.ndarray) -> np.ndarray:
    e = np.asarray(extrinsics, dtype=np.float64)
    return np.einsum("nij,nj->ni", e[:, :, :3].transpose(0, 2, 1), -e[:, :, 3])


def _median_scene_depth(model) -> float:
    """How far the room is from the cameras, in model units."""
    if len(model.points) == 0:
        return float("nan")
    e = model.extrinsics[len(model.extrinsics) // 2]
    cam = (e[:, :3] @ model.points.T).T + e[:, 3]
    z = cam[:, 2]
    z = z[np.isfinite(z) & (z > 0)]
    return float(np.median(z)) if len(z) else float("nan")


def _choose_pairs(centres: np.ndarray, depth: float) -> list[tuple[int, int]]:
    """For each view, the nearest later view with a usable baseline.

    Searching forward rather than taking a fixed stride is what makes this work
    on a real walk: someone pauses, and for those seconds consecutive frames
    have no baseline at all while frames a second later have plenty.
    """
    lo, hi = MIN_BASELINE_RATIO * depth, MAX_BASELINE_RATIO * depth
    pairs: list[tuple[int, int]] = []
    n = len(centres)
    for i in range(n - 1):
        for j in range(i + 1, min(i + 40, n)):
            b = float(np.linalg.norm(centres[j] - centres[i]))
            if b < lo:
                continue
            if b > hi:
                break
            pairs.append((i, j))
            break
    return pairs


def _pair_points(cv2, matcher, model, image_dir: Path, i: int, j: int, *, scale: float, max_depth: float):
    """Rectify one pair, match it, and lift the disparity into world points.

    The pair is ordered so the second camera sits to the *right* of the first
    after rectification. Block matchers search positive disparities only, so a
    pair whose baseline points the other way matches pure noise -- every true
    correspondence has negative disparity and is unreachable. A walk produces
    both orderings about equally, which without this normalisation silently
    discards half the sweep and poisons the rest.
    """
    for a, b in ((i, j), (j, i)):
        got = _rectify(cv2, model, image_dir, a, b, scale)
        if got is None:
            continue
        proj2, rest = got
        # CALIB_ZERO_DISPARITY puts the signed baseline in proj2's third
        # column: negative x means "second camera to the right", which is the
        # geometry the matcher can search. A dominant y-term is a vertical
        # baseline -- someone raising the phone -- and no horizontal matcher
        # can use that pair at all.
        if abs(proj2[1, 3]) > abs(proj2[0, 3]):
            return None
        if proj2[0, 3] < 0:
            break
    else:
        return None
    img1, img2, rect1, r1w, t1, q, map1, map2 = rest

    warp1 = cv2.remap(img1, map1[0], map1[1], cv2.INTER_LINEAR)
    warp2 = cv2.remap(img2, map2[0], map2[1], cv2.INTER_LINEAR)

    gray1 = cv2.cvtColor(warp1, cv2.COLOR_BGR2GRAY)
    disp = matcher.compute(gray1, cv2.cvtColor(warp2, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0

    valid = disp > (SGBM_MIN_DISPARITY + 0.5)
    # Keep disparity only where the image had something to match on. On a
    # blank wall SGBM's smoothness term invents a plausible-looking surface;
    # honest is returning nothing there and letting the carve-based completion
    # label the gap, exactly as it does for a wall nobody filmed.
    gx = cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray1, cv2.CV_32F, 0, 1, ksize=3)
    valid &= np.hypot(gx, gy) >= TEXTURE_MIN_GRADIENT
    if valid.sum() < 500:
        return None

    pts_rect = cv2.reprojectImageTo3D(disp, q)
    z = pts_rect[..., 2]
    valid &= np.isfinite(z) & (z > 0) & (z < max_depth)
    if valid.sum() < 500:
        return None

    # Rectified camera frame -> the left camera's own frame -> world.
    sel = pts_rect[valid]
    cam_left = (rect1.T @ sel.T).T
    world = (r1w.T @ (cam_left - t1).T).T
    colors = cv2.cvtColor(warp1, cv2.COLOR_BGR2RGB)[valid].astype(np.float64)
    return np.hstack([world, colors])


def _rectify(cv2, model, image_dir: Path, i: int, j: int, scale: float):
    """Load, scale and rectify one ordered pair. Returns (proj2, unpacked state)."""
    img1 = cv2.imread(str(image_dir / model.names[i]), cv2.IMREAD_COLOR)
    img2 = cv2.imread(str(image_dir / model.names[j]), cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        return None
    if scale != 1.0:
        img1 = cv2.resize(img1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = img1.shape[:2]

    k1 = model.intrinsics[i].copy() * scale
    k2 = model.intrinsics[j].copy() * scale
    k1[2, 2] = k2[2, 2] = 1.0
    zero = np.zeros((5, 1))

    r1w, t1 = model.extrinsics[i][:, :3], model.extrinsics[i][:, 3]
    r2w, t2 = model.extrinsics[j][:, :3], model.extrinsics[j][:, 3]
    # Relative pose taking camera i's frame to camera j's. The translation
    # must be a column vector: OpenCV 5's stereoRectify rejects a flat (3,).
    rel_r = r2w @ r1w.T
    rel_t = (t2 - rel_r @ t1).reshape(3, 1)

    try:
        rect1, rect2, proj1, proj2, q, _, _ = cv2.stereoRectify(
            k1, zero, k2, zero, (w, h), rel_r, rel_t,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )
        map1 = cv2.initUndistortRectifyMap(k1, zero, rect1, proj1, (w, h), cv2.CV_32FC1)
        map2 = cv2.initUndistortRectifyMap(k2, zero, rect2, proj2, (w, h), cv2.CV_32FC1)
    except cv2.error:
        return None
    return proj2, (img1, img2, rect1, r1w, t1, q, map1, map2)
