"""The video front-end, checked without running COLMAP or decoding a real sweep.

Everything here runs in a second or two, which is the point: the parts of the
video path that can be *wrong in a way that matters* are not the stereo
matcher's dense output -- that is what it is -- but the arithmetic wrapped
around it. Which frames get chosen. Which way the camera says is up. What
number turns reconstruction units into metres, and how much of an error bar
goes on it.

Each of those is tested against a synthetic case with a known answer, and the
answer is constructed independently of the code under test rather than read
back out of it. A gravity estimate built from cameras tilted 20 degrees has to
come back 20 degrees off, not 0.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from locaish.formats import ScanImport
from locaish.scan.ingest import is_video
from locaish.types import PointCloud
from locaish.video import colmap, frames, metric

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("room.mov", True),
        ("room.MOV", True),
        ("room.mp4", True),
        ("room.MP4", True),
        ("room.mkv", True),
        ("room.ply", False),
        ("room.glb", False),
        ("room.obj", False),
        ("room", False),
    ],
)
def test_video_sources_route_to_reconstruction(name, expected):
    assert is_video(name) is expected


# ---------------------------------------------------------------------------
# frame selection
# ---------------------------------------------------------------------------


def test_bucket_picker_covers_the_whole_timeline():
    """Coverage is the property that must hold even when sharpness is adversarial.

    The scores here rise monotonically, so a picker that simply took the
    sharpest frames would return the last eight and reconstruct the end of the
    sweep only. Spreading the picks is what stops that.
    """
    scores = np.linspace(0.0, 1.0, 200)
    picked = frames._pick_per_bucket(scores, 8)

    assert len(picked) == 8
    assert picked[0] < 25, "the first bucket's frame must come from the start of the video"
    assert picked[-1] > 175, "the last bucket's frame must come from the end of the video"
    gaps = np.diff(picked)
    assert gaps.min() > 0, "picks must be strictly increasing"


def test_bucket_picker_takes_the_sharpest_within_each_bucket():
    scores = np.zeros(100)
    scores[7] = scores[33] = scores[61] = scores[95] = 1.0
    picked = frames._pick_per_bucket(scores, 4)
    assert picked == [7, 33, 61, 95]


def test_bucket_picker_keeps_everything_when_there_is_not_enough():
    assert frames._pick_per_bucket(np.array([0.2, 0.9, 0.4]), 8) == [0, 1, 2]


def test_sharpness_ranks_a_blurred_frame_below_its_original(tmp_path):
    """The measure only has to be monotonic in blur, which is all the picker uses."""
    from PIL import Image, ImageFilter

    rng = np.random.default_rng(0)
    sharp = Image.fromarray(rng.integers(0, 255, (256, 256), dtype=np.uint8))
    blurred = sharp.filter(ImageFilter.GaussianBlur(3))

    a, b = tmp_path / "sharp.png", tmp_path / "blur.png"
    sharp.save(a)
    blurred.save(b)

    assert frames._sharpness(a) > frames._sharpness(b) * 5


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg is not installed")
def test_probe_reads_what_the_container_declares(tmp_path):
    src = _synthetic_video(tmp_path / "clip.mp4", seconds=2, fps=30, size="320x240")
    info = frames.probe(src)

    assert info.width == 320 and info.height == 240
    assert info.fps == pytest.approx(30, abs=0.5)
    assert info.duration_s == pytest.approx(2.0, abs=0.3)


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg is not installed")
def test_extract_frames_returns_the_requested_count_in_order(tmp_path):
    src = _synthetic_video(tmp_path / "clip.mp4", seconds=4, fps=30, size="320x240")
    fs = frames.extract_frames(src, tmp_path / "work", count=6, candidate_fps=6)

    assert len(fs) == 6
    assert all(p.exists() for p in fs.paths)
    assert fs.timestamps == sorted(fs.timestamps)
    assert fs.timestamps[-1] > 2.0, "the selection must reach the end of the clip"
    assert not (tmp_path / "work" / "candidates").exists(), "candidates should be cleaned up"


def test_probe_rejects_a_file_that_is_not_there(tmp_path):
    with pytest.raises(frames.VideoError):
        frames.probe(tmp_path / "nope.mov")


# ---------------------------------------------------------------------------
# gravity from camera poses
# ---------------------------------------------------------------------------


def test_up_is_recovered_from_an_upright_camera_path():
    """A phone held upright, panning around the room, must report world up.

    The extrinsics are built from a known world up rather than derived from the
    function's own convention, so a sign or transpose error in `_up_from_cameras`
    fails this test instead of cancelling out of it.
    """
    world_up = np.array([0.0, 0.0, 1.0])
    extr = np.stack([_extrinsic(yaw, world_up) for yaw in np.linspace(0, 2 * np.pi, 16)])

    up, coherence, note = colmap.up_from_cameras(extr)

    assert note is None
    assert coherence > 0.99
    assert np.dot(up, world_up) > 0.999


def test_a_consistently_tilted_camera_reports_a_tilted_up():
    """The estimate has to carry the operator's tilt, not silently level it."""
    true_up = np.array([0.0, np.sin(np.radians(20)), np.cos(np.radians(20))])
    true_up /= np.linalg.norm(true_up)
    extr = np.stack([_extrinsic(yaw, true_up) for yaw in np.linspace(0, np.pi, 12)])

    up, coherence, _ = colmap.up_from_cameras(extr)

    assert coherence > 0.99
    assert np.degrees(np.arccos(np.clip(np.dot(up, np.array([0.0, 0.0, 1.0])), -1, 1))) == pytest.approx(20, abs=1.0)


def test_an_incoherently_held_camera_withholds_the_hint():
    """Rotating the phone through every orientation must produce no hint at all.

    Withholding beats downweighting here: a confidently wrong up would outvote
    the room's own geometry, and the geometry is the fallback that works.
    """
    rng = np.random.default_rng(3)
    ups = rng.normal(size=(20, 3))
    ups /= np.linalg.norm(ups, axis=1, keepdims=True)
    extr = np.stack([_extrinsic(0.0, u) for u in ups])

    up, coherence, note = colmap.up_from_cameras(extr)

    assert up is None
    assert coherence < colmap.UP_COHERENCE_FLOOR
    assert note and "consistent orientation" in note


def test_scan_import_normalises_a_declared_up():
    scan = ScanImport(points=PointCloud(np.zeros((4, 3))), up_hint=np.array([0.0, 0.0, 7.0]))
    assert np.allclose(scan.up_hint, [0.0, 0.0, 1.0])


def test_scan_import_drops_a_degenerate_up():
    scan = ScanImport(points=PointCloud(np.zeros((4, 3))), up_hint=np.zeros(3))
    assert scan.up_hint is None


# ---------------------------------------------------------------------------
# scale
# ---------------------------------------------------------------------------


def test_declared_metres_can_carry_less_than_total_confidence():
    """A computed unit must not be laundered into a declared one.

    This is the whole reason `hint_confidence` exists: the video path really is
    in metres, so the unit inference must not re-run its priors over it, but the
    metres came from an estimate and QA has to be told so.
    """
    from locaish.scan.scale import infer_unit_scale

    cloud = PointCloud(np.random.default_rng(0).normal(size=(2000, 3)))

    declared = infer_unit_scale(cloud, hint="m")
    estimated = infer_unit_scale(cloud, hint="m", hint_confidence=0.62, hint_evidence=["solved"])

    assert declared.confidence == 1.0
    assert declared.unit == estimated.unit == "m"
    assert estimated.factor == declared.factor
    assert estimated.confidence == pytest.approx(0.62)
    assert estimated.evidence == ["solved"]


# ---------------------------------------------------------------------------
# gravity hint, through the real resolver
# ---------------------------------------------------------------------------


def test_up_hint_settles_a_room_that_cannot_tell_floor_from_ceiling():
    """An empty box is genuinely ambiguous; the hint is what breaks the tie.

    Without furniture there is no clutter asymmetry, both horizontal surfaces
    are sampled identically, and the sign vote is a coin toss. This is exactly
    the case a phone's own orientation should decide, so both directions are
    asked for and each must be obeyed.
    """
    from locaish.geom.align import find_up
    from locaish.geom.planes import detect_planes
    from locaish.geom.normals import estimate_normals

    cloud = _empty_box()
    normals = estimate_normals(cloud)
    cloud = PointCloud(xyz=cloud.xyz, normals=normals)
    planes = detect_planes(cloud, normals=normals, distance_thresh=0.02, seed=0)

    up = find_up(cloud, planes, up_hint=np.array([0.0, 0.0, 1.0]))
    down = find_up(cloud, planes, up_hint=np.array([0.0, 0.0, -1.0]))

    assert up.up[2] > 0.9, "a hint of +Z must produce a twin standing up"
    assert down.up[2] < -0.9, "a hint of -Z must be obeyed just as readily"


def test_up_hint_cannot_lay_a_furnished_room_on_its_side():
    """A hint at right angles to the truth must lose to the room itself.

    The device is trusted, not obeyed. A phone filmed pointing at the floor the
    whole time would report a sideways up, and a pipeline that took that at face
    value would turn a perfectly good scan into a twin lying on its side.
    """
    from locaish.geom.align import find_up
    from locaish.geom.planes import detect_planes
    from locaish.geom.normals import estimate_normals

    from locaish import fixtures

    fx = fixtures.build("clean")
    cloud = fx.points
    normals = estimate_normals(cloud)
    cloud = PointCloud(xyz=cloud.xyz, rgb=cloud.rgb, normals=normals)
    planes = detect_planes(cloud, normals=normals, distance_thresh=0.03, seed=0)

    solution = find_up(cloud, planes, up_hint=np.array([1.0, 0.0, 0.0]))

    assert abs(solution.up[2]) > 0.99, "the room's own geometry must win"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _extrinsic(yaw: float, world_up: np.ndarray) -> np.ndarray:
    """A world-to-camera [R|t] for a camera at the origin, upright about `world_up`.

    Built the long way round -- choose a look direction perpendicular to the
    given up, then form the OpenCV camera basis (x right, y down, z forward) --
    so that the matrix encodes the intended up by construction rather than by
    reusing the convention the code under test assumes.
    """
    world_up = world_up / np.linalg.norm(world_up)
    seed = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    forward = seed - world_up * np.dot(seed, world_up)
    if np.linalg.norm(forward) < 1e-6:
        forward = np.cross(world_up, [1.0, 0.0, 0.0])
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = -world_up
    R = np.stack([right, down, forward])  # rows: camera axes in world coordinates
    return np.concatenate([R, np.zeros((3, 1))], axis=1)


def _empty_box(n: int = 40_000, w: float = 4.0, d: float = 3.0, h: float = 2.6) -> PointCloud:
    """A featureless rectangular shell: right shape, no clue which end is the floor."""
    rng = np.random.default_rng(0)
    faces = [
        (np.array([w, d, 0.0]), np.array([0.0, 0.0, 0.0])),   # floor
        (np.array([w, d, 0.0]), np.array([0.0, 0.0, h])),     # ceiling
        (np.array([w, 0.0, h]), np.array([0.0, 0.0, 0.0])),
        (np.array([w, 0.0, h]), np.array([0.0, d, 0.0])),
        (np.array([0.0, d, h]), np.array([0.0, 0.0, 0.0])),
        (np.array([0.0, d, h]), np.array([w, 0.0, 0.0])),
    ]
    per = n // len(faces)
    pts = []
    for span, origin in faces:
        u = rng.random((per, 3))
        pts.append(origin + u * span)
    return PointCloud(np.concatenate(pts))


def _synthetic_video(path, *, seconds: int, fps: int, size: str):
    """A moving test pattern, so consecutive frames genuinely differ."""
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}:duration={seconds}",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )
    return path


# ---------------------------------------------------------------------------
# the second, independent scale estimator
# ---------------------------------------------------------------------------


def _room_in_network_units(unit_m: float, *, w=4.0, d=3.0, h=2.6, cam_h=1.55, n=60_000):
    """A room of known metric size, expressed in an arbitrary reconstruction unit.

    Everything is built in metres and divided by `unit_m` at the end, so the
    factor the estimator must recover is `unit_m` and nothing in the
    construction encodes it twice.
    """
    rng = np.random.default_rng(7)
    floor = np.column_stack([rng.uniform(0, w, n // 3), rng.uniform(0, d, n // 3), np.zeros(n // 3)])
    ceil = np.column_stack([rng.uniform(0, w, n // 3), rng.uniform(0, d, n // 3), np.full(n // 3, h)])
    wall = np.column_stack([rng.uniform(0, w, n // 3), np.zeros(n // 3), rng.uniform(0, h, n // 3)])
    pts = np.concatenate([floor, ceil, wall]) / unit_m
    cams = np.column_stack([
        np.linspace(0.5, w - 0.5, 20),
        np.full(20, d / 2),
        np.full(20, cam_h),
    ]) / unit_m
    return pts, cams


def test_camera_height_recovers_a_known_factor():
    pts, cams = _room_in_network_units(0.37)

    est = metric.scale_from_camera_height(pts, cams, np.array([0.0, 0.0, 1.0]))

    assert est is not None
    assert est.factor == pytest.approx(0.37, rel=0.03)
    assert est.source == "camera-height"


def test_camera_height_needs_gravity():
    pts, cams = _room_in_network_units(0.5)
    assert metric.scale_from_camera_height(pts, cams, None) is None


def test_camera_height_refuses_when_the_path_sits_absurdly_in_the_volume():
    """A camera below the floor or up at the ceiling is not a person walking.

    That happens when the floor was never captured and the lowest points are
    something else, and the right answer is no estimate rather than a wrong one.
    """
    pts, cams = _room_in_network_units(0.4, cam_h=2.55, h=2.6)
    assert metric.scale_from_camera_height(pts, cams, np.array([0.0, 0.0, 1.0])) is None


# ---------------------------------------------------------------------------
# combining estimators -- the honesty property
# ---------------------------------------------------------------------------


def _est(factor, spread, bias, source):
    return metric.ScaleEstimate(
        factor=factor, confidence=0.5, log_spread=spread, prior_bias=bias, source=source
    )


def test_agreeing_estimators_produce_a_tighter_bar_than_either_alone():
    a = _est(2.00, 0.02, 0.10, "camera-height")
    b = _est(2.03, 0.02, 0.10, "door-height")

    combined = metric.combine_scales([a, b])

    assert 2.0 <= combined.factor <= 2.03
    assert combined.log_spread < a.total_log_sigma
    assert combined.source == "combined"


def test_disagreeing_estimators_widen_the_bar_instead_of_picking_a_winner():
    """The property this whole design exists for.

    Two estimators, each individually confident to ~10%, that differ by a factor
    of 2.2. A combiner that just averaged them would report a wrong answer with
    a 7% error bar. What has to happen instead is that the error bar grows until
    it covers the disagreement, because at that point we genuinely do not know.
    """
    a = _est(1.00, 0.02, 0.10, "camera-height")
    b = _est(2.20, 0.02, 0.10, "door-height")

    combined = metric.combine_scales([a, b])

    assert 1.0 < combined.factor < 2.2, "the answer must lie between the estimates"
    assert combined.log_spread > a.total_log_sigma * 3, "the bar must cover the argument"
    assert combined.relative_error > 0.25
    assert combined.confidence < 0.3, "a disputed scale must not read as confident"
    assert any("disagree" in w for w in combined.warnings)


def test_internal_agreement_alone_cannot_produce_a_confident_scale():
    """Precision is not accuracy, and the combiner must not confuse them.

    An estimator that agrees with itself perfectly still carries the bias of
    the prior that produced it. Reporting that zero spread as certainty is
    exactly the failure this module is built to prevent.
    """
    import math

    bias = math.log(1.30)
    flawless = _est(1.0, 0.0, bias, "camera-height")

    combined = metric.combine_scales([flawless])

    assert combined.log_spread >= bias
    assert combined.relative_error > 0.2
    assert any("single estimator" in w for w in combined.warnings)


def test_combining_nothing_is_an_error_not_a_default_of_one():
    with pytest.raises(RuntimeError, match="no scale estimate"):
        metric.combine_scales([])


# ---------------------------------------------------------------------------
# the studio's view of a finished twin
# ---------------------------------------------------------------------------


def test_studio_summary_reads_a_real_twin(clean):
    """The drop-page's result card must survive contact with an actual QAReport.

    It reaches across into `Twin`, `Structure` and `QAReport` to build six
    numbers, and it runs at the very end of a two-minute reconstruction — so a
    typo in a field name shows up as a job that says "failed" after every
    expensive step already succeeded. Cheaper to catch here.
    """
    from locaish.serve import _summarise

    result, _fixture = clean
    summary = _summarise(result.twin, result)

    assert summary["verdict"] in {"pass", "warn", "fail", "unknown"}
    assert summary["points"] > 0
    assert summary["floor_area_m2"] > 0
    assert summary["ceiling_height_m"] > 0
    assert isinstance(summary["checks"]["fail"], list)
    assert isinstance(summary["checks"]["warn"], list)
    assert all(isinstance(name, str) for name in summary["checks"]["warn"])
    # A scan-file twin has no video stage, and the card must cope rather than
    # assume every twin came from footage.
    assert summary["scale_relative_error"] is None
    assert summary["frames"] is None


def test_studio_summary_reports_the_video_error_bar(clean):
    """When the twin did come from footage, the card must show the scale's spread.

    The card is the only place a person using the drop page ever sees how much
    to trust the size, so a summary that silently omits it turns an honest
    estimate back into an unqualified number.
    """
    from locaish.serve import _summarise

    result, _fixture = clean
    result.steps = dict(result.steps)
    result.steps["video"] = {
        "frames": {"used": 24},
        "scale": {"relative_error": 0.357, "confidence": 0.05, "source": "combined"},
    }
    try:
        summary = _summarise(result.twin, result)
    finally:
        result.steps.pop("video")

    assert summary["scale_relative_error"] == pytest.approx(0.357)
    assert summary["scale_confidence"] == pytest.approx(0.05)
    assert summary["frames"] == 24


# ---------------------------------------------------------------------------
# the third scale estimator: doorways
# ---------------------------------------------------------------------------


def _opening(height, width, sill, confidence=0.8):
    from locaish.types import Opening

    return Opening(
        center=np.zeros(3),
        width=width,
        height=height,
        normal=np.array([1.0, 0.0, 0.0]),
        sill_height=sill,
        confidence=confidence,
    )


def test_a_doorway_recovers_a_known_scale_error():
    """The room was measured 1.7x too big, so its doors came out 1.7x too tall.

    Everything is built from that premise and nothing tells the estimator what
    the error was, so recovering it is a real inversion rather than a restated
    input.
    """
    error = 1.7
    doors = [
        _opening(2.03 * error, 0.90 * error, 0.0),
        _opening(1.98 * error, 0.86 * error, 0.01 * error),
    ]

    est = metric.scale_from_doors(doors, current_factor=2.0)

    assert est is not None
    assert est.source == "door-height"
    assert est.factor == pytest.approx(2.0 / error, rel=0.05)
    assert est.frames_used == 2


def test_windows_are_not_doors():
    """A sill off the floor disqualifies an aperture however tall it is."""
    windows = [_opening(1.4, 1.2, 0.9), _opening(0.92, 2.01, 0.53)]
    assert metric.scale_from_doors(windows, current_factor=2.0) is None


def test_a_wide_low_gap_is_not_a_door():
    """Reconstruction noise leaves floor-level gaps that are nothing like doors."""
    assert metric.scale_from_doors([_opening(0.44, 1.35, 0.0)], current_factor=2.0) is None


def test_a_tall_pass_through_does_not_drag_the_scale_down():
    """The median, not the maximum -- an open archway is taller than a door.

    Anchoring on the tallest aperture would read every archway as a two-metre
    door and shrink the whole room to suit.
    """
    apertures = [
        _opening(2.05, 0.90, 0.0),
        _opening(2.03, 0.88, 0.0),
        _opening(2.60, 1.10, 0.0),   # an archway, taller than any door
    ]

    est = metric.scale_from_doors(apertures, current_factor=1.0)

    assert est.factor == pytest.approx(2.03 / 2.05, rel=0.02)


def test_low_confidence_apertures_are_ignored():
    faint = [_opening(2.03, 0.9, 0.0, confidence=0.05)]
    assert metric.scale_from_doors(faint, current_factor=2.0) is None


def test_disagreeing_doors_widen_their_own_error_bar():
    agree = [_opening(2.03, 0.9, 0.0), _opening(2.05, 0.88, 0.0)]
    argue = [_opening(1.7, 0.9, 0.0), _opening(2.6, 0.88, 0.0)]

    tight = metric.scale_from_doors(agree, current_factor=1.0)
    loose = metric.scale_from_doors(argue, current_factor=1.0)

    assert loose.log_spread > tight.log_spread
    assert loose.confidence < tight.confidence


def test_the_door_anchor_is_scale_free_in_its_shape_test():
    """Door-shapedness must not depend on what a metre currently means.

    The same room, measured at two wildly different provisional scales, must
    recognise the same aperture as a door -- otherwise the estimator only works
    when the scale is already roughly right, which is when it is least needed.
    """
    for error in (0.4, 1.0, 3.0):
        doors = [_opening(2.03 * error, 0.9 * error, 0.0)]
        est = metric.scale_from_doors(doors, current_factor=1.0)
        assert est is not None, f"a door went unrecognised at {error}x scale"
        assert est.factor == pytest.approx(1.0 / error, rel=0.02)
