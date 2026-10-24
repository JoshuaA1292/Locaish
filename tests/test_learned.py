"""The learned-geometry stage, checked without loading a model.

What can go wrong here is not the network's output -- that is what it is --
but the arithmetic that ties it to the reconstruction: which way the units
go, whether a hallucinated patch survives the cross-view check, whether the
scale anchor's error bar carries the model's bias. Each of those is tested on
a scene rendered analytically, so the right answer is known before the code
runs and is not read back out of it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from locaish.video import colmap, learned, metric

# A room in COLMAP units: U units per metre. A wall facing the camera at
# world z = WALL, a floor at world y = FLOOR (camera looks +z, +y is down).
U = 2.0
WALL_M = 3.0
# Low enough that the floor enters the frame before the wall does: the
# vertical half-field is 90/260 of the focal, and a floor 0.8 m down crosses
# that at z = 2.3 m, inside the wall.
FLOOR_M = 0.8
W, H = 320, 180
F = 260.0


def _K():
    return np.array([[F, 0.0, W / 2], [0.0, F, H / 2], [0.0, 0.0, 1.0]])


def _extrinsic(cam_world):
    """World -> camera for an un-rotated camera at `cam_world` (units)."""
    E = np.zeros((3, 4))
    E[:, :3] = np.eye(3)
    E[:, 3] = -np.asarray(cam_world, dtype=np.float64)
    return E


def _render_depth_m(cam_world):
    """Analytic z-depth in METRES of the wall+floor scene from this camera."""
    cx, cy, cz = (np.asarray(cam_world) / U)
    vs, us = np.mgrid[0:H, 0:W]
    dx = (us + 0.5 - W / 2) / F
    dy = (vs + 0.5 - H / 2) / F
    z_wall = np.full((H, W), WALL_M - cz)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_floor = np.where(dy > 1e-6, (FLOOR_M - cy) / dy, np.inf)
    return np.minimum(z_wall, z_floor).astype(np.float32)


def _frame(name, cam_world, depth_m=None):
    d = _render_depth_m(cam_world) if depth_m is None else depth_m
    return learned.FrameDepth(
        name=name, depth=d, mask=np.ones_like(d, dtype=bool), K=_K(), size=(W, H)
    )


def _sparse(seed=0, n=400):
    """Points on the wall and floor, in units."""
    rng = np.random.default_rng(seed)
    wall = np.column_stack(
        [rng.uniform(-1.4, 1.4, n), rng.uniform(-0.8, 0.8, n), np.full(n, WALL_M)]
    )
    floor = np.column_stack(
        [rng.uniform(-1.4, 1.4, n), np.full(n, FLOOR_M), rng.uniform(2.35, 2.95, n)]
    )
    return np.concatenate([wall, floor]) * U


# ---------------------------------------------------------------------------
# alignment: the model's metres against the solve's units
# ---------------------------------------------------------------------------


def test_align_frame_recovers_units_per_metre():
    cam = np.array([0.0, 0.0, 0.0])
    got = learned.align_frame(_frame("a", cam), _extrinsic(cam), _K(), (W, H), _sparse())

    assert got is not None
    s, mad, n = got
    assert s == pytest.approx(U, rel=0.01)
    assert mad < 0.02
    assert n > 100


def test_shell_collapses_onto_its_mid_surface():
    """A wall predicted as a 6 cm thick shell must come out as a sheet at the
    shell's middle, and a point far off any sheet must be dropped."""
    rng = np.random.default_rng(5)
    n = 6000
    xyz = np.column_stack([rng.uniform(0, 3, n), rng.uniform(0, 2.5, n), rng.uniform(-0.03, 0.03, n)])
    xyz = np.vstack([xyz, [[1.5, 1.2, 0.20]]])
    pts = np.hstack([xyz, np.full((len(xyz), 3), 128.0)])

    out = learned.collapse_shell(pts, units_per_m=1.0)

    assert len(out) == n, "the lone point 20 cm off the sheet was dropped"
    # a uniform +/-3 cm shell has 1.7 cm RMS; the sheet comes out under 1 cm,
    # with the only residual at the sheet's edges where neighbourhoods are
    # lopsided
    assert float(np.sqrt(np.mean(out[:, 2] ** 2))) < 0.008
    assert np.abs(out[:, 2]).max() < 0.025


def test_align_frame_refuses_too_few_points():
    cam = np.array([0.0, 0.0, 0.0])
    few = _sparse()[:10]
    assert learned.align_frame(_frame("a", cam), _extrinsic(cam), _K(), (W, H), few) is None


def test_align_frame_refuses_depth_that_does_not_describe_the_scene():
    """Noise is not a depth map, and the median of noise is not a scale."""
    cam = np.array([0.0, 0.0, 0.0])
    rng = np.random.default_rng(3)
    noise = rng.uniform(0.3, 9.0, (H, W)).astype(np.float32)
    assert (
        learned.align_frame(_frame("a", cam, noise), _extrinsic(cam), _K(), (W, H), _sparse())
        is None
    )


# ---------------------------------------------------------------------------
# the anchor
# ---------------------------------------------------------------------------


def _aligned(units_per_metre, n=12, jitter=0.02, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        s = units_per_metre * math.exp(rng.normal(0.0, jitter))
        out.append(
            learned.AlignedFrame(
                index=i, frame=_frame(f"f{i}", (0, 0, 0)), units_per_metre=s,
                log_mad=0.03, points_used=200,
            )
        )
    return out


def test_scale_anchor_is_metres_per_unit_and_carries_the_model_bias():
    est = learned.scale_from_metric_depth(_aligned(U))

    assert est is not None
    assert est.factor == pytest.approx(1.0 / U, rel=0.03)
    assert est.source == "metric-depth-model"
    assert est.model == learned.MODEL_ID
    assert est.frames_used == 12
    # a dozen frames that agree pin the median tightly...
    assert est.log_spread < 0.02
    # ...but the estimator still inherits its prior's bias when it stands alone
    assert est.total_log_sigma >= learned.MODEL_LOG_BIAS
    combined = metric.combine_scales([est])
    assert combined.relative_error >= 0.1


def test_scale_anchor_needs_a_distribution():
    assert learned.scale_from_metric_depth(_aligned(U, n=2)) is None


def test_anchor_tightens_the_bar_when_it_agrees_with_the_camera_height():
    cam_h = metric.ScaleEstimate(
        factor=0.52, confidence=0.6, log_spread=0.0, prior_bias=metric.CAMERA_HEIGHT_LOG_BIAS,
        source="camera-height",
    )
    model = learned.scale_from_metric_depth(_aligned(U))
    both = metric.combine_scales([cam_h, model])

    assert 0.5 <= both.factor <= 0.52
    assert both.log_spread < cam_h.total_log_sigma
    assert both.log_spread < model.total_log_sigma
    assert not any("disagree" in w for w in both.warnings)


def test_anchor_widens_the_bar_when_it_argues_with_the_camera_height():
    cam_h = metric.ScaleEstimate(
        factor=0.85, confidence=0.6, log_spread=0.0, prior_bias=metric.CAMERA_HEIGHT_LOG_BIAS,
        source="camera-height",
    )
    model = learned.scale_from_metric_depth(_aligned(U))
    both = metric.combine_scales([cam_h, model])

    assert 0.5 < both.factor < 0.85
    assert any("disagree" in w for w in both.warnings)
    assert both.relative_error > 0.15


# ---------------------------------------------------------------------------
# dense fill: predictions survive only when other views agree
# ---------------------------------------------------------------------------


def _three_views(depth0=None):
    cams = [np.array([-0.6, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.6, 0.0, 0.0])]
    frames = [
        _frame("f0", cams[0], depth0),
        _frame("f1", cams[1]),
        _frame("f2", cams[2]),
    ]
    aligned = [
        learned.AlignedFrame(index=i, frame=f, units_per_metre=U, log_mad=0.01, points_used=300)
        for i, f in enumerate(frames)
    ]
    extr = np.stack([_extrinsic(c) for c in cams])
    return aligned, extr


def _on_a_plane(pts):
    """Distance of each point (units) from the nearer of wall and floor."""
    return np.minimum(np.abs(pts[:, 2] - WALL_M * U), np.abs(pts[:, 1] - FLOOR_M * U))


def test_fusion_keeps_the_surface_other_views_confirm():
    aligned, extr = _three_views()
    pts = learned.fuse_depth_maps(aligned, extr, None, stride=4, min_views=2)

    assert len(pts) > 2000
    assert np.percentile(_on_a_plane(pts), 99) < 1e-3
    # both surfaces made it, not just the easy one
    assert (np.abs(pts[:, 2] - WALL_M * U) < 1e-3).sum() > 500
    assert (np.abs(pts[:, 1] - FLOOR_M * U) < 1e-3).sum() > 500


def test_fusion_drops_a_patch_only_one_view_believes():
    """A blob hallucinated at half depth in one frame reprojects into the
    neighbours at the wrong depth and must not survive."""
    d0 = _render_depth_m(np.array([-0.6, 0.0, 0.0]))
    d0[40:80, 120:200] *= 0.5
    aligned, extr = _three_views(d0)
    pts = learned.fuse_depth_maps(aligned, extr, None, stride=4, min_views=2)

    assert len(pts) > 1000
    assert _on_a_plane(pts).max() < 1e-3, "a point off both planes is the blob"


def test_fusion_needs_enough_views_to_vote():
    aligned, extr = _three_views()
    assert len(learned.fuse_depth_maps(aligned[:2], extr[:2], None, min_views=2)) == 0
    # three views cannot give any point three *other* witnesses
    assert len(learned.fuse_depth_maps(aligned, extr, None, stride=4, min_views=3)) == 0


def test_grazing_predictions_are_dropped_before_the_vote():
    """A floor 0.3 m below the camera is seen almost edge-on at the far end of
    the frame; those pixels are the model's least trustworthy and must not
    reach the fusion, while the wall straight ahead must."""
    cam = np.array([0.0, 0.0, 0.0])
    cx, cy, cz = 0.0, 0.0, 0.0
    vs, us = np.mgrid[0:H, 0:W]
    dy = (vs + 0.5 - H / 2) / F
    with np.errstate(divide="ignore"):
        z_floor = np.where(dy > 1e-6, 0.3 / dy, np.inf)
    depth = np.minimum(np.full((H, W), WALL_M), z_floor).astype(np.float32)
    fd = _frame("g", cam, depth)
    pts, us_k, vs_k = learned._backproject(fd, U, 0.0, stride=2)

    floor = np.abs(pts[:, 1] - 0.3 * U) < 1e-3
    wall = np.abs(pts[:, 2] - WALL_M * U) < 1e-3
    assert wall.sum() > 2000
    # the floor pixels that survive are the steep near ones, never the far
    # edge-on ones (incidence |cos| = dy / |ray| < 0.2 there)
    ray = pts[floor] / np.linalg.norm(pts[floor], axis=1, keepdims=True)
    assert (np.abs(ray[:, 1]) >= learned.INCIDENCE_MIN_COS - 1e-6).all()
    assert floor.sum() < (vs_k.size)  # some floor was dropped


def test_gap_fill_keeps_only_what_stereo_did_not_cover():
    rng = np.random.default_rng(1)
    learned_pts = np.column_stack([rng.uniform(0, 10, 500), np.zeros(500), np.zeros(500), *[np.full(500, 100.0)] * 3])
    stereo = np.column_stack([np.linspace(0, 5, 2000), np.zeros(2000), np.zeros(2000)])

    kept = learned.fill_gaps(learned_pts, stereo, gap=0.1)

    assert len(kept) < 500
    assert kept[:, 0].min() > 5.0
    assert len(learned.fill_gaps(learned_pts, np.zeros((0, 3)), gap=0.1)) == 500


def test_learned_fill_conforms_to_the_measured_surface():
    """A learned sheet 3 cm above a stereo wall with a hole in it: over the
    wall the sheet lands ON the wall (or is dropped as redundant), over the
    hole it stands where predicted, and a point 15 cm off the wall -- inside
    the conform radius but past the offset limit -- is discarded as clutter
    the model smoothed over."""
    rng = np.random.default_rng(2)
    n = 20000
    stereo = np.column_stack([rng.uniform(0, 4, n), rng.uniform(0, 2.5, n), rng.normal(0, 0.005, n)])
    stereo = stereo[~((stereo[:, 0] > 1.5) & (stereo[:, 0] < 2.5))]  # the hole
    m = 3000
    sheet = np.column_stack([rng.uniform(0, 4, m), rng.uniform(0, 2.5, m), np.full(m, 0.03)])
    sheet = np.vstack([sheet, [[0.5, 1.0, 0.15]]])
    learned_pts = np.hstack([sheet, np.full((len(sheet), 3), 100.0)])

    out, tally, standalone = learned.conform_to_stereo(learned_pts, stereo, units_per_m=1.0)
    assert standalone.sum() == tally["kept"]

    # Over the wall every sheet point is within 6 cm of stereo: redundant,
    # gone. In the 20 cm band inside the hole's edges the sheet has stereo
    # neighbours and is conformed onto the wall. In the hole's middle it
    # stands where the model put it.
    over_hole = (out[:, 0] > 1.75) & (out[:, 0] < 2.25)
    over_wall = (out[:, 0] < 1.4) | (out[:, 0] > 2.6)
    edge_band = ((out[:, 0] > 1.5) & (out[:, 0] < 1.68)) | ((out[:, 0] > 2.32) & (out[:, 0] < 2.5))
    assert over_hole.sum() > 300
    assert np.allclose(out[over_hole, 2], 0.03), "the hole is filled where the model put it"
    assert over_wall.sum() == 0, "a sheet on top of measured surface is redundant"
    # in the band, a point either stands as predicted (its stereo
    # neighbours were a line along the edge, no plane to conform to) or was
    # moved onto the wall -- never left a few centimetres off it
    moved = edge_band & (np.abs(out[:, 2] - 0.03) > 1e-9)
    assert moved.sum() > 50
    assert np.abs(out[moved, 2]).max() < 0.015, "at the hole's edges the fill is moved onto the wall"
    assert tally["redundant"] > 1500 and tally["conformed"] > 50 and tally["kept"] > 300
    assert tally["dropped"] >= 1
    assert not ((np.abs(out[:, 0] - 0.5) < 1e-6) & (np.abs(out[:, 1] - 1.0) < 1e-6)).any()
    kept_all, t, sa = learned.conform_to_stereo(learned_pts, np.zeros((0, 3)), 1.0)
    assert len(kept_all) == len(learned_pts) and t["kept"] == len(learned_pts) and sa.all()


def test_the_model_may_fill_a_ceiling_but_not_invent_a_wall():
    """Two stand-alone learned sheets with no stereo near them: a horizontal
    one (a ceiling) survives, a vertical one (an invented wall) does not;
    with no gravity to tell them apart, both are kept."""
    rng = np.random.default_rng(9)
    n = 4000
    ceiling = np.column_stack([rng.uniform(0, 3, n), rng.uniform(0, 3, n), np.full(n, 2.5) + rng.normal(0, 0.002, n)])
    wall = np.column_stack([np.full(n, 3.0) + rng.normal(0, 0.002, n), rng.uniform(0, 3, n), rng.uniform(0, 2.4, n)])
    pts = np.hstack([np.vstack([ceiling, wall]), np.full((2 * n, 3), 90.0)])
    standalone = np.ones(2 * n, dtype=bool)

    kept, dropped = learned.drop_standalone_walls(pts, standalone, np.array([0.0, 0.0, 1.0]))

    assert dropped > 0.9 * n
    assert (np.abs(kept[:, 2] - 2.5) < 0.05).mean() > 0.95, "what survives is the ceiling"
    kept2, dropped2 = learned.drop_standalone_walls(pts, standalone, None)
    assert dropped2 == 0 and len(kept2) == 2 * n
    # a conformed or redundant row is never touched, whatever its orientation
    kept3, dropped3 = learned.drop_standalone_walls(pts, np.zeros(2 * n, dtype=bool), np.array([0.0, 0.0, 1.0]))
    assert dropped3 == 0


def test_voxel_fusion_averages_repeated_sightings():
    pts = np.array([
        [0.001, 0.0, 0.0, 10, 10, 10],
        [0.004, 0.0, 0.0, 30, 30, 30],
        [1.000, 0.0, 0.0, 50, 50, 50],
    ])
    fused = learned.fuse_voxels(pts, voxel=0.01)
    assert len(fused) == 2
    near = fused[np.argmin(fused[:, 0])]
    assert near[0] == pytest.approx(0.0025)
    assert near[3] == pytest.approx(20.0)


def test_frames_are_spread_over_the_sweep():
    picked = learned.choose_frames(300, 10)
    assert picked[0] == 0 and picked[-1] == 299 and len(picked) == 10
    assert learned.choose_frames(5, 10) == [0, 1, 2, 3, 4]


def test_stage_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("LOCAISH_LEARNED", "0")
    assert not learned.enabled()
    assert not learned.available()


# ---------------------------------------------------------------------------
# detector and matcher selection
# ---------------------------------------------------------------------------


def test_sift_config_keeps_the_dense_frame_budget(monkeypatch):
    monkeypatch.delenv("LOCAISH_FEATURES", raising=False)
    monkeypatch.delenv("LOCAISH_MATCHER", raising=False)
    cfg = colmap.feature_config("sift")
    assert cfg["features"] == "sift"
    assert cfg["fps"] == colmap.CLASSICAL_FPS
    assert cfg["long_side"] == colmap.CLASSICAL_LONG_SIDE
    assert colmap.matching_type(cfg) == "SIFT_BRUTEFORCE"
    assert "--FeatureExtraction.type" in colmap.extraction_args(cfg)


def test_aliked_is_chosen_when_the_build_has_it(monkeypatch):
    monkeypatch.delenv("LOCAISH_FEATURES", raising=False)
    monkeypatch.delenv("LOCAISH_MATCHER", raising=False)
    monkeypatch.setattr(colmap, "supports_aliked", lambda exe=None: True)
    cfg = colmap.feature_config()
    assert cfg["features"] == "aliked"
    assert cfg["matcher"] == "bruteforce"
    assert cfg["fps"] == colmap.ALIKED_FPS
    assert cfg["long_side"] == colmap.ALIKED_LONG_SIDE
    args = colmap.extraction_args(cfg)
    assert "ALIKED_N16ROT" in args
    assert colmap.matching_type(cfg) == "ALIKED_BRUTEFORCE"
    assert colmap.matching_type(colmap.feature_config(matcher="lightglue")) == "ALIKED_LIGHTGLUE"


def test_aliked_falls_back_to_sift_and_says_so(monkeypatch):
    monkeypatch.setattr(colmap, "supports_aliked", lambda exe=None: False)
    cfg = colmap.feature_config("aliked")
    assert cfg["features"] == "sift"
    assert cfg["note"] and "SIFT" in cfg["note"]
    assert colmap.feature_config()["features"] == "sift"


def test_scale_components_are_listed_for_the_header():
    from locaish.video.reconstruct import _scale_components

    a = metric.ScaleEstimate(factor=0.5, confidence=0.6, log_spread=0.0, prior_bias=0.1, source="camera-height")
    b = metric.ScaleEstimate(factor=0.51, confidence=0.6, log_spread=0.01, prior_bias=0.1, source="metric-depth-model")
    assert _scale_components(metric.combine_scales([a, b])) == ["camera-height", "metric-depth-model"]
    assert _scale_components(metric.combine_scales([a])) == ["camera-height"]
    assert _scale_components(None) == []
