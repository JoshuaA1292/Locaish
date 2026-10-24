"""Set pieces: a labelled object lifts into a solid, movable piece whose
geometry is exactly the returns the scan put there, and every map and sweep
recomputes against the moved room."""

from __future__ import annotations

import json

import numpy as np
import pytest

from locaish.film import render as R
from locaish.film import space, sweep
from locaish.product import setpieces as S
from locaish.types import Opening, PointCloud, Structure, Twin


def _box(fx, i):
    """Truth box i of a fixture as (lo, hi, label dict with the anchor on the top face)."""
    lo, hi = fx.truth.boxes[i]
    anchor = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, hi[2]]
    return lo, hi, {"id": f"lbl{i}", "label": "sofa", "kind": "furniture", "xyz": anchor}


@pytest.fixture(scope="module")
def long_box(clean):
    """The 0.6 x 1.8 x 0.72 m box against the door wall, lifted once."""
    twin = clean[0].twin
    lo, hi, label = _box(clean[1], 0)
    piece = S.lift_from_label(twin, label, save=False)
    return twin, lo, hi, piece


def test_lift_grows_exactly_the_box(long_box):
    twin, lo, hi, piece = long_box
    assert piece.source == "scan"
    assert piece.point_count >= S.MIN_POINTS
    truth = sorted((hi - lo).tolist())
    got = list(piece.footprint.dims_sorted())
    for t, g in zip(truth, got):
        assert abs(g - t) / t < 0.05, f"dims {got} vs truth {truth}"
    assert np.allclose([piece.footprint.cx, piece.footprint.cy], ((lo + hi) / 2.0)[:2], atol=0.05)
    # the long axis names the heading: this box runs along Y
    assert abs(abs(piece.footprint.yaw_deg) - 90.0) < 3.0


def test_lift_does_not_swallow_walls_or_the_door_frame(long_box):
    twin, lo, hi, piece = long_box
    pts = twin.points.xyz[piece.point_mask]
    outside = np.maximum(pts - hi, lo - pts).max(axis=1)
    assert (outside > 0.05).sum() == 0, "lifted points leak out of the furniture box"
    # and nothing it took belongs to the fitted architecture
    assert not S.architecture_mask(twin, pts).any()


def test_cube_reads_as_a_square_not_a_diamond(clean):
    twin = clean[0].twin
    lo, hi, label = _box(clean[1], 2)
    piece = S.lift_from_label(twin, label, save=False)
    fp = piece.footprint
    assert abs(fp.width_m - 0.9) < 0.05 and abs(fp.depth_m - 0.9) < 0.05
    assert abs(fp.height_m - 1.15) < 0.05


def test_move_shifts_only_the_piece_and_never_the_input(long_box):
    twin, lo, hi, piece = long_box
    before = twin.points.xyz.copy()
    S.move(piece, 1.0, 0.0, 0.0)
    try:
        applied = S.apply(twin, [piece])
        assert applied is not twin and applied.points is not twin.points
        assert np.array_equal(twin.points.xyz, before)
        delta = applied.points.xyz[piece.point_mask] - twin.points.xyz[piece.point_mask]
        assert np.allclose(delta, [1.0, 0.0, 0.0])
        rest = ~piece.point_mask
        assert np.array_equal(applied.points.xyz[rest], twin.points.xyz[rest])
        assert applied.structure is twin.structure
    finally:
        S.reset(piece)
    assert not piece.moved
    assert np.array_equal(S.apply(twin, [piece]).points.xyz, twin.points.xyz)


def test_turning_keeps_the_centre_and_swaps_the_axes(long_box):
    twin, lo, hi, piece = long_box
    S.move(piece, 0.0, 0.0, 90.0)
    try:
        applied = S.apply(twin, [piece])
        moved = applied.points.xyz[piece.point_mask]
        fp = S.oriented_footprint(moved, twin.structure.floor_z)
        assert np.allclose([fp.cx, fp.cy], [piece.footprint.cx, piece.footprint.cy], atol=0.02)
        assert abs(fp.width_m - piece.footprint.width_m) < 0.02
        assert abs(abs(fp.yaw_deg) - abs(piece.footprint_now().yaw_deg)) < 1.0
        assert abs(abs(piece.footprint_now().yaw_deg) - 0.0) < 3.0  # ran along Y, now along X
    finally:
        S.reset(piece)


def test_standable_floor_recomputes_where_the_piece_was_and_is(long_box):
    twin, lo, hi, piece = long_box
    # the box's own floor, half a metre down from its centre so the door
    # frame it stands beside does not decide the cell
    was = ((lo + hi) / 2.0)[:2] + np.array([0.0, -0.5])
    now = was + np.array([1.0, 0.0])
    S.move(piece, 1.0, 0.0, 0.0)
    try:
        m0 = space.floor_maps(twin)
        m1 = space.floor_maps(S.apply(twin, [piece]))
        s0, s1 = m0.standable(), m1.standable()
        assert not m0.sample(s0, was)[0] and m1.sample(s1, was)[0]
        assert m0.sample(s0, now)[0] and not m1.sample(s1, now)[0]
    finally:
        S.reset(piece)


def test_exclusion_mask_names_the_moved_piece_only(long_box):
    twin, lo, hi, piece = long_box
    assert S.exclusion_mask(twin, [piece]).sum() == 0
    assert S.exclusion_mask(twin, [piece], moved_only=False).sum() == piece.point_count
    S.move(piece, 0.5, 0.0)
    try:
        ex = S.exclusion_mask(twin, [piece])
        assert ex.sum() == piece.point_count and ex.shape == (len(twin.points),)
        fig = S.figure(piece, twin)
        assert isinstance(fig, R.Figure) and len(fig.xyz) == piece.point_count
        assert np.allclose(fig.xyz.mean(axis=0)[:2] - twin.points.xyz[piece.point_mask].mean(axis=0)[:2], [0.5, 0.0], atol=1e-6)
        img = R.render_shot(twin, (0.0, 1.5, 1.4), (-1.0, 0.0), 24.0, figures=[fig], exclude_points=ex)
        assert img.size[0] > 0
    finally:
        S.reset(piece)


def test_sweep_recomputes_against_the_moved_room(clean):
    twin = clean[0].twin
    lo, hi, label = _box(clean[1], 2)
    piece = S.lift_from_label(twin, label, save=False)
    S.move(piece, -1.0, 0.0)
    before = sweep.sweep(twin)
    after = sweep.sweep(S.apply(twin, [piece]))
    d = S.sweep_diff(before, after)
    assert d["visible_before"] == int(before.columns["visible"].sum())
    assert d["visible_after"] == int(after.columns["visible"].sum())
    assert d["delta"] != 0
    assert set(d["by_focal_mm"]) == {"16", "25", "35", "50", "75", "100"}
    assert d["sentences"] and all(isinstance(s, str) for s in d["sentences"])


def test_door_fit_arithmetic_on_a_hand_built_opening():
    door = Opening(center=[-2.0, 0.0, 1.0], width=0.9, height=2.0, normal=[1, 0, 0], sill_height=0.0, kind="door", confidence=0.9)
    window = Opening(center=[0.0, 2.0, 1.5], width=1.4, height=1.2, normal=[0, -1, 0], sill_height=0.9, kind="window")
    twin = Twin(name="t", points=PointCloud(xyz=np.zeros((1, 3))),
                structure=Structure(openings=[door, window]))
    sofa = S.SetPiece(id="a", label="sofa", kind="furniture", anchor=[0, 0, 0.5],
                      footprint=S.Footprint(0, 0, 1.8, 0.6, 0.72))
    fit = S.door_fit(sofa, twin)
    assert len(fit["doors"]) == 1  # the window is not a door
    assert fit["passes_m"] == [0.6, 0.72]
    assert fit["doors"][0]["fits"] is True
    assert fit["doors"][0]["clearance_m"] == pytest.approx(0.3, abs=1e-6)   # min(0.9-0.6, 2.0-0.72)
    assert fit["tightest_clearance_m"] == pytest.approx(0.3, abs=1e-6)
    assert fit["fits_all"] is True

    crate = S.SetPiece(id="b", label="crate", kind="other", anchor=[0, 0, 0.5],
                       footprint=S.Footprint(0, 0, 1.0, 1.0, 1.0))
    fit = S.door_fit(crate, twin)
    assert fit["doors"][0]["fits"] is False
    assert fit["doors"][0]["clearance_m"] == pytest.approx(-0.1, abs=1e-6)

    # a door is taller than it is wide, so upright always wins there; a wide,
    # low hatch is where a piece has to go through on its side
    hatch = Twin(name="h", points=PointCloud(xyz=np.zeros((1, 3))),
                 structure=Structure(openings=[Opening(center=[0, 2, 0.6], width=2.0, height=0.9,
                                                       normal=[0, -1, 0], sill_height=0.15, kind="door")]))
    low = S.SetPiece(id="c", label="chest", kind="furniture", anchor=[0, 0, 0.5],
                     footprint=S.Footprint(0, 0, 1.9, 0.85, 0.7))
    fit = S.door_fit(low, hatch)
    assert fit["doors"][0]["fits"] is True and fit["doors"][0]["orientation"] == "on its side"
    assert fit["doors"][0]["clearance_m"] == pytest.approx(0.2, abs=1e-6)   # min(2.0-0.85, 0.9-0.7)

    bare = Twin(name="t", points=PointCloud(xyz=np.zeros((1, 3))))
    fit = S.door_fit(sofa, bare)
    assert fit["doors"] == [] and fit["fits_all"] is None and fit["note"]


def test_sweep_diff_on_synthetic_columns():
    before = {
        "visible": np.array([1, 1, 0, 1, 0, 0], dtype=np.uint8),
        "shot_size": np.array(["ls", "ls", "ls", "cu", "cu", "ms"], dtype=object),
        "focal_mm": np.array([50, 50, 50, 35, 35, 50], dtype=np.float32),
    }
    after = {
        "visible": np.array([1, 1, 1, 0, 0, 1, 1], dtype=np.uint8),
        "shot_size": np.array(["ls", "ls", "ls", "cu", "cu", "ms", "ms"], dtype=object),
        "focal_mm": np.array([50, 50, 50, 35, 35, 50, 50], dtype=np.float32),
    }
    d = S.sweep_diff(before, after)
    assert d["visible_before"] == 3 and d["visible_after"] == 5 and d["delta"] == 2
    assert d["by_shot_size"]["ls"] == {"before": 2, "after": 3, "delta": 1}
    assert d["by_shot_size"]["cu"] == {"before": 1, "after": 0, "delta": -1}
    assert d["by_shot_size"]["ms"] == {"before": 0, "after": 2, "delta": 2}
    assert d["by_focal_mm"]["50"] == {"before": 2, "after": 5, "delta": 3}
    assert d["by_focal_mm"]["35"] == {"before": 1, "after": 0, "delta": -1}
    text = " ".join(d["sentences"])
    assert "a 50 mm medium shot that had no clear setup now has 2" in text
    assert "a 35 mm close-up lost all 1 of its clear setups" in text


def test_store_round_trips_json_and_masks(long_box, tmp_path):
    twin, lo, hi, piece = long_box
    S.move(piece, 0.4, -0.2, 15.0)
    try:
        S.save_pieces(tmp_path, [piece])
        raw = json.loads((tmp_path / "setpieces.json").read_text())
        rec = raw["pieces"][0]
        assert rec["mask_path"] == f"setpieces/{piece.id}.npy"
        assert (tmp_path / rec["mask_path"]).exists()
        assert set(rec["footprint"]) == {"cx", "cy", "width_m", "depth_m", "height_m", "yaw_deg", "z_base"}
        assert set(rec["pose"]) == {"dx", "dy", "dyaw_deg"}
        back = S.load_pieces(tmp_path)
        assert len(back) == 1
        b = back[0]
        assert b.id == piece.id and b.label == piece.label and b.kind == piece.kind
        assert b.pose == {"dx": 0.4, "dy": -0.2, "dyaw_deg": 15.0}
        assert np.array_equal(b.point_mask, piece.point_mask)
        assert b.point_count == piece.point_count
        assert np.allclose(b.anchor, piece.anchor, atol=1e-4)
        assert abs(b.footprint.width_m - piece.footprint.width_m) < 1e-3
        # the same move, applied from the reloaded piece, lands the same points
        assert np.allclose(S.apply(twin, [b]).points.xyz, S.apply(twin, [piece]).points.xyz, atol=1e-6)
        # upsert replaces by id, remove deletes the mask file
        assert len(S.upsert_piece(tmp_path, b)) == 1
        assert S.remove_piece(tmp_path, b.id) is True
        assert S.load_pieces(tmp_path) == [] and not (tmp_path / rec["mask_path"]).exists()
    finally:
        S.reset(piece)


def test_lift_from_label_saves_into_the_workdir(clean, tmp_path):
    twin = clean[0].twin
    lo, hi, label = _box(clean[1], 1)
    piece = S.lift_from_label(twin, label, tmp_path)
    back = S.load_pieces(tmp_path)
    assert [p.id for p in back] == [piece.id]
    assert back[0].label_id == "lbl1"


def test_catalogue_default_when_nothing_grows(clean, tmp_path):
    twin = clean[0].twin
    label = {"id": "air", "label": "sofa", "kind": "furniture", "xyz": [0.0, 0.0, 1.0]}  # mid-air
    piece = S.lift_from_label(twin, label, tmp_path)
    assert piece.source == "catalogue-default"
    assert piece.point_mask is None and piece.point_count == 0
    assert (piece.footprint.width_m, piece.footprint.depth_m, piece.footprint.height_m) == S.CATALOGUE["sofa"]
    assert any("catalogue default" in n for n in piece.notes)
    # drawn, but never moves a point
    fig = S.figure(piece, twin, tmp_path)
    assert len(fig.xyz) > 100 and "catalogue default" in fig.label
    S.move(piece, 1.0, 0.0)
    assert np.array_equal(S.apply(twin, [piece]).points.xyz, twin.points.xyz)
    assert S.exclusion_mask(twin, [piece]).sum() == 0
    desc = S.describe([piece], twin)[0]
    assert desc["measured"] is False and desc["door_fit"]["doors"]
    back = S.load_pieces(tmp_path)
    assert back[0].source == "catalogue-default" and back[0].point_mask is None


def test_attached_gaussians_ride_the_pose(long_box, tmp_path):
    twin, lo, hi, piece = long_box
    n = 40
    xyz = twin.points.xyz[piece.point_mask][:n]
    cov = np.tile(np.diag([4e-4, 1e-4, 1e-4]).astype(np.float32), (n, 1, 1))
    np.savez(tmp_path / "obj.npz", xyz=xyz, rgb=np.full((n, 3), 0.5, np.float32),
             opacity=np.full(n, 0.9, np.float32), cov=cov)
    S.attach_gaussians(piece, tmp_path, tmp_path / "obj.npz")
    try:
        assert piece.source == "sam3d"
        assert piece.gaussians_path == f"setpieces/{piece.id}_gaussians.npz"
        assert (tmp_path / piece.gaussians_path).exists()
        S.move(piece, 0.0, 0.0, 90.0)
        fig = S.figure(piece, twin, tmp_path)
        assert len(fig.xyz) == n and fig.cov.shape == (n, 3, 3) and fig.opacity.shape == (n,)
        # the splat turned with the piece: the long covariance axis swapped from X to Y
        assert np.allclose(fig.cov[0][0, 0], 1e-4, atol=1e-6) and np.allclose(fig.cov[0][1, 1], 4e-4, atol=1e-6)
        # positions match the moved points
        assert np.allclose(fig.xyz, S.apply(twin, [piece]).points.xyz[piece.point_mask][:n])
    finally:
        S.reset(piece)
        piece.gaussians_path = None
        piece.source = "scan"


def test_describe_speaks_in_metres(long_box):
    twin, lo, hi, piece = long_box
    S.move(piece, 0.3, 0.4, -10.0)
    try:
        d = S.describe([piece], twin)[0]
        assert d["label"] == "sofa" and d["measured"] is True
        assert d["moved"] is True and d["moved_by"] == {"dx_m": 0.3, "dy_m": 0.4, "dyaw_deg": -10.0}
        assert "moved 0.50 m" in d["moved_text"] and "turned -10 deg" in d["moved_text"]
        assert d["door_fit"]["doors"] and isinstance(d["door_fit"]["tightest_clearance_m"], float)
        assert np.allclose(d["position_m"], np.array(d["scan_position_m"]) + [0.3, 0.4], atol=1e-3)
    finally:
        S.reset(piece)
