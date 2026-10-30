"""The parametric human figure: a mesh from a height, posed, and drawn as a
shaded solid that the twin's measured geometry occludes."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from locaish.film import render as R
from locaish.product import cast as C
from locaish.product import figures as FG

RED = (230, 80, 60)


def _shoulder_axis_extent(V: np.ndarray, h: float) -> tuple[float, float]:
    band = V[(V[:, 2] > 0.78 * h) & (V[:, 2] < 0.84 * h)]
    return float(np.ptp(band[:, 0])), float(np.ptp(band[:, 1]))


def _feet(V: np.ndarray) -> np.ndarray:
    return V[V[:, 2] < 0.05]


# ---------------------------------------------------------------------------
# the mesh
# ---------------------------------------------------------------------------


def test_mesh_is_sound_and_at_height():
    V, F = FG.figure_mesh(1.75)
    assert np.isfinite(V).all()
    assert len(V) > 200 and len(F) > 400
    assert 1000 <= len(F) <= 4000
    assert F.min() >= 0 and F.max() < len(V)
    assert abs(V[:, 2].max() - 1.75) < 0.02
    assert abs(V[:, 2].min()) < 1e-9
    # The shoulder line, deltoid to deltoid, is the reference 0.46 m torso
    # width plus the shoulders themselves: half a metre, give or take.
    sx, _ = _shoulder_axis_extent(V, 1.75)
    assert FG.SHOULDER_WIDTH_M - 0.02 < sx < FG.SHOULDER_WIDTH_M + 0.10


@pytest.mark.parametrize("h", [1.45, 1.62, 1.95])
def test_height_scales_the_whole_body(h):
    V, _ = FG.figure_mesh(h)
    ref, _ = FG.figure_mesh(1.75)
    assert abs(V[:, 2].max() - h) < 0.02
    sx, _ = _shoulder_axis_extent(V, h)
    sr, _ = _shoulder_axis_extent(ref, 1.75)
    assert sx / sr == pytest.approx(h / 1.75, rel=0.03)


def test_deterministic():
    a = FG.figure_mesh(1.7, pose="walking", phase=0.3, facing_deg=40.0, xy=(1, 2), floor_z=0.1)
    b = FG.figure_mesh(1.7, pose="walking", phase=0.3, facing_deg=40.0, xy=(1, 2), floor_z=0.1)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_facing_rotates_the_shoulder_axis():
    V90, _ = FG.figure_mesh(1.75, facing_deg=90.0)      # canonical: shoulders along X
    V0, _ = FG.figure_mesh(1.75, facing_deg=0.0)        # turned to +X: shoulders along Y
    sx, sy = _shoulder_axis_extent(V90, 1.75)
    assert sx > 2.0 * sy
    sx, sy = _shoulder_axis_extent(V0, 1.75)
    assert sy > 2.0 * sx
    # Facing +Y puts the face (and the toes) ahead of the body centre in +Y.
    feet = _feet(V90)
    assert feet[:, 1].max() > 0.12


def test_placement_moves_feet_to_the_mark():
    V, _ = FG.figure_mesh(1.75, xy=(2.0, -1.0), floor_z=0.3)
    feet = V[V[:, 2] < 0.3 + 0.05]
    assert abs(feet[:, 0].mean() - 2.0) < 0.1 and abs(feet[:, 1].mean() + 1.0) < 0.15
    assert abs(V[:, 2].min() - 0.3) < 1e-9


def test_walking_phase_moves_the_feet_apart():
    stand, _ = FG.figure_mesh(1.75)
    together, _ = FG.figure_mesh(1.75, pose="walking", phase=0.0)
    apart, _ = FG.figure_mesh(1.75, pose="walking", phase=0.25)
    spread = lambda V: float(np.ptp(_feet(V)[:, 1]))  # noqa: E731
    assert spread(apart) > spread(stand) + 0.3
    assert spread(together) < spread(apart) - 0.3
    # Still a person of the right height, give or take the bob.
    assert abs(apart[:, 2].max() - 1.75) < 0.06
    # The other half of the cycle leads with the other leg.
    other, _ = FG.figure_mesh(1.75, pose="walking", phase=0.75)
    la = _feet(apart)
    lo = _feet(other)
    assert np.sign(la[la[:, 0] < 0][:, 1].mean()) != np.sign(lo[lo[:, 0] < 0][:, 1].mean())


def test_sitting_lowers_the_head_and_bends_the_knees():
    stand, _ = FG.figure_mesh(1.75)
    sit, _ = FG.figure_mesh(1.75, pose="sitting")
    assert stand[:, 2].max() - sit[:, 2].max() > 0.3
    # Knees out in front of the hips: the thighs reach forward.
    assert sit[:, 1].max() > stand[:, 1].max() + 0.25
    assert abs(sit[:, 2].min()) < 1e-9


def test_unknown_pose_refused():
    with pytest.raises(ValueError):
        FG.figure_mesh(1.75, pose="dancing")


def test_walk_phase():
    assert FG.walk_phase(3.0, 0.0) == 0.0
    p = [FG.walk_phase(t, 1.0) for t in (0.0, 0.35, 0.7, 1.05)]
    assert all(0.0 <= x < 1.0 for x in p)
    assert p[1] == pytest.approx(0.25) and p[2] == pytest.approx(0.5)
    assert FG.walk_phase(1.4, 1.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# figures for the renderer
# ---------------------------------------------------------------------------


def test_stand_in_is_a_mesh_figure_with_a_body_of_points():
    f = R.stand_in(1.75, (1.0, 2.0), 180.0, 0.0, rgb=RED, label="x")
    assert f.mesh is not None and f.mesh_rgb is not None
    V, F = f.mesh
    assert len(f.mesh_rgb) == len(F)
    assert len(f.xyz) > 1000 and np.allclose(f.rgb[0], np.array(RED) / 255.0, atol=1e-3)
    # The face patch and the shoes are tinted: more than one colour per face.
    assert len(np.unique(np.round(f.mesh_rgb, 3), axis=0)) >= 3
    # The point body is the mesh's surface, so point-only consumers see a body.
    assert abs(f.xyz[:, 2].max() - 1.75) < 0.03 and f.xyz[:, 2].min() > -0.01
    assert abs(f.xyz[:, 0].mean() - 1.0) < 0.1 and abs(f.xyz[:, 1].mean() - 2.0) < 0.15
    g = f.translated(0.5, 0.0, 0.0)
    assert g.mesh is not None and np.isclose(g.mesh[0][:, 0].mean() - V[:, 0].mean(), 0.5)


def test_figure_for_actor_uses_the_cast_colour():
    a = C.Actor(name="Maya", height_m=1.68, colour=(60, 130, 230))
    f = FG.figure_for_actor(a, (0.0, 0.0), 90.0, 0.0, pose="walking", phase=0.4)
    assert f.label == "Maya"
    assert np.allclose(f.rgb[0], np.array([60, 130, 230]) / 255.0, atol=1e-3)
    assert abs(f.mesh[0][:, 2].max() - 1.68) < 0.06


def test_figure_validates_mesh():
    with pytest.raises(ValueError):
        R.Figure(xyz=np.zeros((3, 3)), rgb=np.array([1, 0, 0]), mesh=(np.zeros((2, 3)), np.array([[0, 1, 5]])))
    with pytest.raises(ValueError):
        R.Figure(xyz=np.zeros((3, 3)), rgb=np.array([1, 0, 0]),
                 mesh=(np.zeros((3, 3)), np.array([[0, 1, 2]])), mesh_rgb=np.zeros((2, 3)))


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _changed(twin, cam, subj, figs, **kw):
    base = np.asarray(R.render_shot(twin, cam, subj, 35.0, draw_subject=False, **kw))
    with_fig = np.asarray(R.render_shot(twin, cam, subj, 35.0, draw_subject=False, figures=figs, **kw))
    return base, with_fig, np.any(base != with_fig, axis=-1)


def test_figure_two_metres_from_a_35mm_covers_a_plausible_area(clean):
    twin = clean[0].twin
    fz = float(twin.structure.floor_z)
    fig = R.stand_in(1.75, (0.0, 1.0), 270.0, fz, rgb=RED)
    t0 = time.time()
    base, with_fig, changed = _changed(twin, (0.0, -1.0, 1.2), (0.0, 1.0), [fig])
    dt = time.time() - t0
    print(f"\nfigure render: two 960 px frames (one with a figure) in {dt:.2f} s")
    frac = changed.mean()
    assert 0.01 < frac < 0.25, frac
    # Shaded: the actor's red appears in more than one shade.
    px = with_fig[changed]
    red = px[(px[:, 0] > 1.3 * px[:, 1]) & (px[:, 0] > 1.3 * px[:, 2]) & (px[:, 0] > 50)]
    assert len(red) > 0.5 * changed.sum()
    assert len(np.unique(red[:, 0])) >= 2
    # A real spread of shades, not two values a pixel apart.
    assert np.ptp(red[:, 0]) > 40


def test_two_figures_add_little_time(clean):
    twin = clean[0].twin
    fz = float(twin.structure.floor_z)
    cam = R.camera_at((0.0, -1.0, 1.4), (0.0, 0.6, 1.5), 35.0, width_px=960)
    figs = [R.stand_in(1.75, (-0.45, 0.6), 270.0, fz, rgb=RED),
            R.stand_in(1.68, (0.6, 2.0), 250.0, fz, pose="walking", phase=0.15, rgb=(60, 130, 230))]
    img = np.zeros((cam.height_px, cam.width_px, 3), dtype=np.uint8)
    R.draw_figures(img, twin, cam, figs)  # warm
    t0 = time.time()
    out = R.draw_figures(img, twin, cam, figs)
    dt = time.time() - t0
    print(f"\ndraw_figures: two figures at 960x{cam.height_px} in {dt * 1000:.0f} ms")
    assert out.shape == img.shape and np.any(out != img)


def test_measured_furniture_occludes_a_figure_behind_it(clean):
    # The clean room has a 1.15 m cabinet at x 1.5-2.4, y -1.85 to -0.95.
    # A figure against the far wall behind it, seen from a low camera on
    # the near side, loses its legs; the same figure in the open does not.
    twin = clean[0].twin
    fz = float(twin.structure.floor_z)
    cam_b, mark_b = (1.95, 0.6, 0.8), (1.95, -1.85)
    cam_o, mark_o = (-0.5, 0.6, 0.8), (-0.5, -1.85)
    _, _, behind = _changed(twin, cam_b, mark_b, [R.stand_in(1.75, mark_b, 90.0, fz, rgb=RED)])
    _, _, open_ = _changed(twin, cam_o, mark_o, [R.stand_in(1.75, mark_o, 90.0, fz, rgb=RED)])
    assert open_.sum() > 20_000
    assert behind.sum() < 0.75 * open_.sum(), (behind.sum(), open_.sum())
    # The occlusion is at the bottom: the head row is intact.
    H = behind.shape[0]
    assert behind[: H // 2].sum() > 0.8 * open_[: H // 2].sum()


def test_figure_in_front_of_the_furniture_is_not_occluded(clean):
    twin = clean[0].twin
    fz = float(twin.structure.floor_z)
    cam_f, mark_f = (1.95, 0.9, 0.8), (1.95, -0.6)
    cam_o, mark_o = (-0.5, 0.9, 0.8), (-0.5, -0.6)
    _, _, front = _changed(twin, cam_f, mark_f, [R.stand_in(1.75, mark_f, 90.0, fz, rgb=RED)])
    _, _, open_ = _changed(twin, cam_o, mark_o, [R.stand_in(1.75, mark_o, 90.0, fz, rgb=RED)])
    assert open_.sum() > 20_000
    assert front.sum() > 0.92 * open_.sum(), (front.sum(), open_.sum())


def test_feet_are_not_clipped_by_the_floor(clean):
    twin = clean[0].twin
    fz = float(twin.structure.floor_z)
    # A high camera looking down at a standing figure: the floor behind the
    # feet is nearer in the depth map than it would be flat on, and the
    # tolerance keeps the shoes.
    fig = R.stand_in(1.75, (0.0, 0.8), 270.0, fz, rgb=RED)
    cam_pos, focal, hh = (0.0, -1.6, 2.0), 24.0, 0.6
    base = np.asarray(R.render_shot(twin, cam_pos, (0.0, 0.8), focal, draw_subject=False, subject_height_m=hh))
    with_fig = np.asarray(R.render_shot(twin, cam_pos, (0.0, 0.8), focal, draw_subject=False,
                                        subject_height_m=hh, figures=[fig]))
    changed = np.any(base != with_fig, axis=-1)
    cam = R.camera_at(cam_pos, (0.0, 0.8, fz + 0.94 * hh), focal, width_px=960)
    u, v = cam.project((0.0, 0.8, fz + 0.02))
    assert 20 < v < cam.height_px - 20
    rows = changed[int(v) - 12: int(v) + 4, int(u) - 60: int(u) + 60]
    assert rows.sum() > 50


def test_point_figures_still_take_the_point_path(clean):
    twin = clean[0].twin
    g = np.linspace(-0.3, 0.3, 25)
    X, Z = np.meshgrid(g, g + 1.0)
    xyz = np.stack([X.ravel(), np.full(X.size, 1.0), Z.ravel()], axis=1)
    fig = R.Figure(xyz=xyz, rgb=np.array([255, 0, 255]), label="block")
    assert fig.mesh is None
    im = np.asarray(R.render_shot(twin, (0.0, -1.5, 1.5), (0.0, 1.0), 32.0, draw_subject=False, figures=[fig]))
    magenta = (im[..., 0] > 200) & (im[..., 1] < 60) & (im[..., 2] > 200)
    assert magenta.sum() > 200


REAL_TWIN = Path("twins/studio/d3752e88f78a/IMG_6191.twin")


@pytest.mark.skipif(not REAL_TWIN.exists(), reason="real scan not present")
def test_figure_over_a_gaussian_field_twin():
    from locaish.types import Twin

    twin = Twin.load(REAL_TWIN)
    if R.gaussian_field(twin) is None:
        pytest.skip("twin has no usable field")
    fz = float(twin.structure.floor_z)
    fig = R.stand_in(1.68, (-0.4, 0.0), 270.0, fz, rgb=RED)
    base = np.asarray(R.render_shot(twin, (0.0, -1.6, 1.4), (0.0, 0.0), 35.0, width_px=480, draw_subject=False))
    with_fig = np.asarray(R.render_shot(twin, (0.0, -1.6, 1.4), (0.0, 0.0), 35.0, width_px=480,
                                        draw_subject=False, figures=[fig]))
    changed = np.any(base != with_fig, axis=-1)
    assert 0.01 < changed.mean() < 0.5
