"""Figures in the frame: an actor or a moved set piece drawn as part of the room."""

from __future__ import annotations

import numpy as np

from locaish.film import render as R


def test_figure_normalises_colours_and_becomes_gaussians():
    f = R.Figure(xyz=np.zeros((5, 3)), rgb=np.array([255, 0, 0]), radius_m=0.05, label="a")
    assert f.rgb.shape == (5, 3) and np.allclose(f.rgb[0], [1, 0, 0])
    g = f.as_gaussians()
    assert g.cov.shape == (5, 3, 3) and np.isclose(g.cov[0, 0, 0], 0.0025)
    assert len(g.opacity) == 5
    m = R.merge_fields([g, f.translated(1, 0).as_gaussians()])
    assert len(m.xyz) == 10 and np.isclose(m.xyz[5, 0], 1.0)


def test_figures_paint_into_the_point_render(clean):
    twin = clean[0].twin
    cam = (0.0, -1.5, 1.5)
    subj = (0.0, 1.0)
    base = np.asarray(R.render_shot(twin, cam, subj, 32.0, draw_subject=False))
    # A dense magenta block in front of the camera, at the subject's mark.
    g = np.linspace(-0.3, 0.3, 25)
    X, Z = np.meshgrid(g, g + 1.0)
    xyz = np.stack([X.ravel(), np.full(X.size, 1.0), Z.ravel()], axis=1)
    fig = R.Figure(xyz=xyz, rgb=np.array([255, 0, 255]), label="block")
    with_fig = np.asarray(R.render_shot(twin, cam, subj, 32.0, draw_subject=False, figures=[fig]))
    magenta = (with_fig[..., 0] > 200) & (with_fig[..., 1] < 60) & (with_fig[..., 2] > 200)
    assert magenta.sum() > 200
    assert not np.array_equal(base, with_fig)


def test_subject_height_moves_the_stand_in(clean):
    twin = clean[0].twin
    a = np.asarray(R.render_shot(twin, (0.0, -2.0, 1.4), (0.0, 1.0), 32.0, subject_height_m=1.55))
    b = np.asarray(R.render_shot(twin, (0.0, -2.0, 1.4), (0.0, 1.0), 32.0, subject_height_m=1.95))
    assert not np.array_equal(a, b)
