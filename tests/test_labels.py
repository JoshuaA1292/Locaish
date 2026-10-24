"""Naming what is in the room: the projection chain on synthetic poses, the
ray march against the real clean room, merging, and the labels file."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from locaish.product import labels as L


def _frames(R=None, t=None, factor=0.5, n=3):
    """Pinhole cameras on a ring around the origin, looking inward, in a
    'colmap' frame that the twin transform scales by `factor` and moves."""
    R = np.eye(3) if R is None else R
    t = np.array([1.0, -2.0, 0.3]) if t is None else t
    K = np.array([[800.0, 0, 400.0], [0, 800.0, 300.0], [0, 0, 1.0]])
    extr = []
    for k in range(n):
        a = 2 * np.pi * k / n
        c = np.array([6 * np.cos(a), 6 * np.sin(a), 0.0])        # camera centre (colmap units)
        fwd = -c / np.linalg.norm(c)                               # look at the origin
        up = np.array([0, 0, 1.0])
        right = np.cross(fwd, up); right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        Rc = np.stack([right, down, fwd])                          # rows: camera x, y, z axes in world
        E = np.zeros((3, 4)); E[:, :3] = Rc; E[:, 3] = -Rc @ c
        extr.append(E)
    return L.PosedFrames(names=[f"f{k}.jpg" for k in range(n)], paths=[Path(f"f{k}.jpg") for k in range(n)],
                         extr=np.stack(extr), K=np.stack([K] * n), size=(800, 600), R=R, t=t, factor=factor)


def test_twin_colmap_round_trip_and_camera_centres():
    fr = _frames()
    p = np.array([0.3, -0.2, 0.9])
    assert np.allclose(fr.to_twin(fr.to_colmap(p)), p)
    # Camera centres carried into twin space land where the chain says.
    c0_colmap = np.array([6.0, 0.0, 0.0])
    assert np.allclose(fr.centres_twin[0], fr.to_twin(c0_colmap))


def test_projection_and_ray_are_inverses():
    fr = _frames()
    p_twin = fr.to_twin(np.array([0.4, 0.1, -0.2]))
    u, v, depth = fr.project(p_twin)
    assert np.all(depth > 0), "the point is in front of every ring camera"
    for i in range(len(fr)):
        o, d = fr.ray(i, u[i], v[i])
        # The ray from the camera through the pixel passes through the point.
        to_p = p_twin - o
        dist = np.linalg.norm(to_p - d * np.dot(to_p, d))
        assert dist < 1e-6
        # `depth` is along the optical axis; the ray is longer by the pixel's obliquity.
        K = fr.K[i]
        oblique = np.sqrt(1 + ((u[i] - K[0, 2]) / K[0, 0]) ** 2 + ((v[i] - K[1, 2]) / K[1, 1]) ** 2)
        assert np.isclose(np.dot(to_p, d), depth[i] * oblique, atol=1e-6), "depth is metric along the axis"


def test_best_views_prefer_central_and_skip_offscreen():
    fr = _frames(n=4)
    # A point straight ahead of camera 0 (on the axis from camera 0 to the origin).
    p_twin = fr.to_twin(np.array([2.0, 0.0, 0.0]))
    views = fr.best_views(p_twin, None, k=4)
    assert views and views[0]["frame"] == "f0.jpg"
    assert all(0 <= v["u"] <= 800 and 0 <= v["v"] <= 600 for v in views)
    behind = fr.to_twin(np.array([9.0, 0.0, 0.0]))       # behind camera 0, off-centre for the rest
    assert all(v["frame"] != "f0.jpg" for v in fr.best_views(behind, None, k=4))


def test_spread_picks_sharpest_per_slice():
    fr = _frames(n=3)
    fr.sharpness = np.array([1.0, 5.0, 2.0])
    assert fr.spread(1) == [1]
    assert fr.spread(5) == [0, 1, 2]


def test_ray_march_hits_the_measured_wall(clean):
    result, _fx = clean
    twin = result.twin
    tree = L.surface_tree(twin, cap=200_000)
    s = twin.structure
    # From the middle of the room, straight at the +X wall: the hit lands on it.
    origin = np.array([0.0, 0.0, 1.2])
    hit = L.first_hit(tree, origin, np.array([1.0, 0.0, 0.0]))
    assert hit is not None
    xmax = max(p[0] for p in s.footprint)
    assert abs(hit[0] - xmax) < 0.1
    # The floor between is clear; the wall is not.
    assert not L.occluded(tree, origin, np.array([0.5, 0.0, 1.2]))
    assert L.occluded(tree, origin, np.array([xmax + 1.0, 0.0, 1.2]))


def test_merge_joins_same_name_within_reach():
    dets = [
        {"label": "sofa", "kind": "furniture", "xyz": np.array([0, 0, 0.4]), "frame": "a", "uv": [1, 2]},
        {"label": "sofa", "kind": "furniture", "xyz": np.array([0.3, 0.2, 0.4]), "frame": "b", "uv": [1, 2]},
        {"label": "sofa", "kind": "furniture", "xyz": np.array([3.0, 0, 0.4]), "frame": "c", "uv": [1, 2]},
        {"label": "lamp", "kind": "lighting", "xyz": np.array([0.1, 0, 1.6]), "frame": "a", "uv": [1, 2]},
    ]
    m = L.merge(dets)
    assert [(g["label"], g["n"]) for g in m] == [("sofa", 2), ("sofa", 1), ("lamp", 1)]
    assert np.allclose(m[0]["xyz"], [0.15, 0.1, 0.4])


def test_labels_file_add_confirm_remove(tmp_path):
    assert L.load_labels(tmp_path) == []
    rec = L.add_label(tmp_path, label="  Kitchen   Island ", xyz=[1, 2, 0.9], kind="furniture", frame="f.jpg", uv=[10, 20])
    assert rec["label"] == "kitchen island" and rec["source"] == "user"
    with pytest.raises(ValueError):
        L.add_label(tmp_path, label="   ", xyz=[0, 0, 0])
    model = {**rec, "id": "m1", "source": "gemini", "label": "fridge", "confidence": 2}
    L.save_labels(tmp_path, L.load_labels(tmp_path) + [model])
    assert L.confirm_label(tmp_path, "m1", label="freezer")["source"] == "user"
    assert [l["label"] for l in L.load_labels(tmp_path)] == ["kitchen island", "freezer"]
    assert L.remove_label(tmp_path, rec["id"]) and not L.remove_label(tmp_path, "nope")
    d = L.describe(L.load_labels(tmp_path))
    assert d[0]["status"] == "confirmed"
    svg = L.labels_svg(L.load_labels(tmp_path), lambda x: x * 10, lambda y: y * 10)
    assert any("freezer" in s for s in svg)
    assert json.loads(L.labels_path(tmp_path).read_text())["labels"]


def test_scan_file_twins_have_no_frames(clean, tmp_path):
    result, _fx = clean
    assert L.load_frames(tmp_path, result.twin) is None
