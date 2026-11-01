"""The world builder: the room as it will be on the day, from the twin.

What is tested is the contract the stage and the viewer rely on, not the
look of the result. Cameras stand where a person can stand and look at
things inside the room; the coverage check is honest about what is seen;
a rendered view is exactly the frame the camera model claims, with nothing
burnt into it; the COLMAP poses are the exact inverse of the cameras that
rendered the frames; and the twin that comes out carries the new splat in
the twin frame with the identity transform the viewer and the film renderer
both apply. Brush is exercised once, briefly, when it is installed, because
the one thing a monkeypatched trainer cannot prove is that the dataset
format is accepted.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from locaish.film import render, space
from locaish.product import world as W
from locaish.types import Twin
from locaish.video.splat import SH_C0, brush_binary, read_gaussian_ply
from locaish.viewer import build

HAVE_BRUSH = brush_binary() is not None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _in_box(xy, lo, hi) -> bool:
    return bool(lo[0] <= xy[0] <= hi[0] and lo[1] <= xy[1] <= hi[1])


def _inside_polygon(poly, xy) -> bool:
    return bool(space._points_in_polygon(np.array([xy[0]]), np.array([xy[1]]), poly)[0])


def write_fake_gaussian_ply(twin: Twin, path: Path, count: int = 3000) -> Path:
    """A tiny but valid 3DGS PLY, with the property names `read_gaussian_ply`
    maps by, seeded from the twin's own measured points so it falls inside
    the bounds the viewer and the film renderer filter on."""
    pts = twin.points.measured()
    pick = np.random.default_rng(0).choice(len(pts.xyz), min(count, len(pts.xyz)), replace=False)
    xyz = pts.xyz[pick].astype(np.float32)
    rgb = (pts.rgb[pick].astype(np.float32) / 255.0) if pts.rgb is not None else np.full((len(xyz), 3), 0.6, np.float32)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    arr = np.zeros(len(xyz), dtype=[(k, "<f4") for k in names])
    arr["x"], arr["y"], arr["z"] = xyz.T
    for i in range(3):
        arr[f"f_dc_{i}"] = (rgb[:, i] - 0.5) / SH_C0
        arr[f"scale_{i}"] = np.log(0.03)
    arr["opacity"] = 2.2          # logit: ~0.9
    arr["rot_0"] = 1.0
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(arr)
              + "".join(f"property float {k}\n" for k in names) + "end_header\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header.encode("ascii") + arr.tobytes())
    return path


def _fake_trainer(twin: Twin):
    def train(images_dir, model_dir, out_dir, *, steps=6000, resolution=960, progress=None):
        out_dir = Path(out_dir)
        assert Path(images_dir).is_dir() and (Path(model_dir) / "images.txt").exists()
        return write_fake_gaussian_ply(twin, out_dir / f"splat_{steps}.ply")
    return train


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def test_select_views_stand_in_the_open_and_look_into_the_room(clean):
    result, fx = clean
    twin = result.twin
    maps = space.floor_maps(twin)
    views, cov = W.select_views(twin, maps=maps, n=60)
    assert len(views) == 60
    assert len({v.name for v in views}) == 60 and views[0].name == "view_0000"
    poly = twin.structure.footprint
    stand = maps.standable() & (maps.clearance_m >= W.WALL_CLEARANCE_M)
    for v in views:
        xy = np.array(v.cam[:2])
        # On a standable cell, 0.35 m from anything solid.
        assert maps.sample(stand, xy)[0], (v.name, v.cam)
        # Never inside the furniture the fixture put in the room.
        for lo, hi in fx.truth.boxes:
            assert not _in_box(xy, lo, hi), (v.name, v.cam)
        # Looking at somewhere inside the room.
        assert _inside_polygon(poly, v.aim[:2]), (v.name, v.aim)
        assert v.width_px == 960 and v.height_px == W.frame_size(960)[1]
    heights = {round(v.cam[2] - float(twin.structure.floor_z), 1) for v in views}
    assert {1.0, 1.5, 2.0} <= heights
    assert cov["roles"].get("corner", 0) >= 3


def test_select_views_coverage_sees_every_wall_and_opening_three_times(clean):
    twin = clean[0].twin
    views, cov = W.select_views(twin, n=60)
    assert cov["walls"] >= 4 and cov["openings"] == len(twin.structure.openings) == 3
    assert cov["complete"], cov["uncovered"]
    assert cov["least_seen"] >= 3
    for row in cov["targets"]:
        assert row["views"] >= 3, row
    assert cov["requested"] == 60 and cov["returned"] == 60
    # The coverage dict is JSON, because the stage writes it to disk.
    json.dumps(cov)


def test_select_views_is_deterministic_and_round_trips(clean):
    twin = clean[0].twin
    a, _ = W.select_views(twin, n=20, seed=3)
    b, _ = W.select_views(twin, n=20, seed=3)
    assert [v.to_dict() for v in a] == [v.to_dict() for v in b]
    again = [W.View.from_dict(v.to_dict()) for v in a]
    assert [v.to_dict() for v in again] == [v.to_dict() for v in a]


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def test_render_view_is_the_exact_frame_with_no_caption(clean):
    twin = clean[0].twin
    views, _ = W.select_views(twin, n=12)
    v = views[0]
    im = W.render_view(twin, v)
    cam = render.camera_at(v.cam, v.aim, v.focal_mm, width_px=v.width_px)
    assert im.size == (v.width_px, v.height_px) == (cam.width_px, cam.height_px)
    arr = np.asarray(im)
    # No caption strip: the bottom rows are room, not a bar of background
    # with text on it. A caption would be a band of near-uniform background
    # colour; the floor seen from 1.5 m is not uniform.
    bottom = arr[-24:]
    bg = np.array(render.BACKGROUND, dtype=np.int32)
    is_bg = np.all(np.abs(bottom.astype(np.int32) - bg) < 3, axis=-1)
    assert is_bg.mean() < 0.95
    # A figure in the frame draws into it and keeps the size.
    fz = float(twin.structure.floor_z)
    fig = render.stand_in(1.75, (float(v.aim[0]), float(v.aim[1])), 90.0, fz, rgb=(220, 40, 40), label="A")
    im2 = W.render_view(twin, v, [fig])
    assert im2.size == im.size
    assert np.any(np.asarray(im2) != arr)


def test_render_views_writes_pngs_and_the_finisher_meta(clean, tmp_path):
    twin = clean[0].twin
    views, _ = W.select_views(twin, n=12)
    fz = float(twin.structure.floor_z)
    fig = render.stand_in(1.7, (0.0, 0.0), 0.0, fz, rgb=(200, 50, 50), label="Ana")
    light = {"azimuth_deg": 200.0, "elevation_deg": 30.0, "clock": "16:00", "date": "2027-03-01", "windows_lit": []}
    items = W.render_views(twin, views[:4], tmp_path, figures_for=[fig], light=light, dressing="bare")
    assert len(items) == 4
    for name, path, meta in items:
        assert path.exists() and path.name == f"{name}.png"
        assert meta["name"] == name and meta["camera"]["focal_mm"] == 24.0
        assert meta["light"] == light and meta["dressing"] == "bare"
        assert meta["neighbours"] == []
        assert "Ana" in meta["actors"] and len(meta["actors"]["Ana"]) == 3
        # How much of the frame the twin painted. The synthetic room is a
        # sparse point cloud, so the number is small here; what matters is
        # that it is a fraction the finisher can read before inventing.
        assert 0.0 < meta["painted"] <= 1.0
    saved = json.loads((tmp_path / "views.json").read_text())
    assert len(saved["views"]) == 4 and saved["views"][0]["name"] == views[0].name


# ---------------------------------------------------------------------------
# the camera model
# ---------------------------------------------------------------------------


def _parse_images_txt(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    out = {}
    lines = [l for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    for l in lines:
        parts = l.split()
        if len(parts) < 10:
            continue
        q = np.array([float(x) for x in parts[1:5]])
        t = np.array([float(x) for x in parts[5:8]])
        out[parts[9]] = (q, t, int(parts[8]))
    return out


def _rot_from_quat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def test_write_colmap_model_poses_invert_the_render_cameras(clean, tmp_path):
    twin = clean[0].twin
    views, _ = W.select_views(twin, n=12)
    size = W.frame_size(960)
    W.write_colmap_model(twin, views, size, tmp_path)
    for name in ("cameras.txt", "images.txt", "points3D.txt", "README.txt"):
        assert (tmp_path / name).exists()
    cams = [l.split() for l in (tmp_path / "cameras.txt").read_text().splitlines() if l and not l.startswith("#")]
    assert len(cams) == 1 and cams[0][1] == "PINHOLE"
    w, h, fx, fy, cx, cy = int(cams[0][2]), int(cams[0][3]), *map(float, cams[0][4:8])
    cam0 = render.camera_at(views[0].cam, views[0].aim, 24.0, width_px=960)
    assert (w, h) == size == (cam0.width_px, cam0.height_px)
    assert fx == pytest.approx(cam0.fx, rel=1e-6) and fy == pytest.approx(cam0.fy, rel=1e-6)
    assert (cx, cy) == (w / 2.0, h / 2.0)

    poses = _parse_images_txt(tmp_path / "images.txt")
    assert len(poses) == len(views)
    for v in views:
        q, t, cam_id = poses[f"{v.name}.png"]
        assert cam_id == 1
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-6)
        R = _rot_from_quat(q)
        # The camera's own position lands at the origin of camera space...
        assert np.allclose(R @ np.array(v.cam) + t, 0.0, atol=1e-5)
        # ...and the aim point lands on +z, dead centre.
        p = R @ np.array(v.aim) + t
        assert p[2] > 0 and abs(p[0]) < 1e-5 and abs(p[1]) < 1e-5, p
        # x right, y down: a point to the camera's right has +x, one above has -y.
        cam = render.camera_at(v.cam, v.aim, 24.0, width_px=960)
        r = R @ (np.array(v.cam) + cam.right) + t
        u = R @ (np.array(v.cam) + np.array([0, 0, 1.0])) + t
        assert r[0] > 0.99 and u[1] < 0

    pts = [l.split() for l in (tmp_path / "points3D.txt").read_text().splitlines() if l and not l.startswith("#")]
    assert len(pts) == min(W.POINTS3D_N, len(twin.points.measured()))
    xyz = np.array([[float(x) for x in p[1:4]] for p in pts[:500]])
    lo, hi = twin.points.bounds
    assert np.all(xyz >= lo - 1e-3) and np.all(xyz <= hi + 1e-3)
    assert all(0 <= int(c) <= 255 for p in pts[:50] for c in p[4:7])


# ---------------------------------------------------------------------------
# the world twin and the whole build
# ---------------------------------------------------------------------------


def test_world_twin_is_in_the_twin_frame(clean, tmp_path):
    twin = clean[0].twin
    ply = write_fake_gaussian_ply(twin, tmp_path / "s.ply")
    wt = W.world_twin(twin, ply, world_meta={"date": "2027-03-01", "hour": "16:00", "dressing": "x", "views": 60})
    assert np.allclose(wt.canonical_transform, np.eye(4))
    assert wt.provenance["steps"]["video"]["scale"]["factor"] == 1.0
    assert wt.provenance["splat_ply"] == str(ply.resolve())
    assert wt.provenance["world"]["date"] == "2027-03-01" and wt.provenance["world"]["built_at"]
    assert wt.points is twin.points and wt.structure is twin.structure
    # The source twin is untouched: it still has no splat and the same transform.
    assert "splat_ply" not in (twin.provenance or {}) or twin.provenance["splat_ply"] != str(ply.resolve())
    # Both consumers of the view layer read it as a field in the twin frame.
    fld = render.gaussian_field(wt)
    assert fld is not None and len(fld.xyz) >= 1000
    assert np.allclose(fld.xyz[:5], read_gaussian_ply(ply)["xyz"][:5], atol=1e-5)
    payload = build._splat_payload(wt, np.zeros(3))
    assert payload is not None and payload["count"] >= 1000


def test_build_world_end_to_end_with_the_measured_finisher(clean, tmp_path, monkeypatch):
    twin = clean[0].twin
    monkeypatch.setattr(W, "train_world", _fake_trainer(twin))
    events = []
    res = W.build_world(twin, tmp_path, finish_fn=W.measured_finisher, n_views=12, steps=50,
                        dressing="bare", world_meta={"date": "2027-03-01", "hour": "16:00"},
                        progress=lambda w, d, n: events.append((w, d, n)))
    d = res.to_dict()
    json.dumps(d)
    assert len(res.views) == 12 and len(res.rendered) == 12 and len(res.finished) == 12
    assert (tmp_path / "world.html").exists() and res.html == tmp_path / "world.html"
    assert (tmp_path / "coverage.json").exists() and (tmp_path / "world.json").exists()
    # Finished frames: exactly the view's size, written under the contract's names.
    from PIL import Image

    for v, fp in zip(res.views, res.finished):
        assert fp.name == f"finish_{v.name}.png"
        assert Image.open(fp).size == (v.width_px, v.height_px)
        assert (tmp_path / "colmap" / "images" / f"{v.name}.png").exists()
    assert (tmp_path / "colmap" / "sparse" / "0" / "images.txt").exists()
    # The world twin loads in the twin frame with the new splat.
    wt = Twin.load(tmp_path / "world.twin")
    assert np.allclose(wt.canonical_transform, np.eye(4))
    assert wt.provenance["steps"]["video"]["scale"]["factor"] == 1.0
    assert wt.provenance["splat_ply"] == str(res.ply.resolve())
    assert wt.provenance["world"]["views"] == 12 and wt.provenance["world"]["hour"] == "16:00"
    # The page embeds the splat payload.
    html = (tmp_path / "world.html").read_text()
    assert '"splat":{"count":' in html
    payload = build.twin_to_payload(wt, max_points=600_000)
    assert payload["splat"] is not None and payload["splat"]["count"] >= 1000
    assert {"select", "render", "finish", "model", "train", "html", "total"} <= set(res.seconds)
    assert events and events[-1][0] == "world built"
    assert any(e[0].startswith("rendering") for e in events)


def test_build_world_resizes_what_the_finisher_changed(clean, tmp_path, monkeypatch):
    """A finisher that answers at another size, or forgets a frame, must
    not reach the trainer: the frame is resampled to the pose's exact
    size, and a missing one is stood in for by the measured render."""
    import io

    from PIL import Image

    twin = clean[0].twin
    monkeypatch.setattr(W, "train_world", _fake_trainer(twin))

    def odd_finisher(items):
        out = []
        for k, (name, png, meta) in enumerate(items):
            if k == 0:
                continue                      # forgotten
            im = Image.open(io.BytesIO(png)).resize((1024, 576))
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            out.append((name, buf.getvalue()))
        return out

    res = W.build_world(twin, tmp_path, finish_fn=odd_finisher, n_views=12, steps=50)
    for v, fp in zip(res.views, res.finished):
        assert Image.open(fp).size == (v.width_px, v.height_px)
    assert any("unfinished" in n for n in res.notes)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_flythrough_renders_from_the_world_splat(clean, tmp_path):
    twin = clean[0].twin
    ply = write_fake_gaussian_ply(twin, tmp_path / "s.ply")
    wt = W.world_twin(twin, ply, world_meta={"date": "d", "hour": "h"})
    views, _ = W.select_views(twin, n=12)
    out = W.flythrough(wt, views, tmp_path / "fly.mp4", fps=6, seconds=1.0, width_px=320)
    assert out is not None and out.exists() and out.stat().st_size > 1000


def test_flythrough_skips_cleanly_without_ffmpeg(clean, tmp_path, monkeypatch):
    twin = clean[0].twin
    monkeypatch.setattr(W.shutil, "which", lambda name: None)
    views, _ = W.select_views(twin, n=12)
    assert W.flythrough(twin, views, tmp_path / "fly.mp4", fps=6, seconds=1.0) is None
    assert not (tmp_path / "fly.mp4").exists()


# ---------------------------------------------------------------------------
# the trainer, for real
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_BRUSH, reason="Brush is not installed")
def test_brush_accepts_the_world_dataset(clean, tmp_path):
    """One short real training run: the proof that the text COLMAP model
    and the finished frames are a dataset Brush will load, and that what
    it exports lands in the twin frame rather than a renormalised one."""
    twin = clean[0].twin
    t0 = time.time()
    res = W.build_world(twin, tmp_path, finish_fn=W.measured_finisher, n_views=12, steps=300, width_px=480)
    dt = time.time() - t0
    assert res.ply.exists()
    g = read_gaussian_ply(res.ply)
    assert len(g["xyz"]) > 1000
    lo, hi = twin.points.bounds
    p2, p98 = np.percentile(g["xyz"], 2, axis=0), np.percentile(g["xyz"], 98, axis=0)
    assert np.all(p2 > lo - 0.5) and np.all(p98 < hi + 0.5), (p2, p98, lo, hi)
    wt = Twin.load(tmp_path / "world.twin")
    assert build._splat_payload(wt, np.zeros(3)) is not None
    print(f"\nbrush: 12 views at 480 px, 300 steps, {len(g['xyz']):,} gaussians in {dt:.1f} s "
          f"(train {res.seconds['train']:.1f} s)")
