"""The cast: real heights on real marks, stored, drawn, and swept against."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from locaish.film import render as R
from locaish.film import sweep as sweepmod
from locaish.product import cast as C


def _span(x: np.ndarray) -> float:
    return float(x.max() - x.min())


# ---------------------------------------------------------------------------
# the actor and the store
# ---------------------------------------------------------------------------


def test_actor_validation_and_defaults():
    a = C.Actor("Ana", 1.62)
    assert a.height_m == 1.62
    assert math.isclose(a.eye_height_m, sweepmod.EYELINE_RATIO * 1.62, abs_tol=1e-3)
    assert a.source == "typed" and a.body_path is None and len(a.id) == 10
    with pytest.raises(ValueError, match="metres"):
        C.Actor("Tall", 175)             # centimetres typed by mistake
    with pytest.raises(ValueError):
        C.Actor("Short", 1.1)
    with pytest.raises(ValueError):
        C.Actor("", 1.7)
    with pytest.raises(ValueError):
        C.Actor("Odd eyes", 1.7, eye_height_m=0.4)
    with pytest.raises(ValueError):
        C.Actor("Bad source", 1.7, source="guessed")
    with pytest.raises(ValueError):
        C.Actor("Bad colour", 1.7, colour=(300, 0, 0))


def test_store_round_trip(tmp_path):
    assert C.load_cast(tmp_path) == []
    a = C.add_actor(tmp_path, "Ana", 1.62, headshot="headshots/ana.jpg")
    b = C.add_actor(tmp_path, "Ben", 1.88, eye_height_m=1.75)
    assert a.colour == C.PALETTE[0] and b.colour == C.PALETTE[1]

    raw = json.loads((tmp_path / C.CAST_FILE).read_text())
    assert raw["version"] == C.CAST_VERSION
    assert [x["name"] for x in raw["actors"]] == ["Ana", "Ben"]
    assert raw["actors"][0]["headshot"] == "headshots/ana.jpg"
    assert set(raw["actors"][0]) >= {
        "id", "name", "height_m", "eye_height_m", "colour", "headshot",
        "body_path", "source", "created_at",
    }

    cast = C.load_cast(tmp_path)
    assert [x.id for x in cast] == [a.id, b.id]
    assert cast[1].eye_height_m == 1.75 and cast[0].colour == a.colour
    assert C.get_actor(tmp_path, b.id).name == "Ben"

    assert C.remove_actor(tmp_path, a.id) is True
    assert C.remove_actor(tmp_path, a.id) is False
    assert [x.name for x in C.load_cast(tmp_path)] == ["Ben"]
    # The freed colour is reused before the palette wraps.
    c = C.add_actor(tmp_path, "Cy", 1.70)
    assert c.colour == C.PALETTE[0]


def test_store_survives_a_bad_record(tmp_path):
    (tmp_path / C.CAST_FILE).write_text(json.dumps({
        "version": 1,
        "actors": [{"name": "Ok", "height_m": 1.7}, {"name": "Broken", "height_m": 17.0}],
    }))
    assert [a.name for a in C.load_cast(tmp_path)] == ["Ok"]


# ---------------------------------------------------------------------------
# the capsule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h", [1.55, 1.75, 2.0])
def test_capsule_point_count_and_extents(h):
    pts = C.capsule_points(h)
    assert pts.shape[1] == 3
    assert 1500 <= len(pts) <= 4500
    assert abs(pts[:, 2].min()) < 0.02
    assert abs(pts[:, 2].max() - h) < 0.02
    width = _span(pts[:, 0])          # shoulders plus hanging arms
    depth = _span(pts[:, 1])
    s = h / C.REFERENCE_HEIGHT_M
    assert 0.46 * s < width < 0.75 * s
    assert 0.15 * s < depth < 0.35 * s
    assert width > depth              # facing +Y: wider across than deep
    # The head is the top: points in the last 0.2 m are a sphere ~0.22 m wide.
    head = pts[pts[:, 2] > h - 0.2 * s]
    assert 0.15 * s < _span(head[:, 0]) < 0.26 * s


def test_capsule_is_deterministic_and_spacing_scales_count():
    a = C.capsule_points(1.75)
    b = C.capsule_points(1.75)
    assert np.array_equal(a, b)
    coarse = C.capsule_points(1.75, spacing_m=0.08)
    assert len(coarse) < len(a) / 2


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------


def test_figure_lands_on_the_mark_and_turns_with_facing():
    a = C.Actor("Ana", 1.62, colour=(230, 80, 60))
    f = C.figure(a, (2.0, -3.0), 90.0, 0.4)
    assert isinstance(f, R.Figure) and f.label == "Ana"
    assert np.allclose(f.xyz[:, :2].mean(axis=0), [2.0, -3.0], atol=0.02)
    assert abs(f.xyz[:, 2].min() - 0.4) < 0.02
    assert abs(f.xyz[:, 2].max() - (0.4 + 1.62)) < 0.02
    assert np.allclose(f.rgb[0], np.array([230, 80, 60]) / 255.0)
    assert f.radius_m == pytest.approx(C.FIGURE_RADIUS_M)
    # Facing +Y (90) puts the shoulder axis along X; facing +X (0) swings it to Y.
    assert _span(f.xyz[:, 0]) > _span(f.xyz[:, 1]) * 1.8
    g = C.figure(a, (2.0, -3.0), 0.0, 0.4)
    assert _span(g.xyz[:, 1]) > _span(g.xyz[:, 0]) * 1.8
    assert np.isclose(_span(g.xyz[:, 1]), _span(f.xyz[:, 0]))
    # A 45 degree turn is in between, and the mark and height are untouched.
    h45 = C.figure(a, (2.0, -3.0), 45.0, 0.4)
    assert np.allclose(h45.xyz[:, :2].mean(axis=0), [2.0, -3.0], atol=0.02)
    assert np.isclose(_span(h45.xyz[:, 0]), _span(h45.xyz[:, 1]), rtol=0.1)


def test_figure_reads_floor_from_the_twin(clean):
    twin = clean[0].twin
    a = C.Actor("Ana", 1.62)
    f = C.figure(a, (0.0, 1.0), 270.0, twin=twin)
    assert abs(f.xyz[:, 2].min() - float(twin.structure.floor_z)) < 0.02
    with pytest.raises(ValueError):
        C.figure(a, (0.0, 1.0), 270.0)


# ---------------------------------------------------------------------------
# the body npz
# ---------------------------------------------------------------------------


def _box_body(path, *, extents=(0.4, 0.2, 1.0), rgb=False, height_m=1.0, **extra):
    import trimesh

    box = trimesh.creation.box(extents=extents)
    v = np.asarray(box.vertices, dtype=np.float64)
    v[:, 2] -= v[:, 2].min()              # feet at z=0 per the contract
    v[:, 0] += 0.3                        # off-centre on purpose: figure() recentres
    arrays = {"vertices": v, "faces": np.asarray(box.faces, dtype=np.int64)}
    if height_m is not None:
        arrays["height_m"] = np.float64(height_m)
    if rgb:
        col = np.zeros_like(v)
        col[:, 2] = 1.0                   # a blue body
        arrays["rgb"] = col
    arrays.update(extra)
    np.savez(path, **arrays)
    return path


def test_body_npz_is_scaled_to_the_typed_height(tmp_path):
    npz = _box_body(tmp_path / "ana_body.npz")
    body = C.load_body(npz)
    assert body.height_m == 1.0 and body.rgb is None
    assert body.extent_z_m == pytest.approx(1.0)

    scaled = C.scale_to_height(body, 1.80)
    assert scaled.extent_z_m == pytest.approx(1.80)
    assert scaled.vertices[:, 2].min() == pytest.approx(0.0)
    assert _span(scaled.vertices[:, 0]) == pytest.approx(0.4 * 1.8)   # uniform
    assert np.allclose(scaled.vertices[:, :2].mean(axis=0), 0.0, atol=1e-9)
    assert scaled.height_m == 1.0                                      # the model's claim, kept

    a = C.add_actor(tmp_path, "Ana", 1.80)
    C.attach_body(a, npz, workdir=tmp_path)
    assert a.source == "sam3d-body" and a.body_path == "ana_body.npz"
    stored = C.load_cast(tmp_path)[0]
    assert stored.body_path == "ana_body.npz" and stored.source == "sam3d-body"

    f = C.figure(stored, (1.0, 1.0), 90.0, 0.0, workdir=tmp_path)
    assert 2000 <= len(f.xyz) <= 4000
    assert abs(f.xyz[:, 2].max() - 1.80) < 0.03 and abs(f.xyz[:, 2].min()) < 0.03
    assert np.allclose(f.xyz[:, :2].mean(axis=0), [1.0, 1.0], atol=0.03)
    assert abs(_span(f.xyz[:, 0]) - 0.72) < 0.03                      # 0.4 * 1.8
    assert np.allclose(f.rgb[0], np.asarray(a.colour) / 255.0)

    desc = C.describe([stored], workdir=tmp_path)[0]
    assert desc["body"]["attached"] is True
    assert desc["body"]["model_height_m"] == 1.0
    assert desc["body"]["model_vs_typed_m"] == pytest.approx(-0.8)
    assert desc["default_difference_m"] == pytest.approx(0.05)

    C.detach_body(stored, workdir=tmp_path)
    assert C.load_cast(tmp_path)[0].body_path is None


def test_body_with_rgb_keeps_its_own_colours(tmp_path):
    npz = _box_body(tmp_path / "blue.npz", rgb=True)
    a = C.Actor("Blue", 1.7, colour=(230, 80, 60))
    C.attach_body(a, npz)
    f = C.figure(a, (0.0, 0.0), 90.0, 0.0)
    assert np.allclose(f.rgb, [0.0, 0.0, 1.0])


def test_body_npz_contract_is_enforced(tmp_path):
    import trimesh

    box = trimesh.creation.box(extents=(0.4, 0.2, 1.0))
    np.savez(tmp_path / "nofaces.npz", vertices=np.asarray(box.vertices))
    with pytest.raises(ValueError, match="faces"):
        C.load_body(tmp_path / "nofaces.npz")
    np.savez(tmp_path / "badidx.npz", vertices=np.asarray(box.vertices),
             faces=np.asarray(box.faces) + 100)
    with pytest.raises(ValueError, match="index"):
        C.load_body(tmp_path / "badidx.npz")
    np.savez(tmp_path / "badrgb.npz", vertices=np.asarray(box.vertices),
             faces=np.asarray(box.faces), rgb=np.ones((3, 3)))
    with pytest.raises(ValueError, match="rgb"):
        C.load_body(tmp_path / "badrgb.npz")
    with pytest.raises(FileNotFoundError):
        C.load_body(tmp_path / "missing.npz")
    a = C.Actor("Ana", 1.7)
    with pytest.raises(ValueError):
        C.attach_body(a, tmp_path / "nofaces.npz")
    assert a.body_path is None and a.source == "typed"


# ---------------------------------------------------------------------------
# eyelines
# ---------------------------------------------------------------------------


def test_pair_eyelines_arithmetic():
    ana = C.Actor("Ana", 1.62)
    ben = C.Actor("Ben", 1.88)
    r = C.pair_eyelines(ana, ben)
    assert r["height_difference_m"] == pytest.approx(-0.26)
    eye_diff = sweepmod.EYELINE_RATIO * (1.62 - 1.88)
    assert r["eye_difference_m"] == pytest.approx(round(eye_diff, 4))
    assert r["eyeline_angle_deg"] == pytest.approx(math.degrees(math.atan(abs(eye_diff) / 1.2)), abs=0.02)
    assert r["taller"] == "Ben" and r["shorter"] == "Ana"
    assert r["over_shoulder_ok"] is False
    assert r["apple_box_m"] == 0.1 and r["apple_box_for"] == "Ana"
    assert "apple box" in r["note"]

    # Closer than 0.15 m at the eyes: no box.
    cy = C.Actor("Cy", 1.70)
    ok = C.pair_eyelines(ana, cy)
    assert ok["over_shoulder_ok"] is True and ok["apple_box_m"] is None
    # A full apple when the half is not enough.
    big = C.pair_eyelines(C.Actor("Tiny", 1.50), C.Actor("Huge", 1.90))
    assert big["apple_box_m"] == 0.2 and big["apple_box_for"] == "Tiny"
    # The angle falls with distance; the order of arguments only flips the sign.
    far = C.pair_eyelines(ana, ben, distance_m=3.0)
    assert far["eyeline_angle_deg"] < r["eyeline_angle_deg"]
    assert C.pair_eyelines(ben, ana)["eye_difference_m"] == -r["eye_difference_m"]


def test_describe_is_json_ready():
    cast = [C.Actor("Ana", 1.62), C.Actor("Ben", 1.88, colour=(1, 2, 3))]
    d = C.describe(cast)
    json.dumps(d)
    assert [x["name"] for x in d] == ["Ana", "Ben"]
    assert d[1]["colour"] == [1, 2, 3] and d[1]["body"] is None
    assert d[0]["eye_height_m"] == pytest.approx(round(0.94 * 1.62, 4))


# ---------------------------------------------------------------------------
# the sweep and the frame
# ---------------------------------------------------------------------------


def test_sweep_frames_against_the_real_stature(clean):
    twin = clean[0].twin
    default = sweepmod.sweep(twin)
    short = sweepmod.sweep(twin, subject_height_m=1.55)

    # Same room, same marks, same camera cells: only the person changed.
    assert short.camera_cells == default.camera_cells
    assert np.array_equal(short.subject_marks, default.subject_marks)
    assert list(short.columns) == list(default.columns)

    # The row set changes only where the person's height is the cut. A shorter
    # subject fills less of the frame, so more tight setups survive the fill
    # cut; and her eyes are lower, so a low camera right beside the mark can
    # fall inside the minimum camera-to-eye distance. Every row that differs
    # must be one of those two, and the sweep must have more rows, not fewer.
    key = ("cam_x", "cam_y", "cam_z", "subj_x", "subj_y", "focal_mm")

    def keyed(sw):
        cols = [sw.columns[k].astype(np.float64) for k in key]
        return {tuple(round(float(c[i]), 4) for c in cols): i for i in range(len(sw))}

    kd, ks = keyed(default), keyed(short)
    assert len(short) > len(default)
    extra = [ks[k] for k in set(ks) - set(kd)]
    assert extra, "a 20 cm shorter subject should let more tight setups through the fill cut"
    rescaled = short.columns["subject_fill"][extra] * (sweepmod.SUBJECT_HEIGHT_M / 1.55)
    assert np.all(rescaled > sweepmod.MAX_SUBJECT_FILL - 1e-3)
    lost = [kd[k] for k in set(kd) - set(ks)]
    if lost:
        dx = default.columns["cam_x"][lost] - default.columns["subj_x"][lost]
        dy = default.columns["cam_y"][lost] - default.columns["subj_y"][lost]
        dz = default.columns["cam_z"][lost] - sweepmod.EYELINE_RATIO * 1.55
        assert np.all(np.sqrt(dx * dx + dy * dy + dz * dz) < sweepmod.MIN_DISTANCE_M + 1e-3)

    # On the rows both sweeps share, framing follows the stature exactly...
    common = sorted(set(kd) & set(ks))
    i_d = np.array([kd[k] for k in common])
    i_s = np.array([ks[k] for k in common])
    # fill = stature / framed height, and framed height is set by the distance
    # to the eyes, which also moved; so the ratio is the stature ratio times
    # the distance ratio, exactly.
    ratio = short.columns["subject_fill"][i_s] / default.columns["subject_fill"][i_d]
    expect = (1.55 / sweepmod.SUBJECT_HEIGHT_M) * (
        default.columns["distance_m"][i_d] / short.columns["distance_m"][i_s]
    )
    assert np.allclose(ratio, expect, rtol=1e-3)
    assert np.any(short.columns["shot_size"][i_s] != default.columns["shot_size"][i_d])
    # ...and the aim point moved down to her eyes, so the camera pitches
    # lower and the lens-only columns do not change at all.
    assert np.all(short.columns["pitch_deg"][i_s] <= default.columns["pitch_deg"][i_d] + 1e-4)
    assert np.any(short.columns["pitch_deg"][i_s] < default.columns["pitch_deg"][i_d] - 0.1)
    assert np.array_equal(short.columns["fov_h_deg"][i_s], default.columns["fov_h_deg"][i_d])
    assert np.array_equal(short.columns["focal_mm"][i_s], default.columns["focal_mm"][i_d])

    with pytest.raises(ValueError):
        sweepmod.sweep(twin, subject_height_m=175)


def test_actor_figure_is_drawn_in_the_frame(clean):
    twin = clean[0].twin
    actor = C.Actor("Ana", 1.62, colour=(230, 80, 60))
    cam, subj = (0.0, -1.5, 1.5), (0.0, 1.0)
    fig = C.figure(actor, subj, 270.0, twin=twin)       # facing the camera
    base = np.asarray(R.render_shot(twin, cam, subj, 32.0, draw_subject=False,
                                    subject_height_m=actor.height_m)).astype(int)
    img = np.asarray(R.render_shot(twin, cam, subj, 32.0, draw_subject=False, figures=[fig],
                                   subject_height_m=actor.height_m)).astype(int)
    target = np.array(actor.colour)
    close = np.all(np.abs(img - target) <= 12, axis=-1)
    assert close.sum() > 300
    assert np.all(np.abs(base - target) <= 12, axis=-1).sum() < close.sum() // 10
