"""Mover masks, held to known answers without COLMAP or the network.

Every frame here is a synthetic JPEG whose "person" is a rectangle we drew,
so the right mask is known before the code runs: its area, its position,
how much a dilation of r pixels must grow it, and whether 60% coverage
trips the drop rule. The segmentation client is a fake that answers from a
script keyed on the frame, so the per-image and per-clip paths can both be
driven through the same frames.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from locaish.meta import masks
from locaish.meta.client import Detection, FakeMetaClient
from locaish.video import colmap
from locaish.video import reconstruct as recon
from locaish.video.frames import FrameSet, VideoInfo

W, H = 200, 120


def _jpeg(path: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
    Image.fromarray(a, "RGB").save(path, format="JPEG", quality=90)
    return path


def _mask_png(x0, y0, x1, y1, *, size=(W, H), mode="L") -> bytes:
    """A PNG mask with the object (non-zero) in a pixel rectangle."""
    w, h = size
    a = np.zeros((h, w), dtype=np.uint8)
    a[y0:y1, x0:x1] = 255
    im = Image.fromarray(a, "L")
    if mode == "RGBA":
        rgba = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        rgba.putalpha(im)
        im = rgba
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _frames(tmp_path: Path, n: int = 4) -> list[Path]:
    d = tmp_path / "frames"
    d.mkdir()
    return [_jpeg(d / f"frame_{i:04d}.jpg", i) for i in range(n)]


class ScriptedClient:
    """`segment` answers from {frame bytes: {prompt: [Detection]}}; records calls.

    `video` controls the clip path: None refuses `frames=` with an error,
    "frames" answers with per-frame indices, "noframe" answers without them.
    """

    def __init__(self, script: dict[bytes, dict[str, list[Detection]]], *, video=None, fail_on=None):
        self.script = script
        self.video = video
        self.fail_on = fail_on or set()
        self.log: list[tuple] = []

    def segment(self, prompt, image=None, *, frames=None, model=None):
        if frames is not None:
            self.log.append(("clip", prompt, len(frames)))
            if self.video is None:
                raise RuntimeError("no clip support")
            out = []
            for i, f in enumerate(frames):
                for d in self.script.get(f, {}).get(prompt, []):
                    out.append(Detection(d.label, d.box, d.score, d.mask_png,
                                         frame=(i if self.video == "frames" else None)))
            return out
        self.log.append(("image", prompt))
        if image in self.fail_on:
            raise RuntimeError("api down")
        return list(self.script.get(image, {}).get(prompt, []))


# ---------------------------------------------------------------------------
# the per-image path
# ---------------------------------------------------------------------------


def test_masks_are_written_per_frame_with_colmap_names_and_polarity(tmp_path):
    frames = _frames(tmp_path, 3)
    # frame 1 has a person in a 40x60 rectangle; the others are empty
    png = _mask_png(80, 30, 120, 90)
    script = {frames[1].read_bytes(): {"person": [Detection("person", (0.4, 0.25, 0.6, 0.75), mask_png=png)]}}
    client = ScriptedClient(script)
    out = tmp_path / "masks"
    report = masks.mask_frames(frames, out, client=client, prompts=("person",), progress=lambda s: None)

    for f in frames:
        assert (out / f"{f.name}.png").exists()
    with Image.open(out / "frame_0000.jpg.png") as im:
        assert im.mode == "L" and im.size == (W, H)
        assert np.asarray(im).min() == 255          # nothing found: all white
    with Image.open(out / "frame_0001.jpg.png") as im:
        a = np.asarray(im)
    assert a[60, 100] == 0                          # the person is black
    assert a[5, 5] == 255                           # the room is white
    assert set(np.unique(a)) <= {0, 255}

    assert report.counts == {"person": 1}
    assert report.frames_masked == 1
    assert report.masked_fraction["frame_0000.jpg"] == 0.0
    assert report.dropped == []
    assert report.provider == "meta:sam-3.1"
    # one call per frame per prompt
    assert client.log.count(("image", "person")) == 3


def test_dilation_grows_the_black_area_by_the_expected_margin(tmp_path):
    frames = _frames(tmp_path, 1)
    x0, y0, x1, y1 = 80, 30, 120, 90
    png = _mask_png(x0, y0, x1, y1)
    client = ScriptedClient({frames[0].read_bytes(): {"person": [Detection("person", (0.4, 0.25, 0.6, 0.75), mask_png=png)]}})
    masks.mask_frames(frames, tmp_path / "m", client=client, prompts=("person",))
    mover = masks.read_mask(tmp_path / "m" / "frame_0000.jpg.png")
    r = int(round(masks.DILATE_FRACTION * W))      # 4 px at 200 wide
    raw_area = (x1 - x0) * (y1 - y0)
    expected = (x1 - x0 + 2 * r) * (y1 - y0 + 2 * r)
    assert r >= 1
    assert mover.sum() == expected > raw_area
    assert mover[y0 - r, x0 - r] and not mover[y0 - r - 1, x0 - r - 1]


def test_a_box_without_a_mask_fills_the_box(tmp_path):
    frames = _frames(tmp_path, 1)
    client = ScriptedClient({frames[0].read_bytes(): {"dog": [Detection("dog", (0.0, 0.0, 0.5, 0.5), mask_png=None)]}})
    masks.mask_frames(frames, tmp_path / "m", client=client, prompts=("dog",))
    mover = masks.read_mask(tmp_path / "m" / "frame_0000.jpg.png")
    r = int(round(masks.DILATE_FRACTION * W))
    assert mover[0, 0] and mover[H // 2 - 1, W // 2 - 1]
    assert mover[H // 2 + r - 1, W // 2 + r - 1]       # grown by the dilation
    assert not mover[H // 2 + r + 1, W // 2 + r + 1]
    assert not mover[H - 1, W - 1]


def test_a_mostly_mover_frame_is_reported_for_dropping(tmp_path):
    frames = _frames(tmp_path, 2)
    # frame 0: 60% of the picture; frame 1: a small person
    big = _mask_png(0, 0, W, int(0.6 * H))
    small = _mask_png(10, 10, 30, 40)
    script = {
        frames[0].read_bytes(): {"person": [Detection("person", (0, 0, 1, 0.6), mask_png=big)]},
        frames[1].read_bytes(): {"person": [Detection("person", (0.05, 0.08, 0.15, 0.33), mask_png=small)]},
    }
    report = masks.mask_frames(frames, tmp_path / "m", client=ScriptedClient(script), prompts=("person",))
    assert report.dropped == ["frame_0000.jpg"]
    assert report.masked_fraction["frame_0000.jpg"] > masks.DROP_FRACTION
    assert report.masked_fraction["frame_0001.jpg"] < 0.1
    note = report.note()
    assert note and "2 frames had people masked out of the camera solve" in note
    assert "1 frame was dropped as mostly movers" in note


def test_the_report_is_json_serialisable_and_fingerprinted(tmp_path):
    frames = _frames(tmp_path, 2)
    png = _mask_png(0, 0, 50, 50, mode="RGBA")       # alpha-carried mask decodes too
    script = {frames[0].read_bytes(): {"cat": [Detection("cat", (0, 0, 0.25, 0.42), mask_png=png)]}}
    report = masks.mask_frames(frames, tmp_path / "m", client=ScriptedClient(script), prompts=("person", "cat"))
    d = json.loads(json.dumps(report.to_dict()))
    assert d["provider"] == "meta:sam-3.1"
    assert d["prompts"] == ["person", "cat"]
    assert d["frames_masked"] == 1 and d["frames_dropped"] == []
    assert d["counts"] == {"cat": 1}
    assert 0.0 < d["median_masked_fraction"] < 0.2
    assert len(d["fingerprint"]) == 16 and d["fingerprint"] == report.fingerprint()
    s = report.summary()
    json.dumps(s)
    assert set(s) == {"provider", "frames_masked", "median_masked_fraction", "frames_dropped", "prompts", "fingerprint"}
    # the written report round-trips
    p = masks.write_report(report, tmp_path / "m" / "report.json")
    assert json.loads(p.read_text())["fingerprint"] == report.fingerprint()


def test_a_failed_call_leaves_the_frame_all_white_and_logs_it(tmp_path):
    frames = _frames(tmp_path, 2)
    png = _mask_png(0, 0, 50, 50)
    bad = frames[0].read_bytes()
    script = {bad: {"person": [Detection("person", (0, 0, 0.25, 0.42), mask_png=png)]}}
    said: list[str] = []
    report = masks.mask_frames(frames, tmp_path / "m", client=ScriptedClient(script, fail_on={bad}),
                               prompts=("person",), progress=said.append)
    mover = masks.read_mask(tmp_path / "m" / "frame_0000.jpg.png")
    assert not mover.any()
    assert report.failed == ["frame_0000.jpg"]
    assert report.masked_fraction["frame_0000.jpg"] == 0.0
    assert any("failed" in s for s in said)
    assert report.note() is None or "could not be masked" in report.note()


def test_no_movers_anywhere_means_no_note(tmp_path):
    frames = _frames(tmp_path, 2)
    report = masks.mask_frames(frames, tmp_path / "m", client=FakeMetaClient(detections=[]), prompts=("person",))
    assert report.note() is None
    assert report.frames_masked == 0 and report.median_masked_fraction == 0.0


def test_a_mask_at_another_resolution_is_resized_to_the_frame(tmp_path):
    frames = _frames(tmp_path, 1)
    png = _mask_png(0, 0, 25, 25, size=(50, 30))      # half-size answer, top-left quarter-ish
    client = ScriptedClient({frames[0].read_bytes(): {"person": [Detection("person", (0, 0, 0.5, 0.83), mask_png=png)]}})
    masks.mask_frames(frames, tmp_path / "m", client=client, prompts=("person",))
    mover = masks.read_mask(tmp_path / "m" / "frame_0000.jpg.png")
    assert mover.shape == (H, W)
    assert mover[0, 0] and mover[90, 90] and not mover[H - 1, W - 1]


def test_mover_prompts_come_from_the_environment(monkeypatch):
    monkeypatch.delenv("LOCAISH_MOVER_PROMPTS", raising=False)
    assert masks.mover_prompts() == masks.MOVER_PROMPTS == ("person", "dog", "cat", "car", "bicycle")
    monkeypatch.setenv("LOCAISH_MOVER_PROMPTS", "pigeon, forklift ,")
    assert masks.mover_prompts() == ("pigeon", "forklift")


# ---------------------------------------------------------------------------
# the clip path
# ---------------------------------------------------------------------------


def test_the_clip_path_chunks_and_maps_frame_indices_back(tmp_path):
    frames = _frames(tmp_path, 5)
    png = _mask_png(0, 0, 100, 120)                    # left half
    script = {frames[3].read_bytes(): {"person": [Detection("person", (0, 0, 0.5, 1), mask_png=png)]}}
    client = ScriptedClient(script, video="frames")
    report = masks.mask_video_frames(frames, tmp_path / "m", client=client, prompts=("person", "dog"), chunk=2)
    # 5 frames in chunks of 2 -> 3 chunks x 2 prompts = 6 clip calls, no image calls
    assert [c[0] for c in client.log] == ["clip"] * 6
    assert [c[2] for c in client.log] == [2, 2, 2, 2, 1, 1]
    assert report.frames_masked == 1
    assert report.masked_fraction["frame_0003.jpg"] > 0.5
    assert report.dropped == ["frame_0003.jpg"]
    assert report.masked_fraction["frame_0002.jpg"] == 0.0
    assert all((tmp_path / "m" / f"{f.name}.png").exists() for f in frames)


def test_the_clip_path_falls_back_per_image_when_frames_are_not_indexed(tmp_path):
    frames = _frames(tmp_path, 3)
    png = _mask_png(0, 0, 40, 40)
    script = {frames[1].read_bytes(): {"person": [Detection("person", (0, 0, 0.2, 0.33), mask_png=png)]}}
    client = ScriptedClient(script, video="noframe")
    report = masks.mask_video_frames(frames, tmp_path / "m", client=client, prompts=("person",), chunk=60)
    assert client.log[0][0] == "clip"
    assert client.log.count(("image", "person")) == 3
    assert report.frames_masked == 1 and report.masked_fraction["frame_0001.jpg"] > 0


def test_the_clip_path_falls_back_per_image_when_the_clip_call_fails(tmp_path):
    frames = _frames(tmp_path, 2)
    client = ScriptedClient({}, video=None)
    said: list[str] = []
    report = masks.mask_video_frames(frames, tmp_path / "m", client=client, prompts=("person",), progress=said.append)
    assert client.log.count(("image", "person")) == 2
    assert report.failed == []
    assert any("frame by frame" in s for s in said)


# ---------------------------------------------------------------------------
# wiring into the solve and the cache
# ---------------------------------------------------------------------------


def test_feature_extractor_gets_the_mask_path_only_when_given(tmp_path):
    cfg = colmap.feature_config("sift")
    plain = colmap.feature_extractor_command("colmap", tmp_path / "db", tmp_path / "frames", cfg)
    masked = colmap.feature_extractor_command("colmap", tmp_path / "db", tmp_path / "frames", cfg,
                                              mask_dir=tmp_path / "masks")
    assert "--ImageReader.mask_path" not in plain
    i = masked.index("--ImageReader.mask_path")
    assert masked[i + 1] == str(tmp_path / "masks")
    assert plain[:2] == ["colmap", "feature_extractor"]
    assert "--ImageReader.single_camera" in plain
    for a in colmap.extraction_args(cfg):
        assert a in masked
    # the stamp fingerprints the mask set, None without one
    assert colmap._masks_fingerprint(None) is None
    (tmp_path / "masks").mkdir()
    (tmp_path / "masks" / "frame_0000.jpg.png").write_bytes(b"x" * 10)
    assert colmap._masks_fingerprint(tmp_path / "masks") == [["frame_0000.jpg.png", 10]]


def test_cache_key_differs_with_and_without_movers(tmp_path):
    src = tmp_path / "sweep.mov"
    src.write_bytes(b"\x00" * 100)
    cfg = {"features": "sift", "matcher": "bruteforce"}
    off = recon._cache_key(src, 8.0, 1000, None, None, None, cfg)
    off2 = recon._cache_key(src, 8.0, 1000, None, None, None, cfg, movers="off")
    on = recon._cache_key(src, 8.0, 1000, None, None, None, cfg, movers="on")
    assert off == off2
    assert on != off
    assert on["movers"] == "on" and on["mask_fingerprint"] is None
    json.dumps(on)

    # a lookup before the masks exist matches a stored masked key, never an unmasked one
    stored = dict(on, mask_fingerprint="abc123")
    assert recon._keys_match(stored, on)
    assert not recon._keys_match(off, on)
    assert not recon._keys_match(stored, off)
    assert recon._keys_match(stored, dict(stored))
    assert not recon._keys_match(stored, dict(on, mask_fingerprint="other"))


def test_movers_option_resolves_against_the_environment(monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("LOCAISH_MOVERS", raising=False)
    assert recon._resolve_movers("auto", None) == ("off", None)
    assert recon._resolve_movers("off", None) == ("off", None)
    with pytest.raises(recon.VideoError, match="META_API_KEY"):
        recon._resolve_movers("on", None)
    fake = FakeMetaClient()
    assert recon._resolve_movers("on", fake) == ("on", fake)
    assert recon._resolve_movers("auto", fake) == ("on", fake)
    monkeypatch.setenv("LOCAISH_MOVERS", "off")
    assert recon._resolve_movers("auto", fake) == ("off", None)
    with pytest.raises(ValueError):
        recon._resolve_movers("maybe", None)


def test_dropped_frames_leave_the_set_and_are_kept_aside(tmp_path):
    paths = _frames(tmp_path, 3)
    info = VideoInfo(tmp_path / "v.mov", 1.0, 30.0, W, H, 0.0, "h264", 30)
    fs = FrameSet(paths=paths, timestamps=[0.0, 0.5, 1.0], sharpness=[1.0, 2.0, 3.0],
                  info=info, candidates_considered=9, warnings=["w"])
    out = recon._drop_frames(fs, {"frame_0001.jpg"}, tmp_path / "aside")
    assert [p.name for p in out.paths] == ["frame_0000.jpg", "frame_0002.jpg"]
    assert out.timestamps == [0.0, 1.0] and out.sharpness == [1.0, 3.0]
    assert not paths[1].exists() and (tmp_path / "aside" / "frame_0001.jpg").exists()
    assert out.warnings == ["w"] and out.info is info
    assert recon._drop_frames(fs, set(), tmp_path / "aside") is fs
