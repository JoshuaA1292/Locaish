"""The vision provider: Muse Spark reads frames through the same prompts
Gemini did, the switch picks one from the environment, and every label the
model leaves carries its provider. No network: everything takes a
`FakeMetaClient`."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from locaish.meta import FakeMetaClient, model_for
from locaish.meta import vision as V
from locaish.product import labels as L
from locaish.scan import semantic as S
from test_labels import _frames

_KEYS = ("META_API_KEY", "MODEL_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
         "GOOGLE_GENAI_USE_VERTEXAI", "LOCAISH_VISION_PROVIDER", "LOCAISH_META_VISION_MODEL")


@pytest.fixture
def no_keys(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _jpeg(colour=(90, 90, 90)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (16, 12), colour).save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------


def test_provider_auto_prefers_meta_then_gemini_then_none(no_keys):
    assert V.provider() == "none" and not V.available()
    no_keys.setenv("GOOGLE_API_KEY", "g")
    assert V.provider() == "gemini" and V.available()
    no_keys.setenv("META_API_KEY", "m")
    assert V.provider() == "meta" and V.available()
    assert V.display_name() == "Muse Spark"
    # Meta's own recipes call the key MODEL_API_KEY; that spelling counts too.
    no_keys.delenv("META_API_KEY")
    no_keys.setenv("MODEL_API_KEY", "m")
    assert V.provider() == "meta"


def test_provider_env_forces_a_choice_even_when_unconfigured(no_keys):
    no_keys.setenv("META_API_KEY", "m")
    no_keys.setenv("GOOGLE_API_KEY", "g")
    no_keys.setenv("LOCAISH_VISION_PROVIDER", "gemini")
    assert V.provider() == "gemini" and V.available()
    no_keys.setenv("LOCAISH_VISION_PROVIDER", "none")
    assert V.provider() == "none" and not V.available()
    # Asked for Meta without a key: say so, do not quietly use Gemini.
    no_keys.delenv("META_API_KEY")
    no_keys.setenv("LOCAISH_VISION_PROVIDER", "META")
    assert V.provider() == "meta" and not V.available()
    # Garbage is auto.
    no_keys.setenv("LOCAISH_VISION_PROVIDER", "claude")
    assert V.provider() == "gemini"


def test_labels_and_semantic_answer_through_the_switch(no_keys):
    assert not L.vision_available() and not S.available()
    assert L.gemini_available is L.vision_available
    no_keys.setenv("META_API_KEY", "m")
    assert L.vision_available() and S.available()


# ---------------------------------------------------------------------------
# objects in a frame
# ---------------------------------------------------------------------------


def test_detect_objects_normalises_boxes_and_shapes():
    fake = FakeMetaClient(chat_answers=[
        # 0-1000 as asked, with a kind that needs lower-casing
        [{"label": "Kitchen  Island", "kind": "Furniture", "box_2d": [450, 100, 650, 400]}],
        # unit coordinates, under a different key
        [{"label": "sofa", "kind": "furniture", "box": [0.1, 0.2, 0.3, 0.6]}],
        # wrapped in an object, with one entry to drop for having no label
        {"objects": [{"label": "radiator", "kind": "fixture", "box_2d": [700, 50, 900, 300]},
                     {"label": "", "kind": "other", "box_2d": [0, 0, 10, 10]},
                     {"label": "mystery", "kind": "gadget", "box_2d": [0, 0, 100, 100]}]},
    ])
    a = V.detect_objects(_jpeg(), client=fake)
    assert a == [{"label": "kitchen island", "kind": "furniture", "box": [450.0, 100.0, 650.0, 400.0]}]
    b = V.detect_objects(_jpeg(), client=fake)
    assert b[0]["label"] == "sofa"
    assert np.allclose(b[0]["box"], [100, 200, 300, 600])
    c = V.detect_objects(_jpeg(), client=fake)
    assert [o["label"] for o in c] == ["radiator", "mystery"]
    assert c[1]["kind"] == "other", "an unknown kind falls back rather than failing the frame"
    # The shape is exactly what labels.detect_in_frame hands to lift().
    for o in a + b + c:
        assert set(o) == {"label", "kind", "box"} and len(o["box"]) == 4
    # Same prompt as the Gemini path, one image per call, the vision model.
    assert all(e["text"].startswith(L._PROMPT) and e["images"] == 1 for e in fake.log)
    assert fake.log[0]["model"] == model_for("vision")


def test_detect_objects_honours_model_override_and_ignores_rubbish():
    fake = FakeMetaClient(chat_answers=["not json at all {", [{"label": "lamp", "box_2d": [1, 2, 3]}]])
    with pytest.raises(ValueError):
        V.detect_objects(_jpeg(), client=fake, model="muse-spark-x")
    assert fake.log[0]["model"] == "muse-spark-x"
    assert V.detect_objects(_jpeg(), client=fake) == [], "a three-number box is no box"


def test_detect_in_frame_dispatches_to_meta(tmp_path):
    p = tmp_path / "f.jpg"
    p.write_bytes(_jpeg())
    fake = FakeMetaClient(chat_answers=[[{"label": "door", "kind": "door", "box_2d": [0, 0, 1000, 200]}]])
    out = L.detect_in_frame(p, client=fake, provider="meta")
    assert out == [{"label": "door", "kind": "door", "box": [0.0, 0.0, 1000.0, 200.0]}]
    with pytest.raises(ValueError):
        L.detect_in_frame(p, client=fake, provider="none")


# ---------------------------------------------------------------------------
# the room across frames
# ---------------------------------------------------------------------------


def test_crosscheck_frames_parses_the_observation():
    fake = FakeMetaClient(chat_answers=[{
        "room_type": "kitchen", "window_count": "2", "door_count": 1.0, "ceiling_visible": True,
        "ceiling_height_estimate_m": "2.7", "capture_issues": ["motion blur", 5],
    }])
    obs = V.crosscheck_frames([_jpeg(), _jpeg(), _jpeg()], client=fake)
    assert obs == {
        "model": f"meta:{model_for('vision')}", "frames_used": 3, "room_type": "kitchen",
        "window_count": 2, "door_count": 1, "ceiling_visible": True,
        "ceiling_height_estimate_m": 2.7, "capture_issues": ["motion blur", "5"],
    }
    assert fake.log[0]["text"] == S._PROMPT and fake.log[0]["images"] == 3
    assert S.who(obs) == "Muse Spark"
    assert S.who({"model": "gemini:gemini-3.6-flash"}) == "Gemini"
    assert S.who({"model": "gemini-3.6-flash"}) == "Gemini", "observations recorded before the provider prefix"
    with pytest.raises(ValueError):
        V.crosscheck_frames([_jpeg()], client=FakeMetaClient(chat_answers=[[1, 2]]))
    with pytest.raises(ValueError):
        V.crosscheck_frames([], client=fake)


def test_semantic_crosscheck_reads_a_frames_dir_through_meta(tmp_path):
    for i in range(9):
        (tmp_path / f"frame_{i:03d}.jpg").write_bytes(_jpeg())
    (tmp_path / "notes.txt").write_text("not a frame")
    fake = FakeMetaClient(chat_answers=[{"room_type": "office", "window_count": 1, "door_count": 1,
                                         "ceiling_visible": False, "ceiling_height_estimate_m": None,
                                         "capture_issues": []}])
    obs = S.crosscheck(tmp_path, client=fake, provider="meta")
    assert obs["frames_used"] == S.MAX_FRAMES and obs["room_type"] == "office"
    assert obs["ceiling_height_estimate_m"] is None and obs["model"].startswith("meta:")
    assert S.crosscheck(tmp_path / "missing", client=fake, provider="meta") is None
    with pytest.raises(ValueError):
        S.crosscheck(tmp_path, client=fake, provider="none")


# ---------------------------------------------------------------------------
# the whole semantic pass, on a synthetic wall
# ---------------------------------------------------------------------------


def _wall_frames(tmp_path: Path):
    """Three ring cameras looking at the origin, with real JPEGs on disk and a
    measured wall through the origin that camera 0 faces square on."""
    fr = _frames(n=3)
    for i, name in enumerate(fr.names):
        p = tmp_path / name
        p.write_bytes(_jpeg((40 * i, 80, 120)))
        fr.paths[i] = p
    # The wall: a 4 m x 4 m plane x = t[0] in twin space, 2 cm point spacing.
    ys, zs = np.meshgrid(np.arange(-2, 2, 0.02), np.arange(-2, 2, 0.02))
    wall = np.stack([np.full(ys.size, fr.t[0]), fr.t[1] + ys.ravel(), fr.t[2] + zs.ravel()], axis=1)
    return fr, cKDTree(wall)


def test_detect_end_to_end_labels_carry_meta_as_source(tmp_path, no_keys):
    fr, tree = _wall_frames(tmp_path)
    # Labels already there: a person's, and two stale model passes.
    L.add_label(tmp_path, label="piano", xyz=fr.t + np.array([0.0, 1.5, 0.4]), kind="furniture")
    stale = [{"id": "g1", "label": "fridge", "kind": "appliance", "xyz": list(fr.t + [0, -1.5, 0.9]),
              "source": "gemini", "frame": "f0.jpg", "uv": None, "note": "", "confidence": 1, "created_at": "x"},
             {"id": "m1", "label": "fridge", "kind": "appliance", "xyz": list(fr.t + [0, -1.5, 0.9]),
              "source": "meta", "frame": "f0.jpg", "uv": None, "note": "", "confidence": 1, "created_at": "x"}]
    L.save_labels(tmp_path, L.load_labels(tmp_path) + stale)

    centre = {"label": "sofa", "kind": "furniture", "box_2d": [450, 450, 550, 550]}   # image centre
    fake = FakeMetaClient(chat_answers=[[centre], [centre], [centre, {"label": "lamp", "kind": "lighting",
                                                                       "box_2d": [0.45, 0.45, 0.55, 0.55]}]])
    seen = []
    res = L.detect(tmp_path, None, frames=fr, tree=tree, client=fake, provider="meta", progress=seen.append)

    assert res["provider"] == "meta" and res["model"] == model_for("vision")
    assert res["frames_looked_at"] == 3 and res["detections"] == 4 and res["placed"] == 4
    assert fake.calls == 3 and seen[0] == "looking at frame 1 of 3"
    labels = res["labels"]
    assert [l["label"] for l in labels] == ["piano", "sofa"], "person kept, stale model labels of both providers dropped"
    sofa = labels[1]
    assert sofa["source"] == "meta" and sofa["confidence"] == 3, "three frames agreed on one object"
    assert np.allclose(sofa["xyz"], fr.t, atol=0.06), "the box's centre ray lands on the wall at the origin"
    assert sofa["frame"] == "f0.jpg" and sofa["kind"] == "furniture"
    # The lamp sat on the same spot as the sofa under a different name: merge
    # keeps names apart, but a thing already placed within reach is not re-added.
    assert "lamp" not in [l["label"] for l in labels]
    # Written to disk, and the describe/svg views treat it as provisional.
    on_disk = L.load_labels(tmp_path)
    assert [l["source"] for l in on_disk] == ["user", "meta"]
    d = L.describe(on_disk)
    assert d[0]["status"] == "confirmed" and d[1]["status"].startswith("seen by the model in 3 frames")
    svg = "".join(L.labels_svg(on_disk, lambda x: x, lambda y: y))
    assert "sofa?" in svg and "piano<" in svg


def test_detect_refuses_without_a_configured_provider(tmp_path, no_keys):
    fr, tree = _wall_frames(tmp_path)
    with pytest.raises(ValueError, match="no vision model"):
        L.detect(tmp_path, None, frames=fr, tree=tree)
    with pytest.raises(ValueError, match="no vision model"):
        L.detect(tmp_path, None, frames=fr, tree=tree, provider="meta")   # asked for, no key, no client
    # One failing frame is not a failed pass.
    fake = FakeMetaClient(chat_answers=["garbage", [], []])
    res = L.detect(tmp_path, None, frames=fr, tree=tree, client=fake, provider="meta")
    assert res["objects"] == 0 and res["frames_looked_at"] == 3 and res["provider"] == "meta"
