"""The Meta Model API client: wire format out, tolerant parsing in, no network."""

from __future__ import annotations

import base64
import json

import httpx
import numpy as np
import pytest

from locaish.meta import client as mc


def _transport(handler):
    return httpx.MockTransport(handler)


def test_chat_sends_openai_shape_and_parses_tool_calls():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "model": "muse-wonder-1.0",
            "choices": [{"finish_reason": "tool_calls", "message": {
                "content": None,
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": "measure", "arguments": "{\"x1\": 1.5}"}}],
            }}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        })

    c = mc.MetaClient("k", base_url="https://api.meta.ai/v1", transport=_transport(handler))
    res = c.chat([{"role": "user", "content": "hi"}], tools=[mc.tool_schema("measure", "d", {"type": "object"})])
    assert seen["path"].endswith("/chat/completions")
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["tool_choice"] == "auto"
    assert seen["body"]["model"] == mc.DEFAULT_MODELS["agent"]
    assert res.tool_calls[0].name == "measure" and res.tool_calls[0].arguments == {"x1": 1.5}
    assert res.usage["prompt_tokens"] == 10 and res.text == ""


def test_chat_json_puts_images_before_text_and_parses_fenced_json(monkeypatch):
    monkeypatch.setenv("LOCAISH_META_VISION_MODEL", "muse-spark-9")
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "```json\n{\"a\": 1}\n```"}}]})

    c = mc.MetaClient("k", transport=_transport(handler))
    out = c.chat_json("what is this", images=[mc._blank_png()])
    assert out == {"a": 1}
    content = seen["body"]["messages"][0]["content"]
    assert content[0]["type"] == "image_url" and content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[-1]["type"] == "text"
    assert seen["body"]["model"] == "muse-spark-9"
    assert seen["body"]["response_format"] == {"type": "json_object"}


def test_errors_carry_status_and_body_and_retry_on_429():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down", headers={"retry-after": "0"})
        return httpx.Response(400, text="bad request: tool_choice")

    c = mc.MetaClient("k", transport=_transport(handler))
    with pytest.raises(mc.MetaError) as exc:
        c.chat([{"role": "user", "content": "x"}])
    assert exc.value.status == 400 and "tool_choice" in exc.value.body
    assert calls["n"] == 2


def test_generate_image_decodes_b64_and_uses_edit_path_with_references():
    seen = {}
    png = mc._blank_png(4, 4)

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png).decode()}]})

    c = mc.MetaClient("k", transport=_transport(handler))
    out = c.generate_image("a room", references=[png, png])
    assert out == png
    assert seen["path"].endswith("/images/edits")
    assert len(seen["body"]["images"]) == 2 and seen["body"]["model"] == mc.DEFAULT_MODELS["image"]
    assert seen["body"]["images"][0]["image_url"].startswith("data:image/png;base64,")
    c.generate_image("a room")
    assert seen["path"].endswith("/images/generations")


def test_segment_parses_boxes_wherever_they_sit():
    png = mc._blank_png(4, 4)
    answer = {"output": [{"content": [{"type": "sam_result", "width": 200, "height": 100, "detections": [
        {"label": "person", "box": [20, 10, 120, 90], "score": 0.9,
         "mask": base64.b64encode(png).decode(), "frame": 3, "track_id": 7},
        {"bbox": {"x": 0.5, "y": 0.5, "w": 0.25, "h": 0.25}},
        {"box": [0, 0, 0, 0]},
    ]}]}]}

    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "sam-3.1"
        assert body["input"][0]["content"][0] == {"type": "input_text", "text": "person"}
        return httpx.Response(200, json=answer)

    c = mc.MetaClient("k", transport=_transport(handler))
    dets = c.segment("person", png)
    assert len(dets) == 2
    d = dets[0]
    assert d.label == "person" and d.mask_png == png and d.frame == 3 and d.track_id == 7
    assert np.allclose(d.box, (0.1, 0.1, 0.6, 0.9))
    assert dets[1].label == "person" and np.allclose(dets[1].box, (0.5, 0.5, 0.75, 0.75))


def test_fake_client_scripts_answers_and_records_calls():
    fake = mc.FakeMetaClient(chat_answers=[{"objects": []}, "plain text"],
                             detections=[mc.Detection("sofa", (0.1, 0.1, 0.5, 0.5))])
    assert fake.chat_json("look", images=[b"\x89PNG\r\n\x1a\n"]) == {"objects": []}
    assert fake.chat([{"role": "user", "content": "x"}]).text == "plain text"
    assert fake.segment("sofa", b"x")[0].label == "sofa"
    assert fake.generate_image("x")[:8] == b"\x89PNG\r\n\x1a\n"
    assert fake.calls == 4 and [e["kind"] for e in fake.log] == ["chat_json", "chat", "segment", "image"]


def test_configured_and_model_for(monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    assert not mc.configured()
    monkeypatch.setenv("MODEL_API_KEY", "k")
    assert mc.configured()
    assert mc.model_for("segment") == "sam-3.1"
    with pytest.raises(KeyError):
        mc.model_for("nope")
    with pytest.raises(mc.MetaError):
        monkeypatch.delenv("MODEL_API_KEY")
        mc.MetaClient()


def test_sam_mask_payload_decodes_through_metas_parser():
    """The protocol's own 5x5 example (meta-models/meta-sam, protocol/sam3.md)
    decodes to a raster and lands in the frame at the box; needs the
    meta-sam-parser package, otherwise the box stands in."""
    import io

    import numpy as np
    import pytest
    from PIL import Image

    from locaish.meta.client import _sam_mask_png, parse_sam_text, sam_masks_decodable

    if not sam_masks_decodable():
        pytest.skip("meta-sam-parser is not installed")
    png = _sam_mask_png("5,5,!!!!!(QO(0lu8?", 10, 20, 14, 24, 100, 80)
    m = np.asarray(Image.open(io.BytesIO(png)))
    assert m.shape == (80, 100) and m.dtype == np.uint8
    inside = m[20:25, 10:15]
    assert inside.max() == 255 and m.sum() == inside.sum(), "the mask lives inside its box and nowhere else"
    text = "<7f>0<|box;x1=10;y1=20;x2=14;y2=24;w=100;h=80|><|mask;x=0;y=0;data=5,5,!!!!!(QO(0lu8?|>"
    dets = parse_sam_text(text, default_label="cup")
    assert len(dets) == 1 and dets[0].mask_png is not None, "a decodable mask rides with its box"
