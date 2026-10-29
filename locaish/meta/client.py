"""The Meta Model API, over plain HTTP.

One client, four verbs: `chat` (Muse Spark, Muse Wonder: text and images in,
text or tool calls out, OpenAI Chat Completions protocol), `chat_json` (the
same with a JSON object demanded and parsed), `generate_image` (Muse Image:
a prompt plus reference images in, PNG bytes out) and `segment` (SAM 3.1: a
noun phrase plus an image in, boxes and masks out, through the Responses
endpoint). The API speaks the OpenAI wire format at `https://api.meta.ai/v1`
and `tool_choice` accepts only `auto`, so that is all this client ever sends.

Nothing here retries forever or hides a failure: a non-2xx answer is a
`MetaError` with the body, and the caller decides whether that is fatal (a
scout agent with no model) or best-effort (one frame of thirty failed to
label). The tests never touch the network; they take a `FakeMetaClient`.

Configuration: `META_API_KEY` (or `MODEL_API_KEY`, the name Meta's own
recipes use). `LOCAISH_META_BASE_URL` overrides the host. Model ids come
from `DEFAULT_MODELS` and are overridable per job with
`LOCAISH_META_<JOB>_MODEL`, so a cheaper or newer id is one variable away.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

DEFAULT_BASE_URL = "https://api.meta.ai/v1"

# Which model does which job. Keys are jobs, not products, so a swap to a
# newer id touches one line here or one env var per deployment.
DEFAULT_MODELS: dict[str, str] = {
    "vision": "muse-spark-1.3",      # name what a frame shows, score a frame
    "agent": "muse-wonder-1.0",      # the scout, the planner, the rehearsal
    "breakdown": "muse-spark-1.3",   # scene to shots: one structured call
    "image": "muse-image-1.0",       # finish a frame photoreal
    "segment": "sam-3.1",            # boxes and masks from a noun phrase
}

_TIMEOUT_S = float(os.environ.get("LOCAISH_META_TIMEOUT_S", "120"))

# Muse Spark and Muse Wonder reason before they answer, and the reasoning
# is billed as output and counted against max_tokens: a 20-token budget
# comes back empty with finish_reason "length". So a caller's max_tokens is
# the budget for the *answer*, and this allowance is added for the thinking;
# the effort knob (minimal, low, medium, high, xhigh, max) keeps the
# thinking short where the job is reading a frame or a form.
REASONING_ALLOWANCE = int(os.environ.get("LOCAISH_META_REASONING_ALLOWANCE", "2048"))
DEFAULT_REASONING = {
    "vision": os.environ.get("LOCAISH_META_VISION_EFFORT", "low"),
    "breakdown": os.environ.get("LOCAISH_META_BREAKDOWN_EFFORT", "low"),
    "agent": os.environ.get("LOCAISH_META_AGENT_EFFORT", "medium"),
}
_RETRIES = 2
_RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


class MetaError(RuntimeError):
    """The API answered with an error, or did not answer."""

    def __init__(self, message: str, *, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


def api_key() -> str | None:
    return os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY") or None


def configured() -> bool:
    """Whether the Meta Model API can be called from this process."""
    return bool(api_key())


def model_for(job: str) -> str:
    """The model id for a job, honouring `LOCAISH_META_<JOB>_MODEL`."""
    if job not in DEFAULT_MODELS:
        raise KeyError(f"unknown model job {job!r}; one of {sorted(DEFAULT_MODELS)}")
    return os.environ.get(f"LOCAISH_META_{job.upper()}_MODEL") or DEFAULT_MODELS[job]


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = ""


@dataclass
class Detection:
    """One thing SAM found: a label, a box in normalised xyxy (0-1), a score,
    and the mask as PNG bytes when the API returned one."""

    label: str
    box: tuple[float, float, float, float]
    score: float = 1.0
    mask_png: bytes | None = None
    frame: int | None = None       # video: which frame this belongs to
    track_id: int | None = None    # video: which tracked object


# ---------------------------------------------------------------------------
# message helpers
# ---------------------------------------------------------------------------


def data_url(image: bytes, mime: str | None = None) -> str:
    mime = mime or _sniff(image)
    return f"data:{mime};base64," + base64.b64encode(image).decode("ascii")


def _sniff(b: bytes) -> str:
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if b[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def user_message(text: str, images: list[bytes] | None = None) -> dict[str, Any]:
    """A chat message with the text last, after the pictures it refers to."""
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": data_url(im)}} for im in (images or [])
    ]
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def tool_schema(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


def _default_effort(model: str) -> str | None:
    """The reasoning effort for a model id, by the job it is registered for."""
    for job, mid in DEFAULT_MODELS.items():
        if model_for(job) == model or mid == model:
            return DEFAULT_REASONING.get(job) or None
    return DEFAULT_REASONING.get("vision") if "spark" in model else DEFAULT_REASONING.get("agent")


def parse_json_text(text: str) -> Any:
    """JSON out of a model answer that may be fenced or prefixed with prose."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
        t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # The first balanced object or array in the text.
    for open_, close in (("{", "}"), ("[", "]")):
        i = t.find(open_)
        j = t.rfind(close)
        if i != -1 and j > i:
            try:
                return json.loads(t[i : j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("the model's answer held no JSON")


# ---------------------------------------------------------------------------
# the client
# ---------------------------------------------------------------------------


class MetaClient:
    """Synchronous client over httpx. Thread-safe: one connection pool, no state."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout_s: float = _TIMEOUT_S,
        transport: Any = None,
    ) -> None:
        import httpx

        key = api_key or globals()["api_key"]()
        if not key:
            raise MetaError("META_API_KEY is not set")
        self.base_url = (base_url or os.environ.get("LOCAISH_META_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=timeout_s,
            transport=transport,
        )
        self.calls: int = 0

    # -- transport --------------------------------------------------------

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        import httpx

        last: Exception | None = None
        for attempt in range(_RETRIES + 1):
            try:
                self.calls += 1
                r = self._http.post(path, json=body)
            except httpx.HTTPError as exc:
                last = exc
                if attempt < _RETRIES:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise MetaError(f"{path}: {type(exc).__name__}: {exc}") from exc
            if r.status_code in _RETRY_STATUSES and attempt < _RETRIES:
                retry_after = r.headers.get("retry-after")
                try:
                    wait = float(retry_after) if retry_after else 1.5 * (attempt + 1)
                except ValueError:
                    wait = 1.5 * (attempt + 1)
                time.sleep(min(wait, 20.0))
                continue
            if r.status_code >= 400:
                raise MetaError(
                    f"{path} returned {r.status_code}: {r.text[:400]}",
                    status=r.status_code, body=r.text,
                )
            try:
                return r.json()
            except ValueError as exc:
                raise MetaError(f"{path}: the answer was not JSON: {r.text[:200]}") from exc
        raise MetaError(f"{path}: gave up after {_RETRIES + 1} attempts: {last}")

    # -- chat ---------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> ChatResult:
        body: dict[str, Any] = {
            "model": model or model_for("agent"),
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"      # the only value the API accepts
        if response_format:
            body["response_format"] = response_format
        if max_tokens:
            body["max_tokens"] = int(max_tokens) + REASONING_ALLOWANCE
        effort = reasoning_effort or _default_effort(body["model"])
        if effort:
            body["reasoning_effort"] = effort
        data = self._post("/chat/completions", body)
        return _parse_chat(data)

    def chat_json(
        self,
        text: str,
        images: list[bytes] | None = None,
        *,
        model: str | None = None,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """One question, one JSON answer. Images are shown before the text."""
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(user_message(text, images))
        fmt: dict[str, Any] = {"type": "json_object"}
        if schema:
            fmt = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
        try:
            res = self.chat(messages, model=model or model_for("vision"), response_format=fmt,
                            max_tokens=max_tokens)
        except MetaError as exc:
            # Some ids reject json_schema; json_object is universally honoured.
            if schema and exc.status == 400:
                res = self.chat(messages, model=model or model_for("vision"),
                                response_format={"type": "json_object"}, max_tokens=max_tokens)
            else:
                raise
        return parse_json_text(res.text)

    # -- images -------------------------------------------------------------

    def generate_image(
        self,
        prompt: str,
        *,
        references: list[bytes] | None = None,
        model: str | None = None,
        size: str | None = None,
    ) -> bytes:
        """Muse Image. With references it is an instruction edit: the first
        reference is the frame to change, the rest are what to match. The
        answer is PNG (or whatever the API encoded) bytes.

        The endpoint shape follows the OpenAI images protocol the API mirrors
        (`/images/generations`, `/images/edits`); `LOCAISH_META_IMAGE_PATH`
        overrides the path if Meta's docs name a different one.
        """
        body: dict[str, Any] = {"model": model or model_for("image"), "prompt": prompt, "n": 1,
                                "response_format": "b64_json"}
        if size:
            body["size"] = size
        if references:
            # The JSON form of the edit endpoint takes `images`, each an
            # image_url item; `image` is the multipart form's field.
            body["images"] = [{"image_url": data_url(r)} for r in references]
            path = os.environ.get("LOCAISH_META_IMAGE_EDIT_PATH", "/images/edits")
        else:
            path = os.environ.get("LOCAISH_META_IMAGE_PATH", "/images/generations")
        data = self._post(path, body)
        return _image_bytes(data)

    # -- segmentation -------------------------------------------------------

    def segment(
        self,
        prompt: str,
        image: bytes | None = None,
        *,
        frames: list[bytes] | None = None,
        model: str | None = None,
    ) -> list[Detection]:
        """SAM 3.1: every match for a short noun phrase, as boxes and masks.

        One image, or a list of frames for tracking through a clip. The
        Responses endpoint is the documented route; the parser is tolerant of
        where in the answer the detections sit, because that is the part of
        the wire format most likely to move.
        """
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        if image is not None:
            content.append({"type": "input_image", "image_url": data_url(image)})
        for i, f in enumerate(frames or []):
            content.append({"type": "input_image", "image_url": data_url(f), "frame": i})
        body = {"model": model or model_for("segment"), "input": [{"role": "user", "content": content}]}
        data = self._post("/responses", body)
        text = _responses_text(data)
        if text and "<|box" in text:
            return parse_sam_text(text, default_label=prompt)
        return parse_detections(data, default_label=prompt)


# ---------------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------------


def _parse_chat(data: dict[str, Any]) -> ChatResult:
    choices = data.get("choices") or []
    if not choices:
        raise MetaError(f"chat answer had no choices: {json.dumps(data)[:300]}")
    msg = choices[0].get("message") or {}
    text = msg.get("content")
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    calls: list[ToolCall] = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments") or "{}"
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        calls.append(ToolCall(id=str(tc.get("id") or ""), name=str(fn.get("name") or ""), arguments=args))
    usage = data.get("usage") or {}
    return ChatResult(
        text=text or "",
        tool_calls=calls,
        model=str(data.get("model") or ""),
        usage={k: int(v) for k, v in usage.items() if isinstance(v, (int, float))},
        raw=data,
        finish_reason=str(choices[0].get("finish_reason") or ""),
    )


def _image_bytes(data: dict[str, Any]) -> bytes:
    items = data.get("data") or data.get("images") or []
    if isinstance(items, dict):
        items = [items]
    for it in items:
        if not isinstance(it, dict):
            continue
        b64 = it.get("b64_json") or it.get("b64") or it.get("image_base64")
        if b64:
            return base64.b64decode(b64)
        url = it.get("url") or it.get("image_url")
        if url and str(url).startswith("data:"):
            return base64.b64decode(str(url).split(",", 1)[1])
        if url:
            import httpx

            r = httpx.get(str(url), timeout=60.0)
            r.raise_for_status()
            return r.content
    raise MetaError(f"image answer held no image: {json.dumps(data)[:300]}")


def _responses_text(data: Any) -> str:
    """The concatenated output_text of a Responses-protocol answer."""
    parts: list[str] = []
    for item in (data.get("output") or []) if isinstance(data, dict) else []:
        for c in (item.get("content") or []) if isinstance(item, dict) else []:
            if isinstance(c, dict) and c.get("type") in ("output_text", "text") and c.get("text"):
                parts.append(str(c["text"]))
    return "".join(parts)


_BOX_RE = re.compile(r"<\|box;([^|]*)\|>")
_MASK_RE = re.compile(r"<\|mask;([^|]*)\|>")
_FIELD_RE = re.compile(r"([a-z0-9]+)=([^;]*)")


def parse_sam_text(text: str, *, default_label: str = "") -> list[Detection]:
    """SAM 3.1 answers in a token grammar, not JSON: for each match a
    `<|box;x1=..;y1=..;x2=..;y2=..;w=W;h=H|>` in pixels of a WxH frame,
    followed by `<|mask;x=..;y=..;data=h,w,<rle>|>`, the mask of the box's
    crop in COCO's compressed run-length string, placed at (x, y) inside the
    box. Boxes come back normalised and the mask as a full-frame PNG, white
    where the thing is, so the rest of the product reads them as any other
    detection."""
    out: list[Detection] = []
    boxes = [(m.start(), m.group(1)) for m in _BOX_RE.finditer(text)]
    masks = [(m.start(), m.group(1)) for m in _MASK_RE.finditer(text)]
    for i, (pos, fields) in enumerate(boxes):
        f = dict(_FIELD_RE.findall(fields))
        try:
            x1, y1, x2, y2 = (float(f[k]) for k in ("x1", "y1", "x2", "y2"))
            W, H = float(f["w"]), float(f["h"])
        except (KeyError, ValueError):
            continue
        if W <= 0 or H <= 0:
            continue
        box = (max(0.0, x1 / W), max(0.0, y1 / H), min(1.0, x2 / W), min(1.0, y2 / H))
        # The mask that follows this box and precedes the next one.
        nxt = boxes[i + 1][0] if i + 1 < len(boxes) else len(text) + 1
        mfields = next((mf for mp, mf in masks if pos < mp < nxt), None)
        mask_png = None
        if mfields is not None:
            mf = dict(_FIELD_RE.findall(mfields.split("data=", 1)[0])) if "data=" in mfields else {}
            data = mfields.split("data=", 1)[1] if "data=" in mfields else ""
            try:
                mask_png = _sam_mask_png(data, int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)),
                                         int(W), int(H), int(float(mf.get("x", 0))), int(float(mf.get("y", 0))))
            except Exception:  # noqa: BLE001 - a mask that will not decode leaves the box
                mask_png = None
        out.append(Detection(label=default_label, box=box, score=1.0, mask_png=mask_png))
    return out


def _coco_rle_counts(s: str) -> list[int]:
    """pycocotools' string encoding of run lengths: 5-bit groups, LEB-style,
    with every third count stored as a delta from the one two back."""
    counts: list[int] = []
    p, m = 0, 0
    n = len(s)
    while p < n:
        x, k, more = 0, 0, 1
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if m > 2:
            x += counts[m - 2]
        counts.append(x)
        m += 1
    return counts


def _sam_mask_png(data: str, x1: int, y1: int, x2: int, y2: int, W: int, H: int, ox: int = 0, oy: int = 0) -> bytes:
    import numpy as np
    from PIL import Image

    head, rle = data.split(",", 2)[:2], data.split(",", 2)[2]
    h, w = int(head[0]), int(head[1])
    counts = _coco_rle_counts(rle)
    flat = np.zeros(h * w, dtype=np.uint8)
    pos, val = 0, 0
    for c in counts:
        c = max(0, int(c))
        if val:
            flat[pos : pos + c] = 1
        pos += c
        val ^= 1
        if pos >= h * w:
            break
    crop = flat.reshape((w, h)).T          # COCO lays the mask out column-major
    full = np.zeros((H, W), dtype=np.uint8)
    X0, Y0 = x1 + ox, y1 + oy
    hh, ww = min(h, H - Y0), min(w, W - X0)
    if hh > 0 and ww > 0:
        full[Y0 : Y0 + hh, X0 : X0 + ww] = crop[:hh, :ww]
    buf = io.BytesIO()
    Image.fromarray(full * 255, mode="L").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def parse_detections(data: Any, *, default_label: str = "") -> list[Detection]:
    """Find the detections wherever the answer put them.

    Accepts the shapes seen in SAM-style APIs: a top-level `detections` or
    `objects` list, or the same list nested under `output[*].content[*]`.
    Each item needs a box (`box`, `bbox`, `box_2d`, xyxy or xywh, normalised
    or in pixels given `width`/`height`) and may carry a `mask` (base64 PNG),
    `label`, `score`, `frame` and `track_id`/`object_id`.
    """
    found: list[dict[str, Any]] = []
    _walk(data, found)
    out: list[Detection] = []
    W = _num(_dig(data, "width")) or None
    H = _num(_dig(data, "height")) or None
    for d in found:
        box = _box(d, W, H)
        if box is None:
            continue
        mask = d.get("mask") or d.get("mask_png") or d.get("mask_base64")
        if isinstance(mask, dict):
            mask = mask.get("png") or mask.get("data") or mask.get("b64")
        mask_bytes = None
        if isinstance(mask, str) and mask:
            try:
                mask_bytes = base64.b64decode(mask.split(",", 1)[-1])
            except Exception:  # noqa: BLE001
                mask_bytes = None
        elif isinstance(mask, (bytes, bytearray)):
            mask_bytes = bytes(mask)
        out.append(Detection(
            label=str(d.get("label") or d.get("text") or d.get("prompt") or default_label),
            box=box,
            score=float(d.get("score") or d.get("confidence") or 1.0),
            mask_png=mask_bytes,
            frame=_int(d.get("frame") if "frame" in d else d.get("frame_index")),
            track_id=_int(d.get("track_id") if "track_id" in d else d.get("object_id")),
        ))
    return out


def _walk(obj: Any, out: list[dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        if any(k in obj for k in ("box", "bbox", "box_2d", "bounding_box")) and not out.__contains__(obj):
            out.append(obj)
            return
        for k in ("detections", "objects", "results", "masks", "output", "content", "data", "items", "annotations"):
            if k in obj:
                _walk(obj[k], out)
        for k, v in obj.items():
            if k not in ("detections", "objects", "results", "masks", "output", "content", "data", "items", "annotations") \
                    and isinstance(v, (dict, list)):
                _walk(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, out)


def _dig(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _dig(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _dig(v, key)
            if r is not None:
                return r
    return None


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(v: Any) -> int | None:
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def _box(d: dict[str, Any], W: float | None, H: float | None) -> tuple[float, float, float, float] | None:
    raw = d.get("box") or d.get("bbox") or d.get("box_2d") or d.get("bounding_box")
    if isinstance(raw, dict):
        if all(k in raw for k in ("x", "y", "w", "h")):
            vals = [raw["x"], raw["y"], raw["x"] + raw["w"], raw["y"] + raw["h"]]
        elif all(k in raw for k in ("x1", "y1", "x2", "y2")):
            vals = [raw["x1"], raw["y1"], raw["x2"], raw["y2"]]
        elif all(k in raw for k in ("xmin", "ymin", "xmax", "ymax")):
            vals = [raw["xmin"], raw["ymin"], raw["xmax"], raw["ymax"]]
        else:
            return None
    elif isinstance(raw, (list, tuple)) and len(raw) == 4:
        vals = list(raw)
        fmt = str(d.get("box_format") or d.get("format") or "").lower()
        if fmt in ("xywh", "x,y,w,h"):
            vals = [vals[0], vals[1], vals[0] + vals[2], vals[1] + vals[3]]
    else:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in vals)
    except (TypeError, ValueError):
        return None
    if max(x1, y1, x2, y2) > 1.5:
        # Pixels, or the Gemini-style 0-1000 convention.
        if W and H:
            x1, x2, y1, y2 = x1 / W, x2 / W, y1 / H, y2 / H
        else:
            x1, y1, x2, y2 = x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0
    x1, x2 = sorted((min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)))
    y1, y2 = sorted((min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)))
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# the fake, for tests
# ---------------------------------------------------------------------------


def _blank_png(w: int = 8, h: int = 8, colour=(120, 120, 120)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format="PNG")
    return buf.getvalue()


class FakeMetaClient:
    """Stands in for `MetaClient`: records every call, answers from a script.

    `chat_answers` is a list consumed in order (each a `ChatResult`, a plain
    string, or a dict that becomes a JSON text answer) or a callable taking
    the messages. `detections` is what `segment` returns (a list, or a
    callable of the prompt). `image` is what `generate_image` returns.
    """

    def __init__(
        self,
        *,
        chat_answers: list[Any] | Callable[..., Any] | None = None,
        detections: list[Detection] | Callable[[str], list[Detection]] | None = None,
        image: bytes | None = None,
    ) -> None:
        self.base_url = "fake://meta"
        self._answers = chat_answers
        self._detections = detections if detections is not None else []
        self._image = image or _blank_png()
        self.calls = 0
        self.log: list[dict[str, Any]] = []

    def _next_answer(self, messages) -> ChatResult:
        if callable(self._answers):
            a = self._answers(messages)
        elif self._answers:
            a = self._answers.pop(0)
        else:
            a = "{}"
        if isinstance(a, ChatResult):
            return a
        if isinstance(a, (dict, list)):
            return ChatResult(text=json.dumps(a), model="fake")
        return ChatResult(text=str(a), model="fake")

    def chat(self, messages, *, model=None, tools=None, response_format=None, temperature=0.0,
             max_tokens=None, reasoning_effort=None) -> ChatResult:
        self.calls += 1
        self.log.append({"kind": "chat", "model": model, "messages": messages, "tools": tools,
                         "response_format": response_format})
        return self._next_answer(messages)

    def chat_json(self, text, images=None, *, model=None, system=None, schema=None, max_tokens=None):
        self.calls += 1
        self.log.append({"kind": "chat_json", "model": model, "text": text,
                         "images": len(images or []), "schema": schema})
        return parse_json_text(self._next_answer([user_message(text, images)]).text)

    def generate_image(self, prompt, *, references=None, model=None, size=None) -> bytes:
        self.calls += 1
        self.log.append({"kind": "image", "model": model, "prompt": prompt,
                         "references": len(references or [])})
        return self._image

    def segment(self, prompt, image=None, *, frames=None, model=None) -> list[Detection]:
        self.calls += 1
        self.log.append({"kind": "segment", "model": model, "prompt": prompt,
                         "image": image is not None, "frames": len(frames or [])})
        if callable(self._detections):
            return list(self._detections(prompt))
        return list(self._detections)
