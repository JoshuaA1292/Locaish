"""The vision provider: which model reads the frames, and the two readings.

Two places in Locaish show a model a photograph and ask it to name or count
what it sees: `product.labels` (what objects are in the room, each with a box
that geometry then lifts into the twin) and `scan.semantic` (what kind of
room, how many windows and doors, roughly how high the ceiling). Neither
measures anything -- the model names and judges, the geometry answers -- so
the model behind them is a choice, not a dependency. This module makes that
choice in one place.

Muse Spark through the Meta Model API is the default. Gemini stays wired in
as the alternative, because the prompts were written against it and a studio
that already holds a Google key should keep working unchanged. `provider()`
is the single switch: `LOCAISH_VISION_PROVIDER` names one outright, and
`auto` takes Meta when a key is present, Gemini when only a Google key is,
and `none` otherwise -- in which case both callers decline quietly and the
pipeline runs exactly as it did before any model was attached.

The Meta readings here reuse the callers' prompts verbatim (imported, not
copied) so the two providers are asked the same question and their answers
are comparable. Every label and observation that leaves here carries its
provider as `source` or in `model`, so a producer reading a package can tell
a model's guess from a person's word and one model's guess from another's.
"""

from __future__ import annotations

import os
from typing import Any

from . import client as _client

#: What `LOCAISH_VISION_PROVIDER` may say.
PROVIDERS = ("auto", "meta", "gemini", "none")

#: The sources a model writes on a label. A person's labels are "user" and
#: are never replaced by a model pass, whichever provider ran it.
MODEL_PROVIDERS = frozenset({"meta", "gemini"})

#: How each provider is named in the studio.
DISPLAY_NAMES = {"meta": "Muse Spark", "gemini": "Gemini", "none": "no vision model"}


def _gemini_configured() -> bool:
    """The check `labels.gemini_available` made: a Google key (or Vertex) and
    the SDK importable."""
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")):
        return False
    try:
        from google import genai  # noqa: F401
    except Exception:  # noqa: BLE001 - no SDK is simply no Gemini
        return False
    return True


def configured_for(name: str) -> bool:
    """Whether one named provider has what it needs to be called."""
    if name == "meta":
        return _client.configured()
    if name == "gemini":
        return _gemini_configured()
    return False


def provider() -> str:
    """Which provider reads the frames: "meta", "gemini" or "none".

    A provider named outright by `LOCAISH_VISION_PROVIDER` is returned as
    asked, configured or not, so a misconfigured studio says "meta" and
    `available()` says False rather than silently falling back to a model
    the operator did not choose. An unrecognised value is treated as auto.
    """
    want = (os.environ.get("LOCAISH_VISION_PROVIDER") or "auto").strip().lower()
    if want in PROVIDERS and want != "auto":
        return want
    if _client.configured():
        return "meta"
    if _gemini_configured():
        return "gemini"
    return "none"


def available() -> bool:
    """Whether the chosen provider can actually be called from this process."""
    return configured_for(provider())


def display_name(name: str | None = None) -> str:
    return DISPLAY_NAMES.get(name or provider(), str(name or ""))


def _meta_client(client: Any = None) -> Any:
    return client if client is not None else _client.MetaClient()


# ---------------------------------------------------------------------------
# objects in one frame
# ---------------------------------------------------------------------------


def _box_1000(raw: Any) -> list[float] | None:
    """A [ymin, xmin, ymax, xmax] box on the 0-1000 grid `labels.lift` reads.

    Muse Spark is asked for 0-1000 like Gemini, but vision models drift
    towards unit coordinates when the prompt and their training disagree, so
    a box whose every edge is within [0, 1] is taken as 0-1 and scaled. The
    corner case -- a genuinely 0-1000 box smaller than a thousandth of the
    frame -- is nothing a film crew would care about anyway.
    """
    if isinstance(raw, dict):
        keys = ("ymin", "xmin", "ymax", "xmax")
        if not all(k in raw for k in keys):
            return None
        raw = [raw[k] for k in keys]
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        vals = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    if all(0.0 <= v <= 1.0 for v in vals):
        vals = [v * 1000.0 for v in vals]
    vals = [min(max(v, 0.0), 1000.0) for v in vals]
    ymin, ymax = sorted((vals[0], vals[2]))
    xmin, xmax = sorted((vals[1], vals[3]))
    if ymax - ymin <= 0 or xmax - xmin <= 0:
        return None
    return [ymin, xmin, ymax, xmax]


def detect_objects(image: bytes, *, client: Any = None, model: str | None = None) -> list[dict]:
    """Muse Spark names what is in one frame.

    Returns what `labels.detect_in_frame` returns: a list of
    `{"label", "kind", "box": [ymin, xmin, ymax, xmax]}` with the box on the
    0-1000 grid, so the caller's ray march does not know which model it is
    downstream of. The answer may be a bare list or wrapped in an object
    (`objects`, `items`, `detections`); both are read.
    """
    from ..product.labels import KINDS, _PROMPT, normalise

    client = _meta_client(client)
    # The API's JSON mode wants an object at the top, and asked for "a list"
    # Muse Spark answers with the first object alone; wrapping the list under
    # one key gives it a shape it keeps to.
    data = client.chat_json(
        _PROMPT + '\nWrap the list as {"objects": [ ... ]} and include every object you can see.',
        [image], model=model or _client.model_for("vision"), max_tokens=3000,
    )
    if isinstance(data, dict) and "box_2d" in data and not any(
            isinstance(data.get(k), list) for k in ("objects", "items", "detections", "labels", "results")):
        data = [data]
    if isinstance(data, dict):
        inner = None
        for key in ("objects", "items", "detections", "labels", "results"):
            if isinstance(data.get(key), list):
                inner = data[key]
                break
        data = inner if inner is not None else ([data] if "label" in data else [])
    out: list[dict] = []
    for d in data if isinstance(data, list) else []:
        if not isinstance(d, dict):
            continue
        box = _box_1000(d.get("box_2d", d.get("box", d.get("bbox"))))
        label = normalise(d.get("label"))
        if box is None or not label:
            continue
        kind = str(d.get("kind") or "other").strip().lower()
        out.append({"label": label, "kind": kind if kind in KINDS else "other", "box": box})
    return out


# ---------------------------------------------------------------------------
# the room, across several frames
# ---------------------------------------------------------------------------


def crosscheck_frames(frames: list[bytes], *, client: Any = None, model: str | None = None) -> dict[str, Any]:
    """Muse Spark reads the room off a handful of frames.

    Returns the observation `scan.semantic.crosscheck` returns, with `model`
    naming the provider as well as the id ("meta:muse-spark-1.3") so a QA
    report says whose eyes it is quoting. The prompt is `scan.semantic`'s
    own and carries nothing from the twin: independence is the whole value
    of the check, and a cross-check that could see the answer it is checking
    would read zero and could not fail.
    """
    from ..scan import semantic

    frames = [f for f in frames if f]
    if not frames:
        raise ValueError("a cross-check needs at least one frame")
    client = _meta_client(client)
    model_id = model or _client.model_for("vision")
    data = client.chat_json(semantic._PROMPT, frames, model=model_id)
    if not isinstance(data, dict):
        raise ValueError(f"Muse Spark returned {type(data).__name__}, not an object")
    return semantic.observation(data, model=f"meta:{model_id}", frames_used=len(frames))


__all__ = [
    "PROVIDERS", "MODEL_PROVIDERS", "DISPLAY_NAMES",
    "provider", "available", "configured_for", "display_name", "detect_objects", "crosscheck_frames",
]
