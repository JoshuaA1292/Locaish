"""A second, independent reading of the room, from the pictures themselves.

Every check in `scan.qa` measures the twin against itself or against an
architectural prior, and all of them share one blind spot: they are downstream
of the same reconstruction. A wall the stereo never saw is invisible to a
residual; a "window" hallucinated by a gap in the returns satisfies every
geometric test, because geometry is all it ever was.

The frames are the one source of truth the reconstruction cannot contaminate.
This module shows a handful of them to Gemini and asks what a person would be
asked: what kind of room is this, how many windows and doors do you actually
see, roughly how high is the ceiling. The answers are compared with what the
geometry claims, and disagreement is reported -- never silently reconciled.

Independence is the entire value, so it is enforced structurally: the prompt
contains no numbers from the twin, no detected openings, no footprint. Gemini
answers from pixels alone, the comparison happens here afterwards, and a
cross-check that could see the answer it is checking would be worthless -- a
residual derived from its own subject reads zero and cannot fail.

The check is advisory. It can warn, it cannot fail a twin: a model reading six
frames is evidence, not a measurement, and the report says which is which.

Compliance note: this pipeline is bound (Agentic Cinema rules) to Google Cloud
AI tools only. The one model called here is Gemini via `google-genai`; if that
SDK or a `GEMINI_API_KEY`/`GOOGLE_API_KEY` is absent the module declines
quietly and the pipeline runs exactly as before.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..types import Twin

#: Frames sent per cross-check. Six evenly spaced frames cover a sweep's arc
#: without turning an advisory check into a token bill.
MAX_FRAMES = 6

DEFAULT_MODEL = os.environ.get("LOCAISH_GEMINI_MODEL", "gemini-3.6-flash")

#: Above this disagreement the ceiling comparison warns rather than informs.
CEILING_DISAGREE_M = 0.5

_FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png"}

_PROMPT = """\
These are frames from one handheld video sweep of a single room.
Answer from what you can SEE in the frames only. Respond with JSON only,
exactly these keys:

{
  "room_type": "<kitchen | bedroom | living room | bathroom | office | \
hallway | studio | garage | other>",
  "window_count": <integer, distinct windows visible across all frames>,
  "door_count": <integer, distinct doorways or doors visible>,
  "ceiling_visible": <true if any frame shows the ceiling>,
  "ceiling_height_estimate_m": <number or null, your rough estimate>,
  "capture_issues": ["<short phrases: motion blur, too fast, poor light, \
never looks at walls, never looks up, reflective surfaces, ...>"]
}

Count a window or door once even if it appears in several frames. Use null
when you genuinely cannot tell. Do not include any text outside the JSON.
"""


def available() -> bool:
    """Whether a cross-check could run at all: key present, SDK importable."""
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        return False
    try:
        from google import genai  # noqa: F401
    except Exception:
        return False
    return True


def _pick_frames(frames_dir: Path, count: int = MAX_FRAMES) -> list[Path]:
    frames = sorted(
        p
        for p in frames_dir.iterdir()
        if p.suffix.lower() in _FRAME_EXTENSIONS and p.is_file()
    )
    if len(frames) <= count:
        return frames
    # evenly spaced across the sweep, ends included, so the sample sees the
    # whole arc rather than the first six seconds
    idx = [round(i * (len(frames) - 1) / (count - 1)) for i in range(count)]
    return [frames[i] for i in dict.fromkeys(idx)]


def crosscheck(frames_dir: str | Path, *, model: str | None = None) -> dict[str, Any] | None:
    """Ask Gemini to read the room off the frames. Returns the observation
    dict, or None when there was nothing to ask about.

    Raises on API failure; the caller treats the whole check as best-effort.
    """
    frames_dir = Path(frames_dir)
    if not frames_dir.is_dir():
        return None
    frames = _pick_frames(frames_dir)
    if len(frames) < 2:
        return None

    from google import genai
    from google.genai import types as gtypes

    client = genai.Client()
    parts: list[Any] = [
        gtypes.Part.from_bytes(
            data=p.read_bytes(),
            mime_type="image/png" if p.suffix.lower() == ".png" else "image/jpeg",
        )
        for p in frames
    ]
    parts.append(_PROMPT)
    response = client.models.generate_content(
        model=model or os.environ.get("LOCAISH_GEMINI_MODEL", DEFAULT_MODEL),
        contents=parts,
        config=gtypes.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    raw = (response.text or "").strip()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Gemini returned {type(data).__name__}, not an object")

    return {
        "model": model or os.environ.get("LOCAISH_GEMINI_MODEL", DEFAULT_MODEL),
        "frames_used": len(frames),
        "room_type": str(data.get("room_type") or "other"),
        "window_count": _int_or_none(data.get("window_count")),
        "door_count": _int_or_none(data.get("door_count")),
        "ceiling_visible": bool(data.get("ceiling_visible")),
        "ceiling_height_estimate_m": _float_or_none(data.get("ceiling_height_estimate_m")),
        "capture_issues": [str(x) for x in (data.get("capture_issues") or [])][:8],
    }


def _int_or_none(v: Any) -> int | None:
    try:
        return None if v is None else max(0, int(v))
    except (TypeError, ValueError):
        return None


def _float_or_none(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def apply(twin: Twin, observation: dict[str, Any]) -> None:
    """Compare Gemini's reading with the geometry and record the verdicts.

    Adds `semantic_*` checks to the twin's QA report -- info when the two
    agree, warn when they plainly do not -- and re-finalises the report.
    Nothing here ever alters the geometry: the pictures get a vote on whether
    the twin is believable, not on what it says.
    """
    struct = twin.structure
    checks: list[tuple[str, str, str]] = []

    geo_windows = sum(1 for o in struct.openings if o.kind == "window")
    geo_doors = sum(1 for o in struct.openings if o.kind == "door")
    seen_windows = observation.get("window_count")
    seen_doors = observation.get("door_count")

    if seen_windows is not None:
        if geo_windows > seen_windows:
            checks.append(
                (
                    "semantic_openings",
                    "warn",
                    f"The geometry reports {geo_windows} window(s) but Gemini "
                    f"sees {seen_windows} in the frames; a window the pictures "
                    "do not show is usually a gap in the returns wearing a "
                    "window's shape.",
                )
            )
        elif seen_windows > geo_windows:
            checks.append(
                (
                    "semantic_openings",
                    "info",
                    f"Gemini sees {seen_windows} window(s) where the geometry "
                    f"found {geo_windows}; glass returns poorly, so a missed "
                    "window is expected rather than alarming.",
                )
            )
        else:
            checks.append(
                (
                    "semantic_openings",
                    "info",
                    f"Windows agree: the frames and the geometry both say "
                    f"{geo_windows}."
                    + (
                        f" Doors: frames {seen_doors}, geometry {geo_doors}."
                        if seen_doors is not None
                        else ""
                    ),
                )
            )

    est = observation.get("ceiling_height_estimate_m")
    height = struct.ceiling_height
    if est is not None and height is not None:
        gap = abs(est - height)
        status = "warn" if gap > CEILING_DISAGREE_M else "info"
        checks.append(
            (
                "semantic_ceiling",
                status,
                f"The measured ceiling is {height:.2f} m; Gemini's eyeball "
                f"estimate from the frames is {est:.1f} m"
                + (
                    ", which is close enough to corroborate the scale."
                    if status == "info"
                    else " -- a disagreement this large usually means the "
                    "metric scale is wrong, not the ceiling."
                ),
            )
        )

    issues = observation.get("capture_issues") or []
    if issues:
        checks.append(
            (
                "semantic_capture",
                "info",
                "Gemini flags the footage itself: " + "; ".join(issues) + ".",
            )
        )

    for name, status, message in checks:
        twin.qa.add(name, status, message)
    if checks:
        twin.qa.finalize()


__all__ = ["available", "crosscheck", "apply", "MAX_FRAMES", "DEFAULT_MODEL"]
