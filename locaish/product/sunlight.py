"""Where the sun lands on the floor, on the shoot date, at a given hour.

The question a DP asks on a recce is not "which way does the window face"
but "at three o'clock on the fourteenth, where is the patch of hard sun, and
is it on the actor or on the wall behind her". That is a projection: the
window is a rectangle in space, the sun is a direction, the floor is a
plane. Each window corner is carried along the sun's ray until it meets the
floor, and the four landing points are the patch.

No photometry is attempted. The patch says where direct sun *can* fall if
the sky is clear and nothing outside the window shades it; it does not say
how bright. That is the honest limit of a twin that has never seen outside.
"""

from __future__ import annotations

import math
from datetime import date as _date
from datetime import datetime, timedelta, timezone

import numpy as np

from ..film import daylight
from ..types import Twin

# Which hours of the day a package shows, in local solar time. Three is the
# usual crew question ("morning, midday, afternoon"); callers pick others.
DEFAULT_HOURS = (9.0, 12.0, 15.0)


def sun_at(twin: Twin, day: _date | str, hour_solar: float) -> dict:
    """The sun's azimuth and elevation at a solar hour, and its direction
    in twin space (unit vector *toward* the sun)."""
    geo = twin.georeference
    if geo is None:
        raise ValueError("this twin has no georeference, so the sun cannot be placed")
    if isinstance(day, str):
        day = _date.fromisoformat(day)
    base = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    when = base + timedelta(hours=hour_solar - geo.utc_offset_hours)
    az, el = daylight.solar_position(geo.latitude, geo.longitude, when)
    a, e = math.radians(az), math.radians(el)
    toward_enu = np.array([math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)])
    # enu_from_twin maps twin -> ENU; its transpose maps back.
    toward_twin = geo.enu_from_twin().T @ toward_enu
    return {
        "hour": float(hour_solar),
        "azimuth_deg": round(az, 1),
        "elevation_deg": round(el, 1),
        "compass": daylight.COMPASS[int(round(az / 22.5)) % 16],
        "toward_sun_twin": toward_twin,
    }


def window_corners(opening) -> np.ndarray:
    """The four corners of an opening's rectangle in twin space (4, 3).

    The opening's frame is its centre, its into-room normal, gravity up and
    the tangent across the wall; width runs along the tangent, height along
    up. Openings are detected on vertical walls, so up is the twin's +Z.
    """
    c = np.asarray(opening.center, dtype=np.float64).reshape(3)
    n = np.asarray(opening.normal, dtype=np.float64).reshape(3)
    up = np.array([0.0, 0.0, 1.0])
    t = np.cross(up, n)
    norm = float(np.linalg.norm(t))
    if norm < 1e-9:
        t = np.array([1.0, 0.0, 0.0])
    else:
        t = t / norm
    hw, hh = float(opening.width) / 2.0, float(opening.height) / 2.0
    return np.array([
        c - t * hw - up * hh,
        c + t * hw - up * hh,
        c + t * hw + up * hh,
        c - t * hw + up * hh,
    ])


def sun_patches(twin: Twin, day: _date | str, hour_solar: float) -> dict:
    """Direct-sun patches on the floor from every window, at one solar hour.

    Returns the sun's position and a list of patches, one per window the sun
    is in front of; each patch is the four floor points (x, y) the window's
    corners project to, in the window's corner order, plus the window index
    so the page can label them.
    """
    sun = sun_at(twin, day, hour_solar)
    toward = sun["toward_sun_twin"]
    el = sun["elevation_deg"]
    out = {k: v for k, v in sun.items() if k != "toward_sun_twin"}
    out["patches"] = []
    floor_z = float(twin.structure.floor_z)
    if el <= daylight.HORIZON_DEG or toward[2] <= 1e-6:
        out["note"] = "sun below the horizon"
        return out
    # Light travels from the sun: the ray through a corner heads away from it.
    d = -toward
    openings = getattr(twin.structure, "openings", None) or []
    for j, op in enumerate(openings):
        if getattr(op, "kind", "window") == "door":
            continue
        n = np.asarray(op.normal, dtype=np.float64).reshape(3)
        # Into-room normal dotted with the ray direction: positive means the
        # light is coming through the pane into the room, not hitting the
        # wall from inside.
        incidence = float(np.dot(d, n))
        if incidence <= daylight.MIN_INCIDENCE:
            continue
        corners = window_corners(op)
        t = (corners[:, 2] - floor_z) / (-d[2])
        landed = corners + t[:, None] * d[None, :]
        out["patches"].append({
            "window": j,
            "kind": op.kind,
            "incidence": round(incidence, 2),
            "throw_m": round(float(np.linalg.norm(landed[:, :2].mean(axis=0) - corners[:, :2].mean(axis=0))), 2),
            "floor_xy": [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in landed],
        })
    return out


def north_in_twin(twin: Twin) -> tuple[float, float] | None:
    """Unit vector in twin XY pointing true north, or None without a heading."""
    geo = twin.georeference
    if geo is None:
        return None
    h = math.radians(geo.heading_deg)
    return (math.cos(h), math.sin(h))


def patches_svg(twin: Twin, patches: dict, X, Y, *, clip_id: str | None = None) -> list[str]:
    """SVG fragments for the patches, in the plan's coordinate functions."""
    parts = []
    if not patches.get("patches"):
        return parts
    clip = f' clip-path="url(#{clip_id})"' if clip_id else ""
    parts.append(f"<g{clip}>")
    for p in patches["patches"]:
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in p["floor_xy"])
        parts.append(
            f'<polygon points="{pts}" fill="#ffcf5a" fill-opacity="0.45" stroke="#ffb020" '
            f'stroke-width="1.2" stroke-opacity="0.9"/>'
        )
    parts.append("</g>")
    return parts
