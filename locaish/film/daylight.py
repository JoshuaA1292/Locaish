"""When the sun is where, and which glass it comes through: pure astronomy.

The gaffer's questions about a location -- when is golden hour, when does hard
sun rake that west window, is the morning workable for the dialogue scene --
are ephemeris questions. Nothing here is estimated from the scan's appearance:
the sun's position is computed (the standard low-accuracy solar ephemeris,
good to ~0.01 radians, which is far tighter than a cloud forecast), and the
windows it can enter through are the apertures the twin actually detected,
oriented by the georeference's measured or declared heading.

Times are reported in *local solar time* -- the offset implied by longitude --
and labelled as such. Civil timezones need a tz database and disagree with
themselves across DST; the sun does not. Solar noon is 12:00 by construction,
which is also how a crew actually reasons about light.

The honesty rule from the rest of the pipeline carries through: a heading
whose source is `assumed` produces a schedule that says so, because "the
window faces south-west" is a measurement only if the heading was one.
"""

from __future__ import annotations

import math
from datetime import date as _date
from datetime import datetime, timezone

import numpy as np

from ..types import Twin

# Sun centre at -0.833 deg elevation is the conventional sunrise/sunset
# (refraction plus half a solar disc). Golden hour is the band where the light
# is low, warm and directional; the usual working definition is sun elevation
# between -4 and +6 degrees.
HORIZON_DEG = -0.833
GOLDEN_LOW_DEG = -4.0
GOLDEN_HIGH_DEG = 6.0

# Direct sun "on the glass" needs the sun in front of the pane. Below this
# cosine of incidence the beam is a graze that lights nothing.
MIN_INCIDENCE = 0.10

SAMPLE_MINUTES = 4

COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")


def solar_position(lat_deg: float, lon_deg: float, when_utc) -> tuple[float, float]:
    """(azimuth_deg from true north clockwise, elevation_deg) at a UTC moment."""
    az, el = _solar_vec(lat_deg, lon_deg, np.array([_julian_centuries(when_utc)]))
    return float(az[0]), float(el[0])


def sun_schedule(twin: Twin, on: str | _date | None = None) -> dict:
    """The day's light at this location: rise, set, golden hours, sun-on-glass.

    `on` is an ISO date (defaults to today). Requires the twin to carry a
    georeference; the per-window part additionally needs the heading, and the
    result says how trustworthy that heading is.
    """
    geo = twin.georeference
    if geo is None:
        raise ValueError(
            "this twin has no georeference -- re-ingest with --lat and --lon "
            "(and --heading for per-window answers) to unlock the sun schedule"
        )
    if on is None:
        day = _date.today()
    elif isinstance(on, str):
        day = _date.fromisoformat(on)
    else:
        day = on

    # Sample the whole day in true solar time, converting each sample to UTC
    # via the longitude offset.
    minutes = np.arange(0, 24 * 60, SAMPLE_MINUTES)
    offset_h = geo.utc_offset_hours
    base = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    t_utc = np.array([
        _julian_centuries(base) + (m / 60.0 - offset_h) / 24.0 / 36525.0
        for m in minutes
    ])
    az, el = _solar_vec(geo.latitude, geo.longitude, t_utc)

    out: dict = {
        "date": day.isoformat(),
        "times_are": "local solar time (longitude offset "
        f"{offset_h:+.1f} h from UTC); solar noon is 12:00 by definition",
        "sun_up": _intervals(minutes, el > HORIZON_DEG),
        "golden_hour": _intervals(
            minutes, (el > GOLDEN_LOW_DEG) & (el < GOLDEN_HIGH_DEG)
        ),
        "max_elevation_deg": round(float(el.max()), 1),
    }

    openings = getattr(twin.structure, "openings", None) or []
    glass = [o for o in openings if getattr(o, "kind", "window") != "door"]
    if not glass:
        out["windows"] = []
        out["note"] = "no windows were detected in this twin"
        return out

    heading_note = None
    if geo.heading_source == "assumed":
        heading_note = (
            "the heading was assumed, not measured, so every bearing below "
            "carries that assumption; check one window against a compass"
        )

    enu = geo.enu_from_twin()
    sun_e = np.cos(np.radians(el)) * np.sin(np.radians(az))
    sun_n = np.cos(np.radians(el)) * np.cos(np.radians(az))
    sun_u = np.sin(np.radians(el))
    sun_enu = np.stack([sun_e, sun_n, sun_u], axis=1)      # (T, 3), toward sun

    windows = []
    for o in glass:
        normal = np.asarray(o.normal, dtype=np.float64).reshape(3)
        outward = -(enu @ normal)                           # into-room -> outward, in ENU
        norm = float(np.linalg.norm(outward))
        if norm < 1e-9:
            continue
        outward = outward / norm
        bearing = (math.degrees(math.atan2(outward[0], outward[1]))) % 360.0
        incidence = sun_enu @ outward
        lit = (el > HORIZON_DEG) & (incidence > MIN_INCIDENCE)
        windows.append({
            "size_m": [round(float(o.width), 2), round(float(o.height), 2)],
            "sill_m": round(float(o.sill_height), 2),
            "faces": COMPASS[int(round(bearing / 22.5)) % 16],
            "bearing_deg": round(bearing, 0),
            "direct_sun": _intervals(minutes, lit),
            "peak_incidence": round(float(incidence[lit].max()), 2) if lit.any() else 0.0,
        })
    out["windows"] = windows
    if heading_note:
        out["heading_caveat"] = heading_note
    return out


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _julian_centuries(when_utc: datetime) -> float:
    """Julian centuries since J2000.0 for a timezone-aware UTC datetime."""
    if when_utc.tzinfo is None:
        when_utc = when_utc.replace(tzinfo=timezone.utc)
    epoch = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
    days = (when_utc - epoch).total_seconds() / 86400.0
    return days / 36525.0


def _solar_vec(lat_deg: float, lon_deg: float, t_centuries: np.ndarray):
    """Vectorised low-accuracy solar ephemeris (Meeus). Returns (az_deg, el_deg).

    Azimuth is from true north, clockwise. Elevation includes no refraction --
    the horizon constant used by callers accounts for it in the conventional
    lump.
    """
    t = np.asarray(t_centuries, dtype=np.float64)
    d = t * 36525.0                                   # days since J2000

    g = np.radians((357.529 + 0.98560028 * d) % 360.0)          # mean anomaly
    q = (280.459 + 0.98564736 * d) % 360.0                      # mean longitude
    lam = np.radians(q + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    e = np.radians(23.439 - 0.00000036 * d)                     # obliquity

    ra = np.arctan2(np.cos(e) * np.sin(lam), np.cos(lam))       # right ascension
    dec = np.arcsin(np.sin(e) * np.sin(lam))                    # declination

    gmst = (18.697374558 + 24.06570982441908 * d) % 24.0        # hours
    lst = gmst + lon_deg / 15.0                                 # local sidereal
    ha = np.radians((lst * 15.0) % 360.0) - ra                  # hour angle

    lat = math.radians(lat_deg)
    el = np.arcsin(
        np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)
    )
    az = np.arctan2(
        -np.sin(ha),
        np.tan(dec) * np.cos(lat) - np.sin(lat) * np.cos(ha),
    )
    return (np.degrees(az) % 360.0), np.degrees(el)


def _intervals(minutes: np.ndarray, mask: np.ndarray) -> list[dict]:
    """Contiguous true-runs of `mask` as {'from': 'HH:MM', 'to': 'HH:MM'}."""
    out = []
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return out
    edges = np.flatnonzero(np.diff(m.astype(np.int8)))
    starts = [0] if m[0] else []
    ends = []
    for e in edges:
        if m[e]:
            ends.append(e)
        else:
            starts.append(e + 1)
    if m[-1]:
        ends.append(len(m) - 1)

    def hhmm(i):
        v = int(minutes[i])
        return f"{v // 60:02d}:{v % 60:02d}"

    for s, e in zip(starts, ends):
        out.append({"from": hhmm(s), "to": hhmm(e)})
    return out
