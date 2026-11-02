"""The location package: the document a scout hands in, with numbers in it.

A scout's deliverable today is a folder of photos and a page of notes --
address, contact, parking, power, "faces south-west". This is that page,
except that the dimensions were measured, the sun on the shoot date was
computed through the actual windows onto the actual floor, the dolly list is
real equipment against real clearance, and every number says how far it can
be trusted. It prints to A4, forwards as one link, and opens on a phone.

The page is light-on-white on purpose: it is read on paper and in email,
beside the client's deck, not in the studio.
"""

from __future__ import annotations

import html
import json
from datetime import date as _date
from pathlib import Path
from typing import Any

from ..film import coverage as cov
from ..film import optics
from ..film import report as reportmod
from ..types import Twin
from . import sunlight

SIZE_NAMES = {s.key: s.name for s in optics.SHOT_SIZES}


# ---------------------------------------------------------------------------
# trust, in plain words
# ---------------------------------------------------------------------------


def trust_statement(twin: Twin) -> dict[str, Any]:
    """What the QA report means for someone who will quote these numbers.

    The pipeline's own verdict is a list of named checks; a producer needs
    one sentence that says whether to trust the metres. The bands follow the
    scale anchors: confidence above 0.8 means independent anchors agreed
    closely, 0.6 to 0.8 is a typical phone-video twin, below 0.4 is shape
    without size.
    """
    qa = twin.qa
    conf = qa.metrics.get("scale_confidence")
    conf = None if conf is None or conf != conf else round(float(conf), 3)
    source = (twin.provenance or {}).get("source_format") or ""
    from_video = str(source).lower() in ("video", "mov", "mp4", "mkv", "webm", "m4v", "avi")
    failing = [c["name"] for c in qa.checks if c.get("status") == "fail"]
    warning = [c["name"] for c in qa.checks if c.get("status") == "warn"]

    if conf is None:
        band, pct = "unknown", None
    elif conf >= 0.8:
        band, pct = "measured", 3
    elif conf >= 0.6:
        band, pct = "metric", 10
    elif conf >= 0.4:
        band, pct = "approximate", 20
    else:
        band, pct = "proportional", None

    if band == "measured":
        line = "Dimensions are measured: independent scale anchors agree to within a few percent."
    elif band == "metric":
        line = ("Dimensions are metric to within roughly ten percent. The room's shape is "
                "measured; its size rests on the camera height and door-height anchors "
                "agreeing. Check one wall with a tape before ordering track.")
    elif band == "approximate":
        line = ("Dimensions are approximate (roughly twenty percent). Treat lens and "
                "distance figures as a first pass and tape one length on the tech scout.")
    elif band == "proportional":
        line = ("The room's proportions are measured but its absolute size is not: one "
                "taped length would fix every number here.")
    else:
        line = "The twin carries no scale confidence; treat distances as proportional."

    notes = []
    if from_video:
        notes.append("Built from a phone video, not a LiDAR scan: a LiDAR export from the same room tightens every figure.")
    if twin.structure.ceiling_height is None:
        notes.append("No ceiling was captured; nothing here answers overhead rigging or boom clearance.")
    if "coverage" in failing:
        notes.append("Parts of the room were not seen by the camera; the floor plan marks the surveyed ground.")
    if "drift" in failing:
        notes.append("The capture drifted along its path, which can bow a long wall by a few centimetres.")
    if "gravity" in failing or "yaw" in failing:
        notes.append("The room's alignment was contested in the self-check; wall angles may be a few degrees off.")
    return {
        "verdict": qa.verdict,
        "band": band,
        "scale_confidence": conf,
        "plus_minus_pct": pct,
        "line": line,
        "notes": notes,
        "failing": failing,
        "warning": warning,
    }


# ---------------------------------------------------------------------------
# the package as data
# ---------------------------------------------------------------------------


def build(
    twin: Twin,
    *,
    job_id: str,
    name: str,
    details: dict | None = None,
    shoot_date: str | None = None,
    hours: tuple[float, ...] = sunlight.DEFAULT_HOURS,
    plans: list[dict] | None = None,
    report: reportmod.ScoutReport | None = None,
    viewer_url: str = "",
    package_url: str = "",
    labels: list[dict] | None = None,
) -> dict[str, Any]:
    """Everything the package page shows, as one JSON-able dict."""
    from . import labels as labelsmod

    details = details or {}
    labels = labels or []
    report = report or reportmod.build(twin)
    rd = report.to_dict()
    day = shoot_date or _date.today().isoformat()
    try:
        _date.fromisoformat(day)
    except ValueError:
        day = _date.today().isoformat()

    st = twin.structure
    fp = st.footprint
    dims = None
    if fp is not None and len(fp) >= 3:
        import numpy as np

        arr = np.asarray(fp, dtype=np.float64)
        ext = arr.max(axis=0) - arr.min(axis=0)
        dims = {"long_m": round(float(ext.max()), 2), "short_m": round(float(ext.min()), 2)}

    out: dict[str, Any] = {
        "job_id": job_id,
        "name": details.get("display_name") or name,
        "generated_at": _date.today().isoformat(),
        "shoot_date": day,
        "shoot_date_is_today": shoot_date is None,
        "details": {k: details.get(k) for k in ("address", "contact", "power", "parking", "access",
                                                 "sound_notes", "restrictions", "fee", "notes")
                    if details.get(k)},
        "trust": trust_statement(twin),
        "space": {**rd["space"], "footprint_dims": dims},
        "camera": rd["camera"],
        "grip": rd["grip"],
        "sound": rd["sound"],
        "light": rd.get("light") or {},
        "caveats": rd["caveats"],
        "viewer_url": viewer_url,
        "package_url": package_url,
        "sun": None,
        "floor_plans": [],
        "contents": labelsmod.describe(labels),
    }
    label_overlay = (lambda X, Y: labelsmod.labels_svg(labels, X, Y)) if labels else None

    # The sun on the shoot date, and where it lands by the hour.
    if twin.georeference is not None:
        from ..film import daylight

        try:
            sched = daylight.sun_schedule(twin, day)
        except ValueError as exc:
            sched = {"error": str(exc)}
        out["sun"] = sched
        out["heading"] = {"deg": twin.georeference.heading_deg,
                          "source": twin.georeference.heading_source}
        up = sched.get("sun_up") or []
        if up:
            for h in hours:
                try:
                    patches = sunlight.sun_patches(twin, day, h)
                except ValueError:
                    continue
                svg = cov.floor_plan_svg(
                    twin, None,
                    overlay=lambda X, Y, p=patches: sunlight.patches_svg(twin, p, X, Y)
                    + (label_overlay(X, Y) if label_overlay else []),
                    caption=f"{_hhmm(h)} solar · sun {patches['compass']} {patches['elevation_deg']:.0f}° up",
                )
                out["floor_plans"].append({
                    "hour": h, "label": _hhmm(h), "svg": svg,
                    "sun": {k: patches[k] for k in ("azimuth_deg", "elevation_deg", "compass")},
                    "patches": patches["patches"],
                })
    if not out["floor_plans"]:
        out["floor_plans"].append({"hour": None, "label": "plan", "svg": cov.floor_plan_svg(twin, None, overlay=label_overlay),
                                   "sun": None, "patches": []})

    # Frames from the best plan already run on this room: the shots the
    # producer has already seen, with their measured setups.
    best = None
    for p in plans or []:
        if best is None or (p.get("planned", 0), p.get("created_at", "")) > (best.get("planned", 0), best.get("created_at", "")):
            best = p
    if best:
        out["plan"] = {
            "plan_id": best.get("plan_id"),
            "title": best.get("title"),
            "planned": best.get("planned"),
            "total": len(best.get("shots") or []),
            "floor_plan_svg": best.get("floor_plan_svg") or "",
            "shots": [
                {
                    "number": s["shot"]["number"],
                    "size": SIZE_NAMES.get(s["shot"].get("size"), s["shot"].get("size")),
                    "subject": s["shot"].get("subject"),
                    "description": s["shot"].get("description"),
                    "frame": s.get("frame"),
                    "focal_mm": (s.get("setup") or {}).get("focal_mm"),
                    "distance_m": (s.get("setup") or {}).get("distance_m"),
                    "cam_z": (s.get("setup") or {}).get("cam_z"),
                    "why": s.get("why"),
                    "held": bool(s.get("setup")),
                }
                for s in best.get("shots") or []
            ],
        }
    return out


def _hhmm(h: float) -> str:
    m = int(round(h * 60))
    return f"{m // 60:02d}:{m % 60:02d}"


# ---------------------------------------------------------------------------
# the package as a page
# ---------------------------------------------------------------------------

_CSS = """
:root { --ink:#15181d; --dim:#5b6470; --mute:#8a94a3; --line:#e4e7eb; --paper:#fff; --soft:#f5f6f8;
        --accent:#15181d; --ok:#1f9d5a; --warn:#b7791f; --bad:#c23b4a; --sun:#f3b63c; --render:#b8853f;
        --ui:system-ui, -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
        --serif:ui-serif, "Iowan Old Style", Charter, Georgia, serif; }
* { box-sizing: border-box; }
body { margin:0; background:var(--soft); color:var(--ink); font: 15px/1.55 var(--ui); }
a { color: var(--ink); text-decoration: underline; text-underline-offset: 2px; }
.sheet { max-width: 900px; margin: 24px auto; background: var(--paper); padding: 40px 44px 48px;
         border: 1px solid var(--line); border-radius: 0; }
header { display:flex; justify-content:space-between; align-items:flex-start; gap:24px; margin-bottom: 18px; }
header h1 { font: 400 36px/1.1 var(--serif); letter-spacing:-.01em; margin: 0 0 4px; }
header .sub { color: var(--dim); font-size: 13px; }
header .brand { display:flex; align-items:center; gap:8px; color:var(--dim); font-size:12px; font-weight:500;
                white-space:nowrap; text-decoration:none; }
.hero { display:grid; grid-template-columns: 1.3fr 1fr; gap: 18px; margin: 10px 0 22px; }
.hero img { width:100%; aspect-ratio: 16/9; object-fit: cover; border-radius: 0; background:#111; display:block; }
.facts { display:grid; grid-template-columns: 1fr 1fr; gap: 10px 14px; align-content:start; }
.fact b { display:block; font: 400 24px/1.2 var(--serif); letter-spacing:-.01em; font-variant-numeric: tabular-nums; }
.fact span { color: var(--dim); font-size: 12px; }
.trust { border-left: 2px solid var(--accent); background: var(--soft); padding: 12px 14px; border-radius: 0; margin: 0 0 22px; }
.trust.approximate, .trust.proportional, .trust.unknown { border-color: var(--warn); }
.trust.measured { border-color: var(--ok); }
.trust b { font-size: 13px; }
.trust ul { margin: 6px 0 0 18px; padding:0; color: var(--dim); font-size: 12.5px; }
h2 { font: 500 11px/1.2 var(--ui); letter-spacing:.08em; text-transform: uppercase; color: var(--mute); margin: 28px 0 10px;
     padding-bottom: 6px; border-bottom: 1px solid var(--line); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th { text-align:left; color: var(--mute); font-weight: 500; font-size: 11.5px; padding: 6px 8px; border-bottom: 1px solid var(--line); }
td { padding: 7px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.tag { display:inline-block; font-size: 11px; padding: 1px 7px; border-radius: 0; border:1px solid var(--line); color: var(--dim); }
.tag.ok { color: var(--ok); border-color: #bfe6cf; } .tag.warn { color: var(--warn); border-color: #efd9a8; } .tag.bad { color: var(--bad); border-color: #f0c0c6; }
.sunrow { display:grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.sunrow figure { margin:0; }
.sunrow figure svg { width:100%; height:auto; border-radius: 0; display:block; }
.sunrow figcaption { font-size: 12px; color: var(--dim); margin-top: 6px; }
.sunrow figcaption b { color: var(--ink); }
.kv { display:grid; grid-template-columns: 140px 1fr; gap: 6px 14px; font-size: 13px; }
.kv dt { color: var(--mute); } .kv dd { margin:0; white-space: pre-wrap; }
.shots { display:grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.shot img { width:100%; aspect-ratio: 16/9; object-fit: cover; border-radius: 0; background:#111; display:block; }
.shot b { display:block; font-size: 13px; margin-top: 6px; }
.shot span { color: var(--dim); font-size: 12px; }
.planimg svg { width:100%; height:auto; border-radius: 0; display:block; }
.caveats { color: var(--dim); font-size: 12.5px; }
.caveats li { margin-bottom: 4px; }
footer { margin-top: 34px; padding-top: 14px; border-top: 1px solid var(--line); color: var(--mute); font-size: 11.5px;
         display:flex; justify-content:space-between; gap: 12px; flex-wrap: wrap; }
.actions { display:flex; gap: 8px; justify-content:flex-end; max-width: 900px; margin: 0 auto 8px; padding: 0 4px; }
.actions a, .actions button { font: inherit; font-size: 13px; padding: 7px 12px; border-radius: 6px; border: 1px solid var(--ink);
                 background: #fff; color: var(--ink); text-decoration:none; cursor:pointer; }
.actions .primary { background: var(--ink); color:#fff; border-color: var(--ink); }
.hours { display:flex; gap:6px; margin: 0 0 10px; }
.hours button { font: inherit; font-size: 12px; padding: 4px 10px; border-radius: 6px; border:1px solid var(--line); background:#fff; cursor:pointer; }
.hours button[aria-pressed="true"] { background: var(--sun); border-color: var(--sun); }
.legend { font-size: 12px; color: var(--dim); margin-top: 8px; }
.legend i { display:inline-block; width: 12px; height: 12px; background: rgba(255,207,90,.6); border:1px solid #ffb020; vertical-align: -2px; margin-right: 4px; }
@media (max-width: 700px) { .sheet { padding: 22px 16px; margin: 0; border-radius:0; } .hero, .sunrow, .shots { grid-template-columns: 1fr; } .kv { grid-template-columns: 1fr; } }
@media print { body { background:#fff; } .sheet { border:0; margin:0; max-width:none; padding: 0; } .actions { display:none; }
               h2 { break-after: avoid; } .sunrow, .shots, table { break-inside: avoid; } a { color: inherit; text-decoration: none; } }
"""


def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def _m(v, d=2, unit=" m") -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{d}f}{unit}"
    except (TypeError, ValueError):
        return "—"


def _intervals(iv) -> str:
    if not iv:
        return "none"
    return ", ".join(f"{g['from']}–{g['to']}" for g in iv)


def render_html(pkg: dict[str, Any], *, image_base: str = "", print_only: bool = False) -> str:
    """The package as a standalone page. `image_base` prefixes image URLs
    (a share link serves images under its own token)."""
    t = pkg["trust"]
    sp = pkg["space"]
    d = pkg.get("details") or {}
    parts: list[str] = []
    add = parts.append

    add("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
    add("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    add(f"<title>{_e(pkg['name'])} · location package</title>")
    add(f"<style>{_CSS}</style></head><body>")
    if not print_only:
        add("<div class='actions'>")
        if pkg.get("viewer_url"):
            add(f"<a href='{_e(pkg['viewer_url'])}' target='_blank'>Walk the 3D twin</a>")
        add("<button onclick='window.print()' class='primary'>Print / save PDF</button></div>")
    add("<div class='sheet'>")

    # -- header -----------------------------------------------------------
    add("<header><div>")
    add(f"<h1>{_e(pkg['name'])}</h1>")
    sub = []
    if d.get("address"):
        sub.append(_e(d["address"]))
    sub.append(f"shoot date {_e(pkg['shoot_date'])}" + (" (today: set the shoot date for the real sun)" if pkg.get("shoot_date_is_today") else ""))
    add(f"<div class='sub'>{' · '.join(sub)}</div></div>")
    add("<a class='brand' href='/'><svg viewBox='0 0 32 32' width='16' height='16' aria-hidden='true'><path d='M11 4H8a4 4 0 0 0-4 4v3M21 4h3a4 4 0 0 1 4 4v3M28 21v3a4 4 0 0 1-4 4h-3M11 28H8a4 4 0 0 1-4-4v-3' fill='none' stroke='#2f6fed' stroke-width='2.6' stroke-linecap='round'/><rect x='11' y='11' width='10' height='10' rx='2.6' transform='rotate(45 16 16)' fill='#2f6fed'/></svg>Location package · Locaish</a>")
    add("</header>")

    # -- hero: still + facts ---------------------------------------------
    add("<div class='hero'>")
    add(f"<img src='{_e(image_base)}/thumb/{_e(pkg['job_id'])}' alt='the room'>")
    add("<div class='facts'>")
    dims = sp.get("footprint_dims")
    add(f"<div class='fact'><b>{_m(sp.get('floor_area_m2'), 1, ' m²')}</b><span>floor area"
        + (f" · {dims['long_m']:.1f} × {dims['short_m']:.1f} m" if dims else "") + "</span></div>")
    add(f"<div class='fact'><b>{_m(sp.get('ceiling_height_m'))}</b><span>ceiling{'' if sp.get('ceiling_height_m') else ' (not captured)'}</span></div>")
    add(f"<div class='fact'><b>{_m(sp.get('workable_floor_m2'), 1, ' m²')}</b><span>workable floor</span></div>")
    wins = [o for o in sp.get("openings", []) if o["kind"] != "door"]
    add(f"<div class='fact'><b>{len(wins)}</b><span>window{'s' if len(wins) != 1 else ''} detected</span></div>")
    cam = pkg.get("camera") or {}
    add(f"<div class='fact'><b>{_m(cam.get('longest_camera_run_m'), 1)}</b><span>longest camera run</span></div>")
    so = pkg.get("sound") or {}
    rt = (so.get("rt60_s") or {}).get("typical")
    add(f"<div class='fact'><b>{_e(so.get('verdict') or '—')}</b><span>sound{(' · RT60 ~' + str(rt) + ' s') if rt else ''}</span></div>")
    add("</div></div>")

    # -- trust ------------------------------------------------------------
    add(f"<div class='trust {_e(t['band'])}'><b>{_e(t['line'])}</b>")
    if t.get("notes"):
        add("<ul>" + "".join(f"<li>{_e(n)}</li>" for n in t["notes"]) + "</ul>")
    add("</div>")

    # -- practical --------------------------------------------------------
    if d:
        add("<h2>Practical</h2><dl class='kv'>")
        for k, label in (("address", "Address"), ("contact", "Contact"), ("access", "Access & load-in"),
                         ("parking", "Parking"), ("power", "Power"), ("sound_notes", "Sound"),
                         ("restrictions", "Restrictions"), ("fee", "Fee"), ("notes", "Notes")):
            if d.get(k):
                add(f"<dt>{label}</dt><dd>{_e(d[k])}</dd>")
        add("</dl>")

    # -- what is in the room ----------------------------------------------
    contents = pkg.get("contents") or []
    if contents:
        add("<h2>In the room</h2>")
        add("<table><tr><th>Object</th><th>Kind</th><th class='num'>Height</th><th>Status</th></tr>")
        for c in contents:
            add(f"<tr><td>{_e(c['label'])}</td><td>{_e(c['kind'])}</td><td class='num'>{_m(c['height_m'])}</td>"
                f"<td><span class='tag {'ok' if c['source'] == 'user' else ''}'>{_e(c['status'])}</span></td></tr>")
        add("</table><p class='caveats'>Confirmed items were named by a person looking at the photographs; the "
            "rest were named by a vision model from the capture and are marked with a question mark on the plans.</p>")

    # -- sun --------------------------------------------------------------
    sun = pkg.get("sun")
    add(f"<h2>Light on {_e(pkg['shoot_date'])}</h2>")
    if not sun:
        add("<p class='caveats'>No coordinates on this twin, so there is no sun schedule. Set the address "
            "and heading in the studio to unlock it.</p>")
    elif sun.get("error"):
        add(f"<p class='caveats'>{_e(sun['error'])}</p>")
    else:
        up = sun.get("sun_up") or []
        gold = sun.get("golden_hour") or []
        add("<dl class='kv'>")
        add(f"<dt>Sun up</dt><dd>{_e(_intervals(up))} · peak {_e(sun.get('max_elevation_deg'))}° · {_e(sun.get('times_are'))}</dd>")
        add(f"<dt>Golden hour</dt><dd>{_e(_intervals(gold))}</dd>")
        for i, w in enumerate(sun.get("windows") or [], 1):
            add(f"<dt>Window {i}</dt><dd>{w['size_m'][0]} × {w['size_m'][1]} m, sill {w['sill_m']} m, faces "
                f"<b>{_e(w['faces'])}</b> ({w['bearing_deg']:.0f}°) · direct sun {_e(_intervals(w['direct_sun']))}</dd>")
        if sun.get("heading_caveat"):
            add(f"<dt></dt><dd class='caveats'>{_e(sun['heading_caveat'])}</dd>")
        add("</dl>")
    fps = pkg.get("floor_plans") or []
    if fps:
        add("<div class='sunrow'>")
        for f in fps:
            cap = ""
            if f.get("sun"):
                n = len(f.get("patches") or [])
                cap = (f"<b>{_e(f['label'])}</b> solar · sun {_e(f['sun']['compass'])}, {f['sun']['elevation_deg']:.0f}° up · "
                       + (f"direct sun through {n} window{'s' if n != 1 else ''}" if n else "no direct sun inside"))
            add(f"<figure>{f['svg']}<figcaption>{cap}</figcaption></figure>")
        add("</div>")
        if any(f.get("sun") for f in fps):
            add("<div class='legend'><i></i>where direct sun lands on the floor through each window, clear sky, nothing "
                "outside shading the glass. Green dashes: ground the capture surveyed."
                + (" Named objects are marked; a question mark means the model named it, not a person." if contents else "")
                + "</div>")

    # -- camera -----------------------------------------------------------
    add("<h2>Camera</h2>")
    if cam.get("shots"):
        add(f"<p class='caveats'>Lenses that hold each framing from the far end of the longest run "
            f"({_m(cam.get('longest_camera_run_m'), 1)}), {_e(cam.get('sensor'))} sensor.</p>")
        add("<table><tr><th>Framing</th><th>Covers</th><th class='num'>Lens from the far end</th><th class='num'>Closest on an 18 mm</th></tr>")
        for sh in cam["shots"]:
            a = sh["at_max_distance"]
            lens = f"{a['nearest_prime_mm']:g} mm" if a.get("nearest_prime_mm") else f"{a['focal_mm']:.0f} mm (longer than the prime set)"
            add(f"<tr><td>{_e(sh['shot'])}</td><td>{_e(sh['covers'])}</td><td class='num'>{_e(lens)}</td><td class='num'>{sh['min_distance_m']} m</td></tr>")
        add("</table>")
    else:
        add(f"<p class='caveats'>{_e(cam.get('note') or 'no camera positions')}</p>")

    # -- grip -------------------------------------------------------------
    grip = pkg.get("grip") or {}
    add("<h2>Grip: what fits</h2>")
    add("<table><tr><th>Equipment</th><th>Verdict</th><th class='num'>Floor it fits on</th><th class='num'>Best clearance</th></tr>")
    for f in grip.get("fits", []):
        cls = "ok" if f["verdict"] == "fits" else ("warn" if f["verdict"] == "marginal" else "bad")
        add(f"<tr><td>{_e(f['gear'])}</td><td><span class='tag {cls}'>{_e(f['verdict'])}</span></td>"
            f"<td class='num'>{f['area_inside_surveyed_m2']} m²</td><td class='num'>{f['best_clearance_m']} m</td></tr>")
    add("</table>")
    if grip.get("note"):
        add(f"<p class='caveats'>{_e(grip['note'])}</p>")

    # -- sound ------------------------------------------------------------
    if so and "error" not in so:
        add("<h2>Sound</h2>")
        r = so.get("rt60_s") or {}
        add(f"<p>Volume {so.get('volume_m3')} m³ · reverberation (RT60) {r.get('softest')}–{r.get('hardest')} s depending on finishes, "
            f"typical {r.get('typical')} s · <b>{_e(so.get('verdict'))}</b> for dialogue.</p>")
        for w in so.get("warnings") or []:
            add(f"<p class='caveats'>{_e(w)}</p>")

    # -- the scene, if one was planned here ------------------------------
    plan = pkg.get("plan")
    if plan and plan.get("shots"):
        add(f"<h2>Scene: {_e(plan.get('title') or 'coverage')} · {plan.get('planned')} of {plan.get('total')} shots hold</h2>")
        add("<div class='shots'>")
        for s in plan["shots"]:
            img = (f"<img src='{_e(image_base)}/plan-image/{_e(pkg['job_id'])}/{_e(plan['plan_id'])}/{_e(s['frame'])}' alt=''>"
                   if s.get("frame") else "<div style='aspect-ratio:16/9;background:#f5f6f8'></div>")
            meta = (f"{s['focal_mm']:.0f} mm · {s['distance_m']:.1f} m · h {s['cam_z']:.2f} m"
                    if s.get("held") and s.get("focal_mm") else "no clear setup in this room")
            add(f"<div class='shot'>{img}<b>#{s['number']} · {_e(s['size'])} of {_e(s['subject'])}</b><span>{_e(meta)}</span></div>")
        add("</div>")
        if plan.get("floor_plan_svg"):
            add(f"<div class='planimg' style='margin-top:12px'>{plan['floor_plan_svg']}</div>")

    # -- caveats ----------------------------------------------------------
    if pkg.get("caveats"):
        add("<h2>Read before quoting any number above</h2><ul class='caveats'>")
        for c in pkg["caveats"]:
            add(f"<li>{_e(c)}</li>")
        add("</ul>")

    add("<footer><span>Generated by Locaish from a phone capture. Every figure traces to a measurement of the twin, "
        "a solar ephemeris or a catalogue dimension; nothing is estimated by a language model.</span>"
        f"<span>{_e(pkg.get('generated_at'))}</span></footer>")
    add("</div></body></html>")
    return "".join(parts)


def write(pkg: dict[str, Any], directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "package.json").write_text(json.dumps(pkg, indent=2, default=str))
    out = directory / "package.html"
    out.write_text(render_html(pkg))
    return out
