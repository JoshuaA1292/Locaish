"""The scout report: what a location manager would hand back after the recce.

Structured by department, because that is how a tech scout is actually run and
how the answers get used -- the gaffer does not read the grip section. Each
department's entry answers the questions its head would have asked standing in
the room, and every figure is either measured from the twin or explicitly
labelled as an assumption.

The report deliberately refuses to summarise itself into a score. A location is
not better or worse in the abstract; it is right or wrong for a particular
scene, and collapsing "3.2 m of headroom, 1.4 s of reverb, one power outlet"
into a number out of ten throws away the only part anyone can act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..types import Twin
from . import acoustics as acousticsmod
from . import equipment as equipmod
from . import optics
from . import space as spacemod


@dataclass
class ScoutReport:
    """Everything the recce established, by department."""

    name: str
    trust: dict[str, Any] = field(default_factory=dict)
    space: dict[str, Any] = field(default_factory=dict)
    camera: dict[str, Any] = field(default_factory=dict)
    grip: dict[str, Any] = field(default_factory=dict)
    sound: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "location": self.name,
            "trust": self.trust,
            "space": self.space,
            "camera": self.camera,
            "grip": self.grip,
            "sound": self.sound,
            "caveats": self.caveats,
        }


def build(twin: Twin, *, cell: float = spacemod.DEFAULT_CELL_M) -> ScoutReport:
    """Survey a twin as a location, and write it up."""
    maps = spacemod.floor_maps(twin, cell=cell)
    report = ScoutReport(name=twin.name)
    caveats: list[str] = []

    # -- how far to trust any of this ------------------------------------
    qa = twin.qa
    report.trust = {
        "qa_verdict": qa.verdict,
        "failing_checks": [c["name"] for c in qa.checks if c["status"] == "fail"],
        "scale_confidence": round(float(qa.metrics.get("scale_confidence", float("nan"))), 3),
        "surveyed_fraction": round(
            float(maps.surveyed[maps.inside].mean()) if maps.inside.any() else 0.0, 3
        ),
    }
    if qa.verdict == "fail":
        caveats.append(
            "the twin failed QA, so every dimension below is provisional; the "
            "failing checks are listed under trust and each one names what it "
            "makes untrustworthy"
        )
    conf = qa.metrics.get("scale_confidence")
    if conf is not None and conf < 0.4:
        caveats.append(
            f"scale confidence is {conf:.2f}: the room's *shape* is measured but "
            "its size is inferred, so treat every distance here as proportional "
            "rather than absolute until one length is checked with a tape"
        )
    inferred = None
    if twin.mesh is not None and twin.mesh.filled is not None:
        inferred = float((twin.mesh.filled > 0.5).mean())
        if inferred > 0.02:
            caveats.append(
                f"{inferred:.0%} of the surface was completed from where the camera "
                "could see rather than from returns; nothing is measured off it"
            )

    # -- the space itself --------------------------------------------------
    standable = maps.standable()
    cell_area = maps.cell ** 2
    height = twin.structure.ceiling_height
    report.space = {
        "floor_area_m2": round(float(twin.structure.floor_area), 1),
        "ceiling_height_m": None if height is None else round(float(height), 2),
        "workable_floor_m2": round(float(standable.sum()) * cell_area, 1),
        "workable_fraction": round(
            float(standable.sum() / max(maps.inside.sum(), 1)), 2
        ),
        "openings": [
            {
                "kind": o.kind,
                "width_m": round(float(o.width), 2),
                "height_m": round(float(o.height), 2),
                "sill_m": round(float(o.sill_height), 2),
            }
            for o in twin.structure.openings
        ],
    }
    if height is None:
        caveats.append(
            "no ceiling was captured, so there is no answer here about lighting "
            "rigs, boom clearance or anything overhead"
        )

    # -- camera ------------------------------------------------------------
    report.camera = _camera_section(twin, maps, standable)

    # -- grip --------------------------------------------------------------
    report.grip = _grip_section(maps, height)

    # -- sound -------------------------------------------------------------
    try:
        ac = acousticsmod.estimate(twin)
        report.sound = ac.to_dict()
        caveats.extend(ac.warnings)
    except ValueError as exc:
        report.sound = {"error": str(exc)}

    report.caveats = caveats
    return report


def _camera_section(twin: Twin, maps, standable) -> dict:
    """What lens the room wants, and how far back the camera can actually get.

    The useful camera answer on a recce is not a list of positions -- it is the
    longest clear sightline in the room, because that is the hard limit on how
    wide a shot can be and therefore on which lenses are worth bringing.
    """
    idx = np.argwhere(standable)
    out: dict[str, Any] = {"sensor": optics.DEFAULT_SENSOR}
    if len(idx) < 2:
        out["note"] = "not enough workable floor to place a camera anywhere"
        return out

    pts = maps.world_of(idx)
    # The two workable cells furthest apart bound every shot in the room.
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    longest = float(np.hypot(*(hi - lo)))
    sensor = optics.SENSORS[optics.DEFAULT_SENSOR]

    out["longest_camera_run_m"] = round(longest, 2)
    out["shots"] = []
    for shot in optics.SHOT_SIZES:
        if shot.key in ("ecu", "els"):
            continue
        # What lens holds this shot from the far end of the room?
        focal = optics.focal_for_framing(sensor, longest, shot.framed_height_m)
        prime = optics.nearest_prime(focal)
        # A prime is only a recommendation if it is actually close. Past the
        # long end of the set the nearest prime is simply the longest one, and
        # reporting that as the answer would put two different shots on the same
        # lens and be wrong about both.
        covered = abs(focal - prime) <= 0.15 * focal
        out["shots"].append({
            "shot": shot.name,
            "covers": shot.covers,
            "at_max_distance": {
                "distance_m": round(longest, 2),
                "focal_mm": round(focal, 1),
                "nearest_prime_mm": prime if covered else None,
                "beyond_prime_set": not covered,
            },
            "min_distance_m": round(
                optics.distance_for_framing(sensor, 18.0, shot.framed_height_m), 2
            ),
        })
    out["note"] = (
        "focal lengths are what holds each framing from the longest clear run in "
        "the room; min_distance_m is how close the camera would sit on an 18 mm, "
        "which is the practical wide end before a face starts to distort"
    )
    return out


def _grip_section(maps, ceiling_height) -> dict:
    """What fits, and where. One entry per dolly and support in the catalogue."""
    out: dict[str, Any] = {"fits": [], "unverified_dimensions": []}
    cell_area = maps.cell ** 2
    for gear in equipmod.of_kind("dolly") + equipmod.of_kind("support"):
        headroom = min(gear.height_m, 1.9)
        ok = spacemod.fits_mask(maps, gear.footprint_m, min_headroom_m=headroom)
        ok_surveyed = ok & maps.surveyed
        best = float(maps.clearance_m[ok].max()) if ok.any() else 0.0
        entry = {
            "gear": gear.name,
            "needs_clear_radius_m": round(0.5 * float(np.hypot(*gear.footprint_m)), 2),
            "area_m2": round(float(ok.sum()) * cell_area, 1),
            "area_inside_surveyed_m2": round(float(ok_surveyed.sum()) * cell_area, 1),
            "best_clearance_m": round(best, 2),
            "verdict": (
                "fits" if ok_surveyed.any()
                else ("marginal" if ok.any() else "does not fit")
            ),
        }
        if not gear.verified:
            out["unverified_dimensions"].append(gear.name)
        out["fits"].append(entry)

    if out["unverified_dimensions"]:
        out["note"] = (
            "dimensions for " + ", ".join(out["unverified_dimensions"]) + " are "
            "class-typical rather than published; confirm against the rental "
            "house's spec sheet before committing to them"
        )
    if ceiling_height is not None:
        out["ceiling_height_m"] = round(float(ceiling_height), 2)
    return out


def render_text(report: ScoutReport) -> str:
    """The report as a location manager would type it up."""
    d = report.to_dict()
    lines: list[str] = []
    add = lines.append

    add(f"LOCATION: {d['location']}")
    t = d["trust"]
    add(f"  twin QA: {t['qa_verdict']}"
        + (f"  (failing: {', '.join(t['failing_checks'])})" if t["failing_checks"] else "")
        + f"   surveyed {t['surveyed_fraction']:.0%} of the floor")

    s = d["space"]
    add("")
    add("SPACE")
    add(f"  floor {s['floor_area_m2']} m2, of which {s['workable_floor_m2']} m2 is workable "
        f"({s['workable_fraction']:.0%})")
    add(f"  ceiling: {s['ceiling_height_m'] or 'not captured'}"
        + (" m" if s["ceiling_height_m"] else ""))
    if s["openings"]:
        for o in s["openings"]:
            add(f"  {o['kind']}: {o['width_m']} x {o['height_m']} m, sill {o['sill_m']} m")
    else:
        add("  no windows or doors were detected")

    c = d["camera"]
    add("")
    add("CAMERA")
    if "longest_camera_run_m" in c:
        add(f"  longest run in the room: {c['longest_camera_run_m']} m  (sensor: {c['sensor']})")
        for sh in c["shots"]:
            a = sh["at_max_distance"]
            lens = (
                f"{a['nearest_prime_mm']:g} mm" if a["nearest_prime_mm"]
                else f"{a['focal_mm']:.0f} mm (longer than the prime set)"
            )
            add(f"  {sh['shot']:<20s} {sh['covers']:<30s} "
                f"{lens} from {a['distance_m']} m, "
                f"or step in to {sh['min_distance_m']} m on an 18")
    else:
        add(f"  {c.get('note', 'no camera positions')}")

    g = d["grip"]
    add("")
    add("GRIP")
    for f in g["fits"]:
        add(f"  {f['gear']:<40s} {f['verdict']:<14s} "
            f"{f['area_inside_surveyed_m2']} m2 of surveyed floor "
            f"(needs {f['needs_clear_radius_m']} m radius, best here "
            f"{f['best_clearance_m']} m)")

    so = d["sound"]
    add("")
    add("SOUND")
    if "error" in so:
        add(f"  {so['error']}")
    else:
        r = so["rt60_s"]
        add(f"  volume {so['volume_m3']} m3   RT60 {r['softest']}-{r['hardest']} s "
            f"(typical {r['typical']} s)")
        add(f"  verdict: {so['verdict']}")

    if d["caveats"]:
        add("")
        add("READ THIS BEFORE QUOTING ANY NUMBER ABOVE")
        for c_ in d["caveats"]:
            add(f"  - {c_}")
    return "\n".join(lines)
