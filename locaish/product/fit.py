"""Which of these rooms holds the scene: the bid-window question, answered.

A commercial production considers five to ten locations per job and can
afford to drive the department heads to one or two. The ranking here is the
measurement engine's answer to "scout six, drive to two": for every room in
the project, every shot in the brief is placed by the deterministic planner
(no model, so the same brief gives the same answer on every machine), and
the room is scored on how many shots it holds outright, how many only with
a compromise, and how many not at all.

The score is deliberately simple and explained in the row that carries it.
A producer will not trust a number out of ten; they will trust "holds 5 of
6 shots, the close-up of JON only with a window behind him".
"""

from __future__ import annotations

from typing import Any

from ..film import coverage as cov
from ..film import report as reportmod
from ..types import Twin

# A shot placed only after relaxing a constraint counts half: it is a shot
# the DP can get, with a caveat they would rather not have.
RELAXED_WEIGHT = 0.5


def shots_from_brief(brief: str) -> list[cov.Shot]:
    """The brief as shots. A typed shot list parses directly; prose does not
    (that is the agent's job), so prose yields nothing and the caller says so."""
    return cov.parse_shot_lines(brief or "")


def fit_location(
    twin: Twin,
    source: cov.SetupSource,
    shots: list[cov.Shot],
    *,
    shoot_date: str | None = None,
    report: reportmod.ScoutReport | None = None,
) -> dict[str, Any]:
    """Score one room against a shot list. Pure: no files, no rendering."""
    out: dict[str, Any] = {"location": twin.name, "shots": [], "held": 0, "relaxed": 0,
                           "missed": 0, "score": 0.0, "warnings": []}
    if shots:
        try:
            plan = cov.plan(twin, shots, source, render=False, title="fit")
        except Exception as exc:  # noqa: BLE001 - a room the sweep declined still gets a row
            out["warnings"].append(f"{type(exc).__name__}: {exc}")
            plan = None
        if plan is not None:
            for ps in plan.shots:
                st = ps.setup
                if st is None:
                    verdict = "missed"
                elif ps.relaxed or st.get("window_behind_subject"):
                    verdict = "relaxed"
                else:
                    verdict = "held"
                out[verdict] += 1
                entry = {
                    "number": ps.shot.number,
                    "size": ps.shot.size,
                    "subject": ps.shot.subject,
                    "verdict": verdict,
                    "relaxed": list(ps.relaxed),
                }
                if st is not None:
                    entry.update({
                        "focal_mm": st.get("focal_mm"),
                        "distance_m": st.get("distance_m"),
                        "cam_z": st.get("cam_z"),
                        "window_behind": bool(st.get("window_behind_subject")),
                        "key_quality": st.get("key_quality"),
                        "background_depth_m": st.get("background_depth_m"),
                    })
                    if st.get("window_behind_subject") and "window behind" not in " ".join(ps.relaxed):
                        entry["relaxed"].append("window behind the subject")
                out["shots"].append(entry)
            out["warnings"].extend(plan.warnings)
            total = max(len(plan.shots), 1)
            out["score"] = round((out["held"] + RELAXED_WEIGHT * out["relaxed"]) / total, 3)
            out["total"] = len(plan.shots)

    # The room on its own merits, from the same report the package prints.
    if report is None:
        try:
            report = reportmod.build(twin)
        except Exception as exc:  # noqa: BLE001
            out["warnings"].append(f"survey failed: {exc}")
            report = None
    if report is not None:
        d = report.to_dict()
        out["trust"] = d["trust"]
        out["floor_area_m2"] = d["space"].get("floor_area_m2")
        out["workable_floor_m2"] = d["space"].get("workable_floor_m2")
        out["ceiling_height_m"] = d["space"].get("ceiling_height_m")
        out["windows"] = sum(1 for o in d["space"].get("openings", []) if o["kind"] != "door")
        out["dollies_fit"] = [f["gear"] for f in d["grip"].get("fits", []) if f["verdict"] == "fits"]
        out["sound"] = (d.get("sound") or {}).get("verdict")
        out["longest_run_m"] = d["camera"].get("longest_camera_run_m")
    if shoot_date and twin.georeference is not None:
        from ..film import daylight

        try:
            sun = daylight.sun_schedule(twin, shoot_date)
            out["sun"] = {
                "date": shoot_date,
                "sun_up": sun.get("sun_up"),
                "golden_hour": sun.get("golden_hour"),
                "windows_with_direct_sun": sum(1 for w in sun.get("windows", []) if w["direct_sun"]),
                "heading_assumed": "heading_caveat" in sun,
            }
        except ValueError:
            pass
    return out


def rank(rows: list[dict]) -> list[dict]:
    """Rooms ordered by fit, then by how much floor there is to work with.

    Ties on the score are broken by workable floor because, between two
    rooms that hold the same shots, the crew wants the one with room for
    the dolly and the lights.
    """
    def key(r):
        return (-(r.get("score") or 0.0), -(r.get("held") or 0), -(r.get("workable_floor_m2") or 0.0))

    out = sorted(rows, key=key)
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out


def summary_line(row: dict) -> str:
    """The sentence a producer reads: what holds, what does not, why."""
    total = row.get("total")
    if not total:
        return "no shot list to test against"
    bits = [f"holds {row.get('held', 0)} of {total} shots"]
    if row.get("relaxed"):
        bits.append(f"{row['relaxed']} with a compromise")
    if row.get("missed"):
        bits.append(f"{row['missed']} not possible")
    caveats = []
    for s in row.get("shots", []):
        if s["verdict"] == "relaxed" and s.get("relaxed"):
            caveats.append(f"#{s['number']} {s['size'].upper()} of {s['subject']}: {s['relaxed'][0]}")
        elif s["verdict"] == "missed":
            caveats.append(f"#{s['number']} {s['size'].upper()} of {s['subject']}: no clear setup")
    line = ", ".join(bits)
    if caveats:
        line += " — " + "; ".join(caveats[:3])
    return line
