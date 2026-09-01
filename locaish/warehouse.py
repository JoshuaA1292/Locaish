"""The shot table lives in ClickHouse; this module is the only thing that writes it.

The division of labour is deliberate. Loading is done here, over
`clickhouse-connect`, in bulk and column-oriented -- a sweep is a few hundred
thousand rows and inserting them row-wise through an agent would be theatre.
*Reading* is done by the agent, at runtime, through the official ClickHouse MCP
server (`mcp-clickhouse`), which stays read-only exactly as it ships. The agent
never sees this module; it sees a database.

Configuration comes from the same `CLICKHOUSE_*` environment variables the MCP
server reads, so one set of variables configures both halves:

    CLICKHOUSE_HOST      required
    CLICKHOUSE_USER      default: "default"
    CLICKHOUSE_PASSWORD  default: ""
    CLICKHOUSE_PORT      default: 8443 when secure, 8123 when not
    CLICKHOUSE_SECURE    default: "true" (ClickHouse Cloud); "false" for local
    CLICKHOUSE_DATABASE  default: "locaish"

The schema is chosen for the one query shape the table exists to serve --
selective filters (location, shot size, lens, booleans) followed by top-N on a
score or a measurement. Hence the sort key `(location, shot_size, focal_mm)`:
the brief always names a location, almost always names a framing, and often
names glass; everything else is a range filter that ClickHouse skips granules
with. Partitioning by location makes replacing one room's sweep a partition
drop instead of a mutation.
"""

from __future__ import annotations

import os

import numpy as np

TABLE = "shot_setups"

DDL = f"""
CREATE TABLE IF NOT EXISTS {{db}}.{TABLE} (
    location               LowCardinality(String),
    setup_id               UInt32,
    cam_x                  Float32,
    cam_y                  Float32,
    cam_z                  Float32,
    subj_x                 Float32,
    subj_y                 Float32,
    distance_m             Float32,
    yaw_deg                Float32,
    pitch_deg              Float32,
    focal_mm               Float32,
    sensor                 LowCardinality(String),
    shot_size              LowCardinality(String),
    size_fit               Float32,
    framed_height_m        Float32,
    subject_fill           Float32,
    fov_h_deg              Float32,
    dof_near_m             Float32,
    dof_far_m              Float32,
    dof_infinite           UInt8,
    visible                UInt8,
    surveyed               UInt8,
    clearance_m            Float32,
    headroom_m             Float32,
    window_in_frame        UInt8,
    window_behind_subject  UInt8,
    background_depth_m     Float32,
    backup_room_m          Float32,
    key_angle_deg          Float32,
    key_quality            LowCardinality(String),
    axis_wall_angle_deg    Float32,
    portrait_ok            UInt8,
    score                  Float32
)
ENGINE = MergeTree
PARTITION BY location
ORDER BY (location, shot_size, focal_mm, setup_id)
"""

# Columns added after the table first shipped. `ensure_schema` adds any that
# an existing table lacks, so a warehouse created by an older sweep keeps
# working -- its old rows read as zero/"none" on the new columns, which is
# what "unmeasured" should look like until the location is re-swept.
_ADDED_COLUMNS = (
    ("background_depth_m", "Float32"),
    ("backup_room_m", "Float32"),
    ("key_angle_deg", "Float32"),
    ("key_quality", "LowCardinality(String)"),
    ("axis_wall_angle_deg", "Float32"),
    ("portrait_ok", "UInt8"),
)


class WarehouseError(RuntimeError):
    """ClickHouse is not configured, not reachable, or refused the load."""


def configured() -> bool:
    """Whether the environment names a ClickHouse to talk to at all."""
    return bool(os.environ.get("CLICKHOUSE_HOST"))


def connection_env() -> dict[str, str]:
    """The `CLICKHOUSE_*` variables as the MCP server should see them.

    Defaults are resolved here so the MCP subprocess and this module cannot
    drift: both describe the same server or neither does.
    """
    secure = os.environ.get("CLICKHOUSE_SECURE", "true").strip().lower() not in (
        "false", "0", "no",
    )
    return {
        "CLICKHOUSE_HOST": os.environ.get("CLICKHOUSE_HOST", ""),
        "CLICKHOUSE_PORT": os.environ.get(
            "CLICKHOUSE_PORT", "8443" if secure else "8123"
        ),
        "CLICKHOUSE_USER": os.environ.get("CLICKHOUSE_USER", "default"),
        "CLICKHOUSE_PASSWORD": os.environ.get("CLICKHOUSE_PASSWORD", ""),
        "CLICKHOUSE_SECURE": "true" if secure else "false",
        "CLICKHOUSE_DATABASE": database(),
    }


def database() -> str:
    return os.environ.get("CLICKHOUSE_DATABASE", "locaish")


def client():
    """A connected clickhouse-connect client, or a clear refusal."""
    if not configured():
        raise WarehouseError(
            "CLICKHOUSE_HOST is not set. Point it at a ClickHouse Cloud "
            "instance (with CLICKHOUSE_PASSWORD) or a local server "
            "(CLICKHOUSE_SECURE=false), and the shot search comes alive."
        )
    import clickhouse_connect

    env = connection_env()
    try:
        return clickhouse_connect.get_client(
            host=env["CLICKHOUSE_HOST"],
            port=int(env["CLICKHOUSE_PORT"]),
            username=env["CLICKHOUSE_USER"],
            password=env["CLICKHOUSE_PASSWORD"],
            secure=env["CLICKHOUSE_SECURE"] == "true",
        )
    except Exception as exc:
        raise WarehouseError(f"could not reach ClickHouse: {exc}") from exc


def ensure_schema(ch=None) -> None:
    ch = ch or client()
    db = database()
    ch.command(f"CREATE DATABASE IF NOT EXISTS {db}")
    ch.command(DDL.format(db=db))
    for name, typ in _ADDED_COLUMNS:
        ch.command(f"ALTER TABLE {db}.{TABLE} ADD COLUMN IF NOT EXISTS {name} {typ} AFTER window_behind_subject")


def load_sweep(sweep, ch=None, progress=None) -> int:
    """Replace this location's setups with a fresh sweep. Returns rows loaded.

    Replacement is a partition drop, which is atomic and instant, rather than a
    DELETE mutation that would leave old and new rows visible together while it
    runs in the background.
    """
    ch = ch or client()
    ensure_schema(ch)
    db = database()

    cols = sweep.columns
    names = list(cols.keys())
    n = len(sweep)
    if n == 0:
        return 0

    ch.command(
        f"ALTER TABLE {db}.{TABLE} DROP PARTITION %(p)s",
        parameters={"p": sweep.location},
    )

    if progress:
        progress(f"loading {n:,} setups into ClickHouse")
    data = []
    for name in names:
        col = cols[name]
        if col.dtype == object:
            data.append([str(v) for v in col])
        elif col.dtype == np.uint8:
            data.append(col.astype(np.uint8))
        else:
            data.append(np.ascontiguousarray(col))
    ch.insert(
        f"{db}.{TABLE}",
        data,
        column_names=names,
        column_oriented=True,
    )
    return n


def location_counts(ch=None) -> dict[str, int]:
    """Rows per location currently in the table -- the studio's status line."""
    ch = ch or client()
    ensure_schema(ch)
    res = ch.query(
        f"SELECT location, count() FROM {database()}.{TABLE} GROUP BY location"
    )
    return {row[0]: int(row[1]) for row in res.result_rows}


# ---------------------------------------------------------------------------
# planned coverage: what the planner decided, kept where it can be compared
# ---------------------------------------------------------------------------

PLANS_TABLE = "shot_plans"

# One row per planned shot. The sweep table says what a room *could* hold;
# this one says what a scene *asked* of it and what the room answered --
# which setup, how many candidates there were, and what Gemini thought of
# the frame. Kept per location so the question "which of our scanned
# locations holds this scene" is a GROUP BY rather than a re-plan.
PLANS_DDL = f"""
CREATE TABLE IF NOT EXISTS {{db}}.{PLANS_TABLE} (
    location               LowCardinality(String),
    plan_id                String,
    plan_title             String,
    planner                LowCardinality(String),
    planned_at             DateTime,
    shot_no                UInt16,
    description            String,
    subject                String,
    second_subject         String,
    wanted_size            LowCardinality(String),
    wanted_lens_mm         Float32,
    wanted_height          LowCardinality(String),
    placed                 UInt8,
    setup_id               UInt32,
    shot_size              LowCardinality(String),
    focal_mm               Float32,
    cam_x                  Float32,
    cam_y                  Float32,
    cam_z                  Float32,
    subj_x                 Float32,
    subj_y                 Float32,
    distance_m             Float32,
    yaw_deg                Float32,
    dof_near_m             Float32,
    dof_far_m              Float32,
    dof_infinite           UInt8,
    window_in_frame        UInt8,
    window_behind_subject  UInt8,
    size_fit               Float32,
    score                  Float32,
    candidates             UInt32,
    relaxed                String,
    attempts               UInt8,
    review_score           Float32,
    review_verdict         LowCardinality(String),
    review_notes           String
)
ENGINE = MergeTree
PARTITION BY location
ORDER BY (location, plan_id, shot_no)
"""


def ensure_plans_schema(ch=None) -> None:
    ch = ch or client()
    db = database()
    ch.command(f"CREATE DATABASE IF NOT EXISTS {db}")
    ch.command(PLANS_DDL.format(db=db))


def load_plan(plan, ch=None) -> int:
    """Append a coverage plan's shots. Returns rows written."""
    from datetime import datetime, timezone

    ch = ch or client()
    ensure_plans_schema(ch)
    db = database()
    try:
        when = datetime.fromisoformat(plan.created_at)
    except Exception:  # noqa: BLE001
        when = datetime.now(timezone.utc)
    when = when.replace(tzinfo=None)

    rows = []
    for ps in plan.shots:
        sh = ps.shot
        st = ps.setup or {}
        rv = ps.review
        rows.append([
            plan.location, plan.plan_id, plan.title, plan.planner, when,
            int(sh.number), sh.description, sh.subject, sh.second_subject or "",
            sh.size, float(sh.lens_mm or 0.0), sh.height or "",
            1 if ps.setup else 0,
            int(st.get("setup_id", 0)), str(st.get("shot_size", "")), float(st.get("focal_mm", 0.0)),
            float(st.get("cam_x", 0.0)), float(st.get("cam_y", 0.0)), float(st.get("cam_z", 0.0)),
            float(st.get("subj_x", 0.0)), float(st.get("subj_y", 0.0)),
            float(st.get("distance_m", 0.0)), float(st.get("yaw_deg", 0.0)),
            float(st.get("dof_near_m", 0.0)), float(st.get("dof_far_m", 0.0)), int(st.get("dof_infinite", 0)),
            int(st.get("window_in_frame", 0)), int(st.get("window_behind_subject", 0)),
            float(st.get("size_fit", 0.0)), float(st.get("score", 0.0)),
            int(ps.candidates), "; ".join(ps.relaxed), int(ps.attempts),
            float(rv.score) if rv else -1.0, rv.verdict if rv else "", rv.notes if rv else "",
        ])
    if not rows:
        return 0
    names = [
        "location", "plan_id", "plan_title", "planner", "planned_at",
        "shot_no", "description", "subject", "second_subject",
        "wanted_size", "wanted_lens_mm", "wanted_height",
        "placed", "setup_id", "shot_size", "focal_mm",
        "cam_x", "cam_y", "cam_z", "subj_x", "subj_y", "distance_m", "yaw_deg",
        "dof_near_m", "dof_far_m", "dof_infinite", "window_in_frame", "window_behind_subject",
        "size_fit", "score", "candidates", "relaxed", "attempts",
        "review_score", "review_verdict", "review_notes",
    ]
    ch.insert(f"{db}.{PLANS_TABLE}", rows, column_names=names)
    return len(rows)


def capacity(location: str, ch=None) -> dict:
    """What a room holds, by framing and lens -- the studio's coverage heatmap.

    `clean` is the number a DP actually cares about: a clear sightline and
    no window behind the subject. The per-location ranking underneath is the
    same count for every room in the table, which is the multi-location
    question answered without planning anything.
    """
    ch = ch or client()
    ensure_schema(ch)
    db = database()
    res = ch.query(
        f"""
        SELECT shot_size, focal_mm, count() AS total,
               countIf(visible = 1 AND window_behind_subject = 0) AS clean,
               countIf(visible = 1 AND window_behind_subject = 1) AS backlit
        FROM {db}.{TABLE}
        WHERE location = %(loc)s
        GROUP BY shot_size, focal_mm
        ORDER BY shot_size, focal_mm
        """,
        parameters={"loc": location},
    )
    cells = [
        {"shot_size": r[0], "focal_mm": float(r[1]), "total": int(r[2]), "clean": int(r[3]), "backlit": int(r[4])}
        for r in res.result_rows
    ]
    locs = ch.query(
        f"""
        SELECT location, count() AS setups,
               countIf(visible = 1 AND window_behind_subject = 0) AS clean
        FROM {db}.{TABLE}
        GROUP BY location
        ORDER BY clean DESC
        """
    )
    locations = [{"location": r[0], "setups": int(r[1]), "clean": int(r[2])} for r in locs.result_rows]
    sizes = ["ecu", "bcu", "cu", "mcu", "ms", "mls", "ls", "els"]
    lenses = sorted({c["focal_mm"] for c in cells})
    return {
        "location": location,
        "total": sum(c["total"] for c in cells),
        "clean": sum(c["clean"] for c in cells),
        "sizes": sizes,
        "lenses": lenses,
        "cells": cells,
        "locations": locations,
    }


def nearest_setup(location: str, x: float, y: float, z: float, focal_mm: float, ch=None) -> dict | None:
    """The swept setup nearest a free camera pose on the same lens, with its distance.

    What the viewfinder asks when the user has dragged the camera somewhere:
    is there a scored row for roughly here, and what does it say.
    """
    ch = ch or client()
    db = database()
    res = ch.query(
        f"""
        SELECT setup_id, cam_x, cam_y, cam_z, subj_x, subj_y, distance_m, yaw_deg, pitch_deg,
               focal_mm, shot_size, size_fit, visible, window_behind_subject, window_in_frame,
               clearance_m, score,
               sqrt(pow(cam_x - %(x)s, 2) + pow(cam_y - %(y)s, 2) + pow(cam_z - %(z)s, 2)) AS away_m
        FROM {db}.{TABLE}
        WHERE location = %(loc)s AND abs(focal_mm - %(f)s) < 0.01
        ORDER BY away_m ASC
        LIMIT 1
        """,
        parameters={"loc": location, "x": float(x), "y": float(y), "z": float(z), "f": float(focal_mm)},
    )
    if not res.result_rows:
        return None
    row = dict(zip(res.column_names, res.result_rows[0]))
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}


def plans_by_location(ch=None) -> list[dict]:
    """Every plan in the table, summarised: the cross-location comparison."""
    ch = ch or client()
    ensure_plans_schema(ch)
    db = database()
    res = ch.query(
        f"""
        SELECT location, plan_id, any(plan_title), any(planner), max(planned_at),
               count() AS n_shots, countIf(placed = 1) AS n_placed,
               avgIf(review_score, review_score >= 0) AS gemini_avg,
               countIf(placed = 1 AND window_behind_subject = 0) AS n_clean
        FROM {db}.{PLANS_TABLE}
        GROUP BY location, plan_id
        ORDER BY max(planned_at) DESC
        """
    )
    out = []
    for r in res.result_rows:
        out.append({
            "location": r[0], "plan_id": r[1], "title": r[2], "planner": r[3],
            "planned_at": str(r[4]), "shots": int(r[5]), "placed": int(r[6]),
            "gemini_avg": None if r[7] is None or r[7] != r[7] else round(float(r[7]), 1),
            "clean": int(r[8]),
        })
    return out
