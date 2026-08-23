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
    score                  Float32
)
ENGINE = MergeTree
PARTITION BY location
ORDER BY (location, shot_size, focal_mm, setup_id)
"""


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
