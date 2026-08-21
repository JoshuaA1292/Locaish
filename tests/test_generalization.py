"""Rooms the pipeline was never tuned against.

The README's non-negotiable constraint is that every claim has to generalise to
a room we have never seen. The catalogue fixtures cannot prove that: they are
eight rooms, and every threshold in the codebase has had the opportunity to
learn them. So this module generates rooms from a seed the tuning never saw --
random dimensions, random unit, random tilt and yaw, random noise, density and
occlusion -- and holds the pipeline to its promises.

The promise being tested is deliberately not "every room is measured
correctly". Some rooms are genuinely ambiguous: a nearly cubical space gives
almost nothing to distinguish up from sideways, and a room whose dimensions in
inches are as architecturally plausible as its dimensions in centimetres cannot
be told apart from the geometry alone. Insisting those come out right would
mean guessing, and a confident guess is the failure this whole project is built
to avoid.

What is promised instead, and what these tests enforce:

    a twin whose QA verdict is `pass` is accurate.

A room the pipeline cannot handle must degrade to `warn` or `fail`, and then it
is allowed to be wrong. That is the contract a location manager can actually
use, and the last test in this file -- the one that says nothing may be wrong
quietly -- is the one that must never be relaxed.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from locaish import fixtures
from locaish.formats.ply import write_ply
from locaish.scan.dimensions import measure_room
from locaish.scan.ingest import IngestOptions, ingest

#: Rooms per sweep, and the seed nothing was tuned on. Fixed so any failure is
#: reproducible by room number.
SWEEP_SIZE = 25
SWEEP_SEED = 20260816

DIMENSION_TOL_M = 0.015
CEILING_TOL_M = 0.015
OPENING_TOL_M = 0.08

#: How far wrong a twin has to be before calling it "pass" counts as a lie
#: rather than as ordinary measurement error.
SILENTLY_WRONG_M = 0.05

UNITS = [("m", 1.0), ("cm", 100.0), ("mm", 1000.0), ("in", 39.3701), ("ft", 3.28084)]


@dataclass
class Room:
    label: str
    width: float
    depth: float
    height: float
    unit: str
    result: object
    truth: object

    @property
    def twin(self):
        return self.result.twin

    @property
    def verdict(self) -> str:
        return self.twin.qa.verdict

    @property
    def trusted(self) -> bool:
        """Whether the twin claims to be trustworthy. Only these are held to
        the accuracy targets; the rest have already said not to rely on them."""
        return self.verdict == "pass"

    def describe(self) -> str:
        return (
            f"{self.label} ({self.width:.1f}x{self.depth:.1f}x{self.height:.1f} m "
            f"in {self.unit}, verdict {self.verdict})"
        )


def _recipes(n: int = SWEEP_SIZE, seed: int = SWEEP_SEED) -> list[dict]:
    """Room recipes spanning what a location scout would actually walk into.

    Wider than the catalogue on every axis, and deliberately including rooms
    taller than they are narrow, since that shape is what exposed the gravity
    axis failure that the catalogue could not.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        unit, factor = UNITS[int(rng.integers(0, len(UNITS)))]
        out.append(
            dict(
                label=f"r{i}",
                unit=unit,
                width=float(rng.uniform(2.4, 14.0)),
                depth=float(rng.uniform(2.2, 10.0)),
                height=float(rng.uniform(2.1, 5.5)),
                unit_scale=factor,
                seed=900 + i,
                noise_m=float(rng.uniform(0.003, 0.015)),
                occlusion_patches=int(rng.integers(0, 16)),
                density=float(rng.uniform(300, 1200)),
                tilt_deg=float(rng.uniform(0.0, 10.0)),
                yaw_deg=float(rng.uniform(0.0, 360.0)),
                translate=tuple(float(rng.uniform(-80, 80)) for _ in range(3)),
            )
        )
    return out


_SWEEP: list[Room] | None = None


def sweep() -> list[Room]:
    """Ingest every random room once, through a real PLY, and cache it."""
    global _SWEEP
    if _SWEEP is not None:
        return _SWEEP

    rooms = []
    for recipe in _recipes():
        label, unit = recipe.pop("label"), recipe.pop("unit")
        w, d, h = recipe["width"], recipe["depth"], recipe["height"]
        fx = fixtures.make_room(**recipe)
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / f"{label}.ply"
            write_ply(src, fx.points, None)
            result = ingest(src, IngestOptions(name=label, seed=0))
        rooms.append(
            Room(label=label, width=w, depth=d, height=h, unit=unit, result=result, truth=fx.truth)
        )
    _SWEEP = rooms
    return _SWEEP


@pytest.fixture(scope="session")
def rooms():
    return sweep()


def test_no_room_is_silently_wrong(rooms):
    """The one that must never be relaxed.

    It is acceptable for a hard room to defeat the pipeline. It is not
    acceptable for a hard room to defeat the pipeline quietly, because a twin
    that reports `pass` is one a location manager will build a shoot around.
    """
    silent = []
    for r in rooms:
        got_long, got_short = measure_room(r.twin.points, r.twin.structure).plan
        want_long, want_short = sorted([r.width, r.depth], reverse=True)
        h = r.twin.structure.ceiling_height
        wrong = (
            abs(got_long - want_long) > SILENTLY_WRONG_M
            or abs(got_short - want_short) > SILENTLY_WRONG_M
            or h is None
            or abs(h - r.height) > SILENTLY_WRONG_M
        )
        if wrong and r.trusted:
            silent.append(
                f"{r.describe()}: long {(got_long - want_long) * 1000:+.0f} mm, "
                f"short {(got_short - want_short) * 1000:+.0f} mm, ceiling "
                + ("missing" if h is None else f"{(h - r.height) * 1000:+.0f} mm")
            )
    assert not silent, (
        f"{len(silent)} of {len(rooms)} twins were wrong while reporting 'pass':\n  "
        + "\n  ".join(silent)
    )


def test_most_unseen_rooms_are_trusted(rooms):
    """The pipeline is allowed to refuse a room, but not most of them.

    Without this, everything above could be satisfied by failing every scan,
    which is honest and useless.

    Measured at 15 of 25 on this sweep. Most of the refusals are the ceiling
    plausibility check firing on rooms 5 m and taller, which are correctly
    measured but sit outside the band that catches a misread unit -- see
    CEILING_MAX_M in scan/qa.py for why that trade is deliberate. The bar here
    is set below the measured rate so a real regression trips it, not so it
    flatters the current build.
    """
    trusted = [r for r in rooms if r.trusted]
    assert len(trusted) >= 0.55 * len(rooms), (
        f"only {len(trusted)}/{len(rooms)} rooms were trusted; refused: "
        + ", ".join(r.describe() for r in rooms if not r.trusted)
    )


def test_plan_dimensions_generalize(rooms):
    """The core 1:1 claim, on every room that claims to be trustworthy."""
    failures = []
    for r in rooms:
        if not r.trusted:
            continue
        got_long, got_short = measure_room(r.twin.points, r.twin.structure).plan
        want_long, want_short = sorted([r.width, r.depth], reverse=True)
        el, es = (got_long - want_long) * 1000, (got_short - want_short) * 1000
        if abs(el) > DIMENSION_TOL_M * 1000 or abs(es) > DIMENSION_TOL_M * 1000:
            failures.append(f"{r.describe()}: long {el:+.0f} mm, short {es:+.0f} mm")
    assert not failures, "trusted rooms outside the +/-15 mm target:\n  " + "\n  ".join(failures)


def test_units_generalize(rooms):
    """A room in feet must not come back as a room in metres."""
    wrong = []
    for r in rooms:
        if not r.trusted:
            continue
        got = r.twin.provenance["steps"]["canonicalize"]["unit"]
        if got != r.unit:
            wrong.append(f"{r.describe()}: read {got!r}")
    assert not wrong, "trusted rooms with the wrong unit:\n  " + "\n  ".join(wrong)


def test_ceiling_height_generalizes(rooms):
    failures = []
    for r in rooms:
        if not r.trusted:
            continue
        h = r.twin.structure.ceiling_height
        if h is None:
            failures.append(f"{r.describe()}: no ceiling found in a {r.height:.1f} m room")
        elif abs(h - r.height) > CEILING_TOL_M:
            failures.append(f"{r.describe()}: ceiling {(h - r.height) * 1000:+.0f} mm off")
    assert not failures, "trusted rooms with a wrong ceiling:\n  " + "\n  ".join(failures)


def test_gravity_generalizes(rooms):
    """Floor at z = 0 and level, on rooms tilted up to ten degrees.

    Measured by refitting the floor from the finished twin, independently of
    whatever the aligner reported about its own work.
    """
    from locaish.geom.planes import fit_plane

    failures = []
    for r in rooms:
        if not r.trusted:
            continue
        twin = r.twin
        if abs(twin.structure.floor_z) > 0.01:
            failures.append(f"{r.describe()}: floor at z={twin.structure.floor_z:.3f}")
        z = twin.points.xyz[:, 2]
        low = twin.points.xyz[z < np.percentile(z, 3)]
        n, _ = fit_plane(low)
        tilt = np.degrees(np.arccos(np.clip(abs(n[2]), -1, 1)))
        if tilt > 0.5:
            failures.append(f"{r.describe()}: floor is {tilt:.2f} deg off level")
    assert not failures, "gravity failed on trusted rooms:\n  " + "\n  ".join(failures)


def test_openings_generalize(rooms):
    """Windows and doors, which Phase 3's daylight study consumes directly.

    The fixture only counts openings that actually fit on their wall, since a
    window running off the end of a wall is not an aperture and the pipeline is
    right to refuse it.

    This is the weakest capability in Phase 1 and the bar reflects that rather
    than hiding it. The known residual failure is a sensor shadow sitting
    directly beside a real doorway: it is the size and shape of a small window,
    it has nothing behind it exactly as a window does, and it borrows the
    doorway's reveal as its own evidence. Depth cannot separate those and
    neither can shape, so a small false positive survives roughly one room in
    twenty-five. It is recorded in the README's limitations rather than tuned
    away against this fixture.
    """
    wrong = []
    trusted = [r for r in rooms if r.trusted]
    for r in trusted:
        found, want = r.twin.structure.openings, r.truth.openings
        if len(found) != len(want):
            kinds = ", ".join(f"{o.kind} {o.width:.2f}x{o.height:.2f}" for o in found)
            wrong.append(f"{r.describe()}: found {len(found)} of {len(want)} [{kinds}]")
    assert len(wrong) <= 0.12 * len(trusted), (
        f"opening detection missed on {len(wrong)}/{len(trusted)} trusted rooms, "
        f"past the 12% allowance:\n  " + "\n  ".join(wrong)
    )


def test_opening_sizes_generalize(rooms):
    """Sizes, not just counts -- a window found but measured 300 mm too wide
    would let the daylight study through an aperture that does not exist."""
    bad = []
    for r in rooms:
        if not r.trusted or len(r.twin.structure.openings) != len(r.truth.openings):
            continue
        remaining = list(range(len(r.twin.structure.openings)))
        for spec in r.truth.openings:
            if not remaining:
                break
            i = min(
                remaining,
                key=lambda j: abs(r.twin.structure.openings[j].width - spec.width)
                + abs(r.twin.structure.openings[j].height - spec.height),
            )
            remaining.remove(i)
            o = r.twin.structure.openings[i]
            err = max(abs(o.width - spec.width), abs(o.height - spec.height))
            if err > OPENING_TOL_M:
                bad.append(
                    f"{r.describe()}: {spec.kind} measured {o.width:.2f}x{o.height:.2f} "
                    f"against {spec.width:.2f}x{spec.height:.2f} ({err * 1000:.0f} mm out)"
                )
    assert not bad, "openings mis-measured on trusted rooms:\n  " + "\n  ".join(bad)
