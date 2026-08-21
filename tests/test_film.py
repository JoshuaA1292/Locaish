"""The film layer: lens geometry, what fits, what can be seen, and what rings.

The danger in this package is different from the rest of Locaish. Phase 1 can be
checked against a fixture whose dimensions are known exactly; here the outputs
are recommendations -- a focal length, a dolly position, a reverberation time --
and a recommendation that is confidently wrong looks exactly like one that is
right. A scout who drives out because the twin said the dolly fits has been
failed by a number nobody could see was wrong.

So the optics are checked against closed-form values computed independently in
the test, the spatial answers against rooms built with obstructions in known
places, and the acoustics against Sabine's equation worked by hand. Where the
honest answer is a refusal -- no ceiling, no floor seen, gear that does not fit
-- that refusal is asserted too.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from locaish.film import acoustics, equipment, moves, optics, report, space
from locaish.types import Opening, Plane, PointCloud, QAReport, Structure, Twin


# ---------------------------------------------------------------------------
# optics
# ---------------------------------------------------------------------------


def test_field_of_view_matches_the_closed_form():
    s = optics.SENSORS["super35"]
    h, v = optics.field_of_view_deg(s, 32.0)

    assert h == pytest.approx(2 * math.degrees(math.atan(24.89 / 64.0)), abs=1e-9)
    assert v == pytest.approx(2 * math.degrees(math.atan(14.00 / 64.0)), abs=1e-9)


def test_a_wider_lens_sees_more():
    s = optics.SENSORS["super35"]
    assert optics.field_of_view_deg(s, 18)[0] > optics.field_of_view_deg(s, 50)[0]


def test_framing_and_distance_are_inverses():
    """The two questions a recce asks are the same equation read both ways."""
    s = optics.SENSORS["fullframe"]
    for focal in (18, 35, 85):
        for distance in (1.2, 4.0, 11.0):
            framed = optics.framed_height_m(s, focal, distance)
            assert optics.distance_for_framing(s, focal, framed) == pytest.approx(distance)
            assert optics.focal_for_framing(s, distance, framed) == pytest.approx(focal)


def test_framing_scales_the_way_similar_triangles_do():
    s = optics.SENSORS["super35"]
    near = optics.framed_height_m(s, 50, 2.0)
    far = optics.framed_height_m(s, 50, 6.0)
    assert far == pytest.approx(3.0 * near)

    long_lens = optics.framed_height_m(s, 100, 2.0)
    assert long_lens == pytest.approx(near / 2.0)


def test_shot_sizes_are_named_by_what_fills_the_frame():
    assert optics.classify_framing(0.60).key == "cu"
    assert optics.classify_framing(1.20).key == "ms"
    assert optics.classify_framing(2.10).key == "ls"


def test_shot_classification_is_multiplicative_not_linear():
    """0.6 and 0.9 are as far apart as 2.1 and 3.2; a linear match says otherwise.

    Halfway between a close-up and a medium close-up in *log* space is 0.735 m.
    A linear nearest-neighbour would put the boundary at 0.75 m, so a framing of
    0.74 m is the case that separates the two rules.
    """
    boundary = math.sqrt(0.60 * 0.90)
    assert optics.classify_framing(boundary * 0.98).key == "cu"
    assert optics.classify_framing(boundary * 1.02).key == "mcu"


def test_depth_of_field_brackets_the_focus_distance():
    s = optics.SENSORS["super35"]
    near, far, hyper = optics.depth_of_field(s, 32, 2.8, 3.0)
    assert near < 3.0 < far
    assert hyper > 3.0, "3 m should be inside the hyperfocal for a 32mm at f/2.8"


def test_stopping_down_deepens_the_focus():
    s = optics.SENSORS["super35"]
    wide = optics.depth_of_field(s, 50, 1.4, 3.0)
    stopped = optics.depth_of_field(s, 50, 8.0, 3.0)
    assert (stopped[1] - stopped[0]) > (wide[1] - wide[0])


def test_focus_runs_to_infinity_past_the_hyperfocal():
    s = optics.SENSORS["super35"]
    _, _, hyper = optics.depth_of_field(s, 18, 5.6, 5.0)
    _, far, _ = optics.depth_of_field(s, 18, 5.6, hyper * 1.2)
    assert far == float("inf")


def test_primes_are_chosen_in_log_space():
    """Lens sets are spaced multiplicatively, so the nearest one is too.

    117 mm is the case that separates the rules: it is 17 mm above the 100 and
    18 mm below the 135, so a linear match picks the 100 -- but it is a smaller
    *ratio* from the 135, which is what actually governs how different two
    lenses look.
    """
    assert optics.nearest_prime(117) == 135
    assert abs(117 - 100) < abs(117 - 135), "the test is vacuous unless linear disagrees"

    assert optics.nearest_prime(26) == 27
    assert optics.nearest_prime(47) == 50
    assert optics.nearest_prime(1000) == 135, "past the set it saturates, and callers must notice"


def test_bad_optics_inputs_are_rejected():
    s = optics.SENSORS["super35"]
    for bad in (0, -5):
        with pytest.raises(ValueError):
            optics.framed_height_m(s, bad, 3.0)
        with pytest.raises(ValueError):
            optics.framed_height_m(s, 32, bad)


# ---------------------------------------------------------------------------
# equipment
# ---------------------------------------------------------------------------


def test_unverified_gear_is_flagged_as_such():
    """A class-typical dimension must never be indistinguishable from a published one."""
    unverified = {g.key for g in equipment.unverified()}
    assert "fisher-11" in unverified, "its length is an estimate and must say so"
    assert equipment.get("super-peewee").verified, "its dimensions are published"
    assert all(g.source for g in equipment.CATALOGUE.values()), "every entry needs a provenance"


def test_swept_footprint_includes_room_to_work():
    g = equipment.get("super-peewee")
    swept = g.swept_footprint_m
    assert swept[0] > g.footprint_m[0] and swept[1] > g.footprint_m[1]
    assert swept[0] == pytest.approx(g.footprint_m[0] + 2 * g.clearance_m)


def test_an_unknown_key_is_an_error_not_a_default():
    with pytest.raises(KeyError):
        equipment.get("imaginary-crane")


# ---------------------------------------------------------------------------
# a room to survey
# ---------------------------------------------------------------------------


def _box_twin(w=6.0, d=4.0, h=2.7, *, obstacle=None, ceiling=True, n=60_000, name="room"):
    """A rectangular room, optionally with a solid block standing in it."""
    rng = np.random.default_rng(0)
    parts = [
        np.column_stack([rng.uniform(0, w, n), rng.uniform(0, d, n), np.zeros(n)]),
        np.column_stack([rng.uniform(0, w, n // 2), np.zeros(n // 2), rng.uniform(0, h, n // 2)]),
        np.column_stack([rng.uniform(0, w, n // 2), np.full(n // 2, d), rng.uniform(0, h, n // 2)]),
        np.column_stack([np.zeros(n // 2), rng.uniform(0, d, n // 2), rng.uniform(0, h, n // 2)]),
        np.column_stack([np.full(n // 2, w), rng.uniform(0, d, n // 2), rng.uniform(0, h, n // 2)]),
    ]
    if ceiling:
        parts.append(
            np.column_stack([rng.uniform(0, w, n), rng.uniform(0, d, n), np.full(n, h)])
        )
    if obstacle is not None:
        (x0, x1, y0, y1, z1) = obstacle
        k = 30_000
        parts.append(np.column_stack([
            rng.uniform(x0, x1, k), rng.uniform(y0, y1, k), rng.uniform(0, z1, k)
        ]))
    xyz = np.concatenate(parts)

    footprint = np.array([[0, 0], [w, 0], [w, d], [0, d]], dtype=float)
    twin = Twin(
        name=name,
        points=PointCloud(xyz),
        structure=Structure(
            floor_z=0.0,
            ceiling_z=h if ceiling else None,
            footprint=footprint,
            planes=[Plane(normal=[0, 0, 1], offset=0.0, kind="floor", area=w * d)],
        ),
        qa=QAReport(verdict="pass"),
    )
    return twin


# ---------------------------------------------------------------------------
# space
# ---------------------------------------------------------------------------


def test_floor_maps_find_the_room():
    twin = _box_twin()
    maps = space.floor_maps(twin, cell=0.1)

    area = float(maps.inside.sum()) * maps.cell**2
    assert area == pytest.approx(24.0, rel=0.05), "6 x 4 m of floor"
    assert np.nanmedian(maps.headroom_m[maps.inside]) == pytest.approx(2.7, abs=0.1)


def test_an_obstacle_eats_the_clearance_around_it():
    """A block in the middle must be visible in the clearance map, not just the cloud."""
    clear = space.floor_maps(_box_twin(), cell=0.1)
    blocked = space.floor_maps(
        _box_twin(obstacle=(2.5, 3.5, 1.5, 2.5, 1.0)), cell=0.1
    )

    at_block = np.array([[3.0, 2.0]])
    assert clear.sample(clear.clearance_m, at_block)[0] > 1.0
    assert blocked.sample(blocked.clearance_m, at_block)[0] < 0.2


def test_a_low_obstacle_blocks_a_dolly_and_not_a_head():
    """Headroom and clearance ask different questions and must not be conflated.

    A table blocks the floor and leaves the air above it free. A pipe under the
    ceiling does the reverse. A single "is this cell free" map cannot express
    either, which is why there are two.
    """
    maps = space.floor_maps(_box_twin(obstacle=(2.5, 3.5, 1.5, 2.5, 1.0)), cell=0.1)
    at_block = np.array([[3.0, 2.0]])

    assert maps.sample(maps.clearance_m, at_block)[0] < 0.2, "the dolly is blocked"
    assert maps.sample(maps.headroom_m, at_block)[0] > 2.0, "a head is not"


def test_standable_floor_excludes_the_obstacle():
    maps = space.floor_maps(_box_twin(obstacle=(2.5, 3.5, 1.5, 2.5, 1.0)), cell=0.1)
    stand = maps.standable()
    assert not maps.sample(stand, np.array([[3.0, 2.0]]))[0]
    assert maps.sample(stand, np.array([[1.0, 2.0]]))[0]


def test_gear_only_fits_where_there_is_room_for_it():
    maps = space.floor_maps(_box_twin(), cell=0.1)
    small = space.fits_mask(maps, (0.5, 0.5))
    large = space.fits_mask(maps, (3.0, 3.0))
    assert small.sum() > large.sum(), "a bigger footprint must fit in fewer places"

    huge = space.fits_mask(maps, (20.0, 20.0))
    assert huge.sum() == 0, "gear larger than the room must fit nowhere"


def test_placements_prefer_the_most_room():
    maps = space.floor_maps(_box_twin(), cell=0.1)
    spots = space.placements(maps, (0.6, 1.0), limit=50)
    assert spots
    clearances = [s.clearance_m for s in spots]
    assert clearances == sorted(clearances, reverse=True)


# ---------------------------------------------------------------------------
# sightlines
# ---------------------------------------------------------------------------


def test_a_clear_line_is_clear_and_a_blocked_one_is_not():
    """The whole sightline machinery in one assertion pair.

    Same two endpoints, same room, one with a pillar between them. If this does
    not discriminate, every eyeline answer in the report is decoration.
    """
    empty = space.occupancy(_box_twin(), cell=0.08)
    walled = space.occupancy(_box_twin(obstacle=(2.9, 3.1, 0.0, 4.0, 2.5)), cell=0.08)

    a, b = np.array([1.0, 2.0, 1.2]), np.array([5.0, 2.0, 1.2])
    assert space.visible(*empty, a, b)
    assert not space.visible(*walled, a, b)


def test_a_sightline_is_not_blocked_by_its_own_endpoints():
    """Both ends sit on something -- a camera on sticks, a head on a body."""
    occ = space.occupancy(_box_twin(obstacle=(1.0, 1.4, 1.8, 2.2, 1.6)), cell=0.08)
    # From just beside the block to across the room.
    a = np.array([1.2, 2.0, 1.55])
    b = np.array([5.0, 2.0, 1.55])
    assert space.visible(occ[0], occ[1], occ[2], a, b, slack=0.35)


# ---------------------------------------------------------------------------
# moves
# ---------------------------------------------------------------------------


def _rig(twin, cell=0.1):
    return space.floor_maps(twin, cell=cell), space.occupancy(twin, cell=0.08)


def test_a_push_in_tightens_the_framing():
    """The defining property of a push-in, asserted rather than assumed."""
    twin = _box_twin()
    maps, occ = _rig(twin)
    result = moves.simulate(
        maps, occ,
        moves.straight([1.0, 2.0, 1.1], [4.0, 2.0, 1.1]),
        subject_path=np.array([[5.2, 2.0, 0.0]]),
        focal_mm=32,
    )

    assert result.beats[0].distance_m > result.beats[-1].distance_m
    assert result.beats[0].framed_height_m > result.beats[-1].framed_height_m
    assert all(b.subject_in_frame for b in result.beats)


def test_the_camera_reports_where_in_frame_the_subject_sits():
    twin = _box_twin()
    maps, occ = _rig(twin)
    subject = np.array([[3.0, 2.0, 0.0]])

    centred = moves.simulate(
        maps, occ, moves.straight([1.0, 2.0, 1.1], [1.5, 2.0, 1.1]),
        subject_path=subject, focal_mm=32,
    )
    assert abs(centred.beats[0].frame_uv[0]) < 0.05, "aiming at the subject centres it"

    # Aim at a fixed point well to one side and the subject must slide off centre.
    offset = moves.simulate(
        maps, occ, moves.straight([1.0, 2.0, 1.1], [1.5, 2.0, 1.1]),
        subject_path=subject, focal_mm=32, aim_at=np.array([3.0, 3.6, 1.6]),
    )
    assert abs(offset.beats[0].frame_uv[0]) > abs(centred.beats[0].frame_uv[0])


def test_a_move_that_loses_the_subject_says_so():
    twin = _box_twin()
    maps, occ = _rig(twin)
    result = moves.simulate(
        maps, occ, moves.straight([1.0, 2.0, 1.1], [2.0, 2.0, 1.1]),
        subject_path=np.array([[3.0, 2.0, 0.0]]),
        focal_mm=32, aim_at=np.array([1.0, 0.1, 1.1]),
    )
    assert not result.feasible
    assert any("outside the frame" in p for b in result.beats for p in b.problems)


def test_a_move_behind_a_pillar_reports_the_blocked_beats():
    """The case a storyboard cannot catch: clean at both ends, blocked in the middle."""
    twin = _box_twin(obstacle=(2.9, 3.1, 1.7, 2.3, 2.4))
    maps, occ = _rig(twin)
    result = moves.simulate(
        maps, occ,
        moves.straight([1.0, 2.0, 1.1], [2.2, 2.0, 1.1]),
        subject_path=np.array([[5.2, 2.0, 0.0]]),
        focal_mm=32,
    )
    assert not result.feasible
    assert any("between the camera and the subject" in p
               for b in result.beats for p in b.problems)


def test_gear_too_big_for_the_position_is_refused():
    twin = _box_twin()
    maps, occ = _rig(twin)
    # Hard against the wall, where a dolly cannot go.
    result = moves.simulate(
        maps, occ, moves.straight([0.15, 2.0, 1.1], [0.4, 2.0, 1.1]),
        subject_path=np.array([[5.0, 2.0, 0.0]]),
        gear=equipment.get("super-peewee"),
    )
    assert not result.feasible
    assert any("clear radius" in p for b in result.beats for p in b.problems)


def test_an_uneven_floor_fails_the_track_check():
    """Track is wedged level; a floor that wanders needs cribbing, and that is time."""
    rng = np.random.default_rng(1)
    twin = _box_twin()
    # Raise a strip of floor by 8 cm, well past the 20 mm a wedge covers.
    xyz = twin.points.xyz
    lift = (xyz[:, 2] < 0.05) & (xyz[:, 0] > 3.0)
    xyz[lift, 2] += 0.08
    twin.points = PointCloud(xyz)

    maps, occ = _rig(twin)
    result = moves.simulate(
        maps, occ, moves.straight([2.0, 2.0, 1.1], [4.5, 2.0, 1.1]),
        subject_path=np.array([[5.5, 2.0, 0.0]]), on_track=True,
    )
    assert result.track_level_range_m > 0.05
    assert not result.feasible
    assert any("cribbing" in p for b in result.beats for p in b.problems)


def test_a_level_floor_passes_the_track_check():
    twin = _box_twin()
    maps, occ = _rig(twin)
    result = moves.simulate(
        maps, occ, moves.straight([2.0, 2.0, 1.1], [4.5, 2.0, 1.1]),
        subject_path=np.array([[5.5, 2.0, 0.0]]), on_track=True,
    )
    assert result.track_level_range_m < 0.02
    assert result.feasible, [p for b in result.beats for p in b.problems]


def test_camera_and_subject_paths_are_walked_in_step():
    """Beat i of the camera must be the same instant as beat i of the actor.

    Sampled independently they have different lengths, and zipping them without
    resampling silently compares the start of the move against the end of the
    walk -- which produces a plausible report of a shot that was never simulated.
    """
    twin = _box_twin()
    maps, occ = _rig(twin)
    camera = moves.straight([1.0, 1.0, 1.1], [1.0, 3.0, 1.1])
    walk = moves.through([[4.0, 1.0, 0.0], [4.0, 3.0, 0.0]])

    result = moves.simulate(maps, occ, camera, subject_path=walk, focal_mm=32)

    assert len(result.beats) == len(camera)
    # Both travel the same way at the same rate, so the distance holds steady.
    spread = max(b.distance_m for b in result.beats) - min(b.distance_m for b in result.beats)
    assert spread < 0.2, f"distance wandered by {spread:.2f} m in a parallel track"


def test_a_move_needs_somewhere_to_go():
    twin = _box_twin()
    maps, occ = _rig(twin)
    with pytest.raises(ValueError):
        moves.simulate(maps, occ, np.array([[1.0, 1.0, 1.1]]))


# ---------------------------------------------------------------------------
# acoustics
# ---------------------------------------------------------------------------


def test_sabine_matches_the_hand_calculation():
    """RT60 = 0.161 V / A, worked independently for the hardest surfaces."""
    twin = _box_twin(w=6.0, d=4.0, h=2.7)
    est = acoustics.estimate(twin)

    volume = 6.0 * 4.0 * 2.7
    walls = 2 * (6.0 + 4.0) * 2.7
    absorption = 24.0 * 0.02 + walls * 0.02 + 24.0 * 0.02      # floor, wall, ceiling
    expected = 0.161 * volume / absorption

    assert est.volume_m3 == pytest.approx(volume, rel=0.02)
    assert est.rt60_hardest_s == pytest.approx(expected, rel=0.05)


def test_softer_finishes_give_a_shorter_tail():
    est = acoustics.estimate(_box_twin())
    assert est.rt60_softest_s < est.rt60_typical_s < est.rt60_hardest_s


def test_a_bigger_room_rings_longer():
    small = acoustics.estimate(_box_twin(w=4.0, d=3.0, h=2.4))
    large = acoustics.estimate(_box_twin(w=12.0, d=9.0, h=2.4))
    assert large.rt60_typical_s > small.rt60_typical_s


def test_the_verdict_is_judged_on_the_hard_end():
    """A room assumed carpeted and found tiled is a problem found on the day."""
    est = acoustics.estimate(_box_twin(w=14.0, d=10.0, h=4.0))
    assert est.verdict in {"difficult", "problem", "depends on the finishes"}
    assert est.rt60_hardest_s > acoustics.RT60_CLEAN_DIALOGUE_S


def test_a_missing_ceiling_is_declared_not_assumed():
    est = acoustics.estimate(_box_twin(ceiling=False))
    assert any("no ceiling" in w for w in est.warnings)


def test_openings_absorb_everything_that_reaches_them():
    twin = _box_twin()
    sealed = acoustics.estimate(twin)
    twin.structure.openings = [
        Opening(center=[0.0, 2.0, 1.0], width=1.2, height=2.1,
                normal=[1, 0, 0], sill_height=0.0, kind="door", confidence=0.9)
    ]
    opened = acoustics.estimate(twin)
    assert opened.rt60_hardest_s < sealed.rt60_hardest_s


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------


def test_the_report_covers_every_department():
    built = report.build(_box_twin(name="Unit 4, ground floor"))
    d = built.to_dict()
    for section in ("trust", "space", "camera", "grip", "sound"):
        assert d[section], f"{section} came back empty"
    assert d["location"] == "Unit 4, ground floor"
    assert d["space"]["ceiling_height_m"] == pytest.approx(2.7, abs=0.05)


def test_the_report_refuses_to_answer_overhead_questions_without_a_ceiling():
    built = report.build(_box_twin(ceiling=False))
    assert built.space["ceiling_height_m"] is None
    assert any("no ceiling" in c for c in built.caveats)


def test_the_report_names_gear_whose_dimensions_are_estimates():
    built = report.build(_box_twin())
    assert built.grip["unverified_dimensions"]
    assert "note" in built.grip


def test_a_long_lens_is_reported_as_beyond_the_set_rather_than_clamped():
    """Two different shots must not both come back as the longest prime.

    Saturating at 135 mm made a big close-up and a close-up read identically,
    which is a recommendation that is wrong about both.
    """
    built = report.build(_box_twin(w=6.0, d=4.0))
    beyond = [s for s in built.camera["shots"] if s["at_max_distance"]["beyond_prime_set"]]
    assert beyond, "a tight shot from across the room needs more than a 135"
    for s in beyond:
        assert s["at_max_distance"]["nearest_prime_mm"] is None
        assert s["at_max_distance"]["focal_mm"] > 135


def test_the_report_renders_to_text():
    text = report.render_text(report.build(_box_twin(name="Location A")))
    for heading in ("LOCATION: Location A", "SPACE", "CAMERA", "GRIP", "SOUND"):
        assert heading in text
