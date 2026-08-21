"""Joining reconstruction windows: the arithmetic, and the case that breaks it.

Each window the network solves comes back in its own arbitrary frame at its own
arbitrary size, so a wrong join does not look wrong -- it looks like a room that
is slightly bent, or a room that is fine except the far end is rolled ten
degrees. Nothing downstream can detect that; the walls are still planar and the
floor is still flat, they are just no longer the same walls and floor.

So the transform is checked against similarities built from known parameters,
and the assertions are on recovering those parameters rather than on the join
looking plausible. The straight-line walk gets its own tests because it is both
the most common capture anyone makes and the one where the textbook method
silently fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from locaish.video import chunks


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _extrinsics(centres, rotations):
    """World-to-camera [R | -R c] for cameras at `centres` oriented by `rotations`."""
    out = np.zeros((len(centres), 3, 4))
    for i, (c, r) in enumerate(zip(centres, rotations)):
        out[i, :, :3] = r
        out[i, :, 3] = -r @ c
    return out


def _pose_pair(centres, rng, scale, rotation, translation):
    """The same cameras written in two frames related by a known similarity.

    Frame b is built *from* frame a by inverting `p_a = s R p_b + t`, so nothing
    in the construction reuses the estimator's own conventions -- a sign error
    in `register` cannot cancel against a matching one here.
    """
    rotations = [_rotation(rng) for _ in centres]
    a = _extrinsics(centres, rotations)
    centres_b = ((rotation.T @ (centres - translation).T) / scale).T
    b = _extrinsics(centres_b, [r @ rotation for r in rotations])
    return a, b, centres_b


def _umeyama_centres_only(src, dst):
    """The textbook similarity from point correspondences, for comparison."""
    src_c, dst_c = src - src.mean(0), dst - dst.mean(0)
    cov = dst_c.T @ src_c / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1, -1] = -1
    rot = u @ d @ vt
    scale = float(np.trace(np.diag(s) @ d) / (src_c**2).sum() * len(src))
    return scale, rot, dst.mean(0) - scale * rot @ src.mean(0)


CURVED = np.column_stack([
    np.linspace(0, 3, 8), np.sin(np.linspace(0, 2, 8)), np.full(8, 1.5)
])
STRAIGHT = np.column_stack([np.linspace(0, 3, 8), np.zeros(8), np.full(8, 1.5)])


# ---------------------------------------------------------------------------
# windowing
# ---------------------------------------------------------------------------


def test_a_short_sweep_is_one_window():
    assert chunks.windows(24, 24, 8) == [(0, 24)]
    assert chunks.windows(5, 24, 8) == [(0, 5)]


def test_windows_cover_every_frame():
    for count in (25, 40, 64, 100, 233):
        spans = chunks.windows(count, 24, 8)
        covered = set()
        for lo, hi in spans:
            covered |= set(range(lo, hi))
        assert covered == set(range(count)), f"{count} frames left gaps"


def test_neighbouring_windows_share_enough_to_register():
    spans = chunks.windows(100, 24, 8)
    for (alo, ahi), (blo, bhi) in zip(spans, spans[1:]):
        shared = len(set(range(alo, ahi)) & set(range(blo, bhi)))
        assert shared >= chunks.MIN_OVERLAP, f"only {shared} shared frames"


def test_the_last_window_is_a_full_window():
    """A short final window reconstructs the end of the sweep with less context."""
    for count in (30, 41, 57, 99):
        spans = chunks.windows(count, 24, 8)
        lo, hi = spans[-1]
        assert hi == count
        assert hi - lo == 24


def test_a_bad_chunk_size_is_an_error():
    with pytest.raises(ValueError):
        chunks.windows(50, 0, 8)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("centres, name", [(CURVED, "curved"), (STRAIGHT, "straight")])
def test_registration_recovers_a_known_similarity(centres, name):
    rng = np.random.default_rng(0)
    scale, rotation = 2.3, _rotation(rng)
    translation = np.array([4.0, -2.0, 0.7])
    a, b, centres_b = _pose_pair(centres, rng, scale, rotation, translation)

    sim = chunks.register(a, b)

    assert sim.scale == pytest.approx(scale, rel=1e-9)
    assert np.allclose(sim.rotation, rotation, atol=1e-9)
    assert np.allclose(sim.translation, translation, atol=1e-9)
    assert np.allclose(sim.apply(centres_b), centres, atol=1e-9)
    assert sim.residual_m < 1e-9


def test_a_straight_walk_defeats_the_textbook_method():
    """Why rotation is averaged from orientations instead of fitted to centres.

    Someone walking a straight line gives collinear camera positions, and a
    similarity fitted to those is free to roll about the walking axis -- the
    correspondences cannot see the difference. The room comes back barrel-rolled
    and every surface in it is still perfectly planar, so nothing downstream
    will ever notice.
    """
    rng = np.random.default_rng(1)
    scale, rotation = 2.3, _rotation(rng)
    translation = np.array([4.0, -2.0, 0.7])
    a, b, centres_b = _pose_pair(STRAIGHT, rng, scale, rotation, translation)

    textbook = _umeyama_centres_only(centres_b, STRAIGHT)
    ours = chunks.register(a, b)

    assert not np.allclose(textbook[1], rotation, atol=1e-3), (
        "the test is vacuous unless the centres-only fit really is ambiguous here"
    )
    assert np.allclose(ours.rotation, rotation, atol=1e-9)


def test_registration_round_trips_the_extrinsics():
    """Transformed poses must equal what the other window solved, not merely look like it."""
    rng = np.random.default_rng(2)
    scale, rotation = 0.4, _rotation(rng)
    translation = np.array([-1.0, 3.0, 2.0])
    a, b, _ = _pose_pair(CURVED, rng, scale, rotation, translation)

    moved = chunks.register(a, b).apply_extrinsics(b)

    assert np.allclose(moved, a, atol=1e-9)


def test_the_extrinsic_translation_carries_the_scale():
    """Dropping the scale on `t` leaves cameras aimed right and placed wrong.

    It is the easiest term in the derivation to lose, and losing it produces
    poses that still look like poses.
    """
    rng = np.random.default_rng(3)
    scale, rotation = 3.0, _rotation(rng)
    translation = np.array([1.0, 1.0, 1.0])
    a, b, _ = _pose_pair(CURVED, rng, scale, rotation, translation)
    sim = chunks.register(a, b)

    moved = sim.apply_extrinsics(b)
    wrong = np.array(moved, copy=True)
    wrong[:, :, 3] /= sim.scale

    assert np.allclose(chunks.camera_centres(moved), CURVED, atol=1e-9)
    assert not np.allclose(chunks.camera_centres(wrong), CURVED, atol=1e-2)


def test_a_direction_is_rotated_but_not_scaled_or_shifted():
    rng = np.random.default_rng(4)
    rotation = _rotation(rng)
    sim = chunks.Similarity(scale=7.0, rotation=rotation, translation=np.array([9.0, 9.0, 9.0]))

    up = np.array([0.0, 0.0, 1.0])
    moved = sim.apply_direction(up)

    assert np.allclose(moved, rotation @ up)
    assert np.linalg.norm(moved) == pytest.approx(1.0)


def test_the_residual_reports_a_join_that_did_not_work():
    """Two windows that disagree about where the shared cameras were must say so."""
    rng = np.random.default_rng(5)
    scale, rotation = 1.0, np.eye(3)
    translation = np.zeros(3)
    a, b, _ = _pose_pair(CURVED, rng, scale, rotation, translation)

    clean = chunks.register(a, b)
    # Shove one window's cameras apart by a quarter of a metre each, at random.
    noisy = np.array(b, copy=True)
    for i in range(len(noisy)):
        noisy[i, :, 3] += rng.normal(0, 0.25, 3)
    broken = chunks.register(a, noisy)

    assert clean.residual_m < 1e-9
    assert broken.residual_m > 0.05, "a scrambled join reported itself as clean"


def test_registration_needs_matching_pairs():
    rng = np.random.default_rng(6)
    a, b, _ = _pose_pair(CURVED, rng, 1.0, np.eye(3), np.zeros(3))

    with pytest.raises(ValueError, match="matching pose pairs"):
        chunks.register(a[:4], b)
    with pytest.raises(ValueError, match="matching pose pairs"):
        chunks.register(a[:1], b[:1])


def test_camera_centres_invert_the_extrinsics():
    rng = np.random.default_rng(7)
    rotations = [_rotation(rng) for _ in CURVED]
    assert np.allclose(chunks.camera_centres(_extrinsics(CURVED, rotations)), CURVED)


def test_joins_compose_without_drifting_on_exact_input():
    """Three windows chained: the far end must land where it belongs.

    Error accumulates along the chain by design -- there is no bundle adjustment
    here -- so the floor for that accumulation has to be zero when the input is
    exact, or every real sweep starts with a handicap.
    """
    rng = np.random.default_rng(8)
    rotations = [_rotation(rng) for _ in CURVED]
    a = _extrinsics(CURVED, rotations)

    current = a
    composed = CURVED
    for scale, shift in ((1.7, 2.0), (0.6, -3.0)):
        rotation = _rotation(rng)
        translation = np.array([shift, 0.5 * shift, 0.25 * shift])
        centres = chunks.camera_centres(current)
        nxt_centres = ((rotation.T @ (centres - translation).T) / scale).T
        nxt = _extrinsics(nxt_centres, [c[:, :3] @ rotation for c in current])
        sim = chunks.register(current, nxt)
        composed = sim.apply(nxt_centres)
        assert np.allclose(composed, centres, atol=1e-8)
        current = sim.apply_extrinsics(nxt)

    assert np.allclose(chunks.camera_centres(current), CURVED, atol=1e-8)
