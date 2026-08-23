"""Recovering the one number a monocular reconstruction cannot know: metres.

Structure from a moving camera is fixed only up to a similarity transform. The
shape of the room, the angles between its walls, the ratio of its length to its
height -- all of that is recoverable from parallax alone. Its *size* is not. A
kitchen and a scale model of a kitchen produce literally identical video, so no
amount of better reconstruction will ever tell them apart. This is not a
limitation of the network; it is a property of the problem.

Locaish cannot ship a twin with an unknown scale, because every downstream
claim -- will the dolly fit, how long is the light on that wall -- is a
statement in metres. So the scale has to come from outside the geometry, and
this module gets it from two physical anchors that have nothing to do with
each other:

**How high the phone was.** A person filming a room holds it somewhere around
chest height. Once gravity is known -- and the camera poses give that away --
the distance from the camera path down to the floor is a length in the
reconstruction's arbitrary units whose value in metres we already know to
within a few centimetres.

**Any doorway in the room.** A door leaf is built to a standard almost
everywhere on earth, which makes it the tightest anchor available in a room
nobody measured. It is recognised by shape rather than size -- see
`scale_from_doors` -- and runs as a second pass, because apertures are found
in the canonical frame and the canonical frame needs a scale to exist.

Neither is a tape measure. The point of having both is that they fail
*differently*: camera height is wrong only if the operator was crouching or
holding the phone overhead, while a door anchor is wrong only if the aperture
it locked onto was not actually a door. When two estimates built from
unrelated evidence agree, that agreement is real information; when they
disagree, the honest response is a wider error bar, not a choice.
`combine_scales` does exactly that, and the spread it reports is what becomes
the twin's `scale_confidence`.

The failure this module exists to prevent is subtle and worth naming: a scale
solved from one estimator, checked against itself, reports beautiful internal
agreement and can still be off by a factor of two. Self-consistency measures
precision. Only a second, independent estimator measures anything like
accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# What a hand-held phone rides at while someone walks a room, and how much that
# varies between people and postures. Roughly chest to eye height: the operator
# is looking at the screen, not through a viewfinder.
CAMERA_HEIGHT_M = 1.55
CAMERA_HEIGHT_LOG_BIAS = math.log(1.12)

# A door leaf is one of the few dimensions in a building that is effectively
# standardised: 1981 mm in the UK, 2032 mm across Europe and North America, and
# the structural opening a few centimetres above the leaf. The spread that
# matters is not the door's though -- it is the error in measuring an aperture
# that was inferred from missing geometry in the first place, which is a few
# centimetres on a good detection. Eight percent covers both.
DOOR_HEIGHT_M = 2.03
DOOR_LOG_BIAS = math.log(1.08)

# What counts as door-shaped, in ratios only. Using metres here would be
# circular: the whole point of this estimator is that we do not yet know what a
# metre is, and an aperture classified as a door by a height threshold has
# already assumed the answer. A door is instead recognised by sitting on the
# floor and being between one and a half and three and a half times as tall as
# it is wide -- true of every door ever hung, and false of windows, hatches and
# the long low gaps that reconstruction noise leaves along a skirting board.
DOOR_MAX_SILL_RATIO = 0.08
DOOR_MIN_ASPECT = 1.5
DOOR_MAX_ASPECT = 3.5
DOOR_MIN_CONFIDENCE = 0.3

# Sanity envelope on the camera-height estimator. Outside this the recovered
# height is not a person holding a phone -- it is a bad floor estimate, or a
# capture where the floor was never seen.
MIN_CAMERA_HEIGHT_FRACTION = 0.15
MAX_CAMERA_HEIGHT_FRACTION = 0.85


@dataclass
class ScaleEstimate:
    """A metres-per-reconstruction-unit factor, with its own error bars."""

    factor: float
    confidence: float
    log_spread: float
    source: str = "unspecified"
    prior_bias: float = 0.0
    per_frame: list[float] = field(default_factory=list)
    frames_used: int = 0
    frames_rejected: int = 0
    model: str = ""
    components: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def relative_error(self) -> float:
        """One-sigma fractional uncertainty on the factor, e.g. 0.08 for +/-8%."""
        return float(np.expm1(self.log_spread))

    @property
    def total_log_sigma(self) -> float:
        """Measured spread and estimator bias together -- what a combiner needs."""
        return float(math.hypot(self.log_spread, self.prior_bias))

    def to_dict(self) -> dict:
        d = {
            "factor": self.factor,
            "confidence": self.confidence,
            "log_spread": self.log_spread,
            "relative_error": self.relative_error,
            "source": self.source,
            "frames_used": self.frames_used,
            "frames_rejected": self.frames_rejected,
            "model": self.model,
        }
        if self.components:
            d["components"] = self.components
        return d


def scale_from_camera_height(
    points: np.ndarray,
    cameras: np.ndarray,
    up: np.ndarray | None,
) -> ScaleEstimate | None:
    """Find the factor from how far the phone rode above the floor.

    Independent of the depth network in every respect that matters: it uses the
    camera trajectory and the floor, not appearance, and its prior comes from
    human anatomy rather than a training set. That independence is the entire
    reason it exists -- two estimators sharing a failure mode would agree
    beautifully and mean nothing.

    Returns None rather than a guess when the evidence is not there: no gravity,
    no camera path, or a recovered height that is not a plausible fraction of
    the room's own vertical extent, which is what happens when the floor was
    never actually captured and the lowest points are something else entirely.
    """
    if up is None or cameras is None or len(cameras) < 2 or len(points) < 1000:
        return None
    up = np.asarray(up, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(up))
    if norm < 1e-9:
        return None
    up = up / norm

    t_pts = np.asarray(points, dtype=np.float64) @ up
    t_cam = np.asarray(cameras, dtype=np.float64).reshape(-1, 3) @ up
    if not (np.isfinite(t_pts).all() and np.isfinite(t_cam).all()):
        t_pts = t_pts[np.isfinite(t_pts)]
        t_cam = t_cam[np.isfinite(t_cam)]
        if len(t_pts) < 1000 or len(t_cam) < 2:
            return None

    # A low percentile rather than the minimum: reconstruction floaters live
    # below the floor, and one of them would otherwise set the datum.
    floor = float(np.percentile(t_pts, 1.0))
    top = float(np.percentile(t_pts, 99.5))
    span = top - floor
    height = float(np.median(t_cam) - floor)
    if span <= 0 or height <= 0:
        return None

    fraction = height / span
    warnings: list[str] = []
    if not (MIN_CAMERA_HEIGHT_FRACTION <= fraction <= MAX_CAMERA_HEIGHT_FRACTION):
        return None
    if fraction > 0.7:
        warnings.append(
            "the camera path sits high in the captured volume, which usually "
            "means the floor was not properly captured rather than that the "
            "operator was tall; the camera-height scale estimate is weak here"
        )

    return ScaleEstimate(
        factor=CAMERA_HEIGHT_M / height,
        confidence=_confidence(CAMERA_HEIGHT_LOG_BIAS),
        log_spread=0.0,
        source="camera-height",
        prior_bias=CAMERA_HEIGHT_LOG_BIAS,
        frames_used=len(t_cam),
        warnings=warnings,
    )


def scale_from_doors(openings, current_factor: float) -> ScaleEstimate | None:
    """Find the factor from the height of any doorway in the room.

    The third estimator, and the sharpest of the three when it fires. Camera
    height is a prior on human anatomy and the depth network is a prior on how
    rooms tend to look; a door is a manufactured object built to a standard, so
    the same aperture in any building on the continent is the same height to
    within a few centimetres.

    It has to be plumbed as a second pass, because openings are found in the
    canonical frame and the canonical frame needs a scale to exist -- so this
    reads apertures that were measured with the *provisional* scale and returns
    the corrected one. `current_factor` is the metres-per-reconstruction-unit
    that produced those measurements, and the result is that factor multiplied
    by however wrong the doors say it was.

    Returns None rather than a guess when nothing door-shaped was found. A room
    with no door in shot is ordinary, and inventing an anchor there would be
    worse than having two estimators instead of three.
    """
    candidates = []
    for o in openings or []:
        height = float(getattr(o, "height", 0.0))
        width = float(getattr(o, "width", 0.0))
        sill = float(getattr(o, "sill_height", 0.0))
        conf = float(getattr(o, "confidence", 0.0))
        if height <= 0 or width <= 0 or conf < DOOR_MIN_CONFIDENCE:
            continue
        if sill > DOOR_MAX_SILL_RATIO * height:
            continue
        aspect = height / width
        if not (DOOR_MIN_ASPECT <= aspect <= DOOR_MAX_ASPECT):
            continue
        candidates.append(height)

    if not candidates:
        return None

    heights = np.array(candidates, dtype=np.float64)
    # The median across doors, not the tallest: a pass-through with no door in
    # it is taller than a door and would drag the scale down every time.
    implied = DOOR_HEIGHT_M / np.median(heights)
    spread = 0.0
    if len(heights) > 1:
        logs = np.log(DOOR_HEIGHT_M / heights)
        spread = float(np.median(np.abs(logs - np.median(logs))) * 1.4826)

    return ScaleEstimate(
        factor=float(current_factor * implied),
        confidence=_confidence(math.hypot(spread, DOOR_LOG_BIAS)),
        log_spread=spread,
        source="door-height",
        prior_bias=DOOR_LOG_BIAS,
        per_frame=[float(h) for h in heights],
        frames_used=len(heights),
        warnings=[],
    )


def combine_scales(estimates: list[ScaleEstimate]) -> ScaleEstimate:
    """Merge independent scale estimates, widening the error bar when they argue.

    Inverse-variance weighting in log space, because scale is multiplicative: an
    estimator that is twice too big and one that is half too big should average
    to right, not to 1.25x.

    The part that matters is the last step. If the estimates disagree by more
    than their stated uncertainties allow, then at least one of those
    uncertainties is understated, and the combined error is inflated by the
    square root of the reduced chi-square until it is consistent with the
    scatter actually observed. This is the standard treatment of discrepant
    measurements, and it is the mechanism that stops two confident estimators
    from producing one confident wrong answer: disagreement can only ever widen
    the bar, never narrow it.
    """
    estimates = [e for e in estimates if e is not None and e.factor > 0]
    if not estimates:
        raise RuntimeError("no scale estimate survived; the twin has no size")
    if len(estimates) == 1:
        only = estimates[0]
        sigma = only.total_log_sigma
        return ScaleEstimate(
            factor=only.factor,
            confidence=_confidence(sigma),
            log_spread=sigma,
            source=only.source,
            prior_bias=only.prior_bias,
            per_frame=only.per_frame,
            frames_used=only.frames_used,
            frames_rejected=only.frames_rejected,
            model=only.model,
            components=[only.to_dict()],
            warnings=list(only.warnings)
            + [
                f"scale rests on a single estimator ({only.source}); with nothing "
                "to cross-check it against, its error bar is a prior rather than "
                "a measurement"
            ],
        )

    logs = np.array([math.log(e.factor) for e in estimates])
    sigmas = np.array([max(e.total_log_sigma, 1e-6) for e in estimates])
    weights = 1.0 / sigmas**2
    mu = float((weights * logs).sum() / weights.sum())
    sigma = float(1.0 / math.sqrt(weights.sum()))

    chi2 = float((weights * (logs - mu) ** 2).sum())
    dof = len(estimates) - 1
    inflation = max(1.0, math.sqrt(chi2 / dof))
    sigma *= inflation

    warnings: list[str] = []
    for e in estimates:
        warnings += e.warnings
    if inflation > 1.5:
        pairs = ", ".join(f"{e.source} {e.factor:.4g}" for e in estimates)
        warnings.append(
            f"the scale estimators disagree by more than their own error bars "
            f"allow ({pairs}), so the combined uncertainty was widened "
            f"{inflation:.1f}x to cover the disagreement; this twin's size is a "
            "range, not a measurement, and one length checked with a tape and "
            "passed as --scale-factor would settle it"
        )

    rel = float(np.expm1(sigma))
    if rel > 0.15:
        warnings.append(
            f"scale is uncertain to about +/-{rel:.0%}; every distance in this "
            "twin carries that error"
        )

    return ScaleEstimate(
        factor=float(math.exp(mu)),
        confidence=_confidence(sigma),
        log_spread=sigma,
        source="combined",
        prior_bias=0.0,
        frames_used=max(e.frames_used for e in estimates),
        model=next((e.model for e in estimates if e.model), ""),
        components=[e.to_dict() for e in estimates],
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _confidence(log_sigma: float) -> float:
    """Map a log-space uncertainty onto the 0-1 confidence the pipeline speaks.

    The 0.30 is the point at which a scale is worthless rather than merely
    uncertain: a room reported +/-35% could be a bedroom or a hall, and no
    decision downstream survives that.
    """
    rel = float(np.expm1(max(log_sigma, 0.0)))
    return float(np.clip(1.0 - rel / 0.30, 0.05, 0.95))
