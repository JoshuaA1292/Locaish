"""The cast: who stands on the marks, and how tall they really are.

A film plan that assumes every actor is 1.75 m gets the eyeline, the headroom
and the over-the-shoulder wrong by exactly the difference nobody typed in. An
`Actor` here is a name, a real height, and optionally a body: either the
proportioned capsule this module builds from the height alone, or a mesh from
SAM 3D Body that another module produced from a headshot. Either way the typed
height is the truth. A body mesh is scaled to it, never the other way round,
and the model's own height estimate is only reported as a cross-check.

Actors feed three places. The sweep takes `actor.height_m` as its
`subject_height_m`, so framing and eyeline are measured against this person.
The renderer draws `figure(actor, mark, facing, floor_z)` as a `Figure` of
point gaussians standing on the mark. The planner's notes read
`pair_eyelines(a, b)` for the apple-box arithmetic of a two-shot.

The cast is a `cast.json` in the job's workdir, written atomically like the
rest of the product store. Nothing in here enters a measurement of the room:
a body is drawn into frames and used for framing arithmetic, and that is all.

Body npz contract (consumed only; produced by the SAM 3D Body module):

    vertices  (V, 3) float   metres; upright along +Z, feet at z=0, facing +Y
    faces     (F, 3) int     triangle indices into `vertices`
    rgb       (V, 3) float   optional, 0-1 per vertex
    height_m  ()   float     the stature the model measured, for cross-check
"""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..film import render
from ..film.sweep import EYELINE_RATIO, SUBJECT_HEIGHT_M

CAST_FILE = "cast.json"
CAST_VERSION = 1

# Stature bounds for an adult or child actor in metres. Outside this the typed
# number is a typo (centimetres, feet) and the plan built on it would be wrong
# by a factor, so it is refused rather than clamped.
MIN_HEIGHT_M = 1.2
MAX_HEIGHT_M = 2.2

# The stature every proportion below is written against. A 1.75 m figure has
# a 0.11 m head radius and 0.46 m shoulders; everyone else is that, scaled.
REFERENCE_HEIGHT_M = 1.75

# Fixed palette so the same actor is the same colour in every frame, table and
# legend. Saturated and far apart so two bodies read as two people in a grey
# gaussian render. RGB 0-255.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (230, 80, 60),     # red
    (60, 130, 230),    # blue
    (240, 180, 40),    # amber
    (70, 190, 110),    # green
    (190, 90, 210),    # violet
    (240, 120, 200),   # pink
    (40, 190, 200),    # teal
    (250, 140, 50),    # orange
)

# Over-the-shoulder: when the eyes differ by more than this, the taller
# actor's shoulder crowds the shorter one's face or the camera has to tilt;
# the grip's fix is an apple box under the shorter actor.
OTS_MAX_DIFF_M = 0.15
# The two boxes a grip reaches for, in metres: half apple and full apple.
APPLE_BOXES_M = (0.1, 0.2)

# Figure point samples: a body mesh gets this many surface samples; the
# capsule gets one per `spacing_m` cell, twice over, because random surface
# sampling leaves holes at one-per-cell and holes are what a render shows.
BODY_SAMPLES = 3000
CAPSULE_SPACING_M = 0.04
FIGURE_RADIUS_M = 0.025

SOURCES = ("typed", "sam3d-body")

_lock = threading.RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# the actor and the store
# ---------------------------------------------------------------------------


@dataclass
class Actor:
    """A named person at a real height. `height_m` is required and bounded;
    `eye_height_m` defaults to the anthropometric 0.94 of stature the sweep
    uses, and can be overridden when someone measured it."""

    name: str
    height_m: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    eye_height_m: float | None = None
    colour: tuple[int, int, int] = PALETTE[0]
    headshot: str | None = None
    body_path: str | None = None
    source: str = "typed"
    created_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if not self.name:
            raise ValueError("an actor needs a name")
        try:
            self.height_m = float(self.height_m)
        except (TypeError, ValueError):
            raise ValueError(f"height_m must be a number in metres, got {self.height_m!r}") from None
        if not math.isfinite(self.height_m) or not (MIN_HEIGHT_M <= self.height_m <= MAX_HEIGHT_M):
            raise ValueError(
                f"height_m={self.height_m} is outside {MIN_HEIGHT_M}-{MAX_HEIGHT_M} m; "
                "heights are in metres (1.75, not 175)"
            )
        if self.eye_height_m is None:
            self.eye_height_m = round(EYELINE_RATIO * self.height_m, 4)
        self.eye_height_m = float(self.eye_height_m)
        if not (0.5 * self.height_m <= self.eye_height_m <= self.height_m):
            raise ValueError(
                f"eye_height_m={self.eye_height_m} is not between half and all of a "
                f"{self.height_m} m stature"
            )
        c = tuple(int(v) for v in self.colour)
        if len(c) != 3 or any(not 0 <= v <= 255 for v in c):
            raise ValueError("colour must be an RGB triple 0-255")
        self.colour = c
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        if self.body_path is not None:
            self.body_path = str(self.body_path)
        if self.headshot is not None:
            self.headshot = str(self.headshot)

    @property
    def colour_rgb01(self) -> np.ndarray:
        return np.asarray(self.colour, dtype=np.float32) / 255.0

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["colour"] = list(self.colour)
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Actor":
        known = {k: d.get(k) for k in cls.__dataclass_fields__ if k in d}
        if "colour" in known and known["colour"] is not None:
            known["colour"] = tuple(known["colour"])
        return cls(**known)


def _cast_path(workdir: str | Path) -> Path:
    return Path(workdir) / CAST_FILE


def load_cast(workdir: str | Path) -> list[Actor]:
    """The job's cast, in the order they were added; empty when there is none."""
    path = _cast_path(workdir)
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    actors = raw.get("actors", []) if isinstance(raw, dict) else raw
    out: list[Actor] = []
    for d in actors:
        try:
            out.append(Actor.from_json(d))
        except (TypeError, ValueError):
            # One bad record must not take the whole cast down with it.
            continue
    return out


def save_cast(workdir: str | Path, cast: list[Actor]) -> Path:
    """Write the cast atomically: temp file then rename, so a crash mid-write
    never leaves a half-written cast."""
    path = _cast_path(workdir)
    obj = {"version": CAST_VERSION, "updated_at": _now(), "actors": [a.to_json() for a in cast]}
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(obj, indent=2))
        os.replace(tmp, path)
    return path


def next_colour(cast: list[Actor]) -> tuple[int, int, int]:
    """The first palette colour nobody in the cast has; round-robin after that."""
    used = {a.colour for a in cast}
    for c in PALETTE:
        if c not in used:
            return c
    return PALETTE[len(cast) % len(PALETTE)]


def add_actor(
    workdir: str | Path,
    name: str,
    height_m: float,
    *,
    eye_height_m: float | None = None,
    headshot: str | None = None,
) -> Actor:
    """Add a typed actor to the job's cast and persist it."""
    with _lock:
        cast = load_cast(workdir)
        actor = Actor(
            name=name,
            height_m=height_m,
            eye_height_m=eye_height_m,
            colour=next_colour(cast),
            headshot=headshot,
        )
        cast.append(actor)
        save_cast(workdir, cast)
    return actor


def remove_actor(workdir: str | Path, actor_id: str) -> bool:
    """Drop an actor by id. True when someone was removed."""
    with _lock:
        cast = load_cast(workdir)
        kept = [a for a in cast if a.id != actor_id]
        if len(kept) == len(cast):
            return False
        save_cast(workdir, kept)
    return True


def get_actor(workdir: str | Path, actor_id: str) -> Actor | None:
    for a in load_cast(workdir):
        if a.id == actor_id:
            return a
    return None


# ---------------------------------------------------------------------------
# the body from SAM 3D Body
# ---------------------------------------------------------------------------


@dataclass
class Body:
    """A body mesh in the contract's frame: metres, upright +Z, feet at 0,
    facing +Y. `height_m` is what the model measured, kept for the cross-check
    in `describe`; the mesh itself is scaled to the actor's typed height."""

    vertices: np.ndarray
    faces: np.ndarray
    rgb: np.ndarray | None = None
    height_m: float | None = None

    @property
    def extent_z_m(self) -> float:
        return float(self.vertices[:, 2].max() - self.vertices[:, 2].min())


BODY_REQUIRED_KEYS = ("vertices", "faces")
BODY_OPTIONAL_KEYS = ("rgb", "height_m")


def load_body(path: str | Path) -> Body:
    """Read and validate a body npz against the contract in the module doc."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"body npz not found: {path}")
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        missing = [k for k in BODY_REQUIRED_KEYS if k not in keys]
        if missing:
            raise ValueError(f"body npz {path.name} lacks {missing}; needs {BODY_REQUIRED_KEYS}")
        verts = np.asarray(z["vertices"], dtype=np.float64)
        faces = np.asarray(z["faces"])
        rgb = np.asarray(z["rgb"], dtype=np.float64) if "rgb" in keys else None
        height = float(np.asarray(z["height_m"]).reshape(-1)[0]) if "height_m" in keys else None

    if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) < 4:
        raise ValueError(f"vertices must be (V, 3) with V >= 4, got {verts.shape}")
    if not np.all(np.isfinite(verts)):
        raise ValueError("vertices contain non-finite values")
    if faces.dtype.kind not in "iu" or faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 1:
        raise ValueError(f"faces must be an integer (F, 3) array, got {faces.dtype} {faces.shape}")
    faces = faces.astype(np.int64)
    if faces.min() < 0 or faces.max() >= len(verts):
        raise ValueError("faces index outside the vertex array")
    if rgb is not None:
        if rgb.shape != verts.shape:
            raise ValueError(f"rgb must be (V, 3) matching vertices, got {rgb.shape}")
        if rgb.min() < 0.0 or rgb.max() > 1.0:
            # Tolerate 0-255 from a careless writer; refuse anything stranger.
            if rgb.max() <= 255.0 and rgb.min() >= 0.0:
                rgb = rgb / 255.0
            else:
                raise ValueError("rgb must be 0-1 (or 0-255) per vertex")
    body = Body(vertices=verts, faces=faces, rgb=rgb, height_m=height)
    if body.extent_z_m < 1e-6:
        raise ValueError("body mesh has no height along +Z; is it upright?")
    if height is not None and not (0.3 <= height <= 3.0):
        raise ValueError(f"height_m={height} in the npz is not a stature in metres")
    return body


def scale_to_height(body: Body, height_m: float) -> Body:
    """Uniform scale so the mesh's z-extent equals `height_m`, feet at z=0.

    The typed height is the truth. Only x and y are recentred (the mesh's
    footprint centroid goes to the origin, so a mark is under the body); z is
    shifted so the lowest vertex is the floor. The facing (+Y) is untouched.
    """
    v = body.vertices
    s = float(height_m) / body.extent_z_m
    out = v.copy()
    out[:, 2] -= v[:, 2].min()
    centre_xy = out[:, :2].mean(axis=0)
    out[:, :2] -= centre_xy
    out *= s
    return Body(vertices=out, faces=body.faces, rgb=body.rgb, height_m=body.height_m)


def attach_body(
    actor: Actor,
    npz_path: str | Path,
    *,
    workdir: str | Path | None = None,
) -> Actor:
    """Give an actor a SAM 3D Body mesh. The npz is validated now so a bad
    file fails at attach time, not at the first render. When `workdir` is
    given the path is stored relative to it and the cast is saved."""
    npz_path = Path(npz_path)
    load_body(npz_path)
    stored = str(npz_path)
    if workdir is not None:
        try:
            stored = str(npz_path.resolve().relative_to(Path(workdir).resolve()))
        except ValueError:
            stored = str(npz_path.resolve())
    actor.body_path = stored
    actor.source = "sam3d-body"
    if workdir is not None:
        with _lock:
            cast = load_cast(workdir)
            replaced = False
            for i, a in enumerate(cast):
                if a.id == actor.id:
                    cast[i] = actor
                    replaced = True
            if not replaced:
                cast.append(actor)
            save_cast(workdir, cast)
    return actor


def detach_body(actor: Actor, *, workdir: str | Path | None = None) -> Actor:
    actor.body_path = None
    actor.source = "typed"
    if workdir is not None:
        with _lock:
            cast = load_cast(workdir)
            cast = [actor if a.id == actor.id else a for a in cast]
            save_cast(workdir, cast)
    return actor


def resolve_body_path(actor: Actor, workdir: str | Path | None = None) -> Path | None:
    if not actor.body_path:
        return None
    p = Path(actor.body_path)
    if not p.is_absolute() and workdir is not None:
        p = Path(workdir) / p
    return p


# ---------------------------------------------------------------------------
# the capsule: a human-shaped volume from height alone
# ---------------------------------------------------------------------------


def capsule_mesh(height_m: float):
    """A proportioned stand-in built from trimesh primitives: head sphere,
    neck, torso ellipsoid, two legs, two arms hanging. Upright along +Z,
    feet at z=0, facing +Y, shoulders along X. Proportions are the reference
    1.75 m figure scaled by stature."""
    import trimesh

    h = float(height_m)
    s = h / REFERENCE_HEIGHT_M
    parts = []

    head_r = 0.11 * s
    head = trimesh.creation.icosphere(subdivisions=3, radius=head_r)
    head.apply_translation([0.0, 0.0, h - head_r])
    parts.append(head)

    shoulder_z = 1.45 * s
    hip_z = 0.90 * s
    neck_h = (h - 2 * head_r) - shoulder_z
    if neck_h > 1e-3:
        neck = trimesh.creation.cylinder(radius=0.05 * s, height=neck_h, sections=16)
        neck.apply_translation([0.0, 0.0, shoulder_z + neck_h / 2.0])
        parts.append(neck)

    torso = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    torso.apply_scale([0.23 * s, 0.12 * s, (shoulder_z - hip_z) / 2.0 + 0.03 * s])
    torso.apply_translation([0.0, 0.0, (shoulder_z + hip_z) / 2.0])
    parts.append(torso)

    leg_h = hip_z + 0.05 * s
    for sx in (-1.0, 1.0):
        leg = trimesh.creation.cylinder(radius=0.075 * s, height=leg_h, sections=20)
        leg.apply_translation([sx * 0.10 * s, 0.0, leg_h / 2.0])
        parts.append(leg)

    arm_top = shoulder_z - 0.02 * s
    arm_h = 0.67 * s
    for sx in (-1.0, 1.0):
        arm = trimesh.creation.cylinder(radius=0.045 * s, height=arm_h, sections=16)
        arm.apply_translation([sx * 0.255 * s, 0.0, arm_top - arm_h / 2.0])
        parts.append(arm)

    return trimesh.util.concatenate(parts)


def capsule_points(height_m: float, *, spacing_m: float = CAPSULE_SPACING_M, seed: int = 0) -> np.ndarray:
    """Surface samples of the capsule, (N, 3) in metres: feet at z=0, head
    top at `height_m`, facing +Y. About 2400 points at the default spacing
    for 1.75 m; the count scales with surface area."""
    from trimesh.sample import sample_surface

    mesh = capsule_mesh(height_m)
    count = int(math.ceil(2.0 * float(mesh.area) / (float(spacing_m) ** 2)))
    pts, _ = sample_surface(mesh, max(count, 100), seed=seed)
    pts = np.asarray(pts, dtype=np.float64)
    # The samples sit on the primitive surfaces; pin the extremes exactly so
    # the figure's top and feet are the typed stature whatever the sampler drew.
    pts = np.vstack([pts, [[0.0, 0.0, float(height_m)]], [[0.0, 0.0, 0.0]]])
    return pts


# ---------------------------------------------------------------------------
# the figure the renderer draws
# ---------------------------------------------------------------------------


def _rotation_z(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def body_points(body: Body, *, samples: int = BODY_SAMPLES, seed: int = 0) -> tuple[np.ndarray, np.ndarray | None]:
    """Surface samples of a body mesh and, when it carries colour, the colour
    at each sample (mean of the face's vertex colours)."""
    import trimesh
    from trimesh.sample import sample_surface

    mesh = trimesh.Trimesh(vertices=body.vertices, faces=body.faces, process=False)
    pts, fidx = sample_surface(mesh, int(samples), seed=seed)
    pts = np.asarray(pts, dtype=np.float64)
    rgb = None
    if body.rgb is not None:
        rgb = body.rgb[body.faces[np.asarray(fidx)]].mean(axis=1).astype(np.float32)
    return pts, rgb


def figure(
    actor: Actor,
    mark_xy,
    facing_deg: float,
    floor_z: float | None = None,
    *,
    twin=None,
    workdir: str | Path | None = None,
    radius_m: float = FIGURE_RADIUS_M,
) -> render.Figure:
    """The actor standing on `mark_xy`, feet on `floor_z`, looking along
    `facing_deg` (degrees anticlockwise from +X in plan, so 90 is the body's
    canonical +Y). A body mesh when one is attached, scaled to the typed
    height; the capsule otherwise. Coloured with the actor's palette colour,
    except that a body carrying its own rgb keeps it."""
    if floor_z is None:
        if twin is None:
            raise ValueError("figure needs floor_z or a twin to read it from")
        floor_z = float(twin.structure.floor_z)
    mark = np.asarray(mark_xy, dtype=np.float64).reshape(2)

    rgb: np.ndarray | None = None
    path = resolve_body_path(actor, workdir)
    if path is not None:
        body = scale_to_height(load_body(path), actor.height_m)
        pts, rgb = body_points(body)
    else:
        pts = capsule_points(actor.height_m)

    R = _rotation_z(float(facing_deg) - 90.0)
    xyz = pts @ R.T
    xyz[:, 0] += mark[0]
    xyz[:, 1] += mark[1]
    xyz[:, 2] += float(floor_z)

    colours = rgb if rgb is not None else actor.colour_rgb01
    return render.Figure(xyz=xyz, rgb=colours, radius_m=float(radius_m), label=actor.name)


# ---------------------------------------------------------------------------
# arithmetic for the planner's notes
# ---------------------------------------------------------------------------


def pair_eyelines(a: Actor, b: Actor, *, distance_m: float = 1.2) -> dict[str, Any]:
    """Two actors facing each other `distance_m` apart: how their eyelines
    meet, and whether a standard over-the-shoulder works without a box.

    Pure arithmetic. The angle is the tilt of the sightline between the two
    pairs of eyes: the taller looks down by it, the shorter looks up. Beyond
    `OTS_MAX_DIFF_M` of eye difference, the fix is an apple box (half or full)
    under the shorter actor, chosen to bring the difference inside the band.
    """
    distance_m = float(distance_m)
    if distance_m <= 0:
        raise ValueError("distance_m must be positive")
    height_diff = a.height_m - b.height_m
    eye_diff = a.eye_height_m - b.eye_height_m
    angle = math.degrees(math.atan2(abs(eye_diff), distance_m))
    taller, shorter = (a, b) if eye_diff >= 0 else (b, a)
    ots_ok = abs(eye_diff) <= OTS_MAX_DIFF_M
    box = None
    if not ots_ok:
        for candidate in APPLE_BOXES_M:
            if abs(eye_diff) - candidate <= OTS_MAX_DIFF_M:
                box = candidate
                break
        if box is None:
            box = APPLE_BOXES_M[-1]
    residual = abs(eye_diff) - (box or 0.0)
    if ots_ok:
        note = (f"{a.name} and {b.name} meet eyes within {abs(eye_diff):.2f} m; "
                f"a standard over-the-shoulder works as they stand")
    else:
        note = (f"{taller.name} is {abs(eye_diff):.2f} m above {shorter.name} at the eyes "
                f"({angle:.0f} deg over {distance_m:.1f} m); a {box:.1f} m apple box under "
                f"{shorter.name} leaves {max(residual, 0.0):.2f} m")
    return {
        "a": a.name,
        "b": b.name,
        "distance_m": distance_m,
        "height_difference_m": round(height_diff, 4),
        "eye_difference_m": round(eye_diff, 4),
        "taller": taller.name,
        "shorter": shorter.name if taller is not shorter else None,
        "eyeline_angle_deg": round(angle, 2),
        "over_shoulder_ok": ots_ok,
        "apple_box_m": box,
        "apple_box_for": shorter.name if box else None,
        "residual_after_box_m": round(max(residual, 0.0), 4),
        "note": note,
    }


def describe(cast: list[Actor], *, workdir: str | Path | None = None) -> list[dict[str, Any]]:
    """The cast as the agent reads it: heights, eyes, colours, and for a body
    the model's own stature beside the typed one. Each entry is JSON-ready."""
    out: list[dict[str, Any]] = []
    for a in cast:
        d: dict[str, Any] = {
            "id": a.id,
            "name": a.name,
            "height_m": round(a.height_m, 4),
            "eye_height_m": round(a.eye_height_m, 4),
            "default_difference_m": round(a.height_m - SUBJECT_HEIGHT_M, 4),
            "colour": list(a.colour),
            "source": a.source,
            "headshot": a.headshot,
            "body": None,
        }
        path = resolve_body_path(a, workdir)
        if path is not None:
            body: dict[str, Any] = {"path": str(a.body_path), "attached": True}
            try:
                b = load_body(path)
                body["vertices"] = int(len(b.vertices))
                body["faces"] = int(len(b.faces))
                body["has_rgb"] = b.rgb is not None
                body["model_height_m"] = None if b.height_m is None else round(b.height_m, 4)
                body["model_vs_typed_m"] = (
                    None if b.height_m is None else round(b.height_m - a.height_m, 4)
                )
                body["note"] = "typed height is the truth; the model's height is a cross-check only"
            except (OSError, ValueError) as e:
                body["attached"] = False
                body["error"] = str(e)
            d["body"] = body
        out.append(d)
    return out
