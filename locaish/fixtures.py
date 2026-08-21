"""Synthetic rooms with exact ground truth.

The whole "1:1" claim rests on being able to check the pipeline against a room
whose true dimensions we know to the millimetre. Real scans can't do that --
you'd be comparing against a tape measure held by a person. So we build rooms
analytically, sample them the way a handheld LiDAR sweep samples a room (noisy,
uneven density, occluded, drifting), hand the result to the pipeline in an
arbitrary pose and unit, and assert that what comes back matches the numbers we
started from.

A fixture deliberately arrives *wrong* in every way an export can be wrong:

  - rotated and tilted (the phone was never level, and Polycam's floor is not
    always our floor)
  - in centimetres or inches instead of metres
  - with a translated origin far from the room
  - with holes where a body occluded a wall
  - with per-point noise, and optional slow drift across the sweep

Everything the canonicaliser is supposed to undo, `RoomTruth` remembers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .types import Mesh, PointCloud, rotation_z


# ---------------------------------------------------------------------------
# ground truth
# ---------------------------------------------------------------------------


#: How much solid wall an opening needs on every side to be a real aperture
#: rather than the end of the wall. Below this there is no jamb for the scanner
#: to catch and nothing for hole detection to close against.
_OPENING_MARGIN_M = 0.10


@dataclass
class OpeningSpec:
    """A rectangular hole in a wall, in the room's own frame.

    `wall` is one of -x, +x, -y, +y. `offset` is the centre's position along
    that wall measured from the room's centre, `sill` is the bottom edge height
    above the floor.
    """

    wall: str
    offset: float
    sill: float
    width: float
    height: float
    kind: str = "window"


@dataclass
class RoomTruth:
    """Everything we know exactly about a synthetic room, in room frame.

    Room frame: floor at z=0, room centred on the origin in XY, walls axis
    aligned. This is precisely the canonical twin frame, which is the point --
    a correct pipeline maps the mangled export back onto these numbers.
    """

    width: float  # X extent, m
    depth: float  # Y extent, m
    height: float  # Z extent, m
    openings: list[OpeningSpec] = field(default_factory=list)
    boxes: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    # the transform the fixture applied on top of room frame, so tests can
    # reason about what the canonicaliser has to undo
    applied_transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    applied_unit_scale: float = 1.0
    seed: int = 0

    @property
    def floor_area(self) -> float:
        return float(self.width * self.depth)

    @property
    def volume(self) -> float:
        return float(self.width * self.depth * self.height)

    def opening_area(self, kind: str | None = None) -> float:
        return float(
            sum(o.width * o.height for o in self.openings if kind is None or o.kind == kind)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "depth": self.depth,
            "height": self.height,
            "floor_area": self.floor_area,
            "volume": self.volume,
            "openings": [
                {
                    "wall": o.wall,
                    "offset": o.offset,
                    "sill": o.sill,
                    "width": o.width,
                    "height": o.height,
                    "kind": o.kind,
                }
                for o in self.openings
            ],
            "applied_unit_scale": self.applied_unit_scale,
            "seed": self.seed,
        }


@dataclass
class Fixture:
    """A synthetic scan plus the truth it was generated from."""

    points: PointCloud
    truth: RoomTruth
    mesh: Mesh | None = None
    camera_positions: np.ndarray | None = None


# ---------------------------------------------------------------------------
# surface sampling
# ---------------------------------------------------------------------------


def _sample_rect(
    rng: np.random.Generator,
    origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    u_len: float,
    v_len: float,
    density: float,
    holes: list[tuple[float, float, float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Poisson-ish uniform samples over a rectangle, minus any hole rectangles.

    Returns (points, uv) where uv is the 2D coordinate within the rectangle, so
    callers can decide what a sample means (e.g. which side of a window it is).
    `holes` are (umin, vmin, umax, vmax) boxes cut out of the rectangle.
    """
    n = max(1, int(u_len * v_len * density))
    uu = rng.random(n) * u_len
    vv = rng.random(n) * v_len
    if holes:
        keep = np.ones(n, dtype=bool)
        for umin, vmin, umax, vmax in holes:
            keep &= ~((uu >= umin) & (uu <= umax) & (vv >= vmin) & (vv <= vmax))
        uu, vv = uu[keep], vv[keep]
    pts = origin[None, :] + uu[:, None] * u[None, :] + vv[:, None] * v[None, :]
    return pts, np.stack([uu, vv], axis=1)


def _box_surface(
    rng: np.random.Generator, lo: np.ndarray, hi: np.ndarray, density: float
) -> np.ndarray:
    """Sample the outside of an axis-aligned box (furniture, a pillar)."""
    out = []
    size = hi - lo
    faces = [
        (np.array([lo[0], lo[1], lo[2]]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), size[0], size[1]),
        (np.array([lo[0], lo[1], hi[2]]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), size[0], size[1]),
        (np.array([lo[0], lo[1], lo[2]]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), size[0], size[2]),
        (np.array([lo[0], hi[1], lo[2]]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), size[0], size[2]),
        (np.array([lo[0], lo[1], lo[2]]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), size[1], size[2]),
        (np.array([hi[0], lo[1], lo[2]]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), size[1], size[2]),
    ]
    for origin, u, v, ul, vl in faces:
        if ul <= 0 or vl <= 0:
            continue
        pts, _ = _sample_rect(rng, origin, u, v, ul, vl, density)
        out.append(pts)
    return np.concatenate(out) if out else np.zeros((0, 3))


# ---------------------------------------------------------------------------
# the generator
# ---------------------------------------------------------------------------


def make_room(
    width: float = 5.2,
    depth: float = 4.1,
    height: float = 2.85,
    *,
    openings: list[OpeningSpec] | None = None,
    furniture: bool = True,
    density: float = 900.0,
    noise_m: float = 0.006,
    occlusion_patches: int = 6,
    drift_m: float = 0.0,
    unit_scale: float = 1.0,
    tilt_deg: float = 0.0,
    yaw_deg: float = 0.0,
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    seed: int = 0,
    with_mesh: bool = False,
) -> Fixture:
    """Build one synthetic room scan.

    `density` is samples per square metre of surface -- 900/m^2 is roughly what
    a careful iPhone LiDAR sweep of a small room yields after Polycam's
    decimation. `noise_m` is per-point Gaussian range noise. `occlusion_patches`
    punches circular holes in surfaces the way a person standing in the room
    does. `drift_m` applies a slow spatial warp across the sweep, which is the
    error mode that makes a room come out as a subtly non-rectangular box.

    `unit_scale`, `tilt_deg`, `yaw_deg` and `translate` are the export-side
    mangling: the returned cloud is in room frame transformed by
    `scale * R_tilt * R_yaw` then translated.
    """
    rng = np.random.default_rng(seed)
    if openings is None:
        openings = [
            OpeningSpec("+y", offset=-0.8, sill=0.95, width=1.4, height=1.25, kind="window"),
            OpeningSpec("+y", offset=1.6, sill=0.95, width=0.9, height=1.25, kind="window"),
            OpeningSpec("-x", offset=0.4, sill=0.0, width=0.86, height=2.04, kind="door"),
        ]

    hw, hd = width / 2.0, depth / 2.0

    # Drop openings that do not fit on the wall they were assigned to. The
    # default set is placed by offset from the wall's centre, so a narrow room
    # pushes a window off the end of its own wall, and a window that runs off
    # the edge is not an aperture at all -- it is a wall that stops. The
    # pipeline correctly refuses to report those (a hole touching the edge of
    # the scanned wall is unscanned wall, not a window), so leaving them in the
    # truth would score the pipeline for the generator's mistake. The margin is
    # the width of jamb needed for the hole to be enclosed on both sides.
    kept: list[OpeningSpec] = []
    for op in openings:
        wall_len = width if op.wall in ("+y", "-y") else depth
        centre = wall_len / 2.0 + op.offset
        lo, hi = centre - op.width / 2.0, centre + op.width / 2.0
        fits_across = lo >= _OPENING_MARGIN_M and hi <= wall_len - _OPENING_MARGIN_M
        fits_up = op.sill >= 0.0 and op.sill + op.height <= height - _OPENING_MARGIN_M
        if fits_across and fits_up:
            kept.append(op)
    openings = kept

    chunks: list[np.ndarray] = []

    # floor and ceiling
    floor, _ = _sample_rect(
        rng, np.array([-hw, -hd, 0.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
        width, depth, density,
    )
    ceiling, _ = _sample_rect(
        rng, np.array([-hw, -hd, height]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
        width, depth, density * 0.7,
    )
    chunks += [floor, ceiling]

    # walls, with openings cut out. u runs along the wall, v runs up.
    wall_defs = {
        "-x": (np.array([-hw, -hd, 0.0]), np.array([0, 1.0, 0]), depth),
        "+x": (np.array([hw, -hd, 0.0]), np.array([0, 1.0, 0]), depth),
        "-y": (np.array([-hw, -hd, 0.0]), np.array([1.0, 0, 0]), width),
        "+y": (np.array([-hw, hd, 0.0]), np.array([1.0, 0, 0]), width),
    }
    for wall, (origin, u, u_len) in wall_defs.items():
        holes = []
        for op in openings:
            if op.wall != wall:
                continue
            # `offset` is measured from the wall's centre along +u
            centre_u = u_len / 2.0 + op.offset
            holes.append(
                (
                    centre_u - op.width / 2.0,
                    op.sill,
                    centre_u + op.width / 2.0,
                    op.sill + op.height,
                )
            )
        pts, _ = _sample_rect(
            rng, origin, u, np.array([0, 0, 1.0]), u_len, height, density, holes
        )
        chunks.append(pts)
        # a real scan catches the reveal -- the depth of the wall around the
        # opening. Without it, opening detection has no edge to lock onto.
        for umin, vmin, umax, vmax in holes:
            for du in (umin, umax):
                edge = origin + du * u
                jamb, _ = _sample_rect(
                    rng, edge + np.array([0, 0, vmin]),
                    np.cross(u, np.array([0, 0, 1.0])), np.array([0, 0, 1.0]),
                    0.12, vmax - vmin, density * 1.5,
                )
                chunks.append(jamb)

    if furniture:
        boxes = [
            (np.array([-hw + 0.15, -0.9, 0.0]), np.array([-hw + 0.75, 0.9, 0.72])),
            (np.array([0.6, hd - 0.85, 0.0]), np.array([2.1, hd - 0.15, 0.45])),
            (np.array([hw - 1.1, -hd + 0.2, 0.0]), np.array([hw - 0.2, -hd + 1.1, 1.15])),
        ]
        boxes = [
            (lo, hi) for lo, hi in boxes
            if np.all(hi > lo) and hi[0] <= hw and hi[1] <= hd and lo[0] >= -hw and lo[1] >= -hd
        ]
        for lo, hi in boxes:
            chunks.append(_box_surface(rng, lo, hi, density))
    else:
        boxes = []

    xyz = np.concatenate(chunks)

    # occlusion: a person standing in the room leaves round shadows
    for _ in range(occlusion_patches):
        centre = xyz[rng.integers(len(xyz))]
        radius = rng.uniform(0.15, 0.45)
        keep = np.linalg.norm(xyz - centre, axis=1) > radius
        xyz = xyz[keep]

    # drift: a slow, smooth, low-frequency warp -- the SLAM error mode
    if drift_m > 0:
        phase = rng.random(3) * 2 * math.pi
        span = np.maximum(xyz.max(axis=0) - xyz.min(axis=0), 1e-6)
        t = (xyz - xyz.min(axis=0)) / span
        warp = np.stack(
            [np.sin(2 * math.pi * t[:, (i + 1) % 3] + phase[i]) for i in range(3)], axis=1
        )
        xyz = xyz + warp * drift_m

    if noise_m > 0:
        xyz = xyz + rng.normal(0.0, noise_m, xyz.shape)

    rgb = _colorize(xyz, height, rng)

    truth = RoomTruth(
        width=width, depth=depth, height=height, openings=list(openings),
        boxes=boxes, seed=seed,
    )

    # the camera path: a person walking a loop at chest height
    cams = _camera_path(rng, width, depth)

    # export-side mangling
    transform = np.eye(4)
    rot = rotation_z(math.radians(yaw_deg))
    if tilt_deg:
        # tilt about a random horizontal axis, which is what a phone's IMU bias
        # or a scanner that guessed the floor wrong actually produces
        axis_angle = rng.random() * 2 * math.pi
        axis = np.array([math.cos(axis_angle), math.sin(axis_angle), 0.0])
        rot = _axis_angle(axis, math.radians(tilt_deg)) @ rot
    transform[:3, :3] = rot * unit_scale
    transform[:3, 3] = np.asarray(translate, dtype=np.float64)

    truth.applied_transform = transform
    truth.applied_unit_scale = unit_scale

    cloud = PointCloud(xyz=xyz, rgb=rgb).transformed(transform)
    cams_t = cams @ transform[:3, :3].T + transform[:3, 3]

    mesh = None
    if with_mesh:
        mesh = _room_mesh(width, depth, height, openings).transformed(transform)

    return Fixture(points=cloud, truth=truth, mesh=mesh, camera_positions=cams_t)


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def _colorize(xyz: np.ndarray, height: float, rng: np.random.Generator) -> np.ndarray:
    """Plausible per-point colour: darker floor, lighter walls, near-white
    ceiling, with enough variation that a texture-based check has signal."""
    t = np.clip(xyz[:, 2] / max(height, 1e-6), 0, 1)
    base = np.stack(
        [110 + 120 * t, 105 + 125 * t, 100 + 130 * t], axis=1
    )
    base += rng.normal(0, 12, base.shape)
    return np.clip(base, 0, 255).astype(np.uint8)


def _camera_path(rng: np.random.Generator, width: float, depth: float) -> np.ndarray:
    """A loop inset from the walls at roughly chest height, jittered."""
    n = 160
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rx = max(width / 2.0 - 0.9, 0.3)
    ry = max(depth / 2.0 - 0.9, 0.3)
    xy = np.stack([rx * np.cos(t), ry * np.sin(t)], axis=1)
    xy += rng.normal(0, 0.05, xy.shape)
    z = 1.45 + rng.normal(0, 0.06, n)
    return np.concatenate([xy, z[:, None]], axis=1)


def _room_mesh(
    width: float, depth: float, height: float, openings: list[OpeningSpec]
) -> Mesh:
    """A clean box mesh for the room shell, openings ignored.

    This exists so the mesh-carrying import path has something to chew on; it
    is not the reference for opening detection, the point cloud is.
    """
    hw, hd = width / 2.0, depth / 2.0
    v = np.array(
        [
            [-hw, -hd, 0.0], [hw, -hd, 0.0], [hw, hd, 0.0], [-hw, hd, 0.0],
            [-hw, -hd, height], [hw, -hd, height], [hw, hd, height], [-hw, hd, height],
        ]
    )
    # wound so normals point into the room
    f = np.array(
        [
            [0, 2, 1], [0, 3, 2],          # floor, normal +z
            [4, 5, 6], [4, 6, 7],          # ceiling, normal -z
            [0, 1, 5], [0, 5, 4],          # -y wall
            [2, 3, 7], [2, 7, 6],          # +y wall
            [3, 0, 4], [3, 4, 7],          # -x wall
            [1, 2, 6], [1, 6, 5],          # +x wall
        ]
    )
    return Mesh(vertices=v, faces=f)


# ---------------------------------------------------------------------------
# named fixtures used by the tests and by `locaish demo`
# ---------------------------------------------------------------------------


def catalogue() -> dict[str, dict[str, Any]]:
    """The rooms we validate against, and what each one is here to break."""
    return {
        "clean": dict(
            kwargs=dict(seed=1, noise_m=0.003, occlusion_patches=2),
            breaks="nothing -- the happy path, tightest tolerances",
        ),
        "tilted": dict(
            kwargs=dict(seed=2, tilt_deg=7.5, yaw_deg=41.0, translate=(31.2, -18.7, 4.4)),
            breaks="gravity alignment and yaw normalisation",
        ),
        "centimetres": dict(
            kwargs=dict(seed=3, unit_scale=100.0, yaw_deg=15.0),
            breaks="unit inference -- a room 520 units wide is not 520 metres",
        ),
        "inches": dict(
            kwargs=dict(seed=4, unit_scale=39.3701, tilt_deg=3.0),
            breaks="unit inference against a non-metric factor",
        ),
        "drifty": dict(
            kwargs=dict(seed=5, drift_m=0.035, noise_m=0.012, tilt_deg=4.0),
            breaks="SLAM drift -- walls stop being planar, QA must notice",
        ),
        "sparse": dict(
            kwargs=dict(seed=6, density=180.0, occlusion_patches=14),
            breaks="thin coverage -- QA must warn rather than silently guess",
        ),
        "tall": dict(
            kwargs=dict(seed=7, width=9.5, depth=7.2, height=4.6, yaw_deg=63.0),
            breaks="a room whose ceiling is outside the usual 2.3-3.2 m prior",
        ),
        "noceiling": dict(
            kwargs=dict(seed=8, height=2.7, yaw_deg=22.0),
            breaks="an open-topped capture; ceiling_z must come back None",
            post="strip_ceiling",
        ),
    }


def build(name: str, **overrides: Any) -> Fixture:
    """Build a named fixture from the catalogue."""
    entry = catalogue().get(name)
    if entry is None:
        raise KeyError(f"unknown fixture {name!r}; have {sorted(catalogue())}")
    kwargs = dict(entry["kwargs"])
    kwargs.update(overrides)
    fx = make_room(**kwargs)
    if entry.get("post") == "strip_ceiling":
        fx = _strip_ceiling(fx)
    return fx


def _strip_ceiling(fx: Fixture) -> Fixture:
    """Drop the top 12% of the room, simulating a capture that never looked up."""
    inv = np.linalg.inv(fx.truth.applied_transform)
    local = fx.points.xyz @ inv[:3, :3].T + inv[:3, 3]
    keep = local[:, 2] < fx.truth.height * 0.88
    return Fixture(
        points=fx.points.subset(keep),
        truth=fx.truth,
        mesh=None,
        camera_positions=fx.camera_positions,
    )
