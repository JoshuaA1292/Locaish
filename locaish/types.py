"""Core data model for Locaish twins.

Coordinate convention (canonical twin space), obeyed by everything downstream:

    Z is up, right-handed. The floor plane is z = 0. Positive Z is away from
    gravity. Units are metres, always. After yaw normalisation the dominant
    wall direction is parallel to +X, so a room in a typical building has
    axis-aligned walls and `Structure.footprint` is close to a rectangle.

    The origin (x=0, y=0) is the centroid of the floor footprint, so a twin is
    roughly centred and small coordinates mean "middle of the room".

The mapping back to the real world lives in `Georeference`: `heading_deg` is
the true-north bearing of the twin's +X axis, which together with lat/lon puts
every point on Earth.

Nothing in this module computes anything. It defines the shapes that the
readers, the canonicaliser, the analysers and the viewer all agree on, and the
on-disk format. Keep it dependency-light: numpy only.
"""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# geometry containers
# ---------------------------------------------------------------------------


@dataclass
class PointCloud:
    """A dense point cloud in whatever frame the owner says it is in.

    `xyz` is (N, 3) float32/float64. `rgb` is (N, 3) uint8 or None. `normals`
    is (N, 3) float32 unit vectors or None -- readers usually leave it None and
    `geom.normals.estimate` fills it in.

    `inferred` is (N,) float32 in [0, 1] or None: how much of each point was
    resampled from a model of the room rather than measured off it -- the
    point-cloud twin of `Mesh.filled`. None means nothing was inferred, which
    is not the same as zeros: one says the question was never asked, the other
    says it was asked of every point and answered no. Anything measuring off
    this cloud must exclude points with `inferred > 0.5`.
    """

    xyz: np.ndarray
    rgb: np.ndarray | None = None
    normals: np.ndarray | None = None
    inferred: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=np.float64).reshape(-1, 3)
        if self.rgb is not None:
            self.rgb = np.asarray(self.rgb, dtype=np.uint8).reshape(-1, 3)
            if len(self.rgb) != len(self.xyz):
                raise ValueError(
                    f"rgb has {len(self.rgb)} rows but xyz has {len(self.xyz)}"
                )
        if self.normals is not None:
            self.normals = np.asarray(self.normals, dtype=np.float64).reshape(-1, 3)
            if len(self.normals) != len(self.xyz):
                raise ValueError(
                    f"normals has {len(self.normals)} rows but xyz has {len(self.xyz)}"
                )
        if self.inferred is not None:
            self.inferred = np.asarray(self.inferred, dtype=np.float32).reshape(-1)
            if len(self.inferred) != len(self.xyz):
                raise ValueError(
                    f"inferred has {len(self.inferred)} entries but xyz has "
                    f"{len(self.xyz)} rows"
                )

    def __len__(self) -> int:
        return len(self.xyz)

    @property
    def bounds(self) -> np.ndarray:
        """(2, 3) array of [min_xyz, max_xyz]. Empty cloud gives zeros."""
        if len(self.xyz) == 0:
            return np.zeros((2, 3))
        return np.stack([self.xyz.min(axis=0), self.xyz.max(axis=0)])

    @property
    def extent(self) -> np.ndarray:
        b = self.bounds
        return b[1] - b[0]

    def transformed(self, matrix: np.ndarray) -> "PointCloud":
        """Apply a 4x4 homogeneous transform. Normals get the rotation only."""
        matrix = np.asarray(matrix, dtype=np.float64)
        rot = matrix[:3, :3]
        xyz = self.xyz @ rot.T + matrix[:3, 3]
        normals = None
        if self.normals is not None:
            # normal transform is the inverse-transpose; for a similarity
            # transform that is the rotation up to a scale we renormalise away
            normals = self.normals @ np.linalg.inv(rot)
            n = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.where(n == 0, 1.0, n)
        return PointCloud(xyz=xyz, rgb=self.rgb, normals=normals, inferred=self.inferred)

    def subset(self, index: np.ndarray) -> "PointCloud":
        return PointCloud(
            xyz=self.xyz[index],
            rgb=None if self.rgb is None else self.rgb[index],
            normals=None if self.normals is None else self.normals[index],
            inferred=None if self.inferred is None else self.inferred[index],
        )

    def measured(self) -> "PointCloud":
        """The subset that was actually observed, for anything taking a number.

        Identity when nothing was inferred, so callers can use it
        unconditionally.
        """
        if self.inferred is None:
            return self
        return self.subset(self.inferred <= 0.5)


@dataclass
class Mesh:
    """A triangle mesh, optionally textured.

    `vertices` (V, 3) float64, `faces` (F, 3) int32 into vertices.
    `vertex_colors` (V, 3) uint8 or None. `uv` (V, 2) float32 or None.
    `texture` is a PIL-decodable image byte blob (PNG/JPEG) or None; keeping it
    as bytes means we never make Pillow a hard import in the data model.

    `filled` is (V,) float32 in [0, 1] or None: how much of each vertex was
    inferred rather than measured, 0 for a vertex sitting on real returns and 1
    for one in the middle of a bridged hole. None means nothing was filled, and
    is not the same as an array of zeros -- one says the question was never
    asked, the other says it was asked and the answer was no. Anything that
    quotes a distance off this mesh has to look here first.
    """

    vertices: np.ndarray
    faces: np.ndarray
    vertex_colors: np.ndarray | None = None
    uv: np.ndarray | None = None
    texture: bytes | None = None
    texture_format: str = "png"
    filled: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int32).reshape(-1, 3)
        if self.vertex_colors is not None:
            self.vertex_colors = np.asarray(self.vertex_colors, dtype=np.uint8).reshape(-1, 3)
        if self.uv is not None:
            self.uv = np.asarray(self.uv, dtype=np.float32).reshape(-1, 2)
        if self.filled is not None:
            self.filled = np.asarray(self.filled, dtype=np.float32).reshape(-1)
            if len(self.filled) != len(self.vertices):
                raise ValueError(
                    f"filled has {len(self.filled)} entries but there are "
                    f"{len(self.vertices)} vertices"
                )

    def __len__(self) -> int:
        return len(self.faces)

    @property
    def bounds(self) -> np.ndarray:
        if len(self.vertices) == 0:
            return np.zeros((2, 3))
        return np.stack([self.vertices.min(axis=0), self.vertices.max(axis=0)])

    @property
    def face_normals(self) -> np.ndarray:
        """(F, 3) unit normals, right-hand rule over the vertex winding."""
        v = self.vertices[self.faces]
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.where(ln == 0, 1.0, ln)

    @property
    def face_areas(self) -> np.ndarray:
        v = self.vertices[self.faces]
        return 0.5 * np.linalg.norm(
            np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1
        )

    @property
    def area(self) -> float:
        return float(self.face_areas.sum())

    def transformed(self, matrix: np.ndarray) -> "Mesh":
        matrix = np.asarray(matrix, dtype=np.float64)
        verts = self.vertices @ matrix[:3, :3].T + matrix[:3, 3]
        # a mirroring transform flips winding; keep normals outward
        faces = self.faces
        if np.linalg.det(matrix[:3, :3]) < 0:
            faces = faces[:, ::-1].copy()
        return Mesh(
            vertices=verts,
            faces=faces,
            vertex_colors=self.vertex_colors,
            uv=self.uv,
            texture=self.texture,
            texture_format=self.texture_format,
        )

    def sample_surface(self, count: int, seed: int = 0) -> PointCloud:
        """Area-weighted uniform sample of the surface. Used to make a point
        cloud out of a mesh-only import so the rest of the pipeline is uniform."""
        if len(self.faces) == 0 or count <= 0:
            return PointCloud(np.zeros((0, 3)))
        rng = np.random.default_rng(seed)
        areas = self.face_areas
        total = areas.sum()
        if total <= 0:
            return PointCloud(np.zeros((0, 3)))
        pick = rng.choice(len(areas), size=count, p=areas / total)
        tri = self.vertices[self.faces[pick]]
        u = rng.random((count, 1))
        v = rng.random((count, 1))
        over = (u + v) > 1.0
        u[over] = 1.0 - u[over]
        v[over] = 1.0 - v[over]
        xyz = tri[:, 0] + u * (tri[:, 1] - tri[:, 0]) + v * (tri[:, 2] - tri[:, 0])
        rgb = None
        if self.vertex_colors is not None:
            c = self.vertex_colors[self.faces[pick]].astype(np.float64)
            w = np.concatenate([1.0 - u - v, u, v], axis=1)[:, :, None]
            rgb = np.clip((c * w).sum(axis=1), 0, 255).astype(np.uint8)
        normals = self.face_normals[pick]
        return PointCloud(xyz=xyz, rgb=rgb, normals=normals)


# ---------------------------------------------------------------------------
# structure: what the geometry means
# ---------------------------------------------------------------------------


@dataclass
class Plane:
    """An oriented plane fitted to a subset of the cloud.

    `normal` is a unit vector, `offset` is d in `dot(normal, p) = offset`, so
    the signed distance of a point is `dot(normal, p) - offset`. Normals point
    *into* the room (a floor's normal is +Z, a ceiling's is -Z, a wall's points
    at the room interior), which is what makes clearance queries sign-stable.

    `kind` is one of: floor, ceiling, wall, other.
    """

    normal: np.ndarray
    offset: float
    kind: str = "other"
    inlier_count: int = 0
    area: float = 0.0
    # axis-aligned extent of the inliers projected into the plane's own 2D
    # frame, as [umin, vmin, umax, vmax]; None when not computed
    extent_2d: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        self.normal = np.asarray(self.normal, dtype=np.float64).reshape(3)
        n = np.linalg.norm(self.normal)
        if n == 0:
            raise ValueError("plane normal must be non-zero")
        self.normal = self.normal / n
        self.offset = float(self.offset)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64).reshape(-1, 3) @ self.normal - self.offset

    def to_dict(self) -> dict[str, Any]:
        return {
            "normal": self.normal.tolist(),
            "offset": self.offset,
            "kind": self.kind,
            "inlier_count": int(self.inlier_count),
            "area": float(self.area),
            "extent_2d": None if self.extent_2d is None else list(self.extent_2d),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Plane":
        e = d.get("extent_2d")
        return cls(
            normal=np.array(d["normal"], dtype=np.float64),
            offset=float(d["offset"]),
            kind=d.get("kind", "other"),
            inlier_count=int(d.get("inlier_count", 0)),
            area=float(d.get("area", 0.0)),
            extent_2d=None if e is None else tuple(float(x) for x in e),
        )


@dataclass
class Opening:
    """A hole in a wall plane -- a window, a door, or a pass-through.

    `center` is the 3D centre in twin space, `width`/`height` are metres in the
    wall's own frame, `normal` is the wall normal at that point (pointing into
    the room). `sill_height` is the height of the bottom edge above the floor;
    a door has a sill near 0, a window does not. `kind` is window|door|opening.
    `confidence` in [0, 1] -- openings are inferred from missing geometry, so
    downstream code should be allowed to know how sure we are.
    """

    center: np.ndarray
    width: float
    height: float
    normal: np.ndarray
    sill_height: float
    kind: str = "opening"
    confidence: float = 0.0

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64).reshape(3)
        self.normal = np.asarray(self.normal, dtype=np.float64).reshape(3)
        n = np.linalg.norm(self.normal)
        if n:
            self.normal = self.normal / n

    @property
    def area(self) -> float:
        return float(self.width * self.height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": self.center.tolist(),
            "width": float(self.width),
            "height": float(self.height),
            "normal": self.normal.tolist(),
            "sill_height": float(self.sill_height),
            "kind": self.kind,
            "confidence": float(self.confidence),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Opening":
        return cls(
            center=np.array(d["center"], dtype=np.float64),
            width=float(d["width"]),
            height=float(d["height"]),
            normal=np.array(d["normal"], dtype=np.float64),
            sill_height=float(d["sill_height"]),
            kind=d.get("kind", "opening"),
            confidence=float(d.get("confidence", 0.0)),
        )


@dataclass
class Fixture:
    """A solid the returns found standing in the room, fitted as a prism.

    `footprint` is a closed (M, 2) XY polygon in twin space -- for the
    rectilinear fits the solver produces it is a rectangle aligned to the
    room's wall directions -- extruded from `z0` to `z1`. This is deliberately
    the vocabulary of a CAD massing model, not a mesh: a counter is a box, an
    island is a box, a fridge is a box, and a box is something a person can
    measure, occlude against, and light. `support` is how many returns stand
    on the solid, and `fill` is how much of the prism's plan the returns
    actually occupy -- both kept so a consumer can tell a confident island
    from a box drawn around a wisp of noise.
    """

    footprint: np.ndarray
    z0: float
    z1: float
    support: int = 0
    fill: float = 0.0
    kind: str = "fixture"

    def __post_init__(self) -> None:
        self.footprint = np.asarray(self.footprint, dtype=np.float64).reshape(-1, 2)

    @property
    def height(self) -> float:
        return float(self.z1 - self.z0)

    @property
    def plan_area(self) -> float:
        p = self.footprint
        q = np.roll(p, -1, axis=0)
        return float(abs(np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1])) / 2.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "footprint": self.footprint.tolist(),
            "z0": float(self.z0),
            "z1": float(self.z1),
            "support": int(self.support),
            "fill": float(self.fill),
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Fixture":
        return cls(
            footprint=np.array(d["footprint"], dtype=np.float64),
            z0=float(d["z0"]),
            z1=float(d["z1"]),
            support=int(d.get("support", 0)),
            fill=float(d.get("fill", 0.0)),
            kind=d.get("kind", "fixture"),
        )


@dataclass
class Structure:
    """The room read as architecture rather than as points.

    `floor_z` is 0.0 by construction (the canonicaliser puts it there) but is
    kept explicit so a twin that skipped canonicalisation still round-trips.
    `ceiling_z` is None for an unroofed capture. `footprint` is an (M, 2) XY
    polygon in CCW order describing the floor outline.
    """

    floor_z: float = 0.0
    ceiling_z: float | None = None
    planes: list[Plane] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)
    footprint: np.ndarray | None = None
    # A ceiling the sweep bounded but never returned from. It is kept out of
    # `ceiling_z` so that every consumer which measures keeps reading None and
    # keeps saying "not captured"; it exists so that the ones which *draw* the
    # room, or bound its volume, have something better than the top of the
    # points to use -- and have to name it as inference when they do.
    ceiling_z_inferred: float | None = None
    #: returns | carve | none -- where `ceiling_z`/`ceiling_z_inferred` came from
    ceiling_source: str = "none"
    #: Where the footprint polygon came from, which decides what may be drawn
    #: from it. "cells": the boundary of the wall-arrangement's interior cells,
    #: straight-walled by construction and fit to enclose a room. "raster": a
    #: traced occupancy contour, adequate on a dense pose-less import.
    #: "raster-sparse": the same contour on a capture that HAD poses but whose
    #: room solve declined -- good enough for an area estimate, and expressly
    #: not good enough to extrude a room shell from. "none": no footprint.
    footprint_source: str = "raster"
    #: One label per footprint edge (edge i is footprint[i] -> footprint[i+1]):
    #: returns | carve | frontier. None when the footprint is not cell-derived.
    footprint_edge_sources: list[str] | None = None
    #: Per-vertex inferred weight in [0, 1] for the footprint, aligned with
    #: `footprint`. None means the question was never asked.
    footprint_inferred: np.ndarray | None = None
    #: Solid contents the returns located inside the room -- counters,
    #: islands, appliances -- fitted as extruded prisms. Empty when the
    #: question was never asked (pose-less imports, declined room solves).
    fixtures: list[Fixture] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.footprint is not None:
            self.footprint = np.asarray(self.footprint, dtype=np.float64).reshape(-1, 2)
        if self.footprint_inferred is not None:
            self.footprint_inferred = np.asarray(
                self.footprint_inferred, dtype=np.float32
            ).reshape(-1)

    @property
    def ceiling_height(self) -> float | None:
        if self.ceiling_z is None:
            return None
        return float(self.ceiling_z - self.floor_z)

    @property
    def drawable_ceiling_z(self) -> float | None:
        """The best available cap on the room, measured or not.

        Anything that renders or bounds the room should use this and say which
        it got; anything that reports a dimension should use `ceiling_z` and
        report nothing when it is None.
        """
        return self.ceiling_z if self.ceiling_z is not None else self.ceiling_z_inferred

    @property
    def floor_area(self) -> float:
        """Shoelace area of the footprint polygon, m^2. Zero if unknown."""
        if self.footprint is None or len(self.footprint) < 3:
            return 0.0
        p = self.footprint
        q = np.roll(p, -1, axis=0)
        return float(abs(np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1])) / 2.0)

    def walls(self) -> list[Plane]:
        return [p for p in self.planes if p.kind == "wall"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "floor_z": float(self.floor_z),
            "ceiling_z": None if self.ceiling_z is None else float(self.ceiling_z),
            "planes": [p.to_dict() for p in self.planes],
            "openings": [o.to_dict() for o in self.openings],
            "footprint": None if self.footprint is None else self.footprint.tolist(),
            "ceiling_z_inferred": (
                None if self.ceiling_z_inferred is None else float(self.ceiling_z_inferred)
            ),
            "ceiling_source": self.ceiling_source,
            "footprint_source": self.footprint_source,
            "footprint_edge_sources": (
                None if self.footprint_edge_sources is None
                else list(self.footprint_edge_sources)
            ),
            "footprint_inferred": (
                None if self.footprint_inferred is None
                else [float(v) for v in self.footprint_inferred]
            ),
            "fixtures": [f.to_dict() for f in self.fixtures],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Structure":
        fp = d.get("footprint")
        fpi = d.get("footprint_inferred")
        return cls(
            floor_z=float(d.get("floor_z", 0.0)),
            ceiling_z=None if d.get("ceiling_z") is None else float(d["ceiling_z"]),
            planes=[Plane.from_dict(p) for p in d.get("planes", [])],
            openings=[Opening.from_dict(o) for o in d.get("openings", [])],
            footprint=None if fp is None else np.array(fp, dtype=np.float64),
            ceiling_z_inferred=(
                None if d.get("ceiling_z_inferred") is None else float(d["ceiling_z_inferred"])
            ),
            ceiling_source=d.get("ceiling_source", "none"),
            footprint_source=d.get("footprint_source", "raster"),
            footprint_edge_sources=(
                None if d.get("footprint_edge_sources") is None
                else list(d["footprint_edge_sources"])
            ),
            footprint_inferred=(
                None if fpi is None else np.array(fpi, dtype=np.float32)
            ),
            fixtures=[Fixture.from_dict(f) for f in d.get("fixtures", [])],
        )


# ---------------------------------------------------------------------------
# where on Earth, and where the scanner walked
# ---------------------------------------------------------------------------


@dataclass
class Georeference:
    """Ties twin space to the planet.

    `heading_deg` is the true-north bearing of the twin's +X axis, clockwise,
    in degrees: 0 means +X points north, 90 means +X points east. Everything
    solar downstream needs this and lat/lon; nothing else does.

    `elevation_m` is the floor's height above sea level, used only for horizon
    and terrain work later. `heading_source` records how we know the heading
    (`user`, `arkit`, `assumed`) because an assumed heading must not be quoted
    as a measurement.
    """

    latitude: float
    longitude: float
    heading_deg: float = 0.0
    elevation_m: float = 0.0
    heading_source: str = "assumed"

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError(f"latitude {self.latitude} out of range")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError(f"longitude {self.longitude} out of range")
        self.heading_deg = float(self.heading_deg) % 360.0

    @property
    def utc_offset_hours(self) -> float:
        """Solar-time offset implied by longitude. Not a civil timezone -- it
        is deliberately the sun's offset, because that is what daylight work
        needs and it never disagrees with itself across a DST boundary."""
        return self.longitude / 15.0

    def enu_from_twin(self) -> np.ndarray:
        """3x3 rotation taking twin XYZ into local East-North-Up.

        With heading h (bearing of +X), +X maps to (sin h, cos h, 0) in ENU and
        +Y, being 90 degrees counter-clockwise from +X in twin space, maps to
        the bearing h - 90.
        """
        h = math.radians(self.heading_deg)
        x_enu = np.array([math.sin(h), math.cos(h), 0.0])
        y_enu = np.array([math.sin(h - math.pi / 2), math.cos(h - math.pi / 2), 0.0])
        z_enu = np.array([0.0, 0.0, 1.0])
        return np.stack([x_enu, y_enu, z_enu], axis=1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Georeference":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


@dataclass
class CaptureBounds:
    """Where the scanner actually was, i.e. what the twin is entitled to claim.

    Outside this volume the geometry is extrapolation, and the sweep must not
    propose a tripod there. `hull_xy` is an (M, 2) CCW polygon of the traversed
    floor area; `z_range` is the (min, max) height of the camera path.
    `camera_positions` is kept when the export gave us poses, else None.
    """

    hull_xy: np.ndarray
    z_range: tuple[float, float] = (0.0, 0.0)
    camera_positions: np.ndarray | None = None
    source: str = "inferred"  # inferred | poses

    def __post_init__(self) -> None:
        self.hull_xy = np.asarray(self.hull_xy, dtype=np.float64).reshape(-1, 2)
        self.z_range = (float(self.z_range[0]), float(self.z_range[1]))
        if self.camera_positions is not None:
            self.camera_positions = np.asarray(
                self.camera_positions, dtype=np.float64
            ).reshape(-1, 3)

    @property
    def area(self) -> float:
        if len(self.hull_xy) < 3:
            return 0.0
        p = self.hull_xy
        q = np.roll(p, -1, axis=0)
        return float(abs(np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1])) / 2.0)

    def contains(self, points_xy: np.ndarray) -> np.ndarray:
        """Boolean mask of which XY points fall inside the hull polygon.

        Ray-crossing test, vectorised over points. Points exactly on an edge
        are not guaranteed either way, which is fine for a 'is this a legal
        standpoint' question.
        """
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        poly = self.hull_xy
        if len(poly) < 3:
            return np.zeros(len(pts), dtype=bool)
        inside = np.zeros(len(pts), dtype=bool)
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            straddles = (yi > pts[:, 1]) != (yj > pts[:, 1])
            with np.errstate(divide="ignore", invalid="ignore"):
                x_cross = (xj - xi) * (pts[:, 1] - yi) / (yj - yi) + xi
            inside ^= straddles & (pts[:, 0] < x_cross)
            j = i
        return inside

    def to_dict(self) -> dict[str, Any]:
        return {
            "hull_xy": self.hull_xy.tolist(),
            "z_range": list(self.z_range),
            "source": self.source,
            "has_camera_positions": self.camera_positions is not None,
        }


# ---------------------------------------------------------------------------
# quality: the part that lets us say "1:1" without lying
# ---------------------------------------------------------------------------


@dataclass
class QAReport:
    """Evidence that the twin is metrically trustworthy, or evidence it isn't.

    Every field is a measurement of the twin against itself or against known
    architectural priors. `checks` holds named pass/warn/fail verdicts with a
    human-readable message; `metrics` holds the raw numbers. The CLI prints
    this and `verdict` gates whether we call the twin filming-ready.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    checks: list[dict[str, str]] = field(default_factory=list)
    verdict: str = "unknown"  # pass | warn | fail | unknown

    def add(self, name: str, status: str, message: str) -> None:
        if status not in {"pass", "warn", "fail", "info"}:
            raise ValueError(f"bad check status {status!r}")
        self.checks.append({"name": name, "status": status, "message": message})

    def finalize(self) -> "QAReport":
        statuses = {c["status"] for c in self.checks}
        if "fail" in statuses:
            self.verdict = "fail"
        elif "warn" in statuses:
            self.verdict = "warn"
        elif statuses:
            self.verdict = "pass"
        else:
            self.verdict = "unknown"
        return self

    def failures(self) -> list[dict[str, str]]:
        return [c for c in self.checks if c["status"] == "fail"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "checks": self.checks,
            "verdict": self.verdict,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QAReport":
        return cls(
            metrics=dict(d.get("metrics", {})),
            checks=list(d.get("checks", [])),
            verdict=d.get("verdict", "unknown"),
        )


# ---------------------------------------------------------------------------
# the twin
# ---------------------------------------------------------------------------


@dataclass
class Twin:
    """A canonicalised, metric digital twin of one space.

    `name` is a slug. `points` is always present. `mesh` is present when the
    source had one or we reconstructed one. `canonical_transform` is the 4x4
    that was applied to the *source* geometry to get here, kept so we can map a
    measurement back to the original file's coordinates.

    `provenance` records where it came from and what we did: source path,
    source format, detected unit scale, software, timestamps, versions.
    """

    name: str
    points: PointCloud
    mesh: Mesh | None = None
    structure: Structure = field(default_factory=Structure)
    georeference: Georeference | None = None
    capture_bounds: CaptureBounds | None = None
    qa: QAReport = field(default_factory=QAReport)
    canonical_transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    provenance: dict[str, Any] = field(default_factory=dict)
    # The video projected back onto the room shell: a JPEG atlas plus the
    # layout that says which atlas rect belongs to which shell panel, in the
    # panel-uv convention `geom.shell.build_shell` emits. Optional because it
    # only exists for video captures whose room solve produced a shell; the
    # viewer falls back to provenance tints without it. Nothing measures off
    # a texture -- it is paint on fitted architecture, and the layout carries
    # per-panel coverage so the page can say where the video ran out.
    shell_texture: bytes | None = None
    shell_texture_format: str = "jpg"
    shell_texture_layout: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.canonical_transform = np.asarray(
            self.canonical_transform, dtype=np.float64
        ).reshape(4, 4)

    # -- convenience ------------------------------------------------------

    @property
    def bounds(self) -> np.ndarray:
        if self.mesh is not None and len(self.mesh.vertices):
            b = np.stack([self.points.bounds, self.mesh.bounds])
            return np.stack([b[:, 0].min(axis=0), b[:, 1].max(axis=0)])
        return self.points.bounds

    @property
    def extent(self) -> np.ndarray:
        b = self.bounds
        return b[1] - b[0]

    def summary(self) -> dict[str, Any]:
        b = self.bounds
        return {
            "name": self.name,
            "points": len(self.points),
            "faces": 0 if self.mesh is None else len(self.mesh),
            "extent_m": [round(float(v), 3) for v in (b[1] - b[0])],
            "ceiling_height_m": (
                None
                if self.structure.ceiling_height is None
                else round(self.structure.ceiling_height, 3)
            ),
            "floor_area_m2": round(self.structure.floor_area, 2),
            "openings": len(self.structure.openings),
            "verdict": self.qa.verdict,
        }

    # -- persistence ------------------------------------------------------
    #
    # A .twin is a zip: manifest.json plus raw .npy arrays. Zip because it is
    # one file to hand around, npy because loading a 20M-point cloud should be
    # a memcpy and not a JSON parse.

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix != ".twin":
            path = path.with_suffix(".twin")
        path.parent.mkdir(parents=True, exist_ok=True)

        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "structure": self.structure.to_dict(),
            "georeference": None
            if self.georeference is None
            else self.georeference.to_dict(),
            "capture_bounds": None
            if self.capture_bounds is None
            else self.capture_bounds.to_dict(),
            "qa": self.qa.to_dict(),
            "canonical_transform": self.canonical_transform.tolist(),
            "provenance": self.provenance,
            "arrays": [],
            "has_mesh": self.mesh is not None,
            "texture_format": None if self.mesh is None else self.mesh.texture_format,
            "shell_texture_format": (
                None if self.shell_texture is None else self.shell_texture_format
            ),
            "shell_texture_layout": self.shell_texture_layout,
        }

        arrays: dict[str, np.ndarray] = {"points/xyz": self.points.xyz.astype(np.float32)}
        if self.points.rgb is not None:
            arrays["points/rgb"] = self.points.rgb
        if self.points.normals is not None:
            arrays["points/normals"] = self.points.normals.astype(np.float32)
        if self.points.inferred is not None:
            arrays["points/inferred"] = self.points.inferred.astype(np.float32)
        if self.mesh is not None:
            arrays["mesh/vertices"] = self.mesh.vertices.astype(np.float32)
            arrays["mesh/faces"] = self.mesh.faces.astype(np.int32)
            if self.mesh.vertex_colors is not None:
                arrays["mesh/vertex_colors"] = self.mesh.vertex_colors
            if self.mesh.uv is not None:
                arrays["mesh/uv"] = self.mesh.uv.astype(np.float32)
            if self.mesh.filled is not None:
                arrays["mesh/filled"] = self.mesh.filled.astype(np.float32)
        if self.capture_bounds is not None and self.capture_bounds.camera_positions is not None:
            arrays["capture/camera_positions"] = self.capture_bounds.camera_positions.astype(
                np.float32
            )
        manifest["arrays"] = sorted(arrays)

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            for key, arr in arrays.items():
                import io as _io

                buf = _io.BytesIO()
                np.save(buf, arr, allow_pickle=False)
                zf.writestr(f"{key}.npy", buf.getvalue())
            if self.mesh is not None and self.mesh.texture is not None:
                zf.writestr(f"mesh/texture.{self.mesh.texture_format}", self.mesh.texture)
            if self.shell_texture is not None:
                zf.writestr(
                    f"shell/texture.{self.shell_texture_format}", self.shell_texture
                )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Twin":
        path = Path(path)
        with zipfile.ZipFile(path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
            version = manifest.get("schema_version", 0)
            if version > SCHEMA_VERSION:
                raise ValueError(
                    f"{path.name} is schema v{version}; this build understands v{SCHEMA_VERSION}"
                )
            names = set(zf.namelist())

            def arr(key: str) -> np.ndarray | None:
                fn = f"{key}.npy"
                if fn not in names:
                    return None
                import io as _io

                return np.load(_io.BytesIO(zf.read(fn)), allow_pickle=False)

            points = PointCloud(
                xyz=arr("points/xyz"),
                rgb=arr("points/rgb"),
                normals=arr("points/normals"),
                inferred=arr("points/inferred"),
            )
            mesh = None
            if manifest.get("has_mesh"):
                tex_fmt = manifest.get("texture_format") or "png"
                tex_name = f"mesh/texture.{tex_fmt}"
                mesh = Mesh(
                    vertices=arr("mesh/vertices"),
                    faces=arr("mesh/faces"),
                    vertex_colors=arr("mesh/vertex_colors"),
                    uv=arr("mesh/uv"),
                    filled=arr("mesh/filled"),
                    texture=zf.read(tex_name) if tex_name in names else None,
                    texture_format=tex_fmt,
                )
            cb = None
            if manifest.get("capture_bounds"):
                d = manifest["capture_bounds"]
                cb = CaptureBounds(
                    hull_xy=np.array(d["hull_xy"], dtype=np.float64).reshape(-1, 2),
                    z_range=tuple(d.get("z_range", (0.0, 0.0))),
                    camera_positions=arr("capture/camera_positions"),
                    source=d.get("source", "inferred"),
                )
            geo = (
                Georeference.from_dict(manifest["georeference"])
                if manifest.get("georeference")
                else None
            )
            shell_fmt = manifest.get("shell_texture_format") or "jpg"
            shell_name = f"shell/texture.{shell_fmt}"
            return cls(
                name=manifest["name"],
                points=points,
                mesh=mesh,
                structure=Structure.from_dict(manifest.get("structure", {})),
                georeference=geo,
                capture_bounds=cb,
                qa=QAReport.from_dict(manifest.get("qa", {})),
                canonical_transform=np.array(
                    manifest.get("canonical_transform", np.eye(4).tolist())
                ),
                provenance=manifest.get("provenance", {}),
                shell_texture=zf.read(shell_name) if shell_name in names else None,
                shell_texture_format=shell_fmt,
                shell_texture_layout=manifest.get("shell_texture_layout"),
            )


# ---------------------------------------------------------------------------
# small shared helpers -- here rather than in a util module because every
# geometry module needs them and this is the one file they all already import
# ---------------------------------------------------------------------------


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 rotation taking unit vector `a` onto unit vector `b` by the shortest
    arc. Handles the antiparallel case, which the naive Rodrigues form does not.
    """
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # antiparallel: rotate pi about any axis orthogonal to a
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, axis)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def rotation_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def homogeneous(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    m = np.eye(4)
    if rotation is not None:
        m[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if translation is not None:
        m[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return m


def chunked(n: int, size: int) -> Iterator[slice]:
    """Slices covering range(n) in `size`-sized pieces. Point clouds from a
    LiDAR sweep are big enough that the naive (N, N) intermediate is fatal."""
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))
