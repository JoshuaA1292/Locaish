"""The room as a surface, so that a room looks like one.

A twin made of points is a measurement and not a place. Nobody looking at two
million stereo returns floating in the dark can tell whether the far side of
the room is a wall or the end of the sweep, and on a capture where the walls
were textureless there is nothing at all where the walls are -- the honest
picture of that data is a shelf of jars hanging in space, which is exactly what
it looked like.

The shell is the other half of the same twin: the architecture the fitter
found, closed into a surface. It is built from the footprint rather than from
the wall planes directly, because a footprint is a closed polygon by
construction and therefore so is anything extruded from it -- there is no way
for this to produce a room with a hole in the corner, whatever the capture did.

Every face carries where it came from. A wall the returns located is drawn as
measurement; a wall the carve located and the returns never reached, a ceiling
that was inferred, a stretch of outline that was completed across a corner
nobody filmed -- all of it is drawn as inference and says so. The rule is the
one the rest of the pipeline lives by: a twin may contain inference, and it may
not let inference pass for measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..types import Opening, Structure

# How close an opening's centre has to be to a wall's line, and how well its
# normal has to agree, before the opening is cut out of that wall. Openings are
# fitted on the wall plane and the footprint edge is that plane, so the tolerance
# only has to absorb the difference between the fitted plane and the rastered
# outline that was snapped onto it.
OPENING_MATCH_M = 0.35
OPENING_MATCH_COS = 0.86

# An opening is trimmed back this far from the ends and the head of its wall, so
# that a doorway detected flush with a corner still leaves a sliver of jamb and
# the shell stays a closed surface rather than an open-sided box.
OPENING_INSET_M = 0.02

# Fallback height for a room with no ceiling of any kind. Only used for drawing
# and always marked inferred; nothing measures off it.
DEFAULT_WALL_HEIGHT_M = 2.40


@dataclass
class Shell:
    """A closed room surface, with provenance per face.

    Each vertex also carries which *panel* of the room it belongs to
    (`uv_part`: "floor", "ceiling", or "wall_<edge index>") and where it sits
    on that panel (`uv`, in metres in the panel's own 2D frame). `part_frames`
    maps each panel to its 3D frame as (origin, u_axis, v_axis), so a panel
    point (u, v) is `origin + u * u_axis + v * v_axis`. Together these let a
    texture baked per panel land on the shell without re-deriving any geometry:
    the baker and the renderer read the same numbers off the same object.
    """

    vertices: np.ndarray
    faces: np.ndarray
    # 0.0 where a face sits on measured returns, 1.0 where it was inferred
    inferred: np.ndarray
    # floor | ceiling | wall, one per face
    parts: list[str] = field(default_factory=list)
    # per-vertex panel key and panel-local (u, v) in metres
    uv_part: list[str] = field(default_factory=list)
    uv: np.ndarray | None = None
    # panel key -> (origin(3,), u_axis(3,), v_axis(3,))
    part_frames: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)
        self.inferred = np.asarray(self.inferred, dtype=np.float32).reshape(-1)
        if self.uv is not None:
            self.uv = np.asarray(self.uv, dtype=np.float64).reshape(-1, 2)

    @property
    def area(self) -> float:
        v = self.vertices
        a, b, c = v[self.faces[:, 0]], v[self.faces[:, 1]], v[self.faces[:, 2]]
        return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())

    def measured_fraction(self) -> float:
        """Share of the shell's area that stands on returns."""
        if not len(self.faces):
            return 0.0
        v = self.vertices
        a, b, c = v[self.faces[:, 0]], v[self.faces[:, 1]], v[self.faces[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        weight = 1.0 - self.inferred[self.faces].mean(axis=1)
        total = float(areas.sum())
        return float((areas * weight).sum() / total) if total > 0 else 0.0


def _signed_area(poly: np.ndarray) -> float:
    q = np.roll(poly, -1, axis=0)
    return float(np.sum(poly[:, 0] * q[:, 1] - q[:, 0] * poly[:, 1]) / 2.0)


def triangulate(poly: np.ndarray) -> np.ndarray:
    """Ear clipping for a simple polygon, returned CCW.

    Written out rather than pulled in because the alternatives all arrive with
    a geometry stack behind them, and a room outline is a few dozen vertices of
    a simple polygon -- the one case ear clipping handles without conditions.
    Delaunay is not a substitute: it triangulates the convex hull, which quietly
    fills in the notch of an L-shaped room and reports the missing corner as
    floor.
    """
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    n = len(p)
    if n < 3:
        return np.zeros((0, 3), dtype=np.int64)
    if _signed_area(p) < 0:
        p = p[::-1]
        flip = True
    else:
        flip = False

    idx = list(range(n))
    out: list[tuple[int, int, int]] = []
    guard = 0
    while len(idx) > 3 and guard < 4 * n * n:
        guard += 1
        clipped = False
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = p[i0], p[i1], p[i2]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 1e-12:
                continue  # reflex or degenerate: not an ear
            others = [j for j in idx if j not in (i0, i1, i2)]
            if others and _any_inside(p[others], a, b, c):
                continue
            out.append((i0, i1, i2))
            idx.pop(k)
            clipped = True
            break
        if not clipped:
            break  # self-intersecting outline; keep what we have
    if len(idx) == 3:
        out.append((idx[0], idx[1], idx[2]))

    tris = np.array(out, dtype=np.int64) if out else np.zeros((0, 3), dtype=np.int64)
    if flip and len(tris):
        tris = (n - 1) - tris
        tris = tris[:, ::-1]
    return tris


def _any_inside(pts: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    v0, v1 = c - a, b - a
    v2 = pts - a
    d00 = float(v0 @ v0)
    d01 = float(v0 @ v1)
    d11 = float(v1 @ v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-18:
        return False
    d02 = v2 @ v0
    d12 = v2 @ v1
    u = (d11 * d02 - d01 * d12) / denom
    v = (d00 * d12 - d01 * d02) / denom
    return bool(np.any((u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)))


def _openings_on(
    openings: list[Opening], a: np.ndarray, b: np.ndarray
) -> list[tuple[float, float, float, float]]:
    """Openings cut into the wall a->b, as (u0, u1, v0, v1) in wall coordinates."""
    seg = b - a
    length = float(np.linalg.norm(seg))
    if length < 1e-6:
        return []
    d = seg / length
    inward = np.array([-d[1], d[0]])

    out = []
    for op in openings:
        # Only walk-through openings are cut out of the shell. A window hole
        # in a photographed wall deletes the very pixels that show whether it
        # was glass or backsplash -- and the detector's windows are reasoned
        # from absent returns, which a reflective splashback fakes perfectly.
        # The opening overlay still outlines every detection either way.
        if op.kind != "door":
            continue
        n = np.asarray(op.normal, dtype=np.float64)[:2]
        norm = float(np.linalg.norm(n))
        if norm < 1e-9 or abs(float(n @ inward) / norm) < OPENING_MATCH_COS:
            continue
        rel = np.asarray(op.center, dtype=np.float64)[:2] - a
        along = float(rel @ d)
        across = abs(float(rel @ inward))
        if across > OPENING_MATCH_M:
            continue
        u0 = along - op.width / 2.0
        u1 = along + op.width / 2.0
        if u1 <= OPENING_INSET_M or u0 >= length - OPENING_INSET_M:
            continue
        out.append(
            (
                max(OPENING_INSET_M, u0),
                min(length - OPENING_INSET_M, u1),
                float(op.sill_height),
                float(op.sill_height + op.height),
            )
        )
    return sorted(out)


def _wall_panels(
    a: np.ndarray,
    b: np.ndarray,
    z0: float,
    z1: float,
    cuts: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """The wall a->b as rectangles in (u, v), with the openings taken out.

    Slicing vertically rather than triangulating a polygon with holes: a wall
    is a rectangle and an opening is a rectangle, so the difference is always a
    handful of rectangles and never needs a general boolean.
    """
    length = float(np.linalg.norm(b - a))
    height = z1 - z0
    if length <= 1e-6 or height <= 1e-6:
        return []
    if not cuts:
        return [(0.0, length, 0.0, height)]

    panels: list[tuple[float, float, float, float]] = []
    u = 0.0
    for u0, u1, v0, v1 in cuts:
        u0 = max(u0, u)
        if u0 > u:
            panels.append((u, u0, 0.0, height))
        v0 = max(0.0, min(height, v0))
        v1 = max(0.0, min(height, v1))
        if v0 > 1e-6:
            panels.append((u0, u1, 0.0, v0))  # under the sill
        if v1 < height - 1e-6:
            panels.append((u0, u1, v1, height))  # over the head
        u = max(u, u1)
    if u < length:
        panels.append((u, length, 0.0, height))
    return [p for p in panels if p[1] - p[0] > 1e-6 and p[3] - p[2] > 1e-6]


def wall_provenance(
    footprint: np.ndarray, walls: list, *, tol_m: float = 0.30
) -> dict[int, bool]:
    """Which footprint edges stand on a wall the returns actually located.

    Matching by geometry rather than by bookkeeping: the footprint was snapped
    onto these planes, so an edge that came from one lies along it to within a
    few centimetres, and an edge that is the frontier of the sweep lies along
    nothing. `inlier_count` is zero on a wall the carve placed and the returns
    never reached, which is exactly the distinction the shell has to draw.
    """
    poly = np.asarray(footprint, dtype=np.float64).reshape(-1, 2)
    out: dict[int, bool] = {}
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        seg = b - a
        length = float(np.linalg.norm(seg))
        if length < 1e-6:
            continue
        d = seg / length
        inward = np.array([-d[1], d[0]])
        mid = 0.5 * (a + b)
        for wall in walls:
            wn = np.asarray(wall.normal, dtype=np.float64)[:2]
            if float(wn @ inward) < OPENING_MATCH_COS:
                continue
            if abs(float(mid @ wn) - float(wall.offset)) > tol_m:
                continue
            out[i] = int(getattr(wall, "inlier_count", 0)) > 0
            break
    return out


def build_shell(
    structure: Structure,
    *,
    wall_measured: dict[int, bool] | None = None,
    footprint_inferred: np.ndarray | None = None,
    include_fixtures: bool = False,
) -> Shell | None:
    """Close the fitted room into a surface.

    `wall_measured` maps footprint edge index to whether that wall stands on
    returns; anything unnamed is treated as inference, which is the safe
    direction to be wrong in.
    """
    poly = structure.footprint
    if poly is None or len(poly) < 3:
        return None
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if _signed_area(poly) < 0:
        poly = poly[::-1]
        if footprint_inferred is not None:
            footprint_inferred = np.asarray(footprint_inferred)[::-1]

    floor_z = float(structure.floor_z)
    cap = structure.drawable_ceiling_z
    ceiling_inferred = structure.ceiling_z is None
    if cap is None or cap - floor_z < 0.5:
        cap = floor_z + DEFAULT_WALL_HEIGHT_M
        ceiling_inferred = True

    verts: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    inferred: list[float] = []
    parts: list[str] = []
    uv_part: list[str] = []
    uvs: list[tuple[float, float]] = []
    part_frames: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # -- floor: the outline, triangulated, facing up ------------------------
    tris = triangulate(poly)
    if len(tris):
        floor_pts = np.column_stack([poly, np.full(len(poly), floor_z)])
        fw = (
            np.asarray(footprint_inferred, dtype=np.float64)
            if footprint_inferred is not None and len(footprint_inferred) == len(poly)
            else np.zeros(len(poly))
        )
        base = len(verts)
        verts.extend(floor_pts)
        inferred.extend(fw.tolist())
        uv_part.extend(["floor"] * len(poly))
        uvs.extend((float(x), float(y)) for x, y in poly)
        for tri in tris:
            faces.append((base + int(tri[0]), base + int(tri[1]), base + int(tri[2])))
            parts.append("floor")
        part_frames["floor"] = (
            np.array([0.0, 0.0, floor_z]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )

        # -- ceiling: the same outline, flipped ----------------------------
        ceil_pts = np.column_stack([poly, np.full(len(poly), cap)])
        base = len(verts)
        verts.extend(ceil_pts)
        weight = 1.0 if ceiling_inferred else 0.0
        inferred.extend([weight] * len(ceil_pts))
        uv_part.extend(["ceiling"] * len(poly))
        uvs.extend((float(x), float(y)) for x, y in poly)
        for tri in tris:
            faces.append((base + int(tri[2]), base + int(tri[1]), base + int(tri[0])))
            parts.append("ceiling")
        part_frames["ceiling"] = (
            np.array([0.0, 0.0, cap]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )

    # -- walls: one extrusion per edge, openings cut out --------------------
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        seg = b - a
        length = float(np.linalg.norm(seg))
        if length < 1e-6:
            continue
        d = seg / length
        key = f"wall_{i}"
        part_frames[key] = (
            np.array([a[0], a[1], floor_z]),
            np.array([d[0], d[1], 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        cuts = _openings_on(list(structure.openings), a, b)
        weight = 0.0 if (wall_measured or {}).get(i, False) else 1.0
        for u0, u1, v0, v1 in _wall_panels(a, b, floor_z, cap, cuts):
            p00 = np.array([a[0] + d[0] * u0, a[1] + d[1] * u0, floor_z + v0])
            p10 = np.array([a[0] + d[0] * u1, a[1] + d[1] * u1, floor_z + v0])
            p11 = np.array([a[0] + d[0] * u1, a[1] + d[1] * u1, floor_z + v1])
            p01 = np.array([a[0] + d[0] * u0, a[1] + d[1] * u0, floor_z + v1])
            quad = np.stack([p00, p10, p11, p01])
            base = len(verts)
            verts.extend(quad)
            inferred.extend([weight] * 4)
            uv_part.extend([key] * 4)
            uvs.extend([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
            # wound so the normal points into the room, which is where the
            # camera is and therefore the side that has to be lit
            for tri in np.array([[0, 2, 1], [0, 3, 2]]):
                faces.append((base + int(tri[0]), base + int(tri[1]), base + int(tri[2])))
                parts.append("wall")

    # -- fixtures: the room's solid contents, as extruded prisms ------------
    #
    # Off by default: with a dense enough cloud the returns themselves carry
    # the contents, and a massing prism drawn over them reads as a box, not
    # a counter. The fitted fixtures stay in the Structure for consumers
    # that need solids (camera planning, occlusion); a caller that wants
    # them drawn asks. Wound with their normals pointing OUT of the solid --
    # the opposite of the envelope, and both are correct under the same
    # back-face cull. Sides get a wall-style (u, v) frame so the texture
    # baker can paint them exactly like walls.
    fixtures = (getattr(structure, "fixtures", []) or []) if include_fixtures else []
    for fi, fx in enumerate(fixtures):
        fpoly = np.asarray(fx.footprint, dtype=np.float64).reshape(-1, 2)
        if len(fpoly) < 3:
            continue
        if _signed_area(fpoly) < 0:
            fpoly = fpoly[::-1]
        z0, z1 = float(fx.z0), float(fx.z1)
        if z1 - z0 < 1e-3:
            continue
        m = len(fpoly)
        for e in range(m):
            a, b = fpoly[e], fpoly[(e + 1) % m]
            seg = b - a
            length = float(np.linalg.norm(seg))
            if length < 1e-6:
                continue
            d = seg / length
            key = f"fix{fi}_s{e}"
            part_frames[key] = (
                np.array([a[0], a[1], z0]),
                np.array([d[0], d[1], 0.0]),
                np.array([0.0, 0.0, 1.0]),
            )
            p00 = np.array([a[0], a[1], z0])
            p10 = np.array([b[0], b[1], z0])
            p11 = np.array([b[0], b[1], z1])
            p01 = np.array([a[0], a[1], z1])
            base = len(verts)
            verts.extend([p00, p10, p11, p01])
            inferred.extend([0.0] * 4)
            uv_part.extend([key] * 4)
            uvs.extend([(0.0, 0.0), (length, 0.0), (length, z1 - z0), (0.0, z1 - z0)])
            for tri in ((0, 1, 2), (0, 2, 3)):  # outward
                faces.append((base + tri[0], base + tri[1], base + tri[2]))
                parts.append("fixture")
        ftris = triangulate(fpoly)
        if len(ftris):
            key = f"fix{fi}_top"
            part_frames[key] = (
                np.array([0.0, 0.0, z1]),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
            )
            base = len(verts)
            verts.extend(np.column_stack([fpoly, np.full(m, z1)]))
            inferred.extend([0.0] * m)
            uv_part.extend([key] * m)
            uvs.extend((float(px), float(py)) for px, py in fpoly)
            for tri in ftris:  # CCW from above = facing up = outward
                faces.append((base + int(tri[0]), base + int(tri[1]), base + int(tri[2])))
                parts.append("fixture")
            if z0 > floor_z + 0.01:  # floating solids need an underside
                key = f"fix{fi}_bot"
                part_frames[key] = (
                    np.array([0.0, 0.0, z0]),
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                )
                base = len(verts)
                verts.extend(np.column_stack([fpoly, np.full(m, z0)]))
                inferred.extend([0.0] * m)
                uv_part.extend([key] * m)
                uvs.extend((float(px), float(py)) for px, py in fpoly)
                for tri in ftris:
                    faces.append((base + int(tri[2]), base + int(tri[1]), base + int(tri[0])))
                    parts.append("fixture")

    if not faces:
        return None
    return Shell(
        vertices=np.array(verts),
        faces=np.array(faces),
        inferred=np.array(inferred, dtype=np.float32),
        parts=parts,
        uv_part=uv_part,
        uv=np.array(uvs, dtype=np.float64),
        part_frames=part_frames,
    )


def shell_from_structure(structure: Structure) -> Shell | None:
    """The shell exactly as the viewer will draw it.

    One definition of "which edges are measured" shared by everything that
    consumes a shell -- the texture baker runs at ingest and the payload
    builder runs at render, and a texel baked for wall panel 3 must land on
    the wall panel 3 the page draws. Centralising the construction is what
    makes that a tautology instead of a synchronisation problem.
    """
    if structure.footprint is None or len(structure.footprint) < 3:
        return None
    if structure.footprint_source == "raster-sparse":
        return None
    if (
        structure.footprint_edge_sources is not None
        and len(structure.footprint_edge_sources) == len(structure.footprint)
    ):
        wall_measured = {
            i: src == "returns" for i, src in enumerate(structure.footprint_edge_sources)
        }
    else:
        wall_measured = wall_provenance(structure.footprint, structure.walls())
    return build_shell(
        structure,
        wall_measured=wall_measured,
        footprint_inferred=structure.footprint_inferred,
    )


__all__ = ["Shell", "build_shell", "shell_from_structure", "triangulate", "wall_provenance"]
