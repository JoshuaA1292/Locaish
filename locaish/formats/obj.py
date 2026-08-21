"""Wavefront OBJ reader, including the material and texture chain.

OBJ is what Polycam, RoomPlan and every desktop tool export when the user picks
"mesh", and it is the format most likely to arrive half broken: indices may be
negative and relative to the file position, faces may be n-gons, the same
vertex may carry different texcoords in different faces, and the texture is not
in the file at all -- it is named by a `map_Kd` line in a `.mtl` sitting beside
it, which is exactly the file a user forgets to include when they zip up an
export.

Parsing is done in one pass that only classifies lines, followed by bulk numeric
conversion: `np.array(text.split(), float)` moves the per-vertex work into C,
which matters because a Polycam room OBJ is routinely a million lines long.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from ..types import Mesh, PointCloud
from . import ScanImport
from .detect import detect_software, detect_unit_hint
from .ply import fan_triangulate

_TEXTURE_SUFFIXES = {
    ".png": "png",
    ".jpg": "jpg",
    ".jpeg": "jpg",
    ".webp": "webp",
    ".bmp": "bmp",
    ".tif": "tiff",
    ".tiff": "tiff",
}

# map_Kd can be prefixed with options ("-s 1 1 1 -o 0 0 0 tex.png"); the value
# options take numeric arguments, so the filename is what survives after the
# option tokens are stripped.
_MTL_OPTION_ARGS = {
    "-blendu": 1, "-blendv": 1, "-boost": 1, "-mm": 2, "-o": 3, "-s": 3,
    "-t": 3, "-texres": 1, "-clamp": 1, "-bm": 1, "-imfchan": 1, "-type": 1,
}

_REF_SPLIT = re.compile(r"[\s/]+")


def read_obj(path: str | Path) -> ScanImport:
    """Parse an OBJ (plus its .mtl and texture) into a ScanImport."""
    path = Path(path)
    warnings: list[str] = []
    raw = path.read_bytes().decode("utf-8", errors="replace")
    if "\t" in raw:
        raw = raw.replace("\t", " ")

    v_lines: list[str] = []
    vt_lines: list[str] = []
    vn_lines: list[str] = []
    f_lines: list[str] = []
    f_base: list[tuple[int, int, int]] = []
    comments: list[str] = []
    mtllibs: list[str] = []
    groups: list[str] = []
    usemtl: list[tuple[str, int]] = []
    counts = [0, 0, 0]  # v, vt, vn seen so far

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line[0] == "#":
            if len(comments) < 64:
                comments.append(line[1:].strip())
            continue
        tag, _, rest = line.partition(" ")
        rest = rest.strip()
        if tag == "v":
            v_lines.append(rest)
            counts[0] += 1
        elif tag == "vt":
            vt_lines.append(rest)
            counts[1] += 1
        elif tag == "vn":
            vn_lines.append(rest)
            counts[2] += 1
        elif tag == "f":
            f_lines.append(rest)
            f_base.append((counts[0], counts[1], counts[2]))
        elif tag == "mtllib":
            mtllibs.extend(rest.split())
        elif tag == "usemtl":
            usemtl.append((rest, len(f_lines)))
        elif tag in ("o", "g"):
            if rest and len(groups) < 64:
                groups.append(rest)

    xyz, colors = _parse_vertices(v_lines, warnings)
    uv_table = _parse_floats(vt_lines, 2, warnings, "vt")
    del vn_lines  # `Mesh` has no per-vertex normal slot, so vn is dead weight

    raw_header: dict[str, Any] = {
        "comments": comments,
        "mtllib": mtllibs,
        "groups": groups,
        "materials": [m for m, _ in usemtl],
        "counts": {"v": counts[0], "vt": counts[1], "vn": counts[2], "f": len(f_lines)},
    }

    mesh = None
    if f_lines:
        mesh = _build_mesh(
            f_lines, f_base, xyz, colors, uv_table, counts, warnings, raw_header
        )
    if mesh is not None:
        texture, tex_fmt, mtl_meta = _load_texture(
            path, mtllibs, usemtl, len(f_lines), warnings
        )
        raw_header["mtl"] = mtl_meta
        if texture is not None:
            mesh.texture = texture
            mesh.texture_format = tex_fmt

    points = PointCloud(xyz=xyz, rgb=colors)
    software = detect_software(*comments, *groups, path.name)
    return ScanImport(
        points=points,
        mesh=mesh,
        source_path=path,
        source_format="obj",
        software=software,
        camera_positions=None,
        unit_hint=detect_unit_hint(*comments),
        warnings=warnings,
        raw_header=raw_header,
    )


# ---------------------------------------------------------------------------
# numeric blocks
# ---------------------------------------------------------------------------


def _parse_floats(
    lines: list[str], width: int, warnings: list[str], tag: str
) -> np.ndarray:
    """Bulk-convert a block of equal-width float lines, truncating to `width`."""
    if not lines:
        return np.zeros((0, width))
    flat = np.array(" ".join(lines).split(), dtype=np.float64)
    if flat.size % len(lines):
        warnings.append(f"OBJ {tag} lines have inconsistent field counts; parsed row by row")
        rows = np.zeros((len(lines), width))
        for i, ln in enumerate(lines):
            vals = np.array(ln.split()[:width], dtype=np.float64)
            rows[i, : len(vals)] = vals
        return rows
    found = flat.size // len(lines)
    table = flat.reshape(len(lines), found)
    if found < width:
        pad = np.zeros((len(lines), width - found))
        return np.concatenate([table, pad], axis=1)
    return table[:, :width]


def _parse_vertices(
    lines: list[str], warnings: list[str]
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read `v` lines, picking up the unofficial trailing-RGB extension.

    `v x y z r g b` is not in the OBJ spec but MeshLab, CloudCompare and several
    scanners emit it, and dropping the colour would cost the texture-based
    checks their only signal on an untextured export. The 0..1 versus 0..255
    convention is decided by the observed range, as it is in PLY.
    """
    if not lines:
        return np.zeros((0, 3)), None
    flat = np.array(" ".join(lines).split(), dtype=np.float64)
    if flat.size % len(lines):
        warnings.append("OBJ v lines have inconsistent field counts; parsed row by row")
        rows = np.zeros((len(lines), 3))
        for i, ln in enumerate(lines):
            vals = np.array(ln.split()[:3], dtype=np.float64)
            rows[i, : len(vals)] = vals
        return rows, None
    width = flat.size // len(lines)
    table = flat.reshape(len(lines), width)
    xyz = table[:, :3]
    if width >= 6:
        raw = table[:, 3:6]
        peak = float(np.nanmax(np.abs(raw))) if raw.size else 0.0
        if peak <= 1.0 + 1e-6:
            warnings.append("OBJ vertex colours are floats in 0..1; scaled to 0..255")
            rgb = np.clip(raw * 255.0, 0, 255).astype(np.uint8)
        else:
            warnings.append(f"OBJ vertex colours are floats peaking at {peak:.1f}; read as 0..255")
            rgb = np.clip(raw, 0, 255).astype(np.uint8)
        return xyz, rgb
    if width == 4:
        # homogeneous w; dividing through is what the spec says it means
        w = table[:, 3]
        safe = np.where(np.abs(w) < 1e-12, 1.0, w)
        if not np.allclose(safe, 1.0):
            warnings.append("OBJ v lines carry a w component; divided through")
            xyz = xyz / safe[:, None]
    return xyz, None


# ---------------------------------------------------------------------------
# faces
# ---------------------------------------------------------------------------


def _build_mesh(
    f_lines: list[str],
    f_base: list[tuple[int, int, int]],
    xyz: np.ndarray,
    colors: np.ndarray | None,
    uv_table: np.ndarray,
    counts: list[int],
    warnings: list[str],
    raw_header: dict[str, Any],
) -> Mesh | None:
    refs: list[str] = []
    arity = np.empty(len(f_lines), dtype=np.int64)
    for i, line in enumerate(f_lines):
        toks = line.split()
        arity[i] = len(toks)
        refs.extend(toks)
    if not refs:
        return None

    table = _parse_refs(refs, warnings)
    if table is None:
        return None
    fields = table.shape[1]

    base = np.repeat(np.array(f_base, dtype=np.int64), arity, axis=0)
    v_idx = _resolve(table[:, 0], base[:, 0])
    vt_idx = (
        _resolve(table[:, 1], base[:, 1])
        if fields >= 2
        else np.full(len(table), -1, dtype=np.int64)
    )

    bad = (v_idx < 0) | (v_idx >= len(xyz))
    if np.any(bad):
        warnings.append(
            f"{int(bad.sum())} OBJ face corners reference vertices outside the file; clamped"
        )
        v_idx = np.clip(v_idx, 0, max(len(xyz) - 1, 0))

    has_uv = bool(len(uv_table)) and bool(np.any(vt_idx >= 0))
    if has_uv:
        vt_idx = np.where((vt_idx >= 0) & (vt_idx < len(uv_table)), vt_idx, -1)
        pairs = np.stack([v_idx, vt_idx], axis=1)
        uniq, inverse = np.unique(pairs, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        vertices = xyz[uniq[:, 0]]
        uv = np.zeros((len(uniq), 2), dtype=np.float32)
        valid = uniq[:, 1] >= 0
        uv[valid] = uv_table[uniq[valid, 1]][:, :2]
        vertex_colors = None if colors is None else colors[uniq[:, 0]]
        corner_index = inverse
        if len(uniq) != len(xyz):
            raw_header["uv_split_vertices"] = int(len(uniq) - len(xyz))
    else:
        vertices = xyz
        uv = None
        vertex_colors = colors
        corner_index = v_idx

    faces = fan_triangulate(arity, corner_index)
    ngons = int(np.count_nonzero(arity > 3))
    if ngons:
        warnings.append(f"{ngons} OBJ faces were quads or n-gons and were fan-triangulated")
    degenerate = int(np.count_nonzero(arity < 3))
    if degenerate:
        warnings.append(f"{degenerate} OBJ faces had fewer than three corners and were dropped")
    if len(faces) == 0:
        return None
    return Mesh(
        vertices=vertices,
        faces=faces.astype(np.int32),
        vertex_colors=vertex_colors,
        uv=uv,
    )


def _parse_refs(refs: list[str], warnings: list[str]) -> np.ndarray | None:
    """Turn "v", "v/vt", "v//vn", "v/vt/vn" corner tokens into an (M, k) array.

    The fast path rewrites "//" as "/0/" so that a single regex split over the
    whole joined block yields a rectangular integer table -- one C-level parse
    for a million corners. It is only taken when the resulting token count
    matches exactly, which is the check that catches a file mixing reference
    forms between faces; those fall back to the per-token parse.
    """
    joined = " ".join(refs)
    if "/" not in joined:
        values = np.array(joined.split(), dtype=np.int64)
        if values.size == len(refs):
            return values.reshape(-1, 1)
    else:
        fields = refs[0].count("/") + 1
        tokens = _REF_SPLIT.split(joined.replace("//", "/0/").strip())
        if len(tokens) == len(refs) * fields:
            try:
                return np.array(tokens, dtype=np.int64).reshape(-1, fields)
            except ValueError:
                pass
    warnings.append("OBJ face references mix reference forms; parsed corner by corner")
    rows = np.zeros((len(refs), 3), dtype=np.int64)
    for i, ref in enumerate(refs):
        parts = ref.split("/")
        for j in range(min(3, len(parts))):
            rows[i, j] = int(parts[j]) if parts[j] else 0
    return rows


def _resolve(raw: np.ndarray, base: np.ndarray) -> np.ndarray:
    """OBJ indices are 1-based, and a negative one counts back from however many
    elements had been declared *at that line*, not from the end of the file.

    Getting that wrong only shows up on multi-object exports, where it silently
    stitches the wrong vertices together into a mesh that still looks plausible,
    so the per-line base count is carried through rather than assumed.
    """
    out = np.where(raw > 0, raw - 1, np.where(raw < 0, base + raw, -1))
    return out.astype(np.int64)


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------


def _load_texture(
    obj_path: Path,
    mtllibs: list[str],
    usemtl: list[tuple[str, int]],
    n_faces: int,
    warnings: list[str],
) -> tuple[bytes | None, str, dict[str, Any]]:
    """Follow mtllib -> newmtl -> map_Kd and load the image bytes.

    `Mesh` holds a single texture, so when the export uses several materials we
    take the one covering the most faces and warn about the rest; that keeps the
    dominant surface (the walls) correctly coloured instead of whichever
    material happened to be declared first.
    """
    meta: dict[str, Any] = {"files": [], "materials": {}, "missing": []}
    materials: dict[str, str] = {}
    for name in mtllibs:
        mtl_path = _resolve_sibling(obj_path, name)
        if mtl_path is None:
            meta["missing"].append(name)
            warnings.append(f"OBJ references {name!r} but no such .mtl is beside it")
            continue
        meta["files"].append(mtl_path.name)
        materials.update(_parse_mtl(mtl_path))
    meta["materials"] = dict(materials)
    if not materials:
        return None, "png", meta

    usage: dict[str, int] = {}
    for i, (name, start) in enumerate(usemtl):
        end = usemtl[i + 1][1] if i + 1 < len(usemtl) else n_faces
        usage[name] = usage.get(name, 0) + max(end - start, 0)
    ranked = sorted(materials, key=lambda m: -usage.get(m, 0))
    if len({materials[m] for m in ranked if m in materials}) > 1:
        warnings.append(
            "OBJ uses several diffuse textures; kept the one covering the most faces"
        )

    for name in ranked:
        rel = materials.get(name)
        if not rel:
            continue
        img = _resolve_sibling(obj_path, rel)
        if img is None:
            meta["missing"].append(rel)
            warnings.append(f"OBJ material {name!r} names texture {rel!r}, which is missing")
            continue
        suffix = _TEXTURE_SUFFIXES.get(img.suffix.lower(), "png")
        return img.read_bytes(), suffix, meta
    return None, "png", meta


def _parse_mtl(path: Path) -> dict[str, str]:
    materials: dict[str, str] = {}
    current: str | None = None
    text = path.read_bytes().decode("utf-8", errors="replace").replace("\t", " ")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tag, _, rest = line.partition(" ")
        tag = tag.lower()
        rest = rest.strip()
        if tag == "newmtl":
            current = rest
            materials.setdefault(current, "")
        elif tag in ("map_kd", "map_ka") and current is not None:
            if not materials.get(current) or tag == "map_kd":
                materials[current] = _strip_map_options(rest)
    return {k: v for k, v in materials.items()}


def _strip_map_options(value: str) -> str:
    tokens = value.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("-"):
            i += 1 + _MTL_OPTION_ARGS.get(tok.lower(), 1)
            continue
        out.append(tok)
        i += 1
    # a filename may contain spaces, in which case everything left is the name
    return " ".join(out).strip()


def _resolve_sibling(obj_path: Path, name: str) -> Path | None:
    """Find a referenced file, tolerating Windows paths and case differences.

    Exports zipped on one OS and unzipped on another routinely break the exact
    path, and a missing texture is not worth failing an import over, but a
    texture found by a slightly fuzzy match is worth taking.
    """
    if not name:
        return None
    name = name.replace("\\", "/")
    base = obj_path.parent
    candidates = [base / name, base / Path(name).name]
    for cand in candidates:
        if cand.is_file():
            return cand
    target = Path(name).name.lower()
    for directory in (base, base / "textures", base / "tex", base / "materials"):
        if not directory.is_dir():
            continue
        for entry in sorted(directory.iterdir()):
            if entry.is_file() and entry.name.lower() == target:
                return entry
    return None
