"""A first-party PLY reader and writer.

PLY is the format every consumer scanner eventually offers, and it is the one
format we refuse to delegate. Third-party loaders quietly normalise things we
need to see: they drop unknown properties (which is exactly where a gaussian
splat keeps its colour), they cast float colours to bytes with the wrong
convention (Luma writes 0..1, everyone else writes 0..255), they discard the
comment block that names the capture app, and several of them silently
triangulate or reorder faces. A twin that claims to be 1:1 cannot be built on a
reader whose losses it cannot enumerate, so this module parses the header
itself and reports what it found in `raw_header`.

The parser is buffer-oriented rather than record-oriented: a fixed-width
element becomes one `np.frombuffer` over a structured dtype, and a face element
with list properties is read in runs of equal list length, which makes the
common all-triangles or all-quads mesh a single vectorised read as well.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..types import Mesh, PointCloud
from . import ScanImport
from .detect import detect_software, detect_unit_hint

# The zeroth-order spherical-harmonic coefficient. A 3D gaussian splat stores
# its base colour as an SH DC term, and this is the constant that turns it back
# into linear RGB.
C0 = 0.28209479177387814

# Every scalar spelling the PLY spec allows plus the sized aliases that newer
# exporters (and every splat trainer) emit.
_SCALAR_DTYPES: dict[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "long": "i8",
    "int64": "i8",
    "ulong": "u8",
    "uint64": "u8",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

_BYTE_ORDER = {
    "ascii": "=",
    "binary_little_endian": "<",
    "binary_big_endian": ">",
}

# Names an exporter might use for the same quantity, lowercased.
_ALIASES: dict[str, tuple[str, ...]] = {
    "x": ("x", "px", "posx", "position_x"),
    "y": ("y", "py", "posy", "position_y"),
    "z": ("z", "pz", "posz", "position_z"),
    "nx": ("nx", "normal_x", "normalx"),
    "ny": ("ny", "normal_y", "normaly"),
    "nz": ("nz", "normal_z", "normalz"),
    "red": ("red", "r", "diffuse_red", "c_red"),
    "green": ("green", "g", "diffuse_green", "c_green"),
    "blue": ("blue", "b", "diffuse_blue", "c_blue"),
    "alpha": ("alpha", "a", "diffuse_alpha"),
}

_FACE_INDEX_NAMES = ("vertex_indices", "vertex_index", "vertex_indexes", "indices")

# Above this many distinct run lengths in one face element we stop paying the
# numpy setup cost per run and finish the element with a plain struct loop. A
# real mesh has one or two runs; a file that needs thousands is pathological
# and the loop is the cheaper of two bad options.
_MAX_LIST_RUNS = 2048

_HEADER_LIMIT = 4 * 1024 * 1024


@dataclass
class PlyProperty:
    """One `property` line. `is_list` records the `property list c v name` form,
    where every record carries its own element count."""

    name: str
    value_type: str
    is_list: bool = False
    count_type: str | None = None

    @property
    def value_np(self) -> str:
        return _SCALAR_DTYPES[self.value_type]

    @property
    def count_np(self) -> str:
        assert self.count_type is not None
        return _SCALAR_DTYPES[self.count_type]


@dataclass
class PlyElement:
    name: str
    count: int
    properties: list[PlyProperty] = field(default_factory=list)

    @property
    def has_lists(self) -> bool:
        return any(p.is_list for p in self.properties)


@dataclass
class PlyHeader:
    fmt: str
    version: str
    elements: list[PlyElement]
    comments: list[str]
    obj_info: list[str]
    data_offset: int
    text: str

    @property
    def order(self) -> str:
        return _BYTE_ORDER[self.fmt]

    def element(self, *names: str) -> PlyElement | None:
        lookup = {e.name.lower(): e for e in self.elements}
        for n in names:
            if n in lookup:
                return lookup[n]
        return None


# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------


def parse_header(path: str | Path) -> PlyHeader:
    """Read the text header and return where the payload starts.

    The header is read in chunks rather than by lines because the payload that
    follows it may be binary, and a line-oriented read over binary data can
    consume an arbitrary amount of it looking for a newline that is really a
    coordinate byte.
    """
    path = Path(path)
    buf = bytearray()
    with open(path, "rb") as fh:
        while b"end_header" not in buf:
            chunk = fh.read(65536)
            if not chunk:
                raise ValueError(f"{path.name}: no end_header, not a PLY file")
            buf += chunk
            if len(buf) > _HEADER_LIMIT:
                raise ValueError(f"{path.name}: PLY header exceeds {_HEADER_LIMIT} bytes")
    marker = buf.index(b"end_header")
    nl = buf.index(b"\n", marker)
    data_offset = nl + 1
    text = bytes(buf[:data_offset]).decode("utf-8", errors="replace")

    fmt = "ascii"
    version = "1.0"
    elements: list[PlyElement] = []
    comments: list[str] = []
    obj_info: list[str] = []

    lines = [ln.strip("\r\n").strip() for ln in text.splitlines()]
    if not lines or lines[0].strip().lower() != "ply":
        raise ValueError(f"{path.name}: first line is not 'ply'")
    for line in lines[1:]:
        if not line:
            continue
        tag, _, rest = line.partition(" ")
        tag = tag.lower()
        rest = rest.strip()
        if tag == "format":
            parts = rest.split()
            fmt = parts[0].lower()
            if fmt not in _BYTE_ORDER:
                raise ValueError(f"{path.name}: unknown PLY format {fmt!r}")
            version = parts[1] if len(parts) > 1 else "1.0"
        elif tag == "comment":
            comments.append(rest)
        elif tag == "obj_info":
            obj_info.append(rest)
        elif tag == "element":
            parts = rest.split()
            elements.append(PlyElement(name=parts[0], count=int(parts[1])))
        elif tag == "property":
            if not elements:
                raise ValueError(f"{path.name}: property before any element")
            parts = rest.split()
            if parts[0].lower() == "list":
                ctype, vtype, name = parts[1].lower(), parts[2].lower(), parts[3]
                _require_type(ctype, path)
                _require_type(vtype, path)
                elements[-1].properties.append(
                    PlyProperty(name=name, value_type=vtype, is_list=True, count_type=ctype)
                )
            else:
                vtype, name = parts[0].lower(), parts[1]
                _require_type(vtype, path)
                elements[-1].properties.append(PlyProperty(name=name, value_type=vtype))
        elif tag == "end_header":
            break
    return PlyHeader(
        fmt=fmt,
        version=version,
        elements=elements,
        comments=comments,
        obj_info=obj_info,
        data_offset=data_offset,
        text=text,
    )


def _require_type(name: str, path: Path) -> None:
    if name not in _SCALAR_DTYPES:
        raise ValueError(f"{path.name}: unsupported PLY scalar type {name!r}")


# ---------------------------------------------------------------------------
# payload
# ---------------------------------------------------------------------------


def read_ply(path: str | Path) -> ScanImport:
    """Parse a PLY into a ScanImport, keeping every property we understand.

    Handles the three PLY encodings, arbitrary property order, list-typed face
    indices with fan triangulation of quads and n-gons, byte and float colour
    conventions, and gaussian-splat clouds whose colour lives in `f_dc_*`.
    """
    path = Path(path)
    header = parse_header(path)
    warnings: list[str] = []
    data = _read_payload(path, header, warnings)

    raw_header: dict[str, Any] = {
        "format": header.fmt,
        "version": header.version,
        "comments": list(header.comments),
        "obj_info": list(header.obj_info),
        "elements": {
            e.name: {
                "count": e.count,
                "properties": [p.name for p in e.properties],
            }
            for e in header.elements
        },
    }

    vertex_elem = header.element("vertex", "vertices", "point", "points")
    if vertex_elem is None:
        raise ValueError(f"{path.name}: no vertex element in PLY header")
    vertex = data[vertex_elem.name]
    points = _vertex_cloud(vertex_elem, vertex, warnings, raw_header)

    mesh = _face_mesh(header, data, points, warnings)

    software = detect_software(*header.comments, *header.obj_info)
    if raw_header.get("is_gaussian_splat") and software is None:
        software = "gaussian_splat"

    return ScanImport(
        points=points,
        mesh=mesh,
        source_path=path,
        source_format="ply",
        software=software,
        camera_positions=_camera_positions(header, data),
        unit_hint=detect_unit_hint(*header.comments, *header.obj_info),
        warnings=warnings,
        raw_header=raw_header,
    )


def _read_payload(
    path: Path, header: PlyHeader, warnings: list[str]
) -> dict[str, dict[str, Any]]:
    """Return {element_name: {property_name: array}} for the whole file.

    List properties come back as a `(counts, flat_values)` tuple so that a face
    element with mixed arities survives to the triangulator intact.
    """
    with open(path, "rb") as fh:
        fh.seek(header.data_offset)
        payload = fh.read()
    if header.fmt == "ascii":
        return _read_ascii(payload, header, warnings)
    return _read_binary(payload, header, warnings)


# -- ascii ------------------------------------------------------------------


def _read_ascii(
    payload: bytes, header: PlyHeader, warnings: list[str]
) -> dict[str, dict[str, Any]]:
    text = payload.decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    out: dict[str, dict[str, Any]] = {}
    cursor = 0
    for elem in header.elements:
        take = lines[cursor : cursor + elem.count]
        if len(take) < elem.count:
            warnings.append(
                f"ascii PLY truncated: element {elem.name} declared {elem.count} rows, "
                f"found {len(take)}"
            )
            elem.count = len(take)
        cursor += len(take)
        if elem.has_lists:
            out[elem.name] = _ascii_list_element(elem, take)
        else:
            out[elem.name] = _ascii_fixed_element(elem, take, warnings)
    return out


def _ascii_fixed_element(
    elem: PlyElement, lines: list[str], warnings: list[str]
) -> dict[str, Any]:
    """Bulk-parse a rectangular ASCII block.

    Everything is read as float64 first and cast afterwards: it is one C-level
    conversion for the whole element instead of a per-field Python parse, and
    float64 is exact for every PLY integer type up to 2**53, which no vertex
    index or colour byte comes close to.
    """
    ncols = len(elem.properties)
    if not lines or ncols == 0:
        return {p.name: np.zeros(0, dtype=p.value_np) for p in elem.properties}
    flat = np.array(" ".join(lines).split(), dtype=np.float64)
    if flat.size != len(lines) * ncols:
        warnings.append(
            f"ascii PLY element {elem.name}: expected {ncols} fields per row, "
            f"got {flat.size} values over {len(lines)} rows; parsing row by row"
        )
        rows = np.zeros((len(lines), ncols), dtype=np.float64)
        for i, ln in enumerate(lines):
            vals = ln.split()[:ncols]
            rows[i, : len(vals)] = np.array(vals, dtype=np.float64)
        flat = rows.reshape(-1)
    table = flat.reshape(len(lines), ncols)
    return {
        p.name: table[:, i].astype(p.value_np, copy=False)
        for i, p in enumerate(elem.properties)
    }


def _ascii_list_element(elem: PlyElement, lines: list[str]) -> dict[str, Any]:
    """Row-by-row parse for ASCII elements with variable-length properties.

    There is no rectangular shape to exploit here, but ASCII PLY meshes are
    small by construction -- nobody ships a million-face room as text -- so the
    per-row cost is acceptable where it would not be for the binary path.
    """
    counts: dict[str, list[int]] = {p.name: [] for p in elem.properties if p.is_list}
    values: dict[str, list[np.ndarray]] = {p.name: [] for p in elem.properties if p.is_list}
    scalars: dict[str, list[float]] = {p.name: [] for p in elem.properties if not p.is_list}
    for line in lines:
        toks = line.split()
        pos = 0
        for p in elem.properties:
            if p.is_list:
                k = int(float(toks[pos]))
                pos += 1
                values[p.name].append(np.array(toks[pos : pos + k], dtype=np.float64))
                counts[p.name].append(k)
                pos += k
            else:
                scalars[p.name].append(float(toks[pos]))
                pos += 1
    out: dict[str, Any] = {}
    for p in elem.properties:
        if p.is_list:
            flat = (
                np.concatenate(values[p.name])
                if values[p.name]
                else np.zeros(0, dtype=np.float64)
            )
            out[p.name] = (
                np.array(counts[p.name], dtype=np.int64),
                flat.astype(p.value_np, copy=False),
            )
        else:
            out[p.name] = np.array(scalars[p.name], dtype=p.value_np)
    return out


# -- binary -----------------------------------------------------------------


def _read_binary(
    payload: bytes, header: PlyHeader, warnings: list[str]
) -> dict[str, dict[str, Any]]:
    order = header.order
    out: dict[str, dict[str, Any]] = {}
    offset = 0
    for elem in header.elements:
        if elem.has_lists:
            cols, offset = _binary_list_element(payload, offset, elem, order, warnings)
        else:
            cols, offset = _binary_fixed_element(payload, offset, elem, order, warnings)
        out[elem.name] = cols
    return out


def _binary_fixed_element(
    payload: bytes, offset: int, elem: PlyElement, order: str, warnings: list[str]
) -> tuple[dict[str, Any], int]:
    names = _unique_names([p.name for p in elem.properties])
    dt = np.dtype([(n, order + p.value_np) for n, p in zip(names, elem.properties)])
    available = (len(payload) - offset) // max(dt.itemsize, 1)
    count = min(elem.count, available)
    if count < elem.count:
        warnings.append(
            f"binary PLY truncated: element {elem.name} declared {elem.count} rows, "
            f"only {count} present"
        )
        elem.count = count
    view = np.frombuffer(payload, dtype=dt, count=count, offset=offset)
    cols = {p.name: np.ascontiguousarray(view[n]) for n, p in zip(names, elem.properties)}
    return cols, offset + count * dt.itemsize


def _binary_list_element(
    payload: bytes, offset: int, elem: PlyElement, order: str, warnings: list[str]
) -> tuple[dict[str, Any], int]:
    """Read an element containing list properties.

    Fast path: exactly one list property, which is the shape of every face
    element in the wild. Records are consumed in runs that share a list length,
    so an all-triangle mesh is one `frombuffer` and a mesh of triangles then
    quads is two. Anything more tangled -- several list properties, or list
    lengths that alternate constantly -- falls back to a struct loop, which is
    slow but is the only thing that is actually correct there.
    """
    lists = [p for p in elem.properties if p.is_list]
    if len(lists) != 1:
        return _binary_list_slow(payload, offset, elem, order, warnings, start=0)

    lp = lists[0]
    idx = elem.properties.index(lp)
    pre = elem.properties[:idx]
    post = elem.properties[idx + 1 :]
    pre_size = sum(np.dtype(p.value_np).itemsize for p in pre)
    post_size = sum(np.dtype(p.value_np).itemsize for p in post)
    csize = np.dtype(lp.count_np).itemsize
    vsize = np.dtype(lp.value_np).itemsize

    counts_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []
    scalar_chunks: dict[str, list[np.ndarray]] = {p.name: [] for p in pre + post}

    done = 0
    runs = 0
    while done < elem.count:
        if offset + pre_size + csize > len(payload):
            warnings.append(
                f"binary PLY truncated inside element {elem.name} at record {done}"
            )
            elem.count = done
            break
        k = int(
            np.frombuffer(
                payload, dtype=order + lp.count_np, count=1, offset=offset + pre_size
            )[0]
        )
        stride = pre_size + csize + k * vsize + post_size
        if stride <= 0:
            raise ValueError(f"PLY element {elem.name}: zero-length record")
        available = (len(payload) - offset) // stride
        take = min(elem.count - done, available)
        if take <= 0:
            warnings.append(
                f"binary PLY truncated inside element {elem.name} at record {done}"
            )
            elem.count = done
            break

        fields: dict[str, Any] = {"names": [], "formats": [], "offsets": []}
        pos = 0
        for p in pre:
            fields["names"].append(p.name)
            fields["formats"].append(order + p.value_np)
            fields["offsets"].append(pos)
            pos += np.dtype(p.value_np).itemsize
        fields["names"].append("__n")
        fields["formats"].append(order + lp.count_np)
        fields["offsets"].append(pos)
        pos += csize
        fields["names"].append("__v")
        fields["formats"].append((order + lp.value_np, (k,)) if k else (order + lp.value_np, (0,)))
        fields["offsets"].append(pos)
        pos += k * vsize
        for p in post:
            fields["names"].append(p.name)
            fields["formats"].append(order + p.value_np)
            fields["offsets"].append(pos)
            pos += np.dtype(p.value_np).itemsize
        fields["names"] = _unique_names(fields["names"])
        dt = np.dtype({**fields, "itemsize": stride})

        view = np.frombuffer(payload, dtype=dt, count=take, offset=offset)
        run_counts = view["__n"]
        mismatch = np.flatnonzero(run_counts != k)
        if mismatch.size:
            take = int(mismatch[0])
            if take == 0:
                # cannot happen -- record 0's count is what defined k -- but
                # refusing to spin forever is cheaper than trusting that.
                return _binary_list_slow(
                    payload, offset, elem, order, warnings, start=done
                )
            view = view[:take]

        counts_chunks.append(np.full(take, k, dtype=np.int64))
        value_chunks.append(np.ascontiguousarray(view["__v"]).reshape(take, k))
        for n, p in zip(dt.names, [*pre, None, None, *post]):
            if p is not None:
                scalar_chunks[p.name].append(np.ascontiguousarray(view[n]))

        offset += take * stride
        done += take
        runs += 1
        if runs > _MAX_LIST_RUNS and done < elem.count:
            warnings.append(
                f"PLY element {elem.name}: list lengths change every few records; "
                "finishing with the slow per-record reader"
            )
            rest, offset = _binary_list_slow(
                payload, offset, elem, order, warnings, start=done
            )
            counts_chunks.append(rest[lp.name][0])
            value_chunks.append(rest[lp.name][1])
            for p in pre + post:
                scalar_chunks[p.name].append(rest[p.name])
            done = elem.count
            break

    counts = (
        np.concatenate(counts_chunks) if counts_chunks else np.zeros(0, dtype=np.int64)
    )
    flat = (
        np.concatenate([c.reshape(-1) for c in value_chunks])
        if value_chunks
        else np.zeros(0, dtype=lp.value_np)
    )
    cols: dict[str, Any] = {lp.name: (counts, flat)}
    for p in pre + post:
        chunks = scalar_chunks[p.name]
        cols[p.name] = (
            np.concatenate(chunks) if chunks else np.zeros(0, dtype=p.value_np)
        )
    return cols, offset


def _binary_list_slow(
    payload: bytes,
    offset: int,
    elem: PlyElement,
    order: str,
    warnings: list[str],
    *,
    start: int,
) -> tuple[dict[str, Any], int]:
    """Per-record reader for elements the vectorised path cannot describe."""
    endian = "<" if order in "<=" else ">"
    counts: dict[str, list[int]] = {}
    values: dict[str, list[Any]] = {}
    scalars: dict[str, list[Any]] = {}
    for p in elem.properties:
        (counts if p.is_list else scalars).setdefault(p.name, [])
        if p.is_list:
            values[p.name] = []

    fmt_of = {t: _struct_code(t) for t in _SCALAR_DTYPES}
    for _ in range(start, elem.count):
        for p in elem.properties:
            if p.is_list:
                code = endian + fmt_of[p.count_type or "uchar"]
                (k,) = struct.unpack_from(code, payload, offset)
                offset += struct.calcsize(code)
                vcode = endian + str(k) + fmt_of[p.value_type]
                vals = struct.unpack_from(vcode, payload, offset)
                offset += struct.calcsize(vcode)
                counts[p.name].append(int(k))
                values[p.name].extend(vals)
            else:
                code = endian + fmt_of[p.value_type]
                (v,) = struct.unpack_from(code, payload, offset)
                offset += struct.calcsize(code)
                scalars[p.name].append(v)
    out: dict[str, Any] = {}
    for p in elem.properties:
        if p.is_list:
            out[p.name] = (
                np.array(counts[p.name], dtype=np.int64),
                np.array(values[p.name], dtype=p.value_np),
            )
        else:
            out[p.name] = np.array(scalars[p.name], dtype=p.value_np)
    return out, offset


def _struct_code(ply_type: str) -> str:
    return {
        "i1": "b",
        "u1": "B",
        "i2": "h",
        "u2": "H",
        "i4": "i",
        "u4": "I",
        "i8": "q",
        "u8": "Q",
        "f4": "f",
        "f8": "d",
    }[_SCALAR_DTYPES[ply_type]]


def _unique_names(names: list[str]) -> list[str]:
    """numpy structured dtypes reject duplicate field names; PLY does not."""
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}__{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


# ---------------------------------------------------------------------------
# interpretation
# ---------------------------------------------------------------------------


def _lookup(cols: dict[str, Any], key: str) -> str | None:
    lower = {k.lower(): k for k in cols}
    for alias in _ALIASES.get(key, (key,)):
        if alias in lower:
            return lower[alias]
    return None


def _vertex_cloud(
    elem: PlyElement,
    cols: dict[str, Any],
    warnings: list[str],
    raw_header: dict[str, Any],
) -> PointCloud:
    kx, ky, kz = (_lookup(cols, k) for k in ("x", "y", "z"))
    if not (kx and ky and kz):
        raise ValueError("PLY vertex element has no x/y/z properties")
    xyz = np.stack(
        [np.asarray(cols[kx], dtype=np.float64) for kx in (kx, ky, kz)], axis=1
    )

    prop_types = {p.name.lower(): p.value_type for p in elem.properties}
    names = {p.name.lower() for p in elem.properties}

    splat_props = sorted(
        n
        for n in names
        if n.startswith(("f_dc_", "f_rest_", "scale_", "rot_")) or n == "opacity"
    )
    is_splat = all(f"f_dc_{i}" in names for i in range(3)) and "opacity" in names
    if is_splat:
        raw_header["is_gaussian_splat"] = True
        raw_header["splat_properties"] = splat_props
        raw_header["splat_sh_degree"] = _sh_degree(names)

    rgb = _vertex_colors(cols, prop_types, warnings)
    if rgb is None and is_splat:
        f_dc = np.stack(
            [np.asarray(cols[_key(cols, f"f_dc_{i}")], dtype=np.float64) for i in range(3)],
            axis=1,
        )
        rgb = (np.clip(0.5 + C0 * f_dc, 0.0, 1.0) * 255.0).astype(np.uint8)
        warnings.append(
            "gaussian-splat PLY: colour reconstructed from the f_dc spherical-harmonic "
            "DC term, view-dependent terms ignored"
        )
    elif rgb is not None and is_splat:
        raw_header["splat_had_explicit_rgb"] = True

    normals = None
    knx, kny, knz = (_lookup(cols, k) for k in ("nx", "ny", "nz"))
    if knx and kny and knz:
        normals = np.stack(
            [np.asarray(cols[k], dtype=np.float64) for k in (knx, kny, knz)], axis=1
        )
        length = np.linalg.norm(normals, axis=1, keepdims=True)
        degenerate = int(np.count_nonzero(length < 1e-9))
        if degenerate:
            warnings.append(
                f"{degenerate} vertex normals in the PLY are zero-length; left unnormalised"
            )
        normals = normals / np.where(length < 1e-9, 1.0, length)

    if is_splat and "opacity" in names:
        opacity = np.asarray(cols[_key(cols, "opacity")], dtype=np.float64)
        # trainers store logit-opacity; a sigmoid below 0.1 is a floater
        alpha = 1.0 / (1.0 + np.exp(-opacity))
        faint = float(np.mean(alpha < 0.1))
        raw_header["splat_faint_fraction"] = faint
        if faint > 0.2:
            warnings.append(
                f"{faint:.0%} of splats have opacity below 0.1; they are kept but are "
                "likely reconstruction floaters rather than surface"
            )

    return PointCloud(xyz=xyz, rgb=rgb, normals=normals)


def _key(cols: dict[str, Any], name: str) -> str:
    lower = {k.lower(): k for k in cols}
    return lower[name]


def _sh_degree(names: set[str]) -> int:
    rest = sum(1 for n in names if n.startswith("f_rest_"))
    # (deg+1)^2 - 1 coefficients per channel, three channels
    for deg in range(0, 5):
        if rest == 3 * ((deg + 1) ** 2 - 1):
            return deg
    return -1


def _vertex_colors(
    cols: dict[str, Any], prop_types: dict[str, str], warnings: list[str]
) -> np.ndarray | None:
    """Recover 0..255 colour, detecting the float 0..1 convention.

    Luma and most splat trainers write colour as float, and the convention is
    not declared anywhere in the header. Byte-typed properties are unambiguous;
    for every other width it is the *observed* range that decides, never the
    declared one, and either branch is recorded so that a washed-out twin can
    be traced back to this decision.

    Two failure modes are being defended against, both of which used to produce
    a confidently wrong image. The first is a PLY that stores ordinary 0..255
    colour in an int, uint or ushort property: dividing by the declared width
    turns the whole cloud black while reporting a correct-sounding "downscaled
    to 8-bit". The second is an export whose three channels have different
    widths, where taking the dtype from red alone and casting the stacked array
    to uint8 wraps the wider channels modulo 256 in silence, so a green of 300
    becomes 44. Both are cured by letting the observed range decide and by
    clipping instead of casting; the mixed-width file is additionally decoded a
    channel at a time, because a file that disagrees with itself about its own
    colour depth is the one case where the channels are genuinely on different
    scales, and it is reported rather than quietly repaired.
    """
    keys = [_lookup(cols, k) for k in ("red", "green", "blue")]
    if not all(keys):
        return None
    stack = np.stack([np.asarray(cols[k]) for k in keys], axis=1)
    declared = [prop_types.get(str(k).lower(), "uchar") for k in keys]
    kinds = [_SCALAR_DTYPES.get(name, "u1") for name in declared]
    if any(kind in ("f4", "f8") for kind in kinds):
        stack = stack.astype(np.float64)
        peak = float(np.nanmax(np.abs(stack))) if stack.size else 0.0
        if peak <= 1.0 + 1e-6:
            warnings.append(
                "PLY vertex colours are float in 0..1; scaled to 0..255"
            )
            return np.clip(stack * 255.0, 0, 255).astype(np.uint8)
        warnings.append(
            f"PLY vertex colours are float with peak {peak:.1f}; read as 0..255"
        )
        return np.clip(stack, 0, 255).astype(np.uint8)

    wide = [name for name, kind in zip(declared, kinds) if np.dtype(kind).itemsize > 1]
    if len(set(kinds)) > 1:
        widths = ", ".join(
            f"{chan}={name}" for chan, name in zip(("red", "green", "blue"), declared)
        )
        warnings.append(
            f"PLY vertex colour channels have mixed integer widths ({widths}); "
            "each channel decoded on its own observed range"
        )
        out = np.zeros(stack.shape, dtype=np.uint8)
        for i, chan in enumerate(("red", "green", "blue")):
            out[:, i] = _integer_channel_to_byte(stack[:, i], f" ({chan})", warnings)
        return out

    out = _integer_channel_to_byte(stack, "", warnings)
    if wide and out.size and int(np.max(stack)) <= 255:
        warnings.append(
            f"PLY vertex colours are declared {wide[0]} but never exceed 255; "
            "read as 0..255 rather than downscaled to black"
        )
    return out


def _integer_channel_to_byte(
    values: np.ndarray, label: str, warnings: list[str]
) -> np.ndarray:
    """Fit integer colour into 0..255 using the range the file actually uses.

    The declared property width is not evidence of the colour depth -- writers
    pad 8-bit colour into int and uint all the time -- so the shift comes from
    the largest value present. That also covers the 10- and 12-bit channels
    some photogrammetry tools emit, which a fixed byte-wise divide would darken
    by a factor of four or sixteen. Negatives have no colour meaning and are
    floored at black by the clip rather than being allowed to wrap.
    """
    peak = int(np.max(values)) if values.size else 0
    if peak > 255:
        bits = peak.bit_length()
        warnings.append(
            f"PLY vertex colours{label} peak at {peak}, i.e. {bits}-bit; "
            "downscaled to 8-bit"
        )
        values = values / float(1 << (bits - 8))
    return np.clip(values, 0, 255).astype(np.uint8)


def fan_triangulate(counts: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """Turn variable-length polygon index runs into an (F, 3) triangle array.

    Fan triangulation from each polygon's first vertex, vectorised over all
    polygons at once. It is only exact for convex polygons, but exported room
    geometry has no concave n-gons in practice and every alternative needs a
    plane fit per face, which is a geometry decision this module has no business
    making.
    """
    counts = np.asarray(counts, dtype=np.int64)
    flat = np.asarray(flat)
    if counts.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    keep = counts >= 3
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    counts, starts = counts[keep], starts[keep]
    if counts.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    tris = counts - 2
    total = int(tris.sum())
    base = np.repeat(starts, tris)
    run_start = np.repeat(np.concatenate([[0], np.cumsum(tris)[:-1]]), tris)
    step = np.arange(total, dtype=np.int64) - run_start
    idx = np.stack([base, base + 1 + step, base + 2 + step], axis=1)
    return flat[idx].astype(np.int64)


def _face_mesh(
    header: PlyHeader,
    data: dict[str, dict[str, Any]],
    points: PointCloud,
    warnings: list[str],
) -> Mesh | None:
    elem = header.element("face", "faces")
    if elem is None:
        return None
    cols = data.get(elem.name, {})
    key = None
    lower = {k.lower(): k for k in cols}
    for candidate in _FACE_INDEX_NAMES:
        if candidate in lower:
            key = lower[candidate]
            break
    if key is None:
        for p in elem.properties:
            if p.is_list:
                key = p.name
                break
    if key is None or not isinstance(cols.get(key), tuple):
        warnings.append("PLY face element has no list-typed index property; mesh skipped")
        return None
    counts, flat = cols[key]
    if counts.size == 0:
        return None
    faces = fan_triangulate(counts, flat)
    ngons = int(np.count_nonzero(counts > 3))
    if ngons:
        warnings.append(
            f"{ngons} PLY faces were quads or n-gons and were fan-triangulated"
        )
    nv = len(points.xyz)
    if faces.size and (faces.min() < 0 or faces.max() >= nv):
        bad = int(np.count_nonzero((faces < 0) | (faces >= nv)))
        warnings.append(
            f"{bad} face indices fall outside the {nv} vertices; those faces dropped"
        )
        faces = faces[np.all((faces >= 0) & (faces < nv), axis=1)]
    if len(faces) == 0:
        return None
    uv = _face_uv(cols, counts, flat, nv, warnings)
    return Mesh(
        vertices=points.xyz,
        faces=faces.astype(np.int32),
        vertex_colors=points.rgb,
        uv=uv,
    )


def _face_uv(
    cols: dict[str, Any],
    counts: np.ndarray,
    flat: np.ndarray,
    nvertices: int,
    warnings: list[str],
) -> np.ndarray | None:
    """Per-corner texcoords collapsed onto vertices, when the file has them.

    A PLY texture lives on the face corners, so a vertex shared by two faces
    with different UVs cannot be represented by `Mesh.uv` at all. Splitting the
    vertex would fix the UV and break the promise that `mesh.vertices` and
    `points.xyz` are the same array, which the whole pipeline relies on, so the
    last corner wins and the warning says so.
    """
    lower = {k.lower(): k for k in cols}
    key = next((lower[n] for n in ("texcoord", "texcoords") if n in lower), None)
    if key is None or not isinstance(cols.get(key), tuple):
        return None
    tc_counts, tc_flat = cols[key]
    if tc_flat.size == 0 or tc_counts.shape != counts.shape or not np.all(tc_counts == counts * 2):
        warnings.append("PLY texcoord list does not match the face arity; UVs dropped")
        return None
    corner_uv = np.asarray(tc_flat, dtype=np.float32).reshape(-1, 2)
    corner_vertex = np.asarray(flat, dtype=np.int64).reshape(-1)
    if len(corner_vertex) != len(corner_uv):
        warnings.append("PLY texcoord count does not match the index count; UVs dropped")
        return None
    inside = (corner_vertex >= 0) & (corner_vertex < nvertices)
    uv = np.zeros((nvertices, 2), dtype=np.float32)
    uv[corner_vertex[inside]] = corner_uv[inside]
    shared = len(corner_vertex) - len(np.unique(corner_vertex))
    warnings.append(
        "PLY carried per-corner texcoords; collapsed to per-vertex UVs"
        + (f", {shared} corners overwrote a shared vertex" if shared else "")
    )
    return uv


def _camera_positions(
    header: PlyHeader, data: dict[str, dict[str, Any]]
) -> np.ndarray | None:
    """Bundler-style PLYs carry a `camera` element with the viewpoint in it.

    It is the only place a PLY ever states where the scanner was, and knowing
    that is what stops the sweep planner proposing a tripod inside a wall.
    """
    elem = header.element("camera", "cameras", "view")
    if elem is None:
        return None
    cols = data.get(elem.name, {})
    lower = {k.lower(): k for k in cols}
    trio = [lower.get(n) for n in ("view_px", "view_py", "view_pz")]
    if not all(trio):
        trio = [lower.get(n) for n in ("x", "y", "z")]
    if not all(trio):
        return None
    try:
        return np.stack([np.asarray(cols[k], dtype=np.float64).reshape(-1) for k in trio], axis=1)
    except (KeyError, ValueError):
        return None


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def write_ply(
    path: str | Path,
    points: PointCloud | None = None,
    mesh: Mesh | None = None,
    *,
    binary: bool = True,
    big_endian: bool = False,
    double: bool = False,
) -> Path:
    """Write a PLY. Exactly one of `points` or `mesh` supplies the vertices.

    A PLY has a single vertex element, so a call carrying both a cloud and a
    mesh has no unambiguous meaning and is rejected rather than resolved by
    picking one and dropping the other on the floor.

    Coordinates default to float32, which is what every viewer expects and
    which costs about 2 micrometres of precision at room scale; `double=True`
    gives an exact round trip when a test needs one. ASCII output is printed
    with `%.9g`, the shortest format that recovers a float32 exactly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if points is not None and mesh is not None:
        raise ValueError(
            "write_ply takes points or mesh, not both: a PLY has one vertex element"
        )
    if points is None and mesh is None:
        raise ValueError("write_ply needs either points or mesh")

    if mesh is not None:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        colors = mesh.vertex_colors
        normals = None
        faces = np.asarray(mesh.faces, dtype=np.int32).reshape(-1, 3)
    else:
        assert points is not None
        vertices = np.asarray(points.xyz, dtype=np.float64)
        colors = points.rgb
        normals = points.normals
        faces = None

    vt = "double" if double else "float"
    vnp = "f8" if double else "f4"
    order = ">" if (binary and big_endian) else "<"
    fmt = (
        "ascii"
        if not binary
        else ("binary_big_endian" if big_endian else "binary_little_endian")
    )

    head = ["ply", f"format {fmt} 1.0", "comment created by locaish"]
    head.append(f"element vertex {len(vertices)}")
    head += [f"property {vt} {axis}" for axis in ("x", "y", "z")]
    if normals is not None:
        head += [f"property {vt} {axis}" for axis in ("nx", "ny", "nz")]
    if colors is not None:
        head += [f"property uchar {c}" for c in ("red", "green", "blue")]
    if faces is not None:
        head.append(f"element face {len(faces)}")
        head.append("property list uchar int vertex_indices")
    head.append("end_header")
    header_bytes = ("\n".join(head) + "\n").encode("ascii")

    columns: list[tuple[str, np.ndarray]] = [
        (name, vertices[:, i].astype(vnp)) for i, name in enumerate(("x", "y", "z"))
    ]
    if normals is not None:
        n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        columns += [(name, n[:, i].astype(vnp)) for i, name in enumerate(("nx", "ny", "nz"))]
    if colors is not None:
        c = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        columns += [
            (name, c[:, i]) for i, name in enumerate(("red", "green", "blue"))
        ]

    with open(path, "wb") as fh:
        fh.write(header_bytes)
        if binary:
            dt = np.dtype(
                [
                    (name, order + ("u1" if name in ("red", "green", "blue") else vnp))
                    for name, _ in columns
                ]
            )
            table = np.empty(len(vertices), dtype=dt)
            for name, column in columns:
                table[name] = column
            fh.write(table.tobytes())
            if faces is not None:
                fdt = np.dtype([("n", "u1"), ("v", order + "i4", (3,))])
                ftab = np.empty(len(faces), dtype=fdt)
                ftab["n"] = 3
                ftab["v"] = faces
                fh.write(ftab.tobytes())
        else:
            table = np.column_stack([c.astype(np.float64) for _, c in columns])
            fmts = ["%.9g"] * (3 + (3 if normals is not None else 0))
            fmts += ["%d"] * (3 if colors is not None else 0)
            np.savetxt(fh, table, fmt=fmts, delimiter=" ")
            if faces is not None:
                ftab = np.concatenate(
                    [np.full((len(faces), 1), 3, dtype=np.int64), faces.astype(np.int64)],
                    axis=1,
                )
                np.savetxt(fh, ftab, fmt="%d", delimiter=" ")
    return path
