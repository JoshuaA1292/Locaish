"""glTF, GLB, STL and COLLADA imports, plus the PLY reader of last resort.

These are container formats rather than scan formats: a Polycam GLB is a scene
graph whose room is spread over several nodes, each with its own transform and
its own material, and reading only the first mesh gives a twin missing three
walls. So trimesh is used strictly as a decoder -- it turns bytes into arrays
and nothing else -- and this module does the flattening: every geometry is
pulled into world space through its node transform and concatenated into one
`Mesh`, which is the only shape the rest of the pipeline accepts.

The glTF JSON is parsed a second time, directly, for the two things trimesh
discards and we care about: the `asset.generator` banner that names the capture
app, and the camera nodes, which are the only record of where the scanner
actually stood.
"""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from ..types import Mesh, PointCloud
from . import ScanImport
from .detect import detect_software, detect_unit_hint

_TRIMESH_FILE_TYPE = {
    "glb": "glb",
    "gltf": "gltf",
    "stl": "stl",
    "dae": "dae",
    "off": "off",
    "ply": "ply",
    "obj": "obj",
}


def read_with_trimesh(path: str | Path, source_format: str | None = None) -> ScanImport:
    """Decode `path` with trimesh and flatten the result into a ScanImport."""
    import trimesh  # imported lazily: it is the slowest import in the tree

    path = Path(path)
    fmt = (source_format or path.suffix.lstrip(".")).lower()
    warnings: list[str] = []
    raw_header: dict[str, Any] = {"reader": "trimesh", "trimesh_version": trimesh.__version__}

    loaded = trimesh.load(
        str(path),
        file_type=_TRIMESH_FILE_TYPE.get(fmt),
        process=False,   # process=True merges vertices, which silently changes geometry
        force=None,
    )

    vertices, faces, colors, uv, texture, tex_fmt, cloud_xyz, cloud_rgb = _flatten(
        loaded, warnings
    )

    mesh = None
    if len(faces):
        mesh = Mesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=colors,
            uv=uv,
            texture=texture,
            texture_format=tex_fmt or "png",
        )

    if len(cloud_xyz):
        points = PointCloud(xyz=cloud_xyz, rgb=cloud_rgb)
    elif mesh is not None:
        points = PointCloud(xyz=mesh.vertices, rgb=mesh.vertex_colors)
    else:
        points = PointCloud(xyz=np.zeros((0, 3)))

    software = None
    cameras = None
    unit_hint = None
    if fmt in ("glb", "gltf"):
        doc = _gltf_json(path, warnings)
        if doc is not None:
            asset = doc.get("asset", {})
            raw_header["gltf_asset"] = asset
            raw_header["gltf_extensions_used"] = doc.get("extensionsUsed", [])
            # glTF declares metres, but consumer scanners violate that often
            # enough that treating it as a measurement would be a lie; the fact
            # is recorded and `scale` still has to prove it.
            raw_header["gltf_declares_metres"] = True
            # glTF is Y-up by definition. No axis swap happens here: `geom.align`
            # finds gravity from the floor plane, and a reader that pre-rotated
            # the cloud would leave it disagreeing with a scanner that ignored
            # the convention.
            raw_header["gltf_y_up_by_spec"] = True
            software = detect_software(
                asset.get("generator"), asset.get("copyright"), json.dumps(asset.get("extras", {}))
            )
            unit_hint = detect_unit_hint(asset.get("generator"), json.dumps(asset.get("extras", {})))
            cameras = _gltf_camera_positions(doc, warnings)
    if software is None:
        software = detect_software(
            *_metadata_strings(getattr(loaded, "metadata", None)), path.name
        )

    return ScanImport(
        points=points,
        mesh=mesh,
        source_path=path,
        source_format=fmt,
        software=software,
        camera_positions=cameras,
        unit_hint=unit_hint,
        warnings=warnings,
        raw_header=raw_header,
    )


# ---------------------------------------------------------------------------
# scene flattening
# ---------------------------------------------------------------------------


def _flatten(loaded: Any, warnings: list[str]) -> tuple[
    np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, bytes | None, str | None,
    np.ndarray, np.ndarray | None,
]:
    """Concatenate every geometry in a scene into one vertex/face array pair.

    Each node's world transform is applied on the way in. A scene mixing meshes
    and point clouds keeps them apart -- the points become the cloud, the
    triangles become the mesh -- because sampling a mesh that already has a
    cloud beside it would double-count the same surface.
    """
    import trimesh

    parts: list[tuple[Any, np.ndarray]] = []
    if isinstance(loaded, trimesh.Scene):
        for node in loaded.graph.nodes_geometry:
            transform, geom_name = loaded.graph[node]
            geom = loaded.geometry.get(geom_name)
            if geom is not None:
                parts.append((geom, np.asarray(transform, dtype=np.float64)))
        if not parts:
            warnings.append("scene contains no geometry nodes")
    elif loaded is not None:
        parts.append((loaded, np.eye(4)))

    v_chunks: list[np.ndarray] = []
    f_chunks: list[np.ndarray] = []
    c_chunks: list[np.ndarray | None] = []
    uv_chunks: list[np.ndarray | None] = []
    p_chunks: list[np.ndarray] = []
    pc_chunks: list[np.ndarray | None] = []
    texture: bytes | None = None
    tex_fmt: str | None = None
    textures_seen = 0
    offset = 0

    for geom, transform in parts:
        if isinstance(geom, trimesh.PointCloud):
            xyz = np.asarray(geom.vertices, dtype=np.float64)
            p_chunks.append(xyz @ transform[:3, :3].T + transform[:3, 3])
            pc_chunks.append(_vertex_colors(geom))
            continue
        if not isinstance(geom, trimesh.Trimesh):
            warnings.append(f"ignored scene geometry of type {type(geom).__name__}")
            continue
        xyz = np.asarray(geom.vertices, dtype=np.float64)
        faces = np.asarray(geom.faces, dtype=np.int64)
        v_chunks.append(xyz @ transform[:3, :3].T + transform[:3, 3])
        if np.linalg.det(transform[:3, :3]) < 0:
            faces = faces[:, ::-1]
        f_chunks.append(faces + offset)
        offset += len(xyz)
        c_chunks.append(_vertex_colors(geom))
        uv_chunks.append(_uv(geom))
        blob, fmt = _texture_bytes(geom)
        if blob is not None:
            textures_seen += 1
            if texture is None:
                texture, tex_fmt = blob, fmt

    if textures_seen > 1:
        warnings.append(
            f"scene carried {textures_seen} textures; kept the first, "
            "the Mesh contract holds one"
        )

    vertices = np.concatenate(v_chunks) if v_chunks else np.zeros((0, 3))
    faces = np.concatenate(f_chunks) if f_chunks else np.zeros((0, 3), dtype=np.int64)
    colors = _stack_optional(c_chunks, v_chunks, 3, np.uint8, 255, warnings, "vertex colours")
    uv = _stack_optional(uv_chunks, v_chunks, 2, np.float32, 0, warnings, "texture coordinates")
    cloud = np.concatenate(p_chunks) if p_chunks else np.zeros((0, 3))
    cloud_rgb = _stack_optional(pc_chunks, p_chunks, 3, np.uint8, 255, warnings, "point colours")
    return vertices, faces, colors, uv, texture, tex_fmt, cloud, cloud_rgb


def _stack_optional(
    chunks: list[np.ndarray | None],
    geometry: list[np.ndarray],
    width: int,
    dtype: Any,
    fill: int,
    warnings: list[str],
    what: str,
) -> np.ndarray | None:
    """Concatenate a per-geometry attribute, padding the geometries that lack it.

    A scene where one node is textured and another is plain would otherwise have
    to drop the attribute entirely; padding keeps the node that has it, and the
    warning stops anyone reading the filler as data.
    """
    if not chunks or all(c is None for c in chunks):
        return None
    missing = 0
    out = []
    for chunk, geom in zip(chunks, geometry):
        if chunk is None:
            missing += len(geom)
            out.append(np.full((len(geom), width), fill, dtype=dtype))
        else:
            out.append(np.asarray(chunk, dtype=dtype).reshape(-1, width))
    if missing:
        warnings.append(f"{missing} vertices had no {what}; filled with {fill}")
    return np.concatenate(out) if out else None


def _vertex_colors(geom: Any) -> np.ndarray | None:
    visual = getattr(geom, "visual", None)
    if visual is None:
        return None
    try:
        if getattr(visual, "kind", None) not in ("vertex", "face"):
            return None
        colors = np.asarray(visual.vertex_colors)
    except Exception:  # trimesh raises a family of its own errors here
        return None
    if colors.ndim != 2 or len(colors) != len(geom.vertices):
        return None
    return colors[:, :3].astype(np.uint8)


def _uv(geom: Any) -> np.ndarray | None:
    uv = getattr(getattr(geom, "visual", None), "uv", None)
    if uv is None:
        return None
    uv = np.asarray(uv, dtype=np.float32)
    if uv.ndim != 2 or uv.shape[1] < 2 or len(uv) != len(geom.vertices):
        return None
    return uv[:, :2]


def _texture_bytes(geom: Any) -> tuple[bytes | None, str | None]:
    """Re-encode the material's diffuse image as bytes for `Mesh.texture`.

    JPEG sources are written back as JPEG so a 4K room texture does not become
    a 40 MB PNG inside every saved twin; everything else becomes PNG, which is
    lossless and always decodable.
    """
    material = getattr(getattr(geom, "visual", None), "material", None)
    if material is None:
        return None, None
    image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    if image is None:
        return None, None
    buf = io.BytesIO()
    fmt = (getattr(image, "format", None) or "PNG").upper()
    try:
        if fmt in ("JPEG", "JPG"):
            image.convert("RGB").save(buf, format="JPEG", quality=95)
            return buf.getvalue(), "jpg"
        image.save(buf, format="PNG")
        return buf.getvalue(), "png"
    except Exception:
        return None, None


def _metadata_strings(metadata: Any) -> list[str]:
    if not isinstance(metadata, dict):
        return []
    out = []
    for key, value in metadata.items():
        if isinstance(value, str):
            out.append(f"{key}: {value}")
        elif isinstance(value, dict):
            out.extend(_metadata_strings(value))
    return out


# ---------------------------------------------------------------------------
# glTF document
# ---------------------------------------------------------------------------


def _gltf_json(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    """Return the glTF JSON document, unwrapping the GLB container if needed."""
    try:
        raw = path.read_bytes()
        if raw[:4] == b"glTF":
            if len(raw) < 20:
                return None
            offset = 12
            while offset + 8 <= len(raw):
                length, kind = struct.unpack_from("<II", raw, offset)
                offset += 8
                if kind == 0x4E4F534A:  # 'JSON'
                    return json.loads(raw[offset : offset + length].decode("utf-8"))
                offset += length
            return None
        return json.loads(raw.decode("utf-8", errors="replace"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        warnings.append(f"could not read the glTF JSON for metadata: {exc}")
        return None


def _gltf_camera_positions(doc: dict[str, Any], warnings: list[str]) -> np.ndarray | None:
    """World-space translations of every node that has a camera attached.

    Poses are worth chasing because `CaptureBounds` built from real camera
    positions is a measurement, and one inferred from the point cloud is a
    guess -- the difference decides whether a filming position is legal.
    """
    nodes = doc.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return None
    scenes = doc.get("scenes") or []
    scene_index = doc.get("scene", 0)
    roots: list[int]
    if 0 <= scene_index < len(scenes):
        roots = list(scenes[scene_index].get("nodes", []))
    else:
        roots = list(range(len(nodes)))

    found: list[np.ndarray] = []
    seen: set[int] = set()
    stack: list[tuple[int, np.ndarray]] = [(i, np.eye(4)) for i in reversed(roots)]
    while stack:
        index, parent = stack.pop()
        if index in seen or not 0 <= index < len(nodes):
            continue
        seen.add(index)
        node = nodes[index]
        world = parent @ _node_matrix(node)
        if "camera" in node:
            found.append(world[:3, 3])
        for child in reversed(node.get("children", [])):
            stack.append((child, world))
    if not found:
        return None
    if len(seen) < len(nodes):
        warnings.append(
            f"{len(nodes) - len(seen)} glTF nodes are not reachable from the active scene"
        )
    return np.stack(found)


def _node_matrix(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        # glTF matrices are column-major
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    m = np.eye(4)
    if "rotation" in node:
        x, y, z, w = (float(v) for v in node["rotation"])
        m[:3, :3] = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
    if "scale" in node:
        m[:3, :3] = m[:3, :3] @ np.diag(np.asarray(node["scale"], dtype=np.float64))
    if "translation" in node:
        m[:3, 3] = np.asarray(node["translation"], dtype=np.float64)
    return m
