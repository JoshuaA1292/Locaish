"""Deciding what a file actually is, before trusting anything it claims.

A consumer scanner's "export" arrives with whatever extension the user's share
sheet felt like. Polycam hands out `.glb` files that are really glTF-JSON, mail
clients rename `.ply` to `.txt`, and a `.obj` dropped out of a zip may have lost
its extension entirely. So the sniffer reads the first few kilobytes and
believes the bytes over the filename, falling back to the extension only when
the content is genuinely ambiguous.

The same module carries the two other "read the header, don't guess" jobs that
every reader needs: recognising the capture app from whatever free-text banner
it left behind, and spotting an explicit unit declaration. Both return None
rather than a plausible-looking default, because downstream has to be able to
tell a declaration from an assumption.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

# 4 KiB is comfortably more than any header we care about -- a 60-property
# gaussian-splat PLY header is about 1.5 KiB -- and small enough that sniffing
# a directory of exports costs nothing.
_PROBE_BYTES = 4096

# Extensions we recognise. Formats past the readable set are listed anyway so
# that read_scan can fail with "we know what this is and don't support it"
# rather than "unknown file".
EXTENSION_MAP: dict[str, str] = {
    ".ply": "ply",
    ".obj": "obj",
    ".glb": "glb",
    ".gltf": "gltf",
    ".stl": "stl",
    ".dae": "dae",
    ".off": "off",
    ".xyz": "xyz",
    ".pts": "xyz",
    ".asc": "xyz",
    ".txt": "xyz",
    ".csv": "xyz",
    ".pcd": "pcd",
    ".splat": "splat",
    ".e57": "e57",
    ".las": "las",
    ".laz": "las",
    ".usdz": "usdz",
    ".usd": "usdz",
    ".fbx": "fbx",
    ".3mf": "3mf",
}

# Formats read_scan can actually turn into geometry.
READABLE = frozenset({"ply", "obj", "glb", "gltf", "stl", "dae", "off", "xyz"})

# Ordered most-specific first: a Polycam export processed through MeshLab still
# says Polycam somewhere, and the capture app is the one that determines scale
# and axis conventions, so it wins over whatever edited the file afterwards.
_SOFTWARE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("polycam", r"poly\s*cam|poly\.cam"),
    ("scaniverse", r"scaniverse"),
    ("roomplan", r"room\s*plan"),
    ("luma", r"luma\s*(ai|labs)|lumalabs"),
    ("record3d", r"record\s*3d"),
    ("3dscannerapp", r"3d\s*scanner\s*app"),
    ("matterport", r"matterport"),
    ("colmap", r"colmap"),
    ("realitycapture", r"reality\s*capture"),
    ("metashape", r"agisoft|metashape"),
    ("kiri", r"kiri\s*engine"),
    ("meshroom", r"meshroom|alicevision"),
    ("meshlab", r"meshlab|vcglib"),
    ("cloudcompare", r"cloud\s*compare"),
    ("blender", r"blender"),
    ("sketchfab", r"sketchfab"),
    ("trimesh", r"trimesh"),
    ("locaish", r"locaish"),
)

# A unit token only counts when the surrounding text is talking about units or
# scale. Plenty of files contain the letter sequence "in" or the word "meter"
# for unrelated reasons, and a wrong unit hint is worse than no unit hint.
_UNIT_CONTEXT = re.compile(r"unit|units|scale|measure|dimension", re.IGNORECASE)

# Spelled-out unit names. These are unambiguous wherever they appear on a
# unit-or-scale line, so they are tested first and, within the group, the
# longer name before the shorter one it contains: "centimetres" ends in
# "metres" and "millimetres" ends in "metres", and reading either as metres is
# a 10x or 1000x error that nothing downstream can catch.
_UNIT_WORDS: tuple[tuple[str, str], ...] = (
    ("mm", r"\bmillimet(?:er|re)s?\b"),
    ("cm", r"\bcentimet(?:er|re)s?\b"),
    ("in", r"\binch(?:es)?\b"),
    ("ft", r"\b(?:feet|foot)\b"),
    ("m", r"\bmet(?:er|re)s?\b"),
)

# Words that introduce the *value* of a unit declaration. An abbreviation is
# only believed when one of these (or a colon or equals sign) stands
# immediately in front of it, because "in" is also the commonest preposition in
# English and "m" is a common variable name: without this guard "scale in feet"
# reads as inches, a 12x error announced as a declaration, and an explicit hint
# outranks every geometric measurement in scale.infer_unit_scale.
_VALUE_PREFIX = (
    r"(?:[:=]|\b(?:in|to|per|is|are|of|as|using|unit|units|uom|scale|scaled"
    r"|measure|measured|measurement|dimension|dimensions)\b)"
)

# Between the introducer and the unit there may be punctuation, a scale factor
# or an opening bracket -- "1 unit = 1 m", "units (0.001) mm" -- but never
# another word, which would mean the abbreviation is not the declared value.
_VALUE_GAP = r"[^A-Za-z\n]{0,12}"

# The abbreviation must also end the declaration: end of line, or punctuation.
# "measured in a single pass" is not a declaration of inches.
_VALUE_END = r"(?=\s*(?:$|[^\sA-Za-z]))"

# Two-letter abbreviations, again longest-first among the ones that share a
# suffix so that "mm" is never truncated to "m".
_UNIT_ABBREVIATIONS: tuple[tuple[str, str], ...] = (
    ("mm", r"mm"),
    ("cm", r"cm"),
    ("ft", r"ft"),
    ("in", r"in"),
    ("m", r"m"),
)

_UNIT_DECLARATIONS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    [(unit, re.compile(pattern, re.IGNORECASE)) for unit, pattern in _UNIT_WORDS]
    + [
        (unit, re.compile(_VALUE_PREFIX + _VALUE_GAP + abbrev + r"\b" + _VALUE_END, re.IGNORECASE))
        for unit, abbrev in _UNIT_ABBREVIATIONS
    ]
)


def sniff(path: str | Path) -> str:
    """Return the format token for `path`: ply, obj, glb, gltf, stl, xyz, ...

    Magic bytes first, extension second. The awkward case is STL, whose ASCII
    flavour begins with the word "solid" and whose binary flavour is free to
    begin with the same word inside its 80-byte comment header; the only
    reliable discriminator is that a binary STL's length is exactly
    84 + 50 * triangle_count, so that arithmetic is what we test.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        probe = fh.read(_PROBE_BYTES)
    size = path.stat().st_size
    ext = EXTENSION_MAP.get(path.suffix.lower())

    if probe[:4] == b"glTF":
        return "glb"
    if _first_token(probe) == b"ply":
        return "ply"
    if probe[:5].lower() == b"solid":
        # ASCII STL if it goes on to describe facets, otherwise a binary STL
        # that happened to start with the same five bytes.
        if b"facet" in probe[:1024].lower():
            return "stl"
        if _is_binary_stl(path, size):
            return "stl"
        return "stl"
    if _is_binary_stl(path, size):
        return "stl"
    if probe[:11] == b"# .PCD v0.7" or probe[:5] == b"# .PC":
        return "pcd"

    text = probe.decode("utf-8", errors="replace").lstrip("﻿").lstrip()
    if text[:1] == "{" and '"asset"' in text.replace(" ", ""):
        return "gltf"
    if text[:5].lower() == "<?xml" or text[:9].lower() == "<collada":
        if "collada" in text[:2048].lower():
            return "dae"
    if text[:3].upper() == "OFF":
        return "off"

    guess = _sniff_text_geometry(text)
    if guess is not None:
        return guess
    if ext is not None:
        return ext
    return "unknown"


def detect_software(*texts: str | None) -> str | None:
    """Name the capture app from header comments, or None if nothing says.

    Returning None matters: `software` drives per-app workarounds downstream,
    and inventing one would apply a fix for a bug the file does not have.
    """
    blob = "\n".join(t for t in texts if t)
    if not blob:
        return None
    for name, pattern in _SOFTWARE_PATTERNS:
        if re.search(pattern, blob, re.IGNORECASE):
            return name
    return None


def detect_unit_hint(*texts: str | None) -> str | None:
    """Return a declared unit ("m", "cm", "mm", "in", "ft") or None.

    Only a line that is explicitly about units or scale is considered, and only
    the first such line, so that a stray word elsewhere in a comment block
    cannot fabricate a hint that `scale.infer_unit_scale` would then trust over
    its own measurement of the geometry.

    Within such a line a unit has to look like a declaration rather than merely
    like the letters of one. Spelled-out names are checked before the
    two-letter abbreviations, and each abbreviation has to sit in value
    position -- after a colon, an equals sign or a word such as "in" -- and end
    the declaration. The case this defends against is the header comment
    "scale in feet", where the bare-word reading of "in" turns a foot into an
    inch: a silent factor of twelve that outranks every later measurement
    because an explicit hint wins over geometric inference.
    """
    for text in texts:
        if not text:
            continue
        for line in text.splitlines():
            if not _UNIT_CONTEXT.search(line):
                continue
            for unit, pattern in _UNIT_DECLARATIONS:
                if pattern.search(line):
                    return unit
    return None


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _first_token(probe: bytes) -> bytes:
    """First whitespace-delimited token, lowercased. PLY's magic is a line, not
    a fixed-width field, and CRLF-terminated files are common on Windows."""
    head = probe[:64].lstrip()
    for sep in (b"\r", b"\n", b" ", b"\t"):
        head = head.split(sep)[0]
    return head.lower()


def _is_binary_stl(path: Path, size: int) -> bool:
    if size < 84:
        return False
    with open(path, "rb") as fh:
        fh.seek(80)
        raw = fh.read(4)
    if len(raw) != 4:
        return False
    (count,) = struct.unpack("<I", raw)
    return size == 84 + 50 * count


def _sniff_text_geometry(text: str) -> str | None:
    """Tell an OBJ from a bare XYZ point list by looking at the line tags."""
    obj_tags = 0
    numeric_rows = 0
    inspected = 0
    for line in text.splitlines()[:200]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        inspected += 1
        tag = line.split(None, 1)[0]
        if tag in {"v", "vn", "vt", "f", "mtllib", "usemtl", "o", "g", "s", "vp"}:
            obj_tags += 1
        elif _looks_numeric(line):
            numeric_rows += 1
        if inspected >= 40:
            break
    if obj_tags >= 2 and obj_tags >= numeric_rows:
        return "obj"
    if numeric_rows >= 2:
        return "xyz"
    return None


def _looks_numeric(line: str) -> bool:
    parts = line.replace(",", " ").split()
    if not 3 <= len(parts) <= 12:
        return False
    for p in parts:
        try:
            float(p)
        except ValueError:
            return False
    return True
