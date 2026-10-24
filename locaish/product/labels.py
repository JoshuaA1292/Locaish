"""What is in the room: objects named by a vision model, confirmed or
corrected by a person who clicks on them.

Geometry says where the surfaces are; it does not know that the box against
the north wall is a fridge or that the thing under the window is a radiator
the gaffer will have to flag. A producer reading a package wants both. So
after the twin is built a vision model is shown a spread of the capture's
own frames and asked to name what it sees, each answer with a box in the
picture. The box becomes a ray from that frame's camera, the ray is marched
into the measured cloud, and the first surface it meets is where the thing
stands in the room. Names from the model are provisional and say so.

When the model is wrong or silent the person is the sensor: click anywhere
in the twin, and the frames that actually looked at that spot are brought
up with the spot marked, so what is being named is seen in a photograph and
not guessed from a cloud. Every label, whoever gave it, carries its 3D
position, which frame it came from, and its source.

The projection chain is the one the viewer and the renderer already use:
twin = R @ (factor * colmap) + t, with R, t the twin's canonical transform
and factor the solved metres-per-unit. Camera poses are COLMAP's
world-to-camera [R_c | t_c] over the frames the solve registered.
"""

from __future__ import annotations

import io
import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..meta import vision as _vision
from ..types import Twin

#: How a labelled thing is classed, for the floor plan and the package.
KINDS = ("furniture", "appliance", "fixture", "window", "door", "lighting", "plant",
         "artwork", "equipment", "structure", "other")

#: Two detections of the same name closer than this are one object.
MERGE_M = 0.7

#: Occlusion test: a measured point this close to the sightline, and this far
#: short of the target, means the frame was looking at something in front.
OCCLUDE_RADIUS_M = 0.04
OCCLUDE_MARGIN_M = 0.25

#: Ray march for lifting a 2D box into the room.
MARCH_STEP_M = 0.04
MARCH_HIT_M = 0.05
MARCH_MIN_M = 0.3
MARCH_MAX_M = 12.0

#: The cloud the rays are tested against is thinned to this many points: a
#: KD-tree over two million returns builds in seconds and answers no better
#: for a question asked at 4 cm.
TREE_POINTS = 400_000

# Detection is a different job from planning: Roboflow's July 2026 evals
# found gemini-3.6-flash regressed below 3.5-flash on bounding boxes (one
# loose box in crowded scenes), so the detector defaults to 3.5-flash and
# is its own knob.
DEFAULT_MODEL = os.environ.get("LOCAISH_DETECT_MODEL", "gemini-3.5-flash")
DETECT_FRAMES = 8

_PROMPT = """\
This is one frame from a handheld video walkthrough of a room being considered
as a filming location. List every distinct physical object a film crew would
care about: furniture, appliances, fixed fixtures (radiators, counters, sinks,
fireplaces, columns, beams), windows, doors, lamps and light fittings, plants,
artwork, mirrors, electrical panels and sockets, anything that would have to be
moved, dressed, flagged or lit. Ignore small clutter and anything you cannot
identify. Respond with JSON only: a list of objects, each exactly
{"label": "<two or three words, lower case, singular>",
 "kind": "<furniture|appliance|fixture|window|door|lighting|plant|artwork|equipment|structure|other>",
 "box_2d": [ymin, xmin, ymax, xmax]}
with box coordinates normalised to 0-1000 over the image. Do not include any
text outside the JSON.
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# posed frames
# ---------------------------------------------------------------------------


@dataclass
class PosedFrames:
    """The capture's registered frames, with the maths to look through them."""

    names: list[str]
    paths: list[Path]
    extr: np.ndarray            # (N, 3, 4) COLMAP world -> camera
    K: np.ndarray               # (N, 3, 3)
    size: tuple[int, int]       # (width, height) of the frames as posed
    R: np.ndarray               # twin = R @ (factor * colmap) + t
    t: np.ndarray
    factor: float
    sharpness: np.ndarray | None = None
    centres_twin: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    def __post_init__(self) -> None:
        Rc = self.extr[:, :, :3]
        tc = self.extr[:, :, 3]
        centres = -np.einsum("nji,nj->ni", Rc, tc)          # -Rcᵀ tc
        self.centres_twin = (centres * self.factor) @ self.R.T + self.t

    def __len__(self) -> int:
        return len(self.names)

    # -- twin <-> colmap -------------------------------------------------
    def to_colmap(self, p_twin: np.ndarray) -> np.ndarray:
        p = np.asarray(p_twin, dtype=np.float64)
        return ((p - self.t) @ self.R) / self.factor      # Rᵀ (p - t) / factor

    def to_twin(self, p_colmap: np.ndarray) -> np.ndarray:
        p = np.asarray(p_colmap, dtype=np.float64)
        return (p * self.factor) @ self.R.T + self.t

    # -- projection -------------------------------------------------------
    def project(self, p_twin) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pixel (u, v) and metric depth of one twin point in every frame."""
        pc = self.to_colmap(np.asarray(p_twin, dtype=np.float64).reshape(3))
        x = np.einsum("nij,j->ni", self.extr[:, :, :3], pc) + self.extr[:, :, 3]
        z = x[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.K[:, 0, 0] * x[:, 0] / z + self.K[:, 0, 2]
            v = self.K[:, 1, 1] * x[:, 1] / z + self.K[:, 1, 2]
        return u, v, z * self.factor

    def ray(self, i: int, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """Origin and unit direction, in twin space, of the pixel's sightline."""
        K = self.K[i]
        d_cam = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
        d_world = self.extr[i, :, :3].T @ d_cam
        d_twin = self.R @ d_world
        d_twin = d_twin / max(float(np.linalg.norm(d_twin)), 1e-12)
        return self.centres_twin[i].copy(), d_twin

    # -- choosing frames ----------------------------------------------------
    def best_views(self, p_twin, tree=None, *, k: int = 4, margin: float = 0.08) -> list[dict]:
        """The frames that looked at this point, best first.

        Scored on how central the point sits, how close the camera was (a
        metre to four metres reads best), and sharpness when known; frames
        whose sightline to the point is blocked by measured surface are
        dropped, so a label is typed over a photograph of the thing itself.
        """
        u, v, depth = self.project(p_twin)
        w, h = self.size
        ok = (depth > 0.3) & (u > margin * w) & (u < (1 - margin) * w) & (v > margin * h) & (v < (1 - margin) * h)
        idx = np.flatnonzero(ok)
        if not len(idx):
            return []
        cu = np.abs(u[idx] / w - 0.5) + np.abs(v[idx] / h - 0.5)          # 0 at centre, 1 at corner
        dist = np.abs(np.log(np.clip(depth[idx], 0.2, 20.0) / 2.0))      # 0 at 2 m
        score = 1.0 - 0.6 * cu - 0.25 * np.minimum(dist, 2.0) / 2.0
        if self.sharpness is not None and len(self.sharpness) == len(self):
            s = self.sharpness[idx]
            score += 0.15 * (s / max(float(np.max(self.sharpness)), 1e-9))
        order = idx[np.argsort(-score)]
        out = []
        target = np.asarray(p_twin, dtype=np.float64).reshape(3)
        for i in order:
            if tree is not None and occluded(tree, self.centres_twin[i], target):
                continue
            out.append({"index": int(i), "frame": self.names[i], "u": float(u[i]), "v": float(v[i]),
                        "depth_m": round(float(depth[i]), 2), "score": round(float(score[np.searchsorted(idx, i)]), 3)})
            if len(out) >= k:
                break
        return out

    def spread(self, count: int) -> list[int]:
        """`count` frames spread over the sweep, the sharpest in each slice."""
        n = len(self)
        if n <= count:
            return list(range(n))
        edges = np.linspace(0, n, count + 1).astype(int)
        out = []
        for a, b in zip(edges[:-1], edges[1:]):
            if self.sharpness is not None and len(self.sharpness) == n:
                out.append(int(a + np.argmax(self.sharpness[a:b])))
            else:
                out.append(int((a + b) // 2))
        return out

    # -- pictures -----------------------------------------------------------
    def crop(self, i: int, u: float, v: float, *, box_px: int = 480, out_px: int = 520,
             mark: bool = True) -> bytes:
        """A square crop around a pixel with the pixel marked, as JPEG bytes."""
        from PIL import Image, ImageDraw

        im = Image.open(self.paths[i]).convert("RGB")
        sx, sy = im.width / self.size[0], im.height / self.size[1]
        cx, cy = u * sx, v * sy
        half = box_px * max(sx, sy) / 2
        left, top = int(max(cx - half, 0)), int(max(cy - half, 0))
        right, bottom = int(min(cx + half, im.width)), int(min(cy + half, im.height))
        crop = im.crop((left, top, right, bottom))
        if mark:
            d = ImageDraw.Draw(crop)
            px, py = cx - left, cy - top
            r = max(crop.width, crop.height) * 0.05
            d.ellipse((px - r, py - r, px + r, py + r), outline=(255, 207, 90), width=max(2, int(r / 6)))
            d.line((px - 2 * r, py, px - r * 1.2, py), fill=(255, 207, 90), width=3)
            d.line((px + r * 1.2, py, px + 2 * r, py), fill=(255, 207, 90), width=3)
        if crop.width > out_px:
            crop = crop.resize((out_px, int(crop.height * out_px / crop.width)))
        buf = io.BytesIO()
        crop.save(buf, "JPEG", quality=85)
        return buf.getvalue()


def load_frames(workdir: Path, twin: Twin) -> PosedFrames | None:
    """The registered frames of a video job, or None for scan-file twins."""
    from ..video import colmap as colmapmod

    workdir = Path(workdir)
    recon = workdir / "recon"
    steps = (twin.provenance or {}).get("steps", {})
    factor = ((steps.get("video") or {}).get("scale") or {}).get("factor")
    if not factor or twin.canonical_transform is None:
        return None
    sparse = recon / "colmap" / "sparse"
    text_dirs = sorted(sparse.glob("*_txt")) if sparse.is_dir() else []
    text_dir = next((d for d in text_dirs if (d / "images.txt").exists()), None)
    if text_dir is None:
        return None
    frames_dir = recon / "frames"
    if not frames_dir.is_dir():
        return None
    try:
        cameras = colmapmod._read_cameras_text(text_dir / "cameras.txt")
        images = colmapmod._read_images_text(text_dir)
    except Exception:  # noqa: BLE001 - an unreadable model is simply no frames
        return None
    names, paths, extr, K = [], [], [], []
    for name, q, tr, cid in zip(images["names"], images["quats"], images["trans"], images["cam_ids"]):
        p = frames_dir / name
        if not p.exists() or cid not in cameras:
            continue
        E = np.zeros((3, 4))
        E[:, :3] = colmapmod._quat_to_rotation(q)
        E[:, 3] = tr
        names.append(name)
        paths.append(p)
        extr.append(E)
        K.append(cameras[cid]["K"])
    if not names:
        return None
    first = cameras[images["cam_ids"][0]]
    sharp = None
    npz = recon / "reconstruction.npz"
    if npz.exists():
        try:
            d = np.load(npz, allow_pickle=True)
            fp = [Path(p).name for p in json.loads(str(d["frame_paths"]))]
            fs = json.loads(str(d["frame_sharpness"]))
            lookup = dict(zip(fp, fs))
            sharp = np.array([float(lookup.get(n, 0.0)) for n in names])
        except Exception:  # noqa: BLE001
            sharp = None
    M = np.asarray(twin.canonical_transform, dtype=np.float64)
    return PosedFrames(names=names, paths=paths, extr=np.stack(extr), K=np.stack(K),
                       size=(int(first["width"]), int(first["height"])),
                       R=M[:3, :3], t=M[:3, 3], factor=float(factor), sharpness=sharp)


# ---------------------------------------------------------------------------
# the cloud as something a ray can hit
# ---------------------------------------------------------------------------


def surface_tree(twin: Twin, *, cap: int = TREE_POINTS, seed: int = 0):
    from scipy.spatial import cKDTree

    xyz = np.asarray(twin.points.measured().xyz, dtype=np.float64)
    if len(xyz) > cap:
        rng = np.random.default_rng(seed)
        xyz = xyz[rng.choice(len(xyz), cap, replace=False)]
    return cKDTree(xyz)


def occluded(tree, origin, target) -> bool:
    o = np.asarray(origin, dtype=np.float64)
    p = np.asarray(target, dtype=np.float64)
    d = p - o
    length = float(np.linalg.norm(d))
    if length < OCCLUDE_MARGIN_M + MARCH_STEP_M:
        return False
    n = int((length - OCCLUDE_MARGIN_M) / MARCH_STEP_M)
    ts = np.linspace(MARCH_MIN_M, length - OCCLUDE_MARGIN_M, max(n, 2))
    samples = o[None, :] + ts[:, None] * (d / length)[None, :]
    dist, _ = tree.query(samples, k=1, distance_upper_bound=OCCLUDE_RADIUS_M)
    return bool(np.isfinite(dist).any())


def first_hit(tree, origin, direction, *, max_m: float = MARCH_MAX_M) -> np.ndarray | None:
    """Where a ray first meets measured surface, or None."""
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    ts = np.arange(MARCH_MIN_M, max_m, MARCH_STEP_M)
    samples = o[None, :] + ts[:, None] * d[None, :]
    dist, idx = tree.query(samples, k=1, distance_upper_bound=MARCH_HIT_M)
    hits = np.flatnonzero(np.isfinite(dist))
    if not len(hits):
        return None
    return np.asarray(tree.data[idx[hits[0]]], dtype=np.float64)


# ---------------------------------------------------------------------------
# the labels file
# ---------------------------------------------------------------------------


_LOCK = threading.Lock()


def labels_path(workdir: Path) -> Path:
    return Path(workdir) / "labels.json"


def load_labels(workdir: Path) -> list[dict]:
    p = labels_path(workdir)
    try:
        d = json.loads(p.read_text())
        return list(d.get("labels") or [])
    except (OSError, ValueError):
        return []


def save_labels(workdir: Path, labels: list[dict]) -> None:
    p = labels_path(workdir)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"labels": labels, "updated_at": _now()}, indent=2))
    os.replace(tmp, p)


def normalise(label: str) -> str:
    s = " ".join(str(label or "").lower().split())[:60]
    return s


def add_label(workdir: Path, *, label: str, xyz, source: str = "user", kind: str = "other",
              frame: str | None = None, uv=None, note: str = "", confidence: float | None = None) -> dict:
    name = normalise(label)
    if not name:
        raise ValueError("a label needs a name")
    rec = {
        "id": uuid.uuid4().hex[:8],
        "label": name,
        "kind": kind if kind in KINDS else "other",
        "xyz": [round(float(v), 3) for v in np.asarray(xyz, dtype=np.float64).reshape(3)],
        "source": source,
        "frame": frame,
        "uv": [round(float(uv[0]), 1), round(float(uv[1]), 1)] if uv is not None else None,
        "note": str(note or "")[:500],
        "confidence": confidence,
        "created_at": _now(),
    }
    with _LOCK:
        labels = load_labels(workdir)
        labels.append(rec)
        save_labels(workdir, labels)
    return rec


def remove_label(workdir: Path, label_id: str) -> bool:
    with _LOCK:
        labels = load_labels(workdir)
        kept = [l for l in labels if l.get("id") != label_id]
        if len(kept) == len(labels):
            return False
        save_labels(workdir, kept)
        return True


def confirm_label(workdir: Path, label_id: str, *, label: str | None = None, kind: str | None = None) -> dict | None:
    """A person agrees with (or corrects) a model's label: it becomes theirs."""
    with _LOCK:
        labels = load_labels(workdir)
        for rec in labels:
            if rec.get("id") == label_id:
                if label:
                    rec["label"] = normalise(label)
                if kind in KINDS:
                    rec["kind"] = kind
                rec["source"] = "user"
                rec["confirmed_at"] = _now()
                save_labels(workdir, labels)
                return rec
    return None


# ---------------------------------------------------------------------------
# the model's first pass
# ---------------------------------------------------------------------------


#: Sources a model writes. Anything else on a label is a person's word.
MODEL_SOURCES = _vision.MODEL_PROVIDERS


def vision_available() -> bool:
    """Whether the chosen vision provider (`meta.vision.provider()`) can run."""
    return _vision.available()


#: The name this check had when Gemini was the only provider; kept so older
#: callers keep working. It answers for whichever provider is chosen now.
gemini_available = vision_available


def _model_name(provider: str, model: str | None) -> str:
    if provider == "meta":
        from ..meta import model_for

        return model or model_for("vision")
    return model or DEFAULT_MODEL


def detect_in_frame(path: Path, *, model: str | None = None, client=None,
                    provider: str | None = None) -> list[dict]:
    """Ask the vision model what is in one frame. Boxes are 0-1000 normalised
    [ymin, xmin, ymax, xmax] in the frame as stored on disk.

    `provider` is "meta" or "gemini" (default: whatever `meta.vision` chose);
    `client` is that provider's client to reuse across frames."""
    provider = provider or _vision.provider()
    if provider == "meta":
        return _vision.detect_objects(Path(path).read_bytes(), client=client, model=model)
    if provider != "gemini":
        raise ValueError("no vision model is configured (set META_API_KEY, or GOOGLE_API_KEY for Gemini)")
    return _detect_in_frame_gemini(Path(path), model=model, client=client)


def _detect_in_frame_gemini(path: Path, *, model: str | None = None, client=None) -> list[dict]:
    from google import genai
    from google.genai import types as gtypes

    client = client or genai.Client()
    response = client.models.generate_content(
        model=model or DEFAULT_MODEL,
        contents=[gtypes.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg"), _PROMPT],
        config=gtypes.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
    )
    data = json.loads((response.text or "[]").strip() or "[]")
    if isinstance(data, dict):
        data = data.get("objects") or data.get("items") or []
    out = []
    for d in data if isinstance(data, list) else []:
        try:
            box = [float(v) for v in d["box_2d"]]
            if len(box) != 4:
                continue
            out.append({"label": normalise(d.get("label")), "kind": str(d.get("kind") or "other"), "box": box})
        except (KeyError, TypeError, ValueError):
            continue
    return [o for o in out if o["label"]]


def lift(frames: PosedFrames, tree, i: int, box: list[float]) -> np.ndarray | None:
    """A 2D box in frame `i` becomes the surface its centre looks at."""
    from PIL import Image

    ymin, xmin, ymax, xmax = box
    # Boxes are normalised over the stored image; poses are over the frame
    # as COLMAP saw it (same picture, possibly a different pixel size).
    w, h = frames.size
    u = (xmin + xmax) / 2000.0 * w
    v = (ymin + ymax) / 2000.0 * h
    o, d = frames.ray(i, u, v)
    return first_hit(tree, o, d)


def merge(detections: list[dict]) -> list[dict]:
    """Detections of one name within MERGE_M of each other are one object,
    placed at their mean and credited with how many frames agreed."""
    groups: list[dict] = []
    for det in detections:
        placed = False
        for g in groups:
            if g["label"] == det["label"] and np.linalg.norm(np.asarray(g["xyz"]) - np.asarray(det["xyz"])) < MERGE_M:
                n = g["n"]
                g["xyz"] = ((np.asarray(g["xyz"]) * n + np.asarray(det["xyz"])) / (n + 1)).tolist()
                g["n"] = n + 1
                g["frames"].append(det["frame"])
                placed = True
                break
        if not placed:
            groups.append({**det, "xyz": list(det["xyz"]), "n": 1, "frames": [det["frame"]]})
    return groups


def detect(workdir: Path, twin: Twin, *, frames: PosedFrames | None = None, tree=None,
           max_frames: int = DETECT_FRAMES, model: str | None = None, progress=None,
           client=None, provider: str | None = None) -> dict:
    """The semantic pass: name what the frames show, place it in the room,
    and record it as provisional. Replaces earlier model labels -- from any
    provider -- and keeps every label a person gave.

    `provider` and `client` default to the configured vision provider and a
    fresh client for it; the tests pass "meta" and a `FakeMetaClient`."""
    frames = frames or load_frames(workdir, twin)
    if frames is None:
        raise ValueError("this twin has no posed frames to look at (scan-file twins carry no pictures)")
    provider = provider or _vision.provider()
    if provider not in MODEL_SOURCES or (client is None and not _vision.configured_for(provider)):
        raise ValueError("no vision model is configured (set META_API_KEY, or GOOGLE_API_KEY for Gemini)")
    if client is None:
        if provider == "meta":
            from ..meta import MetaClient

            client = MetaClient()
        else:
            from google import genai

            client = genai.Client()
    model_name = _model_name(provider, model)

    tree = tree if tree is not None else surface_tree(twin)
    picks = frames.spread(max_frames)
    detections, raw = [], 0
    for n, i in enumerate(picks, 1):
        if progress:
            progress(f"looking at frame {n} of {len(picks)}")
        try:
            found = detect_in_frame(frames.paths[i], model=model, client=client, provider=provider)
        except Exception as exc:  # noqa: BLE001 - one bad frame is not a failed pass
            if progress:
                progress(f"frame {frames.names[i]}: {type(exc).__name__}: {exc}")
            continue
        raw += len(found)
        for f in found:
            p = lift(frames, tree, i, f["box"])
            if p is None:
                continue
            detections.append({"label": f["label"], "kind": f["kind"], "xyz": p, "frame": frames.names[i],
                               "uv": [(f["box"][1] + f["box"][3]) / 2000.0 * frames.size[0],
                                      (f["box"][0] + f["box"][2]) / 2000.0 * frames.size[1]]})
    merged = merge(detections)
    with _LOCK:
        kept = [l for l in load_labels(workdir) if l.get("source") not in MODEL_SOURCES]
        for g in merged:
            # A thing a person already named within reach is not re-added.
            if any(np.linalg.norm(np.asarray(k["xyz"]) - np.asarray(g["xyz"])) < MERGE_M for k in kept):
                continue
            kept.append({
                "id": uuid.uuid4().hex[:8], "label": g["label"],
                "kind": g["kind"] if g["kind"] in KINDS else "other",
                "xyz": [round(float(v), 3) for v in g["xyz"]], "source": provider,
                "frame": g["frames"][0], "uv": [round(float(v), 1) for v in g["uv"]],
                "note": "", "confidence": g["n"], "created_at": _now(),
            })
        save_labels(workdir, kept)
    return {"frames_looked_at": len(picks), "detections": raw, "placed": len(detections),
            "objects": len(merged), "labels": kept, "model": model_name, "provider": provider}


# ---------------------------------------------------------------------------
# drawing and describing
# ---------------------------------------------------------------------------


def labels_svg(labels: list[dict], X, Y) -> list[str]:
    parts = []
    for l in labels:
        x, y = float(l["xyz"][0]), float(l["xyz"][1])
        colour = "#e8eaed" if l.get("source") == "user" else "#a78bfa"
        parts.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" fill="{colour}" fill-opacity="0.9"/>')
        parts.append(f'<text x="{X(x) + 6:.1f}" y="{Y(y) + 4:.1f}" fill="{colour}" font-size="10">'
                     f'{_esc(l["label"])}{"" if l.get("source") == "user" else "?"}</text>')
    return parts


def describe(labels: list[dict]) -> list[dict]:
    """Labels as a package lists them: name, how sure, where (metres)."""
    out = []
    for l in sorted(labels, key=lambda r: (r.get("kind") or "", r.get("label") or "")):
        out.append({
            "label": l["label"], "kind": l.get("kind") or "other",
            "x_m": l["xyz"][0], "y_m": l["xyz"][1], "height_m": l["xyz"][2],
            "source": l.get("source"),
            "status": "confirmed" if l.get("source") == "user" else f"seen by the model in {l.get('confidence') or 1} frame{'s' if (l.get('confidence') or 1) != 1 else ''}",
        })
    return out


def _esc(t: str) -> str:
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
