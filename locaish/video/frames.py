"""Turning a hand-held video into the handful of frames worth reconstructing.

A 30-second room sweep is 900 frames of which maybe 25 carry independent
information. The rest are either duplicates of a frame we already have -- the
phone barely moved -- or motion-blurred smears taken mid-pan. Feeding all 900
to a reconstruction network is not "more data"; it is the same data plus the
blur, at quadratic cost in attention memory, and the blurred frames actively
poison the geometry because a smeared edge still produces confident-looking
depth.

So selection has two jobs, and they pull against each other:

*Coverage* -- the chosen frames have to span the whole sweep, or the twin is a
reconstruction of one corner of the room. This argues for spreading picks
evenly along the timeline.

*Sharpness* -- within any short window, the sharpest frame is strictly better
than its neighbours, and sharpness varies violently frame to frame during a
pan.

The resolution is to bucket the timeline into as many equal slices as we want
frames, and take the sharpest frame in each slice. Coverage is then guaranteed
by construction rather than hoped for, and sharpness is optimised where it is
free to do so. A frame that is the sharpest in its bucket and still blurry is
kept, with a warning -- dropping it would silently punch a hole in the sweep,
which is the worse failure.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Video containers we will hand to ffmpeg. The sniffing is deliberately by
# extension only: unlike the scan formats, we are not parsing these ourselves,
# and ffmpeg's own demuxer probe is far better than anything we would write.
VIDEO_EXTENSIONS = frozenset(
    {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm", ".hevc", ".mts", ".m2ts", ".3gp"}
)

# Candidates are pulled at this rate before selection. Higher than the final
# frame count so that every timeline bucket has several frames to choose the
# sharpest from; low enough that a two-minute video does not decode into
# thousands of JPEGs.
CANDIDATE_FPS = 6.0

# Reconstruction runs at 518 px; decoding at more than about twice that wastes
# time and disk without adding a pixel of usable detail. It does help the
# sharpness measure, which is why it is not simply 518.
CANDIDATE_LONG_SIDE = 1024

# Below this relative sharpness -- measured against the sharpest frame in the
# whole video, so it is exposure- and content-independent -- a frame is
# reported as blurred. It is a warning, never a rejection.
BLUR_WARN_RATIO = 0.15


class VideoError(RuntimeError):
    """The video could not be read, or carried no usable frames."""


@dataclass
class VideoInfo:
    """What the container says about itself, before we decode anything."""

    path: Path
    duration_s: float
    fps: float
    width: int
    height: int
    rotation: float
    codec: str
    frame_count: int

    def summary(self) -> dict:
        return {
            "path": str(self.path),
            "duration_s": round(self.duration_s, 2),
            "fps": round(self.fps, 3),
            "resolution": f"{self.width}x{self.height}",
            "rotation_deg": self.rotation,
            "codec": self.codec,
            "frames": self.frame_count,
        }


@dataclass
class FrameSet:
    """The frames we chose, and the honest story of how they were chosen."""

    paths: list[Path]
    timestamps: list[float]
    sharpness: list[float]
    info: VideoInfo
    candidates_considered: int
    warnings: list[str]

    def __len__(self) -> int:
        return len(self.paths)


def require_ffmpeg() -> tuple[str, str]:
    """Locate ffmpeg and ffprobe, or explain how to get them.

    Raised as a hard error rather than degraded gracefully: there is no second
    way to decode a video, so continuing would only move the failure later.
    """
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise VideoError(
            "ffmpeg and ffprobe are required to read video. Install with "
            "`brew install ffmpeg` (macOS) or `apt install ffmpeg` (Debian)."
        )
    return ffmpeg, ffprobe


def probe(path: str | Path) -> VideoInfo:
    """Read the container's own account of the video.

    Every field here is a *declaration*, in the same sense the scan readers use
    the word: `nb_frames` is frequently absent or wrong, `duration` on a
    variable-frame-rate phone recording is approximate, and rotation lives in a
    side-data matrix that half the tools in the world ignore. So each is parsed
    defensively and, where it matters, cross-checked against what actually
    decodes rather than trusted outright.
    """
    path = Path(path)
    if not path.exists():
        raise VideoError(f"{path} does not exist")
    _, ffprobe = require_ffmpeg()

    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,codec_name,duration:"
        "stream_side_data=rotation:format=duration",
        "-of", "json", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VideoError(f"ffprobe could not read {path.name}: {proc.stderr.strip()[:300]}")
    try:
        meta = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - ffprobe is well behaved
        raise VideoError(f"ffprobe returned unparseable output for {path.name}") from exc

    streams = meta.get("streams") or []
    if not streams:
        raise VideoError(f"{path.name} has no video stream")
    st = streams[0]

    fps = _ratio(st.get("avg_frame_rate")) or _ratio(st.get("r_frame_rate")) or 0.0
    duration = _float(st.get("duration")) or _float((meta.get("format") or {}).get("duration")) or 0.0
    rotation = 0.0
    for sd in st.get("side_data_list") or []:
        if "rotation" in sd:
            rotation = float(sd["rotation"])
    n = int(_float(st.get("nb_frames")) or 0)
    if n <= 0 and fps > 0 and duration > 0:
        n = int(round(fps * duration))

    return VideoInfo(
        path=path,
        duration_s=float(duration),
        fps=float(fps),
        width=int(st.get("width") or 0),
        height=int(st.get("height") or 0),
        rotation=rotation,
        codec=str(st.get("codec_name") or "unknown"),
        frame_count=n,
    )


def extract_frames(
    path: str | Path,
    out_dir: str | Path,
    *,
    count: int = 24,
    candidate_fps: float = CANDIDATE_FPS,
    long_side: int = CANDIDATE_LONG_SIDE,
    start_s: float | None = None,
    end_s: float | None = None,
    progress=None,
) -> FrameSet:
    """Decode candidates, score them, and keep the sharpest per timeline bucket."""
    info = probe(path)
    ffmpeg, _ = require_ffmpeg()
    out_dir = Path(out_dir)
    cand_dir = out_dir / "candidates"
    if cand_dir.exists():
        shutil.rmtree(cand_dir)
    cand_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    if info.duration_s <= 0:
        warnings.append(
            "the container declares no duration, so frame timestamps are "
            "derived from the decode order rather than read"
        )

    # -velocity note: `-vf fps=` resamples rather than seeks, which is what we
    # want -- seeking per frame on a long-GOP HEVC phone recording is both slow
    # and inaccurate, and the accuracy matters because timestamps become the
    # ordering the reconstruction relies on.
    vf = [f"fps={candidate_fps:g}", f"scale='if(gt(iw,ih),{long_side},-2)':'if(gt(iw,ih),-2,{long_side})'"]
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    if start_s is not None:
        cmd += ["-ss", f"{start_s:g}"]
    if end_s is not None and start_s is not None:
        cmd += ["-t", f"{max(0.0, end_s - start_s):g}"]
    elif end_s is not None:
        cmd += ["-to", f"{end_s:g}"]
    cmd += ["-i", str(path), "-vf", ",".join(vf), "-q:v", "2", str(cand_dir / "cand_%05d.jpg")]

    if progress:
        progress("decode")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VideoError(f"ffmpeg failed to decode {info.path.name}: {proc.stderr.strip()[:400]}")

    candidates = sorted(cand_dir.glob("cand_*.jpg"))
    if not candidates:
        raise VideoError(f"{info.path.name} produced no decodable frames")

    if progress:
        progress("score")
    scores = np.array([_sharpness(p) for p in candidates], dtype=np.float64)
    offset = float(start_s or 0.0)
    stamps = np.array([offset + i / candidate_fps for i in range(len(candidates))])

    keep = _pick_per_bucket(scores, count)
    peak = float(scores.max()) or 1.0
    ratios = scores[keep] / peak
    blurry = int((ratios < BLUR_WARN_RATIO).sum())
    if blurry:
        warnings.append(
            f"{blurry} of {len(keep)} selected frames are heavily motion-blurred "
            f"(under {BLUR_WARN_RATIO:.0%} of the sharpest frame in the video); "
            "they were kept because dropping them would leave a gap in the sweep, "
            "but the geometry they contribute is the least trustworthy in the twin"
        )
    if len(candidates) < count:
        warnings.append(
            f"only {len(candidates)} frames could be decoded, fewer than the {count} "
            "requested, so the reconstruction sees less of the room than intended"
        )

    frames_dir = out_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    chosen: list[Path] = []
    for rank, idx in enumerate(keep):
        dest = frames_dir / f"frame_{rank:04d}.jpg"
        shutil.copyfile(candidates[idx], dest)
        chosen.append(dest)
    shutil.rmtree(cand_dir, ignore_errors=True)

    return FrameSet(
        paths=chosen,
        timestamps=[float(stamps[i]) for i in keep],
        sharpness=[float(scores[i]) for i in keep],
        info=info,
        candidates_considered=len(candidates),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _pick_per_bucket(scores: np.ndarray, count: int) -> list[int]:
    """Sharpest frame from each of `count` equal slices of the timeline."""
    n = len(scores)
    if n <= count:
        return list(range(n))
    edges = np.linspace(0, n, count + 1).round().astype(int)
    picks: list[int] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        picks.append(int(lo + np.argmax(scores[lo:hi])))
    return sorted(set(picks))


def _sharpness(path: Path) -> float:
    """Variance of the Laplacian, the standard cheap focus measure.

    Computed on the luma channel at a fixed downscale so that the number is
    comparable between frames of different content -- what we need is a
    *ranking* within one video, not an absolute focus score, and the variance
    of a second-derivative operator is monotonic in blur for a fixed scene,
    which is exactly the comparison the bucket picker makes.
    """
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("L")
        im.thumbnail((512, 512))
        a = np.asarray(im, dtype=np.float32)
    if a.size == 0:
        return 0.0
    lap = (
        -4.0 * a[1:-1, 1:-1]
        + a[:-2, 1:-1] + a[2:, 1:-1]
        + a[1:-1, :-2] + a[1:-1, 2:]
    )
    return float(lap.var())


def _ratio(text: str | None) -> float | None:
    if not text or text in {"0/0", "N/A"}:
        return None
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else None
        except ValueError:
            return None
    return _float(text)


def _float(text) -> float | None:
    try:
        v = float(text)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None
