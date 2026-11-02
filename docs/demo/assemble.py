"""Assemble the demo from a timeline: clips, stills, cards, captions, narration.

    python docs/demo/assemble.py timeline.json out.mp4

timeline.json:
{
 "width": 1920, "height": 1080, "fps": 30,
 "segments": [
  {"video": "build/phone_8x.mp4", "seconds": 8, "in": 0,          # a clip, trimmed (looped/held if short)
   "caption": "one phone · 48 seconds", "caption_at": [1.0, 7.5],
   "voice": "build/voice/01.wav", "voice_at": 0.5,
   "label": "render"},                                            # optional corner tag
  {"card": {"title": "Locaish", "sub": "shoot the scene before you book the room"}, "seconds": 3},
  {"image": "path.png", "seconds": 4, "zoom": 1.06},               # a still with a slow push
  {"end": {"title": "Locaish", "lines": ["Muse Spark · Muse Wonder · Muse Image · SAM 3.1", "..."]}, "seconds": 6}
 ]
}
Every segment becomes seg_NN.mp4 (video + stereo audio, exact length), then
they are concatenated with 0.4 s crossfades of video and audio.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SERIF = "/System/Library/Fonts/NewYork.ttf"
SANS = "/System/Library/Fonts/SFNS.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"
ACCENT = (232, 160, 72)


def font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except Exception:  # noqa: BLE001
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia.ttf" if "NewYork" in path
                                  else "/System/Library/Fonts/Helvetica.ttc", size)


def caption_png(text: str, W: int, H: int, out: Path) -> Path:
    """A lower-third caption: a quiet band at the bottom left, serif, white."""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = font(SERIF, 40)
    pad = 26
    tw = d.textlength(text, font=f)
    x0, y0 = 72, H - 72 - 40 - 2 * pad
    d.rounded_rectangle([x0, y0, x0 + tw + 2 * pad, y0 + 40 + 2 * pad], radius=6, fill=(0, 0, 0, 170))
    d.text((x0 + pad, y0 + pad - 4), text, font=f, fill=(255, 255, 255, 255))
    im.save(out)
    return out


def label_png(text: str, W: int, H: int, out: Path) -> Path:
    """The product's own tag, top right: A RENDER / A MEASUREMENT."""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = font(SANS, 22)
    t = text.upper()
    tw = d.textlength(t, font=f)
    pad = 12
    x1, y0 = W - 72, 56
    d.rounded_rectangle([x1 - tw - 2 * pad, y0, x1, y0 + 22 + 2 * pad], radius=4, fill=(0, 0, 0, 190))
    d.text((x1 - tw - pad, y0 + pad - 2), t, font=f, fill=ACCENT + (255,))
    im.save(out)
    return out


def card_png(title: str, sub: str | None, W: int, H: int, out: Path, lines: list[str] | None = None) -> Path:
    im = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    ft = font(SERIF, 148)
    tw = d.textlength(title, font=ft)
    y = H * 0.40 - 74
    d.text(((W - tw) / 2, y), title, font=ft, fill=(255, 255, 255))
    y += 180
    if sub:
        fs = font(SANS, 34)
        sw = d.textlength(sub, font=fs)
        d.text(((W - sw) / 2, y), sub, font=fs, fill=(190, 190, 190))
        y += 70
    for ln in lines or []:
        fl = font(MONO, 26)
        lw = d.textlength(ln, font=fl)
        d.text(((W - lw) / 2, y), ln, font=fl, fill=(150, 150, 150))
        y += 46
    im.save(out)
    return out


def probe_seconds(p: str | Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip() or 0)


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def build_segment(seg: dict, i: int, W: int, H: int, fps: int, root: Path, work: Path) -> Path:
    secs = float(seg["seconds"])
    out = work / f"seg_{i:02d}.mp4"
    inputs: list[str] = []
    vf: list[str] = []
    # ---- the picture
    if "card" in seg or "end" in seg:
        c = seg.get("card") or seg.get("end")
        png = card_png(c["title"], c.get("sub"), W, H, work / f"card_{i:02d}.png", lines=c.get("lines"))
        inputs += ["-loop", "1", "-framerate", str(fps), "-t", f"{secs:.3f}", "-i", str(png)]
        chain = "[0:v]format=yuv420p"
        if seg.get("fade", True):
            chain += f",fade=t=in:st=0:d=0.6,fade=t=out:st={max(secs - 0.6, 0):.2f}:d=0.6"
        vf.append(chain + "[v0]")
    elif "image" in seg:
        src = root / seg["image"]
        z = float(seg.get("zoom", 1.0))
        frames = int(secs * fps)
        if z > 1.0:
            # One input frame; zoompan emits `frames` frames from it with a
            # slow push in (the window shrinks as `on` runs 0..frames).
            inputs += ["-i", str(src)]
            vf.append(f"[0:v]scale={W * 2}:{H * 2}:force_original_aspect_ratio=increase,crop={W * 2}:{H * 2},"
                      f"zoompan=z='1+({z - 1:.4f})*on/{frames}':d={frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps={fps},"
                      f"trim=duration={secs:.3f},setpts=PTS-STARTPTS,format=yuv420p[v0]")
        else:
            inputs += ["-loop", "1", "-framerate", str(fps), "-t", f"{secs:.3f}", "-i", str(src)]
            vf.append(f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,format=yuv420p[v0]")
    else:
        src = root / seg["video"]
        start = float(seg.get("in", 0.0))
        speed = float(seg.get("speed", 1.0))
        inputs += ["-ss", f"{start:.3f}", "-i", str(src)]
        chain = f"[0:v]setpts=PTS/{speed}," if speed != 1.0 else "[0:v]"
        chain += (f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"
                  f"fps={fps},tpad=stop_mode=clone:stop_duration={secs:.2f},trim=duration={secs:.3f},setpts=PTS-STARTPTS,format=yuv420p[v0]")
        vf.append(chain)
    n_in = 1
    last = "v0"
    # ---- overlays
    if seg.get("caption"):
        a, b = seg.get("caption_at", [0.6, secs - 0.4])
        png = caption_png(seg["caption"], W, H, work / f"cap_{i:02d}.png")
        inputs += ["-loop", "1", "-framerate", str(fps), "-t", f"{secs:.3f}", "-i", str(png)]
        vf.append(f"[{n_in}:v]format=rgba,fade=t=in:st={a:.2f}:d=0.35:alpha=1,fade=t=out:st={max(b - 0.35, a):.2f}:d=0.35:alpha=1[c{i}]")
        vf.append(f"[{last}][c{i}]overlay=0:0:format=auto[v{n_in}]")
        last = f"v{n_in}"
        n_in += 1
    if seg.get("label"):
        png = label_png(seg["label"], W, H, work / f"lab_{i:02d}.png")
        inputs += ["-loop", "1", "-framerate", str(fps), "-t", f"{secs:.3f}", "-i", str(png)]
        vf.append(f"[{last}][{n_in}:v]overlay=0:0:format=auto[v{n_in}]")
        last = f"v{n_in}"
        n_in += 1
    # Video only: the narration is laid on the finished picture as one
    # track, so a sentence may run past the segment it starts in.
    cmd = ["ffmpeg", "-y", "-v", "error"] + inputs + ["-filter_complex", ";".join(vf), "-map", f"[{last}]",
           "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-r", str(fps), "-an", "-t", f"{secs:.3f}", str(out)]
    run(cmd)
    return out


def concat(segs: list[Path], out: Path, xfade: float = 0.4) -> tuple[Path, list[float]]:
    """Crossfade the video segments into one silent picture. Returns the
    output and each segment's start time on the finished timeline."""
    durs = [probe_seconds(s) for s in segs]
    starts = [0.0]
    for k in range(1, len(segs)):
        starts.append(starts[-1] + durs[k - 1] - xfade)
    if len(segs) == 1:
        out.write_bytes(segs[0].read_bytes())
        return out, starts
    inputs: list[str] = []
    for s in segs:
        inputs += ["-i", str(s)]
    vf: list[str] = []
    vprev = "0:v"
    for k in range(1, len(segs)):
        vf.append(f"[{vprev}][{k}:v]xfade=transition=fade:duration={xfade}:offset={starts[k]:.3f}[v{k}]")
        vprev = f"v{k}"
    run(["ffmpeg", "-y", "-v", "error"] + inputs + ["-filter_complex", ";".join(vf), "-map", f"[{vprev}]",
         "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "-an", str(out)])
    return out, starts


def lay_voices(video: Path, voices: list[tuple[float, Path]], out: Path) -> Path:
    """One narration track over the picture: each clip at its absolute
    second, mixed without normalisation so a sentence keeps its level."""
    total = probe_seconds(video)
    inputs = ["-i", str(video), "-f", "lavfi", "-t", f"{total:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
    vf = ["[1:a]anull[s0]"]
    labels = ["s0"]
    for k, (at, path) in enumerate(voices):
        inputs += ["-i", str(path)]
        ms = int(round(at * 1000))
        vf.append(f"[{k + 2}:a]aformat=sample_rates=48000:channel_layouts=stereo,adelay={ms}|{ms}[n{k}]")
        labels.append(f"n{k}")
    vf.append("".join(f"[{l}]" for l in labels) + f"amix=inputs={len(labels)}:normalize=0:duration=first[a]")
    run(["ffmpeg", "-y", "-v", "error"] + inputs + ["-filter_complex", ";".join(vf), "-map", "0:v", "-map", "[a]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(out)])
    return out


def main() -> None:
    tl = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    root = Path(sys.argv[1]).resolve().parent
    work = out.resolve().parent / "work"
    work.mkdir(parents=True, exist_ok=True)
    W, H, fps = int(tl.get("width", 1920)), int(tl.get("height", 1080)), int(tl.get("fps", 30))
    segs = [build_segment(s, i, W, H, fps, root, work) for i, s in enumerate(tl["segments"])]
    silent, starts = concat(segs, work / "picture.mp4", float(tl.get("xfade", 0.4)))
    voices = [(starts[i] + float(s.get("voice_at", 0.3)), root / s["voice"])
              for i, s in enumerate(tl["segments"]) if s.get("voice")]
    lay_voices(silent, voices, out)
    for (at, path), i in zip(voices, [i for i, s in enumerate(tl["segments"]) if s.get("voice")]):
        end = at + probe_seconds(path)
        nxt = next((starts[j] + float(tl["segments"][j].get("voice_at", 0.3))
                    for j in range(i + 1, len(tl["segments"])) if tl["segments"][j].get("voice")), None)
        if nxt is not None and end > nxt:
            print(f"warning: {path.name} runs {end - nxt:.1f} s into the next narration")
    print(out, f"{probe_seconds(out):.1f} s")


if __name__ == "__main__":
    main()
