# The reconstruction stack

*What turns a phone video into a measured twin, which tool does each step,
what each one is licensed under, and what was measured when the choice was
made. Updated October 2026, when the hackathon's "no AI models" rule stopped
applying and the pipeline was rebuilt around the best free tools for the job.*

## The principle

Geometry first; models where geometry has nothing to say; every model output
verified against the geometry before it is believed. A pose is a
least-squares solve over pixel correspondences. A surface point comes from
multi-view stereo wherever the frames had texture to match, and from a
learned prediction only where they did not, and only when neighbouring views
predict the same depth. The room's size comes from three estimators that
share no evidence, combined so that disagreement can only widen the error
bar. Nothing here changes the twin's bargain: a `pass` verdict means the
numbers are right, and the report says where every number came from.

## The stages

| Stage | Tool | Licence | Runs on |
|---|---|---|---|
| Decode and frame selection | ffmpeg; sharpest frame per timeline slice | LGPL/GPL (system binary) | CPU |
| Keypoints | **ALIKED** (`aliked-n16rot`, ONNX, inside COLMAP 4.x); SIFT fallback | BSD-3 (ALIKED); BSD-3 (COLMAP) | CPU (ONNX runtime) |
| Matching | COLMAP brute-force descriptor matcher; **LightGlue** ONNX opt-in | BSD-3; Apache-2.0 (LightGlue) | CPU, or GPU on a CUDA build |
| Loop closure | COLMAP vocabulary tree, **built per sweep** from its own ALIKED descriptors | BSD-3 | CPU |
| Camera solve | **GLOMAP** global mapper (inside COLMAP 4.x); incremental mapper fallback | BSD-3 | CPU |
| Dense stereo | COLMAP PatchMatch (CUDA) → remote GPU PatchMatch → **OpenMVS** (CPU) → OpenCV SGBM | BSD-3; **AGPL-3.0 (OpenMVS)**; Apache-2.0 (OpenCV) | GPU or CPU |
| Metric scale and learned fill | **MoGe-2** (`Ruicheng/moge-2-vitl`, 326M params) | MIT (code and weights); DINOv2 encoder Apache-2.0 | Apple GPU, CUDA, or CPU |
| Scale anchors | camera height, door height, metric model; inverse-variance fusion in log space | (ours) | CPU |
| Cleaning and refinement | ray-contradiction filter, kNN trim, moving-least-squares projection | (ours) | CPU |
| Photoreal view layer | **Brush** 3D Gaussian Splatting, per scene | Apache-2.0 | Apple GPU (Metal), CUDA, or WebGPU |

**On OpenMVS.** It is the strongest CPU stereo and it is AGPL-3.0. It runs
as a separate executable with no code linked into Locaish, which is the
conventional boundary, but a hosted service built on it should take a view
on AGPL before shipping. The CUDA PatchMatch path is COLMAP's own (BSD) and
is the right stereo for a GPU host; `docs/GCP_GPU_DENSE.md` describes
shipping that one stage to a GPU instance.

## What was measured

All on this machine (Apple M4, 32 GB) unless stated.

**Features, six 2400 px frames of a real kitchen, five sequential pairs each.**

| Detector / matcher | Keypoints per frame | Verified matches per pair | Extraction | Matching |
|---|---|---|---|---|
| SIFT (16384 budget) / brute force | 2,682 | 117 | 0.9 s | 0.4 s |
| ALIKED (min score 0.2) / LightGlue | 908 | 302 | 6.7 s | 3.9 s |
| ALIKED 2048 at 1600 px / brute force | 2,043 | 524 | ~0.7 s per frame | 0.03 s per pair |
| ALIKED 1024 at 1600 px / LightGlue | 1,022 | 456 | | 0.28 s per pair |
| ALIKED 3332 at 1600 px / LightGlue | 3,332 | 915 | | 1.8 s per pair |

LightGlue's cost is quadratic in keypoints, which is why the brute-force
matcher is the default: at 2048 ALIKED keypoints it returns 4.5 times SIFT's
verified matches for a thirtieth of a second a pair. LightGlue is the
opt-in (`LOCAISH_MATCHER=lightglue`) for a GPU host where that cost is
nothing.

**Loop closure.** COLMAP's distributed vocabulary tree (built on SIFT) with
ALIKED descriptors: zero pairs, every time. A 4096-word tree built from the
sweep's own descriptors (80k descriptors, 44 s) found 13 loop pairs across 24
frames on top of 89 sequential ones. The pipeline builds one per sweep,
capped at 120k descriptors.

**Metric scale, kitchen sweep (`IMG_6086`), MoGe-2 ViT-L on the Apple GPU.**
The camera-height prior had put the scale at 0.696 m per reconstruction
unit with a ±12 % bar and nothing to check it against (no door in shot).
Three frames of MoGe-2 depth, each aligned to the sparse points COLMAP
triangulated in that frame, read 0.742, 0.673 and 0.641, with a per-frame
log-MAD of 0.03 to 0.08 against the sparse depth. MoGe-2's own estimate of
the focal length (1019 to 1126 px at 2400 px wide) bracketed COLMAP's
calibrated 1047 px. Inference is 1.5 s a frame at 1024 px on the M4 GPU
after a one-time 1.3 GB weight download.

**End to end, both showcase rooms, October 2026.** The twins the hackathon
shipped had their scale resting on the camera-height prior alone (no door
in shot), a ±12 % bar, and a `fail` verdict each.

| | Kitchen (`IMG_6086`, 45 s) | Room two (`IMG_6062`, 39 s) |
|---|---|---|
| Detector / frames registered | SIFT, 300 of 300 (poses reused, then bundle-adjusted to 0.55 px) | ALIKED at 2400 px and 6 fps, 236 of 236, 1,715 verified pairs at 460 matches each, 0.76 px after bundle adjustment |
| Camera solve | 211 s | 386 s including the per-sweep vocabulary tree and bundle adjustment |
| Dense stereo (OpenMVS, CPU) | 4.3 M points, 324 s | 3.7 M points, 226 s |
| Model frames aligned | 64 of 64, median log-MAD 0.039 | 64 of 64, median log-MAD 0.059 |
| Camera-height anchor | 0.693 m per unit | 0.256 |
| Model anchor | 0.676 (2.5 % apart) | 0.275 (7 % apart) |
| Scale confidence, before → after | 0.60 → 0.72 | 0.60 → 0.72 |
| Ceiling | unknown → 2.62 m | unknown → 2.61 m |
| Floor coverage | 48 % → 50 % | 24 % → 57 % |
| Gravity residual | 1.22° → 0.31° | 0.22° → 0.20° |
| QA checks failing, before → after | 5 → 2 | 3 → 2 |
| Learned stage time (Apple M4 GPU) | 150 to 210 s | 130 to 230 s |

The drift metric (worst wall bowing over 0.5 m tiles) is the one number
the fill can move the wrong way: on room two it reads 5 cm without the fill
and 11 cm with it, because a learned sheet extending a measured wall
inherits the model's low-frequency bend. On the kitchen it improved (11.5
to 8 cm). Both rooms still fail that check as they did before.

## Why these and not the others

| Candidate | Why not (today) |
|---|---|
| Depth Anything 3 metric-large | Apache-2.0 and good, but the package needs xformers, which means CUDA; the Giant and Nested checkpoints are CC BY-NC. Worth revisiting for a CUDA host. |
| VGGT, VGGT-Ω | Research weights are CC BY-NC; VGGT's commercial checkpoint is separate and the model is feed-forward multi-view, which overlaps COLMAP rather than complementing it. |
| MapAnything | An Apache-2.0 checkpoint exists (`facebook/map-anything-apache`) and it is metric and multi-view; heavier than MoGe-2 and not on PyPI. The strongest candidate to replace COLMAP's solve on a GPU host; not needed for the room-scale wins this stage targets. |
| MASt3R, DUSt3R | CC BY-NC. |
| MoGe-3 | Needs Triton (no macOS build). MoGe-2 is pinned to the last commit before MoGe-3 landed. |
| SuperPoint | Non-commercial licence. ALIKED is BSD and at least as good on interiors. |
| Original INRIA 3DGS trainer | Non-commercial. Brush is Apache-2.0 and runs on Metal. |

## Installing

```bash
brew install colmap ffmpeg            # COLMAP 4.x with ONNX: ALIKED + GLOMAP
pip install -e ".[video,learned]"     # OpenCV; PyTorch + MoGe-2 (pinned)
```

The first run downloads the MoGe-2 weights (about 1.3 GB) from Hugging Face
and the ALIKED ONNX models (a few MB) from COLMAP's releases, both cached
under `~/.cache`. Brush is a separate binary (`LOCAISH_BRUSH` or on PATH);
OpenMVS is optional on the CPU (`docs/BUILD_OPENMVS_MACOS.md`).

## Knobs

| Variable | Default | Meaning |
|---|---|---|
| `LOCAISH_FEATURES` | `auto` | `aliked` when the COLMAP build has it, else `sift`; force either |
| `LOCAISH_MATCHER` | `auto` | brute force; `lightglue` for the ONNX LightGlue matcher |
| `LOCAISH_LEARNED` | `1` | `0` disables the metric model stage entirely |
| `LOCAISH_LEARNED_FILL` | `1` | `0` keeps the scale anchor and leaves the cloud exactly as stereo measured it |
| `LOCAISH_METRIC_MODEL` | `Ruicheng/moge-2-vitl` | any MoGe-2 checkpoint; `moge-2-vitb-normal` is three times faster on a CPU |
| `LOCAISH_LEARNED_FRAMES` | 64 on a GPU, 24 on a CPU | registered frames handed to the model, spread over the sweep |
| `LOCAISH_MVS_LEVEL`, `LOCAISH_MVS_VIEWS`, `LOCAISH_MVS_VIEWS_FUSE` | 1, 8, 3 | OpenMVS resolution level and view counts |
| `LOCAISH_SPLAT`, `LOCAISH_SPLAT_STEPS`, `LOCAISH_SPLAT_RESOLUTION` | 1, 12000, 1280 | the Brush view layer |
| `--fps`, `--scale-factor` | per detector; none | frame rate pulled from the clip; a tape-measured override for the scale |

Every stage records what it did in `<recon>/manifest.json`: detector,
matcher, mapper, stereo backend, the model's per-frame scale readings, how
many frames aligned and how many were rejected, how many learned points were
kept and how many were left to stereo.

## What the learned stage does, precisely

1. **Choose frames.** Up to `LOCAISH_LEARNED_FRAMES` registered views, spread
   evenly over the sweep.
2. **Predict.** MoGe-2 at 1024 px long side, given the calibrated horizontal
   field of view from the COLMAP solve so its metres are tied to the real
   focal. Depth, validity mask and intrinsics are cached per frame under
   `<recon>/learned/`, stamped with the model id.
3. **Align.** Every sparse point in front of the camera is projected into the
   frame and the model's depth is read at that pixel. The median ratio of
   COLMAP depth to model depth is that frame's "units per metre"; the
   log-MAD of the ratios is its scatter. Frames with fewer than 40 points or
   a scatter over 0.25 are rejected.
4. **Anchor.** The median over aligned frames is the model's scale reading;
   the standard error of that median is the measured part of its error bar
   and the model's systematic bias (`log 1.12`) is the prior part. It goes
   into `metric.combine_scales` beside the camera-height and door anchors.
5. **Verify.** Each aligned frame's depth map is lifted to world points
   (every second pixel), pixels whose ray meets the predicted surface at
   more than about 78 degrees off its normal are dropped, and the rest are
   projected into the four aligned frames on either side. A point is kept
   when at least three of them predict the same depth along its ray to
   within 3 %.
6. **Collapse.** What survives is a shell a few centimetres thick around the
   true surface (measured on a real room: about 5 cm RMS against
   stereo-fitted planes, under 3 cm of bias, under 0.4 degrees of tilt),
   because the model's per-pixel noise is a few percent of distance and
   every view agrees on it. Moving least squares over 24 neighbours projects
   the shell onto its mid-surface; points more than 12 cm off a locally
   planar neighbourhood are dropped.
7. **Fill.** The sheet is averaged per 1.5 cm voxel and kept only where no
   stereo point lies within 6 cm. The count is written to the manifest and
   the twin's header; the warning text in the QA report says how many points
   came from the model and how many were left to stereo.

**Is the fill worth having?** Measured by switching it off
(`LOCAISH_LEARNED_FILL=0`) with everything else identical, on the kitchen:

| Kitchen | hackathon twin | anchor only, no fill | anchor and fill (default) |
|---|---|---|---|
| QA checks failing | 5 | 5 | 2 |
| Ceiling | unknown | unknown | 2.62 m, 39 % covered |
| Floor coverage | 48 % | 49 % | 50 % |
| Gravity residual | 1.22° | 1.06° | 0.31° |
| Worst wall RMS | 2.5 cm | 2.3 cm | 2.6 cm |
| Facing-wall parallelism | 2.1° | 8.0° | 2.9° |
| Drift | 11.5 cm | 7.5 cm | 8.3 cm |

| Room two | hackathon twin | anchor only, no fill | anchor and fill (default) |
|---|---|---|---|
| QA checks failing | 3 | 3 | 2 |
| Ceiling | unknown | unknown | 2.61 m, 53 % covered |
| Floor coverage | 24 % | 25 % | 57 % |
| Gravity residual | 0.22° | 0.27° | 0.20° |
| Worst wall RMS | 1.9 cm | 1.9 cm | 2.4 cm |
| Facing-wall parallelism | 3.4° | 3.8° | 3.5° |
| Drift | 7.0 cm | 5.3 cm | 10.7 cm |

The wall-arrangement numbers on the kitchen move by several degrees between
runs whose clouds differ only slightly, with or without the fill; that is
the capture's own drift meeting a solver that votes over cabinet fronts and
through-door surfaces, and it is the next thing to fix. The fill is on by
default because on both rooms it leaves fewer checks failing than its own
no-fill twin, and it is the only source of a ceiling on a capture that never
looked up. Its cost is honest and visible: half a centimetre on the worst
wall's RMS, and on room two a worse drift reading.

Two things were tried and reverted, and the reasons are kept in the code:
a per-frame scale-plus-shift alignment (on frames with a narrow depth range
the two are collinear; fitted shifts reached 13 % of depth and whole depth
maps moved), and a looser vote of two views at 4 % (it thickened the walls
stereo had already measured by about a centimetre RMS on both test rooms).

After the global camera solve a full COLMAP bundle adjustment now runs over
every pose, point and the shared intrinsics; GLOMAP's own refinement is
lighter, and a room walk's residual drift lives in exactly the long-range
consistency a final joint solve recovers.

## What is next (stage 2)

- **Metric poses from the phone.** An ARKit capture path that records camera
  poses and LiDAR depth makes scale, gravity and heading a measurement rather
  than three estimators, and leaves COLMAP as a refinement. The single most
  valuable item for the product, independent of anything here.
- **A GPU host.** CUDA PatchMatch replaces OpenMVS (and its AGPL question),
  LightGlue becomes free, MoGe-2 drops to 60 ms a frame, and MapAnything
  becomes a candidate to solve poses and geometry in one pass.
- **Exteriors.** A different reconstruction problem; nothing here assumes
  it, nothing here solves it.
