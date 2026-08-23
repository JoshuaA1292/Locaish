# Locaish

**Scan any room. Get a filming-ready digital twin and an instant tech scout
report — camera angles, sun schedule, equipment fit, acoustics — before anyone
drives out.**

Built for [Agentic Cinema: The Blockbuster Hackathon](https://agentic-cinema.devpost.com/),
ClickHouse track. Gemini (via Google Cloud Agent Builder / Vertex AI) drives
an agent over a real measurement engine — every number in the output traces
to a ray cast, an ephemeris lookup, or a ClickHouse query, never to the model.

## The problem

A location scout's day rate exists because someone has to physically stand in
a room and answer: does the light do what the script needs, does the dolly
track fit, will the room ring on the boom mic. That trip happens *after* a
location is already a finalist — which means every finalist got picked on a
photo and a guess.

## What it does
1. **Scan** — any room, any scanner that exports a point cloud or gaussian
   splat (Scaniverse, Polycam, Luma, or raw COLMAP output), *or* just a video
   of someone walking around the space, from any phone with any camera.
2. **Reconstruct** — the scan becomes a metric, gravity-aligned, georeferenced
   digital twin. This has to work on *any* room handed to it, not just the
   ones we built test fixtures for — see Phase 1.
3. **Insight** — one command produces the report a location manager charges
   for: photography windows, lighting schedule through actual apertures,
   grip fit against real equipment dimensions, an acoustic estimate.
4. **Search** — an agent sweeps thousands of camera setups against a shot
   brief and returns a rendered frame with the physical reasoning behind it.

## Hackathon alignment

- **Partner (ClickHouse):** the shot-search sweep is a genuine ClickHouse
  workload — selective filters (light phase, height band) plus top-N ranking
  across hundreds of thousands of scored candidates. Schema, sort key, and
  indexes are chosen for that access pattern, not decorative.
- **Google Cloud:** the agent runs on Gemini through Vertex AI / Agent
  Builder, not a bare AI Studio key — this has to be true in the demo, not
  just "supported."
- **Agentic, not scripted:** the model is given a tool surface (search,
  diagnose, study daylight, render) and decides what to call. It never
  touches geometry directly; every claim it makes traces to a tool result.

## Non-negotiable constraint

**Every claim has to generalize to a room we've never seen before.** The
demo must run on a scan taken live or on camera, not only on fixtures built
to make the pipeline look good. If a capability only works on the sample
room, it doesn't ship in the pitch — it goes in a "known limitations"
section instead, same as before, but that section has to be small.

---

# Phase 1 — Scan (built)

Scan a room with an iPhone or iPad Pro, export it, and get back a **metric,
gravity-aligned, yaw-normalised twin** with a report that says how far to trust
it. See [CAPTURE.md](CAPTURE.md) for how to take the scan.

```bash
pip install -e .

locaish demo clean                     # whole pipeline on a room with known truth
locaish ingest room.ply --lat 51.5074 --lon -0.1278 --heading 212
locaish ingest sweep.mov --view         # a video of the room, reconstructed
locaish inspect twins/room.twin        # summary and QA report
locaish view twins/room.twin           # self-contained WebGL viewer, no build step
locaish measure twins/room.twin --from -1,0,1 --to 1,0,1
locaish export twins/room.twin -o room.glb
locaish studio                          # drop a video on a page instead
```

Reads PLY (binary LE/BE and ASCII, including gaussian-splat exports), OBJ+MTL,
GLB/GLTF and STL, from Scaniverse, Polycam, Luma, RoomPlan or COLMAP — and MOV,
MP4, MKV and friends, which get reconstructed first.

## From video

No LiDAR, no scanning app, no export step: film the room and hand over the file.

```bash
locaish ingest sweep.mov --view
```

The reconstruction is **classical, end to end** — no neural networks anywhere
in the pipeline. Frames are pulled densely from the sweep, SIFT features are
matched between neighbouring frames, bundle adjustment solves every camera pose
and a sparse cloud (COLMAP), and photometric stereo densifies the result — GPU
PatchMatch where CUDA exists, semi-global block matching on the CPU everywhere
else. SIFT is from 1999 and bundle adjustment is older; there is no model file,
nothing was trained, and every point traces to a corner detected in an image
and a least-squares solve over reprojection error.

That choice is load-bearing, not aesthetic: the Agentic Cinema rules permit
only Google Cloud AI tools and prohibit any other AI model regardless of
vendor, while explicitly allowing open-source non-AI software. A pretrained
reconstruction network is an AI model whoever wrote it; SIFT is not. This path
is compliant by construction rather than by argument.

Two things are worth knowing before trusting the output.

**Video has no scale.** A kitchen and a doll's-house kitchen produce identical
footage; parallax recovers shape, never size. So the metres come from outside
the geometry — from two physical anchors that share no evidence:

- **how high the phone was.** Gravity is known from the camera poses, so the
  drop from the camera path to the floor is a length whose value in metres we
  already know to within a few centimetres;
- **any doorway in the room.** A door leaf is 1981–2032 mm almost everywhere on
  earth, which makes it the tightest anchor available in a room nobody measured.
  It is recognised by shape rather than by size — sitting on the floor, between
  1.5 and 3.5 times as tall as it is wide — because a height threshold would
  assume the very answer being solved for. This one runs as a second pass, since
  apertures are found in the canonical frame and the canonical frame needs a
  scale to exist.

They are combined by inverse-variance weighting in log space, and — this is the
part that matters — if they disagree by more than their own error bars allow,
the combined uncertainty is inflated until it covers the disagreement. Two
confident estimators cannot produce one confident wrong answer. An estimator
that agrees with itself perfectly can still be off by a factor of two:
self-consistency measures precision, and reporting precision as accuracy is how
a twin ends up claiming a 5.9 m ceiling, to ±4%, in a room that is 2.6 m tall.
That is not hypothetical — an earlier version of this pipeline did it during
development, and the independent-anchor design is what caught it.

The result lands in the QA report as `scale_confidence` rather than being
laundered into a declared unit. If you need better, tape-measure one length in
the room and pass `--scale-factor`.

**Gravity comes from how you held the phone.** Feature matching knows nothing
about up — a room filmed upside down reconstructs perfectly happily upside
down. But the camera poses give it away: averaging each frame's own down-axis
recovers gravity directly, and the frames' agreement about it is checked before
the hint is used at all. It breaks ties in the vertical-axis choice and votes
hard on which end is the floor, but it can never outvote the room's own
geometry — film pointing at the floor the whole time and the furniture still
wins.

**Holes get closed by carving, not by guessing.** A capture by someone who is
not a surveyor comes back with the floor hidden under furniture, walls glanced
at once, and usually no ceiling at all. Rather than interpolate a surface across
those gaps, the pipeline asks where the camera *proved* there was nothing: every
camera-to-point ray sweeps out empty space, and the boundary of the volume they
sweep is closed by construction. Where the sweep met a wall the boundary sits on
the wall; where it never looked, it sits at the frontier of what was seen, and
every such vertex is labelled — `Mesh.filled`, muted colours, and a line in the
QA report. It cannot seal a doorway you walked through; that is a test, not a
hope. Without camera poses it declines entirely rather than guess.

**Classical SfM needs frames that chain.** Matching two views of a blank
painted wall taken a second apart finds nothing, so a sweep sampled sparsely in
time fragments into disconnected pieces. Measured on a real capture, 72 frames
produced four fragments whose largest held 28 of them, while 251 frames of the
same clip registered 240 into a single model with a mean reprojection error
under a pixel. So frames are pulled densely — `--fps`, default 8 — and the
binding constraint is temporal density, not coverage. When the chain does
break, the largest fragment is reconstructed and the QA report says how many
frames were left out, rather than joining unrelated coordinate frames on a
guess.

The reconstruction is cached beside the twin, so re-ingesting with different
options costs seconds rather than minutes. `--refresh` forces it to run again.

If a terminal is the wrong interface for the person holding the phone:

```bash
locaish studio
```

serves a loopback page you drag the video onto, streams the pipeline's own
stage names back as it works, and hands you the twin and the viewer at the end.
Same code path as `ingest`, same QA, no extra dependencies.

Requires `ffmpeg` and `colmap` on PATH (`brew install colmap ffmpeg`) and the
optional extra: `pip install -e ".[video]"` (OpenCV, for the CPU stereo
fallback). No weights to download, no GPU required — CUDA merely upgrades the
dense stage from block matching to PatchMatch.

## What a twin is

Z up, metres, floor at `z = 0`, dominant wall parallel to +X, origin in the
middle of the floor. Point cloud, mesh, fitted floor/ceiling/wall planes,
detected windows and doors, the floor footprint polygon, capture bounds (where
the scanner actually went), a georeference, and a QA report. One `.twin` file.

## The promise

> **A twin whose QA verdict is `pass` is accurate.**

Not "every room works". Some rooms are genuinely ambiguous — a nearly cubical
space gives almost nothing to distinguish up from sideways, and a room whose
dimensions in inches are as plausible as its dimensions in centimetres cannot
be told apart from geometry alone. Those degrade to `warn` or `fail`, and then
they are allowed to be wrong. A confident guess is the one thing that isn't
allowed.

## Measured

25 rooms generated from a seed nothing was tuned on — random dimensions 2.4–14 m
wide and 2.1–5.5 m tall, random unit (m/cm/mm/in/ft), tilted up to 10°, rotated,
translated up to 80 m, 3–15 mm noise, 300–1200 points/m², up to 16 occlusion
patches. Reproduce with `pytest tests/test_generalization.py`.

| | result |
|---|---|
| Twins that were wrong while reporting `pass` | **0 of 25** |
| Rooms trusted (`pass`) | 15 of 25 |
| Worst plan-dimension error, trusted twins | **1 mm** |
| Worst ceiling-height error, trusted twins | **2 mm** |
| Opening count exact, trusted twins | 14 of 15 |
| Opening size error | 8 mm median, 41 mm worst |

The ten refusals are honest ones: the ceiling-plausibility check firing on rooms
5 m and taller, and low scale confidence where the unit was genuinely ambiguous.

## Known limitations

- **Tall rooms get flagged.** The ceiling check warns outside 2.1–5.0 m. It
  reads as an opinion about architecture but is really the check that catches a
  misread unit, since a wrong unit puts the ceiling 2.5× or 3.3× off. Widening
  it was tried and reverted — it let genuinely wrong twins report `pass`. A
  sound stage or a warehouse will need the warning cleared by hand.
- **cm and inches are not always separable.** They differ by 2.54×, and some
  rooms are architecturally plausible at both. Those come back at low confidence
  and fail the scale check rather than guessing.
- **One opening false positive in ~25 rooms.** A sensor shadow beside a doorway
  is the size and shape of a small window, has nothing behind it exactly as a
  window does, and borrows the doorway's reveal as evidence. Neither depth nor
  shape separates them.
- **A twin from video is metric to tens of percent, not ±10 mm.** The shape is
  measured; the size is inferred, and how well the two independent estimators
  agree is what sets the error bar — which on a real capture has been wide. The
  QA report carries it as `scale_confidence`. Every accuracy number in the table
  above is for scan-file input. Video needs `--scale-factor` from a tape measure
  before any claim at centimetre precision.
- **A video sweep has to actually sweep.** Reconstruction only knows what the
  frames saw, and a pan from one spot gives feature matching no baseline to
  triangulate from — the twin fails the coverage check, correctly. See
  [CAPTURE.md](CAPTURE.md).
- **A broken matching chain drops frames.** A pause on a textureless wall can
  fragment the reconstruction; the largest fragment wins and the rest of the
  sweep is honestly absent rather than stitched on a guess. Film with steady
  overlap and the chain holds.
- **The CPU stereo fallback is weaker on blank walls.** Semi-global matching
  returns nothing where there is nothing to match, so an undecorated room
  densifies thinly without CUDA. Honest sparsity, but sparsity.
- **Nobody films the ceiling**, so `ceiling_z` usually comes back unknown and no
  ceiling height is reported. The completion pass closes the room at the
  frontier of what was swept and labels it inferred; that is not a measurement
  of a ceiling and is not offered as one.
- **Camera poses are not read from Polycam or Scaniverse archives.** Only glTF
  camera nodes and Bundler-style PLY. Without poses, capture bounds are inferred
  from reconstructed floor, which is more generous than a record of where you
  actually stood.
- **An interior scan is a shell, not a solid.** Reconstructed meshes are not
  watertight, and the QA report says so.
- Not read yet: E57, LAS/LAZ, PCD, USDZ, FBX.

## How it's verified

`pytest tests/` — 156 tests. `test_contract.py` pins the frozen data model,
`test_accuracy.py` checks eight catalogue fixtures to ±15 mm, `test_qa.py`
checks the report discriminates rather than decorates, `test_video.py` holds the
video front-end to known answers without running COLMAP — a gravity estimate
built from cameras tilted 20° has to come back 20° off, not 0 — and
`test_generalization.py` is the one that matters: rooms built from a seed the
tuning never saw, holding the pipeline to the promise above.

Ground truth comes from `locaish/fixtures.py`, which builds rooms analytically
and then hands them over deliberately broken — tilted, rotated, in the wrong
unit, occluded, drifting — so "1:1" is a measurement against a known number
rather than an assertion.

