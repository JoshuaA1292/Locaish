<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
  <img src="assets/logo.svg" alt="Locaish" width="240">
</picture>

**Live demo:** [locaish-564007129146.us-central1.run.app](https://locaish-564007129146.us-central1.run.app) — two scanned rooms, the Gemini scout, and the ClickHouse shot table, hosted on Cloud Run.

**Scan any room. Get a filming-ready digital twin and an instant tech scout
report — camera angles, sun schedule, equipment fit, acoustics — before anyone
drives out.**

Built for [Agentic Cinema: The Blockbuster Hackathon](https://agentic-cinema.devpost.com/),
ClickHouse track. Gemini (through the Agent Development Kit, served by
Vertex AI on the hosted demo) drives
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

## Coverage: the tedious day, done

The job a DP spends the recce on is *coverage*: for every shot the scene
needs, where the camera stands, at what height, on which lens, whether the
sightline is clean, whether there is room to back up, whether a window ends
up behind the actor -- then the overhead camera diagram and the typed shot
list. Locaish does that day:

1. **Breakdown** -- paste the scene (or a shot list). A Gemini agent turns it
   into shots and puts each character on a mark the room actually has.
2. **Placement loop** -- an ADK `LoopAgent` takes one shot at a time: asks
   the ClickHouse shot table for candidates (the compiled filter, or its own
   SQL through the official ClickHouse MCP server), places the best, renders
   the frame from the twin's gaussian field, and **shows the frame to
   Gemini**, which scores it as a DP would and can send the planner back to
   the table for a different height or lens.
3. **The deliverables** -- shot cards with the measured setup (lens, height,
   distance, depth of field, backlight), an overhead camera plan (SVG), a
   shot list (.txt), and every placed shot written to a `shot_plans` table
   in ClickHouse so "which of our locations holds this scene" is a query.
4. **The viewfinder** -- click any card to look through exactly that lens
   from exactly that spot in the 3D twin; `[` `]` change lens, drag to
   re-aim, and the page tells you the nearest swept setup to wherever you
   ended up.

**What it knows.** The ranking is not a guess: every rule is a working
convention of cinematography and location scouting with a source, reduced
to a measurement the twin makes — the 180-degree line as a cross-product
predicate, matched reverses (same lens, same distance), over-the-shoulder
geometry, the window as a three-quarter key (`key_quality`), depth behind
the subject and shooting into corners (`background_depth_m`,
`axis_wall_angle_deg`), room to back up, faces no closer than 1.4 m, and
height from mood. See [docs/CINEMATOGRAPHY.md](docs/CINEMATOGRAPHY.md).
What the table cannot hold — headroom, look room, what is actually in the
background — Gemini judges by looking at the rendered frame.

Without Gemini the same engine still plans a typed shot list
deterministically (`locaish coverage room.twin --brief scene.txt`); without
ClickHouse it plans against the in-memory sweep. Both backends answer the
same predicates, so a plan is the same plan wherever it was computed.

## Hackathon alignment

- **Partner (ClickHouse):** the shot-search sweep is a genuine ClickHouse
  workload — selective filters (shot size, lens, sightline, backlight) plus
  top-N ranking across hundreds of thousands of scored candidates per room.
  Schema, sort key and partitioning are chosen for that access pattern
  (`locaish/warehouse.py`). The agent's *only* database access is the
  official ClickHouse MCP server (`mcp-clickhouse`), spawned at runtime,
  read-only as it ships; bulk loading goes around it over
  `clickhouse-connect`, column-oriented.
- **Google Cloud:** the agent is a `google-adk` `LlmAgent` running Gemini
  through Vertex AI (`GOOGLE_GENAI_USE_VERTEXAI=TRUE` + application-default
  credentials), not a bare AI Studio key — see `locaish/agent/core.py`.
- **Agentic, not scripted:** the model is given a tool surface — SQL over the
  shot table via MCP, the scout report, a tape measure, a dolly simulator, a
  frame renderer — and decides what to call. It never touches geometry
  directly; every claim it makes traces to a tool result.
- **No other AI models, anywhere.** The reconstruction is classical
  structure-from-motion (COLMAP) and photometric stereo. Gemini is the only
  model in the system.

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
pip install -e ".[video]"              # plus: brew install colmap ffmpeg

locaish studio                          # the product: drop a room, ask the scout
locaish studio --showcase               # hosted mode: gallery of scanned rooms, uploads off
# Approve a scan in the studio to link it into twins/showcase; --showcase serves that root.
locaish demo clean                     # whole pipeline on a room with known truth
locaish ingest room.ply --lat 51.5074 --lon -0.1278 --heading 212
locaish ingest sweep.mov --view         # a video of the room, reconstructed
locaish inspect twins/room.twin        # summary and QA report
locaish view twins/room.twin           # self-contained WebGL viewer, no build step
locaish measure twins/room.twin --from -1,0,1 --to 1,0,1
locaish sweep twins/room.twin          # every camera setup, scored, into ClickHouse
locaish coverage twins/room.twin --brief scene.txt --agent   # plan a scene's coverage
locaish export twins/room.twin -o room.glb
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
  sweep is honestly absent rather than stitched on a guess. The chain is built
  to hold — dense high-resolution features, wide sequential overlap, loop
  closure when the walk circles back — and the QA report says how many frames
  fell out when it doesn't.
- **Install OpenMVS.** The dense stage runs COLMAP's PatchMatch on CUDA,
  OpenMVS's patch-match on any CPU (see `docs/BUILD_OPENMVS_MACOS.md`; the
  Dockerfile builds it automatically), and falls back to block matching only
  when neither exists — and block matching is markedly weaker on the blank
  walls rooms are made of.
- **Install Brush for the photoreal layer (optional).** The gaussian-splat
  view layer is trained per scene with [Brush](https://github.com/ArthurBrussee/brush)
  — download a release binary and either put `brush_app` on PATH or point
  `LOCAISH_BRUSH` at it. Without it the pipeline still produces the full
  measured twin; the viewer just has no photoreal mode and the coverage
  planner renders frames from the point cloud instead.
- **A video twin's surfaces are centimetres thick, not millimetres.** The
  pipeline widens its plane tolerances to the reconstruction's declared noise
  (`noise_hint_m`), which recovers the floor and walls — but the finer
  structure detections tuned against LiDAR accuracy, ceilings and window/door
  openings especially, usually decline on video input rather than guess. A
  scanner-app export gets all of them.
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


---

# Phase 2 — Insight (built)

One command turns a twin into the report a location manager charges for
(`locaish/film/`). It is structured by department because that is how a tech
scout is run: **space** (dimensions, standable area, what the capture actually
surveyed), **camera** (longest sightlines, which framings the room physically
allows), **grip** (which dollies, sliders and jibs fit, from a catalogue of
real equipment dimensions, and where), **sound** (an RT60 estimate from the
room's surfaces and volume), **light** (sunrise, golden hours and per-window
direct-sun intervals from real solar ephemeris — `locaish/film/daylight.py`,
pure astronomy, requires a georeference and says so when the heading was
assumed rather than measured). Every figure is measured from the twin or
labelled as an assumption, and the report opens with how far to trust it —
the QA verdict and scale confidence travel with every number downstream.

The report deliberately refuses to summarise itself into a score out of ten.
A location is right or wrong *for a particular scene*, and collapsing
"3.2 m of headroom, 1.4 s of reverb" into "7/10" throws away the only part
anyone can act on.

---

# Phase 3 — Search (built)

The shot brief — "a clean close-up, longer lens, nothing hot behind her" — is
answered in three parts.

**The sweep** (`locaish/film/sweep.py`). Every standable camera position, at
three working heights, on six primes, against every plausible subject mark:
each combination is checked against the twin's real geometry — sightline
marched through the occupancy grid, clearance and headroom read from the floor
maps, framing and depth of field from thin-lens arithmetic, backlight risk
from the detected windows. A room comes out as 30,000–300,000 scored rows,
in about a second.

**The warehouse** (`locaish/warehouse.py`). Those rows go into ClickHouse —
`MergeTree`, sorted `(location, shot_size, focal_mm)`, partitioned by
location so re-scanning a room is a partition drop. The access pattern the
schema serves is exactly a shot brief: selective filters, then top-N.

**The scout** (`locaish/agent/`). A Gemini agent on `google-adk` with two
kinds of tools: SQL over the shot table through the official `mcp-clickhouse`
MCP server (read-only), and measurement tools over the twin itself — the
scout report, a tape measure with line-of-sight, a dolly-move simulator, and
a renderer that draws the actual frame a proposed setup would capture from
the twin's own points. The instruction is blunt about the contract: a number
the model did not get from a tool is a number it may not state.

Ask it "find the cleanest 75 mm medium shot with no window behind the
subject" and it writes the filter, queries the table, sanity-checks the
winner, renders the frame, and explains the physical reasoning — with every
step visible in the chat's activity feed.

---

# Running the full product

Two external services, both free-tier friendly. Everything else is
`pip install -e ".[video]"` plus `colmap` and `ffmpeg` on PATH.

**Gemini via Vertex AI** (the hackathon configuration):

```bash
gcloud auth application-default login
export GOOGLE_GENAI_USE_VERTEXAI=TRUE
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=global   # gemini-3.6-flash lives on the global endpoint
# optional: export LOCAISH_GEMINI_MODEL=gemini-3.6-flash   (the default)
```

**ClickHouse** — either ClickHouse Cloud:

```bash
export CLICKHOUSE_HOST=xxxxx.clickhouse.cloud
export CLICKHOUSE_PASSWORD=...
```

or a local server for development:

```bash
docker run -d --name locaish-ch -p 8123:8123 \
  -e CLICKHOUSE_USER=default -e CLICKHOUSE_PASSWORD=locaish \
  clickhouse/clickhouse-server
export CLICKHOUSE_HOST=localhost CLICKHOUSE_PORT=8123 \
       CLICKHOUSE_USER=default CLICKHOUSE_PASSWORD=locaish \
       CLICKHOUSE_SECURE=false
```

Then:

```bash
locaish studio
```

Drop a video. When the twin lands, the sweep is loaded automatically and the
chat goes live. No room handy? Drop `examples/IMG_6086.twin` on the studio
instead: a finished scan of a real kitchen (29 m², 102,852 swept setups)
that skips reconstruction and lights up the whole product in a couple of
minutes. Without `CLICKHOUSE_HOST` the studio still works — the agent
says plainly that the shot table is offline and answers what the measurement
tools can. Without Google credentials the chat explains what to set.

**Deploying the showcase** (how the live demo ships): approve rooms in the
studio, stage `deploy/ctx`, then `./deploy/deploy.sh <gcp-project>`. The
image bundles the approved rooms, their splats, and a ClickHouse server
loaded from baked dumps; Gemini runs through Vertex AI on the Cloud Run
service identity, so no API key ships. Grant the service account
`roles/aiplatform.user` once.

**Deploying the full pipeline** — the root `Dockerfile` builds a Cloud
Run-ready image with the reconstruction toolchain (CPU-only is fine; dense
stereo falls back from CUDA PatchMatch to semi-global matching):

```bash
gcloud run deploy locaish --source . --region us-central1 \
  --memory 8Gi --cpu 4 --timeout 3600 \
  --set-env-vars GOOGLE_GENAI_USE_VERTEXAI=TRUE,GOOGLE_CLOUD_LOCATION=global,CLICKHOUSE_HOST=...,CLICKHOUSE_PASSWORD=...
```
