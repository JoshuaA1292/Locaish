<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
  <img src="assets/logo.svg" alt="Locaish" width="240">
</picture>

**Scout six locations. Drive to two.**

A location scout films a room with a phone. Locaish turns the walkthrough
into a measured 3D twin and hands back the tech scout's answers before
anyone books a van: the sun through every window on the shoot date, which
dollies fit, the lens that holds each framing, the room's acoustics, and --
for a scene -- which of your candidate rooms physically holds every shot.

**Live demo:** [locaish-564007129146.us-central1.run.app](https://locaish-564007129146.us-central1.run.app)
-- two scanned rooms, the scout agent, and the shot table, hosted on Cloud Run.
**Pricing and pilots:** `/pricing` on any deployment. **What stands between this and revenue:** [docs/BOTTLENECK_RESEARCH.md](docs/BOTTLENECK_RESEARCH.md). Built in 17 days for
[Agentic Cinema](https://agentic-cinema.devpost.com/), won its track, now a
product.

## What you get

| | For whom | What it is |
|---|---|---|
| **Location package** | the scout, handing in | One page per room, printable, forwardable: dimensions with an honest error bar, where direct sun lands on the floor at 9, 12 and 15 on the shoot date, what gear fits, lens table, RT60, the practical fields (address, power, parking, access, contact) and a link into the 3D twin. `/package/<room>?date=YYYY-MM-DD` |
| **Projects** | the head of production, bidding | One job, one shot list, one shoot date, every candidate room. Every shot is placed in every room by the measurement engine and the rooms are ranked: holds it, holds it with a compromise, cannot. `/projects` |
| **Share links** | the director, in another city | A read-only URL that opens the package, the ranking and the 3D twin on a phone. No account. Revocable. `/s/<token>` |
| **Capture page** | the scout, standing in the room | The five-step recipe and a camera button; the upload starts when filming stops. `/capture` |
| **What is in the room** | the scout, the art department | After the twin is built a vision model names what the capture's frames show (sofa, radiator, fridge, ceiling fan) and each name is placed in the room by casting the frame's ray into the measured cloud. Anything it missed or got wrong: click the thing in the twin, pick the photograph that saw it, type the name. Confirmed and provisional names are kept apart everywhere they appear. |
| **Coverage planner** | the DP | Paste the scene; the agent breaks it into shots, places each one, renders the frame, and draws the overhead camera plan. |
| **Viewfinder** | everyone | Stand in any setup in the 3D twin, change lens, drag to re-aim. |

Every figure traces to a measurement of the twin, a solar ephemeris, or a
catalogue dimension. The twin grades its own accuracy and the package says,
in plain words, how far to trust the metres. Nothing is estimated by a
language model; the model plans and reviews, the geometry answers.

## The scene in this room (Meta Global AI Developer Hackathon, Oct 26 to Nov 4 2026)

**A real room, real people in it at real heights, cameras you place and
inspect, and every angle rendered as if it were photographed.** CAD-like to
work in, real life to look at. On the studio page, under *The scene in this
room*:

- **People.** An actor is a name, a height, a description and a wardrobe.
  In the twin and in every measured frame they are articulated bodies with
  limbs and shading, standing, walking or sitting, cut correctly by the
  furniture in front of them. A headshot adds their face.
- **Objects.** A labelled sofa, desk or chair lifts into a solid you can
  move; the floor and the shot table are re-measured where it now stands.
- **Angles.** Paste a scene and every shot is placed from the measured shot
  table, with the other actors treated as obstacles, so nobody's back fills
  the lens. Or key a camera move yourself on a real rig, drag the keys in
  the twin, scrub it, look through it. Or tell the operator what the camera
  should do: Muse Wonder builds the move through the engine's tools and Muse
  Spark watches the frames as a DP until it holds.
- **Photographed.** Every frame is finished by Muse Image from the measured
  render, the scan's own photographs of that view, and the cast's looks:
  the real room, these people, this angle, consistent from frame to frame.
  The measured frame stays one toggle away and every finished frame is
  stamped a render. A move is cut to video both ways.
- **Walk into the location as it will be on the day.** Pick a date and an
  hour and say how the room will be dressed. Sixty views of the measured
  room, chosen to cover every wall, are finished by Muse Image with the
  sun through the real window at that hour, then painted onto the scan's
  own gaussians: where the model and the measurement agree on the
  structure the finished pixel is kept; where they disagree the scan keeps
  its colour and takes only the light. The walls stay where they were
  measured. Walk it in the browser (WASD) or in a headset (WebXR); stand
  where a view was finished and the photograph itself fades in. This is
  what a location scout would have brought back, before anyone drove there.

Meta's models do every job that reads, names, judges, plans or paints: Muse
Spark (`locaish/meta/vision.py`, the breakdown, the DP reviews of frames,
filmstrips and video), Muse Wonder (the operator, `locaish/agent/sim.py`;
the scout and planner through `locaish/meta/adk.py`), Muse Image
(`locaish/meta/finish.py`), SAM 3.1 (`locaish/meta/masks.py`) and the
open-weight SAM 3D models on Modal (`locaish/meta/sam3d.py`,
`docs/SAM3D_MODAL.md`) for bodies from a photograph. The reconstruction
stays geometry; nothing a model says becomes a measurement except the
stature anchor, through the fusion. The bodies are `locaish/product/figures.py`
and `locaish/film/render.py`; the engine is `locaish/film/sim.py`; the
rehearsal is `locaish/film/rehearsal.py`; the world is
`locaish/product/world.py`; the studio's stage is
`locaish/product/stage.py`. Plan and status: [docs/META_HACKATHON.md](docs/META_HACKATHON.md).

```bash
export META_API_KEY=...            # Muse Spark, Muse Wonder, Muse Image, SAM 3.1 (or put it in .env)
export LOCAISH_SAM3D_URL=... LOCAISH_SAM3D_TOKEN=...   # optional: modal deploy locaish/meta/modal_app.py
locaish studio                     # the scene appears under every finished twin
```

## The problem

A location scout's day rate exists because someone has to physically stand in
a room and answer: does the light do what the script needs, does the dolly
track fit, will the room ring on the boom mic. That trip happens *after* a
location is already a finalist -- which means every finalist got picked on a
photo and a guess. In commercials the decision is made during the bid, before
the job is awarded, by a director who is often in another city.

## How it works
1. **Capture** -- a phone video of the room, or a LiDAR export from Polycam,
   Scaniverse or Luma (PLY, GLB, GLTF, OBJ) for a twin metric to centimetres.
2. **Reconstruct** -- the capture becomes a metric, gravity-aligned,
   georeferenced digital twin with a self-grading QA report. It has to work
   on *any* room, not only the ones we built fixtures for -- see Phase 1.
3. **Survey** -- the scout report by department: space, camera, grip, sound,
   light. Lighting through the actual apertures from a solar ephemeris.
4. **Sweep** -- every standable camera position at three heights on six
   primes against every subject mark, scored, into ClickHouse (or kept
   in memory): the index of what a camera can do in this room.
5. **Deliver** -- the package, the project ranking, the share link.

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

## Built on

- **ClickHouse** holds the shot index: selective filters (shot size, lens,
  sightline, backlight) plus top-N ranking across hundreds of thousands of
  scored candidates per room; schema and partitioning in `locaish/warehouse.py`.
  Without it the same predicates run in numpy over the in-memory sweep, so a
  laptop plans the same plan the warehouse would.
- **Gemini through the Agent Development Kit** plans coverage and judges
  rendered frames; its only database access is the official ClickHouse MCP
  server. The product works without it (typed shot lists, deterministic
  placement, every package and project feature).
- **COLMAP, OpenMVS, MoGe-2, Brush** reconstruct; all open source and free
  for commercial use -- see [docs/RECONSTRUCTION.md](docs/RECONSTRUCTION.md).

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

The reconstruction is **geometry first, models where geometry has nothing to
say**. Frames are pulled from the sweep, ALIKED keypoints (a small learned
detector shipped inside COLMAP) are matched between neighbouring frames,
GLOMAP solves every camera pose jointly and a sparse cloud (COLMAP), and
multi-view stereo densifies the result — GPU PatchMatch where CUDA exists,
OpenMVS on the CPU elsewhere. Then a metric monocular geometry model
(MoGe-2) runs over a spread of the registered frames and contributes two
things, each verified against the geometry before it is believed: a third,
independent reading of the room's scale, and surface for the painted walls
stereo could not match, kept only where neighbouring frames predict the same
depth and only where stereo left a gap. Every pose still traces to pixel
correspondences and a least-squares solve; every learned point carries a
label saying where it came from. The full stack, with every licence, is in
[docs/RECONSTRUCTION.md](docs/RECONSTRUCTION.md).

Two things are worth knowing before trusting the output.

**Video has no scale.** A kitchen and a doll's-house kitchen produce identical
footage; parallax recovers shape, never size. So the metres come from outside
the geometry — from three anchors that share no evidence:

- **how high the phone was.** Gravity is known from the camera poses, so the
  drop from the camera path to the floor is a length whose value in metres we
  already know to within a few centimetres;
- **any doorway in the room.** A door leaf is 1981–2032 mm almost everywhere on
  earth, which makes it the tightest anchor available in a room nobody measured.
  It is recognised by shape rather than by size — sitting on the floor, between
  1.5 and 3.5 times as tall as it is wide — because a height threshold would
  assume the very answer being solved for. This one runs as a second pass, since
  apertures are found in the canonical frame and the canonical frame needs a
  scale to exist;
- **what a metric depth model makes of each frame.** MoGe-2 has learned what
  size things are from millions of rooms and answers in metres per frame;
  tied to the sparse points COLMAP triangulated in that same frame, each
  frame yields an independent reading of metres per reconstruction unit, and
  tens of frames give a median and a spread. Anatomy, carpentry and
  appearance: three priors with nothing in common.

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

**Structure from motion needs frames that chain.** Matching two views of a
blank painted wall taken a second apart finds nothing, so a sweep sampled
sparsely in time fragments into disconnected pieces. With SIFT, 72 frames of a
real capture produced four fragments whose largest held 28 of them, while 251
frames of the same clip registered 240 into one model. ALIKED holds the chain
across a wider baseline — on the same kitchen frames it returns four to five
times as many geometrically verified matches per pair as SIFT, from a third
as many keypoints — so its frame budget is lighter (`--fps`, default 5 at
1600 px, against SIFT's 8 at 2400 px). Loop closure for ALIKED uses a
vocabulary tree built from the sweep's own descriptors, since the tree COLMAP
distributes is built on SIFT. When the chain does break, the largest fragment
is reconstructed and the QA report says how many frames were left out, rather
than joining unrelated coordinate frames on a guess.

The reconstruction is cached beside the twin, so re-ingesting with different
options costs seconds rather than minutes. `--refresh` forces it to run again.

If a terminal is the wrong interface for the person holding the phone:

```bash
locaish studio
```

serves a loopback page you drag the video onto, streams the pipeline's own
stage names back as it works, and hands you the twin and the viewer at the end.
Same code path as `ingest`, same QA, no extra dependencies.

Requires `ffmpeg` and `colmap` on PATH (`brew install colmap ffmpeg`; COLMAP
4.x with ONNX support unlocks ALIKED, older builds fall back to SIFT) and the
optional extras: `pip install -e ".[video,learned]"` — OpenCV for the CPU
stereo fallback, and PyTorch plus MoGe-2 for the learned stage, whose weights
(about 1.3 GB) are fetched from Hugging Face on first use. No GPU required:
the model runs on an Apple GPU or a CPU, and CUDA merely upgrades the dense
stage to PatchMatch. Without the `learned` extra the pipeline runs as before
and the twin says so.

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
- **A twin from video is metric to a few percent, not ±10 mm.** The shape is
  measured; the size is inferred, and how well the independent estimators
  agree is what sets the error bar. On the two showcase rooms the metric
  depth model and the camera-height prior landed 2.5 % and 7 % apart, which
  puts the combined bar near ±8 % and the QA's `scale_confidence` at 0.72 to
  0.81; before the model stage it was ±12 % resting on one prior. Every
  accuracy number in the table above is for scan-file input. Video still
  needs `--scale-factor` from a tape measure before any claim at centimetre
  precision.
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
chat goes live.

## Running it for a customer

A design partner gets their own deployment and one passcode. Everything the
product layer stores -- projects, location details, share links, pilot
requests -- is JSON under `<root>/_product/` and `<job>/details.json`, so the
studio root is the whole state: back it up, and the deployment is backed up.

```bash
export LOCAISH_PASSCODE='their-team-passcode'      # gates every page except share links, /pricing, /capture
export LOCAISH_LEAD_WEBHOOK='https://hooks.slack.com/...'   # optional: pilot requests from /pricing
export LOCAISH_STRIPE_SCOUT_URL='https://buy.stripe.com/...'       # optional: payment links on /pricing
export LOCAISH_STRIPE_PRODUCTION_URL='https://buy.stripe.com/...'
export LOCAISH_CONTACT_EMAIL='you@locaish.com'
locaish studio --no-open --root /data/studio
```

Share links (`/s/<token>`) bypass the passcode by design: the producer sends
one to the director, the director opens it on a phone, nobody signs in. Revoke
from the room or project page. The hosted gallery (`--showcase`) keeps uploads
and project edits off and share links on. No room handy? Drop `examples/IMG_6086.twin` on the studio
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
