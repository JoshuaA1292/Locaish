<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
  <img src="assets/logo.svg" alt="Locaish" width="240">
</picture>

**Scan a room with a phone. Shoot the scene before anyone drives there.**

One fifty-second walk through a room becomes a measured 3D twin. Locaish then
does the tech scout's day in minutes: which lens holds each shot, where a
dolly fits, where the sun lands on the shoot date, and what every frame of
the scene will look like, in the real room, with the cast in it. Meta's
models read, plan, direct and paint inside that measurement; nothing they
say ever becomes a measurement.

**Live:** [locaish-564007129146.us-central1.run.app](https://locaish-564007129146.us-central1.run.app) (a scanned dorm, look-only)
· **Hackathon write-up:** [docs/SUBMISSION_META.md](docs/SUBMISSION_META.md)
· **The long version:** [docs/REFERENCE.md](docs/REFERENCE.md)

## Who it's for

Producers, DPs and location managers on commercials, indie film and TV.
Today a location is picked from photographs, then someone flies out with a
tape measure, then the tech scout finds out the dolly doesn't fit. Locaish
replaces the scouting trip, the tape measure, the pre-vis artist and the
guesswork.

## What it does

| | |
|---|---|
| **Measures the room** | Every camera pose of the walk is recovered and a metric twin is built around the path (COLMAP, GLOMAP, OpenMVS, a MoGe-2 metric anchor). Every dimension traces to the footage and the twin grades its own accuracy. |
| **Names what's in it** | Muse Spark looks at the frames and names the furniture; each name is pinned to a measured point by casting the frame's ray into the cloud. |
| **Sweeps every setup** | Thousands of candidate camera positions (the dorm: 9,666), each tested for lens, shot size, clear sightline, headroom and whether a dolly fits. Ask "where can a close-up stand" and it answers in metres. |
| **Breaks down a scene** | Paste the scene. Muse Spark turns it into beats and shots, the room places each one from the sweep and says which cannot be held ("this dolly needs 68 cm of clear floor; there are 30"). |
| **Stands the cast up** | Actors are names, heights and wardrobes. They stand as bodies on their marks in the twin and in every measured frame, cut by the real furniture. |
| **Operates the camera** | Key a rig yourself and look through the lens, or tell the operator what you want: Muse Wonder drives the camera through the engine's tools and Muse Spark reviews every frame as a DP. |
| **Photographs the scene** | The scan's own photograph nearest each planned camera is cropped to the lens, the measured bodies are placed into it by the geometry, and Muse Image paints only the people. Muse Spark checks that nothing else changed. |
| **Dresses the set** | SAM 3.1 masks a labelled thing in the frame that saw it best, SAM 3D lifts it into a solid, and the shot table is recomputed when you move it. |
| **Shows the day** | Pick the date and the hour. Sixty views of the room are finished with the sun where it will be and painted onto the scan; walk it in a browser or a headset. |

## Meta's models, inside a measurement

| Model | Job | Code |
|---|---|---|
| Muse Spark | names the room, breaks the scene into shots, reviews camera moves as a DP, judges every finished still against the real photographs | `locaish/meta/vision.py`, `locaish/agent/rehearsal.py`, `locaish/agent/sim.py`, `locaish/meta/finish.py` |
| Muse Wonder | the camera operator and the scout agent, through the engine's tools | `locaish/agent/sim.py`, `locaish/meta/adk.py` |
| Muse Image | paints the people into the scan's photographs; finishes the sixty views of the day | `locaish/meta/finish.py`, `locaish/product/world.py` |
| SAM 3.1 | masks movers out of the camera solve; masks a set piece | `locaish/meta/masks.py` |
| SAM 3D Objects, SAM 3D Body | lifts a set piece or a person into a solid, on Modal | `locaish/meta/sam3d.py`, `locaish/meta/modal_app.py` |

Every model output is checked against something real: geometry (labels,
shots, dolly fits), the scan's photographs (the finish), or another model
acting as judge that sends work back. The measured frame is always one
toggle away and every finished frame is stamped a render.

## What is new for the Meta hackathon

Locaish is a startup entry: the reconstruction is pre-existing work, the
scene layer is new. To be exact:

| Pre-existing (built June to September 2026 for Agentic Cinema) | New for this hackathon (24 October to 4 November 2026) |
|---|---|
| The 3D reconstruction from phone video and scan files: COLMAP, GLOMAP, OpenMVS, the MoGe-2 metric anchor, the canonical twin, structure and QA (`locaish/video/`, `locaish/scan/`, `locaish/geom/`, `locaish/formats/`) | Everything that reads, plans, directs, paints and lifts inside the twin with Meta's models (`locaish/meta/`): Muse Spark labels pinned to geometry, the scene breakdown and the review of frames, Muse Wonder as camera operator, Muse Image finishing (the photograph base, the day's sixty views), SAM 3.1 masks, SAM 3D on Modal |
| The scout report, the shot sweep and its ClickHouse table, the Gemini coverage planner, the location package, projects, share links, the WebGL viewer (`locaish/film/{sweep,report,coverage}.py`, `locaish/warehouse.py`, `locaish/agent/{core,coverage}.py`, `locaish/product/{package,pages,store}.py`) | The rehearsal (a scene placed, timed and walked inside the room with the cast as bodies), the camera simulator and real rigs, the set pieces, the world, the studio's scene page, and the finish contract that holds the painter to the measurement and the photographs (`locaish/agent/{rehearsal,sim}.py`, `locaish/film/{rehearsal,sim,render}.py`, `locaish/product/{stage,cast,figures,labels,setpieces,world}.py`, `locaish/viewer/stage.js`) |

The new work is 32 commits from 24 October onwards; `git log --since=2026-10-24` lists them, and
[docs/META_HACKATHON.md](docs/META_HACKATHON.md) carries the day-by-day status table.

## Run it

```bash
pip install -e ".[video,learned]"      # plus: brew install colmap ffmpeg
export META_API_KEY=...                 # Muse Spark, Muse Wonder, Muse Image, SAM 3.1
locaish studio                          # drop a phone video; the scene page appears under the twin
```

Optional: `LOCAISH_SAM3D_URL` and `LOCAISH_SAM3D_TOKEN` for SAM 3D on Modal
(`modal deploy locaish/meta/modal_app.py`); `CLICKHOUSE_HOST` to keep the
shot table in ClickHouse instead of memory. No room handy? Drop
`examples/IMG_6086.twin` on the studio.

From the terminal:

```bash
locaish ingest sweep.mov --view         # video to twin, with the viewer
locaish inspect twins/room.twin         # dimensions and the QA report
locaish sweep twins/room.twin           # every camera setup, scored
locaish coverage twins/room.twin --brief scene.txt   # a shot list, placed
```

## How much to trust it

On 25 rooms built from a seed nothing was tuned on (random size, unit,
tilt, noise, occlusion): no twin was wrong while reporting `pass`, and the
trusted twins' worst plan error was 1 mm. A twin from video is metric to a
few percent; the QA report says how far to trust the metres, and a tape
measure plus `--scale-factor` tightens it. Full study, method and every
known limitation: [docs/REFERENCE.md](docs/REFERENCE.md). `pytest tests`
runs 637 tests.

## Repository

```
locaish/video/       reconstruction: frames, COLMAP/GLOMAP, dense stereo, MoGe-2, movers masked
locaish/scan/        the twin: canonical frame, structure, QA
locaish/film/        the sweep, the scout report, the rehearsal, the simulator, daylight
locaish/meta/        Muse Spark, Muse Wonder, Muse Image, SAM 3.1, SAM 3D
locaish/agent/       the planner, the rehearsal agent, the camera operator
locaish/product/     labels, cast, set pieces, the stage, the world, packages, projects
locaish/viewer/      the WebGL viewer and the studio's scene page
docs/demo/           the demo film, rendered from a job's own data
```

Docs: [reconstruction](docs/RECONSTRUCTION.md) · [cinematography rules](docs/CINEMATOGRAPHY.md)
· [how to film a room](CAPTURE.md) · [SAM 3D on Modal](docs/SAM3D_MODAL.md)
· [hackathon plan and status](docs/META_HACKATHON.md) · [business plan](BUSINESS_PLAN.md)

The reconstruction was built for [Agentic Cinema](https://agentic-cinema.devpost.com/)
(won its track); the scene layer for the Meta Global AI Developer Hackathon,
October to November 2026. MIT.
