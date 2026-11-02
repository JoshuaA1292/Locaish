# Meta Global AI Developer Hackathon: the Rehearsal

*Written 24 October 2026. The hackathon runs 26 October to 4 November (submissions
9:00 PM ET on 4 November, judging to 8 November, finals 9 November).
**Applications close 25 October, 11:59 PM ET, or earlier if 20,000 spots fill.**
Sources: [dev.meta.ai event page](https://dev.meta.ai/events/global-hackathon),
[Cerebral Valley page](https://cerebralvalley.ai/e/meta-ai-developer-hackathon),
[Meta Model API model list](https://dev.meta.ai/docs/models). Official rules were not
published at time of writing; team size and submission format are unconfirmed.
Unofficial guides expect a 2 to 3 minute demo video, a repo and a write-up.*

---

## The one-line idea

**Shoot the scene before you book the room.** A 45-second phone video of a real
room becomes a measured set you can rehearse in: the furniture is solid and
movable, the cast stands on their marks at their real heights, the sun comes
through the real windows at the real shoot hour, the camera rides a dolly that
physically fits, and the shoot-day schedule falls out of the simulation.
Previsualization has always happened in sets someone built by hand in Unreal or
Previs Pro. This is previs inside a reconstruction of a place that exists, with
every centimetre traceable, built from a phone.

Working name: **Rehearsal** (a mode of Locaish). The twin stops being a thing you
look at and becomes a thing you *run*.

## Why this and not something else

- **It is the unthinkable version of what we already have.** The coverage planner
  already places every shot against measured geometry and renders frames from a
  gaussian field. Rehearsal adds the three things a DP says are missing from any
  scan: bodies, time, and the freedom to move the furniture.
- **It uses five Meta models for five jobs none of them could do alone**, and every
  one of them is verified against geometry before it is believed, which is the
  house rule and the thing judges on the startup track are told to look for
  ("technical execution, holds up in production").
- **Nobody has it.** Polycam, Scaniverse, Luma, Matterport: none ships blocking,
  bodies, or a sun-driven schedule. Previs tools: none starts from a phone scan of
  a real room. The combination is the moat, and it compounds the Google win rather
  than restarting.
- **Theme fit.** "Build for Your World" (we know the domain; we won the last
  hackathon in it and have a priced product). The scan-through-a-crowd and
  stature-anchor pieces lean toward "Push the Frontier".

## Which Meta model does which job

| Job | Model | Where it runs | What checks it |
|---|---|---|---|
| Scan a busy room: mask people, dogs, cars out of the walkthrough so the camera solve ignores movers | **SAM 3.1** (`sam-3.1`, text-prompted video masks, streaming) | Meta Model API | Masks feed COLMAP's feature-extraction masks; a frame whose masked area exceeds a threshold is dropped; registration count and bundle residual are reported as now |
| A fourth, independent scale anchor from the people in the scan | **SAM 3D Body** (single image to Momentum Human Rig mesh) | Modal GPU (open weights, SAM License) | Adult stature prior (mean ~1.70 m, sd ~0.09 m) enters the existing inverse-variance fusion in log space; wide bar by design, and disagreement widens the bar, never narrows it |
| Lift each labelled object (sofa, table, fridge) into a solid, textured set piece at its measured position | **SAM 3D Objects** (image plus mask to a textured **gaussian splat** with pose and layout) | Modal GPU | Position from the ray-marched hit the labels pipeline already computes; metric size from the box extent at that depth; the object's splat is cropped against the twin's own surface points before it is accepted |
| Cast from a headshot: the actor's body on a mark at their real height | **SAM 3D Body** | Modal GPU | Height typed by the user (or stature prior); eye height drives eyeline and headroom predicates in the sweep instead of the fixed 1.7 m |
| Name what is in the room; cross-check frames; read the script; score the rendered frame as a DP | **Muse Spark 1.3** (`muse-spark-1.3`: text, image, video, audio, PDF in, 1M context, tools) | Meta Model API | Replaces Gemini in `product/labels.py`, `scan/semantic.py`, and the ADK review loop. The model plans and judges; the geometry answers, as before |
| The long-horizon agent: scene to beats to blocking to timeline to shoot day; remembers the production across sessions | **Muse Wonder** (`muse-wonder-1.0`, agentic, Active Memory) | Meta Model API | Every tool it calls is a Locaish measurement or a ClickHouse query; it cannot invent a number |
| Finish the animatic frame photoreal from the gaussian render plus the actor's headshot and a wardrobe or mood reference | **Muse Image** (`muse-image-1.0`, instruction edit, up to 10 reference images, $0.01 a frame) | Meta Model API | Labelled "render, not measurement" in the UI and the package. The measured frame (lens, distance, headroom) stays beside it |
| Optional: replace or cross-check the COLMAP solve on a GPU host with a Meta metric multi-view model | **MapAnything** (`facebook/map-anything-apache`, Apache 2.0) | Modal GPU | Only as a fourth scale reading or a GPU fast path; COLMAP remains the reference solve |

API compatibility: base URL `https://api.meta.ai/v1`, OpenAI Chat Completions and
Responses protocols, so the ADK agent swaps to a LiteLLM model wrapper or a plain
OpenAI client. Known limit: `tool_choice` accepts only `auto`. Pricing seen:
Muse Spark 1.3 $1.25 / $4.25 per M tokens; Muse Wonder $4 / $20; Muse Image
$0.01 per image; SAM 3.1 about $0.20 per 1,000 video frames (secondary source).
Selected participants get $150 in Model API credits; Modal and Baseten credits
($100 each) go to "select participants" only, so budget a few dollars of our own
GPU time.

## What the demo shows (the 2 to 3 minute video)

1. **Scan through the crowd.** A café with people walking through it, filmed on a
   phone. SAM 3.1 masks the movers; the twin registers cleanly; the QA report shows
   the stature anchor agreeing with the door and camera-height anchors.
2. **The room becomes a set.** Click the sofa. It lifts into a solid set piece. Drag
   it to the wall. The standable floor recomputes; the ClickHouse sweep re-runs;
   the shot table says a 50 mm wide that could not exist now holds.
3. **Cast it.** Drop two headshots and type two heights. Two bodies stand on marks.
   Headroom, eyeline and over-the-shoulder geometry are measured against them.
4. **Rehearse it.** Paste the scene. Muse Wonder breaks it into beats, blocks each
   walk on the standable floor around the real furniture at walking pace, times the
   lines, puts the camera on a dolly from the catalogue that fits the track, and
   plays it back: an animatic of the real room with the sun where it will be at
   10:40 on the shoot date.
5. **Finish the frame.** One click on any animatic frame: Muse Image turns the grey
   bodies and the gaussian render into a photoreal still of the actor in the room,
   labelled as a render. Muse Spark scores it as a DP and sends the planner back to
   the table if the window is behind her.
6. **The shoot day falls out.** The schedule page: "Scene 4 window coverage holds
   09:40 to 11:10; three setups at 25 minutes; strike the sofa before setup 2."
   Exported as the location package, shareable by link, on a phone.

Close on the sentence: *every number on screen traces to a measurement, a solar
ephemeris, or a catalogue; the models planned and judged, the geometry answered.*

## Status, 24 October (before the window opens)

Built on branch `meta-hackathon`, 564 tests green (29 Oct); the Meta layer runs live, nothing run against a live
Meta key or a deployed Modal endpoint yet:

| Piece | Where | State |
|---|---|---|
| Meta Model API client, fake for tests | `locaish/meta/client.py` | done; wire format unverified live |
| Muse Spark/Wonder as ADK models; provider switch | `locaish/meta/adk.py`, `locaish/agent/core.py`, `agent/coverage.py` | done; `tool_choice=auto` only |
| Muse Spark vision: labels, cross-check, DP review | `locaish/meta/vision.py`, `product/labels.py`, `scan/semantic.py` | done |
| SAM 3.1 mover masks into COLMAP; frame drops; cache keys | `locaish/meta/masks.py`, `video/colmap.py`, `video/reconstruct.py` | done; mask wire shape unverified |
| Stature scale anchor from the people seen | `locaish/meta/stature.py`, `video/metric.py`, `video/reconstruct.py` | done, fused with sanity window against consensus |
| Cast with real heights, capsule or SAM 3D Body figure | `locaish/product/cast.py` | done |
| Set pieces lifted from the scan, moved, re-swept, door fit | `locaish/product/setpieces.py` | done; 1 to 7 % of truth on fixtures, 1.81 x 0.89 x 0.76 m sofa on a real scan |
| SAM 3D Objects and Body on Modal; fit to footprint; body npz | `locaish/meta/sam3d.py`, `meta/modal_app.py`, `docs/SAM3D_MODAL.md` | written, not deployed (needs Modal account, HF gate approval) |
| Rehearsal engine: blocking, timing, camera, dolly check, sun, schedule, animatic, blocking SVG, call sheet | `locaish/film/rehearsal.py` | done, deterministic |
| Muse Spark breakdown, placement from the shot table, notes | `locaish/agent/rehearsal.py` | done against a scripted model |
| Muse Image frame finishing, stamped | `locaish/meta/finish.py` | done; endpoint path unverified |
| Camera simulator: timeline, eight rigs, solve, per-frame checks, segments, from-rehearsal | `locaish/film/sim.py` | done, 30 tests |
| Simulate mode in the viewer: timeline bar, keys, drag, frustum, rails, figures, camera-view playback | `locaish/viewer/template.html` | done, screenshot-verified (SwiftShader) |
| Simulator routes: evaluate on edit, figures, frame at playhead, video render (ffmpeg), hand-off files | `locaish/product/stage.py`, `serve.py`, `viewer/stage.js` | done, 9 route tests |
| The operator: Muse Wonder on the engine's tools, Muse Spark reading frames, directing route with events | `locaish/agent/sim.py` | done against a scripted model, 4 tests |
| Articulated bodies: parametric mesh, standing/walking/sitting, z-buffered against the measured room; the same in the viewer | `locaish/product/figures.py`, `film/render.py`, `viewer/template.html` | done, 29 Oct |
| Actors as obstacles in placement, the rehearsal and the simulator | `agent/rehearsal.py`, `film/rehearsal.py`, `film/sim.py` | done, 29 Oct |
| Photoreal finishing of whole sequences with the scan's own photographs of the view, cast looks and continuity; finished video | `locaish/meta/finish.py`, `product/stage.py`, `viewer/stage.js` | done, verified live 29 Oct: the real room, consistent people |
| Operator watches its move (filmstrip) and a rendered move is reviewed as video | `locaish/agent/sim.py` | done, verified live |
| Studio stage: routes, page, animatic scrubber, finish button | `locaish/product/stage.py`, `locaish/viewer/stage.js`, `serve.py` | done, screenshot-verified on a real room |
| Directed scenes: beats carry pose, business, eyeline and composition; occluder-aware placement; light words for the hour; the scan's photographs in every finish | `agent/rehearsal.py`, `film/rehearsal.py`, `meta/finish.py`, `product/figures.py` | done 1 Nov, 620 tests |
| The world: sixty views of the measured room chosen for coverage, finished by Muse Image for a date and hour (sun through the real window) and a dressing, then **baked onto the scan's own gaussians** | `locaish/product/world.py`, `video/splat.py` | done 1 Nov; bake 7 s, 3 min with relighting; see below |
| Walk mode and WebXR in the viewer; photographs pinned where they were taken fade in underfoot; `?photo=view_0012` deep link; N steps to the next; the studio strip stands you on a click | `locaish/viewer/template.html`, `viewer/stage.js`, `product/stage.py` | done 1 Nov, screenshot-verified (SwiftShader) |
| The studio in black and white: tokens, no gradients, the scene first on the page | `locaish/product/pages.py`, `serve.py`, `viewer/stage.js` | done 1 Nov |
| The studio set like a printed programme: serif titles, the picture first, one number, captions folded; phone width holds | `serve.py`, `product/pages.py`, `viewer/stage.js`, `viewer/template.html` | done 1 Nov PM, 625 tests |
| SAM 3.1 masks decoded through Meta's `meta-sam-parser` (base85 over an arithmetic coder); boxes stand in without it | `locaish/meta/client.py` | done 1 Nov, verified on a live sample |
| **SAM 3D Objects and SAM 3D Body deployed on Modal** (L40S / L4); the chair lifted from the dorm scan to a 263k-gaussian textured splat on its measured footprint in 87 s; a body to an 18k-vertex mesh in 84 s | `locaish/meta/modal_app.py`, `meta/sam3d.py`, `product/stage.py` | **live 1 Nov**, eight deploys to get the images to build (see commit b747cb2) |
| Demo video tooling: script, DevTools screencast/poll recorder, ffmpeg assembler with cards and captions, Jamie (Premium) narration | `docs/demo/` | built 1 Nov; the cut is assembled from `docs/demo/timeline.json` |

| Muse Spark reviews every finished frame beside its render (head count, a seat under anyone seated, the framing, the furniture) and a rejected frame is finished again with the correction; a seated person is told which seat | `locaish/meta/finish.py` | done 2 Nov, 3 tests; the dorm's eleven frames re-finished through it |
| The finish stops inventing the room: the frame's layout from the twin's labels and openings projected through its camera; the photographs' light read once by Muse Spark (a night scan finished for the afternoon paints daylit windows); photographs that look the way the camera looks; actors with no stand-in are out of frame; setup-by-setup continuity; the reviewer sees the photographs, the anchor, the light and the layout, judges every attempt and keeps the best; the world's views reviewed too | `locaish/meta/finish.py`, `locaish/product/stage.py` | done 2 Nov after a 31-agent audit of the hallucinations; 7 contract tests; the dorm's scene re-rehearsed (a doubtful window opening dropped from the twin) and finished through it |
| The dorm (IMG_6208) staged end to end: 18 labels, 8-beat scene, 11 finished frames, the operator's move, the bunk bed lifted by SAM 3D (479k gaussians) and moved (0.5 m north gains 176 clear setups, 0.5 m south loses 2,706), the world at 16:40 on 14 Nov | `twins/studio/511430a1306a`, `docs/demo/shoot.py` | done 2 Nov |
| Demo v4: the film walked shot by shot with slates, labels on the photographs, the move, the grid of sixty views; Samantha narration | `docs/demo/timeline4.json`, `docs/demo/film.py`, `docs/demo/script.md` | cut 2 Nov, 3:02 |
| Hosted dorm for judges: the showcase image with the staged room, the Meta key and the SAM 3D endpoint set on the service | `deploy/deploy.sh`, `deploy/ctx` | staged 2 Nov; deploy with `./deploy/deploy.sh <project>` |

### 1 November: the world, and why it is baked rather than trained

The first world trained a fresh splat (Brush, 6000 steps) on the sixty finished
views. It was a smear: the finished views are sixty opinions about one room,
and a trainer learns the disagreements as fog while throwing away the one thing
the scan had, crisp geometry. The world now keeps every gaussian the scan
trained exactly where it is and changes only its colour. For each finished
view, every gaussian on the measured surface (depth within tolerance of the
measured depth at that pixel) is compared with the render the finisher was
shown: where the model and the measurement agree on the structure (local
correlation of the two luminances) the finished pixel is painted, because it
is the room's own detail relit and dressed; where they disagree (a bed the
model moved, a wall it invented) the gaussian keeps the scan's colour and
takes only the light, the finished view over the render at low frequency,
which is where the sun patches and the hour's colour live. Views are weighted
by that agreement and by nearness. The result walks like the scan and is lit
like the hour. People are not baked: a figure painted into sixty views
disagrees from view to view and averages into a ghost, so a world is the empty
location by default and the viewer draws the cast as bodies on their marks.

Between the cameras the baked field is what there is to see; on one of them
the photograph itself is better than anything a field can show, so the viewer
fades it in as the walker reaches the spot and looks the way the camera did,
drawing the field through the same lens so the two agree where they meet. The
world is honest about what it is: a label on every page, a render note in the
provenance, and the scan one click away.

**Next, in order:** get a META_API_KEY and run the vision, breakdown and
finish paths live (fix wire shapes); deploy the Modal app and lift one sofa
and one body; film the crowd clip; record the demo; the ClickHouse path for
the dressed room's sweep (today the dressed sweep is in memory); drag set
pieces in the 3D viewer rather than nudge buttons.

## Build plan (ten days, one person plus Claude)

| Day | Deliverable | Notes |
|---|---|---|
| 26 Oct | Meta API layer: `locaish/agent/meta.py` client; ADK model swapped to Muse Spark 1.3 via LiteLLM; detector in `product/labels.py` and cross-check in `scan/semantic.py` on Muse Spark; tests green with the Gemini path kept behind an env flag | Apply for Modal/Baseten credits in the kickoff; set up Modal app skeleton |
| 27 Oct | SAM 3.1 video masks into the COLMAP front end (`video/frames.py`, `video/colmap.py`): mask PNGs per frame, over-masked frames dropped, QA line for masked fraction | Test on a clip with a person walking through |
| 28 Oct | SAM 3D Body on Modal; stature anchor in `scan/scale.py` as a fourth estimator in the existing fusion; QA reports it | Honest bar; must never narrow the fused bar on its own |
| 29 Oct | SAM 3D Objects on Modal; `product/setpieces.py`: lift a labelled object to a gaussian splat placed and scaled in the twin; viewer shows set pieces as a layer | Output is gaussians, same representation as the Brush layer, so `film/render.py` composites by concatenation |
| 30 Oct | Movable set pieces: drag in the viewer, occupancy and standable maps recomputed from the twin minus the original object points plus the moved piece, re-sweep, ClickHouse table refreshed, "what became possible" diff; door-fit check for load-in | Reuse `film/space.py` maps and `film/sweep.py` |
| 31 Oct | Casting: headshot to body mesh to point samples at a typed height; bodies on marks; sweep predicates read per-actor eye height | `video/splat.py::sample_surface` already turns a mesh into splats |
| 1 Nov | Rehearsal engine `film/rehearsal.py`: beats, A* blocking on the standable grid, walking-pace timing, line timing, dolly track fit from `film/moves.py` and `film/equipment.py`, a timeline of camera and bodies; animatic frames from `film/render.py` | Deterministic; works without any model given a typed beat list |
| 2 Nov | Muse Wonder agent drives the rehearsal with tools; Muse Image finishing with references; Muse Spark DP review loop; schedule page from `film/daylight.py` sun intervals plus setup durations; package and share link carry it | Tool calls are measurements only |
| 3 Nov | Deploy to Cloud Run with Modal endpoints; record demo on a live scan; write-up; README section | Buffer day for whatever broke |
| 4 Nov | Submit by 9:00 PM ET | Do not touch the pipeline on submission day |

Stretch, only if 1 and 2 November land early: drop a whole screenplay and let
Muse Wonder schedule every interior scene across the index of twins; read the
walkthrough's own audio with Muse Spark 1.2 for a noise floor beside the RT60.

## What stays honest

- The twin's bargain does not change. A `pass` verdict still means the numbers are
  right. Set pieces, bodies and finished frames are layers on top, each labelled
  with its source, none of them feeding a measurement except the stature anchor,
  which enters the fusion the same way the other three anchors do.
- The Muse Image still is a render. It sits beside the measured frame, never
  instead of it, and the package says so in words.
- Nothing ships that only works on the sample room. Day 3's demo is filmed on a
  room the pipeline has not seen.

## Risks and the fallback for each

| Risk | Fallback |
|---|---|
| SAM 3D Objects or Body are slow or fiddly on Modal | Pre-compute for the demo rooms; keep the live path for one object and one body; both are open weights so no API dependency |
| Modal credits are not granted | Pay-as-you-go; a few dollars of A10G time covers the demo |
| `tool_choice=auto` only | The ADK loop never forces a tool today; the deterministic planner works without the model |
| SAM 3.1 streaming or frame limits | Run on the subsampled frame set (about 250 frames at 5 fps), not the raw video |
| Hackathon rules demand "indie" means solo and the team is one person | Enter indie if the startup track requires a registered company; the product still enters as "works, ambitious, creative" |
| SAM License commercial terms | Permits research and commercial use with acceptable-use conditions per secondary sources; read the LICENSE in both repos on day 1 and note it in `docs/RECONSTRUCTION.md` |

## Application text (due 25 October, 11:59 PM ET)

**Project idea (clarity, feasibility, theme fit):**

> Locaish turns a phone video of a room into a measured 3D twin and plans a
> film scene's camera coverage inside it; it won its track at Google's Agentic
> Cinema hackathon in September and is now a priced product for commercial
> production companies. For this hackathon we make the twin something you can
> rehearse in before anyone books the room. SAM 3.1 masks people and traffic out
> of the walkthrough so busy locations scan cleanly. SAM 3D Objects lifts each
> piece of furniture into a solid, movable set piece at its measured position, and
> the camera-setup index recomputes when the art department moves it. SAM 3D Body
> puts the cast on their marks at their real heights from a headshot. Muse Wonder
> reads the scene, blocks the actors on the real floor around the real furniture,
> fits a dolly from a catalogue of real equipment, and plays the scene back with
> the sun through the actual windows at the shoot hour, from a solar ephemeris.
> Muse Image finishes any frame photoreal from the actor's reference, labelled as a
> render; Muse Spark reviews it as a DP would. Because the moves and the sun are
> timed, the shoot-day schedule falls out of the simulation. Every number traces
> to a measurement, an ephemeris, or a catalogue; the models plan and judge, the
> geometry answers. Theme: Build for Your World. Track: Startup.

**Technical skill (an AI project already built):**

> Locaish (github link, live demo link). Phone video to metric twin: COLMAP with
> ALIKED features and GLOMAP, OpenMVS dense stereo, MoGe-2 metric depth used only
> where stereo has nothing and only where neighbouring frames agree; scale from
> three independent anchors fused by inverse variance in log space so disagreement
> can only widen the error bar; a self-grading QA report held to a 25-room
> generalization suite the tuning never saw. Every standable camera position at
> three heights on six primes is scored against the geometry into ClickHouse
> (30k to 300k rows a room); an agent plans a scene's coverage, renders each
> frame from a per-scene gaussian splat, and a vision model reviews the frame.
> 291 tests. Deployed on Cloud Run. Won its track at Agentic Cinema 2026.

## Decisions for the user

1. **Track.** Startup fits the product and the judging criteria; confirm it does not
   require a registered company once the official rules land.
2. **GPU host.** Modal is the plan (credits plus serverless GPUs). Say if GCP is
   preferred since the dense stage already has a GCP path.
3. **The crowd scan.** Needs one short clip of a real room with people moving
   through it, filmed before 27 October.
