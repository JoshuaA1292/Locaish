# Locaish demo, Meta Global AI Developer Hackathon (v5, about 2:50)

v5 scraps the screen-recorded studio cut. Every beat is rendered from the
job's own data with `docs/demo/v5/*.py` (shared look in `hud.py`: black
ground, ice-blue data, amber for anything a model painted, the cast's red
and blue), cut to a music bed built from GarageBand's royalty-free Epoch
loops, with Samantha narrating short lines. Scan IMG_6208, job 511430a1306a,
rehearsal 330f8554 ("They Know": everyone on their feet, re-rehearsed after a doubtful window opening was dropped from the twin), sim 978a99f5,
world 5b86571e, set piece 2beb74c2.

| # | Clip | Renderer | What it shows |
|---|---|---|---|
| 1 | hook (10 s) | titles.py | Three typed lines on black: a crew picks a location from photographs; someone flies out with a tape measure; this room was scanned with a phone in 50 seconds. The music hit lands at 10.0 s. |
| 2 | title (4 s) | titles.py | LOCAISH · the location, before anyone drives there. |
| 3 | cloud (27 s) | cloud.py | The hero shot: the measured room assembles itself in the order the phone's frames saw it, around the real COLMAP path and frustum; the phone frame inset; frames, points, scale readouts; then Muse Spark's names drop onto the geometry. |
| 4 | sweep (16 s) | sweep.py | A tactical plan of the room; 9,666 setups light up (6 primes, 3 heights); "where can a close-up stand" answered with fans; the Doorway dolly fits on 2,564 of 8,409 clear setups. |
| 5 | scene (22 s) | scene_hud.py | The scene types in; Muse Spark's 8-beat breakdown; the shot table with the room's verdicts, each with its measured frame; Shot 4's dolly fails (needs 68 cm, has 30); the shoot day against the sun. |
| 6 | bodies (12 s) | bodies.py | The simulation: red Maya, blue Theo at their heights on their marks inside the crisp cloud, every shot's camera drawn, the live measured frame inset, the lines as they land. |
| 7 | operator (14 s) | operator_hud.py | Muse Wonder's tool calls and Muse Spark's DP verdicts streamed from the real run (4 minutes in 14 s), with the frames it looked at. |
| 8 | film (15 s) | film.py | Shot by shot, side by side: the measured bodies on the left, the scan's own photograph with the stand-ins placed by the geometry on the right; the photograph wipes into the people and fills the frame; slates and lines; "A RENDER" on every finished frame. Rehearsal c03fdb72, the one the time-lapse recorded. |
| 9 | lift (12.5 s) | lift_hud.py | The bunk bed's own points (the set piece's SAM 3.1 mask) glow in the cloud, rise out of the scan, move 0.5 m north, settle; the shot table's answer (+176 clear setups). |
| 10 | world (10 s) | world_hud.py | Sixty measured views wipe into Muse Image's finished photographs for 14 November 16:40; one fills the frame. |
| 11 | run (14 s) | timelapse.py | One real run of the studio page, polled every 2 s and played at 20×: the scene in, Muse Spark's breakdown, the shots, the frames finished and reviewed. |
| 12 | end (7 s) | titles.py | LOCAISH · nothing a model says becomes a measurement · which model does what · the repo. |

Narration (docs/demo/build/v5/voice/lines.txt), one line per beat from 3 to 11.
Timeline: docs/demo/timeline5.json; music: build/v5/bed.wav (mix.py in the
session scratch, Epoch family at 122 BPM, −18 LUFS, ducked under the voice).

## Build

```
.venv/bin/python docs/demo/v5/titles.py docs/demo/build/v5/titles
.venv/bin/python docs/demo/v5/cloud.py 511430a1306a docs/demo/build/v5/cloud
.venv/bin/python docs/demo/v5/sweep.py
.venv/bin/python docs/demo/v5/scene_hud.py 511430a1306a 330f8554 docs/demo/build/v5/scene
.venv/bin/python docs/demo/v5/bodies.py 511430a1306a 330f8554 docs/demo/build/v5/bodies_r
.venv/bin/python docs/demo/v5/operator_hud.py 511430a1306a 978a99f5 docs/demo/build/v5/operator_events.json docs/demo/build/v5/operator
.venv/bin/python docs/demo/v5/film.py 511430a1306a 330f8554 docs/demo/build/v5/film
.venv/bin/python docs/demo/v5/move_hud.py docs/demo/build/v5/move
.venv/bin/python docs/demo/v5/world_hud.py 511430a1306a 5b86571e docs/demo/build/v5/world
.venv/bin/python docs/demo/assemble.py docs/demo/timeline5.json docs/demo/build/v5/locaish_demo_v5.mp4
```

## What changed in the product for this cut

- A standing person is told so in words the painter cannot misread ("on
  both feet at full height; a waist-up framing is a standing person"), and
  the composition sentence no longer says a subject "sits" in a third of
  the frame. Muse Spark's review now fails a crouch, a kneel or a perch on
  anyone staged on their feet.
- The frame's layout is said from the twin itself: the labels and window
  openings projected through the frame's camera, left to right with
  distances, which windows are in frame (or that none is, and where they
  are instead), and what is not in frame. A doubtful opening (confidence
  under 0.8, no window label near it) is not announced as a window.
- The photographs' light is read once per job by Muse Spark (night or day,
  what lights the room, what the glass shows) and the prompt says so, so a
  night scan finished for 15:30 paints daylit windows and no ceiling tube.
- The two room photographs must look the way the planned camera looks
  (within 60 degrees); before, a photograph of the opposite wall was common
  and the painter flipped the room.
- An actor on the marks but with no stand-in in the render is out of frame
  and the painter is told not to paint them; the people count comes from
  the stand-ins.
- Frames are finished setup by setup, each later frame of a setup carrying
  its setup's first finished frame as the background it must keep.
- Muse Spark's review sees the photographs and the scene's anchor and
  fails invented architecture, a mirrored room, a thing the layout lists as
  absent, black windows in daylight, a ceiling light burning by day; every
  attempt is judged and the best is kept; the verdicts are written beside
  each frame as `finish_<name>.json`.
- The renderer's caption strip is really cut off before the painter sees
  it, and the answer is centre-cropped to the render's aspect.
- The same contract for the world's sixty views, with a review and one
  retry per view.
- The photograph as the base (`stage._photo_base_fn`, `finish.build_photo_prompt`):
  the scan's own frame nearest the planned camera (within 1.6 m and 60
  degrees), cropped to the planned lens around the planned aim, with the
  cast drawn in as flat stand-ins where the measured geometry puts them
  (the photograph's pose is known, so they land where the people stand).
  Muse Image then changes the people and nothing else; the reviewer holds
  the frame to the photograph. The base is saved beside the frame as
  `finish_<name>_base.png`. With no close photograph the render is the base
  as before.
- The reviewer counts only the cast: a world with no cast has nobody in it,
  whatever marks the meta carries (the first world re-run painted two
  strangers into every view because of this).
