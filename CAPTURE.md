# Capture protocol

The pipeline can only recover what the scanner recorded. Every accuracy target
in `docs/PHASE1_SPEC.md` assumes a capture that actually saw the floor, the
ceiling, all four walls and the reveal around every opening, at a range the
sensor can honestly resolve, with the loop closed. A twin that measures true and
a twin that drifts four centimetres across a room are usually the same room,
the same phone and the same app — the difference is technique.

This is the recipe. Follow it before the pipeline ever runs.

If you have no LiDAR and no scanning app — you are just going to film the room
on a phone — most of this still applies, but the failure modes are different
enough to be worth their own section: see [§12](#12-filming-instead-of-scanning).

Numbers labelled **(measured)** come from running against
`locaish/fixtures.py`, the synthetic rooms with exact ground truth. They
quantify how the pipeline responds to a capture mistake, not how the sensor
behaves in the field; the fixtures have no sensor bias. Numbers labelled
**(vendor)** or **(third party)** are cited at the end, and anything we could
not verify is marked **unverified** rather than stated confidently.

---

## 1. What the sensor actually is

The LiDAR on an iPhone or iPad Pro is a low-resolution time-of-flight depth
sensor built for autofocus and AR occlusion. It is not a survey instrument, and
treating it as one is how people end up quoting a wall to the millimetre that is
wrong by five centimetres.

| property | value | source |
|---|---|---|
| depth map resolution | 256 x 192 (about 49,000 depth samples per frame) | third party |
| maximum range | about 5 m; Polycam encodes its raw depth maps as 16-bit millimetres with a stated maximum of 5 m | Polycam raw data spec |
| honest working range | 3–4 m; beyond that the return weakens, density drops and depth noise dominates | third party |
| single-shot range precision | roughly 0.4 mm std at 0.25 m rising to about 1.0 mm at 5 m in one lab study; another study put facade RMSE at 4.89 mm against a terrestrial scanner's 3.44 mm | third party |
| room-scale accuracy | dominated by SLAM drift, not by range noise | see below |

The last row is the one that matters. In a 2023 comparison of four iPhone
scanning apps against a reference survey, most apps landed within about 10 cm
RMSE over a building interior, with roughly 70–83% of points inside 5 cm, and
one app was far worse; the authors attribute the failures to drift accumulating
over the sweep and showing up as split surfaces where two passes overlap and
where loops end. Those specific per-app figures are from one 2023 test on an
iPhone 13 Pro and are stale as a ranking — do not quote them as current app
quality. The lesson that survives is structural: **your technique and the app's
pose graph decide the error, not the chip.**

Which devices have the sensor (compiled from third-party device lists — confirm
your exact model before you drive out):

| has LiDAR | does not |
|---|---|
| iPhone 12/13/14/15/16/17 **Pro** and **Pro Max** | every non-Pro iPhone: base, Plus, mini, e, Air |
| iPad Pro 11-inch 2nd gen (2020) and later; iPad Pro 12.9-inch 4th gen (2020) and later; iPad Pro 13-inch (M4, M5) | iPad, iPad Air, iPad mini |

Without LiDAR you can still capture — photogrammetry and gaussian splatting run
on any recent phone — but see the scale warning in section 2. Assume a
non-LiDAR capture will need the scale ritual in section 7 to be trusted at all.

---

## 2. Choosing the app, the mode and the export

Our readers accept **PLY, OBJ, GLB/GLTF, STL**. Nothing else. USDZ, FBX and LAS
are not ingest paths, however good the scan is, so the export menu decides
whether the capture was worth taking.

| | Scaniverse | Polycam | Apple RoomPlan |
|---|---|---|---|
| capture we want | Mesh (LiDAR) mode | LiDAR mode (or Room mode) | the only mode there is |
| also offers | splat mode; photogrammetry beyond LiDAR range | photo mode (photogrammetry), splat, Room mode, 360 | — |
| export we want | **GLB** | **GLTF** | none we can read |
| available free | yes — free and unlimited per the support page (see caveat) | yes — the free plan exports `.gltf` and nothing else (vendor pricing page) | yes, it is an OS framework |
| metric scale | yes from LiDAR mode | yes from LiDAR/Room mode | yes |
| colour | textured mesh | textured mesh | untextured parametric boxes |
| camera poses | none documented (unverified) | yes, via developer mode raw data | in-session only, not in the export |

Exact menu items:

- **Scaniverse.** Capture in a mesh/LiDAR mode, not splat mode. Then
  Share → **Export Model** → under *Export Model As* choose **GLB** →
  **Save to Files**. The App Store listing for 5.2.6 names SPZ, PLY, GLB and
  FBX; several secondary sources list a wider mesh menu (OBJ, FBX, GLB, USDZ,
  STL, PLY, LAS) and the official support page lists a narrower one (OBJ, FBX,
  USDZ, LAS). The menu clearly varies by version — **check your build's menu on
  a throwaway scan before you drive to a location.** If your build offers only
  USDZ, FBX or LAS for meshes, it cannot feed this pipeline; use Polycam.
- **Polycam.** Capture in **LiDAR** mode. Export → **glTF (.gltf)**. That is the
  one format the free plan gives you, and it is one we read. OBJ, STL and the
  point-cloud formats (PLY, LAS, PTS, XYZ) are behind a paid plan.
- **Polycam camera poses.** Settings → scroll to **Developer mode** → on. This
  adds a **raw data** export for LiDAR and Room captures. It is **not
  retroactive** — turn it on before you scan. The archive carries
  `keyframes/cameras/*.json` with per-frame intrinsics and a row-major extrinsic
  matrix in ARKit's gravity-aligned convention (+Y up, −Z forward),
  `keyframes/depth/` as 16-bit millimetre PNGs capped at 5 m, and per-pixel
  confidence maps (0 low / 127 medium / 255 high). Polycam globally optimises
  the ARKit poses before meshing and says the optimised poses are as good as or
  better than SfM. Keep this archive next to the mesh — but note the current
  limitation: `ScanImport` has a `camera_positions` field, and `CaptureBounds`
  uses it when it is populated, yet **no reader ingests a Polycam raw-data
  archive today**. Poses are read only from glTF camera nodes and a
  Bundler-style PLY camera element, neither of which Polycam or Scaniverse
  writes. So keeping the archive costs nothing and buys nothing yet; it is
  worth doing only because poses turn `CaptureBounds` from an inference into a
  record of where you actually stood, and that reader is a small piece of work
  we have not done.
- **Apple RoomPlan** (any app built on it; Polycam's Room mode is similar,
  though whether it uses RoomPlan is **unverified**). Gives you labelled walls,
  doors, windows, openings and furniture as a parametric model with dimensions
  and confidences, exported as USD/USDA/USDZ. What it gives up is real geometry:
  walls become idealised planes, furniture becomes cuboids, a bay window becomes
  a flat rectangle, and door swing direction is lost. It is robust where mesh
  scanning is weak (featureless rooms, fast turnaround) and it is **not an
  ingest path for us** — we cannot read USDZ. Use it as a **cross-check**: run a
  RoomPlan scan alongside the mesh scan and compare its wall lengths, ceiling
  height and window count against `locaish inspect`. Two independent estimates
  that agree are worth far more than one that is confident. Apple's own limits:
  LiDAR device required, rooms up to about 9 x 9 m (30 x 30 ft), at least 50 lux,
  and no scan longer than five minutes.

Three traps worth naming:

1. **A gaussian-splat PLY is not a point cloud.** Splat PLYs store gaussian
   means plus opacity, scale, rotation and spherical-harmonic colour
   coefficients. A reader that takes the vertex positions gets a fuzzy,
   non-surface sampling with colours that are not colours. Export a mesh, or
   accept that the splat is for looking at, not measuring.
2. **A photogrammetry-only capture is scale-free unless the app injects
   scale.** Structure-from-motion recovers geometry up to an unknown similarity
   transform; phones fix it from VIO or LiDAR. If you capture in photo mode on a
   device without LiDAR, treat the metric scale as unproven until section 7's
   ritual passes.
3. **Scaniverse's processing location has changed between versions** — older
   documentation says all processing is on-device and needs no connection, the
   current App Store listing describes cloud processing for splats. If offline
   or confidential capture matters for a location, verify on your build first.
   **Unverified** which versions do which.

---

## 3. Pre-flight

**Light the room for the camera, not for the sensor.** The LiDAR works in the
dark. The RGB texture and, more importantly, the visual-inertial tracking that
stitches the sweep together do not. Apple asks for at least 50 lux — roughly a
living room in the evening — and a room lit below that will drift.

- Turn on every practical light. Open blinds unless the twin is meant to record
  them closed. Whatever you choose, write it down: Phase 3 shines the real sun
  through these apertures and needs to know what state the room was in.
- Kill everything that moves: ceiling fans, air conditioning that stirs
  curtains, TVs and monitors, screensavers, pets, other people. A moving surface
  becomes a smear that the plane fitter has to reject.
- Move your own kit out of the room: bags, cases, tripods. Leave the furniture
  a production would actually find there. This is a scout document, not a
  showroom photograph.
- Decide every door and window: open or closed, and stay consistent. A door left
  ajar reads as neither a wall nor a clean opening.
- Cover mirrors if you are allowed to (a sheet, paper, gaffer tape). Note any
  you could not cover.
- Put the scale reference on the floor near the middle of the room, flat, in
  the open (section 7).
- Phone: lenses clean, Do Not Disturb on, Low Power Mode off, battery above 50%,
  device cool to the touch, several GB free. A long LiDAR capture with raw data
  enabled is large — the exact size is **unverified**, budget generously.
- Take the compass reading now, before you start walking (section 8).

---

## 4. The walk

**Speed.** Walk at roughly half normal walking pace — about 0.3 m/s — and pan
slower than you walk. Our rule of thumb, not a vendor figure: never rotate the
phone faster than you can read a wall socket as you pass it. Scaniverse's own
guidance is "move your device steadily, avoid sudden movements to prevent blur
or position tracking loss." Blur costs you tracking, and lost tracking costs you
the whole sweep.

**Distance.** Hold the phone **1.0–2.5 m** from whatever surface you are
currently claiming, and never rely on returns past 4 m. In the synthetic 9.5 x
7.2 x 4.6 m room, a single perimeter loop leaves **35.1%** of the wall and
ceiling points more than 3 m from the nearest camera position, with a worst case
of **4.02 m** and a median of 2.43 m **(measured)** — past the honest range, in a
room only just inside Apple's 9 x 9 m guidance. In the small 5.2 x 4.1 m room the
same loop keeps everything within **2.28 m**, with nothing beyond 3 m
**(measured)**. Big room means extra interior passes, not a wider loop.

**Sweep pattern.** Four passes, in this order:

1. **Eye-height perimeter loop.** Walk the room with a wall on your shoulder,
   phone held level, capturing walls and the top of the furniture. This is the
   pass the wall planes and yaw normalisation come from, so give every wall
   comparable time — a wall you rushed becomes the wall that skews +X.
2. **Floor pass.** Same loop, phone tilted down about 45 degrees, catching the
   floor, skirting boards and the base of the furniture. The floor plane sets
   `z = 0` and gravity alignment for the entire twin.
3. **Ceiling pass.** Same loop, phone tilted up about 45 degrees. See section 5.
4. **Interior fill.** Cross the middle of the room in a serpentine, and walk
   around anything free-standing. This matters most when your export carries
   camera poses: with poses, `CaptureBounds` is the hull of where you actually
   walked, and a perimeter loop alone encloses **63.9%** of the floor in the
   21 m² fixture and **69.3%** in the 68 m² one **(measured)**. The shot search
   refuses to place a tripod outside that region, so a perimeter-only scan
   produces a twin that will not let you stand in the middle of the room.
   Without poses the bound is inferred from reconstructed floor instead and
   comes out near the full room (98.2% and 99.5% on the same two fixtures), which
   is more generous but is an inference about what you saw rather than a record
   of where you stood.

Keep previously captured geometry in frame as you move; overlap is what lets the
tracker connect one view to the next. Never walk sideways while pointing at a
blank wall — see section 6.

**Doorways.** Decide first whether the next room is in scope.

- Not in scope: close the door and scan it as a surface, or open it flat against
  the wall and sweep the reveal (the 5–15 cm depth of the jamb) from inside the
  room, from both sides of the doorway, at 1–1.5 m. **Do not walk through.**
  Crossing a threshold starts a corridor's worth of drift and drags the geometry
  of a room you are not claiming into the footprint, which inflates floor area
  and confuses wall fitting.
- In scope: scan it as a **separate capture**, with its own loop and its own
  closure. Two clean twins beat one drifted one.

Opening detection works on holes in wall planes plus the reveal edge around
them, so a door leaf that is neither flat open nor properly closed is the worst
case: not a wall, not a hole, no clean edge.

**Close the loop.** Finish where you started, facing what you faced at the
start, and hold there for five to ten seconds so the tracker re-observes the
features it began with. That is what lets the app's pose graph recognise it has
returned and distribute the accumulated error around the loop instead of leaving
it piled up at the seam.

What happens if you do not: in the `drifty` fixture, a 35 mm slow warp across
the sweep — modest by SLAM standards — bows the walls that should be planes from
**4.9 mm to 16.4 mm RMS**, a factor of 3.3 measured over the same four walls
**(measured)**. Nothing about that scan looks wrong on the phone. It looks wrong
three weeks later when the dolly track does not fit. `locaish ingest` reports it
as the `drift` check, which is the one number in the report that a re-scan can
fix and post-processing cannot.

---

## 5. Height discipline

Cover three bands on every wall: **floor (0 to 0.5 m), eye (0.5 to 1.8 m),
ceiling (1.8 m to the ceiling)**. In a well-covered synthetic capture those
bands hold **39.3%, 27.3% and 33.4%** of the points respectively **(measured)** —
about a third of the whole cloud lives above head height, and the single
densest band is the one under your knees.

A scan taken entirely at chest height gives the pipeline no ceiling and no clean
floor, and something has to fill the gap. The `noceiling` fixture — a capture
that stopped 12% short of the top — has its highest returns at 2.376 m in a room
whose true ceiling is 2.700 m. Anything that reads "the top of the cloud is the
ceiling" gets **345 mm low** at the 99.5th percentile, against **+5 mm** on a
properly covered scan **(measured)**. That is an invented ceiling: a number that
looks like a measurement, is off by a third of a metre, and silently wrecks the
daylight study, the boom-mic headroom and every lighting-rig clearance.

Locaish's defence is to return `ceiling_z = None` rather than guess, and
`Structure.ceiling_height` then reports nothing at all. That is the correct
behaviour and it is still a failed capture. Look up.

Same argument at the bottom. The floor plane defines `z = 0` and gravity for the
entire twin, and a floor seen only at grazing angles from standing height fits
worse than one seen from above at 45 degrees. Give the floor a full loop of its
own.

---

## 6. The known enemies

| enemy | what the sensor does | what you get | what to do |
|---|---|---|---|
| **Windows and glass** | near-infrared passes through, or reflects away from the receiver | no return: a clean hole in the wall plane | Nothing. This is why opening detection works — the hole *is* the window. Sweep the reveal and jamb from 1–1.5 m so there is a hard edge to lock onto. Expect junk points from whatever is outside; do not linger pointing out of the window. |
| **Mirrors** | returns come back from the reflected room | a phantom duplicate room behind the wall, and tracking that can jump | Cover it. If you cannot, scan that wall last, briefly, from an oblique angle, and record the mirror's position and size by hand. |
| **Glass railings and balustrades** | invisible, exactly like a window | an open floor edge where there is a barrier — a safety-relevant lie | Tape or drape an opaque strip along a section so at least the line is recorded, and note it in the location file. Never let a twin imply a mezzanine edge is walkable. |
| **Dark and matte black surfaces** | absorb the IR pulse | sparse, noisy or entirely missing geometry | Close to about 1 m, slow, multiple passes from different angles. |
| **Glossy and polished surfaces** (marble floor, gloss paint, stainless, TV screens) | specular return leaves at the mirror angle | false depth, waviness, dropouts | Approach obliquely and vary the angle; never scan a gloss surface straight on. |
| **Large featureless walls** | depth is fine, the *visual* tracker has nothing to lock onto | drift along the wall, which becomes a bowed or split wall | Always keep a corner, doorframe, socket, switch or piece of furniture in frame. Never track sideways along a blank wall. |
| **Thin objects** (cables, chair legs, stands, curtain rods) | thinner than the depth footprint at range | missing, or bloated into a blob | Get inside 1 m if you need them. Do not trust the twin for anything thinner than about 3 cm. |
| **Direct sunlight** | solar IR swamps the return; RGB clips | dropouts and range noise in the sunlit patch, blown texture | Scan that area when the patch has moved, or diffuse it with sheers, and record the time of capture so the daylight study can tell a real bright patch from a sensor artefact. |
| **Anything that moves** | sampled in two places | smeared ghosts that survive into the mesh | Turn it off, close it, or remove it before you start. |
| **Rooms larger than about 9 x 9 m** | fine locally, drifts globally | walls that are individually flat and collectively wrong | Split into overlapping captures with their own closed loops, or accept a `warn` verdict and say so out loud. |

---

## 7. The scale verification ritual

Never ship a twin whose scale you have not checked against something you
measured with a tape.

**Before the scan.** Put a rigid reference of known length on the floor, near
the middle, flat, in the open, matte, at least 1.0 m long, and **not thin** — a
folding rule opened flat, a taped-down tape measure, an A1 board, a marked
plank. At the fixtures' surface density of about 950 points per m²
**(measured)**, a 1.0 x 0.1 m rule collects roughly 95 points and a 0.6 x 0.6 m
board roughly 340, which is enough to place each end within a couple of
centimetres; a 20 mm round rod collects almost nothing and is exactly the thin
object section 6 warns about. Then tape-measure and write down two more real
lengths: **a door leaf width** and **the floor-to-ceiling height in one corner**.

**After ingest.** The CLI is being written in parallel; the commands below match
the flags in `locaish/cli.py` today, but treat the exact spelling as
**provisional** and check `locaish ingest --help`.

```
locaish ingest scans/studio.glb --name studio \
    --lat 51.5074 --lon -0.1278 --heading 287.4
locaish inspect twins/studio.twin --metrics
locaish measure twins/studio.twin --from -1.20,0.35,0.01 --to -0.20,0.35,0.01
locaish view twins/studio.twin
```

The intent, flag by flag: `ingest` turns the export into a `.twin` and prints
the QA report; `--heading` is section 8 and without it every solar number
downstream is an assumption; `inspect --metrics` prints the full metrics table
behind the verdict; `measure --from/--to` gives the distance between two points
in twin space with an uncertainty attached; `view` writes a self-contained HTML
viewer you use to find the endpoints of your reference object and read off their
coordinates.

**Acceptance.** Measure the reference, the door leaf and the ceiling height in
the twin, and compare against the tape:

| check | accept |
|---|---|
| 1 m reference | within ±15 mm |
| door leaf width | within ±30 mm |
| corner ceiling height | within ±25 mm |

**Read the pattern, not just the pass/fail.** These two failure modes need
different fixes and look identical if you only measure one thing:

- **Every length is off by the same percentage** → unit or scale error. The
  export was in centimetres, inches, or scale-free photogrammetry. Re-ingest
  with `--unit m|cm|mm|in|ft`, or re-export from a LiDAR mesh mode. A 1% scale
  error is 52 mm on a 5.2 m wall **(measured)** and it multiplies, so it does
  not average out over a big room, it grows.
- **Short lengths are right and long ones are wrong, or the error depends on
  which direction you measure** → drift. No flag fixes this. Rescan with a
  closed loop, and if the room is large, split it.

---

## 8. Capturing the heading

`Georeference.heading_deg` is the true-north bearing of the twin's **+X** axis,
clockwise, in degrees. It plus lat/lon is the entire link between this geometry
and the real sun. Get it wrong and every photography window, every shadow, every
golden-hour claim is confidently wrong: 1 degree of heading error puts the end
of a 4 m shadow 7 cm sideways, 5 degrees puts it 35 cm, and 11 degrees — a
perfectly ordinary magnetic declination — puts it 78 cm **(measured)**.

You cannot know in advance which wall the canonicaliser picks as +X, so record a
bearing you can convert afterwards.

1. **In the room, before you scan.** Stand at your start point. Pick a long
   wall. Sight along it — phone held flat, its long edge parallel to the wall
   face — and read the bearing from the Compass app. Write down the bearing,
   which wall, and which way along it you were sighting. Photograph the room
   from that spot.
2. **Take three readings** a metre apart and use the median. Hold the phone away
   from steel doors, radiators, laptops, speakers, magnetic phone mounts and
   MagSafe cases. If the three disagree by more than about 5 degrees the room is
   magnetically dirty: step outside and take the bearing against the same wall
   line from the exterior.
3. **Magnetic is not true.** A compass points at magnetic north; the difference
   from true north is the local declination, positive east, and it reaches 20
   degrees in some places. iOS can apply it for you — Settings, Compass, *Use
   True North* (the exact path has moved between iOS versions and is
   **unverified** for yours). If you cannot confirm the setting is on, take the
   magnetic bearing and add the declination from NOAA's calculator for your
   coordinates and today's date; the field drifts, so a value from an old map
   is not good enough.
4. **After ingest**, open `locaish view` and find your reference wall. Then:

   | +X runs | heading |
   |---|---|
   | along your reference wall, in the direction you sighted | the bearing |
   | along it, the other way | bearing + 180 |
   | along the perpendicular wall | bearing ± 90 — resolve the sign in the viewer |

   Re-run `locaish ingest ... --heading <value>` with the answer.
5. **If you did not measure it, do not invent it.** `Georeference.heading_source`
   exists precisely so an assumed heading cannot be quoted as a measurement.
   `user` means you stood in the room with a compass. `assumed` means nobody
   did, and every solar result derived from it must be labelled as an
   assumption downstream.

---

## 9. Pre-flight checklist

```
[ ] Device has LiDAR, confirmed by model, not by hope
[ ] App set to a LiDAR/mesh mode, not splat, not photo
[ ] Export format confirmed on a throwaway scan: GLB or GLTF (or PLY/OBJ/STL)
[ ] Polycam only: Developer mode ON before scanning, if you want camera poses
[ ] Battery > 50%, device cool, several GB free, Do Not Disturb on, lenses clean
[ ] Every light on; room at least 50 lux
[ ] Fans, AC, TVs, monitors off; pets and people out
[ ] Doors and windows set deliberately; state written down
[ ] Mirrors covered, or listed as uncovered
[ ] Own kit removed from the room
[ ] Scale reference, at least 1 m, flat and matte, on the floor near the middle
[ ] Tape-measured: door leaf width, corner ceiling height, written down
[ ] Compass bearing taken: three readings, median, wall and direction noted
[ ] Start point chosen and memorable — you have to come back to it
```

## 10. Post-scan acceptance checklist

```
[ ] Four passes done: eye-height loop, floor loop, ceiling loop, interior fill
[ ] Loop closed at the start point, held 5-10 s
[ ] Scan under 5 minutes; if not, it should have been two scans
[ ] Nothing captured from beyond 4 m that the twin is expected to measure
[ ] Every wall got comparable time; no wall was rushed
[ ] Every opening's reveal swept from both sides, from 1-1.5 m
[ ] Scale reference visible in the app's preview before you leave
[ ] Exported in a format we read, and the file opened without error
[ ] locaish ingest run: verdict is pass (warn accepted only with a written note)
[ ] locaish measure on the 1 m reference: within +/- 15 mm
[ ] Ceiling height present and within +/- 25 mm of the tape
[ ] Opening count matches what you counted by eye
[ ] Floor area within a few percent of length x width by tape
[ ] --heading supplied and heading_source reads "user"
[ ] You are still on site while checking this list
```

That last line is the whole point of the list. Every failure below is cheap to
fix in the room and expensive to fix from an office.

---

## 11. Troubleshooting

`locaish inspect` prints the QA report. The exact check names are
**provisional** — `locaish/scan/qa.py` is being written in parallel — so the
symptom column describes what you will see rather than a literal string.

| symptom in the QA report | likely capture mistake | what to do differently |
|---|---|---|
| Unit inferred wrongly, or scale confidence low | Export carried no metric scale: splat PLY, photo-mode capture, or a CAD-unit export | Re-export from a LiDAR mesh mode. As a stopgap, `locaish ingest --unit cm` and then re-verify with the 1 m reference. |
| `ceiling_height` comes back `None` | You never looked up; the capture is open-topped | Add the ceiling pass: same perimeter loop, phone tilted up 45 degrees. |
| Ceiling height 200–400 mm short | The top band was thin and the highest returns were a wardrobe, a cornice or a beam | The `noceiling` fixture reads 2.355 m against a true 2.700 m **(measured)**. Ceiling pass, slower, closer to the walls. |
| Gravity residual above 0.25 deg | The floor was seen only at grazing angles, or from one side of the room | Dedicated floor loop at 45 degrees down, all the way round. |
| Yaw residual high, walls not axis-aligned | One long wall was captured far more thinly than the others, so the dominant direction was fitted to the wrong evidence | Give every wall equal time. If the room genuinely is not rectangular, expect this and say so. |
| Wall planarity or box residual high; dimensions inflated a few centimetres | The loop was never closed, so drift stayed piled up at the seam | 35 mm of drift added +108 mm to width and quadrupled wall flatness RMS **(measured)**. Rescan, close the loop, hold at the start point. |
| Fewer openings than the room has | Blinds or curtains closed over a window, or the reveal was swept from too far away | Open them, sweep each opening's jamb from 1–1.5 m, from both sides. |
| More openings than the room has | Furniture against a wall shadowed it and the occlusion read as a hole | Sweep around and behind free-standing furniture, or move it. |
| Capture bounds tiny; the shot search rejects standpoints | You only walked the perimeter | A perimeter loop encloses 31.5% of the floor in a 21 m² room **(measured)**. Add the interior serpentine. |
| Coverage or density warning | You walked too fast, or the room is too big for one capture | Half pace. Split rooms above roughly 9 x 9 m into overlapping captures. |
| Floor area far larger than the room | The sweep escaped through a doorway or a window into the next space | Close doors that are out of scope, keep the sweep inside, scan adjacent rooms separately. |
| A duplicate room behind a wall | A mirror | Cover it and rescan that wall. |
| A floor edge that should have a barrier | A glass balustrade the sensor saw straight through | Drape or tape the glass and rescan that section; record the barrier by hand. |
| Verdict `pass` but `measure` disagrees with the tape | Either the reference is not in the scan, or you picked the wrong endpoints in the viewer | Check both, then apply section 7's scale-versus-drift test. |
| Verdict `warn` or `fail` and you cannot rescan | Nothing to fix from here | Ship it labelled. A wrong number that announces itself is acceptable; a wrong number that does not is not. |

---

## 12. Filming instead of scanning

`locaish ingest sweep.mov` reconstructs a room from ordinary video. No depth
sensor is involved: geometry comes from parallax between frames, which changes
what a good capture looks like.

**Translate, do not pan.** This is the one rule that matters more than all the
others combined. A pan from a single spot gives the reconstruction no parallax,
and no parallax means no depth — the frames are consistent with a room of any
size at any distance. *Walk* through the space while filming. A slow circuit of
the room beats a sweep from the doorway every time, and a sweep from the doorway
is the single most common way to get a twin that fails coverage.

**Move slowly enough not to blur.** Frame selection picks the sharpest frame in
each slice of the timeline, but it can only choose among the frames you gave it.
If every frame in a two-second window is smeared, one of them still gets picked
and the geometry it contributes is the worst in the twin. The ingest reports how
many selected frames were blurred; if it is more than one or two, walk slower.

**Sixty to ninety seconds is plenty, and spend it moving.** Frames are chosen
evenly across the whole clip, so a five-minute video does not get you more of
them — it gets you the same ones, further apart. Length is not what buys
coverage; *distinct viewpoints* are. Ninety seconds of walking beats five
minutes of standing still every time.

**Raise `--frames` if the twin has holes.** The default of 24 is one window's
worth, which is what the network holds at once. Past that the sweep is
reconstructed in overlapping chunks and joined, so `--frames 72` covers
considerably more of the room at roughly three times the runtime. On a test
capture it took wall coverage from 0.36 to 0.46. It cannot invent viewpoints you
never filmed, so it rewards a thorough walk and does little for a short one.

**Hold the phone the way you normally would.** Gravity is recovered from the
camera poses, which works because people hold phones upright. Portrait or
landscape are both fine, and consistency matters more than which. Rotating
between the two mid-clip, or filming with the phone pointed straight down, makes
the estimate incoherent — the pipeline detects that and falls back to the room's
own geometry, but the fallback is the thing that was hard in the first place.

**Overlap generously.** Consecutive selected frames need to see some of the same
surfaces. A circuit where you sweep past a wall once, quickly, gives the
reconstruction one look at it; a circuit where each part of the room appears in
several frames from different positions gives it the redundancy that makes the
depth agree with itself.

**Get the floor and the ceiling in shot.** Not by pointing at them — by keeping
the phone level and letting a normal field of view catch both. A capture that
only ever saw the middle band of the walls leaves the pipeline to infer the
floor and ceiling planes from very little, and it will say so.

**Expect ±10% on size, and fix it if you need better.** Video cannot know scale;
it is inferred from a monocular metric depth prior and reported as
`scale_confidence` with an explicit error bar. If the twin needs to be right to
the centimetre, measure one length in the room with a tape, compare it against
`locaish measure`, and re-run with `--scale-factor` scaled by the ratio. That is
the same ritual as [§7](#7-the-scale-verification-ritual), and it is worth doing
before anyone plans a dolly move around the result.

**Look up, at least once.** Nobody points a phone at the ceiling, which is why
almost every video twin comes back with `ceiling_z` unknown and no ceiling
height. Two or three seconds of tilting up as you walk is enough to fix it, and
without them no amount of processing can recover a surface that was never in
frame — the completion pass will close the room at the frontier of what you saw
and label the result inferred, which is honest but is not a ceiling.

**What good looks like.** After ingest, `locaish inspect` should show floor and
ceiling coverage above 0.6, a capture coverage ratio that is not a small
fraction, no blurred-frame warning, and a small inferred-surface fraction. A
verdict of `fail` on coverage almost always means the walk was too short or too
static, not that the room was difficult.

## Sources

Verified against these, on 16 August 2026. Vendor pages change; re-check the
export menus before a shoot.

- [Polycam — What File Types Can Polycam Export?](https://learn.poly.cam/hc/en-us/articles/27756102599572-What-File-Types-Can-Polycam-Export)
- [Polycam — Pricing (free plan exports .gltf only)](https://poly.cam/pricing)
- [Polycam — How to Extract Raw Data and What Is Included](https://learn.poly.cam/hc/en-us/articles/38276871185044-How-to-Extract-Raw-Data-and-What-Is-Included)
- [PolyCam/polyform — raw data layout, camera pose JSON, depth and confidence maps](https://github.com/PolyCam/polyform)
- [Scaniverse — support and FAQ (modes, formats, free use)](https://dev.scaniverse.com/support)
- [Scaniverse — App Store listing 5.2.6 (SPZ, PLY, GLB, FBX)](https://apps.apple.com/us/app/scaniverse-3d-scanner/id1541433223)
- [Niantic Spatial — Scaniverse scanning techniques](https://www.nianticspatial.com/docs/scaniverse/techniques/)
- [Apple — Create parametric 3D room scans with RoomPlan (WWDC22): device, 9 x 9 m, 50 lux, 5 minutes, CapturedRoom, USD export](https://developer.apple.com/videos/play/wwdc2022/10127/)
- [Apple Machine Learning Research — 3D Parametric Room Representation with RoomPlan](https://machinelearning.apple.com/research/roomplan)
- [it-jim — RoomPlan in practice, absolute versus relative accuracy, door and window limitations](https://www.it-jim.com/blog/roomplan-framework-by-apple/)
- [MDPI — Use of Smartphone Lidar Technology for Low-Cost 3D Building Documentation with iPhone 13 Pro (app comparison, RMSE, drift and split surfaces)](https://www.mdpi.com/2673-7418/3/4/30)
- [ScienceDirect — Evaluating the accuracy and quality of an iPad Pro's built-in lidar for 3D indoor mapping](https://www.sciencedirect.com/science/article/pii/S2666165923000510)
- [ResearchGate — Evaluating the Accuracy of iPhone Lidar Sensor for Building Facades Conservation](https://www.researchgate.net/publication/378351112_Evaluating_the_Accuracy_of_iPhone_Lidar_Sensor_for_Building_Facades_Conservation)
- [SimplyWise — Which iPhones have LiDAR (device list)](https://www.simplywise.com/blog/which-iphones-have-lidar/)
- [RoomSketcher — Does My Phone or Tablet Have LiDAR (device list)](https://help.roomsketcher.com/hc/en-us/articles/29949063142045-Does-My-Phone-or-Tablet-Have-LiDAR)
- [NOAA NCEI — Magnetic declination calculator and help](https://www.ncei.noaa.gov/products/magnetic-declination)
- [OpenELAB — What LiDAR cannot detect: glass, mirrors, fog, black objects](https://openelab.io/blogs/learn/what-can-lidar-not-detect-blind-spots-sensor-fusion-guide)
