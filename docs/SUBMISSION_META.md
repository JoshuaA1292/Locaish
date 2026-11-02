# Submission: Meta Global AI Developer Hackathon (Startup track)

Paste-ready. Deadline 4 November 2026, 9:00 PM ET.

**Repo:** https://github.com/JoshuaA1292/Locaish (public, MIT)
**Hosted dorm, look around it:** https://locaish-564007129146.us-central1.run.app (look-only: the rehearsal, the camera move, the finished frames and the world made here; nothing a judge does spends)
**Video:** _paste link_

## Project name

Locaish

## Tagline

Meta's models, inside a measured room: shoot the scene before anyone drives
to the location.

## Project description (paste this)

**Locaish turns one phone walk through a room into a measured 3D location, then puts Meta's models to work inside it, so a film crew can shoot the scene before anyone drives there.**

**Who it's for.** Producers, DPs and location managers on commercials, indie film and TV. Today a location is chosen from photographs, then someone flies out with a tape measure, then a tech scout finds out the dolly doesn't fit. Locaish replaces the scouting trip, the tape measure, the pre-vis artist and the guesswork: scan the room once (50 seconds on a phone), and plan everything in metres.

**What it does.**
- Builds a measured twin from the walk (every camera pose recovered, every dimension traceable to the footage) and sweeps it: 9,666 candidate camera setups tested for lens, shot size, clear sightline and whether a dolly fits.
- **Muse Spark** reads the room and names what's in it, each name pinned to measured geometry; breaks a pasted scene into beats and shots, which the room then checks ("this dolly needs 68 cm of clear floor, there are 30"); reviews every camera move as a DP and every finished still against the real photographs.
- **Muse Wonder** operates the camera through the engine's own tools from a sentence of direction.
- **Muse Image** paints the cast into the scan's own photographs: the measured bodies are placed by the geometry and cropped to the planned lens, and the model is allowed to change the people and nothing else. It also finishes sixty views of the room for the shoot date and hour, with the sun where it will actually be.
- **SAM 3.1 and SAM 3D** lift the furniture out of the scan as solids; move the bunk bed half a metre and the shot table is recomputed (+176 clear setups).

**Where it pushes the boundary.** The models never work from a prompt alone: they work inside a measurement. Every output is checked against something real, by geometry (labels, shots, dolly fits), by the scan's photographs (the finish), or by another model acting as judge (Muse Spark reviews Muse Image and Muse Wonder, and sends work back). The painter's job was cut down until it could not invent a room, and the result is pre-vis that is faithful to the actual space. Nothing a model says ever becomes a measurement.

## Which Meta model does which job

| Model | Job | Code |
|---|---|---|
| Muse Spark | names the room; breaks the scene into beats and shots; reviews every simulated frame as a DP; reviews every finished frame beside its render | `locaish/meta/vision.py`, `locaish/agent/rehearsal.py`, `locaish/agent/sim.py`, `locaish/meta/finish.py` |
| Muse Wonder | the operator: builds camera moves through the engine's tools; the scout and planner agents | `locaish/agent/sim.py`, `locaish/meta/adk.py` |
| Muse Image | finishes frames and the sixty views of the world from the measured render and the scan's photographs | `locaish/meta/finish.py`, `locaish/product/world.py` |
| SAM 3.1 | masks movers out of the camera solve; masks a labelled thing in the frame that saw it best | `locaish/meta/masks.py`, `locaish/meta/client.py` (masks decoded with `meta-sam-parser`) |
| SAM 3D Objects | lifts the masked thing into a textured solid fitted to its measured footprint (on Modal) | `locaish/meta/sam3d.py`, `locaish/meta/modal_app.py` |
| SAM 3D Body | stands a person up from one photograph at their typed height | `locaish/meta/sam3d.py` |

## What stays honest

Nothing a model says becomes a measurement. The twin is classical
reconstruction (COLMAP, GLOMAP, OpenMVS) with a metric depth anchor; every
dimension traces to it. Finished frames carry the product's own "a render"
stamp and the measured frame is one toggle away. The world keeps the scan's
walls where they were measured and takes only the light from the finish.

## Built during the window

The Meta layer (`locaish/meta/`, `locaish/agent/sim.py`,
`locaish/agent/rehearsal.py`, `locaish/product/{cast,figures,setpieces,
stage,world}.py`, `locaish/film/{sim,rehearsal}.py`, the studio's stage and
the viewer's simulate and walk modes) was built 26 October to 4 November
2026 on branch `meta-hackathon`. The reconstruction pipeline and the shot
sweep predate it (they won the Google Agentic Cinema hackathon); the
Official Rules' statement on pre-existing work is what we read before
submitting.

## Try it

Open the hosted dorm. Walk the twin (WASD), read what Muse Spark named, open
the rehearsal and scrub the finished frames, play the camera move and read
the operator's log, open the world and stand on the photographs. To rehearse
your own scene live, clone the repo and run the studio with a Meta Model API
key (`META_API_KEY`); the hosted copy is deliberately look-only.
