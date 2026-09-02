# Devpost submission: Agentic Cinema, ClickHouse track

Everything below is paste-ready. The only field left blank is the video URL.

**Hosted URL:** https://locaish-564007129146.us-central1.run.app
**Repo:** https://github.com/JoshuaA1292/Locaish (public, MIT license at top)
**Video:** _paste YouTube link here_
**Deadline:** September 9, 2026, 2:00 PM PT. Late means disqualified, no drama, just gone.

---

## Project name

Locaish

## Tagline

The location scout you can send anywhere.

## Devpost description (paste as-is, edit to taste)

### Inspiration

A location scout's day rate exists because someone has to physically stand
in a room and answer real questions: does the dolly move fit, where does the
sun come in, which lens works from which corner. That trip happens after a
location is already a finalist. Which means every finalist got picked from a
photo and a guess. We wanted the guess gone.

### What it does

Locaish turns a sixty-second phone walkthrough into a metric, gravity-aligned
3D twin of a room, then hires a Gemini agent as its tech scout.

Every physically possible camera setup in the room gets swept into
ClickHouse: position, height, lens, subject distance, sightline, depth of
field, background depth, where the window light lands. One of our demo rooms
holds 102,852 of them. Describe a scene in plain prose and the agent breaks
it into coverage, queries ClickHouse for candidates, renders the actual view
through each chosen lens, and then looks at that frame the way a
cinematographer would. If the framing is wrong it says so, changes its
constraints, and tries again. We have watched it reject its own pick with
"16 mm is far too wide for an emotional close-up, go to the 35." It was
right.

The deliverable is what a crew actually uses: rendered frames, an overhead
camera plan, lenses and marks, a written shot list. And a viewfinder, so you
can stand in any setup yourself and look through the lens before anyone
rents a van.

Film craft is not vibes here. The 180 degree line is a SQL predicate.
Matched reverses are a lens and distance constraint. Every rule in the
ranking is sourced from working cinematography practice (docs/CINEMATOGRAPHY.md
lists the citations) and compiled once into both SQL and numpy, so the
database and the fallback planner cannot disagree.

### How we built it

The reconstruction is deliberately classical: COLMAP structure-from-motion,
OpenMVS dense stereo, and a per-scene gaussian splat fit with Brush for the
photoreal view layer. No pretrained models, no neural reconstruction. Partly
because the rules say Google AI only, mostly because a measurement you can't
audit is just a confident guess. The twin ships with a QA report that grades
its own scan and says how far to trust it.

The agent side is `google-adk`: a breakdown agent, a per-shot placer, and a
frame reviewer, all Gemini, served through Vertex AI on the hosted demo
(the Cloud Run service account is the credential, so no key ships). The
agent's only database access is the official `mcp-clickhouse` MCP server,
read-only, spawned as a stdio toolset. Its instructions forbid stating any
number that did not come from a tool.

Hosting is one Cloud Run container: the studio in showcase mode, a local
ClickHouse loaded from baked dumps at startup, and the scanned rooms.
Scanning itself runs locally (it wants a GPU and fifteen minutes), so the
hosted gallery serves rooms we scanned and approved, and anyone who clones
the repo can scan their own.

### Challenges we ran into

Metric scale from video without a depth network: we anchor on camera height
and door leaves, because a door is a manufactured standard hiding in every
room. Hand-held stereo on textureless walls: most of what stereo returns
there is correlated noise, so we learned to delete it and let the splat
carry the look while the crisp points carry the measurements. Cost: our
first agent loop resent the whole conversation every shot and burned 85
cents a scout; fresh sessions per shot and trimmed tool results got it to 31
cents. Cloud Run: it caps responses at 32 MiB and our viewer page was 130 MB,
so the hosted rooms serve an 800k-point render, pre-gzipped, under the cap.

### Accomplishments we're proud of

The agent argues with the database and wins for the right reasons. The
craft rules survive being read by a film person. The whole thing runs
end to end on a phone video of an ordinary kitchen. And when the scan is
bad, Locaish says so instead of decorating a guess.

### What we learned

A self-check that reuses the thing it is checking will happily confirm an
answer that is 90 degrees wrong; our gravity cross-check now compares two
quantities that share no arithmetic. Phone cameras know which way is up
better than sparse geometry does. ClickHouse partition drops make "reload
this room's 100k setups" feel instant. And the fastest way to make an LLM
trustworthy is to take away every excuse to guess: give it tools that
measure, and instructions that forbid numbers from anywhere else.

### What's next

Scale the shot space: a production location library is thousands of rooms
at 100k setups each, which is exactly the columnar search ClickHouse was
built for. Multi-location answers ("which of our five candidates holds this
scene" is already a GROUP BY away). Heading from ARKit so sun schedules work
on every scan. And dolly moves as first-class citizens: sweep the track, not
just the tripod.

## Built with

`python` `gemini` `google-adk` `google-genai` `clickhouse` `mcp-clickhouse`
`cloud-run` `colmap` `openmvs` `brush-3dgs` `numpy` `scipy`

## Data sources

Everything is generated by the pipeline from our own phone captures: the
twins, the swept setups, the plans. The cinematography ranking rules are
sourced from public film-craft references, cited in docs/CINEMATOGRAPHY.md.
Sun position comes from a solar ephemeris calculation, not an API.

## Checklist (rule by rule)

- [x] Only Google AI models: Gemini via `google-adk` and `google-genai`,
  both on the accepted package list. The hosted demo serves Gemini through
  Vertex AI using Cloud Run's service identity, so no API key ships with
  the service. Reconstruction is classical, no other vendor's models or
  APIs anywhere.
- [x] ClickHouse at runtime via the official `mcp-clickhouse` MCP server,
  read-only, self-hosted cluster.
- [x] Hosted URL live and public, uploads disabled, two scanned rooms,
  scout and chat fully working: https://locaish-564007129146.us-central1.run.app
- [x] Public repo, MIT LICENSE detectable at top, partner usage visible in
  code (`locaish/agent/core.py`, `locaish/warehouse.py`).
- [x] Runnable from a fresh clone with no scanning: `examples/IMG_6086.twin`
  is a finished scan of a real room; drop it on the studio and the sweep,
  scout and viewfinder all come alive.
- [x] Built inside the contest window (first commit 2026-08-21).
- [ ] Video: 3 minutes or less, public on YouTube or Vimeo, English.
  Record the scout on the hosted URL so the address bar does the proving.
- [ ] Paste this description into the Devpost form, add the video link,
  submit before 2:00 PM PT on September 9.
