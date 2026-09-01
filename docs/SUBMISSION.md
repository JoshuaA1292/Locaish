# Devpost submission checklist — Agentic Cinema, ClickHouse track

**Hosted URL:** https://locaish-564007129146.us-central1.run.app

Deadline: **September 9, 2026, 2:00 PM PT**. Judging is two-stage; stage one
just verifies the thing runs and uses what it claims to use.

## Must be true at submission (rule-by-rule)

- [x] **Only Google AI models.** Gemini via `google-adk` is the only model in
  the system; reconstruction is classical COLMAP + OpenCV. No other AI
  vendor's code or weights anywhere in the repo or at runtime.
- [x] **Gemini + Agent Builder.** `google-adk` `LlmAgent` on Vertex AI
  (`locaish/agent/core.py`), an accepted package under the rules.
- [x] **ClickHouse at runtime via official MCP server.** The agent's database
  access is `mcp-clickhouse` spawned as a stdio toolset, read-only.
- [x] **New project, built inside the contest window** (first commit
  2026-08-21; window opened 2026-07-27).
- [x] **Open-source license at repo top** (LICENSE).
- [x] **Public repo** with all source and run instructions (README).
- [ ] **Billing on the Vertex project** — as of 2026-09-01 Gemini calls on
  gen-lang-client-0144049817 return 403 "billing must be enabled"; the
  studio's coverage planner and scout chat need it (or `GOOGLE_API_KEY`).
- [ ] **Hosted project URL.** Deploy with `gcloud run deploy` (command in
  README, Dockerfile included). Needs: a GCP project with Vertex AI enabled,
  and a ClickHouse Cloud instance (or any reachable ClickHouse).
- [ ] **Demo video, ≤3 minutes,** on YouTube/Vimeo, public, English or
  English subtitles. Only the first 3 minutes are judged.
- [ ] **Written description** on Devpost: features, technologies, data
  sources, learnings.
- [ ] Team ≤4, all eligible; confirm no Google/partner employment conflicts.

## Suggested 3-minute video beats

1. **(0:00–0:20)** The problem: a location scout's day rate exists because
   someone has to stand in the room. Every finalist got picked from a photo.
2. **(0:20–1:00)** Phone video of a real room dropped on the studio page.
   Quiet progress line; the twin appears with its QA verdict — point out that
   the pipeline says *how far to trust it*.
3. **(1:00–2:20)** Scout the scene. Paste the scene page, hit *Scout the
   scene*, and watch the twin: the viewfinder flies through the candidates
   ClickHouse returned for each shot, settles on the pick, and the caption
   gives the reason in a DP's words ("35 mm from 1.5 m; the window keys from
   three-quarter; matches the reverse's lens and distance"). Gemini's
   verdict on the rendered frame ("adjust: too tight, go to the 35") and the
   retry show in the caption. Then the cards, the camera plan, "what this
   room holds", and *How it decided* with the SQL. One line: every placed
   shot is a row in `shot_plans`, so "which of our five locations holds this
   scene" is a GROUP BY. Name-drop docs/CINEMATOGRAPHY.md: the rules are
   sourced, not invented.
4. **(2:20–3:00)** One sentence on architecture (every number traces to a ray
   cast, an ephemeris, or a ClickHouse query — never to the model), and the
   scale: hundreds of thousands of physically-checked setups per room,
   searchable in milliseconds.

## Draft Devpost description (edit to taste)

**Inspiration.** A location scout's trip happens after a location is already
a finalist — which means every finalist got picked on a photo and a guess.

**What it does.** Locaish turns a phone video of any room into a metric,
gravity-aligned digital twin, then puts a Gemini agent in it as a virtual
tech scout. It sweeps every physically-possible camera setup — position,
height, lens, subject mark, sightline, depth of field, backlight — into
ClickHouse, and answers shot briefs with real setups, rendered frames, and
the physical reasoning behind them. Sun schedules come from solar ephemeris
through the windows the twin actually detected.

**How we built it.** Classical structure-from-motion (COLMAP) and photometric
stereo — no neural reconstruction, by rule and by design. The twin carries a
QA report that refuses to be confident when the evidence isn't there. The
agent is a `google-adk` LlmAgent on Vertex AI whose only database access is
the official ClickHouse MCP server; its instruction forbids stating any number
that didn't come from a tool.

**Challenges.** Recovering metric scale from video without a depth network
(camera-height and doorway anchors, combined with honest error bars); stereo
pair ordering on hand-held walks; keeping the agent's claims traceable.

**What's next.** Multi-location search ("which of our five candidates can
hold this dolly move"), heading from ARKit exports, E57/LAS ingest.
