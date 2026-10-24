# Locaish: from hackathon win to company

*Prepared 19 October 2026. Research sources are linked inline; anything marked **estimate** is my own arithmetic from those sources, not a published figure.*

---

## 0. The short version

**What Locaish is today.** A phone walkthrough of a room becomes a metric, gravity-aligned digital twin with a self-grading QA report. From that twin the software produces the deliverable a tech scout exists to produce: sun through each window by the hour, which dollies and jibs fit, an acoustic estimate, every physically possible camera setup swept into a searchable table, and an agent that plans a scene's coverage and renders the frames. As of 20 October 2026 it also produces the three things a production company buys (Section 1.4): a **location package** per room with the sun on the shoot date drawn on the floor plan, a **project** that ranks every candidate room against the shot list, and a **share link** the director opens on a phone without an account. It is roughly 34,000 lines of Python, 291 tests, one Cloud Run deployment, and it won its track at Agentic Cinema.

**The market to start in.** Commercial and branded-content production in the US and UK: the roughly 400 [AICP](https://aicp.com/membership-information/member-directory/general-member-directory) member companies that make 80 to 85 percent of US commercials, their heads of production, and the freelance location scouts they hire. Not features, not television, not real estate. Section 3 explains why.

**The company.** Locaish becomes *the index of filmable space*. Every scouted location, every marketplace listing, every film-commission library entry gets a twin and a sweep, so "which of our locations holds this scene" becomes a query rather than a van ride. The wedge is the scout's deliverable. The endgame is the Matterport of production, where Matterport indexed space by floor plan and sold for [$1.6 billion](https://www.inman.com/2025/02/28/costar-closes-1-6b-matterport-acquisition-ahead-of-schedule/) to a real-estate data company, and Locaish indexes space by what a camera can do in it.

**The honest evaluation.** Score 6.5 out of 10 as a venture today. The technology is real and differentiated, the problem is real, the beachhead is reachable by one person with a phone and a list of 400 companies. The weaknesses are a contracting core industry, a small beachhead (an **estimate** of $20 to 30 million a year of addressable spend), a solo founder, and a reconstruction path whose metric scale from plain video is still "tens of percent" without a tape measure. All four have concrete fixes in Section 8.

**The immediate decision.** YC Winter 2027 applications close [2 November 2026 at 8pm PT](https://x.com/ycombinator/status/2111183078072979886). That is 14 days from today. Section 9 says what to do with those 14 days and what to submit.

---

## 1. What Locaish actually does (and what it does not)

### 1.1 The pipeline, as built

| Stage | What happens | What it is worth commercially |
|---|---|---|
| **Capture** | Any scanner export (Scaniverse, Polycam, Luma, RoomPlan, COLMAP) or a plain phone video | Zero hardware ask. A scout already walks the room with a phone. |
| **Reconstruct** | Classical SfM (COLMAP) plus dense stereo (OpenMVS), gaussian-splat view layer (Brush). Metric scale from two independent anchors: camera height above floor and door-leaf height. Gravity from camera poses. Holes closed by free-space carving, with every inferred vertex labelled. | No learned model in the measurement path means every number is auditable. This matters more to a buyer than it sounds: the alternative products sell pictures, Locaish sells numbers it can defend. |
| **QA** | Twin grades itself `pass`, `warn`, `fail` with a `scale_confidence`. On a held-out synthetic benchmark of 25 rooms, 0 twins were wrong while reporting `pass`, and trusted twins measured plan dimensions to 1 mm and ceilings to 2 mm (scan-file input). | "A twin whose QA verdict is pass is accurate" is a product promise competitors do not make. |
| **Scout report** | By department: space, camera (longest sightlines, allowed framings), grip (which real dollies, sliders and jibs fit, from a catalogue of real equipment dimensions), sound (RT60 estimate), light (sunrise, golden hour, per-window direct-sun intervals from solar ephemeris, given a georeference and heading). | This is the thing a tech scout day is paid for. |
| **Sweep** | Every standable camera position at three heights on six primes against every subject mark, scored on sightline, clearance, headroom, framing, depth of field, backlight, background depth, key-light angle, wall angle. 30,000 to 300,000 rows per room in about a second, into ClickHouse. | The index. This is the data asset that compounds. |
| **Coverage agent** | Paste a scene. Agent breaks it into shots, enforces the 180-degree line as a SQL predicate, matched reverses, OTS geometry, queries the sweep, renders each frame from the splat, has a vision model judge it the way a DP would, iterates. Outputs shot cards, overhead camera plan, shot list, and writes the plan back to the index. | The "tedious day done" for a DP. Measured at about $0.31 of model cost and 200 seconds per 6-shot scene. |
| **Viewfinder** | Stand in any setup in the 3D twin, change lens with `[` `]`, drag to re-aim, snap to the nearest swept setup. | The demo moment. Directors understand this in ten seconds. |

### 1.4 The product, v1 (built 20 October 2026)

The engine above was a hackathon demo: one room, a chat box, a coverage planner. What a production company pays for is a deliverable, a decision and a way to send it. Research on what a scout actually hands in ([checklists](https://bpb-us-w2.wpmucdn.com/hawksites.newpaltz.edu/dist/7/20978/files/2025/04/Location-Scout-Checklist.pdf), [templates](https://artsandsciences.csuohio.edu/sites/default/files/Filmsourcing_Location_Scouting_v1.3.pdf)) agrees on the contents: photos, dimensions, power, parking, access, contact, and the sun on the shoot day, for which scouts carry a sun-path app and visit at the planned hour. None of that is spatial; all of it is now one page.

| Feature | What it does | Why it sells |
|---|---|---|
| **Location package** (`/package/<room>?date=`) | One printable page per room: dimensions with a plain-English error bar, the practical fields the scout types in, lens table, gear verdicts, RT60, and three floor plans showing *where direct sun lands on the floor through each window* at 09:00, 12:00 and 15:00 solar on the shoot date. | It is the scout's existing deliverable with numbers in it. The sun-on-the-floor drawing is something no sun-path app can do, because none of them know where the windows and walls are. |
| **Projects** (`/projects`) | A job: client, brief as a shot list, shoot date, candidate rooms. "Compare the rooms" places every shot in every room with the deterministic planner and ranks them: holds, holds with a compromise (lens, height, window behind the subject), cannot. Three real rooms compare in under three seconds. | This is "scout six, drive to two" as a button. The ranking explains itself in a sentence a producer can repeat to a director. |
| **Share links** (`/s/<token>`) | Read-only, unguessable, revocable URLs for a package, a project or the 3D twin. No account for the recipient. Opens are counted. | The director is in another city and will not create an account. Forwardability is the distribution. |
| **Capture page** (`/capture`) | The five-step recipe plus a camera button that uploads when filming stops; works on the phone in the room. | Removes the "how do I film it" call. The LiDAR-export path (Polycam, Scaniverse) is recommended on the page for metric-to-centimetre twins today, without waiting for our own app. |
| **What is in the room** (21 October) | A semantic pass after reconstruction: a vision model is shown eight frames of the capture and names what it sees with a box; each box becomes a ray from that frame's camera into the measured cloud, and the first surface it meets is where the thing is. Names are provisional until a person confirms them; anything missed is named by clicking it in the twin, which brings up the photographs that saw that spot. Names appear on the floor plans, in the package, and in the scout agent's answers. On a real room, three frames placed 20 objects at plausible heights (sofa 0.36 m, coffee table 0.15 m, ceiling fan 2.28 m) in 19 seconds for about a cent. | The art department and the gaffer read a package by what is in the room, not by its area. A model that names things and a person who corrects it is cheaper than either alone. |
| **Details** | Address, contact, power, parking, access, restrictions, fee, and the compass heading and coordinates. The heading the scout types in overrides the capture's assumed one and turns the sun schedule from an assumption into a measurement. | The sun section is only worth money if the heading is right; a compass reading on one window is a thirty-second fix. |
| **Passcode deployment** | One passcode per customer deployment; share links, pricing and capture stay public. State is JSON beside the jobs, so a studio root is a complete backup. | Lets ten design partners run on ten private deployments without an auth system, which is the honest amount of infrastructure for ten customers. |
| **Pricing and pilot intake** (`/pricing`) | The plans below, Stripe payment links when configured, and a design-partner form that writes to disk and a webhook. | The first paying customer needs somewhere to pay. |

Deliberately not built yet: accounts and multi-tenant billing (one deployment per customer is enough for the first ten), the ARKit capture app (LiDAR exports from existing apps cover the metric problem for anyone with a Pro iPhone), civil-time sun schedules (times are solar, labelled as such; a timezone database is a small follow-up).

### 1.2 What it is not, stated plainly

- **Not survey grade from video.** A scan-file twin measures to millimetres. A video twin is "metric to tens of percent" unless the user supplies one tape measurement. This is the single most important technical gap for a commercial product and it has a known fix (Section 8.1).
- **Not exteriors.** The room model assumes floor, walls, apertures. Streets, fields, and building facades are a different reconstruction problem.
- **Not ceilings from video.** Nobody films the ceiling, so ceiling height is usually unknown on video input.
- **Not fast.** Reconstruction plus splat training is about 15 minutes on a laptop GPU. Fine for a scout filing a report that evening, not fine for "scan and see it now."
- **Not a mobile app.** Capture is "film it however, upload it here." There is no in-app capture guidance, no ARKit pose stream, no LiDAR use.
- **Not free of hackathon scar tissue.** The Google-only model rule and the ClickHouse-via-MCP requirement shaped the architecture. The rule no longer applies. The ClickHouse index is worth keeping because it *is* the product; the Gemini-only constraint is not.

### 1.3 What is genuinely hard to copy

1. **The craft compiler.** Twenty-odd cinematography conventions, each sourced to the standard texts, each reduced to a column or a predicate, emitted once as both SQL and numpy so the database and the fallback planner cannot disagree. A competitor can write "find a clean 75mm medium" as a prompt; Locaish can answer it as a query over a hundred thousand measured candidates. That took a reading of Brown, Katz, Mascelli and Bowen plus a month of geometry, and it is documented in `docs/CINEMATOGRAPHY.md`.
2. **Honesty as a feature.** Independent scale anchors, inflated uncertainty on disagreement, labelled inferred geometry, refusal over guessing. This is the opposite of how every consumer scanning app behaves and it is exactly what a line producer needs before betting a shoot day on a room.
3. **The index.** Every twin adds rows. "Which of our 400 locations holds this scene" is a `GROUP BY` today. No one else has a shot-level index of real places.

---

## 2. The problem, with numbers

A location gets chosen on photos, then a crew drives out to find out whether the photos lied.

**What the trip costs.** A tech scout brings the department heads: director, producer or UPM, DP, gaffer, production designer, location manager, often the 1st AD and key grip. Sources put the group at [3 to 5 people for small jobs and up to 15 for large ones](https://www.wrapbook.com/blog/tech-scout). Day rates for the people on that van, from industry guides (not union tables, treat as indicative): location scouts [$400 to $1,200 a day](https://getcameracrew.com/resources/learn/what-is-a-location-manager) depending on budget tier, location managers [$3,000 to $6,000 a week](https://saturation.io/film-crew-positions/location-manager) on union features. A DP, gaffer, and production designer each cost more than a scout. A fully loaded commercial tech scout day is an **estimate** of $6,000 to $15,000 including transport, and it only confirms or rejects a location that was already a finalist. Official rates for commercials are in the [AICP and Teamsters Local 399 location managers agreement, 2023 to 2026](https://www.ht399.org/app/uploads/2024/12/AICP-Local-399-Locations-Managers-2023-2026-CBA.final_.pdf); verify against that before quoting a number to a buyer.

**What a wrong pick costs.** No published study quantifies it, which is itself a finding: the industry treats it as weather. Practitioner accounts agree on the mechanism. John August's rule was that he [never signed off on a location unless his DP had seen it](https://johnaugust.com/?p=806). MINDCASTLE's DP scouts [at the hour they plan to shoot](https://musicbed.com/blog/?p=2866) because light changes the picture. Zacuto describes companies that [send a PA with an iPhone](https://zacuto.com/blogs/blog/9-things-to-consider-when-scouting-a-movie-location) and discover on the day that the room is unusable. The cost of the wrong pick is a lost shoot day, and in commercials a shoot day is most of the budget: the [median US 30-second spot costs about $342,000](https://greenfroglabs.com/blog/30-second-commercial-cost) to produce (a figure attributed to a 2025 cost survey and not independently verified) and is typically shot in one or two days.

**What is being spent around the problem.** The global [film location rental market is put at $4.2 billion in 2025](https://dataintelo.com/report/film-location-rental-market) by one research house. UK film and high-end TV spend hit a record [£6.8 billion in 2025, up 22 percent](https://www.bfi.org.uk/news/official-bfi-statistics-2025). The LED-volume virtual production market, which consumes scanned real locations as backgrounds, is around [$3.7 billion in 2026](https://www.globenewswire.com/news-release/2026/10/14/3380872/0/en/virtual-production-market-research-company-expansions-and-latest-industry-news-global-market-outlook-led-studio-investments-and-ai-driven-production-trends.html) across several estimates. Locaish does not capture any of these directly. It captures the decision cost that sits in front of all of them.

**The counter-signal, stated up front.** Los Angeles on-location production fell [16.1 percent in 2025](https://deadline.com/2026/01/los-angeles-film-tv-on-location-production-2025-q4-filmla-1236684296/) to 19,694 shoot days, and commercials fell hardest in the back half: [down 17.9 percent in Q3](https://www.cined.com/la-film-production-declines-13-2-in-q3-2025-incentive-backed-projects-signal-hope/) and [23.2 percent in Q4](https://deadline.com/2026/01/los-angeles-film-tv-on-location-production-2025-q4-filmla-1236684296/). US high-budget production spend fell [20 percent to $12.1 billion in 2025](https://www.thewrap.com/industry-news/industry-trends/us-film-production-2026/) while the UK rose 15 percent. The industry Locaish serves is shrinking in its home city and shifting geographically. Section 3 argues this helps the wedge; it does not help the TAM.

---

## 3. Market selection

I scored eight candidate markets against six criteria. The criteria, in the order a seed-stage company should weigh them: frequency of the decision Locaish improves, pain per decision, reachability of the buyer by a solo founder, fit with what is built today, willingness to pay, and expansion path.

| Market | Decision frequency | Pain per decision | Reachable | Fit today | Pays | Expands to | Verdict |
|---|---|---|---|---|---|---|---|
| **Commercial and branded content production (US, UK)** | Very high: 5 to 10 locations considered per job, dozens of jobs a year per company, 2 to 3 week prep windows | High: one shoot day is most of the budget; directors bid remotely and often cannot visit | High: ~400 AICP companies, named heads of production, trade press that covers tools | Strong: interiors, prime lenses, dialogue coverage, daylight through windows | Yes: already pays scouts $400 to $1,200 a day and scanning bureaus £3,000 to £8,000 a location | Scouts' libraries, marketplaces, commissions | **Beachhead** |
| Freelance location scouts and their libraries | High: scouts hold libraries of hundreds to thousands of locations | Medium: their deliverable is a JPEG folder; a twin is a better deliverable and a competitive edge | High: they are on every job above; SuperScout proves they will adopt software | Strong | Lower individual ACV but they are the capture workforce | Everything | **Beachhead channel** (same motion, same people) |
| Feature and episodic location departments | Medium | Very high per pick, but they already budget for tech scouts and have [about 800 unionised location managers in LA](https://www.yahoo.com/entertainment/hollywood-teamsters-local-399-reaches-183502334.html) whose day rates the product threatens | Medium: gatekept, union, slow | Strong for interiors | Yes but procurement is per-production | Studios' location archives | **Second market**, enter via DPs and production designers who sat on commercial tech scouts |
| Location marketplaces (Giggster [45,000+ listings](https://www.eventsinminutes.com/blog/best-peerspace-alternatives-2026/), Peerspace [40,000](https://en.wikipedia.org/wiki/Peerspace), SetVenue) | Very high across the platform | Low per listing, high in aggregate: reviewers complain listings have [poor photos and hosts who do not understand production](https://setvenue.com/blog/setvenue-vs-giggster-vs-peerspace-comparison) | Medium: partnership, not sales | Strong | Hosts already pay [$150 to $600 for a real-estate 3D tour](https://travvir.com/blog/en/real-estate-virtual-tour-cost) | Platform-scale index | **Phase 3 distribution**, not a starting point: no product to partner with until the wedge works |
| Film commissions ([360+ AFCI members](https://en.wikipedia.org/wiki/Association_of_Film_Commissioners_International), libraries run on [Reel-Scout](https://www.reel-scout.com/clients)) | High | Medium | Low: government procurement, annual budgets | Partial: libraries are exterior-heavy | Yes, by subscription, slowly | Regional index | **Phase 3**, via a Reel-Scout integration or by displacing it |
| Virtual production stages (about [200 LED volumes](https://www.futuresource-consulting.com/the-source/industry-pulse/led-volume-making-waves-in-virtual-production/) at end of 2023, growing 18 percent a year) | Medium | High | Medium | Weak: they want relightable splats of exteriors, not room measurements; Mo-Sys, XGRIDS and bureaus already serve this | Yes | Export path | **Feature, not market**: ship a splat export to Unreal and OpenUSD, which [ratified gaussian splat support in April 2026](https://radiancefields.com/gaussian-splatting-statistics) |
| Real estate | Very high | Low | Low | Weak | Commoditised | None for Locaish | **No**: CoStar owns Matterport and 12 million spaces |
| Architecture, construction, insurance | High | High | Medium | Weak: they need survey grade and exteriors | Yes | None near-term | **No** |

### 3.1 Why commercials specifically

1. **The decision happens before the money exists.** A commercial production company bids on a board, and the bid includes a location plan. They scout during the bid, before the job is awarded, and lose the scouting cost on every bid they lose. A tool that turns a scout's phone video into a tech report shifts the bid from guess to measurement at near-zero marginal cost. Nobody is building for the bid window.
2. **The director is often not in the city.** Commercial directors are booked from rosters and bid on jobs across countries. The treatment is written from photos. A viewfinder they can stand in from London for a Los Angeles kitchen is a direct substitute for a flight.
3. **Prep is too short for two trips.** Two to three weeks from award to shoot means one scout, one tech scout, one shoot. A location that fails at the tech scout has no time to be replaced well.
4. **The buyer is concentrated and listed.** The AICP directory is public. Heads of production and executive producers make tooling decisions without a studio procurement process.
5. **The shrinkage helps the pitch.** When shoot days fall 20 percent, the companies that survive are the ones that cut prep cost and win more bids per dollar. "Scout six, drive to two" is a survival pitch in 2026, not a convenience pitch.
6. **What is built fits it.** Interiors, dialogue coverage, prime lenses, daylight. A commercial is a kitchen, a living room, an office, a studio stage. Locaish was accidentally built for exactly these rooms.

### 3.2 The buyer, the user, the payer

- **User:** the freelance location scout (captures) and the DP or director (consumes the viewfinder and the coverage plan).
- **Buyer:** head of production or executive producer at the production company.
- **Payer:** the production company, as a monthly subscription with per-twin processing, billed to the job as a line item the way scouts already are.
- **Champion to recruit first:** the scout. They are on every job, they are competitive with each other, and a twin makes their deliverable obviously better than the next scout's folder of photos. Giving scouts the tool free for the first six months is the cheapest distribution available.

---

## 4. Competitive landscape

| Who | What they sell | What they lack that Locaish has | Threat level |
|---|---|---|---|
| [SuperScout](https://superscout.ai/team/) (UK, launched Dec 2023, grew out of a space-booking platform) | Database management for scouts' existing photo libraries: map search, keyword tagging | No 3D, no measurement, no sun, no grip fit, no shot search. Photos in, photos out. | Medium. Closest in *customer*. Natural acquirer or partner, or the company Locaish out-executes. |
| [Reel-Scout](https://www.reel-scout.com/clients) | Film-commission location and project databases, annual subscription, clients include California, Houston, Austin, Screen Scotland | Same: photo libraries. No spatial data. | Low near-term. Integration target in Phase 3. |
| Giggster, Peerspace, SetVenue | Marketplaces, 15 to 20 percent of booking | Listing quality is their known weakness. They need what Locaish makes. | Low as competitors, high as channel. |
| Matterport (CoStar), [$169.7 million revenue 2024](https://www.inman.com/2025/02/28/costar-closes-1-6b-matterport-acquisition-ahead-of-schedule/), 12 million spaces | Real-estate twins, dollhouse views, floor plans | No film semantics: no lens, no sun through apertures, no equipment fit, no coverage. Owned by a real-estate data company with no reason to enter production. | Low. The comparable, not the competitor. |
| Polycam ([$18 million Series A 2024](https://radiancefields.com/polycam-announces-18m-series-a), 100,000+ paying subscribers), Scaniverse / [Niantic Spatial ($250 million, 2025)](https://en.wikipedia.org/wiki/Niantic_Spatial), Luma | Capture apps and splat platforms | No measurement promise, no QA, no film layer. They are inputs to Locaish today. | Medium long-term: Niantic Spatial names entertainment as a target vertical. Locaish must own the film layer before a capture platform bolts one on. |
| Cine Tracer ($65 to $90 one-time), Previs Pro ($40 a month), Unreal | Previs in synthetic sets | No real location. You previs a room that does not exist, then go find one like it. | Low. Complementary; export a twin into them. |
| Scanning bureaus (Visual Skies, Clear Angle, SetScan): [£3,000 to £8,000 per location, £30,000 to £150,000 per production](https://visualskies.com/3d-scanning-for-film-and-tv/how-much-does-it-cost-to-lidar-scan-a-film-set-v3/) | Survey-grade LiDAR for VFX and VP | Price. Locaish is a phone and a hundred dollars. | None at the beachhead; they serve $50 million features. |
| Vū / SceneForge, Cuebric | Virtual production previs and AI backgrounds | Stage-side, not location-side | Low. |
| AI filmmaking startups (YC's [Martini W26](https://www.ycombinator.com/companies/martini), Flick, Koyal; Higgsfield at a [reported $5.4 billion](https://fortune.com/2026/09/30/cio-intelligence-sept-30/)) | Generating footage | They make pixels. Locaish measures places where real cameras go. Different axis entirely. | None, but they absorb investor attention in "AI for film." Position away from them. |

**The positioning sentence:** every tool above either shows you pictures of a place or lets you imagine a place. Locaish measures a real place and tells you what a camera can do in it.

---

## 5. The extended mission

**Mission:** index the physical world by what a camera can do in it.

**Why that framing.** Matterport proved that a twin per space is a billion-dollar data asset when an industry's decisions depend on space. Real estate's decision is "would I live here." Production's decision is "can I shoot this here," and it is asked millions of times a year across commercials, features, television, music videos, photography, events and live broadcast, always answered today by a person standing in a room. Locaish answers it from an index.

**The ladder, each rung paying for the next:**

1. **The scout's deliverable (2026 to 2027).** A scout's phone video becomes a twin plus a tech report. Sold to commercial production companies. Proves the unit economics and builds the first thousand twins.
2. **The company's library (2027).** Every location a production company or scout has ever captured, searchable by shot: "a 75mm two-shot with a window key at 4pm in November, within 30 miles of the stage." This is the ClickHouse `GROUP BY` the README already describes, turned into the product. Retention comes from the library.
3. **The market's index (2027 to 2028).** Marketplaces and film commissions put twins on listings. Hosts pay for capture the way they pay real-estate photographers today. Locaish becomes the shot-search layer under Giggster, Peerspace and the commission libraries, and the first place a director looks.
4. **Beyond the room (2028+).** Exteriors (a new reconstruction path), dolly and crane moves as first-class swept objects, relightable splats into Unreal and OpenUSD for virtual production, and the same index serving event planners (sightlines and AV), broadcast (camera positions in venues), and photographers. The corpus of consented, measured interior spaces also has a second life as simulation data for robotics, which is where Niantic Spatial is already pointing Scaniverse; that is an option, not a plan, and it must be consent-first.

**The one-line version for a YC application:** "Locaish turns a phone video of a room into a measured 3D twin and tells filmmakers exactly where the camera can go. We are building the index of every filmable space on earth."

---

## 6. Business model and pricing

Pricing is anchored to what the buyer already pays, which is the only anchor that works in production.

| Plan | Who | Price | Anchor |
|---|---|---|---|
| **Scout** (free for 6 months, then $29 a month) | Freelance scouts and location managers | 10 packages a month, share links, personal library | SuperScout charges £20 to £50; the research of 21 October (docs/BOTTLENECK_RESEARCH.md) put the earlier $49 above the whole band scouts demonstrably pay |
| **Production, per job** ($249 a job) | Production companies bidding | Up to eight candidate rooms ranked against the shot list, packages and share links for 60 days | Billed to the job the way the scout is; a sample AICP bid carries ~$2,610 of scouting lines on a $95,000 spot, so one job's package is a tenth of what it already spends to decide |
| **Production** ($499 a month) | Production companies with steady volume | Unlimited jobs, team passcode, company library, shot search, coverage agent | AICP members average about five shoot days a month; two jobs a month makes the subscription cheaper than per-job |
| **Studio and Enterprise** (from $2,500 a month) | Studios, large commercial houses, commissions, marketplaces | API, SSO, bulk ingest, custom equipment catalogues, white-label viewfinder on listings | Scanning bureaus charge £3,000 to £8,000 per location |
| **Single package** ($29 per location, no subscription) | Scouts and producers on one job | One location package with share link, kept for 90 days | A scout bills per job; the first purchase should not require a monthly decision |
| **Capture service** ($350 to $750 per location, year one only) | Any of the above | A Locaish-trained scout captures and delivers the twin | Real-estate hosts already pay $150 to $600 for a 3D tour; this is "AI-native service company" in YC's own Summer 2026 language, and it dilutes to software as the app matures |
| **Host listing twin** ($199 one-time, Phase 3) | Marketplace hosts | Twin and sweep on the listing | Same real-estate anchor |

**Unit economics (estimate).** Cloud GPU reconstruction plus splat training is roughly $0.50 to $2.00 per twin on a rented GPU; the coverage agent is about $0.31 per scene at current model prices; storage is cents. Gross margin on the software tiers should exceed 85 percent. The capture service runs at 40 to 50 percent margin and exists to seed the index, not to make money.

**Bottom-up size of the beachhead (estimate, show the arithmetic).** FilmLA counted about 2,800 commercial shoot days in Los Angeles in 2025 (summing the quarterly figures above). Los Angeles is perhaps a quarter of US commercial production, giving something like 10,000 US commercial shoot days a year. If each job considers 5 to 10 locations, that is 50,000 to 100,000 location evaluations a year in US commercials alone. At $29 to $49 per twin, that is $1.5 to $5 million of processing revenue, plus subscriptions: 400 AICP companies at $499 a month is $2.4 million a year at full penetration, and an **estimate** of 2,000 US freelance scouts at $49 a month is another $1.2 million. Add the UK at roughly a third of the US. The beachhead is a $20 to 30 million a year market at saturation. That is enough to build a company on and not enough to excite a growth investor, which is why the ladder in Section 5 exists and why the application must lead with the index, not the scout tool.

**Top-down context for the ladder.** 85,000 marketplace listings at $199 is $17 million of one-time capture revenue before any search subscription. 360 film commissions at $30,000 a year is $11 million. Matterport reached $170 million of revenue on 12 million spaces in an industry where the twin is a marketing asset; in production the twin is a decision asset, which should price higher per space.

---

## 7. Go-to-market plan

### Phase 0: the next 14 days (19 October to 2 November 2026)

Goal: a credible YC Winter 2027 application and the first five design-partner conversations. Do not build features.

1. **Fix the story, not the code.** ~~Rewrite the README's first screen for a producer, not a judge. Remove every sentence about hackathon rules.~~ Done 20 October, together with the product layer in Section 1.4: the hosted demo now opens on locations, projects and packages, not on the architecture.
2. **Twenty conversations.** Pull 40 names from the AICP member directory (Los Angeles, New York, London) and the LMGI member directory. Cold email heads of production and working scouts with the hosted demo link and one sentence: "Scan a room with your phone, get the tech scout report before anyone drives out." Ask for 20 minutes. Record every objection verbatim. Target: 20 calls booked, 5 completed before 2 November.
3. **Three live captures.** Get three scouts or producers to film a real room on a real job and send the video. Process it. Send the report and viewfinder back within 24 hours. These three become the application's traction section whether or not they pay.
4. **Find the cofounder conversation.** YC funds solo founders, but [the pattern it funds is a technical founder paired with someone who has sold into the market](https://www.wisp.blog/blog/does-y-combinator-accept-solo-founders-heres-what-you-need-to-know). A working commercial location scout or line producer who wants out of the van is the ideal partner. Do not add a cofounder for the application; do start the conversation and say so in the application.
5. **Submit by 2 November 8pm PT.** Section 9 drafts the answers.

### Phase 1: design partners (November 2026 to February 2027)

Goal: 10 companies using it on real jobs, 3 paying, one written case study with a dollar figure.

- **Offer:** free Production plan for 90 days in exchange for one captured room per week and a 15-minute call every two weeks.
- **Build only what the partners block on.** The share link and the package (the "PDF report that looks like the one a scout hands in") shipped on 20 October. Expected next: an iPhone capture flow using ARKit poses and LiDAR depth, which solves scale, gravity and heading in one move (Section 8.1); civil-time sun schedules; whatever the first three partners say in week one.
- **Measure the money metric in the product.** The project page records every comparison; the number of rooms ranked "cannot" or "with a compromise" and then *not visited* is the number of vans that did not leave. Ask each partner to mark which rooms they drove to.
- **Capture service as the forcing function.** Offer to send someone to capture for $350. If partners pay for that, the demand is real and the software's job is to make the $350 go to zero.
- **Measure:** twins per week, percent of twins that replaced a physical visit (ask every time), hours from upload to report, number of locations rejected from the twin alone (this is the money metric: every rejection is a van that did not leave).
- **Pricing test:** charge from day 91. Expect two of ten to churn, three to pay, five to negotiate. Learn the number.

### Phase 2: the library (March to August 2027)

Goal: $15,000 to $30,000 of monthly recurring revenue, 50 paying companies and scouts, 2,000 twins in the index, a batch or a seed round.

- **Ship the company library and shot search.** This is the retention product. A company with 200 twins in Locaish does not leave.
- **Scouts as the channel.** Free Scout plan, referral credit when a scout's twin is opened by a company that then subscribes. Scouts carry the tool from job to job and company to company.
- **Trade presence.** AICP Week, the LMGI Awards (which has an Outstanding Film Commission category, meaning commissions are in the room), British Cinematographer and Deadline's production desks. One case study with a named company and a dollar figure beats any ad.
- **London in parallel with Los Angeles.** UK production spend is up 22 percent while LA is down 16 percent; the UK Guild of Location Managers is smaller and more approachable than Local 399; SuperScout is there, which means scouts there already buy software.

### Phase 3: the index (September 2027 onward)

- **Marketplace partnership.** Approach Giggster first (production-focused, 45,000 listings, known listing-quality problem): twins on listings, shot search in their UI, Locaish paid per capture and per search seat. Peerspace second.
- **Film commissions.** Pilot with one Reel-Scout customer that wants a differentiator (Screen Scotland and Ontario have been public about tooling). Integrate before competing.
- **Virtual production export.** Splat to OpenUSD and Unreal. Not a market; a reason stage owners send captures.
- **Features and episodic.** Enter through DPs and production designers who used it on commercials. Sell to the production, not the location department.

---

## 8. Honest evaluation

### 8.1 Technical risks and their fixes

| Risk | Severity | Fix | Cost |
|---|---|---|---|
| Video twins are metric only to tens of percent without a tape measurement | **High.** A tech report with a 20 percent error bar on the room width is not a tech report. | **Today:** the capture page steers scouts with a Pro iPhone to a Polycam or Scaniverse LiDAR export, which the pipeline already reads and which is metric to centimetres; the package prints the error band in words so a video twin never pretends. **Next:** an iPhone capture app that streams ARKit camera poses and LiDAR depth. Metric scale, gravity and compass heading come from the device; COLMAP becomes a refinement step rather than the only source. iPhone Pro models have had LiDAR since 2020 and every scout owns one. The hackathon's "no learned models" rule is gone, so a depth prior is also allowed now, but prefer the sensor. | 4 to 6 weeks for a Swift capture app plus pipeline changes. The single highest-value engineering item. |
| 15-minute reconstruction | Medium | Cloud GPU workers (Modal, RunPod, or GCP with the GPU dense path already documented in `docs/GCP_GPU_DENSE.md`). Target under 5 minutes to a measured twin, splat trained in the background. | 2 weeks. |
| No exteriors | Medium for features, low for commercials | Out of scope until Phase 3. Say so. | n/a |
| Ceilings unknown from video | Low | Capture guidance in the app ("tilt up once"); LiDAR path recovers it. | Comes with the app. |
| Splats are visual, not measurement; the measured cloud is what matters | Low, already handled (splat points are flagged inferred and excluded from measurement, fixed September 2026) | Keep the separation; it is a selling point. | none |
| Google-only model dependency | Low | Swap the agent to a model-agnostic layer; keep Gemini where it is good (vision review is good). | 1 week. |
| ClickHouse as a hard dependency at tiny scale | Low | Keep it. It is the index and the hosted deploy already bakes it into one container. | none |

### 8.2 Market risks

| Risk | Assessment |
|---|---|
| Core industry contracting (LA down 16 percent, commercials down 18 to 23 percent in late 2025) | Real. Mitigation is geographic (UK up 22 percent, [Q1 2026 LA commercials up 35 percent quarter-on-quarter](https://filmla.com/encouraging-signs-for-la-area-filming-as-select-production-categories-begin-trending-upward/) off the Q4 low) and positional: in a contraction the pitch is survival, not convenience. |
| Beachhead is small ($20 to 30 million) | Real. This is a wedge market and must be presented as one. The application leads with the index. |
| Location managers see a threat to their day rates | Real for features, weaker for commercials where scouts are freelance and compete on deliverables. Position as the scout's tool. Never say "replace the scout." Say "scout six, drive to two." |
| "The director needs to feel the room" | Partly true and permanent. Locaish does not replace the final visit; it chooses which room gets the final visit. Make this the explicit promise. |
| A capture platform (Niantic Spatial, Polycam) adds a film layer | Possible within 24 months. Defence is the craft compiler plus the index plus being first with the customers. Move fast on Phase 1. |
| Privacy and consent for scanned private homes | Real and under-discussed. Twins of private homes are sensitive data. Build consent capture into the app from day one, give hosts deletion rights, never train on twins without explicit opt-in. This is also a sales advantage with commissions and marketplaces. |

### 8.3 Founder and company risks

- **Solo founder.** YC's acceptance rate is [around 1 to 2 percent](https://www.joinleland.com/library/a/yc-acceptance-rate) and solo technical founders without market traction are the most common rejection. The fix is traction (three real jobs in 14 days is traction) and an honest cofounder plan.
- **No stated film-industry network.** Unknown from here. If none exists, Phase 0's twenty conversations are also the start of building one, and the hackathon win is the opening line.
- **Hackathon code is not product code.** 31,000 lines built in 17 days by one person with an agent is impressive and brittle in the places that were not demoed. Budget a month of hardening in Phase 1 and do not promise uptime until it is done.

### 8.4 Scorecard

| Dimension | Score (10) | Why |
|---|---|---|
| Problem severity | 7 | Real, daily, expensive, unquantified by the industry itself, which cuts both ways |
| Solution fit | 8 | Measures the exact things a tech scout measures; the honesty layer is unusual and valuable |
| Technical differentiation | 8 | Craft compiler, independent-anchor scale, shot index. Hard to copy quickly |
| Market size (beachhead) | 4 | $20 to 30 million at saturation |
| Market size (mission) | 7 | Matterport-scale comparable, plausible ladder, large adjacent markets |
| Timing | 7 | LiDAR phones, splat standards ratified in 2026, contraction pressure, remote directing normalised, agents able to do the reasoning |
| Go-to-market reachability | 8 | Listed buyers, free channel through scouts, concentrated trade press |
| Competition | 7 | Nobody does this; several could try; none has the customer and the tech together |
| Team | 4 | Solo, technical, proven builder, unknown industry access |
| Current traction | 3 | A hackathon win, a live demo, zero customers |
| **Overall** | **6.5** | **Fundable at pre-seed with three live customers; a YC candidate with ten** |

### 8.4b The bottleneck, researched (21 October 2026)

Five independent research sweeps and a bull-and-bear stress test (full memo with every source: `docs/BOTTLENECK_RESEARCH.md`) changed four things in this plan:

- **The bottleneck is a sale nobody has made yet, not a feature.** No source shows any production company or scout paying for a scan-derived location package. Every comparable got its first customers through founder conversations (SuperScout: 2 subscriptions from ~400 conversations; Filmustage: 350 paid of 4,400 registered after two years and $550K). The next unit of progress is a conversation.
- **The accuracy bar is about 3 cm**, which departments "treat as truth"; plain video at ±10 to 20 percent cannot answer the dolly-through-the-door question (8.9 cm of margin for a Fisher 10 on a 30-inch door) and a Polycam or Scaniverse LiDAR export (4 to 10 cm in independent tests) can. The LiDAR export is the metric product; video is the demo. The ARKit app remains the fix that removes the dependency on someone else's app.
- **The beachhead is smaller and shrinking faster than assumed.** The AICP directory shows at least 262 general members, not 400; FilmLA Q2 2026 commercial shoot days were down 21.5 percent year on year. The sale is "fewer return trips and a measured package", never "a replaced scout": every practitioner source says the final visit stays.
- **The first ten customers** are location managers who already LiDAR-scan rooms for the art department, location agencies and libraries that already sell 360 recces (UK Locations, SCENA, EXT), and one film office, bought as $29 packages, $249 jobs and $350 capture batches. Ten of them is roughly $2,500 to $4,000 a month (**estimate**); $10,000 needs about eight Production regulars plus ten capture jobs plus two agency batches a month.

### 8.5 What would make me wrong

- If, after twenty conversations, heads of production say scouting cost is noise in the budget and the wrong-pick problem "never happens to us," the wedge is wrong and the better entry is marketplaces (Phase 3 first).
- If scouts refuse to capture video on a job because clients forbid it or because it adds ten minutes they are not paid for, the capture workforce assumption is wrong and the capture service becomes the product for longer.
- If the ARKit capture app does not get video twins to under 3 percent dimensional error on real rooms, the product is a viewfinder, not a tech scout, and the price drops by 5x.

---

## 9. The YC application, drafted

Deadline 2 November 2026, 8pm PT. Decisions by 11 December. Batch runs January to March 2027 in San Francisco. Projected Spring 2027 deadline is [February 2027](https://www.roundfunded.com/en/blogs/yc-batches-2026-dates-acceptance-rate) if this round misses.

**Describe what your company does in 50 characters or less.**
Phone video of a room → measured 3D twin → where the camera goes.

**What is your company going to make?**
Locaish turns a phone walkthrough of a room into a metric 3D twin and tells filmmakers what a camera can do in it: sun through every window by the hour, which dollies and jibs fit, every physically possible camera position scored and searchable, and a scene's coverage planned and rendered before anyone drives out. The twin grades its own accuracy and refuses to guess. We sell it to commercial production companies and their location scouts, who today choose locations from photos and then pay a van full of department heads to find out if the photos lied. Every room we process joins an index, and the company we are building is the index of every filmable space.

**Why did you pick this idea to work on?**
I built it in 17 days for Agentic Cinema and it won its track. The part that surprised me was not the reconstruction but how much of cinematography is measurable: the 180-degree line is a cross product, a matched reverse is a lens and a distance, a window key is an angle. Once those are columns, choosing a location is a query. Nobody had made it one.

**Who are your competitors, and what do you understand about your business that they don't?**
SuperScout and Reel-Scout manage photo libraries for scouts and film commissions. Giggster and Peerspace list spaces with photos. Matterport makes twins for real estate. Polycam, Scaniverse and Luma make scans without a film layer. Scanning bureaus charge £3,000 to £8,000 per location for VFX-grade LiDAR. What we understand: the decision is made during the bid, before the money exists, by a director who is usually in another city, and the only thing that changes that decision is a number you can defend. Pictures do not change it and survey-grade scans are a hundred times too expensive for it.

**How do or will you make money? How much could you make?**
Subscriptions to production companies ($499 a month) and scouts ($49 a month) with per-twin processing, a capture service in year one, and per-listing twins for marketplaces and commissions from year two. The commercial beachhead in the US and UK is $20 to 30 million a year. The index across marketplaces, commissions and studios is the Matterport comparable: 12 million spaces, $170 million revenue, acquired for $1.6 billion by the company that owned the industry's data.

**How far along are you?**
Live hosted product: scanned rooms, a working agent, a shot index, and the three things a production company buys -- a location package per room with the sun on the shoot date drawn on the floor plan, a project view that ranks candidate rooms against the shot list, and share links a director opens on a phone. Private passcode deployments for design partners and a pricing page that takes pilots. [Fill in on 2 November: N rooms captured on real jobs for N companies, N of which replaced a physical visit, N paying.]

**Progress in the next 14 days to fill the blanks above:** three live job captures, five completed calls, one letter of intent.

---

## 10. Ninety-day plan, week by week

| Week | Milestone |
|---|---|
| 1 (20 Oct) | **Done 20 Oct:** README rewritten for producers; location package, projects with room ranking, share links, phone capture page, passcode deployments, pricing and pilot intake built and tested (Section 1.4). **Still to do this week:** 40 names pulled, 40 emails sent with a share link to a real package. |
| 2 (27 Oct) | 5 calls done. 3 real-job videos received and processed. YC application submitted 2 November. |
| 3 to 4 | ARKit capture app spec. Cloud GPU worker. Model-agnostic agent layer. Design-partner offer sent to everyone who took a call. |
| 5 to 8 | Capture app alpha in the hands of 3 scouts. 10 design partners signed. First capture-service job delivered for cash. |
| 9 to 10 | Report PDF and share link polished. First "we didn't drive out" story written up with a dollar figure. YC interview, if it comes, is in these weeks. |
| 11 to 12 | Charge the first three partners. Company library and shot search scoped from what partners asked for. Decide: YC batch, or raise a $500,000 to $1 million pre-seed on the same deck from angels with production backgrounds. |
| 13 (mid-January 2027) | 30 twins on real jobs, 3 paying, one case study, one London partner, cofounder conversation resolved either way. |

---

## 11. Sources

Market and industry
- [FilmLA Q4 and full-year 2025 via Deadline](https://deadline.com/2026/01/los-angeles-film-tv-on-location-production-2025-q4-filmla-1236684296/), [Q3 2025 via CineD](https://www.cined.com/la-film-production-declines-13-2-in-q3-2025-incentive-backed-projects-signal-hope/), [Q1 2026 FilmLA](https://filmla.com/encouraging-signs-for-la-area-filming-as-select-production-categories-begin-trending-upward/)
- [BFI 2025 statistics, £6.8 billion UK spend](https://www.bfi.org.uk/news/official-bfi-statistics-2025)
- [ProdPro via TheWrap, US spend down 20 percent in 2025](https://www.thewrap.com/industry-news/industry-trends/us-film-production-2026/)
- [Film location rental market, $4.2 billion 2025 (Dataintelo)](https://dataintelo.com/report/film-location-rental-market)
- [Virtual production market, $3.7 billion 2026 (GlobeNewswire)](https://www.globenewswire.com/news-release/2026/10/14/3380872/0/en/virtual-production-market-research-company-expansions-and-latest-industry-news-global-market-outlook-led-studio-investments-and-ai-driven-production-trends.html); [200 LED volumes, Futuresource](https://www.futuresource-consulting.com/the-source/industry-pulse/led-volume-making-waves-in-virtual-production/)
- [Commercial cost figures (vendor blog citing a 2025 survey, unverified)](https://greenfroglabs.com/blog/30-second-commercial-cost); [AICP directory](https://aicp.com/membership-information/member-directory/general-member-directory)

Rates and costs
- [Location manager day rates (Camera Crew)](https://getcameracrew.com/resources/learn/what-is-a-location-manager); [location manager weekly (Saturation)](https://saturation.io/film-crew-positions/location-manager); [location scout (Saturation)](https://saturation.io/film-crew-positions/location-scout)
- [AICP and Teamsters Local 399 location managers agreement 2023 to 2026 (official rates)](https://www.ht399.org/app/uploads/2024/12/AICP-Local-399-Locations-Managers-2023-2026-CBA.final_.pdf); [about 800 LA location managers](https://www.yahoo.com/entertainment/hollywood-teamsters-local-399-reaches-183502334.html)
- [Tech scout attendees (Wrapbook)](https://www.wrapbook.com/blog/tech-scout); [No Film School](https://nofilmschool.com/what-is-a-tech-scout)
- [LiDAR set scanning costs (Visual Skies)](https://visualskies.com/3d-scanning-for-film-and-tv/how-much-does-it-cost-to-lidar-scan-a-film-set-v3/)
- [Real-estate 3D tour pricing (Travvir)](https://travvir.com/blog/en/real-estate-virtual-tour-cost)

Competitors and comparables
- [Matterport acquisition closed, $1.6 billion, $169.7 million 2024 revenue (Inman)](https://www.inman.com/2025/02/28/costar-closes-1-6b-matterport-acquisition-ahead-of-schedule/)
- [Polycam Series A](https://radiancefields.com/polycam-announces-18m-series-a); [Niantic Spatial](https://en.wikipedia.org/wiki/Niantic_Spatial)
- [SuperScout team page](https://superscout.ai/team/); [Reel-Scout clients](https://www.reel-scout.com/clients)
- [Giggster and Peerspace listing counts](https://www.eventsinminutes.com/blog/best-peerspace-alternatives-2026/); [Peerspace](https://en.wikipedia.org/wiki/Peerspace); [marketplace comparison](https://setvenue.com/blog/setvenue-vs-giggster-vs-peerspace-comparison)
- [AFCI, 360+ commissions](https://en.wikipedia.org/wiki/Association_of_Film_Commissioners_International)
- [Previs tool pricing](https://mstudio.ai/insights/best-previs-software-2026); [Cine Tracer](https://www.cined.com/cine-tracer-cineamtography-simulator-created-unreal-engine/)
- [Gaussian splat standards 2026 (OpenUSD, Khronos)](https://radiancefields.com/gaussian-splatting-statistics); [splats in virtual production (SIGGRAPH)](https://blog.siggraph.org/2026/07/gaussian-splatting-in-virtual-production-how-we-put-a-human-sized-can-into-a-childhood-bedroom.html/)
- [YC Martini](https://www.ycombinator.com/companies/martini); [Higgsfield valuation (Fortune)](https://fortune.com/2026/09/30/cio-intelligence-sept-30/)

Practitioner accounts of the problem
- [John August on location sign-off](https://johnaugust.com/?p=806); [MINDCASTLE scouting (Musicbed)](https://musicbed.com/blog/?p=2866); [Zacuto, nine things to consider](https://zacuto.com/blogs/blog/9-things-to-consider-when-scouting-a-movie-location)

YC
- [Winter 2027 deadline, 2 November 2026 (YC on X)](https://x.com/ycombinator/status/2111183078072979886); [batch dates](https://www.roundfunded.com/en/blogs/yc-batches-2026-dates-acceptance-rate); [Requests for Startups, Fall 2026](https://www.ycombinator.com/rfs); [acceptance rate](https://www.joinleland.com/library/a/yc-acceptance-rate); [solo founders](https://www.wisp.blog/blog/does-y-combinator-accept-solo-founders-heres-what-you-need-to-know)
