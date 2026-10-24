# What stands between Locaish and $10,000 a month

*21 October 2026. Five independent research sweeps (demand, comparables, accuracy and turnaround, semantic labelling, distribution) over about 450 web lookups, then a bull case and a bear case argued from that evidence alone. Everything below carries its source; anything computed is marked **estimate**. The gaps the research could not close are listed at the end, because they are the next things to find out by talking to people, not by searching.*

## The verdict

$10,000 a month is reachable, but not the way the plan was written. The constraint is not a missing feature. It is that nobody has yet been shown paying for a scan-derived location package, and the product's central promise, measured dimensions, is only trusted at about 3 cm, which plain phone video cannot deliver and a LiDAR export from a free app can.

The bottleneck, named: **one technical founder hand-selling a measurement nobody has bought before, into a market that is shrinking this year, at a price band above what its closest comparable charges.** Every comparable got its first customers one conversation at a time. The product is now good enough that the next unit of progress is a conversation, not a commit.

## What the evidence says

### Demand is real, indirect, and never framed as "skip the scout"

- A working BC location scout costs a two-day, 15-person scout at about CAD $40,000 against about CAD $4,000 for a 360 tour, and in the same post says practical scouting "will always be the gold standard" ([LMGI Tech Talk](https://locationmanagers.org/tech-talk-disruption-scouting-with-360/)).
- UK Locations added 360 recces to its 300+ property library in 2020 and reports in-person visits per location fell from about 5 to about 3 ([UK Locations](https://www.uklocations.co.uk/journal/ukl-home-of-the-360-virtual-recce/)).
- Matterport's own LocationsHub case study says scans cut candidate visits from 25 to 35 sites to 2 or 3, and concedes on-site scouting will never be fully replaced ([Matterport](https://matterport.com/blog/locationshub-production-scouting-matterport)).
- A peer-reviewed study of 20 directors, producers and DPs found they want sun simulation for the shoot date and hour, lens and camera-move previs, and logistics answers, and still consider one physical visit vital; it reports no willingness-to-pay data ([Bodini et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9471056)).
- December 2025 trade press: location managers scan rooms with Polycam for the art department, Marvel-type productions request LiDAR scans, departments "treat it as truth" at about 3 cm, and "you still have to feel a place, but VR gets you to the shortlist much faster" ([The Location Guide](https://www.thelocationguide.com/spotlight-report/spotlight-report-from-footwork-to-framework-how-tech-is-transforming-location-scouting)).
- No source anywhere shows a production company paying specifically to avoid a tech scout. That is the hole in the plan, and only a customer can fill it.

### The money on the line item is small

A sample AICP-format bid for a $95,000 spot carries a $1,200 DP tech-scout day, a location scout at $500 a day for two days and a location manager at $700 a day: about $2,610 of scouting, roughly 2.7 percent of the budget (**estimate**, [Storiara template](https://storiara.com/templates/commercial-production-budget)). A $499 a month subscription is a fifth of one job's scouting spend before it has saved anything. The sale is "fewer return trips and a measured package", never "a replaced scout".

### What scouts actually pay for software

SuperScout, the closest comparable, charges £20 to £50 a month for scouts and £150 to £250 for film offices ([pricing](https://www.superscoutpro.com/pricing/)), and its founder had spoken to about 400 scouts over four months and sold two subscriptions at MVP stage, with SEO as the main discovery channel ([founder interview, Feb 2025](https://artstechnology.substack.com/p/conversation-with-gabriel-isserlis)). Filmustage needed about two years, a Product Hunt launch and $550K to reach 350 paid users on 4,400 registrations at $49 to $149 a month, which is about $17,000 a month ([TechCrunch](https://techcrunch.com/2023/03/20/filmustage-leverages-ai-to-breakdown-film-scripts-create-shooting-schedules-and-more/)). Team tiers for production SaaS sit at $150 to $400 a month (StudioBinder, Yamdu). Matterport's paid ARPU is about $112 a month (**estimate** from its [FY2024 10-K](https://www.sec.gov/Archives/edgar/data/1819394/000181939425000007/mttr-20241231.htm)); Polycam had nearly 100,000 paying customers in February 2024 at $150 to $1,200 a year ([TechCrunch](https://techcrunch.com/2024/02/07/3d-scanning-app-polycam-gets-backing-from-youtube-co-founder)).

### The accuracy bar, and where Locaish sits

| Method | Room-scale error, independent tests |
|---|---|
| iPhone LiDAR, Polycam | 5 cm mean, 10 cm RMSE in a 200 m² lab ([HCU Hamburg 2023](https://www.repos.hcu-hamburg.de/bitstream/hcu/955/1/geomatics-03-00030.pdf)) |
| iPhone LiDAR, Scaniverse | 4 to 5 cm RMSE on a 6 × 13 m classroom ([Yanchuk 2026](https://geodetski-vestnik.com/arhiv/70/1/47_Yanchuk.pdf)); 44 cm mean in the Hamburg test when tracking failed |
| Gaussian splat from video with a physical scale reference | 1.3 to 3.5 cm mean ([Plain Concepts, vendor-side](https://www.plainconcepts.com/digital-twins-3d-gaussian-splatting/)) |
| Monocular metric depth (MoGe-2) | 4.4 to 13.6 percent relative ([paper](https://arxiv.org/html/2507.02546v1)) |
| Locaish, plain video today | ±10 to 20 percent scale band (our own QA) |
| What departments trust | about 3 cm ([Location Guide](https://www.thelocationguide.com/spotlight-report/spotlight-report-from-footwork-to-framework-how-tech-is-transforming-location-scouting)); a Fisher 10 through a 30-inch door has 8.9 cm to spare ([Expressway Cine](https://blog.expresswaycine.com/new-product-fisher-10-dolly/)) |

The dolly-fit promise is only keepable from a LiDAR export or a scale reference. That path already exists in the product; the plan has to say it is the product, and the video path is the demo.

### The semantic phase is cheap and defensible, not a bottleneck

Polycam, Scaniverse and Luma ship no object semantics as of October 2026; Matterport's Property Intelligence and Tags are the only precedent ([Matterport](https://matterport.com/blog/harnessing-the-power-of-ai-automated-measurements-and-layouts-now-available), [Tags](https://support.matterport.com/s/article/Create-a-Mattertag-Post?language=en_US)). Gemini as a detector is YOLOv4-class (mAP 0.41 on COCO for 3 Pro; 3.6 Flash regressed below 3.5 Flash in [Roboflow's July 2026 evals](https://roboflow.com/blog/gemini-3-6-flash-for-vision), per [this benchmark](https://simedw.com/2025/07/10/gemini-bounding-boxes/)), at about $0.002 a frame, which is why the click-then-type correction loop is the right design and why the detector now defaults to 3.5 Flash. Commercially usable 3D alternatives exist for later (SAM 3 at $0.005 a request, SAGA click-to-segment under Apache-2.0); LangSplat and Boxer are non-commercial.

### The market is contracting and the buyer list is shorter than assumed

FilmLA counted 543 commercial shoot days in Q2 2026, down 21.5 percent year on year and 36 percent below the five-year average ([TheWrap](https://www.thewrap.com/creative-content/movies/filmla-production-decline-q2-2026/)); UK film and HETV spend fell 15.6 percent in H1 2026 ([Screen Daily](https://www.screendaily.com/news/total-uk-feature-production-spend-drops-6-in-first-half-of-2026-but-local-films-and-co-pros-enjoy-boosts/5219283.article)). The AICP general-member directory shows at least 262 companies, not the 400 the plan assumed. Giggster and Peerspace have no public API or partner programme. The indie segment budgets $1,500 for all location scouting on a $102,000 feature ([No Film School](https://nofilmschool.com/example-line-budget-independent-film)) and will not carry a subscription.

## The arithmetic for $10,000 a month (all **estimates**)

| Mix | Customers |
|---|---|
| Production only | 21 at $499, about 6 to 8 percent of AICP general members |
| Scout only | 345 at $29, more than a tenth of the estimated 1,500 to 3,500 US and UK location professionals |
| Per-job | 40 jobs a month at $249 |
| Capture service | 29 locations a month at $350 |
| Blend the evidence supports | 8 Production or per-job regulars (~$3,000) + 10 capture jobs (~$3,500) + 2 location agencies or film offices on capture batches (~$3,500) |

## What changed in the product and the plan because of this

1. **Pricing.** Scout drops to $29 a month (SuperScout's band is £20 to £50) with a $29 single package. Production is offered per job at $249, billed to the job the way the scout is, with $499 a month for unlimited jobs. Capture service stays at $350.
2. **The first ten customers** are not AICP heads of production cold. They are location managers who already Polycam-scan rooms (the Location Guide archetype), location agencies and libraries that already sell 360 recces (UK Locations, SCENA, EXT), and one film office, all reached in person and by direct email, because no comparable found a channel that worked at this stage.
3. **The LiDAR export is the product's metric path**; video is the zero-friction demo. The capture page and the package already say so; the pitch must too.
4. **The detector** defaults to Gemini 3.5 Flash with the correction loop as the design, not an afterthought.

## What only conversations can answer

- Whether any production company or agency will put $249 or $350 on a job against the $2,610 of scouting lines in its own bid.
- Whether LiDAR-export twins hit under 5 cm on real customer rooms (measure one wall per delivered package and keep the log).
- Conversion from direct outreach: SuperScout's 2 paid from about 400 conversations is the floor to beat; Filmustage's 8 percent of registrations is the ceiling seen.
- Whether productions pay scouts a premium for scans, or merely request them.
- FilmLA Q3 and Q4 2026 commercial shoot days.

## What the research could not retrieve

Reddit was blocked to the crawler, so no first-hand scout threads on paying for tools. SuperScout's live customer count, StudioBinder's official pricing, an LMGI member count, any production-software sales-cycle data, UK commercials statistics (APA), and any survey of indie location budgets. Each is a question for a person, not a search.
