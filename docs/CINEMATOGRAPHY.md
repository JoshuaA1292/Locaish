# What the planner knows about pictures — and where it learned it

Locaish's coverage planner is not a scoring heuristic dreamed up in a
notebook. Each rule below is a working convention of narrative
cinematography and location scouting, grounded in the standard texts of the
craft (Brown's *Cinematography: Theory and Practice*; Katz's *Film Directing
Shot by Shot*; Mascelli's *The Five C's of Cinematography*; Bowen's *Grammar
of the Shot*; Block's *The Visual Story*; Alton's *Painting with Light*; the
*American Cinematographer Manual*), with a linked source kept wherever a
phrase is quoted, and then the *measurement* the twin makes to apply it. Every rule reduces to a column in
the `shot_setups` table (`locaish/film/sweep.py`) or a predicate in the
planner (`locaish/film/coverage.py`); the Gemini agent is told the rules in
the same words (`locaish/agent/coverage.py`) and can overrule the ranking
for a stated reason, after looking at the frame.

## 1. Coverage: what a dialogue scene needs

*Convention.* Standard coverage of a two-person scene is a master, a pair of
over-the-shoulders and a pair of singles, intercut; the master carries the
scene's geography and the singles carry its emotion, so the singles want the
closer camera.
— Katz, *Film Directing Shot by Shot* (the dialogue-staging chapters);
Mascelli, *The Five C's of Cinematography* ("Camera Angles");
[Sony Cine: The Cinematographer and Scene Blocking](https://sony-cinematography.com/articles/the-cinematographer-and-scene-blocking-part-2/).

*In Locaish.* The breakdown agent is instructed to design exactly this when
handed prose: master, OTS pair, singles at matching sizes, inserts for what
the text singles out. A typed shot list is honoured as written.

## 2. The 180-degree line

*Convention.* Draw a line through the two characters; keep the camera on one
side of it, or the cut flips who is looking left and who is looking right.
— Katz, *Film Directing Shot by Shot* (continuity and the line of action);
Brown, *Cinematography: Theory and Practice*, 3rd ed. (screen direction);
Mascelli, *The Five C's* ("Continuity").

*In Locaish.* `PlanContext.line` is the line through the two marks; the
first placed shot on either character fixes `line_side` from the sign of
the cross product `(B−A) × (cam−A)`. Every later shot on those characters
carries the predicate `sign(...) = side` — in SQL, so ClickHouse enforces
it. It is the *last* thing the planner relaxes, and the shot list says
"crossed the line of action" when it had to.

## 3. Reverses that cut

*Convention.* Compose the two sides of a conversation with "the same shot
size, distance from the subject, focal length, horizon and depth of field"
and put both cameras "at a similar distance from the axis"; a standard or
medium-telephoto lens connects the eyelines better than a wide.
— [Backstage: Shot/Reverse Shot](https://www.backstage.com/magazine/article/what-is-shot-reverse-shot-film-examples-75550/) (the quoted checklist);
Brown, *Cinematography: Theory and Practice* (shooting dialogue);
Katz, *Shot by Shot* (matching reverses).

*In Locaish.* When a single of A at size *s* has been placed and the planner
reaches a single of B at size *s*, two soft predicates are added: `focal_mm
= f` and `|distance − d| ≤ 20 %`. They are the first preferences dropped if
the room refuses, and the card says "matches the reverse's lens and
distance" when they held.

## 4. Over the shoulder

*Convention.* An OTS is a medium close-up on one actor with part of the
other's shoulder in the near foreground; it is shot as a pair.
— Katz, *Shot by Shot* (the A/B dialogue patterns);
Bowen, *Grammar of the Shot*, 4th ed. (shot types).

*In Locaish.* `Shot.ots`: the camera stands 0.45–1.6 m from the foreground
actor's mark and nearer to it than to the subject, with the foreground mark
inside the full half-field-of-view (edge of frame). Pure geometry, in SQL.

## 5. Faces and distance

*Convention.* Perspective on a face is set by camera-to-subject distance,
not focal length: the same close-up from a metre and from three metres
renders the features differently, and the near one enlarges the nose. Rules
of thumb put a flattering distance at 8 ft and more.
— [American Cinematographer: Understanding Lens Distortion](https://theasc.com/article/understanding-lens-distortion/);
Brown, *Cinematography: Theory and Practice* (lens perspective and the
close-up); *American Cinematographer Manual*, 11th ed. (optics and
depth-of-field tables).

*In Locaish.* `portrait_ok` = 1 for tight framings (ECU–MCU) only when
`distance_m ≥ 1.4`. It is 30 % of the tight-shot ranking, so the planner
reaches for the 50 from farther back before the 25 from close. In a room
with no farther-back position it says so on the card.

## 6. Windows are the key light

*Convention.* "The main thing to look for in an interior are windows and
knowing what direction they are facing." A key at ~45° to the camera is
the standard interview setup; short-side key (light from the side away from
camera) gives dimension, front light is flat, a window behind the subject is
a silhouette unless you want one.
— [StudioBinder: Location Scouting Checklist](https://www.studiobinder.com/blog/ultimate-location-scouting-checklist-for-producers-and-ads/) (the quoted scouting advice);
Brown, *Cinematography: Theory and Practice* (lighting direction and key
placement); Alton, *Painting with Light* (single-source interiors);
[Neil Oseman (BSC-listed DP): Introduction to Short Key Lighting](https://neiloseman.com/introduction-short-key-lighting/).

*In Locaish.* `key_angle_deg` is the angle at the subject between the
camera and the largest-looking window; `key_quality` names the band
(front < 25° < three-quarter < 70° < side < 110° < rim < 150° < back). The
tight-shot ranking weights three-quarter 1.0, side 0.8, rim 0.6, front 0.3,
back 0 — inverted when the brief asks for a silhouette. `window_behind_subject`
remains the hard flag for "you are shooting into the glass".
`sun_schedule` (ephemeris, `film/daylight.py`) says *when* that glass is hot.

## 7. Depth, and shooting into corners

*Convention.* The longest sightline in a rectangular room runs into a
corner; shooting corner-to-corner adds roughly 20 % of perceived depth,
while "shooting straight across the narrow axis of a room makes it feel
confined." Pull the subject off the wall to push the background back.
— [Peek at This: Small Spaces, Cinematic Results](https://peekatthis.com/shooting-in-small-spaces-tips-limited-locations/) (the quoted 20% figure);
Block, *The Visual Story*, 2nd ed. (deep space and depth cues);
Katz, *Shot by Shot* (staging in depth).

*In Locaish.* `background_depth_m` marches the lens axis on past the
subject through the twin's occupancy grid to the first surface (12 m means
it left the capture). `axis_wall_angle_deg` is the angle between the lens
axis and the footprint wall it ends on: 0 = square onto a wall, 45 = into
the corner. Wides weight depth 0.5 and corner 0.3.

## 8. Room to work

*Convention.* Tech-scout checklists ask for "space for camera movement,
depth for wide shots, angles and vantage points" and whether the room holds
the planned dolly moves.
— [No Film School: Tech Scout Guide](https://nofilmschool.com/tech-scout),
[Wrapbook: Nail Your Next Tech Scout](https://www.wrapbook.com/blog/tech-scout).

*In Locaish.* `backup_room_m` marches backwards from the camera at lens
height to the first obstruction; `clearance_m` and `headroom_m` were
already in the sweep; `check_dolly_move` simulates the track.

## 9. Height and mood

*Convention.* Eye level is neutral; a low angle makes a character powerful,
threatening or mythic; a high angle makes them small, fragile or judged.
— Katz, *Shot by Shot* (camera height and point of view);
Mascelli, *The Five C's* ("Camera Angles");
Brown, *Cinematography: Theory and Practice* (the frame and power).

*In Locaish.* The breakdown maps mood to height; the shot-list parser reads
"looms / dominant / menacing" as low and "vulnerable / cornered / small" as
high. The sweep's heights are 0.45, 1.05 and 1.55 m; "high" resolves to the
top of that and the card says so.

## 10. What is left to the eye

Headroom, look room (nose room) and the actual content of the background
are compositional judgements the table cannot hold.
— Bowen, *Grammar of the Shot* (composition: headroom and look room);
[Neil Oseman: Lead Room, Nose Room or Looking Space](https://neiloseman.com/lead-room-nose-room-or-looking-space/).

*In Locaish.* Every placed setup is rendered from the gaussian field and
shown to Gemini with the shot brief; its verdict (keep / adjust / reject,
with a suggested height or lens) can send the agent back to the table. The
director's-viewfinder practice this mirrors — framing a lens on a scout
with Artemis or Cadrage and exporting the shot list — is what the studio's
viewfinder mode reproduces inside the twin.
— [CineD: Cadrage](https://www.cined.com/cadrage-the-better-directors-viewfinder-app/),
[Bill Zarchy: Artemis Director's Viewfinder](https://billzarchy.com/blog/production-apptitude-artemis-directors-viewfinder/).
