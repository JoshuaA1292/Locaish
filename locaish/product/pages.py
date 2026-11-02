"""The product's pages beyond the room: projects, pricing, capture, login.

Each is a self-contained HTML string in the studio's visual language (black,
white, one hairline, no accent but the primary action), rendered server-side with the data inlined so the page
works the moment it arrives. The project page is the one a head of
production lives in during a bid; the capture page is the one a scout opens
on a phone standing in the room; pricing is the one a founder sends.
"""

from __future__ import annotations

import html
import json
from typing import Any

MARK = ("<svg class='mark' viewBox='0 0 16 16' width='14' height='14' aria-hidden='true'>"
        "<rect x='1' y='1' width='14' height='14' rx='3.5' fill='none' stroke='currentColor' stroke-width='1.5'/>"
        "<circle cx='8' cy='8' r='2.6' fill='currentColor'/></svg>")

# The same system as the studio page: black, white, two greys, one hairline,
# white for the one primary action, a warm tone only where a render is named.
_SHELL_CSS = """
:root { --bg:#000000; --surface:#0a0a0a; --surface2:#111111; --line:#1f1f1f; --line2:#2a2a2a; --ink:#ffffff;
        --dim:#9a9a9a; --mute:#5a5a5a; --accent:#ffffff; --accent-ink:#000000; --ok:#ffffff; --warn:#9a9a9a; --bad:#ffffff;
        --sun:#ffffff; --render:#d9a86c; --r:12px; --r2:8px; --r3:6px; --gap:64px; --pad:24px; --measure:560px; --col:1280px; --edge:40px;
        --sans:-apple-system, "Helvetica Neue", Inter, sans-serif;
        --ui:system-ui, -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
        --serif:ui-serif, "Iowan Old Style", Charter, Georgia, serif;
        --mono:ui-monospace, "SF Mono", Menlo, monospace; }
* { box-sizing:border-box; } html,body { height:100%; }
body { margin:0; background:var(--bg); color:var(--ink); -webkit-font-smoothing:antialiased; font:15px/1.55 var(--ui); }
a { color:inherit; } .hidden { display:none !important; }
b, strong { font-weight:500; }
main { max-width:var(--col); margin:0 auto; padding:0 var(--edge) 120px; }
#top { position:sticky; top:0; z-index:5; display:flex; align-items:center; gap:24px; height:56px; margin:0 calc(-1 * var(--edge)) 48px;
       padding:0 var(--edge); background:var(--bg); border-bottom:1px solid var(--line); }
#top .brand { font:500 15px/1 var(--ui); text-decoration:none; display:inline-flex; align-items:center; gap:8px; }
#top nav { display:flex; gap:24px; } #top nav a { color:var(--dim); text-decoration:none; font-size:13px; }
#top nav a:hover, #top nav a.on { color:var(--ink); }
#top .right { margin-left:auto; display:flex; gap:20px; align-items:center; }
h1 { font:400 36px/1.15 var(--serif); letter-spacing:-.01em; margin:0 0 8px; }
h1[contenteditable] { outline:none; }
h2 { font:500 13px/1.4 var(--ui); letter-spacing:0; text-transform:none; color:var(--dim); margin:0 0 16px; }
h2.mt0 { margin:0; }
p.lead { color:var(--dim); font-size:15px; margin:0 0 40px; max-width:var(--measure); }
.card { background:transparent; border:0; border-top:1px solid var(--line); border-radius:0; padding:24px 0; min-width:0; overflow-wrap:anywhere; }
.btn { display:inline-flex; align-items:center; justify-content:center; background:var(--ink); color:var(--bg); border:1px solid var(--ink); border-radius:var(--r3);
       height:36px; padding:0 16px; font:500 13px/1 var(--ui); cursor:pointer; text-decoration:none; white-space:nowrap; }
.btn:hover { background:#e6e6e6; border-color:#e6e6e6; } .btn:disabled { opacity:.4; cursor:default; }
.btn.quiet { background:transparent; color:var(--ink); border:1px solid var(--ink); }
.btn.quiet:hover { background:var(--surface2); }
.btn.small { height:28px; padding:0 10px; font-size:12px; }
.btn.block { display:flex; width:100%; height:48px; }
input, textarea, select { width:100%; background:var(--bg); color:var(--ink); border:1px solid var(--line2); border-radius:var(--r3); height:36px;
       padding:0 12px; font:400 14px/1.4 var(--ui); font-variant-numeric:tabular-nums; outline:none; max-width:100%; min-width:0; }
textarea { height:auto; min-height:96px; padding:10px 12px; line-height:1.55; resize:vertical; }
input:focus, textarea:focus, select:focus { border-color:var(--ink); }
input::placeholder, textarea::placeholder { color:var(--mute); }
input[type=date]:invalid, input[type=time]:invalid { color:var(--mute); }
input.empty::-webkit-datetime-edit, input.empty::-webkit-datetime-edit-text, input.empty::-webkit-datetime-edit-year-field, input.empty::-webkit-datetime-edit-month-field, input.empty::-webkit-datetime-edit-day-field { color:var(--mute); }
input[type=date]::-webkit-calendar-picker-indicator { filter:invert(1); opacity:.5; }
input[type=checkbox] { accent-color:#fff; width:auto; height:auto; }
label { display:block; color:var(--dim); font-size:13px; margin-bottom:6px; }
label.row { color:var(--dim); cursor:pointer; margin-top:10px; }
label.pick { display:inline; color:var(--ink); cursor:pointer; text-decoration:underline; margin:0; }
.field { margin-bottom:16px; }
.grid2 { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:0 40px; align-items:start; }
.plans { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:0 40px; }
@media (max-width:800px) { .grid2, .plans { grid-template-columns:minmax(0,1fr); } #top nav { display:none; } main { padding:0 20px 96px; } #top { margin:0 -20px 32px; padding:0 20px; } }
.price { font:400 48px/1.1 var(--serif); font-variant-numeric:tabular-nums; letter-spacing:-.01em; margin:0 0 12px; }
.price .muted { font:13px/1.4 var(--ui); color:var(--dim); }
.pill { font:12px/1.4 var(--ui); color:var(--dim); border:0; padding:0; white-space:nowrap; }
.pill.on { color:var(--ink); }
table { border-collapse:collapse; width:100%; font-size:14px; }
th { text-align:left; color:var(--dim); font-weight:500; font-size:13px; letter-spacing:0; text-transform:none; padding:8px 16px 8px 0; border-bottom:1px solid var(--line); }
td { padding:12px 16px 12px 0; border-bottom:1px solid var(--line); vertical-align:top; }
tr:last-child td { border-bottom:0; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; padding-right:0; }
td a, .room .b a { text-decoration:underline; text-underline-offset:3px; text-decoration-color:var(--line2); }
.v { display:inline-block; width:22px; height:22px; line-height:20px; text-align:center; border-radius:var(--r3); font-size:12px; font-weight:500; border:1px solid var(--line2); }
.v.held { background:var(--ink); color:var(--bg); border-color:var(--ink); } .v.relaxed { color:var(--ink); } .v.missed { color:var(--mute); }
.note { color:var(--dim); font-size:13px; } .err { color:var(--ink); }
.muted { color:var(--mute); font-size:13px; }
.foot { max-width:60ch; font-size:14px; line-height:1.6; }
.mt { margin-top:12px; } .mt2 { margin-top:26px; } .narrow { max-width:420px; margin:12vh auto 0; } .wide { max-width:720px; }
.row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; } .row.between { justify-content:space-between; align-items:baseline; }
.scroll { overflow-x:auto; }
pre.note { white-space:pre-wrap; font:inherit; margin:0 0 12px; max-width:var(--measure); }
.plain { text-decoration:none; }
ul.note, ol.note { line-height:1.7; padding-left:20px; }
.rooms { display:grid; grid-template-columns:repeat(auto-fill, minmax(230px, 1fr)); gap:24px; }
.room { display:block; text-decoration:none; background:transparent; border:0; border-radius:0; }
.room img { width:100%; aspect-ratio:16/9; object-fit:cover; display:block; background:var(--surface); }
.room .b { display:block; padding:10px 0 0; } .room b { display:block; font-size:15px; font-weight:500; } .room span { color:var(--dim); font-size:13px; }
.room .b.links { display:flex; gap:12px; padding-top:4px; font-size:13px; }
.room .b a { color:var(--dim); } .room .b a:hover { color:var(--ink); }
.prog .track { height:1px; background:var(--line); margin-top:8px; overflow:hidden; }
.prog .track i { display:block; height:100%; width:0; background:var(--ink); transition:width .5s; }
.toast { position:fixed; bottom:22px; left:50%; transform:translateX(-50%); background:var(--surface2); border:1px solid var(--line2);
         border-radius:var(--r3); padding:9px 14px; font-size:13px; }
"""


# An empty date input shows its mask in the placeholder grey, not as a value.
_DATE_JS = ("(function(){var f=function(i){i.classList.toggle('empty',!i.value);};"
            "var all=function(){document.querySelectorAll('input[type=date],input[type=time]').forEach(f);};"
            "['input','change'].forEach(function(ev){document.addEventListener(ev,function(e){var i=e.target;"
            "if(i&&(i.type==='date'||i.type==='time'))f(i);});});all();})();")


def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def shell(title: str, body: str, *, nav: str = "", script: str = "", extra_css: str = "") -> str:
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_e(title)}</title><link rel='icon' type='image/svg+xml' href='/favicon.svg'>"
        f"<style>{_SHELL_CSS}{extra_css}</style></head><body><main>"
        f"<div id='top'><a class='brand' href='/'>{MARK}Locaish</a>"
        f"<nav><a href='/' {'class=on' if nav == 'rooms' else ''}>Locations</a>"
        f"<a href='/projects' {'class=on' if nav == 'projects' else ''}>Projects</a>"
        f"<a href='/capture' {'class=on' if nav == 'capture' else ''}>Capture</a>"
        f"<a href='/pricing' {'class=on' if nav == 'pricing' else ''}>Pricing</a></nav>"
        "<div class='right'></div></div>"
        f"{body}</main><script>{_DATE_JS}{script}</script></body></html>"
    )


# ---------------------------------------------------------------------------
# projects
# ---------------------------------------------------------------------------


def projects_page(projects: list[dict], rooms: list[dict]) -> str:
    rows = "".join(
        f"<tr><td><a href='/project/{_e(p['id'])}' class='plain'>{_e(p['title'])}</a>"
        f"<div class='muted'>{_e(p.get('client') or '')}</div></td>"
        f"<td>{_e(p.get('shoot_date') or '—')}</td><td class='num'>{len(p.get('locations') or [])}</td>"
        f"<td class='num'>{_e((p.get('updated_at') or '')[:10])}</td></tr>"
        for p in projects
    ) or "<tr><td colspan='4' class='note'>No projects yet. A project is one job: the brief, the shoot date, and every room you are considering.</td></tr>"
    body = f"""
<h1>Projects</h1>
<p class="lead">One job, one brief, one shoot date, every candidate room. Locaish tells you which rooms hold the scene before anyone drives out.</p>
<div class="grid2">
  <div class="card">
    <h2>Start a project</h2>
    <form id="newp">
      <div class="field"><label>Job</label><input name="title" placeholder="Acme spring campaign — kitchen spot" required></div>
      <div class="grid2">
        <div class="field"><label>Client / agency</label><input name="client" placeholder="optional"></div>
        <div class="field"><label>Shoot date</label><input name="shoot_date" type="date"></div>
      </div>
      <button class="btn" type="submit">Create project</button>
    </form>
  </div>
  <div class="card">
    <h2>Open projects</h2>
    <table><tr><th>Job</th><th>Shoot</th><th class="num">Rooms</th><th class="num">Updated</th></tr>{rows}</table>
  </div>
</div>
"""
    script = """
document.getElementById('newp').onsubmit = e => {
  e.preventDefault();
  const f = e.target, body = {title: f.title.value, client: f.client.value, shoot_date: f.shoot_date.value};
  fetch('/projects', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)})
    .then(r => r.json()).then(r => { if (r.id) location.href = '/project/' + r.id; else alert(r.error || 'could not create'); });
};
"""
    return shell("Projects · Locaish", body, nav="projects", script=script)


def project_page(project: dict, rooms: list[dict], *, share_base: str = "") -> str:
    """The bid page. `rooms` are every finished job (id, name, summary)."""
    data = json.dumps({"project": project, "rooms": rooms}).replace("</", "<\\/")
    body = f"""
<div id="phead"></div>
<div class="grid2">
  <div class="card">
    <h2>The scene</h2>
    <div class="field"><label>Shot list, one per line (or paste the scene; the deterministic compare reads shot lines)</label>
      <textarea id="brief" rows="7" placeholder="1. Master wide: MAYA at the counter, JON enters&#10;2. Over JON's shoulder onto MAYA, medium close-up&#10;3. Close-up MAYA, no window behind"></textarea></div>
    <div class="grid2">
      <div class="field"><label>Shoot date</label><input id="shoot_date" type="date"></div>
      <div class="field"><label>Client / agency</label><input id="client"></div>
    </div>
    <div class="field"><label>Notes</label><textarea id="notes" rows="2"></textarea></div>
    <div class="row">
      <button class="btn quiet" id="save">Save</button>
      <button class="btn" id="compare">Compare the rooms</button>
      <span class="note" id="status"></span>
    </div>
  </div>
  <div class="card">
    <h2>Candidate rooms</h2>
    <div id="cands"></div>
    <div class="row mt">
      <select id="addroom"></select><button class="btn quiet" id="addbtn">Add</button>
    </div>
    <p class="muted mt">Every finished scan in this studio can be a candidate. Scan more from the <a href="/">Locations</a> page or the <a href="/capture">phone capture</a> page.</p>
  </div>
</div>
<section id="fitwrap" class="hidden mt2">
  <div class="card">
    <div class="row between">
      <h2 class="mt0">Which rooms hold the scene</h2>
      <span class="note" id="fitmeta"></span>
    </div>
    <div class="scroll mt"><table id="fit"></table></div>
    <p class="muted mt"><span class="v held">✓</span> holds the shot with a clear sightline &nbsp;
      <span class="v relaxed">~</span> holds it with a compromise (lens, height, or a window behind the subject) &nbsp;
      <span class="v missed">✗</span> no setup in the room frames it. Placed by the deterministic planner over every swept camera position; the same brief gives the same answer every time.</p>
  </div>
</section>
<section class="mt2">
  <div class="card">
    <div class="row between">
      <h2 class="mt0">Send it to the director</h2>
      <button class="btn small quiet" id="sharebtn">Create share link</button>
    </div>
    <p class="note" id="sharenote">A read-only page with the ranking and every room's location package. No account needed to open it; revoke it any time.</p>
  </div>
</section>
"""
    script = """
const D = """ + data + """;
const $ = id => document.getElementById(id);
let P = D.project; const ROOMS = D.rooms; const byId = {}; ROOMS.forEach(r => byId[r.id] = r);
const SHARE_BASE = """ + json.dumps(share_base) + """;
function esc(t){ const d = document.createElement('div'); d.textContent = t == null ? '' : String(t); return d.innerHTML; }
function roomName(id){ const r = byId[id]; if (!r) return id; return String(r.name || id).replace(/\\.[a-z0-9]+$/i,'').replace(/_+/g,' '); }
function paintHead(){
  $('phead').innerHTML = '<a href="/projects" class="muted plain">← Projects</a>'
    + '<h1 contenteditable="true" id="title" spellcheck="false">' + esc(P.title) + '</h1>'
    + '<p class="lead">' + (P.client ? esc(P.client) + ' · ' : '') + (P.shoot_date ? 'shooting ' + esc(P.shoot_date) : 'no shoot date yet — set one so the sun schedule is real')
    + ' · ' + P.locations.length + ' candidate room' + (P.locations.length === 1 ? '' : 's') + '</p>';
  $('title').onblur = () => save({title: $('title').textContent.trim()});
}
function paintCands(){
  const box = $('cands');
  if (!P.locations.length) { box.innerHTML = '<p class="note">No rooms yet. Add the scans you are considering for this job.</p>'; }
  else box.innerHTML = '<div class="rooms">' + P.locations.map(id => '<div class="room">'
    + '<a href="/?job=' + id + '" class="plain"><img loading="lazy" src="/thumb/' + id + '" alt="" onerror="this.remove()"><span class="b"><b>' + esc(roomName(id)) + '</b>'
    + '<span>' + esc((byId[id] && byId[id].meta) || '') + '</span></span></a>'
    + '<span class="b links"><a href="/package/' + id + (P.shoot_date ? '?date=' + P.shoot_date : '') + '" target="_blank">package</a>'
    + '<a href="#" data-rm="' + id + '">remove</a></span></div>').join('') + '</div>';
  box.querySelectorAll('[data-rm]').forEach(a => a.onclick = e => { e.preventDefault(); save({remove_location: a.dataset.rm}); });
  const sel = $('addroom'); sel.innerHTML = '';
  ROOMS.filter(r => !P.locations.includes(r.id)).forEach(r => { const o = document.createElement('option'); o.value = r.id; o.textContent = roomName(r.id) + (r.meta ? ' · ' + r.meta : ''); sel.appendChild(o); });
  if (!sel.options.length) { const o = document.createElement('option'); o.textContent = 'every scanned room is already in'; o.disabled = true; sel.appendChild(o); }
}
function fill(){ $('brief').value = P.brief || ''; $('shoot_date').value = P.shoot_date || ''; $('client').value = P.client || ''; $('notes').value = P.notes || ''; }
function save(changes){
  const body = Object.assign({brief: $('brief').value, shoot_date: $('shoot_date').value, client: $('client').value, notes: $('notes').value}, changes || {});
  $('status').textContent = 'saving…';
  return fetch('/project/' + P.id, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)})
    .then(r => r.json()).then(r => { if (r.error) { $('status').textContent = r.error; return; } P = r; paintHead(); paintCands(); $('status').textContent = 'saved'; setTimeout(() => $('status').textContent = '', 1500); if (P.fit) paintFit(P.fit); });
}
$('save').onclick = () => save();
$('addbtn').onclick = () => { const v = $('addroom').value; if (v) save({add_location: v}); };
$('compare').onclick = () => {
  $('compare').disabled = true; $('status').textContent = 'placing every shot in every room…';
  save().then(() => fetch('/project/' + P.id + '/compare', {method:'POST'})).then(r => r.json()).then(r => {
    $('compare').disabled = false;
    if (r.error) { $('status').textContent = r.error; return; }
    P.fit = r; paintFit(r); $('status').textContent = '';
    $('fitwrap').scrollIntoView({behavior:'smooth', block:'start'});
  }).catch(e => { $('compare').disabled = false; $('status').textContent = String(e); });
};
function paintFit(fit){
  const rows = fit.rows || [];
  $('fitwrap').classList.remove('hidden');
  const shots = fit.shots || [];
  $('fitmeta').textContent = (shots.length ? shots.length + ' shots · ' : 'no shot list: rooms ranked on floor alone · ') + rows.length + ' rooms · ' + (fit.seconds || 0) + ' s' + (fit.shoot_date ? ' · sun on ' + fit.shoot_date : '');
  let h = '<tr><th>#</th><th>Room</th>' + shots.map(s => '<th title="' + esc(s.description || '') + '">' + esc(String(s.size || '').toUpperCase()) + '<br><span class="muted">' + esc(s.subject || '') + '</span></th>').join('')
    + '<th class="num">Workable floor</th><th>Dollies</th><th>Sun on the day</th><th>Trust</th><th></th></tr>';
  rows.forEach(r => {
    const cells = shots.map(s => { const f = (r.shots || []).find(x => x.number === s.number); if (!f) return '<td></td>';
      const t = f.verdict === 'held' ? '✓' : f.verdict === 'relaxed' ? '~' : '✗';
      const tip = f.verdict === 'missed' ? 'no clear setup' : (Math.round(f.focal_mm) + ' mm from ' + Number(f.distance_m).toFixed(1) + ' m' + (f.relaxed && f.relaxed.length ? ' · ' + f.relaxed.join('; ') : ''));
      return '<td><span class="v ' + f.verdict + '" title="' + esc(tip) + '">' + t + '</span></td>'; }).join('');
    const sun = r.sun ? (r.sun.windows_with_direct_sun + ' window' + (r.sun.windows_with_direct_sun === 1 ? '' : 's') + ' lit' + (r.sun.heading_assumed ? ' <span class="muted">(heading assumed)</span>' : '')) : '<span class="muted">no coordinates</span>';
    const trust = r.trust ? (r.trust.qa_verdict === 'pass' ? '<span class="pill on">measured</span>' : '<span class="pill">scale ±' + (r.trust.scale_confidence >= 0.8 ? '3' : r.trust.scale_confidence >= 0.6 ? '10' : '20') + '%</span>') : '';
    h += '<tr><td class="num"><b>' + r.rank + '</b></td><td><a href="/?job=' + r.job_id + '" class="plain">' + esc(r.display_name || roomName(r.job_id)) + '</a><div class="muted">' + esc(r.summary || '') + '</div></td>'
      + cells + '<td class="num">' + (r.workable_floor_m2 != null ? r.workable_floor_m2 + ' m²' : '—') + '</td><td>' + esc((r.dollies_fit || []).filter(d => /dolly|PeeWee|Fisher/i.test(d)).length + ' of 3') + '</td>'
      + '<td>' + sun + '</td><td>' + trust + '</td><td><a href="/package/' + r.job_id + (fit.shoot_date ? '?date=' + fit.shoot_date : '') + '" target="_blank">package</a></td></tr>';
  });
  $('fit').innerHTML = h;
}
$('sharebtn').onclick = () => {
  fetch('/share', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({kind:'project', target: P.id})})
    .then(r => r.json()).then(r => {
      if (r.error) { $('sharenote').textContent = r.error; return; }
      const url = (SHARE_BASE || location.origin) + '/s/' + r.token;
      $('sharenote').innerHTML = 'Share link: <a href="' + url + '" target="_blank">' + url + '</a> · opened ' + (r.opens || 0) + ' time' + (r.opens === 1 ? '' : 's') + ' · <a href="#" id="revoke">revoke</a>';
      navigator.clipboard && navigator.clipboard.writeText(url).catch(() => {});
      $('revoke').onclick = e => { e.preventDefault(); fetch('/share/' + r.token, {method:'DELETE'}).then(() => $('sharenote').textContent = 'Link revoked.'); };
    });
};
paintHead(); paintCands(); fill(); if (P.fit) paintFit(P.fit);
"""
    return shell(f"{project['title']} · Locaish", body, nav="projects", script=script)


def shared_project_page(project: dict, fit: dict | None, rooms: dict[str, dict], token: str) -> str:
    """Read-only: the ranking and a package link per room, under the token."""
    rows = (fit or {}).get("rows") or []
    shots = (fit or {}).get("shots") or []
    head = "".join(f"<th>{_e(str(s.get('size') or '').upper())}<br><span class='muted'>{_e(s.get('subject'))}</span></th>" for s in shots)
    body_rows = []
    for r in rows:
        cells = []
        for s in shots:
            f = next((x for x in r.get("shots", []) if x["number"] == s["number"]), None)
            if not f:
                cells.append("<td></td>")
                continue
            sym = {"held": "✓", "relaxed": "~", "missed": "✗"}[f["verdict"]]
            cells.append(f"<td><span class='v {f['verdict']}'>{sym}</span></td>")
        pkg = f"/s/{_e(token)}/package/{_e(r['job_id'])}"
        body_rows.append(
            f"<tr><td class='num'><b>{r.get('rank')}</b></td><td><a href='{pkg}' class='plain'>{_e(r.get('display_name') or r['job_id'])}</a>"
            f"<div class='muted'>{_e(r.get('summary'))}</div></td>{''.join(cells)}"
            f"<td class='num'>{_e(r.get('workable_floor_m2'))} m²</td><td><a href='{pkg}'>package</a> · <a href='/s/{_e(token)}/view/{_e(r['job_id'])}' target='_blank'>3D twin</a></td></tr>"
        )
    if not rows:
        body_rows.append("<tr><td colspan='99' class='note'>The rooms have not been compared yet.</td></tr>")
        for jid in project.get("locations") or []:
            r = rooms.get(jid) or {}
            body_rows.append(f"<tr><td></td><td><a href='/s/{_e(token)}/package/{_e(jid)}' class='plain'>{_e(r.get('name') or jid)}</a></td>"
                             f"<td colspan='98'><a href='/s/{_e(token)}/package/{_e(jid)}'>package</a> · <a href='/s/{_e(token)}/view/{_e(jid)}' target='_blank'>3D twin</a></td></tr>")
    body = f"""
<h1>{_e(project['title'])}</h1>
<p class="lead">{_e(project.get('client') or '')}{' · ' if project.get('client') else ''}{('shooting ' + _e(project['shoot_date'])) if project.get('shoot_date') else ''}</p>
<div class="card">
  <h2>Which rooms hold the scene</h2>
  {('<pre class="note">' + _e(project.get('brief')) + '</pre>') if project.get('brief') else ''}
  <div class="scroll"><table><tr><th>#</th><th>Room</th>{head}<th class="num">Workable floor</th><th></th></tr>{''.join(body_rows)}</table></div>
  <p class="muted mt"><span class="v held">✓</span> clear sightline &nbsp; <span class="v relaxed">~</span> with a compromise &nbsp; <span class="v missed">✗</span> not possible in this room. Measured from phone captures by Locaish; open a package for the numbers and their error bars.</p>
</div>
"""
    return shell(f"{project['title']} · shared", body)


# ---------------------------------------------------------------------------
# pricing and pilots
# ---------------------------------------------------------------------------


def pricing_page(*, stripe_scout: str = "", stripe_production: str = "", contact_email: str = "") -> str:
    def cta(url: str, label: str, plan: str) -> str:
        if url:
            return f"<a class='btn' href='{_e(url)}'>{_e(label)}</a>"
        return f"<a class='btn' href='#pilot' data-plan='{plan}'>Start a pilot</a>"

    body = f"""
<h1>Scout six. Drive to two.</h1>
<p class="lead">A tech scout day costs a van of department heads. Locaish answers the tech scout's questions from a phone walkthrough, so the van only goes to the finalists. Priced against what you already pay.</p>
<div class="plans">
  <div class="card">
    <h2>Scout</h2>
    <div class="price">$29<span class="muted"> / month</span></div>
    <p class="note">Free for your first six months, or $29 for a single package with no subscription. For freelance scouts and location managers: a better deliverable than the next scout's folder of JPEGs.</p>
    <ul class="note"><li>10 location packages a month</li><li>Sun on the shoot date, through every window</li><li>Share links the director opens on a phone</li><li>Your personal library</li></ul>
    {cta(stripe_scout, "Start free", "scout")}
  </div>
  <div class="card" style="border-color:var(--ink)">
    <h2>Production</h2>
    <div class="price">$249<span class="muted"> / job, or $499 / month</span></div>
    <p class="note">For production companies bidding jobs, billed to the job like the scout is: up to eight candidate rooms ranked against the shot list, packages and share links for 60 days. Unlimited jobs at $499 a month.</p>
    <ul class="note"><li>Projects: every candidate room ranked against the brief</li><li>Coverage planning with rendered frames</li><li>Company library and shot search</li><li>Team passcode, private deployment</li></ul>
    {cta(stripe_production, "Subscribe", "production")}
  </div>
  <div class="card">
    <h2>Capture service</h2>
    <div class="price">$350<span class="muted"> / location</span></div>
    <p class="note">We send a trained scout with a LiDAR phone and deliver the package within 24 hours. The fastest way to find out whether this is worth it to you.</p>
    <ul class="note"><li>Los Angeles, New York, London</li><li>Package, twin, and coverage plan</li><li>$750 for a multi-room location</li></ul>
    <a class="btn quiet" href="#pilot" data-plan="capture">Book a capture</a>
  </div>
</div>
<section id="pilot" class="mt2">
  <div class="card wide">
    <h2>Design-partner pilot</h2>
    <p class="note">Ten companies, ninety days, free Production plan. In return: one captured room a week and a fifteen-minute call every two weeks. We build what you block on.</p>
    <form id="lead">
      <div class="grid2">
        <div class="field"><label>Name</label><input name="name" required></div>
        <div class="field"><label>Email</label><input name="email" type="email" required></div>
        <div class="field"><label>Company</label><input name="company"></div>
        <div class="field"><label>Role</label><input name="role" placeholder="head of production, scout, DP…"></div>
      </div>
      <input type="hidden" name="plan" value="pilot">
      <div class="field"><label>What are you bidding or scouting right now?</label><textarea name="message" rows="3"></textarea></div>
      <button class="btn" type="submit">Request a pilot</button> <span class="note" id="leadstatus"></span>
    </form>
  </div>
</section>
<p class="muted mt2 foot">Every number Locaish produces traces to a measurement, a solar ephemeris or a catalogue dimension, and every twin grades its own accuracy. We do not replace the final visit; we choose which room gets it.{(' Questions: <a href="mailto:' + _e(contact_email) + '">' + _e(contact_email) + '</a>') if contact_email else ''}</p>
"""
    script = """
document.querySelectorAll('[data-plan]').forEach(a => a.onclick = () => { document.querySelector('[name=plan]').value = a.dataset.plan; });
document.getElementById('lead').onsubmit = e => {
  e.preventDefault(); const f = e.target; const body = {};
  ['name','email','company','role','plan','message'].forEach(k => body[k] = f[k].value);
  document.getElementById('leadstatus').textContent = 'sending…';
  fetch('/lead', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)}).then(r => r.json())
    .then(r => { document.getElementById('leadstatus').textContent = r.error ? r.error : 'Thanks — we reply within a day.'; if (!r.error) f.reset(); })
    .catch(() => document.getElementById('leadstatus').textContent = 'could not send; email us instead');
};
"""
    return shell("Pricing · Locaish", body, nav="pricing", script=script)


# ---------------------------------------------------------------------------
# capture, on a phone, in the room
# ---------------------------------------------------------------------------


def capture_page(*, showcase: bool = False) -> str:
    body = f"""
<h1>Capture a room</h1>
<p class="lead">Sixty seconds of walking with a phone. Follow the five steps and the twin comes back metric, with the sun schedule and the shot table.</p>
<div class="grid2">
  <div class="card">
    <h2>The recipe</h2>
    <ol class="note">
      <li><b>Hold the phone at chest height</b>, landscape, and keep it there. The camera height is one of the scale anchors.</li>
      <li><b>Walk the whole room slowly</b>, along the walls, keeping every wall in frame as you pass it. Turn corners with the camera, not your feet.</li>
      <li><b>Include a door</b>, fully, top to bottom. A door leaf is the same height almost everywhere on earth and it is the tightest anchor we have.</li>
      <li><b>Tilt up once</b> to see the ceiling and down once to see the floor at your feet, mid-walk.</li>
      <li><b>End where you started</b>, looking at the same wall. A closed loop is what stops the room from drifting.</li>
    </ol>
    <p class="note"><b>Better still:</b> an iPhone Pro with Polycam or Scaniverse in LiDAR mode, exported as GLTF or GLB, gives a twin that is metric to centimetres. Drop the export here instead of a video. <a href="https://github.com/JoshuaA1292/Locaish/blob/main/CAPTURE.md">Full capture protocol</a>.</p>
  </div>
  <div class="card">
    <h2>Film it now</h2>
    {"<p class='note'>Uploads are off on this gallery. Run the studio locally or on your own deployment to scan.</p>" if showcase else '''
    <p class="note">Tap to open the camera, film the walk, and the upload starts when you stop. Keep this page open; the report is ready in about fifteen minutes.</p>
    <label class="btn block" for="cam">Open the camera</label>
    <input id="cam" type="file" accept="video/*" capture="environment" class="hidden">
    <p class="note mt">Or <label for="pick" class="pick">choose a file</label> already on this phone (video, or a Polycam / Scaniverse export).</p>
    <input id="pick" type="file" accept="video/*,.ply,.obj,.glb,.gltf,.stl,.mov,.mp4,.mkv,.twin" class="hidden">
    <label class="row"><input type="checkbox" id="usegeo" checked> use this phone's location so the sun schedule is real</label>
    <div id="prog" class="hidden prog mt"><div id="stage"></div><div class="track"><i id="bar"></i></div><div id="done" class="note mt"></div></div>
    '''}
  </div>
</div>
"""
    script = "" if showcase else """
const $ = id => document.getElementById(id);
function go(f){
  if (!f) return;
  $('prog').classList.remove('hidden'); $('stage').textContent = 'reading ' + f.name;
  const send = (lat, lon) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/upload?name=' + encodeURIComponent(f.name) + (lat != null ? '&lat=' + lat + '&lon=' + lon : ''));
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.upload.onprogress = e => { if (e.lengthComputable) { $('bar').style.width = (10 * e.loaded / e.total) + '%'; $('stage').textContent = 'uploading ' + Math.round(100 * e.loaded / e.total) + '%'; } };
    xhr.onload = () => { let r = {}; try { r = JSON.parse(xhr.responseText); } catch (_) {} if (!r.id) { $('stage').textContent = r.error || 'upload failed'; return; } listen(r.id); };
    xhr.onerror = () => $('stage').textContent = 'upload failed';
    xhr.send(f);
  };
  if ($('usegeo').checked && navigator.geolocation) {
    let sent = false; const once = (a, b) => { if (!sent) { sent = true; send(a, b); } };
    setTimeout(() => once(null, null), 6000);
    navigator.geolocation.getCurrentPosition(p => once(p.coords.latitude, p.coords.longitude), () => once(null, null), {timeout: 6000});
  } else send(null, null);
}
function listen(id){
  const es = new EventSource('/events/' + id); let n = 10;
  $('done').innerHTML = 'Job <code>' + id + '</code> is running. You can close this page: the room will be at <a href="/?job=' + id + '">/?job=' + id + '</a> when it finishes.';
  es.onmessage = ev => { const m = JSON.parse(ev.data);
    if (m.kind === 'stage') { $('stage').textContent = m.text; n = Math.min(n + 3, 95); $('bar').style.width = n + '%'; }
    else if (m.kind === 'error') { es.close(); $('stage').textContent = m.text; }
    else if (m.kind === 'done') { es.close(); $('bar').style.width = '100%'; $('stage').textContent = 'ready'; $('done').innerHTML = '<a class="btn" href="/?job=' + id + '">Open the room</a> &nbsp; <a class="btn quiet" href="/package/' + id + '">Location package</a>'; } };
  es.onerror = () => { es.close(); setTimeout(() => listen(id), 2500); };
}
$('cam').onchange = () => go($('cam').files[0]);
$('pick').onchange = () => go($('pick').files[0]);
"""
    return shell("Capture · Locaish", body, nav="capture", script=script)


# ---------------------------------------------------------------------------
# login
# ---------------------------------------------------------------------------


def login_page(*, error: str = "", next_url: str = "/") -> str:
    body = f"""
<div class="narrow">
  <div class="card">
    <h2>This studio is private</h2>
    <p class="note">Enter the team passcode. Share links sent from here open without one.</p>
    <form method="post" action="/login">
      <input type="hidden" name="next" value="{_e(next_url)}">
      <div class="field"><label>Passcode</label><input name="passcode" type="password" autofocus required></div>
      {"<p class='err'>" + _e(error) + "</p>" if error else ""}
      <button class="btn" type="submit">Enter</button>
    </form>
  </div>
</div>
"""
    return shell("Sign in · Locaish", body)
