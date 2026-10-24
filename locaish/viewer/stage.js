/* The stage: cast, set pieces and the rehearsal, on the studio page.
 *
 * Served at /stage.js and loaded after the page's own script, so it can use
 * its helpers ($, esc, postViewer, lookAt, caption, SIZE_NAMES). Everything
 * here talks to the Stage routes in locaish/product/stage.py and never
 * computes a number of its own: it shows what the server measured.
 */

let stJob = null, stStatus = {}, stCast = [], stPieces = [], stLabels = [], stRuns = [];
let stCurrent = null, stFrames = [], stFrameIdx = 0, stPlaying = null, stBusy = false;

function initStage(id) {
  stJob = id;
  stCurrent = null; stFrames = []; stFrameIdx = 0; stopAnimatic();
  $('st_add').onclick = addActor;
  $('st_height').onkeydown = e => { if (e.key === 'Enter') addActor(); };
  $('st_lift').onclick = liftPiece;
  $('st_run').onclick = startRehearsal;
  $('st_example_typed').onclick = e => { e.preventDefault(); $('st_scene').value = typedExample(); };
  $('st_example_prose').onclick = e => { e.preventDefault(); $('st_scene').value = proseExample(); };
  $('st_scrub').oninput = () => showFrame(Number($('st_scrub').value));
  $('st_play').onclick = togglePlay;
  const d = $('d_shoot_date'); if (d && d.value) $('st_date').value = d.value;
  fetch('/stage-status/' + id).then(r => r.json()).then(s => { stStatus = s || {}; paintStatus(); }).catch(() => {});
  loadCast(); loadPieces(); loadRuns();
}

function paintStatus() {
  const bits = [];
  bits.push(stStatus.breakdown ? 'Muse Spark breaks the scene down' : 'no Meta key: type the scene as beats');
  bits.push(stStatus.sam3d ? 'SAM 3D lifts bodies and set pieces' : 'SAM 3D not deployed: capsules and scan points stand in');
  bits.push(stStatus.finish ? 'Muse Image finishes frames' : '');
  $('st_status').textContent = bits.filter(Boolean).join(' · ');
  $('st_run').textContent = stStatus.breakdown ? 'Rehearse the scene' : 'Rehearse the typed scene';
}

/* --- cast --------------------------------------------------------------- */

function loadCast() {
  fetch('/cast/' + stJob).then(r => r.json()).then(r => { stCast = r.cast || []; paintCast(); }).catch(() => {});
}

function paintCast() {
  const box = $('st_cast');
  if (!stCast.length) { box.innerHTML = '<span class="dim">no one cast yet: a name and a height in metres</span>'; return; }
  box.innerHTML = stCast.map(a => {
    const col = 'rgb(' + (a.colour || [120, 120, 120]).join(',') + ')';
    const body = a.body && a.body.attached ? '<span class="st_tag">body from headshot</span>' : (a.headshot ? '<span class="st_tag dim">capsule</span>' : '');
    const img = a.headshot ? '<img class="st_head" src="/headshot/' + stJob + '/' + a.id + '" alt="">' : '<span class="st_head st_blank" style="background:' + col + '"></span>';
    return '<div class="st_actor" data-id="' + a.id + '">' + img
      + '<div class="st_actorbody"><b>' + esc(a.name) + '</b> <span class="dim">' + Number(a.height_m).toFixed(2) + ' m · eyes ' + Number(a.eye_height_m).toFixed(2) + ' m</span> ' + body
      + '<div class="st_acts"><label class="st_file">headshot<input type="file" accept="image/*" data-head="' + a.id + '"></label>'
      + '<button class="st_x" data-rm="' + a.id + '" title="remove">×</button></div></div></div>';
  }).join('');
  box.querySelectorAll('input[data-head]').forEach(inp => inp.onchange = () => uploadHeadshot(inp.dataset.head, inp.files[0]));
  box.querySelectorAll('[data-rm]').forEach(b => b.onclick = () => fetch('/cast/' + stJob + '/' + b.dataset.rm, {method: 'DELETE'}).then(loadCast));
}

function addActor() {
  const name = $('st_name').value.trim(), h = $('st_height').value.trim();
  if (!name || !h) return;
  fetch('/cast/' + stJob, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({name, height_m: Number(h)})})
    .then(r => r.json()).then(r => {
      if (r.error) { $('st_caststatus').textContent = r.error; return; }
      $('st_name').value = ''; $('st_height').value = ''; $('st_caststatus').textContent = '';
      loadCast();
    });
}

function uploadHeadshot(aid, file) {
  if (!file) return;
  $('st_caststatus').textContent = stStatus.sam3d ? 'SAM 3D Body is lifting ' + file.name + '…' : 'saving the headshot…';
  fetch('/cast/' + stJob + '/' + aid + '/headshot', {method: 'POST', headers: {'Content-Type': file.type || 'image/jpeg'}, body: file})
    .then(r => r.json()).then(r => {
      $('st_caststatus').textContent = r.error || r.note || (r.body ? 'body lifted: ' + r.body.vertices + ' vertices, scaled to the typed height' : '');
      loadCast();
    }).catch(e => { $('st_caststatus').textContent = String(e); });
}

/* --- set pieces ---------------------------------------------------------- */

function loadPieces() {
  Promise.all([
    fetch('/setpieces/' + stJob).then(r => r.json()),
    fetch('/labels/' + stJob).then(r => r.json()),
  ]).then(([p, l]) => { stPieces = p.pieces || []; stLabels = l.labels || []; paintPieces(); paintPlan(); }).catch(() => {});
}

function paintPieces() {
  const sel = $('st_liftsel');
  const lifted = new Set(stPieces.map(p => p.label_id).filter(Boolean));
  const cands = stLabels.filter(l => !lifted.has(l.id));
  sel.innerHTML = cands.length ? cands.map(l => '<option value="' + l.id + '">' + esc(l.label) + '</option>').join('') : '<option value="">name something in the room first</option>';
  $('st_lift').disabled = !cands.length;
  const box = $('st_pieces');
  if (!stPieces.length) { box.innerHTML = '<span class="dim">nothing lifted yet: pick a labelled thing and lift it into a set piece</span>'; return; }
  box.innerHTML = stPieces.map(p => {
    const fit = p.door_fit && p.door_fit.note ? '<div class="dim">' + esc(p.door_fit.note) + '</div>' : '';
    const meas = p.measured ? p.size + ' measured from ' + p.points.toLocaleString() + ' returns' : p.size + ' <span class="warnc">catalogue size: the scan had too few returns here</span>';
    const lift3d = stStatus.sam3d && stStatus.meta && p.source !== 'sam3d' ? '<button class="lensbtn" data-l3="' + p.id + '">Lift to 3D</button>' : '';
    const tag = p.source === 'sam3d' ? '<span class="st_tag">SAM 3D splat</span>' : '';
    return '<div class="st_piece" data-id="' + p.id + '"><div><b>' + esc(p.label) + '</b> ' + tag + '<div class="dim">' + meas + '</div><div class="' + (p.moved ? 'st_moved' : 'dim') + '">' + esc(p.moved_text) + '</div>' + fit + '</div>'
      + '<div class="st_nudge">'
      + '<button data-mv="' + p.id + '|0|0.25|0" title="north 25 cm">↑</button><button data-mv="' + p.id + '|-0.25|0|0" title="west 25 cm">←</button><button data-mv="' + p.id + '|0.25|0|0" title="east 25 cm">→</button><button data-mv="' + p.id + '|0|-0.25|0" title="south 25 cm">↓</button>'
      + '<button data-mv="' + p.id + '|0|0|15" title="turn 15°">⟲</button><button data-mv="' + p.id + '|0|0|-15" title="turn -15°">⟳</button>'
      + '<button data-reset="' + p.id + '" title="back where the scan found it">reset</button>' + lift3d
      + '<button class="st_x" data-rmp="' + p.id + '" title="remove">×</button></div></div>';
  }).join('');
  box.querySelectorAll('[data-mv]').forEach(b => b.onclick = () => {
    const [id, dx, dy, dyaw] = b.dataset.mv.split('|');
    movePiece(id, {dx: Number(dx), dy: Number(dy), dyaw_deg: Number(dyaw), relative: true});
  });
  box.querySelectorAll('[data-reset]').forEach(b => b.onclick = () => movePiece(b.dataset.reset, {reset: true}));
  box.querySelectorAll('[data-rmp]').forEach(b => b.onclick = () => fetch('/setpieces/' + stJob + '/' + b.dataset.rmp, {method: 'DELETE'}).then(loadPieces));
  box.querySelectorAll('[data-l3]').forEach(b => b.onclick = () => {
    $('st_piecestatus').textContent = 'SAM 3.1 is masking it in the frame that saw it best; SAM 3D Objects is lifting it…';
    b.disabled = true;
    fetch('/setpieces/' + stJob + '/' + b.dataset.l3 + '/lift3d', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'})
      .then(r => r.json()).then(r => { $('st_piecestatus').textContent = r.error || ('lifted: ' + r.gaussians.toLocaleString() + ' gaussians fitted to the measured footprint'); loadPieces(); });
  });
}

function liftPiece() {
  const id = $('st_liftsel').value;
  if (!id) return;
  $('st_piecestatus').textContent = 'growing the piece from the scan’s own returns…';
  fetch('/setpieces/' + stJob + '/lift', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({label_id: id})})
    .then(r => r.json()).then(r => { $('st_piecestatus').textContent = r.error || (r.piece.size + (r.piece.measured ? ' measured' : ', catalogue size')); loadPieces(); });
}

function movePiece(id, body) {
  $('st_piecestatus').textContent = 're-sweeping the room as dressed…';
  fetch('/setpieces/' + stJob + '/' + id + '/move', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})
    .then(r => r.json()).then(r => {
      if (r.error) { $('st_piecestatus').textContent = r.error; return; }
      const d = r.diff;
      $('st_piecestatus').innerHTML = d && d.sentences && d.sentences.length ? d.sentences.slice(0, 3).map(esc).join('<br>') : (d ? 'the shot table did not change' : '');
      loadPieces();
    });
}

function paintPlan() {
  fetch('/stage-plan/' + stJob).then(r => r.text()).then(svg => { $('st_plan').innerHTML = svg; }).catch(() => {});
}

/* --- the rehearsal ------------------------------------------------------- */

function typedExample() {
  return 'TITLE: The kitchen, morning\nMARKS: A=(0.4,0.8) B=(-1.0,-0.4) DOOR=(1.6,-1.2)\nCAST: Maya=1.68 Theo=1.84\n'
    + '1. Theo enters at B\n2. Maya enters at DOOR   [cam (0.2,-2.0) h1.5 32mm on Maya]\n3. Maya moves to A\n'
    + '4. Theo: "You came back."   [cam (-0.6,-1.9) h1.45 50mm on Theo]\n5. Maya: "I never left."   [cam (0.9,-1.8) h1.4 35mm on Maya dolly to (0.3,-1.8)]\n6. hold 2s\n';
}

function proseExample() {
  return 'INT. KITCHEN - MORNING\n\nTHEO is at the counter with his back to the window. MAYA comes in through the door and crosses to the table.\n\n'
    + 'THEO\nYou came back.\n\nMAYA\nI never left.\n\nShe sits. He does not turn around.';
}

function loadRuns() {
  fetch('/rehearsals/' + stJob).then(r => r.json()).then(r => {
    stRuns = r.rehearsals || [];
    const box = $('st_prev');
    if (!stRuns.length) { box.classList.add('hidden'); return; }
    box.classList.remove('hidden');
    box.innerHTML = '<span class="dim">earlier:</span> ' + stRuns.filter(x => x.state === 'done').map(x =>
      '<a href="#" data-run="' + x.rehearsal_id + '">' + esc(x.title) + ' · ' + x.shots + ' setups · ' + Math.round(x.duration_s || 0) + ' s</a>').join(' ');
    box.querySelectorAll('[data-run]').forEach(a => a.onclick = e => { e.preventDefault(); openRun(a.dataset.run); });
    // Coming back to a room shows its last rehearsal without a click.
    const done = stRuns.filter(x => x.state === 'done');
    if (!stCurrent && done.length) openRun(done[done.length - 1].rehearsal_id);
  }).catch(() => {});
}

function startRehearsal() {
  const scene = $('st_scene').value.trim();
  if (!scene || stBusy) return;
  stBusy = true; $('st_run').disabled = true;
  $('st_log').innerHTML = ''; $('st_out').classList.add('hidden');
  stLog('rehearsing…');
  const body = {scene, shoot_date: $('st_date').value || null, shoot_start: $('st_start').value || null, setup_minutes: Number($('st_setup').value) || 25};
  fetch('/rehearse/' + stJob, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})
    .then(r => r.json()).then(r => {
      if (!r.rehearsal_id) { rehearsalFail(r.error || 'could not start'); return; }
      listenRehearsal(r.rehearsal_id);
    }).catch(e => rehearsalFail(String(e)));
}

function rehearsalFail(text) { stLog(text, 'error'); stBusy = false; $('st_run').disabled = false; }

function stLog(text, cls) {
  const d = document.createElement('div'); d.textContent = text; if (cls) d.className = cls;
  $('st_log').appendChild(d); $('st_log').scrollTop = 1e9;
}

function listenRehearsal(rid) {
  const es = new EventSource('/rehearsal-events/' + stJob + '/' + rid);
  let settled = false;
  es.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.kind === 'stage') { stLog(m.text); caption('Rehearsing', m.text); }
    else if (m.kind === 'note') stLog(m.text, 'note');
    else if (m.kind === 'error') { settled = true; es.close(); rehearsalFail(m.text); }
    else if (m.kind === 'done') { settled = true; es.close(); stBusy = false; $('st_run').disabled = false; stLog('done'); loadRuns(); openRun(rid); }
  };
  es.onerror = () => { es.close(); if (!settled) setTimeout(() => { if (!settled) listenRehearsal(rid); }, 2000); };
}

function openRun(rid) {
  fetch('/rehearsal/' + stJob + '/' + rid).then(r => r.json()).then(r => {
    if (r.error) { stLog(r.error, 'error'); return; }
    stCurrent = {id: rid, data: r};
    renderRun();
  });
}

function renderRun() {
  const r = stCurrent.data, rid = stCurrent.id, reh = r.rehearsal || {}, sc = r.scene || {};
  $('st_out').classList.remove('hidden');
  const prov = r.provider === 'meta' ? ' · broken down by Muse Spark' : ' · typed scene';
  $('st_head').innerHTML = '<b>' + esc(sc.title || 'Rehearsal') + '</b><span>' + Math.round(reh.duration_s || 0) + ' s · ' + (reh.shots || []).length + ' setups · ' + (sc.beats || []).length + ' beats' + prov + '</span>';
  // frames
  fetch('/rehearsal/' + stJob + '/' + rid + '/animatic.json').then(x => x.json()).then(a => {
    stFrames = (a.frames || []).filter(f => f.file);
    $('st_scrub').max = Math.max(0, stFrames.length - 1); $('st_scrub').value = 0;
    $('st_strip').innerHTML = stFrames.map((f, i) => '<img data-i="' + i + '" src="/rehearsal-image/' + stJob + '/' + rid + '/' + encodeURIComponent(f.file) + '" loading="lazy" alt="">').join('');
    $('st_strip').querySelectorAll('img').forEach(im => im.onclick = () => { stopAnimatic(); showFrame(Number(im.dataset.i)); });
    showFrame(0);
  }).catch(() => { stFrames = []; $('st_strip').innerHTML = ''; });
  // shots
  $('st_shots').innerHTML = (reh.shots || []).map(s => {
    const st = s.setup || {};
    const probs = (s.problems || []).length ? '<div class="relax">' + s.problems.map(esc).join('; ') + '</div>' : '';
    const sizes = (s.sizes || []).map(z => esc(z.name || z.size) + ' ' + Number(z.from).toFixed(1) + '–' + Number(z.to).toFixed(1) + ' s').join(', ');
    return '<div class="st_shot"><b>Shot ' + s.number + '</b> · ' + Math.round(st.focal_mm || 0) + ' mm at ' + Number(st.cam_z || 0).toFixed(2) + ' m on ' + esc(st.subject || '') + (st.dolly_to ? ' · dolly' : '')
      + ' · ' + Number(s.t_start).toFixed(1) + '–' + Number(s.t_end).toFixed(1) + ' s<div class="dim">' + sizes + '</div>' + probs + '</div>';
  }).join('') || '<div class="dim">no camera in this scene: add [cam …] to a beat, or let the breakdown choose</div>';
  // placements that failed
  const failed = (r.placements || []).filter(p => !p.placed);
  const probs = (reh.problems || []).concat(failed.map(p => 'shot ' + p.shot + ': ' + p.why));
  $('st_problems').innerHTML = probs.length ? '<h4>What does not work</h4>' + probs.map(p => '<div class="note">' + esc(p) + '</div>').join('') : '';
  // schedule
  const sch = r.schedule;
  $('st_sched').innerHTML = sch ? '<h4>The shoot day</h4>' + (sch.entries || []).map(e => '<div><b>' + e.start + '–' + e.end + '</b> ' + esc(e.reason || ('shot ' + e.shot)) + '</div>').join('')
    + (sch.sentences || []).map(s => '<div class="dim">' + esc(s) + '</div>').join('') : '<div class="dim">set a shoot date (and a georeference in the package) to lay the shots out against the sun</div>';
  $('st_notes').innerHTML = r.notes ? '<h4>Notes</h4><p>' + esc(r.notes) + '</p>' : '';
  $('st_blocking').innerHTML = r.blocking_svg || '';
  const base = '/rehearsal/' + stJob + '/' + rid;
  $('st_links').innerHTML = '<a href="' + base + '/callsheet.txt" target="_blank">Call sheet (.txt)</a><a href="' + base + '/blocking.svg" target="_blank">Blocking (.svg)</a><a href="' + base + '/rehearsal.json">Rehearsal (.json)</a>';
  $('st_out').scrollIntoView({behavior: 'smooth', block: 'start'});
}

function showFrame(i) {
  if (!stFrames.length) { $('st_frame').innerHTML = '<div class="dim">no frames: the scene has no camera</div>'; return; }
  stFrameIdx = Math.max(0, Math.min(i, stFrames.length - 1));
  const f = stFrames[stFrameIdx], rid = stCurrent.id;
  $('st_scrub').value = stFrameIdx;
  const cam = f.camera || {};
  const who = Object.keys(f.actors || {}).join(', ');
  const sun = f.sun ? (f.sun.direct_sun ? 'direct sun through window ' + (f.sun.windows_lit || []).map(w => w + 1).join(', ') : 'no direct sun') : '';
  const finishBtn = stStatus.finish ? '<button class="lensbtn" id="st_finish">Finish this frame with Muse Image</button>' : '';
  $('st_frame').innerHTML = '<div class="st_framerow"><div><img src="/rehearsal-image/' + stJob + '/' + rid + '/' + encodeURIComponent(f.file) + '" alt="frame"><div class="dim">measured frame · t ' + Number(f.t).toFixed(1) + ' s' + (f.clock ? ' · ' + f.clock : '') + (cam.focal_mm ? ' · ' + Math.round(cam.focal_mm) + ' mm · shot ' + cam.shot : '') + (who ? ' · ' + esc(who) : '') + (sun ? ' · ' + sun : '') + '</div></div>'
    + '<div id="st_finished"></div></div><div class="st_acts"><button class="lensbtn" id="st_look">Look through it in the twin</button>' + finishBtn + '</div>';
  $('st_strip').querySelectorAll('img').forEach(im => im.classList.toggle('active', Number(im.dataset.i) === stFrameIdx));
  $('st_look').onclick = () => {
    if (!cam.pos) return;
    lookAt({cam: cam.pos, subject: [cam.aim[0], cam.aim[1]], focal_mm: cam.focal_mm}, 'Rehearsal · shot ' + cam.shot + ' · ' + Math.round(cam.focal_mm) + ' mm');
    $('viewerwrap').scrollIntoView({behavior: 'smooth', block: 'start'});
  };
  if (stStatus.finish) $('st_finish').onclick = () => finishFrame(f);
}

function finishFrame(f) {
  const rid = stCurrent.id;
  $('st_finished').innerHTML = '<div class="dim">Muse Image is finishing the frame from the cast’s headshots…</div>';
  fetch('/rehearsal/' + stJob + '/' + rid + '/finish/' + encodeURIComponent(f.file), {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({})})
    .then(r => r.json()).then(r => {
      if (r.error) { $('st_finished').innerHTML = '<div class="error">' + esc(r.error) + '</div>'; return; }
      $('st_finished').innerHTML = '<img src="/rehearsal-image/' + stJob + '/' + rid + '/' + encodeURIComponent(r.file) + '?t=' + Date.now() + '" alt="finished"><div class="dim">' + esc(r.note) + ' · ' + esc(r.model) + (r.headshots ? ' · ' + r.headshots + ' headshot(s)' : '') + '</div>';
    }).catch(e => { $('st_finished').innerHTML = '<div class="error">' + esc(String(e)) + '</div>'; });
}

function togglePlay() { if (stPlaying) stopAnimatic(); else { $('st_play').textContent = 'Pause'; stPlaying = setInterval(() => showFrame((stFrameIdx + 1) % Math.max(1, stFrames.length)), 1000); } }
function stopAnimatic() { if (stPlaying) clearInterval(stPlaying); stPlaying = null; const b = $('st_play'); if (b) b.textContent = 'Play'; }
