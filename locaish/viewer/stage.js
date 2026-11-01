/* The stage: cast, set pieces and the rehearsal, on the studio page.
 *
 * Served at /stage.js and loaded after the page's own script, so it can use
 * its helpers ($, esc, postViewer, lookAt, caption, SIZE_NAMES). Everything
 * here talks to the Stage routes in locaish/product/stage.py and never
 * computes a number of its own: it shows what the server measured.
 */

let stJob = null, stStatus = {}, stCast = [], stPieces = [], stLabels = [], stRuns = [];
let stCurrent = null, stFrames = [], stFrameIdx = 0, stPlaying = null, stBusy = false;
// The finished frames of the open rehearsal (file names under finish/), which
// of the two the big frame shows, and the finish-all run in flight.
let stFinished = new Set(), stView = 'finished', stFinishRun = null, stFinishText = '';
const AUTO_KEY = 'locaish.autofinish';
function autoFinish() { try { return localStorage.getItem(AUTO_KEY) !== '0'; } catch (e) { return true; } }

function initStage(id) {
  stJob = id;
  stCurrent = null; stFrames = []; stFrameIdx = 0; stopAnimatic();
  stFinished = new Set(); stFinishRun = null; stFinishText = '';
  $('st_add').onclick = addActor;
  $('st_height').onkeydown = e => { if (e.key === 'Enter') addActor(); };
  $('st_lift').onclick = liftPiece;
  $('st_run').onclick = startRehearsal;
  $('st_example_typed').onclick = e => { e.preventDefault(); $('st_scene').value = typedExample(); };
  $('st_example_prose').onclick = e => { e.preventDefault(); $('st_scene').value = proseExample(); };
  $('st_scrub').oninput = () => showFrame(Number($('st_scrub').value));
  $('st_play').onclick = togglePlay;
  const d = $('d_shoot_date'); if (d && d.value) $('st_date').value = d.value;
  fetch('/stage-status/' + id).then(r => r.json()).then(s => { stStatus = s || {}; paintStatus(); if (typeof initDirect === 'function') initDirect(); }).catch(() => {});
  loadCast(); loadPieces(); loadRuns();
  if (typeof initSim === 'function') initSim(id);
  if (typeof initWorld === 'function') initWorld(id);
}

function paintStatus() {
  const bits = [];
  bits.push(stStatus.breakdown ? 'Muse Spark breaks the scene down' : 'no Meta key: type the scene as beats');
  bits.push(stStatus.sam3d ? 'SAM 3D lifts bodies and set pieces' : 'SAM 3D not deployed: capsules and scan points stand in');
  bits.push(stStatus.finish ? 'Muse Image finishes frames' : '');
  $('st_status').textContent = bits.filter(Boolean).join(' · ');
  $('st_run').textContent = stStatus.breakdown ? 'Rehearse the scene' : 'Rehearse the typed scene';
  const wh = $('wd_hint'); if (wh) wh.textContent = stStatus.finish ? 'Muse Image finishes every view from the scan\u2019s own photographs, about a cent a view; the splat is trained on the finished views.' : 'No Meta key on this studio: a world needs Muse Image to finish its views (set META_API_KEY).';
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
      + '<div class="st_words"><input data-words="description" data-actor="' + a.id + '" placeholder="looks: 19, short brown hair, slim" value="' + esc(a.description || '') + '" title="how they look, for the finish">'
      + '<input data-words="wardrobe" data-actor="' + a.id + '" placeholder="wardrobe: grey t-shirt, white trousers" value="' + esc(a.wardrobe || '') + '" title="what they wear, for the finish"></div>'
      + '<div class="st_acts"><label class="st_file">headshot<input type="file" accept="image/*" data-head="' + a.id + '"></label>'
      + '<button class="st_x" data-rm="' + a.id + '" title="remove">×</button></div></div></div>';
  }).join('');
  box.querySelectorAll('input[data-head]').forEach(inp => inp.onchange = () => uploadHeadshot(inp.dataset.head, inp.files[0]));
  box.querySelectorAll('input[data-words]').forEach(inp => {
    inp.onchange = () => saveWords(inp.dataset.actor, inp.dataset.words, inp.value);
    inp.onkeydown = e => { if (e.key === 'Enter') inp.blur(); };
  });
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

function saveWords(aid, field, value) {
  const body = {}; body[field] = value.trim();
  fetch('/cast/' + stJob + '/' + aid, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})
    .then(r => r.json()).then(r => {
      if (r.error) { $('st_caststatus').textContent = r.error; return; }
      const a = stCast.find(x => x.id === aid); if (a) a[field] = r.actor[field];
      $('st_caststatus').textContent = '';
    }).catch(e => { $('st_caststatus').textContent = String(e); });
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
    else if (m.kind === 'done') {
      settled = true; es.close(); stBusy = false; $('st_run').disabled = false; stLog('done'); loadRuns();
      openRun(rid).then(() => { if (stStatus.finish && stCast.length && autoFinish()) startFinishAll(false); });
    }
  };
  es.onerror = () => { es.close(); if (!settled) setTimeout(() => { if (!settled) listenRehearsal(rid); }, 2000); };
}

function openRun(rid) {
  return fetch('/rehearsal/' + stJob + '/' + rid).then(r => r.json()).then(r => {
    if (r.error) { stLog(r.error, 'error'); return; }
    stCurrent = {id: rid, data: r};
    if (stFinishRun && stFinishRun.rid !== rid) stFinishText = '';
    renderRun();
  });
}

function renderRun() {
  const r = stCurrent.data, rid = stCurrent.id, reh = r.rehearsal || {}, sc = r.scene || {};
  $('st_out').classList.remove('hidden');
  const prov = r.provider === 'meta' ? ' · broken down by Muse Spark' : ' · typed scene';
  $('st_head').innerHTML = '<b>' + esc(sc.title || 'Rehearsal') + '</b><span>' + Math.round(reh.duration_s || 0) + ' s · ' + (reh.shots || []).length + ' setups · ' + (sc.beats || []).length + ' beats' + prov + '</span>';
  // frames: the finished ones first, so the strip shows them where they exist
  Promise.all([
    fetch('/rehearsal/' + stJob + '/' + rid + '/animatic.json').then(x => x.json()),
    fetch('/rehearsal/' + stJob + '/' + rid + '/finished').then(x => x.json()).catch(() => ({files: []})),
  ]).then(([a, fin]) => {
    stFrames = (a.frames || []).filter(f => f.file);
    stFinished = new Set(fin.files || []);
    $('st_scrub').max = Math.max(0, stFrames.length - 1); $('st_scrub').value = 0;
    $('st_strip').innerHTML = stFrames.map((f, i) => '<img data-i="' + i + '" src="' + frameSrc(f, rid) + '" class="' + (hasFinished(f) ? 'finished' : '') + '" loading="lazy" alt="">').join('');
    $('st_strip').querySelectorAll('img').forEach(im => im.onclick = () => { stopAnimatic(); showFrame(Number(im.dataset.i)); });
    paintFinishBar();
    showFrame(0);
  }).catch(() => { stFrames = []; stFinished = new Set(); $('st_strip').innerHTML = ''; paintFinishBar(); });
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
  const finishBtn = stStatus.finish ? '<button class="lensbtn" id="st_finish">' + (hasFinished(f) ? 'Finish this frame again' : 'Finish this frame with Muse Image') + '</button>' : '';
  const showFin = hasFinished(f) && stView === 'finished';
  const facts = 't ' + Number(f.t).toFixed(1) + ' s' + (f.clock ? ' · ' + f.clock : '') + (cam.focal_mm ? ' · ' + Math.round(cam.focal_mm) + ' mm · shot ' + cam.shot : '') + (who ? ' · ' + esc(who) : '') + (sun ? ' · ' + sun : '');
  const big = showFin
    ? '<div class="st_frame_finished"><img src="/rehearsal-image/' + stJob + '/' + rid + '/' + encodeURIComponent(finishedName(f)) + '" alt="finished frame"><span class="st_stamp">finished · Muse Image · a render</span></div><div class="dim">finished frame, the measured one underneath it · ' + facts + '</div>'
    : '<img src="/rehearsal-image/' + stJob + '/' + rid + '/' + encodeURIComponent(f.file) + '" alt="frame"><div class="dim">measured frame · ' + facts + '</div>';
  $('st_frame').innerHTML = '<div class="st_framerow' + (showFin ? ' has_fin' : '') + '"><div>' + big + '</div>'
    + '<div id="st_finished">' + (showFin ? '<img src="/rehearsal-image/' + stJob + '/' + rid + '/' + encodeURIComponent(f.file) + '" alt="measured frame"><div class="dim">the measurement it was finished from</div>' : '') + '</div></div>'
    + '<div class="st_acts"><button class="lensbtn" id="st_look">Look through it in the twin</button>' + finishBtn + '</div>';
  $('st_strip').querySelectorAll('img').forEach(im => im.classList.toggle('active', Number(im.dataset.i) === stFrameIdx));
  paintToggle();
  $('st_look').onclick = () => {
    if (!cam.pos) return;
    lookAt({cam: cam.pos, subject: [cam.aim[0], cam.aim[1]], focal_mm: cam.focal_mm}, 'Rehearsal · shot ' + cam.shot + ' · ' + Math.round(cam.focal_mm) + ' mm');
    $('viewerwrap').scrollIntoView({behavior: 'smooth', block: 'start'});
  };
  if (stStatus.finish) $('st_finish').onclick = () => finishFrame(f);
}

function finishedName(f) { return 'finish_' + f.file.replace(/\.[^.]+$/, '') + '.png'; }
function hasFinished(f) { return stFinished.has(finishedName(f)); }
function frameSrc(f, rid, bust) {
  const name = hasFinished(f) ? finishedName(f) : f.file;
  return '/rehearsal-image/' + stJob + '/' + rid + '/' + encodeURIComponent(name) + (bust ? '?t=' + Date.now() : '');
}

function finishFrame(f) {
  const rid = stCurrent.id;
  $('st_finished').innerHTML = '<div class="dim">Muse Image is finishing the frame: the cast’s words and headshots, the measured geometry kept…</div>';
  fetch('/rehearsal/' + stJob + '/' + rid + '/finish/' + encodeURIComponent(f.file), {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({})})
    .then(r => r.json()).then(r => {
      if (r.error) { $('st_finished').innerHTML = '<div class="error">' + esc(r.error) + '</div>'; return; }
      landFinished(r.file, rid);
    }).catch(e => { $('st_finished').innerHTML = '<div class="error">' + esc(String(e)) + '</div>'; });
}

// A finished frame arrived: the strip and, if it is the one on screen, the big frame swap to it.
function landFinished(file, rid) {
  if (!stCurrent || stCurrent.id !== rid) return;
  stFinished.add(file);
  const i = stFrames.findIndex(f => finishedName(f) === file);
  if (i < 0) return;
  const im = $('st_strip').querySelector('img[data-i="' + i + '"]');
  if (im) { im.src = frameSrc(stFrames[i], rid, true); im.classList.add('finished'); }
  if (i === stFrameIdx) { stView = 'finished'; showFrame(i); }
  else paintToggle();
}

/* --- the whole scene through Muse Image ---------------------------------- */

function paintFinishBar() {
  const bar = $('st_finishbar');
  if (!bar) return;
  if (!stFrames.length) { bar.innerHTML = ''; return; }
  const n = stFrames.filter(hasFinished).length;
  const can = !!stStatus.finish;
  bar.innerHTML = '<span class="st_toggle"><button data-view="measured">measured</button><button data-view="finished">finished</button></span>'
    + '<span class="dim" id="st_fincount"></span>'
    + (can ? '<button class="lensbtn" id="st_finishall">' + (n === stFrames.length && n ? 'Finish the whole scene again' : 'Finish the whole scene with Muse Image') + '</button>'
      + '<label><input type="checkbox" id="st_autofin"' + (autoFinish() ? ' checked' : '') + '> finish automatically</label>'
      : '<span class="dim">no Meta key on this studio: the measured frames stand</span>')
    + '<span class="st_progress" id="st_finishline"></span>';
  bar.querySelectorAll('[data-view]').forEach(b => b.onclick = () => { stView = b.dataset.view; showFrame(stFrameIdx); });
  if (can) {
    $('st_finishall').onclick = () => startFinishAll(n > 0);
    $('st_autofin').onchange = () => { try { localStorage.setItem(AUTO_KEY, $('st_autofin').checked ? '1' : '0'); } catch (e) {} };
  }
  paintToggle();
}

function paintToggle() {
  const bar = $('st_finishbar');
  if (!bar || !stFrames.length) return;
  const n = stFrames.filter(hasFinished).length;
  const cur = stFrames[stFrameIdx];
  const canFin = cur && hasFinished(cur);
  bar.querySelectorAll('[data-view]').forEach(b => {
    b.disabled = b.dataset.view === 'finished' && !canFin;
    b.classList.toggle('on', (b.dataset.view === 'finished') === (canFin && stView === 'finished'));
  });
  const c = $('st_fincount'); if (c) c.textContent = n ? n + ' of ' + stFrames.length + ' finished' : (stCast.length ? '' : 'cast someone first: the finish needs people to put in the room');
  const line = $('st_finishline'); if (line) { line.textContent = stFinishText; line.classList.toggle('error', /^(error|failed)/i.test(stFinishText)); }
  const btn = $('st_finishall'); if (btn) btn.disabled = !!stFinishRun;
}

function setFinishText(t) { stFinishText = t; const line = $('st_finishline'); if (line) { line.textContent = t; line.classList.toggle('error', /^(error|failed)/i.test(t)); } }

function startFinishAll(force) {
  if (!stCurrent || stFinishRun) return;
  const rid = stCurrent.id;
  stFinishRun = {rid};
  setFinishText('starting…'); paintToggle();
  fetch('/rehearsal/' + stJob + '/' + rid + '/finish-all', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({force: !!force})})
    .then(r => r.json()).then(r => {
      if (!r.run) { finishAllEnd('error: ' + (r.error || 'could not start')); return; }
      const es = new EventSource(r.events);
      es.onmessage = e => {
        const m = JSON.parse(e.data);
        if (m.kind === 'stage') setFinishText(m.text);
        else if (m.kind === 'frame') { setFinishText(m.text); if (m.file) landFinished(m.file, rid); }
        else if (m.kind === 'note') setFinishText(m.text);
        else if (m.kind === 'done') { es.close(); finishAllEnd((m.finished || 0) + ' of ' + (m.total || 0) + ' frames finished in ' + (m.seconds || 0) + ' s' + (m.failed ? ' · ' + m.failed + ' failed, measured frames stand in' : '')); }
        else if (m.kind === 'error') { es.close(); finishAllEnd('error: ' + m.text); }
      };
      es.onerror = () => { es.close(); finishAllEnd('error: the finishing stream closed'); };
    }).catch(e => finishAllEnd('error: ' + String(e)));
}

function finishAllEnd(text) { stFinishRun = null; setFinishText(text); paintFinishBar(); }

function togglePlay() { if (stPlaying) stopAnimatic(); else { $('st_play').textContent = 'Pause'; stPlaying = setInterval(() => showFrame((stFrameIdx + 1) % Math.max(1, stFrames.length)), 1000); } }
function stopAnimatic() { if (stPlaying) clearInterval(stPlaying); stPlaying = null; const b = $('st_play'); if (b) b.textContent = 'Play'; }

/* --- the camera simulation ----------------------------------------------- */

// The page is the relay between the engine (over HTTP) and the viewer (over
// postMessage): a key moved in the twin is PUT to the server, the evaluation
// comes back, and the viewer is told; the playhead moving in the twin fetches
// the cast at that moment and the frame the camera would see.
let smJob = null, smSims = [], smCur = null, smRun = null, smBusy = false, smMode = false;
let smFrameTimer = null, smFigTimer = null, smT = 0, smListening = false;

function initDirect() {
  const btn = $('sm_directbtn'), inp = $('sm_instruction');
  if (!btn) return;
  btn.onclick = directCamera;
  inp.onkeydown = e => { if (e.key === 'Enter') directCamera(); };
  if (!stStatus.meta) { btn.disabled = true; $('sm_directnote').textContent = 'no Meta key on this studio: key the move in the twin instead'; }
  else $('sm_directnote').textContent = 'Muse Wonder operates the camera through the engine; Muse Spark looks at the frames.';
}

function directCamera() {
  const text = $('sm_instruction').value.trim();
  if (!text || !smCur || smDirecting) return;
  smDirecting = true; $('sm_directbtn').disabled = true;
  $('sm_directlog').innerHTML = ''; $('sm_looks').innerHTML = ''; $('sm_directnotes').innerHTML = '';
  dlog('directing: ' + text);
  fetch('/sim/' + stJob + '/' + smCur.id + '/direct', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({instruction: text})})
    .then(r => r.json()).then(r => {
      if (!r.run) { dlog(r.error || 'could not start', 'error'); smDirecting = false; $('sm_directbtn').disabled = false; return; }
      listenDirect(r.events);
    }).catch(e => { dlog(String(e), 'error'); smDirecting = false; $('sm_directbtn').disabled = false; });
}

function dlog(text, cls) { const d = document.createElement('div'); d.textContent = text; if (cls) d.className = cls; $('sm_directlog').appendChild(d); $('sm_directlog').scrollTop = 1e9; }

function listenDirect(url) {
  const es = new EventSource(url);
  let settled = false;
  es.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.kind === 'stage') dlog(m.text);
    else if (m.kind === 'call') dlog('→ ' + m.tool + (m.summary ? ': ' + m.summary : ''));
    else if (m.kind === 'agent') dlog('operator: ' + (m.text || ''), 'ag');
    else if (m.kind === 'look') {
      const base = '/sim-file/' + stJob + '/' + smCur.id + '/';
      const f = document.createElement('figure');
      f.innerHTML = '<img src="' + base + encodeURIComponent(m.file) + '?t=' + Date.now() + '" alt="look"><figcaption>looked at t ' + Number(m.t).toFixed(1) + ' s</figcaption>';
      f.dataset.t = Number(m.t).toFixed(2);
      $('sm_looks').appendChild(f);
    }
    else if (m.kind === 'review') {
      dlog('DP at ' + Number(m.t).toFixed(1) + ' s: ' + (m.score != null ? m.score + '/10 ' : '') + (m.verdict || '') + ' — ' + (m.text || ''), 'note');
      const fig = [...$('sm_looks').querySelectorAll('figure')].find(x => x.dataset.t === Number(m.t).toFixed(2));
      if (fig) fig.querySelector('figcaption').innerHTML = '<b>' + (m.score != null ? m.score + '/10 · ' : '') + esc(m.verdict || '') + '</b> ' + esc(m.text || '');
    }
    else if (m.kind === 'error') { settled = true; es.close(); dlog(m.text, 'error'); smDirecting = false; $('sm_directbtn').disabled = false; }
    else if (m.kind === 'done') {
      settled = true; es.close(); smDirecting = false; $('sm_directbtn').disabled = false;
      dlog('done: ' + (m.model_calls || 0) + ' model calls, ' + (m.looks || 0) + ' looks, ' + (m.seconds || 0) + ' s' + (m.finished ? '' : ' · the operator ran out of calls'));
      if (m.notes) $('sm_directnotes').innerHTML = '<h4>The operator\u2019s notes</h4><p>' + esc(m.notes) + '</p>';
      if (m.timeline) { takeSim(m); loadSims(); }
    }
  };
  es.onerror = () => { es.close(); if (!settled) setTimeout(() => { if (!settled) listenDirect(url); }, 2000); };
}

let smDirecting = false;

function initSim(id) {
  initDirect();
  smJob = id; smCur = null; smRun = null; smBusy = false; smMode = false; smT = 0;
  if (!$('sm_list')) return;
  $('sm_open').classList.add('hidden');
  $('sm_new').onclick = newSim;
  $('sm_fromreh').onchange = () => { const rid = $('sm_fromreh').value; $('sm_fromreh').value = ''; if (rid) simFromRehearsal(rid); };
  $('sm_twin').onclick = openInTwin;
  $('sm_render').onclick = renderMove;
  if (!smListening) { smListening = true; window.addEventListener('message', simMessage); }
  loadSims();
  loadSimRehearsals();
}

function loadSims() {
  fetch('/sims/' + smJob).then(r => r.json()).then(r => {
    smSims = r.sims || []; paintSims();
    // Coming back to a room shows its last camera move without a click.
    if (!smCur && smSims.length) openSim(smSims[smSims.length - 1].sim_id);
  }).catch(() => {});
}

function loadSimRehearsals() {
  fetch('/rehearsals/' + smJob).then(r => r.json()).then(r => {
    const done = (r.rehearsals || []).filter(x => x.state === 'done');
    $('sm_fromreh').innerHTML = '<option value="">from rehearsal ▾</option>' + done.map(x =>
      '<option value="' + x.rehearsal_id + '">' + esc(x.title) + ' · ' + x.shots + ' setups · ' + Math.round(x.duration_s || 0) + ' s</option>').join('');
    $('sm_fromreh').disabled = !done.length;
  }).catch(() => {});
}

function paintSims() {
  const box = $('sm_list');
  if (!smSims.length) { box.innerHTML = '<span class="dim">no camera moves yet: start one, or build one from a rehearsal</span>'; return; }
  box.innerHTML = smSims.map(s =>
    '<span class="sm_item' + (smCur && smCur.timeline.id === s.sim_id ? ' active' : '') + '" data-open="' + s.sim_id + '"><b>' + esc(s.title) + '</b><span class="dim">' + esc(s.rig) + ' · ' + Number(s.duration_s || 0).toFixed(1) + ' s</span>'
    + (s.problems ? '<span class="sm_probs">' + s.problems + ' problem' + (s.problems === 1 ? '' : 's') + '</span>' : '')
    + '<button class="st_x" data-del="' + s.sim_id + '" title="delete">×</button></span>').join('');
  box.querySelectorAll('[data-open]').forEach(el => el.onclick = e => { if (e.target.dataset.del) return; openSim(el.dataset.open); });
  box.querySelectorAll('[data-del]').forEach(b => b.onclick = e => {
    e.stopPropagation();
    fetch('/sim/' + smJob + '/' + b.dataset.del, {method: 'DELETE'}).then(() => {
      if (smCur && smCur.timeline.id === b.dataset.del) { smCur = null; $('sm_open').classList.add('hidden'); if (smMode) postViewer({type: 'locaish:sim-mode', on: false}); }
      loadSims();
    });
  });
}

function simFail(text) { $('sm_status').textContent = text || ''; smBusy = false; }

function newSim() {
  if (smBusy) return;
  smBusy = true; $('sm_status').textContent = 'placing a tripod where the sweep says a camera can stand…';
  fetch('/sims/' + smJob, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'})
    .then(r => r.json()).then(r => { if (r.error) return simFail(r.error); simFail(''); takeSim(r); loadSims(); }).catch(e => simFail(String(e)));
}

function simFromRehearsal(rid) {
  if (smBusy) return;
  smBusy = true; $('sm_status').textContent = 'turning the rehearsal into a timeline…';
  fetch('/sims/' + smJob, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'})
    .then(r => r.json()).then(r => {
      if (r.error) throw new Error(r.error);
      return fetch('/sim/' + smJob + '/' + r.timeline.id + '/from-rehearsal', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({rehearsal_id: rid})});
    }).then(r => r.json()).then(r => { if (r.error) return simFail(r.error); simFail(''); takeSim(r); loadSims(); }).catch(e => simFail(String(e)));
}

function openSim(id) {
  fetch('/sim/' + smJob + '/' + id).then(r => r.json()).then(r => { if (r.error) return simFail(r.error); takeSim(r); paintSims(); }).catch(e => simFail(String(e)));
}

function takeSim(r) {
  smCur = r; smT = 0; smRun = null; smFinishRun = null;
  $('sm_video').innerHTML = ''; $('sm_log').innerHTML = ''; $('sm_finishline').textContent = '';
  $('sm_open').classList.remove('hidden');
  paintSim();
  paintSimVideos();
  if (smMode) postViewer({type: 'locaish:sim', timeline: smCur.timeline, evaluation: smCur.evaluation});
  refreshSimFrame(0);
}

function fmtVal(v) {
  if (v == null) return '—';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(2);
  if (Array.isArray(v)) return v.map(fmtVal).join(', ');
  if (typeof v === 'object') return Object.keys(v).map(k => k.replace(/_/g, ' ') + ' ' + fmtVal(v[k])).join(' · ');
  return String(v);
}

function paintSim() {
  const tl = smCur.timeline, ev = smCur.evaluation || {};
  const rig = tl.rig || {}, params = rig.params || {};
  const gear = Object.keys(params).map(k => k.replace(/_/g, ' ') + ' ' + fmtVal(params[k])).join(' · ');
  $('sm_rig').innerHTML = '<b>' + esc(tl.title || 'Camera move') + '</b> · ' + esc(rig.kind || 'tripod') + ' · ' + Number(tl.duration_s || 0).toFixed(1) + ' s at ' + (tl.fps || 24) + ' fps · ' + esc(String(tl.sensor || '')) + ' f/' + (tl.aperture_f || '')
    + (gear ? '<div class="dim">' + esc(gear) + '</div>' : '') + (smCur.rig_text ? '<p>' + esc(smCur.rig_text) + '</p>' : '')
    + paramDocs(smCur.rig_info)
    + '<div class="dim">' + ((tl.camera || {}).keys || []).length + ' camera key(s) · ' + Object.keys(tl.actors || {}).length + ' actor(s) · ' + (tl.pieces || []).length + ' moved piece(s)</div>';
  const segs = ev.segments || [];
  $('sm_problems').innerHTML = segs.length ? segs.map(s => {
    const sev = String(s.severity || '').toLowerCase();
    const cls = /red|bad|error|fail/.test(sev) ? 'red' : (/amber|warn/.test(sev) ? 'amber' : '');
    return '<div class="sm_seg ' + cls + '"><b>' + Number(s.t_start).toFixed(1) + '–' + Number(s.t_end).toFixed(1) + ' s</b><span>' + esc(s.text || '') + '</span></div>';
  }).join('') : '<div class="dim">the move works the whole way</div>';
  // The summary: what the rig line does not already say, the headline numbers first.
  const sum = ev.summary || {};
  const skip = new Set(['title', 'rig', 'duration_s', 'fps', 'frames', 'worst_severity', 'notes']);
  const lead = {camera_travel_m: 'camera travels', size_range: 'size range', distance_range_m: 'distance to subject, m', focal_range_mm: 'lenses, mm', worst: 'worst problem'};
  const keys = Object.keys(lead).filter(k => k in sum).concat(Object.keys(sum).filter(k => !(k in lead) && !skip.has(k)));
  $('sm_summary').innerHTML = keys.filter(k => sum[k] != null && !(Array.isArray(sum[k]) && !sum[k].length)).map(k =>
    '<div>' + esc(lead[k] || k.replace(/_/g, ' ')) + ' <b>' + esc(k === 'camera_travel_m' ? Number(sum[k]).toFixed(2) + ' m' : fmtVal(sum[k])) + '</b></div>').join('');
  const base = '/sim-file/' + smJob + '/' + tl.id + '/';
  $('sm_links').innerHTML = '<a href="' + base + 'blender.py">Blender (.py)</a><a href="' + base + 'room.glb">Room (.glb)</a><a href="' + base + 'timeline.json">Timeline (.json)</a><a href="' + base + 'evaluation.json" target="_blank">Evaluation (.json)</a>';
  $('sm_twin').textContent = smMode ? 'Editing in the twin' : 'Open in the twin';
}

function paramDocs(info) {
  const docs = info && info.params;
  if (!docs || typeof docs !== 'object') return '';
  return Object.keys(docs).map(k => {
    const d = docs[k] || {};
    return '<div class="dim">' + esc(k.replace(/_/g, ' ')) + (d.default != null ? ' · default ' + esc(fmtVal(d.default)) : '') + (d.doc ? ' · ' + esc(d.doc) : '') + '</div>';
  }).join('');
}

function simFrameAt(t) {
  const ev = smCur && smCur.evaluation, frames = ev && ev.frames || [];
  if (!frames.length) return null;
  const fps = ev.fps || (ev.summary && ev.summary.fps) || (smCur.timeline.fps || 24);
  const i = Math.max(0, Math.min(frames.length - 1, Math.round(t * fps)));
  return frames[i];
}

function refreshSimFrame(t) {
  if (!smCur) return;
  clearTimeout(smFrameTimer);
  smFrameTimer = setTimeout(() => {
    const id = smCur.timeline.id;
    $('sm_frameimg').src = '/sim-frame/' + smJob + '/' + id + '?t=' + t.toFixed(3) + '&w=640&v=' + encodeURIComponent(smCur.evaluation && smCur.evaluation.frames ? smCur.evaluation.frames.length : 0);
    const f = simFrameAt(t);
    $('sm_frametext').innerHTML = f ? 't ' + Number(f.t).toFixed(2) + ' s · ' + Math.round(f.focal_mm || 0) + ' mm · ' + esc(String(f.shot_size || '')) + (f.distance_m != null ? ' · ' + Number(f.distance_m).toFixed(1) + ' m' : '')
      + ((f.problems || []).length ? ' · <span class="error">' + f.problems.map(esc).join('; ') + '</span>' : '') : '';
  }, 300);
}

function openInTwin() {
  if (!smCur) return;
  smMode = true;
  postViewer({type: 'locaish:sim-mode', on: true});
  postViewer({type: 'locaish:sim', timeline: smCur.timeline, evaluation: smCur.evaluation});
  postViewer({type: 'locaish:sim-rig', rig: smCur.timeline.rig});
  $('sm_twin').textContent = 'Editing in the twin';
  const vw = $('viewerwrap'); if (vw) vw.scrollIntoView({behavior: 'smooth', block: 'start'});
}

function simMessage(ev) {
  const m = ev.data;
  if (!m || typeof m.type !== 'string' || !m.type.startsWith('locaish:sim') || !smCur) return;
  if (m.type === 'locaish:sim-edit' && m.timeline) {
    const id = smCur.timeline.id;
    fetch('/sim/' + smJob + '/' + id, {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({timeline: m.timeline})})
      .then(r => r.json()).then(r => {
        if (r.error) { $('sm_status').textContent = r.error; return; }
        $('sm_status').textContent = '';
        smCur = Object.assign(smCur, r);
        paintSim(); loadSims();
        postViewer({type: 'locaish:sim', timeline: smCur.timeline, evaluation: smCur.evaluation});
        refreshSimFrame(smT);
      }).catch(e => { $('sm_status').textContent = String(e); });
  } else if (m.type === 'locaish:sim-time') {
    smT = Number(m.t) || 0;
    clearTimeout(smFigTimer);
    smFigTimer = setTimeout(() => {
      fetch('/sim/' + smJob + '/' + smCur.timeline.id + '/figures?t=' + smT.toFixed(3)).then(r => r.json())
        .then(r => { if (r.figures) postViewer({type: 'locaish:figures', figures: r.figures}); }).catch(() => {});
    }, 150);
    clearTimeout(smFrameTimer);
    smFrameTimer = setTimeout(() => refreshSimFrame(smT), 400);
  } else if (m.type === 'locaish:sim-mode') {
    smMode = !!m.on;
    $('sm_twin').textContent = smMode ? 'Editing in the twin' : 'Open in the twin';
  } else if (m.type === 'locaish:sim-select') {
    const k = ((smCur.timeline.camera || {}).keys || [])[m.key_index];
    if (k) { smT = Number(k.t) || 0; refreshSimFrame(smT); }
  }
}

function renderMove() {
  if (!smCur || smRun) return;
  const id = smCur.timeline.id;
  $('sm_render').disabled = true; $('sm_video').innerHTML = ''; $('sm_log').innerHTML = '';
  smLog('rendering the move…');
  fetch('/sim/' + smJob + '/' + id + '/render', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({width_px: 640})})
    .then(r => r.json()).then(r => {
      if (!r.run) { smLog(r.error || 'could not start', 'error'); $('sm_render').disabled = false; return; }
      smRun = r.run;
      const es = new EventSource(r.events);
      es.onmessage = e => {
        const m = JSON.parse(e.data);
        if (m.kind === 'stage') smLog(m.text);
        else if (m.kind === 'progress') { smLog(m.text); if (m.frame) $('sm_frameimg').src = m.frame; }
        else if (m.kind === 'done') {
          es.close(); smRun = null; $('sm_render').disabled = false;
          smLog((m.frames || 0) + ' frames in ' + (m.seconds || 0) + ' s' + (m.note ? ' · ' + m.note : ''));
          smCur.files = Object.assign({}, smCur.files || {}, {move_mp4: !!m.mp4, rendered: m.frames || 0, move_finished_mp4: false, finished: 0});
          paintSimVideos();
          if (stStatus.finish && stCast.length && autoFinish() && m.frames) finishMove();
          refreshSimFrame(smT);
        } else if (m.kind === 'error') { es.close(); smRun = null; $('sm_render').disabled = false; smLog(m.text, 'error'); }
      };
      es.onerror = () => { es.close(); smRun = null; $('sm_render').disabled = false; };
    }).catch(e => { smLog(String(e), 'error'); $('sm_render').disabled = false; });
}

function smLog(text, cls) {
  const d = document.createElement('div'); d.textContent = text; if (cls) d.className = cls;
  $('sm_log').appendChild(d); $('sm_log').scrollTop = 1e9;
}

/* --- the move through Muse Image ------------------------------------------ */

let smFinishRun = null;

// The measured move and, when it exists, the finished one beside it.
function paintSimVideos() {
  const files = (smCur && smCur.files) || {};
  const base = '/sim-file/' + smJob + '/' + smCur.timeline.id + '/';
  const v = Date.now();
  const parts = [];
  if (files.move_mp4) parts.push('<figure><video controls src="' + base + 'move.mp4?v=' + v + '"></video><figcaption>measured move · <a href="' + base + 'move.mp4" download="move.mp4">download</a></figcaption></figure>');
  if (files.move_finished_mp4) parts.push('<figure><video controls src="' + base + 'move_finished.mp4?v=' + v + '"></video><figcaption>finished with Muse Image · a render, not a measurement · <a href="' + base + 'move_finished.mp4" download="move_finished.mp4">download</a></figcaption></figure>');
  $('sm_video').innerHTML = parts.length ? '<div class="sm_pair' + (parts.length === 1 ? ' single' : '') + '">' + parts.join('') + '</div>' : '';
  const acts = $('sm_finishacts');
  if (!acts) return;
  if (files.rendered && stStatus.finish) {
    acts.innerHTML = '<button class="lensbtn" id="sm_finish">' + (files.move_finished_mp4 ? 'Finish the move again' : 'Finish the move with Muse Image') + '</button>'
      + '<span class="dim">every 12th rendered frame, held between</span>';
    $('sm_finish').onclick = () => finishMove(!!files.move_finished_mp4);
    $('sm_finish').disabled = !!smFinishRun;
  } else if (files.rendered) {
    acts.innerHTML = '<span class="dim">no Meta key on this studio: the measured move stands</span>';
  } else acts.innerHTML = '';
}

function finishMove(force) {
  if (!smCur || smFinishRun) return;
  const id = smCur.timeline.id;
  smFinishRun = {id};
  const line = $('sm_finishline');
  line.textContent = 'starting…';
  const btn = $('sm_finish'); if (btn) btn.disabled = true;
  fetch('/sim/' + smJob + '/' + id + '/finish', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({every: 12, force: !!force})})
    .then(r => r.json()).then(r => {
      if (!r.run) { finishMoveEnd('error: ' + (r.error || 'could not start')); return; }
      const es = new EventSource(r.events);
      es.onmessage = e => {
        const m = JSON.parse(e.data);
        if (m.kind === 'stage' || m.kind === 'frame' || m.kind === 'note') line.textContent = m.text;
        else if (m.kind === 'done') {
          es.close();
          if (smCur && smCur.timeline.id === id) smCur.files = Object.assign({}, smCur.files || {}, {move_finished_mp4: !!m.mp4, finished: m.finished || 0});
          finishMoveEnd((m.finished || 0) + ' of ' + (m.total || 0) + ' frames finished in ' + (m.seconds || 0) + ' s' + (m.note ? ' · ' + m.note : ''));
        } else if (m.kind === 'error') { es.close(); finishMoveEnd('error: ' + m.text); }
      };
      es.onerror = () => { es.close(); finishMoveEnd('error: the finishing stream closed'); };
    }).catch(e => finishMoveEnd('error: ' + String(e)));
}

function finishMoveEnd(text) { smFinishRun = null; $('sm_finishline').textContent = text; if (smCur) paintSimVideos(); }

/* --- the world: the room as it will be on the day ---------------------------
 *
 * A shoot date, an hour and a dressing note go to POST /world/<job>; the run
 * streams its stages and its counted steps; when it is done the page shows
 * the world's own viewer in a frame the size of the hero's, with a toggle
 * back to the scan, and the finished views in a strip. Everything shown here
 * is labelled a render: the measured room is the frame beside it.
 */

let wdJob = null, wdRun = null, wdCurrent = null, wdWorlds = [], wdMode = 'world';

function initWorld(id) {
  wdJob = id; wdRun = null; wdCurrent = null; wdWorlds = []; wdMode = 'world';
  const d = $('d_shoot_date');
  $('wd_date').value = (d && d.value) || new Date().toISOString().slice(0, 10);
  if (d) d.addEventListener('change', () => { if (d.value) $('wd_date').value = d.value; });
  $('wd_hour').oninput = paintWorldHour; paintWorldHour();
  $('wd_build').onclick = buildWorld;
  $('wd_build').disabled = false;
  $('wd_dressing').onkeydown = e => { if (e.key === 'Enter') buildWorld(); };
  $('wd_scanned').onclick = () => setWorldMode('scan');
  $('wd_asis').onclick = () => setWorldMode('world');
  $('wd_out').classList.add('hidden'); $('wd_log').innerHTML = ''; $('wd_progress').innerHTML = ''; $('wd_strip').innerHTML = '';
  $('wd_frame').removeAttribute('src');
  loadWorlds();
}

function wdClock(h) {
  const m = Math.round(Number(h || 0) * 60);
  return String(Math.floor(m / 60)).padStart(2, '0') + ':' + String(m % 60).padStart(2, '0');
}
function paintWorldHour() { $('wd_hourout').textContent = wdClock($('wd_hour').value); }
function wdLog(text, cls) {
  const d = document.createElement('div'); d.textContent = text; if (cls) d.className = cls;
  $('wd_log').appendChild(d); $('wd_log').scrollTop = 1e9;
}
function wdProgress(text, frac, cls) {
  const box = $('wd_progress');
  box.className = 'wd_progress' + (cls ? ' ' + cls : '');
  box.innerHTML = '<span>' + esc(text || '') + '</span>'
    + (frac == null ? '' : '<div class="wd_meter"><i style="width:' + Math.round(100 * Math.min(1, Math.max(0, frac))) + '%"></i></div>');
}

function loadWorlds() {
  fetch('/worlds/' + wdJob).then(r => r.json()).then(r => {
    wdWorlds = (r.worlds || []).filter(w => w.state === 'done');
    paintWorldList();
    // Coming back to a room shows its latest world without a click.
    if (!wdCurrent && !wdRun && wdWorlds.length) openWorld(wdWorlds[wdWorlds.length - 1]);
  }).catch(() => {});
}

function paintWorldList() {
  const box = $('wd_prev');
  if (!wdWorlds.length) { box.classList.add('hidden'); box.innerHTML = ''; return; }
  box.classList.remove('hidden');
  box.innerHTML = '<span class="dim">worlds built:</span> ' + wdWorlds.map(w =>
    '<a href="#" data-world="' + w.world_id + '"' + (wdCurrent && wdCurrent.world_id === w.world_id ? ' class="active"' : '') + '>'
    + esc((w.date || '') + ' ' + wdClock(w.hour)) + (w.dressing ? ' · ' + esc(w.dressing.length > 40 ? w.dressing.slice(0, 40) + '…' : w.dressing) : '')
    + ' · ' + (w.finished || 0) + ' of ' + (w.views || 0) + ' views</a>').join('');
  box.querySelectorAll('[data-world]').forEach(a => a.onclick = e => {
    e.preventDefault(); openWorld(wdWorlds.find(w => w.world_id === a.dataset.world));
  });
}

function buildWorld() {
  if (wdRun) return;
  const body = {date: $('wd_date').value || null, hour: Number($('wd_hour').value), dressing: $('wd_dressing').value.trim()};
  wdRun = {}; $('wd_build').disabled = true; $('wd_log').innerHTML = '';
  wdProgress('starting…', 0);
  fetch('/world/' + wdJob, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})
    .then(r => r.json()).then(r => {
      if (!r.run) { worldFail(r.error || 'could not start'); return; }
      wdRun = {id: r.run, events: r.events};
      listenWorld(r.events);
    }).catch(e => worldFail(String(e)));
}

function worldFail(text) {
  wdRun = null; $('wd_build').disabled = false;
  wdProgress(text, null, 'error'); wdLog(text, 'error');
}

function listenWorld(url) {
  const es = new EventSource(url);
  let settled = false;
  es.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.kind === 'stage') { wdLog(m.text); wdProgress(m.text); }
    else if (m.kind === 'progress') {
      const frac = m.total ? m.done / m.total : null;
      wdProgress((m.what ? m.what + ': ' : '') + (m.text || (m.done + ' of ' + m.total)), frac);
      if (m.url && /finish_view_/.test(m.url)) wdThumb(m.url, m.name || m.file || '');
    }
    else if (m.kind === 'note') wdLog(m.text, 'note');
    else if (m.kind === 'error') { settled = true; es.close(); worldFail('error: ' + m.text); }
    else if (m.kind === 'done') {
      settled = true; es.close(); wdRun = null; $('wd_build').disabled = false;
      wdProgress('built in ' + (m.seconds || 0) + ' s: ' + (m.finished || 0) + ' of ' + (m.views || 0) + ' views finished, splat trained on them', 1);
      wdLog('done');
      openWorld(m);
      loadWorlds();
    }
  };
  // A dropped stream is not a dead run: the server replays the terminal event.
  es.onerror = () => { es.close(); if (!settled) setTimeout(() => { if (!settled) listenWorld(url); }, 2000); };
}

function worldHtml(w) { return w.world_html || ('/world-file/' + wdJob + '/' + w.world_id + '/world.html'); }

function openWorld(w) {
  if (!w) return;
  wdCurrent = w; wdMode = 'world';
  $('wd_out').classList.remove('hidden');
  const light = w.light;
  const sun = light ? (light.windows_lit && light.windows_lit.length
      ? 'direct sun through window ' + light.windows_lit.map(i => i + 1).join(', ')
      : (light.elevation_deg > 0 ? 'no direct sun through a window' : 'sun below the horizon'))
    : 'no georeference: the light as described';
  $('wd_meta').textContent = (w.date || '') + ' at ' + wdClock(w.hour) + ' solar · ' + sun
    + (w.dressing ? ' · ' + w.dressing : '') + ' · ' + (w.finished || 0) + ' of ' + (w.views || 0) + ' views finished';
  paintWorldFrame();
  paintWorldList();
  const strip = $('wd_strip');
  strip.innerHTML = '';
  const n = Math.min(Number(w.views || 0), 200);
  for (let i = 0; i < n; i++) {
    const name = 'finish_view_' + String(i).padStart(4, '0') + '.png';
    wdThumb('/world-file/' + wdJob + '/' + w.world_id + '/' + name, name);
  }
}

function wdThumb(url, name) {
  const strip = $('wd_strip');
  if (strip.querySelector('img[data-url="' + url + '"]')) return;
  const img = document.createElement('img');
  img.loading = 'lazy'; img.alt = 'finished view: a render'; img.title = (name || '') + ' · render';
  img.dataset.url = url; img.src = url;
  img.onerror = () => img.remove();             // a view the model failed on has no finished file
  img.onclick = () => window.open(url, '_blank');
  strip.appendChild(img);
}

function paintWorldFrame() {
  const w = wdCurrent; if (!w) return;
  const page = wdMode === 'scan' ? '/view/' + wdJob : worldHtml(w);
  const src = page + '?embed=1';
  const f = $('wd_frame');
  if (f.getAttribute('src') !== src) f.src = src;
  $('wd_full').href = page;
  $('wd_scanned').classList.toggle('on', wdMode === 'scan');
  $('wd_asis').classList.toggle('on', wdMode !== 'scan');
  $('wd_tag').classList.toggle('hidden', wdMode === 'scan');
  $('wd_tag').textContent = wdMode === 'scan' ? '' : 'render';
}

function setWorldMode(m) { wdMode = m; paintWorldFrame(); }
