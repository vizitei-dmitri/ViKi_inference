// Read-only replay of causal observations and measured joints on a shared clock.
import { api, log } from './core.js';
import { create } from './review_scene.js';

let root = null, scene = null, metadata = null, sessions = [], frame = 0, lastFrame = null;
let generation = 0, loadSerial = 0, frameSerial = 0, playing = false, animation = 0;
let pending = false, cloudFrame = -1;
const frames = new Map(), clouds = new Map();
const role = name => root?.querySelector(`[data-role="${name}"]`);
const text = (name, value) => { const element = role(name); if (element) element.textContent = value; };
const number = value => Number.isFinite(value) ? value.toFixed(1) : '—';
const url = suffix => `/api/live/reviews/${encodeURIComponent(metadata.id)}${suffix}`;

function pause() {
  playing = false; cancelAnimationFrame(animation); animation = 0;
  text('play', '▶');
}
function clearCaches() { frames.clear(); clouds.clear(); cloudFrame = -1; }
function cached(cache, key, factory, cap) {
  if (!cache.has(key)) {
    const promise = factory().catch(error => { cache.delete(key); throw error; });
    cache.set(key, promise);
    if (cache.size > cap) cache.delete(cache.keys().next().value);
  }
  return cache.get(key);
}
function fetchFrame(index) {
  return cached(frames, index, () => api('GET', url(`/frames/${index}`)), 160);
}
function fetchCloud(index) {
  return cached(clouds, index, async () => {
    const response = await fetch(url(`/cloud/${index}`));
    if (!response.ok) throw Error('Point cloud unavailable: ' + response.status);
    return response.arrayBuffer();
  }, 48);
}
function showMetrics(value) {
  const h = value.hand, r = value.robot;
  text('state', r?.state || 'waiting for robot state');
  role('state').classList.toggle('bad', r?.state === 'fault');
  role('state').classList.toggle('active', r?.state === 'active');
  text('reason', r?.reason || (!h?.visible ? h?.reason || 'warming up / stale hand' : '—'));
  text('age', h ? `${number(h.age_ms)} ms` : '—');
  text('processing', h ? `${number(h.processing_ms)} ms` : '—');
  text('valid', h?.visible ? `${h.observed.filter(Boolean).length} / 21` : 'unavailable');
  text('orientation', h?.visible && h.orientation_valid ? 'valid' : 'unavailable');
  text('solver', `${number(r?.solver_ms)} ms`);
  text('tracking', Number.isFinite(r?.tracking_error_rad) ? `${r.tracking_error_rad.toFixed(4)} rad` : '—');
  text('sequence', `hand ${h?.sequence ?? '—'} · control input ${r?.input_sequence ?? '—'}`);
  text('time', `${value.time_s.toFixed(2)} s · source ${value.source_time_s.toFixed(2)} s · frame ${value.source_frame+1}/${metadata.source_frames}`);
  text('counter', `${frame+1} / ${metadata.n_frames}`);
  role('timeline').value = frame;
}
async function setFrame(index) {
  if (!metadata || !root) return;
  const serial = ++frameSerial, id = metadata.id, gen = generation;
  const selected = Math.max(0, Math.min(metadata.n_frames - 1, Math.round(index)));
  try {
    const value = await fetchFrame(selected);
    const buffer = metadata.cloud && value.source_frame !== cloudFrame ? await fetchCloud(value.source_frame) : null;
    if (!root || gen !== generation || serial !== frameSerial || metadata?.id !== id) return;
    frame = selected; lastFrame = value;
    if (buffer) { scene.setCloud(buffer); cloudFrame = value.source_frame; }
    if (!metadata.cloud) scene.setCloud(null);
    scene.setFrame(value); showMetrics(value);
    for (let d=1; d<=3 && selected+d<metadata.n_frames; d++) fetchFrame(selected+d).catch(() => {});
  } catch (error) {
    if (gen === generation && root && serial === frameSerial) { pause(); text('error', error.message); }
  }
}
function play() {
  if (!metadata || playing) return;
  playing = true; text('play', '❚❚');
  const gen = generation, serial = loadSerial;
  const started = performance.now(), firstFrame = frame;
  const playbackSpeed = Number(role('speed').value);
  async function tick(now) {
    if (!root || !playing || gen !== generation || serial !== loadSerial) return;
    const elapsed = firstFrame + Math.floor((now-started) * metadata.fps * playbackSpeed / 1000);
    if (!pending && elapsed !== frame) {
      if (elapsed >= metadata.n_frames && !role('loop').checked) { await setFrame(metadata.n_frames-1); pause(); return; }
      pending = true;
      try { await setFrame(elapsed % metadata.n_frames); }
      finally { if(gen===generation && serial===loadSerial) pending = false; }
    }
    if (root && playing && gen === generation && serial === loadSerial) animation = requestAnimationFrame(tick);
  }
  animation = requestAnimationFrame(tick);
}
async function load(id) {
  pause(); const serial = ++loadSerial, gen = generation;
  ++frameSerial; metadata = null; lastFrame = null; frame = 0; clearCaches(); scene.setCloud(null);
  scene.setMetadata(null); scene.setFrame({});
  text('error', ''); text('status', 'Loading recorded online run…');
  try {
    const value = await api('GET', `/api/live/reviews/${encodeURIComponent(id)}`);
    if (!root || gen !== generation || serial !== loadSerial) return;
    metadata = value; scene.setMetadata(value);
    role('timeline').max = value.n_frames-1;
    text('status', `${value.episode_id} · ${value.backend} · ${value.duration_s.toFixed(2)} s · ${value.phase}`);
    text('profile', value.profile);
    text('mapping', `Motion ×${value.scale}; input speed ×${value.recorded_speed}; ${value.orientation ? 'position + orientation' : 'position only'}`);
    text('alignment', value.alignment);
    text('clock', value.clock_method);
    text('outcome', `${value.phase}: ${value.reason}`);
    if (!value.cloud) text('error', 'Point cloud is missing. Build it with viki cloud for this reference episode.');
    const link = role('link');
    const location = new URL(window.location.href); location.searchParams.set('online_session', value.id);
    location.hash = 'online-review'; link.href = location.toString();
    await setFrame(0);
  } catch(error) { if(root && gen===generation && serial===loadSerial) text('error',error.message); }
}
async function refresh(preferred) {
  const gen = generation;
  try {
    const result = await api('GET','/api/live/reviews');
    if (!root || gen !== generation) return;
    sessions = result.sessions || [];
    const picker = role('session'); picker.replaceChildren();
    for (const item of sessions) {
      const option = document.createElement('option'); option.value = item.id;
      option.textContent = `${item.episode_id} · ${item.id.slice(0,8)} · input ×${item.recorded_speed} · ${item.orientation ? '6D' : 'position'} · ${item.phase}`;
      option.title = option.textContent + ' · ' + item.reason;
      picker.appendChild(option);
    }
    if (!sessions.length) { text('status','No completed recorded online runs.'); return; }
    if(preferred && !sessions.some(s=>s.id===preferred)) {
      text('error','Requested online run was not found. Select a run explicitly.');
      picker.selectedIndex=-1; return;
    }
    picker.value = preferred || sessions[0].id;
    await load(picker.value);
  } catch(error) { if(root && gen===generation) text('error',error.message); }
}

export function mount(view) {
  generation++; root = document.createElement('div'); root.className = 'viewer-tab review-tab';
  root.innerHTML = `
    <div class="viewer-canvas" data-role="canvas"><div class="review-legend">
      <b>Recorded ONLINE inference</b>
      <label><input data-layer="cloud" type="checkbox" checked> Source point cloud</label>
      <label><input data-layer="hand" type="checkbox" checked> <span style="color:#ffd166">Online filtered hand</span></label>
      <label><input data-layer="robot" type="checkbox" checked> <span style="color:#8aafff">Measured robot joints</span></label>
      <label><input data-layer="trails" type="checkbox" checked> <span style="color:#f472b6">Target</span> / <span style="color:#22d3ee">achieved TCP</span></label>
      <small>One shared runtime clock · no offline plan</small>
    </div></div>
    <aside class="viewer-side">
      <label class="viewer-field">Recorded online run<select data-role="session"></select></label>
      <button data-role="refresh">Refresh runs</button>
      <div data-role="status" class="viewer-status">Loading…</div>
      <div data-role="state" class="live-status"></div>
      <div data-role="error" class="live-error" role="alert"></div>
      <dl class="live-metrics">
        <dt>Observed points</dt><dd data-role="valid">—</dd>
        <dt>Palm orientation</dt><dd data-role="orientation">—</dd>
        <dt>Observation age</dt><dd data-role="age">—</dd>
        <dt>Source → estimate</dt><dd data-role="processing">—</dd>
        <dt>IK</dt><dd data-role="solver">—</dd>
        <dt>Joint tracking</dt><dd data-role="tracking">—</dd>
        <dt>Sequences</dt><dd data-role="sequence">—</dd>
        <dt>Reason</dt><dd data-role="reason">—</dd>
      </dl>
      <p class="hint" data-role="mapping"></p><p class="hint" data-role="alignment"></p>
      <p class="hint">Hand appears only after its estimate was ready. Missing or stale points are hidden. Braking and unbound intervals remain in the recording.</p>
      <p class="hint" data-role="profile"></p><p class="hint" data-role="clock"></p><p class="hint" data-role="outcome"></p>
      <a data-role="link">Link to this run</a>
      <button data-role="fit">Fit robot + hand</button><button data-role="focus-hand">Hand detail</button>
      <p class="hint">UR10e arm only; no gripper actuation.</p>
      <label class="viewer-field">Playback speed<select data-role="speed"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option></select></label>
      <label><input data-role="loop" type="checkbox" checked> Loop</label>
      <p class="viewer-help">Drag to orbit · wheel to zoom · right-drag to pan.<br>Read-only: this view never controls a robot.</p>
    </aside>
    <div class="viewer-timeline">
      <button data-role="stop">■</button><button data-role="prev">◄</button><button data-role="play">▶</button><button data-role="next">►</button>
      <input data-role="timeline" type="range" min="0" max="0" value="0" step="1">
      <span data-role="counter">0 / 0</span><span class="review-time" data-role="time">—</span>
    </div>`;
  view.appendChild(root); scene = create(role('canvas'));
  root.addEventListener('click', event => {
    const action = event.target.closest('button')?.dataset.role;
    if (action==='play') playing ? pause() : play();
    else if(action==='stop'){pause();setFrame(0);}
    else if(action==='prev'||action==='next'){pause();setFrame(frame+(action==='prev'?-1:1));}
    else if(action==='fit')scene.fit();
    else if(action==='focus-hand')scene.focusHand();
    else if(action==='refresh')refresh(role('session').value);
  });
  root.addEventListener('change', event => {
    if(event.target.dataset.role==='session')load(event.target.value);
    if(event.target.dataset.layer){scene.setLayer(event.target.dataset.layer,event.target.checked);if(lastFrame)scene.setFrame(lastFrame);}
    if(event.target.dataset.role==='speed' && playing){pause();play();}
  });
  root.addEventListener('input', event => {
    if(event.target.dataset.role==='timeline'){pause();setFrame(Number(event.target.value));}
  });
  refresh(new URLSearchParams(location.search).get('online_session'));
}

export function unmount() {
  generation++; loadSerial++; frameSerial++; pause(); pending=false;
  scene?.dispose(); scene=null; root?.remove(); root=null; metadata=null; lastFrame=null;
  clearCaches();
}
