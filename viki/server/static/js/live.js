// Live observations and explicit following; tab navigation never stops a session.
import { sessionGet, sessionSet, log } from './core.js';
import { create as createScene } from './live_scene.js';

let root = null, humanScene = null, robotScene = null, status = null;
let generation = 0, pollTimer = 0, pollAbort = null, actionAbort = null, busy = false;
let packetHostUs = null, packetBrowserMs = 0;
let pendingRequest = null;
const number = value => Number.isFinite(value) ? value.toFixed(1) : '—';
const vector = value => Array.isArray(value) && value.length === 3 && value.every(Number.isFinite)
  ? value.map(v => v.toFixed(3)).join(', ') : '—';
const role = name => root?.querySelector(`[data-role="${name}"]`);

// Page-owned badge survives panel disposal so motion remains visible elsewhere.
function watchNavigation() {
  const badge = document.querySelector('[data-live-nav-state]');
  if (!badge) return;
  let closed = false, timer = 0, abort = null, epoch = 0;
  const tick = async () => {
    const token = epoch;
    abort = new AbortController();
    try {
      const current = await request('GET', '/api/live/status', null, abort.signal);
      if (closed || token !== epoch) return;
      const controller = current.controller || {};
      const running = current.running || controller.running;
      const state = controller.state === 'active' ? 'tracking' : controller.state === 'braking'
        ? 'braking' : current.phase === 'fault' || controller.state === 'fault' ? 'fault' : current.phase || 'observing';
      badge.hidden = !running && state !== 'fault';
      badge.textContent = ` · ${state}`;
      badge.className = state === 'tracking' ? 'live-nav-state active'
        : state === 'fault' ? 'live-nav-state bad' : 'live-nav-state';
    } catch (error) {
      if (token === epoch && error.name !== 'AbortError' && !badge.hidden) {
        badge.textContent = ' · status unavailable';
        badge.className = 'live-nav-state bad';
      }
    } finally {
      if (!closed && token === epoch) timer = setTimeout(tick, 2000);
    }
  };
  window.addEventListener('pagehide', () => {
    closed = true;
    epoch += 1;
    clearTimeout(timer);
    abort?.abort();
  });
  window.addEventListener('pageshow', () => {
    if (closed) { closed = false; tick(); }
  });
  tick();
}
watchNavigation();

async function request(method, path, body, signal) {
  const response = await fetch(path, { method, signal,
    headers: { 'Content-Type': 'application/json' },
    ...(body ? { body: JSON.stringify(body) } : {}) });
  const data = await response.json();
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(Array.isArray(detail) ? detail.map(d => d.msg || String(d)).join('; ')
      : detail || 'Live request failed');
  }
  return data;
}

function settings() {
  return {
    reference: role('reference').value.trim(), source: role('source').value,
    backend: role('backend').value, landmark_mode: role('landmarks').value,
    perception_hz: Number(role('rate').value), seconds: Number(role('seconds').value),
    recorded_speed: Number(role('speed').value), scale: Number(role('scale').value),
    orientation: role('orientation').checked, host: role('host').value.trim(),
    mapping_json: role('mapping').value.trim(),
  };
}

function backendFields() {
  const physical = role('backend').value === 'ur10e';
  role('hardware-fields').hidden = !physical;
  role('speed-field').hidden = role('source').value !== 'recorded';
  role('action-help').textContent = physical
    ? 'Bind & follow enables motion on the physical UR10e. Clutch releases following; Stop ends the session.'
    : 'Bind & follow starts relative following in MuJoCo. Clutch releases following; Stop ends the session.';
}

function saveForm() {
  if (!root) return;
  sessionSet('live-form-v1', settings());
  backendFields();
}

function text(name, value) {
  const element = role(name);
  if (element) element.textContent = value;
}

function renderStatus(next) {
  if (!root) return;
  status = next;
  const hand = next.hand, controller = next.controller || {};
  const running = Boolean(next.running || controller.running);
  const state = controller.state || next.phase || 'idle';
  const physical = next.backend === 'ur10e';
  if (next.running && next.reference) {
    role('reference').value = next.reference;
    if (next.source) role('source').value = next.source;
    if (next.backend) role('backend').value = next.backend;
    backendFields();
  }
  text('state', `${physical ? 'UR10e · ' : 'MuJoCo · '}${next.phase || 'idle'} / ${state}`);
  role('state').className = state === 'fault' || next.phase === 'fault' ? 'live-status bad'
    : state === 'active' ? 'live-status active' : 'live-status';
  text('session', next.session_id ? next.session_id.slice(0, 12) : '—');
  text('points', hand ? `${(hand.observed || []).filter(Boolean).length} / 21` : '—');
  text('position-valid', hand ? hand.position_valid ? 'valid' : 'unavailable' : '—');
  text('orientation-valid', hand ? hand.orientation_valid ? 'valid' : 'unavailable' : '—');
  text('processing', `${number(next.processing_ms?.median)} / ${number(next.processing_ms?.p95)} ms`);
  text('solver', `${number(controller.solver_ms)} ms`);
  text('reason', controller.reason || hand?.reason || next.reason || '—');
  text('hand-position', vector(hand?.position));
  text('robot-position', vector(controller.tcp_position));
  text('tracking', Number.isFinite(controller.tracking_error_rad)
    ? `${controller.tracking_error_rad.toFixed(4)} rad` : '—');
  text('directory', next.directory || '—');
  const requestResult = controller.request_result;
  if (pendingRequest && controller.request_id !== pendingRequest.id) {
    if (!running || state === 'fault') {
      text('request-result', `${pendingRequest.action}: session ended before acknowledgement`);
      pendingRequest = null;
    } else text('request-result', `${pendingRequest.action}: pending`);
  } else if (requestResult) {
    text('request-result', requestResult.accepted ? `${requestResult.command}: accepted`
      : `${requestResult.command}: ${requestResult.reason || 'rejected'}`);
    if (pendingRequest && controller.request_id === pendingRequest.id) pendingRequest = null;
  }
  role('start').disabled = busy || running;
  role('bind').disabled = busy || Boolean(pendingRequest) || !next.running || next.phase !== 'observing'
    || !hand?.position_valid || (role('orientation').checked && !hand?.orientation_valid) || state !== 'unbound';
  role('clutch').disabled = busy || Boolean(pendingRequest) || !controller.running || !['active', 'braking'].includes(state);
  role('stop').disabled = busy || !running;
  for (const input of root.querySelectorAll('[data-session-setting]')) input.disabled = busy || running;
  role('active-note').hidden = !running;
  role('robot-title').textContent = physical ? 'UR10e · robot-base frame' : 'MuJoCo · robot-base frame';
  const hostUs = controller.host_monotonic_us || hand?.generated_time_us;
  if (Number.isFinite(hostUs) && hostUs !== packetHostUs) {
    packetHostUs = hostUs;
    packetBrowserMs = performance.now();
  }
  if (hand) humanScene?.update({ points: hand.points, observed: hand.observed,
    position: hand.position, rotation: hand.rotation, poseValid: hand.position_valid,
    rotationValid: hand.orientation_valid, binding: controller.binding_human_position });
  robotScene?.update({ points: controller.robot_points || [], edges: controller.robot_edges || [],
    position: controller.tcp_position, rotation: controller.tcp_rotation,
    poseValid: Array.isArray(controller.tcp_position), rotationValid: true,
    binding: controller.binding_robot_position });
  updateAge();
}

function updateAge() {
  if (!root) return;
  const hand = status?.hand;
  const age = hand && Number.isFinite(packetHostUs)
    ? Math.max(0, (packetHostUs - hand.source_time_us) / 1000 + performance.now() - packetBrowserMs) : null;
  text('age', age == null ? '—' : `${age.toFixed(0)} ms`);
  role('human-canvas').classList.toggle('stale', age == null || age > 250 || !status?.running);
}

async function poll(id) {
  if (!root || id !== generation) return;
  pollAbort = new AbortController();
  try {
    const result = await request('GET', '/api/live/status', null, pollAbort.signal);
    if (id === generation && root) renderStatus(result);
  } catch (error) {
    if (error.name !== 'AbortError' && id === generation && root) text('error', error.message);
  } finally {
    if (id === generation && root) {
      pollAbort = null;
      // Schedule only after the preceding request resolves: no status backlog.
      pollTimer = setTimeout(() => poll(id), 200);
    }
  }
}

async function action(name) {
  if (!root || busy) return;
  const id = generation;
  let body = null;
  if (name === 'start') {
    if (!role('reference').reportValidity()) return;
    const values = settings();
    if (values.backend === 'ur10e' && !role('execute').checked) {
      text('error', 'Enable physical motion before starting a UR10e session.');
      return;
    }
    if (values.backend === 'ur10e' && (!values.mapping_json || !values.host)) {
      text('error', 'UR10e requires its IP and a confirmed camera-to-robot mapping file.');
      return;
    }
    for (const input of root.querySelectorAll('input[type="number"]')) {
      if (!input.reportValidity()) return;
    }
    body = { reference: values.reference, source: values.source, backend: values.backend,
      seconds: values.seconds, perception_hz: values.perception_hz,
      recorded_speed: values.recorded_speed, landmark_mode: values.landmark_mode,
      mapping_json: values.mapping_json || null,
      options: { scale: values.scale, orientation: values.orientation,
        host: values.host || null, hardware_enabled: values.backend === 'ur10e' && role('execute').checked } };
    saveForm();
  }
  busy = true;
  text('error', '');
  text('request-result', `${name}: requested`);
  if (status) renderStatus(status);
  actionAbort = new AbortController();
  try {
    const result = await request('POST', `/api/live/${name}`, body, actionAbort.signal);
    if (id !== generation || !root) return;
    if (name === 'start' || name === 'stop') renderStatus(result);
    pendingRequest = result.request_id ? { id: result.request_id, action: name } : null;
    text('request-result', `${name}: ${pendingRequest ? 'pending' : 'submitted'}`);
    log(name === 'start' ? 'Live observation started' : name === 'bind' ? 'Live bind requested'
      : name === 'clutch' ? 'Live following released' : 'Live stop requested', 'ok');
  } catch (error) {
    if (id === generation && root && error.name !== 'AbortError') text('error', error.message);
  } finally {
    if (id === generation && root) {
      busy = false;
      actionAbort = null;
      if (status) renderStatus(status);
    }
  }
}

export function mount(view) {
  generation += 1;
  const id = generation;
  status = { running: false, phase: 'idle' };
  busy = false;
  packetHostUs = null;
  pendingRequest = null;
  root = document.createElement('div');
  root.className = 'live-tab';
  // Only fixed markup. All persisted/API/user strings enter value/textContent.
  root.innerHTML = `
    <aside class="live-sidebar">
      <h2>Live human following</h2>
      <div data-role="state" class="live-status">Loading…</div>
      <label class="live-field">Reference episode<input data-session-setting data-role="reference"
        type="text" required placeholder="data/datasets/40mm-pnp/2026-10-06_11-15-00"></label>
      <div class="live-form-pair">
        <label class="live-field">Input<select data-session-setting data-role="source">
          <option value="recorded">Recorded RGB-D</option><option value="cameras">Live cameras</option></select></label>
        <label class="live-field">Robot<select data-session-setting data-role="backend">
          <option value="mujoco">MuJoCo</option><option value="ur10e">Physical UR10e</option></select></label>
      </div>
      <div class="live-form-pair">
        <label class="live-field">Landmarks<select data-session-setting data-role="landmarks">
          <option value="all21">All 21 points</option><option value="arm7">7 control anchors</option></select></label>
        <label class="live-field">Processing, Hz<input data-session-setting data-role="rate"
          type="number" min="1" max="30" step="1"></label>
      </div>
      <div class="live-form-pair">
        <label class="live-field">Session limit, s<input data-session-setting data-role="seconds"
          type="number" min="1" max="3600" step="1"></label>
        <label class="live-field" data-role="speed-field">Input speed<input data-session-setting data-role="speed"
          type="number" min="0.1" max="2" step="0.1"></label>
      </div>
      <label class="live-field">Motion scale<input data-session-setting data-role="scale"
        type="number" min="0.01" max="1" step="0.01"></label>
      <label class="live-check"><input data-session-setting data-role="orientation" type="checkbox"> Follow palm orientation</label>
      <div data-role="hardware-fields" class="live-hardware" hidden>
        <label class="live-field">UR10e IP<input data-session-setting data-role="host" type="text" placeholder="192.168.0.100"></label>
        <label class="live-field">Confirmed mapping JSON<input data-session-setting data-role="mapping"
          type="text" placeholder="data/live_mapping.json"></label>
        <label class="live-check"><input data-session-setting data-role="execute" type="checkbox"> Enable physical motion on UR10e</label>
      </div>
      <div class="live-actions">
        <button data-role="start" class="primary">Start observing</button>
        <button data-role="bind" class="primary" disabled>Bind &amp; follow</button>
        <button data-role="clutch" disabled>Clutch · release</button>
        <button data-role="stop" class="danger" disabled>Stop</button>
      </div>
      <p class="hint" data-role="action-help"></p>
      <p class="hint" data-role="active-note" hidden>The session continues when you switch tabs.</p>
      <div class="live-error" data-role="error" role="alert"></div>
      <dl class="live-metrics">
        <dt>Session</dt><dd data-role="session">—</dd>
        <dt>Observed points</dt><dd data-role="points">—</dd>
        <dt>Position</dt><dd data-role="position-valid">—</dd>
        <dt>Palm orientation</dt><dd data-role="orientation-valid">—</dd>
        <dt>Source age</dt><dd data-role="age">—</dd>
        <dt>Processing median / p95</dt><dd data-role="processing">—</dd>
        <dt>IK</dt><dd data-role="solver">—</dd>
        <dt>Tracking</dt><dd data-role="tracking">—</dd>
        <dt>Last reason</dt><dd data-role="reason">—</dd>
        <dt>Operator request</dt><dd data-role="request-result">—</dd>
        <dt>Logs</dt><dd data-role="directory">—</dd>
      </dl>
    </aside>
    <section class="live-views">
      <div class="live-view"><div class="live-view-title">Human hand · rig frame <span data-role="hand-position">—</span></div>
        <div class="live-canvas" data-role="human-canvas"></div></div>
      <div class="live-view"><div class="live-view-title"><b data-role="robot-title">MuJoCo · robot-base frame</b>
        <span data-role="robot-position">—</span></div><div class="live-canvas" data-role="robot-canvas"></div></div>
      <p class="hint live-view-help">Drag to orbit · wheel to zoom · RGB axes show palm/TCP orientation · yellow markers show the current and bound anchors.</p>
    </section>`;
  view.appendChild(root);
  const stored = sessionGet('live-form-v1', {});
  const defaults = { reference: '', source: 'recorded', backend: 'mujoco', landmark_mode: 'all21',
    perception_hz: 15, seconds: 60, recorded_speed: 1, scale: .25, host: '', mapping_json: '', orientation: false };
  const values = { ...defaults, ...stored };
  const names = { reference: 'reference', source: 'source', backend: 'backend', landmark_mode: 'landmarks',
    perception_hz: 'rate', seconds: 'seconds', recorded_speed: 'speed', scale: 'scale', host: 'host', mapping_json: 'mapping' };
  for (const [key, field] of Object.entries(names)) role(field).value = values[key];
  role('orientation').checked = Boolean(values.orientation);
  // Enabling motion is an explicit per-session choice, never a saved preference.
  role('execute').checked = false;
  root.addEventListener('change', saveForm);
  root.addEventListener('click', event => {
    const button = event.target.closest('button[data-role]');
    if (button && ['start', 'bind', 'clutch', 'stop'].includes(button.dataset.role)) action(button.dataset.role);
  });
  backendFields();
  try {
    humanScene = createScene(role('human-canvas'), { human: true });
    robotScene = createScene(role('robot-canvas'));
  } catch (error) {
    text('error', '3D preview unavailable: ' + error.message);
  }
  renderStatus(status);
  poll(id);
}

export function unmount() {
  generation += 1;
  clearTimeout(pollTimer);
  pollAbort?.abort(); actionAbort?.abort();
  pollAbort = actionAbort = null;
  humanScene?.dispose(); robotScene?.dispose();
  humanScene = robotScene = null;
  root?.remove(); root = null;
  busy = false;
  // No /stop request: the operator may inspect other panels during following.
}
