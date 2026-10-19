// One rig-frame scene. Geometry arrives from a recorded causal run only.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

const EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],[0,17]];
const point = p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite);

export function create(container) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0d10);
  const camera = new THREE.PerspectiveCamera(45, 1, .005, 100);
  camera.up.set(0, 0, 1);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  container.appendChild(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.minDistance = .04;
  const content = new THREE.Group(); scene.add(content);
  const grid = new THREE.GridHelper(4, 40, 0x344052, 0x1e2632);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid, new THREE.AxesHelper(.2), new THREE.AmbientLight(0xffffff, 2));
  const light = new THREE.DirectionalLight(0xffffff, 2);
  light.position.set(1, -2, 3); scene.add(light);
  const cloud = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({
    size: .004, vertexColors: true, sizeAttenuation: true }));
  cloud.frustumCulled = false; content.add(cloud);
  const sphere = new THREE.SphereGeometry(1, 10, 8);
  const cylinder = new THREE.CylinderGeometry(1, 1, 1, 8);
  function skeleton(color, count, radius) {
    const material = new THREE.MeshStandardMaterial({ color });
    const joints = Array.from({ length: count }, () => {
      const m = new THREE.Mesh(sphere, material); m.visible = false;
      m.scale.setScalar(radius); content.add(m); return m;
    });
    const bones = Array.from({ length: 32 }, () => {
      const m = new THREE.Mesh(cylinder, material); m.visible = false;
      content.add(m); return m;
    });
    return { joints, bones };
  }
  const hand = skeleton(0xffd166, 21, .0045);
  const robot = skeleton(0x8aafff, 32, .018);
  const handAxes = new THREE.AxesHelper(.07), tcpAxes = new THREE.AxesHelper(.1);
  content.add(handAxes, tcpAxes);
  function marker(color, radius) {
    const mesh = new THREE.Mesh(sphere, new THREE.MeshBasicMaterial({ color }));
    mesh.scale.setScalar(radius); mesh.visible = false; content.add(mesh); return mesh;
  }
  const target = marker(0xf472b6, .012), actual = marker(0x22d3ee, .01);
  function trail(color) {
    const line = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color }));
    line.frustumCulled = false; content.add(line); return line;
  }
  const actualTrail = trail(0x22d3ee), targetTrail = trail(0xf472b6);
  const a = new THREE.Vector3(), b = new THREE.Vector3(), direction = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0);
  let metadata = null, dirty = true, disposed = false, animation = 0;
  const visible = { cloud: true, hand: true, robot: true, trails: true };
  const changed = () => { dirty = true; };
  controls.addEventListener('change', changed);
  function drawSkeleton(objects, points, edges, radius) {
    objects.joints.forEach((mesh, i) => {
      mesh.visible = point(points?.[i]);
      if (mesh.visible) mesh.position.fromArray(points[i]);
    });
    objects.bones.forEach((mesh, i) => {
      const edge = edges?.[i];
      mesh.visible = Boolean(edge && point(points?.[edge[0]]) && point(points?.[edge[1]]));
      if (!mesh.visible) return;
      a.fromArray(points[edge[0]]); b.fromArray(points[edge[1]]); direction.subVectors(b, a);
      const length = direction.length();
      if (length < 1e-9) { mesh.visible = false; return; }
      mesh.position.copy(a).add(b).multiplyScalar(.5);
      mesh.quaternion.setFromUnitVectors(up, direction.multiplyScalar(1 / length));
      mesh.scale.set(radius, length, radius);
    });
  }
  function axes(object, position, rotation, valid) {
    object.visible = Boolean(valid && point(position) && Array.isArray(rotation)
      && rotation.length === 3 && rotation.every(point));
    if (!object.visible) return;
    object.position.fromArray(position);
    object.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(
      ...rotation[0], 0, ...rotation[1], 0, ...rotation[2], 0, 0, 0, 0, 1));
  }
  function path(line, positions, index) {
    const vertices = [];
    for (let i = 1; i <= index; i++) {
      if (point(positions?.[i - 1]) && point(positions?.[i])) vertices.push(...positions[i - 1], ...positions[i]);
    }
    line.geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    line.visible = visible.trails;
  }
  function fit() {
    const box = new THREE.Box3();
    const initial = metadata?.robot_initial_points_rig;
    if (initial?.length) {
      initial.filter(point).forEach(p => box.expandByPoint(new THREE.Vector3(...p)));
      if(point(metadata?.hand_anchor_rig))box.expandByPoint(new THREE.Vector3(...metadata.hand_anchor_rig));
      box.expandByScalar(.15);
    } else if (metadata?.cloud?.bounds?.length === 6) {
      const bounds = metadata.cloud.bounds;
      box.expandByPoint(new THREE.Vector3(...bounds.slice(0, 3)));
      box.expandByPoint(new THREE.Vector3(...bounds.slice(3)));
    }
    if (point(metadata?.robot_base_rig)) box.expandByPoint(new THREE.Vector3(...metadata.robot_base_rig));
    if (box.isEmpty()) { box.min.set(-1,-1,0); box.max.set(1,1,1.5); }
    box.applyMatrix4(content.matrix);
    const center = box.getCenter(new THREE.Vector3());
    const span = Math.max(.5, box.getSize(new THREE.Vector3()).length());
    controls.target.copy(center);
    camera.position.copy(center).add(new THREE.Vector3(span*.7, -span*.9, span*.6));
    controls.update(); dirty = true;
  }
  function setMetadata(value) {
    metadata = value;
    content.matrixAutoUpdate = false;
    if(value?.display_transform)content.matrix.set(...value.display_transform.flat());
    else content.matrix.identity();
    content.matrixWorldNeedsUpdate = true;
    fit();
  }
  function focusHand() {
    if(!point(metadata?.hand_anchor_rig))return;
    const center = new THREE.Vector3(...metadata.hand_anchor_rig).applyMatrix4(content.matrix);
    controls.target.copy(center);
    camera.position.copy(center).add(new THREE.Vector3(.35,-.45,.3));
    controls.update(); dirty=true;
  }
  function setFrame(value) {
    const h = value.hand, r = value.robot;
    const points = h?.visible && visible.hand
      ? h.points.map((p, i) => h.observed[i] ? p : null) : [];
    drawSkeleton(hand, points, EDGES, .0025);
    drawSkeleton(robot, visible.robot ? r?.points : [], metadata?.robot_edges, .012);
    axes(handAxes, h?.position, h?.rotation, visible.hand && h?.visible && h?.orientation_valid);
    axes(tcpAxes, r?.tcp_position, r?.tcp_rotation, visible.robot);
    actual.visible = visible.robot && point(r?.tcp_position);
    target.visible = visible.robot && point(r?.target_position);
    if (actual.visible) actual.position.fromArray(r.tcp_position);
    if (target.visible) target.position.fromArray(r.target_position);
    path(actualTrail, metadata?.actual_trajectory, r?.sample_index ?? -1);
    path(targetTrail, metadata?.target_trajectory, r?.sample_index ?? -1);
    dirty = true;
  }
  function setCloud(buffer) {
    if (!buffer) { cloud.geometry.setDrawRange(0, 0); dirty = true; return; }
    const count = new DataView(buffer).getInt32(0, true);
    if (count < 0 || buffer.byteLength !== 4 + count * 15) throw Error('Invalid point-cloud frame');
    cloud.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(buffer, 4, count*3), 3));
    cloud.geometry.setAttribute('color', new THREE.BufferAttribute(new Uint8Array(buffer, 4+count*12, count*3), 3, true));
    cloud.geometry.setDrawRange(0, count); cloud.visible = visible.cloud; dirty = true;
  }
  const resize = new ResizeObserver(() => {
    const width = Math.max(1, container.clientWidth), height = Math.max(1, container.clientHeight);
    renderer.setSize(width, height); camera.aspect = width / height;
    camera.updateProjectionMatrix(); dirty = true;
  });
  resize.observe(container);
  function render() {
    if (disposed) return;
    animation = requestAnimationFrame(render);
    if (dirty) { renderer.render(scene, camera); dirty = false; }
  }
  render(); fit();
  return { setMetadata, setFrame, setCloud, fit, focusHand,
    setLayer(name, on) { visible[name] = on; cloud.visible = visible.cloud; dirty = true; },
    dispose() {
      disposed = true; cancelAnimationFrame(animation); resize.disconnect();
      controls.removeEventListener('change', changed); controls.dispose();
      const geometries = new Set(), materials = new Set();
      scene.traverse(o => { if(o.geometry)geometries.add(o.geometry); if(o.material)materials.add(o.material); });
      geometries.forEach(g => g.dispose()); materials.forEach(m => m.dispose());
      renderer.dispose(); renderer.forceContextLoss(); renderer.domElement.remove();
    },
  };
}
