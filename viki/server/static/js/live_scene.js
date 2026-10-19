// Small browser-only skeleton views. Display never participates in control.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

const HAND_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
  [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const finitePoint = p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite);
const finiteRotation = r => Array.isArray(r) && r.length === 3 && r.every(finitePoint);

export function create(container, { human = false } = {}) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0d10);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.005, 30);
  camera.up.set(0, 0, 1);
  camera.position.set(1.8, -2.4, 1.6);
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'low-power' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  container.appendChild(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, .5);
  controls.minDistance = .08;
  controls.maxDistance = 12;
  const grid = new THREE.GridHelper(2, 20, 0x344052, 0x1e2632);
  grid.rotation.x = Math.PI / 2;
  const originAxes = new THREE.AxesHelper(.15);
  scene.add(grid, originAxes);
  scene.add(new THREE.AmbientLight(0xffffff, 2));
  const light = new THREE.DirectionalLight(0xffffff, 2);
  light.position.set(1, -2, 3);
  scene.add(light);

  const group = new THREE.Group();
  scene.add(group);
  const sphere = new THREE.SphereGeometry(1, 10, 8);
  const cylinder = new THREE.CylinderGeometry(1, 1, 1, 8);
  const material = new THREE.MeshStandardMaterial({ color: human ? 0x4ade80 : 0x7da3ff });
  const targetMaterial = new THREE.MeshStandardMaterial({ color: 0xfbbf24 });
  const joints = Array.from({ length: 32 }, () => {
    const object = new THREE.Mesh(sphere, material);
    object.visible = false;
    group.add(object);
    return object;
  });
  const bones = Array.from({ length: 32 }, () => {
    const object = new THREE.Mesh(cylinder, material);
    object.visible = false;
    group.add(object);
    return object;
  });
  const marker = new THREE.Mesh(sphere, targetMaterial);
  marker.scale.setScalar(human ? .006 : .018);
  marker.visible = false;
  group.add(marker);
  const axes = new THREE.AxesHelper(human ? .06 : .12);
  axes.visible = false;
  group.add(axes);
  const bindingMarker = new THREE.Mesh(sphere, targetMaterial);
  bindingMarker.scale.setScalar(human ? .004 : .010);
  bindingMarker.visible = false;
  scene.add(bindingMarker);
  const a = new THREE.Vector3(), b = new THREE.Vector3(), direction = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0);
  let centered = !human, disposed = false, dirty = true, animation = 0;
  const changed = () => { dirty = true; };
  controls.addEventListener('change', changed);
  const resize = new ResizeObserver(() => {
    if (disposed) return;
    const width = Math.max(1, container.clientWidth), height = Math.max(1, container.clientHeight);
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    dirty = true;
  });
  resize.observe(container);

  function render() {
    if (disposed) return;
    animation = requestAnimationFrame(render);
    if (dirty) {
      controls.update();
      renderer.render(scene, camera);
      dirty = false;
    }
  }
  render();

  function update({ points = [], observed = [], edges = [], position = null,
    rotation = null, poseValid = false, rotationValid = false, binding = null } = {}) {
    const accepted = index => finitePoint(points[index]) && (!human || observed[index] === true);
    for (let i = 0; i < joints.length; i++) {
      joints[i].visible = accepted(i);
      if (joints[i].visible) {
        joints[i].position.fromArray(points[i]);
        joints[i].scale.setScalar(human ? .0045 : .018);
      }
    }
    const links = human ? HAND_EDGES : edges;
    for (let i = 0; i < bones.length; i++) {
      const link = links[i];
      const valid = Array.isArray(link) && link.length === 2 && link.every(Number.isInteger)
        && accepted(link[0]) && accepted(link[1]);
      bones[i].visible = valid;
      if (!valid) continue;
      a.fromArray(points[link[0]]); b.fromArray(points[link[1]]);
      direction.subVectors(b, a);
      const length = direction.length();
      if (length < 1e-8) { bones[i].visible = false; continue; }
      bones[i].position.copy(a).add(b).multiplyScalar(.5);
      bones[i].quaternion.setFromUnitVectors(up, direction.multiplyScalar(1 / length));
      const radius = human ? .0025 : .013;
      bones[i].scale.set(radius, length, radius);
    }
    marker.visible = poseValid && finitePoint(position);
    axes.visible = marker.visible && rotationValid && finiteRotation(rotation);
    if (marker.visible) marker.position.fromArray(position);
    if (axes.visible) {
      axes.position.fromArray(position);
      const m = new THREE.Matrix4().set(
        rotation[0][0], rotation[0][1], rotation[0][2], 0,
        rotation[1][0], rotation[1][1], rotation[1][2], 0,
        rotation[2][0], rotation[2][1], rotation[2][2], 0, 0, 0, 0, 1);
      axes.quaternion.setFromRotationMatrix(m);
    }
    bindingMarker.visible = finitePoint(binding);
    if (bindingMarker.visible) bindingMarker.position.fromArray(binding);
    if (!centered && finitePoint(position)) {
      controls.target.fromArray(position);
      camera.position.copy(controls.target).add(new THREE.Vector3(.35, -.40, .30));
      controls.update();
      centered = true;
    }
    dirty = true;
  }

  return { update, dispose() {
    disposed = true;
    cancelAnimationFrame(animation);
    resize.disconnect();
    controls.removeEventListener('change', changed);
    controls.dispose();
    sphere.dispose(); cylinder.dispose(); material.dispose(); targetMaterial.dispose();
    for (const object of [grid, originAxes, axes]) {
      object.geometry?.dispose();
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      for (const item of materials) item?.dispose();
    }
    renderer.dispose();
    renderer.forceContextLoss();
    renderer.domElement.remove();
  } };
}
