import {
  AmbientLight,
  BufferGeometry,
  Color,
  DirectionalLight,
  DoubleSide,
  Float32BufferAttribute,
  Mesh,
  MeshStandardMaterial,
  PerspectiveCamera,
  Scene,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransvoxelMesher, TransitionFace } from "../src/surface-extractor/transvoxel-extractor";
import { FunctionalVolume } from "../src/volume/volume-data";
import { Vector3i } from "../src/math/vector3i";

const mount = document.querySelector<HTMLDivElement>("#app");
if (!mount) {
  throw new Error("Mount element with id \"app\" is missing.");
}

const lodIndex = 1;
const blockExtent = 16 * (1 << lodIndex);
const blockCenter = blockExtent * 0.5;

const scene = new Scene();
scene.background = new Color("#04060F");

const camera = new PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 500);
camera.position.set(blockExtent * 1.8, blockExtent * 1.4, blockExtent * 1.8);

const renderer = new WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
mount.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(blockCenter, blockCenter, blockCenter);
controls.enableDamping = true;
controls.update();

const keyLight = new DirectionalLight(0xffffff, 1.2);
keyLight.position.set(blockExtent, blockExtent * 1.2, blockExtent);
scene.add(keyLight);

const rimLight = new DirectionalLight(0x6ab0ff, 0.5);
rimLight.position.set(-blockExtent, blockExtent * 0.5, -blockExtent * 1.2);
scene.add(rimLight);

scene.add(new AmbientLight(0x0f1425, 0.9));

const volume = new FunctionalVolume((x, y, z) => {
  const nx = (x - blockCenter) / blockExtent;
  const ny = (y - blockCenter) / blockExtent;
  const nz = (z - blockCenter) / blockExtent;
  const sphere = nx * nx + ny * ny + nz * nz - 0.18;
  const folds = Math.sin(nx * 8.0) * 0.35 + Math.cos(ny * 6.0) * 0.35 + Math.sin(nz * 7.0) * 0.35;
  const density = sphere + 0.25 * folds;
  return Math.floor(density * 127);
});

const mesher = new TransvoxelMesher(volume);
const transitionFaces: TransitionFace[] = [
  "negativeX",
  "positiveX",
  "negativeY",
  "positiveY",
  "negativeZ",
  "positiveZ",
];

const meshData = mesher.extractBlock({
  origin: new Vector3i(0, 0, 0),
  lodIndex,
  cellSize: 1,
  transitionFaces,
});

const geometry = new BufferGeometry();
const vertexCount = meshData.vertices.length;
const positions = new Float32Array(vertexCount * 3);
const normals = new Float32Array(vertexCount * 3);

meshData.vertices.forEach((vertex, index) => {
  const base = index * 3;
  positions[base] = vertex.primary.x;
  positions[base + 1] = vertex.primary.y;
  positions[base + 2] = vertex.primary.z;

  normals[base] = vertex.normal.x;
  normals[base + 1] = vertex.normal.y;
  normals[base + 2] = vertex.normal.z;
});

geometry.setAttribute("position", new Float32BufferAttribute(positions, 3));
geometry.setAttribute("normal", new Float32BufferAttribute(normals, 3));
geometry.setIndex(meshData.indices);
geometry.computeBoundingSphere();

const material = new MeshStandardMaterial({
  color: 0x5bc7ff,
  metalness: 0.08,
  roughness: 0.42,
  side: DoubleSide,
});

const mesh = new Mesh(geometry, material);
scene.add(mesh);

const animate = (time: number) => {
  mesh.rotation.y = time * 0.00018;
  controls.update();
  renderer.render(scene, camera);
};

renderer.setAnimationLoop(animate);

window.addEventListener("resize", () => {
  const { innerWidth, innerHeight } = window;
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
});
