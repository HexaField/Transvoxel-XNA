import {
  AmbientLight,
  Color,
  DirectionalLight,
  FogExp2,
  Mesh,
  MeshStandardMaterial,
  PerspectiveCamera,
  Scene,
  WebGLRenderer,
} from "three";
import type { Vector3 } from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransvoxelExtractor, TransvoxelMesher } from "../src/surface-extractor/transvoxel-extractor";
import { Vector3i } from "../src/math/vector3i";
import { meshDataToGeometry } from "./mesh-utils";
import {
  createChunkFieldSampler,
  type DensityFunction,
} from "../src/volume/volume-data";
import {
  QuadChunkManager,
  type ChunkDescriptor,
  type ChunkRequest,
  type ChunkPlan,
  type TransitionFace,
  type LodLevel,
} from "./quadChunks";

const mount = document.querySelector<HTMLDivElement>("#app");
if (!mount) {
  throw new Error("Missing #app mount point.");
}

const statsLabel = document.querySelector<HTMLParagraphElement>("#chunk-stats");

const scene = new Scene();
scene.background = new Color("#03050c");
scene.fog = new FogExp2("#03050c", 0.0012);

const camera = new PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  1500
);
camera.position.set(60, 34, 60);

const renderer = new WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
mount.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 18, 0);
controls.update();

scene.add(new AmbientLight(0xbfd5ff, 0.45));

const sunLight = new DirectionalLight(0xfff2d6, 1.2);
sunLight.position.set(120, 160, 40);
sunLight.castShadow = true;
scene.add(sunLight);

const rimLight = new DirectionalLight(0x5bc0ff, 0.3);
rimLight.position.set(-80, 60, -100);
scene.add(rimLight);

const BLOCK_WIDTH = TransvoxelExtractor.BlockWidth;
const CELL_SIZE = 1;
const LOD_LEVELS: LodLevel[] = [
  { lodIndex: 0, color: 0xff0000, maxDistance: 48 }, // red - fine detail
  { lodIndex: 1, color: 0x00ff00, maxDistance: 120 }, // green - medium detail
  { lodIndex: 2, color: 0x6666ff, maxDistance: 360 }, // blue - coarse detail
];

const MAX_LOD_INDEX = LOD_LEVELS[LOD_LEVELS.length - 1].lodIndex;
const LOD_LOOKUP = new Map(LOD_LEVELS.map((level) => [level.lodIndex, level]));

const chunkWorldSize = (lodIndex: number): number =>
  CELL_SIZE * (BLOCK_WIDTH << lodIndex);
const chunkDiagonalRadius = (lodIndex: number): number =>
  (chunkWorldSize(lodIndex) * Math.SQRT2) / 2;
const chunkKey = (lodIndex: number, chunkX: number, chunkZ: number): string =>
  `${lodIndex}:${chunkX}:${chunkZ}`;

const lerp01 = (a: number, b: number, t: number): number => a + (b - a) * t;
const smoothstep01 = (t: number): number => t * t * (3 - 2 * t);

const hash2dNoise = (x: number, z: number): number => {
  const s = Math.sin(x * 127.1 + z * 311.7) * 43758.5453;
  return s - Math.floor(s);
};

const valueNoise2d = (x: number, z: number): number => {
  const xi = Math.floor(x);
  const zi = Math.floor(z);
  const xf = x - xi;
  const zf = z - zi;

  const v00 = hash2dNoise(xi, zi);
  const v10 = hash2dNoise(xi + 1, zi);
  const v01 = hash2dNoise(xi, zi + 1);
  const v11 = hash2dNoise(xi + 1, zi + 1);

  const u = smoothstep01(xf);
  const v = smoothstep01(zf);

  const top = lerp01(v00, v10, u);
  const bottom = lerp01(v01, v11, u);
  return lerp01(top, bottom, v);
};

const fbm2d = (x: number, z: number): number => {
  let value = 0;
  let amplitude = 1;
  let frequency = 1;
  for (let i = 0; i < 4; i++) {
    value += amplitude * valueNoise2d(x * frequency, z * frequency);
    amplitude *= 0.5;
    frequency *= 2.0;
  }
  return value;
};

const ridgeNoise2d = (x: number, z: number): number => {
  const n = valueNoise2d(x, z);
  return 1 - Math.abs(2 * n - 1);
};

const terrainHeightEstimate = (x: number, z: number): number => {
  const hills = fbm2d(x * 0.04, z * 0.04) * 28;
  const ridges = ridgeNoise2d(x * 0.02, z * 0.02) * 16;
  const dunes = Math.sin(x * 0.01) * 3 + Math.cos(z * 0.012) * 3;
  return 8 + hills + ridges + dunes;
};

const estimateOriginY = (
  lodIndex: number,
  chunkX: number,
  chunkZ: number
): number => {
  const chunkSize = BLOCK_WIDTH << lodIndex;
  const centerX = (chunkX + 0.5) * chunkSize;
  const centerZ = (chunkZ + 0.5) * chunkSize;
  const estimatedSurface = terrainHeightEstimate(centerX, centerZ);
  const verticalPadding = Math.max(16, chunkSize * 0.75);
  return Math.floor(estimatedSurface - verticalPadding);
};

type ChunkWorkerRequest = {
  key: string;
  lodIndex: number;
  chunkX: number;
  chunkZ: number;
  originY: number;
  requestId: number;
};

type ChunkWorkerResponse = {
  key: string;
  minX: number;
  minY: number;
  minZ: number;
  size: number;
  buffer: ArrayBuffer;
  requestId: number;
};

interface ChunkRecord {
  mesh: Mesh;
  material: MeshStandardMaterial;
  key: string;
  transitionHash: string;
  lodIndex: number;
  chunkX: number;
  chunkZ: number;
  color: number;
  originY: number;
}

interface PendingChunkRequest {
  descriptor: ChunkDescriptor;
  requestId: number;
}


const mesher = new TransvoxelMesher();
const chunkWorker: Worker = createChunkWorker();
const activeChunks = new Map<string, ChunkRecord>();
const stagedChunks = new Map<string, ChunkRecord>();
const pendingBuilds = new Map<string, PendingChunkRequest>();
const buildQueue: ChunkRequest[] = [];
let inflightBuilds = 0;
const MAX_INFLIGHT_BUILDS = 1;
const currentCameraXZ = { x: 0, z: 0 };
const chunkScheduler = new QuadChunkManager({
  blockWidth: BLOCK_WIDTH,
  cellSize: CELL_SIZE,
  lodLevels: LOD_LEVELS,
  estimateOriginY,
});

chunkWorker.onmessage = (event: MessageEvent<ChunkWorkerResponse>) => {
  handleWorkerMessage(event.data as ChunkWorkerResponse);
};

function updateTerrain(cameraPosition: Vector3): void {
  currentCameraXZ.x = cameraPosition.x;
  currentCameraXZ.z = cameraPosition.z;
  const plan = chunkScheduler.update({
    x: cameraPosition.x,
    z: cameraPosition.z,
  });
  applyChunkPlan(plan);
}

function applyChunkPlan(plan: ChunkPlan): void {
  for (const key of plan.cancels) {
    removeFromBuildQueue(key);
    pendingBuilds.delete(key);
    discardStagedChunk(key);
  }

  for (const key of plan.releases) {
    promoteChildrenForParent(key);
    discardStagedChunk(key);
    disposeChunk(key);
  }

  for (const request of plan.requests) {
    enqueueBuildRequest(request);
  }

  reorderBuildQueue();
  dispatchBuilds();
  updateTerrainStats();
}

function enqueueBuildRequest(request: ChunkRequest): void {
  const key = request.descriptor.key;
  if (pendingBuilds.has(key)) {
    return;
  }
  removeFromBuildQueue(key);
  buildQueue.push(request);
}

function removeFromBuildQueue(key: string): void {
  const index = buildQueue.findIndex((queued) => queued.descriptor.key === key);
  if (index !== -1) {
    buildQueue.splice(index, 1);
  }
}

function reorderBuildQueue(): void {
  buildQueue.sort((a, b) => {
    const lodDiff = a.descriptor.lodIndex - b.descriptor.lodIndex;
    if (lodDiff !== 0) {
      return lodDiff;
    }
    const distanceDiff =
      chunkRequestDistance(a) - chunkRequestDistance(b);
    if (distanceDiff !== 0) {
      return distanceDiff;
    }
    return a.descriptor.key.localeCompare(b.descriptor.key);
  });
}

function chunkRequestDistance(request: ChunkRequest): number {
  const size = chunkWorldSize(request.descriptor.lodIndex);
  const centerX = (request.descriptor.chunkX + 0.5) * size;
  const centerZ = (request.descriptor.chunkZ + 0.5) * size;
  return Math.hypot(centerX - currentCameraXZ.x, centerZ - currentCameraXZ.z);
}

function dispatchBuilds(): void {
  while (inflightBuilds < MAX_INFLIGHT_BUILDS && buildQueue.length > 0) {
    const request = buildQueue.shift()!;
    pendingBuilds.set(request.descriptor.key, {
      descriptor: request.descriptor,
      requestId: request.requestId,
    });
    const workerRequest: ChunkWorkerRequest = {
      key: request.descriptor.key,
      lodIndex: request.descriptor.lodIndex,
      chunkX: request.descriptor.chunkX,
      chunkZ: request.descriptor.chunkZ,
      originY: request.descriptor.originY,
      requestId: request.requestId,
    };
    chunkWorker.postMessage(workerRequest);
    inflightBuilds++;
  }
}

function handleWorkerMessage(message: ChunkWorkerResponse): void {
  const pending = pendingBuilds.get(message.key);
  pendingBuilds.delete(message.key);
  if (inflightBuilds > 0) {
    inflightBuilds--;
  }
  dispatchBuilds();
  updateTerrainStats();

  if (!pending || pending.requestId !== message.requestId) {
    return;
  }

  const sampler = createChunkFieldSampler({
    data: new Int8Array(message.buffer),
    minX: message.minX,
    minY: message.minY,
    minZ: message.minZ,
    size: message.size,
  });

  const descriptor = pending.descriptor;
  const record = buildChunkRecord(descriptor, sampler);
  const outcome = record ? "mesh" : "empty";

  if (record) {
    stageChunkRecord(record);
  }

  const finalizePlan = chunkScheduler.finalizeRequest({
    descriptor,
    requestId: pending.requestId,
    outcome,
  });
  applyChunkPlan(finalizePlan);
}

function buildChunkRecord(
  descriptor: ChunkDescriptor,
  sampler: DensityFunction
): ChunkRecord | null {
  const cellScale = 1 << descriptor.lodIndex;
  const samplesPerAxis = BLOCK_WIDTH * cellScale;
  const origin = new Vector3i(
    descriptor.chunkX * samplesPerAxis,
    descriptor.originY,
    descriptor.chunkZ * samplesPerAxis
  );

  const meshData = mesher.extractBlock(sampler, {
    origin,
    lodIndex: descriptor.lodIndex,
    cellSize: CELL_SIZE,
    transitionFaces: descriptor.transitionFaces,
  });

  if (meshData.indices.length === 0) {
    return null;
  }

  const built = meshDataToGeometry(meshData);
  const material = new MeshStandardMaterial({
    color: descriptor.color,
    roughness: 0.9,
    metalness: 0.05,
    flatShading: true,
  });
  const mesh = new Mesh(built.geometry, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;

  return {
    mesh,
    material,
    key: descriptor.key,
    transitionHash: transitionSignature(descriptor.transitionFaces),
    lodIndex: descriptor.lodIndex,
    chunkX: descriptor.chunkX,
    chunkZ: descriptor.chunkZ,
    color: descriptor.color,
    originY: descriptor.originY,
  };
}

function addChunkToScene(record: ChunkRecord): void {
  scene.add(record.mesh);
  activeChunks.set(record.key, record);
}

function disposeChunk(key: string): void {
  const record = activeChunks.get(key);
  if (!record) {
    return;
  }

  releaseChunkResources(record);
  activeChunks.delete(key);
}

function releaseChunkResources(record: ChunkRecord): void {
  scene.remove(record.mesh);
  record.mesh.geometry.dispose();
  record.material.dispose();
}

function stageChunkRecord(record: ChunkRecord): void {
  stagedChunks.set(record.key, record);
  if (!shouldHoldForParent(record)) {
    showStagedChunk(record.key);
  }
}

function promoteChildrenForParent(parentKey: string): void {
  const parentInfo = parseChunkKey(parentKey);
  if (!parentInfo || parentInfo.lodIndex === 0) {
    return;
  }
  const childKeys = childKeysForParent(parentInfo);
  if (childKeys.every((key) => stagedChunks.has(key))) {
    childKeys.forEach((key) => showStagedChunk(key));
  }
}

function showStagedChunk(key: string): void {
  const record = stagedChunks.get(key);
  if (!record) {
    return;
  }
  stagedChunks.delete(key);
  addChunkToScene(record);
}

function discardStagedChunk(key: string): void {
  const record = stagedChunks.get(key);
  if (!record) {
    return;
  }
  stagedChunks.delete(key);
  releaseChunkResources(record);
}

function shouldHoldForParent(record: ChunkRecord): boolean {
  const parentKey = parentKeyFor(record);
  if (!parentKey) {
    return false;
  }
  return activeChunks.has(parentKey);
}

function parentKeyFor(record: ChunkRecord): string | null {
  if (record.lodIndex >= MAX_LOD_INDEX) {
    return null;
  }
  const parentLod = record.lodIndex + 1;
  const parentX = Math.floor(record.chunkX / 2);
  const parentZ = Math.floor(record.chunkZ / 2);
  return chunkKey(parentLod, parentX, parentZ);
}

function parseChunkKey(
  key: string
): { lodIndex: number; chunkX: number; chunkZ: number } | null {
  const parts = key.split(":");
  if (parts.length !== 3) {
    return null;
  }
  const lodIndex = Number(parts[0]);
  const chunkX = Number(parts[1]);
  const chunkZ = Number(parts[2]);
  if (Number.isNaN(lodIndex) || Number.isNaN(chunkX) || Number.isNaN(chunkZ)) {
    return null;
  }
  return { lodIndex, chunkX, chunkZ };
}

function childKeysForParent(parent: {
  lodIndex: number;
  chunkX: number;
  chunkZ: number;
}): string[] {
  if (parent.lodIndex === 0) {
    return [];
  }
  const childLod = parent.lodIndex - 1;
  const baseX = parent.chunkX * 2;
  const baseZ = parent.chunkZ * 2;
  return [
    chunkKey(childLod, baseX, baseZ),
    chunkKey(childLod, baseX + 1, baseZ),
    chunkKey(childLod, baseX, baseZ + 1),
    chunkKey(childLod, baseX + 1, baseZ + 1),
  ];
}

function updateTerrainStats(): void {
  if (statsLabel) {
    const building = pendingBuilds.size + buildQueue.length;
    statsLabel.textContent = `${activeChunks.size} active chunks (${building} building)`;
  }
}

function transitionSignature(faces: TransitionFace[]): string {
  if (faces.length === 0) {
    return "none";
  }
  return faces.slice().sort().join(",");
}
const animate = () => {
  controls.update();
  updateTerrain(controls.target);
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
};

requestAnimationFrame(animate);

window.addEventListener("resize", () => {
  const { innerWidth, innerHeight } = window;
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
});

function createChunkWorker(): Worker {
  const workerSource = `
    const BLOCK_WIDTH = ${TransvoxelExtractor.BlockWidth};

    self.onmessage = (event) => {
      const { key, lodIndex, chunkX, chunkZ, originY, requestId } = event.data;
      const range = BLOCK_WIDTH << lodIndex;
      const transitionReach = lodIndex === 0 ? 1 : ((1 << (lodIndex - 1)) * 2 + 1);
      const padding = Math.max(1, transitionReach);
      const sampleSize = range + padding * 2 + 1;
      const minX = chunkX * range - padding;
      const minY = originY - padding;
      const minZ = chunkZ * range - padding;
      const totalSamples = sampleSize * sampleSize * sampleSize;
      const data = new Int8Array(totalSamples);
      let index = 0;

      for (let z = 0; z < sampleSize; z++) {
        const worldZ = minZ + z;
        for (let y = 0; y < sampleSize; y++) {
          const worldY = minY + y;
          for (let x = 0; x < sampleSize; x++) {
            const worldX = minX + x;
            data[index++] = sampleTerrainDensity(worldX, worldY, worldZ);
          }
        }
      }

      self.postMessage({ key, minX, minY, minZ, size: sampleSize, buffer: data.buffer, requestId }, [data.buffer]);
    };

    function sampleTerrainDensity(x, y, z) {
      const height = terrainHeight(x, z);
      const strata = Math.sin((x + z) * 0.05) * 2.5;
      const density = height + strata - y;
      return clampToByte(density * 6);
    }

    function terrainHeight(x, z) {
      const hills = fbm(x * 0.04, z * 0.04) * 28;
      const ridges = ridgeNoise(x * 0.02, z * 0.02) * 16;
      const dunes = Math.sin(x * 0.01) * 3 + Math.cos(z * 0.012) * 3;
      return 8 + hills + ridges + dunes;
    }

    function fbm(x, z) {
      let value = 0;
      let amplitude = 1;
      let frequency = 1;
      for (let i = 0; i < 4; i++) {
        value += amplitude * valueNoise(x * frequency, z * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
      }
      return value;
    }

    function ridgeNoise(x, z) {
      const n = valueNoise(x, z);
      return 1 - Math.abs(2 * n - 1);
    }

    function valueNoise(x, z) {
      const xi = Math.floor(x);
      const zi = Math.floor(z);
      const xf = x - xi;
      const zf = z - zi;

      const v00 = hash2d(xi, zi);
      const v10 = hash2d(xi + 1, zi);
      const v01 = hash2d(xi, zi + 1);
      const v11 = hash2d(xi + 1, zi + 1);

      const u = smoothstep(xf);
      const v = smoothstep(zf);

      const top = lerp(v00, v10, u);
      const bottom = lerp(v01, v11, u);
      return lerp(top, bottom, v);
    }

    function hash2d(x, z) {
      const s = Math.sin(x * 127.1 + z * 311.7) * 43758.5453;
      return s - Math.floor(s);
    }

    const smoothstep = (t) => t * t * (3 - 2 * t);
    const lerp = (a, b, t) => a + (b - a) * t;

    function clampToByte(value) {
      return Math.max(-127, Math.min(127, Math.floor(value)));
    }
  `;

  const blob = new Blob([workerSource], { type: "application/javascript" });
  const url = URL.createObjectURL(blob);
  const worker = new Worker(url, { name: "terrain-chunk-worker" });
  URL.revokeObjectURL(url);
  return worker;
}
