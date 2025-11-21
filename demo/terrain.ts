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
import {
  TransvoxelExtractor,
  TransvoxelMesher,
  type TransitionFace,
} from "../src/surface-extractor/transvoxel-extractor";
import { Vector3i } from "../src/math/vector3i";
import { Vector3f } from "../src/math/vector3f";
import { meshDataToGeometry } from "./mesh-utils";
import {
  createChunkFieldSampler,
  type DensityFunction,
} from "../src/volume/volume-data";

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
const CHUNK_ORIGIN_Y = 0;

interface LodLevel {
  lodIndex: number;
  color: number;
  maxDistance: number;
}

const LOD_LEVELS: LodLevel[] = [
  { lodIndex: 0, color: 0xff0000, maxDistance: 48 }, // red - fine detail
  { lodIndex: 1, color: 0x00ff00, maxDistance: 120 }, // green - medium detail
  { lodIndex: 2, color: 0x6666ff, maxDistance: 260 }, // blue - coarse detail
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

type ChunkWorkerRequest = {
  key: string;
  lodIndex: number;
  chunkX: number;
  chunkZ: number;
  originY: number;
};

type ChunkWorkerResponse = {
  key: string;
  minX: number;
  minY: number;
  minZ: number;
  size: number;
  buffer: ArrayBuffer;
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

interface ChunkDescriptor {
  lodIndex: number;
  chunkX: number;
  chunkZ: number;
  key: string;
  color: number;
  transitionFaces: TransitionFace[];
  originY: number;
}

interface QuadNode {
  lodIndex: number;
  chunkX: number;
  chunkZ: number;
}

interface CoverageCell {
  ownerKey: string;
  lodIndex: number;
}

class ChunkManager {
  private readonly mesher = new TransvoxelMesher();
  private readonly worker = createChunkWorker();
  private readonly chunks = new Map<string, ChunkRecord>();
  private readonly pending = new Map<string, ChunkDescriptor>();
  private readonly lastDesiredDescriptors = new Map<string, string>();
  private lastCameraPosition: { x: number; z: number } | null = null;

  constructor(
    private readonly root: Scene,
    private readonly statsTarget: HTMLParagraphElement | null
  ) {
    this.worker.onmessage = (event: MessageEvent<ChunkWorkerResponse>) => {
      this.handleWorkerMessage(event.data as ChunkWorkerResponse);
    };
  }

  update(cameraPosition: Vector3): void {
    this.lastCameraPosition = { x: cameraPosition.x, z: cameraPosition.z };
    const desired = this.collectDesiredChunks(cameraPosition);
    if (this.hasDesiredChanged(desired)) {
      this.reconcileChunks(desired);
      this.captureDesiredDescriptors(desired);
    }
    this.updateStats();
  }

  private collectDesiredChunks(
    cameraPosition: Vector3
  ): Map<string, ChunkDescriptor> {
    const desired = new Map<string, ChunkDescriptor>();
    const coverage = new Map<string, CoverageCell>();
    const queue = this.buildInitialNodes(cameraPosition);

    while (queue.length) {
      const node = queue.pop()!;
      const level = LOD_LOOKUP.get(node.lodIndex);
      if (!level) {
        continue;
      }

      const center = this.chunkCenter(node);
      const distance = Math.hypot(
        center.x - cameraPosition.x,
        center.z - cameraPosition.z
      );
      const radius = chunkDiagonalRadius(node.lodIndex);

      if (distance - radius > level.maxDistance) {
        continue;
      }

      if (this.shouldSubdivide(node, distance, radius)) {
        this.subdivideNode(node).forEach((child) => queue.push(child));
        continue;
      }

      const key = chunkKey(node.lodIndex, node.chunkX, node.chunkZ);
      desired.set(key, {
        lodIndex: node.lodIndex,
        chunkX: node.chunkX,
        chunkZ: node.chunkZ,
        key,
        color: level.color,
        transitionFaces: [],
        originY: this.estimateOriginY(node.lodIndex, node.chunkX, node.chunkZ),
      });
      this.markCoverage(coverage, node.chunkX, node.chunkZ, node.lodIndex, key);
    }

    this.assignTransitionFaces(desired, coverage);
    return desired;
  }

  private buildInitialNodes(cameraPosition: Vector3): QuadNode[] {
    const nodes: QuadNode[] = [];
    const coarseSize = chunkWorldSize(MAX_LOD_INDEX);
    const radiusChunks =
      Math.ceil(LOD_LEVELS[LOD_LEVELS.length - 1].maxDistance / coarseSize) + 2;
    const baseX = Math.floor(cameraPosition.x / coarseSize);
    const baseZ = Math.floor(cameraPosition.z / coarseSize);

    for (let dx = -radiusChunks; dx <= radiusChunks; dx++) {
      for (let dz = -radiusChunks; dz <= radiusChunks; dz++) {
        nodes.push({
          lodIndex: MAX_LOD_INDEX,
          chunkX: baseX + dx,
          chunkZ: baseZ + dz,
        });
      }
    }

    return nodes;
  }

  private shouldSubdivide(
    node: QuadNode,
    distance: number,
    radius: number
  ): boolean {
    if (node.lodIndex === 0) {
      return false;
    }

    const childLevel = LOD_LOOKUP.get(node.lodIndex - 1);
    if (!childLevel) {
      return false;
    }

    return distance - radius < childLevel.maxDistance;
  }

  private subdivideNode(node: QuadNode): QuadNode[] {
    const childLod = node.lodIndex - 1;
    const baseX = node.chunkX * 2;
    const baseZ = node.chunkZ * 2;
    return [
      { lodIndex: childLod, chunkX: baseX, chunkZ: baseZ },
      { lodIndex: childLod, chunkX: baseX + 1, chunkZ: baseZ },
      { lodIndex: childLod, chunkX: baseX, chunkZ: baseZ + 1 },
      { lodIndex: childLod, chunkX: baseX + 1, chunkZ: baseZ + 1 },
    ];
  }

  private chunkCenter(node: QuadNode): { x: number; z: number } {
    const size = chunkWorldSize(node.lodIndex);
    return {
      x: (node.chunkX + 0.5) * size,
      z: (node.chunkZ + 0.5) * size,
    };
  }

  private estimateOriginY(
    lodIndex: number,
    chunkX: number,
    chunkZ: number
  ): number {
    const size = chunkWorldSize(lodIndex);
    const minX = chunkX * size;
    const minZ = chunkZ * size;
    const samplePoints: Array<{ x: number; z: number }> = [
      { x: minX, z: minZ },
      { x: minX + size, z: minZ },
      { x: minX, z: minZ + size },
      { x: minX + size, z: minZ + size },
      { x: minX + size * 0.5, z: minZ + size * 0.5 },
    ];

    let minHeight = Infinity;
    let maxHeight = -Infinity;
    for (const point of samplePoints) {
      const height = terrainHeightEstimate(point.x, point.z);
      if (height < minHeight) {
        minHeight = height;
      }
      if (height > maxHeight) {
        maxHeight = height;
      }
    }

    if (!Number.isFinite(minHeight) || !Number.isFinite(maxHeight)) {
      return CHUNK_ORIGIN_Y;
    }

    const margin = Math.min(size * 0.25, 12);
    const desiredMin = minHeight - margin;
    const desiredMax = maxHeight + margin;
    const span = desiredMax - desiredMin;

    if (span <= size) {
      return Math.floor(desiredMin);
    }

    const shift = (span - size) * 0.5;
    return Math.floor(desiredMin + shift);
  }

  private markCoverage(
    coverage: Map<string, CoverageCell>,
    chunkX: number,
    chunkZ: number,
    lodIndex: number,
    ownerKey: string
  ): void {
    const scale = 1 << lodIndex;
    const startX = chunkX * scale;
    const startZ = chunkZ * scale;
    for (let x = 0; x < scale; x++) {
      for (let z = 0; z < scale; z++) {
        const key = this.baseCoverageKey(startX + x, startZ + z);
        coverage.set(key, { ownerKey, lodIndex });
      }
    }
  }

  private assignTransitionFaces(
    desired: Map<string, ChunkDescriptor>,
    coverage: Map<string, CoverageCell>
  ): void {
    for (const descriptor of desired.values()) {
      if (descriptor.lodIndex === 0) {
        descriptor.transitionFaces = [];
        continue;
      }

      const faces: TransitionFace[] = [];
      if (this.hasHigherDetailNeighbor(descriptor, coverage, "x", -1)) {
        faces.push("negativeX");
      }
      if (this.hasHigherDetailNeighbor(descriptor, coverage, "x", 1)) {
        faces.push("positiveX");
      }
      if (this.hasHigherDetailNeighbor(descriptor, coverage, "z", -1)) {
        faces.push("negativeZ");
      }
      if (this.hasHigherDetailNeighbor(descriptor, coverage, "z", 1)) {
        faces.push("positiveZ");
      }

      descriptor.transitionFaces = faces;
    }
  }

  private hasHigherDetailNeighbor(
    descriptor: ChunkDescriptor,
    coverage: Map<string, CoverageCell>,
    axis: "x" | "z",
    direction: -1 | 1
  ): boolean {
    const scale = 1 << descriptor.lodIndex;
    const startX = descriptor.chunkX * scale;
    const startZ = descriptor.chunkZ * scale;
    const endX = startX + scale - 1;
    const endZ = startZ + scale - 1;

    if (axis === "x") {
      const neighborX = direction === -1 ? startX - 1 : endX + 1;
      for (let z = startZ; z <= endZ; z++) {
        const cell = coverage.get(this.baseCoverageKey(neighborX, z));
        if (cell && cell.lodIndex < descriptor.lodIndex) {
          return true;
        }
      }
    } else {
      const neighborZ = direction === -1 ? startZ - 1 : endZ + 1;
      for (let x = startX; x <= endX; x++) {
        const cell = coverage.get(this.baseCoverageKey(x, neighborZ));
        if (cell && cell.lodIndex < descriptor.lodIndex) {
          return true;
        }
      }
    }

    return false;
  }

  private baseCoverageKey(x: number, z: number): string {
    return `${x}:${z}`;
  }

  private reconcileChunks(desired: Map<string, ChunkDescriptor>): void {
    for (const existingKey of Array.from(this.chunks.keys())) {
      if (!desired.has(existingKey)) {
        const record = this.chunks.get(existingKey);
        if (!record) {
          continue;
        }

        const replacement = this.findOverlappingDescriptor(record, desired);
        if (replacement) {
          if (replacement.lodIndex !== record.lodIndex) {
            this.disposeChunk(existingKey);
          }
          continue;
        }

        if (this.shouldCullChunk(record)) {
          this.disposeChunk(existingKey);
        }
      }
    }

    for (const pendingKey of Array.from(this.pending.keys())) {
      if (!desired.has(pendingKey)) {
        this.pending.delete(pendingKey);
      }
    }

    for (const descriptor of desired.values()) {
      const desiredHash = this.transitionSignature(descriptor.transitionFaces);
      const existing = this.chunks.get(descriptor.key);
      if (existing) {
        if (existing.transitionHash !== desiredHash) {
          this.disposeChunk(descriptor.key);
        } else {
          continue;
        }
      }

      const pendingDescriptor = this.pending.get(descriptor.key);
      if (pendingDescriptor) {
        const pendingHash = this.transitionSignature(
          pendingDescriptor.transitionFaces
        );
        if (pendingHash !== desiredHash) {
          this.pending.set(descriptor.key, {
            ...descriptor,
            transitionFaces: [...descriptor.transitionFaces],
          });
        }
        continue;
      }
      this.requestChunk(descriptor);
    }
  }

  private hasDesiredChanged(desired: Map<string, ChunkDescriptor>): boolean {
    if (desired.size !== this.lastDesiredDescriptors.size) {
      return true;
    }

    for (const [key, descriptor] of desired.entries()) {
      const signature = this.transitionSignature(descriptor.transitionFaces);
      if (this.lastDesiredDescriptors.get(key) !== signature) {
        return true;
      }
    }

    return false;
  }

  private captureDesiredDescriptors(
    desired: Map<string, ChunkDescriptor>
  ): void {
    this.lastDesiredDescriptors.clear();
    for (const [key, descriptor] of desired.entries()) {
      this.lastDesiredDescriptors.set(
        key,
        this.transitionSignature(descriptor.transitionFaces)
      );
    }
  }

  private requestChunk(descriptor: ChunkDescriptor): void {
    this.pending.set(descriptor.key, {
      ...descriptor,
      transitionFaces: [...descriptor.transitionFaces],
    });
    const request: ChunkWorkerRequest = {
      key: descriptor.key,
      lodIndex: descriptor.lodIndex,
      chunkX: descriptor.chunkX,
      chunkZ: descriptor.chunkZ,
      originY: descriptor.originY,
    };
    this.worker.postMessage(request);
  }

  private handleWorkerMessage(message: ChunkWorkerResponse): void {
    const descriptor = this.pending.get(message.key);
    if (!descriptor) {
      return;
    }

    this.pending.delete(message.key);
    if (this.chunks.has(message.key)) {
      return;
    }
    const sampler = createChunkFieldSampler({
      data: new Int8Array(message.buffer),
      minX: message.minX,
      minY: message.minY,
      minZ: message.minZ,
      size: message.size,
    });
    this.buildChunk(descriptor, sampler);
  }

  private buildChunk(
    descriptor: ChunkDescriptor,
    sampler: DensityFunction
  ): void {
    const cellScale = 1 << descriptor.lodIndex;
    const samplesPerAxis = BLOCK_WIDTH * cellScale;
    const origin = new Vector3i(
      descriptor.chunkX * samplesPerAxis,
      descriptor.originY,
      descriptor.chunkZ * samplesPerAxis
    );

    const meshData = this.mesher.extractBlock(sampler, {
      origin,
      offset: Vector3f.zero,
      lodIndex: descriptor.lodIndex,
      cellSize: CELL_SIZE,
      transitionFaces: descriptor.transitionFaces,
    });

    if (meshData.indices.length === 0) {
      return;
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

    this.root.add(mesh);
    this.chunks.set(descriptor.key, {
      mesh,
      material,
      key: descriptor.key,
      transitionHash: this.transitionSignature(descriptor.transitionFaces),
      lodIndex: descriptor.lodIndex,
      chunkX: descriptor.chunkX,
      chunkZ: descriptor.chunkZ,
      color: descriptor.color,
      originY: descriptor.originY,
    });
  }

  private disposeChunk(key: string): void {
    const record = this.chunks.get(key);
    if (!record) {
      return;
    }

    this.root.remove(record.mesh);
    record.mesh.geometry.dispose();
    record.material.dispose();
    this.chunks.delete(key);
  }

  private updateStats(): void {
    if (this.statsTarget) {
      this.statsTarget.textContent = `${this.chunks.size} active chunks (${this.pending.size} building)`;
    }
  }

  private findOverlappingDescriptor(
    record: ChunkRecord,
    desired: Map<string, ChunkDescriptor>
  ): ChunkDescriptor | null {
    const bounds = this.chunkBounds(
      record.lodIndex,
      record.chunkX,
      record.chunkZ
    );
    for (const descriptor of desired.values()) {
      const other = this.chunkBounds(
        descriptor.lodIndex,
        descriptor.chunkX,
        descriptor.chunkZ
      );
      if (this.boundsIntersect(bounds, other)) {
        return descriptor;
      }
    }
    return null;
  }

  private chunkBounds(
    lodIndex: number,
    chunkX: number,
    chunkZ: number
  ): {
    minX: number;
    maxX: number;
    minZ: number;
    maxZ: number;
  } {
    const size = chunkWorldSize(lodIndex);
    const minX = chunkX * size;
    const minZ = chunkZ * size;
    return {
      minX,
      maxX: minX + size,
      minZ,
      maxZ: minZ + size,
    };
  }

  private boundsIntersect(
    a: { minX: number; maxX: number; minZ: number; maxZ: number },
    b: { minX: number; maxX: number; minZ: number; maxZ: number }
  ): boolean {
    return !(
      a.maxX <= b.minX ||
      a.minX >= b.maxX ||
      a.maxZ <= b.minZ ||
      a.minZ >= b.maxZ
    );
  }

  private shouldCullChunk(record: ChunkRecord): boolean {
    if (!this.lastCameraPosition) {
      return false;
    }

    const centerX = (record.chunkX + 0.5) * chunkWorldSize(record.lodIndex);
    const centerZ = (record.chunkZ + 0.5) * chunkWorldSize(record.lodIndex);
    const distance = Math.hypot(
      centerX - this.lastCameraPosition.x,
      centerZ - this.lastCameraPosition.z
    );
    const level = LOD_LOOKUP.get(record.lodIndex);
    const maxDistance = level
      ? level.maxDistance
      : chunkWorldSize(record.lodIndex) * 4;
    return distance - chunkDiagonalRadius(record.lodIndex) > maxDistance * 1.5;
  }

  private transitionSignature(faces: TransitionFace[]): string {
    if (faces.length === 0) {
      return "none";
    }
    return faces.slice().sort().join(",");
  }
}

const chunkManager = new ChunkManager(scene, statsLabel);

const animate = () => {
  controls.update();
  chunkManager.update(controls.target);
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
      const { key, lodIndex, chunkX, chunkZ, originY } = event.data;
      const range = BLOCK_WIDTH << lodIndex;
      const sampleSize = range + 3;
      const minX = chunkX * range - 1;
      const minY = originY - 1;
      const minZ = chunkZ * range - 1;
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

      self.postMessage({ key, minX, minY, minZ, size: sampleSize, buffer: data.buffer }, [data.buffer]);
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
