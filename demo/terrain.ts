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
import { Vector3f } from "../src/math/vector3f";
import { meshDataToGeometry } from "./mesh-utils";
import { createChunkFieldSampler, type DensityFunction } from "../src/volume/volume-data";

const mount = document.querySelector<HTMLDivElement>("#app");
if (!mount) {
  throw new Error("Missing #app mount point.");
}

const statsLabel = document.querySelector<HTMLParagraphElement>("#chunk-stats");

const scene = new Scene();
scene.background = new Color("#03050c");
scene.fog = new FogExp2("#03050c", 0.0012);

const camera = new PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1500);
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
}

interface ChunkDescriptor {
  lodIndex: number;
  chunkX: number;
  chunkZ: number;
  key: string;
  color: number;
}

const lodBands = [
  { lodIndex: 0, radius: 1, color: 0x8ed081 },
  { lodIndex: 1, radius: 2, color: 0x6b9d60 },
  { lodIndex: 2, radius: 3, color: 0x4a6c3d },
];

class ChunkManager {
  private readonly mesher = new TransvoxelMesher();
  private readonly worker = createChunkWorker();
  private readonly chunks = new Map<string, ChunkRecord>();
  private readonly pending = new Map<string, ChunkDescriptor>();
  private readonly lastDesiredKeys = new Set<string>();

  constructor(private readonly root: Scene, private readonly statsTarget: HTMLParagraphElement | null) {
    this.worker.onmessage = (event: MessageEvent<ChunkWorkerResponse>) => {
      this.handleWorkerMessage(event.data as ChunkWorkerResponse);
    };
  }

  update(cameraPosition: Vector3): void {
    const desired = this.collectDesiredChunks(cameraPosition);
    if (this.hasDesiredChanged(desired)) {
      this.reconcileChunks(desired);
      this.captureDesiredKeys(desired);
    }
    this.updateStats();
  }

  private collectDesiredChunks(cameraPosition: Vector3): Map<string, ChunkDescriptor> {
    const desired = new Map<string, ChunkDescriptor>();
    for (const band of lodBands) {
      const chunkSize = BLOCK_WIDTH << band.lodIndex;
      const baseX = Math.floor(cameraPosition.x / chunkSize);
      const baseZ = Math.floor(cameraPosition.z / chunkSize);

      for (let dx = -band.radius; dx <= band.radius; dx++) {
        for (let dz = -band.radius; dz <= band.radius; dz++) {
          if (Math.hypot(dx, dz) > band.radius + 0.2) {
            continue;
          }

          const chunkX = baseX + dx;
          const chunkZ = baseZ + dz;
          const key = `${band.lodIndex}:${chunkX}:${chunkZ}`;
          desired.set(key, {
            lodIndex: band.lodIndex,
            chunkX,
            chunkZ,
            key,
            color: band.color,
          });
        }
      }
    }

    return desired;
  }

  private reconcileChunks(desired: Map<string, ChunkDescriptor>): void {
    for (const existingKey of Array.from(this.chunks.keys())) {
      if (!desired.has(existingKey)) {
        this.disposeChunk(existingKey);
      }
    }

    for (const pendingKey of Array.from(this.pending.keys())) {
      if (!desired.has(pendingKey)) {
        this.pending.delete(pendingKey);
      }
    }

    for (const descriptor of desired.values()) {
      if (this.chunks.has(descriptor.key) || this.pending.has(descriptor.key)) {
        continue;
      }
      this.requestChunk(descriptor);
    }
  }

  private hasDesiredChanged(desired: Map<string, ChunkDescriptor>): boolean {
    if (desired.size !== this.lastDesiredKeys.size) {
      return true;
    }

    for (const key of desired.keys()) {
      if (!this.lastDesiredKeys.has(key)) {
        return true;
      }
    }

    return false;
  }

  private captureDesiredKeys(desired: Map<string, ChunkDescriptor>): void {
    this.lastDesiredKeys.clear();
    for (const key of desired.keys()) {
      this.lastDesiredKeys.add(key);
    }
  }

  private requestChunk(descriptor: ChunkDescriptor): void {
    this.pending.set(descriptor.key, descriptor);
    const request: ChunkWorkerRequest = {
      key: descriptor.key,
      lodIndex: descriptor.lodIndex,
      chunkX: descriptor.chunkX,
      chunkZ: descriptor.chunkZ,
      originY: CHUNK_ORIGIN_Y,
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

  private buildChunk(descriptor: ChunkDescriptor, sampler: DensityFunction): void {
    const cellScale = 1 << descriptor.lodIndex;
    const samplesPerAxis = BLOCK_WIDTH * cellScale;
    const origin = new Vector3i(descriptor.chunkX * samplesPerAxis, CHUNK_ORIGIN_Y, descriptor.chunkZ * samplesPerAxis);
    const offset = new Vector3f(origin.x * CELL_SIZE, origin.y * CELL_SIZE, origin.z * CELL_SIZE);

    const meshData = this.mesher.extractRegularBlock(sampler, {
      origin,
      offset,
      lodIndex: descriptor.lodIndex,
      cellSize: CELL_SIZE,
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
    this.chunks.set(descriptor.key, { mesh, material, key: descriptor.key });
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
      const sampleSize = range + 2;
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
