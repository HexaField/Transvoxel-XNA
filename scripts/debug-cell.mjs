import { TransvoxelExtractor } from "../dist/surface-extractor/transvoxel-extractor.js";
import { RegularCache } from "../dist/surface-extractor/cache.js";
import { Vector3i } from "../dist/math/vector3i.js";
import { Vector3f } from "../dist/math/vector3f.js";
import { getRenderablePosition } from "../dist/surface-extractor/vertex.js";

const BLOCK_WIDTH = TransvoxelExtractor.BlockWidth;
const lodIndex = 1;
const cellSize = 1;
const lodScale = 1 << lodIndex;
const blockExtent = BLOCK_WIDTH * lodScale * cellSize;
const blockCenter = blockExtent * 0.5;

const createSampleVolume = () => (x, y, z) => {
  const nx = (x - blockCenter) / blockExtent;
  const ny = (y - blockCenter) / blockExtent;
  const nz = (z - blockCenter) / blockExtent;
  const sphere = nx * nx + ny * ny + nz * nz - 0.18;
  const folds = Math.sin(nx * 8.0) * 0.35 + Math.cos(ny * 6.0) * 0.35 + Math.sin(nz * 7.0) * 0.35;
  const density = sphere + 0.25 * folds;
  return Math.floor(density * 127);
};

const volume = createSampleVolume();
const blockOrigin = Vector3i.zero;
const blockOffset = new Vector3f(blockOrigin.x * cellSize, blockOrigin.y * cellSize, blockOrigin.z * cellSize);
const target = { x: 8, y: 14, z: 9 };

function collectWithCache() {
  const cache = new RegularCache(BLOCK_WIDTH);
  const verts = [];
  const indices = [];

  for (let x = 0; x < BLOCK_WIDTH; x++) {
    for (let y = 0; y < BLOCK_WIDTH; y++) {
      for (let z = 0; z < BLOCK_WIDTH; z++) {
        const xyz = new Vector3i(x, y, z);
        const min = blockOrigin.add(xyz.multiplyScalar(lodScale));
        const beforeVerts = verts.length;
        const beforeIndices = indices.length;

        TransvoxelExtractor.polygonizeRegularCell(
          min,
          blockOffset,
          xyz,
          blockOrigin,
          volume,
          lodIndex,
          cellSize,
          verts,
          indices,
          cache
        );

        if (x === target.x && y === target.y && z === target.z) {
          const newVerts = verts.slice(beforeVerts);
          const newIndices = indices.slice(beforeIndices);
          return { newVerts, newIndices, verts, indices };
        }
      }
    }
  }
  throw new Error("Target cell not processed");
}

function collectIsolated() {
  const cache = new RegularCache(BLOCK_WIDTH);
  const verts = [];
  const indices = [];
  const xyz = new Vector3i(target.x, target.y, target.z);
  const min = blockOrigin.add(xyz.multiplyScalar(lodScale));
  TransvoxelExtractor.polygonizeRegularCell(
    min,
    blockOffset,
    xyz,
    blockOrigin,
    volume,
    lodIndex,
    cellSize,
    verts,
    indices,
    cache
  );
  return { verts, indices };
}

function trianglesFrom(verts, indices) {
  const tris = [];
  for (let t = 0; t < indices.length; t += 3) {
    const tri = [indices[t], indices[t + 1], indices[t + 2]].map((index) => {
      const v = verts[index];
      return getRenderablePosition(v);
    });
    tris.push(tri);
  }
  return tris;
}

const cached = collectWithCache();
const isolated = collectIsolated();

console.log("cached triangles:");
console.log(trianglesFrom(cached.verts ?? cached.newVerts, cached.newIndices));

console.log("isolated triangles:");
console.log(trianglesFrom(isolated.verts, isolated.indices));
