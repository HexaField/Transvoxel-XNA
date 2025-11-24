import { describe, it } from "vitest";
import { TransvoxelExtractor } from "../src/surface-extractor/transvoxel-extractor";
import { RegularCache } from "../src/surface-extractor/cache";
import type { DensityFunction } from "../src/volume/volume-data";
import {
  type Vector3i,
  addVector3i,
  createVector3i,
  equalsVector3i,
  multiplyVector3iScalar,
  vector3iZero,
} from "../src/math/vector3i";
import { type Vector3f, vector3fZero } from "../src/math/vector3f";
import { getRenderablePosition, TransvoxelVertex } from "../src/surface-extractor/vertex";

const BLOCK_WIDTH = TransvoxelExtractor.BlockWidth;
const lodIndex = 1;
const cellSize = 1;
const lodScale = 1 << lodIndex;
const blockExtent = BLOCK_WIDTH * lodScale * cellSize;
const blockCenter = blockExtent * 0.5;

const createSampleVolume = (): DensityFunction => (x, y, z) => {
  const nx = (x - blockCenter) / blockExtent;
  const ny = (y - blockCenter) / blockExtent;
  const nz = (z - blockCenter) / blockExtent;
  const sphere = nx * nx + ny * ny + nz * nz - 0.18;
  const folds =
    Math.sin(nx * 8.0) * 0.35 + Math.cos(ny * 6.0) * 0.35 + Math.sin(nz * 7.0) * 0.35;
  const density = sphere + 0.25 * folds;
  return Math.floor(density * 127);
};

const blockOrigin = vector3iZero;
const blockOffset = vector3fZero;
const target = createVector3i(8, 14, 9);

const trianglesFrom = (verts: TransvoxelVertex[], indices: number[]) => {
  const tris: Array<{ a: Vector3f; b: Vector3f; c: Vector3f }> = [];
  for (let i = 0; i < indices.length; i += 3) {
    const tri = {
      a: getRenderablePosition(verts[indices[i]]),
      b: getRenderablePosition(verts[indices[i + 1]]),
      c: getRenderablePosition(verts[indices[i + 2]]),
    };
    tris.push(tri);
  }
  return tris;
};

describe("debug cell", () => {
  it("compare cached vs isolated", () => {
    const volume = createSampleVolume();

    const cachedCache = new RegularCache(BLOCK_WIDTH);
    const cachedVerts: TransvoxelVertex[] = [];
    const cachedIndices: number[] = [];
    let cachedSlice: { verts: TransvoxelVertex[]; indices: number[]; beforeVerts: number } | null = null;

    outer: for (let x = 0; x < BLOCK_WIDTH; x++) {
      for (let y = 0; y < BLOCK_WIDTH; y++) {
        for (let z = 0; z < BLOCK_WIDTH; z++) {
          const xyz = createVector3i(x, y, z);
          const min = addVector3i(blockOrigin, multiplyVector3iScalar(xyz, lodScale));
          const beforeVerts = cachedVerts.length;
          const beforeIndices = cachedIndices.length;
          TransvoxelExtractor.polygonizeRegularCell(
            min,
            blockOffset,
            xyz,
            blockOrigin,
            volume,
            lodIndex,
            cellSize,
            cachedVerts,
            cachedIndices,
            cachedCache
          );
          if (equalsVector3i(xyz, target)) {
            cachedSlice = {
              verts: cachedVerts.slice(),
              indices: cachedIndices.slice(beforeIndices),
              beforeVerts,
            };
            break outer;
          }
        }
      }
    }

    if (!cachedSlice) {
      throw new Error("Target cell not found");
    }

    const isolatedCache = new RegularCache(BLOCK_WIDTH);
    const isolatedVerts: TransvoxelVertex[] = [];
    const isolatedIndices: number[] = [];
    const min = addVector3i(blockOrigin, multiplyVector3iScalar(target, lodScale));
    TransvoxelExtractor.polygonizeRegularCell(
      min,
      blockOffset,
      target,
      blockOrigin,
      volume,
      lodIndex,
      cellSize,
      isolatedVerts,
      isolatedIndices,
      isolatedCache
    );

    const annotatedCached = cachedSlice.indices.reduce<Array<{ index: number; reused: boolean; position: Vector3f }>>(
      (acc, idx) => {
        acc.push({
          index: idx,
          reused: idx < cachedSlice!.beforeVerts,
          position: getRenderablePosition(cachedSlice!.verts[idx]),
        });
        return acc;
      },
      []
    );

    console.log("cached", trianglesFrom(cachedSlice.verts, cachedSlice.indices));
    console.log("cached vertices", annotatedCached);
    console.log("isolated", trianglesFrom(isolatedVerts, isolatedIndices));
  });
});
