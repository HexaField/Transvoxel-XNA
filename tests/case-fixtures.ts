import { Tables } from "../src/lengyel/tables";
import { matrix3x3FromColumns, multiplyMatrix3x3Vector3i } from "../src/math/matrix3x3";
import { vector3fZero, fromVector3i as vector3fFromVector3i } from "../src/math/vector3f";
import {
  type Vector3i,
  addVector3i,
  createVector3i,
  multiplyVector3iScalar,
  vector3iZero,
} from "../src/math/vector3i";
import { RegularCache, TransitionCache } from "../src/surface-extractor/cache";
import { MeshData } from "../src/surface-extractor/mesh-data";
import { TransvoxelExtractor } from "../src/surface-extractor/transvoxel-extractor";
import { TransvoxelVertex } from "../src/surface-extractor/vertex";
import type { DensityFunction } from "../src/volume/volume-data";

const clamp = (value: number, min: number, max: number): number => Math.max(min, Math.min(max, value));
const vectorKey = (vector: Vector3i): string => `${vector.x},${vector.y},${vector.z}`;

const BLOCK_WIDTH = TransvoxelExtractor.BlockWidth;
const TRANSITION_LOD_INDEX = 1;
const TRANSITION_CELL_SIZE = 1;
const TRANSITION_LOD_SCALE = 1 << TRANSITION_LOD_INDEX;

const transitionDescriptor = {
  axis: 2 as 0 | 1 | 2,
  direction: 1 as -1 | 1,
  originOffset: createVector3i(0, 0, BLOCK_WIDTH),
  localX: createVector3i(1, 0, 0),
  localY: createVector3i(0, 1, 0),
  localZ: createVector3i(0, 0, -1),
};

const transitionCellOrigin = multiplyVector3iScalar(transitionDescriptor.originOffset, TRANSITION_LOD_SCALE);

const transitionCoordinates = [
  createVector3i(0, 0, 0),
  createVector3i(1, 0, 0),
  createVector3i(2, 0, 0),
  createVector3i(0, 1, 0),
  createVector3i(1, 1, 0),
  createVector3i(2, 1, 0),
  createVector3i(0, 2, 0),
  createVector3i(1, 2, 0),
  createVector3i(2, 2, 0),
  createVector3i(0, 0, 2),
  createVector3i(2, 0, 2),
  createVector3i(0, 2, 2),
  createVector3i(2, 2, 2),
];

const canonicalTransitionPositions = buildTransitionPositions(
  transitionDescriptor,
  vector3iZero,
  TRANSITION_LOD_INDEX,
  TRANSITION_CELL_SIZE
);

const transitionBitOrder = [0, 1, 2, 5, 8, 7, 6, 3, 4];

const transitionCoordinateBitLookup = new Map<string, number>();
transitionBitOrder.forEach((positionIndex, bitIndex) => {
  const key = vectorKey(canonicalTransitionPositions[positionIndex]);
  transitionCoordinateBitLookup.set(key, bitIndex);
});
[
  [9, 0],
  [10, 2],
  [11, 6],
  [12, 8],
].forEach(([duplicateIndex, sourceIndex]) => {
  const sourceKey = vectorKey(canonicalTransitionPositions[sourceIndex]);
  const bitIndex = transitionCoordinateBitLookup.get(sourceKey);
  if (bitIndex !== undefined) {
    const duplicateKey = vectorKey(canonicalTransitionPositions[duplicateIndex]);
    transitionCoordinateBitLookup.set(duplicateKey, bitIndex);
  }
});

const regularCornerLookup = new Map<string, number>();
Tables.CornerIndex.forEach((corner, index) => {
  regularCornerLookup.set(vectorKey(corner), index);
});

const createRegularCaseSampler = (caseCode: number): DensityFunction =>
  (x, y, z) => {
    const clamped = createVector3i(clamp(x, 0, 1), clamp(y, 0, 1), clamp(z, 0, 1));
    const key = vectorKey(clamped);
    const cornerIndex = regularCornerLookup.get(key);
    if (cornerIndex === undefined) {
      return 1;
    }
    const filled = ((caseCode >> cornerIndex) & 1) === 1;
    return filled ? -1 : 1;
  };

const createTransitionCaseSampler = (caseCode: number): DensityFunction =>
  (x, y, z) => {
    const bitIndex = transitionCoordinateBitLookup.get(`${x},${y},${z}`);
    if (bitIndex === undefined) {
      return 1;
    }
    const filled = ((caseCode >> bitIndex) & 1) === 1;
    return filled ? -1 : 1;
  };

export function buildRegularCaseMesh(caseCode: number, cache: RegularCache): MeshData {
  cache.reset();
  const vertices: TransvoxelVertex[] = [];
  const indices: number[] = [];
  TransvoxelExtractor.polygonizeRegularCell(
    vector3iZero,
    vector3fZero,
    vector3iZero,
    vector3iZero,
    createRegularCaseSampler(caseCode),
    0,
    1,
    vertices,
    indices,
    cache
  );
  return new MeshData(vertices, indices);
}

export function buildTransitionCaseMesh(caseCode: number, cache: TransitionCache): MeshData {
  cache.reset();
  const vertices: TransvoxelVertex[] = [];
  const indices: number[] = [];
  TransvoxelExtractor.polygonizeTransitionCell(
    vector3fZero,
    transitionCellOrigin,
    transitionDescriptor.localX,
    transitionDescriptor.localY,
    transitionDescriptor.localZ,
    0,
    0,
    1,
    TRANSITION_LOD_INDEX,
    transitionDescriptor.axis,
    0,
    createTransitionCaseSampler(caseCode),
    vertices,
    indices,
    cache,
    vector3iZero
  );
  return new MeshData(vertices, indices);
}

function buildTransitionPositions(
  descriptor: {
    originOffset: Vector3i;
    localX: Vector3i;
    localY: Vector3i;
    localZ: Vector3i;
  },
  origin: Vector3i,
  lodIndex: number,
  cellSize: number
): Vector3i[] {
  const lodScale = 1 << lodIndex;
  const sampleStep = 1 << (lodIndex - 1);
  const faceOrigin = addVector3i(origin, multiplyVector3iScalar(descriptor.originOffset, lodScale));
  const mx = multiplyVector3iScalar(descriptor.localX, sampleStep * cellSize);
  const my = multiplyVector3iScalar(descriptor.localY, sampleStep * cellSize);
  const mz = multiplyVector3iScalar(descriptor.localZ, sampleStep * cellSize);
  const basis = matrix3x3FromColumns(
    vector3fFromVector3i(mx),
    vector3fFromVector3i(my),
    vector3fFromVector3i(mz)
  );
  return transitionCoordinates.map((coord) => addVector3i(faceOrigin, multiplyMatrix3x3Vector3i(basis, coord)));
}

