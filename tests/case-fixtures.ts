import { Tables } from "../src/lengyel/tables";
import { Matrix3x3 } from "../src/math/matrix3x3";
import { Vector3f } from "../src/math/vector3f";
import { Vector3i } from "../src/math/vector3i";
import { RegularCache, TransitionCache } from "../src/surface-extractor/cache";
import { MeshData } from "../src/surface-extractor/mesh-data";
import { TransvoxelExtractor } from "../src/surface-extractor/transvoxel-extractor";
import { TransvoxelVertex } from "../src/surface-extractor/vertex";
import { VolumeData } from "../src/volume/volume-data";

const clamp = (value: number, min: number, max: number): number => Math.max(min, Math.min(max, value));
const vectorKey = (vector: Vector3i): string => `${vector.x},${vector.y},${vector.z}`;

const BLOCK_WIDTH = TransvoxelExtractor.BlockWidth;
const TRANSITION_LOD_INDEX = 1;
const TRANSITION_CELL_SIZE = 1;
const TRANSITION_LOD_SCALE = 1 << TRANSITION_LOD_INDEX;

const transitionDescriptor = {
  axis: 2 as 0 | 1 | 2,
  direction: 1 as -1 | 1,
  originOffset: new Vector3i(0, 0, BLOCK_WIDTH),
  localX: new Vector3i(1, 0, 0),
  localY: new Vector3i(0, 1, 0),
  localZ: new Vector3i(0, 0, -1),
};

const transitionCellOrigin = transitionDescriptor.originOffset.multiplyScalar(TRANSITION_LOD_SCALE);

const transitionCoordinates = [
  new Vector3i(0, 0, 0),
  new Vector3i(1, 0, 0),
  new Vector3i(2, 0, 0),
  new Vector3i(0, 1, 0),
  new Vector3i(1, 1, 0),
  new Vector3i(2, 1, 0),
  new Vector3i(0, 2, 0),
  new Vector3i(1, 2, 0),
  new Vector3i(2, 2, 0),
  new Vector3i(0, 0, 2),
  new Vector3i(2, 0, 2),
  new Vector3i(0, 2, 2),
  new Vector3i(2, 2, 2),
];

const canonicalTransitionPositions = buildTransitionPositions(
  transitionDescriptor,
  Vector3i.zero,
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

class RegularCaseVolume implements VolumeData {
  constructor(private readonly caseCode: number) {}

  sample(x: number, y: number, z: number): number {
    const clamped = new Vector3i(clamp(x, 0, 1), clamp(y, 0, 1), clamp(z, 0, 1));
    const key = vectorKey(clamped);
    const cornerIndex = regularCornerLookup.get(key);
    if (cornerIndex === undefined) {
      return 1;
    }
    const filled = ((this.caseCode >> cornerIndex) & 1) === 1;
    return filled ? -1 : 1;
  }
}

class TransitionCaseVolume implements VolumeData {
  constructor(private readonly caseCode: number) {}

  sample(x: number, y: number, z: number): number {
    const bitIndex = transitionCoordinateBitLookup.get(`${x},${y},${z}`);
    if (bitIndex === undefined) {
      return 1;
    }
    const filled = ((this.caseCode >> bitIndex) & 1) === 1;
    return filled ? -1 : 1;
  }
}

export function buildRegularCaseMesh(caseCode: number, cache: RegularCache): MeshData {
  cache.reset();
  const vertices: TransvoxelVertex[] = [];
  const indices: number[] = [];
  TransvoxelExtractor.polygonizeRegularCell(
    Vector3i.zero,
    Vector3f.zero,
    Vector3i.zero,
    new RegularCaseVolume(caseCode),
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
    Vector3f.zero,
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
    new TransitionCaseVolume(caseCode),
    vertices,
    indices,
    cache
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
  const faceOrigin = origin.add(descriptor.originOffset.multiplyScalar(lodScale));
  const mx = descriptor.localX.multiplyScalar(sampleStep * cellSize);
  const my = descriptor.localY.multiplyScalar(sampleStep * cellSize);
  const mz = descriptor.localZ.multiplyScalar(sampleStep * cellSize);
  const basis = Matrix3x3.fromColumns(
    Vector3f.fromVector3i(mx),
    Vector3f.fromVector3i(my),
    Vector3f.fromVector3i(mz)
  );
  return transitionCoordinates.map((coord) => faceOrigin.add(basis.multiplyVector3i(coord)));
}

