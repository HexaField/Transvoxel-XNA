import { Tables, RegularCell } from "../lengyel/tables";
import {
  type Vector3f,
  addVector3f,
  combineVector3f,
  createVector3f,
  divideVector3fScalar,
  fromVector3i as vector3fFromVector3i,
  multiplyVector3fScalar,
  normalizeVector3f,
  subtractVector3f,
  setVector3f,
} from "../math/vector3f";
import {
  type Vector3i,
  addVector3i,
  createVector3i,
  multiplyVector3iScalar,
  setVector3i,
  subtractVector3i,
  vector3iComponent,
  vector3iUnitX,
  vector3iUnitY,
  vector3iUnitZ,
} from "../math/vector3i";
import {
  createMatrix3x3,
  matrix3x3FromColumns,
  multiplyMatrix3x3Vector3f,
  multiplyMatrix3x3Vector3i,
  setMatrix3x3,
  type Matrix3x3,
} from "../math/matrix3x3";
import { DensityFunction } from "../volume/volume-data";
import { RegularCache, TransitionCache } from "./cache";
import { MeshData } from "./mesh-data";
import { TransvoxelVertex, unusedVertexPosition } from "./vertex";

const BLOCK_WIDTH = 16;
const S = 1.0 / 256.0;

const hiNibble = (value: number): number => (value >> 4) & 0x0f;
const loNibble = (value: number): number => value & 0x0f;
const signBit = (value: number): number => (value < 0 ? 1 : 0);

const toVector3f = (vector: Vector3i): Vector3f => vector3fFromVector3i(vector);
const vertexDeltaScratch = createVector3f();
const vertexProjectionScratch = createVector3f();
const transitionDeltaScratch = createVector3f();
const transitionProjectionScratch = createVector3f();
const transitionSecondaryBaseScratch = createVector3f();

const setAxisComponent = (vector: Vector3f, axis: number, value: number): Vector3f => {
  switch (axis) {
    case 0:
      return createVector3f(value, vector.y, vector.z);
    case 1:
      return createVector3f(vector.x, value, vector.z);
    case 2:
      return createVector3f(vector.x, vector.y, value);
    default:
      throw new RangeError(`Axis ${axis} is out of range for Vector3f.`);
  }
};

const sample = (sampler: DensityFunction, position: Vector3i): number => sampler(position.x, position.y, position.z);

const intDiv = (numerator: number, denominator: number): number => {
  if (denominator === 0) {
    return 0;
  }
  return Math.trunc(numerator / denominator);
};

const computeDelta = (v: Vector3f, k: number, s: number, out: Vector3f = createVector3f()): Vector3f => {
  if (k < 1) {
    return setVector3f(out, 0, 0, 0);
  }

  const p2k = Math.pow(2, k);
  const wk = Math.pow(2, k - 2);
  const p2mk = Math.pow(2, -k);

  const componentDelta = (p: number): number => {
    if (p < p2k) {
      return (1.0 - p2mk * p) * wk;
    }
    if (p > p2k * (s - 1)) {
      return ((p2k * s) - 1.0 - p) * wk;
    }
    return 0;
  };

  return setVector3f(out, componentDelta(v.x), componentDelta(v.y), componentDelta(v.z));
};

const projectionMatrix = createMatrix3x3();
const projectNormal = (n: Vector3f, delta: Vector3f, out: Vector3f = createVector3f()): Vector3f => {
  setMatrix3x3(
    projectionMatrix,
    1.0 - n.x * n.x,
    -n.x * n.y,
    -n.x * n.z,
    -n.x * n.y,
    1.0 - n.y * n.y,
    -n.y * n.z,
    -n.x * n.z,
    -n.y * n.z,
    1.0 - n.z * n.z
  );

  return multiplyMatrix3x3Vector3f(projectionMatrix, delta, out);
};

const prevOffset = (dir: number): Vector3i =>
  createVector3i(-(dir & 1), -((dir >> 1) & 1), -((dir >> 2) & 1));

const computeNormal = (n0: Vector3f, n1: Vector3f, t0: number, t1: number): Vector3f =>
  normalizeVector3f(combineVector3f(n0, t0, n1, t1));

const createVertex = (
  pi: Vector3f,
  normal: Vector3f,
  near: number,
  offset: Vector3f,
  lodIndex: number,
  cellSize: number
): TransvoxelVertex => {
  const scaledPi = multiplyVector3fScalar(pi, cellSize, vertexDeltaScratch);
  const primary = createVector3f(offset.x + scaledPi.x, offset.y + scaledPi.y, offset.z + scaledPi.z);
  if (near > 0) {
    const delta = computeDelta(pi, lodIndex, BLOCK_WIDTH, vertexDeltaScratch);
    const projected = multiplyVector3fScalar(projectNormal(normal, delta, vertexProjectionScratch), cellSize, vertexProjectionScratch);
    return {
      primary,
      secondary: createVector3f(primary.x + projected.x, primary.y + projected.y, primary.z + projected.z),
      normal,
      near,
    };
  }

  return {
    primary,
    secondary: unusedVertexPosition,
    normal,
    near,
  };
};

const interpolate = (
  v0: Vector3f,
  v1: Vector3f,
  p0: Vector3i,
  p1: Vector3i,
  samples: DensityFunction,
  lodIndex = 0
): Vector3f => {
  let vStart = v0;
  let vEnd = v1;
  let pStart = p0;
  let pEnd = p1;

  const shouldSwap =
    pEnd.x < pStart.x ||
    (pEnd.x === pStart.x && pEnd.y < pStart.y) ||
    (pEnd.x === pStart.x && pEnd.y === pStart.y && pEnd.z < pStart.z);

  if (shouldSwap) {
    vStart = v1;
    vEnd = v0;
    pStart = p1;
    pEnd = p0;
  }

  let s0 = sample(samples, pStart);
  let s1 = sample(samples, pEnd);

  let t = intDiv(s1 << 8, s1 - s0);
  let u = 0x0100 - t;

  if ((t & 0x00ff) === 0) {
    return t === 0 ? vEnd : vStart;
  }

  for (let i = 0; i < lodIndex; i++) {
    const vm = divideVector3fScalar(addVector3f(vStart, vEnd), 2);
    const pm = createVector3i((pStart.x + pEnd.x) >> 1, (pStart.y + pEnd.y) >> 1, (pStart.z + pEnd.z) >> 1);
    const sm = sample(samples, pm);

    if (signBit(s0) !== signBit(sm)) {
      vEnd = vm;
      pEnd = pm;
      s1 = sm;
    } else {
      vStart = vm;
      pStart = pm;
      s0 = sm;
    }
  }

  t = intDiv(s1 << 8, s1 - s0);
  u = 0x0100 - t;

  return combineVector3f(vStart, t * S, vEnd, u * S);
};

const buildCornerPositions = (min: Vector3i, lodScale: number): Vector3i[] =>
  Tables.CornerIndex.map((corner) => addVector3i(min, multiplyVector3iScalar(corner, lodScale)));

const transitionCoordinates: ReadonlyArray<Vector3i> = [
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
const transitionPositionsScratch = transitionCoordinates.map(() => createVector3i());
const transitionSampleIndexMap = [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 2, 6, 8];
const transitionScaledCoordinate = createVector3i();
const transitionNormalsScratch = Array.from({ length: 13 }, () => createVector3f());
const transitionBasisColumnXScratch = createVector3i();
const transitionBasisColumnYScratch = createVector3i();
const transitionBasisColumnZScratch = createVector3i();
const transitionBasisColumnXFloat = createVector3f();
const transitionBasisColumnYFloat = createVector3f();
const transitionBasisColumnZFloat = createVector3f();

const buildCornerNormals = (positions: Vector3i[], samples: DensityFunction): Vector3f[] =>
  positions.map((p) => {
    const nx = (sample(samples, addVector3i(p, vector3iUnitX)) - sample(samples, subtractVector3i(p, vector3iUnitX))) * 0.5;
    const ny = (sample(samples, addVector3i(p, vector3iUnitY)) - sample(samples, subtractVector3i(p, vector3iUnitY))) * 0.5;
    const nz = (sample(samples, addVector3i(p, vector3iUnitZ)) - sample(samples, subtractVector3i(p, vector3iUnitZ))) * 0.5;
    return normalizeVector3f(createVector3f(nx, ny, nz));
  });

const toVector3fArray = (vectors: Vector3i[]): Vector3f[] => vectors.map((v) => toVector3f(v));

export interface ExtractBlockOptions {
  origin: Vector3i;
  offset?: Vector3f;
  lodIndex?: number;
  cellSize?: number;
}

export type TransitionFace =
  | "negativeX"
  | "positiveX"
  | "negativeY"
  | "positiveY"
  | "negativeZ"
  | "positiveZ";

interface TransitionFaceDescriptor {
  axis: 0 | 1 | 2;
  direction: -1 | 1;
  originOffset: Vector3i;
  localX: Vector3i;
  localY: Vector3i;
  localZ: Vector3i;
}

const blockVector = (x: number, y: number, z: number): Vector3i => createVector3i(x, y, z);

const crossVector3i = (a: Vector3i, b: Vector3i): Vector3i =>
  createVector3i(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  );

const dotVector3i = (a: Vector3i, b: Vector3i): number => a.x * b.x + a.y * b.y + a.z * b.z;

const transitionFaceDescriptors: Record<TransitionFace, TransitionFaceDescriptor> = {
  negativeX: {
    axis: 0,
    direction: -1,
    originOffset: blockVector(0, 0, BLOCK_WIDTH),
    localX: blockVector(0, 0, -1),
    localY: blockVector(0, 1, 0),
    localZ: blockVector(1, 0, 0),
  },
  positiveX: {
    axis: 0,
    direction: 1,
    originOffset: blockVector(BLOCK_WIDTH, 0, 0),
    localX: blockVector(0, 0, 1),
    localY: blockVector(0, 1, 0),
    localZ: blockVector(-1, 0, 0),
  },
  negativeY: {
    axis: 1,
    direction: -1,
    originOffset: blockVector(0, 0, BLOCK_WIDTH),
    localX: blockVector(1, 0, 0),
    localY: blockVector(0, 0, -1),
    localZ: blockVector(0, 1, 0),
  },
  positiveY: {
    axis: 1,
    direction: 1,
    originOffset: blockVector(0, BLOCK_WIDTH, 0),
    localX: blockVector(1, 0, 0),
    localY: blockVector(0, 0, 1),
    localZ: blockVector(0, -1, 0),
  },
  negativeZ: {
    axis: 2,
    direction: -1,
    originOffset: blockVector(0, 0, 0),
    localX: blockVector(1, 0, 0),
    localY: blockVector(0, 1, 0),
    localZ: blockVector(0, 0, 1),
  },
  positiveZ: {
    axis: 2,
    direction: 1,
    originOffset: blockVector(0, BLOCK_WIDTH, BLOCK_WIDTH),
    localX: blockVector(1, 0, 0),
    localY: blockVector(0, -1, 0),
    localZ: blockVector(0, 0, -1),
  },
};

export interface TransvoxelMesherOptions {
  regularCache?: RegularCache;
  transitionCache?: TransitionCache;
}

export class TransvoxelMesher {
  private readonly regularCache: RegularCache;
  private readonly transitionCache: TransitionCache;

  constructor(options: TransvoxelMesherOptions = {}) {
    this.regularCache = options.regularCache ?? new RegularCache(BLOCK_WIDTH);
    this.transitionCache = options.transitionCache ?? new TransitionCache(BLOCK_WIDTH);
  }

  getRegularCache(): RegularCache {
    return this.regularCache;
  }

  getTransitionCache(): TransitionCache {
    return this.transitionCache;
  }

  extractBlock(
    samples: DensityFunction,
    options: ExtractBlockOptions & { transitionFaces?: TransitionFace[] }
  ): MeshData {
    const regularMesh = this.extractRegularBlock(samples, options);
    const faces = options.transitionFaces?.filter(Boolean) ?? [];
    if (faces.length === 0) {
      return regularMesh;
    }

    const transitionMesh = this.extractTransitionFaces(samples, { ...options, faces });
    if (transitionMesh.vertices.length === 0) {
      return regularMesh;
    }

    const vertexOffset = regularMesh.vertices.length;
    transitionMesh.vertices.forEach((vertex) => regularMesh.vertices.push(vertex));
    transitionMesh.indices.forEach((index) => regularMesh.indices.push(index + vertexOffset));
    return regularMesh;
  }

  extractRegularBlock(
    samples: DensityFunction,
    { origin, offset, lodIndex = 0, cellSize = 1 }: ExtractBlockOptions
  ): MeshData {
    const vertices: TransvoxelVertex[] = [];
    const indices: number[] = [];
    this.regularCache.reset();
    const lodScale = 1 << lodIndex;

    const blockOffset =
      offset ?? createVector3f(origin.x * cellSize, origin.y * cellSize, origin.z * cellSize);

    const xyz = createVector3i();
    const scaled = createVector3i();
    const min = createVector3i();

    for (let x = 0; x < BLOCK_WIDTH; x++) {
      for (let y = 0; y < BLOCK_WIDTH; y++) {
        for (let z = 0; z < BLOCK_WIDTH; z++) {
          setVector3i(xyz, x, y, z);
          multiplyVector3iScalar(xyz, lodScale, scaled);
          addVector3i(origin, scaled, min);
          TransvoxelExtractor.polygonizeRegularCell(
            min,
            blockOffset,
            xyz,
            origin,
            samples,
            lodIndex,
            cellSize,
            vertices,
            indices,
            this.regularCache
          );
        }
      }
    }

    return new MeshData(vertices, indices);
  }

  extractTransitionFaces(
    samples: DensityFunction,
    {
      origin,
      offset,
      lodIndex = 0,
      cellSize = 1,
      faces,
    }: ExtractBlockOptions & { faces: TransitionFace[] }
  ): MeshData {
    if (lodIndex < 1) {
      throw new RangeError("Transition faces require lodIndex >= 1.");
    }

    const uniqueFaces = Array.from(new Set(faces));
    if (uniqueFaces.length === 0) {
      return new MeshData();
    }

    const blockOffset =
      offset ?? createVector3f(origin.x * cellSize, origin.y * cellSize, origin.z * cellSize);

    const vertices: TransvoxelVertex[] = [];
    const indices: number[] = [];

    for (const face of uniqueFaces) {
      const descriptor = transitionFaceDescriptors[face];
      if (!descriptor) {
        throw new Error(`Unknown transition face "${face}".`);
      }

      this.transitionCache.reset();
      this.generateTransitionFace(
        descriptor,
        origin,
        blockOffset,
        lodIndex,
        cellSize,
        samples,
        vertices,
        indices
      );
    }

    return new MeshData(vertices, indices);
  }

  private generateTransitionFace(
    descriptor: TransitionFaceDescriptor,
    origin: Vector3i,
    blockOffset: Vector3f,
    lodIndex: number,
    cellSize: number,
    samples: DensityFunction,
    verts: TransvoxelVertex[],
    indices: number[]
  ): void {
    const lodScale = 1 << lodIndex;
    const sampleStep = 1 << (lodIndex - 1);
    const stride = sampleStep << 1;
    const scaledOriginOffset = createVector3i();
    const faceOrigin = createVector3i();
    addVector3i(origin, multiplyVector3iScalar(descriptor.originOffset, lodScale, scaledOriginOffset), faceOrigin);
    const orientation = dotVector3i(crossVector3i(descriptor.localX, descriptor.localY), descriptor.localZ);
    const invertWinding = orientation > 0;

    const rowOrigin = createVector3i();
    const cellOrigin = createVector3i();
    const stepX = createVector3i();
    const stepY = createVector3i();

    for (let y = 0; y < BLOCK_WIDTH; y++) {
      multiplyVector3iScalar(descriptor.localY, y * stride, stepY);
      addVector3i(faceOrigin, stepY, rowOrigin);
      for (let x = 0; x < BLOCK_WIDTH; x++) {
        multiplyVector3iScalar(descriptor.localX, x * stride, stepX);
        addVector3i(rowOrigin, stepX, cellOrigin);
        const directionMask = (x > 0 ? 1 : 0) | ((y > 0 ? 1 : 0) << 1);
        TransvoxelExtractor.polygonizeTransitionCell(
          blockOffset,
          cellOrigin,
          descriptor.localX,
          descriptor.localY,
          descriptor.localZ,
          x,
          y,
          cellSize,
          lodIndex,
          descriptor.axis,
          directionMask,
          samples,
          verts,
          indices,
          this.transitionCache,
          origin,
          invertWinding
        );
      }
    }
  }
}

export class TransvoxelExtractor {
  static readonly BlockWidth = BLOCK_WIDTH;

  static polygonizeRegularCell(
    min: Vector3i,
    offset: Vector3f,
    xyz: Vector3i,
    blockOrigin: Vector3i,
    samples: DensityFunction,
    lodIndex: number,
    cellSize: number,
    verts: TransvoxelVertex[],
    indices: number[],
    cache: RegularCache
  ): number {
    const lodScale = 1 << lodIndex;
    const last = 15 * lodScale;
    const blockOriginF = vector3fFromVector3i(blockOrigin);
    const directionMask =
      (xyz.x > 0 ? 1 : 0) | ((xyz.y > 0 ? 1 : 0) << 1) | ((xyz.z > 0 ? 1 : 0) << 2);
    let near = 0;
    const localMin = subtractVector3i(min, blockOrigin);

    for (let i = 0; i < 3; i++) {
      if (vector3iComponent(localMin, i) === 0) {
        near |= 1 << (i * 2);
      }
      if (vector3iComponent(localMin, i) === last) {
        near |= 1 << (i * 2 + 1);
      }
    }

    const cornerPositions = buildCornerPositions(min, lodScale);
    const cornerSamples = cornerPositions.map((pos) => sample(samples, pos));
    const cornerNormals = buildCornerNormals(cornerPositions, samples);

    const caseCode =
      (signBit(cornerSamples[0]) << 0) |
      (signBit(cornerSamples[1]) << 1) |
      (signBit(cornerSamples[2]) << 2) |
      (signBit(cornerSamples[3]) << 3) |
      (signBit(cornerSamples[4]) << 4) |
      (signBit(cornerSamples[5]) << 5) |
      (signBit(cornerSamples[6]) << 6) |
      (signBit(cornerSamples[7]) << 7);

    const cacheCell = cache.getCellByVector(xyz);
    cacheCell.caseIndex = caseCode;
    if (caseCode === 0 || caseCode === 0xff) {
      return 0;
    }

    const cellClass = Tables.RegularCellClass[caseCode];
    const cellData = Tables.RegularCellData[cellClass];

    const triangleCount = cellData.getTriangleCount();
    const vertexCount = cellData.getVertexCount();
    const localVertexMapping = new Array<number>(12).fill(-1);

    for (let i = 0; i < vertexCount; i++) {
      const edgeCode = Tables.RegularVertexData[caseCode][i];
      const v0 = hiNibble(edgeCode & 0xff);
      const v1 = loNibble(edgeCode & 0xff);
      const p0 = cornerPositions[v0];
      const p1 = cornerPositions[v1];
      const n0 = cornerNormals[v0];
      const n1 = cornerNormals[v1];
      const d0 = sample(samples, p0);
      const d1 = sample(samples, p1);
      let t = intDiv(d1 << 8, d1 - d0);
      let u = 0x0100 - t;
      const t0 = t * S;
      const t1 = u * S;

      if ((t & 0x00ff) !== 0) {
        const dir = hiNibble((edgeCode >> 8) & 0xff);
        const idx = loNibble((edgeCode >> 8) & 0xff);
        const axisBits = dir & 0x07;
        let present = axisBits !== 0 && (axisBits & directionMask) === axisBits;

        if (present) {
          const prev = cache.getCellByVector(addVector3i(xyz, prevOffset(dir)));
          if (
            prev.caseIndex === 0 ||
            prev.caseIndex === 255 ||
            prev.dirs[idx] !== axisBits
          ) {
            localVertexMapping[i] = -1;
          } else {
            localVertexMapping[i] = prev.verts[idx];
          }
        }

        if (!present || localVertexMapping[i] < 0) {
          const interpolated = interpolate(toVector3f(p0), toVector3f(p1), p0, p1, samples, lodIndex);
          const pi = subtractVector3f(interpolated, blockOriginF);
          const normal = computeNormal(n0, n1, t0, t1);
          const vertex = createVertex(pi, normal, near, offset, lodIndex, cellSize);
          localVertexMapping[i] = verts.push(vertex) - 1;

          if ((dir & 8) !== 0) {
            cacheCell.verts[idx] = localVertexMapping[i];
            cacheCell.dirs[idx] = axisBits;
          }
        }

      } else if (t === 0 && v1 === 7) {
        const pi = subtractVector3f(combineVector3f(toVector3f(p1), t0, toVector3f(p1), t1), blockOriginF);
        const normal = computeNormal(n0, n1, t0, t1);
        const vertex = createVertex(pi, normal, near, offset, lodIndex, cellSize);
        localVertexMapping[i] = verts.push(vertex) - 1;
        cacheCell.verts[0] = localVertexMapping[i];
        cacheCell.dirs[0] = 0;
      } else {
        const dir = t === 0 ? (v1 ^ 7) : (v0 ^ 7);
        const axisBits = dir & 0x07;
        let present = axisBits !== 0 && (axisBits & directionMask) === axisBits;
        let reusedFrom = -1;

        if (present) {
          const prev = cache.getCellByVector(addVector3i(xyz, prevOffset(dir)));
          if (
            prev.caseIndex === 0 ||
            prev.caseIndex === 255 ||
            prev.dirs[0] !== axisBits
          ) {
            localVertexMapping[i] = -1;
          } else {
            localVertexMapping[i] = prev.verts[0];
            reusedFrom = prev.verts[0];
          }
        }

        if (!present || localVertexMapping[i] < 0) {
          const pi = subtractVector3f(
            combineVector3f(toVector3f(p0), t0, toVector3f(p1), t1),
            blockOriginF
          );
          const normal = computeNormal(n0, n1, t0, t1);
          const vertex = createVertex(pi, normal, near, offset, lodIndex, cellSize);
          localVertexMapping[i] = verts.push(vertex) - 1;
        }
      }
    }

    for (let t = 0; t < triangleCount; t++) {
      for (let i = 0; i < 3; i++) {
        indices.push(localVertexMapping[cellData.indices()[t * 3 + i]]);
      }
    }

    return triangleCount;
  }

  static polygonizeTransitionCell(
    offset: Vector3f,
    origin: Vector3i,
    localX: Vector3i,
    localY: Vector3i,
    localZ: Vector3i,
    x: number,
    y: number,
    cellSize: number,
    lodIndex: number,
    axis: number,
    directionMask: number,
    samples: DensityFunction,
    verts: TransvoxelVertex[],
    indices: number[],
    cache: TransitionCache,
    blockOrigin: Vector3i,
    invertWinding = false
  ): number {
    if (lodIndex < 1) {
      throw new RangeError("Transition cells require lodIndex >= 1.");
    }

    const lodScale = 1 << lodIndex;
    const sampleStep = 1 << (lodIndex - 1);
    const last = 16 * lodScale;
    const blockOriginF = vector3fFromVector3i(blockOrigin);
    const localOrigin = subtractVector3i(origin, blockOrigin);
    let near = 0;

    for (let i = 0; i < 3; i++) {
      if (vector3iComponent(localOrigin, i) === 0) {
        near |= 1 << (i * 2);
      }
      if (vector3iComponent(localOrigin, i) === last) {
        near |= 1 << (i * 2 + 1);
      }
    }

    const mx = multiplyVector3iScalar(localX, sampleStep, transitionBasisColumnXScratch);
    const my = multiplyVector3iScalar(localY, sampleStep, transitionBasisColumnYScratch);
    const mz = multiplyVector3iScalar(localZ, sampleStep, transitionBasisColumnZScratch);
    const basis = matrix3x3FromColumns(
      setVector3f(transitionBasisColumnXFloat, mx.x, mx.y, mx.z),
      setVector3f(transitionBasisColumnYFloat, my.x, my.y, my.z),
      setVector3f(transitionBasisColumnZFloat, mz.x, mz.y, mz.z)
    );

    const positions = transitionPositionsScratch;
    for (let i = 0; i < transitionCoordinates.length; i++) {
      const transformed = multiplyMatrix3x3Vector3i(basis, transitionCoordinates[i], transitionScaledCoordinate);
      setVector3i(
        positions[i],
        origin.x + transformed.x,
        origin.y + transformed.y,
        origin.z + transformed.z
      );
    }
    const normals = transitionNormalsScratch;
    for (let i = 0; i < 9; i++) {
      const p = positions[i];
      const nx =
        (sample(samples, addVector3i(p, vector3iUnitX)) - sample(samples, subtractVector3i(p, vector3iUnitX))) * 0.5;
      const ny =
        (sample(samples, addVector3i(p, vector3iUnitY)) - sample(samples, subtractVector3i(p, vector3iUnitY))) * 0.5;
      const nz =
        (sample(samples, addVector3i(p, vector3iUnitZ)) - sample(samples, subtractVector3i(p, vector3iUnitZ))) * 0.5;
      setVector3f(normals[i], nx, ny, nz);
      normalizeVector3f(normals[i], normals[i]);
    }

    normals[0x9] = normals[0];
    normals[0xA] = normals[2];
    normals[0xB] = normals[6];
    normals[0xC] = normals[8];

    const caseCode =
      (signBit(sample(samples, positions[0])) * 0x001) |
      (signBit(sample(samples, positions[1])) * 0x002) |
      (signBit(sample(samples, positions[2])) * 0x004) |
      (signBit(sample(samples, positions[5])) * 0x008) |
      (signBit(sample(samples, positions[8])) * 0x010) |
      (signBit(sample(samples, positions[7])) * 0x020) |
      (signBit(sample(samples, positions[6])) * 0x040) |
      (signBit(sample(samples, positions[3])) * 0x080) |
      (signBit(sample(samples, positions[4])) * 0x100);

    if (caseCode === 0 || caseCode === 0x1ff) {
      return 0;
    }

    const cacheCell = cache.getCell(x, y);
    cacheCell.caseIndex = caseCode;

    const classIndex = Tables.TransitionCellClass[caseCode];
    const cellData = Tables.TransitionRegularCellData[classIndex & 0x7f];
    const inverse = (classIndex & 0x80) !== 0;
    const finalInverse = inverse !== invertWinding;
    const localVertexMapping = new Array<number>(12).fill(-1);
    const vertexCount = cellData.getVertexCount();
    const triangleCount = cellData.getTriangleCount();

    const sampleIndexMap = transitionSampleIndexMap;
    for (let i = 0; i < vertexCount; i++) {
      const edgeCode = Tables.TransitionVertexData[caseCode][i];
      const v0 = hiNibble(edgeCode & 0xff);
      const v1 = loNibble(edgeCode & 0xff);
      const lowside = v0 > 8 && v1 > 8;
      const d0 = sample(samples, positions[sampleIndexMap[v0]]);
      const d1 = sample(samples, positions[sampleIndexMap[v1]]);
      let t = intDiv(d1 << 8, d1 - d0);
      let u = 0x0100 - t;
      const t0 = t * S;
      const t1 = u * S;
      const n0 = normals[v0];
      const n1 = normals[v1];
      const normal = computeNormal(n0, n1, t0, t1);

      if ((t & 0x00ff) !== 0) {
        const dir = hiNibble((edgeCode >> 8) & 0xff);
        const idx = loNibble((edgeCode >> 8) & 0xff);
        const present = (dir & directionMask) === dir;

        if (present) {
          const prev = cache.getCell(x - (dir & 1), y - ((dir >> 1) & 1));
          if (prev.caseIndex === 0 || prev.caseIndex === 0x1ff) {
            localVertexMapping[i] = -1;
          } else {
            localVertexMapping[i] = prev.verts[idx];
          }
        }

        if (!present || localVertexMapping[i] < 0) {
          const pi = interpolate(
            toVector3f(positions[v0]),
            toVector3f(positions[v1]),
            positions[sampleIndexMap[v0]],
            positions[sampleIndexMap[v1]],
            samples,
            lowside ? lodIndex : lodIndex - 1
          );
          const piLocal = subtractVector3f(pi, blockOriginF);

          let vertex: TransvoxelVertex;
          if (lowside) {
            const axisBoundary = vector3iComponent(origin, axis) - vector3iComponent(blockOrigin, axis);
            const adjusted = setAxisComponent(piLocal, axis, axisBoundary);
            const delta = computeDelta(adjusted, lodIndex, BLOCK_WIDTH, transitionDeltaScratch);
            const projected = multiplyVector3fScalar(
              projectNormal(normal, delta, transitionProjectionScratch),
              cellSize,
              transitionProjectionScratch
            );
            const secondaryBase = multiplyVector3fScalar(adjusted, cellSize, transitionSecondaryBaseScratch);
            const baseX = offset.x + secondaryBase.x;
            const baseY = offset.y + secondaryBase.y;
            const baseZ = offset.z + secondaryBase.z;
            vertex = {
              primary: unusedVertexPosition,
              secondary: createVector3f(baseX + projected.x, baseY + projected.y, baseZ + projected.z),
              normal,
              near,
            };
          } else {
            vertex = createVertex(piLocal, normal, 0, offset, lodIndex - 1, cellSize);
          }

          localVertexMapping[i] = verts.push(vertex) - 1;

          if ((dir & 8) !== 0) {
            cacheCell.verts[idx] = localVertexMapping[i];
          }
        }
      } else {
        const v = t === 0 ? v1 : v0;
        const cornerData = Tables.TransitionCornerData[v];
        const dir = hiNibble(cornerData);
        const idx = loNibble(cornerData);
        const present = (dir & directionMask) === dir;

        if (present) {
          const prev = cache.getCell(x - (dir & 1), y - ((dir >> 1) & 1));
          if (prev.caseIndex === 0 || prev.caseIndex === 0x1ff) {
            localVertexMapping[i] = -1;
          } else {
            localVertexMapping[i] = prev.verts[idx];
          }
        }

        if (!present || localVertexMapping[i] < 0) {
          let pi = subtractVector3f(toVector3f(positions[v]), blockOriginF);
          let vertex: TransvoxelVertex;

          if (v > 8) {
            const axisBoundary = vector3iComponent(origin, axis) - vector3iComponent(blockOrigin, axis);
            const adjusted = setAxisComponent(pi, axis, axisBoundary);
            const delta = computeDelta(adjusted, lodIndex, BLOCK_WIDTH, transitionDeltaScratch);
            const projected = multiplyVector3fScalar(
              projectNormal(normal, delta, transitionProjectionScratch),
              cellSize,
              transitionProjectionScratch
            );
            const secondaryBase = multiplyVector3fScalar(adjusted, cellSize, transitionSecondaryBaseScratch);
            const baseX = offset.x + secondaryBase.x;
            const baseY = offset.y + secondaryBase.y;
            const baseZ = offset.z + secondaryBase.z;
            vertex = {
              primary: unusedVertexPosition,
              secondary: createVector3f(baseX + projected.x, baseY + projected.y, baseZ + projected.z),
              normal,
              near,
            };
          } else {
            vertex = createVertex(pi, normal, 0, offset, lodIndex - 1, cellSize);
          }

          localVertexMapping[i] = verts.push(vertex) - 1;
          cacheCell.verts[idx] = localVertexMapping[i];
        }
      }
    }

    const cellIndices = cellData.indices();
    for (let t = 0; t < triangleCount; t++) {
      if (finalInverse) {
        indices.push(
          localVertexMapping[cellIndices[t * 3 + 2]],
          localVertexMapping[cellIndices[t * 3 + 1]],
          localVertexMapping[cellIndices[t * 3 + 0]]
        );
      } else {
        indices.push(
          localVertexMapping[cellIndices[t * 3 + 0]],
          localVertexMapping[cellIndices[t * 3 + 1]],
          localVertexMapping[cellIndices[t * 3 + 2]]
        );
      }
    }

    return triangleCount;
  }
}
