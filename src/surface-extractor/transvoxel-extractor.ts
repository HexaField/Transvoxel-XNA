import { Tables, RegularCell } from "../lengyel/tables";
import { Vector3f } from "../math/vector3f";
import { Vector3i } from "../math/vector3i";
import { Matrix3x3 } from "../math/matrix3x3";
import { VolumeData } from "../volume/volume-data";
import { RegularCache, TransitionCache } from "./cache";
import { MeshData } from "./mesh-data";
import { TransvoxelVertex, unusedVertexPosition } from "./vertex";

const BLOCK_WIDTH = 16;
const S = 1.0 / 256.0;

const hiNibble = (value: number): number => (value >> 4) & 0x0f;
const loNibble = (value: number): number => value & 0x0f;
const signBit = (value: number): number => (value >> 7) & 1;

const toVector3f = (vector: Vector3i): Vector3f => new Vector3f(vector.x, vector.y, vector.z);

const setAxisComponent = (vector: Vector3f, axis: number, value: number): Vector3f => {
  switch (axis) {
    case 0:
      return new Vector3f(value, vector.y, vector.z);
    case 1:
      return new Vector3f(vector.x, value, vector.z);
    case 2:
      return new Vector3f(vector.x, vector.y, value);
    default:
      throw new RangeError(`Axis ${axis} is out of range for Vector3f.`);
  }
};

const sample = (volume: VolumeData, position: Vector3i): number => volume.sample(position.x, position.y, position.z);

const intDiv = (numerator: number, denominator: number): number => {
  if (denominator === 0) {
    return 0;
  }
  return Math.trunc(numerator / denominator);
};

const computeDelta = (v: Vector3f, k: number, s: number): Vector3f => {
  if (k < 1) {
    return Vector3f.zero;
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

  return new Vector3f(componentDelta(v.x), componentDelta(v.y), componentDelta(v.z));
};

const projectNormal = (n: Vector3f, delta: Vector3f): Vector3f => {
  const mat = new Matrix3x3(
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

  return mat.multiplyVector3f(delta);
};

const prevOffset = (dir: number): Vector3i =>
  new Vector3i(-(dir & 1), -((dir >> 1) & 1), -((dir >> 2) & 1));

const computeNormal = (n0: Vector3f, n1: Vector3f, t0: number, t1: number): Vector3f =>
  n0.multiplyScalar(t0).add(n1.multiplyScalar(t1)).normalize();

const createVertex = (
  pi: Vector3f,
  normal: Vector3f,
  near: number,
  offset: Vector3f,
  lodIndex: number
): TransvoxelVertex => {
  const primary = offset.add(pi);
  if (near > 0) {
    const delta = computeDelta(pi, lodIndex, BLOCK_WIDTH);
    const projected = projectNormal(normal, delta);
    return {
      primary,
      secondary: primary.add(projected),
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
  samples: VolumeData,
  lodIndex = 0
): Vector3f => {
  let s0 = sample(samples, p0);
  let s1 = sample(samples, p1);

  let t = intDiv(s1 << 8, s1 - s0);
  let u = 0x0100 - t;

  if ((t & 0x00ff) === 0) {
    return t === 0 ? v1 : v0;
  }

  for (let i = 0; i < lodIndex; i++) {
    const vm = v0.add(v1).divideScalar(2);
    const pm = new Vector3i((p0.x + p1.x) >> 1, (p0.y + p1.y) >> 1, (p0.z + p1.z) >> 1);
    const sm = sample(samples, pm);

    if (signBit(s0) !== signBit(sm)) {
      v1 = vm;
      p1 = pm;
      s1 = sm;
    } else {
      v0 = vm;
      p0 = pm;
      s0 = sm;
    }
  }

  t = intDiv(s1 << 8, s1 - s0);
  u = 0x0100 - t;

  return v0.multiplyScalar(t * S).add(v1.multiplyScalar(u * S));
};

const buildCornerPositions = (min: Vector3i, lodScale: number): Vector3i[] =>
  Tables.CornerIndex.map((corner) => min.add(corner.multiplyScalar(lodScale)));

const transitionCoordinates: ReadonlyArray<Vector3i> = [
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

const buildCornerNormals = (positions: Vector3i[], samples: VolumeData): Vector3f[] =>
  positions.map((p) => {
    const nx = (sample(samples, p.add(Vector3i.unitX)) - sample(samples, p.subtract(Vector3i.unitX))) * 0.5;
    const ny = (sample(samples, p.add(Vector3i.unitY)) - sample(samples, p.subtract(Vector3i.unitY))) * 0.5;
    const nz = (sample(samples, p.add(Vector3i.unitZ)) - sample(samples, p.subtract(Vector3i.unitZ))) * 0.5;
    return new Vector3f(nx, ny, nz).normalize();
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

const blockVector = (x: number, y: number, z: number): Vector3i =>
  new Vector3i(x, y, z);

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
    originOffset: blockVector(0, 0, BLOCK_WIDTH),
    localX: blockVector(1, 0, 0),
    localY: blockVector(0, 1, 0),
    localZ: blockVector(0, 0, -1),
  },
};

export class TransvoxelMesher {
  private readonly regularCache = new RegularCache(BLOCK_WIDTH);
  private readonly transitionCache = new TransitionCache(BLOCK_WIDTH);

  constructor(private readonly volume: VolumeData) {}

  extractBlock(options: ExtractBlockOptions & { transitionFaces?: TransitionFace[] }): MeshData {
    const regularMesh = this.extractRegularBlock(options);
    const faces = options.transitionFaces?.filter(Boolean) ?? [];
    if (faces.length === 0) {
      return regularMesh;
    }

    const transitionMesh = this.extractTransitionFaces({ ...options, faces });
    if (transitionMesh.vertices.length === 0) {
      return regularMesh;
    }

    const vertexOffset = regularMesh.vertices.length;
    transitionMesh.vertices.forEach((vertex) => regularMesh.vertices.push(vertex));
    transitionMesh.indices.forEach((index) => regularMesh.indices.push(index + vertexOffset));
    return regularMesh;
  }

  extractRegularBlock({ origin, offset, lodIndex = 0, cellSize = 1 }: ExtractBlockOptions): MeshData {
    const vertices: TransvoxelVertex[] = [];
    const indices: number[] = [];
    this.regularCache.reset();

    const blockOffset =
      offset ?? new Vector3f(origin.x * cellSize, origin.y * cellSize, origin.z * cellSize);

    for (let x = 0; x < BLOCK_WIDTH; x++) {
      for (let y = 0; y < BLOCK_WIDTH; y++) {
        for (let z = 0; z < BLOCK_WIDTH; z++) {
          const xyz = new Vector3i(x, y, z);
          const min = origin.add(xyz.multiplyScalar(1 << lodIndex));
          TransvoxelExtractor.polygonizeRegularCell(
            min,
            blockOffset,
            xyz,
            this.volume,
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

  extractTransitionFaces({
    origin,
    offset,
    lodIndex = 0,
    cellSize = 1,
    faces,
  }: ExtractBlockOptions & { faces: TransitionFace[] }): MeshData {
    if (lodIndex < 1) {
      throw new RangeError("Transition faces require lodIndex >= 1.");
    }

    const uniqueFaces = Array.from(new Set(faces));
    if (uniqueFaces.length === 0) {
      return new MeshData();
    }

    const blockOffset =
      offset ?? new Vector3f(origin.x * cellSize, origin.y * cellSize, origin.z * cellSize);

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
    verts: TransvoxelVertex[],
    indices: number[]
  ): void {
    const lodScale = 1 << lodIndex;
    const sampleStep = 1 << (lodIndex - 1);
    const stride = sampleStep << 1;

    const faceOrigin = origin.add(descriptor.originOffset.multiplyScalar(lodScale));

    for (let y = 0; y < BLOCK_WIDTH; y++) {
      const rowOrigin = faceOrigin.add(descriptor.localY.multiplyScalar(y * stride));
      for (let x = 0; x < BLOCK_WIDTH; x++) {
        const cellOrigin = rowOrigin.add(descriptor.localX.multiplyScalar(x * stride));
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
          this.volume,
          verts,
          indices,
          this.transitionCache
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
    samples: VolumeData,
    lodIndex: number,
    cellSize: number,
    verts: TransvoxelVertex[],
    indices: number[],
    cache: RegularCache
  ): number {
    const lodScale = 1 << lodIndex;
    const last = 15 * lodScale;
    const directionMask =
      (xyz.x > 0 ? 1 : 0) | ((xyz.y > 0 ? 1 : 0) << 1) | ((xyz.z > 0 ? 1 : 0) << 2);
    let near = 0;

    for (let i = 0; i < 3; i++) {
      if (min.component(i) === 0) {
        near |= 1 << (i * 2);
      }
      if (min.component(i) === last) {
        near |= 1 << (i * 2 + 1);
      }
    }

    const cornerPositions = buildCornerPositions(min, lodScale);
    const cornerSamples = cornerPositions.map((pos) => sample(samples, pos));
    const cornerNormals = buildCornerNormals(cornerPositions, samples);

    const caseCode =
      ((cornerSamples[0] >> 7) & 0x01) |
      ((cornerSamples[1] >> 6) & 0x02) |
      ((cornerSamples[2] >> 5) & 0x04) |
      ((cornerSamples[3] >> 4) & 0x08) |
      ((cornerSamples[4] >> 3) & 0x10) |
      ((cornerSamples[5] >> 2) & 0x20) |
      ((cornerSamples[6] >> 1) & 0x40) |
      (cornerSamples[7] & 0x80);

    const cacheCell = cache.getCellByVector(xyz);
    cacheCell.caseIndex = caseCode;
    if ((caseCode ^ ((cornerSamples[7] >> 7) & 0xff)) === 0) {
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
        let present = (dir & directionMask) === dir;

        if (present) {
          const prev = cache.getCellByVector(xyz.add(prevOffset(dir)));
          if (prev.caseIndex === 0 || prev.caseIndex === 255) {
            localVertexMapping[i] = -1;
          } else {
            localVertexMapping[i] = prev.verts[idx];
          }
        }

        if (!present || localVertexMapping[i] < 0) {
          const pi = interpolate(toVector3f(p0), toVector3f(p1), p0, p1, samples, lodIndex);
          const normal = computeNormal(n0, n1, t0, t1);
          const vertex = createVertex(pi, normal, near, offset, lodIndex);
          localVertexMapping[i] = verts.push(vertex) - 1;

          if ((dir & 8) !== 0) {
            cacheCell.verts[idx] = localVertexMapping[i];
          }
        }
      } else if (t === 0 && v1 === 7) {
        const pi = toVector3f(p1).multiplyScalar(t0).add(toVector3f(p1).multiplyScalar(t1));
        const normal = computeNormal(n0, n1, t0, t1);
        const vertex = createVertex(pi, normal, near, offset, lodIndex);
        localVertexMapping[i] = verts.push(vertex) - 1;
        cacheCell.verts[0] = localVertexMapping[i];
      } else {
        const dir = t === 0 ? (v1 ^ 7) : (v0 ^ 7);
        let present = (dir & directionMask) === dir;

        if (present) {
          const prev = cache.getCellByVector(xyz.add(prevOffset(dir)));
          if (prev.caseIndex === 0 || prev.caseIndex === 255) {
            localVertexMapping[i] = -1;
          } else {
            localVertexMapping[i] = prev.verts[0];
          }
        }

        if (!present || localVertexMapping[i] < 0) {
          const pi = toVector3f(p0).multiplyScalar(t0).add(toVector3f(p1).multiplyScalar(t1));
          const normal = computeNormal(n0, n1, t0, t1);
          const vertex = createVertex(pi, normal, near, offset, lodIndex);
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
    samples: VolumeData,
    verts: TransvoxelVertex[],
    indices: number[],
    cache: TransitionCache
  ): number {
    if (lodIndex < 1) {
      throw new RangeError("Transition cells require lodIndex >= 1.");
    }

    const lodScale = 1 << lodIndex;
    const sampleStep = 1 << (lodIndex - 1);
    const last = 16 * lodScale;
    let near = 0;

    for (let i = 0; i < 3; i++) {
      if (origin.component(i) === 0) {
        near |= 1 << (i * 2);
      }
      if (origin.component(i) === last) {
        near |= 1 << (i * 2 + 1);
      }
    }

    const mx = localX.multiplyScalar(sampleStep);
    const my = localY.multiplyScalar(sampleStep);
    const mz = localZ.multiplyScalar(sampleStep);
    const basis = Matrix3x3.fromColumns(
      Vector3f.fromVector3i(mx),
      Vector3f.fromVector3i(my),
      Vector3f.fromVector3i(mz)
    );

    const positions = transitionCoordinates.map((coord) =>
      origin.add(basis.multiplyVector3i(coord))
    );

    const normals = new Array<Vector3f>(13);
    for (let i = 0; i < 9; i++) {
      const p = positions[i];
      const nx = (sample(samples, p.add(Vector3i.unitX)) - sample(samples, p.subtract(Vector3i.unitX))) * 0.5;
      const ny = (sample(samples, p.add(Vector3i.unitY)) - sample(samples, p.subtract(Vector3i.unitY))) * 0.5;
      const nz = (sample(samples, p.add(Vector3i.unitZ)) - sample(samples, p.subtract(Vector3i.unitZ))) * 0.5;
      normals[i] = new Vector3f(nx, ny, nz).normalize();
    }

    normals[0x9] = normals[0];
    normals[0xA] = normals[2];
    normals[0xB] = normals[6];
    normals[0xC] = normals[8];

    const samplePositions = [
      positions[0],
      positions[1],
      positions[2],
      positions[3],
      positions[4],
      positions[5],
      positions[6],
      positions[7],
      positions[8],
      positions[0],
      positions[2],
      positions[6],
      positions[8],
    ];

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
    const localVertexMapping = new Array<number>(12).fill(-1);
    const vertexCount = cellData.getVertexCount();
    const triangleCount = cellData.getTriangleCount();

    for (let i = 0; i < vertexCount; i++) {
      const edgeCode = Tables.TransitionVertexData[caseCode][i];
      const v0 = hiNibble(edgeCode & 0xff);
      const v1 = loNibble(edgeCode & 0xff);
      const lowside = v0 > 8 && v1 > 8;
      const d0 = sample(samples, samplePositions[v0]);
      const d1 = sample(samples, samplePositions[v1]);
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
            samplePositions[v0],
            samplePositions[v1],
            samples,
            lowside ? lodIndex : lodIndex - 1
          );

          let vertex: TransvoxelVertex;
          if (lowside) {
            const adjusted = setAxisComponent(pi, axis, origin.component(axis));
            const delta = computeDelta(adjusted, lodIndex, BLOCK_WIDTH);
            const projected = projectNormal(normal, delta);
            vertex = {
              primary: unusedVertexPosition,
              secondary: offset.add(adjusted).add(projected),
              normal,
              near,
            };
          } else {
            vertex = createVertex(pi, normal, 0, offset, lodIndex - 1);
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
          let pi = toVector3f(positions[v]);
          let vertex: TransvoxelVertex;

          if (v > 8) {
            const adjusted = setAxisComponent(pi, axis, origin.component(axis));
            const delta = computeDelta(adjusted, lodIndex, BLOCK_WIDTH);
            const projected = projectNormal(normal, delta);
            vertex = {
              primary: unusedVertexPosition,
              secondary: offset.add(adjusted).add(projected),
              normal,
              near,
            };
          } else {
            vertex = createVertex(pi, normal, 0, offset, lodIndex - 1);
          }

          localVertexMapping[i] = verts.push(vertex) - 1;
          cacheCell.verts[idx] = localVertexMapping[i];
        }
      }
    }

    const cellIndices = cellData.indices();
    for (let t = 0; t < triangleCount; t++) {
      if (inverse) {
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
