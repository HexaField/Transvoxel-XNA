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

export class TransvoxelMesher {
  private readonly regularCache = new RegularCache(BLOCK_WIDTH);
  private readonly transitionCache = new TransitionCache(BLOCK_WIDTH);

  constructor(private readonly volume: VolumeData) {}

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
}
