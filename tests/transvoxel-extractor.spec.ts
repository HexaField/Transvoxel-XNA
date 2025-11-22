import { describe, expect, it } from "vitest";

import { TransvoxelMesher, TransvoxelExtractor, TransitionFace } from "../src/surface-extractor/transvoxel-extractor";
import type { DensityFunction } from "../src/volume/volume-data";
import { Vector3i } from "../src/math/vector3i";
import { Vector3f } from "../src/math/vector3f";
import { RegularCache, TransitionCache } from "../src/surface-extractor/cache";
import { TransvoxelVertex, getRenderablePosition, unusedVertexPosition } from "../src/surface-extractor/vertex";
import { Tables } from "../src/lengyel/tables";
import { buildRegularCaseMesh, buildTransitionCaseMesh } from "./case-fixtures";
import { MeshData } from "../src/surface-extractor/mesh-data";

const createSphereVolume = (radius = 7, center = 8): DensityFunction => (x, y, z) => {
  const dx = x - center;
  const dy = y - center;
  const dz = z - center;
  const value = radius * radius - (dx * dx + dy * dy + dz * dz);
  return Math.floor(value);
};



type AxisBounds = { x: number; y: number; z: number };

interface SampleBounds {
  min: AxisBounds;
  max: AxisBounds;
}

const computeMeshBounds = (vertices: TransvoxelVertex[]): SampleBounds => {
  const min = { x: Infinity, y: Infinity, z: Infinity };
  const max = { x: -Infinity, y: -Infinity, z: -Infinity };

  for (const vertex of vertices) {
    const position = getRenderablePosition(vertex);
    min.x = Math.min(min.x, position.x);
    min.y = Math.min(min.y, position.y);
    min.z = Math.min(min.z, position.z);
    max.x = Math.max(max.x, position.x);
    max.y = Math.max(max.y, position.y);
    max.z = Math.max(max.z, position.z);
  }

  return { min, max };
};

const triangleArea = (a: Vector3f, b: Vector3f, c: Vector3f): number => {
  const ab = b.subtract(a);
  const ac = c.subtract(a);
  return ab.cross(ac).length() * 0.5;
};

const hasDegenerateTriangles = (mesh: MeshData, epsilon = 1e-5): boolean => {
  const { vertices, indices } = mesh;
  for (let i = 0; i < indices.length; i += 3) {
    const a = getRenderablePosition(vertices[indices[i]]);
    const b = getRenderablePosition(vertices[indices[i + 1]]);
    const c = getRenderablePosition(vertices[indices[i + 2]]);
    if (triangleArea(a, b, c) <= epsilon) {
      return true;
    }
  }
  return false;
};

const expectVerticesNearSurface = (
  mesh: MeshData,
  sdf: (position: Vector3f) => number,
  tolerance: number
): void => {
  for (const vertex of mesh.vertices) {
    const position = getRenderablePosition(vertex);
    expect(Math.abs(sdf(position))).toBeLessThanOrEqual(tolerance);
  }
};

const axisKeyForFace = (face: TransitionFace): keyof AxisBounds => {
  if (face.endsWith("X")) {
    return "x";
  }
  if (face.endsWith("Y")) {
    return "y";
  }
  return "z";
};

const directionForFace = (face: TransitionFace): -1 | 1 =>
  face.startsWith("negative") ? -1 : 1;

const tangentialAxes: Record<keyof AxisBounds, Array<keyof AxisBounds>> = {
  x: ["y", "z"],
  y: ["x", "z"],
  z: ["x", "y"],
};

const newBounds = (): SampleBounds => ({
  min: { x: Number.POSITIVE_INFINITY, y: Number.POSITIVE_INFINITY, z: Number.POSITIVE_INFINITY },
  max: { x: Number.NEGATIVE_INFINITY, y: Number.NEGATIVE_INFINITY, z: Number.NEGATIVE_INFINITY },
});

class RecordingSampler {
  private bounds: SampleBounds = newBounds();
  readonly sampler: DensityFunction;

  constructor(private readonly fn: DensityFunction) {
    this.sampler = (x, y, z) => {
      this.bounds.min.x = Math.min(this.bounds.min.x, x);
      this.bounds.min.y = Math.min(this.bounds.min.y, y);
      this.bounds.min.z = Math.min(this.bounds.min.z, z);
      this.bounds.max.x = Math.max(this.bounds.max.x, x);
      this.bounds.max.y = Math.max(this.bounds.max.y, y);
      this.bounds.max.z = Math.max(this.bounds.max.z, z);
      return this.fn(x, y, z);
    };
  }

  reset(): void {
    this.bounds = newBounds();
  }

  getBounds(): SampleBounds {
    return this.bounds;
  }
}

describe("TransvoxelExtractor", () => {
  it("returns zero geometry for homogeneous densities", () => {
    const volume: DensityFunction = () => 64;
    const cache = new RegularCache(1);
    const verts: TransvoxelVertex[] = [];
    const indices: number[] = [];

    const triangles = TransvoxelExtractor.polygonizeRegularCell(
      Vector3i.zero,
      Vector3f.zero,
      Vector3i.zero,
      Vector3i.zero,
      volume,
      0,
      1,
      verts,
      indices,
      cache
    );

    expect(triangles).toBe(0);
    expect(verts).toHaveLength(0);
    expect(indices).toHaveLength(0);
  });

  it("produces deterministic regular block meshes", () => {
    const volume = createSphereVolume();
    const mesher = new TransvoxelMesher();
    const options = { origin: Vector3i.zero, lodIndex: 0 as const, cellSize: 1 };

    const meshA = mesher.extractRegularBlock(volume, options);
    const meshB = mesher.extractRegularBlock(volume, options);

    expect(meshA.vertices.length).toBeGreaterThan(0);
    expect(meshA.indices.length % 3).toBe(0);
    expect(meshA.vertices.length).toBe(meshB.vertices.length);
    expect(meshA.indices.length).toBe(meshB.indices.length);
  });

  it("adds seam data when transition faces are extracted", () => {
    const planeOffset = 24;
    const scale = 16;
    const planeField: DensityFunction = (x, y, z) => Math.floor((x + y + z - planeOffset) * scale);
    const mesher = new TransvoxelMesher();
    const faces: TransitionFace[] = [
      "negativeX",
      "positiveX",
      "negativeY",
      "positiveY",
      "negativeZ",
      "positiveZ",
    ];

    const regularOnly = mesher.extractRegularBlock(planeField, { origin: Vector3i.zero, lodIndex: 1, cellSize: 1 });
    const withTransitions = mesher.extractBlock(planeField, {
      origin: Vector3i.zero,
      lodIndex: 1,
      cellSize: 1,
      transitionFaces: faces,
    });

    expect(regularOnly.vertices.length).toBeGreaterThan(0);
    expect(withTransitions.vertices.length).toBeGreaterThan(regularOnly.vertices.length);
    expect(withTransitions.indices.length).toBeGreaterThan(regularOnly.indices.length);

    const hasSecondary = withTransitions.vertices.some((vertex) => !vertex.secondary.equals(unusedVertexPosition));
    expect(hasSecondary).toBe(true);
  });

  it("samples transition faces within the block bounds", () => {
    const volume = new RecordingSampler(() => 64);
    const mesher = new TransvoxelMesher();
    const lodIndex = 2;
    const lodScale = 1 << lodIndex;
    const extent = TransvoxelExtractor.BlockWidth * lodScale;
    const margin = 1;
    const faces: TransitionFace[] = [
      "negativeX",
      "positiveX",
      "negativeY",
      "positiveY",
      "negativeZ",
      "positiveZ",
    ];

    for (const face of faces) {
      volume.reset();
      mesher.extractTransitionFaces(volume.sampler, {
        origin: Vector3i.zero,
        lodIndex,
        cellSize: 1,
        faces: [face],
      });

      const { min, max } = volume.getBounds();
      expect(min.x).toBeGreaterThanOrEqual(-margin);
      expect(min.y).toBeGreaterThanOrEqual(-margin);
      expect(min.z).toBeGreaterThanOrEqual(-margin);
      expect(max.x).toBeLessThanOrEqual(extent + margin);
      expect(max.y).toBeLessThanOrEqual(extent + margin);
      expect(max.z).toBeLessThanOrEqual(extent + margin);

      const axisKey = axisKeyForFace(face);
      const direction = directionForFace(face);
      const axisBoundary = direction === -1 ? 0 : extent;

      expect(min[axisKey]).toBeGreaterThanOrEqual(axisBoundary - (lodScale + margin));
      expect(max[axisKey]).toBeLessThanOrEqual(axisBoundary + (lodScale + margin));

      const center = extent / 2;
      for (const tangential of tangentialAxes[axisKey]) {
        expect(min[tangential]).toBeLessThanOrEqual(center + margin);
        expect(max[tangential]).toBeGreaterThanOrEqual(center - margin);
      }
    }
  });

  it("matches the coarse cell depth along the normal axis", () => {
    const mesher = new TransvoxelMesher();
    const lodIndex = 2;
    const lodScale = 1 << lodIndex;
    const extent = TransvoxelExtractor.BlockWidth * lodScale;
    const coarseDepth = extent / 2;
    const margin = 2.5;
    const volume = createSphereVolume(coarseDepth + 4, extent / 2);
    const faces: TransitionFace[] = [
      "negativeX",
      "positiveX",
      "negativeY",
      "positiveY",
      "negativeZ",
      "positiveZ",
    ];
    for (const face of faces) {
      const mesh = mesher.extractTransitionFaces(volume, {
        origin: Vector3i.zero,
        lodIndex,
        cellSize: 1,
        faces: [face],
      });

      const axisKey = axisKeyForFace(face);
      expect(mesh.vertices.length).toBeGreaterThan(0);
      const bounds = computeMeshBounds(mesh.vertices);
      const direction = directionForFace(face);
      const axisBoundary = direction === -1 ? 0 : extent;

      expect(bounds.min[axisKey]).toBeLessThanOrEqual(axisBoundary + margin);
      expect(bounds.max[axisKey]).toBeGreaterThanOrEqual(axisBoundary - margin);
      expect(bounds.min[axisKey]).toBeGreaterThanOrEqual(axisBoundary - coarseDepth - margin);
      expect(bounds.max[axisKey]).toBeLessThanOrEqual(axisBoundary + coarseDepth + margin);
    }
  });

  it("reuses cached vertices along the Z axis", () => {
    const volume: DensityFunction = (_x, _y, z) => Math.round(Math.cos(z * Math.PI) * 127);
    const cache = new RegularCache(2);
    const verts: TransvoxelVertex[] = [];
    const indices: number[] = [];
    const lodIndex = 0;
    const cellSize = 1;
    const offset = Vector3f.zero;

    TransvoxelExtractor.polygonizeRegularCell(
      Vector3i.zero,
      offset,
      Vector3i.zero,
      Vector3i.zero,
      volume,
      lodIndex,
      cellSize,
      verts,
      indices,
      cache
    );

    const afterFirstCell = verts.length;

    TransvoxelExtractor.polygonizeRegularCell(
      new Vector3i(0, 0, 1),
      offset,
      new Vector3i(0, 0, 1),
      Vector3i.zero,
      volume,
      lodIndex,
      cellSize,
      verts,
      indices,
      cache
    );

    expect(afterFirstCell).toBeGreaterThan(0);
    expect(verts.length).toBe(afterFirstCell + 4);
  });

  it("positions regular block vertices on the planar isosurface", () => {
    const planeOffset = 24;
    const scale = 16;
    const signedDistance = (x: number, y: number, z: number): number => x + y + z - planeOffset;
    const planeField: DensityFunction = (x, y, z) => Math.floor(signedDistance(x, y, z) * scale);
    const mesher = new TransvoxelMesher();

    const mesh = mesher.extractRegularBlock(planeField, {
      origin: Vector3i.zero,
      lodIndex: 0,
      cellSize: 1,
    });

    expect(mesh.vertices.length).toBeGreaterThan(0);
    expectVerticesNearSurface(mesh, (position) => signedDistance(position.x, position.y, position.z), 0.75);
  });

  it("places translated regular blocks at the correct world coordinates", () => {
    const planeOffset = 24;
    const scale = 16;
    const signedDistance = (x: number, y: number, z: number): number => x + y + z - planeOffset;
    const planeField: DensityFunction = (x, y, z) => Math.floor(signedDistance(x, y, z) * scale);
    const translation = new Vector3i(16, -8, -8);
    const mesher = new TransvoxelMesher();

    const base = mesher.extractRegularBlock(planeField, {
      origin: Vector3i.zero,
      lodIndex: 0,
      cellSize: 1,
    });
    const translated = mesher.extractRegularBlock(planeField, {
      origin: translation,
      lodIndex: 0,
      cellSize: 1,
    });

    const baseBounds = computeMeshBounds(base.vertices);
    const translatedBounds = computeMeshBounds(translated.vertices);
    const translationComponents: AxisBounds = {
      x: translation.x,
      y: translation.y,
      z: translation.z,
    };

    expect(base.vertices.length).toBeGreaterThan(0);
    expect(translated.vertices.length).toBeGreaterThan(0);

    (Object.keys(baseBounds.min) as (keyof AxisBounds)[]).forEach((axis) => {
      expect(Math.abs(translatedBounds.min[axis] - (baseBounds.min[axis] + translationComponents[axis]))).toBeLessThan(0.01);
      expect(Math.abs(translatedBounds.max[axis] - (baseBounds.max[axis] + translationComponents[axis]))).toBeLessThan(0.01);
    });
  });

  it("scales meshes according to the provided cell size", () => {
    const sphereField = createSphereVolume(7, 24);
    const mesher = new TransvoxelMesher();

    const unitMesh = mesher.extractRegularBlock(sphereField, {
      origin: Vector3i.zero,
      lodIndex: 0,
      cellSize: 1,
    });

    const scaledMesh = mesher.extractRegularBlock(sphereField, {
      origin: Vector3i.zero,
      lodIndex: 0,
      cellSize: 2,
    });

    const unitBounds = computeMeshBounds(unitMesh.vertices);
    const scaledBounds = computeMeshBounds(scaledMesh.vertices);
    expect(scaledBounds.min.x).toBeCloseTo(unitBounds.min.x * 2, 5);
    expect(scaledBounds.max.x).toBeCloseTo(unitBounds.max.x * 2, 5);
    expect(scaledBounds.min.y).toBeCloseTo(unitBounds.min.y * 2, 5);
    expect(scaledBounds.max.y).toBeCloseTo(unitBounds.max.y * 2, 5);
    expect(scaledBounds.min.z).toBeCloseTo(unitBounds.min.z * 2, 5);
    expect(scaledBounds.max.z).toBeCloseTo(unitBounds.max.z * 2, 5);
  });

  it("marks translated boundary cells as near", () => {
    const planeField: DensityFunction = (x, _y, _z) => Math.floor((x - 17) * 16);
    const mesher = new TransvoxelMesher();
    const mesh = mesher.extractRegularBlock(planeField, {
      origin: new Vector3i(16, 0, 0),
      lodIndex: 0,
      cellSize: 1,
    });

    expect(mesh.vertices.length).toBeGreaterThan(0);
    const hasSecondary = mesh.vertices.some(
      (vertex) => vertex.near !== 0 && !vertex.secondary.equals(unusedVertexPosition)
    );
    expect(hasSecondary).toBe(true);
  });

  it("prefers secondary coordinates for renderable positions", () => {
    const secondary = new Vector3f(1, 2, 3);
    const vertex: TransvoxelVertex = {
      primary: Vector3f.zero,
      secondary,
      normal: Vector3f.zero,
      near: 1,
    };

    expect(getRenderablePosition(vertex)).toBe(secondary);

    const fallback: TransvoxelVertex = {
      primary: secondary,
      secondary: unusedVertexPosition,
      normal: Vector3f.zero,
      near: 0,
    };

    expect(getRenderablePosition(fallback)).toBe(secondary);
  });
});

describe("Reference case tables", () => {
  const regularCache = new RegularCache(TransvoxelExtractor.BlockWidth);
  const transitionCache = new TransitionCache(TransvoxelExtractor.BlockWidth);

  it("matches every regular case entry", () => {
    const mismatches: Array<{
      caseCode: number;
      classIndex: number;
      expectedVertices: number;
      actualVertices: number;
      expectedTriangles: number;
      actualTriangles: number;
    }> = [];
    const degenerateCases: number[] = [];

    for (let caseCode = 0; caseCode < 256; caseCode++) {
      const mesh = buildRegularCaseMesh(caseCode, regularCache);
      const actualVertices = mesh.vertices.length;
      const actualTriangles = mesh.indices.length / 3;
      const classIndex = Tables.RegularCellClass[caseCode];
      const entry = Tables.RegularCellData[classIndex];
      const expectedVertices = entry.getVertexCount();
      const expectedTriangles = entry.getTriangleCount();

      if (actualVertices !== expectedVertices || actualTriangles !== expectedTriangles) {
        mismatches.push({
          caseCode,
          classIndex,
          expectedVertices,
          actualVertices,
          expectedTriangles,
          actualTriangles,
        });
      }

      if (mesh.indices.length > 0 && hasDegenerateTriangles(mesh)) {
        degenerateCases.push(caseCode);
      }
    }

    if (mismatches.length > 0) {
      console.table(
        mismatches.map((mismatch) => ({
          case: `0x${mismatch.caseCode.toString(16).toUpperCase().padStart(2, "0")}`,
          classIndex: mismatch.classIndex,
          expected: `${mismatch.expectedVertices}/${mismatch.expectedTriangles}`,
          actual: `${mismatch.actualVertices}/${mismatch.actualTriangles}`,
        }))
      );
    }

    expect(mismatches).toHaveLength(0);
    expect(degenerateCases).toHaveLength(0);
  });

  it("matches every transition case entry", () => {
    const mismatches: Array<{
      caseCode: number;
      classIndex: number;
      inverted: boolean;
      expectedVertices: number;
      actualVertices: number;
      expectedTriangles: number;
      actualTriangles: number;
    }> = [];
    const degenerateCases: number[] = [];

    for (let caseCode = 0; caseCode < 512; caseCode++) {
      const mesh = buildTransitionCaseMesh(caseCode, transitionCache);
      const actualVertices = mesh.vertices.length;
      const actualTriangles = mesh.indices.length / 3;
      const rawClass = Tables.TransitionCellClass[caseCode];
      const classIndex = rawClass & 0x7f;
      const inverted = (rawClass & 0x80) !== 0;
      const entry = Tables.TransitionRegularCellData[classIndex];
      const expectedVertices = entry.getVertexCount();
      const expectedTriangles = entry.getTriangleCount();

      if (actualVertices !== expectedVertices || actualTriangles !== expectedTriangles) {
        mismatches.push({
          caseCode,
          classIndex,
          inverted,
          expectedVertices,
          actualVertices,
          expectedTriangles,
          actualTriangles,
        });
      }

      if (mesh.indices.length > 0 && hasDegenerateTriangles(mesh)) {
        degenerateCases.push(caseCode);
      }
    }

    if (mismatches.length > 0) {
      console.table(
        mismatches.map((mismatch) => ({
          case: `0x${mismatch.caseCode.toString(16).toUpperCase().padStart(3, "0")}`,
          classIndex: mismatch.classIndex,
          inverted: mismatch.inverted,
          expected: `${mismatch.expectedVertices}/${mismatch.expectedTriangles}`,
          actual: `${mismatch.actualVertices}/${mismatch.actualTriangles}`,
        }))
      );
    }

    expect(mismatches).toHaveLength(0);
    expect(degenerateCases).toHaveLength(0);
  });
});
