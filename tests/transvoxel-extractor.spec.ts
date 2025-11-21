import { describe, expect, it } from "vitest";

import { TransvoxelMesher, TransvoxelExtractor, TransitionFace } from "../src/surface-extractor/transvoxel-extractor";
import { FunctionalVolume, type DensityFunction, type VolumeData } from "../src/volume/volume-data";
import { Vector3i } from "../src/math/vector3i";
import { Vector3f } from "../src/math/vector3f";
import { RegularCache, TransitionCache } from "../src/surface-extractor/cache";
import { TransvoxelVertex, getRenderablePosition, unusedVertexPosition } from "../src/surface-extractor/vertex";
import { Tables } from "../src/lengyel/tables";
import { buildRegularCaseMesh, buildTransitionCaseMesh } from "./case-fixtures";

const createSphereVolume = (radius = 7, center = 8) =>
  new FunctionalVolume((x, y, z) => {
    const dx = x - center;
    const dy = y - center;
    const dz = z - center;
    const value = radius * radius - (dx * dx + dy * dy + dz * dz);
    return Math.floor(value);
  });

type AxisBounds = { x: number; y: number; z: number };

interface SampleBounds {
  min: AxisBounds;
  max: AxisBounds;
}

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

class RecordingVolume implements VolumeData {
  private bounds: SampleBounds = newBounds();

  constructor(private readonly fn: DensityFunction) {}

  sample(x: number, y: number, z: number): number {
    this.bounds.min.x = Math.min(this.bounds.min.x, x);
    this.bounds.min.y = Math.min(this.bounds.min.y, y);
    this.bounds.min.z = Math.min(this.bounds.min.z, z);
    this.bounds.max.x = Math.max(this.bounds.max.x, x);
    this.bounds.max.y = Math.max(this.bounds.max.y, y);
    this.bounds.max.z = Math.max(this.bounds.max.z, z);
    return this.fn(x, y, z);
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
    const volume = new FunctionalVolume(() => 64);
    const cache = new RegularCache(1);
    const verts: TransvoxelVertex[] = [];
    const indices: number[] = [];

    const triangles = TransvoxelExtractor.polygonizeRegularCell(
      Vector3i.zero,
      Vector3f.zero,
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
    const mesher = new TransvoxelMesher(volume);
    const options = { origin: Vector3i.zero, lodIndex: 0 as const, cellSize: 1 };

    const meshA = mesher.extractRegularBlock(options);
    const meshB = mesher.extractRegularBlock(options);

    expect(meshA.vertices.length).toBeGreaterThan(0);
    expect(meshA.indices.length % 3).toBe(0);
    expect(meshA.vertices.length).toBe(meshB.vertices.length);
    expect(meshA.indices.length).toBe(meshB.indices.length);
  });

  it("adds seam data when transition faces are extracted", () => {
    const volume = createSphereVolume();
    const mesher = new TransvoxelMesher(volume);
    const faces: TransitionFace[] = [
      "negativeX",
      "positiveX",
      "negativeY",
      "positiveY",
      "negativeZ",
      "positiveZ",
    ];

    const regularOnly = mesher.extractRegularBlock({ origin: Vector3i.zero, lodIndex: 1, cellSize: 1 });
    const withTransitions = mesher.extractBlock({
      origin: Vector3i.zero,
      lodIndex: 1,
      cellSize: 1,
      transitionFaces: faces,
    });

    expect(withTransitions.vertices.length).toBeGreaterThan(regularOnly.vertices.length);
    expect(withTransitions.indices.length).toBeGreaterThan(regularOnly.indices.length);

    const hasSecondary = withTransitions.vertices.some((vertex) => !vertex.secondary.equals(unusedVertexPosition));
    expect(hasSecondary).toBe(true);
  });

  it("samples transition faces within the block bounds", () => {
    const volume = new RecordingVolume(() => 64);
    const mesher = new TransvoxelMesher(volume);
    const lodIndex = 1;
    const extent = TransvoxelExtractor.BlockWidth * (1 << lodIndex);
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
      mesher.extractTransitionFaces({
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
      const axisExpected = directionForFace(face) === -1 ? 0 : extent;
      expect(min[axisKey]).toBeGreaterThanOrEqual(axisExpected - margin);
      expect(max[axisKey]).toBeLessThanOrEqual(axisExpected + margin);

      const center = extent / 2;
      for (const tangential of tangentialAxes[axisKey]) {
        expect(min[tangential]).toBeLessThanOrEqual(center + margin);
        expect(max[tangential]).toBeGreaterThanOrEqual(center - margin);
      }
    }
  });

  it("reuses cached vertices along the Z axis", () => {
    const volume = new FunctionalVolume((_, __, z) => Math.round((z - 0.5) * 127));
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

  it("returns secondary coordinates for renderable positions", () => {
    const secondary = new Vector3f(1, 2, 3);
    const vertex: TransvoxelVertex = {
      primary: unusedVertexPosition,
      secondary,
      normal: Vector3f.zero,
      near: 0,
    };

    expect(getRenderablePosition(vertex)).toBe(secondary);
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
  });
});
