import { describe, it, expect } from "vitest";

import {
  TransvoxelExtractor,
  TransvoxelMesher,
  type TransitionFace,
} from "../src/surface-extractor/transvoxel-extractor";
import type { DensityFunction } from "../src/volume/volume-data";
import { Vector3i } from "../src/math/vector3i";
import { getRenderablePosition } from "../src/surface-extractor/vertex";
import type { MeshData } from "../src/surface-extractor/mesh-data";
import type { Vector3f } from "../src/math/vector3f";

const BLOCK_WIDTH = TransvoxelExtractor.BlockWidth;
const COARSE_LOD = 1;
const FINE_LOD = 0;
const coarseExtent = BLOCK_WIDTH << COARSE_LOD;
const fineExtent = BLOCK_WIDTH << FINE_LOD;

const sphereField = createSphereField(coarseExtent * 0.75, coarseExtent * 0.5);
const mesher = new TransvoxelMesher();

const axisForFace = (face: TransitionFace): AxisKey => {
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

const tangentialAxes: Record<AxisKey, [AxisKey, AxisKey]> = {
  x: ["y", "z"],
  y: ["x", "z"],
  z: ["x", "y"],
};

type AxisKey = "x" | "y" | "z";

describe("transition seams", () => {
  const faces: TransitionFace[] = [
    "negativeX",
    "positiveX",
    "negativeY",
    "positiveY",
    "negativeZ",
    "positiveZ",
  ];

  for (const face of faces) {
    it(`bridges the seam on ${face}`, () => {
      const axis = axisForFace(face);
      const direction = directionForFace(face);
      const seamCoordinate = direction === -1 ? 0 : coarseExtent;
      const coarseOrigin = Vector3i.zero;
      const highOrigin = buildHighOrigin(axis, direction);

      const transitionMesh = mesher.extractTransitionFaces(sphereField, {
        origin: coarseOrigin,
        lodIndex: COARSE_LOD,
        cellSize: 1,
        faces: [face],
      });

      expect(transitionMesh.vertices.length).toBeGreaterThan(0);

      const highMesh = mesher.extractRegularBlock(sphereField, {
        origin: highOrigin,
        lodIndex: FINE_LOD,
        cellSize: 1,
      });

      expect(highMesh.vertices.length).toBeGreaterThan(0);

      const tangentialBounds = buildTangentialBounds(axis, highOrigin);
      const transitionEdges = collectSeamEdges(
        transitionMesh,
        axis,
        seamCoordinate,
        tangentialBounds
      );
      const highEdges = collectSeamEdges(
        highMesh,
        axis,
        seamCoordinate,
        tangentialBounds
      );

      expect(transitionEdges.size).toBeGreaterThan(0);
      expect(highEdges.size).toBeGreaterThan(0);

      const missing: string[] = [];
      transitionEdges.forEach((_count, key) => {
        if (!highEdges.has(key)) {
          missing.push(key);
        }
      });

      expect(missing).toHaveLength(0);
    });
  }
});

function createSphereField(radius: number, center: number): DensityFunction {
  const r2 = radius * radius;
  return (x, y, z) => {
    const dx = x - center;
    const dy = y - center;
    const dz = z - center;
    return Math.floor(r2 - (dx * dx + dy * dy + dz * dz));
  };
}

function buildHighOrigin(axis: AxisKey, direction: -1 | 1): Vector3i {
  const offset = direction === -1 ? -fineExtent : coarseExtent;
  switch (axis) {
    case "x":
      return new Vector3i(offset, 0, 0);
    case "y":
      return new Vector3i(0, offset, 0);
    default:
      return new Vector3i(0, 0, offset);
  }
}

type Bounds = Record<AxisKey, { min: number; max: number }>;

function buildTangentialBounds(axis: AxisKey, highOrigin: Vector3i): Bounds {
  const bounds: Bounds = {
    x: { min: 0, max: 0 },
    y: { min: 0, max: 0 },
    z: { min: 0, max: 0 },
  };
  const [t0, t1] = tangentialAxes[axis];
  bounds[t0] = {
    min: getAxisValue(highOrigin, t0),
    max: getAxisValue(highOrigin, t0) + fineExtent,
  };
  bounds[t1] = {
    min: getAxisValue(highOrigin, t1),
    max: getAxisValue(highOrigin, t1) + fineExtent,
  };
  return bounds;
}

function getAxisValue(vector: Vector3i, axis: AxisKey): number {
  switch (axis) {
    case "x":
      return vector.x;
    case "y":
      return vector.y;
    default:
      return vector.z;
  }
}

function collectSeamEdges(
  mesh: MeshData,
  axis: AxisKey,
  seamCoordinate: number,
  tangentialBounds: Bounds
): Map<string, number> {
  const edges = new Map<string, number>();
  const { vertices, indices } = mesh;
  const epsilon = 1e-3;
  for (let i = 0; i < indices.length; i += 3) {
    const a = getRenderablePosition(vertices[indices[i]]);
    const b = getRenderablePosition(vertices[indices[i + 1]]);
    const c = getRenderablePosition(vertices[indices[i + 2]]);
    processEdge(a, b, edges, axis, seamCoordinate, tangentialBounds, epsilon);
    processEdge(b, c, edges, axis, seamCoordinate, tangentialBounds, epsilon);
    processEdge(c, a, edges, axis, seamCoordinate, tangentialBounds, epsilon);
  }
  return edges;
}

function processEdge(
  start: Vector3f,
  end: Vector3f,
  edges: Map<string, number>,
  axis: AxisKey,
  seamCoordinate: number,
  tangentialBounds: Bounds,
  epsilon: number
): void {
  if (!isOnSeam(start, axis, seamCoordinate, epsilon)) {
    return;
  }
  if (!isOnSeam(end, axis, seamCoordinate, epsilon)) {
    return;
  }

  const [t0, t1] = tangentialAxes[axis];
  if (!withinBounds(start, t0, tangentialBounds, epsilon)) {
    return;
  }
  if (!withinBounds(start, t1, tangentialBounds, epsilon)) {
    return;
  }
  if (!withinBounds(end, t0, tangentialBounds, epsilon)) {
    return;
  }
  if (!withinBounds(end, t1, tangentialBounds, epsilon)) {
    return;
  }

  const key = edgeKey(start, end);
  edges.set(key, (edges.get(key) ?? 0) + 1);
}

function withinBounds(
  point: Vector3f,
  axis: AxisKey,
  bounds: Bounds,
  epsilon: number
): boolean {
  const value = getAxisComponent(point, axis);
  const { min, max } = bounds[axis];
  return value >= min - epsilon && value <= max + epsilon;
}

function isOnSeam(point: Vector3f, axis: AxisKey, seamCoordinate: number, epsilon: number): boolean {
  return Math.abs(getAxisComponent(point, axis) - seamCoordinate) <= epsilon;
}

function getAxisComponent(point: Vector3f, axis: AxisKey): number {
  switch (axis) {
    case "x":
      return point.x;
    case "y":
      return point.y;
    default:
      return point.z;
  }
}

const quantize = (value: number): number => Math.round(value * 256) / 256;

function edgeKey(a: Vector3f, b: Vector3f): string {
  const pa = vertexKey(a);
  const pb = vertexKey(b);
  return pa < pb ? `${pa}|${pb}` : `${pb}|${pa}`;
}

function vertexKey(point: Vector3f): string {
  return `${quantize(point.x)},${quantize(point.y)},${quantize(point.z)}`;
}
