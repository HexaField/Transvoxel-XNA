import { BufferGeometry, Float32BufferAttribute } from "three";
import { MeshData } from "../src/surface-extractor/mesh-data";
import { getRenderablePosition } from "../src/surface-extractor/vertex";
import type { DensityFunction } from "../src/volume/volume-data";

export interface BuiltGeometry {
  geometry: BufferGeometry;
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
}

export const meshDataToGeometry = (meshData: MeshData): BuiltGeometry => {
  const vertexCount = meshData.vertices.length;
  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);

  meshData.vertices.forEach((vertex, index) => {
    const base = index * 3;
    const position = getRenderablePosition(vertex);
    positions[base] = position.x;
    positions[base + 1] = position.y;
    positions[base + 2] = position.z;

    normals[base] = vertex.normal.x;
    normals[base + 1] = vertex.normal.y;
    normals[base + 2] = vertex.normal.z;
  });

  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute(positions, 3));
  geometry.setAttribute("normal", new Float32BufferAttribute(normals, 3));
  geometry.setIndex(meshData.indices);

  const indices = new Uint32Array(meshData.indices);

  return { geometry, positions, normals, indices };
};

export const createSampleVolume = (blockCenter: number, blockExtent: number): DensityFunction =>
  (x, y, z) => {
    const nx = (x - blockCenter) / blockExtent;
    const ny = (y - blockCenter) / blockExtent;
    const nz = (z - blockCenter) / blockExtent;
    const sphere = nx * nx + ny * ny + nz * nz - 0.18;
    const folds = Math.sin(nx * 8.0) * 0.35 + Math.cos(ny * 6.0) * 0.35 + Math.sin(nz * 7.0) * 0.35;
    const density = sphere + 0.25 * folds;
    return Math.floor(density * 127);
  };
