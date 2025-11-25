import { BufferAttribute, BufferGeometry, Float32BufferAttribute } from 'three'
import { MeshData } from '../src/surface-extractor/mesh-data'
import type { DensityFunction } from '../src/volume/volume-data'

export interface BuiltGeometry {
  geometry: BufferGeometry
  positions: Float32Array
  normals: Float32Array
  indices: Uint32Array
}

export const meshDataToGeometry = (meshData: MeshData): BuiltGeometry => {
  const { positions, normals, indices } = meshData
  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3))
  geometry.setAttribute('normal', new Float32BufferAttribute(normals, 3))
  geometry.setIndex(new BufferAttribute(indices, 1))

  return { geometry, positions, normals, indices }
}

export const createSampleVolume =
  (blockCenter: number, blockExtent: number): DensityFunction =>
  (x, y, z) => {
    const nx = (x - blockCenter) / blockExtent
    const ny = (y - blockCenter) / blockExtent
    const nz = (z - blockCenter) / blockExtent
    const sphere = nx * nx + ny * ny + nz * nz - 0.18
    const folds = Math.sin(nx * 8.0) * 0.35 + Math.cos(ny * 6.0) * 0.35 + Math.sin(nz * 7.0) * 0.35
    const density = sphere + 0.25 * folds
    return Math.floor(density * 127)
  }
