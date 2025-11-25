import { getRenderablePosition, type TransvoxelVertex } from './vertex'

export interface MeshData {
  positions: Float32Array
  normals: Float32Array
  indices: Uint32Array
}

export const buildMeshData = (vertices: TransvoxelVertex[], indices: number[]): MeshData => {
  const positions = new Float32Array(vertices.length * 3)
  const normals = new Float32Array(vertices.length * 3)

  vertices.forEach((vertex, index) => {
    const base = index * 3
    const position = getRenderablePosition(vertex)
    positions[base] = position.x
    positions[base + 1] = position.y
    positions[base + 2] = position.z

    normals[base] = vertex.normal.x
    normals[base + 1] = vertex.normal.y
    normals[base + 2] = vertex.normal.z
  })

  return {
    positions,
    normals,
    indices: new Uint32Array(indices)
  }
}
