import { type Vector3f, createVector3f } from '../math/vector3f'

export interface TransvoxelVertex {
  primary: Vector3f
  secondary: Vector3f
  normal: Vector3f
  near: number
}

export const unusedVertexPosition: Vector3f = Object.freeze(createVector3f(1000, 1000, 1000)) as Vector3f

export const getRenderablePosition = (vertex: TransvoxelVertex): Vector3f =>
  vertex.primary === unusedVertexPosition ? vertex.secondary : vertex.primary
