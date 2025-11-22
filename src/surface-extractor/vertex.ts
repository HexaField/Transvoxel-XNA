import { Vector3f } from "../math/vector3f";

export interface TransvoxelVertex {
  primary: Vector3f;
  secondary: Vector3f;
  normal: Vector3f;
  near: number;
}

export const unusedVertexPosition = new Vector3f(1000, 1000, 1000);

export const getRenderablePosition = (vertex: TransvoxelVertex): Vector3f =>
  vertex.secondary === unusedVertexPosition ? vertex.primary : vertex.secondary;
