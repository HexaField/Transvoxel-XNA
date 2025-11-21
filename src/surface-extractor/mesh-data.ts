import { TransvoxelVertex } from "./vertex";

export class MeshData {
  constructor(
    public readonly vertices: TransvoxelVertex[] = [],
    public readonly indices: number[] = []
  ) {}

  addVertex(vertex: TransvoxelVertex): number {
    this.vertices.push(vertex);
    return this.vertices.length - 1;
  }

  addTriangle(a: number, b: number, c: number): void {
    this.indices.push(a, b, c);
  }
}
