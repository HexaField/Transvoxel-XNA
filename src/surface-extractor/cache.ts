import { Vector3i } from "../math/vector3i";

export class RegularCacheCell {
  caseIndex = 0;
  verts = new Int32Array(4).fill(-1);

  reset(): void {
    this.caseIndex = 0;
    this.verts.fill(-1);
  }
}

export class RegularCache {
  private readonly cells: RegularCacheCell[][][];

  constructor(private readonly blockWidth: number) {
    this.cells = Array.from({ length: blockWidth }, () =>
      Array.from({ length: blockWidth }, () =>
        Array.from({ length: blockWidth }, () => new RegularCacheCell())
      )
    );
  }

  getCell(x: number, y: number, z: number): RegularCacheCell {
    if (x < 0 || y < 0 || z < 0 || x >= this.blockWidth || y >= this.blockWidth || z >= this.blockWidth) {
      throw new RangeError(`RegularCache coordinates (${x}, ${y}, ${z}) out of bounds.`);
    }
    return this.cells[x][y][z];
  }

  getCellByVector(v: Vector3i): RegularCacheCell {
    return this.getCell(v.x, v.y, v.z);
  }

  reset(): void {
    for (let x = 0; x < this.blockWidth; x++) {
      for (let y = 0; y < this.blockWidth; y++) {
        for (let z = 0; z < this.blockWidth; z++) {
          this.cells[x][y][z].reset();
        }
      }
    }
  }
}

export class TransitionCacheCell {
  caseIndex = 0;
  verts = new Int32Array(12).fill(-1);

  reset(): void {
    this.caseIndex = 0;
    this.verts.fill(-1);
  }
}

export class TransitionCache {
  private readonly cells: TransitionCacheCell[][];

  constructor(private readonly blockWidth: number) {
    this.cells = Array.from({ length: blockWidth }, () =>
      Array.from({ length: blockWidth }, () => new TransitionCacheCell())
    );
  }

  getCell(x: number, y: number): TransitionCacheCell {
    if (x < 0 || y < 0 || x >= this.blockWidth || y >= this.blockWidth) {
      throw new RangeError(`TransitionCache coordinates (${x}, ${y}) out of bounds.`);
    }
    return this.cells[x][y];
  }

  reset(): void {
    for (let x = 0; x < this.blockWidth; x++) {
      for (let y = 0; y < this.blockWidth; y++) {
        this.cells[x][y].reset();
      }
    }
  }
}
