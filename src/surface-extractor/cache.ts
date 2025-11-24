import { Vector3i } from '../math/vector3i'

export interface RegularCacheCellView {
  caseIndex: number
  readonly verts: Int32Array
  readonly dirs: Int8Array
  reset(): void
}

class RegularCacheCell implements RegularCacheCellView {
  private readonly caseIndexArray: Int16Array
  private readonly caseIndexIndex: number
  private readonly vertsView: Int32Array
  private readonly dirsView: Int8Array

  constructor(caseIndexArray: Int16Array, vertsArray: Int32Array, dirsArray: Int8Array, cellIndex: number) {
    this.caseIndexArray = caseIndexArray
    this.caseIndexIndex = cellIndex
    const vertsStart = cellIndex * 4
    this.vertsView = vertsArray.subarray(vertsStart, vertsStart + 4)
    this.dirsView = dirsArray.subarray(vertsStart, vertsStart + 4)
  }

  get caseIndex(): number {
    return this.caseIndexArray[this.caseIndexIndex]
  }

  set caseIndex(value: number) {
    this.caseIndexArray[this.caseIndexIndex] = value
  }

  get verts(): Int32Array {
    return this.vertsView
  }

  get dirs(): Int8Array {
    return this.dirsView
  }

  reset(): void {
    this.caseIndex = 0
    this.vertsView.fill(-1)
    this.dirsView.fill(-1)
  }
}

export class RegularCache {
  private readonly caseIndex: Int16Array
  private readonly verts: Int32Array
  private readonly dirs: Int8Array
  private readonly cells: RegularCacheCell[][][]

  constructor(private readonly blockWidth: number) {
    const cellCount = blockWidth * blockWidth * blockWidth
    this.caseIndex = new Int16Array(cellCount)
    this.verts = new Int32Array(cellCount * 4).fill(-1)
    this.dirs = new Int8Array(cellCount * 4).fill(-1)
    this.cells = Array.from({ length: blockWidth }, (_, x) =>
      Array.from({ length: blockWidth }, (_, y) =>
        Array.from(
          { length: blockWidth },
          (_, z) => new RegularCacheCell(this.caseIndex, this.verts, this.dirs, this.indexOf(x, y, z))
        )
      )
    )
  }

  getCell(x: number, y: number, z: number): RegularCacheCellView {
    if (x < 0 || y < 0 || z < 0 || x >= this.blockWidth || y >= this.blockWidth || z >= this.blockWidth) {
      throw new RangeError(`RegularCache coordinates (${x}, ${y}, ${z}) out of bounds.`)
    }
    return this.cells[x][y][z]
  }

  getCellByVector(v: Vector3i): RegularCacheCellView {
    return this.getCell(v.x, v.y, v.z)
  }

  reset(): void {
    this.caseIndex.fill(0)
    this.verts.fill(-1)
    this.dirs.fill(-1)
  }

  private indexOf(x: number, y: number, z: number): number {
    return x + this.blockWidth * (y + this.blockWidth * z)
  }
}

export interface TransitionCacheCellView {
  caseIndex: number
  readonly verts: Int32Array
  reset(): void
}

class TransitionCacheCell implements TransitionCacheCellView {
  private readonly caseIndexArray: Int16Array
  private readonly caseIndexIndex: number
  private readonly vertsView: Int32Array

  constructor(caseIndexArray: Int16Array, vertsArray: Int32Array, cellIndex: number) {
    this.caseIndexArray = caseIndexArray
    this.caseIndexIndex = cellIndex
    const start = cellIndex * 12
    this.vertsView = vertsArray.subarray(start, start + 12)
  }

  get caseIndex(): number {
    return this.caseIndexArray[this.caseIndexIndex]
  }

  set caseIndex(value: number) {
    this.caseIndexArray[this.caseIndexIndex] = value
  }

  get verts(): Int32Array {
    return this.vertsView
  }

  reset(): void {
    this.caseIndex = 0
    this.vertsView.fill(-1)
  }
}

export class TransitionCache {
  private readonly caseIndex: Int16Array
  private readonly verts: Int32Array
  private readonly cells: TransitionCacheCell[][]

  constructor(private readonly blockWidth: number) {
    const cellCount = blockWidth * blockWidth
    this.caseIndex = new Int16Array(cellCount)
    this.verts = new Int32Array(cellCount * 12).fill(-1)
    this.cells = Array.from({ length: blockWidth }, (_, x) =>
      Array.from(
        { length: blockWidth },
        (_, y) => new TransitionCacheCell(this.caseIndex, this.verts, this.indexOf(x, y))
      )
    )
  }

  getCell(x: number, y: number): TransitionCacheCellView {
    if (x < 0 || y < 0 || x >= this.blockWidth || y >= this.blockWidth) {
      throw new RangeError(`TransitionCache coordinates (${x}, ${y}) out of bounds.`)
    }
    return this.cells[x][y]
  }

  reset(): void {
    this.caseIndex.fill(0)
    this.verts.fill(-1)
  }

  private indexOf(x: number, y: number): number {
    return x + this.blockWidth * y
  }
}
