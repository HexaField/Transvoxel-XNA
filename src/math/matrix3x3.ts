import { createVector3f, setVector3f, type MutableVector3f, type Vector3f } from './vector3f'
import { createVector3i, setVector3i, type MutableVector3i, type Vector3i } from './vector3i'

export interface Matrix3x3 {
  m11: number
  m12: number
  m13: number
  m21: number
  m22: number
  m23: number
  m31: number
  m32: number
  m33: number
}

export type MutableMatrix3x3 = Matrix3x3
export type ReadonlyMatrix3x3 = Readonly<Matrix3x3>

export const createMatrix3x3 = (
  m11 = 1,
  m12 = 0,
  m13 = 0,
  m21 = 0,
  m22 = 1,
  m23 = 0,
  m31 = 0,
  m32 = 0,
  m33 = 1
): MutableMatrix3x3 => ({
  m11,
  m12,
  m13,
  m21,
  m22,
  m23,
  m31,
  m32,
  m33
})

export const setMatrix3x3 = (
  out: MutableMatrix3x3,
  m11: number,
  m12: number,
  m13: number,
  m21: number,
  m22: number,
  m23: number,
  m31: number,
  m32: number,
  m33: number
): MutableMatrix3x3 => {
  out.m11 = m11
  out.m12 = m12
  out.m13 = m13
  out.m21 = m21
  out.m22 = m22
  out.m23 = m23
  out.m31 = m31
  out.m32 = m32
  out.m33 = m33
  return out
}

export const matrix3x3FromColumns = (
  col1: Vector3f,
  col2: Vector3f,
  col3: Vector3f,
  out: MutableMatrix3x3 = createMatrix3x3()
): MutableMatrix3x3 => setMatrix3x3(out, col1.x, col2.x, col3.x, col1.y, col2.y, col3.y, col1.z, col2.z, col3.z)

export const multiplyMatrix3x3Vector3f = (
  matrix: Matrix3x3,
  vector: Vector3f,
  out: MutableVector3f = createVector3f()
): MutableVector3f =>
  setVector3f(
    out,
    matrix.m11 * vector.x + matrix.m12 * vector.y + matrix.m13 * vector.z,
    matrix.m21 * vector.x + matrix.m22 * vector.y + matrix.m23 * vector.z,
    matrix.m31 * vector.x + matrix.m32 * vector.y + matrix.m33 * vector.z
  )

export const multiplyMatrix3x3Vector3i = (
  matrix: Matrix3x3,
  vector: Vector3i,
  out: MutableVector3i = createVector3i()
): MutableVector3i =>
  setVector3i(
    out,
    toInt(matrix.m11 * vector.x + matrix.m12 * vector.y + matrix.m13 * vector.z),
    toInt(matrix.m21 * vector.x + matrix.m22 * vector.y + matrix.m23 * vector.z),
    toInt(matrix.m31 * vector.x + matrix.m32 * vector.y + matrix.m33 * vector.z)
  )

const toInt = (value: number): number => (value < 0 ? Math.ceil(value) : Math.floor(value))
