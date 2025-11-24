export interface Vector3i {
  x: number
  y: number
  z: number
}

export type MutableVector3i = Vector3i
export type ReadonlyVector3i = Readonly<Vector3i>

export const createVector3i = (x = 0, y = 0, z = 0): MutableVector3i => ({ x, y, z })

const freeze = (vector: MutableVector3i): ReadonlyVector3i => Object.freeze({ ...vector })

export const vector3iZero = freeze(createVector3i(0, 0, 0))
export const vector3iOne = freeze(createVector3i(1, 1, 1))
export const vector3iUnitX = freeze(createVector3i(1, 0, 0))
export const vector3iUnitY = freeze(createVector3i(0, 1, 0))
export const vector3iUnitZ = freeze(createVector3i(0, 0, 1))

export const vector3iFromArray = ([x, y, z]: [number, number, number]): MutableVector3i => createVector3i(x, y, z)

export const setVector3i = (out: MutableVector3i, x: number, y: number, z: number): MutableVector3i => {
  out.x = x
  out.y = y
  out.z = z
  return out
}

export const addVector3i = (a: Vector3i, b: Vector3i, out: MutableVector3i = createVector3i()): MutableVector3i =>
  setVector3i(out, a.x + b.x, a.y + b.y, a.z + b.z)

export const subtractVector3i = (a: Vector3i, b: Vector3i, out: MutableVector3i = createVector3i()): MutableVector3i =>
  setVector3i(out, a.x - b.x, a.y - b.y, a.z - b.z)

export const multiplyVector3iScalar = (
  vector: Vector3i,
  scalar: number,
  out: MutableVector3i = createVector3i()
): MutableVector3i => setVector3i(out, vector.x * scalar, vector.y * scalar, vector.z * scalar)

export const divideVector3iScalar = (
  vector: Vector3i,
  scalar: number,
  out: MutableVector3i = createVector3i()
): MutableVector3i => setVector3i(out, vector.x / scalar, vector.y / scalar, vector.z / scalar)

export const multiplyVector3iComponents = (
  a: Vector3i,
  b: Vector3i,
  out: MutableVector3i = createVector3i()
): MutableVector3i => setVector3i(out, a.x * b.x, a.y * b.y, a.z * b.z)

export const divideVector3iComponents = (
  a: Vector3i,
  b: Vector3i,
  out: MutableVector3i = createVector3i()
): MutableVector3i => setVector3i(out, a.x / b.x, a.y / b.y, a.z / b.z)

export const vector3iComponent = (vector: Vector3i, index: number): number => {
  switch (index) {
    case 0:
      return vector.x
    case 1:
      return vector.y
    case 2:
      return vector.z
    default:
      throw new RangeError(`Vector3i component ${index} is out of range.`)
  }
}

export const vector3iToArray = (vector: Vector3i): [number, number, number] => [vector.x, vector.y, vector.z]

export const equalsVector3i = (a: Vector3i, b: Vector3i): boolean => a.x === b.x && a.y === b.y && a.z === b.z

export const vector3iToString = (vector: Vector3i): string => `(${vector.x}, ${vector.y}, ${vector.z})`
