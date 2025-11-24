import type { Vector3i } from "./vector3i";

export interface Vector3f {
  x: number;
  y: number;
  z: number;
}

export type MutableVector3f = Vector3f;
export type ReadonlyVector3f = Readonly<Vector3f>;

export const createVector3f = (x = 0, y = 0, z = 0): MutableVector3f => ({ x, y, z });

const freeze = (vector: MutableVector3f): ReadonlyVector3f => Object.freeze({ ...vector });

export const vector3fZero = freeze(createVector3f(0, 0, 0));
export const vector3fOne = freeze(createVector3f(1, 1, 1));
export const vector3fUnitX = freeze(createVector3f(1, 0, 0));
export const vector3fUnitY = freeze(createVector3f(0, 1, 0));
export const vector3fUnitZ = freeze(createVector3f(0, 0, 1));

export const cloneVector3f = (vector: Vector3f, out: MutableVector3f = createVector3f()): MutableVector3f => {
  out.x = vector.x;
  out.y = vector.y;
  out.z = vector.z;
  return out;
};

export const fromVector3i = ({ x, y, z }: Vector3i): MutableVector3f => createVector3f(x, y, z);

export const setVector3f = (
  out: MutableVector3f,
  x: number,
  y: number,
  z: number
): MutableVector3f => {
  out.x = x;
  out.y = y;
  out.z = z;
  return out;
};

export const addVector3f = (
  a: Vector3f,
  b: Vector3f,
  out: MutableVector3f = createVector3f()
): MutableVector3f => setVector3f(out, a.x + b.x, a.y + b.y, a.z + b.z);

export const subtractVector3f = (
  a: Vector3f,
  b: Vector3f,
  out: MutableVector3f = createVector3f()
): MutableVector3f => setVector3f(out, a.x - b.x, a.y - b.y, a.z - b.z);

export const multiplyVector3fScalar = (
  vector: Vector3f,
  scalar: number,
  out: MutableVector3f = createVector3f()
): MutableVector3f => setVector3f(out, vector.x * scalar, vector.y * scalar, vector.z * scalar);

export const divideVector3fScalar = (
  vector: Vector3f,
  scalar: number,
  out: MutableVector3f = createVector3f()
): MutableVector3f => setVector3f(out, vector.x / scalar, vector.y / scalar, vector.z / scalar);

export const multiplyVector3fComponents = (
  a: Vector3f,
  b: Vector3f,
  out: MutableVector3f = createVector3f()
): MutableVector3f => setVector3f(out, a.x * b.x, a.y * b.y, a.z * b.z);

export const combineVector3f = (
  a: Vector3f,
  scaleA: number,
  b: Vector3f,
  scaleB: number,
  out: MutableVector3f = createVector3f()
): MutableVector3f =>
  setVector3f(
    out,
    a.x * scaleA + b.x * scaleB,
    a.y * scaleA + b.y * scaleB,
    a.z * scaleA + b.z * scaleB
  );

export const lengthSquaredVector3f = (vector: Vector3f): number =>
  vector.x * vector.x + vector.y * vector.y + vector.z * vector.z;

export const lengthVector3f = (vector: Vector3f): number => Math.sqrt(lengthSquaredVector3f(vector));

export const normalizeVector3f = (
  vector: Vector3f,
  out: MutableVector3f = createVector3f()
): MutableVector3f => {
  const len = lengthVector3f(vector);
  if (len === 0) {
    return cloneVector3f(vector, out);
  }
  const inv = 1 / len;
  return setVector3f(out, vector.x * inv, vector.y * inv, vector.z * inv);
};

export const dotVector3f = (a: Vector3f, b: Vector3f): number => a.x * b.x + a.y * b.y + a.z * b.z;

export const crossVector3f = (
  a: Vector3f,
  b: Vector3f,
  out: MutableVector3f = createVector3f()
): MutableVector3f =>
  setVector3f(
    out,
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  );

export const lerpVector3f = (
  a: Vector3f,
  b: Vector3f,
  t: number,
  out: MutableVector3f = createVector3f()
): MutableVector3f => {
  if (t < 0 || t > 1) {
    throw new RangeError("Interpolation parameter t must be between 0 and 1.");
  }
  const oneMinusT = 1 - t;
  return setVector3f(out, a.x * oneMinusT + b.x * t, a.y * oneMinusT + b.y * t, a.z * oneMinusT + b.z * t);
};

export const equalsVector3f = (a: Vector3f, b: Vector3f): boolean =>
  a.x === b.x && a.y === b.y && a.z === b.z;

export const vector3fToString = (vector: Vector3f): string => `(${vector.x}, ${vector.y}, ${vector.z})`;
