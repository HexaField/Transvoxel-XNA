export class Vector3f {
  constructor(public readonly x: number, public readonly y: number, public readonly z: number) {}

  static readonly zero = new Vector3f(0, 0, 0);
  static readonly one = new Vector3f(1, 1, 1);
  static readonly unitX = new Vector3f(1, 0, 0);
  static readonly unitY = new Vector3f(0, 1, 0);
  static readonly unitZ = new Vector3f(0, 0, 1);

  static fromVector3i({ x, y, z }: { x: number; y: number; z: number }): Vector3f {
    return new Vector3f(x, y, z);
  }

  add(other: Vector3f): Vector3f {
    return new Vector3f(this.x + other.x, this.y + other.y, this.z + other.z);
  }

  subtract(other: Vector3f): Vector3f {
    return new Vector3f(this.x - other.x, this.y - other.y, this.z - other.z);
  }

  multiplyScalar(scalar: number): Vector3f {
    return new Vector3f(this.x * scalar, this.y * scalar, this.z * scalar);
  }

  divideScalar(scalar: number): Vector3f {
    return new Vector3f(this.x / scalar, this.y / scalar, this.z / scalar);
  }

  multiplyComponents(other: Vector3f): Vector3f {
    return new Vector3f(this.x * other.x, this.y * other.y, this.z * other.z);
  }

  lengthSquared(): number {
    return this.x * this.x + this.y * this.y + this.z * this.z;
  }

  length(): number {
    return Math.sqrt(this.lengthSquared());
  }

  normalize(): Vector3f {
    const len = this.length();
    if (len === 0) {
      return this;
    }
    const inv = 1 / len;
    return new Vector3f(this.x * inv, this.y * inv, this.z * inv);
  }

  dot(other: Vector3f): number {
    return this.x * other.x + this.y * other.y + this.z * other.z;
  }

  cross(other: Vector3f): Vector3f {
    return new Vector3f(
      this.y * other.z - this.z * other.y,
      this.z * other.x - this.x * other.z,
      this.x * other.y - this.y * other.x
    );
  }

  lerp(other: Vector3f, t: number): Vector3f {
    if (t < 0 || t > 1) {
      throw new RangeError("Interpolation parameter t must be between 0 and 1.");
    }
    const oneMinusT = 1 - t;
    return new Vector3f(
      this.x * oneMinusT + other.x * t,
      this.y * oneMinusT + other.y * t,
      this.z * oneMinusT + other.z * t
    );
  }

  equals(other: Vector3f): boolean {
    return this.x === other.x && this.y === other.y && this.z === other.z;
  }

  toString(): string {
    return `(${this.x}, ${this.y}, ${this.z})`;
  }
}
