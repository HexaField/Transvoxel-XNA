export class Vector3i {
  constructor(public readonly x: number, public readonly y: number, public readonly z: number) {}

  static readonly zero = new Vector3i(0, 0, 0);
  static readonly one = new Vector3i(1, 1, 1);
  static readonly unitX = new Vector3i(1, 0, 0);
  static readonly unitY = new Vector3i(0, 1, 0);
  static readonly unitZ = new Vector3i(0, 0, 1);

  static fromArray([x, y, z]: [number, number, number]): Vector3i {
    return new Vector3i(x, y, z);
  }

  add(other: Vector3i): Vector3i {
    return new Vector3i(this.x + other.x, this.y + other.y, this.z + other.z);
  }

  subtract(other: Vector3i): Vector3i {
    return new Vector3i(this.x - other.x, this.y - other.y, this.z - other.z);
  }

  multiplyScalar(scalar: number): Vector3i {
    return new Vector3i(this.x * scalar, this.y * scalar, this.z * scalar);
  }

  divideScalar(scalar: number): Vector3i {
    return new Vector3i(this.x / scalar, this.y / scalar, this.z / scalar);
  }

  multiplyComponents(other: Vector3i): Vector3i {
    return new Vector3i(this.x * other.x, this.y * other.y, this.z * other.z);
  }

  divideComponents(other: Vector3i): Vector3i {
    return new Vector3i(this.x / other.x, this.y / other.y, this.z / other.z);
  }

  component(index: number): number {
    switch (index) {
      case 0:
        return this.x;
      case 1:
        return this.y;
      case 2:
        return this.z;
      default:
        throw new RangeError(`Vector3i component ${index} is out of range.`);
    }
  }

  toArray(): [number, number, number] {
    return [this.x, this.y, this.z];
  }

  equals(other: Vector3i): boolean {
    return this.x === other.x && this.y === other.y && this.z === other.z;
  }

  toString(): string {
    return `(${this.x}, ${this.y}, ${this.z})`;
  }
}
