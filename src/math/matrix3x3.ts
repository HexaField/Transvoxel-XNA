import { Vector3f } from "./vector3f";
import { Vector3i } from "./vector3i";

export class Matrix3x3 {
  constructor(
    public readonly m11: number,
    public readonly m12: number,
    public readonly m13: number,
    public readonly m21: number,
    public readonly m22: number,
    public readonly m23: number,
    public readonly m31: number,
    public readonly m32: number,
    public readonly m33: number
  ) {}

  static fromColumns(col1: Vector3f, col2: Vector3f, col3: Vector3f): Matrix3x3 {
    return new Matrix3x3(
      col1.x,
      col2.x,
      col3.x,
      col1.y,
      col2.y,
      col3.y,
      col1.z,
      col2.z,
      col3.z
    );
  }

  multiplyVector3f(v: Vector3f): Vector3f {
    return new Vector3f(
      this.m11 * v.x + this.m12 * v.y + this.m13 * v.z,
      this.m21 * v.x + this.m22 * v.y + this.m23 * v.z,
      this.m31 * v.x + this.m32 * v.y + this.m33 * v.z
    );
  }

  multiplyVector3i(v: Vector3i): Vector3i {
    const x = this.m11 * v.x + this.m12 * v.y + this.m13 * v.z;
    const y = this.m21 * v.x + this.m22 * v.y + this.m23 * v.z;
    const z = this.m31 * v.x + this.m32 * v.y + this.m33 * v.z;
    return new Vector3i(Matrix3x3.toInt(x), Matrix3x3.toInt(y), Matrix3x3.toInt(z));
  }

  private static toInt(value: number): number {
    return value < 0 ? Math.ceil(value) : Math.floor(value);
  }
}
