export interface VolumeData {
  sample(x: number, y: number, z: number): number;
}

export type DensityFunction = (x: number, y: number, z: number) => number;

export class FunctionalVolume implements VolumeData {
  constructor(private readonly fn: DensityFunction) {}

  sample(x: number, y: number, z: number): number {
    return this.fn(x, y, z);
  }
}
