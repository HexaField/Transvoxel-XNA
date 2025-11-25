export type DensityFunction = (x: number, y: number, z: number) => number

export interface ChunkDensityField {
  /** Inclusive minimum world-space X sample coordinate represented in the field. */
  minX: number
  /** Inclusive minimum world-space Y sample coordinate represented in the field. */
  minY: number
  /** Inclusive minimum world-space Z sample coordinate represented in the field. */
  minZ: number
  /** Number of samples along a single axis (cube assumed). */
  size: number
  /** Signed density samples stored in X-fastest order. */
  data: Int8Array
}

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value))

export const createChunkFieldSampler = (field: ChunkDensityField): DensityFunction => {
  const { data, size, minX, minY, minZ } = field
  if (size <= 0) {
    throw new RangeError('Chunk field size must be positive.')
  }

  const slice = size
  const area = size * size

  return (x: number, y: number, z: number): number => {
    const ix = clamp(x - minX, 0, size - 1)
    const iy = clamp(y - minY, 0, size - 1)
    const iz = clamp(z - minZ, 0, size - 1)
    return data[iz * area + iy * slice + ix]
  }
}
