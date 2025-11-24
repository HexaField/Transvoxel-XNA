export type DensityGeneratorId = 'terrain' | 'plateaus' | 'spheres'

export type ChunkFieldRequest = {
  key: string
  lodIndex: number
  chunkX: number
  chunkY: number
  chunkZ: number
  originY: number
  requestId: number
  generatorId: DensityGeneratorId
  generatorToken: number
}

export type ChunkFieldResponse = {
  key: string
  minX: number
  minY: number
  minZ: number
  size: number
  buffer: ArrayBuffer
  requestId: number
  generatorId: DensityGeneratorId
  generatorToken: number
}
