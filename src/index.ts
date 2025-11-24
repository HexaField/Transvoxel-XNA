import { RegularCache, TransitionCache } from './surface-extractor/cache'
import { MeshData } from './surface-extractor/mesh-data'
import {
  ExtractBlockOptions,
  TransitionFace,
  TransvoxelMesher,
  TransvoxelMesherOptions
} from './surface-extractor/transvoxel-extractor'
import { DensityFunction } from './volume/volume-data'

export { Tables } from './lengyel/tables'
export * from './math/matrix3x3'
export * from './math/vector3f'
export * from './math/vector3i'
export { RegularCache, TransitionCache } from './surface-extractor/cache'
export { MeshData } from './surface-extractor/mesh-data'
export { TransvoxelMesher, type TransvoxelMesherOptions } from './surface-extractor/transvoxel-extractor'
export type { ExtractBlockOptions, TransitionFace } from './surface-extractor/transvoxel-extractor'
export type { TransvoxelVertex } from './surface-extractor/vertex'
export type { DensityFunction } from './volume/volume-data'

export interface FunctionalMesher {
  readonly regularCache: RegularCache
  readonly transitionCache: TransitionCache
  extractBlock(samples: DensityFunction, options: ExtractBlockOptions): MeshData
  extractTransitionFaces(samples: DensityFunction, options: ExtractBlockOptions & { faces: TransitionFace[] }): MeshData
}

export type CreateMesherOptions = TransvoxelMesherOptions

export const createTransvoxelMesher = (options: CreateMesherOptions = {}): FunctionalMesher => {
  const mesher = new TransvoxelMesher(options)
  const regularCache = mesher.getRegularCache()
  const transitionCache = mesher.getTransitionCache()

  return {
    regularCache,
    transitionCache,
    extractBlock: (samples, extractOptions) => {
      if (!extractOptions) {
        throw new Error('extractBlock options are required.')
      }
      return mesher.extractBlock(samples, extractOptions)
    },
    extractTransitionFaces: (samples, options) => mesher.extractTransitionFaces(samples, options)
  }
}
