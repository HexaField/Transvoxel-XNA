export type ChunkFieldRequest = {
  key: string;
  lodIndex: number;
  chunkX: number;
  chunkZ: number;
  originY: number;
  requestId: number;
};

export type ChunkFieldResponse = {
  key: string;
  minX: number;
  minY: number;
  minZ: number;
  size: number;
  buffer: ArrayBuffer;
  requestId: number;
};
