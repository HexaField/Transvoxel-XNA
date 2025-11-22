/// <reference types="@webgpu/types" />

import type {
  ChunkFieldRequest,
  ChunkFieldResponse,
  DensityGeneratorId,
} from "./chunk-field-types";

const WORKGROUP_SIZE = 4;
const CHUNK_SAMPLER_SHADER = /* wgsl */ `
  struct ChunkParams {
    minCoords : vec4<f32>,
  };

  @group(0) @binding(0) var<uniform> params : ChunkParams;
  @group(0) @binding(1) var<storage, read_write> outValues : array<i32>;

  fn smoothstep01(t : f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
  }

  fn lerp(a : f32, b : f32, t : f32) -> f32 {
    return a + (b - a) * t;
  }

  fn hash2d(x : f32, z : f32) -> f32 {
    let s = sin(x * 127.1 + z * 311.7) * 43758.5453;
    return s - floor(s);
  }

  fn valueNoise(x : f32, z : f32) -> f32 {
    let xi = floor(x);
    let zi = floor(z);
    let xf = x - xi;
    let zf = z - zi;

    let v00 = hash2d(xi, zi);
    let v10 = hash2d(xi + 1.0, zi);
    let v01 = hash2d(xi, zi + 1.0);
    let v11 = hash2d(xi + 1.0, zi + 1.0);

    let u = smoothstep01(xf);
    let v = smoothstep01(zf);

    let top = lerp(v00, v10, u);
    let bottom = lerp(v01, v11, u);
    return lerp(top, bottom, v);
  }

  fn fbm(x : f32, z : f32) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    for (var i = 0; i < 4; i = i + 1) {
      value = value + amplitude * valueNoise(x * frequency, z * frequency);
      amplitude = amplitude * 0.5;
      frequency = frequency * 2.0;
    }
    return value;
  }

  fn ridgeNoise(x : f32, z : f32) -> f32 {
    let n = valueNoise(x, z);
    return 1.0 - abs(2.0 * n - 1.0);
  }

  fn terrainHeight(x : f32, z : f32) -> f32 {
    let hills = fbm(x * 0.04, z * 0.04) * 28.0;
    let ridges = ridgeNoise(x * 0.02, z * 0.02) * 16.0;
    let dunes = sin(x * 0.01) * 3.0 + cos(z * 0.012) * 3.0;
    return 8.0 + hills + ridges + dunes;
  }

  @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
  fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let sampleSize = u32(params.minCoords.w);
    if (global_id.x >= sampleSize || global_id.y >= sampleSize || global_id.z >= sampleSize) {
      return;
    }

    let worldPos = vec3<f32>(global_id) + params.minCoords.xyz;
    let height = terrainHeight(worldPos.x, worldPos.z);
    let strata = sin((worldPos.x + worldPos.z) * 0.05) * 2.5;
    let densityValue = (height + strata - worldPos.y) * 6.0;
    let floored = floor(densityValue);
    let clamped = clamp(floored, -127.0, 127.0);

    let linearIndex = (global_id.z * sampleSize + global_id.y) * sampleSize + global_id.x;
    outValues[linearIndex] = i32(clamped);
  }
`;

const alignTo = (value: number, multiple: number): number => Math.ceil(value / multiple) * multiple;

export class WebGPUChunkGenerator {
  private readonly blockWidth: number;
  private device: GPUDevice | null = null;
  private pipeline: GPUComputePipeline | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;
  private paramsBuffer: GPUBuffer | null = null;
  private readyPromise: Promise<void> | null = null;

  constructor(blockWidth: number) {
    this.blockWidth = blockWidth;
  }

  static isSupported(): boolean {
    return typeof navigator !== "undefined" && typeof navigator.gpu !== "undefined";
  }

  supportsGenerator(generatorId: DensityGeneratorId): boolean {
    return generatorId === "terrain";
  }

  async generateChunkData(request: ChunkFieldRequest): Promise<ChunkFieldResponse> {
    await this.ensureReady();

    if (!this.supportsGenerator(request.generatorId)) {
      throw new Error(`Generator ${request.generatorId} is not supported by the WebGPU path.`);
    }

    const device = this.device!;
    const pipeline = this.pipeline!;
    const bindGroupLayout = this.bindGroupLayout!;
    const paramsBuffer = this.paramsBuffer!;

    const { minX, minY, minZ, sampleSize, totalSamples } = this.computeSampleRegion(request);

    const uniformData = new Float32Array([minX, minY, minZ, sampleSize]);
    device.queue.writeBuffer(paramsBuffer, 0, uniformData.buffer);

    const outputBytes = alignTo(totalSamples * 4, 4);
    const storageBuffer = device.createBuffer({
      size: outputBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readbackBuffer = device.createBuffer({
      size: outputBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: storageBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const groups = Math.ceil(sampleSize / WORKGROUP_SIZE);
    pass.dispatchWorkgroups(groups, groups, groups);
    pass.end();

    encoder.copyBufferToBuffer(storageBuffer, 0, readbackBuffer, 0, outputBytes);
    device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const mapped = readbackBuffer.getMappedRange();
    const copy = mapped.slice(0);
    readbackBuffer.unmap();
    storageBuffer.destroy();
    readbackBuffer.destroy();

    const int32View = new Int32Array(copy);
    const int8Result = new Int8Array(totalSamples);
    for (let i = 0; i < totalSamples; i++) {
      int8Result[i] = int32View[i];
    }

    return {
      key: request.key,
      minX,
      minY,
      minZ,
      size: sampleSize,
      buffer: int8Result.buffer,
      requestId: request.requestId,
      generatorId: request.generatorId,
      generatorToken: request.generatorToken,
    };
  }

  private computeSampleRegion(request: ChunkFieldRequest) {
    const range = this.blockWidth << request.lodIndex;
    const transitionReach = request.lodIndex === 0 ? 1 : (1 << (request.lodIndex - 1)) * 2 + 1;
    const padding = Math.max(1, transitionReach);
    const sampleSize = range + padding * 2 + 1;
    const minX = request.chunkX * range - padding;
      const chunkOriginY = request.originY ?? request.chunkY * range;
      const minY = chunkOriginY - padding;
    const minZ = request.chunkZ * range - padding;
    const totalSamples = sampleSize * sampleSize * sampleSize;
    return { minX, minY, minZ, sampleSize, totalSamples };
  }

  private ensureReady(): Promise<void> {
    if (this.readyPromise) {
      return this.readyPromise;
    }

    if (!WebGPUChunkGenerator.isSupported()) {
      return Promise.reject(new Error("WebGPU is not available in this environment."));
    }

    this.readyPromise = this.initialize();
    return this.readyPromise;
  }

  private async initialize(): Promise<void> {
    const adapter = await navigator.gpu!.requestAdapter();
    if (!adapter) {
      throw new Error("WebGPU adapter is unavailable.");
    }

    this.device = await adapter.requestDevice();
    const shaderModule = this.device.createShaderModule({ code: CHUNK_SAMPLER_SHADER });
    this.pipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });
    this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
    this.paramsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }
}
