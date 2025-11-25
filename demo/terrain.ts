import type { Vector3 } from 'three'
import {
  AmbientLight,
  AxesHelper,
  Color,
  DirectionalLight,
  FogExp2,
  Group,
  InstancedMesh,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Object3D,
  PerspectiveCamera,
  Scene,
  SphereGeometry,
  WebGLRenderer
} from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { createVector3i } from '../src/math/vector3i'
import { TransvoxelExtractor, TransvoxelMesher } from '../src/surface-extractor/transvoxel-extractor'
import { createChunkFieldSampler, type DensityFunction } from '../src/volume/volume-data'
import { type ChunkFieldRequest, type ChunkFieldResponse, type DensityGeneratorId } from './chunk-field-types'
import { meshDataToGeometry } from './mesh-utils'
import {
  OctreeChunkManager,
  type ChunkDescriptor,
  type ChunkPlan,
  type ChunkRequest,
  type LodLevel
} from './octreeChunks'
import { WebGPUChunkGenerator } from './webgpu-chunk-generator'

const mount = document.querySelector<HTMLDivElement>('#app')
if (!mount) {
  throw new Error('Missing #app mount point.')
}

const statsLabel = document.querySelector<HTMLParagraphElement>('#chunk-stats')
const generatorStatusLabel = document.querySelector<HTMLSpanElement>('#generator-status')
const generatorControlsRoot = document.querySelector<HTMLDivElement>('#generator-controls')
const generatorButtons = new Map<DensityGeneratorId, HTMLButtonElement>()
const densityToggle = document.querySelector<HTMLInputElement>('#toggle-density')

const scene = new Scene()
scene.background = new Color('#03050c')
scene.fog = new FogExp2('#03050c', 0.0012)

const camera = new PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1500)
camera.position.set(60, 34, 60)

const renderer = new WebGLRenderer({ antialias: true })
renderer.setSize(window.innerWidth, window.innerHeight)
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
mount.appendChild(renderer.domElement)

const controls = new OrbitControls(camera, renderer.domElement)
controls.enableDamping = true
controls.target.set(0, 18, 0)
controls.update()

scene.add(new AmbientLight(0xbfd5ff, 0.45))

const sunLight = new DirectionalLight(0xfff2d6, 1.2)
sunLight.position.set(120, 160, 40)
sunLight.castShadow = true
scene.add(sunLight)

const rimLight = new DirectionalLight(0x5bc0ff, 0.3)
rimLight.position.set(-80, 60, -100)
scene.add(rimLight)

const densityGroup = new Group()
densityGroup.visible = false
scene.add(densityGroup)

scene.add(new AxesHelper(10))

const BLOCK_WIDTH = TransvoxelExtractor.BlockWidth
const CELL_SIZE = 1
const VERTEX_DENSITY_RATIO = 1.5
const DENSITY_SAMPLE_STRIDE_BASE = 2

const densitySphereGeometry = new SphereGeometry(0.35, 6, 6)
const densitySphereMaterial = new MeshBasicMaterial({
  color: new Color('#ffffff'),
  transparent: true,
  opacity: 0.7,
  depthWrite: false,
  vertexColors: false,
  toneMapped: false
})
const densityColorPositive = new Color('#7ad9ff')
const densityColorNegative = new Color('#ff8c8c')
const densityColorNeutral = new Color('#7b7b7b')
const densityColorScratch = new Color()
const densityTransformScratch = new Object3D()

const chunkWorldSize = (lodIndex: number): number => CELL_SIZE * (BLOCK_WIDTH << lodIndex)
const chunkDiagonalRadius = (lodIndex: number): number => (chunkWorldSize(lodIndex) * Math.sqrt(3)) / 2

const deriveLodLevels = (): LodLevel[] => {
  const entries = [
    { lodIndex: 0, color: 0xff0000 },
    { lodIndex: 1, color: 0x00ff00 },
    { lodIndex: 2, color: 0x6666ff }
  ]
  return entries.map((entry) => ({
    ...entry,
    maxDistance: chunkWorldSize(entry.lodIndex) * VERTEX_DENSITY_RATIO
  }))
}

const LOD_LEVELS: LodLevel[] = deriveLodLevels()
const MAX_LOD_INDEX = LOD_LEVELS[LOD_LEVELS.length - 1].lodIndex
const LOD_LOOKUP = new Map(LOD_LEVELS.map((level) => [level.lodIndex, level]))
const chunkKey = (lodIndex: number, chunkX: number, chunkY: number, chunkZ: number): string =>
  `${lodIndex}:${chunkX}:${chunkY}:${chunkZ}`

interface DensityGenerator {
  id: DensityGeneratorId
  label: string
  supportsGpu: boolean
}

interface ScheduledChunkRequest extends ChunkRequest {
  generatorId: DensityGeneratorId
  generatorToken: number
}

const lerp01 = (a: number, b: number, t: number): number => a + (b - a) * t
const smoothstep01 = (t: number): number => t * t * (3 - 2 * t)

const hash2dNoise = (x: number, z: number): number => {
  const s = Math.sin(x * 127.1 + z * 311.7) * 43758.5453
  return s - Math.floor(s)
}

const valueNoise2d = (x: number, z: number): number => {
  const xi = Math.floor(x)
  const zi = Math.floor(z)
  const xf = x - xi
  const zf = z - zi

  const v00 = hash2dNoise(xi, zi)
  const v10 = hash2dNoise(xi + 1, zi)
  const v01 = hash2dNoise(xi, zi + 1)
  const v11 = hash2dNoise(xi + 1, zi + 1)

  const u = smoothstep01(xf)
  const v = smoothstep01(zf)

  const top = lerp01(v00, v10, u)
  const bottom = lerp01(v01, v11, u)
  return lerp01(top, bottom, v)
}

const fbm2d = (x: number, z: number): number => {
  let value = 0
  let amplitude = 1
  let frequency = 1
  for (let i = 0; i < 4; i++) {
    value += amplitude * valueNoise2d(x * frequency, z * frequency)
    amplitude *= 0.5
    frequency *= 2.0
  }
  return value
}

const ridgeNoise2d = (x: number, z: number): number => {
  const n = valueNoise2d(x, z)
  return 1 - Math.abs(2 * n - 1)
}

const terrainHeightEstimate = (x: number, z: number): number => {
  const hills = fbm2d(x * 0.04, z * 0.04) * 28
  const ridges = ridgeNoise2d(x * 0.02, z * 0.02) * 16
  const dunes = Math.sin(x * 0.01) * 3 + Math.cos(z * 0.012) * 3
  return 8 + hills + ridges + dunes
}

const PLATEAU_BASE_HEIGHT = 20
const PLATEAU_SIN_FREQ_X = 0.025
const PLATEAU_SIN_FREQ_Z = 0.03
const PLATEAU_SIN_AMPLITUDE = 6
const plateauHeightEstimate = (x: number, z: number): number =>
  PLATEAU_BASE_HEIGHT +
  Math.sin(x * PLATEAU_SIN_FREQ_X) * PLATEAU_SIN_AMPLITUDE +
  Math.cos(z * PLATEAU_SIN_FREQ_Z) * PLATEAU_SIN_AMPLITUDE

const FLOATING_SPHERE_SPACING = 42
const FLOATING_SPHERE_RADIUS = 14
const FLOATING_SPHERE_BASE_HEIGHT = 32
const FLOATING_SPHERE_SWAY_AMPLITUDE = 6
const FLOATING_SPHERE_SWAY_FREQUENCY = 0.02

const floatingSphereCenterY = (x: number, z: number): number =>
  FLOATING_SPHERE_BASE_HEIGHT + Math.sin((x + z) * FLOATING_SPHERE_SWAY_FREQUENCY) * FLOATING_SPHERE_SWAY_AMPLITUDE

const floatingSphereSurfaceEstimate = (x: number, z: number): number =>
  floatingSphereCenterY(x, z) + FLOATING_SPHERE_RADIUS

const DENSITY_GENERATORS: DensityGenerator[] = [
  {
    id: 'terrain',
    label: 'Procedural Terrain',
    supportsGpu: true
  },
  {
    id: 'plateaus',
    label: 'Rippled Plateau',
    supportsGpu: false
  },
  {
    id: 'spheres',
    label: 'Floating Spheres',
    supportsGpu: false
  }
]

const generatorLookup = new Map(DENSITY_GENERATORS.map((generator) => [generator.id, generator]))

let activeGenerator: DensityGenerator = DENSITY_GENERATORS[0]
let generatorToken = 0
let densityVisualizationEnabled = false

const createChunkScheduler = (): OctreeChunkManager =>
  new OctreeChunkManager({
    blockWidth: BLOCK_WIDTH,
    cellSize: CELL_SIZE,
    lodLevels: LOD_LEVELS
  })

type ChunkWorkerRequest = ChunkFieldRequest
type ChunkWorkerResponse = ChunkFieldResponse

interface ChunkRecord {
  mesh?: Mesh
  material?: MeshStandardMaterial
  key: string
  lodIndex: number
  chunkX: number
  chunkY: number
  chunkZ: number
  color: number
  generatorId: DensityGeneratorId
  descriptor: ChunkDescriptor
  densityMesh?: InstancedMesh
}

interface PendingChunkRequest {
  descriptor: ChunkDescriptor
  requestId: number
  generatorId: DensityGeneratorId
  generatorToken: number
}

const mesher = new TransvoxelMesher()
const chunkFieldGenerator = WebGPUChunkGenerator.isSupported() ? new WebGPUChunkGenerator(BLOCK_WIDTH) : null
let chunkWorker: Worker | null = chunkFieldGenerator ? null : createChunkWorker()
let suppressGpuPath = false
const activeChunks = new Map<string, ChunkRecord>()
const stagedChunks = new Map<string, ChunkRecord>()
const pendingBuilds = new Map<string, PendingChunkRequest>()
const buildQueue: ScheduledChunkRequest[] = []
let inflightBuilds = 0
const MAX_INFLIGHT_BUILDS = 1
const currentCameraPosition = { x: 0, y: 0, z: 0 }
let chunkScheduler = createChunkScheduler()

const attachWorkerHandler = (worker: Worker): void => {
  worker.onmessage = (event: MessageEvent<ChunkWorkerResponse>) => {
    handleWorkerMessage(event.data as ChunkWorkerResponse)
  }
}

const ensureChunkWorker = (): Worker => {
  if (!chunkWorker) {
    chunkWorker = createChunkWorker()
    attachWorkerHandler(chunkWorker)
  }
  return chunkWorker
}

if (chunkWorker) {
  attachWorkerHandler(chunkWorker)
}

const setupGeneratorControls = (): void => {
  if (!generatorControlsRoot) {
    updateGeneratorUI()
    return
  }

  generatorControlsRoot.replaceChildren()
  generatorButtons.clear()
  DENSITY_GENERATORS.forEach((generator) => {
    const button = document.createElement('button')
    button.type = 'button'
    button.className = 'generator-button'
    button.textContent = generator.label
    button.addEventListener('click', () => setActiveGenerator(generator.id))
    generatorControlsRoot.appendChild(button)
    generatorButtons.set(generator.id, button)
  })

  updateGeneratorUI()
}

const updateGeneratorUI = (): void => {
  generatorButtons.forEach((button, id) => {
    button.classList.toggle('is-active', id === activeGenerator.id)
  })
  if (generatorStatusLabel) {
    generatorStatusLabel.textContent = activeGenerator.label
  }
}

const resetTerrainState = (): void => {
  activeChunks.forEach((record) => releaseChunkResources(record))
  activeChunks.clear()
  stagedChunks.forEach((record) => releaseChunkResources(record))
  stagedChunks.clear()
  pendingBuilds.clear()
  buildQueue.length = 0
  inflightBuilds = 0
  densityGroup.clear()
  updateTerrainStats()
}

const setActiveGenerator = (generatorId: DensityGeneratorId): void => {
  const next = generatorLookup.get(generatorId)
  if (!next || next.id === activeGenerator.id) {
    return
  }
  activeGenerator = next
  generatorToken++
  chunkScheduler = createChunkScheduler()
  resetTerrainState()
  updateGeneratorUI()
  updateTerrain(controls.target)
}

function updateTerrain(cameraPosition: Vector3): void {
  currentCameraPosition.x = cameraPosition.x
  currentCameraPosition.y = cameraPosition.y
  currentCameraPosition.z = cameraPosition.z
  const plan = chunkScheduler.update(currentCameraPosition)
  applyChunkPlan(plan)
}

function applyChunkPlan(plan: ChunkPlan): void {
  for (const key of plan.cancels) {
    removeFromBuildQueue(key)
    pendingBuilds.delete(key)
    discardStagedChunk(key)
  }

  for (const key of plan.releases) {
    promoteChildrenForParent(key)
    discardStagedChunk(key)
    disposeChunk(key)
  }

  for (const request of plan.requests) {
    enqueueBuildRequest(request)
  }

  reorderBuildQueue()
  dispatchBuilds()
  updateTerrainStats()
}

function enqueueBuildRequest(request: ChunkRequest): void {
  const key = request.descriptor.key
  if (pendingBuilds.has(key)) {
    return
  }
  removeFromBuildQueue(key)
  const scheduled: ScheduledChunkRequest = {
    descriptor: request.descriptor,
    requestId: request.requestId,
    generatorId: activeGenerator.id,
    generatorToken
  }
  buildQueue.push(scheduled)
}

function removeFromBuildQueue(key: string): void {
  const index = buildQueue.findIndex((queued) => queued.descriptor.key === key)
  if (index !== -1) {
    buildQueue.splice(index, 1)
  }
}

function reorderBuildQueue(): void {
  buildQueue.sort((a, b) => {
    const lodDiff = a.descriptor.lodIndex - b.descriptor.lodIndex
    if (lodDiff !== 0) {
      return lodDiff
    }
    const distanceDiff = chunkRequestDistance(a) - chunkRequestDistance(b)
    if (distanceDiff !== 0) {
      return distanceDiff
    }
    return a.descriptor.key.localeCompare(b.descriptor.key)
  })
}

function chunkRequestDistance(request: ChunkRequest): number {
  const size = chunkWorldSize(request.descriptor.lodIndex)
  const centerX = (request.descriptor.chunkX + 0.5) * size
  const centerY = (request.descriptor.chunkY + 0.5) * size
  const centerZ = (request.descriptor.chunkZ + 0.5) * size
  const dx = centerX - currentCameraPosition.x
  const dy = centerY - currentCameraPosition.y
  const dz = centerZ - currentCameraPosition.z
  return Math.hypot(dx, dy, dz)
}

function dispatchBuilds(): void {
  while (inflightBuilds < MAX_INFLIGHT_BUILDS && buildQueue.length > 0) {
    const request = buildQueue.shift()!
    pendingBuilds.set(request.descriptor.key, {
      descriptor: request.descriptor,
      requestId: request.requestId,
      generatorId: request.generatorId,
      generatorToken: request.generatorToken
    })
    const workerRequest: ChunkWorkerRequest = {
      key: request.descriptor.key,
      lodIndex: request.descriptor.lodIndex,
      chunkX: request.descriptor.chunkX,
      chunkY: request.descriptor.chunkY,
      chunkZ: request.descriptor.chunkZ,
      requestId: request.requestId,
      generatorId: request.generatorId,
      generatorToken: request.generatorToken
    }
    inflightBuilds++
    const canUseGpu =
      chunkFieldGenerator && !suppressGpuPath && chunkFieldGenerator.supportsGenerator(request.generatorId)
    if (canUseGpu) {
      chunkFieldGenerator
        .generateChunkData(workerRequest)
        .then((response) => handleWorkerMessage(response))
        .catch((error) => {
          console.error('WebGPU chunk generation failed; falling back to worker', error)
          suppressGpuPath = true
          const worker = ensureChunkWorker()
          worker.postMessage(workerRequest)
        })
    } else {
      const worker = ensureChunkWorker()
      worker.postMessage(workerRequest)
    }
  }
}

function handleWorkerMessage(message: ChunkWorkerResponse): void {
  const pending = pendingBuilds.get(message.key)
  pendingBuilds.delete(message.key)
  if (pending && pending.generatorToken === message.generatorToken && inflightBuilds > 0) {
    inflightBuilds--
  } else if (!pending && inflightBuilds > 0) {
    inflightBuilds--
  }
  dispatchBuilds()
  updateTerrainStats()

  if (!pending || pending.requestId !== message.requestId || pending.generatorToken !== message.generatorToken) {
    return
  }

  const sampler = createChunkFieldSampler({
    data: new Int8Array(message.buffer),
    minX: message.minX,
    minY: message.minY,
    minZ: message.minZ,
    size: message.size
  })

  const descriptor = pending.descriptor
  const record = buildChunkRecord(descriptor, sampler, pending.generatorId)
  const outcome = record ? 'mesh' : 'empty'

  if (record) {
    stageChunkRecord(record)
  }

  const finalizePlan = chunkScheduler.finalizeRequest({
    descriptor,
    requestId: pending.requestId,
    outcome
  })
  applyChunkPlan(finalizePlan)
}

function buildChunkRecord(
  descriptor: ChunkDescriptor,
  sampler: DensityFunction,
  generatorId: DensityGeneratorId
): ChunkRecord | null {
  const cellScale = 1 << descriptor.lodIndex
  const samplesPerAxis = BLOCK_WIDTH * cellScale
  const originY = descriptor.chunkY * samplesPerAxis
  const origin = createVector3i(
    descriptor.chunkX * samplesPerAxis,
    originY,
    descriptor.chunkZ * samplesPerAxis
  )

  const transitionFaces = descriptor.lodIndex > 0 ? descriptor.transitionFaces : undefined

  const mergedMeshData = mesher.extractBlock(sampler, {
    origin,
    lodIndex: descriptor.lodIndex,
    cellSize: CELL_SIZE,
    transitionFaces
  })

  if (mergedMeshData.indices.length === 0) {
    return null
  }

  const built = meshDataToGeometry(mergedMeshData)
  const material = new MeshStandardMaterial({
    color: descriptor.color,
    roughness: 0.9,
    metalness: 0.05,
    flatShading: true
  })
  const mesh = new Mesh(built.geometry, material)
  mesh.castShadow = true
  mesh.receiveShadow = true

  return {
    mesh,
    material,
    key: descriptor.key,
    lodIndex: descriptor.lodIndex,
    chunkX: descriptor.chunkX,
    chunkY: descriptor.chunkY,
    chunkZ: descriptor.chunkZ,
    color: descriptor.color,
    generatorId,
    descriptor
  }
}

function addChunkToScene(record: ChunkRecord): void {
  if (record.mesh) {
    scene.add(record.mesh)
  }
  activeChunks.set(record.key, record)
  refreshDensityForRecord(record)
}

function disposeChunk(key: string): void {
  const record = activeChunks.get(key)
  if (!record) {
    return
  }

  releaseChunkResources(record)
  activeChunks.delete(key)
}

function releaseChunkResources(record: ChunkRecord): void {
  if (record.mesh) {
    scene.remove(record.mesh)
    record.mesh.geometry.dispose()
    record.mesh = undefined
  }
  if (record.material) {
    record.material.dispose()
    record.material = undefined
  }
  if (record.densityMesh) {
    densityGroup.remove(record.densityMesh)
    record.densityMesh = undefined
  }
}

function stageChunkRecord(record: ChunkRecord): void {
  stagedChunks.set(record.key, record)
  if (!shouldHoldForParent(record)) {
    showStagedChunk(record.key)
  }
}

function promoteChildrenForParent(parentKey: string): void {
  const parentInfo = parseChunkKey(parentKey)
  if (!parentInfo || parentInfo.lodIndex === 0) {
    return
  }
  const childKeys = childKeysForParent(parentInfo)
  if (childKeys.every((key) => stagedChunks.has(key))) {
    childKeys.forEach((key) => showStagedChunk(key))
  }
}

function showStagedChunk(key: string): void {
  const record = stagedChunks.get(key)
  if (!record) {
    return
  }
  stagedChunks.delete(key)
  addChunkToScene(record)
}

function discardStagedChunk(key: string): void {
  const record = stagedChunks.get(key)
  if (!record) {
    return
  }
  stagedChunks.delete(key)
  releaseChunkResources(record)
}

function shouldHoldForParent(record: ChunkRecord): boolean {
  const parentKey = parentKeyFor(record)
  if (!parentKey) {
    return false
  }
  return activeChunks.has(parentKey)
}

function parentKeyFor(record: ChunkRecord): string | null {
  if (record.lodIndex >= MAX_LOD_INDEX) {
    return null
  }
  const parentLod = record.lodIndex + 1
  const parentX = Math.floor(record.chunkX / 2)
  const parentY = Math.floor(record.chunkY / 2)
  const parentZ = Math.floor(record.chunkZ / 2)
  return chunkKey(parentLod, parentX, parentY, parentZ)
}

function parseChunkKey(key: string): { lodIndex: number; chunkX: number; chunkY: number; chunkZ: number } | null {
  const parts = key.split(':')
  if (parts.length !== 4) {
    return null
  }
  const lodIndex = Number(parts[0])
  const chunkX = Number(parts[1])
  const chunkY = Number(parts[2])
  const chunkZ = Number(parts[3])
  if (Number.isNaN(lodIndex) || Number.isNaN(chunkX) || Number.isNaN(chunkY) || Number.isNaN(chunkZ)) {
    return null
  }
  return { lodIndex, chunkX, chunkY, chunkZ }
}

function childKeysForParent(parent: { lodIndex: number; chunkX: number; chunkY: number; chunkZ: number }): string[] {
  if (parent.lodIndex === 0) {
    return []
  }
  const childLod = parent.lodIndex - 1
  const baseX = parent.chunkX * 2
  const baseY = parent.chunkY * 2
  const baseZ = parent.chunkZ * 2
  const keys: string[] = []
  for (let dz = 0; dz < 2; dz++) {
    for (let dy = 0; dy < 2; dy++) {
      for (let dx = 0; dx < 2; dx++) {
        keys.push(chunkKey(childLod, baseX + dx, baseY + dy, baseZ + dz))
      }
    }
  }
  return keys
}

function updateTerrainStats(): void {
  if (statsLabel) {
    const building = pendingBuilds.size + buildQueue.length
    statsLabel.textContent = `${activeChunks.size} active chunks (${building} building) | ${activeGenerator.label}`
  }
}

const setDensityVisualizationEnabled = (enabled: boolean): void => {
  densityVisualizationEnabled = enabled
  densityGroup.visible = enabled
  if (densityToggle && densityToggle.checked !== enabled) {
    densityToggle.checked = enabled
  }
  if (!enabled) {
    activeChunks.forEach((record) => {
      if (record.densityMesh) {
        densityGroup.remove(record.densityMesh)
        record.densityMesh = undefined
      }
    })
    return
  }
  activeChunks.forEach((record) => refreshDensityForRecord(record))
}

const refreshDensityForRecord = (record: ChunkRecord): void => {
  if (!densityVisualizationEnabled) {
    if (record.densityMesh) {
      densityGroup.remove(record.densityMesh)
      record.densityMesh = undefined
    }
    return
  }
  if (!record.densityMesh) {
    const sampler = densitySamplerFor(record.generatorId)
    const mesh = buildDensityVisualizationMesh(record.descriptor, sampler)
    if (mesh) {
      record.densityMesh = mesh
    }
  }
  if (record.densityMesh && record.densityMesh.parent !== densityGroup) {
    densityGroup.add(record.densityMesh)
  }
}

const buildDensityVisualizationMesh = (descriptor: ChunkDescriptor, sampler: DensityFunction): InstancedMesh | null => {
  if (!densityVisualizationEnabled) {
    return null
  }
  const cellScale = 1 << descriptor.lodIndex
  const samplesPerAxis = BLOCK_WIDTH * cellScale
  const stride = Math.max(DENSITY_SAMPLE_STRIDE_BASE, cellScale)
  const originX = descriptor.chunkX * samplesPerAxis
  const originY = descriptor.chunkY * samplesPerAxis
  const originZ = descriptor.chunkZ * samplesPerAxis
  const samples: Array<{ x: number; y: number; z: number; density: number }> = []
  for (let z = 0; z <= samplesPerAxis; z += stride) {
    const worldZ = originZ + z
    for (let y = 0; y <= samplesPerAxis; y += stride) {
      const worldY = originY + y
      for (let x = 0; x <= samplesPerAxis; x += stride) {
        const worldX = originX + x
        const density = sampler(worldX, worldY, worldZ)
        samples.push({ x: worldX, y: worldY, z: worldZ, density })
      }
    }
  }
  if (samples.length === 0) {
    return null
  }
  const mesh = new InstancedMesh(densitySphereGeometry, densitySphereMaterial, samples.length)
  samples.forEach((sample, index) => {
    densityTransformScratch.position.set(sample.x * CELL_SIZE, sample.y * CELL_SIZE, sample.z * CELL_SIZE)
    const scale = densityScaleForValue(sample.density)
    densityTransformScratch.scale.setScalar(scale)
    densityTransformScratch.updateMatrix()
    mesh.setMatrixAt(index, densityTransformScratch.matrix)
    mesh.setColorAt(index, densityColorForValue(sample.density))
  })
  mesh.instanceMatrix.needsUpdate = true
  if (mesh.instanceColor) {
    mesh.instanceColor.needsUpdate = true
  }
  return mesh
}

const densityScaleForValue = (value: number): number => {
  const normalized = Math.min(1, Math.abs(value) / 80)
  return 0.08 + normalized * 0.35
}

const densityColorForValue = (value: number): Color => {
  const strength = Math.min(1, Math.abs(value) / 80)
  const target = value >= 0 ? densityColorPositive : densityColorNegative
  return densityColorScratch.copy(densityColorNeutral).lerp(target, strength)
}

const clampDensity = (value: number): number => Math.max(-127, Math.min(127, Math.floor(value)))

const sampleTerrainDensityValue: DensityFunction = (x, y, z) => {
  const height = terrainHeightEstimate(x, z)
  const strata = Math.sin((x + z) * 0.05) * 2.5
  return clampDensity((height + strata - y) * 6)
}

const samplePlateauDensityValue: DensityFunction = (x, y, z) => {
  const height = plateauHeightEstimate(x, z)
  return clampDensity((height - y) * 12)
}

const nearestSphereCenter = (value: number): number =>
  Math.round(value / FLOATING_SPHERE_SPACING) * FLOATING_SPHERE_SPACING

const sampleFloatingSpheresDensityValue: DensityFunction = (x, y, z) => {
  const centerX = nearestSphereCenter(x)
  const centerZ = nearestSphereCenter(z)
  const centerY = floatingSphereCenterY(centerX, centerZ)
  const dx = x - centerX
  const dy = y - centerY
  const dz = z - centerZ
  const dist = Math.hypot(dx, dy, dz)
  return clampDensity((FLOATING_SPHERE_RADIUS - dist) * 12)
}

const densitySamplerFor = (generatorId: DensityGeneratorId): DensityFunction => {
  switch (generatorId) {
    case 'plateaus':
      return samplePlateauDensityValue
    case 'spheres':
      return sampleFloatingSpheresDensityValue
    default:
      return sampleTerrainDensityValue
  }
}

setupGeneratorControls()
if (densityToggle) {
  densityToggle.addEventListener('change', () => setDensityVisualizationEnabled(densityToggle.checked))
  setDensityVisualizationEnabled(densityToggle.checked)
} else {
  setDensityVisualizationEnabled(false)
}

const animate = () => {
  controls.update()
  updateTerrain(controls.target)
  renderer.render(scene, camera)
  requestAnimationFrame(animate)
}

requestAnimationFrame(animate)

window.addEventListener('resize', () => {
  const { innerWidth, innerHeight } = window
  camera.aspect = innerWidth / innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(innerWidth, innerHeight)
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
})

function createChunkWorker(): Worker {
  const workerSource = `
    const BLOCK_WIDTH = ${TransvoxelExtractor.BlockWidth};
    const PLATEAU_BASE_HEIGHT = ${PLATEAU_BASE_HEIGHT};
    const PLATEAU_SIN_FREQ_X = ${PLATEAU_SIN_FREQ_X};
    const PLATEAU_SIN_FREQ_Z = ${PLATEAU_SIN_FREQ_Z};
    const PLATEAU_SIN_AMPLITUDE = ${PLATEAU_SIN_AMPLITUDE};
    const FLOATING_SPHERE_SPACING = ${FLOATING_SPHERE_SPACING};
    const FLOATING_SPHERE_RADIUS = ${FLOATING_SPHERE_RADIUS};
    const FLOATING_SPHERE_BASE_HEIGHT = ${FLOATING_SPHERE_BASE_HEIGHT};
    const FLOATING_SPHERE_SWAY_AMPLITUDE = ${FLOATING_SPHERE_SWAY_AMPLITUDE};
    const FLOATING_SPHERE_SWAY_FREQUENCY = ${FLOATING_SPHERE_SWAY_FREQUENCY};

    self.onmessage = (event) => {
      const { key, lodIndex, chunkX, chunkY, chunkZ, requestId, generatorId, generatorToken } = event.data;
      const range = BLOCK_WIDTH << lodIndex;
      const transitionReach = lodIndex === 0 ? 1 : ((1 << (lodIndex - 1)) * 2 + 1);
      const padding = Math.max(1, transitionReach);
      const sampleSize = range + padding * 2 + 1;
      const minX = chunkX * range - padding;
      const minY = chunkY * range - padding;
      const minZ = chunkZ * range - padding;
      const totalSamples = sampleSize * sampleSize * sampleSize;
      const data = new Int8Array(totalSamples);
      let index = 0;
      const sampler = resolveSampler(generatorId);

      for (let z = 0; z < sampleSize; z++) {
        const worldZ = minZ + z;
        for (let y = 0; y < sampleSize; y++) {
          const worldY = minY + y;
          for (let x = 0; x < sampleSize; x++) {
            const worldX = minX + x;
            data[index++] = sampler(worldX, worldY, worldZ);
          }
        }
      }

      self.postMessage({ key, minX, minY, minZ, size: sampleSize, buffer: data.buffer, requestId, generatorId, generatorToken }, [data.buffer]);
    };

    function resolveSampler(generatorId) {
      switch (generatorId) {
        case "plateaus":
          return samplePlateauDensity;
        case "spheres":
          return sampleFloatingSpheresDensity;
        default:
          return sampleTerrainDensity;
      }
    }

    function sampleTerrainDensity(x, y, z) {
      const height = terrainHeight(x, z);
      const strata = Math.sin((x + z) * 0.05) * 2.5;
      const density = height + strata - y;
      return clampToByte(density * 6);
    }

    function samplePlateauDensity(x, y, z) {
      const height = PLATEAU_BASE_HEIGHT + Math.sin(x * PLATEAU_SIN_FREQ_X) * PLATEAU_SIN_AMPLITUDE + Math.cos(z * PLATEAU_SIN_FREQ_Z) * PLATEAU_SIN_AMPLITUDE;
      const density = height - y;
      return clampToByte(density * 12);
    }

    function sphereCenterY(x, z) {
      return FLOATING_SPHERE_BASE_HEIGHT + Math.sin((x + z) * FLOATING_SPHERE_SWAY_FREQUENCY) * FLOATING_SPHERE_SWAY_AMPLITUDE;
    }

    function sampleFloatingSpheresDensity(x, y, z) {
      const centerX = Math.round(x / FLOATING_SPHERE_SPACING) * FLOATING_SPHERE_SPACING;
      const centerZ = Math.round(z / FLOATING_SPHERE_SPACING) * FLOATING_SPHERE_SPACING;
      const centerY = sphereCenterY(centerX, centerZ);
      const dx = x - centerX;
      const dy = y - centerY;
      const dz = z - centerZ;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const density = FLOATING_SPHERE_RADIUS - dist;
      return clampToByte(density * 12);
    }

    function terrainHeight(x, z) {
      const hills = fbm(x * 0.04, z * 0.04) * 28;
      const ridges = ridgeNoise(x * 0.02, z * 0.02) * 16;
      const dunes = Math.sin(x * 0.01) * 3 + Math.cos(z * 0.012) * 3;
      return 8 + hills + ridges + dunes;
    }

    function fbm(x, z) {
      let value = 0;
      let amplitude = 1;
      let frequency = 1;
      for (let i = 0; i < 4; i++) {
        value += amplitude * valueNoise(x * frequency, z * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
      }
      return value;
    }

    function ridgeNoise(x, z) {
      const n = valueNoise(x, z);
      return 1 - Math.abs(2 * n - 1);
    }

    function valueNoise(x, z) {
      const xi = Math.floor(x);
      const zi = Math.floor(z);
      const xf = x - xi;
      const zf = z - zi;

      const v00 = hash2d(xi, zi);
      const v10 = hash2d(xi + 1, zi);
      const v01 = hash2d(xi, zi + 1);
      const v11 = hash2d(xi + 1, zi + 1);

      const u = smoothstep(xf);
      const v = smoothstep(zf);

      const top = lerp(v00, v10, u);
      const bottom = lerp(v01, v11, u);
      return lerp(top, bottom, v);
    }

    function hash2d(x, z) {
      const s = Math.sin(x * 127.1 + z * 311.7) * 43758.5453;
      return s - Math.floor(s);
    }

    const smoothstep = (t) => t * t * (3 - 2 * t);
    const lerp = (a, b, t) => a + (b - a) * t;

    function clampToByte(value) {
      return Math.max(-127, Math.min(127, Math.floor(value)));
    }
  `

  const blob = new Blob([workerSource], { type: 'application/javascript' })
  const url = URL.createObjectURL(blob)
  const worker = new Worker(url, { name: 'terrain-chunk-worker' })
  URL.revokeObjectURL(url)
  return worker
}
