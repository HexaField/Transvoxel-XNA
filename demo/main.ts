import {
  AmbientLight,
  BoxGeometry,
  Color,
  DirectionalLight,
  DoubleSide,
  Group,
  InstancedMesh,
  LineSegments,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Object3D,
  PerspectiveCamera,
  Scene,
  Vector3,
  WebGLRenderer
} from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { Tables } from '../src/lengyel/tables'
import { createVector3f } from '../src/math/vector3f'
import { addVector3i, createVector3i, multiplyVector3iScalar, type Vector3i } from '../src/math/vector3i'
import { RegularCache } from '../src/surface-extractor/cache'
import { MeshData } from '../src/surface-extractor/mesh-data'
import {
  TransvoxelExtractor,
  TransvoxelMesher,
  type TransitionFace
} from '../src/surface-extractor/transvoxel-extractor'
import type { TransvoxelVertex } from '../src/surface-extractor/vertex'
import type { DensityFunction } from '../src/volume/volume-data'
import { OrientationStats, analyzeOrientation, createInvertedHelper, createNormalHelper } from './debug-helpers'
import { createSampleVolume, meshDataToGeometry, type BuiltGeometry } from './mesh-utils'

const mount = document.querySelector<HTMLDivElement>('#app')
if (!mount) {
  throw new Error('Mount element with id "app" is missing.')
}

const BLOCK_WIDTH = TransvoxelExtractor.BlockWidth
const blockOrigin = createVector3i(0, 0, 0)
const initialExtent = BLOCK_WIDTH * 2

const scene = new Scene()
scene.background = new Color('#04060F')

const camera = new PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 500)
camera.position.set(initialExtent * 1.8, initialExtent * 1.4, initialExtent * 1.8)

const renderer = new WebGLRenderer({ antialias: true, alpha: true })
renderer.setSize(window.innerWidth, window.innerHeight)
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
mount.appendChild(renderer.domElement)

const controls = new OrbitControls(camera, renderer.domElement)
controls.target.set(initialExtent * 0.5, initialExtent * 0.5, initialExtent * 0.5)
controls.enableDamping = true
controls.update()

const keyLight = new DirectionalLight(0xffffff, 1.2)
keyLight.position.set(initialExtent, initialExtent * 1.2, initialExtent)
scene.add(keyLight)

const rimLight = new DirectionalLight(0x6ab0ff, 0.5)
rimLight.position.set(-initialExtent, initialExtent * 0.5, -initialExtent * 1.2)
scene.add(rimLight)

scene.add(new AmbientLight(0xffffff, 0.4))

const transitionFaces: TransitionFace[] = ['negativeX', 'positiveX', 'negativeY', 'positiveY', 'negativeZ', 'positiveZ']

interface MeshAccumulator {
  positions: number[]
  normals: number[]
  indices: number[]
}

const createMeshAccumulator = (): MeshAccumulator => ({
  positions: [],
  normals: [],
  indices: []
})

const finalizeAccumulator = (accumulator: MeshAccumulator): MeshData => ({
  positions: Float32Array.from(accumulator.positions),
  normals: Float32Array.from(accumulator.normals),
  indices: Uint32Array.from(accumulator.indices)
})

const CELL_SIZE = 1
const VERTEX_DENSITY_RATIO = 1.5
const chunkWorldSize = (lodIndex: number): number => CELL_SIZE * (BLOCK_WIDTH << lodIndex)
const deriveLodDistance = (lodIndex: number): number => chunkWorldSize(lodIndex) * VERTEX_DENSITY_RATIO

interface LodLevel {
  lodIndex: number
  maxDistance: number
}

const LOD_LEVELS: LodLevel[] = [0, 1, 2].map((lodIndex) => ({
  lodIndex,
  maxDistance: deriveLodDistance(lodIndex)
}))

interface DemoSettings {
  volumeScale: number
  gridMultiplier: number
  lodRadius: number
}

interface DebugControlState {
  showRegular: boolean
  showTransition: boolean
  showNormals: boolean
  showInverted: boolean
  wireframe: boolean
}

const settings: DemoSettings = {
  volumeScale: 1,
  gridMultiplier: 1,
  lodRadius: LOD_LEVELS[1].maxDistance
}

const debugControlState: DebugControlState = {
  showRegular: true,
  showTransition: true,
  showNormals: false,
  showInverted: false,
  wireframe: false
}

const blockWorldOrigin = new Vector3(0, 0, 0)
const mesher = new TransvoxelMesher()
const chunkGroup = new Group()
scene.add(chunkGroup)

let bundles: DebugBundle[] = []
let cubeGrid: InstancedMesh | null = null
let selectedCube = createSelectedCube(1)
scene.add(selectedCube)
let currentCellWorldSize = 1
let currentVolume: DensityFunction = createSampleVolume(8, 16)
let gridCellCount = BLOCK_WIDTH
let currentChunkStride = BLOCK_WIDTH

const selectionState: SelectionState = {
  x: Math.floor(BLOCK_WIDTH / 2),
  y: Math.floor(BLOCK_WIDTH / 2),
  z: Math.floor(BLOCK_WIDTH / 2)
}

const diagnosticCache = new RegularCache(BLOCK_WIDTH)
const diagnosticContext: DiagnosticContext = {
  lodScale: 1,
  lodIndex: 0,
  cellSize: settings.volumeScale,
  cellWorldSize: currentCellWorldSize,
  blockOrigin,
  volume: currentVolume,
  gridCellCount,
  chunkStride: currentChunkStride
}

const overlay = document.querySelector<HTMLDivElement>('.overlay')
const overlayDescription = overlay?.querySelector<HTMLParagraphElement>('p')
const debugPanel = document.querySelector<HTMLDivElement>('#debug-panel')

const scheduleRebuild = (() => {
  let pending = false
  return () => {
    if (pending) {
      return
    }
    pending = true
    requestAnimationFrame(() => {
      pending = false
      rebuildDemo()
    })
  }
})()

setupControlPanel(overlay, settings, scheduleRebuild)

updateSelectedCubeTransform(selectedCube, selectionState, currentCellWorldSize, blockWorldOrigin)
window.addEventListener('keydown', (event) =>
  handleSelectionKey(
    event,
    selectionState,
    selectedCube,
    diagnosticContext.cellWorldSize,
    blockWorldOrigin,
    diagnosticCache,
    diagnosticContext
  )
)

scheduleRebuild()

const animate = (): void => {
  controls.update()
  renderer.render(scene, camera)
}

renderer.setAnimationLoop(animate)

const movementMap: Record<string, { x: number; y: number; z: number }> = {
  q: { x: 0, y: 1, z: 0 },
  e: { x: 0, y: -1, z: 0 },
  w: { x: 0, y: 0, z: 1 },
  s: { x: 0, y: 0, z: -1 },
  a: { x: -1, y: 0, z: 0 },
  d: { x: 1, y: 0, z: 0 }
}

function rebuildDemo(): void {
  const lodIndex = computeLodIndex(settings.lodRadius)
  const lodScale = 1 << lodIndex
  const cellSize = settings.volumeScale
  currentCellWorldSize = lodScale * cellSize
  gridCellCount = BLOCK_WIDTH * settings.gridMultiplier
  currentChunkStride = BLOCK_WIDTH * lodScale
  const blockExtent = gridCellCount * currentCellWorldSize
  const blockCenter = blockExtent * 0.5
  currentVolume = createSampleVolume(blockCenter, blockExtent)

  const aggregatedRegular = createMeshAccumulator()
  const aggregatedTransition = createMeshAccumulator()
  const includeTransitions = lodIndex >= 1

  for (let gx = 0; gx < settings.gridMultiplier; gx++) {
    for (let gz = 0; gz < settings.gridMultiplier; gz++) {
      const origin = createVector3i(gx * currentChunkStride, 0, gz * currentChunkStride)
      const regular = mesher.extractRegularBlock(currentVolume, {
        origin,
        lodIndex,
        cellSize
      })
      appendMeshData(aggregatedRegular, regular)
      if (includeTransitions) {
        const transitionData = mesher.extractTransitionFaces(currentVolume, {
          origin,
          lodIndex,
          cellSize,
          faces: transitionFaces
        })
        appendMeshData(aggregatedTransition, transitionData)
      }
    }
  }

  const regularMeshData = finalizeAccumulator(aggregatedRegular)
  const transitionMeshData = finalizeAccumulator(aggregatedTransition)

  disposeBundles(bundles)
  bundles = []

  if (regularMeshData.indices.length > 0) {
    bundles.push(createDebugBundle('Regular', regularMeshData, 0x5bc7ff))
  }
  if (transitionMeshData.indices.length > 0) {
    bundles.push(createDebugBundle('Transition', transitionMeshData, 0xffb347))
  }

  bundles.forEach((bundle) => {
    chunkGroup.add(bundle.mesh)
    chunkGroup.add(bundle.normalHelper)
    if (bundle.invertedHelper) {
      chunkGroup.add(bundle.invertedHelper)
    }
  })

  if (cubeGrid) {
    scene.remove(cubeGrid)
    cubeGrid.geometry.dispose()
    ;(cubeGrid.material as MeshBasicMaterial).dispose()
  }
  cubeGrid = createCubeGrid(gridCellCount, currentCellWorldSize, blockWorldOrigin)
  scene.add(cubeGrid)

  scene.remove(selectedCube)
  selectedCube.geometry.dispose()
  ;(selectedCube.material as MeshBasicMaterial).dispose()
  selectedCube = createSelectedCube(currentCellWorldSize)
  scene.add(selectedCube)

  selectionState.x = clamp(selectionState.x, 0, gridCellCount - 1)
  selectionState.y = clamp(selectionState.y, 0, BLOCK_WIDTH - 1)
  selectionState.z = clamp(selectionState.z, 0, gridCellCount - 1)
  updateSelectedCubeTransform(selectedCube, selectionState, currentCellWorldSize, blockWorldOrigin)

  diagnosticContext.lodScale = lodScale
  diagnosticContext.lodIndex = lodIndex
  diagnosticContext.cellSize = cellSize
  diagnosticContext.cellWorldSize = currentCellWorldSize
  diagnosticContext.blockOrigin = blockOrigin
  diagnosticContext.volume = currentVolume
  diagnosticContext.gridCellCount = gridCellCount
  diagnosticContext.chunkStride = currentChunkStride

  const targetHeight = blockExtent * 0.5
  const frameRadius = blockExtent * 1.5 + 10
  controls.target.set(blockCenter, targetHeight, blockCenter)
  camera.position.set(blockCenter + frameRadius, targetHeight + frameRadius * 0.4, blockCenter + frameRadius)
  controls.update()

  keyLight.position.set(blockExtent, blockExtent * 1.2, blockExtent)
  rimLight.position.set(-blockExtent, blockExtent * 0.5, -blockExtent * 1.2)

  if (overlayDescription) {
    overlayDescription.innerHTML = `LOD <code>${lodIndex}</code> · ${
      settings.gridMultiplier
    }&times; grid · ${gridCellCount} cells · radius ${Math.round(settings.lodRadius)}`
  }

  if (debugPanel) {
    renderDebugPanel(debugPanel, bundles, debugControlState, applyDebugState)
  }
  applyDebugState()

  if (bundles.length > 0) {
    console.table(
      bundles.map((bundle) => ({
        mesh: bundle.name,
        vertices: bundle.meshData.positions.length / 3,
        triangles: bundle.meshData.indices.length / 3,
        inverted: bundle.stats.invertedCount,
        minDot: bundle.stats.minDot.toFixed(3),
        maxDot: bundle.stats.maxDot.toFixed(3)
      }))
    )
  }

  logSelectedCellDiagnostics(selectionState, diagnosticCache, diagnosticContext)
}

function applyDebugState(): void {
  bundles.forEach((bundle) => {
    const showBundle = bundle.name === 'Regular' ? debugControlState.showRegular : debugControlState.showTransition
    bundle.material.wireframe = debugControlState.wireframe
    bundle.mesh.visible = showBundle
    bundle.normalHelper.visible = debugControlState.showNormals && showBundle
    if (bundle.invertedHelper) {
      bundle.invertedHelper.visible = debugControlState.showInverted && showBundle
    }
  })
}

function appendMeshData(target: MeshAccumulator, source: MeshData): void {
  const baseIndex = target.positions.length / 3
  for (let i = 0; i < source.positions.length; i++) {
    target.positions.push(source.positions[i])
  }
  for (let i = 0; i < source.normals.length; i++) {
    target.normals.push(source.normals[i])
  }
  for (let i = 0; i < source.indices.length; i++) {
    target.indices.push(baseIndex + source.indices[i])
  }
}

function disposeBundles(bundleList: DebugBundle[]): void {
  bundleList.forEach((bundle) => {
    chunkGroup.remove(bundle.mesh)
    chunkGroup.remove(bundle.normalHelper)
    if (bundle.invertedHelper) {
      chunkGroup.remove(bundle.invertedHelper)
    }
    bundle.mesh.geometry.dispose()
    bundle.material.dispose()
    bundle.normalHelper.geometry.dispose()
    const normalMaterial = bundle.normalHelper.material
    if (Array.isArray(normalMaterial)) {
      normalMaterial.forEach((mat) => mat.dispose())
    } else {
      normalMaterial.dispose()
    }
    if (bundle.invertedHelper) {
      bundle.invertedHelper.geometry.dispose()
      const invertedMaterial = bundle.invertedHelper.material
      if (Array.isArray(invertedMaterial)) {
        invertedMaterial.forEach((mat) => mat.dispose())
      } else {
        invertedMaterial.dispose()
      }
    }
  })
}

function computeLodIndex(radius: number): number {
  for (const level of LOD_LEVELS) {
    if (radius <= level.maxDistance) {
      return level.lodIndex
    }
  }
  return LOD_LEVELS[LOD_LEVELS.length - 1].lodIndex
}

function renderDebugPanel(
  panel: HTMLDivElement,
  bundleList: DebugBundle[],
  state: DebugControlState,
  onStateChange: () => void
): void {
  const transitionAvailable = bundleList.some((bundle) => bundle.name === 'Transition')

  const rows = bundleList
    .map((bundle) => {
      const triangles = bundle.meshData.indices.length / 3
      return `<tr>
        <td>${bundle.name}</td>
        <td>${bundle.meshData.positions.length / 3}</td>
        <td>${triangles}</td>
        <td>${bundle.stats.invertedCount}</td>
        <td>${bundle.stats.minDot.toFixed(3)}</td>
        <td>${bundle.stats.maxDot.toFixed(3)}</td>
      </tr>`
    })
    .join('')

  panel.innerHTML = `
    <h2>Diagnostics</h2>
    <section>
      <table>
        <thead>
          <tr>
            <th>Mesh</th>
            <th>Verts</th>
            <th>Tris</th>
            <th>Inverted</th>
            <th>Min Dot</th>
            <th>Max Dot</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </section>
    <section class="debug-controls">
      <label><input type="checkbox" id="toggle-regular" ${
        state.showRegular ? 'checked' : ''
      }/> Show regular mesh</label>
      <label><input type="checkbox" id="toggle-transition" ${
        state.showTransition ? 'checked' : ''
      } ${transitionAvailable ? '' : 'disabled'}/> Show transition mesh</label>
      <label><input type="checkbox" id="toggle-wireframe" ${
        state.wireframe ? 'checked' : ''
      }/> Wireframe shading</label>
      <label><input type="checkbox" id="toggle-normals" ${
        state.showNormals ? 'checked' : ''
      }/> Display vertex normals</label>
      <label><input type="checkbox" id="toggle-inverted" ${
        state.showInverted ? 'checked' : ''
      }/> Highlight inverted faces</label>
      <button id="download-report" type="button">Download mesh report</button>
    </section>
  `

  panel.querySelector<HTMLInputElement>('#toggle-regular')?.addEventListener('change', (event) => {
    state.showRegular = (event.target as HTMLInputElement).checked
    onStateChange()
  })

  panel.querySelector<HTMLInputElement>('#toggle-transition')?.addEventListener('change', (event) => {
    state.showTransition = (event.target as HTMLInputElement).checked
    onStateChange()
  })

  panel.querySelector<HTMLInputElement>('#toggle-wireframe')?.addEventListener('change', (event) => {
    state.wireframe = (event.target as HTMLInputElement).checked
    onStateChange()
  })

  panel.querySelector<HTMLInputElement>('#toggle-normals')?.addEventListener('change', (event) => {
    state.showNormals = (event.target as HTMLInputElement).checked
    onStateChange()
  })

  panel.querySelector<HTMLInputElement>('#toggle-inverted')?.addEventListener('change', (event) => {
    state.showInverted = (event.target as HTMLInputElement).checked
    onStateChange()
  })

  panel
    .querySelector<HTMLButtonElement>('#download-report')
    ?.addEventListener('click', () => downloadReport(bundleList))
}

function downloadReport(bundleList: DebugBundle[]): void {
  const payload = bundleList.reduce<Record<string, unknown>>((acc, bundle) => {
    acc[bundle.name.toLowerCase()] = {
      vertices: bundle.meshData.positions.length / 3,
      triangles: bundle.meshData.indices.length / 3,
      diagnostics: {
        invertedTriangles: bundle.stats.invertedCount,
        minDot: bundle.stats.minDot,
        maxDot: bundle.stats.maxDot
      },
      positions: Array.from(bundle.positions),
      normals: Array.from(bundle.normals),
      indices: Array.from(bundle.indices)
    }
    return acc
  }, {})

  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: 'application/json'
  })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = 'transvoxel-mesh-report.json'
  anchor.click()
  URL.revokeObjectURL(url)
}

function setupControlPanel(
  overlayRoot: HTMLDivElement | null,
  currentSettings: DemoSettings,
  onChange: () => void
): void {
  const style = document.createElement('style')
  style.textContent = `
    .control-panel {
      margin-top: 0.8rem;
      padding-top: 0.75rem;
      border-top: 1px solid rgba(122, 142, 255, 0.25);
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
    }
    .control-panel label {
      display: flex;
      flex-direction: column;
      font-size: 0.8rem;
      gap: 0.15rem;
      color: #d5dcff;
    }
    .control-panel input[type="range"] {
      width: 100%;
    }
    .control-value {
      font-size: 0.75rem;
      color: #9fb6ff;
    }
  `
  document.head.appendChild(style)

  const host = overlayRoot ?? document.body
  const panel = document.createElement('section')
  panel.className = 'control-panel'
  panel.innerHTML = `
    <h2>Mesh Controls</h2>
    <label>
      Volume scale
      <span id="volume-scale-value" class="control-value">${currentSettings.volumeScale.toFixed(2)}×</span>
      <input type="range" id="volume-scale" min="0.5" max="2" step="0.05" value="${currentSettings.volumeScale}" />
    </label>
    <label>
      Grid copies
      <span id="grid-copies-value" class="control-value">${
        currentSettings.gridMultiplier
      }× (${currentSettings.gridMultiplier * BLOCK_WIDTH} cells)</span>
      <input type="range" id="grid-copies" min="1" max="4" step="1" value="${currentSettings.gridMultiplier}" />
    </label>
    <label>
      LOD radius
      <span id="lod-radius-value" class="control-value">LOD ${computeLodIndex(
        currentSettings.lodRadius
      )} · radius ${Math.round(currentSettings.lodRadius)}</span>
      <input type="range" id="lod-radius" min="24" max="${
        LOD_LEVELS[LOD_LEVELS.length - 1].maxDistance
      }" step="2" value="${currentSettings.lodRadius}" />
    </label>
  `
  host.appendChild(panel)

  const volumeSlider = panel.querySelector<HTMLInputElement>('#volume-scale')
  const volumeLabel = panel.querySelector<HTMLSpanElement>('#volume-scale-value')
  volumeSlider?.addEventListener('input', (event) => {
    const value = parseFloat((event.target as HTMLInputElement).value)
    currentSettings.volumeScale = value
    if (volumeLabel) {
      volumeLabel.textContent = `${value.toFixed(2)}×`
    }
    onChange()
  })

  const gridSlider = panel.querySelector<HTMLInputElement>('#grid-copies')
  const gridLabel = panel.querySelector<HTMLSpanElement>('#grid-copies-value')
  gridSlider?.addEventListener('input', (event) => {
    const value = parseInt((event.target as HTMLInputElement).value, 10)
    currentSettings.gridMultiplier = value
    if (gridLabel) {
      gridLabel.textContent = `${value}× (${value * BLOCK_WIDTH} cells)`
    }
    onChange()
  })

  const lodSlider = panel.querySelector<HTMLInputElement>('#lod-radius')
  const lodLabel = panel.querySelector<HTMLSpanElement>('#lod-radius-value')
  lodSlider?.addEventListener('input', (event) => {
    const value = parseFloat((event.target as HTMLInputElement).value)
    currentSettings.lodRadius = value
    if (lodLabel) {
      lodLabel.textContent = `LOD ${computeLodIndex(value)} · radius ${Math.round(value)}`
    }
    onChange()
  })
}

const createDebugBundle = (name: string, meshData: MeshData, color: number): DebugBundle => {
  const built = meshDataToGeometry(meshData)
  const stats = analyzeOrientation(built.positions, built.normals, built.indices)

  const material = new MeshStandardMaterial({
    color,
    metalness: 0.08,
    roughness: 0.42,
    flatShading: true,
    side: DoubleSide
  })

  const meshInstance = new Mesh(built.geometry, material)
  meshInstance.name = `${name}Mesh`

  const normalHelper = createNormalHelper(built.positions, built.normals, color)
  normalHelper.visible = false

  const invertedHelper =
    stats.invertedCount > 0 ? createInvertedHelper(built.positions, built.indices, stats.flaggedTriangles) : null
  if (invertedHelper) {
    invertedHelper.visible = false
  }

  return {
    name,
    meshData,
    material,
    mesh: meshInstance,
    normalHelper,
    invertedHelper,
    stats,
    ...built
  }
}

function createCubeGrid(cellCount: number, cellSize: number, origin: Vector3): InstancedMesh {
  const geometry = new BoxGeometry(cellSize, cellSize, cellSize)
  const material = new MeshBasicMaterial({
    color: 0x4a537a,
    wireframe: true,
    transparent: true,
    opacity: 0.1,
    depthWrite: false
  })
  const totalInstances = cellCount ** 3
  const grid = new InstancedMesh(geometry, material, totalInstances)
  const dummy = new Object3D()
  let instanceIndex = 0

  for (let x = 0; x < cellCount; x++) {
    for (let y = 0; y < cellCount; y++) {
      for (let z = 0; z < cellCount; z++) {
        dummy.position.set(
          origin.x + x * cellSize + cellSize * 0.5,
          origin.y + y * cellSize + cellSize * 0.5,
          origin.z + z * cellSize + cellSize * 0.5
        )
        dummy.updateMatrix()
        grid.setMatrixAt(instanceIndex++, dummy.matrix)
      }
    }
  }

  grid.instanceMatrix.needsUpdate = true
  grid.frustumCulled = false
  grid.renderOrder = -1
  return grid
}

function createSelectedCube(cellSize: number): Mesh {
  const geometry = new BoxGeometry(cellSize * 0.98, cellSize * 0.98, cellSize * 0.98)
  const material = new MeshBasicMaterial({
    color: 0xff5e99,
    transparent: true,
    opacity: 0.35,
    depthWrite: false,
    depthTest: false,
    wireframe: false
  })
  const mesh = new Mesh(geometry, material)
  mesh.renderOrder = 10
  mesh.name = 'SelectedCell'
  return mesh
}

function handleSelectionKey(
  event: KeyboardEvent,
  selection: SelectionState,
  selectedMesh: Mesh,
  cellSize: number,
  origin: Vector3,
  cache: RegularCache,
  context: DiagnosticContext
): void {
  const movement = movementMap[event.key.toLowerCase()]
  if (!movement) {
    return
  }

  event.preventDefault()
  const nextX = clamp(selection.x + movement.x, 0, context.gridCellCount - 1)
  const nextY = clamp(selection.y + movement.y, 0, BLOCK_WIDTH - 1)
  const nextZ = clamp(selection.z + movement.z, 0, context.gridCellCount - 1)
  if (nextX === selection.x && nextY === selection.y && nextZ === selection.z) {
    return
  }

  selection.x = nextX
  selection.y = nextY
  selection.z = nextZ
  updateSelectedCubeTransform(selectedMesh, selection, cellSize, origin)
  logSelectedCellDiagnostics(selection, cache, context)
}

function updateSelectedCubeTransform(
  selectedMesh: Mesh,
  selection: SelectionState,
  cellSize: number,
  origin: Vector3
): void {
  selectedMesh.position.copy(getCellCenter(selection, cellSize, origin))
}

function logSelectedCellDiagnostics(selection: SelectionState, cache: RegularCache, context: DiagnosticContext): void {
  const diagnostics = evaluateRegularCellDiagnostics(selection, cache, context)
  const row = {
    cell: `(${selection.x}, ${selection.y}, ${selection.z})`,
    caseCode: diagnostics.caseCodeHex,
    classIndex: diagnostics.classIndex,
    filled: `${diagnostics.filledCorners}/8`,
    mask: diagnostics.binaryMask,
    expected: `${diagnostics.expectedVertices}/${diagnostics.expectedTriangles}`,
    actual: `${diagnostics.actualVertices}/${diagnostics.actualTriangles}`,
    matches: diagnostics.matches ? '✅' : '⚠️'
  }

  console.group(`Regular cell diagnostics ${row.cell}`)
  console.table([row])
  console.groupEnd()
}

function evaluateRegularCellDiagnostics(
  selection: SelectionState,
  cache: RegularCache,
  context: DiagnosticContext
): {
  caseCode: number
  caseCodeHex: string
  classIndex: number
  expectedVertices: number
  expectedTriangles: number
  actualVertices: number
  actualTriangles: number
  binaryMask: string
  filledCorners: number
  matches: boolean
} {
  cache.reset()
  const chunkX = Math.floor(selection.x / BLOCK_WIDTH)
  const chunkZ = Math.floor(selection.z / BLOCK_WIDTH)
  const localX = selection.x % BLOCK_WIDTH
  const localZ = selection.z % BLOCK_WIDTH
  const xyz = createVector3i(localX, selection.y, localZ)
  const chunkOrigin = createVector3i(chunkX * context.chunkStride, 0, chunkZ * context.chunkStride)
  const min = addVector3i(chunkOrigin, multiplyVector3iScalar(xyz, context.lodScale))
  const offset = createVector3f(
    chunkOrigin.x * context.cellSize,
    chunkOrigin.y * context.cellSize,
    chunkOrigin.z * context.cellSize
  )

  const vertices: TransvoxelVertex[] = []
  const indices: number[] = []

  TransvoxelExtractor.polygonizeRegularCell(
    min,
    offset,
    xyz,
    chunkOrigin,
    context.volume,
    context.lodIndex,
    context.cellSize,
    vertices,
    indices,
    cache
  )

  const cacheCell = cache.getCellByVector(xyz)
  const caseCode = cacheCell.caseIndex & 0xff
  const classIndex = Tables.RegularCellClass[caseCode]
  const entry = Tables.RegularCellData[classIndex]
  const expectedVertices = entry.getVertexCount()
  const expectedTriangles = entry.getTriangleCount()
  const actualVertices = vertices.length
  const actualTriangles = indices.length / 3
  const binaryMask = formatBinary(caseCode, 8)
  const filledCorners = countBits(caseCode)
  const matches = expectedVertices === actualVertices && expectedTriangles === actualTriangles

  return {
    caseCode,
    caseCodeHex: `0x${formatHex(caseCode, 2)}`,
    classIndex,
    expectedVertices,
    expectedTriangles,
    actualVertices,
    actualTriangles,
    binaryMask,
    filledCorners,
    matches
  }
}

function getCellCenter(selection: SelectionState, cellSize: number, origin: Vector3): Vector3 {
  return new Vector3(
    origin.x + selection.x * cellSize + cellSize * 0.5,
    origin.y + selection.y * cellSize + cellSize * 0.5,
    origin.z + selection.z * cellSize + cellSize * 0.5
  )
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function formatBinary(value: number, width: number): string {
  return value.toString(2).padStart(width, '0')
}

function formatHex(value: number, width: number): string {
  return value.toString(16).toUpperCase().padStart(width, '0')
}

function countBits(value: number): number {
  let v = value
  let count = 0
  while (v) {
    count += v & 1
    v >>>= 1
  }
  return count
}

interface DebugBundle extends BuiltGeometry {
  name: string
  meshData: MeshData
  material: MeshStandardMaterial
  mesh: Mesh
  normalHelper: LineSegments
  invertedHelper: LineSegments | null
  stats: OrientationStats
}

interface SelectionState {
  x: number
  y: number
  z: number
}

interface DiagnosticContext {
  lodScale: number
  lodIndex: number
  cellSize: number
  cellWorldSize: number
  blockOrigin: Vector3i
  volume: DensityFunction
  gridCellCount: number
  chunkStride: number
}
