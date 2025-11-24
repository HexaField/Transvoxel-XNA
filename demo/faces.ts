import {
  AmbientLight,
  AxesHelper,
  Box3,
  Box3Helper,
  Color,
  DirectionalLight,
  DoubleSide,
  LineSegments,
  Material,
  Mesh,
  MeshStandardMaterial,
  PerspectiveCamera,
  Scene,
  Vector3,
  WebGLRenderer
} from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { Tables } from '../src/lengyel/tables'
import { matrix3x3FromColumns, multiplyMatrix3x3Vector3i } from '../src/math/matrix3x3'
import { fromVector3i as vector3fFromVector3i, vector3fZero } from '../src/math/vector3f'
import { type Vector3i, addVector3i, createVector3i, multiplyVector3iScalar, vector3iZero } from '../src/math/vector3i'
import { RegularCache, TransitionCache } from '../src/surface-extractor/cache'
import { MeshData } from '../src/surface-extractor/mesh-data'
import { TransvoxelExtractor } from '../src/surface-extractor/transvoxel-extractor'
import { TransvoxelVertex } from '../src/surface-extractor/vertex'
import type { DensityFunction } from '../src/volume/volume-data'
import { analyzeOrientation, createInvertedHelper, createNormalHelper } from './debug-helpers'
import { meshDataToGeometry } from './mesh-utils'

type CaseKind = 'regular' | 'transition'

interface CaseDefinition {
  kind: CaseKind
  caseCode: number
  bitLength: number
  classIndex: number
  expectedVertices: number
  expectedTriangles: number
  summary: string
  binaryMask: string
  title: string
  inverted: boolean
}

interface CaseStats {
  meshData: MeshData
  actualVertices: number
  actualTriangles: number
  matchesExpected: boolean
}

interface CaseDebugControls {
  element: HTMLElement
  handlePreviewReady(preview: FacePreview): void
}

const sectionsRoot = document.querySelector<HTMLDivElement>('#case-sections')
if (!sectionsRoot) {
  throw new Error('Unable to locate case sections container.')
}

const BLOCK_WIDTH = TransvoxelExtractor.BlockWidth
const REGULAR_BIT_LENGTH = 8
const TRANSITION_BIT_LENGTH = 9
const TRANSITION_LOD_INDEX = 1
const TRANSITION_CELL_SIZE = 1

const regularCaseDefinitions = Array.from({ length: 256 }, (_, code) => createRegularCaseDefinition(code))
const transitionCaseDefinitions = Array.from({ length: 512 }, (_, code) => createTransitionCaseDefinition(code))

const definitionLookup = new Map<string, CaseDefinition>()
;[...regularCaseDefinitions, ...transitionCaseDefinitions].forEach((def) => {
  definitionLookup.set(caseKey(def.kind, def.caseCode), def)
})

const statsCache = new Map<string, CaseStats>()
const regularCache = new RegularCache(BLOCK_WIDTH)
const transitionCache = new TransitionCache(BLOCK_WIDTH)
const debugControlsRegistry = new WeakMap<HTMLElement, CaseDebugControls>()

const transitionDescriptor = {
  axis: 2 as 0 | 1 | 2,
  direction: 1 as -1 | 1,
  originOffset: createVector3i(0, 0, BLOCK_WIDTH),
  localX: createVector3i(1, 0, 0),
  localY: createVector3i(0, 1, 0),
  localZ: createVector3i(0, 0, -1)
}

const transitionCoordinates = [
  createVector3i(0, 0, 0),
  createVector3i(1, 0, 0),
  createVector3i(2, 0, 0),
  createVector3i(0, 1, 0),
  createVector3i(1, 1, 0),
  createVector3i(2, 1, 0),
  createVector3i(0, 2, 0),
  createVector3i(1, 2, 0),
  createVector3i(2, 2, 0),
  createVector3i(0, 0, 2),
  createVector3i(2, 0, 2),
  createVector3i(0, 2, 2),
  createVector3i(2, 2, 2)
]

const canonicalTransitionPositions = buildTransitionPositions(
  transitionDescriptor,
  vector3iZero,
  TRANSITION_LOD_INDEX,
  TRANSITION_CELL_SIZE
)

const regularCornerLookup = new Map<string, number>()
Tables.CornerIndex.forEach((corner, index) => {
  regularCornerLookup.set(vectorKey(corner), index)
})

const transitionBitOrder = [0, 1, 2, 5, 8, 7, 6, 3, 4]
const transitionCoordinateBitLookup = new Map<string, number>()
transitionBitOrder.forEach((positionIndex, bitIndex) => {
  const key = vectorKey(canonicalTransitionPositions[positionIndex])
  transitionCoordinateBitLookup.set(key, bitIndex)
})

;[
  [9, 0],
  [10, 2],
  [11, 6],
  [12, 8]
].forEach(([duplicateIndex, sourceIndex]) => {
  const sourceKey = vectorKey(canonicalTransitionPositions[sourceIndex])
  const bitIndex = transitionCoordinateBitLookup.get(sourceKey)
  if (bitIndex !== undefined) {
    const duplicateKey = vectorKey(canonicalTransitionPositions[duplicateIndex])
    transitionCoordinateBitLookup.set(duplicateKey, bitIndex)
  }
})

const createRegularCaseSampler =
  (caseCode: number): DensityFunction =>
  (x, y, z) => {
    const clamped = createVector3i(clamp(x, 0, 1), clamp(y, 0, 1), clamp(z, 0, 1))
    const cornerIndex = regularCornerLookup.get(vectorKey(clamped))
    if (cornerIndex === undefined) {
      return 1
    }
    const filled = ((caseCode >> cornerIndex) & 1) === 1
    return filled ? -1 : 1
  }

const createTransitionCaseSampler =
  (caseCode: number): DensityFunction =>
  (x, y, z) => {
    const bitIndex = transitionCoordinateBitLookup.get(`${x},${y},${z}`)
    if (bitIndex === undefined) {
      return 1
    }
    const filled = ((caseCode >> bitIndex) & 1) === 1
    return filled ? -1 : 1
  }

const previewObserver = new IntersectionObserver(handlePreviewIntersection, {
  rootMargin: '200px',
  threshold: 0.2
})

const sections = [
  {
    title: 'Regular Cases',
    subtitle: 'Standard marching-cubes cells processed at full resolution.',
    definitions: regularCaseDefinitions
  },
  {
    title: 'Transition Cases',
    subtitle: 'Boundary cells that stitch adjacent LODs together.',
    definitions: transitionCaseDefinitions
  }
]

sections.forEach((section) => sectionsRoot.appendChild(renderCaseSection(section)))

function renderCaseSection(section: { title: string; subtitle: string; definitions: CaseDefinition[] }): HTMLElement {
  const wrapper = document.createElement('section')
  wrapper.className = 'case-section'

  const header = document.createElement('div')
  header.className = 'case-section-header'

  const title = document.createElement('h2')
  title.textContent = `${section.title} (${section.definitions.length})`
  header.appendChild(title)

  const subtitle = document.createElement('span')
  subtitle.textContent = section.subtitle
  header.appendChild(subtitle)

  wrapper.appendChild(header)

  const list = document.createElement('div')
  list.className = 'case-list'
  section.definitions.forEach((def) => list.appendChild(createCaseCard(def)))
  wrapper.appendChild(list)

  return wrapper
}

function createCaseCard(def: CaseDefinition): HTMLElement {
  const stats = getCaseStats(def)

  const card = document.createElement('article')
  card.className = 'case-card'
  if (!stats.matchesExpected) {
    card.dataset.alert = 'mismatch'
  }

  const status = document.createElement('span')
  status.className = `status-pill ${stats.matchesExpected ? 'status-pass' : 'status-fail'}`
  status.textContent = stats.matchesExpected ? 'Matches tables' : 'Mismatch'
  card.appendChild(status)

  const title = document.createElement('h3')
  title.textContent = def.title
  card.appendChild(title)

  const summary = document.createElement('p')
  summary.className = 'case-meta'
  summary.textContent = def.summary
  card.appendChild(summary)

  const statsList = document.createElement('div')
  statsList.className = 'case-stats'
  statsList.appendChild(createStatBlock('Expected', `${def.expectedVertices} verts · ${def.expectedTriangles} tris`))
  statsList.appendChild(createStatBlock('Actual', `${stats.actualVertices} verts · ${stats.actualTriangles} tris`))
  card.appendChild(statsList)

  const preview = document.createElement('div')
  preview.className = 'case-preview'
  preview.dataset.kind = def.kind
  preview.dataset.caseCode = String(def.caseCode)
  card.appendChild(preview)

  if (stats.actualVertices === 0) {
    preview.textContent = 'No geometry for this mask.'
    preview.dataset.rendered = 'true'
  } else {
    previewObserver.observe(preview)
    const debugControls = createCaseDebugControls(def)
    card.appendChild(debugControls.element)
    debugControlsRegistry.set(preview, debugControls)
  }

  const actions = document.createElement('div')
  actions.className = 'case-actions'

  const downloadButton = document.createElement('button')
  downloadButton.type = 'button'
  downloadButton.textContent = 'Download case JSON'
  downloadButton.addEventListener('click', () => downloadCaseReport(def, stats))
  actions.appendChild(downloadButton)

  const copyButton = document.createElement('button')
  copyButton.type = 'button'
  copyButton.textContent = 'Copy summary'
  copyButton.addEventListener('click', () => copyCaseSummary(def, stats, copyButton))
  actions.appendChild(copyButton)

  card.appendChild(actions)

  return card
}

function createCaseDebugControls(def: CaseDefinition): CaseDebugControls {
  const container = document.createElement('div')
  container.className = 'case-debug-controls'

  const state = {
    showNormals: false,
    showInverted: false
  }

  let previewRef: FacePreview | null = null

  const buildToggle = (id: string, labelText: string): { label: HTMLLabelElement; input: HTMLInputElement } => {
    const input = document.createElement('input')
    input.type = 'checkbox'
    input.id = id
    input.disabled = false

    const label = document.createElement('label')
    label.htmlFor = id
    label.appendChild(input)
    const text = document.createElement('span')
    text.textContent = labelText
    label.appendChild(text)
    return { label, input }
  }

  const normalsId = `normals-${def.kind}-${def.caseCode}`
  const normals = buildToggle(normalsId, 'Display vertex normals')
  normals.input.addEventListener('change', () => {
    state.showNormals = normals.input.checked
    previewRef?.setNormalsVisible(state.showNormals)
  })
  container.appendChild(normals.label)

  const invertedId = `inverted-${def.kind}-${def.caseCode}`
  const inverted = buildToggle(invertedId, 'Highlight inverted faces')
  const invertedNote = document.createElement('span')
  invertedNote.className = 'case-debug-note'
  invertedNote.textContent = 'No inverted faces detected'
  invertedNote.hidden = true
  inverted.label.appendChild(invertedNote)
  inverted.input.addEventListener('change', () => {
    state.showInverted = inverted.input.checked
    previewRef?.setInvertedVisible(state.showInverted)
  })
  container.appendChild(inverted.label)

  const handlePreviewReady = (instance: FacePreview): void => {
    previewRef = instance
    normals.input.disabled = false
    previewRef.setNormalsVisible(state.showNormals)

    if (instance.hasInvertedHelper()) {
      inverted.input.disabled = false
      invertedNote.hidden = true
      previewRef.setInvertedVisible(state.showInverted)
    } else {
      state.showInverted = false
      inverted.input.checked = false
      inverted.input.disabled = true
      invertedNote.hidden = false
      previewRef.setInvertedVisible(false)
    }
  }

  return { element: container, handlePreviewReady }
}

function createStatBlock(label: string, value: string): HTMLElement {
  const wrapper = document.createElement('div')
  const dt = document.createElement('dt')
  dt.textContent = label
  const dd = document.createElement('dd')
  dd.textContent = value
  wrapper.appendChild(dt)
  wrapper.appendChild(dd)
  wrapper.className = 'case-stat'
  return wrapper
}

function getCaseStats(def: CaseDefinition): CaseStats {
  const key = caseKey(def.kind, def.caseCode)
  const cached = statsCache.get(key)
  if (cached) {
    return cached
  }

  const meshData = def.kind === 'regular' ? buildRegularCaseMesh(def.caseCode) : buildTransitionCaseMesh(def.caseCode)

  const actualVertices = meshData.vertices.length
  const actualTriangles = meshData.indices.length / 3
  const matchesExpected = actualVertices === def.expectedVertices && actualTriangles === def.expectedTriangles

  const stats: CaseStats = {
    meshData,
    actualVertices,
    actualTriangles,
    matchesExpected
  }
  statsCache.set(key, stats)
  return stats
}

function buildRegularCaseMesh(caseCode: number): MeshData {
  regularCache.reset()
  const vertices: TransvoxelVertex[] = []
  const indices: number[] = []
  TransvoxelExtractor.polygonizeRegularCell(
    vector3iZero,
    vector3fZero,
    vector3iZero,
    vector3iZero,
    createRegularCaseSampler(caseCode),
    0,
    1,
    vertices,
    indices,
    regularCache
  )
  return new MeshData(vertices, indices)
}

function buildTransitionCaseMesh(caseCode: number): MeshData {
  transitionCache.reset()
  const vertices: TransvoxelVertex[] = []
  const indices: number[] = []
  TransvoxelExtractor.polygonizeTransitionCell(
    vector3fZero,
    vector3iZero,
    transitionDescriptor.localX,
    transitionDescriptor.localY,
    transitionDescriptor.localZ,
    0,
    0,
    1,
    TRANSITION_LOD_INDEX,
    transitionDescriptor.axis,
    0,
    createTransitionCaseSampler(caseCode),
    vertices,
    indices,
    transitionCache,
    vector3iZero
  )
  return new MeshData(vertices, indices)
}

class FacePreview {
  private readonly scene = new Scene()
  private readonly camera = new PerspectiveCamera(40, 1, 0.05, 2000)
  private readonly renderer = new WebGLRenderer({ antialias: true, alpha: true })
  private readonly controls: OrbitControls
  private readonly extent = BLOCK_WIDTH * (1 << TRANSITION_LOD_INDEX)
  private readonly previewCenter = new Vector3(0, 0, 0)
  private helperExtent = this.extent
  private normalHelper: LineSegments | null = null
  private invertedHelper: LineSegments | null = null

  constructor(container: HTMLElement, meshData: MeshData, color: number) {
    this.scene.background = new Color('#03040a')
    this.previewCenter.setScalar(this.extent * 0.5)

    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    container.appendChild(this.renderer.domElement)

    const { mesh, radius, normalHelper, invertedHelper } = this.createScaledMesh(meshData, color)
    this.helperExtent = Math.max(radius * 2.2, this.extent * 0.5)
    this.normalHelper = normalHelper
    this.normalHelper.visible = true
    this.normalHelper.position.copy(this.previewCenter)
    this.scene.add(this.normalHelper)

    if (invertedHelper) {
      this.invertedHelper = invertedHelper
      this.invertedHelper.visible = true
      this.invertedHelper.position.copy(this.previewCenter)
      this.scene.add(this.invertedHelper)
    }

    this.scene.add(mesh)

    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.target.copy(this.previewCenter)
    this.controls.enableDamping = true

    this.configureHelpers()
    this.configureLights()
    this.positionCamera(radius)

    const animate = () => {
      this.controls.update()
      this.renderer.render(this.scene, this.camera)
    }
    this.renderer.setAnimationLoop(animate)

    const resizeObserver = new ResizeObserver(() => this.handleResize(container))
    resizeObserver.observe(container)
    this.handleResize(container)
  }

  public setNormalsVisible(visible: boolean): void {
    if (this.normalHelper) {
      this.normalHelper.visible = visible
    }
  }

  public setInvertedVisible(visible: boolean): void {
    if (this.invertedHelper) {
      this.invertedHelper.visible = visible
    }
  }

  public hasInvertedHelper(): boolean {
    return Boolean(this.invertedHelper)
  }

  private createScaledMesh(
    meshData: MeshData,
    color: number
  ): { mesh: Mesh; radius: number; normalHelper: LineSegments; invertedHelper: LineSegments | null } {
    const built = meshDataToGeometry(meshData)
    const { geometry } = built
    geometry.computeBoundingBox()
    const boundingBox =
      geometry.boundingBox?.clone() ?? new Box3(new Vector3(-0.5, -0.5, -0.5), new Vector3(0.5, 0.5, 0.5))

    const size = boundingBox.getSize(new Vector3())
    const maxDimension = Math.max(size.x, size.y, size.z, 1e-3)
    geometry.center()
    const targetSize = this.extent * 0.65
    const scale = targetSize / maxDimension
    geometry.scale(scale, scale, scale)
    geometry.computeBoundingSphere()
    const radius = geometry.boundingSphere?.radius ?? targetSize * 0.5

    const orientation = analyzeOrientation(built.positions, built.normals, built.indices)

    const normalHelper = createNormalHelper(built.positions, built.normals, color)
    const invertedHelper =
      orientation.invertedCount > 0
        ? createInvertedHelper(built.positions, built.indices, orientation.flaggedTriangles)
        : null

    const material = new MeshStandardMaterial({
      color,
      metalness: 0.08,
      roughness: 0.4,
      flatShading: true,
      side: DoubleSide
    })

    const mesh = new Mesh(geometry, material)
    mesh.position.copy(this.previewCenter)
    return { mesh, radius, normalHelper, invertedHelper }
  }

  private configureHelpers(): void {
    const halfExtent = this.helperExtent * 0.5
    const min = new Vector3(
      this.previewCenter.x - halfExtent,
      this.previewCenter.y - halfExtent,
      this.previewCenter.z - halfExtent
    )
    const max = new Vector3(
      this.previewCenter.x + halfExtent,
      this.previewCenter.y + halfExtent,
      this.previewCenter.z + halfExtent
    )
    const box = new Box3(min, max)
    const boxHelper = new Box3Helper(box, 0x4a537a)
    this.scene.add(boxHelper)
    const boxMaterial = getSingleMaterial(boxHelper.material)
    if (boxMaterial) {
      boxMaterial.opacity = 0.35
      boxMaterial.transparent = true
    }

    const axes = new AxesHelper(this.helperExtent * 0.55)
    axes.position.copy(this.previewCenter)
    const axesMaterial = getSingleMaterial(axes.material)
    if (axesMaterial) {
      axesMaterial.depthTest = false
      axesMaterial.transparent = true
      axesMaterial.opacity = 0.75
    }
    this.scene.add(axes)
  }

  private configureLights(): void {
    this.scene.add(new AmbientLight(0xffffff, 0.7))
    const keyLight = new DirectionalLight(0xffffff, 1.15)
    keyLight.position.set(this.extent, this.extent, this.extent)
    this.scene.add(keyLight)
  }

  private positionCamera(radius: number): void {
    const safeRadius = Math.max(radius, this.extent * 0.18)
    const distance = safeRadius * 3.2
    const offset = new Vector3(1.1, 1.0, 0.9).normalize().multiplyScalar(distance)
    this.camera.position.copy(this.previewCenter.clone().add(offset))
    this.camera.near = Math.max(0.05, safeRadius * 0.05)
    this.camera.far = safeRadius * 20
    this.controls.target.copy(this.previewCenter)
    this.camera.updateProjectionMatrix()
  }

  private handleResize(container: HTMLElement): void {
    const { clientWidth, clientHeight } = container
    this.renderer.setSize(clientWidth, clientHeight, false)
    this.camera.aspect = clientWidth / clientHeight
    this.camera.updateProjectionMatrix()
  }
}

function createRegularCaseDefinition(caseCode: number): CaseDefinition {
  const classIndex = Tables.RegularCellClass[caseCode]
  const entry = Tables.RegularCellData[classIndex]
  const binaryMask = formatBinary(caseCode, REGULAR_BIT_LENGTH)
  const filled = countBits(caseCode)
  return {
    kind: 'regular',
    caseCode,
    bitLength: REGULAR_BIT_LENGTH,
    classIndex,
    expectedVertices: entry.getVertexCount(),
    expectedTriangles: entry.getTriangleCount(),
    summary: `${filled} / 8 corners solid (mask ${binaryMask}). Uses lookup class ${classIndex}.`,
    binaryMask,
    title: `Regular Case 0x${formatHex(caseCode, 2)}`,
    inverted: false
  }
}

function createTransitionCaseDefinition(caseCode: number): CaseDefinition {
  const rawClass = Tables.TransitionCellClass[caseCode]
  const classIndex = rawClass & 0x7f
  const inverted = (rawClass & 0x80) !== 0
  const entry = Tables.TransitionRegularCellData[classIndex]
  const binaryMask = formatBinary(caseCode, TRANSITION_BIT_LENGTH)
  const filled = countBits(caseCode)
  return {
    kind: 'transition',
    caseCode,
    bitLength: TRANSITION_BIT_LENGTH,
    classIndex,
    expectedVertices: entry.getVertexCount(),
    expectedTriangles: entry.getTriangleCount(),
    summary:
      `${filled} / 9 high-res samples solid (mask ${binaryMask}). ` +
      `Lookup class ${classIndex}${inverted ? ' (inverted)' : ''}.`,
    binaryMask,
    title: `Transition Case 0x${formatHex(caseCode, 3)}`,
    inverted
  }
}

function downloadCaseReport(def: CaseDefinition, stats: CaseStats): void {
  const { positions, normals, indices } = meshDataToGeometry(stats.meshData)
  const payload = {
    kind: def.kind,
    caseCode: def.caseCode,
    binaryMask: def.binaryMask,
    expected: {
      vertices: def.expectedVertices,
      triangles: def.expectedTriangles
    },
    actual: {
      vertices: stats.actualVertices,
      triangles: stats.actualTriangles,
      matchesExpected: stats.matchesExpected
    },
    geometry: {
      positions: Array.from(positions),
      normals: Array.from(normals),
      indices: Array.from(indices)
    }
  }

  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `${def.kind}-case-${formatHex(def.caseCode, def.kind === 'regular' ? 2 : 3)}.json`
  anchor.click()
  URL.revokeObjectURL(url)
}

async function copyCaseSummary(def: CaseDefinition, stats: CaseStats, button: HTMLButtonElement): Promise<void> {
  const summary =
    `${def.title} → ${def.summary} Expected ${def.expectedVertices}/${def.expectedTriangles} vs ` +
    `${stats.actualVertices}/${stats.actualTriangles}. ` +
    `${stats.matchesExpected ? 'Matches tables.' : 'Mismatch detected.'}`

  try {
    await navigator.clipboard?.writeText(summary)
    button.textContent = 'Copied'
    setTimeout(() => {
      button.textContent = 'Copy summary'
    }, 1500)
  } catch {
    console.warn('Clipboard not available; falling back to prompt.')
    window.prompt('Copy summary:', summary)
  }
}

function handlePreviewIntersection(entries: IntersectionObserverEntry[]): void {
  for (const entry of entries) {
    if (!entry.isIntersecting) {
      continue
    }
    const container = entry.target as HTMLElement
    if (container.dataset.rendered === 'true') {
      previewObserver.unobserve(container)
      continue
    }
    const kind = container.dataset.kind as CaseKind
    const caseCode = Number(container.dataset.caseCode)
    const def = definitionLookup.get(caseKey(kind, caseCode))
    if (!def) {
      continue
    }
    const stats = getCaseStats(def)
    const previewInstance = new FacePreview(container, stats.meshData, kind === 'regular' ? 0x5bc7ff : 0xffb347)
    debugControlsRegistry.get(container)?.handlePreviewReady(previewInstance)
    container.dataset.rendered = 'true'
    previewObserver.unobserve(container)
  }
}

function buildTransitionPositions(
  descriptor: {
    originOffset: Vector3i
    localX: Vector3i
    localY: Vector3i
    localZ: Vector3i
  },
  origin: Vector3i,
  lodIndex: number,
  cellSize: number
): Vector3i[] {
  const lodScale = 1 << lodIndex
  const sampleStep = 1 << (lodIndex - 1)
  const faceOrigin = addVector3i(origin, multiplyVector3iScalar(descriptor.originOffset, lodScale))
  const cellOrigin = faceOrigin
  const mx = multiplyVector3iScalar(descriptor.localX, sampleStep * cellSize)
  const my = multiplyVector3iScalar(descriptor.localY, sampleStep * cellSize)
  const mz = multiplyVector3iScalar(descriptor.localZ, sampleStep * cellSize)
  const basis = matrix3x3FromColumns(vector3fFromVector3i(mx), vector3fFromVector3i(my), vector3fFromVector3i(mz))
  return transitionCoordinates.map((coord) => addVector3i(cellOrigin, multiplyMatrix3x3Vector3i(basis, coord)))
}

function caseKey(kind: CaseKind, code: number): string {
  return `${kind}:${code}`
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
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

function formatBinary(value: number, width: number): string {
  return value.toString(2).padStart(width, '0')
}

function formatHex(value: number, width: number): string {
  return value.toString(16).toUpperCase().padStart(width, '0')
}

function vectorKey(vector: Vector3i): string {
  return `${vector.x},${vector.y},${vector.z}`
}

function getSingleMaterial(input: Material | Material[]): Material | null {
  if (Array.isArray(input)) {
    return input[0] ?? null
  }
  return input
}
