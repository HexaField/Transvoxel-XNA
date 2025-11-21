import {
  AmbientLight,
  BoxGeometry,
  Color,
  DirectionalLight,
  DoubleSide,
  InstancedMesh,
  LineSegments,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Object3D,
  PerspectiveCamera,
  Scene,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  TransvoxelExtractor,
  TransvoxelMesher,
  TransitionFace,
} from "../src/surface-extractor/transvoxel-extractor";
import { Tables } from "../src/lengyel/tables";
import { Vector3f } from "../src/math/vector3f";
import { Vector3i } from "../src/math/vector3i";
import { RegularCache } from "../src/surface-extractor/cache";
import { MeshData } from "../src/surface-extractor/mesh-data";
import {
  meshDataToGeometry,
  createSampleVolume,
  BuiltGeometry,
} from "./mesh-utils";
import {
  OrientationStats,
  analyzeOrientation,
  createInvertedHelper,
  createNormalHelper,
} from "./debug-helpers";
import { VolumeData } from "../src/volume/volume-data";
import type { TransvoxelVertex } from "../src/surface-extractor/vertex";

const mount = document.querySelector<HTMLDivElement>("#app");
if (!mount) {
  throw new Error('Mount element with id "app" is missing.');
}

const BLOCK_WIDTH = TransvoxelExtractor.BlockWidth;
const lodIndex = 1;
const cellSize = 1;
const lodScale = 1 << lodIndex;
const blockExtent = BLOCK_WIDTH * lodScale * cellSize;
const blockCenter = blockExtent * 0.5;
const blockOrigin = new Vector3i(0, 0, 0);
const cellWorldSize = lodScale * cellSize;
const blockWorldOrigin = new Vector3(
  blockOrigin.x * cellSize,
  blockOrigin.y * cellSize,
  blockOrigin.z * cellSize
);
const blockOffsetVector = new Vector3f(
  blockOrigin.x * cellSize,
  blockOrigin.y * cellSize,
  blockOrigin.z * cellSize
);

const scene = new Scene();
scene.background = new Color("#04060F");

const camera = new PerspectiveCamera(
  45,
  window.innerWidth / window.innerHeight,
  0.1,
  500
);
camera.position.set(blockExtent * 1.8, blockExtent * 1.4, blockExtent * 1.8);

const renderer = new WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
mount.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(blockCenter, blockCenter, blockCenter);
controls.enableDamping = true;
controls.update();

const keyLight = new DirectionalLight(0xffffff, 1.2);
keyLight.position.set(blockExtent, blockExtent * 1.2, blockExtent);
scene.add(keyLight);

const rimLight = new DirectionalLight(0x6ab0ff, 0.5);
rimLight.position.set(-blockExtent, blockExtent * 0.5, -blockExtent * 1.2);
scene.add(rimLight);

scene.add(new AmbientLight(0xffffff, 0.4));

const volume: VolumeData = createSampleVolume(blockCenter, blockExtent);

const mesher = new TransvoxelMesher(volume);
const transitionFaces: TransitionFace[] = [
  "negativeX",
  "positiveX",
  "negativeY",
  "positiveY",
  "negativeZ",
  "positiveZ",
];

const regularMeshData = mesher.extractRegularBlock({
  origin: blockOrigin,
  lodIndex,
  cellSize,
});
const transitionMeshData = mesher.extractTransitionFaces({
  origin: blockOrigin,
  lodIndex,
  cellSize,
  faces: transitionFaces,
});

interface DebugBundle extends BuiltGeometry {
  name: string;
  meshData: MeshData;
  material: MeshStandardMaterial;
  mesh: Mesh;
  normalHelper: LineSegments;
  invertedHelper: LineSegments | null;
  stats: OrientationStats;
}

const createDebugBundle = (
  name: string,
  meshData: MeshData,
  color: number
): DebugBundle => {
  const built = meshDataToGeometry(meshData);
  const stats = analyzeOrientation(
    built.positions,
    built.normals,
    built.indices
  );

  const material = new MeshStandardMaterial({
    color,
    metalness: 0.08,
    roughness: 0.42,
    flatShading: true,
    side: DoubleSide,
  });

  const meshInstance = new Mesh(built.geometry, material);
  meshInstance.name = `${name}Mesh`;

  const normalHelper = createNormalHelper(
    built.positions,
    built.normals,
    color
  );
  normalHelper.visible = false;

  const invertedHelper =
    stats.invertedCount > 0
      ? createInvertedHelper(
          built.positions,
          built.indices,
          stats.flaggedTriangles
        )
      : null;
  if (invertedHelper) {
    invertedHelper.visible = false;
  }

  return {
    name,
    meshData,
    material,
    mesh: meshInstance,
    normalHelper,
    invertedHelper,
    stats,
    ...built,
  };
};

const regularBundle = createDebugBundle("Regular", regularMeshData, 0x5bc7ff);
const transitionBundle = createDebugBundle(
  "Transition",
  transitionMeshData,
  0xffb347
);
const bundles = [regularBundle, transitionBundle];

bundles.forEach((bundle) => {
  scene.add(bundle.mesh);
  scene.add(bundle.normalHelper);
  if (bundle.invertedHelper) {
    scene.add(bundle.invertedHelper);
  }
});

scene.add(createCubeGrid(BLOCK_WIDTH, cellWorldSize, blockWorldOrigin));

const selectedCube = createSelectedCube(cellWorldSize);
scene.add(selectedCube);

const selectionState: SelectionState = {
  x: Math.floor(BLOCK_WIDTH / 2),
  y: Math.floor(BLOCK_WIDTH / 2),
  z: Math.floor(BLOCK_WIDTH / 2),
};

const diagnosticCache = new RegularCache(BLOCK_WIDTH);
const diagnosticContext: DiagnosticContext = {
  lodScale,
  lodIndex,
  cellSize,
  blockOrigin,
  blockOffsetVector,
  volume,
};

updateSelectedCubeTransform(selectedCube, selectionState, cellWorldSize, blockWorldOrigin);
logSelectedCellDiagnostics(selectionState, diagnosticCache, diagnosticContext);
window.addEventListener("keydown", (event) =>
  handleSelectionKey(
    event,
    selectionState,
    selectedCube,
    cellWorldSize,
    blockWorldOrigin,
    diagnosticCache,
    diagnosticContext
  )
);

console.table(
  bundles.map((bundle) => ({
    mesh: bundle.name,
    vertices: bundle.meshData.vertices.length,
    triangles: bundle.meshData.indices.length / 3,
    inverted: bundle.stats.invertedCount,
    minDot: bundle.stats.minDot.toFixed(3),
    maxDot: bundle.stats.maxDot.toFixed(3),
  }))
);

const debugPanel = document.querySelector<HTMLDivElement>("#debug-panel");
if (debugPanel) {
  renderDebugPanel(debugPanel, bundles);
}

const animate = (time: number) => {
  controls.update();
  renderer.render(scene, camera);
};

renderer.setAnimationLoop(animate);

window.addEventListener("resize", () => {
  const { innerWidth, innerHeight } = window;
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
});

function renderDebugPanel(
  panel: HTMLDivElement,
  bundleList: DebugBundle[]
): void {
  const statsRows = bundleList
    .map((bundle) => {
      const triangles = bundle.meshData.indices.length / 3;
      return `<tr>
        <td>${bundle.name}</td>
        <td>${bundle.meshData.vertices.length}</td>
        <td>${triangles}</td>
        <td>${bundle.stats.invertedCount}</td>
        <td>${bundle.stats.minDot.toFixed(3)}</td>
        <td>${bundle.stats.maxDot.toFixed(3)}</td>
      </tr>`;
    })
    .join("");

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
        <tbody>${statsRows}</tbody>
      </table>
    </section>
    <section class="debug-controls">
      <label><input type="checkbox" id="toggle-regular" checked /> Show regular mesh</label>
      <label><input type="checkbox" id="toggle-transition" checked /> Show transition mesh</label>
      <label><input type="checkbox" id="toggle-wireframe" /> Wireframe shading</label>
      <label><input type="checkbox" id="toggle-normals" /> Display vertex normals</label>
      <label><input type="checkbox" id="toggle-inverted" /> Highlight inverted faces</label>
      <button id="download-report" type="button">Download mesh report</button>
    </section>
  `;

  const state = {
    showNormals: false,
    showInverted: false,
  };

  const syncHelpers = () => {
    bundleList.forEach((bundle) => {
      bundle.normalHelper.visible = state.showNormals && bundle.mesh.visible;
      if (bundle.invertedHelper) {
        bundle.invertedHelper.visible =
          state.showInverted && bundle.mesh.visible;
      }
    });
  };

  const setBundleVisibility = (name: string, visible: boolean) => {
    const bundle = bundleList.find((entry) => entry.name === name);
    if (bundle) {
      bundle.mesh.visible = visible;
    }
  };

  const wireframeToggle =
    panel.querySelector<HTMLInputElement>("#toggle-wireframe");
  wireframeToggle?.addEventListener("change", (event) => {
    const checked = (event.target as HTMLInputElement).checked;
    bundleList.forEach((bundle) => {
      bundle.material.wireframe = checked;
    });
  });

  const regularToggle =
    panel.querySelector<HTMLInputElement>("#toggle-regular");
  regularToggle?.addEventListener("change", (event) => {
    const checked = (event.target as HTMLInputElement).checked;
    setBundleVisibility("Regular", checked);
    syncHelpers();
  });

  const transitionToggle =
    panel.querySelector<HTMLInputElement>("#toggle-transition");
  transitionToggle?.addEventListener("change", (event) => {
    const checked = (event.target as HTMLInputElement).checked;
    setBundleVisibility("Transition", checked);
    syncHelpers();
  });

  const normalsToggle =
    panel.querySelector<HTMLInputElement>("#toggle-normals");
  normalsToggle?.addEventListener("change", (event) => {
    state.showNormals = (event.target as HTMLInputElement).checked;
    syncHelpers();
  });

  const invertedToggle =
    panel.querySelector<HTMLInputElement>("#toggle-inverted");
  invertedToggle?.addEventListener("change", (event) => {
    state.showInverted = (event.target as HTMLInputElement).checked;
    syncHelpers();
  });

  const downloadButton =
    panel.querySelector<HTMLButtonElement>("#download-report");
  downloadButton?.addEventListener("click", () => downloadReport(bundleList));
}

function downloadReport(bundleList: DebugBundle[]): void {
  const payload = bundleList.reduce<Record<string, unknown>>((acc, bundle) => {
    acc[bundle.name.toLowerCase()] = {
      vertices: bundle.meshData.vertices.length,
      triangles: bundle.meshData.indices.length / 3,
      diagnostics: {
        invertedTriangles: bundle.stats.invertedCount,
        minDot: bundle.stats.minDot,
        maxDot: bundle.stats.maxDot,
      },
      positions: Array.from(bundle.positions),
      normals: Array.from(bundle.normals),
      indices: Array.from(bundle.indices),
    };
    return acc;
  }, {});

  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "transvoxel-mesh-report.json";
  anchor.click();
  URL.revokeObjectURL(url);
}

interface SelectionState {
  x: number;
  y: number;
  z: number;
}

interface DiagnosticContext {
  lodScale: number;
  lodIndex: number;
  cellSize: number;
  blockOrigin: Vector3i;
  blockOffsetVector: Vector3f;
  volume: VolumeData;
}

const movementMap: Record<string, { x: number; y: number; z: number }> = {
  q: { x: 0, y: 1, z: 0 },
  e: { x: 0, y: -1, z: 0 },
  w: { x: 0, y: 0, z: 1 },
  s: { x: 0, y: 0, z: -1 },
  a: { x: -1, y: 0, z: 0 },
  d: { x: 1, y: 0, z: 0 },
};

function createCubeGrid(blockWidth: number, cellSize: number, origin: Vector3): InstancedMesh {
  const geometry = new BoxGeometry(cellSize, cellSize, cellSize);
  const material = new MeshBasicMaterial({
    color: 0x4a537a,
    wireframe: true,
    transparent: true,
    opacity: 0.1,
    depthWrite: false,
  });
  const totalInstances = blockWidth ** 3;
  const grid = new InstancedMesh(geometry, material, totalInstances);
  const dummy = new Object3D();
  let instanceIndex = 0;

  for (let x = 0; x < blockWidth; x++) {
    for (let y = 0; y < blockWidth; y++) {
      for (let z = 0; z < blockWidth; z++) {
        dummy.position.set(
          origin.x + x * cellSize + cellSize * 0.5,
          origin.y + y * cellSize + cellSize * 0.5,
          origin.z + z * cellSize + cellSize * 0.5
        );
        dummy.updateMatrix();
        grid.setMatrixAt(instanceIndex++, dummy.matrix);
      }
    }
  }

  grid.instanceMatrix.needsUpdate = true;
  grid.frustumCulled = false;
  grid.renderOrder = -1;
  return grid;
}

function createSelectedCube(cellSize: number): Mesh {
  const geometry = new BoxGeometry(cellSize * 0.98, cellSize * 0.98, cellSize * 0.98);
  const material = new MeshBasicMaterial({
    color: 0xff5e99,
    transparent: true,
    opacity: 0.35,
    depthWrite: false,
    depthTest: false,
    wireframe: false,
  });
  const mesh = new Mesh(geometry, material);
  mesh.renderOrder = 10;
  mesh.name = "SelectedCell";
  return mesh;
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
  const movement = movementMap[event.key.toLowerCase()];
  if (!movement) {
    return;
  }

  event.preventDefault();
  const nextX = clamp(selection.x + movement.x, 0, BLOCK_WIDTH - 1);
  const nextY = clamp(selection.y + movement.y, 0, BLOCK_WIDTH - 1);
  const nextZ = clamp(selection.z + movement.z, 0, BLOCK_WIDTH - 1);
  if (nextX === selection.x && nextY === selection.y && nextZ === selection.z) {
    return;
  }

  selection.x = nextX;
  selection.y = nextY;
  selection.z = nextZ;
  updateSelectedCubeTransform(selectedMesh, selection, cellSize, origin);
  logSelectedCellDiagnostics(selection, cache, context);
}

function updateSelectedCubeTransform(
  selectedMesh: Mesh,
  selection: SelectionState,
  cellSize: number,
  origin: Vector3
): void {
  selectedMesh.position.copy(getCellCenter(selection, cellSize, origin));
}

function logSelectedCellDiagnostics(
  selection: SelectionState,
  cache: RegularCache,
  context: DiagnosticContext
): void {
  const diagnostics = evaluateRegularCellDiagnostics(selection, cache, context);
  const row = {
    cell: `(${selection.x}, ${selection.y}, ${selection.z})`,
    caseCode: diagnostics.caseCodeHex,
    classIndex: diagnostics.classIndex,
    filled: `${diagnostics.filledCorners}/8`,
    mask: diagnostics.binaryMask,
    expected: `${diagnostics.expectedVertices}/${diagnostics.expectedTriangles}`,
    actual: `${diagnostics.actualVertices}/${diagnostics.actualTriangles}`,
    matches: diagnostics.matches ? "✅" : "⚠️",
  };

  console.group(`Regular cell diagnostics ${row.cell}`);
  console.table([row]);
  console.groupEnd();
}

function evaluateRegularCellDiagnostics(
  selection: SelectionState,
  cache: RegularCache,
  context: DiagnosticContext
): {
  caseCode: number;
  caseCodeHex: string;
  classIndex: number;
  expectedVertices: number;
  expectedTriangles: number;
  actualVertices: number;
  actualTriangles: number;
  binaryMask: string;
  filledCorners: number;
  matches: boolean;
} {
  cache.reset();
  const xyz = new Vector3i(selection.x, selection.y, selection.z);
  const min = context.blockOrigin.add(xyz.multiplyScalar(context.lodScale));
  const vertices: TransvoxelVertex[] = [];
  const indices: number[] = [];

  TransvoxelExtractor.polygonizeRegularCell(
    min,
    context.blockOffsetVector,
    xyz,
    context.volume,
    context.lodIndex,
    context.cellSize,
    vertices,
    indices,
    cache
  );

  const cacheCell = cache.getCellByVector(xyz);
  const caseCode = cacheCell.caseIndex & 0xff;
  const classIndex = Tables.RegularCellClass[caseCode];
  const entry = Tables.RegularCellData[classIndex];
  const expectedVertices = entry.getVertexCount();
  const expectedTriangles = entry.getTriangleCount();
  const actualVertices = vertices.length;
  const actualTriangles = indices.length / 3;
  const binaryMask = formatBinary(caseCode, 8);
  const filledCorners = countBits(caseCode);
  const matches =
    expectedVertices === actualVertices && expectedTriangles === actualTriangles;

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
    matches,
  };
}

function getCellCenter(selection: SelectionState, cellSize: number, origin: Vector3): Vector3 {
  return new Vector3(
    origin.x + selection.x * cellSize + cellSize * 0.5,
    origin.y + selection.y * cellSize + cellSize * 0.5,
    origin.z + selection.z * cellSize + cellSize * 0.5
  );
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function formatBinary(value: number, width: number): string {
  return value.toString(2).padStart(width, "0");
}

function formatHex(value: number, width: number): string {
  return value.toString(16).toUpperCase().padStart(width, "0");
}

function countBits(value: number): number {
  let v = value;
  let count = 0;
  while (v) {
    count += v & 1;
    v >>>= 1;
  }
  return count;
}

