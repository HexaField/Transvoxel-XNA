import {
  AmbientLight,
  Color,
  DirectionalLight,
  DoubleSide,
  LineSegments,
  Mesh,
  MeshStandardMaterial,
  PerspectiveCamera,
  Scene,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  TransvoxelMesher,
  TransitionFace,
} from "../src/surface-extractor/transvoxel-extractor";
import { Vector3i } from "../src/math/vector3i";
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

const mount = document.querySelector<HTMLDivElement>("#app");
if (!mount) {
  throw new Error('Mount element with id "app" is missing.');
}

const lodIndex = 1;
const cellSize = 1;
const blockExtent = 16 * (1 << lodIndex) * cellSize;
const blockCenter = blockExtent * 0.5;
const blockOrigin = new Vector3i(0, 0, 0);

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

const volume = createSampleVolume(blockCenter, blockExtent);

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

