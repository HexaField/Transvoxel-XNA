import {
  AmbientLight,
  Color,
  DirectionalLight,
  DoubleSide,
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
import { meshDataToGeometry, createSampleVolume } from "./mesh-utils";

const gridElement = document.querySelector<HTMLDivElement>("#face-grid");
if (!gridElement) {
  throw new Error("Face grid container is missing.");
}
const grid = gridElement;

const lodIndex = 1;
const cellSize = 1;
const blockExtent = 16 * (1 << lodIndex) * cellSize;
const blockCenter = blockExtent * 0.5;

const mesher = new TransvoxelMesher(
  createSampleVolume(blockCenter, blockExtent)
);
const faces: TransitionFace[] = [
  "negativeX",
  "positiveX",
  "negativeY",
  "positiveY",
  "negativeZ",
  "positiveZ",
];

class FacePreview {
  private readonly scene: Scene;
  private readonly camera: PerspectiveCamera;
  private readonly renderer: WebGLRenderer;
  private readonly controls: OrbitControls;

  constructor(container: HTMLElement, meshData: MeshData) {
    this.scene = new Scene();
    this.scene.background = new Color("#03040a");

    this.camera = new PerspectiveCamera(40, 1, 0.1, 500);
    this.camera.position.set(
      blockExtent * 0.8,
      blockExtent * 0.8,
      blockExtent * 0.8
    );

    this.renderer = new WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(blockCenter, blockCenter, blockCenter);
    this.controls.enableDamping = true;

    const { geometry } = meshDataToGeometry(meshData);
    const material = new MeshStandardMaterial({
      color: 0x5bc7ff,
      metalness: 0.08,
      roughness: 0.4,
      side: DoubleSide,
    });

    const mesh = new Mesh(geometry, material);
    this.scene.add(mesh);

    this.scene.add(new AmbientLight(0xffffff, 0.6));
    const light = new DirectionalLight(0xffffff, 1.1);
    light.position.set(blockExtent, blockExtent, blockExtent);
    this.scene.add(light);

    const animate = () => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    this.renderer.setAnimationLoop(animate);

    const resizeObserver = new ResizeObserver(() =>
      this.handleResize(container)
    );
    resizeObserver.observe(container);
    this.handleResize(container);
  }

  private handleResize(container: HTMLElement): void {
    const { clientWidth, clientHeight } = container;
    this.renderer.setSize(clientWidth, clientHeight, false);
    this.camera.aspect = clientWidth / clientHeight;
    this.camera.updateProjectionMatrix();
  }
}

function createFaceCard(face: TransitionFace): void {
  const card = document.createElement("section");
  card.className = "face-card";

  const title = document.createElement("h2");
  title.textContent = face;
  card.appendChild(title);

  const stats = document.createElement("p");
  stats.textContent = "Generating…";
  card.appendChild(stats);

  const preview = document.createElement("div");
  preview.className = "face-preview";
  card.appendChild(preview);

  grid.appendChild(card);

  const meshData = mesher.extractTransitionFaces({
    origin: new Vector3i(0, 0, 0),
    lodIndex,
    cellSize,
    faces: [face],
  });

  const triangleCount = meshData.indices.length / 3;
  stats.textContent =
    triangleCount === 0
      ? "No geometry generated in this configuration."
      : `${meshData.vertices.length} verts · ${triangleCount} tris`;

  if (meshData.vertices.length === 0) {
    return;
  }

  new FacePreview(preview, meshData);
}

faces.forEach((face) => createFaceCard(face));
