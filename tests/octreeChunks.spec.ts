import { describe, it, expect } from "vitest";
import {
  OctreeChunkManager,
  type ChunkDescriptor,
  type ChunkPlan,
  type ChunkRequest,
  type ChunkBuildOutcome,
  type LodLevel,
  type OctreeChunkConfig,
} from "../demo/octreeChunks";

const DEFAULT_LODS: LodLevel[] = [
  { lodIndex: 0, color: 0xff0000, maxDistance: 16 },
  { lodIndex: 1, color: 0x00ff00, maxDistance: 64 },
];

const SPLIT_LODS: LodLevel[] = [
  { lodIndex: 0, color: 0xff0000, maxDistance: 2 },
  { lodIndex: 1, color: 0x00ff00, maxDistance: 0.5 },
];

const TELEPORT_LODS: LodLevel[] = [
  { lodIndex: 0, color: 0xff0000, maxDistance: 24 },
  { lodIndex: 1, color: 0x00ff00, maxDistance: 96 },
  { lodIndex: 2, color: 0x0000ff, maxDistance: 160 },
];

const TRANSITION_TEST_LODS: LodLevel[] = [
  { lodIndex: 0, color: 0xff0000, maxDistance: 3.5 },
  { lodIndex: 1, color: 0x00ff00, maxDistance: 128 },
];

const createManager = (
  overrides: Partial<OctreeChunkConfig> = {}
): OctreeChunkManager => {
  const config: OctreeChunkConfig = {
    blockWidth: overrides.blockWidth ?? 4,
    cellSize: overrides.cellSize ?? 1,
    lodLevels: overrides.lodLevels ?? DEFAULT_LODS,
    worldMinY: overrides.worldMinY ?? -64,
    worldMaxY: overrides.worldMaxY ?? 64,
    lodDistanceMultiplier: overrides.lodDistanceMultiplier ?? 1.5,
  };
  return new OctreeChunkManager(config);
};

const chunkKeys = (plan: ChunkPlan): string[] =>
  plan.requests.map((request: ChunkRequest) => request.descriptor.key);

const completeRequest = (
  manager: OctreeChunkManager,
  request: ChunkRequest,
  outcome: ChunkBuildOutcome = "mesh"
): ChunkPlan =>
  manager.finalizeRequest({
    descriptor: request.descriptor,
    requestId: request.requestId,
    outcome,
  });

const childKeysFor = (descriptor: ChunkDescriptor): string[] => {
  const childLod = descriptor.lodIndex - 1;
  const baseX = descriptor.chunkX * 2;
  const baseY = descriptor.chunkY * 2;
  const baseZ = descriptor.chunkZ * 2;
  const keys: string[] = [];
  for (let dz = 0; dz < 2; dz++) {
    for (let dy = 0; dy < 2; dy++) {
      for (let dx = 0; dx < 2; dx++) {
        keys.push(`${childLod}:${baseX + dx}:${baseY + dy}:${baseZ + dz}`);
      }
    }
  }
  return keys;
};

const descriptorFor = (
  blockWidth: number,
  lodIndex: number,
  chunkX: number,
  chunkY: number,
  chunkZ: number,
  color = 0
): ChunkDescriptor => {
  const samplesPerAxis = blockWidth << lodIndex;
  return {
    lodIndex,
    chunkX,
    chunkY,
    chunkZ,
    key: `${lodIndex}:${chunkX}:${chunkY}:${chunkZ}`,
    color,
    transitionFaces: [],
    originY: chunkY * samplesPerAxis,
  };
};

describe("OctreeChunkManager", () => {
  it("requests coverage once for a steady camera", () => {
    const manager = createManager();
    const camera = { x: 0, y: 0, z: 0 };

    const initialPlan = manager.update(camera);
    expect(initialPlan.requests.length).toBeGreaterThan(0);

    const repeatPlan = manager.update(camera);
    expect(repeatPlan.requests).toHaveLength(0);
    expect(repeatPlan.cancels).toHaveLength(0);
  });

  it("cancels pending builds when the camera moves away", () => {
    const manager = createManager();
    const initialPlan = manager.update({ x: 0, y: 0, z: 0 });
    const pending = initialPlan.requests[0];
    expect(pending).toBeDefined();

    const farPlan = manager.update({ x: 512, y: 256, z: 512 });
    expect(farPlan.cancels).toContain(pending!.descriptor.key);
  });

  it("remembers empty completions and skips redundant work", () => {
    const manager = createManager();
    const plan = manager.update({ x: 0, y: 0, z: 0 });
    const request = plan.requests[0];
    expect(request).toBeDefined();

    completeRequest(manager, request!, "empty");

    const repeatPlan = manager.update({ x: 0, y: 0, z: 0 });
    expect(chunkKeys(repeatPlan)).not.toContain(request!.descriptor.key);
  });

  it("requests fine detail when the camera enters a coarse cell", () => {
    const blockWidth = 16;
    const lodLevels: LodLevel[] = [
      { lodIndex: 0, color: 0xff0000, maxDistance: 32 },
      { lodIndex: 1, color: 0x00ff00, maxDistance: 256 },
    ];
    const manager = createManager({ blockWidth, lodLevels });
    const chunkSize = blockWidth << 1;

    const plan = manager.update({ x: chunkSize * 0.4, y: 0, z: chunkSize * 0.4 });
    const hasFinest = plan.requests.some(
      (request: ChunkRequest) => request.descriptor.lodIndex === 0
    );
    expect(hasFinest).toBe(true);
  });

  it("covers the entire vertical range at the coarsest LOD", () => {
    const blockWidth = 8;
    const worldMinY = -32;
    const worldMaxY = 40;
    const manager = createManager({ blockWidth, worldMinY, worldMaxY });
    const plan = manager.update({ x: 0, y: 0, z: 0 });
    const coarseKeys = plan.requests
      .filter((request: ChunkRequest) => request.descriptor.lodIndex === 1)
      .map((request: ChunkRequest) => request.descriptor.chunkY);
    expect(new Set(coarseKeys).size).toBeGreaterThan(1);
  });

  it("splits an active parent chunk and releases it after children finish", () => {
    const blockWidth = 2;
    const manager = createManager({
      blockWidth,
      lodLevels: SPLIT_LODS,
      worldMinY: 0,
      worldMaxY: 4,
    });
    const chunkSize = blockWidth << 1;
    const farCamera = { x: chunkSize * 2, y: 2, z: chunkSize * 2 };
    const nearCamera = { x: chunkSize * 0.25, y: 1, z: chunkSize * 0.25 };

    const coarsePlan = manager.update(farCamera);
    const parentRequest = coarsePlan.requests.find(
      (request: ChunkRequest) => request.descriptor.lodIndex === 1
    );
    expect(parentRequest).toBeDefined();
    completeRequest(manager, parentRequest!);

    const splitPlan = manager.update(nearCamera);
    const expectedChildKeys = childKeysFor(parentRequest!.descriptor);
    expectedChildKeys.forEach((key) => {
      expect(chunkKeys(splitPlan)).toContain(key);
    });
    expect(splitPlan.releases).not.toContain(parentRequest!.descriptor.key);

    const releases = new Set<string>();
    for (const key of expectedChildKeys) {
      const childRequest = splitPlan.requests.find(
        (request: ChunkRequest) => request.descriptor.key === key
      );
      expect(childRequest).toBeDefined();
      const finalizePlan = completeRequest(manager, childRequest!);
      finalizePlan.releases.forEach((releaseKey: string) =>
        releases.add(releaseKey)
      );
    }

    expect(releases.has(parentRequest!.descriptor.key)).toBe(true);
  });

  it("merges children back into their parent when distance increases", () => {
    const blockWidth = 2;
    const manager = createManager({
      blockWidth,
      lodLevels: SPLIT_LODS,
      worldMinY: 0,
      worldMaxY: 4,
    });
    const chunkSize = blockWidth << 1;
    const farCamera = { x: chunkSize * 4, y: 3, z: chunkSize * 4 };
    const nearCamera = { x: chunkSize * 0.2, y: 1, z: chunkSize * 0.2 };

    const coarsePlan = manager.update(farCamera);
    const parentRequest = coarsePlan.requests.find(
      (request: ChunkRequest) => request.descriptor.lodIndex === 1
    );
    expect(parentRequest).toBeDefined();
    completeRequest(manager, parentRequest!);

    const splitPlan = manager.update(nearCamera);
    const childKeys = childKeysFor(parentRequest!.descriptor);
    const childRequests = splitPlan.requests.filter((request: ChunkRequest) =>
      childKeys.includes(request.descriptor.key)
    );
    expect(childRequests).toHaveLength(childKeys.length);
    childRequests.forEach((request: ChunkRequest) =>
      completeRequest(manager, request)
    );

    const mergePlan = manager.update(farCamera);
    expect(chunkKeys(mergePlan)).toContain(parentRequest!.descriptor.key);

    const parentMergeRequest = mergePlan.requests.find(
      (request: ChunkRequest) =>
        request.descriptor.key === parentRequest!.descriptor.key
    );
    expect(parentMergeRequest).toBeDefined();

    const completionPlan = completeRequest(manager, parentMergeRequest!);
    childKeys.forEach((key) => {
      expect(completionPlan.releases).toContain(key);
    });
  });

  it("assigns transition faces to coarse chunks bordering finer neighbors", () => {
    const blockWidth = 2;
    const manager = createManager({
      blockWidth,
      lodLevels: TRANSITION_TEST_LODS,
      worldMinY: 0,
      worldMaxY: blockWidth << 1,
    });
    const desired = new Map<string, ChunkDescriptor>();
    const coarse = descriptorFor(blockWidth, 1, 0, 0, 0, 0x00ff00);
    const finePositiveX = descriptorFor(blockWidth, 0, 2, 0, 0, 0xff0000);
    const finePositiveZ = descriptorFor(blockWidth, 0, 0, 0, 2, 0xff0000);
    desired.set(coarse.key, coarse);
    desired.set(finePositiveX.key, finePositiveX);
    desired.set(finePositiveZ.key, finePositiveZ);

    (manager as any).assignTransitionFaces(desired);

    expect(coarse.transitionFaces).toEqual(["positiveX", "positiveZ"]);
    expect(finePositiveX.transitionFaces).toHaveLength(0);
    expect(finePositiveZ.transitionFaces).toHaveLength(0);
  });

  it("leaves transition faces empty when no finer neighbors touch the chunk", () => {
    const blockWidth = 2;
    const manager = createManager({
      blockWidth,
      lodLevels: TRANSITION_TEST_LODS,
      worldMinY: 0,
      worldMaxY: blockWidth << 1,
    });
    const desired = new Map<string, ChunkDescriptor>();
    const coarse = descriptorFor(blockWidth, 1, 0, 0, 0, 0x00ff00);
    const neighborSameLod = descriptorFor(blockWidth, 1, 1, 0, 0, 0x00ff00);
    desired.set(coarse.key, coarse);
    desired.set(neighborSameLod.key, neighborSameLod);

    (manager as any).assignTransitionFaces(desired);

    expect(coarse.transitionFaces).toHaveLength(0);
    expect(neighborSameLod.transitionFaces).toHaveLength(0);
  });

  it("re-requests an active chunk when its transition faces change", () => {
    const blockWidth = 2;
    const manager = createManager({
      blockWidth,
      lodLevels: TRANSITION_TEST_LODS,
      worldMinY: 0,
      worldMaxY: blockWidth << 1,
    });

    const desired = new Map<string, ChunkDescriptor>();
    const coarse = descriptorFor(blockWidth, 1, 0, 0, 0, 0x00ff00);
    const finePositiveX = descriptorFor(blockWidth, 0, 2, 0, 0, 0xff0000);
    desired.set(coarse.key, { ...coarse });
    desired.set(finePositiveX.key, finePositiveX);

    (manager as any).assignTransitionFaces(desired);

    const desiredCoarse = desired.get(coarse.key)!;
    const managerInternal = manager as any;
    managerInternal.activeChunks.set(coarse.key, {
      ...coarse,
      transitionFaces: [],
    });

    const plan = managerInternal.reconcile(desired, false) as ChunkPlan;
    expect(plan.releases).toContain(coarse.key);

    const refreshed = plan.requests.find(
      (request: ChunkRequest) => request.descriptor.key === coarse.key
    );

    expect(refreshed).toBeDefined();
    expect(refreshed!.descriptor.transitionFaces).toEqual(
      desiredCoarse.transitionFaces
    );
  });
});
