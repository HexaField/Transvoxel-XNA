import { describe, it, expect } from "vitest";
import {
  QuadChunkManager,
  type ChunkDescriptor,
  type ChunkPlan,
  type ChunkRequest,
  type ChunkBuildOutcome,
  type LodLevel,
  type QuadChunkConfig,
} from "../demo/quadChunks";

const DEFAULT_LODS: LodLevel[] = [
  { lodIndex: 0, color: 0xff0000, maxDistance: 12 },
  { lodIndex: 1, color: 0x00ff00, maxDistance: 64 },
];

const SPLIT_LODS: LodLevel[] = [
  { lodIndex: 0, color: 0xff0000, maxDistance: 2 },
  { lodIndex: 1, color: 0x00ff00, maxDistance: 0.5 },
];

const TELEPORT_LODS: LodLevel[] = [
  { lodIndex: 0, color: 0xff0000, maxDistance: 16 },
  { lodIndex: 1, color: 0x00ff00, maxDistance: 64 },
  { lodIndex: 2, color: 0x0000ff, maxDistance: 128 },
];

const createManager = (
  overrides: Partial<QuadChunkConfig> = {}
): QuadChunkManager => {
  const config: QuadChunkConfig = {
    blockWidth: overrides.blockWidth ?? 4,
    cellSize: overrides.cellSize ?? 1,
    lodLevels: overrides.lodLevels ?? DEFAULT_LODS,
    estimateOriginY: overrides.estimateOriginY ?? (() => 0),
  };
  return new QuadChunkManager(config);
};

const chunkKeys = (plan: ChunkPlan): string[] =>
  plan.requests.map((request) => request.descriptor.key);

const completeRequest = (
  manager: QuadChunkManager,
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
  const baseZ = descriptor.chunkZ * 2;
  return [
    `${childLod}:${baseX}:${baseZ}`,
    `${childLod}:${baseX + 1}:${baseZ}`,
    `${childLod}:${baseX}:${baseZ + 1}`,
    `${childLod}:${baseX + 1}:${baseZ + 1}`,
  ];
};

describe("QuadChunkManager", () => {
  it("requests coverage once for a steady camera", () => {
    const manager = createManager();
    const camera = { x: 0, z: 0 };

    const initialPlan = manager.update(camera);
    expect(initialPlan.requests.length).toBeGreaterThan(0);

    const repeatPlan = manager.update(camera);
    expect(repeatPlan.requests).toHaveLength(0);
    expect(repeatPlan.cancels).toHaveLength(0);
  });

  it("cancels pending builds when the camera moves away", () => {
    const manager = createManager();
    const initialPlan = manager.update({ x: 0, z: 0 });
    const pending = initialPlan.requests[0];
    expect(pending).toBeDefined();

    const farPlan = manager.update({ x: 512, z: 512 });
    expect(farPlan.cancels).toContain(pending!.descriptor.key);
  });

  it("remembers empty completions and skips redundant work", () => {
    const manager = createManager();
    const plan = manager.update({ x: 0, z: 0 });
    const request = plan.requests[0];
    expect(request).toBeDefined();

    completeRequest(manager, request!, "empty");

    const repeatPlan = manager.update({ x: 0, z: 0 });
    expect(chunkKeys(repeatPlan)).not.toContain(request!.descriptor.key);
  });

  it("requests fine detail when the camera hugs a coarse chunk edge", () => {
    const blockWidth = 32;
    const lodLevels: LodLevel[] = [
      { lodIndex: 0, color: 0xff0000, maxDistance: 24 },
      { lodIndex: 1, color: 0x00ff00, maxDistance: 400 },
    ];
    const manager = createManager({ blockWidth, lodLevels });
    const chunkSize = blockWidth << 1;

    const plan = manager.update({ x: 0.1, z: chunkSize * 0.5 });
    const hasFinest = plan.requests.some(
      (request) => request.descriptor.lodIndex === 0
    );
    expect(hasFinest).toBe(true);
  });

  it("flushes far coarse chunks when teleporting to a new origin", () => {
    const manager = createManager({ blockWidth: 4, lodLevels: TELEPORT_LODS });
    const startPlan = manager.update({ x: 0, z: 0 });
    const coarseRequests = startPlan.requests.filter(
      (r) => r.descriptor.lodIndex === 2
    );
    expect(coarseRequests.length).toBeGreaterThan(0);
    coarseRequests.forEach((request) => completeRequest(manager, request));

    const teleportPlan = manager.update({ x: 300, z: 0 });
    coarseRequests.forEach((request) => {
      expect(teleportPlan.releases).toContain(request.descriptor.key);
      expect(chunkKeys(teleportPlan)).not.toContain(request.descriptor.key);
    });
  });

  it("cancels pending work from the previous origin after a teleport", () => {
    const manager = createManager({ blockWidth: 4, lodLevels: TELEPORT_LODS });
    const pendingPlan = manager.update({ x: 0, z: 0 });
    const pendingKeys = pendingPlan.requests.map(
      (request) => request.descriptor.key
    );
    expect(pendingKeys.length).toBeGreaterThan(0);

    const teleportPlan = manager.update({ x: 300, z: 150 });
    pendingKeys.forEach((key) => {
      expect(teleportPlan.cancels).toContain(key);
    });
  });

  it("disposes old LOD 2 chunks after a moderate jump", () => {
    const manager = createManager({ blockWidth: 4, lodLevels: TELEPORT_LODS });
    const startPlan = manager.update({ x: 0, z: 0 });
    const coarseRequest = startPlan.requests
      .filter((r) => r.descriptor.lodIndex === 2)
      .sort((a, b) => {
        const aDist = Math.hypot(a.descriptor.chunkX, a.descriptor.chunkZ);
        const bDist = Math.hypot(b.descriptor.chunkX, b.descriptor.chunkZ);
        return aDist - bDist;
      })[0];
    expect(coarseRequest).toBeDefined();
    completeRequest(manager, coarseRequest!);

    const jumpPlan = manager.update({ x: 140, z: 0 });
    expect(jumpPlan.releases).toContain(coarseRequest!.descriptor.key);
  });

  it("splits an active parent chunk and releases it after children finish", () => {
    const blockWidth = 2;
    const manager = createManager({ blockWidth, lodLevels: SPLIT_LODS });
    const chunkSize = blockWidth << 1;
    const farCamera = { x: chunkSize + 2.25, z: chunkSize + 1.5 };
    const nearCamera = { x: chunkSize * 0.25, z: chunkSize * 0.25 };
    const targetParentKey = `1:0:0`;

    const coarsePlan = manager.update(farCamera);
    const parentRequest = coarsePlan.requests.find(
      (r) => r.descriptor.key === targetParentKey
    );
    expect(parentRequest).toBeDefined();

    const parentKey = parentRequest!.descriptor.key;
    completeRequest(manager, parentRequest!);

    const splitPlan = manager.update(nearCamera);
    const expectedChildKeys = childKeysFor(parentRequest!.descriptor);

    expectedChildKeys.forEach((key) => {
      expect(chunkKeys(splitPlan)).toContain(key);
    });
    expect(splitPlan.releases).not.toContain(parentKey);

    const releases = new Set<string>();
    for (const key of expectedChildKeys) {
      const childRequest = splitPlan.requests.find(
        (request) => request.descriptor.key === key
      );
      expect(childRequest).toBeDefined();
      const finalizePlan = completeRequest(manager, childRequest!);
      finalizePlan.releases.forEach((releaseKey) => releases.add(releaseKey));
    }

    expect(releases.has(parentKey)).toBe(true);
  });

  it("merges children back into their parent when distance increases", () => {
    const blockWidth = 2;
    const manager = createManager({ blockWidth, lodLevels: SPLIT_LODS });
    const chunkSize = blockWidth << 1;
    const farCamera = { x: chunkSize + 2.25, z: chunkSize + 1.5 };
    const nearCamera = { x: chunkSize * 0.25, z: chunkSize * 0.25 };
    const targetParentKey = `1:0:0`;

    const coarsePlan = manager.update(farCamera);
    const parentRequest = coarsePlan.requests.find(
      (r) => r.descriptor.key === targetParentKey
    );
    expect(parentRequest).toBeDefined();
    completeRequest(manager, parentRequest!);

    const splitPlan = manager.update(nearCamera);
    const childKeys = childKeysFor(parentRequest!.descriptor);
    const childRequests = splitPlan.requests.filter((request) =>
      childKeys.includes(request.descriptor.key)
    );
    expect(childRequests).toHaveLength(childKeys.length);
    childRequests.forEach((request) => completeRequest(manager, request));

    const mergePlan = manager.update(farCamera);
    expect(chunkKeys(mergePlan)).toContain(parentRequest!.descriptor.key);
    expect(mergePlan.releases.some((key) => childKeys.includes(key))).toBe(
      false
    );

    const parentMergeRequest = mergePlan.requests.find(
      (request) => request.descriptor.key === parentRequest!.descriptor.key
    );
    expect(parentMergeRequest).toBeDefined();

    const completionPlan = completeRequest(manager, parentMergeRequest!);
    childKeys.forEach((key) => {
      expect(completionPlan.releases).toContain(key);
    });
  });
});
