export type TransitionFace =
  | "negativeX"
  | "positiveX"
  | "negativeZ"
  | "positiveZ";

export interface LodLevel {
  lodIndex: number;
  color: number;
  maxDistance?: number;
}

type DerivedLodLevel = LodLevel & { maxDistance: number };

export interface Vector2Like {
  x: number;
  z: number;
}

export interface QuadChunkConfig {
  blockWidth: number;
  cellSize: number;
  lodLevels: LodLevel[];
  lodDistanceMultiplier?: number;
  estimateOriginY: (lodIndex: number, chunkX: number, chunkZ: number) => number;
}

export interface ChunkDescriptor {
  lodIndex: number;
  chunkX: number;
  chunkZ: number;
  key: string;
  color: number;
  transitionFaces: TransitionFace[];
  originY: number;
}

export interface ChunkRequest {
  descriptor: ChunkDescriptor;
  requestId: number;
}

export interface ChunkPlan {
  requests: ChunkRequest[];
  cancels: string[];
  releases: string[];
}

export type ChunkBuildOutcome = "mesh" | "empty";

export interface ChunkBuildCompletion {
  descriptor: ChunkDescriptor;
  requestId: number;
  outcome: ChunkBuildOutcome;
}

interface QuadNode {
  lodIndex: number;
  chunkX: number;
  chunkZ: number;
  forceInclude?: boolean;
}

interface PendingRequest {
  descriptor: ChunkDescriptor;
  requestId: number;
}

export class QuadChunkManager {
  private readonly maxLodIndex: number;
  private readonly lodLookup: Map<number, DerivedLodLevel>;
  private readonly derivedLodLevels: DerivedLodLevel[];
  private readonly lodDistanceMultiplier: number;
  private lastCameraPosition: Vector2Like | null = null;
  private desiredChunkKeys = new Set<string>();
  private nextRequestId = 1;
  private readonly activeChunks = new Map<string, ChunkDescriptor>();
  private readonly pendingRequests = new Map<string, PendingRequest>();
  private readonly retainedParents = new Map<string, Set<string>>();
  private readonly pendingMergeParents = new Map<string, Set<string>>();
  private readonly emptyChunks = new Set<string>();

  constructor(private readonly config: QuadChunkConfig) {
    if (config.lodLevels.length === 0) {
      throw new Error("At least one LOD level is required");
    }
    this.lodDistanceMultiplier = config.lodDistanceMultiplier ?? 1.5;
    this.derivedLodLevels = config.lodLevels
      .map((level) => ({
        ...level,
        maxDistance:
          level.maxDistance ??
          this.chunkWorldSize(level.lodIndex) * this.lodDistanceMultiplier,
      }))
      .sort((a, b) => a.lodIndex - b.lodIndex);
    this.maxLodIndex =
      this.derivedLodLevels[this.derivedLodLevels.length - 1].lodIndex;
    this.lodLookup = new Map(
      this.derivedLodLevels.map((level) => [level.lodIndex, level])
    );
  }

  update(cameraPosition: Vector2Like): ChunkPlan {
    const previousPosition = this.lastCameraPosition;
    const teleported = previousPosition
      ? this.distanceBetween(previousPosition, cameraPosition) >
        this.teleportThreshold()
      : false;
    this.lastCameraPosition = { x: cameraPosition.x, z: cameraPosition.z };
    const desired = this.collectDesiredChunks(cameraPosition);
    this.pruneDistantChunks(desired, cameraPosition, previousPosition);
    const plan = this.reconcile(desired, teleported);
    this.desiredChunkKeys = new Set(desired.keys());
    return plan;
  }

  isRequestPending(key: string, requestId: number): boolean {
    const pending = this.pendingRequests.get(key);
    return Boolean(pending && pending.requestId === requestId);
  }

  finalizeRequest(completion: ChunkBuildCompletion): ChunkPlan {
    const plan = this.createPlan();
    const { descriptor, requestId, outcome } = completion;
    const pending = this.pendingRequests.get(descriptor.key);
    if (!pending || pending.requestId !== requestId) {
      return plan;
    }

    this.pendingRequests.delete(descriptor.key);

    if (!this.desiredChunkKeys.has(descriptor.key)) {
      if (outcome === "mesh") {
        plan.releases.push(descriptor.key);
      }
      return plan;
    }

    if (outcome === "empty") {
      this.emptyChunks.add(descriptor.key);
    } else {
      this.activeChunks.set(descriptor.key, descriptor);
    }

    this.resolveParentRetention(descriptor, plan);
    this.completePendingMerge(descriptor.key, plan);
    return plan;
  }

  getPendingDescriptor(key: string): ChunkDescriptor | undefined {
    return this.pendingRequests.get(key)?.descriptor;
  }

  private reconcile(
    desired: Map<string, ChunkDescriptor>,
    teleported: boolean
  ): ChunkPlan {
    const plan = this.createPlan();

    if (teleported) {
      this.flushAllChunks(plan);
    }

    for (const [key, record] of this.activeChunks.entries()) {
      if (desired.has(key)) {
        this.retainedParents.delete(key);
        continue;
      }

      const childDescriptors = this.collectChildDescriptors(record, desired);
      if (childDescriptors.length === 4) {
        this.scheduleSplit(record, childDescriptors, plan);
        continue;
      } else {
        this.retainedParents.delete(key);
      }

      const parentDescriptor = this.findParentDescriptor(record, desired);
      if (parentDescriptor) {
        this.scheduleMerge(parentDescriptor, plan);
        continue;
      }

      this.disposeChunk(key, plan);
    }

    for (const pendingKey of Array.from(this.pendingRequests.keys())) {
      if (!desired.has(pendingKey)) {
        this.cancelPendingRequest(pendingKey, plan);
      }
    }

    for (const descriptor of desired.values()) {
      if (
        this.activeChunks.has(descriptor.key) ||
        this.pendingRequests.has(descriptor.key) ||
        this.emptyChunks.has(descriptor.key)
      ) {
        continue;
      }
      this.requestChunk(descriptor, plan);
    }

    return plan;
  }

  private collectDesiredChunks(
    cameraPosition: Vector2Like
  ): Map<string, ChunkDescriptor> {
    const desired = new Map<string, ChunkDescriptor>();
    const queue = this.buildInitialNodes(cameraPosition);

    while (queue.length) {
      const node = queue.pop()!;
      const level = this.lodLookup.get(node.lodIndex);
      if (!level) {
        continue;
      }

      const distance = this.chunkDistance(node, cameraPosition);
      const radius = this.chunkDiagonalRadius(node.lodIndex);
      const culled =
        !node.forceInclude && distance > level.maxDistance + radius;
      if (culled) {
        continue;
      }

      if (this.shouldSubdivide(node, distance)) {
        this.subdivideNode(node).forEach((child) =>
          queue.push({ ...child, forceInclude: true })
        );
        continue;
      }

      const key = this.chunkKey(node.lodIndex, node.chunkX, node.chunkZ);
      if (!desired.has(key)) {
        desired.set(
          key,
          this.createDescriptor(node.lodIndex, node.chunkX, node.chunkZ)
        );
      }
    }

    return desired;
  }

  private pruneDistantChunks(
    desired: Map<string, ChunkDescriptor>,
    cameraPosition: Vector2Like,
    previousPosition: Vector2Like | null
  ): void {
    if (!previousPosition || desired.size === 0) {
      return;
    }

    const travelVector = {
      x: cameraPosition.x - previousPosition.x,
      z: cameraPosition.z - previousPosition.z,
    };
    const travelDistance = Math.hypot(travelVector.x, travelVector.z);
    if (travelDistance === 0) {
      return;
    }

    const coarseSize = this.chunkWorldSize(this.maxLodIndex);
    const farthestLevel =
      this.derivedLodLevels[this.derivedLodLevels.length - 1];
    const movementThreshold = Math.max(
      coarseSize * 2,
      farthestLevel.maxDistance * 0.5
    );
    if (travelDistance < movementThreshold) {
      return;
    }

    const level = this.lodLookup.get(this.maxLodIndex);
    if (!level || level.maxDistance < coarseSize) {
      return;
    }

    const dirX = travelVector.x / travelDistance;
    const dirZ = travelVector.z / travelDistance;
    // Drop coarse chunks that now sit well behind the camera after a large move.
    const coarseDescriptors: ChunkDescriptor[] = [];
    for (const descriptor of desired.values()) {
      if (descriptor.lodIndex === this.maxLodIndex) {
        coarseDescriptors.push(descriptor);
      }
    }
    for (const descriptor of this.activeChunks.values()) {
      if (
        descriptor.lodIndex === this.maxLodIndex &&
        !desired.has(descriptor.key)
      ) {
        coarseDescriptors.push(descriptor);
      }
    }

    for (const descriptor of coarseDescriptors) {
      const chunkSize = this.chunkWorldSize(descriptor.lodIndex);
      const centerX = (descriptor.chunkX + 0.5) * chunkSize;
      const centerZ = (descriptor.chunkZ + 0.5) * chunkSize;
      const offsetX = centerX - cameraPosition.x;
      const offsetZ = centerZ - cameraPosition.z;
      const projection = offsetX * dirX + offsetZ * dirZ;
      if (projection >= -chunkSize) {
        continue;
      }
      desired.delete(descriptor.key);
      this.removeDescendantDesiredChunks(desired, descriptor);
    }
  }

  private removeDescendantDesiredChunks(
    desired: Map<string, ChunkDescriptor>,
    parent: ChunkDescriptor
  ): void {
    for (const [childKey, childDescriptor] of Array.from(desired.entries())) {
      if (!childDescriptor || childDescriptor.lodIndex >= parent.lodIndex) {
        continue;
      }
      if (this.isDescendantChunk(parent, childDescriptor)) {
        desired.delete(childKey);
      }
    }
  }

  private isDescendantChunk(
    parent: ChunkDescriptor,
    child: ChunkDescriptor
  ): boolean {
    if (child.lodIndex >= parent.lodIndex) {
      return false;
    }
    const parentSize = this.chunkWorldSize(parent.lodIndex);
    const childSize = this.chunkWorldSize(child.lodIndex);
    const parentMinX = parent.chunkX * parentSize;
    const parentMaxX = parentMinX + parentSize;
    const parentMinZ = parent.chunkZ * parentSize;
    const parentMaxZ = parentMinZ + parentSize;
    const childMinX = child.chunkX * childSize;
    const childMaxX = childMinX + childSize;
    const childMinZ = child.chunkZ * childSize;
    const childMaxZ = childMinZ + childSize;
    return (
      childMinX >= parentMinX &&
      childMaxX <= parentMaxX &&
      childMinZ >= parentMinZ &&
      childMaxZ <= parentMaxZ
    );
  }

  private buildInitialNodes(cameraPosition: Vector2Like): QuadNode[] {
    const coarseSize = this.chunkWorldSize(this.maxLodIndex);
    const farthest =
      this.derivedLodLevels[this.derivedLodLevels.length - 1];
    const radiusChunks = Math.ceil(farthest.maxDistance / coarseSize) + 2;
    const baseX = Math.floor(cameraPosition.x / coarseSize);
    const baseZ = Math.floor(cameraPosition.z / coarseSize);
    const nodes = this.generateSpiralOrder(baseX, baseZ, radiusChunks).map(
      ({ chunkX, chunkZ }) => ({
        lodIndex: this.maxLodIndex,
        chunkX,
        chunkZ,
      })
    );

    return nodes.reverse();
  }

  private generateSpiralOrder(
    centerX: number,
    centerZ: number,
    radius: number
  ): Array<{ chunkX: number; chunkZ: number }> {
    const limit = Math.max(0, Math.floor(radius));
    const coords: Array<{ chunkX: number; chunkZ: number }> = [
      { chunkX: centerX, chunkZ: centerZ },
    ];

    if (limit === 0) {
      return coords;
    }

    const totalCells = (limit * 2 + 1) ** 2;
    let stepLength = 1;
    let x = centerX;
    let z = centerZ;
    const directions: Array<{ dx: number; dz: number }> = [
      { dx: 1, dz: 0 },
      { dx: 0, dz: 1 },
      { dx: -1, dz: 0 },
      { dx: 0, dz: -1 },
    ];
    let directionIndex = 0;

    while (coords.length < totalCells) {
      for (let repeat = 0; repeat < 2; repeat++) {
        const { dx, dz } = directions[directionIndex % directions.length];
        for (
          let step = 0;
          step < stepLength && coords.length < totalCells;
          step++
        ) {
          x += dx;
          z += dz;
          if (
            Math.abs(x - centerX) <= limit &&
            Math.abs(z - centerZ) <= limit
          ) {
            coords.push({ chunkX: x, chunkZ: z });
          }
        }
        directionIndex++;
      }
      stepLength++;
    }

    return coords;
  }

  private shouldSubdivide(node: QuadNode, distance: number): boolean {
    if (node.lodIndex === 0) {
      return false;
    }
    const desiredLod = this.desiredLodForDistance(distance);
    return desiredLod < node.lodIndex;
  }

  private subdivideNode(node: QuadNode): QuadNode[] {
    const childLod = node.lodIndex - 1;
    const baseX = node.chunkX * 2;
    const baseZ = node.chunkZ * 2;
    return [
      { lodIndex: childLod, chunkX: baseX, chunkZ: baseZ },
      { lodIndex: childLod, chunkX: baseX + 1, chunkZ: baseZ },
      { lodIndex: childLod, chunkX: baseX, chunkZ: baseZ + 1 },
      { lodIndex: childLod, chunkX: baseX + 1, chunkZ: baseZ + 1 },
    ];
  }

  private chunkDistance(node: QuadNode, point: Vector2Like): number {
    return this.chunkDistanceToPoint(
      node.lodIndex,
      node.chunkX,
      node.chunkZ,
      point
    );
  }

  private desiredLodForDistance(distance: number): number {
    for (const level of this.derivedLodLevels) {
      if (distance <= level.maxDistance) {
        return level.lodIndex;
      }
    }
    return this.maxLodIndex;
  }

  private requestChunk(descriptor: ChunkDescriptor, plan: ChunkPlan): void {
    const existing = this.pendingRequests.get(descriptor.key);
    if (existing) {
      return;
    }
    const requestId = this.nextRequestId++;
    const clone: ChunkDescriptor = {
      lodIndex: descriptor.lodIndex,
      chunkX: descriptor.chunkX,
      chunkZ: descriptor.chunkZ,
      key: descriptor.key,
      color: descriptor.color,
      transitionFaces: [...descriptor.transitionFaces],
      originY: descriptor.originY,
    };
    this.pendingRequests.set(descriptor.key, { descriptor: clone, requestId });
    plan.requests.push({ descriptor: clone, requestId });
  }

  private cancelPendingRequest(key: string, plan: ChunkPlan): void {
    if (!this.pendingRequests.has(key)) {
      return;
    }
    this.pendingRequests.delete(key);
    this.pendingMergeParents.delete(key);
    plan.cancels.push(key);
  }

  private disposeChunk(key: string, plan: ChunkPlan): void {
    this.retainedParents.delete(key);
    this.pendingMergeParents.delete(key);
    this.emptyChunks.delete(key);
    if (this.activeChunks.delete(key)) {
      plan.releases.push(key);
    }
  }

  private scheduleSplit(
    parent: ChunkDescriptor,
    childDescriptors: ChunkDescriptor[],
    plan: ChunkPlan
  ): void {
    const parentKey = parent.key;
    const missingChildren = new Set<string>();

    childDescriptors.forEach((descriptor) => {
      this.ensureChunkRequested(descriptor, plan);
      if (
        !this.activeChunks.has(descriptor.key) &&
        !this.emptyChunks.has(descriptor.key)
      ) {
        missingChildren.add(descriptor.key);
      }
    });

    if (missingChildren.size === 0) {
      this.retainedParents.delete(parentKey);
      this.disposeChunk(parentKey, plan);
      return;
    }

    this.retainedParents.set(parentKey, missingChildren);
  }

  private scheduleMerge(
    parentDescriptor: ChunkDescriptor,
    plan: ChunkPlan
  ): void {
    this.ensureChunkRequested(parentDescriptor, plan);
    const parentKey = parentDescriptor.key;
    const childKeys = this.computeChildKeys(
      parentDescriptor.lodIndex,
      parentDescriptor.chunkX,
      parentDescriptor.chunkZ
    ).filter((childKey) => this.activeChunks.has(childKey));

    if (childKeys.length === 0) {
      this.pendingMergeParents.delete(parentKey);
      return;
    }

    if (this.emptyChunks.has(parentKey)) {
      childKeys.forEach((childKey) => this.disposeChunk(childKey, plan));
      this.pendingMergeParents.delete(parentKey);
      return;
    }

    if (this.activeChunks.has(parentKey)) {
      childKeys.forEach((childKey) => this.disposeChunk(childKey, plan));
      this.pendingMergeParents.delete(parentKey);
      return;
    }

    this.pendingMergeParents.set(parentKey, new Set(childKeys));
  }

  private resolveParentRetention(
    descriptor: ChunkDescriptor,
    plan: ChunkPlan
  ): void {
    const parentKey = this.getParentKey(
      descriptor.lodIndex,
      descriptor.chunkX,
      descriptor.chunkZ
    );
    if (!parentKey) {
      return;
    }
    const pending = this.retainedParents.get(parentKey);
    if (!pending) {
      return;
    }
    pending.delete(descriptor.key);
    if (pending.size === 0) {
      this.retainedParents.delete(parentKey);
      this.disposeChunk(parentKey, plan);
    }
  }

  private completePendingMerge(parentKey: string, plan: ChunkPlan): void {
    const children = this.pendingMergeParents.get(parentKey);
    if (!children) {
      return;
    }
    children.forEach((childKey) => this.disposeChunk(childKey, plan));
    this.pendingMergeParents.delete(parentKey);
  }

  private ensureChunkRequested(
    descriptor: ChunkDescriptor,
    plan: ChunkPlan
  ): void {
    if (
      this.activeChunks.has(descriptor.key) ||
      this.pendingRequests.has(descriptor.key) ||
      this.emptyChunks.has(descriptor.key)
    ) {
      return;
    }
    this.requestChunk(descriptor, plan);
  }

  private collectChildDescriptors(
    record: ChunkDescriptor,
    desired: Map<string, ChunkDescriptor>
  ): ChunkDescriptor[] {
    if (record.lodIndex === 0) {
      return [];
    }
    const childLod = record.lodIndex - 1;
    const childKeys = this.computeChildKeys(
      record.lodIndex,
      record.chunkX,
      record.chunkZ
    );
    return childKeys
      .map((childKey) => desired.get(childKey))
      .filter((descriptor): descriptor is ChunkDescriptor =>
        Boolean(descriptor && descriptor.lodIndex === childLod)
      )
      .sort((a, b) => a.key.localeCompare(b.key));
  }

  private findParentDescriptor(
    record: ChunkDescriptor,
    desired: Map<string, ChunkDescriptor>
  ): ChunkDescriptor | null {
    if (record.lodIndex >= this.maxLodIndex) {
      return null;
    }
    const parentKey = this.getParentKey(
      record.lodIndex,
      record.chunkX,
      record.chunkZ
    );
    if (!parentKey) {
      return null;
    }
    const descriptor = desired.get(parentKey);
    if (descriptor && descriptor.lodIndex === record.lodIndex + 1) {
      return descriptor;
    }
    return null;
  }

  private getParentKey(
    lodIndex: number,
    chunkX: number,
    chunkZ: number
  ): string | null {
    if (lodIndex >= this.maxLodIndex) {
      return null;
    }
    const parentLod = lodIndex + 1;
    const parentX = Math.floor(chunkX / 2);
    const parentZ = Math.floor(chunkZ / 2);
    return this.chunkKey(parentLod, parentX, parentZ);
  }

  private computeChildKeys(
    lodIndex: number,
    chunkX: number,
    chunkZ: number
  ): string[] {
    if (lodIndex === 0) {
      return [];
    }
    const childLod = lodIndex - 1;
    const baseX = chunkX * 2;
    const baseZ = chunkZ * 2;
    return [
      this.chunkKey(childLod, baseX, baseZ),
      this.chunkKey(childLod, baseX + 1, baseZ),
      this.chunkKey(childLod, baseX, baseZ + 1),
      this.chunkKey(childLod, baseX + 1, baseZ + 1),
    ];
  }

  private chunkWorldSize(lodIndex: number): number {
    return this.config.cellSize * (this.config.blockWidth << lodIndex);
  }

  private chunkDiagonalRadius(lodIndex: number): number {
    return (this.chunkWorldSize(lodIndex) * Math.SQRT2) / 2;
  }

  private chunkKey(lodIndex: number, chunkX: number, chunkZ: number): string {
    return `${lodIndex}:${chunkX}:${chunkZ}`;
  }

  private createDescriptor(
    lodIndex: number,
    chunkX: number,
    chunkZ: number
  ): ChunkDescriptor {
    const level = this.lodLookup.get(lodIndex);
    if (!level) {
      throw new Error(`Missing LOD configuration for level ${lodIndex}`);
    }
    return {
      lodIndex,
      chunkX,
      chunkZ,
      key: this.chunkKey(lodIndex, chunkX, chunkZ),
      color: level.color,
      transitionFaces: [],
      originY: this.config.estimateOriginY(lodIndex, chunkX, chunkZ),
    };
  }

  private chunkDistanceToPoint(
    lodIndex: number,
    chunkX: number,
    chunkZ: number,
    point: Vector2Like
  ): number {
    const size = this.chunkWorldSize(lodIndex);
    const minX = chunkX * size;
    const maxX = minX + size;
    const minZ = chunkZ * size;
    const maxZ = minZ + size;
    const dx =
      point.x < minX ? minX - point.x : point.x > maxX ? point.x - maxX : 0;
    const dz =
      point.z < minZ ? minZ - point.z : point.z > maxZ ? point.z - maxZ : 0;
    return Math.hypot(dx, dz);
  }

  private flushAllChunks(plan: ChunkPlan): void {
    for (const key of Array.from(this.activeChunks.keys())) {
      this.disposeChunk(key, plan);
    }
    for (const key of Array.from(this.pendingRequests.keys())) {
      this.cancelPendingRequest(key, plan);
    }
    this.retainedParents.clear();
    this.pendingMergeParents.clear();
  }

  private distanceBetween(a: Vector2Like, b: Vector2Like): number {
    return Math.hypot(a.x - b.x, a.z - b.z);
  }

  private teleportThreshold(): number {
    const farthest =
      this.derivedLodLevels[this.derivedLodLevels.length - 1];
    const coarseSpan = this.chunkWorldSize(this.maxLodIndex) * 4;
    return Math.max(farthest.maxDistance * 2, coarseSpan);
  }

  private createPlan(): ChunkPlan {
    return { requests: [], cancels: [], releases: [] };
  }
}
