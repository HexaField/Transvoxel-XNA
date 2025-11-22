export type TransitionFace =
  | "negativeX"
  | "positiveX"
  | "negativeY"
  | "positiveY"
  | "negativeZ"
  | "positiveZ";

export interface LodLevel {
  lodIndex: number;
  color: number;
  maxDistance?: number;
}

type DerivedLodLevel = LodLevel & { maxDistance: number };

export interface Vector3Like {
  x: number;
  y: number;
  z: number;
}

export interface OctreeChunkConfig {
  blockWidth: number;
  cellSize: number;
  lodLevels: LodLevel[];
  worldMinY: number;
  worldMaxY: number;
  lodDistanceMultiplier?: number;
}

export interface ChunkDescriptor {
  lodIndex: number;
  chunkX: number;
  chunkY: number;
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

interface OctreeNode {
  lodIndex: number;
  chunkX: number;
  chunkY: number;
  chunkZ: number;
  forceInclude?: boolean;
}

interface PendingRequest {
  descriptor: ChunkDescriptor;
  requestId: number;
}

interface ChunkBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  minZ: number;
  maxZ: number;
}

export class OctreeChunkManager {
  private readonly maxLodIndex: number;
  private readonly lodLookup: Map<number, DerivedLodLevel>;
  private readonly derivedLodLevels: DerivedLodLevel[];
  private readonly lodDistanceMultiplier: number;
  private readonly worldMinY: number;
  private readonly worldMaxY: number;
  private readonly coarseMinChunkY: number;
  private readonly coarseMaxChunkY: number;
  private lastCameraPosition: Vector3Like | null = null;
  private desiredChunkKeys = new Set<string>();
  private nextRequestId = 1;
  private readonly activeChunks = new Map<string, ChunkDescriptor>();
  private readonly pendingRequests = new Map<string, PendingRequest>();
  private readonly retainedParents = new Map<string, Set<string>>();
  private readonly pendingMergeParents = new Map<string, Set<string>>();
  private readonly emptyChunks = new Set<string>();

  constructor(private readonly config: OctreeChunkConfig) {
    if (config.lodLevels.length === 0) {
      throw new Error("At least one LOD level is required");
    }
    if (config.worldMaxY <= config.worldMinY) {
      throw new RangeError("worldMaxY must be greater than worldMinY");
    }
    this.worldMinY = config.worldMinY;
    this.worldMaxY = config.worldMaxY;
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
    const coarseSize = this.chunkWorldSize(this.maxLodIndex);
    this.coarseMinChunkY = Math.floor(this.worldMinY / coarseSize);
    this.coarseMaxChunkY = Math.floor((this.worldMaxY - 1) / coarseSize);
  }

  update(cameraPosition: Vector3Like): ChunkPlan {
    const previousPosition = this.lastCameraPosition;
    const teleported = previousPosition
      ? this.distanceBetween(previousPosition, cameraPosition) >
        this.teleportThreshold()
      : false;
    this.lastCameraPosition = {
      x: cameraPosition.x,
      y: cameraPosition.y,
      z: cameraPosition.z,
    };
    const desired = this.collectDesiredChunks(cameraPosition);
    this.pruneDistantChunks(desired, cameraPosition, previousPosition);
    const plan = this.reconcile(desired, teleported);
    this.desiredChunkKeys = new Set(desired.keys());
    return this.finalizePlan(plan);
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
      return this.finalizePlan(plan);
    }

    this.pendingRequests.delete(descriptor.key);

    if (!this.desiredChunkKeys.has(descriptor.key)) {
      if (outcome === "mesh") {
        plan.releases.push(descriptor.key);
      }
      return this.finalizePlan(plan);
    }

    if (outcome === "empty") {
      this.emptyChunks.add(descriptor.key);
    } else {
      this.activeChunks.set(descriptor.key, descriptor);
    }

    this.resolveParentRetention(descriptor, plan);
    this.completePendingMerge(descriptor.key, plan);
    return this.finalizePlan(plan);
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
      const desiredDescriptor = desired.get(key);
      if (desiredDescriptor) {
        if (
          this.transitionFacesChanged(
            record.transitionFaces,
            desiredDescriptor.transitionFaces
          )
        ) {
          this.disposeChunk(key, plan);
          this.requestChunk(desiredDescriptor, plan);
          continue;
        }
        this.retainedParents.delete(key);
        continue;
      }

      const childDescriptors = this.collectChildDescriptors(record, desired);
      if (childDescriptors.length === 8) {
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
    cameraPosition: Vector3Like
  ): Map<string, ChunkDescriptor> {
    const desired = new Map<string, ChunkDescriptor>();
    const queue = this.buildInitialNodes(cameraPosition);

    while (queue.length) {
      const node = queue.pop()!;
      if (!this.isNodeWithinWorld(node)) {
        continue;
      }
      const level = this.lodLookup.get(node.lodIndex);
      if (!level) {
        continue;
      }

      const distance = this.chunkDistance(node, cameraPosition);
      const radius = this.chunkDiagonalRadius(node.lodIndex);
      const culled =
        !node.forceInclude &&
        node.lodIndex !== this.maxLodIndex &&
        distance > level.maxDistance + radius;
      if (culled) {
        continue;
      }

      if (this.shouldSubdivide(node, distance)) {
        this.subdivideNode(node).forEach((child) =>
          queue.push({ ...child, forceInclude: true })
        );
        continue;
      }

      const key = this.chunkKey(
        node.lodIndex,
        node.chunkX,
        node.chunkY,
        node.chunkZ
      );
      if (!desired.has(key)) {
        desired.set(key, this.createDescriptor(node));
      }
    }

    this.assignTransitionFaces(desired);
    return desired;
  }

  private pruneDistantChunks(
    desired: Map<string, ChunkDescriptor>,
    cameraPosition: Vector3Like,
    previousPosition: Vector3Like | null
  ): void {
    if (!previousPosition || desired.size === 0) {
      return;
    }

    const travelVector = {
      x: cameraPosition.x - previousPosition.x,
      y: cameraPosition.y - previousPosition.y,
      z: cameraPosition.z - previousPosition.z,
    };
    const travelDistance = Math.hypot(
      travelVector.x,
      travelVector.y,
      travelVector.z
    );
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
    const dirY = travelVector.y / travelDistance;
    const dirZ = travelVector.z / travelDistance;
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
      const centerY = (descriptor.chunkY + 0.5) * chunkSize;
      const centerZ = (descriptor.chunkZ + 0.5) * chunkSize;
      const offsetX = centerX - cameraPosition.x;
      const offsetY = centerY - cameraPosition.y;
      const offsetZ = centerZ - cameraPosition.z;
      const projection = offsetX * dirX + offsetY * dirY + offsetZ * dirZ;
      if (projection >= -chunkSize) {
        continue;
      }
      if (this.hasActiveDescendants(descriptor)) {
        continue;
      }
      desired.delete(descriptor.key);
      this.removeDescendantDesiredChunks(desired, descriptor);
    }
  }

  private hasActiveDescendants(descriptor: ChunkDescriptor): boolean {
    for (const record of this.activeChunks.values()) {
      if (
        record.lodIndex < descriptor.lodIndex &&
        this.isDescendantChunk(descriptor, record)
      ) {
        return true;
      }
    }
    for (const pending of this.pendingRequests.values()) {
      const child = pending.descriptor;
      if (
        child.lodIndex < descriptor.lodIndex &&
        this.isDescendantChunk(descriptor, child)
      ) {
        return true;
      }
    }
    return false;
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
    const parentMinY = parent.chunkY * parentSize;
    const parentMaxY = parentMinY + parentSize;
    const parentMinZ = parent.chunkZ * parentSize;
    const parentMaxZ = parentMinZ + parentSize;
    const childMinX = child.chunkX * childSize;
    const childMaxX = childMinX + childSize;
    const childMinY = child.chunkY * childSize;
    const childMaxY = childMinY + childSize;
    const childMinZ = child.chunkZ * childSize;
    const childMaxZ = childMinZ + childSize;
    return (
      childMinX >= parentMinX &&
      childMaxX <= parentMaxX &&
      childMinY >= parentMinY &&
      childMaxY <= parentMaxY &&
      childMinZ >= parentMinZ &&
      childMaxZ <= parentMaxZ
    );
  }

  private buildInitialNodes(cameraPosition: Vector3Like): OctreeNode[] {
    const coarseSize = this.chunkWorldSize(this.maxLodIndex);
    const farthest = this.derivedLodLevels[this.derivedLodLevels.length - 1];
    const radiusChunks = Math.ceil(farthest.maxDistance / coarseSize) + 2;
    const baseX = Math.floor(cameraPosition.x / coarseSize);
    const baseZ = Math.floor(cameraPosition.z / coarseSize);
    const horizontalOrder = this.generateSpiralOrder(
      baseX,
      baseZ,
      radiusChunks
    );
    const hasOrigin = horizontalOrder.some(
      (coords) => coords.chunkX === 0 && coords.chunkZ === 0
    );
    if (!hasOrigin) {
      horizontalOrder.push({ chunkX: 0, chunkZ: 0 });
    }
    const nodes: OctreeNode[] = [];
    for (const coords of horizontalOrder) {
      const isCenter = coords.chunkX === baseX && coords.chunkZ === baseZ;
      const isOrigin = coords.chunkX === 0 && coords.chunkZ === 0;
      for (
        let chunkY = this.coarseMinChunkY;
        chunkY <= this.coarseMaxChunkY;
        chunkY++
      ) {
        nodes.push({
          lodIndex: this.maxLodIndex,
          chunkX: coords.chunkX,
          chunkY,
          chunkZ: coords.chunkZ,
          forceInclude: isCenter || isOrigin,
        });
      }
    }
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

  private shouldSubdivide(node: OctreeNode, distance: number): boolean {
    if (node.lodIndex === 0) {
      return false;
    }
    const desiredLod = this.desiredLodForDistance(distance);
    return desiredLod < node.lodIndex;
  }

  private subdivideNode(node: OctreeNode): OctreeNode[] {
    const childLod = node.lodIndex - 1;
    const baseX = node.chunkX * 2;
    const baseY = node.chunkY * 2;
    const baseZ = node.chunkZ * 2;
    const children: OctreeNode[] = [];
    for (let dz = 0; dz < 2; dz++) {
      for (let dy = 0; dy < 2; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          const child: OctreeNode = {
            lodIndex: childLod,
            chunkX: baseX + dx,
            chunkY: baseY + dy,
            chunkZ: baseZ + dz,
          };
          if (this.isNodeWithinWorld(child)) {
            children.push(child);
          }
        }
      }
    }
    return children;
  }

  private chunkDistance(node: OctreeNode, point: Vector3Like): number {
    return this.chunkDistanceToPoint(
      node.lodIndex,
      node.chunkX,
      node.chunkY,
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
    if (this.pendingRequests.has(descriptor.key)) {
      return;
    }
    console.debug(
      "[OctreeChunkManager] request",
      descriptor.key,
      descriptor.lodIndex,
      descriptor.chunkX,
      descriptor.chunkY,
      descriptor.chunkZ
    );
    const requestId = this.nextRequestId++;
    const clone: ChunkDescriptor = {
      lodIndex: descriptor.lodIndex,
      chunkX: descriptor.chunkX,
      chunkY: descriptor.chunkY,
      chunkZ: descriptor.chunkZ,
      key: descriptor.key,
      color: descriptor.color,
      transitionFaces: [...descriptor.transitionFaces],
      originY: descriptor.originY,
    };
    this.pendingRequests.set(descriptor.key, { descriptor: clone, requestId });
    plan.requests.push({ descriptor: clone, requestId });
  }

  private finalizePlan(plan: ChunkPlan): ChunkPlan {
    plan.requests.sort((a, b) =>
      this.compareDescriptors(a.descriptor, b.descriptor)
    );
    plan.cancels.sort();
    plan.releases.sort();
    return plan;
  }

  private compareDescriptors(a: ChunkDescriptor, b: ChunkDescriptor): number {
    if (a.lodIndex !== b.lodIndex) {
      return b.lodIndex - a.lodIndex;
    }
    const aPriority = this.descriptorPriority(a);
    const bPriority = this.descriptorPriority(b);
    if (aPriority !== bPriority) {
      return aPriority - bPriority;
    }
    return a.key.localeCompare(b.key);
  }

  private descriptorPriority(descriptor: ChunkDescriptor): number {
    return (
      Math.abs(descriptor.chunkX) +
      Math.abs(descriptor.chunkY) +
      Math.abs(descriptor.chunkZ)
    );
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
    console.debug(
      "[OctreeChunkManager] split",
      parent.key,
      childDescriptors.map((child) => child.key)
    );
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
    const childKeys = this.computeChildKeys(parentDescriptor).filter(
      (childKey) => this.activeChunks.has(childKey)
    );
    console.debug("[OctreeChunkManager] schedule merge", parentKey, childKeys);

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
      descriptor.chunkY,
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
    console.debug(
      "[OctreeChunkManager] merge complete",
      parentKey,
      Array.from(children)
    );
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
    const childKeys = this.computeChildKeys(record);
    return childKeys
      .map((childKey) => desired.get(childKey))
      .filter((descriptor): descriptor is ChunkDescriptor =>
        Boolean(descriptor && descriptor.lodIndex === record.lodIndex - 1)
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
      record.chunkY,
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
    chunkY: number,
    chunkZ: number
  ): string | null {
    if (lodIndex >= this.maxLodIndex) {
      return null;
    }
    const parentLod = lodIndex + 1;
    const parentX = Math.floor(chunkX / 2);
    const parentY = Math.floor(chunkY / 2);
    const parentZ = Math.floor(chunkZ / 2);
    return this.chunkKey(parentLod, parentX, parentY, parentZ);
  }

  private computeChildKeys(record: ChunkDescriptor): string[] {
    if (record.lodIndex === 0) {
      return [];
    }
    const childLod = record.lodIndex - 1;
    const baseX = record.chunkX * 2;
    const baseY = record.chunkY * 2;
    const baseZ = record.chunkZ * 2;
    const keys: string[] = [];
    for (let dz = 0; dz < 2; dz++) {
      for (let dy = 0; dy < 2; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          keys.push(
            this.chunkKey(childLod, baseX + dx, baseY + dy, baseZ + dz)
          );
        }
      }
    }
    return keys;
  }

  private chunkWorldSize(lodIndex: number): number {
    return this.config.cellSize * (this.config.blockWidth << lodIndex);
  }

  private chunkDiagonalRadius(lodIndex: number): number {
    return (this.chunkWorldSize(lodIndex) * Math.sqrt(3)) / 2;
  }

  private assignTransitionFaces(desired: Map<string, ChunkDescriptor>): void {
    if (desired.size === 0) {
      return;
    }

    const boundsByKey = new Map<string, ChunkBounds>();
    const levelBuckets = new Map<number, ChunkBounds[]>();

    desired.forEach((descriptor) => {
      const bounds = this.computeDescriptorBounds(descriptor);
      boundsByKey.set(descriptor.key, bounds);
      const bucket = levelBuckets.get(descriptor.lodIndex);
      if (bucket) {
        bucket.push(bounds);
      } else {
        levelBuckets.set(descriptor.lodIndex, [bounds]);
      }
    });

    desired.forEach((descriptor) => {
      descriptor.transitionFaces = this.computeTransitionFaces(
        descriptor,
        boundsByKey,
        levelBuckets
      );
    });
  }

  private computeTransitionFaces(
    descriptor: ChunkDescriptor,
    boundsByKey: Map<string, ChunkBounds>,
    levelBuckets: Map<number, ChunkBounds[]>
  ): TransitionFace[] {
    if (descriptor.lodIndex === 0) {
      return [];
    }

    const bounds = boundsByKey.get(descriptor.key);
    if (!bounds) {
      return [];
    }

    const faces = new Set<TransitionFace>();
    for (let level = descriptor.lodIndex - 1; level >= 0; level--) {
      const bucket = levelBuckets.get(level);
      if (!bucket) {
        continue;
      }
      for (const neighbor of bucket) {
        this.evaluateTransitionAdjacency(bounds, neighbor, faces);
        if (faces.size === 6) {
          return Array.from(faces).sort();
        }
      }
    }
    return Array.from(faces).sort();
  }

  private evaluateTransitionAdjacency(
    coarse: ChunkBounds,
    neighbor: ChunkBounds,
    faces: Set<TransitionFace>
  ): void {
    if (
      neighbor.maxX === coarse.minX &&
      this.rangesOverlap(
        neighbor.minY,
        neighbor.maxY,
        coarse.minY,
        coarse.maxY
      ) &&
      this.rangesOverlap(neighbor.minZ, neighbor.maxZ, coarse.minZ, coarse.maxZ)
    ) {
      faces.add("negativeX");
    }
    if (
      neighbor.minX === coarse.maxX &&
      this.rangesOverlap(
        neighbor.minY,
        neighbor.maxY,
        coarse.minY,
        coarse.maxY
      ) &&
      this.rangesOverlap(neighbor.minZ, neighbor.maxZ, coarse.minZ, coarse.maxZ)
    ) {
      faces.add("positiveX");
    }
    if (
      neighbor.maxY === coarse.minY &&
      this.rangesOverlap(
        neighbor.minX,
        neighbor.maxX,
        coarse.minX,
        coarse.maxX
      ) &&
      this.rangesOverlap(neighbor.minZ, neighbor.maxZ, coarse.minZ, coarse.maxZ)
    ) {
      faces.add("negativeY");
    }
    if (
      neighbor.minY === coarse.maxY &&
      this.rangesOverlap(
        neighbor.minX,
        neighbor.maxX,
        coarse.minX,
        coarse.maxX
      ) &&
      this.rangesOverlap(neighbor.minZ, neighbor.maxZ, coarse.minZ, coarse.maxZ)
    ) {
      faces.add("positiveY");
    }
    if (
      neighbor.maxZ === coarse.minZ &&
      this.rangesOverlap(
        neighbor.minX,
        neighbor.maxX,
        coarse.minX,
        coarse.maxX
      ) &&
      this.rangesOverlap(neighbor.minY, neighbor.maxY, coarse.minY, coarse.maxY)
    ) {
      faces.add("negativeZ");
    }
    if (
      neighbor.minZ === coarse.maxZ &&
      this.rangesOverlap(
        neighbor.minX,
        neighbor.maxX,
        coarse.minX,
        coarse.maxX
      ) &&
      this.rangesOverlap(neighbor.minY, neighbor.maxY, coarse.minY, coarse.maxY)
    ) {
      faces.add("positiveZ");
    }
  }

  private rangesOverlap(
    minA: number,
    maxA: number,
    minB: number,
    maxB: number
  ): boolean {
    return Math.max(minA, minB) < Math.min(maxA, maxB);
  }

  private computeDescriptorBounds(descriptor: ChunkDescriptor): ChunkBounds {
    const scale = this.config.blockWidth << descriptor.lodIndex;
    const minX = descriptor.chunkX * scale;
    const minZ = descriptor.chunkZ * scale;
    const minY = descriptor.originY;
    return {
      minX,
      maxX: minX + scale,
      minY,
      maxY: minY + scale,
      minZ,
      maxZ: minZ + scale,
    };
  }

  private transitionFacesChanged(
    current: TransitionFace[],
    next: TransitionFace[]
  ): boolean {
    if (current.length !== next.length) {
      return true;
    }
    for (let i = 0; i < current.length; i++) {
      if (current[i] !== next[i]) {
        return true;
      }
    }
    return false;
  }

  private chunkKey(
    lodIndex: number,
    chunkX: number,
    chunkY: number,
    chunkZ: number
  ): string {
    return `${lodIndex}:${chunkX}:${chunkY}:${chunkZ}`;
  }

  private createDescriptor(node: OctreeNode): ChunkDescriptor {
    const level = this.lodLookup.get(node.lodIndex);
    if (!level) {
      throw new Error(`Missing LOD configuration for level ${node.lodIndex}`);
    }
    const samplesPerAxis = this.config.blockWidth << node.lodIndex;
    return {
      lodIndex: node.lodIndex,
      chunkX: node.chunkX,
      chunkY: node.chunkY,
      chunkZ: node.chunkZ,
      key: this.chunkKey(node.lodIndex, node.chunkX, node.chunkY, node.chunkZ),
      color: level.color,
      transitionFaces: [],
      originY: node.chunkY * samplesPerAxis,
    };
  }

  private chunkDistanceToPoint(
    lodIndex: number,
    chunkX: number,
    chunkY: number,
    chunkZ: number,
    point: Vector3Like
  ): number {
    const size = this.chunkWorldSize(lodIndex);
    const minX = chunkX * size;
    const minY = chunkY * size;
    const minZ = chunkZ * size;
    const maxX = minX + size;
    const maxY = minY + size;
    const maxZ = minZ + size;
    const dx =
      point.x < minX ? minX - point.x : point.x > maxX ? point.x - maxX : 0;
    const dy =
      point.y < minY ? minY - point.y : point.y > maxY ? point.y - maxY : 0;
    const dz =
      point.z < minZ ? minZ - point.z : point.z > maxZ ? point.z - maxZ : 0;
    return Math.hypot(dx, dy, dz);
  }

  private distanceBetween(a: Vector3Like, b: Vector3Like): number {
    return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
  }

  private teleportThreshold(): number {
    const farthest = this.derivedLodLevels[this.derivedLodLevels.length - 1];
    const coarseSize = this.chunkWorldSize(this.maxLodIndex);
    const coarseSpan = coarseSize * 8;
    const lodSpan = farthest.maxDistance * 4;
    return Math.max(lodSpan, coarseSpan);
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

  private isNodeWithinWorld(node: OctreeNode): boolean {
    const size = this.chunkWorldSize(node.lodIndex);
    const minY = node.chunkY * size;
    const maxY = minY + size;
    return maxY > this.worldMinY && minY < this.worldMaxY;
  }

  private createPlan(): ChunkPlan {
    return { requests: [], cancels: [], releases: [] };
  }
}
