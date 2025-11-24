import { BufferGeometry, Float32BufferAttribute, LineBasicMaterial, LineSegments, Vector3 } from 'three'

export interface OrientationStats {
  triangleCount: number
  invertedCount: number
  minDot: number
  maxDot: number
  flaggedTriangles: boolean[]
}

export function analyzeOrientation(
  positions: Float32Array,
  normals: Float32Array,
  indices: Uint32Array
): OrientationStats {
  const triangleCount = indices.length / 3
  const flaggedTriangles = new Array<boolean>(triangleCount).fill(false)
  let invertedCount = 0
  let minDot = 1
  let maxDot = -1

  const v0 = new Vector3()
  const v1 = new Vector3()
  const v2 = new Vector3()
  const n0 = new Vector3()
  const n1 = new Vector3()
  const n2 = new Vector3()
  const edge1 = new Vector3()
  const edge2 = new Vector3()
  const faceNormal = new Vector3()
  const averagedNormal = new Vector3()

  for (let tri = 0; tri < triangleCount; tri++) {
    const i0 = indices[tri * 3 + 0]
    const i1 = indices[tri * 3 + 1]
    const i2 = indices[tri * 3 + 2]

    setVectorFromArray(v0, positions, i0)
    setVectorFromArray(v1, positions, i1)
    setVectorFromArray(v2, positions, i2)
    setVectorFromArray(n0, normals, i0)
    setVectorFromArray(n1, normals, i1)
    setVectorFromArray(n2, normals, i2)

    edge1.subVectors(v1, v0)
    edge2.subVectors(v2, v0)
    faceNormal.copy(edge1).cross(edge2).normalize()

    averagedNormal.copy(n0).add(n1).add(n2).divideScalar(3).normalize()
    const dot = faceNormal.dot(averagedNormal)
    minDot = Math.min(minDot, dot)
    maxDot = Math.max(maxDot, dot)

    if (dot < 0) {
      flaggedTriangles[tri] = true
      invertedCount++
    }
  }

  return { triangleCount, invertedCount, minDot, maxDot, flaggedTriangles }
}

export function createNormalHelper(positions: Float32Array, normals: Float32Array, color: number): LineSegments {
  const vertexCount = positions.length / 3
  const linePositions = new Float32Array(vertexCount * 2 * 3)
  const normalScale = 0.5

  for (let i = 0; i < vertexCount; i++) {
    const px = positions[i * 3 + 0]
    const py = positions[i * 3 + 1]
    const pz = positions[i * 3 + 2]
    const nx = normals[i * 3 + 0]
    const ny = normals[i * 3 + 1]
    const nz = normals[i * 3 + 2]

    const lineBase = i * 6
    linePositions[lineBase + 0] = px
    linePositions[lineBase + 1] = py
    linePositions[lineBase + 2] = pz
    linePositions[lineBase + 3] = px + nx * normalScale
    linePositions[lineBase + 4] = py + ny * normalScale
    linePositions[lineBase + 5] = pz + nz * normalScale
  }

  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new Float32BufferAttribute(linePositions, 3))
  const material = new LineBasicMaterial({
    color,
    transparent: true,
    opacity: 0.65
  })
  return new LineSegments(geometry, material)
}

export function createInvertedHelper(positions: Float32Array, indices: Uint32Array, flagged: boolean[]): LineSegments {
  const highlighted = flagged.filter(Boolean).length
  const linePositions = new Float32Array(Math.max(1, highlighted) * 2 * 3)
  const v0 = new Vector3()
  const v1 = new Vector3()
  const v2 = new Vector3()
  const edge1 = new Vector3()
  const edge2 = new Vector3()
  const faceNormal = new Vector3()
  const centroid = new Vector3()
  const normalScale = 0.8

  let cursor = 0
  for (let tri = 0; tri < flagged.length; tri++) {
    if (!flagged[tri]) {
      continue
    }

    const i0 = indices[tri * 3 + 0]
    const i1 = indices[tri * 3 + 1]
    const i2 = indices[tri * 3 + 2]
    setVectorFromArray(v0, positions, i0)
    setVectorFromArray(v1, positions, i1)
    setVectorFromArray(v2, positions, i2)

    edge1.subVectors(v1, v0)
    edge2.subVectors(v2, v0)
    faceNormal.copy(edge1).cross(edge2).normalize()

    centroid.copy(v0).add(v1).add(v2).divideScalar(3)

    linePositions[cursor + 0] = centroid.x
    linePositions[cursor + 1] = centroid.y
    linePositions[cursor + 2] = centroid.z
    linePositions[cursor + 3] = centroid.x + faceNormal.x * normalScale
    linePositions[cursor + 4] = centroid.y + faceNormal.y * normalScale
    linePositions[cursor + 5] = centroid.z + faceNormal.z * normalScale
    cursor += 6
  }

  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new Float32BufferAttribute(linePositions, 3))
  const material = new LineBasicMaterial({
    color: 0xff6277,
    transparent: true,
    opacity: 0.9
  })
  return new LineSegments(geometry, material)
}

function setVectorFromArray(target: Vector3, source: Float32Array, index: number): void {
  target.set(source[index * 3 + 0], source[index * 3 + 1], source[index * 3 + 2])
}
