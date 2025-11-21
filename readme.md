# Transvoxel-XNA
Authors: BinaryConstruct, Oggs91

Implmentation of Eric Lengyl's Transvoxel Algorithm in c# for use in MonoGame, Unity, etc.
![transvoxel-xna-test.png](https://raw.githubusercontent.com/BinaryConstruct/Transvoxel-XNA/master/Docs/transvoxel-xna-test.png)

## TypeScript port

An ES2020/TypeScript mesher now lives under `src/`. It exposes a `TransvoxelMesher` that can emit
regular cell geometry as well as the transition cells needed to stitch adjacent LODs.

```ts
import { TransvoxelMesher, TransitionFace } from "./dist/surface-extractor/transvoxel-extractor.js";
import { Vector3i } from "./dist/math/vector3i.js";

const density = (x: number, y: number, z: number) => /* signed density */;
const mesher = new TransvoxelMesher();
const faces: TransitionFace[] = [
	"negativeX",
	"positiveX",
	"negativeY",
	"positiveY",
	"negativeZ",
	"positiveZ",
];

const mesh = mesher.extractBlock(density, {
	origin: new Vector3i(0, 0, 0),
	lodIndex: 1,
	cellSize: 1,
	transitionFaces: faces,
});

console.log(mesh.vertices.length, mesh.indices.length);
```

### Building the library

```
npm install
npm run build
```

### Three.js demo

A small WebGL demo that consumes the TypeScript port via Vite + Three.js is available under
`demo/`.

```
npm run demo:dev     # start a local dev server
npm run demo:build   # produce a static build in demo/dist
```

The demo renders a single LOD 1 chunk with all six transition faces enabled so the seam logic can be
inspected in real time.

## Commit History
https://github.com/BinaryConstruct/Transvoxel-XNA/commits/master

## License
Copyright 2019 BinaryConstruct

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Credits
Eric Lengyel's Transvoxel Algorithm.  
http://www.terathon.com/voxels/