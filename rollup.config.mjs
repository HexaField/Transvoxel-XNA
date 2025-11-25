import { nodeResolve } from "@rollup/plugin-node-resolve";
import typescript from "@rollup/plugin-typescript";
import dts from "rollup-plugin-dts";

const input = "src/index.ts";

export default [
  {
    input,
    output: {
      file: "dist/index.js",
      format: "es",
      sourcemap: true,
    },
    plugins: [
      nodeResolve({ preferBuiltins: false }),
      typescript({
        tsconfig: "./tsconfig.json",
        compilerOptions: {
          declaration: false,
          declarationMap: false,
        },
      }),
    ],
  },
  {
    input,
    output: {
      file: "dist/index.d.ts",
      format: "es",
    },
    plugins: [
      dts({
        tsconfig: "./tsconfig.json",
      }),
    ],
  },
];
