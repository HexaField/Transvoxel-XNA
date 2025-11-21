import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import path from "node:path";

const demoRoot = fileURLToPath(new URL(".", import.meta.url));
const srcDir = fileURLToPath(new URL("../src", import.meta.url));

export default defineConfig({
  root: demoRoot,
  base: "./",
  build: {
    outDir: path.resolve(demoRoot, "dist"),
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      "@": srcDir,
    },
  },
});
