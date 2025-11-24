import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import path from "node:path";
import dotenv from "dotenv";

dotenv.config();

const demoRoot = fileURLToPath(new URL(".", import.meta.url));
const srcDir = fileURLToPath(new URL("../src", import.meta.url));

const host = process.env.VITE_DEV_HOST || undefined;

export default defineConfig({
  root: demoRoot,
  base: "./",
  server: { host },
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
