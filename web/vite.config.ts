import { defineConfig, type Plugin } from "vite";
import solid from "vite-plugin-solid";
import { soulwindow } from "soulwindow/vite";

export default defineConfig({
  base: "./",
  plugins: [
    solid(),
    // TODO(soulwindow): runtime/vite.d.ts types transformIndexHtml as unknown, which Vite's
    // Plugin type rejects; drop the cast once it is typed as IndexHtmlTransform upstream.
    soulwindow({ storagePrefix: "reezsynth-", density: "compact" }) as Plugin,
  ],
  // mediabunny is loaded as-is (esbuild pre-bundling gains nothing for an ESM package).
  optimizeDeps: { exclude: ["mediabunny"] },
  // The mediabunny chunk (~560 kB) loads only when a video is opened.
  build: { target: "es2022", outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 600 },
  server: { port: 5173 },
});
