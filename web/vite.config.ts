import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  base: "./",
  plugins: [
    react(),
    tailwindcss(),
    {
      // The chart library's logo is switched off, so its tooltip text is removed from the bundle too.
      name: "strip-library-badge",
      renderChunk: (code) => ({ code: code.replace(["Produced", "with", "Plotly.js"].join(" "), ""), map: null }),
    },
  ],
  esbuild: { legalComments: "none" },
  build: { chunkSizeWarningLimit: 6000, sourcemap: false },
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  preview: { port: 4173, proxy: { "/api": "http://127.0.0.1:8000" } },
});
