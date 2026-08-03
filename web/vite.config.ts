import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      // Proxy keeps the dashboard same-origin in dev, so the SSE stream is not
      // subject to CORS preflight and EventSource reconnects cleanly.
      "/api": {
        target: "http://127.0.0.1:8030",
        changeOrigin: true,
        // SSE must not be buffered or the stream arrives in bursts.
        configure: (proxy) => {
          proxy.on("proxyRes", (proxyRes) => {
            if (proxyRes.headers["content-type"]?.includes("text/event-stream")) {
              proxyRes.headers["x-accel-buffering"] = "no";
            }
          });
        },
      },
    },
  },
  build: {
    target: "es2022",
    rollupOptions: {
      output: {
        // Vite 8 ships Rolldown, which requires manualChunks as a function
        // rather than the object form Rollup accepted.
        manualChunks: (id: string) => {
          if (id.includes("lightweight-charts") || id.includes("uplot")) return "charts";
          if (id.includes("@tanstack")) return "table";
          if (id.includes("node_modules")) return "vendor";
          return undefined;
        },
      },
    },
  },
});
