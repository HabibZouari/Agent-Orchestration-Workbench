import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiTarget = `http://127.0.0.1:${process.env.API_PORT || 8000}`;
const proxied = ["/health", "/agents", "/tasks", "/documents", "/search"];

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: Object.fromEntries(proxied.map((path) => [path, apiTarget])),
  },
});
