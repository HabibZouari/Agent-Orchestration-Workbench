import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/health": "http://127.0.0.1:8000",
      "/agents": "http://127.0.0.1:8000",
      "/tasks": "http://127.0.0.1:8000",
      "/documents": "http://127.0.0.1:8000",
      "/search": "http://127.0.0.1:8000"
    }
  }
});
