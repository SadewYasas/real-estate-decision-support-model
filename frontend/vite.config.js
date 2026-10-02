import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
// The dev server forwards API calls to the Flask back end (python backend/app.py, port 5000),
// so the browser only ever talks to one origin and needs no CORS set-up.
export default defineConfig({
  plugins: [
    react(),
    tailwindcss()
  ],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:5000',
      '/predict': 'http://127.0.0.1:5000',
    },
  },
})
