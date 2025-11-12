import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Expose environment variables to the client
  // Variables prefixed with VITE_ are available via import.meta.env
  envPrefix: 'VITE_',
})
