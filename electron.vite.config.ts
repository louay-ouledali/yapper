import { resolve } from 'node:path'
import { defineConfig, externalizeDepsPlugin } from 'electron-vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  main: {
    plugins: [externalizeDepsPlugin()]
  },
  preload: {
    plugins: [externalizeDepsPlugin()]
  },
  renderer: {
    // Use a distinct dev port so it never clashes with a deck running on 5173.
    server: { port: 5180 },
    // wllama loads a real .wasm via a ?url import — don't pre-bundle it.
    // vosk-browser IS pre-bundled: it's a UMD module that inlines its worker +
    // wasm as base64, and pre-bundling gives it proper ESM named exports (left
    // un-bundled, the UMD fell back to a window global and `createModel` was
    // missing from the import).
    optimizeDeps: { exclude: ['@wllama/wllama', '@huggingface/transformers'] },
    resolve: {
      alias: {
        '@renderer': resolve('src/renderer/src')
      }
    },
    plugins: [react()]
  }
})
