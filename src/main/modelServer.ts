/**
 * Offline model server — scavenged from Yapper. A loopback HTTP server that
 * serves everything the on-device engines need without a CDN:
 *   - `/hf/<path>`        → a transformers.js model file (Whisper), downloaded
 *                           ONCE from the Hugging Face hub into userData, marked
 *                           `.complete`, and served from disk forever after (this
 *                           is what makes Whisper load offline on a packaged
 *                           `file://` origin, where the browser Cache API is absent)
 *   - `/ort/<file>`       → ONNX Runtime wasm (shipped inside transformers.js)
 *   - `/wllama-compat/…`  → wllama's offline compat build (on-device LLM)
 *   - `/<file>`           → a downloaded model by filename (e.g. the wllama GGUF)
 *
 * Downloads are resumable (`.partial` + Range), stall-guarded, and redirect-aware
 * (HF returns relative redirects for small files).
 */
import { app, BrowserWindow } from 'electron'
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http'
import { createReadStream, createWriteStream, existsSync, mkdirSync, readdirSync, renameSync, rmSync, statSync } from 'node:fs'
import { rm, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { createRequire } from 'node:module'
import https from 'node:https'

const nodeRequire = createRequire(join(__dirname, 'index.js'))

function ortAssetPath(name: string): string | null {
  if (name.includes('/') || name.includes('\\') || name.includes('..')) return null
  try {
    const distDir = dirname(nodeRequire.resolve('@huggingface/transformers'))
    const p = join(distDir, name)
    return existsSync(p) ? p : null
  } catch {
    return null
  }
}
function wllamaCompatPath(name: string): string | null {
  if (name.includes('/') || name.includes('\\') || name.includes('..')) return null
  try {
    const pkgDir = dirname(nodeRequire.resolve('@wllama/wllama-compat'))
    const p = join(pkgDir, 'wasm', name)
    return existsSync(p) ? p : null
  } catch {
    return null
  }
}

// On-device cleanup models (GGUF, run on CPU via wllama). Each downloads ONCE into
// userData over the loopback server and loads offline forever after. Two tiers:
//   floor    — Qwen 0.5B (~400 MB), smallest/fastest, runs anywhere
//   balanced — Qwen 1.5B (~1.1 GB), stronger cleanup, still fully on-device
export const LLM_MODELS = {
  floor: {
    file: 'qwen2.5-0.5b-instruct-q4_k_m.gguf',
    url: 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf'
  },
  balanced: {
    file: 'qwen2.5-1.5b-instruct-q4_k_m.gguf',
    url: 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf'
  }
} as const
/** Resolve the model file+url for a tier (defaults to the floor). Guards arbitrary input. */
export const llmModelFor = (tier?: string): { file: string; url: string } =>
  tier === 'balanced' ? LLM_MODELS.balanced : LLM_MODELS.floor

// Back-compat aliases (the floor is the zero-setup default).
export const LLM_MODEL_FILE = LLM_MODELS.floor.file
export const LLM_MODEL_URL = LLM_MODELS.floor.url

const BUNDLED_MODELS_DIR = app.isPackaged
  ? join(process.resourcesPath, 'resources', 'models')
  : join(__dirname, '../../resources/models')
export const downloadedModelsDir = (): string => join(app.getPath('userData'), 'models')

export function resolveModel(file: string): string | null {
  if (file.includes('/') || file.includes('\\') || file.includes('..')) return null
  const dl = join(downloadedModelsDir(), file)
  if (existsSync(dl) && existsSync(`${dl}.complete`)) return dl
  const bundled = join(BUNDLED_MODELS_DIR, file)
  return existsSync(bundled) ? bundled : null
}

let modelBaseUrl = ''
export const getModelBaseUrl = (): string => modelBaseUrl

// The window that receives download-progress events.
let progressWin: BrowserWindow | null = null
export function setProgressWindow(win: BrowserWindow | null): void {
  progressWin = win
}
const sendProgress = (channel: string, payload: unknown): void => {
  if (progressWin && !progressWin.isDestroyed()) progressWin.webContents.send(channel, payload)
}

/** Stream a resolved local file with MIME, HEAD, and Range support. */
function serveFile(file: string, req: IncomingMessage, res: ServerResponse): void {
  const mime = file.endsWith('.gz')
    ? 'application/gzip'
    : file.endsWith('.wasm')
      ? 'application/wasm'
      : /\.m?js$/.test(file)
        ? 'text/javascript'
        : file.endsWith('.json')
          ? 'application/json'
          : 'application/octet-stream'
  res.setHeader('Content-Type', mime)
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Accept-Ranges', 'bytes')
  let size = 0
  try {
    size = statSync(file).size
  } catch {
    /* best-effort */
  }
  if (req.method === 'HEAD') {
    if (size) res.setHeader('Content-Length', String(size))
    res.end()
    return
  }
  const range = /bytes=(\d*)-(\d*)/.exec(req.headers.range || '')
  if (range && size) {
    const start = range[1] ? parseInt(range[1], 10) : 0
    const end = range[2] ? parseInt(range[2], 10) : size - 1
    if (start <= end && end < size) {
      res.statusCode = 206
      res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`)
      res.setHeader('Content-Length', String(end - start + 1))
      createReadStream(file, { start, end }).pipe(res)
      return
    }
  }
  if (size) res.setHeader('Content-Length', String(size))
  createReadStream(file).pipe(res)
}

export function startModelServer(): void {
  const server = createServer((req, res) => {
    const name = decodeURIComponent((req.url || '').replace(/^\//, '').split('?')[0])
    if (name.startsWith('hf/')) {
      ensureHfFile(name.slice(3), (received, total) =>
        sendProgress('models:hfProgress', {
          path: name.slice(3),
          received,
          total,
          pct: total ? Math.round((received / total) * 100) : 0
        })
      )
        .then((file) => serveFile(file, req, res))
        .catch((e) => {
          res.statusCode = 502
          res.end(`hf fetch failed: ${(e as Error).message}`)
        })
      return
    }
    const file = name.startsWith('ort/')
      ? ortAssetPath(name.slice(4))
      : name.startsWith('wllama-compat/')
        ? wllamaCompatPath(name.slice('wllama-compat/'.length))
        : name
          ? resolveModel(name)
          : null
    if (!file) {
      res.statusCode = 404
      res.end('not found')
      return
    }
    serveFile(file, req, res)
  })
  server.listen(0, '127.0.0.1', () => {
    const addr = server.address()
    if (addr && typeof addr === 'object') modelBaseUrl = `http://127.0.0.1:${addr.port}`
  })
}

/** Resumable, stall-guarded, redirect-aware streaming download. */
export function downloadFile(url: string, dest: string, onProgress: (received: number, total: number) => void): Promise<void> {
  const partial = `${dest}.partial`
  const STALL_MS = 60_000
  return new Promise((resolve, reject) => {
    let file: ReturnType<typeof createWriteStream> | null = null
    let settled = false
    let retriedRange = false
    let stallTimer: ReturnType<typeof setTimeout> | null = null
    const clearStall = (): void => {
      if (stallTimer) clearTimeout(stallTimer)
      stallTimer = null
    }
    const fail = (e: Error): void => {
      if (settled) return
      settled = true
      clearStall()
      file?.destroy()
      reject(e)
    }
    const get = (u: string, depth = 0): void => {
      if (depth > 5) return fail(new Error('too many redirects'))
      let offset = 0
      try {
        offset = statSync(partial).size
      } catch {
        offset = 0
      }
      const headers: Record<string, string> = offset > 0 ? { Range: `bytes=${offset}-` } : {}
      const req = https.get(u, { headers }, (res) => {
        const status = res.statusCode ?? 0
        if (status >= 300 && status < 400 && res.headers.location) {
          res.resume()
          const next = new URL(res.headers.location, u).toString()
          return get(next, depth + 1)
        }
        if (status === 416) {
          res.resume()
          if (retriedRange) return fail(new Error('HTTP 416'))
          retriedRange = true
          rmSync(partial, { force: true })
          return get(u, depth + 1)
        }
        if (status !== 200 && status !== 206) {
          res.resume()
          return fail(new Error(`HTTP ${status}`))
        }
        const resuming = status === 206 && offset > 0
        if (!resuming) offset = 0
        file = createWriteStream(partial, { flags: resuming ? 'a' : 'w' })
        const bodyLen = Number(res.headers['content-length'] || 0)
        const total = bodyLen > 0 ? offset + bodyLen : 0
        let received = offset
        const bumpStall = (): void => {
          clearStall()
          stallTimer = setTimeout(() => {
            res.destroy()
            fail(new Error(`download stalled (no data for ${STALL_MS / 1000}s)`))
          }, STALL_MS)
        }
        bumpStall()
        res.on('data', (c) => {
          received += c.length
          onProgress(received, total)
          bumpStall()
        })
        res.on('error', fail)
        file.on('error', fail)
        res.pipe(file)
        file.on('finish', () => {
          if (settled) return
          clearStall()
          if (total > 0 && received !== total) return fail(new Error(`incomplete download (${received}/${total} bytes)`))
          settled = true
          file!.close(() => {
            const attempt = (n: number): void => {
              try {
                renameSync(partial, dest)
                resolve()
              } catch (e) {
                if (n < 5) setTimeout(() => attempt(n + 1), 250)
                else reject(e as Error)
              }
            }
            attempt(0)
          })
        })
      })
      req.on('error', fail)
      req.setTimeout(30_000, () => req.destroy(new Error('connection timed out')))
    }
    get(url)
  })
}

// ── Hugging Face model cache (transformers.js Whisper) ──
const HF_HOST = 'https://huggingface.co'
const hfCacheDir = (): string => join(downloadedModelsDir(), 'hf')
const hfInflight = new Map<string, Promise<string>>()

function hfCachePath(rel: string): string | null {
  const clean = rel.replace(/^\/+/, '')
  if (!clean || clean.includes('..') || clean.includes('\\')) return null
  return join(hfCacheDir(), clean)
}

export function ensureHfFile(rel: string, onProgress?: (received: number, total: number) => void): Promise<string> {
  const dest = hfCachePath(rel)
  if (!dest) return Promise.reject(new Error('bad hf path'))
  if (existsSync(dest) && existsSync(`${dest}.complete`)) return Promise.resolve(dest)
  const existing = hfInflight.get(dest)
  if (existing) return existing
  const p = (async () => {
    mkdirSync(dirname(dest), { recursive: true })
    await rm(`${dest}.complete`, { force: true }).catch(() => {})
    await downloadFile(`${HF_HOST}/${rel}`, dest, (r, t) => onProgress?.(r, t))
    await writeFile(`${dest}.complete`, '')
    return dest
  })().finally(() => hfInflight.delete(dest))
  hfInflight.set(dest, p)
  return p
}

export function hfModelSize(model: string): number {
  if (model.includes('..') || model.includes('\\')) return 0
  const root = join(hfCacheDir(), model)
  let total = 0
  const walk = (dir: string): void => {
    let entries
    try {
      entries = readdirSync(dir, { withFileTypes: true })
    } catch {
      return
    }
    for (const e of entries) {
      const full = join(dir, e.name)
      if (e.isDirectory()) walk(full)
      else if (!e.name.endsWith('.partial') && !e.name.endsWith('.complete')) {
        try {
          total += statSync(full).size
        } catch {
          /* best-effort */
        }
      }
    }
  }
  walk(root)
  return total
}

/** Download a model file robustly (clear marker → download → write `.complete`). */
export async function fetchModel(url: string, file: string, channel: string): Promise<{ ok: boolean; error?: string }> {
  const dir = downloadedModelsDir()
  mkdirSync(dir, { recursive: true })
  const dest = join(dir, file)
  const marker = `${dest}.complete`
  try {
    await rm(marker, { force: true }).catch(() => {})
    await downloadFile(url, dest, (received, total) =>
      sendProgress(channel, { received, total, pct: total ? Math.round((received / total) * 100) : 0 })
    )
    await writeFile(marker, '')
    sendProgress(channel, { done: true })
    return { ok: true }
  } catch (e) {
    await rm(marker, { force: true }).catch(() => {})
    const error = (e as Error).message
    sendProgress(channel, { error })
    return { ok: false, error }
  }
}
