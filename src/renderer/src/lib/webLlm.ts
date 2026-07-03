/**
 * On-device GPU LLM — web-llm (MLC) compiled to WebGPU. Powers the bigger 'turbo'
 * and 'max' cleanup tiers: a genuinely stronger model, run on the GPU so it stays
 * fast despite the size. Model artifacts (weights + a tiny model-lib wasm) are
 * fetched ONCE from Hugging Face / MLC and cached in the browser (IndexedDB, which
 * works on the packaged file:// origin where the Cache API is absent), then run
 * fully offline. Everything is lazy — web-llm is only imported when a GPU tier runs.
 *
 * If WebGPU or a hardware adapter is missing, the caller falls back to the CPU
 * 'standard' engine (wllama); nothing here ever silently downloads mid-dictation
 * (the overlay / Settings prepare the model first, with a progress bar).
 */
import { CreateMLCEngine, prebuiltAppConfig, hasModelInCache, type MLCEngine, type AppConfig } from '@mlc-ai/web-llm'
import type { ChatMessage, ChatResult } from './llm'

/** IndexedDB caching so downloaded models persist offline on a file:// origin
 *  (the Cache API is absent there, which is why we don't use the default backend). */
const appConfig: AppConfig = { ...prebuiltAppConfig, cacheBackend: 'indexeddb' }

let engine: MLCEngine | null = null
let loadedModel = ''
let loadPromise: Promise<MLCEngine> | null = null

/** True when a real hardware WebGPU adapter is present (web-llm can run). */
export async function webLlmAvailable(): Promise<boolean> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const gpu = (navigator as any)?.gpu
    if (!gpu) return false
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' })
    return Boolean(adapter && !adapter.isFallbackAdapter)
  } catch {
    return false
  }
}

/** Whether the model's artifacts are already cached (so no download is needed). */
export async function webLlmHasModel(modelId: string): Promise<boolean> {
  try {
    return await hasModelInCache(modelId, appConfig)
  } catch {
    return false
  }
}

/** Load (and memoize) the engine for a model, downloading+caching on first use.
 *  `onProgress` reports 0–100 while artifacts download and the GPU shaders compile. */
async function ensureEngine(modelId: string, onProgress?: (pct: number) => void): Promise<MLCEngine> {
  if (engine && loadedModel === modelId) return engine
  if (loadPromise) {
    await loadPromise.catch(() => {})
    if (engine && loadedModel === modelId) return engine
  }
  const initProgressCallback = (r: { progress?: number }): void => onProgress?.(Math.min(100, Math.round((r.progress ?? 0) * 100)))
  loadPromise = (async () => {
    if (engine) {
      // Switching tiers: reload frees the old model's GPU memory first.
      engine.setInitProgressCallback(initProgressCallback)
      await engine.reload(modelId)
    } else {
      engine = await CreateMLCEngine(modelId, { appConfig, initProgressCallback })
    }
    loadedModel = modelId
    return engine
  })().finally(() => {
    loadPromise = null
  })
  return loadPromise
}

/** Ensure a GPU model is downloaded, cached and resident (used by the prepare/first-run flow). */
export async function webLlmPrepare(modelId: string, onProgress?: (pct: number) => void): Promise<void> {
  await ensureEngine(modelId, onProgress)
}

/**
 * Streaming chat completion on the GPU. Tokens arrive via `onToken`; an AbortSignal
 * interrupts generation. `onProgress` fires only if the model still needs loading.
 */
export async function webLlmChatStream(
  messages: ChatMessage[],
  temperature: number,
  onToken: (text: string) => void,
  signal: AbortSignal | undefined,
  maxTokens: number,
  modelId: string,
  onProgress?: (pct: number) => void
): Promise<ChatResult> {
  try {
    const eng = await ensureEngine(modelId, onProgress)
    if (signal?.aborted) return { ok: true, text: '' }
    const onAbort = (): void => {
      try {
        void eng.interruptGenerate()
      } catch {
        /* ignore */
      }
    }
    signal?.addEventListener('abort', onAbort, { once: true })
    let full = ''
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const req: any = { messages, temperature, max_tokens: maxTokens, stream: true }
      const chunks = (await eng.chat.completions.create(req)) as unknown as AsyncIterable<{
        choices?: { delta?: { content?: string } }[]
      }>
      for await (const c of chunks) {
        if (signal?.aborted) break
        const piece = c.choices?.[0]?.delta?.content
        if (piece) {
          full += piece
          onToken(piece)
        }
      }
    } finally {
      signal?.removeEventListener('abort', onAbort)
    }
    return { ok: true, text: full.trim() }
  } catch (e) {
    if (signal?.aborted) return { ok: true, text: '' }
    return { ok: false, text: '', error: (e as Error).message }
  }
}

/** Free the GPU model (e.g. when switching away from the local provider). */
export async function unloadWebLlm(): Promise<void> {
  try {
    await engine?.unload()
  } catch {
    /* ignore */
  }
  engine = null
  loadedModel = ''
}
