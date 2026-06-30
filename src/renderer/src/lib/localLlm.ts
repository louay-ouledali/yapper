/**
 * On-device LLM — a small model that runs entirely in the renderer via wllama
 * (llama.cpp compiled to WASM), so summaries and cue cards never leave the
 * machine and need no key, server, or internet after a one-time model download.
 *
 * Chosen over a native addon for the same reason as Vosk: WASM packages cleanly
 * with no native ABI to break. It runs MULTI-THREADED: the main process sets the
 * `SharedArrayBuffer` switch, so wllama's own support gate (a SharedArrayBuffer
 * postMessage clone) passes and it spins up real worker threads — verified on a
 * 16-core machine (8 threads, warm answers ~1.4 s). We pin `n_threads` to half
 * the cores (capped) for determinism instead of leaving it to auto-detection.
 *
 * The GGUF is downloaded once (see main `localmodel:download`) and served over
 * the same loopback model server; we load it from that local URL. Everything is
 * lazy: wllama is only imported when the user actually runs a local job.
 */
import { Wllama, type ChatCompletionMessage } from '@wllama/wllama'
import wasmUrl from '@wllama/wllama/esm/wasm/wllama.wasm?url'
import type { ChatMessage, ChatResult } from './llm'
import type { LlmTierId } from './llm-shared'

let wllama: Wllama | null = null
let loadPromise: Promise<void> | null = null
let loadedFile = '' // which GGUF is currently resident (so a tier switch reloads)

async function ensureLoaded(tier: LlmTierId = 'floor'): Promise<void> {
  const status = await window.yapper?.localModelStatus(tier)
  if (!status?.installed || !status.url) {
    throw new Error('On-device model not installed — download it in Settings → AI brain')
  }
  // Up to two passes: if a load for a DIFFERENT model is in flight, await it then reload.
  for (let i = 0; i < 2; i++) {
    if (wllama && wllama.isModelLoaded() && loadedFile === status.file) return
    if (!loadPromise) {
      loadPromise = (async () => {
        // Switching tiers: free the previously-loaded model first.
        if (wllama) {
          try {
            await wllama.exit()
          } catch {
            /* ignore */
          }
          wllama = null
          loadedFile = ''
        }
        const inst = new Wllama({ default: wasmUrl }, { allowOffline: true, suppressNativeLog: true })
        // When the runtime lacks WASM JSPI/Memory64 (current Electron), wllama needs a
        // "compat" build it otherwise pulls from a CDN — which our offline CSP blocks.
        // Point it at the copy we serve over loopback so it loads locally instead.
        const base = status.url.slice(0, status.url.lastIndexOf('/'))
        inst.setCompat({ worker: `${base}/wllama-compat/wllama.js`, wasm: `${base}/wllama-compat/wllama.wasm` })
        // Half the logical cores (min 2, max 8): enough to saturate a small model
        // without starving the UI thread or the whisper worker on big machines.
        const nThreads = Math.max(2, Math.min(8, Math.floor((navigator.hardwareConcurrency || 4) / 2)))
        await inst.loadModelFromUrl(status.url, { n_ctx: 4096, n_threads: nThreads })
        wllama = inst
        loadedFile = status.file
      })().finally(() => {
        loadPromise = null
      })
    }
    await loadPromise
  }
}

/** Run a chat completion fully on-device. Returns a clear error if the model
 * isn't downloaded yet or the engine fails to start. */
export async function localChat(messages: ChatMessage[], temperature: number, tier: LlmTierId = 'floor'): Promise<ChatResult> {
  try {
    await ensureLoaded(tier)
    if (!wllama) return { ok: false, text: '', error: 'on-device model unavailable' }
    const res = await wllama.createChatCompletion({
      messages: messages as ChatCompletionMessage[],
      temperature,
      max_tokens: 320,
      stream: false
    })
    const text = res?.choices?.[0]?.message?.content ?? ''
    return { ok: true, text: String(text).trim() }
  } catch (e) {
    return { ok: false, text: '', error: (e as Error).message }
  }
}

/**
 * Streaming chat completion — tokens arrive via `onToken` as they generate, so
 * the Q&A Copilot can paint talking points the moment they exist instead of
 * after the whole answer. An AbortSignal cancels generation mid-stream (a new
 * audience question supersedes the previous one).
 */
export async function localChatStream(
  messages: ChatMessage[],
  temperature: number,
  onToken: (text: string) => void,
  signal?: AbortSignal,
  maxTokens = 320,
  tier: LlmTierId = 'floor'
): Promise<ChatResult> {
  try {
    await ensureLoaded(tier)
    if (!wllama) return { ok: false, text: '', error: 'on-device model unavailable' }
    const stream = await wllama.createChatCompletion({
      messages: messages as ChatCompletionMessage[],
      temperature,
      max_tokens: maxTokens,
      stream: true,
      abortSignal: signal
    })
    let full = ''
    for await (const chunk of stream) {
      if (signal?.aborted) break
      const piece = chunk?.choices?.[0]?.delta?.content
      if (piece) {
        full += piece
        onToken(piece)
      }
    }
    return { ok: true, text: full.trim() }
  } catch (e) {
    if (signal?.aborted) return { ok: true, text: '', error: undefined }
    return { ok: false, text: '', error: (e as Error).message }
  }
}

/** Free the loaded model (e.g. when switching away from the local provider). */
export async function unloadLocal(): Promise<void> {
  try {
    await wllama?.exit()
  } catch {
    /* ignore */
  }
  wllama = null
}
