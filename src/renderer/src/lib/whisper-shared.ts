/**
 * Pure helpers shared by the Whisper main-thread API (whisper.ts) and the
 * dedicated decode worker (whisper.worker.ts). Must stay free of DOM/window
 * references AND of any `@huggingface/transformers` import — whisper.ts (main
 * thread) pulls this in, and importing transformers there would cost ~1s of
 * main-thread parse (the exact freeze the worker design removes).
 */

/** The subset of transformers.js `env` we configure. Passed in by each worker so
 * this module never imports transformers itself. */
export interface TransformersEnvLike {
  allowLocalModels?: boolean
  allowRemoteModels?: boolean
  useBrowserCache?: boolean
  remoteHost?: string
  remotePathTemplate?: string
}

/**
 * Point transformers.js at the main-process loopback HF cache proxy so model
 * weights download ONCE into userData and load offline forever. Without this,
 * transformers.js caches in the browser Cache API — which does not exist on the
 * packaged `file://` origin, so every launch re-downloaded the whole model. The
 * proxy (`/hf/...` in main) maps these requests to a one-time on-disk cache.
 *
 * Must be called before the first `pipeline()` call in each worker, once the
 * loopback base URL is known. A missing baseUrl leaves transformers.js on its
 * defaults (direct hub + Cache API) as a best-effort fallback.
 */
export function configureTransformersCache(env: TransformersEnvLike, baseUrl?: string): void {
  if (!baseUrl) return
  env.allowLocalModels = false
  env.allowRemoteModels = true
  env.useBrowserCache = false
  env.remoteHost = baseUrl
  env.remotePathTemplate = 'hf/{model}/resolve/{revision}/'
}

/**
 * Generation options for a Whisper model — ONLY multilingual checkpoints accept
 * `language`/`task`; passing either to an English-only (`*.en`) model makes
 * transformers.js THROW on every call (the "listening but nothing happens" bug).
 */
export function whisperGenOptions(model: string, language?: string): Record<string, unknown> | undefined {
  if (/\.en$/i.test(model)) return undefined // english-only model: language/task throw
  const opts: Record<string, unknown> = { task: 'transcribe' }
  // 'auto' (or empty) → omit language so Whisper auto-detects; otherwise pin it.
  if (language && language !== 'auto') opts.language = language
  else if (language === undefined) opts.language = 'english' // back-compat default
  return opts
}

/** Linear resample mono PCM to 16 kHz (what Whisper expects). */
export function resampleTo16k(input: Float32Array, inRate: number): Float32Array {
  if (inRate === 16000) return input
  const ratio = inRate / 16000
  const outLen = Math.max(1, Math.floor(input.length / ratio))
  const out = new Float32Array(outLen)
  for (let i = 0; i < outLen; i++) {
    const idx = i * ratio
    const i0 = Math.floor(idx)
    const i1 = Math.min(i0 + 1, input.length - 1)
    out[i] = input[i0] + (input[i1] - input[i0]) * (idx - i0)
  }
  return out
}

export const toWords = (s: string): string[] => (s.trim() ? s.trim().split(/\s+/) : [])

/** Whisper emits bracketed non-speech tokens like "[BLANK_AUDIO]" on silence. */
export const isNonSpeech = (text: string): boolean => /^[[(].*[\])]$/.test(text)

/** Compute preference: auto = GPU when available, CPU otherwise. */
export type WhisperCompute = 'auto' | 'gpu' | 'cpu'

/**
 * Per-model dtype recipe for WebGPU. Whisper encoders are extremely
 * quantization-sensitive on WebGPU (q8/fp16 encoders emit gibberish for the
 * small checkpoints) — HF's own demos pin fp32 encoder + q4 decoder; the
 * large-v3-turbo demo uses fp16 encoder (fp32 would be ~2.5 GB).
 */
export function webgpuDtype(model: string): Record<string, string> {
  const big = /large|turbo/i.test(model)
  return { encoder_model: big ? 'fp16' : 'fp32', decoder_model_merged: 'q4' }
}

/**
 * Approx. largest single GPU storage buffer (bytes) a model's encoder needs at
 * its WebGPU recipe dtype. WebGPU caps a *single* buffer binding at
 * `maxStorageBufferBindingSize`, and exceeding it is exactly the
 * "failed to allocate a buffer of size 439936716" crash (~419 MB = whisper-small's
 * fp32 encoder). We use this to choose GPU vs CPU up front so we never download
 * and attempt a model set the adapter physically can't bind. The warmup decode
 * is the final validator if this estimate is optimistic.
 */
export function encoderGpuBufferBytes(model: string): number {
  if (/large|turbo/i.test(model)) return 800_000_000 // fp16 encoder (large-v3-turbo)
  if (/small/i.test(model)) return 460_000_000 // fp32 encoder (observed ~419 MB)
  return 110_000_000 // base fp32 encoder (fits typical 128 MB-binding iGPUs)
}

/**
 * Decide GPU (webgpu) vs CPU (wasm) for a model. Pure + unit-testable.
 * - `cpu` → always wasm; `gpu` → webgpu when an adapter exists (forced: tries even
 *   if tight, surfacing a clear error rather than silently downgrading).
 * - `auto` → GPU only when the encoder's largest buffer fits the adapter's binding
 *   limit (with 5% headroom), else CPU. `maxStorageBufferBindingSize` of 0 means
 *   no adapter.
 */
export function resolveWhisperDevice(
  model: string,
  compute: WhisperCompute,
  maxStorageBufferBindingSize: number
): 'webgpu' | 'wasm' {
  if (compute === 'cpu') return 'wasm'
  const hasGpu = maxStorageBufferBindingSize > 0
  if (!hasGpu) return 'wasm'
  if (compute === 'gpu') return 'webgpu'
  return encoderGpuBufferBytes(model) <= maxStorageBufferBindingSize * 0.95 ? 'webgpu' : 'wasm'
}

/** Messages into the decode worker. */
export type WhisperWorkerIn =
  | {
      type: 'init'
      baseUrl?: string
      model: string
      language?: string
      threads: number
      compute: WhisperCompute
      /** Skip GPU entirely (a prior session's GPU warmup failed for this model). */
      forceCpu?: boolean
      /** False when Chromium has no HARDWARE WebGPU (software adapter only) — the
       * worker then stays on CPU instead of the slow/flaky software-GPU path. */
      gpuHardware?: boolean
    }
  | { type: 'audio'; buf: ArrayBuffer; rate: number }
  | { type: 'stop' }

/** Messages out of the decode worker. */
export type WhisperWorkerOut =
  | { type: 'progress'; pct: number }
  /** `downgraded` = we wanted GPU but had to fall back to CPU (so the caller can
   *  remember it and stop re-attempting/re-downloading the GPU set next time). */
  | { type: 'ready'; device: 'webgpu' | 'wasm'; downgraded?: boolean }
  | { type: 'partial'; text: string }
  | { type: 'final'; text: string }
  | { type: 'chunk-error'; count: number; message: string }
  | { type: 'init-error'; message: string }
