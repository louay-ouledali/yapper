/**
 * Whisper decode worker — ALL transformers.js / ONNX Runtime work lives here.
 *
 * Why a dedicated worker: ORT-wasm executes (and, with pthreads, busy-waits)
 * on the calling thread, so running the pipeline on the renderer main thread
 * froze the whole UI during model load, warmup, and every chunk decode. In
 * here the heaviest decode costs nothing on the UI thread.
 *
 * The worker owns utterance segmentation: the main thread just pumps raw mic
 * frames. A chunk is finalized on a short silence (or a hard cap) and decoded
 * as a `final`; while speech is still running, the current buffer is decoded
 * on a ~1.1 s cadence and emitted as a `partial` (minus the still-uncertain
 * trailing word) so Voice-Follow tracks within a second instead of waiting
 * out whole utterances. A single pump loop serializes decodes; partial decodes
 * are simply skipped while one is in flight, so latency can't snowball.
 *
 * The loaded pipeline is cached for the life of the worker — stopping Follow
 * and starting it again does not reload the model.
 */
import { pipeline, env, type AutomaticSpeechRecognitionPipeline } from '@huggingface/transformers'
import {
  whisperGenOptions,
  resampleTo16k,
  isNonSpeech,
  webgpuDtype,
  configureTransformersCache,
  resolveWhisperDevice,
  type WhisperWorkerIn,
  type WhisperWorkerOut
} from './whisper-shared'

env.allowLocalModels = false

const MIN_CHUNK_S = 0.8
const MAX_CHUNK_S = 5.0
// Tuned for non-native / accented cadence: such speakers pause mid-sentence to
// compose, and soft/aspirated consonants carry less energy. We wait a touch
// longer before finalizing (so a thinking pause doesn't sever an utterance and
// rob Whisper of context) and lower the energy gates so quiet onsets/endings
// register as speech instead of being clipped. The trailing-word holdback below
// is unchanged.
const SILENCE_HOLD_S = 0.5
const SILENCE_RMS = 0.006
const VOICE_RMS = 0.01
const PARTIAL_EVERY_MS = 1100
/** A buffer with no voiced frame yet is dropped past this age (don't decode room tone). */
const SILENT_DROP_S = 2.5

const post = (m: WhisperWorkerOut): void => {
  ;(globalThis as unknown as Worker).postMessage(m)
}

let pipe: AutomaticSpeechRecognitionPipeline | null = null
/** model@device the current pipeline was loaded for. */
let loadedKey = ''
let activeDevice: 'webgpu' | 'wasm' = 'wasm'
let genOptions: Record<string, unknown> | undefined
let frames: Float32Array[] = []
let frameLen = 0
let inRate = 16000
let silenceS = 0
let hasVoice = false
let busy = false
let wantFinal = false
let wantPartial = false
let lastPartialLen = 0
let chunkErrors = 0
let partialTimer: ReturnType<typeof setInterval> | null = null
let armed = false

function resetBuffer(): void {
  frames = []
  frameLen = 0
  silenceS = 0
  hasVoice = false
  lastPartialLen = 0
}

function concatFrames(): Float32Array {
  const out = new Float32Array(frameLen)
  let off = 0
  for (const f of frames) {
    out.set(f, off)
    off += f.length
  }
  return out
}

async function decode(audio: Float32Array): Promise<string> {
  if (!pipe) return ''
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const p = pipe as any
  const out = genOptions ? await p(audio, genOptions) : await p(audio)
  const text = (Array.isArray(out) ? out.map((o) => o.text).join(' ') : (out?.text ?? '')).trim()
  return text
}

/** Single serialized decode loop — finals take priority over cadence partials. */
async function pump(): Promise<void> {
  if (busy || !pipe || !armed) return
  busy = true
  try {
    for (;;) {
      if (wantFinal) {
        wantFinal = false
        const durS = frameLen / inRate
        const voiced = hasVoice
        const chunk = concatFrames()
        resetBuffer()
        if (durS < MIN_CHUNK_S || !voiced) continue
        try {
          const text = await decode(resampleTo16k(chunk, inRate))
          chunkErrors = 0
          if (text && !isNonSpeech(text) && armed) post({ type: 'final', text })
        } catch (e) {
          post({ type: 'chunk-error', count: ++chunkErrors, message: (e as Error)?.message?.slice(0, 160) || 'decode error' })
        }
        continue
      }
      if (wantPartial) {
        wantPartial = false
        if (!hasVoice || frameLen / inRate < MIN_CHUNK_S || frameLen === lastPartialLen) continue
        const snapshotLen = frameLen
        try {
          const text = await decode(resampleTo16k(concatFrames(), inRate))
          chunkErrors = 0
          lastPartialLen = snapshotLen
          if (text && !isNonSpeech(text) && armed) {
            // Hold back the trailing word — it is usually mid-utterance and
            // still mutating; the final (or the next partial) will deliver it.
            const w = text.split(/\s+/)
            if (w.length > 1) post({ type: 'partial', text: w.slice(0, -1).join(' ') })
          }
        } catch (e) {
          post({ type: 'chunk-error', count: ++chunkErrors, message: (e as Error)?.message?.slice(0, 160) || 'decode error' })
        }
        continue
      }
      break
    }
  } finally {
    busy = false
  }
}

function onAudio(buf: ArrayBuffer, rate: number): void {
  if (!armed || !pipe) return
  inRate = rate
  const f = new Float32Array(buf)
  frames.push(f)
  frameLen += f.length
  let sum = 0
  for (let i = 0; i < f.length; i++) sum += f[i] * f[i]
  const rms = Math.sqrt(sum / f.length)
  if (rms >= VOICE_RMS) hasVoice = true
  const frameS = f.length / rate
  silenceS = rms < SILENCE_RMS ? silenceS + frameS : 0
  const durS = frameLen / rate
  if (!hasVoice && durS >= SILENT_DROP_S) {
    resetBuffer()
    return
  }
  if (durS >= MAX_CHUNK_S || (silenceS >= SILENCE_HOLD_S && durS >= MIN_CHUNK_S && hasVoice)) {
    wantFinal = true
    void pump()
  }
}

/**
 * Probe the WebGPU adapter. Returns its single-buffer binding limit (0 = no
 * adapter) and the adapter handle, so we can both (a) decide GPU-vs-CPU by fit
 * and (b) create a device with the adapter's MAX limits.
 */
async function probeGpu(): Promise<{ maxBinding: number; adapter: unknown }> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const gpu = (navigator as any)?.gpu
    if (!gpu) return { maxBinding: 0, adapter: null }
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter) return { maxBinding: 0, adapter: null }
    // Reject a SOFTWARE fallback adapter: it answers requestAdapter but runs on
    // the CPU (slow + hangs on shader compile). Only a real hardware adapter
    // (isFallbackAdapter === false) is worth using. This gate is timing-
    // independent, unlike getGPUFeatureStatus() which reads `disabled_off` for the
    // first moment after launch and `enabled` later (the "flaky GPU detection").
    if (adapter.isFallbackAdapter) return { maxBinding: 0, adapter: null }
    const maxBinding = Number(adapter.limits?.maxStorageBufferBindingSize ?? 0)
    return { maxBinding, adapter }
  } catch {
    return { maxBinding: 0, adapter: null }
  }
}

/**
 * Create a WebGPU device from the adapter with its MAX limits and hand it to
 * ONNX Runtime. The default WebGPU device caps a single storage-buffer binding
 * at 128 MB; whisper-small's fp32 encoder needs ~419 MB, which is the
 * "failed to allocate a buffer of size 439936716" crash. Raising the limit to
 * the adapter's own maximum lets ORT bind it. Best-effort: if this doesn't take,
 * the estimate-gate + warmup validation still keep us off an unfit GPU.
 */
async function useGpuDevice(adapter: unknown): Promise<void> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const a = adapter as any
    const device = await a.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: a.limits.maxStorageBufferBindingSize,
        maxBufferSize: a.limits.maxBufferSize
      }
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ortEnv = (env.backends as any)?.onnx
    if (ortEnv?.webgpu) ortEnv.webgpu.device = device
  } catch {
    /* ORT will create its own device; warmup validation covers the rest */
  }
}

let lastPct = 0
function loadPipeline(model: string, device: 'webgpu' | 'wasm'): Promise<AutomaticSpeechRecognitionPipeline> {
  // Aggregate per-file download bytes into one smooth percentage. transformers.js
  // discovers files incrementally, so the running total GROWS as new files start —
  // a raw loaded/total then jumps BACKWARDS (the "counter bugging"). Clamp to a
  // monotonic non-decreasing value so the bar only ever moves forward.
  const dl = new Map<string, { loaded: number; total: number }>()
  lastPct = 0
  const opts: Record<string, unknown> = {
    device,
    progress_callback: (p: unknown) => {
      const d = p as { status?: string; file?: string; loaded?: number; total?: number }
      if (d?.status === 'progress' && d.file && d.total) {
        dl.set(d.file, { loaded: d.loaded ?? 0, total: d.total })
        let loaded = 0
        let total = 0
        for (const v of dl.values()) {
          loaded += v.loaded
          total += v.total
        }
        if (total) {
          const raw = Math.min(100, Math.round((loaded / total) * 100))
          lastPct = Math.max(lastPct, raw)
          post({ type: 'progress', pct: lastPct })
        }
      }
    }
  }
  // WebGPU needs the per-model dtype recipe — default quantized encoders emit
  // gibberish there (the original "garbage output" bug). WASM keeps defaults.
  if (device === 'webgpu') opts.dtype = webgpuDtype(model)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const make = pipeline as any
  return make('automatic-speech-recognition', model, opts) as Promise<AutomaticSpeechRecognitionPipeline>
}

async function init(msg: Extract<WhisperWorkerIn, { type: 'init' }>): Promise<void> {
  try {
    // Route model-weight fetches through the main-process loopback cache so they
    // download once and load offline (even on the packaged file:// origin).
    configureTransformersCache(env, msg.baseUrl)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const wasm = (env.backends as any)?.onnx?.wasm
    if (wasm) {
      if (msg.baseUrl) wasm.wasmPaths = `${msg.baseUrl}/ort/`
      wasm.numThreads = msg.threads
    }

    // GPU gate: use WebGPU when a real HARDWARE adapter exists (probeGpu rejects
    // software fallbacks). `forceCpu` (a prior warmup failed) and `compute:'cpu'`
    // keep us on CPU. We deliberately do NOT gate on the main process's
    // getGPUFeatureStatus() — it reports `disabled_off` for the first instant after
    // launch and `enabled` later, which made detection flaky.
    const gpuAllowed = msg.compute !== 'cpu' && !msg.forceCpu
    const probe = gpuAllowed ? await probeGpu() : { maxBinding: 0, adapter: null }
    if (msg.compute === 'gpu' && !probe.adapter) {
      post({ type: 'init-error', message: 'no hardware GPU available — set Compute to Auto or CPU' })
      return
    }
    const want = msg.forceCpu ? 'wasm' : resolveWhisperDevice(msg.model, msg.compute, probe.maxBinding)

    // Load + warm a pipeline for a device. The warmup decode doubles as
    // validation: a buffer-alloc failure or gibberish on GPU shows up here (at
    // decode), not at load — on webgpu that THROWS so the caller can downgrade;
    // on wasm the warmup output is simply irrelevant.
    const loadAndValidate = async (device: 'webgpu' | 'wasm'): Promise<void> => {
      pipe = null
      if (device === 'webgpu' && probe.adapter) await useGpuDevice(probe.adapter)
      pipe = await loadPipeline(msg.model, device)
      genOptions = whisperGenOptions(msg.model, msg.language)
      try {
        // Bound the warmup: a software-GPU shader compile (or a wedged backend)
        // can hang here forever, which presents as the engine stuck on "loading".
        // A timeout on webgpu forces the CPU downgrade; on wasm we just move on.
        const warmupMs = device === 'webgpu' ? 25_000 : 45_000
        await Promise.race([
          decode(new Float32Array(8000)),
          new Promise((_, rej) => setTimeout(() => rej(new Error('warmup timed out')), warmupMs))
        ])
      } catch (e) {
        if (device === 'webgpu') {
          pipe = null
          throw e
        }
      }
      activeDevice = device
      loadedKey = `${msg.model}@${device}`
    }

    let downgraded = false
    const key = `${msg.model}@${want}`
    if (!pipe || loadedKey !== key) {
      try {
        await loadAndValidate(want)
      } catch (e) {
        // Auto: a GPU that can't load/decode this model falls back to CPU and we
        // tell the caller (so it remembers and stops re-attempting the GPU set).
        if (want === 'webgpu' && msg.compute === 'auto') {
          await loadAndValidate('wasm')
          downgraded = true
        } else {
          throw e
        }
      }
    } else {
      genOptions = whisperGenOptions(msg.model, msg.language)
    }

    resetBuffer()
    chunkErrors = 0
    armed = true
    if (!partialTimer) {
      partialTimer = setInterval(() => {
        wantPartial = true
        void pump()
      }, PARTIAL_EVERY_MS)
    }
    post({ type: 'ready', device: activeDevice, downgraded })
  } catch (e) {
    post({ type: 'init-error', message: (e as Error)?.message?.slice(0, 200) || 'model load failed' })
  }
}

;(globalThis as unknown as Worker).onmessage = (ev: MessageEvent<WhisperWorkerIn>) => {
  const msg = ev.data
  if (msg.type === 'audio') onAudio(msg.buf, msg.rate)
  else if (msg.type === 'init') void init(msg)
  else if (msg.type === 'stop') {
    armed = false
    wantFinal = false
    wantPartial = false
    resetBuffer()
    if (partialTimer) {
      clearInterval(partialTimer)
      partialTimer = null
    }
  }
}
