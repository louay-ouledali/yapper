/**
 * Whisper batch transcription (used by the dictation pipeline). Mirrors the live
 * worker's WebGPU setup so GPU actually works for every model:
 *  - probe a real hardware adapter and raise ORT's storage-buffer limit to the
 *    adapter max (else large encoders fail with "couldn't allocate a buffer…"),
 *  - apply the per-model dtype recipe (default quantized encoders emit gibberish
 *    on WebGPU),
 *  - warm up with a tiny decode under a timeout and fall back to CPU if the GPU
 *    can't handle the model (no more "loads to 100% then hangs forever"),
 *  - chunk long audio (chunk_length_s) so clips over 30 s transcribe in FULL.
 */
import { pipeline, env, type AutomaticSpeechRecognitionPipeline } from '@huggingface/transformers'
import { whisperGenOptions, configureTransformersCache, webgpuDtype, resolveWhisperDevice, type WhisperCompute } from './whisper-shared'
import { DEFAULT_WHISPER_MODEL, type WhisperDevice } from './whisper'

env.allowLocalModels = false

// CPU decode is thread-bound; give it most of the cores (leave one for the UI/mic)
// so a Small model on the wasm path isn't needlessly throttled.
const threadCount = (): number =>
  typeof SharedArrayBuffer !== 'undefined' ? Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 2) - 1)) : 1

let ortConfigured = false
async function configureOrt(): Promise<void> {
  if (ortConfigured) return
  ortConfigured = true
  try {
    const info = await window.yapper?.modelInfo()
    configureTransformersCache(env, info?.baseUrl)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const wasm = (env.backends as any)?.onnx?.wasm
    if (wasm) {
      if (info?.baseUrl) wasm.wasmPaths = `${info.baseUrl}/ort/`
      wasm.numThreads = threadCount()
    }
  } catch {
    /* fall back to defaults */
  }
}

function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const t = setTimeout(() => reject(new Error('timeout')), ms)
    p.then(
      (v) => {
        clearTimeout(t)
        resolve(v)
      },
      (e) => {
        clearTimeout(t)
        reject(e)
      }
    )
  })
}

/** Map the user's compute preference to the resolver's vocabulary. */
function toCompute(d?: WhisperDevice): WhisperCompute {
  if (d === 'wasm') return 'cpu'
  if (d === 'webgpu') return 'gpu'
  return 'auto'
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function probeGpu(): Promise<{ maxBinding: number; adapter: any }> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const gpu = (navigator as any)?.gpu
    if (!gpu) return { maxBinding: 0, adapter: null }
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter || adapter.isFallbackAdapter) return { maxBinding: 0, adapter: null }
    return { maxBinding: Number(adapter.limits?.maxStorageBufferBindingSize ?? 0), adapter }
  } catch {
    return { maxBinding: 0, adapter: null }
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function useGpuDevice(adapter: any): Promise<void> {
  try {
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize
      }
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ortEnv = (env.backends as any)?.onnx
    if (ortEnv?.webgpu) ortEnv.webgpu.device = device
  } catch {
    /* ORT makes its own device; warmup validation covers the rest */
  }
}

function loadPipeline(model: string, device: 'webgpu' | 'wasm', onProgress?: (pct: number) => void): Promise<AutomaticSpeechRecognitionPipeline> {
  const dl = new Map<string, { loaded: number; total: number }>()
  let lastPct = 0
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
          lastPct = Math.max(lastPct, Math.min(100, Math.round((loaded / total) * 100)))
          onProgress?.(lastPct)
        }
      }
    }
  }
  if (device === 'webgpu') opts.dtype = webgpuDtype(model)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const make = pipeline as any
  return make('automatic-speech-recognition', model, opts) as Promise<AutomaticSpeechRecognitionPipeline>
}

let cached: { key: string; device: 'webgpu' | 'wasm'; pipe: AutomaticSpeechRecognitionPipeline } | null = null
/** Final device decision per model — set once so we NEVER re-probe or re-attempt a
 *  device that already failed (the bug that re-ran the 12 s GPU warmup every dictation). */
const decided = new Map<string, 'webgpu' | 'wasm'>()
/** Whether a real hardware GPU adapter was seen on the last probe — used to detect a
 *  "wanted GPU but fell back to CPU" downgrade (e.g. the model's encoder is too big to bind). */
let gpuSeen = false

async function loadFor(model: string, device: 'webgpu' | 'wasm', warmupMs: number, onProgress?: (pct: number) => void): Promise<AutomaticSpeechRecognitionPipeline> {
  if (device === 'webgpu') {
    const probe = await probeGpu()
    if (probe.adapter) await useGpuDevice(probe.adapter)
    const pipe = await loadPipeline(model, 'webgpu', onProgress)
    // Validate the GPU can actually run this model — a silent decode under a timeout.
    // Generous during background prewarm (first-run shader compile is slow), short in the
    // foreground. A throw/timeout means the caller pins this model to CPU.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    await withTimeout((pipe as any)(new Float32Array(16000), { chunk_length_s: 30 }), warmupMs)
    return pipe
  }
  return loadPipeline(model, 'wasm', onProgress)
}

/** Load (and memoize) an ASR pipeline. Decides GPU vs CPU ONCE per model and reuses it, so
 *  every later call is a cache hit — no repeated probing or warmups. `warmupMs` is the GPU
 *  validation budget (pass a large value from the background prewarm so WebGPU can compile). */
export async function getTranscriber(
  model: string = DEFAULT_WHISPER_MODEL,
  devicePref: WhisperDevice = 'auto',
  onProgress?: (pct: number) => void,
  warmupMs = 12000
): Promise<{ pipe: AutomaticSpeechRecognitionPipeline; device: 'webgpu' | 'wasm'; downgraded: boolean }> {
  await configureOrt()
  const compute = toCompute(devicePref)

  // Decide the device: explicit CPU → wasm; already decided → reuse; else probe once.
  let device: 'webgpu' | 'wasm'
  if (compute === 'cpu') device = 'wasm'
  else if (decided.has(model)) device = decided.get(model)!
  else {
    const probe = await probeGpu()
    gpuSeen = probe.maxBinding > 0
    device = resolveWhisperDevice(model, compute, probe.maxBinding)
  }
  // "Wanted GPU, running on CPU, and a GPU exists" → a downgrade the user should see
  // (usually the model's encoder is too big to bind on this adapter).
  const downgraded = compute !== 'cpu' && device === 'wasm' && gpuSeen

  if (cached && cached.key === `${model}@${device}`) return { pipe: cached.pipe, device, downgraded }

  if (device === 'webgpu') {
    try {
      const pipe = await loadFor(model, 'webgpu', warmupMs, onProgress)
      decided.set(model, 'webgpu')
      cached = { key: `${model}@webgpu`, device: 'webgpu', pipe }
      return { pipe, device: 'webgpu', downgraded: false }
    } catch {
      decided.set(model, 'wasm') // GPU failed for this model — never retry it this session
    }
  }

  const downgradedFinal = compute !== 'cpu' && gpuSeen
  if (cached && cached.key === `${model}@wasm`) return { pipe: cached.pipe, device: 'wasm', downgraded: downgradedFinal }
  const pipe = await loadFor(model, 'wasm', 0, onProgress)
  decided.set(model, 'wasm')
  cached = { key: `${model}@wasm`, device: 'wasm', pipe }
  return { pipe, device: 'wasm', downgraded: downgradedFinal }
}

/** Transcribe a Float32Array (mono, 16 kHz) or URL. Returns plain text.
 *  `onProgress` = model download %, `onDevice` = the device actually used. */
export async function transcribe(
  audio: string | Float32Array,
  model?: string,
  devicePref?: WhisperDevice,
  onProgress?: (pct: number) => void,
  language?: string,
  onDevice?: (device: 'webgpu' | 'wasm', downgraded: boolean) => void
): Promise<string> {
  const { pipe, device, downgraded } = await getTranscriber(model, devicePref, onProgress)
  onDevice?.(device, downgraded)
  // chunk_length_s makes transformers.js process the WHOLE clip (not just the
  // first 30 s Whisper window); stride overlaps chunks so words aren't lost.
  const genOptions: Record<string, unknown> = {
    ...(whisperGenOptions(model ?? DEFAULT_WHISPER_MODEL, language) ?? {}),
    chunk_length_s: 30,
    stride_length_s: 5
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const out = await (pipe as any)(audio, genOptions)
  const text = Array.isArray(out) ? out.map((o) => o.text).join(' ') : out?.text
  return (text ?? '').trim()
}
