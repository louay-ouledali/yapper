/**
 * Whisper ASR engine (high-accuracy option) via transformers.js / ONNX Runtime Web.
 *
 * The live recognizer runs the entire pipeline in a dedicated Web Worker
 * (whisper.worker.ts): model load, warmup, and every chunk decode happen off the
 * UI thread, because ORT-wasm blocks (and with pthreads, busy-waits) the thread
 * it runs on — on the main thread that froze the whole app. The worker is a
 * module-level singleton so the loaded model survives stop/start: the second
 * Follow session starts instantly.
 *
 * This file keeps only main-thread concerns: microphone capture (frames are
 * transferred to the worker) and status plumbing. It must NOT import
 * transformers.js — that costs ~1s of main-thread parse/compile, which is
 * exactly the freeze this design removes (batch use lives in whisper-batch.ts).
 */
import type { Recognizer, RecognizerHooks } from './recognizer'
import { toWords, type WhisperCompute, type WhisperWorkerIn, type WhisperWorkerOut } from './whisper-shared'
import WhisperWorkerCtor from './whisper.worker?worker&inline'

export type { WhisperCompute } from './whisper-shared'

export { whisperGenOptions } from './whisper-shared'

export type WhisperDevice = 'auto' | 'webgpu' | 'wasm'
export const DEFAULT_WHISPER_MODEL = 'onnx-community/whisper-base'

/** Mic frames in the first instants are clicks/AGC settling — don't decode them. */
const MUTE_IN_S = 0.3

const threadCount = (): number =>
  typeof SharedArrayBuffer !== 'undefined'
    ? Math.max(1, Math.min(4, Math.floor((navigator.hardwareConcurrency || 2) / 2)))
    : 1

// ────────────────────────────────────────────────────────────────────────────
// Live recognizer (worker-backed)

let worker: Worker | null = null

function getWorker(): Worker {
  if (!worker) worker = new WhisperWorkerCtor()
  return worker
}

export interface WhisperOptions {
  model?: string
  device?: WhisperDevice
  /** Compute preference: auto = GPU (WebGPU) when available, CPU otherwise. */
  compute?: WhisperCompute
  /** Whisper language hint (e.g. 'english', 'french') for multilingual models. */
  language?: string
  /** Skip GPU entirely — this model's GPU warmup failed on a prior run. */
  forceCpu?: boolean
  /** Reports the device the engine actually settled on, and whether it had to
   *  downgrade GPU→CPU, so the caller can remember it and surface it. */
  onResolved?: (info: { device: WhisperDevice; downgraded: boolean; model: string }) => void
}

/**
 * Download + load (but do NOT start the mic) a Whisper model, so the one-time
 * weight download happens in Settings instead of mid-presentation. The worker
 * model stays resident, so the next real Follow/Score/Q&A start is instant.
 * Progress for the download itself arrives via the main `onHfProgress` channel;
 * this `onProgress` reports the (fast, local) load percentage afterwards.
 */
export async function prewarmWhisper(
  opts: WhisperOptions,
  onProgress?: (pct: number) => void
): Promise<{ device: WhisperDevice }> {
  const w = getWorker()
  const model = opts.model ?? DEFAULT_WHISPER_MODEL
  const info = await window.yapper?.modelInfo().catch(() => null)
  return new Promise<{ device: WhisperDevice }>((resolve, reject) => {
    const prev = w.onmessage
    w.onmessage = (ev: MessageEvent<WhisperWorkerOut>): void => {
      const m = ev.data
      if (m.type === 'progress') onProgress?.(m.pct)
      else if (m.type === 'ready') {
        w.onmessage = prev
        // Disarm immediately — we only wanted the model resident, not listening.
        w.postMessage({ type: 'stop' } satisfies WhisperWorkerIn)
        opts.onResolved?.({ device: m.device, downgraded: !!m.downgraded, model })
        resolve({ device: m.device })
      } else if (m.type === 'init-error') {
        w.onmessage = prev
        reject(new Error(m.message))
      }
    }
    w.postMessage({
      type: 'init',
      baseUrl: info?.baseUrl,
      model,
      language: opts.language,
      threads: threadCount(),
      compute: opts.compute ?? 'auto',
      forceCpu: opts.forceCpu,
      gpuHardware: info?.gpuHardware
    } satisfies WhisperWorkerIn)
  })
}

/** A microphone Recognizer backed by Whisper — same interface as the Vosk one. */
export function createWhisperRecognizer(opts: WhisperOptions, hooks: RecognizerHooks): Recognizer {
  let media: MediaStream | null = null
  let ctx: AudioContext | null = null
  let node: ScriptProcessorNode | null = null
  let source: MediaStreamAudioSourceNode | null = null
  let stopped = false

  const model = opts.model ?? DEFAULT_WHISPER_MODEL

  return {
    async start() {
      hooks.onStatus?.('loading')
      try {
        const w = getWorker()
        // (assigned inside the ready handler — keep the union un-narrowed)
        let device = 'wasm' as 'webgpu' | 'wasm'
        const ready = new Promise<void>((resolve, reject) => {
          w.onmessage = (ev: MessageEvent<WhisperWorkerOut>) => {
            const m = ev.data
            if (m.type === 'progress') {
              hooks.onStatus?.('loading', `downloading model… ${m.pct}% · first use only`)
            } else if (m.type === 'ready') {
              device = m.device
              opts.onResolved?.({ device: m.device, downgraded: !!m.downgraded, model })
              resolve()
            } else if (m.type === 'init-error') {
              reject(new Error(m.message))
            } else if (m.type === 'partial') {
              if (!stopped) hooks.onPartial?.(toWords(m.text))
            } else if (m.type === 'final') {
              if (!stopped) hooks.onFinal?.(toWords(m.text), m.text)
            } else if (m.type === 'chunk-error') {
              // One failed chunk can be transient; repeated failures mean the
              // engine is effectively dead — say so instead of sitting on
              // "listening" forever.
              console.error('[whisper] transcription error:', m.message)
              if (m.count >= 3 && !stopped) hooks.onStatus?.('error', `Whisper failed: ${m.message}`)
            }
          }
        })
        const info = await window.yapper?.modelInfo().catch(() => null)
        w.postMessage({
          type: 'init',
          baseUrl: info?.baseUrl,
          model,
          language: opts.language,
          threads: threadCount(),
          compute: opts.compute ?? 'auto',
          forceCpu: opts.forceCpu,
          gpuHardware: info?.gpuHardware
        } satisfies WhisperWorkerIn)
        await ready
        if (stopped) return

        hooks.onStatus?.('loading', 'starting microphone…')
        media = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
          video: false
        })
        if (stopped) {
          media.getTracks().forEach((t) => t.stop())
          return
        }
        hooks.onStream?.(media)
        ctx = new AudioContext()
        const rate = ctx.sampleRate
        node = ctx.createScriptProcessor(4096, 1, 1)
        node.onaudioprocess = (e): void => {
          if (stopped || !ctx) return
          const ch = e.inputBuffer.getChannelData(0)
          if (hooks.onLevel) {
            let sum = 0
            for (let i = 0; i < ch.length; i++) sum += ch[i] * ch[i]
            hooks.onLevel(Math.sqrt(sum / ch.length))
          }
          if (ctx.currentTime < MUTE_IN_S) return
          // Copy before transfer — the buffer is neutered by postMessage.
          const f = new Float32Array(ch)
          w.postMessage({ type: 'audio', buf: f.buffer, rate } satisfies WhisperWorkerIn, [f.buffer])
        }
        source = ctx.createMediaStreamSource(media)
        source.connect(node)
        node.connect(ctx.destination)
        hooks.onStatus?.('listening', device === 'webgpu' ? 'listening · GPU' : 'listening · CPU')
      } catch (e) {
        const err = e as { name?: string; message?: string }
        console.error('[whisper] recognizer failed to start:', e)
        const raw = (err?.message || String(e) || '').trim()
        const detail =
          err?.name === 'NotAllowedError'
            ? 'microphone permission denied'
            : /fetch|network|load|model|onnx|wasm/i.test(raw)
              ? `could not load the Whisper model: ${raw || 'unknown error'}`
              : raw || 'whisper engine error'
        hooks.onStatus?.('error', detail)
      }
    },
    async stop() {
      stopped = true
      try {
        if (node) node.onaudioprocess = null
        source?.disconnect()
        node?.disconnect()
        media?.getTracks().forEach((t) => t.stop())
        await ctx?.close()
      } catch {
        /* ignore */
      }
      // The worker stays alive with the model loaded — only its buffers reset.
      worker?.postMessage({ type: 'stop' } satisfies WhisperWorkerIn)
      hooks.onStatus?.('idle')
    }
  }
}
