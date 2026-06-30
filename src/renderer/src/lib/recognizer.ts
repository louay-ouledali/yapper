/**
 * Speech recognizer abstraction. Voice-Follow consumes words from a Recognizer;
 * the engine, drift recovery, and UI don't care where they come from.
 *
 * Implementation: offline Vosk-WASM (`vosk-browser`) streaming the microphone
 * through a ScriptProcessor at the AudioContext sample rate. The model is served
 * locally by the main process (no network). Verified with a live mic by the user
 * (a mic can't be exercised in the build environment), so failures surface as a
 * clear status string and Simulate mode covers the engine path mic-free.
 */
export type RecognizerStatus = 'idle' | 'loading' | 'listening' | 'error'

export interface RecognizerHooks {
  onPartial?: (words: string[]) => void
  onFinal?: (words: string[], text: string) => void
  onStatus?: (status: RecognizerStatus, detail?: string) => void
  /** Live mic level (RMS, ~0..0.5) per audio frame — drive a meter with this. */
  onLevel?: (rms: number) => void
  /** Fires once the microphone stream is open (e.g. to start a recorder on it). */
  onStream?: (stream: MediaStream) => void
}

export interface Recognizer {
  start: () => Promise<void>
  stop: () => Promise<void>
}

const words = (s: string): string[] => (s.trim() ? s.trim().split(/\s+/) : [])

// vosk-browser worker log level (see its Logger): -1 silent, 0 info, 1 verbose,
// 2 debug. `error()` always logs regardless of level, so a failed load is never
// silent; bump to 1 to trace each load step (storage → download → extract → model)
// when diagnosing a stall.
const VOSK_LOG_LEVEL = 0

/** Reject if a promise hasn't settled in `ms` — turns a silent hang into a clear error. */
function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  return Promise.race([
    p,
    new Promise<T>((_, reject) => setTimeout(() => reject(new Error(`${label} timed out after ${ms / 1000}s`)), ms))
  ])
}

/** A loaded vosk model: enough of the surface for us to build a recognizer + clean up. */
interface VoskModel {
  ready?: boolean
  terminate?: () => void
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  KaldiRecognizer: new (sampleRate: number, grammar?: string) => any
}

/**
 * Load a vosk model, resolving on success and REJECTING on failure.
 *
 * vosk-browser's own `createModel` only listens for the worker's `load` event and
 * silently ignores `error`, so any load failure (bad fetch, decode, FS) leaves its
 * promise pending forever — the "hangs for 90s then times out" symptom. We build the
 * `Model` directly instead and settle on whichever of `load`/`error` arrives first.
 */
function loadVoskModel(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ModelCtor: new (url: string, logLevel?: number) => any,
  url: string
): Promise<VoskModel> {
  return new Promise<VoskModel>((resolve, reject) => {
    let settled = false
    const model = new ModelCtor(url, VOSK_LOG_LEVEL)
    const done = (fn: () => void): void => {
      if (settled) return
      settled = true
      fn()
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    model.on('load', (m: any) =>
      done(() => {
        if (m?.result) resolve(model as VoskModel)
        else {
          try {
            model.terminate?.()
          } catch {
            /* ignore */
          }
          reject(new Error('model worker reported load failure'))
        }
      })
    )
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    model.on('error', (m: any) =>
      done(() => {
        try {
          model.terminate?.()
        } catch {
          /* ignore */
        }
        reject(new Error(m?.error || 'model worker error'))
      })
    )
  })
}

export function createVoskRecognizer(modelUrl: string, hooks: RecognizerHooks): Recognizer {
  let model: VoskModel | null = null
  let media: MediaStream | null = null
  let ctx: AudioContext | null = null
  let node: ScriptProcessorNode | null = null
  let source: MediaStreamAudioSourceNode | null = null

  return {
    async start() {
      hooks.onStatus?.('loading')
      try {
        // vosk-browser is a UMD/CJS module. Depending on how it's bundled its
        // exports can land on the namespace, under `.default`, or (UMD global
        // fallback) on `globalThis.Vosk`. Resolve the `Model` ctor from any of them.
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const mod: any = await import('vosk-browser')
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const g = globalThis as any
        const ModelCtor = mod?.Model ?? mod?.default?.Model ?? g?.Vosk?.Model
        if (typeof ModelCtor !== 'function') {
          throw new Error('vosk-browser Model export not found')
        }
        console.info('[voice-follow] loading model from', modelUrl)
        model = await withTimeout(loadVoskModel(ModelCtor, modelUrl), 90000, 'voice model load')
        console.info('[voice-follow] model loaded; requesting microphone')

        media = await withTimeout(
          navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
            video: false
          }),
          20000,
          'microphone access'
        )
        console.info('[voice-follow] microphone granted; starting recognizer')
        hooks.onStream?.(media)
        ctx = new AudioContext()

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const recognizer: any = new model.KaldiRecognizer(ctx.sampleRate)
        recognizer.setWords(true)
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        recognizer.on('result', (m: any) => {
          const text: string = m?.result?.text ?? ''
          if (text.trim()) hooks.onFinal?.(words(text), text)
        })
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        recognizer.on('partialresult', (m: any) => {
          hooks.onPartial?.(words(m?.result?.partial ?? ''))
        })

        node = ctx.createScriptProcessor(4096, 1, 1)
        node.onaudioprocess = (e) => {
          if (hooks.onLevel) {
            const ch = e.inputBuffer.getChannelData(0)
            let sum = 0
            for (let i = 0; i < ch.length; i++) sum += ch[i] * ch[i]
            hooks.onLevel(Math.sqrt(sum / ch.length))
          }
          // Mute-in: the first instants after the mic opens are clicks/AGC
          // settling that produce junk partials — drop them.
          if (ctx && ctx.currentTime < 0.3) return
          try {
            recognizer.acceptWaveform(e.inputBuffer)
          } catch {
            /* transient buffer error — ignore */
          }
        }
        source = ctx.createMediaStreamSource(media)
        source.connect(node)
        node.connect(ctx.destination)
        console.info('[voice-follow] listening')
        hooks.onStatus?.('listening')
      } catch (e) {
        const err = e as { name?: string; message?: string }
        // Surface the real reason — the generic "could not load" hid the cause.
        // Full object goes to the console for deeper diagnosis.
        console.error('[voice-follow] recognizer failed to start:', e)
        const raw = (err?.message || String(e) || '').trim()
        const detail =
          err?.name === 'NotAllowedError'
            ? 'microphone permission denied'
            : /fetch|network|404|load|model/i.test(raw)
              ? `could not load the voice model: ${raw || 'unknown error'}`
              : raw || 'voice engine error'
        hooks.onStatus?.('error', detail)
      }
    },
    async stop() {
      try {
        if (node) node.onaudioprocess = null
        source?.disconnect()
        node?.disconnect()
        media?.getTracks().forEach((t) => t.stop())
        await ctx?.close()
        model?.terminate?.()
      } catch {
        /* ignore */
      }
      hooks.onStatus?.('idle')
    }
  }
}
