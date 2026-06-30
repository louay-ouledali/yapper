/**
 * Yapper's core: record high-quality audio, then DELIBERATELY process it offline —
 * batch Whisper transcription, then an AI cleanup pass (remove fillers/stutters,
 * fix grammar, correct mis-heard words from context). Not real-time on purpose.
 */
import { resampleTo16k } from './whisper-shared'
import { transcribe } from './whisper-batch'
import { DEFAULT_WHISPER_MODEL, type WhisperDevice } from './whisper'
import { assistStream, brainToLlmConfig, type AiBrain, type ChatMessage } from './llm'

/** A mode id is just a string — modes are user-definable (see DEFAULT_MODES + settings). */
export type DictationMode = string

/** The id of the built-in "no AI" mode (returns the untouched transcript). */
export const RAW_MODE_ID = 'raw'

/** A delivery/cleanup mode: the final form the dictation is turned into. An empty `prompt`
 *  means "no AI" (raw). `builtin` modes can be edited & reset but not deleted. */
export interface CleanupMode {
  id: string
  label: string
  prompt: string
  builtin?: boolean
}

/** Shared tail appended to every mode prompt to keep output clean and proportional. */
const PROMPT_TAIL =
  ' Keep the result roughly the same length and structure as the input — do not pad, expand, or add content that was not said. Write in a plain, natural human tone (not flowery or obviously AI). Reply with ONLY the result text — no preamble, no explanation, no quotes, no thinking.'

export const DEFAULT_MODES: CleanupMode[] = [
  {
    id: 'clean',
    label: 'Clean-up',
    builtin: true,
    prompt:
      'Clean up this dictation: remove filler words (um, uh, ah, like), false starts, stutters and accidentally repeated words. When the speaker corrects themselves or contradicts what they just said, keep only the corrected, final version. Fix grammar, capitalization and punctuation, and fix words that are clearly mis-transcribed using the context. Preserve the speaker’s meaning, tone and wording — do not add new content.'
  },
  {
    id: 'prompt',
    label: 'AI prompt',
    builtin: true,
    prompt:
      'Rewrite this dictation as a clear, concise prompt for an AI assistant. State the goal or instruction plainly and keep every requirement the speaker gave, but stay proportional to what they said — a short request stays a short prompt. Do not answer it; output only the prompt.'
  },
  {
    id: 'email',
    label: 'Email',
    builtin: true,
    prompt:
      'Turn this dictation into an email: a short, specific subject line (prefixed with "Subject: "), then a tidy body. Keep the speaker’s intent, tone and length; fix grammar; do not invent facts. Only add a greeting or sign-off if the speaker dictated one.'
  },
  {
    id: 'notes',
    label: 'Notes',
    builtin: true,
    prompt: 'Condense this dictation into concise bullet-point notes capturing the key information. Fix grammar and drop fillers.'
  },
  { id: RAW_MODE_ID, label: 'Raw', builtin: true, prompt: '' }
]

/** Build the cleanup chat messages from a resolved prompt rule. */
export function cleanupMessages(raw: string, prompt: string): ChatMessage[] {
  return [
    {
      role: 'system',
      content:
        'You post-process dictation. A person spoke and an offline speech-to-text engine produced the transcript below — it may contain disfluencies and mis-heard words. ' +
        prompt +
        PROMPT_TAIL
    },
    { role: 'user', content: raw }
  ]
}

/** Strip a model's reasoning/“thinking” and wrapping quotes from an answer.
 *  Conservative: it only removes complete reasoning blocks (or a reply that is
 *  ENTIRELY an unterminated reasoning block) — it never drops trailing content. */
export function stripModelArtifacts(text: string): string {
  let t = text
  // Complete reasoning blocks emitted by hybrid/reasoning models despite think:false.
  t = t.replace(/<think>[\s\S]*?<\/think>/gi, '')
  t = t.replace(/◁think▷[\s\S]*?◁\/think▷/gi, '')
  t = t.replace(/<(thinking|reasoning)>[\s\S]*?<\/\1>/gi, '')
  // Only when the WHOLE reply is an unterminated reasoning block is there no usable
  // answer — return empty so the caller falls back to the raw transcript.
  const lead = t.replace(/^\s+/, '')
  if (/^(<think>|◁think▷|<thinking>|<reasoning>)/i.test(lead) && !/<\/think>|◁\/think▷|<\/thinking>|<\/reasoning>/i.test(t)) t = ''
  t = t.trim()
  // Unwrap a fully quote-wrapped reply.
  if ((t.startsWith('"') && t.endsWith('"')) || (t.startsWith('“') && t.endsWith('”'))) t = t.slice(1, -1).trim()
  return t
}

/** Why a cleanup pass produced the text it did — surfaced in the UI so nothing fails silently. */
export type CleanStatus = 'ok' | 'off' | 'raw' | 'error' | 'empty'
export interface CleanOutcome {
  text: string
  status: CleanStatus
  error?: string
}

/** Run an AI cleanup pass over a transcript with a resolved prompt, reporting WHAT happened.
 *  Falls back to the raw transcript for raw mode / disabled brain / errors — but says so. */
export async function cleanTranscript(transcript: string, brain: AiBrain, prompt?: string, signal?: AbortSignal): Promise<CleanOutcome> {
  if (!transcript.trim()) return { text: transcript, status: 'empty' }
  if (!prompt || !prompt.trim()) return { text: transcript, status: 'raw' }
  if (!brain.enabled) return { text: transcript, status: 'off' }
  try {
    // assistStream (vs assist) lets a cancel abort the pass mid-generation via the signal.
    const res = await assistStream(brainToLlmConfig(brain, 'live'), cleanupMessages(transcript, prompt), () => {}, signal)
    if (res.ok && res.text) {
      const cleaned = stripModelArtifacts(res.text)
      if (cleaned) return { text: cleaned, status: 'ok' }
      return { text: transcript, status: 'error', error: 'AI returned an empty reply' }
    }
    return { text: transcript, status: 'error', error: res.error || 'AI returned nothing' }
  } catch (e) {
    return { text: transcript, status: 'error', error: (e as Error).message }
  }
}

const MIME_CANDIDATES = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus']
function pickMime(): string {
  for (const c of MIME_CANDIDATES) {
    try {
      if (MediaRecorder.isTypeSupported(c)) return c
    } catch {
      /* ignore */
    }
  }
  return ''
}

export interface RecorderOpts {
  /** Live mic level (RMS, ~0–0.4) for a VU meter while recording. */
  onLevel?: (rms: number) => void
}

/** High-quality mic capture via MediaRecorder. The blob is kept (for retention) AND decoded for transcription. */
export class VoiceRecorder {
  private rec: MediaRecorder | null = null
  private chunks: Blob[] = []
  private stream: MediaStream | null = null
  private onLevel?: (rms: number) => void
  private meterCtx: AudioContext | null = null
  private rafId = 0

  constructor(opts?: RecorderOpts) {
    this.onLevel = opts?.onLevel
  }

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: true }
    })
    this.chunks = []
    const mimeType = pickMime()
    this.rec = new MediaRecorder(this.stream, mimeType ? { mimeType } : undefined)
    this.rec.ondataavailable = (e): void => {
      if (e.data && e.data.size) this.chunks.push(e.data)
    }
    this.rec.start()
    this.startMeter()
  }

  /** Stop and return the recording. `tailMs` keeps capturing briefly before stopping so
   *  trailing words aren't clipped when the user releases the key mid-word. */
  stop(tailMs = 250): Promise<Blob> {
    return new Promise((resolve) => {
      const r = this.rec
      if (!r) {
        this.stopMeter()
        return resolve(new Blob())
      }
      r.onstop = (): void => {
        this.stream?.getTracks().forEach((t) => t.stop())
        this.stream = null
        this.rec = null
        resolve(new Blob(this.chunks, { type: r.mimeType || 'audio/webm' }))
      }
      const finalize = (): void => {
        this.stopMeter()
        r.stop()
      }
      if (tailMs > 0) setTimeout(finalize, tailMs)
      else finalize()
    })
  }

  get active(): boolean {
    return this.rec?.state === 'recording'
  }

  private startMeter(): void {
    if (!this.onLevel || !this.stream) return
    try {
      const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
      const ctx = new Ctx()
      this.meterCtx = ctx
      const src = ctx.createMediaStreamSource(this.stream)
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 512
      src.connect(analyser)
      const data = new Float32Array(analyser.fftSize)
      const tick = (): void => {
        analyser.getFloatTimeDomainData(data)
        let sum = 0
        for (let i = 0; i < data.length; i++) sum += data[i] * data[i]
        this.onLevel?.(Math.sqrt(sum / data.length))
        this.rafId = requestAnimationFrame(tick)
      }
      this.rafId = requestAnimationFrame(tick)
    } catch {
      /* meter is best-effort */
    }
  }

  private stopMeter(): void {
    if (this.rafId) cancelAnimationFrame(this.rafId)
    this.rafId = 0
    this.onLevel?.(0)
    void this.meterCtx?.close().catch(() => {})
    this.meterCtx = null
  }
}

/** Decode a recorded blob to mono 16 kHz Float32 samples (what Whisper expects). */
export async function decodeTo16k(blob: Blob): Promise<Float32Array> {
  const buf = await blob.arrayBuffer()
  const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
  const ctx = new Ctx()
  try {
    const audio = await ctx.decodeAudioData(buf.slice(0))
    return resampleTo16k(audio.getChannelData(0), audio.sampleRate)
  } finally {
    await ctx.close().catch(() => {})
  }
}

export interface ProcessOpts {
  brain: AiBrain
  model?: string
  device?: WhisperDevice
  /** Whisper language ('auto' = detect; undefined = English). */
  language?: string
  /** Resolved prompt for the chosen mode. Empty/undefined = raw (no AI). */
  prompt?: string
  onPhase?: (phase: 'transcribing' | 'cleaning') => void
  /** Whisper model download/load progress (0–100), fired only while a model loads. */
  onModelProgress?: (pct: number) => void
  /** The device the transcriber actually settled on (GPU may fall back to CPU). */
  onDevice?: (device: 'webgpu' | 'wasm') => void
  /** Abort the AI cleanup pass (the transcription itself can't be interrupted mid-decode). */
  signal?: AbortSignal
}

export interface DictationResult {
  transcript: string
  cleaned: string
  status: CleanStatus
  error?: string
}

/** The full deliberate pipeline: decode → transcribe → AI clean. */
export async function processAudio(blob: Blob, opts: ProcessOpts): Promise<DictationResult> {
  const samples = await decodeTo16k(blob)
  opts.onPhase?.('transcribing')
  const transcript = (await transcribe(samples, opts.model ?? DEFAULT_WHISPER_MODEL, opts.device ?? 'auto', opts.onModelProgress, opts.language, opts.onDevice)).trim()
  if (!transcript) return { transcript: '', cleaned: '', status: 'empty' }
  if (!opts.prompt || !opts.prompt.trim()) return { transcript, cleaned: transcript, status: 'raw' }
  if (!opts.brain.enabled) return { transcript, cleaned: transcript, status: 'off' }
  opts.onPhase?.('cleaning')
  const r = await cleanTranscript(transcript, opts.brain, opts.prompt, opts.signal)
  return { transcript, cleaned: r.text, status: r.status, error: r.error }
}
