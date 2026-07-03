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

/** How much the model may "think" before answering. Off = fastest (greedy, no
 *  reasoning) — the right default for cleanup. Higher levels only do something on
 *  reasoning-capable models (Ollama/OpenAI/Claude); on-device models don't deliberate. */
export type CleanupEffort = 'off' | 'low' | 'medium' | 'high'
export const DEFAULT_EFFORT: CleanupEffort = 'off'

/** A delivery/cleanup mode: the final form the dictation is turned into. An empty `prompt`
 *  means "no AI" (raw). `builtin` modes can be edited & reset but not deleted. */
export interface CleanupMode {
  id: string
  label: string
  prompt: string
  builtin?: boolean
  /** Per-mode thinking budget (defaults to 'off' = fastest). */
  effort?: CleanupEffort
}

/** Shared tail appended to every mode prompt. Kept short on purpose: this text is
 *  re-processed on every cleanup pass, so a long tail directly adds latency. */
const PROMPT_TAIL = ' Keep it about the same length as the input — don’t pad or expand. Write plainly. Output ONLY the result text: no preamble, quotes, or notes.'

export const DEFAULT_MODES: CleanupMode[] = [
  {
    id: 'clean',
    label: 'Clean-up',
    builtin: true,
    prompt:
      'Clean up this dictated text: remove fillers (um, uh, like) and accidental back-to-back stutters, and fix grammar, punctuation, capitalization and obvious mis-hearings. Keep every point and the speaker’s own wording, tone and meaning — never summarize, add, or drop content. Keep deliberate repetition/emphasis (e.g. "very very", "test, test, test"). If the speaker corrects themselves, keep only the corrected version.'
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

/** Longest input we send to the model in one pass. A long dictation is split into
 *  several of these so the cleanup output is never truncated by the token cap /
 *  context window (the bug where long recordings lost their latter parts). ~3500
 *  chars ≈ ~900 tokens in, leaving ample room for the reply within a 4k context. */
const CLEAN_CHUNK_CHARS = 3500

/** Split a transcript into cleanup-sized chunks on sentence boundaries (greedy pack).
 *  A single over-long sentence is hard-split on whitespace so nothing is dropped. */
export function splitForCleanup(text: string, maxChars = CLEAN_CHUNK_CHARS): string[] {
  const t = text.trim()
  if (t.length <= maxChars) return t ? [t] : []
  const parts = t.match(/[^.!?…]*[.!?…]+["'”’)\]]*\s*|[^.!?…]+$/g) ?? [t]
  const chunks: string[] = []
  let cur = ''
  const flush = (): void => {
    const s = cur.trim()
    if (s) chunks.push(s)
    cur = ''
  }
  for (const part of parts) {
    if (cur && cur.length + part.length > maxChars) flush()
    if (part.length > maxChars) {
      flush()
      let rest = part
      while (rest.length > maxChars) {
        let cut = rest.lastIndexOf(' ', maxChars)
        if (cut < maxChars * 0.6) cut = maxChars
        chunks.push(rest.slice(0, cut).trim())
        rest = rest.slice(cut)
      }
      cur = rest
    } else {
      cur += part
    }
  }
  flush()
  return chunks
}

/** Token budget for a chunk's reply: cleanup is ~length-preserving, so allow the
 *  chunk's own size plus headroom, capped so we never blow past the context window. */
const cleanupBudget = (chunkChars: number): number => Math.min(1400, Math.max(320, Math.ceil((chunkChars / 4) * 1.5) + 96))

/** Options for a cleanup pass: the mode's thinking budget and a live-token sink. */
export interface CleanOpts {
  /** Per-mode thinking budget (defaults to 'off' = fastest, greedy). */
  effort?: CleanupEffort
  /** Receives cleaned text as it streams, so the UI can paint it live. */
  onToken?: (text: string) => void
}

/** Run an AI cleanup pass over a transcript with a resolved prompt, reporting WHAT happened.
 *  Long transcripts are cleaned in chunks and re-joined so nothing is dropped or truncated.
 *  Falls back to the raw transcript for raw mode / disabled brain / errors — but says so. */
export async function cleanTranscript(transcript: string, brain: AiBrain, prompt?: string, signal?: AbortSignal, opts: CleanOpts = {}): Promise<CleanOutcome> {
  if (!transcript.trim()) return { text: transcript, status: 'empty' }
  if (!prompt || !prompt.trim()) return { text: transcript, status: 'raw' }
  if (!brain.enabled) return { text: transcript, status: 'off' }
  const effort = opts.effort ?? DEFAULT_EFFORT
  // Cleanup is a mechanical edit, not a puzzle: 'off' runs greedy (temp 0) for the
  // fastest, most deterministic reply. Higher efforts respect the brain's temperature.
  const temperature = effort === 'off' ? 0 : brain.temperature
  try {
    const cfg = brainToLlmConfig(brain, 'live')
    const chunks = splitForCleanup(transcript)
    const out: string[] = []
    for (let i = 0; i < chunks.length; i++) {
      if (signal?.aborted) break
      if (i > 0) opts.onToken?.('\n\n') // visual break between streamed chunks
      // assistStream (vs assist) lets a cancel abort the pass mid-generation via the signal.
      const res = await assistStream(cfg, cleanupMessages(chunks[i], prompt), opts.onToken ?? (() => {}), signal, {
        maxTokens: cleanupBudget(chunks[i].length),
        effort,
        temperature
      })
      if (res.ok) {
        const cleaned = stripModelArtifacts(res.text)
        // Never lose a segment: if the model returned nothing usable for this chunk, keep the raw chunk.
        out.push(cleaned || chunks[i])
      } else {
        // A real engine/transport error: report it and fall back to the whole raw transcript.
        return { text: transcript, status: 'error', error: res.error || 'AI returned nothing' }
      }
    }
    if (signal?.aborted) return { text: transcript, status: 'error', error: 'cancelled' }
    const joined = out.join(chunks.length > 1 ? '\n\n' : '').trim()
    if (joined) return { text: joined, status: 'ok' }
    return { text: transcript, status: 'error', error: 'AI returned an empty reply' }
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
  /** The device the transcriber settled on, and whether it wanted GPU but fell back to CPU. */
  onDevice?: (device: 'webgpu' | 'wasm', downgraded: boolean) => void
  /** The chosen mode's thinking budget (defaults to 'off' = fastest). */
  effort?: CleanupEffort
  /** Cleaned text as it streams, so the overlay can paint it live. */
  onCleanToken?: (text: string) => void
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
  const r = await cleanTranscript(transcript, opts.brain, opts.prompt, opts.signal, { effort: opts.effort, onToken: opts.onCleanToken })
  return { transcript, cleaned: r.text, status: r.status, error: r.error }
}
