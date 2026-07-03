/**
 * Optional LLM assist — OFF by default. Talks to an OpenAI-compatible endpoint
 * (bring your own key) or a local Ollama server. The whole app works fully
 * without it; with it off or unconfigured every call returns "unavailable".
 *
 * Providers:
 *  - 'local'  : a small model that runs ON THIS MACHINE (wllama / llama.cpp in
 *               WASM) — the zero-setup, private default; nothing leaves the box.
 *  - 'ollama' : a local (or self-hosted) Ollama server.
 *  - 'openai' : any OpenAI-compatible endpoint (bring your own key) — remote.
 *
 * Remote requests are shaped here (pure, testable) and sent through the main
 * process (`llm:request`) to avoid browser CORS and keep the key out of page
 * script. The 'local' provider never touches the network.
 */
import { DEFAULT_LLM_TIER, type LlmTierId } from './llm-shared'
import type { CleanupEffort } from './dictation'

export type LlmProvider = 'local' | 'openai' | 'ollama'

export interface LlmConfig {
  enabled: boolean
  provider: LlmProvider
  baseUrl: string
  apiKey: string
  model: string
  temperature: number
  /** On-device engine tier: 'standard' (wllama CPU) or 'turbo'/'max' (web-llm GPU).
   *  The GPU tiers silently fall back to 'standard' when no capable GPU is available. */
  localTier: LlmTierId
}

export const DEFAULT_LLM: LlmConfig = {
  enabled: false,
  provider: 'local',
  baseUrl: 'http://localhost:11434',
  apiKey: '',
  model: 'llama3.1',
  temperature: 0.4,
  localTier: DEFAULT_LLM_TIER
}

/**
 * The single, user-facing "AI brain" configuration shared by BOTH the assist
 * tools (summary/cue/rephrase/ask/smart-anchors) AND the Q&A Copilot. It is the
 * one source of truth the Settings UI edits; the low-level {@link LlmConfig}
 * transport shape is derived per role with {@link brainToLlmConfig}.
 *
 * Roles: `live` = the fast model used for assist + Q&A live answers; `deep` = the
 * heavier model the Q&A copilot uses for the talk profile, pre-brief, and the
 * refine pass. For the on-device provider there is only ONE engine (the tier),
 * so both roles resolve to it; the two-model split is meaningful for Ollama
 * (and usable for OpenAI-compatible endpoints).
 */
export interface AiBrain {
  enabled: boolean
  provider: LlmProvider
  /** Ollama / OpenAI-compatible endpoint. */
  baseUrl: string
  /** OpenAI-compatible key. */
  apiKey: string
  /** On-device engine tier ('standard' | 'turbo' | 'max'). */
  localTier: LlmTierId
  /** Fast model name (assist + Q&A live answers) — Ollama/OpenAI. */
  liveModel: string
  /** Smart model name (Q&A profile / pre-brief / refine) — Ollama/OpenAI. */
  deepModel: string
  temperature: number
}

export type BrainRole = 'live' | 'deep'

/** Default brain: AI ON, running fully on-device (zero setup) — first use downloads the
 *  ~1.9 GB Standard CPU model once. Users can switch to a GPU tier, Ollama or OpenAI in Settings. */
export const DEFAULT_AI_BRAIN: AiBrain = {
  enabled: true,
  provider: 'local',
  baseUrl: 'http://localhost:11434',
  apiKey: '',
  localTier: DEFAULT_LLM_TIER,
  liveModel: 'qwen3.5:9b',
  deepModel: 'gpt-oss:120b-cloud',
  temperature: 0.4
}

/** The model name for a role, falling back to the other role's model if blank. */
export function brainModel(b: AiBrain, role: BrainRole): string {
  return role === 'deep' ? b.deepModel || b.liveModel : b.liveModel || b.deepModel
}

/** Derive the low-level transport config for a role (pure). */
export function brainToLlmConfig(b: AiBrain, role: BrainRole = 'live'): LlmConfig {
  return {
    enabled: b.enabled,
    provider: b.provider,
    baseUrl: b.baseUrl,
    apiKey: b.apiKey,
    model: brainModel(b, role),
    temperature: b.temperature,
    localTier: b.localTier
  }
}

/** Build an enabled Ollama config for an explicit model (Q&A live/deep + fallback). */
export function ollamaConfig(baseUrl: string, model: string): LlmConfig {
  return { enabled: true, provider: 'ollama', baseUrl, apiKey: '', model, temperature: 0, localTier: 'standard' }
}

/** An Ollama model that runs in Ollama's CLOUD (e.g. `gpt-oss:120b-cloud`). */
export function isCloudModel(name: string): boolean {
  return /cloud/i.test(name)
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export interface LlmHttpInit {
  method: string
  headers: Record<string, string>
  body: string
}

const trimSlash = (s: string): string => s.replace(/\/+$/, '')

export interface RequestShapeOpts {
  stream?: boolean
  /** Token cap (Ollama num_predict / OpenAI max_tokens). */
  maxTokens?: number
  /** Per-task thinking budget → Ollama `think` level / OpenAI `reasoning_effort`.
   *  'off' (default) disables reasoning for the fastest reply. */
  effort?: CleanupEffort
}

/** Build the URL + request init for a chat completion (pure). */
export function llmRequestShape(
  cfg: LlmConfig,
  messages: ChatMessage[],
  opts: RequestShapeOpts = {}
): { url: string; init: LlmHttpInit } {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  const stream = !!opts.stream
  const effort = opts.effort ?? 'off'
  if (cfg.provider === 'openai') {
    if (cfg.apiKey) headers['Authorization'] = `Bearer ${cfg.apiKey}`
    return {
      url: `${trimSlash(cfg.baseUrl)}/chat/completions`,
      init: {
        method: 'POST',
        headers,
        body: JSON.stringify({
          model: cfg.model,
          messages,
          temperature: cfg.temperature,
          stream,
          ...(opts.maxTokens ? { max_tokens: opts.maxTokens } : {}),
          // Only send reasoning_effort when the user explicitly asks for thinking —
          // non-reasoning models reject the field, so 'off' omits it entirely.
          ...(effort !== 'off' ? { reasoning_effort: effort } : {})
        })
      }
    }
  }
  return {
    url: `${trimSlash(cfg.baseUrl)}/api/chat`,
    init: {
      method: 'POST',
      headers,
      // think:false → direct answers (fastest); a level string asks a reasoning model to deliberate.
      body: JSON.stringify({
        model: cfg.model,
        messages,
        stream,
        think: effort === 'off' ? false : effort,
        options: { temperature: cfg.temperature, ...(opts.maxTokens ? { num_predict: opts.maxTokens } : {}) }
      })
    }
  }
}

/** Extract assistant text from a provider response body (pure). Throws on error shape. */
export function parseLlmText(provider: LlmProvider, raw: string): string {
  const data = JSON.parse(raw)
  if (provider === 'openai') {
    const t = data?.choices?.[0]?.message?.content
    if (typeof t === 'string') return t.trim()
    throw new Error(data?.error?.message || 'unexpected response')
  }
  const t = data?.message?.content
  if (typeof t === 'string') return t.trim()
  throw new Error(typeof data?.error === 'string' ? data.error : 'unexpected response')
}

/** Whether a base URL points off this machine: true/false, or null if unparseable. */
export function hostIsRemote(baseUrl: string): boolean | null {
  try {
    const h = new URL(baseUrl).hostname
    return !(h === 'localhost' || h === '127.0.0.1' || h === '::1' || h === '0.0.0.0')
  } catch {
    return null
  }
}

/** True when the configured endpoint is not on this machine. */
export function isRemote(cfg: LlmConfig): boolean {
  if (cfg.provider === 'local') return false
  const r = hostIsRemote(cfg.baseUrl)
  return r === null ? cfg.provider === 'openai' : r
}

/**
 * True when using this brain sends the user's text OFF this machine — drives the
 * privacy footer. On-device never leaves; OpenAI does unless the host is loopback;
 * Ollama leaves when a CLOUD model is selected or the host is non-loopback.
 * Pass a role to check just that model, or omit to consider both (live + deep).
 */
export function brainIsCloud(b: AiBrain, role?: BrainRole): boolean {
  if (b.provider === 'local') return false
  const remoteHost = hostIsRemote(b.baseUrl)
  if (b.provider === 'openai') return remoteHost === null ? true : remoteHost
  // ollama
  const models = role ? [brainModel(b, role)] : [b.liveModel, b.deepModel]
  if (models.some(isCloudModel)) return true
  return remoteHost === true
}

export function hostOf(url: string): string {
  try {
    return new URL(url).host
  } catch {
    return url
  }
}

export interface ChatResult {
  ok: boolean
  text: string
  error?: string
}

const shorten = (s: string, n = 200): string => (s.length > n ? s.slice(0, n) + '…' : s)

/** Run a chat completion through the main-process proxy. */
export async function chat(cfg: LlmConfig, messages: ChatMessage[]): Promise<ChatResult> {
  if (!cfg.enabled) return { ok: false, text: '', error: 'AI assist is off' }
  if (!cfg.model || !cfg.baseUrl) return { ok: false, text: '', error: 'AI assist is not configured' }
  const { url, init } = llmRequestShape(cfg, messages)
  const res = await window.yapper?.llmRequest(url, init)
  if (!res) return { ok: false, text: '', error: 'bridge unavailable' }
  if (res.error) return { ok: false, text: '', error: res.error }
  if (!res.ok) return { ok: false, text: '', error: `HTTP ${res.status}${res.text ? `: ${shorten(res.text)}` : ''}` }
  try {
    return { ok: true, text: parseLlmText(cfg.provider, res.text) }
  } catch (e) {
    return { ok: false, text: '', error: (e as Error).message }
  }
}

/**
 * Entry point the UI calls. Routes to the on-device engine for 'local' (lazy
 * import so wllama is only pulled in when actually used), or the HTTP proxy
 * otherwise. The rest of the app is unaffected if the local engine is absent.
 */
export async function assist(cfg: LlmConfig, messages: ChatMessage[]): Promise<ChatResult> {
  if (!cfg.enabled) return { ok: false, text: '', error: 'AI assist is off' }
  // The local provider always streams under the hood (tier routing + GPU
  // fallback live in one place); non-streaming callers just ignore tokens.
  if (cfg.provider === 'local') return assistStream(cfg, messages, () => {})
  return chat(cfg, messages)
}

/**
 * Streaming variant of {@link assist}: tokens reach `onToken` as they generate.
 * Only the on-device engine truly streams; remote providers go through the
 * non-streaming proxy and deliver their full text as one late "token" — the
 * caller's rendering works identically either way.
 */
export interface AssistStreamOpts {
  /** Generation cap — short surfaces (Q&A cards) finish much faster with ~160. */
  maxTokens?: number
  /** Thinking budget (Ollama `think` / OpenAI `reasoning_effort`); 'off' = fastest. */
  effort?: CleanupEffort
  /** Override cfg.temperature for this call (e.g. 0 = greedy for fast, deterministic cleanup). */
  temperature?: number
}

export async function assistStream(
  cfg: LlmConfig,
  messages: ChatMessage[],
  onToken: (text: string) => void,
  signal?: AbortSignal,
  opts: AssistStreamOpts = {}
): Promise<ChatResult> {
  if (!cfg.enabled) return { ok: false, text: '', error: 'AI assist is off' }
  const maxTokens = opts.maxTokens ?? 320
  const temperature = opts.temperature ?? cfg.temperature
  if (cfg.provider === 'local') {
    try {
      // The tier selects the engine: wllama (CPU) for 'standard', web-llm (GPU) for
      // the bigger tiers — with a CPU fallback. Lazy import: both runtimes are heavy.
      // (On-device models don't reason, so `effort` only affects the temperature here.)
      const { localAssistStream } = await import('./localEngine')
      return await localAssistStream(messages, temperature, onToken, signal, maxTokens, cfg.localTier)
    } catch (e) {
      return { ok: false, text: '', error: (e as Error).message }
    }
  }
  // Remote (Ollama / OpenAI): stream token-by-token through the main proxy so
  // cloud answers paint live. Temperature override + thinking budget flow through.
  const ecfg = opts.temperature != null ? { ...cfg, temperature: opts.temperature } : cfg
  if (window.yapper) {
    return streamRemote(ecfg, messages, onToken, signal, maxTokens, opts.effort)
  }
  const res = await chat(ecfg, messages)
  if (res.ok && res.text && !signal?.aborted) onToken(res.text)
  return res
}

let streamSeq = 0
/** Stream a remote chat completion via the main-process proxy events. */
async function streamRemote(
  cfg: LlmConfig,
  messages: ChatMessage[],
  onToken: (text: string) => void,
  signal: AbortSignal | undefined,
  maxTokens: number,
  effort?: CleanupEffort
): Promise<ChatResult> {
  const id = `s${Date.now()}_${streamSeq++}`
  const { url, init } = llmRequestShape(cfg, messages, { stream: true, maxTokens, effort })
  let full = ''
  let settled = false
  return new Promise<ChatResult>((resolve) => {
    const finish = (r: ChatResult): void => {
      if (settled) return
      settled = true
      off()
      if (onAbort) signal?.removeEventListener('abort', onAbort)
      resolve(r)
    }
    const off = window.yapper!.onLlmStream((m) => {
      if (m.id !== id) return
      if (m.kind === 'chunk' && m.token) {
        full += m.token
        if (!signal?.aborted) onToken(m.token)
      } else if (m.kind === 'done') {
        finish({ ok: true, text: full.trim() })
      } else if (m.kind === 'error') {
        finish({ ok: false, text: full.trim(), error: m.error })
      }
    })
    const onAbort = signal
      ? (): void => {
          void window.yapper?.llmStreamAbort(id)
          finish({ ok: true, text: full.trim() })
        }
      : null
    if (onAbort) signal?.addEventListener('abort', onAbort, { once: true })
    void window.yapper?.llmStream(id, url, init).then((r) => {
      // A transport-level failure with no error event still settles the promise.
      if (!r?.ok) setTimeout(() => finish({ ok: false, text: full.trim(), error: 'stream failed' }), 50)
    })
  })
}

// ---- prompt builders ----
const cap = (s: string, n = 6000): string => (s.length > n ? s.slice(0, n) + '…' : s)

export function summaryMessages(script: string): ChatMessage[] {
  return [
    {
      role: 'system',
      content:
        'You condense a presentation script into a short briefing. Give 3–6 tight bullet points capturing the key message and structure, in the same language as the script. Reply with only the bullets.'
    },
    { role: 'user', content: `Talk script:\n\n${cap(script)}` }
  ]
}

export function cueCardMessages(sectionText: string): ChatMessage[] {
  return [
    {
      role: 'system',
      content:
        'You turn a slide’s lines into a glanceable cue card: 3–5 very short prompt phrases (not full sentences), each under ~8 words, in the same language as the input. Reply with only the bullet phrases.'
    },
    { role: 'user', content: cap(sectionText, 2000) }
  ]
}
export function rephraseMessages(text: string): ChatMessage[] {
  return [
    {
      role: 'system',
      content:
        'You help a presenter polish lines they will say out loud. Rewrite the passage to be clearer and easier to speak, keeping the same meaning, facts, and roughly the same length. Reply with only the rewritten passage — no preamble, no quotes.'
    },
    { role: 'user', content: text }
  ]
}

export function questionMessages(script: string): ChatMessage[] {
  return [
    {
      role: 'system',
      content:
        'You help a presenter prepare for audience and jury questions. From the talk script, list the most likely questions, each with a one- or two-sentence suggested answer. Be concise and concrete.'
    },
    { role: 'user', content: `Talk script:\n\n${script}` }
  ]
}

export function askMessages(question: string, context: string): ChatMessage[] {
  const sys: ChatMessage = {
    role: 'system',
    content: 'You are a concise, helpful presentation assistant. Use the talk context when relevant.'
  }
  const user: ChatMessage = {
    role: 'user',
    content: context ? `Talk context:\n${context}\n\nQuestion: ${question}` : question
  }
  return [sys, user]
}
