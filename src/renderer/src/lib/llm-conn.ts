/**
 * Connection testing for the unified AI brain. The actual decision logic is pure
 * (classifyOllama / classifyOpenai / classifyLocal + tone mapping) so it is
 * unit-testable without Electron; {@link testConnection} is the thin async
 * orchestrator that calls the IPC bridge and hands the results to the classifiers.
 *
 * Why no warm-up during a "test": warming an Ollama model cold-loads it (~10–17 s
 * on this machine), which would make the Test button feel hung. Presence in the
 * tag list is enough to report "reachable + model installed"; liveness shows up
 * the first time the user actually asks.
 */
import { brainModel, isCloudModel, type AiBrain, type BrainRole } from './llm'

export type ConnState =
  | 'ok'
  | 'unreachable'
  | 'auth'
  | 'model-missing'
  | 'not-installed'
  | 'no-model'

export interface ConnResult {
  state: ConnState
  detail: string
  /** Models advertised by the endpoint (Ollama), for the picker. */
  models?: string[]
}

/** Visual tone for the status dot — maps to the existing `.lamp--*` classes. */
export type ConnTone = 'ok' | 'warn' | 'err' | 'idle'
export function connTone(state: ConnState): ConnTone {
  if (state === 'ok') return 'ok'
  if (state === 'unreachable') return 'err'
  return 'warn'
}

/** A model is "present" if its exact name or its base (pre-`:`) name is advertised. */
export function modelPresent(model: string, names: string[]): boolean {
  if (!model) return false
  if (names.includes(model)) return true
  const base = model.split(':')[0]
  return names.some((n) => n === base || n.split(':')[0] === base)
}

export interface TagsResult {
  ok: boolean
  models?: Array<{ name: string; size?: number }>
  error?: string
}

/** Decide the connection state for Ollama from the /api/tags result (pure). */
export function classifyOllama(tags: TagsResult, model: string): ConnResult {
  if (!tags.ok) return { state: 'unreachable', detail: tags.error ? `Ollama unreachable — ${tags.error}` : 'Ollama not reachable' }
  const names = (tags.models ?? []).map((m) => m.name)
  if (!model) return { state: 'no-model', detail: 'pick a model', models: names }
  if (!modelPresent(model, names)) return { state: 'model-missing', detail: `“${model}” is not installed — run \`ollama pull ${model}\``, models: names }
  return { state: 'ok', detail: isCloudModel(model) ? `cloud model ready · ${model}` : `ready · ${model}`, models: names }
}

/** Decide the connection state for an OpenAI-compatible endpoint from a chat result (pure). */
export function classifyOpenai(r: { ok: boolean; error?: string }): ConnResult {
  if (r.ok) return { state: 'ok', detail: 'endpoint reachable' }
  const e = r.error || 'request failed'
  if (/\b401\b|\b403\b|unauthor|invalid api key|incorrect api key|missing.*key/i.test(e)) return { state: 'auth', detail: e }
  return { state: 'unreachable', detail: e }
}

/** Decide the on-device state from the model-install + GPU readiness (pure). */
export function classifyLocal(installed: boolean, tier: AiBrain['localTier'], gpuHardware: boolean | null): ConnResult {
  if (!installed) return { state: 'not-installed', detail: 'on-device model not downloaded yet' }
  const gpuTier = tier === 'turbo' || tier === 'max'
  if (gpuTier && gpuHardware === false) return { state: 'ok', detail: 'on-device ready (no GPU — uses Standard CPU)' }
  return { state: 'ok', detail: gpuTier ? 'on-device ready (GPU)' : 'on-device ready' }
}

/**
 * Test the brain's connection for a role. Async glue over the IPC bridge; all the
 * branching logic lives in the pure classifiers above.
 */
export async function testConnection(b: AiBrain, role: BrainRole = 'live'): Promise<ConnResult> {
  const bridge = window.yapper
  if (!bridge) return { state: 'unreachable', detail: 'bridge unavailable' }

  if (b.provider === 'local') {
    // GPU tiers (web-llm): ready means a hardware GPU + the model cached in-renderer;
    // without a GPU they fall back to the CPU 'standard' model, so check that instead.
    if (b.localTier === 'turbo' || b.localTier === 'max') {
      const [{ webLlmAvailable, webLlmHasModel }, { LLM_TIERS }] = await Promise.all([import('./webLlm'), import('./llm-shared')])
      const model = LLM_TIERS[b.localTier].webllmModel
      const gpu = await webLlmAvailable().catch(() => false)
      if (gpu && model) return classifyLocal(await webLlmHasModel(model).catch(() => false), b.localTier, true)
      const st = await bridge.localModelStatus('standard').catch(() => null)
      return classifyLocal(Boolean(st?.installed), b.localTier, false)
    }
    const st = await bridge.localModelStatus('standard').catch(() => null)
    return classifyLocal(Boolean(st?.installed), b.localTier, null)
  }

  if (b.provider === 'ollama') {
    const tags = await bridge.ollamaTags(b.baseUrl).catch((e: unknown) => ({ ok: false, error: (e as Error).message }))
    return classifyOllama(tags as TagsResult, brainModel(b, role))
  }

  // openai-compatible: a 1-token ping distinguishes auth vs network failure.
  const model = brainModel(b, role)
  if (!model) return { state: 'no-model', detail: 'enter a model name' }
  const { chat, brainToLlmConfig } = await import('./llm')
  const r = await chat(brainToLlmConfig({ ...b, enabled: true }, role), [{ role: 'user', content: 'ping' }])
  return classifyOpenai(r)
}
