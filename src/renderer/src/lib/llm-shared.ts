/**
 * On-device cleanup model tiers. Both run on CPU via wllama (llama.cpp in WASM) —
 * the proven, reliable path: each is a single GGUF downloaded ONCE through the
 * main-process downloader (resumable, progress-reported) and served from the
 * loopback model server. A floor that's tiny and instant, and a balanced tier
 * that's a stronger model for noticeably better cleanup.
 */
export type LlmTierId = 'floor' | 'balanced'

export interface LlmTier {
  id: LlmTierId
  label: string
  /** Approx. download size in MB (for the UI). The actual file+url live in main. */
  approxMB: number
  blurb: string
}

export const LLM_TIERS: Record<LlmTierId, LlmTier> = {
  floor: {
    id: 'floor',
    label: 'Compact',
    approxMB: 400,
    blurb: 'Qwen 0.5B — smallest download, fastest, runs on any machine.'
  },
  balanced: {
    id: 'balanced',
    label: 'Balanced',
    approxMB: 1100,
    blurb: 'Qwen 1.5B — noticeably stronger cleanup, larger download, still fully on-device.'
  }
}

export const DEFAULT_LLM_TIER: LlmTierId = 'floor'
