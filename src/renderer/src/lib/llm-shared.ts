/**
 * On-device cleanup model tiers. Two engines, one setting:
 *   - wllama (llama.cpp in WASM) runs on the CPU — the reliable, runs-anywhere path.
 *     One GGUF, downloaded ONCE through the main-process downloader and served from
 *     the loopback model server. This is the zero-setup DEFAULT.
 *   - web-llm (MLC) runs on the GPU via WebGPU — much faster and lets us run bigger,
 *     smarter models. Its artifacts are fetched once from Hugging Face and cached in
 *     the browser (IndexedDB), then run fully offline. GPU tiers fall back to the CPU
 *     default when no capable GPU is present.
 */
export type LlmEngine = 'wllama' | 'webllm'
export type LlmTierId = 'standard' | 'turbo' | 'max'

export interface LlmTier {
  id: LlmTierId
  label: string
  /** Which on-device engine runs this tier. */
  engine: LlmEngine
  /** For web-llm tiers: the MLC prebuilt model id (the file/url for wllama lives in main). */
  webllmModel?: string
  /** Approx. download size in MB (for the UI). */
  approxMB: number
  /** True for GPU (web-llm) tiers — needs WebGPU; falls back to Standard otherwise. */
  gpu: boolean
  blurb: string
}

export const LLM_TIERS: Record<LlmTierId, LlmTier> = {
  standard: {
    id: 'standard',
    label: 'Standard · CPU',
    engine: 'wllama',
    approxMB: 1900,
    gpu: false,
    blurb: 'Qwen2.5 3B — smart, fully on-device on the CPU. The zero-setup default; runs on any machine and downloads once.'
  },
  turbo: {
    id: 'turbo',
    label: 'Turbo · GPU',
    engine: 'webllm',
    webllmModel: 'Qwen2.5-7B-Instruct-q4f16_1-MLC',
    approxMB: 4700,
    gpu: true,
    blurb: 'Qwen2.5 7B on the GPU (WebGPU) — noticeably stronger, cleaner results and much faster than CPU. Needs a capable GPU; otherwise falls back to Standard.'
  },
  max: {
    id: 'max',
    label: 'Max · GPU',
    engine: 'webllm',
    webllmModel: 'Llama-3.1-8B-Instruct-q4f32_1-MLC',
    approxMB: 6300,
    gpu: true,
    blurb: 'Llama 3.1 8B on the GPU — the most capable: best at long transcripts and tricky context, and least likely to drop or mangle words. Needs a strong GPU (~7 GB VRAM); otherwise falls back to Standard.'
  }
}

export const DEFAULT_LLM_TIER: LlmTierId = 'standard'

/** Coerce any stored/legacy tier id (e.g. old 'floor'/'balanced') to a valid current tier. */
export function normalizeTier(t: unknown): LlmTierId {
  return t === 'turbo' || t === 'max' || t === 'standard' ? t : 'standard'
}
