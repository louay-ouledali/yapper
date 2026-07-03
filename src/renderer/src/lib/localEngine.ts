/**
 * On-device engine dispatcher. Picks the engine for the selected cleanup tier:
 *   - 'standard'      → wllama (CPU)
 *   - 'turbo' | 'max' → web-llm (GPU) when a hardware WebGPU adapter is present,
 *                       otherwise transparently falls back to the CPU 'standard'
 *                       engine so cleanup still works on machines without a GPU.
 * Both engines are lazy-imported so their heavyweight runtimes only load when used.
 */
import { LLM_TIERS, normalizeTier, type LlmTierId } from './llm-shared'
import type { ChatMessage, ChatResult } from './llm'

/** Run an on-device streaming chat completion on the engine for `tier`. */
export async function localAssistStream(
  messages: ChatMessage[],
  temperature: number,
  onToken: (text: string) => void,
  signal: AbortSignal | undefined,
  maxTokens: number,
  tier: LlmTierId
): Promise<ChatResult> {
  const t = LLM_TIERS[normalizeTier(tier)]
  if (t.engine === 'webllm' && t.webllmModel) {
    const { webLlmAvailable, webLlmChatStream } = await import('./webLlm')
    if (await webLlmAvailable()) {
      const res = await webLlmChatStream(messages, temperature, onToken, signal, maxTokens, t.webllmModel)
      // Succeeded (or the user cancelled) — done. A runtime GPU failure falls through to CPU.
      if (res.ok || signal?.aborted) return res
    }
    // No capable GPU, or the GPU attempt failed — fall back to the CPU standard engine.
  }
  const { localChatStream } = await import('./localLlm')
  return localChatStream(messages, temperature, onToken, signal, maxTokens)
}
