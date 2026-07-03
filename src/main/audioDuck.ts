/**
 * "Duck" the system while dictating: mute the speakers so nothing bleeds into the
 * mic (or distracts), and pause any playing media so a video doesn't run on
 * silently — then restore both the instant recording ends.
 *
 * - Mute uses `loudness` (Core Audio on Windows) so we can read the PRIOR mute
 *   state and restore it exactly (never unmute audio the user had already muted).
 * - Media pause/resume sends the system Play/Pause media key via nut.js — a
 *   best-effort toggle that most players (browsers, Spotify, VLC…) honor.
 *
 * Everything is best-effort and wrapped in try/catch: if a piece fails, recording
 * is never affected. All ops are serialized through a queue so a fast start→stop
 * can't race the mute set/restore, and a safety timer restores audio if a stop is
 * ever missed. The renderer calls duckStart on record start and duckStop on every
 * recording-end path (stop / cancel / auto-stop); main also restores on quit.
 */
import loudness from 'loudness'

let ducked = false
let priorMuted: boolean | null = null // the system mute state before WE muted (so we only unmute what we muted)
let pausedMedia = false
let safety: ReturnType<typeof setTimeout> | null = null
let queue: Promise<void> = Promise.resolve()

/** Serialize duck operations so start's mute can't race stop's restore. */
const enqueue = (fn: () => Promise<void>): Promise<void> => {
  queue = queue.then(fn).catch(() => {})
  return queue
}

/** Toggle the system Play/Pause media key (best-effort). */
async function sendMediaPlayPause(): Promise<void> {
  const { keyboard, Key } = await import('@nut-tree-fork/nut-js')
  keyboard.config.autoDelayMs = 0
  await keyboard.type(Key.AudioPause)
}

export interface DuckOpts {
  mute: boolean
  pauseMedia: boolean
}

/** Mute the speakers and pause playing media for the duration of a dictation. */
export function duckStart(opts: DuckOpts): Promise<void> {
  return enqueue(async () => {
    if (ducked || (!opts.mute && !opts.pauseMedia)) return
    ducked = true
    // Restore even if the renderer never sends a stop (past the 30-min recording cap).
    if (safety) clearTimeout(safety)
    safety = setTimeout(() => void duckStop(), 32 * 60_000)
    if (opts.mute) {
      try {
        priorMuted = await loudness.getMuted()
        if (!priorMuted) await loudness.setMuted(true)
      } catch {
        priorMuted = null
      }
    }
    if (opts.pauseMedia) {
      try {
        await sendMediaPlayPause()
        pausedMedia = true
      } catch {
        pausedMedia = false
      }
    }
  })
}

/** Restore the speakers and resume media. Idempotent — safe to call on any end path. */
export function duckStop(): Promise<void> {
  return enqueue(async () => {
    if (!ducked) return
    ducked = false
    if (safety) {
      clearTimeout(safety)
      safety = null
    }
    // Only unmute if WE muted it (it wasn't already muted before dictation).
    if (priorMuted === false) {
      try {
        await loudness.setMuted(false)
      } catch {
        /* ignore */
      }
    }
    priorMuted = null
    if (pausedMedia) {
      try {
        await sendMediaPlayPause()
      } catch {
        /* ignore */
      }
      pausedMedia = false
    }
  })
}
