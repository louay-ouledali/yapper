/**
 * Typed IPC contract — single source of truth for channel names and payloads
 * shared by main, preload, and the renderer. New channels are defined here;
 * the legacy string-literal channels migrate over during the decomposition
 * pass (Phase 5).
 */

export const IPC = {
  sessionCheckpoint: 'session:checkpoint',
  sessionRecover: 'session:recover',
  sessionClearRecovery: 'session:clearRecovery'
} as const

/** Crash-insurance snapshot of a live session, autosaved every ~20 s. */
export interface SessionSnapshot {
  v: 1
  savedAt: number
  projectName: string
  scriptText: string
  cues: Record<number, string>
  /** The session timeline rendered to Markdown — the irreplaceable artifact. */
  sessionMd: string
  detourCount: number
  qaCount: number
}

export function isSessionSnapshot(v: unknown): v is SessionSnapshot {
  const s = v as SessionSnapshot | null
  return Boolean(s && typeof s === 'object' && s.v === 1 && typeof s.scriptText === 'string' && typeof s.savedAt === 'number')
}
