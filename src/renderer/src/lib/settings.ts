import { DEFAULT_AI_BRAIN, type AiBrain } from './llm'
import { DEFAULT_WHISPER_MODEL, type WhisperDevice } from './whisper'
import { DEFAULT_SHORTCUTS, DEFAULT_SHOW_OVERLAY, type KeyBinding, type ModeShortcut } from './keybindings'
import { DEFAULT_MODES, RAW_MODE_ID, type CleanupMode } from './dictation'

export interface YapperSettings {
  brain: AiBrain
  /** The fallback mode (used by a card with no recorded mode / new manual cleans). */
  defaultModeId: string
  /** User-definable delivery modes (built-ins + custom). */
  modes: CleanupMode[]
  whisperModel: string
  device: WhisperDevice
  /** Whisper language ('auto' = detect; otherwise a language name like 'english'). */
  language: string
  /** Delete saved audio files older than this many days (0 = keep forever). Transcripts are kept. */
  retentionDays: number
  /** Keep the original audio recording (lets you replay / re-transcribe later). */
  keepAudio: boolean
  /** Mode-bound global shortcuts (hold = talk, double-tap = latch). */
  shortcuts: ModeShortcut[]
  /** A shortcut that just reveals the overlay (no recording). */
  showOverlayBinding: KeyBinding | null
  /** How shortcuts activate: 'toggle' (tap on/off, reliable) or 'hold' (hold-to-talk + double-tap latch). */
  shortcutActivation: 'toggle' | 'hold'
  /** Paste the cleaned text into the active app (vs. clipboard only). */
  autoInsert: boolean
  /** Restore the previous clipboard contents after an auto-insert. */
  restoreClipboard: boolean
  /** One-time marker: AI was defaulted ON (on-device) for this install. */
  aiDefaulted?: boolean
}

export const WHISPER_MODELS: { id: string; label: string }[] = [
  { id: 'onnx-community/whisper-base', label: 'Base · fast' },
  { id: 'onnx-community/whisper-small', label: 'Small · accurate' },
  { id: 'onnx-community/whisper-large-v3-turbo', label: 'Large v3 turbo · max (GPU)' }
]

export const WHISPER_LANGUAGES: { id: string; label: string }[] = [
  { id: 'auto', label: 'Auto-detect' },
  { id: 'english', label: 'English' },
  { id: 'french', label: 'French' },
  { id: 'arabic', label: 'Arabic' },
  { id: 'spanish', label: 'Spanish' },
  { id: 'german', label: 'German' },
  { id: 'italian', label: 'Italian' },
  { id: 'portuguese', label: 'Portuguese' },
  { id: 'dutch', label: 'Dutch' },
  { id: 'russian', label: 'Russian' },
  { id: 'turkish', label: 'Turkish' },
  { id: 'chinese', label: 'Chinese' },
  { id: 'japanese', label: 'Japanese' },
  { id: 'korean', label: 'Korean' },
  { id: 'hindi', label: 'Hindi' }
]

export const DEFAULT_SETTINGS: YapperSettings = {
  brain: DEFAULT_AI_BRAIN,
  defaultModeId: 'clean',
  modes: DEFAULT_MODES,
  whisperModel: DEFAULT_WHISPER_MODEL,
  device: 'auto',
  language: 'english',
  retentionDays: 30,
  keepAudio: true,
  shortcuts: DEFAULT_SHORTCUTS,
  showOverlayBinding: DEFAULT_SHOW_OVERLAY,
  shortcutActivation: 'toggle',
  autoInsert: true,
  restoreClipboard: true,
  aiDefaulted: true
}

/** Merge stored modes over the built-in defaults: keep user edits + custom modes, and make
 *  sure every built-in still exists (so a new release's built-ins appear). */
function mergeModes(stored?: CleanupMode[], legacyPrompts?: Record<string, string>): CleanupMode[] {
  const byId = new Map<string, CleanupMode>()
  for (const m of DEFAULT_MODES) byId.set(m.id, { ...m })
  // Fold legacy per-mode prompt overrides (previous schema) onto the built-ins.
  if (legacyPrompts) for (const [id, prompt] of Object.entries(legacyPrompts)) if (byId.has(id) && prompt) byId.get(id)!.prompt = prompt
  if (Array.isArray(stored)) {
    for (const m of stored) {
      if (!m || typeof m.id !== 'string') continue
      const base = byId.get(m.id)
      if (base) byId.set(m.id, { ...base, label: m.label ?? base.label, prompt: m.id === RAW_MODE_ID ? '' : m.prompt ?? base.prompt })
      else byId.set(m.id, { id: m.id, label: m.label || m.id, prompt: m.prompt || '' }) // custom mode
    }
  }
  return [...byId.values()]
}

export async function loadSettings(): Promise<YapperSettings> {
  const s = (await window.yapper?.storeGet<Record<string, unknown>>('settings')) || {}
  const modes = mergeModes(s.modes as CleanupMode[] | undefined, s.modePrompts as Record<string, string> | undefined)
  const merged: YapperSettings = {
    ...DEFAULT_SETTINGS,
    ...(s as Partial<YapperSettings>),
    brain: { ...DEFAULT_AI_BRAIN, ...((s.brain as Partial<AiBrain>) || {}) },
    modes,
    shortcuts: Array.isArray(s.shortcuts) ? (s.shortcuts as ModeShortcut[]) : DEFAULT_SHORTCUTS,
    showOverlayBinding: (s.showOverlayBinding as KeyBinding | null) ?? null
  }
  if (!modes.some((m) => m.id === merged.defaultModeId)) merged.defaultModeId = 'clean'
  // One-time: turn AI on (on-device) for installs from before AI defaulted on.
  if (!s.aiDefaulted) {
    merged.brain = { ...merged.brain, enabled: true, provider: 'local' }
    merged.aiDefaulted = true
  }
  return merged
}
export async function saveSettings(s: YapperSettings): Promise<void> {
  await window.yapper?.storeSet('settings', s)
}

/** Find a mode by id (falls back to the default, then clean). */
export function findMode(settings: YapperSettings, id: string | undefined): CleanupMode | undefined {
  return settings.modes.find((m) => m.id === id) ?? settings.modes.find((m) => m.id === settings.defaultModeId) ?? settings.modes.find((m) => m.id === 'clean')
}
/** Resolve the prompt for a mode id ('' = raw / no AI). */
export function modePrompt(settings: YapperSettings, id: string | undefined): string {
  return findMode(settings, id)?.prompt ?? ''
}

export interface HistoryItem {
  id: number
  at: number
  transcript: string
  cleaned: string
  /** The mode id this card currently reflects. */
  mode: string
  /** Snapshot of the mode label (so a deleted custom mode still labels its card). */
  modeLabel?: string
  audioPath?: string
  /** Transcription produced nothing / errored — show a recoverable card, not silence. */
  failed?: boolean
}

export async function loadHistory(): Promise<HistoryItem[]> {
  return (await window.yapper?.storeGet<HistoryItem[]>('history')) || []
}
/** Prepend an item, cap the list. Transcripts are kept regardless of audio retention. */
export async function addHistory(item: HistoryItem): Promise<HistoryItem[]> {
  const list = await loadHistory()
  const next = [item, ...list].slice(0, 500)
  await window.yapper?.storeSet('history', next)
  return next
}
/** Patch a history item in place (e.g. after a re-transcribe / re-clean / edit). */
export async function updateHistory(id: number, patch: Partial<HistoryItem>): Promise<HistoryItem[]> {
  const next = (await loadHistory()).map((h) => (h.id === id ? { ...h, ...patch } : h))
  await window.yapper?.storeSet('history', next)
  return next
}
/** Remove a history item (the caller deletes its audio file separately). */
export async function removeHistory(id: number): Promise<HistoryItem[]> {
  const next = (await loadHistory()).filter((h) => h.id !== id)
  await window.yapper?.storeSet('history', next)
  return next
}
