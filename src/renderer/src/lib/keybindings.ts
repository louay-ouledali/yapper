/**
 * User-recordable global shortcuts for Yapper. A binding is the exact key combo
 * the user pressed while recording. Each shortcut is tied to a cleanup MODE — the
 * mode of a dictation is decided by which shortcut you press. All shortcuts are
 * matched in the main process against uiohook key events (hold = talk while held,
 * double-tap = latch); see src/main/hotkeys.ts. Pure types + helpers so the
 * recorder UI and the main process can share them.
 */
export interface KeyBinding {
  kind: 'key'
  /** DOM KeyboardEvent.key, e.g. 'ArrowRight', ' ', 'a'. */
  key: string
  /** DOM KeyboardEvent.code, e.g. 'Space', 'KeyA' — layout-independent. */
  code: string
  /** Legacy DOM keyCode (kept for the globalShortcut accelerator + label). */
  keyCode: number
  /** uiohook's normalized keycode, captured via the global hook — the reliable matcher. */
  uioKeycode?: number
  shift?: boolean
  ctrl?: boolean
  alt?: boolean
  meta?: boolean
}

/** A global shortcut tied to a cleanup mode. */
export interface ModeShortcut {
  id: string
  binding: KeyBinding
  modeId: string
}

let seq = 0
export const newShortcutId = (): string => `sc-${Date.now().toString(36)}-${seq++}`

const mk = (code: string, key: string, keyCode: number, mods: Partial<KeyBinding> = {}): KeyBinding => ({
  kind: 'key',
  code,
  key,
  keyCode,
  ...mods
})

/** Sensible defaults — Ctrl+Shift combos avoid the OS-reserved Win+key shortcuts. */
export const DEFAULT_SHORTCUTS: ModeShortcut[] = [
  { id: 'sc-clean', binding: mk('Space', ' ', 32, { ctrl: true, shift: true }), modeId: 'clean' },
  { id: 'sc-prompt', binding: mk('KeyP', 'p', 80, { ctrl: true, shift: true }), modeId: 'prompt' },
  { id: 'sc-email', binding: mk('KeyE', 'e', 69, { ctrl: true, shift: true }), modeId: 'email' }
]

export const DEFAULT_SHOW_OVERLAY: KeyBinding | null = null

/** Windows-reserved combos we should warn about (uiohook can't suppress them). */
export function isReservedCombo(b: KeyBinding | null): boolean {
  if (!b || !b.meta) return false // only Win+… combos are the problem set here
  return ['KeyP', 'KeyM', 'KeyD', 'KeyL', 'KeyE', 'KeyR', 'KeyG', 'KeyI', 'KeyA', 'KeyX', 'Tab'].includes(b.code)
}

const KEY_SYMBOL: Record<string, string> = {
  ' ': 'Space',
  Spacebar: 'Space',
  ArrowRight: '→',
  ArrowLeft: '←',
  ArrowUp: '↑',
  ArrowDown: '↓',
  PageDown: 'PageDn',
  PageUp: 'PageUp',
  Enter: 'Enter',
  Escape: 'Esc',
  Backspace: '⌫',
  Delete: 'Del',
  Home: 'Home',
  End: 'End',
  Tab: 'Tab'
}

/** A short human label for a binding ("Ctrl+⇧Space", "F8", "—"). */
export function describeInput(b: KeyBinding | null): string {
  if (!b) return '—'
  const mods = `${b.ctrl ? 'Ctrl+' : ''}${b.alt ? 'Alt+' : ''}${b.shift ? '⇧' : ''}${b.meta ? 'Win+' : ''}`
  const base = KEY_SYMBOL[b.key] ?? (b.key.length === 1 ? b.key.toUpperCase() : b.key)
  return mods + base
}

/** Capture a binding from a DOM keyboard event (recorder UI). */
export function inputFromKeyEvent(e: {
  key: string
  code: string
  keyCode: number
  shiftKey?: boolean
  ctrlKey?: boolean
  altKey?: boolean
  metaKey?: boolean
}): KeyBinding {
  return {
    kind: 'key',
    key: e.key,
    code: e.code,
    keyCode: e.keyCode,
    ...(e.shiftKey ? { shift: true } : {}),
    ...(e.ctrlKey ? { ctrl: true } : {}),
    ...(e.altKey ? { alt: true } : {}),
    ...(e.metaKey ? { meta: true } : {})
  }
}

/** A modifier-only keypress shouldn't be recorded as a binding on its own. */
export function isModifierKey(key: string): boolean {
  return key === 'Shift' || key === 'Control' || key === 'Alt' || key === 'Meta'
}
