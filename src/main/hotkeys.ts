/**
 * Global shortcut engine — reliable for ANY key, including layout-specific symbols
 * like the ISO `<>` key that neither Electron accelerators nor uiohook's named-key
 * table can express.
 *
 * The trick: shortcuts are RECORDED through uiohook itself (recordNext), capturing
 * uiohook's own normalized `keycode` + modifier flags. Matching then compares the
 * exact same `keycode`, so it works for every physical key regardless of layout or
 * produced character — no DOM-code / accelerator / VK mapping required.
 *
 * Toggle mode (default): try Electron `globalShortcut` first (OS-level, suppresses
 * the combo) for keys that map to a clean accelerator; everything else falls back
 * to a uiohook keycode matcher. Hold mode: all uiohook (needs key-up).
 *
 * NB: the uiohook fallback can't SUPPRESS the key, so a fallback combo also reaches
 * the focused app.
 */
import { globalShortcut } from 'electron'
import { uIOhook, UiohookKey, type UiohookKeyboardEvent } from 'uiohook-napi'

export type Activation = 'toggle' | 'hold'

export interface BindingLike {
  code: string
  key?: string
  keyCode?: number
  /** uiohook's normalized keycode (captured at record time) — the reliable matcher. */
  uioKeycode?: number
  shift?: boolean
  ctrl?: boolean
  alt?: boolean
  meta?: boolean
}
export interface ShortcutLike {
  id: string
  binding: BindingLike
  modeId: string
}
export interface HotkeyActions {
  start: (modeId: string) => void
  stop: () => void
  toggle: (modeId: string) => void
  showOverlay: () => void
}

const REPEAT_GAP = 130
const HOLD_MS = 300
const DOUBLE_MS = 450
const COOLDOWN_MS = 350
const GS_DEBOUNCE = 250

const MOD_KEYCODES = new Set<number>([
  UiohookKey.Ctrl,
  UiohookKey.CtrlRight,
  UiohookKey.Alt,
  UiohookKey.AltRight,
  UiohookKey.Shift,
  UiohookKey.ShiftRight,
  UiohookKey.Meta,
  UiohookKey.MetaRight
])

let actionsRef: HotkeyActions | null = null
let activation: Activation = 'toggle'
let capturing = false // true while recordNext is grabbing a key (suppresses matchers)
// Remember the last applied bindings so we can re-register after sleep/resume etc.
let lastShortcuts: ShortcutLike[] = []
let lastShowOverlay: BindingLike | null = null

// ── Electron accelerator (globalShortcut fast path; null = use uiohook fallback) ──
function codeToAccel(code: string): string | null {
  if (/^Key[A-Z]$/.test(code)) return code.slice(3)
  if (/^Digit[0-9]$/.test(code)) return code.slice(5)
  if (/^F([1-9]|1[0-9]|2[0-4])$/.test(code)) return code
  const map: Record<string, string> = {
    Space: 'Space', Enter: 'Enter', NumpadEnter: 'Enter', Tab: 'Tab', Backspace: 'Backspace',
    Delete: 'Delete', Insert: 'Insert', Escape: 'Esc', Home: 'Home', End: 'End', PageUp: 'PageUp',
    PageDown: 'PageDown', ArrowUp: 'Up', ArrowDown: 'Down', ArrowLeft: 'Left', ArrowRight: 'Right'
  }
  return map[code] ?? null
}
function bindingToAccelerator(b: BindingLike): string | null {
  const key = codeToAccel(b.code)
  if (!key) return null
  const mods: string[] = []
  if (b.ctrl) mods.push('Control')
  if (b.alt) mods.push('Alt')
  if (b.shift) mods.push('Shift')
  if (b.meta) mods.push('Super')
  return [...mods, key].join('+')
}

// ── uiohook keycode matchers ────────────────────────────────────────────────────
interface Matcher {
  keycode: number
  ctrl: boolean
  alt: boolean
  shift: boolean
  meta: boolean
  modeId: string
}
function matcherFor(b: BindingLike, modeId: string): Matcher | null {
  if (typeof b.uioKeycode !== 'number') return null
  return { keycode: b.uioKeycode, ctrl: !!b.ctrl, alt: !!b.alt, shift: !!b.shift, meta: !!b.meta, modeId }
}
function modsMatch(e: UiohookKeyboardEvent, m: { ctrl: boolean; alt: boolean; shift: boolean; meta: boolean }): boolean {
  return Boolean(e.ctrlKey) === m.ctrl && Boolean(e.altKey) === m.alt && Boolean(e.shiftKey) === m.shift && Boolean(e.metaKey) === m.meta
}

let toggleMatchers: Matcher[] = []
let holdMatchers: Matcher[] = []
let overlayMatcher: Omit<Matcher, 'modeId'> | null = null

// ── globalShortcut bookkeeping ──────────────────────────────────────────────────
let registered: string[] = []
let gsLastFire = 0
function unregisterGlobal(): void {
  for (const acc of registered) {
    try {
      globalShortcut.unregister(acc)
    } catch {
      /* ignore */
    }
  }
  registered = []
}

// ── uiohook lifecycle (kept running once started; matchers gate what fires) ──────
let listenersBound = false
let hookStarted = false
function ensureHook(): void {
  if (!listenersBound) {
    uIOhook.on('keydown', onKeyDown)
    uIOhook.on('keyup', onKeyUp)
    listenersBound = true
  }
  if (!hookStarted) {
    try {
      uIOhook.start()
      hookStarted = true
    } catch (err) {
      console.error('[hotkeys] uiohook failed to start', err)
    }
  }
}

// ── TOGGLE (uiohook fallback) ───────────────────────────────────────────────────
let heldKey = 0
let cooldownUntil = 0
let releaseTimer: ReturnType<typeof setTimeout> | null = null
function clearRelease(): void {
  if (releaseTimer) clearTimeout(releaseTimer)
  releaseTimer = null
}
function onKeyDownToggle(e: UiohookKeyboardEvent): void {
  if (e.keycode === heldKey) {
    clearRelease()
    return
  }
  if (Date.now() < cooldownUntil) return
  const m = toggleMatchers.find((x) => x.keycode === e.keycode && modsMatch(e, x))
  if (m) {
    heldKey = e.keycode
    cooldownUntil = Date.now() + COOLDOWN_MS
    actionsRef?.toggle(m.modeId)
    return
  }
  if (overlayMatcher && e.keycode === overlayMatcher.keycode && modsMatch(e, overlayMatcher)) actionsRef?.showOverlay()
}
function onKeyUpToggle(e: UiohookKeyboardEvent): void {
  if (e.keycode !== heldKey || releaseTimer) return
  releaseTimer = setTimeout(() => {
    releaseTimer = null
    heldKey = 0
  }, REPEAT_GAP)
}

// ── HOLD (uiohook) ──────────────────────────────────────────────────────────────
type Phase = 'idle' | 'holding' | 'pending' | 'latched'
let phase: Phase = 'idle'
let activeKey = 0
let pressAt = 0
let pendingTimer: ReturnType<typeof setTimeout> | null = null
function clearPending(): void {
  if (pendingTimer) clearTimeout(pendingTimer)
  pendingTimer = null
}
function finalizeStop(): void {
  clearPending()
  clearRelease()
  phase = 'idle'
  cooldownUntil = Date.now() + COOLDOWN_MS
  actionsRef?.stop()
  if (heldKey === 0) activeKey = 0
}
function confirmReleaseHold(): void {
  releaseTimer = null
  heldKey = 0
  if (phase === 'holding') {
    if (Date.now() - pressAt >= HOLD_MS) finalizeStop()
    else {
      phase = 'pending'
      pendingTimer = setTimeout(() => phase === 'pending' && finalizeStop(), DOUBLE_MS)
    }
  } else if (phase === 'idle') activeKey = 0
}
function onKeyDownHold(e: UiohookKeyboardEvent): void {
  if (e.keycode === activeKey && releaseTimer) {
    clearRelease()
    heldKey = e.keycode
    return
  }
  if (phase !== 'idle' && e.keycode === activeKey) {
    heldKey = e.keycode
    if (phase === 'pending') {
      clearPending()
      phase = 'latched'
    } else if (phase === 'latched') finalizeStop()
    return
  }
  if (phase !== 'idle') return
  if (e.keycode === heldKey) return
  if (Date.now() < cooldownUntil) return
  const m = holdMatchers.find((x) => x.keycode === e.keycode && modsMatch(e, x))
  if (m) {
    phase = 'holding'
    activeKey = e.keycode
    heldKey = e.keycode
    pressAt = Date.now()
    actionsRef?.start(m.modeId)
    return
  }
  if (overlayMatcher && e.keycode === overlayMatcher.keycode && modsMatch(e, overlayMatcher)) actionsRef?.showOverlay()
}
function onKeyUpHold(e: UiohookKeyboardEvent): void {
  if (e.keycode !== activeKey || releaseTimer) return
  releaseTimer = setTimeout(confirmReleaseHold, REPEAT_GAP)
}

function onKeyDown(e: UiohookKeyboardEvent): void {
  if (capturing) return
  if (activation === 'hold') onKeyDownHold(e)
  else onKeyDownToggle(e)
}
function onKeyUp(e: UiohookKeyboardEvent): void {
  if (capturing) return
  if (activation === 'hold') onKeyUpHold(e)
  else onKeyUpToggle(e)
}

function resetState(): void {
  clearRelease()
  clearPending()
  phase = 'idle'
  activeKey = 0
  heldKey = 0
}

/** Capture the next physical (non-modifier) key via uiohook for the recorder UI.
 *  Returns uiohook's keycode + the modifier flags held at press, or null on cancel/timeout. */
export function recordNext(timeoutMs = 6000): Promise<{ keycode: number; ctrl: boolean; alt: boolean; shift: boolean; meta: boolean } | null> {
  ensureHook()
  return new Promise((resolve) => {
    let done = false
    // Track which modifier keys are physically held — more reliable than trusting one
    // event's ctrlKey/shiftKey flags (the source of "captures wrong modifiers").
    const held = new Set<number>()
    const has = (a: number, b: number): boolean => held.has(a) || held.has(b)
    const finish = (v: { keycode: number; ctrl: boolean; alt: boolean; shift: boolean; meta: boolean } | null): void => {
      if (done) return
      done = true
      capturing = false
      uIOhook.off('keydown', onKd)
      uIOhook.off('keyup', onKu)
      resolve(v)
    }
    const onKd = (e: UiohookKeyboardEvent): void => {
      if (MOD_KEYCODES.has(e.keycode)) {
        held.add(e.keycode)
        return // wait for a real (non-modifier) key
      }
      if (e.keycode === UiohookKey.Escape) return finish(null)
      finish({
        keycode: e.keycode,
        ctrl: has(UiohookKey.Ctrl, UiohookKey.CtrlRight) || !!e.ctrlKey,
        alt: has(UiohookKey.Alt, UiohookKey.AltRight) || !!e.altKey,
        shift: has(UiohookKey.Shift, UiohookKey.ShiftRight) || !!e.shiftKey,
        meta: has(UiohookKey.Meta, UiohookKey.MetaRight) || !!e.metaKey
      })
    }
    const onKu = (e: UiohookKeyboardEvent): void => {
      held.delete(e.keycode)
    }
    capturing = true
    uIOhook.on('keydown', onKd)
    uIOhook.on('keyup', onKu)
    setTimeout(() => finish(null), timeoutMs)
  })
}

/** (Re)register all shortcuts. Returns ids that couldn't be bound (no accelerator AND no
 *  captured uiohook keycode — i.e. recorded on an old build; user should re-record). */
export function applyShortcuts(shortcuts: ShortcutLike[], showOverlayBinding: BindingLike | null, actions: HotkeyActions, mode: Activation = 'toggle'): string[] {
  actionsRef = actions
  activation = mode === 'hold' ? 'hold' : 'toggle'
  lastShortcuts = shortcuts
  lastShowOverlay = showOverlayBinding
  unregisterGlobal()
  resetState()
  toggleMatchers = []
  holdMatchers = []
  overlayMatcher = null
  const failed: string[] = []

  if (activation === 'toggle') {
    for (const sc of shortcuts) {
      const acc = bindingToAccelerator(sc.binding)
      let ok = false
      if (acc) {
        try {
          ok = globalShortcut.register(acc, () => {
            const now = Date.now()
            if (now - gsLastFire < GS_DEBOUNCE) return
            gsLastFire = now
            actionsRef?.toggle(sc.modeId)
          })
          if (ok) registered.push(acc)
        } catch {
          ok = false
        }
      }
      if (!ok) {
        const m = matcherFor(sc.binding, sc.modeId)
        if (m) toggleMatchers.push(m)
        else failed.push(sc.id)
      }
    }
    if (showOverlayBinding) {
      const acc = bindingToAccelerator(showOverlayBinding)
      let ok = false
      if (acc) {
        try {
          ok = globalShortcut.register(acc, () => actionsRef?.showOverlay())
          if (ok) registered.push(acc)
        } catch {
          ok = false
        }
      }
      if (!ok) overlayMatcher = matcherFor(showOverlayBinding, '')
    }
    // Keep the hook alive so recordNext always works, even if nothing falls back.
    ensureHook()
    return failed
  }

  for (const sc of shortcuts) {
    const m = matcherFor(sc.binding, sc.modeId)
    if (m) holdMatchers.push(m)
    else failed.push(sc.id)
  }
  if (showOverlayBinding) overlayMatcher = matcherFor(showOverlayBinding, '')
  ensureHook()
  return failed
}

/** Re-register the last-applied bindings (e.g. after resume, or if a watchdog finds them gone). */
export function reapply(): void {
  if (actionsRef) applyShortcuts(lastShortcuts, lastShowOverlay, actionsRef, activation)
}

/** True only if every globalShortcut we registered is still registered with the OS. */
export function shortcutsHealthy(): boolean {
  for (const a of registered) {
    try {
      if (!globalShortcut.isRegistered(a)) return false
    } catch {
      return false
    }
  }
  return true
}

/** Stop and restart the uiohook listener, then re-register everything (resume recovery). */
export function restartHook(): void {
  if (hookStarted) {
    try {
      uIOhook.stop()
    } catch {
      /* ignore */
    }
    hookStarted = false
  }
  reapply()
}

export function stopHotkeys(): void {
  unregisterGlobal()
  resetState()
  if (hookStarted) {
    try {
      uIOhook.stop()
    } catch {
      /* ignore */
    }
    hookStarted = false
  }
}
