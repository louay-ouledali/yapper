import { useEffect, useState, type Dispatch, type SetStateAction } from 'react'
import KeyRecorder from './KeyRecorder'
import { isReservedCombo, newShortcutId, type KeyBinding, type ModeShortcut } from '../lib/keybindings'
import type { YapperSettings } from '../lib/settings'

const sig = (b: KeyBinding): string => `${b.code}|${!!b.ctrl}|${!!b.alt}|${!!b.shift}|${!!b.meta}`

export default function ShortcutsManager({ settings, setS }: { settings: YapperSettings; setS: Dispatch<SetStateAction<YapperSettings>> }): JSX.Element {
  const act = settings.shortcutActivation ?? 'toggle'
  const [failed, setFailed] = useState<string[]>([])

  // On mount, just READ which shortcuts failed to register (no re-registering).
  useEffect(() => {
    void window.yapper?.bindingsStatus().then((f) => setFailed(f || []))
  }, [])

  // Commit shortcut changes to settings AND re-register them live in the main process.
  const commit = (shortcuts: ModeShortcut[], showOverlayBinding: KeyBinding | null, activation: 'toggle' | 'hold' = act): void => {
    setS((c) => ({ ...c, shortcuts, showOverlayBinding, shortcutActivation: activation }))
    void window.yapper?.applyBindings(shortcuts, showOverlayBinding, activation).then((f) => setFailed(f || []))
  }
  const update = (id: string, patch: Partial<ModeShortcut>): void =>
    commit(settings.shortcuts.map((s) => (s.id === id ? { ...s, ...patch } : s)), settings.showOverlayBinding)
  const remove = (id: string): void => commit(settings.shortcuts.filter((s) => s.id !== id), settings.showOverlayBinding)
  const add = (): void => {
    const n = Math.min(9, settings.shortcuts.length + 1)
    const binding: KeyBinding = { kind: 'key', code: `Digit${n}`, key: String(n), keyCode: 48 + n, ctrl: true, shift: true }
    commit([...settings.shortcuts, { id: newShortcutId(), binding, modeId: settings.defaultModeId }], settings.showOverlayBinding)
  }

  const counts = new Map<string, number>()
  for (const s of settings.shortcuts) counts.set(sig(s.binding), (counts.get(sig(s.binding)) ?? 0) + 1)

  return (
    <section className="section">
      <div className="section__title">Shortcuts</div>
      <div className="field">
        <span className="k">Activation</span>
        <select className="select" value={act} onChange={(e) => commit(settings.shortcuts, settings.showOverlayBinding, e.target.value as 'toggle' | 'hold')}>
          <option value="toggle">Toggle — tap to start, tap to stop (reliable)</option>
          <option value="hold">Hold — hold to talk, double-tap to latch</option>
        </select>
      </div>
      <p className="note">
        {act === 'toggle'
          ? 'Tap a shortcut to start recording in its mode; tap again to stop. Works from any app.'
          : 'Hold a shortcut to talk while held; double-tap to record hands-free until you tap again.'}
      </p>
      <div className="scuts">
        {settings.shortcuts.map((s) => {
          const dup = (counts.get(sig(s.binding)) ?? 0) > 1
          const reserved = isReservedCombo(s.binding)
          return (
            <div className="scut" key={s.id}>
              <KeyRecorder value={s.binding} onChange={(b) => b && update(s.id, { binding: b })} />
              <span className="muted">→</span>
              <select className="select scut__mode" value={s.modeId} onChange={(e) => update(s.id, { modeId: e.target.value })}>
                {settings.modes.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                  </option>
                ))}
              </select>
              <span className="ov-spacer" />
              {failed.includes(s.id) && <span className="note note--warn" title="This key can't be a global shortcut — try a letter/number/F-key with Ctrl/Alt/Shift">⚠ won’t register</span>}
              {reserved && <span className="note note--warn" title="Windows uses this combo — it will also trigger the OS action">⚠ OS-reserved</span>}
              {dup && <span className="note note--warn">⚠ duplicate</span>}
              <button className="chip ov-x" onClick={() => remove(s.id)} title="Delete shortcut">✕</button>
            </div>
          )
        })}
      </div>
      <div className="row">
        <button className="chip" onClick={add}>+ Add shortcut</button>
      </div>

      <div className="field" style={{ marginTop: 6 }}>
        <span className="k">Show overlay</span>
        <KeyRecorder value={settings.showOverlayBinding} onChange={(b) => commit(settings.shortcuts, b)} />
      </div>
      <p className="note">Tip: avoid Win+key combos (Win+P, Win+M…) — Windows reserves them and Yapper can’t suppress them, so they’ll trigger both. Ctrl+Shift+key combos are safe.</p>
    </section>
  )
}
