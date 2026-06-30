import type { Dispatch, SetStateAction } from 'react'
import { DEFAULT_MODES, RAW_MODE_ID, type CleanupMode } from '../lib/dictation'
import type { YapperSettings } from '../lib/settings'

const newId = (): string => `mode-${Date.now().toString(36)}`

export default function ModesManager({ settings, setS }: { settings: YapperSettings; setS: Dispatch<SetStateAction<YapperSettings>> }): JSX.Element {
  const patchModes = (fn: (modes: CleanupMode[]) => CleanupMode[]): void => setS((c) => ({ ...c, modes: fn(c.modes) }))
  const updateMode = (id: string, patch: Partial<CleanupMode>): void => patchModes((modes) => modes.map((m) => (m.id === id ? { ...m, ...patch } : m)))
  const deleteMode = (id: string): void =>
    setS((c) => ({ ...c, modes: c.modes.filter((m) => m.id !== id), defaultModeId: c.defaultModeId === id ? 'clean' : c.defaultModeId }))
  const resetMode = (id: string): void => {
    const def = DEFAULT_MODES.find((m) => m.id === id)
    if (def) updateMode(id, { label: def.label, prompt: def.prompt })
  }
  const addMode = (): void => patchModes((modes) => [...modes, { id: newId(), label: 'New mode', prompt: '' }])
  const isEdited = (m: CleanupMode): boolean => {
    const def = DEFAULT_MODES.find((d) => d.id === m.id)
    return !!def && (def.label !== m.label || def.prompt !== m.prompt)
  }

  return (
    <section className="section">
      <div className="section__title">Modes</div>
      <p className="note">A mode is the final form a dictation is delivered in. Pick the default below; each shortcut can use a different mode. Edit the instructions, or add your own. “Raw” returns the untouched transcript.</p>
      <div className="modes">
        {settings.modes.map((m) => {
          const isRaw = m.id === RAW_MODE_ID
          return (
            <div className="mode" key={m.id}>
              <div className="mode__head">
                <input className="input mode__label" value={m.label} onChange={(e) => updateMode(m.id, { label: e.target.value })} aria-label="Mode name" />
                <label className="mode__default" title="Use as the default mode">
                  <input type="radio" name="defaultMode" checked={settings.defaultModeId === m.id} onChange={() => setS((c) => ({ ...c, defaultModeId: m.id }))} />
                  <span>default</span>
                </label>
                <span className="ov-spacer" />
                {m.builtin && isEdited(m) && (
                  <button className="chip" onClick={() => resetMode(m.id)} title="Restore the built-in instructions">Reset</button>
                )}
                {!m.builtin && (
                  <button className="chip ov-x" onClick={() => deleteMode(m.id)} title="Delete this mode">✕</button>
                )}
              </div>
              {isRaw ? (
                <p className="note">No AI — copies/inserts the raw transcript as transcribed.</p>
              ) : (
                <textarea
                  className="input"
                  rows={3}
                  value={m.prompt}
                  spellCheck={false}
                  placeholder="Instructions for the AI…"
                  onChange={(e) => updateMode(m.id, { prompt: e.target.value })}
                />
              )}
            </div>
          )
        })}
      </div>
      <div>
        <button className="chip" onClick={addMode}>+ Add mode</button>
      </div>
    </section>
  )
}
