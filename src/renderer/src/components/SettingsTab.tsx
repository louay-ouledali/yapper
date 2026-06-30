import { useEffect, useState, type Dispatch, type SetStateAction } from 'react'
import AiBrainSection from './AiBrainSection'
import ModesManager from './ModesManager'
import ShortcutsManager from './ShortcutsManager'
import { WHISPER_LANGUAGES, WHISPER_MODELS, type YapperSettings } from '../lib/settings'

/** Live WebGPU detection so the user (and we) can see whether the GPU is actually usable. */
function GpuStatus(): JSX.Element {
  const [txt, setTxt] = useState('checking…')
  const [ok, setOk] = useState(false)
  useEffect(() => {
    let alive = true
    void (async () => {
      const info = await window.yapper?.modelInfo().catch(() => null)
      let line = ''
      let good = false
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const g = (navigator as any).gpu
        if (!g) line = 'WebGPU not available in this build'
        else {
          const a = await g.requestAdapter({ powerPreference: 'high-performance' })
          if (!a) line = 'no GPU adapter found'
          else if (a.isFallbackAdapter) line = 'software fallback only (no hardware GPU)'
          else {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            let inf: any = a.info
            if (!inf && a.requestAdapterInfo) inf = await a.requestAdapterInfo()
            const desc = [inf?.vendor, inf?.architecture || inf?.description].filter(Boolean).join(' ')
            line = `hardware GPU ready${desc ? ' · ' + desc : ''}`
            good = true
          }
        }
      } catch (e) {
        line = 'error: ' + (e as Error).message
      }
      if (alive) {
        setOk(good)
        setTxt(`${line}${info?.gpuStatus ? ` · feature: ${info.gpuStatus}` : ''}`)
      }
    })()
    return () => {
      alive = false
    }
  }, [])
  return (
    <span className="set-inline">
      <span className={'lamp ' + (ok ? 'lamp--ok' : 'lamp--warn')} />
      <span className="muted" style={{ fontSize: 12.5 }}>{txt}</span>
    </span>
  )
}

export interface SettingsTabProps {
  settings: YapperSettings
  setS: Dispatch<SetStateAction<YapperSettings>>
  ollamaModels: Array<{ name: string }>
  refreshOllama: () => void
  reloadHistory: () => void
}

export default function SettingsTab({ settings: s, setS, ollamaModels, refreshOllama, reloadHistory }: SettingsTabProps): JSX.Element {
  const [cleanup, setCleanup] = useState<{ busy: boolean; msg: string }>({ busy: false, msg: '' })
  const runCleanup = (): void => {
    setCleanup({ busy: true, msg: '' })
    void window.yapper?.retentionCleanup().then((r) => {
      reloadHistory()
      const n = r?.deleted ?? 0
      setCleanup({ busy: false, msg: `Deleted ${n} expired recording${n === 1 ? '' : 's'}.` })
    })
  }

  return (
    <div className="settings">
      <AiBrainSection brain={s.brain} onChange={(brain) => setS((c) => ({ ...c, brain }))} ollamaModels={ollamaModels} onRefresh={refreshOllama} />

      <ModesManager settings={s} setS={setS} />

      <ShortcutsManager settings={s} setS={setS} />

      <div className="settings__grid">
        <section className="section">
          <div className="section__title">Output</div>
          <label className="row" style={{ cursor: 'pointer' }}>
            <input type="checkbox" checked={s.autoInsert} onChange={(e) => setS((c) => ({ ...c, autoInsert: e.target.checked }))} />
            <span>Auto-insert into the active app (paste at the cursor)</span>
          </label>
          <label className="row" style={{ cursor: s.autoInsert ? 'pointer' : 'default', opacity: s.autoInsert ? 1 : 0.5 }}>
            <input type="checkbox" disabled={!s.autoInsert} checked={s.restoreClipboard} onChange={(e) => setS((c) => ({ ...c, restoreClipboard: e.target.checked }))} />
            <span>Restore my previous clipboard afterwards</span>
          </label>
          <p className="note">When off, the result is just copied to your clipboard.</p>
        </section>

        <section className="section">
          <div className="section__title">Transcription engine</div>
          <div className="field">
            <span className="k">Whisper model</span>
            <select className="select" value={s.whisperModel} onChange={(e) => setS((c) => ({ ...c, whisperModel: e.target.value }))}>
              {WHISPER_MODELS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <span className="k">Compute</span>
            <select className="select" value={s.device} onChange={(e) => setS((c) => ({ ...c, device: e.target.value as YapperSettings['device'] }))}>
              <option value="auto">Auto (GPU when available)</option>
              <option value="webgpu">GPU (WebGPU · force)</option>
              <option value="wasm">CPU (reliable)</option>
            </select>
          </div>
          <div className="field">
            <span className="k">GPU</span>
            <GpuStatus />
          </div>
          <div className="field">
            <span className="k">Language</span>
            <select className="select" value={s.language} onChange={(e) => setS((c) => ({ ...c, language: e.target.value }))}>
              {WHISPER_LANGUAGES.map((l) => (
                <option key={l.id} value={l.id}>
                  {l.label}
                </option>
              ))}
            </select>
          </div>
        </section>

        <section className="section">
          <div className="section__title">Recordings &amp; retention</div>
          <label className="row" style={{ cursor: 'pointer' }}>
            <input type="checkbox" checked={s.keepAudio} onChange={(e) => setS((c) => ({ ...c, keepAudio: e.target.checked }))} />
            <span>Keep the original audio (lets you replay &amp; re-transcribe)</span>
          </label>
          <div className="field">
            <span className="k">Delete audio after</span>
            <span className="set-inline">
              <input className="input" style={{ width: 90 }} type="number" min={0} max={3650} value={s.retentionDays} onChange={(e) => setS((c) => ({ ...c, retentionDays: Math.max(0, Number(e.target.value)) }))} />
              <span className="muted" style={{ fontSize: 12.5 }}>days (0 = forever)</span>
            </span>
          </div>
          <div className="field">
            <span className="k">Cleanup</span>
            <span className="set-inline">
              <button className="chip" disabled={cleanup.busy} onClick={runCleanup}>{cleanup.busy ? 'Cleaning…' : 'Clean up now'}</button>
              {cleanup.msg && <span className="muted" style={{ fontSize: 12.5 }}>{cleanup.msg}</span>}
            </span>
          </div>
          <p className="note">Transcripts are always kept; only audio files expire.</p>
        </section>
      </div>
    </div>
  )
}
