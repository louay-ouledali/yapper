/**
 * The AI-brain (cleanup) settings — provider (on-device / Ollama / OpenAI),
 * endpoint, key, model picker, a real "Test connection" with a status lamp, and
 * on-device model management (Compact CPU download + Balanced GPU warm-up).
 * Adapted from StagePilot's AiBrainSection for Yapper's single-model cleanup use.
 */
import { useEffect, useState } from 'react'
import { brainIsCloud, hostOf, isCloudModel, type AiBrain } from '../lib/llm'
import { connTone, testConnection, type ConnResult } from '../lib/llm-conn'
import { LLM_TIERS, type LlmTierId } from '../lib/llm-shared'

const fmtMB = (mb: number): string => (mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb} MB`)
const TONE_LAMP: Record<ReturnType<typeof connTone>, string> = { ok: 'lamp--ok', warn: 'lamp--warn', err: 'lamp--err', idle: '' }

export interface AiBrainSectionProps {
  brain: AiBrain
  onChange: (b: AiBrain) => void
  ollamaModels: Array<{ name: string }>
  onRefresh: () => void
}

export default function AiBrainSection({ brain, onChange, ollamaModels, onRefresh }: AiBrainSectionProps): JSX.Element {
  const patch = (p: Partial<AiBrain>): void => onChange({ ...brain, ...p })

  const [conn, setConn] = useState<{ busy: boolean; res?: ConnResult }>({ busy: false })
  const [localReady, setLocalReady] = useState(false)
  const [localDl, setLocalDl] = useState<{ pct: number; busy: boolean; error?: string }>({ pct: 0, busy: false })
  const tier = brain.localTier ?? 'floor'

  // Re-check install state whenever the selected tier changes (each tier is its own file).
  useEffect(() => {
    let alive = true
    setLocalDl({ pct: 0, busy: false })
    void window.yapper?.localModelStatus(tier).then((st) => alive && setLocalReady(Boolean(st?.installed)))
    const off = window.yapper?.onLocalModelProgress?.((p) => {
      const d = p as { pct?: number; done?: boolean; error?: string }
      if (d.error) setLocalDl({ busy: false, pct: 0, error: d.error })
      else if (d.done) {
        setLocalDl({ busy: false, pct: 100 })
        setLocalReady(true)
      } else setLocalDl({ busy: true, pct: d.pct ?? 0 })
    })
    return () => {
      alive = false
      off?.()
    }
  }, [tier])

  const downloadLocal = (): void => {
    setLocalDl({ busy: true, pct: 0, error: undefined })
    void window.yapper?.localModelDownload(tier)
  }
  const runTest = (): void => {
    setConn({ busy: true })
    void testConnection(brain, 'live').then((res) => setConn({ busy: false, res }))
  }

  const isLocal = brain.provider === 'local'
  const isOllama = brain.provider === 'ollama'
  const cloud = brainIsCloud(brain, 'live')
  const tone = conn.res ? connTone(conn.res.state) : 'idle'
  const names = ollamaModels.map((m) => m.name)
  const opts = names.includes(brain.liveModel) || !brain.liveModel ? names : [brain.liveModel, ...names]
  const localNames = opts.filter((n) => !isCloudModel(n))
  const cloudNames = opts.filter((n) => isCloudModel(n))

  return (
    <section className="section">
      <div className="section__title">AI brain (cleanup)</div>

      <label className="row" style={{ cursor: 'pointer' }}>
        <input type="checkbox" checked={brain.enabled} onChange={(e) => patch({ enabled: e.target.checked })} />
        <span>Use AI to clean up transcripts</span>
      </label>

      <div className="field">
        <span className="k">Engine</span>
        <select className="select" value={brain.provider} onChange={(e) => patch({ provider: e.target.value as AiBrain['provider'] })}>
          <option value="local">On-device · private</option>
          <option value="ollama">Ollama · local or cloud</option>
          <option value="openai">OpenAI-compatible · cloud</option>
        </select>
      </div>

      {!isLocal && (
        <>
          <div className="field">
            <span className="k">Endpoint</span>
            <input className="input" value={brain.baseUrl} spellCheck={false} onChange={(e) => patch({ baseUrl: e.target.value })} placeholder={isOllama ? 'http://localhost:11434' : 'https://api.example.com/v1'} />
          </div>
          {brain.provider === 'openai' && (
            <div className="field">
              <span className="k">API key</span>
              <input className="input" type="password" value={brain.apiKey} spellCheck={false} placeholder="sk-…" onChange={(e) => patch({ apiKey: e.target.value })} />
            </div>
          )}
          <div className="field">
            <span className="k">Model</span>
            <span className="set-inline" style={{ width: '100%' }}>
              {isOllama ? (
                <select className="select" value={brain.liveModel} onChange={(e) => patch({ liveModel: e.target.value })}>
                  {opts.length === 0 && <option value={brain.liveModel}>{brain.liveModel || '(no models found — is Ollama running?)'}</option>}
                  {localNames.length > 0 && (
                    <optgroup label="On this device">
                      {localNames.map((n) => (
                        <option key={n} value={n}>
                          {n}
                        </option>
                      ))}
                    </optgroup>
                  )}
                  {cloudNames.length > 0 && (
                    <optgroup label="Ollama cloud ☁">
                      {cloudNames.map((n) => (
                        <option key={n} value={n}>
                          {n}
                        </option>
                      ))}
                    </optgroup>
                  )}
                </select>
              ) : (
                <input className="input" value={brain.liveModel} spellCheck={false} placeholder="gpt-4o-mini" onChange={(e) => patch({ liveModel: e.target.value })} />
              )}
              {isOllama && (
                <>
                  <span className="muted" style={{ fontSize: 12.5, whiteSpace: 'nowrap' }}>{isCloudModel(brain.liveModel) ? '☁ cloud' : 'local'}</span>
                  <button className="chip" onClick={onRefresh} title="Refresh the model list from Ollama">
                    ↻
                  </button>
                </>
              )}
            </span>
          </div>
          <div className="field">
            <span className="k">Connection</span>
            <span className="set-inline">
              <span className={'lamp ' + (conn.busy ? 'lamp--busy' : TONE_LAMP[tone])} />
              <span className="muted" style={{ fontSize: 12.5 }}>{conn.busy ? 'testing…' : conn.res ? conn.res.detail : 'not tested'}</span>
              <button className="chip" disabled={conn.busy} onClick={runTest}>
                Test connection
              </button>
            </span>
          </div>
        </>
      )}

      {isLocal && (
        <>
          <div className="field">
            <span className="k">Model</span>
            <select
              className="select"
              value={tier}
              onChange={(e) => patch({ localTier: e.target.value as LlmTierId })}
              title="Both models run fully on this machine (CPU). Balanced is larger but cleans up noticeably better."
            >
              <option value="floor">
                {LLM_TIERS.floor.label} · Qwen 0.5B (~{fmtMB(LLM_TIERS.floor.approxMB)})
              </option>
              <option value="balanced">
                {LLM_TIERS.balanced.label} · Qwen 1.5B (~{fmtMB(LLM_TIERS.balanced.approxMB)})
              </option>
            </select>
          </div>
          <div className="field">
            <span className="k">Status</span>
            <span className="set-inline">
              <span className="muted" style={{ fontSize: 12.5 }}>{localReady ? 'Installed ✓' : 'Not installed'}</span>
              {!localReady && (
                <button className="chip" disabled={localDl.busy} onClick={downloadLocal}>
                  {localDl.busy ? `Downloading… ${localDl.pct}%` : `Download (~${fmtMB(LLM_TIERS[tier].approxMB)})`}
                </button>
              )}
            </span>
          </div>
          <p className="note">{LLM_TIERS[tier].blurb}</p>
          {localDl.error && <p className="note note--warn">Download failed: {localDl.error}</p>}
        </>
      )}

      <p className={'note' + (cloud && brain.enabled ? ' note--warn' : '')}>
        {isLocal
          ? 'Runs entirely on this machine — no key, no server, nothing leaves the box.'
          : cloud
            ? `Heads-up: transcripts go to ${hostOf(brain.baseUrl)}${isOllama ? ' / Ollama cloud' : ''}.`
            : 'Local endpoint — transcripts stay on this machine.'}
      </p>
    </section>
  )
}
