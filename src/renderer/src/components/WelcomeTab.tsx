import { useEffect, useState } from 'react'
import { cleanTranscript, processAudio } from '../lib/dictation'
import { findMode, modePrompt, modeEffort, removeHistory, updateHistory, type HistoryItem, type YapperSettings } from '../lib/settings'

const fmtTime = (t: number): string => new Date(t).toLocaleString()
const copy = (t: string): void => void window.yapper?.clipboardWrite(t)
const bytesToBlob = (bytes: Uint8Array): Blob => new Blob([bytes as unknown as BlobPart], { type: 'audio/webm' })

function HistoryCard({ item, settings, setHistory }: { item: HistoryItem; settings: YapperSettings; setHistory: (h: HistoryItem[]) => void }): JSX.Element {
  const [busy, setBusy] = useState('')
  const [note, setNote] = useState('')
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [hasAudio, setHasAudio] = useState(false)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(item.cleaned)
  const [showRaw, setShowRaw] = useState(false)

  useEffect(() => {
    let alive = true
    if (item.audioPath) void window.yapper?.recordingExists(item.audioPath).then((ok) => alive && setHasAudio(ok))
    else setHasAudio(false)
    return () => {
      alive = false
    }
  }, [item.audioPath])
  useEffect(() => () => void (audioUrl && URL.revokeObjectURL(audioUrl)), [audioUrl])

  const label = item.modeLabel ?? findMode(settings, item.mode)?.label ?? item.mode

  const play = async (): Promise<void> => {
    if (audioUrl || !item.audioPath) return
    const r = await window.yapper?.recordingRead(item.audioPath)
    if (r?.bytes) setAudioUrl(URL.createObjectURL(bytesToBlob(r.bytes)))
  }
  const exportAudio = (): void => {
    if (item.audioPath) void window.yapper?.recordingExport(item.audioPath)
  }
  const reTranscribe = async (): Promise<void> => {
    if (!item.audioPath) return
    const r = await window.yapper?.recordingRead(item.audioPath)
    if (!r?.bytes) return
    setBusy('transcribing')
    const out = await processAudio(bytesToBlob(r.bytes), {
      brain: settings.brain,
      model: settings.whisperModel,
      device: settings.device,
      language: settings.language,
      prompt: modePrompt(settings, item.mode),
      effort: modeEffort(settings, item.mode),
      onPhase: (ph) => setBusy(ph)
    }).catch(() => null)
    setBusy('')
    if (!out) return
    if (out.status === 'off') setNote('AI is off — turn it on in Settings to use this mode.')
    else if (out.status === 'error') setNote(`AI couldn’t run — ${out.error ?? 'check Settings → AI brain'}.`)
    else setNote('')
    setHistory(await updateHistory(item.id, { transcript: out.transcript, cleaned: out.cleaned, failed: !out.transcript.trim() }))
  }
  const reclean = async (modeId: string): Promise<void> => {
    setBusy('cleaning')
    setNote('')
    const r = await cleanTranscript(item.transcript, settings.brain, modePrompt(settings, modeId), undefined, { effort: modeEffort(settings, modeId) })
    setBusy('')
    if (r.status === 'off') setNote('AI is off — turn it on in Settings to use this mode.')
    else if (r.status === 'error') setNote(`AI couldn’t run — ${r.error ?? 'check Settings → AI brain'}.`)
    setHistory(await updateHistory(item.id, { cleaned: r.text, mode: modeId, modeLabel: findMode(settings, modeId)?.label }))
  }
  const saveEdit = async (): Promise<void> => {
    setEditing(false)
    setHistory(await updateHistory(item.id, { cleaned: draft }))
  }
  const del = async (): Promise<void> => {
    if (item.audioPath) await window.yapper?.recordingDelete(item.audioPath)
    setHistory(await removeHistory(item.id))
  }

  const isFailed = Boolean(item.failed) || (!item.transcript && !item.cleaned)

  const audioActions = (
    <>
      {hasAudio ? (
        <>
          {audioUrl ? (
            <audio src={audioUrl} controls className="card__audio" />
          ) : (
            <button className="chip" onClick={() => void play()}>▶ Play</button>
          )}
          <button className="chip" onClick={exportAudio} title="Save the audio file">Export</button>
          <button className="chip" disabled={!!busy} onClick={() => void reTranscribe()}>Re-transcribe</button>
        </>
      ) : item.audioPath ? (
        <span className="muted card__note">audio expired</span>
      ) : (
        <span className="muted card__note">no audio</span>
      )}
      <button className="chip ov-x" onClick={() => void del()} title="Delete">✕</button>
    </>
  )

  return (
    <div className={'card' + (isFailed ? ' card--failed' : '')}>
      <div className="card__meta">
        <span className="card__mode">{isFailed ? 'Failed' : label}</span>
        <span className="muted">{fmtTime(item.at)}</span>
        <span className="ov-spacer" />
        {busy && <span className="muted">{busy}…</span>}
        {!isFailed && (
          <select className="select card__pick" value={item.mode} disabled={!!busy} onChange={(e) => void reclean(e.target.value)} title="Re-process in another mode">
            {settings.modes.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
        )}
      </div>

      {isFailed ? (
        <>
          <div className="card__warn">Couldn’t transcribe this recording. Replay it or try again{hasAudio ? '' : ' (audio not kept)'}.</div>
          {note && <div className="card__warn">{note}</div>}
          <div className="card__actions">{audioActions}</div>
        </>
      ) : editing ? (
        <>
          <textarea className="input" rows={4} value={draft} onChange={(e) => setDraft(e.target.value)} />
          <div className="card__actions">
            <button className="btn" onClick={() => void saveEdit()}>Save</button>
            <button className="chip" onClick={() => { setEditing(false); setDraft(item.cleaned) }}>Cancel</button>
          </div>
        </>
      ) : (
        <>
          <div className="card__text">{item.cleaned}</div>
          {note && <div className="card__warn">{note}</div>}
          {showRaw && item.transcript !== item.cleaned && <div className="card__raw">raw: {item.transcript}</div>}
          <div className="card__actions">
            <button className="btn" onClick={() => copy(item.cleaned)}>Copy</button>
            <button className="chip" onClick={() => { setDraft(item.cleaned); setEditing(true) }}>Edit</button>
            <button className="chip" onClick={() => copy(item.transcript)} title="Copy the untouched transcript">Copy raw</button>
            {item.transcript !== item.cleaned && (
              <button className="chip" onClick={() => setShowRaw((v) => !v)}>{showRaw ? 'Hide raw' : 'Show raw'}</button>
            )}
            <span className="ov-spacer" />
            {audioActions}
          </div>
        </>
      )}
    </div>
  )
}

export default function WelcomeTab({
  settings,
  history,
  setHistory,
  onOpenSettings
}: {
  settings: YapperSettings
  history: HistoryItem[]
  setHistory: (h: HistoryItem[]) => void
  onOpenSettings: () => void
}): JSX.Element {
  return (
    <div className="welcome">
      {!settings.brain.enabled && (
        <div className="banner">
          <div>
            <strong>AI cleanup is off.</strong> Modes like clean-up, email and AI prompt need AI — right now every mode just returns the raw transcript. Turn on the AI brain (on-device, Ollama, or your own key) to use them.
          </div>
          <button className="btn" onClick={onOpenSettings}>Set up AI</button>
        </div>
      )}
      {history.length === 0 ? (
        <div className="empty">
          <div className="empty__big">Hold a shortcut and speak.</div>
          <p className="note">Your dictations land here. Default shortcuts: <span className="kbd">Ctrl ⇧ Space</span> clean-up · <span className="kbd">Ctrl ⇧ P</span> AI prompt · <span className="kbd">Ctrl ⇧ E</span> email. Hold to talk, double-tap to go hands-free.</p>
        </div>
      ) : (
        <div className="cards">
          {history.map((h, i) => (
            <div key={h.id} style={{ animationDelay: `${Math.min(i, 8) * 45}ms` }} className="card-wrap">
              <HistoryCard item={h} settings={settings} setHistory={setHistory} />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
