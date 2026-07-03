import { useEffect, useRef, useState } from 'react'
import { VoiceRecorder, processAudio, cleanTranscript, type CleanupMode, type DictationResult } from '../lib/dictation'
import { getTranscriber } from '../lib/whisper-batch'
import { addHistory, updateHistory, loadSettings, findMode, modePrompt, modeEffort, type YapperSettings } from '../lib/settings'
import { LLM_TIERS, normalizeTier } from '../lib/llm-shared'

type Phase = 'idle' | 'recording' | 'transcribing' | 'cleaning' | 'done' | 'error' | 'downloading'

const DOT: Record<Phase, string> = {
  idle: '',
  recording: ' ov-dot--rec',
  transcribing: ' ov-dot--busy',
  cleaning: ' ov-dot--busy',
  downloading: ' ov-dot--busy',
  done: ' ov-dot--done',
  error: ' ov-dot--rec'
}

/** Human note about why a result is raw (AI off / error). '' when the AI ran fine. */
function noteFor(status: string, error?: string): string {
  if (status === 'off') return 'AI is off — this is the raw transcript. Turn on AI in Settings to use modes.'
  if (status === 'error') return `AI couldn’t run — raw transcript. ${error ?? ''}`.trim()
  return ''
}

export default function Overlay(): JSX.Element {
  const [phase, setPhase] = useState<Phase>('idle')
  const [result, setResult] = useState('')
  const [inserted, setInserted] = useState(false)
  const [error, setError] = useState('')
  const [level, setLevel] = useState(0)
  const [transcript, setTranscript] = useState('')
  const [mode, setMode] = useState('clean')
  const [modes, setModes] = useState<CleanupMode[]>([])
  const [note, setNote] = useState('') // AI off / error feedback
  const [loadPct, setLoadPct] = useState<number | null>(null) // whisper model download/load %
  const [dlPct, setDlPct] = useState(0) // on-device AI model download %
  const [engine, setEngine] = useState('') // active whisper model · device
  const recRef = useRef<VoiceRecorder | null>(null)
  const phaseRef = useRef<Phase>('idle')
  phaseRef.current = phase
  const modeRef = useRef('clean')
  modeRef.current = mode
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const settingsRef = useRef<YapperSettings | null>(null)
  const histIdRef = useRef<number | null>(null)
  // Cancel plumbing: a flag the in-flight pipeline checks after each await, an
  // AbortController for the AI cleanup, and the audio we've already captured so a
  // cancel can save it as a non-transcribed card instead of throwing it away.
  const cancelledRef = useRef(false)
  const abortRef = useRef<AbortController | null>(null)
  const pendingRef = useRef<{ id: number; mode: string; label?: string; audioPath?: string } | null>(null)
  // Auto-stop a recording that hits the configured max length (a forgotten recorder
  // can't run unbounded). autoStopRef flags it so the result card explains why it ended.
  const limitTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const autoStopRef = useRef(false)
  // Accumulates cleaned text as it streams so the pill paints it live; and whether the
  // transcriber wanted GPU but ran on CPU (so we can tell the user why it felt slow).
  const cleanLiveRef = useRef('')
  const downgradedRef = useRef(false)

  const clearHide = (): void => {
    if (hideTimer.current) clearTimeout(hideTimer.current)
    hideTimer.current = null
  }
  const clearLimit = (): void => {
    if (limitTimer.current) clearTimeout(limitTimer.current)
    limitTimer.current = null
  }
  const armHide = (ms = 5000): void => {
    clearHide()
    hideTimer.current = setTimeout(() => void window.yapper?.overlayHide(), ms)
  }

  const start = async (modeId: string): Promise<void> => {
    clearHide()
    clearLimit()
    setResult('')
    setError('')
    setNote('')
    setLoadPct(null)
    cancelledRef.current = false
    autoStopRef.current = false
    downgradedRef.current = false
    cleanLiveRef.current = ''
    pendingRef.current = null
    abortRef.current = null
    modeRef.current = modeId // sync so a fast hold reads the right mode at stop
    setMode(modeId)
    try {
      const rec = new VoiceRecorder({ onLevel: setLevel })
      await rec.start()
      recRef.current = rec
      setPhase('recording')
      // Mute the speakers + pause any playing media for the duration of the dictation.
      void window.yapper?.audioDuckStart({
        mute: settingsRef.current?.muteWhileDictating ?? true,
        pauseMedia: settingsRef.current?.pauseMediaWhileDictating ?? true
      })
      // Cap the recording length — auto-stop + process when the limit is reached.
      const mins = Math.min(30, Math.max(1, settingsRef.current?.maxRecordingMinutes ?? 20))
      limitTimer.current = setTimeout(() => {
        if (phaseRef.current === 'recording') {
          autoStopRef.current = true
          void stopAndProcess()
        }
      }, mins * 60_000)
    } catch (e) {
      setError('Microphone unavailable: ' + (e as Error).message)
      setPhase('error')
    }
  }

  const stopAndProcess = async (): Promise<void> => {
    const rec = recRef.current
    if (!rec) return
    clearLimit()
    void window.yapper?.audioDuckStop() // dictation is over — unmute + resume media now
    recRef.current = null
    const activeMode = modeRef.current
    setPhase('transcribing')
    let blob: Blob
    try {
      blob = await rec.stop()
    } catch (e) {
      setError((e as Error).message)
      setPhase('error')
      return
    }
    const settings = await loadSettings()
    settingsRef.current = settings
    setModes(settings.modes)
    const modelShort = settings.whisperModel.split('/').pop()
    setEngine(`${modelShort} · …`)
    // Save a card even on failure, so it's recoverable on Welcome (replay / re-transcribe).
    const id = Date.now()
    histIdRef.current = id
    const label = findMode(settings, activeMode)?.label
    let audioPath: string | undefined
    if (settings.keepAudio && blob.size) {
      try {
        const bytes = new Uint8Array(await blob.arrayBuffer())
        const r = await window.yapper?.recordingSave(`dictation-${Date.now()}.webm`, bytes)
        audioPath = r?.path
      } catch {
        /* non-fatal */
      }
    }
    // Audio is now persisted — record it so a cancel can keep it as a non-transcribed card.
    pendingRef.current = { id, mode: activeMode, label, audioPath }
    const ac = new AbortController()
    abortRef.current = ac
    if (cancelledRef.current) return // cancelled while we were saving the audio
    // First run with on-device AI: fetch the model once, with a progress bar.
    await ensureLocalReady(settings)
    if (cancelledRef.current) return
    setPhase('transcribing')
    const saveFailed = async (why: string): Promise<void> => {
      setLoadPct(null)
      setError(why)
      setPhase('error')
      if (audioPath || blob.size) {
        await addHistory({ id, at: id, transcript: '', cleaned: '', mode: activeMode, modeLabel: label, audioPath, failed: true })
      }
      armHide(2800)
    }

    let out: DictationResult
    try {
      out = await processAudio(blob, {
        brain: settings.brain,
        model: settings.whisperModel,
        device: settings.device,
        language: settings.language,
        prompt: modePrompt(settings, activeMode),
        effort: modeEffort(settings, activeMode),
        onPhase: (ph) => {
          setPhase(ph)
          if (ph === 'cleaning') {
            cleanLiveRef.current = ''
            setResult('') // clear so streamed cleaned text paints from empty
          }
        },
        onModelProgress: (pct) => setLoadPct(pct),
        onDevice: (dev, downgraded) => {
          downgradedRef.current = downgraded
          setEngine(downgraded ? `${modelShort} · CPU (GPU can’t fit it — Base is faster)` : `${modelShort} · ${dev === 'webgpu' ? 'GPU' : 'CPU'}`)
        },
        onCleanToken: (t) => {
          cleanLiveRef.current += t
          setResult(cleanLiveRef.current)
        },
        signal: ac.signal
      })
    } catch (e) {
      if (cancelledRef.current) return
      await saveFailed('Transcription failed: ' + (e as Error).message)
      return
    }
    if (cancelledRef.current) return // cancelled during transcription — cancel() saved the card
    setLoadPct(null)
    if (!out.transcript.trim()) {
      await saveFailed('Didn’t catch anything — saved so you can replay or re-transcribe.')
      return
    }
    let didInsert = false
    if (settings.autoInsert) {
      const r = await window.yapper?.clipboardPaste(out.cleaned, { restore: settings.restoreClipboard })
      didInsert = Boolean(r?.pasted)
    } else {
      await window.yapper?.clipboardWrite(out.cleaned)
    }
    if (cancelledRef.current) return // cancelled at the very last moment — don't also save a result card
    await addHistory({ id, at: id, transcript: out.transcript, cleaned: out.cleaned, mode: activeMode, modeLabel: label, audioPath })
    setTranscript(out.transcript)
    setInserted(didInsert)
    setResult(out.cleaned)
    setNote(
      [
        downgradedRef.current ? `Ran on CPU — ${modelShort} is heavy for your GPU; pick Base in Settings for speed.` : '',
        autoStopRef.current ? `Reached the ${settings.maxRecordingMinutes}-min recording limit — stopped and processed.` : '',
        noteFor(out.status, out.error)
      ]
        .filter(Boolean)
        .join(' ')
    )
    setPhase('done')
    armHide()
  }

  /**
   * Cancel an in-flight dictation. Stops any active recorder, aborts the AI pass,
   * and SAVES the captured audio as a non-transcribed (failed) card on Welcome so
   * it can be replayed / re-transcribed later — never transcribes it anyway.
   * When nothing is in flight it just hides the overlay.
   */
  const cancel = async (): Promise<void> => {
    const p = phaseRef.current
    if (p === 'idle' || p === 'done' || p === 'error') {
      clearHide()
      void window.yapper?.overlayHide()
      return
    }
    cancelledRef.current = true
    abortRef.current?.abort()
    clearHide()
    clearLimit()
    void window.yapper?.audioDuckStop() // restore speakers + media on cancel too
    const settings = settingsRef.current
    const activeMode = modeRef.current
    let audioPath = pendingRef.current?.audioPath
    let id = pendingRef.current?.id ?? null
    const label = pendingRef.current?.label ?? (settings ? findMode(settings, activeMode)?.label : undefined)
    // Still recording (cancelled before stop): stop now and persist the audio.
    const rec = recRef.current
    if (rec) {
      recRef.current = null
      try {
        const blob = await rec.stop()
        if ((settings?.keepAudio ?? true) && blob.size) {
          const bytes = new Uint8Array(await blob.arrayBuffer())
          const r = await window.yapper?.recordingSave(`dictation-${Date.now()}.webm`, bytes)
          audioPath = r?.path
        }
      } catch {
        /* ignore */
      }
    }
    // Persist a non-transcribed card (the pipeline hasn't written one yet at cancel time).
    if (audioPath) {
      if (id == null) id = Date.now()
      await addHistory({ id, at: id, transcript: '', cleaned: '', mode: activeMode, modeLabel: label, audioPath, failed: true })
    }
    pendingRef.current = null
    abortRef.current = null
    setResult('')
    setError('')
    setNote('')
    setLoadPct(null)
    setLevel(0)
    setPhase('idle')
    void window.yapper?.overlayHide()
  }

  /** Re-run the cleanup with a different mode on the already-captured transcript. */
  const reclean = async (newMode: string): Promise<void> => {
    const s = settingsRef.current
    if (!s || !transcript) return
    clearHide()
    setMode(newMode)
    setPhase('cleaning')
    cleanLiveRef.current = ''
    setResult('')
    const r = await cleanTranscript(transcript, s.brain, modePrompt(s, newMode), undefined, {
      effort: modeEffort(s, newMode),
      onToken: (t) => {
        cleanLiveRef.current += t
        setResult(cleanLiveRef.current)
      }
    })
    await window.yapper?.clipboardWrite(r.text)
    if (histIdRef.current != null) await updateHistory(histIdRef.current, { cleaned: r.text, mode: newMode, modeLabel: findMode(s, newMode)?.label })
    setInserted(false)
    setResult(r.text)
    setNote(noteFor(r.status, r.error))
    setPhase('done')
    armHide()
  }
  const copyRaw = (): void => {
    clearHide()
    void window.yapper?.clipboardWrite(transcript)
  }

  /** Ensure the on-device cleanup model for the chosen tier is ready (first run), showing progress.
   *  GPU tiers (web-llm) prepare in the renderer; the CPU 'standard' model downloads via main —
   *  which also covers the fallback when a GPU tier is selected but no GPU is present. */
  const ensureLocalReady = async (settings: YapperSettings): Promise<void> => {
    if (settings.brain.provider !== 'local' || !settings.brain.enabled) return
    const t = LLM_TIERS[normalizeTier(settings.brain.localTier)]
    if (t.engine === 'webllm' && t.webllmModel) {
      try {
        const { webLlmAvailable, webLlmHasModel, webLlmPrepare } = await import('../lib/webLlm')
        if (await webLlmAvailable()) {
          if (await webLlmHasModel(t.webllmModel)) return
          setDlPct(0)
          setPhase('downloading')
          await webLlmPrepare(t.webllmModel, (pct) => setDlPct(pct))
          return
        }
        // No capable GPU: fall through and ensure the CPU standard model (the fallback engine).
      } catch {
        /* fall through to CPU */
      }
    }
    const st = await window.yapper?.localModelStatus('standard')
    if (st?.installed) return
    setDlPct(0)
    setPhase('downloading')
    await new Promise<void>((resolve) => {
      const off = window.yapper?.onLocalModelProgress?.((p) => {
        const d = p as { pct?: number; done?: boolean; error?: string }
        if (d.error || d.done) {
          off?.()
          resolve()
        } else setDlPct(d.pct ?? 0)
      })
      void window.yapper?.localModelDownload('standard')
    })
  }

  // Pill button: stop if recording, else start a manual capture in the default mode.
  const toggle = (): void => {
    const p = phaseRef.current
    if (p === 'recording') void stopAndProcess()
    else if (p === 'transcribing' || p === 'cleaning' || p === 'downloading') return
    else void start(settingsRef.current?.defaultModeId ?? 'clean')
  }

  // Toggle mode (globalShortcut): press flips start → stop → (while processing) cancel.
  const toggleFromShortcut = (modeId: string): void => {
    const p = phaseRef.current
    if (p === 'recording') void stopAndProcess()
    else if (p === 'transcribing' || p === 'cleaning' || p === 'downloading') void cancel()
    else void start(modeId || settingsRef.current?.defaultModeId || 'clean')
  }
  // Hold mode (uiohook): explicit start on key-down, stop on key-up / latch.
  const startFromShortcut = (modeId: string): void => {
    const p = phaseRef.current
    if (p === 'recording' || p === 'transcribing' || p === 'cleaning' || p === 'downloading') return
    void start(modeId || settingsRef.current?.defaultModeId || 'clean')
  }
  const stopFromShortcut = (): void => {
    if (phaseRef.current === 'recording') void stopAndProcess()
  }

  useEffect(() => {
    document.body.classList.add('overlay')
    void loadSettings().then((s) => {
      settingsRef.current = s
      setModes(s.modes)
      setMode(s.defaultModeId)
      // Prewarm the Whisper model in the background with a generous GPU warmup budget so
      // first-run WebGPU shader compilation can finish and the pipeline caches on the GPU.
      void getTranscriber(s.whisperModel, s.device, undefined, 60000).catch(() => {})
    })
    const offToggle = window.yapper?.onDictateToggle(toggleFromShortcut)
    const offStart = window.yapper?.onDictateStart(startFromShortcut)
    const offStop = window.yapper?.onDictateStop(stopFromShortcut)
    return () => {
      offToggle?.()
      offStart?.()
      offStop?.()
      document.body.classList.remove('overlay')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const modeLabel = modes.find((m) => m.id === mode)?.label ?? mode
  const status =
    phase === 'recording'
      ? `Listening… · ${modeLabel}`
      : phase === 'downloading'
        ? `Downloading AI model… ${dlPct}%`
        : phase === 'transcribing'
          ? loadPct != null
            ? `Loading model… ${loadPct}%`
            : 'Transcribing…'
          : phase === 'cleaning'
            ? 'Cleaning up…'
          : phase === 'done'
            ? inserted
              ? 'Inserted ✓'
              : 'Copied ✓'
            : phase === 'error'
              ? 'Couldn’t process'
              : 'Ready to dictate'
  const sub =
    phase === 'recording'
      ? settingsRef.current?.shortcutActivation === 'hold'
        ? 'release to stop · ✕ to cancel'
        : 'press the shortcut again (or ■) to stop · ✕ to cancel'
      : phase === 'transcribing'
        ? engine
          ? `${engine} · ✕ to cancel`
          : '✕ to cancel'
        : phase === 'cleaning'
          ? '✕ to cancel'
          : phase === 'idle'
            ? 'hold a shortcut to dictate — or click ●'
            : ''

  const busy = phase === 'transcribing' || phase === 'cleaning' || phase === 'downloading'
  const pillMod = phase === 'recording' ? ' ov-pill--rec' : busy ? ' ov-pill--busy' : ''
  const levelPct = Math.min(100, Math.round((Math.min(0.35, level) / 0.35) * 100))

  return (
    <div className="ov">
      <div
        className={'ov-pill' + pillMod}
        onMouseEnter={() => {
          if (phaseRef.current === 'done') clearHide()
        }}
        onMouseLeave={() => {
          if (phaseRef.current === 'done') armHide(2400)
        }}
      >
        <div className="ov-top">
          <span className={'ov-dot' + DOT[phase]} />
          <div>
            <div className="ov-status">{status}</div>
            {sub && <div className="ov-sub">{sub}</div>}
          </div>
          <span className="ov-spacer" />
          <button className={'chip' + (phase === 'recording' ? ' chip--rec' : '')} onClick={toggle} disabled={busy} aria-label={phase === 'recording' ? 'Stop' : 'Record'}>
            {phase === 'recording' ? '■' : '●'}
          </button>
          <button
            className="chip ov-x"
            onClick={() => void cancel()}
            aria-label={busy || phase === 'recording' ? 'Cancel' : 'Hide'}
            title={busy || phase === 'recording' ? 'Cancel — keep the recording, skip transcription' : 'Hide'}
          >
            ✕
          </button>
        </div>
        {phase === 'recording' && (
          <div className="ov-meter">
            <div className="ov-meter__fill" style={{ width: levelPct + '%' }} />
          </div>
        )}
        {error && <div className="ov-text" style={{ color: 'var(--rec)' }}>{error}</div>}
        {!error && result && <div className="ov-text">{result}</div>}
        {phase === 'done' && note && <div className="ov-note">{note}</div>}
        {phase === 'done' && transcript && (
          <div className="ov-actions">
            <select className="select ov-mode" value={mode} onChange={(e) => void reclean(e.target.value)} title="Re-process this dictation a different way">
              {modes.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
            <button className="chip" onClick={copyRaw} title="Copy the untouched transcript">
              Copy raw
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
