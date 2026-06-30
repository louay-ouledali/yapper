import { useEffect, useState } from 'react'
import { type KeyBinding, describeInput, isModifierKey } from '../lib/keybindings'

export interface KeyRecorderProps {
  value: KeyBinding | null
  onChange: (b: KeyBinding | null) => void
}

const modPrefix = (e: { ctrlKey: boolean; altKey: boolean; shiftKey: boolean; metaKey: boolean }): string =>
  `${e.ctrlKey ? 'Ctrl+' : ''}${e.altKey ? 'Alt+' : ''}${e.shiftKey ? '⇧' : ''}${e.metaKey ? 'Win+' : ''}`

/**
 * "Press a key to record" control. The press is captured TWO ways: the DOM keydown
 * drives a live preview + label (code/key); uiohook (main) returns the reliable,
 * layout-agnostic keycode AND the true modifier state used for matching. Modifiers
 * on the stored binding come from uiohook (not the DOM) so the label and matcher
 * always agree. Escape cancels; ✕ clears.
 */
export default function KeyRecorder({ value, onChange }: KeyRecorderProps): JSX.Element {
  const [recording, setRecording] = useState(false)
  const [preview, setPreview] = useState('')

  useEffect(() => {
    if (!recording) return
    setPreview('')
    let dom: { code: string; key: string; keyCode: number } | null = null
    const onKey = (e: KeyboardEvent): void => {
      e.preventDefault()
      e.stopPropagation()
      if (e.key === 'Escape') return
      if (isModifierKey(e.key)) {
        setPreview(modPrefix(e) + '…')
        return
      }
      dom = { code: e.code, key: e.key, keyCode: e.keyCode }
      setPreview(describeInput({ kind: 'key', code: e.code, key: e.key, keyCode: e.keyCode, ctrl: e.ctrlKey, alt: e.altKey, shift: e.shiftKey, meta: e.metaKey }))
    }
    window.addEventListener('keydown', onKey, true)
    let done = false
    void window.yapper?.recordBinding().then((res) => {
      if (done) return
      done = true
      setRecording(false)
      setPreview('')
      if (!res) return // cancelled / timed out
      const d = dom ?? { code: '', key: '', keyCode: 0 }
      onChange({
        kind: 'key',
        code: d.code,
        key: d.key,
        keyCode: d.keyCode,
        uioKeycode: res.keycode,
        ...(res.ctrl ? { ctrl: true } : {}),
        ...(res.alt ? { alt: true } : {}),
        ...(res.shift ? { shift: true } : {}),
        ...(res.meta ? { meta: true } : {})
      })
    })
    return () => {
      done = true
      window.removeEventListener('keydown', onKey, true)
    }
  }, [recording, onChange])

  return (
    <span className="set-inline">
      <button className={'chip' + (recording ? ' chip--rec' : '')} onClick={() => setRecording((r) => !r)}>
        {recording ? 'press shortcut…' : 'Record'}
      </button>
      <span className="kbd">{recording ? preview || '…' : describeInput(value)}</span>
      {value && !recording && (
        <button className="chip ov-x" title="Clear binding" onClick={() => onChange(null)}>
          ✕
        </button>
      )}
    </span>
  )
}
