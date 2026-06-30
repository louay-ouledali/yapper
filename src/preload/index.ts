import { contextBridge, ipcRenderer } from 'electron'

// The renderer bridge is exposed as `window.yapper`; the scavenged voice/LLM
// libs call `window.yapper.*`.
const api = {
  // model server / downloads
  modelInfo: (): Promise<{ baseUrl: string; gpuHardware: boolean; gpuStatus: string }> => ipcRenderer.invoke('models:info'),
  hfModelSize: (model: string): Promise<number> => ipcRenderer.invoke('models:hfSize', model),
  hfEnsure: (rel: string): Promise<{ ok: boolean; error?: string }> => ipcRenderer.invoke('models:hfEnsure', rel),
  onHfProgress: (cb: (p: { path: string; received: number; total: number; pct: number }) => void): (() => void) => {
    const h = (_e: unknown, p: unknown): void => cb(p as { path: string; received: number; total: number; pct: number })
    ipcRenderer.on('models:hfProgress', h)
    return () => ipcRenderer.removeListener('models:hfProgress', h)
  },
  localModelStatus: (tier?: string): Promise<{ installed: boolean; url: string; file: string }> => ipcRenderer.invoke('localmodel:status', tier),
  localModelDownload: (tier?: string): Promise<{ ok: boolean; error?: string }> => ipcRenderer.invoke('localmodel:download', tier),
  onLocalModelProgress: (cb: (p: unknown) => void): (() => void) => {
    const h = (_e: unknown, p: unknown): void => cb(p)
    ipcRenderer.on('localmodel:progress', h)
    return () => ipcRenderer.removeListener('localmodel:progress', h)
  },

  // store
  storeGet: <T>(key: string): Promise<T | null> => ipcRenderer.invoke('store:get', key),
  storeSet: (key: string, value: unknown): Promise<boolean> => ipcRenderer.invoke('store:set', key, value),

  // LLM proxy
  llmRequest: (
    url: string,
    init: { method: string; headers: Record<string, string>; body: string }
  ): Promise<{ ok: boolean; status: number; text: string; error?: string }> => ipcRenderer.invoke('llm:request', url, init),
  llmStream: (
    id: string,
    url: string,
    init: { method: string; headers: Record<string, string>; body: string }
  ): Promise<{ ok: boolean }> => ipcRenderer.invoke('llm:stream', id, url, init),
  llmStreamAbort: (id: string): Promise<boolean> => ipcRenderer.invoke('llm:streamAbort', id),
  onLlmStream: (
    cb: (m: { kind: 'chunk' | 'done' | 'error'; id: string; token?: string; error?: string; aborted?: boolean }) => void
  ): (() => void) => {
    const chunk = (_e: unknown, p: { id: string; token: string }): void => cb({ kind: 'chunk', ...p })
    const done = (_e: unknown, p: { id: string; aborted?: boolean }): void => cb({ kind: 'done', ...p })
    const err = (_e: unknown, p: { id: string; error: string }): void => cb({ kind: 'error', ...p })
    ipcRenderer.on('llm:streamChunk', chunk)
    ipcRenderer.on('llm:streamDone', done)
    ipcRenderer.on('llm:streamError', err)
    return () => {
      ipcRenderer.removeListener('llm:streamChunk', chunk)
      ipcRenderer.removeListener('llm:streamDone', done)
      ipcRenderer.removeListener('llm:streamError', err)
    }
  },
  ollamaTags: (baseUrl: string): Promise<{ ok: boolean; models?: Array<{ name: string; size?: number }>; error?: string }> =>
    ipcRenderer.invoke('ollama:tags', baseUrl),
  ollamaWarm: (baseUrl: string, model: string): Promise<{ ok: boolean; error?: string }> => ipcRenderer.invoke('ollama:warm', baseUrl, model),

  // dictation / overlay / output
  recordingSave: (name: string, bytes: Uint8Array): Promise<{ path?: string; error?: string }> => ipcRenderer.invoke('recording:save', name, bytes),
  recordingRead: (path: string): Promise<{ bytes?: Uint8Array; error?: string }> => ipcRenderer.invoke('recording:read', path),
  recordingExists: (path: string): Promise<boolean> => ipcRenderer.invoke('recording:exists', path),
  recordingDelete: (path: string): Promise<boolean> => ipcRenderer.invoke('recording:delete', path),
  retentionCleanup: (): Promise<{ deleted: number }> => ipcRenderer.invoke('retention:cleanup'),
  onHistoryChanged: (cb: () => void): (() => void) => {
    const h = (): void => cb()
    ipcRenderer.on('history:changed', h)
    return () => ipcRenderer.removeListener('history:changed', h)
  },
  clipboardWrite: (text: string): Promise<boolean> => ipcRenderer.invoke('clipboard:write', text),
  clipboardPaste: (
    text: string,
    opts?: { restore?: boolean }
  ): Promise<{ ok: boolean; pasted: boolean; error?: string }> => ipcRenderer.invoke('clipboard:paste', text, opts),
  overlayHide: (): Promise<boolean> => ipcRenderer.invoke('overlay:hide'),
  overlayResize: (h: number): Promise<boolean> => ipcRenderer.invoke('overlay:resize', h),
  onDictateToggle: (cb: (modeId: string) => void): (() => void) => {
    const h = (_e: unknown, modeId: string): void => cb(modeId)
    ipcRenderer.on('dictate:toggle', h)
    return () => ipcRenderer.removeListener('dictate:toggle', h)
  },
  onDictateStart: (cb: (modeId: string) => void): (() => void) => {
    const h = (_e: unknown, modeId: string): void => cb(modeId)
    ipcRenderer.on('dictate:start', h)
    return () => ipcRenderer.removeListener('dictate:start', h)
  },
  onDictateStop: (cb: () => void): (() => void) => {
    const h = (): void => cb()
    ipcRenderer.on('dictate:stop', h)
    return () => ipcRenderer.removeListener('dictate:stop', h)
  },

  // shortcuts — re-register the mode-bound global shortcuts live when Settings change
  applyBindings: (shortcuts: unknown, showOverlayBinding: unknown, activation: unknown): Promise<string[]> =>
    ipcRenderer.invoke('bindings:apply', shortcuts, showOverlayBinding, activation),
  bindingsStatus: (): Promise<string[]> => ipcRenderer.invoke('bindings:status'),
  recordBinding: (): Promise<{ keycode: number; ctrl: boolean; alt: boolean; shift: boolean; meta: boolean } | null> => ipcRenderer.invoke('binding:record'),

  // export a saved recording to a user-chosen file
  recordingExport: (path: string): Promise<{ ok: boolean; canceled?: boolean; error?: string }> => ipcRenderer.invoke('recording:export', path)
}

export type YapperApi = typeof api
contextBridge.exposeInMainWorld('yapper', api)
