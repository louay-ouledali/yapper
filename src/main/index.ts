import { app, BrowserWindow, ipcMain, session, clipboard, screen, dialog, Tray, Menu, nativeImage, powerMonitor } from 'electron'
import { join, basename } from 'node:path'
import { mkdirSync } from 'node:fs'
import { readFile, writeFile, rename, readdir, stat, unlink, access, copyFile } from 'node:fs/promises'
import {
  startModelServer,
  setProgressWindow,
  getModelBaseUrl,
  ensureHfFile,
  hfModelSize,
  fetchModel,
  resolveModel,
  downloadedModelsDir,
  llmModelFor
} from './modelServer'
import { applyShortcuts, reapply, restartHook, shortcutsHealthy, recordNext, stopHotkeys, type Activation, type BindingLike, type HotkeyActions, type ShortcutLike } from './hotkeys'

const RENDERER_DEV_URL = process.env['ELECTRON_RENDERER_URL']

/** App icon path that resolves in both dev and the packaged app (see extraResources). */
const iconPath = (): string => (app.isPackaged ? join(process.resourcesPath, 'icon.png') : join(app.getAppPath(), 'build', 'icon.png'))

let tray: Tray | null = null
let quitting = false

// Enable WebGPU for the on-device Whisper/LLM GPU path. Electron can ship WebGPU
// gated behind a flag and may blocklist newer GPUs — force it on so `navigator.gpu`
// and a real hardware adapter are available to ONNX Runtime. (No Vulkan override —
// forcing the Vulkan backend can crash launch on some Windows GPU drivers.)
app.commandLine.appendSwitch('enable-unsafe-webgpu')
app.commandLine.appendSwitch('ignore-gpu-blocklist')

// Keep DEV data in its own userData folder so running `npm run dev` never pollutes a
// real install (both default to %APPDATA%/Yapper and would otherwise share history,
// recordings and downloaded models). Must run before the app is ready.
if (!app.isPackaged) {
  try {
    app.setPath('userData', `${app.getPath('userData')}-dev`)
  } catch {
    /* fall back to the default path */
  }
}

let mainWin: BrowserWindow | null = null
let overlayWin: BrowserWindow | null = null

function loadRole(win: BrowserWindow, role: 'main' | 'overlay'): void {
  if (RENDERER_DEV_URL) win.loadURL(`${RENDERER_DEV_URL}?role=${role}`)
  else win.loadFile(join(__dirname, '../renderer/index.html'), { query: { role } })
}

function createMainWindow(): void {
  mainWin = new BrowserWindow({
    width: 1040,
    height: 720,
    minWidth: 720,
    minHeight: 520,
    show: false,
    autoHideMenuBar: true,
    backgroundColor: '#060508', // pitch black; the renderer paints the purple glow
    title: 'Yapper',
    icon: iconPath(),
    webPreferences: { preload: join(__dirname, '../preload/index.js'), sandbox: false, backgroundThrottling: false }
  })
  mainWin.on('ready-to-show', () => mainWin?.show())
  // Close = hide to tray (keep running so global shortcuts stay live); real quit via tray/before-quit.
  mainWin.on('close', (e) => {
    if (!quitting) {
      e.preventDefault()
      mainWin?.hide()
    }
  })
  mainWin.on('closed', () => {
    mainWin = null
  })
  setProgressWindow(mainWin)
  loadRole(mainWin, 'main')
}

/** Reveal + focus the main window (from the tray, or relaunch). */
function showMainWindow(): void {
  if (!mainWin || mainWin.isDestroyed()) createMainWindow()
  else {
    mainWin.show()
    mainWin.focus()
  }
}

/** System-tray presence so Yapper keeps running (and shortcuts keep firing) in the background. */
function createTray(): void {
  if (tray) return
  let img = nativeImage.createFromPath(iconPath())
  if (!img.isEmpty()) img = img.resize({ width: 18, height: 18 })
  tray = new Tray(img.isEmpty() ? nativeImage.createEmpty() : img)
  tray.setToolTip('Yapper — offline dictation')
  const menu = Menu.buildFromTemplate([
    { label: 'Open Yapper', click: () => showMainWindow() },
    { type: 'separator' },
    {
      label: 'Quit Yapper',
      click: () => {
        quitting = true
        app.quit()
      }
    }
  ])
  tray.setContextMenu(menu)
  tray.on('click', () => showMainWindow())
  tray.on('double-click', () => showMainWindow())
}

/** A compact, frameless, always-on-top recorder pill near the bottom of the screen. */
function createOverlayWindow(): void {
  const { workArea } = screen.getPrimaryDisplay()
  // Roomy transparent margin around the pill so its drop-shadow fades fully BEFORE the
  // window edge (a shadow clipped at the window bounds is what drew the faint rectangle).
  const w = 470
  const h = 176
  overlayWin = new BrowserWindow({
    width: w,
    height: h,
    x: Math.round(workArea.x + (workArea.width - w) / 2),
    y: Math.round(workArea.y + workArea.height - h - 48),
    frame: false,
    transparent: true,
    resizable: false,
    movable: true,
    skipTaskbar: true,
    alwaysOnTop: true,
    show: false,
    focusable: true,
    hasShadow: false, // the OS shadow renders as a faint sharp-cornered border on a transparent window
    roundedCorners: true,
    backgroundColor: '#00000000',
    webPreferences: { preload: join(__dirname, '../preload/index.js'), sandbox: false, backgroundThrottling: false }
  })
  overlayWin.setAlwaysOnTop(true, 'screen-saver')
  overlayWin.on('closed', () => {
    overlayWin = null
  })
  loadRole(overlayWin, 'overlay')
}

function showOverlay(): void {
  if (!overlayWin || overlayWin.isDestroyed()) createOverlayWindow()
  overlayWin?.showInactive()
}

/** A shortcut fired: reveal the overlay and start recording in the given mode. */
function startDictation(modeId: string): void {
  showOverlay()
  overlayWin?.webContents.send('dictate:start', modeId)
}
/** Stop recording and process. */
function stopDictation(): void {
  overlayWin?.webContents.send('dictate:stop')
}
/** Toggle (reveal overlay; it flips start/stop based on its own state) in the given mode. */
function toggleDictation(modeId: string): void {
  showOverlay()
  overlayWin?.webContents.send('dictate:toggle', modeId)
}

const hotkeyActions: HotkeyActions = {
  start: startDictation,
  stop: stopDictation,
  toggle: toggleDictation,
  showOverlay: () => showOverlay()
}

/** Default shortcuts before the renderer has ever saved settings (mirrors keybindings.ts). */
const DEFAULT_SHORTCUTS: ShortcutLike[] = [
  { id: 'sc-clean', binding: { code: 'Space', keyCode: 32, ctrl: true, shift: true }, modeId: 'clean' },
  { id: 'sc-prompt', binding: { code: 'KeyP', keyCode: 80, ctrl: true, shift: true }, modeId: 'prompt' },
  { id: 'sc-email', binding: { code: 'KeyE', keyCode: 69, ctrl: true, shift: true }, modeId: 'email' }
]
async function loadShortcuts(): Promise<{ shortcuts: ShortcutLike[]; showOverlay: BindingLike | null; activation: Activation }> {
  const data = await readStore()
  const s = (data['settings'] as { shortcuts?: ShortcutLike[]; showOverlayBinding?: BindingLike | null; shortcutActivation?: Activation } | undefined) || {}
  return {
    shortcuts: Array.isArray(s.shortcuts) ? s.shortcuts : DEFAULT_SHORTCUTS,
    showOverlay: s.showOverlayBinding ?? null,
    activation: s.shortcutActivation === 'hold' ? 'hold' : 'toggle'
  }
}

// ── atomic userData store (serialized read-modify-write) ──────────────────────
const storePath = (): string => join(app.getPath('userData'), 'yapper-store.json')
let storeQueue: Promise<void> = Promise.resolve()
async function readStore(): Promise<Record<string, unknown>> {
  try {
    return JSON.parse(await readFile(storePath(), 'utf8')) as Record<string, unknown>
  } catch {
    return {}
  }
}
function mutateStore(mutate: (data: Record<string, unknown>) => void): Promise<void> {
  storeQueue = storeQueue.then(async () => {
    const data = await readStore()
    mutate(data)
    const tmp = `${storePath()}.tmp`
    await writeFile(tmp, JSON.stringify(data))
    await rename(tmp, storePath())
  })
  return storeQueue
}
ipcMain.handle('store:get', async (_e, key: string) => (await readStore())[key] ?? null)
ipcMain.handle('store:set', async (_e, key: string, value: unknown) => {
  await mutateStore((d) => {
    d[key] = value
  })
  // Any history write (from the overlay or the main window) refreshes open windows.
  if (key === 'history') {
    for (const w of BrowserWindow.getAllWindows()) if (!w.isDestroyed()) w.webContents.send('history:changed')
  }
  return true
})

// ── model info + downloads ────────────────────────────────────────────────────
ipcMain.handle('models:info', () => {
  let gpuStatus = ''
  try {
    gpuStatus = (app.getGPUFeatureStatus() as unknown as Record<string, string>)?.webgpu ?? ''
  } catch {
    /* ignore */
  }
  return { baseUrl: getModelBaseUrl(), gpuHardware: gpuStatus === 'enabled', gpuStatus }
})
ipcMain.handle('models:hfSize', (_e, model: string) => hfModelSize(String(model)))
ipcMain.handle('models:hfEnsure', async (_e, rel: string) => {
  try {
    await ensureHfFile(String(rel))
    return { ok: true }
  } catch (e) {
    return { ok: false, error: (e as Error).message }
  }
})
ipcMain.handle('localmodel:status', (_e, tier?: string) => {
  const { file } = llmModelFor(tier)
  const resolved = resolveModel(file)
  return { installed: Boolean(resolved), url: resolved ? `${getModelBaseUrl()}/${file}` : '', file }
})
ipcMain.handle('localmodel:download', (_e, tier?: string) => {
  const { file, url } = llmModelFor(tier)
  return fetchModel(url, file, 'localmodel:progress')
})

// ── recordings ────────────────────────────────────────────────────────────────
const recordingsDir = (): string => join(app.getPath('userData'), 'recordings')
/** Guard: only ever touch files that live inside our recordings dir. */
const inRecordings = (p: string): boolean => join(recordingsDir(), basename(p)) === p
const fileExists = async (p: string): Promise<boolean> => {
  try {
    await access(p)
    return true
  } catch {
    return false
  }
}

ipcMain.handle('recording:save', async (_e, name: string, bytes: Uint8Array) => {
  try {
    const dir = recordingsDir()
    mkdirSync(dir, { recursive: true })
    const safe = String(name).replace(/[^\w.\-]+/g, '_')
    const path = join(dir, safe)
    await writeFile(path, Buffer.from(bytes))
    return { path }
  } catch (e) {
    return { error: (e as Error).message }
  }
})
ipcMain.handle('recording:read', async (_e, path: string) => {
  try {
    if (!inRecordings(String(path))) return { error: 'outside recordings dir' }
    return { bytes: new Uint8Array(await readFile(String(path))) }
  } catch (e) {
    return { error: (e as Error).message }
  }
})
ipcMain.handle('recording:exists', (_e, path: string) => (inRecordings(String(path)) ? fileExists(String(path)) : false))
ipcMain.handle('recording:delete', async (_e, path: string) => {
  try {
    if (inRecordings(String(path))) await unlink(String(path)).catch(() => {})
    return true
  } catch {
    return false
  }
})
ipcMain.handle('recording:export', async (_e, path: string) => {
  try {
    if (!inRecordings(String(path)) || !(await fileExists(String(path)))) return { ok: false, error: 'recording not found' }
    const res = await dialog.showSaveDialog(mainWin ?? undefined!, {
      title: 'Export voice recording',
      defaultPath: basename(String(path)),
      filters: [{ name: 'Audio', extensions: ['webm'] }]
    })
    if (res.canceled || !res.filePath) return { ok: false, canceled: true }
    await copyFile(String(path), res.filePath)
    return { ok: true }
  } catch (e) {
    return { ok: false, error: (e as Error).message }
  }
})

// ── retention: delete audio older than retentionDays (transcripts are kept) ──────
async function runRetentionCleanup(): Promise<{ deleted: number }> {
  const settings = (await readStore())['settings'] as { retentionDays?: number } | undefined
  const days = settings?.retentionDays ?? 0
  if (!days || days <= 0) return { deleted: 0 } // 0 = keep forever
  const cutoff = Date.now() - days * 86_400_000
  let names: string[] = []
  try {
    names = await readdir(recordingsDir())
  } catch {
    return { deleted: 0 } // no recordings dir yet
  }
  let deleted = 0
  for (const name of names) {
    const p = join(recordingsDir(), name)
    try {
      const st = await stat(p)
      if (st.isFile() && st.mtimeMs < cutoff) {
        await unlink(p)
        deleted++
      }
    } catch {
      /* ignore */
    }
  }
  if (deleted > 0) {
    // Drop audioPath on history items whose file is now gone.
    const hist = ((await readStore())['history'] as Array<{ audioPath?: string }>) || []
    const cleared = await Promise.all(hist.map(async (h) => (h.audioPath && !(await fileExists(h.audioPath)) ? { ...h, audioPath: undefined } : h)))
    await mutateStore((d) => {
      d['history'] = cleared
    })
  }
  return { deleted }
}
ipcMain.handle('retention:cleanup', () => runRetentionCleanup())

// ── clipboard / insertion ────────────────────────────────────────────────────────
ipcMain.handle('clipboard:write', (_e, text: string) => {
  clipboard.writeText(String(text ?? ''))
  return true
})
/**
 * Auto-insert: put the text on the clipboard, then synthesize Ctrl+V into whatever
 * app currently has focus (the overlay is shown without stealing focus). On success
 * we optionally restore the user's previous clipboard a moment later; on failure we
 * leave the text on the clipboard so it can still be pasted manually.
 */
ipcMain.handle('clipboard:paste', async (_e, text: string, opts?: { restore?: boolean }) => {
  const restore = Boolean(opts?.restore)
  const prev = restore ? clipboard.readText() : null
  clipboard.writeText(String(text ?? ''))
  try {
    const { keyboard, Key } = await import('@nut-tree-fork/nut-js')
    keyboard.config.autoDelayMs = 0
    await keyboard.type(Key.LeftControl, Key.V)
  } catch (err) {
    console.error('[paste] nut.js failed — text left on clipboard for manual paste', err)
    return { ok: false, pasted: false, error: (err as Error).message }
  }
  if (prev != null) setTimeout(() => clipboard.writeText(prev), 250)
  return { ok: true, pasted: true }
})

// ── overlay control ─────────────────────────────────────────────────────────────
ipcMain.handle('overlay:hide', () => {
  overlayWin?.hide()
  return true
})
ipcMain.handle('overlay:resize', (_e, h: number) => {
  if (overlayWin && !overlayWin.isDestroyed()) {
    const [w] = overlayWin.getSize()
    overlayWin.setSize(w, Math.max(80, Math.round(h)))
  }
  return true
})

// ── keybindings (re-register live when Settings change) ──────────────────────────
let lastShortcutFailed: string[] = []
ipcMain.handle('bindings:apply', (_e, shortcuts: ShortcutLike[], showOverlay: BindingLike | null, activation: Activation) => {
  // Returns the ids of shortcuts that couldn't be registered (so the UI can warn).
  lastShortcutFailed = applyShortcuts(Array.isArray(shortcuts) ? shortcuts : [], showOverlay ?? null, hotkeyActions, activation === 'hold' ? 'hold' : 'toggle')
  return lastShortcutFailed
})
// Read the failed-to-register ids without re-registering (Settings UI reads this on mount).
ipcMain.handle('bindings:status', () => lastShortcutFailed)
// Capture the next physical key via uiohook for the shortcut recorder (layout-agnostic).
ipcMain.handle('binding:record', () => recordNext())

// ── LLM proxy (OpenAI-compatible / Ollama) — kept out of page script for CORS + key safety ──
ipcMain.handle('llm:request', async (_e, url: string, init: { method: string; headers: Record<string, string>; body: string }) => {
  try {
    const res = await fetch(url, { method: init.method, headers: init.headers, body: init.body })
    return { ok: res.ok, status: res.status, text: await res.text() }
  } catch (e) {
    return { ok: false, status: 0, text: '', error: (e as Error).message }
  }
})

const llmStreams = new Map<string, AbortController>()
ipcMain.handle('llm:stream', async (e, id: string, url: string, init: { method: string; headers: Record<string, string>; body: string }) => {
  const ctrl = new AbortController()
  llmStreams.set(id, ctrl)
  const wc = e.sender
  const send = (channel: string, payload: unknown): void => {
    if (!wc.isDestroyed()) wc.send(channel, payload)
  }
  try {
    const res = await fetch(url, { method: init.method, headers: init.headers, body: init.body, signal: ctrl.signal })
    if (!res.ok || !res.body) {
      const t = res.body ? await res.text().catch(() => '') : ''
      send('llm:streamError', { id, error: `HTTP ${res.status}${t ? `: ${t.slice(0, 200)}` : ''}` })
      return { ok: false }
    }
    const sse = (res.headers.get('content-type') || '').includes('event-stream')
    const reader = res.body.getReader()
    const dec = new TextDecoder()
    let buf = ''
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      let nl: number
      while ((nl = buf.indexOf('\n')) >= 0) {
        let line = buf.slice(0, nl).trim()
        buf = buf.slice(nl + 1)
        if (!line) continue
        if (sse) {
          if (!line.startsWith('data:')) continue
          line = line.slice(5).trim()
          if (line === '[DONE]') continue
          try {
            const tok = JSON.parse(line)?.choices?.[0]?.delta?.content
            if (tok) send('llm:streamChunk', { id, token: tok })
          } catch {
            /* skip */
          }
        } else {
          try {
            const o = JSON.parse(line)
            const tok = o?.message?.content
            if (tok) send('llm:streamChunk', { id, token: tok })
            if (o?.error) send('llm:streamError', { id, error: String(o.error) })
          } catch {
            /* skip */
          }
        }
      }
    }
    send('llm:streamDone', { id })
    return { ok: true }
  } catch (err) {
    if (ctrl.signal.aborted) {
      send('llm:streamDone', { id, aborted: true })
      return { ok: true }
    }
    send('llm:streamError', { id, error: (err as Error).message })
    return { ok: false }
  } finally {
    llmStreams.delete(id)
  }
})
ipcMain.handle('llm:streamAbort', (_e, id: string) => {
  llmStreams.get(id)?.abort()
  return true
})
ipcMain.handle('ollama:tags', async (_e, baseUrl: string) => {
  try {
    const res = await fetch(`${baseUrl.replace(/\/+$/, '')}/api/tags`)
    if (!res.ok) return { ok: false, error: `HTTP ${res.status}` }
    const j = (await res.json()) as { models?: Array<{ name: string; size?: number }> }
    return { ok: true, models: j.models ?? [] }
  } catch (e) {
    return { ok: false, error: (e as Error).message }
  }
})
ipcMain.handle('ollama:warm', async (_e, baseUrl: string, model: string) => {
  try {
    await fetch(`${baseUrl.replace(/\/+$/, '')}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, messages: [{ role: 'user', content: 'hi' }], stream: false, think: false, options: { num_predict: 1 } })
    })
    return { ok: true }
  } catch (e) {
    return { ok: false, error: (e as Error).message }
  }
})

// ── lifecycle ─────────────────────────────────────────────────────────────────
app.whenReady().then(async () => {
  // The overlay records the mic — grant media, deny the rest. All on-device.
  session.defaultSession.setPermissionRequestHandler((_wc, permission, cb) => cb(permission === 'media'))
  session.defaultSession.setPermissionCheckHandler((_wc, permission) => permission === 'media')

  startModelServer()
  createMainWindow()
  createOverlayWindow()
  createTray()

  // Register the user's mode-bound global shortcuts (hold / double-tap) via the
  // uiohook key hook. See hotkeys.ts.
  const sc = await loadShortcuts()
  lastShortcutFailed = applyShortcuts(sc.shortcuts, sc.showOverlay, hotkeyActions, sc.activation)

  // Reliability: OS hotkeys can be dropped after sleep/lock or grabbed transiently.
  // Re-register on resume/unlock, and a light watchdog re-applies if any go missing.
  powerMonitor.on('resume', () => restartHook())
  powerMonitor.on('unlock-screen', () => restartHook())
  setInterval(() => {
    if (!shortcutsHealthy()) reapply()
  }, 15000)

  // Retention: sweep old audio on startup, then every 6 hours.
  const sweep = (): void => {
    void runRetentionCleanup().then((r) => {
      if (r.deleted > 0 && mainWin && !mainWin.isDestroyed()) mainWin.webContents.send('history:changed')
    })
  }
  sweep()
  setInterval(sweep, 6 * 60 * 60 * 1000)

  app.on('activate', () => showMainWindow())
})

app.on('before-quit', () => {
  quitting = true
})
app.on('will-quit', () => stopHotkeys())
// Keep running in the tray when all windows are closed (the overlay stays hidden but
// alive, and global shortcuts keep working). Quit only via the tray menu.
app.on('window-all-closed', () => {
  /* intentionally no-op — Yapper lives in the tray */
})
