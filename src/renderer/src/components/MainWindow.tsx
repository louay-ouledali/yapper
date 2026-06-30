import { useCallback, useEffect, useState } from 'react'
import WelcomeTab from './WelcomeTab'
import SettingsTab from './SettingsTab'
import logo from '../assets/logo.png'
import { DEFAULT_SETTINGS, loadHistory, loadSettings, saveSettings, type HistoryItem, type YapperSettings } from '../lib/settings'

type Tab = 'welcome' | 'settings'

export default function MainWindow(): JSX.Element {
  const [s, setS] = useState<YapperSettings>(DEFAULT_SETTINGS)
  const [loaded, setLoaded] = useState(false)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [ollamaModels, setOllamaModels] = useState<Array<{ name: string }>>([])
  const [tab, setTab] = useState<Tab>('welcome')

  const reloadHistory = useCallback(() => void loadHistory().then(setHistory), [])

  useEffect(() => {
    void (async () => {
      setS(await loadSettings())
      setHistory(await loadHistory())
      setLoaded(true)
    })()
    const off = window.yapper?.onHistoryChanged(() => reloadHistory())
    return () => off?.()
  }, [reloadHistory])

  useEffect(() => {
    if (loaded) void saveSettings(s)
  }, [s, loaded])

  const refreshOllama = useCallback(() => {
    if (s.brain.provider !== 'ollama') return
    void window.yapper?.ollamaTags(s.brain.baseUrl).then((r) => setOllamaModels(r?.ok && r.models ? r.models : []))
  }, [s.brain.provider, s.brain.baseUrl])
  useEffect(() => {
    if (loaded) refreshOllama()
  }, [loaded, refreshOllama])

  return (
    <div className="app">
      <header className="topbar">
        <img src={logo} className="brand-logo" alt="" />
        <span className="mw__title">Yapper</span>
        <span className="mw__tag">offline dictation</span>
        <nav className="tabbar">
          <button className={'tab' + (tab === 'welcome' ? ' tab--on' : '')} onClick={() => setTab('welcome')}>Welcome</button>
          <button className={'tab' + (tab === 'settings' ? ' tab--on' : '')} onClick={() => setTab('settings')}>Settings</button>
        </nav>
      </header>
      <main className="appbody">
        {/* Both tabs stay mounted (hidden via CSS) so cards / in-flight state persist across switches. */}
        <div style={{ display: tab === 'welcome' ? 'block' : 'none' }}>
          <WelcomeTab settings={s} history={history} setHistory={setHistory} onOpenSettings={() => setTab('settings')} />
        </div>
        <div style={{ display: tab === 'settings' ? 'block' : 'none' }}>
          <SettingsTab settings={s} setS={setS} ollamaModels={ollamaModels} refreshOllama={refreshOllama} reloadHistory={reloadHistory} />
        </div>
      </main>
    </div>
  )
}
