import Overlay from './components/Overlay'
import MainWindow from './components/MainWindow'

const role = new URLSearchParams(location.search).get('role') || 'main'

export default function App(): JSX.Element {
  return role === 'overlay' ? <Overlay /> : <MainWindow />
}
