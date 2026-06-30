/// <reference types="vite/client" />

import type { YapperApi } from '../../preload'

declare global {
  interface Window {
    yapper: YapperApi
  }
}

export {}
