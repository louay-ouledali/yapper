// Minimal typings for the bits of node-tar we use (creating a .tar.gz from a
// directory). Avoids pulling @types/tar for a single call site.
declare module 'tar' {
  interface CreateOptions {
    gzip?: boolean
    file?: string
    cwd?: string
    portable?: boolean
  }
  export function create(options: CreateOptions, paths: ReadonlyArray<string>): Promise<void>
  export { create as c }
}
