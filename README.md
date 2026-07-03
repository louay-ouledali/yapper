<p align="center">
  <img src="build/icon.png" width="120" alt="Yapper logo" />
</p>

<h1 align="center">Yapper</h1>

<p align="center">
  <b>Offline, deliberate voice dictation for your desktop.</b><br/>
  Hold a shortcut, speak, and Yapper transcribes and AI-cleans your words into polished text —
  then pastes them straight into whatever app you're in. Everything runs <b>on your machine</b> by default.
</p>

<p align="center">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows%20%7C%20macOS-1f1f24" />
  <img alt="Privacy" src="https://img.shields.io/badge/privacy-on--device-7c4dff" />
  <img alt="License" src="https://img.shields.io/badge/license-MIT-blue" />
</p>

---

## What it is

Yapper is a dictation app in the spirit of Wispr Flow, but **private by default and deliberate by design**.
Instead of streaming a noisy real-time transcript, it records you in high quality, then runs a careful two-pass
pipeline once you're done speaking:

1. **Record** — high-quality mic capture into a frameless overlay "pill".
2. **Transcribe** — batch [Whisper](https://github.com/openai/whisper) (via `transformers.js` / ONNX Runtime),
   GPU-accelerated when your hardware supports it, CPU otherwise.
3. **Clean up** — an AI pass removes filler words, false starts and stutters, fixes grammar and punctuation,
   and corrects mis-heard words from context — without changing your meaning or padding the length.
4. **Deliver** — the result is pasted into the active app (or copied), and saved to your history.

No account, no API key, no internet required. Bring your own cloud model only if you *want* to.

## Highlights

- 🔒 **On-device by default** — transcription and cleanup both run locally. Nothing leaves your machine unless you pick a cloud provider.
- ⌨️ **Global shortcuts** — mode-bound hotkeys that work in any app. *Toggle* (tap to start, tap to stop) or *hold-to-talk* (release to stop, double-tap to latch hands-free).
- 🧹 **Cleanup modes** — built-in **Clean-up**, **AI prompt**, **Email**, **Notes** and **Raw**, plus your own custom modes with custom prompts.
- 🔁 **Re-process anything** — change the mode on any past dictation to re-run the cleanup; edit, copy raw, replay or re-transcribe from the Welcome tab.
- 🧠 **Pick your brain** — on-device (Qwen, two size tiers), a local/cloud **Ollama** server, or any **OpenAI-compatible** endpoint with your own key.
- 🗣️ **15 languages** — auto-detect or pin a language for Whisper.
- 🛟 **Reliable in the background** — lives in the system tray, re-registers shortcuts after sleep/lock, and keeps a watchdog on its hotkeys.
- 🎛️ **Cancel any time** — stop a recording or cancel a transcription mid-way; the audio is kept as a re-transcribable card so nothing is lost.

## Install

### Windows
Grab **`Yapper-Setup-x.y.z.exe`** from the [Releases](../../releases) page and run the installer wizard.
It isn't code-signed yet, so Windows SmartScreen may warn on first run — choose **More info → Run anyway**.

### macOS
A `.dmg` target is configured, but macOS builds must be produced on a Mac (or a macOS CI runner) — see
[Building from source](#building-from-source). The app is unsigned, so on first launch use
**right-click → Open** to get past Gatekeeper.

## Using it

| Action | Default shortcut |
| --- | --- |
| Dictate → **Clean-up** | `Ctrl + Shift + Space` |
| Dictate → **AI prompt** | `Ctrl + Shift + P` |
| Dictate → **Email** | `Ctrl + Shift + E` |

In **Toggle** mode, press the shortcut once to start and again to stop. In **Hold** mode, hold to talk and
release to stop (double-tap to keep recording hands-free). Press the shortcut again — or the **✕** on the pill —
while it's working to cancel; the recording is saved so you can re-transcribe it later.

Everything is configurable in **Settings**: shortcuts, activation style, Whisper model & language, the AI brain,
cleanup modes, auto-paste, and how long to keep audio.

### On-device AI models

Yapper's cleanup runs locally, with two engines behind one **Model** picker in Settings. Each model is a
one-time download, then works fully offline:

| Tier | Model | Engine | Size | Notes |
| --- | --- | --- | --- | --- |
| **Standard** | Qwen 2.5 3B | CPU ([wllama](https://github.com/ngxson/wllama), llama.cpp in WASM) | ~1.9 GB | the zero-setup default — smart cleanup that runs on any machine; downloads on first use |
| **Turbo** | Qwen 2.5 7B | GPU ([web-llm](https://github.com/mlc-ai/web-llm), WebGPU) | ~5.9 GB | noticeably stronger and much faster on a capable GPU |
| **Max** | Llama 3.1 8B | GPU (web-llm, WebGPU) | ~6.3 GB | the most capable: best on long transcripts and tricky context |

The **Turbo/Max** GPU tiers need WebGPU; on a machine without a capable GPU they transparently fall back to the
**Standard** CPU model. GPU models download once from Hugging Face and are then cached and run entirely offline.

Long dictations are cleaned in **chunks and re-joined**, so nothing is dropped no matter how long you talk (a
single recording is capped — 20 min by default, up to 30 — and auto-stops when it hits the limit).

Prefer your own models? Point Yapper at a local/cloud **Ollama** server or any **OpenAI-compatible** endpoint in Settings.

## Privacy

By default, **your voice and text never leave your computer**. Audio is recorded locally, transcribed locally,
and cleaned up by a local model. Saved recordings live in your user-data folder and are auto-pruned on a schedule
you control (transcripts are kept). The only time data is sent off-device is if *you* deliberately choose a cloud
AI provider (a non-loopback Ollama host, an Ollama cloud model, or an OpenAI-compatible endpoint) — and the UI
clearly flags when that's the case.

## Building from source

Requires **Node.js 20+**.

```bash
git clone https://github.com/louay-ouledali/yapper.git
cd yapper
npm install

npm run dev        # run in development
npm run build      # type-check + bundle (electron-vite)
npm run build:win  # build the Windows installer (NSIS) → release/
```

To build the macOS `.dmg`, run on macOS:

```bash
npm run build
npx electron-builder --mac
```

> Native modules (`uiohook-napi`, `@nut-tree-fork/nut-js`) ship prebuilt N-API binaries, so no C++ toolchain is
> needed and `npmRebuild` is disabled in `electron-builder.yml`.

## Tech stack

- [Electron](https://www.electronjs.org/) + [electron-vite](https://electron-vite.org/) + React + TypeScript
- [@huggingface/transformers](https://github.com/huggingface/transformers.js) (Whisper, ONNX Runtime Web, WebGPU)
- [@wllama/wllama](https://github.com/ngxson/wllama) for the on-device cleanup model
- [uiohook-napi](https://github.com/SnosMe/uiohook-napi) for global hotkeys, [@nut-tree-fork/nut-js](https://github.com/nut-tree/nut.js) for paste injection
- [electron-builder](https://www.electron.build/) for packaging

## License

[MIT](LICENSE) © Louay Ouledali
