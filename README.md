# BreezeMate · 微伴

Realtime speech subtitling and translation on Windows.

```
Audio source ──► ┌─ Vosk (offline) ────────► live preview row
(loopback /      │
 microphone)     └─ canonical ASR ──────────► sentence text ──► [translate] ──► GUI / CLI
                    • Vosk (offline)              (any OpenAI-compatible chat
                    • OpenAI Realtime API          endpoint: OpenAI / Groq /
                      (cloud, high accuracy)       DeepSeek / Ollama / ...)
```

Speech recognition has **two configurable backends**:

- **Local Vosk** (default) — fully offline, free, runs on CPU.
  Models are 30–80 MB per language (small variant); 15+ languages
  including English, 中文, 日本語, 한국어, Русский, Français, Deutsch,
  Español, Italiano, Português, Polski, Українська, Türkçe,
  Tiếng Việt, हिन्दी are downloadable from the GUI on demand. No audio
  leaves the machine, no per-minute fee, no API key needed.
- **OpenAI Realtime API** — cloud-hosted `gpt-4o-(mini-)transcribe`
  through OpenAI's WebSocket streaming endpoint. Higher accuracy and
  punctuation than Vosk, billed per minute. When selected, Vosk
  **still runs in parallel as the preview engine** so the user keeps
  the instant words-as-they-are-spoken UX without waiting for the
  network round-trip — Vosk drives the italic preview row, the cloud
  drives the canonical sentence that gets translated.

Translation is the network leg in either case, and is pluggable: pick
OpenAI, DeepSeek, Groq, OpenRouter, Ollama, LM Studio, or any
OpenAI-compatible endpoint.

Two interfaces:

- **GUI**: floating subtitle overlay, control panel, settings dialog
  with provider switching and Vosk model downloads. Launch with
  `breezemate gui` or `breezemate-gui` (legacy aliases `rt-translator gui` /
  `rt-translator-gui` still work).
- **CLI**: terminal-based subtitle stream via `rich.live`. Launch with
  `breezemate` (alias for `breezemate run`).

> ℹ️ The internal Python package is still named `rt_translator`, and the
> per-user data folder is still `%APPDATA%\rt-translator\`. Both are
> kept stable for backwards compatibility with existing installs and
> downloaded Vosk models.

Two audio sources, same dependency tree:

- **Loopback** — capture whatever Windows is playing (videos, video calls,
  Spotify, anything). Uses WASAPI loopback under the hood, **no virtual
  audio cable needed**.
- **Microphone** — capture a physical mic / line-in / USB audio interface,
  for translating sound coming from an external phone, TV, or another
  laptop sitting next to you.

Two modes, switchable at runtime:

- `asr_only` — only source-language subtitles (no LLM cost).
- `translate` — source subtitles + streaming translation into the target language.

## Requirements

- **Windows 10/11** (loopback uses WASAPI; mic mode works on macOS/Linux too).
- **Python 3.11+** (< 3.14).
- For `translate` mode: an API key from any OpenAI-compatible chat
  provider (OpenAI, DeepSeek, Groq, OpenRouter, Ollama, ...). For
  `asr_only` mode: no key needed at all.

## Install

Using [uv](https://docs.astral.sh/uv/) (recommended):

```powershell
cd C:\Projects\realtime-translator
uv sync
copy .env.example .env
notepad .env             # paste your chat-translation API key (optional)
```

Or with plain `pip + venv`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
copy .env.example .env
notepad .env
```

## Double-click .exe build (no Python needed at runtime)

If you just want a clickable Windows app to hand to a non-developer:

```powershell
uv sync --group dev               # installs PyInstaller + Pillow
uv run pyinstaller tools/breezemate.spec --noconfirm --clean
```

Output lives at `dist\BreezeMate\BreezeMate.exe` (~13 MB exe inside a
~225 MB folder of bundled DLLs / Qt plugins / vosk runtime). Double-click
the .exe, right-click → *Send to → Desktop (create shortcut)*, or zip
the whole `dist\BreezeMate\` folder to share. The app self-creates
`%APPDATA%\rt-translator\` on first run and downloads Vosk models on
demand from the settings dialog.

## Quick start (GUI)

```powershell
uv run breezemate gui
# or, after running the PyInstaller build above:
.\dist\BreezeMate\BreezeMate.exe
```

First launch:

1. Open **设置… → 语音识别**, pick a recognition language (English,
   中文, ...), and click **下载模型**. The small model is ~40 MB and
   downloads in seconds; first inference takes ~2 s to warm up.
2. (Optional, only if you want translation) **设置… → 翻译模型**: pick a
   provider preset (OpenAI / DeepSeek / Groq / Ollama / ...) and paste
   the API key. Each preset auto-fills its base URL and model
   suggestions.
3. Back on the main window, choose audio source + device, pick a mode
   (asr_only / translate), click **▶ 开始**.

The control panel offers:

- Audio source dropdown (Loopback / Mic) + device selector.
- Mode toggle (Speech subtitles only / Speech + translation).
- Start / Stop button.
- A floating, frameless, always-on-top **subtitle overlay**. Drag to
  move; right-click for opacity / font-size / click-through / hide.
- System-tray icon: closing the main window minimises to tray; the
  pipeline keeps running.

Settings and the subtitle window's position are persisted to
`%APPDATA%\rt-translator\config.yaml`. API keys live in
`%APPDATA%\rt-translator\secrets.json` (per-user, not encrypted -- if
that matters, use the `.env` / environment-variable path instead).

Vosk models live under `%APPDATA%\rt-translator\vosk-models\<name>\`.
If the in-app download fails (corporate proxy, etc.) you can fetch the
zip manually from <https://alphacephei.com/vosk/models> and unzip it
into that folder.

## Quick start (CLI)

List available audio devices:

```powershell
uv run breezemate devices
```

Run with default settings (will prompt for a device on first launch and save
your choice to `%APPDATA%\rt-translator\device.json`):

```powershell
uv run breezemate
```

Run with explicit overrides:

```powershell
# Speech subtitles only, system audio loopback
uv run breezemate run --mode asr_only --source loopback

# Source + translation, capture from a USB microphone (substring match)
uv run breezemate run --mode translate --source mic --device "USB"

# Override the Vosk model (must already be downloaded via the GUI)
uv run breezemate run --asr-model vosk-model-small-cn-0.22
```

To re-pick the saved device:

```powershell
uv run breezemate devices --select
```

## Supported translation providers

All translators speak the OpenAI Chat Completions protocol, so switching
is purely a base_url + key + model change. Built-in presets:

| Preset id | Free? | Notes |
|---|---|---|
| `openai` | paid | Highest translation quality (GPT-4o family). |
| `deepseek` | paid (cheap) | `deepseek-chat` is ~10x cheaper than gpt-4o-mini. |
| `groq` | free tier | Fastest token output. `llama-3.3-70b-versatile` is solid. |
| `openrouter` | mixed | Models ending in `:free` are free. |
| `together` | trial credits | Hosted open-source models. |
| `ollama` | free (local) | `ollama pull qwen2.5:7b` then point at `localhost:11434/v1`. |
| `lmstudio` | free (local) | Local desktop server. |
| `custom` | -- | Any OpenAI-compatible endpoint. |

## Supported recognition languages

The Vosk model catalog ships entries for:

| Code | Language | Default model |
|---|---|---|
| `en` | English | `vosk-model-small-en-us-0.15` |
| `zh` | 中文 (普通话) | `vosk-model-small-cn-0.22` |
| `ja` | 日本語 | `vosk-model-small-ja-0.22` |
| `ko` | 한국어 | `vosk-model-small-ko-0.22` |
| `ru` | Русский | `vosk-model-small-ru-0.22` |
| `fr` | Français | `vosk-model-small-fr-0.22` |
| `de` | Deutsch | `vosk-model-small-de-0.15` |
| `es` | Español | `vosk-model-small-es-0.42` |
| `it` | Italiano | `vosk-model-small-it-0.22` |
| `pt` | Português | `vosk-model-small-pt-0.3` |
| `pl` | Polski | `vosk-model-small-pl-0.22` |
| `uk` | Українська | `vosk-model-small-uk-v3-small` |
| `tr` | Türkçe | `vosk-model-small-tr-0.3` |
| `vi` | Tiếng Việt | `vosk-model-small-vn-0.4` |
| `hi` | हिन्दी | `vosk-model-small-hi-0.22` |

Each language ships with exactly one model -- Alphacephei's "small"
build. The larger "full" models were dropped after real-world testing:
they load 10x slower and we punt punctuation / grammar fixes to the
downstream LLM polishing step, which already handles whatever extra
accuracy a bigger acoustic model would have bought.

## Configuration

`config.example.yaml` documents every knob. Copy it to `config.yaml` and tweak,
then run with `--config config.yaml`. CLI flags override the file; the file
overrides built-in defaults.

Resolution priority for the audio source:

1. `--source` / `--device` CLI flags
2. `audio.source` / `audio.device_name` in config.yaml
3. Saved selection in `%APPDATA%\rt-translator\device.json`
4. Interactive picker (first run only)

## Cost estimate

ASR is free (offline Vosk). Translation cost is dominated by your
chosen chat provider; with `gpt-4o-mini`, a typical conversational hour
of `translate` mode costs roughly **$0.05–$0.10 / hour**. With Groq
(free tier) or a local Ollama model, translation is also free.

## Latency

| Step | Typical |
|---|---|
| First word appears in the live preview row | ~150 ms after speech starts |
| Sentence locked + sent to translator | ~1.0 s after speaker pauses (configurable, see `local_asr.finalize_after_silence_s`) |
| Translation fully rendered | ~0.8–1.5 s after sentence is locked |

Latency is dominated by:

- `local_asr.finalize_after_silence_s` (sentence boundary detector).
  Lower = snappier translations but may chop sentences at commas.
- Network RTT to your chosen translation endpoint.

## Troubleshooting

### Vosk model didn't download / "model file not found"

In the GUI: 设置 → 语音识别 → 下载模型. If your network blocks
alphacephei.com, click **打开模型目录** and drop a manually-downloaded
zip's contents in there (folder name must match the model id).

### No audio devices, or wrong device selected

```powershell
uv run breezemate devices --select
```

Then pick the device by number. Tip: in `loopback` mode you typically want
your *output* speakers (what Windows is playing through), not your input
mic.

### Loopback works, microphone doesn't

Mic input devices vary wildly in sample-rate support. The capture loop tries
16 kHz first; if your USB interface refuses, raise an issue (the fallback
to 48 kHz can be added in `audio_capture.py`).

### COM / threading errors on capture

The capture thread calls `pythoncom.CoInitialize()` on entry. If you still
see COM errors, make sure `pywin32` was installed (`uv pip list | findstr pywin32`).

### Subtitles freeze on a half-sentence

That's the silence-boundary detector waiting for a real pause. Lower
`local_asr.finalize_after_silence_s` (in the GUI it's "句末静默时长")
to e.g. 0.5 s for snappier sentence breaks.

## Project layout

```
src/rt_translator/
├── __main__.py                  # python -m rt_translator
├── cli.py                       # argparse + subcommands
├── config.py                    # pydantic models + YAML loader
├── events.py                    # TranscriptDelta/Final + TranslationDelta/Final
├── pipeline.py                  # asyncio orchestration
├── audio_capture.py             # soundcard loopback + mic, soxr resampling
├── device_picker.py             # device enumeration + interactive picker
├── paths.py                     # OS-specific appdata dir
├── secrets.py                   # JSON-backed API key store
├── console_sink.py              # rich.live incremental subtitle renderer (CLI)
├── providers/
│   ├── base.py                  # StreamingASRProvider + LLMTranslator Protocols
│   ├── presets.py               # OpenAI / DeepSeek / Groq / OpenRouter / Ollama / ...
│   ├── asr/
│   │   ├── local_vosk.py        # threaded Vosk wrapper (preview + finals)
│   │   ├── vosk_full.py         # StreamingASRProvider w/ silence-boundary detection
│   │   └── vosk_model.py        # multi-language model catalog + downloader
│   └── llm/
│       └── openai_compatible.py # streaming chat translation
└── gui/                         # PySide6 GUI
    ├── app.py                   # breezemate-gui entry point
    ├── main_window.py           # control panel + system tray
    ├── subtitle_window.py       # frameless floating subtitle overlay
    ├── settings_dialog.py       # provider profile + Vosk model picker
    ├── pipeline_controller.py   # QThread + asyncio bridge
    └── signal_sink.py           # pipeline events -> Qt signals adapter

assets/
├── breezemate.png               # 1024x1024 app icon (in-app QIcon)
└── breezemate.ico               # multi-size Windows .ico (exe icon)

tools/
├── breezemate_launcher.py       # PyInstaller entry script
└── breezemate.spec              # PyInstaller one-folder spec
```

The provider Protocols are the official extension points. Future
backends (Deepgram streaming, faster-whisper, etc.) drop in by adding
new files under `providers/` — no other code changes needed.

## License

MIT.
