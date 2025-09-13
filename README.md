# System Audio Copilot

Python CLI tool for Windows 11 that listens to system audio (WASAPI loopback) and provides live transcription with the ability to get hints from an AI assistant.

## Features

- 🎧 **Capture system audio** via WASAPI loopback (not the microphone)
- 🎙️ **Capture microphone** in parallel with system audio
- 📝 **Live transcription** of system audio every 2 seconds
- 🤖 **AI hints** from OpenAI GPT on Enter key press
- ⚡ **Incremental processing** - accumulate text between requests
- 🔧 **Configurable parameters** via CLI arguments and environment variables
 - 💾 **Save session on exit**: prompt/auto-save last N seconds audio + full transcript

## Requirements

- **OS**: Windows 11
- **Python**: 3.8+
- **OpenAI API key** for Whisper and Chat Completions

## Installation

### Download prebuilt Windows binary

- Go to the project [Releases page](https://github.com/Gansager/system_audio_copilot/releases) and download `SystemAudioCopilot.exe`.
- Place a `.env` next to the `.exe` (copy from [env_example.txt](https://github.com/Gansager/system_audio_copilot/blob/main/env_example.txt)), add `OPENAI_API_KEY`.
- Run from CMD/PowerShell:

```powershell
./SystemAudioCopilot.exe --loopback
```

Or continue below to build from source.

### 1. Clone and set up environment

```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-REPLACE_WITH_YOUR_KEY
MODEL_TRANSCRIBE=whisper-1
MODEL_HINTS=gpt-4o-mini
TEMPERATURE=0.2
```

**Important**: Replace `sk-REPLACE_WITH_YOUR_KEY` with your real OpenAI API key.

## Usage

### Basic run

```bash
python main.py
```

### Command-line options

```bash
python main.py --help
```

- `--window-sec FLOAT` - Transcription interval in seconds (default: 2.0)
- `--samplerate INT` - Sample rate (default: 16000)
- `--enter-only` - Do not print live transcription, only accumulate for Enter
- `--vad-frame-ms INT` - VAD sub-frame size in ms (default: 50)
- `--vad-min-voiced-ratio FLOAT` - Minimum fraction of voiced sub-frames to consider the window speech (default: 0.2)
 - `--min-rms-gate-db FLOAT` - RMS gate in dBFS; below this skip STT (default: -45)
 - `--postfilter-min-chars INT` - Minimum content chars to keep a line (default: 3)
 - `--postfilter-min-words INT` - Minimum words required to keep a line (default: 1)
 - `--postfilter-dedup-seconds FLOAT` - Drop identical lines repeated within seconds (default: 5)
 - `--postfilter-drop-punct-only {0,1}` - Drop punctuation/emoji-only lines (default: 1)
 - `--use-webrtc-vad` - Use WebRTC VAD if available as additional gate
 - `--cross-suppress` - Suppress quieter source when other source dominates (default: on)
 - `--cross-suppress-rms-margin-db FLOAT` - Margin to suppress (default: 6)
 - `--cross-suppress-stale-sec FLOAT` - Other source recency to consider (default: 2.0)
 - `--snr-gate-db FLOAT` - Minimum estimated SNR in-window to pass (default: 0)
 - `--postfilter-stopwords CSV` - Single-word stopwords to drop (default includes common fillers)
 - `--postfilter-allowed-scripts CSV` - Allowed scripts (e.g., latin,cyrillic). Empty = all
 - `--postfilter-similarity-threshold FLOAT` - Approx dedup similarity ratio (default: 0.85)
 - `--postfilter-banned-phrases CSV` - Blocklist of phrases to drop (normalized substring)
 - `--no-loopback` - Disable system audio capture
 - `--capture-mic` (default on) / `--no-mic` - Enable/disable microphone capture
 - `--mic-index INT` / `--mic-device STR` - Select microphone by index or substring
 - `--mic-samplerate INT` - Mic sample rate (defaults to `--samplerate`)
 - `--mic-channels {1,2}` - Mic channels (default 1)
 - `--save-on-exit {ask,yes,no}` - Save session on exit (default: `ask`)
 - `--save-dir PATH` - Directory for saved sessions (default: `./sessions`)
 - `--save-audio-seconds INT` - Recent audio seconds to save (default: `30`; set to `0` to save the full session)
 - `--buffer-session-seconds INT` - Max seconds to retain in RAM for the session ring buffer; `0` = unlimited. If not set, defaults to `--save-audio-seconds`.
 - `--save-audio-mode {separate,mix,both}` - Save separate files, mixed, or both (default: `separate`)

### Examples

```bash
# Standard mode with live transcription
python main.py

# Increased transcription interval
python main.py --window-sec 5

# Tune VAD sub-frames and gating
python main.py --vad-frame-ms 30 --vad-min-voiced-ratio 0.15

# "Enter-only" mode (no live transcription)
python main.py --enter-only

# Custom sample rate
python main.py --samplerate 22050

# Auto-save session on exit to custom directory, keep 60s of audio
python main.py --save-on-exit yes --save-dir sessions --save-audio-seconds 60

# Save the entire session (audio + transcript) on exit
python main.py --save-on-exit yes --save-audio-seconds 0 --buffer-session-seconds 0

# Save mixed audio too
python main.py --save-on-exit yes --save-audio-mode both

# Reduce hallucinations on silence
python main.py --min-rms-gate-db -40 --postfilter-dedup-seconds 10 --postfilter-drop-punct-only 1

# Capture only mic
python main.py --no-loopback

# Capture only system audio
python main.py --no-mic

# Select microphone by name substring
python main.py --mic-device "USB Microphone"
```

## Build Windows .exe

1) Run the build script:
```
build_exe.bat
```
After the build you will get `dist\SystemAudioCopilot.exe`.

2) Place a `.env` file next to the `.exe` (you can copy from `env_example.txt`) and set your `OPENAI_API_KEY`.

3) Example run:
```
dist\SystemAudioCopilot.exe --loopback --output-device "Headphones"
```

Notes:
- `.env` is taken from the folder next to the `.exe`.
- To choose a device by index use `--input-index N`.
- For proper Unicode in CMD: `chcp 65001`.

## How it works

### WASAPI Loopback

The app uses **WASAPI loopback** to capture system audio. This means it listens to the sound that is played through your speakers/headphones, not the microphone.

**Audio sources that will be captured:**
- Browser video (YouTube, Netflix, etc.)
- Calls in Zoom, Teams, Discord
- Music from Spotify, Apple Music
- Game and app sounds
- Any other system audio

### Live transcription

1. **Background capture**: The app continuously records system audio
2. **Incremental processing**: Every N seconds (default 2.0) it sends the accumulated audio to OpenAI Whisper, gated by a lightweight VAD
3. **Live output**: Recognized text is printed to the console immediately
4. **Accumulation**: The same text is added to a buffer for future AI requests

### AI hints

1. **Accumulation**: All recognized text since last Enter is saved
2. **Request**: On Enter, all accumulated text is sent to GPT
3. **Response**: AI returns a brief hint (1-2 sentences)
4. **Reset**: The buffer is cleared and the process repeats

### Save session on exit

When the app exits (Ctrl+C/EOF), it can save the session:

- Prompt: `Сохранить сессию (аудио + текст)? [Y/n]` (default on Enter is Yes)
- Audio: last N seconds (default 30) from the session ring buffer. If `--save-audio-seconds` is `0` or negative, the entire session audio is saved.
- Text: full transcript for the session and assistant Q/A pairs
- Output directory: `sessions/YYYY-MM-DD_HHMMSS/` with `session.wav` and `transcript.txt`

You can control behavior via CLI or environment variables:

- `--save-on-exit {ask,yes,no}` | `SAVE_ON_EXIT`
- `--save-dir PATH` | `SAVE_DIR`
- `--save-audio-seconds INT` | `SAVE_AUDIO_SECONDS`
 - `--min-rms-gate-db FLOAT` | `MIN_RMS_GATE_DB`
 - `--postfilter-min-chars INT` | `POSTFILTER_MIN_CHARS`
 - `--postfilter-min-words INT` | `POSTFILTER_MIN_WORDS`
 - `--postfilter-dedup-seconds FLOAT` | `POSTFILTER_DEDUP_SECONDS`
 - `--postfilter-drop-punct-only {0,1}` | `POSTFILTER_DROP_PUNCT_ONLY`
 - `--use-webrtc-vad` (no env) additional gate if library present
 - `--cross-suppress` | `CROSS_SUPPRESS`
 - `--cross-suppress-rms-margin-db` | `CROSS_SUPPRESS_RMS_MARGIN_DB`
 - `--cross-suppress-stale-sec` | `CROSS_SUPPRESS_STALE_SEC`
 - `--snr-gate-db` | `SNR_GATE_DB`
 - `--postfilter-stopwords` | `POSTFILTER_STOPWORDS`
 - `--postfilter-allowed-scripts` | `POSTFILTER_ALLOWED_SCRIPTS`
 - `--postfilter-similarity-threshold` | `POSTFILTER_SIMILARITY_THRESHOLD`
 - `--postfilter-banned-phrases` | `POSTFILTER_BANNED_PHRASES`
- `--buffer-session-seconds INT` | `BUFFER_SESSION_SECONDS`

## Troubleshooting

### Problem: "Failed to initialize output device"

**Solution**: Make sure you have an active audio output device (speakers/headphones) set as default in Windows.

### Problem: No audio is captured

**Possible causes:**
1. **Wrong output device**: Ensure audio is played through the default device
2. **No audio**: Verify that there is audio playing in the system
3. **Permissions**: Some apps may block audio capture

**Solutions**: 
- Route app audio output to the default device
- Increase the system volume
- Restart the app with administrator privileges

### Problem: API errors

**Solutions**: 
- Check the API key in the `.env` file
- Ensure you have access to the OpenAI API
- Check your OpenAI account balance

### Problem: Low transcription quality

**Solutions**:
- Increase system volume
- Reduce background noise
- Try another Whisper model (if available)

## Project structure

```
system_audio_copilot/
├── main.py              # Main CLI interface
├── audio_capture.py     # System audio capture via WASAPI
├── stt.py               # Transcription via OpenAI Whisper
├── llm.py               # AI hints via OpenAI Chat Completions
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create manually)
└── README.md            # Documentation
```

## Security

- **API keys**: Never commit the `.env` file to the repository
- **Data**: Audio data is sent to OpenAI for processing
- **Privacy**: Ensure you agree with OpenAI's privacy policy

## License

This project is intended for educational and research purposes. Use at your own risk.

## Support

If you face issues:
1. Check the "Troubleshooting" section
2. Ensure all dependencies are installed correctly
3. Check error logs in the console (printed to stderr with the `[error]` prefix)
