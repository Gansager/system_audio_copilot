# System Audio Copilot

Python CLI tool for Windows 11 that listens to system audio (WASAPI loopback) and provides live transcription with the ability to get hints from an AI assistant.

## Features

- 🎧 **Capture system audio** via WASAPI loopback (not the microphone)
- 📝 **Live transcription** of system audio every 3 seconds
- 🤖 **AI hints** from OpenAI GPT on Enter key press
- ⚡ **Incremental processing** - accumulate text between requests
- 🔧 **Configurable parameters** via CLI arguments and environment variables

## Requirements

- **OS**: Windows 11
- **Python**: 3.8+
- **OpenAI API key** for Whisper and Chat Completions

## Installation

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

- `--window-sec FLOAT` - Transcription interval in seconds (default: 3.0)
- `--samplerate INT` - Sample rate (default: 16000)
- `--enter-only` - Do not print live transcription, only accumulate for Enter

### Examples

```bash
# Standard mode with live transcription
python main.py

# Increased transcription interval
python main.py --window-sec 5

# "Enter-only" mode (no live transcription)
python main.py --enter-only

# Custom sample rate
python main.py --samplerate 22050
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
2. **Incremental processing**: Every N seconds (default 3) it sends the accumulated audio to OpenAI Whisper
3. **Live output**: Recognized text is printed to the console immediately
4. **Accumulation**: The same text is added to a buffer for future AI requests

### AI hints

1. **Accumulation**: All recognized text since last Enter is saved
2. **Request**: On Enter, all accumulated text is sent to GPT
3. **Response**: AI returns a brief hint (1-2 sentences)
4. **Reset**: The buffer is cleared and the process repeats

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
