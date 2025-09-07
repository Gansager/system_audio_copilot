# Quick Start - System Audio Copilot

## 🚀 Start in 3 steps

### Option A: Download prebuilt .exe
1. Download `SystemAudioCopilot.exe` from Releases
2. Copy `env_example.txt` to the same folder and rename to `.env`
3. Add your `OPENAI_API_KEY` to `.env`
4. Run:
```powershell
./SystemAudioCopilot.exe
```

### Option B: Build from source

#### 1. Install dependencies
```bash
# Create a virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

#### 2. Set up API key
```bash
# Copy example configuration
copy env_example.txt .env

# Edit the .env file and add your OpenAI API key
notepad .env
```

#### 3. Run
```bash
# Simple run
python main.py

# Or use the batch file (Windows)
run.bat
```

## 🎯 What happens

1. **Start**: The app begins listening to system audio
2. **Live transcription**: Every 2 seconds it prints recognized text (gated by VAD to skip silence)
3. **AI hints**: Press Enter to get a hint from AI
4. **Exit**: Ctrl+C to quit cleanly

## 🔧 Useful commands

```bash
# Increase transcription interval
python main.py --window-sec 5

# Adjust VAD sub-frames and gating sensitivity
python main.py --vad-frame-ms 50 --vad-min-voiced-ratio 0.2

# "Enter-only" mode (no live transcription)
python main.py --enter-only

# Help
python main.py --help
```

## ⚠️ Important notes

- **Output device**: Ensure audio is played through the default device
- **API key**: Get a key at https://platform.openai.com/api-keys
- **Permissions**: If needed, run as administrator

## 🆘 If something doesn't work

1. Verify you have an active audio output device
2. Ensure audio is playing in the system
3. Check the API key in the `.env` file
4. See error messages in the console (they start with `[error]`)
