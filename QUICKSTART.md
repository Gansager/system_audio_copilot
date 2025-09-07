# Quick Start - System Audio Copilot

## 🚀 Start in 3 steps

### 1. Install dependencies
```bash
# Create a virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Set up API key
```bash
# Copy example configuration
copy env_example.txt .env

# Edit the .env file and add your OpenAI API key
notepad .env
```

### 3. Run
```bash
# Simple run
python main.py

# Or use the batch file (Windows)
run.bat
```

## 🎯 What happens

1. **Start**: The app begins listening to system audio
2. **Live transcription**: Every 3 seconds it prints recognized text
3. **AI hints**: Press Enter to get a hint from AI
4. **Exit**: Ctrl+C to quit cleanly

## 🔧 Useful commands

```bash
# Increase transcription interval
python main.py --window-sec 5

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
