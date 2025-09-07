# Git repository setup for System Audio Copilot

## 🚀 Quick setup

### Option 1: Automatic setup
Run the batch file:
```cmd
setup_git.bat
```

### Option 2: Manual setup

#### 1. Initialize Git repository
```cmd
git init
```

#### 2. Add files
```cmd
git add .
```

#### 3. First commit
```cmd
git commit -m "Initial commit: System Audio Copilot - Python CLI tool for live system audio transcription with AI assistance"
```

#### 4. Set main branch
```cmd
git branch -M main
```

## 📋 Next steps

### 1. Create a repository on GitHub
1. Go to https://github.com
2. Click "New repository"
3. Name: `system_audio_copilot`
4. Description: `Python CLI tool for live system audio transcription with AI assistance`
5. Choose "Public"
6. Do NOT add README, .gitignore or license (they already exist)
7. Click "Create repository"

### 2. Connect local project to GitHub
```cmd
git remote add origin https://github.com/YOUR_USERNAME/system_audio_copilot.git
```
Replace `YOUR_USERNAME` with your GitHub username.

### 3. Push code to GitHub
```cmd
git push -u origin main
```

## ✅ Expected result

After completing all steps:
- ✅ Public repository `system_audio_copilot` on GitHub
- ✅ All project code pushed
- ✅ `.env` file NOT pushed (ignored)
- ✅ `.venv/` folder NOT pushed (ignored)
- ✅ Python cache ignored

## 📁 Repository structure

```
system_audio_copilot/
├── .gitignore              # Ignore .env, .venv, caches
├── README.md               # Detailed documentation
├── QUICKSTART.md           # Quick start
├── GIT_SETUP.md            # This guide
├── requirements.txt        # Python dependencies
├── env_example.txt         # Config example
├── run.bat                 # Windows run script
├── setup_git.bat           # Git setup script
├── main.py                 # Main CLI entry point
├── audio_capture.py        # System audio capture
├── stt.py                  # Transcription via Whisper
└── llm.py                  # AI hints via GPT
```

## 🔧 Useful Git commands

```cmd
# Check status
git status

# View commit history
git log --oneline

# Add changes
git add .
git commit -m "Describe changes"

# Push changes to GitHub
git push

# Pull changes from GitHub
git pull
```

## ⚠️ Important notes

- **Never commit** the `.env` file with API keys
- **Always check** `.gitignore` before committing
- **Use descriptive** commit messages
- **Commit** for each meaningful change
