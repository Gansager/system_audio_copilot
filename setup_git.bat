@echo off
echo Setting up Git repository for System Audio Copilot
echo =================================================

echo.
echo 1. Initializing Git repository...
git init

echo.
echo 2. Adding all files to Git...
git add .

echo.
echo 3. Making initial commit...
git commit -m "Initial commit: System Audio Copilot - Python CLI tool for live system audio transcription with AI assistance"

echo.
echo 4. Setting up main branch...
git branch -M main

echo.
echo =================================================
echo Git repository initialized successfully!
echo.
echo Next steps:
echo 1. Create a new public repository on GitHub named "system_audio_copilot"
echo 2. Copy the repository URL
echo 3. Run: git remote add origin YOUR_REPO_URL
echo 4. Run: git push -u origin main
echo.
echo Example:
echo git remote add origin https://github.com/YOUR_USERNAME/system_audio_copilot.git
echo git push -u origin main
echo.
pause
