@echo off
setlocal

rem Build Windows .exe with PyInstaller
rem Requires: pip install pyinstaller

if not exist .venv (
  echo [info] Creating venv...
  py -3 -m venv .venv
)
call .venv\Scripts\activate

echo [info] Installing build deps...
pip install --upgrade pip
pip install -r requirements.txt
pip install "pyinstaller>=6.6" pyinstaller-hooks-contrib

echo [info] Building exe...
pyinstaller --noconfirm --onefile --console ^
  --name SystemAudioCopilot ^
  --add-data "env_example.txt;." ^
  --collect-binaries soundfile ^
  --collect-data soundfile ^
  --hidden-import sounddevice ^
  --hidden-import pyaudiowpatch ^
  main.py

echo.
echo [done] Built: dist\SystemAudioCopilot.exe
echo [hint] Put your .env next to the exe (dist\). See .env.example

endlocal

