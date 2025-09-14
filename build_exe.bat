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

rem --- Derive version and generate version.py ---
set VERSION_FILE=version.py
set DEFAULT_VERSION=1.2.3
set EFFECTIVE_VERSION=%DEFAULT_VERSION%
if not "%VERSION%"=="" (
  set EFFECTIVE_VERSION=%VERSION%
) else (
  for /f "usebackq delims=" %%i in (`git describe --tags --abbrev=0 2^>NUL`) do set EFFECTIVE_VERSION=%%i
)
echo [info] App version: %EFFECTIVE_VERSION%
echo __version__ = "%EFFECTIVE_VERSION%" > %VERSION_FILE%

echo [info] Building exe...
pyinstaller --noconfirm --onefile --console ^
  --name SystemAudioCopilot ^
  --add-data "env_example.txt;." ^
  --collect-binaries soundfile ^
  --collect-data soundfile ^
  --hidden-import sounddevice ^
  --hidden-import pyaudiowpatch ^
  --icon "assets\app.ico" ^
  main.py

echo.
echo [done] Built: dist\SystemAudioCopilot.exe
echo [hint] Put your .env next to the exe (dist\). See .env.example

endlocal

