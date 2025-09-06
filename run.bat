@echo off
echo System Audio Copilot - Quick Start
echo ==================================

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Check if .env exists
if not exist ".env" (
    echo.
    echo WARNING: .env file not found!
    echo Please copy env_example.txt to .env and add your OpenAI API key
    echo.
    pause
    exit /b 1
)

REM Run the application
echo.
echo Starting System Audio Copilot...
echo Press Ctrl+C to stop
echo.
python main.py

pause
