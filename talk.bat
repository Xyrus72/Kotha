@echo off
rem Hands-free voice chat in this window. Extra options are passed on, e.g.  talk.bat --lang bn
cd /d "%~dp0"
set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
curl -s -o nul http://127.0.0.1:11434/api/version
if errorlevel 1 if exist "%OLLAMA%" (
  start "" /min "%OLLAMA%" serve
  timeout /t 3 /nobreak >nul
)
".venv\Scripts\python.exe" talk.py %*
pause
