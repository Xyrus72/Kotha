@echo off
title Voice Chat
rem Voice Chat - one click: starts the local LLM service (if needed), the app, and opens the page.
cd /d "%~dp0"

rem already running? just open the page
curl -s -m 3 -o nul http://127.0.0.1:8765/api/ready
if not errorlevel 1 (
  start "" http://127.0.0.1:8765
  exit /b
)

set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
curl -s -o nul http://127.0.0.1:11434/api/version
if errorlevel 1 (
  if exist "%OLLAMA%" (
    echo Starting the local LLM service...
    start "" /min "%OLLAMA%" serve
  ) else (
    echo Ollama is not installed yet - you will get text from your speech but no replies.
  )
)

echo Loading speech models. The page opens by itself when they are ready.
echo Keep this window open while you use Voice Chat; close it to stop.
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 90;$i++){try{$r=Invoke-RestMethod http://127.0.0.1:8765/api/ready -TimeoutSec 5;if($r.stt_ready -or $r.stt_error){break}}catch{};Start-Sleep 1};Start-Process http://127.0.0.1:8765"
".venv\Scripts\python.exe" server.py
echo.
echo Voice Chat stopped.
pause
