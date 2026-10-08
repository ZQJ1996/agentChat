@echo off
cd /d "%~dp0"
start "agentChat-API" cmd /k "%~dp0start-api.bat"
timeout /t 2 /nobreak >nul
start "agentChat-UI" cmd /k "%~dp0start-frontend.bat"
echo API: http://127.0.0.1:8000
echo UI:  http://127.0.0.1:5173
