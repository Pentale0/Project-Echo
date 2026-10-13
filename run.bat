@echo off
cd /d "%~dp0"
set "VENV=%LOCALAPPDATA%\Project Echo\.venv\Scripts\python.exe"
if exist "%VENV%" (
  "%VENV%" app.py
) else (
  if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
  ) else (
    py -3 app.py
  )
)