@echo off
rem Double-click to start FileSense with this folder's .venv.
cd /d "%~dp0"
".venv\Scripts\python.exe" document_query_visuals.py
if errorlevel 1 pause
