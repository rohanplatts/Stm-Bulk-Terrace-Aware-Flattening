@echo off
cd /d "%~dp0"
python stm_flatten_gui.py
if errorlevel 1 pause
