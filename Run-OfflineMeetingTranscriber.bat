@echo off
setlocal
set "APP_DIR=%~dp0"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_DATASETS_OFFLINE=1"
cd /d "%APP_DIR%"
"%APP_DIR%OfflineMeetingTranscriber.exe"
endlocal
