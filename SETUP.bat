@echo off
title ULTRON Sentinel AI — Setup
color 04
echo.
echo  ==========================================
echo    ULTRON SENTINEL AI — First Time Setup
echo  ==========================================
echo.
echo  [1/3] Installing Python libraries...
cd /d "%~dp0backend"
pip install -r requirements.txt
echo.
echo  [2/3] Downloading YOLO11s model...
python -c "from ultralytics import YOLO; YOLO('yolo11s.pt'); print('YOLO11s ready!')"
echo.
echo  [3/3] Done!
echo.
echo  ==========================================
echo    Setup complete! Run START.bat to launch.
echo    Then open: http://localhost:8000
echo  ==========================================
pause
