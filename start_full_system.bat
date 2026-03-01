@echo off
echo ============================================
echo   CardioVision AI - Full System Launcher
echo ============================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo [ERROR] Virtual environment not found!
    echo Please create one first: python -m venv .venv
    pause
    exit /b 1
)

echo [1/3] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [2/3] Starting Flask backend...
cd backend
start "CardioVision Backend" cmd /k "python app.py"
cd ..

timeout /t 3 /nobreak >nul

echo [3/3] Opening frontend in browser...
start "" "frontend\index_enhanced.html"

echo.
echo ============================================
echo   System Started!
echo ============================================
echo.
echo Backend:  http://localhost:5000
echo Frontend: Opening in browser...
echo.
echo Press any key to show testing instructions
pause >nul

echo.
echo ========== TESTING INSTRUCTIONS ==========
echo.
echo 1. CLASSIFICATION MODE:
echo    - Select "Classification" radio button
echo    - Upload ECG from ecg_data/ folder
echo    - View diagnosis and confidence
echo.
echo 2. DIGITIZATION MODE:
echo    - Select "Digitization" radio button  
echo    - Upload any ECG image
echo    - View 12-lead signals (interactive plot)
echo    - Export as JSON or CSV
echo.
echo ==========================================
echo.
echo Backend window is running in background.
echo Close it manually when done testing.
echo.
pause
