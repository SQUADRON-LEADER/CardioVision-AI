@echo off
echo ============================================
echo   CardioVision AI - Starting Backend
echo ============================================
echo.

cd backend

echo Activating virtual environment...
if exist "..\\.venv\\Scripts\\activate.bat" (
    call "..\\.venv\\Scripts\\activate.bat"
) else (
    echo Warning: Virtual environment not found
)

echo.
echo Starting Flask server...
echo Backend will be available at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py

pause
