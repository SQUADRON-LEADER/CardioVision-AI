@echo off
echo ============================================
echo   CardioVision AI - Starting Frontend
echo ============================================
echo.

cd frontend

echo Starting HTTP server on port 5000...
echo Frontend will be available at: http://localhost:5000
echo.
echo Open your browser and navigate to:
echo    http://localhost:5000
echo.
echo Make sure the backend is also running on port 5001!
echo.
echo Press Ctrl+C to stop the server
echo.

python -m http.server 5000

pause
