# CardioVision AI - Backend Startup Script for PowerShell
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  CardioVision AI - Starting Backend" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Set-Location backend

Write-Host "Activating virtual environment..." -ForegroundColor Yellow
if (Test-Path "..\.venv\Scripts\Activate.ps1") {
    & "..\.venv\Scripts\Activate.ps1"
} else {
    Write-Host "Warning: Virtual environment not found" -ForegroundColor Red
}

Write-Host ""
Write-Host "Starting Flask server..." -ForegroundColor Green
Write-Host "Backend will be available at: http://localhost:5000" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

python app.py
