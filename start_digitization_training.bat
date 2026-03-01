@echo off
REM Quick Start Script for ECG Digitization Training
echo ====================================================================
echo ECG DIGITIZATION TRAINING - Quick Start
echo ====================================================================
echo.

REM Activate virtual environment
echo [1/4] Activating virtual environment...
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    echo     Virtual environment activated
) else (
    echo     Warning: .venv not found. Please create virtual environment first:
    echo     python -m venv .venv
    goto :error
)

REM Check if required packages are installed
echo.
echo [2/4] Checking dependencies...
python -c "import torch, numpy, pandas, matplotlib, wfdb" 2>nul
if %errorlevel% neq 0 (
    echo     Installing required packages...
    pip install torch torchvision numpy pandas matplotlib scipy wfdb pillow opencv-python scikit-learn tqdm --quiet
    echo     Dependencies installed
) else (
    echo     Dependencies OK
)

REM Run system test
echo.
echo [3/4] Running system test...
python test_digitization_system.py
if %errorlevel% neq 0 (
    echo     System test failed. Please check the errors above.
    goto :error
)

REM Start training
echo.
echo [4/4] Starting training...
echo.
echo ====================================================================
echo TRAINING OPTIONS:
echo ====================================================================
echo.
echo Choose training mode:
echo   [1] Quick test (5 epochs, small dataset)
echo   [2] Full training (50 epochs, full dataset)
echo   [3] Custom training (specify parameters)
echo.
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Starting QUICK TEST training...
    python train_digitization.py --num_epochs 5 --batch_size 8
) else if "%choice%"=="2" (
    echo.
    echo Starting FULL training...
    python train_digitization.py --num_epochs 50 --batch_size 16 --model_type advanced
) else if "%choice%"=="3" (
    echo.
    echo Enter custom parameters (or press Enter for defaults):
    set /p epochs="Epochs (default: 30): "
    set /p batch="Batch size (default: 16): "
    set /p lr="Learning rate (default: 0.001): "
    
    if "%epochs%"=="" set epochs=30
    if "%batch%"=="" set batch=16
    if "%lr%"=="" set lr=0.001
    
    echo.
    echo Starting CUSTOM training...
    python train_digitization.py --num_epochs %epochs% --batch_size %batch% --lr %lr% --model_type advanced
) else (
    echo Invalid choice. Exiting.
    goto :error
)

echo.
echo ====================================================================
echo Training complete!
echo.
echo Model saved to: digitization_checkpoints\best_digitization_model.pth
echo Training logs: digitization_logs\
echo ====================================================================
goto :end

:error
echo.
echo Training failed or cancelled.
pause
exit /b 1

:end
echo.
pause
