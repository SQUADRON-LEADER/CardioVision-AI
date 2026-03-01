# Quick Start Script for ECG Digitization Training (PowerShell)
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "ECG DIGITIZATION TRAINING - Quick Start" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
Write-Host "[1/4] Activating virtual environment..." -ForegroundColor Yellow
if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .venv\Scripts\Activate.ps1
    Write-Host "    ✓ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "    ✗ .venv not found. Please create virtual environment first:" -ForegroundColor Red
    Write-Host "    python -m venv .venv" -ForegroundColor White
    exit 1
}

# Check dependencies
Write-Host ""
Write-Host "[2/4] Checking dependencies..." -ForegroundColor Yellow
$check = python -c "import torch, numpy, pandas, matplotlib, wfdb" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "    Installing required packages..." -ForegroundColor Yellow
    pip install torch torchvision numpy pandas matplotlib scipy wfdb pillow opencv-python scikit-learn tqdm --quiet
    Write-Host "    ✓ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "    ✓ Dependencies OK" -ForegroundColor Green
}

# Run system test
Write-Host ""
Write-Host "[3/4] Running system test..." -ForegroundColor Yellow
python test_digitization_system.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "    ✗ System test failed. Please check the errors above." -ForegroundColor Red
    exit 1
}

# Start training
Write-Host ""
Write-Host "[4/4] Starting training..." -ForegroundColor Yellow
Write-Host ""
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "TRAINING OPTIONS:" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Choose training mode:"
Write-Host "  [1] Quick test (5 epochs, small dataset)" -ForegroundColor White
Write-Host "  [2] Full training (50 epochs, full dataset)" -ForegroundColor White
Write-Host "  [3] Custom training (specify parameters)" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter choice (1-3)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "Starting QUICK TEST training..." -ForegroundColor Green
        python train_digitization.py --num_epochs 5 --batch_size 8
    }
    "2" {
        Write-Host ""
        Write-Host "Starting FULL training..." -ForegroundColor Green
        python train_digitization.py --num_epochs 50 --batch_size 16 --model_type advanced
    }
    "3" {
        Write-Host ""
        Write-Host "Enter custom parameters (or press Enter for defaults):"
        $epochs = Read-Host "Epochs (default: 30)"
        $batch = Read-Host "Batch size (default: 16)"
        $lr = Read-Host "Learning rate (default: 0.001)"
        
        if ([string]::IsNullOrWhiteSpace($epochs)) { $epochs = "30" }
        if ([string]::IsNullOrWhiteSpace($batch)) { $batch = "16" }
        if ([string]::IsNullOrWhiteSpace($lr)) { $lr = "0.001" }
        
        Write-Host ""
        Write-Host "Starting CUSTOM training..." -ForegroundColor Green
        python train_digitization.py --num_epochs $epochs --batch_size $batch --lr $lr --model_type advanced
    }
    default {
        Write-Host "Invalid choice. Exiting." -ForegroundColor Red
        exit 1
    }
}

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "====================================================================" -ForegroundColor Green
    Write-Host "Training complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Model saved to: digitization_checkpoints\best_digitization_model.pth" -ForegroundColor White
    Write-Host "Training logs: digitization_logs\" -ForegroundColor White
    Write-Host "====================================================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Training failed or cancelled." -ForegroundColor Red
}

Write-Host ""
Read-Host "Press Enter to exit"
