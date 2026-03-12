@echo off
REM U-Net_med Training Launcher
REM Requirements: Python with CUDA 11.8 support, dependencies installed

echo ========================================
echo U-Net Medical Segmentation Trainer
echo ========================================
echo.

REM Check CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')" 2>nul
echo.

REM Activate venv if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [INFO] Virtual environment activated
)

REM Run training
echo [INFO] Starting training...
echo [INFO] Logs: logs\u_net_med\
echo [INFO] Checkpoints: checkpoints\u_net_med\
echo.
python src\train.py

pause
