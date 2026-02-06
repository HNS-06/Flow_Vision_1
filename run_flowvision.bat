@echo off
echo ===================================================
echo FlowVision - Smart Water Leakage Management System
echo ===================================================
echo.

REM Detect Python Command
python --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
) else (
    py --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=py
    ) else (
        echo [ERROR] Python is not found.
        echo Please install Python from https://python.org/downloads
        echo.
        pause
        exit /b 1
    )
)

echo Using Python: %PYTHON_CMD%

echo [0/3] Checking dependencies...
%PYTHON_CMD% -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Warning: Could not install dependencies.
)

echo [1/3] Generating synthetic data...
%PYTHON_CMD% scripts/generate_sample_data.py
if %errorlevel% neq 0 (
    echo Error generating data.
    pause
    exit /b %errorlevel%
)

echo.
echo [2/3] Preprocessing data and training models...
%PYTHON_CMD% ml_pipeline/data_preprocessing.py
%PYTHON_CMD% ml_pipeline/leak_detection.py
%PYTHON_CMD% ml_pipeline/consumption_forecast.py

echo.
echo [3/3] Starting Backend Server...
echo The dashboard will be available at: http://localhost:8000
echo.
%PYTHON_CMD% -m backend.app

pause
