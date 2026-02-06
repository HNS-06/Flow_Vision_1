<<<<<<< HEAD
# Flow_Vision_1
=======
# FlowVision Setup & Run Guide

## Prerequisites
- Python 3.9+
- Pip (Python Package Manager)

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Simply double-click `run_flowvision.bat` or run:

```bash
run_flowvision.bat
```

This script will automatically:
1. Generate synthetic water sensor data
2. Train the ML models (Leak Detection & Forecasting)
3. Start the FastAPI backend server
4. Serve the Dashboard at `http://localhost:8000`

## Manual Execution

If you prefer to run steps manually:

1. **Generate Data:**
   ```bash
   python scripts/generate_sample_data.py
   ```

2. **Process Data & Train Models:**
   ```bash
   python ml_pipeline/data_preprocessing.py
   python ml_pipeline/leak_detection.py
   python ml_pipeline/consumption_forecast.py
   ```

3. **Start Server:**
   ```bash
   python backend/app.py
   ```

## Project Structure
- `backend/`: FastAPI application and API routes
  - `backend/static/`: Zero-build Frontend (HTML/JS/CSS)
- `ml_pipeline/`: Machine Learning models and logic
- `data/`: Generated datasets (Raw & Processed)
- `scripts/`: Utility scripts

## Features
- **Live Monitoring**: Real-time water flow visualization via WebSockets
- **Leak Detection**: AI-powered anomaly detection using Isolation Forest
- **Forecasting**: Next-24h consumption prediction
- **Simulation**: Interactive "Leak Scenario" toggle for demo purposes
- **[View Datasets & Models Documentation](DATA_AND_MODELS.md)**
>>>>>>> 1f4306b (FlowVision 2)
