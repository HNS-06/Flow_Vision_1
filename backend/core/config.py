"""
FlowVision Backend - Configuration
"""

import os
from pathlib import Path

# Base directory (Project Root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
DATABASE_DIR = DATA_DIR / 'database'

# ML directories
ML_DIR = BASE_DIR / 'ml_pipeline'
MODEL_DIR = ML_DIR / 'models'
EVALUATION_DIR = ML_DIR / 'evaluation'

# API Configuration
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 8000))
API_RELOAD = os.getenv('API_RELOAD', 'True').lower() == 'true'

# Database
DATABASE_URL = f"sqlite:///{DATABASE_DIR / 'flowvision.db'}"

# CORS
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
]

# Simulation settings
SIMULATION_INTERVAL = 1.0  # seconds between updates
