"""
FlowVision Backend - Main FastAPI Application
"""

import sys
import os

# Add project root to sys.path to resolve backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.core.config import CORS_ORIGINS
from backend.core.database import init_db
from backend.api import data_routes, ml_routes, simulation_routes
from backend.services.data_service import data_service

# Initialize FastAPI app
app = FastAPI(
    title="FlowVision API",
    description="Smart Water Leakage Management System - AI/ML Backend",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers (they already have prefixes defined)
app.include_router(data_routes.router, tags=["data"])
app.include_router(ml_routes.router, tags=["ml"])
app.include_router(simulation_routes.router, tags=["simulation"])

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def read_index():
    """Serve the frontend index page"""
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("="*60)
    print("FlowVision Backend - Starting Up")
    print("="*60)
    
    # Initialize database
    init_db()
    
    # Load data
    print("Loading datasets...")
    data_service.load_all_data()
    
    print("="*60)
    print("[OK] FlowVision Backend Ready")
    print("="*60)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "data_loaded": data_service.flow_data is not None
    }

if __name__ == "__main__":
    import uvicorn
    from backend.core.config import API_HOST, API_PORT, API_RELOAD
    
    uvicorn.run(
        "backend.app:app", 
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD
    )
