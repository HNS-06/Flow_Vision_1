"""
FlowVision Backend - API Routes for ML
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from backend.services.ml_service import ml_service
from backend.services.data_service import data_service
import pandas as pd

router = APIRouter(prefix="/api/ml", tags=["machine-learning"])

class LeakDetectionRequest(BaseModel):
    hours: int = 24

class ForecastRequest(BaseModel):
    steps: int = 24
    ward_id: int = 1

@router.post("/detect-leak")
async def detect_leak(request: LeakDetectionRequest):
    """Detect leaks in recent flow data"""
    try:
        # Get recent flow data
        flow_data = data_service.flow_data
        if flow_data is None:
            data_service.load_all_data()
            flow_data = data_service.flow_data
        
        # Get recent data
        recent_data = flow_data.tail(request.hours)
        
        # Detect leaks
        result = ml_service.detect_leaks(recent_data)
        
        if result is None:
            raise HTTPException(status_code=500, detail="Leak detection failed")
        
        # Get summary statistics
        avg_probability = result['leak_probability'].mean()
        max_probability = result['leak_probability'].max()
        num_leaks = result['is_leak_detected'].sum()
        
        return {
            "success": True,
            "summary": {
                "average_leak_probability": round(avg_probability, 2),
                "max_leak_probability": round(max_probability, 2),
                "leaks_detected": int(num_leaks),
                "total_records": len(result)
            },
            "recent_readings": result.tail(10)[
                ['timestamp', 'flow_rate', 'leak_probability', 'is_leak_detected']
            ].to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forecast")
async def forecast_consumption(request: ForecastRequest):
    """Forecast future consumption"""
    try:
        # Get flow data
        flow_data = data_service.flow_data
        if flow_data is None:
            data_service.load_all_data()
            flow_data = data_service.flow_data
        
        # Generate forecast
        summary = ml_service.forecast_consumption(flow_data, steps=request.steps, ward_id=request.ward_id)
        
        if summary is None:
            raise HTTPException(status_code=500, detail="Forecasting failed")
        
        return {
            "success": True,
            "forecast": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/insights")
async def get_insights():
    """Get AI-generated insights"""
    try:
        result = ml_service.get_ward_insights()
        
        if result is None:
            raise HTTPException(status_code=500, detail="Insights generation failed")
        
        return {
            "success": True,
            "insights": result['insights'],
            "comparison": result['comparison'],
            "clusters": result.get('clusters', {})
        }
    except Exception as e:
        print(f"Error in insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize")
async def optimize_distribution():
    """Run optimization for water distribution"""
    try:
        result = ml_service.optimize_water_distribution()
        if result is None:
             raise HTTPException(status_code=500, detail="Optimization failed")
        return {"success": True, "optimization": result}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@router.get("/control/{ward_id}")
async def get_control_action(ward_id: int):
    """Get RL agent action for a ward"""
    try:
        # Mock inputs for now - in real world would come from sensors
        import random
        level = random.randint(30, 90)
        demand = random.randint(100, 300)
        hour = 12
        
        action = ml_service.get_rl_control_action(ward_id, level, demand, hour)
        if action is None:
             raise HTTPException(status_code=500, detail="Control action failed")
        return {"success": True, "action": action, "state": {"level": level, "demand": demand}}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))
