from fastapi import APIRouter, HTTPException
from typing import List
import random

router = APIRouter(prefix="/api/data", tags=["data"])

@router.get("/wards")
def get_wards_data():
    """Get aggregated consumption data for all wards"""
    # In a real app, this would query the database.
    # For the demo, we return mock data matching the 4-Ward topology.
    wards = []
    for i in range(1, 5):
        wards.append({
            "ward_id": i,
            "ward_name": f"Ward {i}",
            "avg_daily_consumption_m3": round(random.uniform(120, 180), 1),
            "active_pipes": 4,
            "status": "normal"
        })
    
    return {
        "success": True, 
        "data": wards
    }
