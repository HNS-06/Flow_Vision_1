"""
FlowVision Backend - API Routes for Simulation
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from backend.services.simulation_service import simulation_service

router = APIRouter(prefix="/api/simulation", tags=["simulation"])

class SimulationControl(BaseModel):
    action: str  # "start" or "stop"

class ScenarioControl(BaseModel):
    scenario: str  # "normal" or "leak"

@router.post("/start")
async def start_simulation():
    """Start real-time simulation"""
    simulation_service.start()
    return {"success": True, "message": "Simulation started"}

@router.post("/stop")
async def stop_simulation():
    """Stop real-time simulation"""
    simulation_service.stop()
    return {"success": True, "message": "Simulation stopped"}

@router.post("/scenario")
async def set_scenario(control: ScenarioControl):
    """Set simulation scenario"""
    enable_leak = control.scenario == "leak"
    simulation_service.toggle_leak_scenario(enable_leak)
    return {
        "success": True,
        "message": f"Scenario set to: {control.scenario}",
        "leak_enabled": enable_leak
    }

@router.get("/status")
async def get_simulation_status():
    """Get simulation status"""
    return {
        "success": True,
        "is_running": simulation_service.is_running,
        "leak_scenario": simulation_service.leak_scenario,
        "current_data": simulation_service.current_data
    }

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data streaming"""
    await websocket.accept()
    print("[OK] WebSocket client connected")
    
    try:
        await simulation_service.stream_data(websocket)
    except WebSocketDisconnect:
        print("[OK] WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
