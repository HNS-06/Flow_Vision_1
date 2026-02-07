
import sys
import os
import asyncio
import json

# Add project root to path
sys.path.append(os.getcwd())

from backend.services.simulation_service import simulation_service

def verify():
    print("Verifying Simulation Service Output...")
    
    # Generate one state
    state = simulation_service.generate_system_state()
    
    # Check Ward 1
    ward = state['wards']['ward_1']
    pipes = ward['pipes']
    
    print(f"Ward 1 Pipes: {len(pipes)}")
    if len(pipes) > 0:
        p = pipes[0]
        print("Pipe 0 Data Keys:", p.keys())
        
        # Check for Critical Fields
        required = ['flow_rate', 'pressure']
        missing = [k for k in required if k not in p]
        
        if missing:
            print(f"❌ MISSING COMPATIBILITY FIELDS: {missing}")
            print("Frontend likely broken.")
        else:
            print("✅ Compatibility fields present.")
            
        # Check for New Fields
        new_fields = ['flow_in', 'flow_out', 'analysis']
        present = [k for k in new_fields if k in p]
        print(f"New Fields Present: {present}")
        
    # Check Analysis
    if 'analysis' in p:
        print("Analysis Data:", json.dumps(p['analysis'], indent=2))

if __name__ == "__main__":
    verify()
