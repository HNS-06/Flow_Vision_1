import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import random
import json

class SmartPipeSystem:
    """
    Manages the state of Pipes and Tanks for a single Ward.
    Topology: 1 Tank -> 2 Pairs of Pipes (Primary+Backup)
    """
    def __init__(self, ward_id):
        self.ward_id = ward_id
        # Two pairs of pipes: (1,2) and (3,4). 1 & 3 are Primaries.
        self.pairs = [
            {'primary': 1, 'backup': 2, 'active': 1, 'status': 'normal'},
            {'primary': 3, 'backup': 4, 'active': 3, 'status': 'normal'}
        ]
        self.alerts = []

    def update(self, base_flow, is_leak_scenario):
        """Update state for this step"""
        ward_data = {
            'ward_id': self.ward_id,
            'pipes': [],
            'alerts': []
        }
        
        for pair_idx, pair in enumerate(self.pairs):
            active_pipe = pair['active']
            
            # Simulate Flow
            flow = base_flow + np.random.normal(0, 5)
            pressure = 45 - (flow - 100) / 10 + np.random.normal(0, 2)
            
            # Leak Checking Logic
            # In 'leak scenario', we force a leak on the *current active* pipe of Pair 0
            is_leaking = False
            if is_leak_scenario and pair_idx == 0: 
                # Leak only affects the active pipe if it's the Primary
                if active_pipe == pair['primary']:
                    flow *= 1.8 # Spike flow
                    pressure *= 0.6 # Drop pressure
                    is_leaking = True
            
            # Smart Switch Logic
            if is_leaking and pair['status'] == 'normal':
                # Trigger Switch!
                pair['status'] = 'switched'
                pair['active'] = pair['backup'] # Switch to Backup
                
                alert_msg = f"Leakage detected on pipe {active_pipe} in Tank-1 (Ward-{self.ward_id}) switching to alternate pipe {pair['backup']}."
                ward_data['alerts'].append(alert_msg)
                print(f"[SMART-PIPE] {alert_msg}")
            
            # Reset if scenario turned off
            if not is_leak_scenario and pair['status'] == 'switched':
                pair['status'] = 'normal'
                pair['active'] = pair['primary'] # Reset to Primary

            # Add Pipe Data
            ward_data['pipes'].append({
                'pair_id': pair_idx + 1,
                'active_pipe': pair['active'],
                'primary_pipe': pair['primary'],
                'backup_pipe': pair['backup'],
                'flow_rate': round(max(0, flow), 2),
                'pressure': round(max(0, pressure), 2),
                'is_leaking': is_leaking,
                'status': 'active' if not is_leaking else 'critical'
            })
        
        # Calculate Ward aggregate metrics for Anomaly Score
        any_leak = any(p['is_leaking'] for p in ward_data['pipes'])
        
        if any_leak:
            ward_data['leak_probability'] = random.uniform(85, 99)
            # Re-broadcast critical alert occasionally
            if random.random() < 0.3:
                ward_data['alerts'].append(f"CRITICAL: Active leak in Ward-{self.ward_id} - Maintenance Required")
        else:
            # More variation in normal score (variable between 2-18%)
            base_noise = random.normalvariate(8, 3) 
            ward_data['leak_probability'] = max(2, min(25, base_noise))
            
            # Generate minor warnings if score is elevated
            if ward_data['leak_probability'] > 12 and random.random() < 0.15:
                 ward_data['alerts'].append(f"Warning: Minor pressure instability detected in Ward-{self.ward_id}")
            elif random.random() < 0.05:
                 ward_data['alerts'].append(f"Info: Routine check - Ward-{self.ward_id} nominal")

        return ward_data

class SimulationService:
    """Orchestrates simulation for 4 Wards"""
    
    def __init__(self):
        self.is_running = False
        self.leak_scenario = False
        self.subscribers = []
        
        # Initialize 4 Wards
        self.wards = {
            f"ward_{i}": SmartPipeSystem(i) for i in range(1, 5)
        }
        
        # Load user data for base patterns
        try:
            self.user_data = pd.read_csv('data/raw/user_sensor_data.csv')
            self.data_index = 0
            print(f"[OK] Loaded {len(self.user_data)} base data points")
        except:
            self.user_data = None

    def start(self):
        self.is_running = True
        print("[OK] Simulation started")
    
    def stop(self):
        self.is_running = False
        print("[OK] Simulation stopped")
    
    def toggle_leak_scenario(self, enable: bool):
        self.leak_scenario = enable
        print(f"[OK] Leak scenario: {enable}")
    
    def generate_system_state(self):
        """Generate state for all 4 wards"""
        now = datetime.now()
        
        # Get base flow from user data or random
        base_flow = 120
        if self.user_data is not None:
             row = self.user_data.iloc[self.data_index]
             base_flow = float(row['outlet_flow_lpm'])
             self.data_index = (self.data_index + 1) % len(self.user_data)
        
        system_update = {
            'timestamp': now.isoformat(),
            'wards': {}
        }
        
        for ward_key, ward_system in self.wards.items():
            # Add some variance per ward
            ward_variance = random.uniform(0.9, 1.1)
            ward_update = ward_system.update(base_flow * ward_variance, self.leak_scenario)
            system_update['wards'][ward_key] = ward_update
            
        return system_update
    
    async def stream_data(self, websocket):
        self.subscribers.append(websocket)
        try:
            while self.is_running:
                data = self.generate_system_state()
                await websocket.send_json(data)
                await asyncio.sleep(1)
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            if websocket in self.subscribers:
                self.subscribers.remove(websocket)

simulation_service = SimulationService()
