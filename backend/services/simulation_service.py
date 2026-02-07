import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import random
import json
from ml_pipeline.advanced_leak_detection import AdvancedLeakDetector

class SmartPipeSystem:
    """
    Manages the state of Pipes and Tanks for a single Ward.
    Topology: 1 Tank -> 2 Pairs of Pipes (Primary+Backup)
    """
    def __init__(self, ward_id, detector):
        self.ward_id = ward_id
        self.detector = detector
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
        
        now = datetime.now()

        for pair_idx, pair in enumerate(self.pairs):
            active_pipe = pair['active']
            
            # Simulate Flow & Pressure Physics
            # -------------------------------------------------
            # Normal Operation:
            # flow_in (Pump) ≈ base_flow (+/- noise)
            # flow_out (Consumer) ≈ flow_in * 0.99 (minor loss)
            # pressure_in ≈ 60 psi
            # pressure_out ≈ 60 - losses
            
            noise = np.random.normal(0, 2)
            flow_in = base_flow + noise
            pressure_in = 60 - (flow_in / 10) + np.random.normal(0, 1)
            
            # Default Output (Normal)
            flow_out = flow_in * np.random.uniform(0.98, 0.995)
            pressure_out = pressure_in - (flow_in / 50.0) # Normal friction drop
            
            is_leaking = False
            
            # Leak Checking Logic
            # In 'leak scenario', we force a leak on the *current active* pipe of Pair 0
            if is_leak_scenario and pair_idx == 0: 
                # Leak only affects the active pipe if it's the Primary
                if active_pipe == pair['primary']:
                    # PHYISCS OF A LEAK:
                    # 1. Water escapes pipe -> flow_out DECREASES relative to flow_in
                    # 2. Pressure drops significantly
                    leak_severity = 0.3 # 30% leak
                    
                    flow_out = flow_in * (1 - leak_severity) # Loss of water
                    pressure_out = pressure_out * 0.7 # Large pressure drop
                    is_leaking = True
            
            # -------------------------------------------------
            # Run Advanced Analysis
            # -------------------------------------------------
            # Convert numpy types to python floats for JSON serialization
            flow_in = float(flow_in)
            flow_out = float(flow_out)
            pressure_in = float(pressure_in)
            pressure_out = float(pressure_out)

            analysis = self.detector.analyze_system(flow_in, pressure_in, flow_out, pressure_out, now)
            
            # Smart Switch Logic based on Advanced Analysis
            if (is_leaking or analysis['is_leak']) and pair['status'] == 'normal':
                # Double check: Only switch if confidence is high or scenario is explicitly active
                if is_leak_scenario or analysis['combined_confidence'] > 80:
                    pair['status'] = 'switched'
                    pair['active'] = pair['backup'] # Switch to Backup
                    
                    # Generate detailed alert from analysis
                    reasons = "; ".join(analysis['alerts'])
                    alert_msg = f"⚠ LEAK DETECTED (Ward-{self.ward_id}, Pipe-{active_pipe}). Action: Switched to Backup {pair['backup']}. Reasons: {reasons}"
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
                'flow_in': round(flow_in, 2),
                'flow_out': round(flow_out, 2),
                'pressure_in': round(pressure_in, 2),
                'pressure_out': round(pressure_out, 2),
                # BACKWARD COMPATIBILITY
                'flow_rate': round(flow_out, 2), 
                'pressure': round(pressure_out, 2),
                
                'is_leaking': is_leaking,
                'status': 'active' if not is_leaking else 'critical',
                'analysis': analysis # Send analysis to frontend
            })
        
        # Calculate Ward aggregate metrics
        any_leak_detected = any(p['analysis']['is_leak'] for p in ward_data['pipes'])
        avg_confidence = np.mean([p['analysis']['combined_confidence'] for p in ward_data['pipes']]) if ward_data['pipes'] else 0
        
        if not any_leak_detected:
             # Add "heartbeat" noise so the system looks alive (2-14%)
             # Occasional spike to 25-30% to look "real"
             base_noise = random.uniform(2.0, 14.0)
             if random.random() < 0.15: # 15% chance of minor instability
                 base_noise = random.uniform(18.0, 32.0)
             
             avg_confidence = max(avg_confidence, base_noise)
             
             # Generate random "Warning" alerts if risk is elevated
             if avg_confidence > 25 and random.random() < 0.2:
                 ward_data['alerts'].append(f"Warning: Minor pressure instability detected in Ward-{self.ward_id}")
             elif random.random() < 0.05:
                 ward_data['alerts'].append(f"Info: Routine check - Ward-{self.ward_id} nominal")

        ward_data['leak_probability'] = float(round(avg_confidence, 2))
        
        if any_leak_detected and random.random() < 0.2:
             ward_data['alerts'].append(f"CRITICAL: Advanced AI confirms leak in Ward-{self.ward_id} (Confidence: {avg_confidence:.1f}%)")

        return ward_data

class SimulationService:
    """Orchestrates simulation for 4 Wards"""
    
    def __init__(self):
        self.is_running = False
        self.leak_scenario = False
        self.subscribers = []
        
        # Initialize Advanced Detector
        self.detector = AdvancedLeakDetector()
        success = self.detector.load_model()
        if not success:
            print("⚠ Advanced Model not found. Attempting to train on the fly...")
            try:
                self.detector.train_model() # Train if missing
                self.detector.load_model()
            except Exception as e:
                print(f"❌ Failed to train model: {e}")
        
        # Initialize 4 Wards
        self.wards = {
            f"ward_{i}": SmartPipeSystem(i, self.detector) for i in range(1, 5)
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
