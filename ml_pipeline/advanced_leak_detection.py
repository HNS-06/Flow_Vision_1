"""
FlowVision - Advanced Leak Detection Module
Implements Gradient Boosting (XGBoost) and Pressure Gradient Analysis for high-precision leak detection.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
from datetime import datetime

class AdvancedLeakDetector:
    """
    Advanced leak detection using:
    1. Gradient Boosting (XGBoost): Predicts expected flow_out based on flow_in and time features.
    2. Pressure Gradient Analysis: Physics-based check of pressure drop vs flow rate.
    """
    
    def __init__(self, model_dir='ml_pipeline/models'):
        self.model_dir = model_dir
        self.xgb_model = None
        self.pressure_drop_margin = 5.0 # Margin for pressure drop deviation (psi)
        self.flow_residual_threshold = 15.0 # L/m threshold for flow deviation
        
    def load_model(self):
        """Load trained XGBoost model"""
        model_path = os.path.join(self.model_dir, 'xgboost_leak.model')
        if os.path.exists(model_path):
            self.xgb_model = xgb.XGBRegressor()
            self.xgb_model.load_model(model_path)
            print(f"[OK] Advanced Leak Detection Model loaded: {model_path}")
            return True
        else:
            print(f"⚠ Model not found: {model_path}")
            return False

    def train_model(self, data_path='data/raw/advanced_training_data.csv'):
        """
        Train XGBoost model on VALID/NORMAL data.
        Input features: flow_in, pressure_in, hour, day_of_week
        Target: flow_out (should match flow_in in normal conditions with minor loss)
        """
        print("Training Advanced XGBoost Model...")
        
        if not os.path.exists(data_path):
            print(f"⚠ Training data not found: {data_path}")
            return False
            
        df = pd.read_csv(data_path)
        
        # Feature Engineering
        X = df[['flow_in', 'pressure_in', 'hour', 'day_of_week']]
        y = df['flow_out']
        
        # Train XGBoost
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            objective='reg:squarederror'
        )
        self.xgb_model.fit(X, y)
        
        # Save model
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, 'xgboost_leak.model')
        self.xgb_model.save_model(model_path)
        print(f"[OK] specific XGBoost model saved to {model_path}")
        return True

    def detect_xgboost(self, flow_in, pressure_in, current_time, actual_flow_out):
        """
        Algorithm 1: Gradient Boosting Analysis
        Predicts expected flow_out. If actual << predicted, logic suggests a leak.
        """
        if self.xgb_model is None:
            return None # Model not loaded
            
        # Prepare input vector
        # Features: flow_in, pressure_in, hour, day_of_week
        hour = current_time.hour
        day_of_week = current_time.weekday()
        
        input_data = pd.DataFrame([[flow_in, pressure_in, hour, day_of_week]], 
                                columns=['flow_in', 'pressure_in', 'hour', 'day_of_week'])
        
        # Predict expected output flow
        predicted_flow_out = self.xgb_model.predict(input_data)[0]
        
        # Calculate residual (Expected - Actual)
        # Positive residual means we expected MORE water than we got -> LEAK
        residual = predicted_flow_out - actual_flow_out
        
        is_leak = False
        confidence = 0.0
        
        if residual > self.flow_residual_threshold:
            is_leak = True
            # Calculate confidence score based on magnitude of residual
            # Cap at 100%
            confidence = min(100, (residual / self.flow_residual_threshold) * 50 + 50)
            
        return {
            'algorithm': 'XGBoost',
            'is_leak': is_leak,
            'confidence': confidence,
            'expected_flow_out': float(predicted_flow_out),
            'actual_flow_out': float(actual_flow_out),
            'residual': float(residual)
        }

    def detect_pressure_gradient(self, flow_in, pressure_in, pressure_out, pipe_length=100, diameter=0.5):
        """
        Algorithm 2: Pressure Gradient Analysis
        Uses Darcy-Weisbach / Hazen-Williams approximation logic via simple linear loss model for simulation.
        Checks if pressure drop is significantly higher than expected for the given flow.
        """
        
        # Simplified Physics Model: Pressure Drop ~ k * Flow^2 (Turbulent flow)
        # We learn 'k' or estimate it. For this simulation, we use a calibrated baseline.
        # Expected Drop = Base Friction * Flow_In
        
        # In our simulation logic (SmartPipeSystem): 
        # pressure_out approx pressure_in - (friction_loss)
        # We'll use a simplified heuristic that matches the simulation's "normal" physics
        
        # Expected pressure drop for this flow rate (calibrated to simulation params)
        # Simulation: pressure = 45 - (flow - 100)/10
        # effective drop varies. 
        
        actual_drop = pressure_in - pressure_out
        
        # Heuristic: Normal drop shouldn't exceed X for given flow
        # We'll assume a linear relationship for the "Pipeline Segment" checking
        expected_drop = (flow_in / 50.0) # Dummy physics calibration matching approx simulation scale
        
        deviation = actual_drop - expected_drop
        
        is_leak = False
        confidence = 0.0
        
        if deviation > self.pressure_drop_margin:
            is_leak = True
            confidence = min(95, (deviation / self.pressure_drop_margin) * 60 + 40)
            
        return {
            'algorithm': 'PressureGradient',
            'is_leak': is_leak,
            'confidence': confidence,
            'expected_drop': float(expected_drop),
            'actual_drop': float(actual_drop),
            'deviation': float(deviation)
        }

    def analyze_system(self, flow_in, pressure_in, flow_out, pressure_out, timestamp):
        """
        Run both algorithms and combine results.
        """
        xg_result = self.detect_xgboost(flow_in, pressure_in, timestamp, flow_out)
        pg_result = self.detect_pressure_gradient(flow_in, pressure_in, pressure_out)
        
        alerts = []
        combined_score = 0
        
        # Ensemble Logic
        if xg_result and xg_result['is_leak']:
            alerts.append(f"[XGBoost] Flow discrepancy detected (Res: {xg_result['residual']:.1f} L/m)")
            combined_score += xg_result['confidence'] * 0.6
            
        if pg_result['is_leak']:
            alerts.append(f"[PressureGrad] Abnormal pressure drop (Dev: {pg_result['deviation']:.1f} psi)")
            combined_score += pg_result['confidence'] * 0.4
            
        return {
            'is_leak': (len(alerts) > 0),
            'combined_confidence': min(100, combined_score),
            'alerts': alerts,
            'details': {
                'xgboost': xg_result,
                'pressure_gradient': pg_result
            }
        }
