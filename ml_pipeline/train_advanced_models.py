"""
Script to train the Advanced XGBoost Leak Detection Model
"""
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from datetime import datetime, timedelta

def generate_normal_data(num_samples=1000):
    """
    Generate synthetic 'Normal' operating data for training.
    Normal: flow_out ≈ flow_in (with minor physical loss/noise)
    """
    print(f"Generating {num_samples} samples of normal data...")
    
    data = []
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    
    for i in range(num_samples):
        timestamp = start_time + timedelta(minutes=i*5)
        
        # Base Flow Pattern (Diurnal cycle)
        hour = timestamp.hour
        # Peak at 8am and 7pm
        base_demand = 50 + 50 * np.sin((hour - 6) * np.pi / 12)**2
        
        # Pump Input Flow (matches demand + overhead)
        flow_in = base_demand * 1.05 + np.random.normal(0, 2)
        
        # System Pressure (inverse to flow mostly)
        pressure_in = 60 - (flow_in / 10) + np.random.normal(0, 1)
        
        # Output Flow (Normal = Input - tiny loss)
        # Efficiency ~ 98-99%
        flow_out = flow_in * np.random.uniform(0.98, 0.995)
        
        data.append({
            'timestamp': timestamp,
            'flow_in': flow_in,
            'pressure_in': pressure_in,
            'flow_out': flow_out, # TARGET
            'hour': hour,
            'day_of_week': timestamp.weekday()
        })
        
    return pd.DataFrame(data)

def train():
    model_dir = 'ml_pipeline/models'
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Generate Data
    df = generate_normal_data(2000)
    
    # 2. Train XGBoost
    X = df[['flow_in', 'pressure_in', 'hour', 'day_of_week']]
    y = df['flow_out']
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        objective='reg:squarederror'
    )
    
    print("Training XGBoost model...")
    model.fit(X, y)
    
    # 3. Save
    model_path = os.path.join(model_dir, 'xgboost_leak.model')
    model.save_model(model_path)
    print(f"[SUCCESS] Model saved to {model_path}")
    
    # 4. Verify
    sample = X.iloc[0:1]
    pred = model.predict(sample)[0]
    actual = y.iloc[0]
    print(f"Verification - Input: {sample.values}, Predicted: {pred:.2f}, Actual: {actual:.2f}")

if __name__ == "__main__":
    train()
