"""
FlowVision - Synthetic Data Generator
Generates realistic water distribution datasets for ML training and demo
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
START_DATE = datetime(2025, 7, 1)
END_DATE = datetime(2026, 1, 31)
NUM_WARDS = 12
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def generate_water_flow_data():
    """Generate time-series water flow sensor data with realistic patterns"""
    print("Generating water flow data...")
    
    # Generate hourly timestamps
    timestamps = pd.date_range(start=START_DATE, end=END_DATE, freq='h')
    num_records = len(timestamps)
    
    data = []
    
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        day_of_week = ts.dayofweek
        
        # Base flow rate with daily pattern (higher during day, lower at night)
        base_flow = 150 + 80 * np.sin((hour - 6) * np.pi / 12)
        
        # Weekend pattern (slightly lower consumption)
        if day_of_week >= 5:
            base_flow *= 0.85
        
        # Peak hours (6-9 AM, 6-9 PM)
        if hour in [6, 7, 8, 18, 19, 20]:
            base_flow *= 1.3
        
        # Add seasonal variation
        day_of_year = ts.timetuple().tm_yday
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
        base_flow *= seasonal_factor
        
        # Add random noise
        flow_rate = base_flow + np.random.normal(0, 10)
        
        # Inject leak anomalies (5% of the time, random duration)
        is_leak = False
        if np.random.random() < 0.05:
            # Leak causes higher flow during off-peak hours
            if hour in [0, 1, 2, 3, 4, 5, 22, 23]:
                flow_rate *= 1.5 + np.random.uniform(0, 0.5)
                is_leak = True
        
        # Pressure (inversely related to flow)
        pressure = 45 + np.random.normal(0, 3) - (flow_rate - 150) / 20
        
        # Temperature (seasonal variation)
        temp = 25 + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 2)
        
        data.append({
            'timestamp': ts,
            'flow_rate': max(0.0, float(flow_rate)),  # L/min
            'pressure': max(20.0, min(60.0, float(pressure))),  # PSI
            'temperature': temp,  # Celsius
            'is_leak': is_leak
        })
    
    df = pd.DataFrame(data)
    output_path = os.path.join(DATA_DIR, 'water_flow.csv')
    df.to_csv(output_path, index=False)
    print(f"[OK] Generated {len(df)} flow records -> {output_path}")
    return df

def generate_consumption_daily(flow_df):
    """Generate daily consumption aggregated by ward"""
    print("Generating daily consumption data...")
    
    # Aggregate flow to daily consumption
    flow_df['date'] = flow_df['timestamp'].dt.date
    daily_total = flow_df.groupby('date')['flow_rate'].sum().reset_index()
    daily_total['total_consumption'] = daily_total['flow_rate'] * 60 / 1000  # Convert to cubic meters
    
    data = []
    
    for _, row in daily_total.iterrows():
        date = row['date']
        total = row['total_consumption']
        
        # Distribute consumption across wards with different patterns
        ward_distribution = np.random.dirichlet(np.ones(NUM_WARDS) * 2)
        
        for ward_id in range(1, NUM_WARDS + 1):
            # Ward-specific multiplier (some wards consume more)
            ward_multiplier = 0.7 + (ward_id % 4) * 0.15
            consumption = total * ward_distribution[ward_id - 1] * ward_multiplier
            
            # Add ward-specific noise
            consumption += np.random.normal(0, consumption * 0.05)
            
            data.append({
                'date': date,
                'ward_id': ward_id,
                'consumption_m3': max(0, consumption),
                'num_connections': 500 + ward_id * 50 + np.random.randint(-20, 20)
            })
    
    df = pd.DataFrame(data)
    output_path = os.path.join(DATA_DIR, 'consumption_daily.csv')
    df.to_csv(output_path, index=False)
    print(f"[OK] Generated {len(df)} daily consumption records -> {output_path}")
    return df

def generate_leak_labels(flow_df):
    """Generate labeled leak events for training"""
    print("Generating leak labels...")
    
    # Extract leak events from flow data
    leak_events = flow_df[flow_df['is_leak'] == True].copy()
    
    # Add leak severity and type
    leak_events['leak_severity'] = np.random.choice(
        ['minor', 'moderate', 'severe'], 
        size=len(leak_events),
        p=[0.6, 0.3, 0.1]
    )
    
    leak_events['leak_type'] = np.random.choice(
        ['pipe_burst', 'joint_leak', 'valve_leak', 'corrosion'],
        size=len(leak_events),
        p=[0.2, 0.4, 0.25, 0.15]
    )
    
    # Calculate anomaly score
    leak_events['anomaly_score'] = (
        (leak_events['flow_rate'] - flow_df['flow_rate'].mean()) / 
        flow_df['flow_rate'].std()
    )
    
    leak_events = leak_events[['timestamp', 'flow_rate', 'pressure', 
                                'leak_severity', 'leak_type', 'anomaly_score']]
    
    output_path = os.path.join(DATA_DIR, 'leak_labels.csv')
    leak_events.to_csv(output_path, index=False)
    print(f"[OK] Generated {len(leak_events)} leak labels -> {output_path}")
    return leak_events

def generate_ward_usage():
    """Generate aggregated ward-level statistics"""
    print("Generating ward usage statistics...")
    
    data = []
    
    for ward_id in range(1, NUM_WARDS + 1):
        # Ward characteristics
        population = 5000 + ward_id * 500 + np.random.randint(-500, 500)
        area_km2 = 2 + ward_id * 0.3 + np.random.uniform(-0.2, 0.2)
        
        # Infrastructure age (older wards have higher leak risk)
        infrastructure_age = 10 + ward_id * 2 + np.random.randint(-3, 3)
        
        # Average consumption per capita
        per_capita_consumption = 120 + np.random.normal(0, 15)  # Liters per day
        
        # Leak risk score (based on age and other factors)
        leak_risk = min(100, infrastructure_age * 2 + np.random.uniform(0, 20))
        
        data.append({
            'ward_id': ward_id,
            'ward_name': f'Ward-{ward_id}',
            'population': population,
            'area_km2': round(float(area_km2), 2),
            'infrastructure_age_years': infrastructure_age,
            'avg_daily_consumption_m3': round(float(population * per_capita_consumption / 1000), 2),
            'per_capita_consumption_lpd': round(float(per_capita_consumption), 2),
            'leak_risk_score': round(float(leak_risk), 2),
            'num_sensors': 3 + ward_id % 5
        })
    
    df = pd.DataFrame(data)
    output_path = os.path.join(DATA_DIR, 'ward_usage.csv')
    df.to_csv(output_path, index=False)
    print(f"[OK] Generated {len(df)} ward records -> {output_path}")
    return df

def main():
    """Generate all datasets"""
    print("=" * 60)
    print("FlowVision - Synthetic Data Generation")
    print("=" * 60)
    print(f"Date Range: {START_DATE.date()} to {END_DATE.date()}")
    print(f"Number of Wards: {NUM_WARDS}")
    print(f"Output Directory: {DATA_DIR}")
    print("=" * 60)
    print()
    
    # Generate all datasets
    flow_df = generate_water_flow_data()
    consumption_df = generate_consumption_daily(flow_df)
    leak_df = generate_leak_labels(flow_df)
    ward_df = generate_ward_usage()
    
    print()
    print("=" * 60)
    print("[OK] Data generation complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  - water_flow.csv: {len(flow_df):,} records")
    print(f"  - consumption_daily.csv: {len(consumption_df):,} records")
    print(f"  - leak_labels.csv: {len(leak_df):,} records")
    print(f"  - ward_usage.csv: {len(ward_df):,} records")
    print()

if __name__ == "__main__":
    main()
