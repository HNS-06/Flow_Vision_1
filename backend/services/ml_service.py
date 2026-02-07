"""
FlowVision Backend - ML Service
Wrapper for ML models
"""

import sys
import os
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ml_pipeline.leak_detection import LeakDetector
from ml_pipeline.consumption_forecast import ConsumptionForecaster
from ml_pipeline.ward_analytics import WardAnalytics
from ml_pipeline.clustering_analysis import ClusteringAnalyzer
from ml_pipeline.optimization import OptimizationEngine
from ml_pipeline.rl_control import RLController
import pandas as pd
import numpy as np

class MLService:
    """Service for ML model operations"""
    
    def __init__(self):
        self.leak_detector = LeakDetector()
        self.forecaster = ConsumptionForecaster()
        self.ward_analytics = WardAnalytics()
        self.clustering = ClusteringAnalyzer()
        self.optimizer = OptimizationEngine()
        self.rl_controllers = {
            i: RLController(ward_id=i) for i in range(1, 5) # 4 Wards
        }
        
        # Try to load existing models or train on the fly
        self.leak_detector.load_model()
        self._initialize_models()
        self._load_rl_models()
        
    def _load_rl_models(self):
        for ctrl in self.rl_controllers.values():
            ctrl.load()

    def _initialize_models(self):
        """Initialize and valid models"""
        # Train Clustering if needed
        if not self.clustering.load_model():
            print("Training clustering model on synthetic data...")
            # Generate synthetic ward hourly data for clustering
            # 4 wards, 30 days, hourly
            data = []
            timestamps = pd.date_range('2025-01-01', periods=24*30, freq='h')
            
            for ward_id in [1, 2, 3, 4]:
                # Different patterns
                base = 100 if ward_id % 2 == 0 else 150
                is_industry = ward_id == 2
                
                for ts in timestamps:
                    flow = base
                    # Daily pattern
                    if is_industry:
                        # Industry: Constant high usage, less day/night var
                        flow += np.random.normal(0, 10)
                    else:
                        # Residential: Peak morning/evening
                        h = ts.hour
                        if 6 <= h <= 9 or 18 <= h <= 21:
                            flow *= 1.5
                        elif 0 <= h <= 5:
                            flow *= 0.3
                            
                    data.append({
                        'ward_id': ward_id,
                        'timestamp': ts,
                        'flow_rate': max(0, flow + np.random.normal(0, 5))
                    })
            
            df_train = pd.DataFrame(data)
            self.clustering.train(df_train)
            
        # Train Forecaster (Global and per ward)
        # We'll just train a representative model for now to ensure it works
        if not self.forecaster.models and not self.forecaster.global_model:
             print("Training forecasting model on synthetic data...")
             # Use the same synthetic data (Ward 1 as representative)
             dummy_data = pd.DataFrame({
                 'timestamp': pd.date_range('2025-01-01', periods=24*60, freq='h'),
                 'flow_rate': [150 + 50*np.sin(i/12) + np.random.normal(0,10) for i in range(24*60)]
             })
             self.forecaster.train(dummy_data)

    def detect_leaks(self, flow_data: pd.DataFrame):
        """Detect leaks in flow data"""
        try:
            result = self.leak_detector.ensemble_detection(flow_data)
            return result
        except Exception as e:
            print(f"Error in leak detection: {e}")
            return None
    
    def forecast_consumption(self, flow_data: pd.DataFrame, steps=24, ward_id=1):
        """Forecast future consumption"""
        try:
            # Prepare history (using flow_data as recent history)
            # In a real app, strict filtering by ward_id on flow_data would happen here
            # For now, we assume flow_data is compatible or we use it as seed
            
            forecast = self.forecaster.predict(flow_data, steps=steps, ward_id=ward_id)
            return forecast
        except Exception as e:
            print(f"Error in forecasting: {e}")
            return None
            
    def optimize_water_distribution(self, total_supply=800):
        """Run LP optimization for current state"""
        try:
            # Mock current state data since we don't have a live state DB yet
            # In prod, fetch real levels and demand
            ward_states = []
            for i in range(1, 5):
                # Simulated current demand based on time of day
                # Assume midday
                base_demand = 180
                if i == 2: base_demand = 250 # Industrial
                
                ward_states.append({
                    'ward_id': i,
                    'demand': base_demand + np.random.randint(-20, 20),
                    'current_level': np.random.randint(40, 80),
                    'priority': 5 if i == 2 else 3 # Priority for Industry/Hospital
                })
                
            result = self.optimizer.optimize_distribution(ward_states, total_supply)
            return result
        except Exception as e:
            print(f"Error in optimization: {e}")
            return None

    def get_rl_control_action(self, ward_id, current_level, demand, hour):
        """Get valve action from RL agent"""
        try:
            agent = self.rl_controllers.get(ward_id)
            if not agent: return None
            
            # Get action (0, 1, 2)
            action_idx = agent.get_action(current_level, demand, hour)
            
            # Map to human readable
            actions = ["Close Valve", "Throttle Valve (50%)", "Open Valve (100%)"]
            return {
                'action_id': int(action_idx),
                'action_desc': actions[action_idx],
                'ward_id': ward_id
            }
        except Exception as e:
            print(f"Error in RL control: {e}")
            return None

    def get_ward_insights(self):
        """Get AI-generated insights for wards"""
        try:
            # 1. Get Clusters
            # We need some recent data to classify. 
            # We'll generate a small snapshot or use stored summary if available.
            # For this demo, we'll re-generate the snapshot on the fly to get live clustering
            # In prod, this would be computed daily
            
            # Ward-specific patterns for classification
            mock_data = []
            timestamps = pd.date_range('2025-01-01', periods=24, freq='h')
            
            # Simulate characteristic data for classification
            for w in [1, 2, 3, 4]:
                base = 100 if w % 2 == 0 else 150
                is_industry = w == 2
                
                for ts in timestamps:
                    flow = base
                    if is_industry:
                        flow += np.random.normal(0, 5) # Flat
                    else:
                        h = ts.hour # Residential peaks
                        if 7 <= h <= 9 or 18 <= h <= 20: flow *= 1.6
                        if 1 <= h <= 5: flow *= 0.2
                        
                    mock_data.append({'ward_id': w, 'timestamp': ts, 'flow_rate': flow})
            
            clusters = self.clustering.get_clusters(pd.DataFrame(mock_data))
            
            # 2. Generate Insights
            insights = []
            
            # Add Cluster Info
            for w_id, info in clusters.items():
                insights.append({
                    "type": "info",
                    "title": f"Ward {w_id} Classification",
                    "message": f"Identified as {info['cluster_name']}."
                })
            
            # Default warnings/success
            insights.append({"type": "success", "title": "System Healthy", "message": "Leak detection active across all wards."})
                
            comparison = [
                {"ward_id": 1, "ward_name": "Ward 1", "avg_consumption": 145.2, "trend": "down"},
                {"ward_id": 2, "ward_name": "Ward 2", "avg_consumption": 158.7, "trend": "stable"},
                {"ward_id": 3, "ward_name": "Ward 3", "avg_consumption": 172.3, "trend": "up"},
                {"ward_id": 4, "ward_name": "Ward 4", "avg_consumption": 151.9, "trend": "stable"}
            ]
            return {
                'insights': insights,
                'comparison': comparison,
                'clusters': clusters
            }
        except Exception as e:
            print(f"Error in ward analytics: {e}")
            return None

# Singleton instance
ml_service = MLService()
