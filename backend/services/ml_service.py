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
import pandas as pd

class MLService:
    """Service for ML model operations"""
    
    def __init__(self):
        self.leak_detector = LeakDetector()
        self.forecaster = ConsumptionForecaster()
        self.ward_analytics = WardAnalytics()
        
        # Try to load existing models
        self.leak_detector.load_model()
        self.forecaster.load_models()
    
    def detect_leaks(self, flow_data: pd.DataFrame):
        """Detect leaks in flow data"""
        try:
            result = self.leak_detector.ensemble_detection(flow_data)
            return result
        except Exception as e:
            print(f"Error in leak detection: {e}")
            return None
    
    def forecast_consumption(self, flow_data: pd.DataFrame, steps=24):
        """Forecast future consumption"""
        try:
            # Generate simple forecast data for demo
            import random
            base_flow = 150.0
            forecast_data = []
            for i in range(steps):
                # Add some variation
                prediction = base_flow + random.uniform(-20, 20) + (i * 0.5)
                forecast_data.append({
                    'hour': i + 1,
                    'prediction': round(prediction, 2),
                    'upper_bound': round(prediction * 1.15, 2),
                    'lower_bound': round(prediction * 0.85, 2)
                })
            return forecast_data
        except Exception as e:
            print(f"Error in forecasting: {e}")
            return None
    
    def get_ward_insights(self):
        """Get AI-generated insights for wards"""
        try:
            # Return simple mock insights for demo
            insights = [
                {"type": "warning", "title": "High Consumption Detected", "message": "Ward 3 showing 15% above average consumption"},
                {"type": "info", "title": "Efficiency Improvement", "message": "Ward 1 reduced consumption by 8% this week"},
                {"type": "success", "title": "No Leaks Detected", "message": "All wards operating within normal parameters"},
                {"type": "info", "title": "Peak Hour Analysis", "message": "Highest demand occurs between 6-9 AM across all wards"}
            ]
            comparison = [
                {"ward_id": 1, "ward_name": "Ward 1", "avg_consumption": 145.2, "trend": "down"},
                {"ward_id": 2, "ward_name": "Ward 2", "avg_consumption": 158.7, "trend": "stable"},
                {"ward_id": 3, "ward_name": "Ward 3", "avg_consumption": 172.3, "trend": "up"},
                {"ward_id": 4, "ward_name": "Ward 4", "avg_consumption": 151.9, "trend": "stable"}
            ]
            return {
                'insights': insights,
                'comparison': comparison
            }
        except Exception as e:
            print(f"Error in ward analytics: {e}")
            return None

# Singleton instance
ml_service = MLService()
