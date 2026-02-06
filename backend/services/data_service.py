"""
FlowVision Backend - Data Service
Handles data loading and management
"""

import pandas as pd
import os
from backend.core.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

class DataService:
    """Service for data operations"""
    
    def __init__(self):
        self.flow_data = None
        self.consumption_data = None
        self.ward_data = None
        self.leak_labels = None
    
    def load_all_data(self):
        """Load all datasets"""
        try:
            # Load raw data
            self.flow_data = pd.read_csv(
                os.path.join(RAW_DATA_DIR, 'water_flow.csv'),
                parse_dates=['timestamp']
            )
            self.consumption_data = pd.read_csv(
                os.path.join(RAW_DATA_DIR, 'consumption_daily.csv'),
                parse_dates=['date']
            )
            self.ward_data = pd.read_csv(
                os.path.join(RAW_DATA_DIR, 'ward_usage.csv')
            )
            self.leak_labels = pd.read_csv(
                os.path.join(RAW_DATA_DIR, 'leak_labels.csv'),
                parse_dates=['timestamp']
            )
            
            print("[OK] All data loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def get_recent_flow_data(self, hours=24):
        """Get recent flow data"""
        if self.flow_data is None:
            self.load_all_data()
        
        return self.flow_data.tail(hours).to_dict('records')
    
    def get_consumption_by_ward(self, ward_id=None, days=30):
        """Get consumption data by ward"""
        if self.consumption_data is None:
            self.load_all_data()
        
        data = self.consumption_data.tail(days * 12)  # Approximate
        
        if ward_id:
            data = data[data['ward_id'] == ward_id]
        
        return data.to_dict('records')
    
    def get_ward_statistics(self):
        """Get ward statistics"""
        if self.ward_data is None:
            self.load_all_data()
        
        return self.ward_data.to_dict('records')
    
    def get_leak_history(self, days=7):
        """Get recent leak events"""
        if self.leak_labels is None:
            self.load_all_data()
        
        cutoff_date = self.leak_labels['timestamp'].max() - pd.Timedelta(days=days)
        recent_leaks = self.leak_labels[self.leak_labels['timestamp'] >= cutoff_date]
        
        return recent_leaks.to_dict('records')

# Singleton instance
data_service = DataService()
