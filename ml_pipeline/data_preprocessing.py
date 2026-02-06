"""
FlowVision - Data Preprocessing Pipeline
Handles data cleaning, feature engineering, and preparation for ML models
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

class DataPreprocessor:
    """Preprocesses water flow and consumption data for ML models"""
    
    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir
        self.flow_data = None
        self.consumption_data = None
        
    def load_data(self):
        """Load raw datasets"""
        flow_path = os.path.join(self.data_dir, 'water_flow.csv')
        consumption_path = os.path.join(self.data_dir, 'consumption_daily.csv')
        
        self.flow_data = pd.read_csv(flow_path, parse_dates=['timestamp'])
        self.consumption_data = pd.read_csv(consumption_path, parse_dates=['date'])
        
        if self.flow_data is not None:
             print(f"Loaded {len(self.flow_data)} flow records")
        if self.consumption_data is not None:
             print(f"Loaded {len(self.consumption_data)} consumption records")
        
        return self.flow_data, self.consumption_data
    
    def handle_missing_values(self, df):
        """Handle missing values using forward fill and interpolation"""
        # Forward fill for small gaps
        df = df.ffill(limit=3)
        
        # Interpolate for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
        
        # Drop remaining NaN rows (if any)
        df = df.dropna()
        
        return df
    
    def remove_outliers(self, df, column, n_std=3):
        """Remove outliers using z-score method"""
        mean = df[column].mean()
        std = df[column].std()
        
        # Keep values within n standard deviations
        df = df[np.abs(df[column] - mean) <= n_std * std]
        
        return df
    
    def create_time_features(self, df, timestamp_col='timestamp'):
        """Create time-based features for ML models"""
        df = df.copy()
        
        df['hour'] = df[timestamp_col].dt.hour
        df['day'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_of_year'] = df[timestamp_col].dt.dayofyear
        df['week_of_year'] = df[timestamp_col].dt.isocalendar().week
        
        # Is weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Is peak hour (6-9 AM, 6-9 PM)
        df['is_peak_hour'] = df['hour'].isin([6, 7, 8, 18, 19, 20]).astype(int)
        
        # Time of day category
        df['time_of_day'] = pd.cut(
            df['hour'], 
            bins=[0, 6, 12, 18, 24], 
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        
        # Cyclical encoding for hour (preserves circular nature)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for day of year
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        return df
    
    def create_rolling_features(self, df, column, windows=[3, 6, 12, 24]):
        """Create rolling statistics features"""
        df = df.copy()
        
        for window in windows:
            # Rolling mean
            df[f'{column}_rolling_mean_{window}h'] = df[column].rolling(
                window=window, min_periods=1
            ).mean()
            
            # Rolling std
            df[f'{column}_rolling_std_{window}h'] = df[column].rolling(
                window=window, min_periods=1
            ).std()
            
            # Rolling min/max
            df[f'{column}_rolling_min_{window}h'] = df[column].rolling(
                window=window, min_periods=1
            ).min()
            
            df[f'{column}_rolling_max_{window}h'] = df[column].rolling(
                window=window, min_periods=1
            ).max()
        
        return df
    
    def create_lag_features(self, df, column, lags=[1, 2, 3, 6, 12, 24]):
        """Create lag features for time series prediction"""
        df = df.copy()
        
        for lag in lags:
            df[f'{column}_lag_{lag}h'] = df[column].shift(lag)
        
        return df
    
    def create_difference_features(self, df, column):
        """Create difference features to capture trends"""
        df = df.copy()
        
        # First difference (change from previous hour)
        df[f'{column}_diff_1h'] = df[column].diff(1)
        
        # Second difference (acceleration)
        df[f'{column}_diff_2h'] = df[column].diff(2)
        
        # Percentage change
        df[f'{column}_pct_change'] = df[column].pct_change()
        
        return df
    
    def preprocess_flow_data(self, save_path='data/processed/flow_processed.csv'):
        """Complete preprocessing pipeline for flow data"""
        print("\n" + "="*60)
        print("Preprocessing Flow Data")
        print("="*60)
        
        df = self.flow_data.copy()
        
        # Handle missing values
        print("Handling missing values...")
        df = self.handle_missing_values(df)
        
        # Create time features
        print("Creating time features...")
        df = self.create_time_features(df)
        
        # Create rolling features for flow_rate
        print("Creating rolling features...")
        df = self.create_rolling_features(df, 'flow_rate')
        
        # Create lag features
        print("Creating lag features...")
        df = self.create_lag_features(df, 'flow_rate')
        
        # Create difference features
        print("Creating difference features...")
        df = self.create_difference_features(df, 'flow_rate')
        
        # Remove outliers (optional, be careful with leak detection)
        # df = self.remove_outliers(df, 'flow_rate', n_std=4)
        
        # Drop rows with NaN created by lag/rolling features
        df = df.dropna()
        
        # Save processed data
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        
        print(f"\n[OK] Processed flow data saved: {save_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Features: {len(df.columns)}")
        
        return df
    
    def preprocess_consumption_data(self, save_path='data/processed/consumption_processed.csv'):
        """Complete preprocessing pipeline for consumption data"""
        print("\n" + "="*60)
        print("Preprocessing Consumption Data")
        print("="*60)
        
        df = self.consumption_data.copy()
        
        # Handle missing values
        print("Handling missing values...")
        df = self.handle_missing_values(df)
        
        # Create time features
        print("Creating time features...")
        df = self.create_time_features(df, timestamp_col='date')
        
        # Create lag features by ward
        print("Creating lag features by ward...")
        for ward_id in df['ward_id'].unique():
            ward_mask = df['ward_id'] == ward_id
            ward_data = df[ward_mask].copy()
            
            # Lag features
            for lag in [1, 7, 14, 30]:
                df.loc[ward_mask, f'consumption_lag_{lag}d'] = ward_data['consumption_m3'].shift(lag)
            
            # Rolling mean
            for window in [7, 14, 30]:
                df.loc[ward_mask, f'consumption_rolling_mean_{window}d'] = ward_data['consumption_m3'].rolling(
                    window=window, min_periods=1
                ).mean()
        
        # Drop rows with NaN
        df = df.dropna()
        
        # Save processed data
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        
        print(f"\n[OK] Processed consumption data saved: {save_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Features: {len(df.columns)}")
        
        return df
    
    def get_feature_summary(self, df):
        """Get summary statistics of features"""
        summary = {
            'num_records': len(df),
            'num_features': len(df.columns),
            'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}" if 'timestamp' in df.columns else 'N/A',
            'missing_values': df.isnull().sum().sum(),
            'numeric_features': list(df.select_dtypes(include=[np.number]).columns)
        }
        return summary

def main():
    """Run preprocessing pipeline"""
    preprocessor = DataPreprocessor()
    
    # Load data
    preprocessor.load_data()
    
    # Preprocess flow data
    flow_processed = preprocessor.preprocess_flow_data()
    
    # Preprocess consumption data
    consumption_processed = preprocessor.preprocess_consumption_data()
    
    print("\n" + "="*60)
    print("[OK] Preprocessing Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
