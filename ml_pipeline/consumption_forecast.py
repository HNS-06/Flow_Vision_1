"""
FlowVision - Consumption Forecasting Module
Implements time-series forecasting for water consumption prediction using SARIMA
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import os
import warnings
import random

warnings.filterwarnings('ignore')

class ConsumptionForecaster:
    """Time-series forecasting for water consumption using SARIMAX"""
    
    def __init__(self, model_dir='ml_pipeline/models'):
        self.model_dir = model_dir
        self.models = {} # Dictionary to store models per ward: {ward_id: model_state}
        self.global_model = None
        os.makedirs(self.model_dir, exist_ok=True)
        
    def _prepare_exog(self, df):
        """Prepare exogenous features: day_of_week, hour, etc."""
        # Check if we have temperature
        exog_cols = []
        
        # Create features if they exist or can be derived
        if 'timestamp' in df.columns:
            if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            
            # Holiday flag (Mock: Sunday is holiday-like)
            df['is_holiday'] = df['day_of_week'].apply(lambda x: 1 if x == 6 else 0)
            
            # Cyclical features for hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            exog_cols = ['hour_sin', 'hour_cos', 'day_of_week', 'is_holiday']
            
            if 'temperature' in df.columns:
                exog_cols.append('temperature')
                
        # Ensure numeric
        return df[exog_cols].fillna(0)

    def train(self, df, ward_id=None, target_col='flow_rate'):
        """Train SARIMA model"""
        print(f"Training SARIMA for {'Ward ' + str(ward_id) if ward_id else 'Global'}...")
        
        # Prepare data
        series = df[target_col]
        exog = self._prepare_exog(df.copy())
        
        # SARIMA Config (daily seasonality)
        # Using Regression with ARIMA errors (ARIMAX) instead of full SARIMA 
        # to avoid convergence issues with high-lag seasonality (24)
        order = (1, 1, 1)
        seasonal_order = (0, 0, 0, 0) # rely on exog features (hour_sin/cos)
        
        try:
            model = SARIMAX(series, exog=exog, order=order, seasonal_order=seasonal_order, 
                            enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False)
            
            # Evaluate
            predictions = results.predict()
            mae = mean_absolute_error(series, predictions)
            print(f"  MAE: {mae:.2f}")
            
            model_state = {
                'model_fit': results, # Note: Pickling SARIMAX results can be large
                'exog_columns': exog.columns.tolist(),
                'mae': mae
            }
            
            if ward_id:
                self.models[ward_id] = model_state
                self._save_model(model_state, f'sarima_ward_{ward_id}.pkl')
            else:
                self.global_model = model_state
                self._save_model(model_state, 'sarima_global.pkl')
                
            return {
                'success': True,
                'mae': mae
            }
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return {'success': False, 'error': str(e)}

    def predict(self, df_history, steps=24, ward_id=None):
        """Forecast future consumption"""
        model_state = self.models.get(ward_id) if ward_id else self.global_model
        
        if not model_state:
            # Fallback: Create a mock prediction if model missing
            print(f"⚠ Model not found for Ward {ward_id}. Using fallback.")
            last_val = df_history['flow_rate'].iloc[-1] if not df_history.empty else 150
            return self._fallback_forecast(last_val, steps, ward_id)
            
        results = model_state['model_fit']
        
        # Create future exogenous variables
        last_ts = pd.to_datetime(df_history['timestamp'].iloc[-1])
        future_dates = [last_ts + pd.Timedelta(hours=i+1) for i in range(steps)]
        
        future_df = pd.DataFrame({'timestamp': future_dates})
        # Mock temperature for future (simple sine wave)
        day_of_year = last_ts.dayofyear
        future_df['temperature'] = [25 + 5 * np.sin(2 * np.pi * (day_of_year)/365) for _ in range(steps)]
        
        exog_future = self._prepare_exog(future_df)
        
        # Align columns
        required_cols = model_state['exog_columns']
        for col in required_cols:
            if col not in exog_future.columns:
                exog_future[col] = 0
        exog_future = exog_future[required_cols]
        
        # Forecast
        forecast = results.get_forecast(steps=steps, exog=exog_future)
        mean_forecast = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05) # 95% confidence
        
        output = []
        for i, (val, (lower, upper)) in enumerate(zip(mean_forecast, conf_int.values)):
            output.append({
                'hour': i + 1,
                'prediction': round(float(val), 2),
                'lower_bound': round(float(lower), 2),
                'upper_bound': round(float(upper), 2)
            })
            
        return output

    def _save_model(self, state, filename):
        path = os.path.join(self.model_dir, filename)
        # We only save necessary parts to keep size down if needed, but joblib handles it well
        # Accessing 'model_fit' might be heavy. For production, save parameters only.
        # For this prototype, we'll skip saving heavy objects to disk to avoid 'pickle' issues with some statsmodels versions
        # interacting with uvicorn reload. We keep in memory.
        pass 

    def load_models(self):
        # Stub for loading if we decided to persist
        pass
        
    def _fallback_forecast(self, base_val, steps, ward_id):
        data = []
        # Ward specific multiplier
        multipliers = {1: 1.0, 2: 1.5, 3: 0.6, 4: 1.2}
        mult = multipliers.get(ward_id, 1.0)
        
        random.seed(ward_id if ward_id else 0)
        
        for i in range(steps):
            # Sine wave pattern
            val = base_val * mult + (np.sin(i / 3) * 10) + random.uniform(-5, 5)
            data.append({
                'hour': i + 1,
                'prediction': round(val, 2),
                'lower_bound': round(val * 0.9, 2),
                'upper_bound': round(val * 1.1, 2)
            })
        return data
