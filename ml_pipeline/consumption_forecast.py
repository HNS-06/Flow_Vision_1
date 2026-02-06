"""
FlowVision - Consumption Forecasting Module
Implements time-series forecasting for water consumption prediction
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class ConsumptionForecaster:
    """Time-series forecasting for water consumption"""
    
    def __init__(self, model_dir='ml_pipeline/models'):
        self.model_dir = model_dir
        self.lr_model = None
        self.arima_model = None
        self.feature_columns = None
        
    def prepare_features_lr(self, df):
        """Prepare features for Linear Regression"""
        feature_cols = [
            'hour', 'day', 'month', 'day_of_week', 'day_of_year',
            'is_weekend', 'is_peak_hour',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'flow_rate_lag_1h', 'flow_rate_lag_2h', 'flow_rate_lag_3h',
            'flow_rate_lag_6h', 'flow_rate_lag_12h', 'flow_rate_lag_24h',
            'flow_rate_rolling_mean_3h', 'flow_rate_rolling_mean_6h',
            'flow_rate_rolling_mean_12h', 'flow_rate_rolling_mean_24h'
        ]
        
        # Filter to available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_cols
        
        return df[available_cols]
    
    def train_linear_regression(self, df, target_col='flow_rate'):
        """Train Linear Regression model"""
        print("Training Linear Regression model...")
        
        # Prepare features
        X = self.prepare_features_lr(df)
        y = df[target_col]
        
        # Split train/test (80/20)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        self.lr_model = LinearRegression()
        self.lr_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.lr_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.4f}")
        
        # Save model
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, 'lr_forecast.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.lr_model, f)
        
        print(f"[OK] Model saved: {model_path}")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'model': self.lr_model
        }
    
    def train_arima(self, df, target_col='flow_rate', order=(2, 1, 2), seasonal_order=None):
        """Train ARIMA model"""
        print("Training ARIMA model...")
        
        # Use only target column for ARIMA
        series = df[target_col]
        
        # Split train/test
        split_idx = int(len(series) * 0.8)
        train, test = series[:split_idx], series[split_idx:]
        
        try:
            # Fit ARIMA
            model = ARIMA(train, order=order)
            self.arima_model = model.fit()
            
            # Forecast
            forecast = self.arima_model.forecast(steps=len(test))
            
            # Evaluate
            mae = mean_absolute_error(test, forecast)
            rmse = np.sqrt(mean_squared_error(test, forecast))
            r2 = r2_score(test, forecast)
            
            print(f"  MAE: {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  R²: {r2:.4f}")
            
            # Save model
            model_path = os.path.join(self.model_dir, 'arima_forecast.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.arima_model, f)
            
            print(f"[OK] Model saved: {model_path}")
            
            return {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'model': self.arima_model
            }
        
        except Exception as e:
            print(f"⚠ ARIMA training failed: {e}")
            return None
    
    def load_models(self):
        """Load trained models"""
        lr_path = os.path.join(self.model_dir, 'lr_forecast.pkl')
        arima_path = os.path.join(self.model_dir, 'arima_forecast.pkl')
        
        if os.path.exists(lr_path):
            with open(lr_path, 'rb') as f:
                self.lr_model = pickle.load(f)
            print("[OK] Linear Regression model loaded")
        
        if os.path.exists(arima_path):
            with open(arima_path, 'rb') as f:
                self.arima_model = pickle.load(f)
            print("[OK] ARIMA model loaded")
        
        return self.lr_model is not None or self.arima_model is not None
    
    def predict_next_hour(self, df, method='lr'):
        """Predict consumption for next hour"""
        if method == 'lr' and self.lr_model is None:
            raise ValueError("Linear Regression model not trained")
        
        # Get last row features
        X = self.prepare_features_lr(df.tail(1))
        prediction = self.lr_model.predict(X)[0]
        
        return prediction
    
    def predict_next_day(self, df, hours=24, method='lr'):
        """Predict consumption for next 24 hours"""
        predictions = []
        
        if method == 'lr':
            # Iterative prediction (each prediction feeds into next)
            current_df = df.copy()
            
            for i in range(hours):
                # Predict next hour
                pred = self.predict_next_hour(current_df, method='lr')
                predictions.append(pred)
                
                # Create next row (simplified - in production, update all features)
                # This is a simplified version for demo
                
        elif method == 'arima' and self.arima_model is not None:
            # ARIMA forecast
            forecast = self.arima_model.forecast(steps=hours)
            predictions = forecast.tolist()
        
        return predictions
    
    def predict_with_confidence(self, df, steps=24, confidence=0.95):
        """Predict with confidence intervals"""
        if self.arima_model is None:
            # Use simple std-based confidence for LR
            predictions = self.predict_next_day(df, hours=steps, method='lr')
            std = np.std(predictions)
            
            lower_bound = [p - 1.96 * std for p in predictions]
            upper_bound = [p + 1.96 * std for p in predictions]
            
            return {
                'predictions': predictions,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        else:
            # ARIMA provides confidence intervals
            forecast_result = self.arima_model.get_forecast(steps=steps)
            forecast_df = forecast_result.summary_frame(alpha=1-confidence)
            
            return {
                'predictions': forecast_df['mean'].tolist(),
                'lower_bound': forecast_df['mean_ci_lower'].tolist(),
                'upper_bound': forecast_df['mean_ci_upper'].tolist()
            }
    
    def identify_peak_windows(self, df, threshold_percentile=75):
        """Identify peak consumption windows"""
        # Calculate percentile threshold
        threshold = df['flow_rate'].quantile(threshold_percentile / 100)
        
        # Find peak hours
        peak_hours = df[df['flow_rate'] > threshold].groupby('hour').size()
        peak_hours = peak_hours.sort_values(ascending=False)
        
        return {
            'threshold': threshold,
            'peak_hours': peak_hours.index.tolist()[:5],
            'peak_hour_counts': peak_hours.values.tolist()[:5]
        }
    
    def forecast_summary(self, df, steps=24):
        """Generate comprehensive forecast summary"""
        # Predict next 24 hours
        forecast = self.predict_with_confidence(df, steps=steps)
        
        # Calculate statistics
        avg_forecast = np.mean(forecast['predictions'])
        max_forecast = np.max(forecast['predictions'])
        min_forecast = np.min(forecast['predictions'])
        
        # Compare with historical average
        historical_avg = df['flow_rate'].tail(24 * 7).mean()  # Last week average
        change_pct = ((avg_forecast - historical_avg) / historical_avg) * 100
        
        # Identify peak windows
        peak_info = self.identify_peak_windows(df)
        
        return {
            'forecast': forecast,
            'statistics': {
                'average': avg_forecast,
                'maximum': max_forecast,
                'minimum': min_forecast,
                'historical_average': historical_avg,
                'change_percentage': change_pct
            },
            'peak_windows': peak_info
        }

def main():
    """Demo forecasting pipeline"""
    print("="*60)
    print("FlowVision - Consumption Forecasting Demo")
    print("="*60)
    
    # Load processed data
    data_path = 'data/processed/flow_processed.csv'
    if not os.path.exists(data_path):
        print(f"⚠ Processed data not found: {data_path}")
        print("Please run data_preprocessing.py first")
        return
    
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    print(f"\nLoaded {len(df)} records")
    
    # Initialize forecaster
    forecaster = ConsumptionForecaster()
    
    # Train Linear Regression
    lr_results = forecaster.train_linear_regression(df)
    
    # Train ARIMA (may take longer)
    print("\nTraining ARIMA (this may take a few minutes)...")
    arima_results = forecaster.train_arima(df, order=(1, 1, 1))
    
    # Generate forecast summary
    print("\n" + "="*60)
    print("Forecast Summary")
    print("="*60)
    
    summary = forecaster.forecast_summary(df, steps=24)
    
    print(f"\nNext 24 Hours Forecast:")
    print(f"  Average: {summary['statistics']['average']:.2f} L/min")
    print(f"  Maximum: {summary['statistics']['maximum']:.2f} L/min")
    print(f"  Minimum: {summary['statistics']['minimum']:.2f} L/min")
    print(f"  Change from last week: {summary['statistics']['change_percentage']:+.2f}%")
    
    print(f"\nPeak Hours: {summary['peak_windows']['peak_hours']}")
    
    # Save forecast
    output_path = 'ml_pipeline/evaluation/forecast_results.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    forecast_df = pd.DataFrame({
        'hour': range(1, 25),
        'prediction': summary['forecast']['predictions'],
        'lower_bound': summary['forecast']['lower_bound'],
        'upper_bound': summary['forecast']['upper_bound']
    })
    forecast_df.to_csv(output_path, index=False)
    print(f"\n[OK] Forecast saved: {output_path}")

if __name__ == "__main__":
    main()
