"""
FlowVision - Leak Detection Module
Implements multiple anomaly detection algorithms for water leak identification
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import os

class LeakDetector:
    """Multi-algorithm leak detection system"""
    
    def __init__(self, model_dir='ml_pipeline/models'):
        self.model_dir = model_dir
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def prepare_features(self, df):
        """Prepare features for leak detection"""
        feature_cols = [
            'flow_rate', 'pressure', 'temperature',
            'hour', 'is_weekend', 'is_peak_hour',
            'flow_rate_rolling_mean_3h', 'flow_rate_rolling_std_3h',
            'flow_rate_rolling_mean_6h', 'flow_rate_rolling_std_6h',
            'flow_rate_rolling_mean_12h', 'flow_rate_rolling_std_12h',
            'flow_rate_diff_1h', 'flow_rate_pct_change'
        ]
        
        # Filter to available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_cols
        
        return df[available_cols]
    
    def detect_zscore_anomalies(self, df, threshold=3):
        """Detect anomalies using Z-score method"""
        flow_mean = df['flow_rate'].mean()
        flow_std = df['flow_rate'].std()
        
        # Calculate Z-score
        df['zscore'] = np.abs((df['flow_rate'] - flow_mean) / flow_std)
        
        # Anomaly if Z-score > threshold
        df['zscore_anomaly'] = (df['zscore'] > threshold).astype(int)
        
        # Anomaly score (0-100)
        df['zscore_score'] = np.clip(df['zscore'] / threshold * 100, 0, 100)
        
        return df
    
    def detect_rolling_deviation(self, df, window=24, threshold=2):
        """Detect anomalies using rolling mean deviation"""
        # Calculate rolling statistics
        rolling_mean = df['flow_rate'].rolling(window=window, min_periods=1).mean()
        rolling_std = df['flow_rate'].rolling(window=window, min_periods=1).std()
        
        # Calculate deviation from rolling mean
        df['rolling_deviation'] = np.abs(df['flow_rate'] - rolling_mean) / (rolling_std + 1e-6)
        
        # Anomaly if deviation > threshold
        df['rolling_anomaly'] = (df['rolling_deviation'] > threshold).astype(int)
        
        # Anomaly score (0-100)
        df['rolling_score'] = np.clip(df['rolling_deviation'] / threshold * 100, 0, 100)
        
        return df
    
    def detect_night_consumption_anomaly(self, df, threshold_multiplier=1.5):
        """Detect unusual consumption during night hours (potential leak indicator)"""
        # Night hours: 0-5 AM
        night_mask = df['hour'].isin([0, 1, 2, 3, 4, 5])
        
        # Calculate average night consumption
        night_avg = df.loc[night_mask, 'flow_rate'].mean()
        day_avg = df.loc[~night_mask, 'flow_rate'].mean()
        
        # Anomaly if night consumption is unusually high
        df['night_anomaly'] = 0
        df.loc[night_mask, 'night_anomaly'] = (
            df.loc[night_mask, 'flow_rate'] > night_avg * threshold_multiplier
        ).astype(int)
        
        # Night anomaly score
        df['night_score'] = 0.0
        df.loc[night_mask, 'night_score'] = np.clip(
            (df.loc[night_mask, 'flow_rate'] / (night_avg * threshold_multiplier)) * 100,
            0, 100
        )
        
        return df
    
    def train_isolation_forest(self, df, contamination=0.05):
        """Train Isolation Forest model for anomaly detection"""
        print("Training Isolation Forest...")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.isolation_forest.fit(X_scaled)
        
        # Save model
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, 'isolation_forest.pkl')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.isolation_forest, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"[OK] Model saved: {model_path}")
        
        return self.isolation_forest
    
    def load_model(self):
        """Load trained Isolation Forest model"""
        model_path = os.path.join(self.model_dir, 'isolation_forest.pkl')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            with open(model_path, 'rb') as f:
                self.isolation_forest = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("[OK] Model loaded successfully")
            return True
        else:
            print("⚠ Model not found. Please train first.")
            return False
    
    def predict_isolation_forest(self, df):
        """Predict anomalies using Isolation Forest"""
        if self.isolation_forest is None:
            # Try loading
            if not self.load_model():
                 print("Model not trained or found.")
                 return df # Return original df without predictions if model missing
        
        if self.isolation_forest is None: # Double check
             raise ValueError("Model not trained. Call train_isolation_forest() first.")
        
        # Prepare features
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Predict anomalies (-1 = anomaly, 1 = normal)
        predictions = self.isolation_forest.predict(X_scaled)
        df['if_anomaly'] = (predictions == -1).astype(int)
        
        # Anomaly score (based on decision function)
        scores = self.isolation_forest.decision_function(X_scaled)
        # Normalize to 0-100 (lower score = more anomalous)
        df['if_score'] = np.clip((1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)) * 100, 0, 100)
        
        return df
    
    def ensemble_detection(self, df):
        """Combine multiple detection methods for robust leak detection"""
        # Apply all detection methods
        df = self.detect_zscore_anomalies(df)
        df = self.detect_rolling_deviation(df)
        df = self.detect_night_consumption_anomaly(df)
        
        # If Isolation Forest is trained, use it
        if self.isolation_forest is not None:
            df = self.predict_isolation_forest(df)
            
            # Ensemble score (weighted average)
            df['leak_probability'] = (
                0.3 * df['zscore_score'] +
                0.3 * df['rolling_score'] +
                0.2 * df['night_score'] +
                0.2 * df['if_score']
            )
        else:
            # Without Isolation Forest
            df['leak_probability'] = (
                0.4 * df['zscore_score'] +
                0.4 * df['rolling_score'] +
                0.2 * df['night_score']
            )
        
        # Final leak classification (threshold at 60)
        df['is_leak_detected'] = (df['leak_probability'] > 60).astype(int)
        
        return df
    
    def get_feature_importance(self):
        """Get feature importance for explainability"""
        if self.isolation_forest is None or self.feature_columns is None:
            return None
        
        # For Isolation Forest, we can't directly get feature importance
        # But we can return the features used
        return {
            'features_used': self.feature_columns,
            'num_features': len(self.feature_columns)
        }
    
    def evaluate_detection(self, df, ground_truth_col='is_leak'):
        """Evaluate detection performance if ground truth is available"""
        if ground_truth_col not in df.columns:
            print("⚠ Ground truth not available for evaluation")
            return None
        
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        y_true = df[ground_truth_col]
        y_pred = df['is_leak_detected']
        
        print("\n" + "="*60)
        print("Leak Detection Evaluation")
        print("="*60)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Leak']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        
        # ROC AUC Score
        if 'leak_probability' in df.columns:
            # Handle NaNs in probability
            probs = df['leak_probability'].fillna(0)
            auc = roc_auc_score(y_true, probs / 100)
            print(f"\nROC AUC Score: {auc:.4f}")
        
        return {
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }

def main():
    """Demo leak detection pipeline"""
    print("="*60)
    print("FlowVision - Leak Detection Demo")
    print("="*60)
    
    # Load processed data
    data_path = 'data/processed/flow_processed.csv'
    if not os.path.exists(data_path):
        print(f"⚠ Processed data not found: {data_path}")
        print("Please run data_preprocessing.py first")
        return
    
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    print(f"\nLoaded {len(df)} records")
    
    # Initialize detector
    detector = LeakDetector()
    
    # Train Isolation Forest
    detector.train_isolation_forest(df, contamination=0.05)
    
    # Perform ensemble detection
    df = detector.ensemble_detection(df)
    
    # Evaluate if ground truth available
    if 'is_leak' in df.columns:
        detector.evaluate_detection(df, 'is_leak')
    
    # Save results
    output_path = 'ml_pipeline/evaluation/leak_detection_results.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Results saved: {output_path}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("Detection Summary")
    print("="*60)
    print(f"Total records analyzed: {len(df)}")
    print(f"Leaks detected: {df['is_leak_detected'].sum()}")
    print(f"Detection rate: {df['is_leak_detected'].sum() / len(df) * 100:.2f}%")
    print(f"Average leak probability: {df['leak_probability'].mean():.2f}")

if __name__ == "__main__":
    main()
