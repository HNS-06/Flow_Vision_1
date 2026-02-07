
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

class ClusteringAnalyzer:
    """
    Analyzes water consumption patterns to cluster wards into groups
    (e.g., Residential, Industrial, Mixed)
    """
    
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.model_path = os.path.join(os.path.dirname(__file__), 'models', 'ward_clustering.pkl')
        self.scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'ward_scaler.pkl')
        
    def _extract_features(self, df):
        """
        Extract features for clustering from raw consumption data.
        Expected input df columns: ['ward_id', 'timestamp', 'flow_rate']
        """
        if df.empty:
            return pd.DataFrame()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        
        features_list = []
        
        for ward_id, group in df.groupby('ward_id'):
            # 1. Average Daily Consumption
            avg_daily = group.groupby(group['timestamp'].dt.date)['flow_rate'].mean().mean()
            
            # 2. Peak Hour (Hour with max average flow)
            hourly_avg = group.groupby('hour')['flow_rate'].mean()
            peak_hour = hourly_avg.idxmax()
            
            # 3. Usage Variation (Coefficient of Variation)
            variation = group['flow_rate'].std() / group['flow_rate'].mean() if group['flow_rate'].mean() > 0 else 0
            
            # 4. Night vs Day Ratio (Night: 0-6, Day: 6-18)
            night_flow = group[group['hour'].isin(range(0, 6))]['flow_rate'].mean()
            day_flow = group[group['hour'].isin(range(6, 18))]['flow_rate'].mean()
            night_day_ratio = night_flow / day_flow if day_flow > 0 else 0
            
            features_list.append({
                'ward_id': ward_id,
                'avg_daily': avg_daily,
                'peak_hour': peak_hour,
                'variation': variation,
                'night_day_ratio': night_day_ratio
            })
            
        return pd.DataFrame(features_list).set_index('ward_id')

    def train(self, raw_data):
        """
        Train the clustering model
        """
        features_df = self._extract_features(raw_data)
        
        if features_df.empty:
            print("Not enough data to train clustering model.")
            return
            
        # Scale features
        X = self.scaler.fit_transform(features_df)
        
        # Train KMeans
        self.model.fit(X)
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"Clustering model trained on {len(features_df)} wards.")
        
        return self.get_clusters(raw_data)

    def load_model(self):
        """Load trained model"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            return True
        except:
            return False

    def get_clusters(self, raw_data):
        """
        Assign clusters to wards based on data
        """
        features_df = self._extract_features(raw_data)
        if features_df.empty:
            return {}
            
        X = self.scaler.transform(features_df)
        labels = self.model.predict(X)
        
        results = {}
        for ward_id, label in zip(features_df.index, labels):
            # Interpret clusters based on features (Simple heuristic)
            # In a real app, you'd analyze the centroids
            row = features_df.loc[ward_id]
            
            # Heuristic naming
            cluster_name = f"Cluster {label}"
            if row['night_day_ratio'] > 0.8:
                cluster_name = "Industrial/24h" # High night usage
            elif row['peak_hour'] in [7, 8, 9, 18, 19, 20]:
                cluster_name = "Residential" # Typical peaks
            else:
                cluster_name = "Mixed Usage"
                
            results[ward_id] = {
                'cluster_id': int(label),
                'cluster_name': cluster_name,
                'features': row.to_dict()
            }
            
        return results
