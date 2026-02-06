"""
FlowVision - Ward Analytics Module
Generates ward-level insights and analytics
"""

import pandas as pd
import numpy as np
import os

class WardAnalytics:
    """Ward-level analytics and insights generation"""
    
    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir
        self.ward_data = None
        self.consumption_data = None
        
    def load_data(self):
        """Load ward and consumption data"""
        ward_path = os.path.join(self.data_dir, 'ward_usage.csv')
        consumption_path = os.path.join(self.data_dir, 'consumption_daily.csv')
        
        self.ward_data = pd.read_csv(ward_path)
        self.consumption_data = pd.read_csv(consumption_path, parse_dates=['date'])
        
        return self.ward_data, self.consumption_data
    
    def calculate_ward_statistics(self):
        """Calculate comprehensive ward statistics"""
        stats = []
        
        if self.ward_data is None or self.consumption_data is None:
            return []
            
        for ward_id in self.ward_data['ward_id']:
            ward_info = self.ward_data[self.ward_data['ward_id'] == ward_id].iloc[0]
            ward_consumption = self.consumption_data[
                self.consumption_data['ward_id'] == ward_id
            ]
            
            # Calculate statistics
            avg_consumption = ward_consumption['consumption_m3'].mean()
            std_consumption = ward_consumption['consumption_m3'].std()
            max_consumption = ward_consumption['consumption_m3'].max()
            min_consumption = ward_consumption['consumption_m3'].min()
            
            # Trend (last 30 days vs previous 30 days)
            recent = ward_consumption.tail(30)['consumption_m3'].mean()
            previous = ward_consumption.iloc[-60:-30]['consumption_m3'].mean() if len(ward_consumption) >= 60 else recent
            trend = ((recent - previous) / previous * 100) if previous > 0 else 0
            
            stats.append({
                'ward_id': ward_id,
                'ward_name': ward_info['ward_name'],
                'population': ward_info['population'],
                'avg_consumption': avg_consumption,
                'std_consumption': std_consumption,
                'max_consumption': max_consumption,
                'min_consumption': min_consumption,
                'consumption_trend_pct': trend,
                'leak_risk_score': ward_info['leak_risk_score'],
                'infrastructure_age': ward_info['infrastructure_age_years']
            })
        
        return pd.DataFrame(stats)
    
    def identify_high_risk_wards(self, threshold=70):
        """Identify wards with high leak risk"""
        high_risk = self.ward_data[
            self.ward_data['leak_risk_score'] > threshold
        ].sort_values('leak_risk_score', ascending=False)
        
        return high_risk[['ward_id', 'ward_name', 'leak_risk_score', 'infrastructure_age_years']]
    
    def identify_abnormal_consumption(self, z_threshold=2):
        """Identify wards with abnormal consumption patterns"""
        stats = self.calculate_ward_statistics()
        
        # Calculate z-score for consumption
        mean_consumption = stats['avg_consumption'].mean()
        std_consumption = stats['avg_consumption'].std()
        
        stats['consumption_zscore'] = (
            (stats['avg_consumption'] - mean_consumption) / std_consumption
        )
        
        # Abnormal if |z-score| > threshold
        abnormal = stats[np.abs(stats['consumption_zscore']) > z_threshold]
        
        return abnormal[['ward_id', 'ward_name', 'avg_consumption', 'consumption_zscore']]
    
    def generate_insights(self):
        """Generate natural language insights"""
        insights = []
        
        # Load data if not already loaded
        if self.ward_data is None:
            self.load_data()
        
        stats = self.calculate_ward_statistics()
        
        # Insight 1: High risk wards
        high_risk = self.identify_high_risk_wards(threshold=70)
        if len(high_risk) > 0:
            ward_names = ', '.join(high_risk['ward_name'].head(3).tolist())
            insights.append({
                'type': 'warning',
                'category': 'leak_risk',
                'message': f"High leak risk detected in {ward_names} due to aging infrastructure",
                'severity': 'high',
                'wards': high_risk['ward_id'].tolist()
            })
        
        # Insight 2: Consumption trends
        increasing_wards = stats[stats['consumption_trend_pct'] > 10]
        if len(increasing_wards) > 0:
            ward_names = ', '.join(increasing_wards['ward_name'].head(3).tolist())
            avg_increase = increasing_wards['consumption_trend_pct'].mean()
            insights.append({
                'type': 'info',
                'category': 'consumption_trend',
                'message': f"{ward_names} showing {avg_increase:.1f}% increase in consumption over last 30 days",
                'severity': 'medium',
                'wards': increasing_wards['ward_id'].tolist()
            })
        
        # Insight 3: Abnormal consumption
        abnormal = self.identify_abnormal_consumption(z_threshold=2)
        if len(abnormal) > 0:
            for _, ward in abnormal.iterrows():
                direction = "higher" if ward['consumption_zscore'] > 0 else "lower"
                insights.append({
                    'type': 'warning',
                    'category': 'abnormal_consumption',
                    'message': f"{ward['ward_name']} shows abnormally {direction} consumption pattern",
                    'severity': 'medium',
                    'wards': [ward['ward_id']]
                })
        
        # Insight 4: Peak consumption ward
        max_ward = stats.loc[stats['avg_consumption'].idxmax()]
        insights.append({
            'type': 'info',
            'category': 'peak_consumption',
            'message': f"{max_ward['ward_name']} has highest average consumption at {max_ward['avg_consumption']:.2f} m³/day",
            'severity': 'low',
            'wards': [max_ward['ward_id']]
        })
        
        # Insight 5: Efficiency recommendation
        high_per_capita = stats.nlargest(3, 'avg_consumption')
        ward_names = ', '.join(high_per_capita['ward_name'].tolist())
        insights.append({
            'type': 'recommendation',
            'category': 'efficiency',
            'message': f"Consider water conservation programs in {ward_names} to reduce consumption",
            'severity': 'low',
            'wards': high_per_capita['ward_id'].tolist()
        })
        
        return insights
    
    def get_ward_comparison(self) -> dict:
        """Get comparative analytics across wards"""
        stats = self.calculate_ward_statistics()
        
        comparison = {
            'total_wards': len(stats),
            'highest_consumption': {
                'ward': stats.loc[stats['avg_consumption'].idxmax(), 'ward_name'],
                'value': stats['avg_consumption'].max()
            },
            'lowest_consumption': {
                'ward': stats.loc[stats['avg_consumption'].idxmin(), 'ward_name'],
                'value': stats['avg_consumption'].min()
            },
            'highest_risk': {
                'ward': stats.loc[stats['leak_risk_score'].idxmax(), 'ward_name'],
                'score': stats['leak_risk_score'].max()
            },
            'average_consumption': stats['avg_consumption'].mean(),
            'total_population': stats['population'].sum()
        }
        
        return comparison

def main():
    """Demo ward analytics"""
    print("="*60)
    print("FlowVision - Ward Analytics Demo")
    print("="*60)
    
    # Initialize analytics
    analytics = WardAnalytics()
    analytics.load_data()
    
    # Calculate statistics
    print("\nCalculating ward statistics...")
    stats = analytics.calculate_ward_statistics()
    print(stats[['ward_name', 'avg_consumption', 'leak_risk_score', 'consumption_trend_pct']])
    
    # Generate insights
    print("\n" + "="*60)
    print("AI-Generated Insights")
    print("="*60)
    
    insights = analytics.generate_insights()
    for i, insight in enumerate(insights, 1):
        icon = "[WARNING]" if insight['type'] == 'warning' else "[INFO]" if insight['type'] == 'info' else "[TIP]"
        print(f"\n{i}. {icon} [{insight['category'].upper()}]")
        print(f"   {insight['message']}")
        print(f"   Severity: {insight['severity']}")
    
    # Ward comparison
    print("\n" + "="*60)
    print("Ward Comparison")
    print("="*60)
    
    comparison = analytics.get_ward_comparison()
    print(f"\nTotal Wards: {comparison['total_wards']}")
    print(f"Total Population: {comparison['total_population']:,}")
    print(f"Average Consumption: {comparison['average_consumption']:.2f} m³/day")
    print(f"\nHighest Consumption: {comparison['highest_consumption']['ward']} ({comparison['highest_consumption']['value']:.2f} m³/day)")
    print(f"Lowest Consumption: {comparison['lowest_consumption']['ward']} ({comparison['lowest_consumption']['value']:.2f} m³/day)")
    print(f"Highest Risk: {comparison['highest_risk']['ward']} (Score: {comparison['highest_risk']['score']:.2f})")

if __name__ == "__main__":
    main()
