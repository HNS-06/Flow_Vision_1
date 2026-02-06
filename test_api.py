"""
Quick test script to verify all API endpoints
"""
import requests
import json

BASE_URL = "http://localhost:8000"

print("="*60)
print("FlowVision API Endpoint Test")
print("="*60)

# Test 1: Health check
print("\n1. Testing /health...")
try:
    r = requests.get(f"{BASE_URL}/health", timeout=2)
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.json()}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 2: Ward data
print("\n2. Testing /api/data/wards...")
try:
    r = requests.get(f"{BASE_URL}/api/data/wards", timeout=2)
    print(f"   Status: {r.status_code}")
    data = r.json()
    print(f"   Success: {data.get('success')}")
    print(f"   Wards: {len(data.get('data', []))}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 3: ML Insights
print("\n3. Testing /api/ml/insights...")
try:
    r = requests.get(f"{BASE_URL}/api/ml/insights", timeout=2)
    print(f"   Status: {r.status_code}")
    data = r.json()
    print(f"   Success: {data.get('success')}")
    if data.get('insights'):
        print(f"   Insights keys: {list(data['insights'].keys())}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 4: Forecast
print("\n4. Testing /api/ml/forecast...")
try:
    r = requests.post(f"{BASE_URL}/api/ml/forecast", 
                     json={"steps": 24}, 
                     headers={"Content-Type": "application/json"},
                     timeout=5)
    print(f"   Status: {r.status_code}")
    data = r.json()
    print(f"   Success: {data.get('success')}")
    print(f"   Forecast points: {len(data.get('forecast', []))}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 5: Start simulation
print("\n5. Testing /api/simulation/start...")
try:
    r = requests.post(f"{BASE_URL}/api/simulation/start", timeout=2)
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.json()}")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "="*60)
print("Test Complete")
print("="*60)
