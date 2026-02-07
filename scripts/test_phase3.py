
import requests
import json
import time
import traceback

BASE_URL = "http://localhost:8000/api/ml"

def test_optimization():
    print("\nTesting Optimization Endpoint...")
    try:
        response = requests.post(f"{BASE_URL}/optimize", timeout=5)
        if response.status_code == 200:
            print("✅ Optimization Success")
            data = response.json()
            print(json.dumps(data, indent=2))
        else:
            print(f"❌ Optimization Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        traceback.print_exc()

def test_control(ward_id):
    print(f"\nTesting Control Endpoint for Ward {ward_id}...")
    try:
        response = requests.get(f"{BASE_URL}/control/{ward_id}", timeout=5)
        if response.status_code == 200:
            print("✅ Control Success")
            data = response.json()
            print(json.dumps(data, indent=2))
        else:
            print(f"❌ Control Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Wait a bit if server is just starting
    time.sleep(2)
    test_optimization()
    test_control(1)
