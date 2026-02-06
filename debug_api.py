import urllib.request
import urllib.parse
import json
import sys

BASE_URL = "http://localhost:8000"

def test_endpoint(name, url, method="GET", body=None):
    print(f"\n--- Testing {name} ---")
    try:
        req = urllib.request.Request(url, method=method)
        req.add_header('Content-Type', 'application/json')
        
        if body:
            data = json.dumps(body).encode('utf-8')
            req.data = data
            
        with urllib.request.urlopen(req) as response:
            status = response.status
            content = response.read().decode('utf-8')
            
            print(f"Status: {status}")
            if status == 200:
                data = json.loads(content)
                print("Response fragments:")
                print(json.dumps(data, indent=2)[:500] + "...")
                
                if name == "Forecast":
                    print("\nForecast Keys:", data.keys())
                    if 'forecast' in data:
                        print("Forecast object keys:", data['forecast'].keys())
                        if 'forecast' in data['forecast']:
                            print("Nested forecast keys:", data['forecast']['forecast'].keys())
                
                if name == "Insights":
                    print("\nInsights Keys:", data.keys())
                    if 'insights' in data:
                        print(f"Number of insights: {len(data['insights'])}")
            else:
                print("Error:", content)

    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
        print(e.read().decode('utf-8'))
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_endpoint("Forecast", f"{BASE_URL}/api/ml/forecast", "POST", {"steps": 24})
    test_endpoint("Insights", f"{BASE_URL}/api/ml/insights")
