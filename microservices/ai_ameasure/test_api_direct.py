#!/usr/bin/env python3
"""
APIのdisplacement_analysis エンドポイントを直接テスト
"""

import requests
import json

def test_api_analyze_whole():
    """analyze-whole エンドポイントをテスト"""
    url = "http://localhost:8000/api/v1/displacement-analysis/analyze-whole"
    
    payload = {
        "folder_name": "01-hokkaido-akan",
        "model_name": "Random Forest", 
        "max_distance_from_face": 100
        # td is optional, don't include it if None
    }
    
    try:
        print("APIエンドポイントをテスト中...")
        response = requests.post(url, json=payload, timeout=300)  # 5分タイムアウト
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"API Response Status: {data.get('status')}")
            print(f"API Response Message: {data.get('message')}")
            
            # training_metricsチェック
            training_metrics = data.get('training_metrics', {})
            print(f"Training metrics keys: {list(training_metrics.keys())}")
            
            # scatter_dataチェック  
            scatter_data = data.get('scatter_data', {})
            print(f"Scatter data keys: {list(scatter_data.keys())}")
            
            if scatter_data:
                train_actual = scatter_data.get('train_actual', [])
                train_predictions = scatter_data.get('train_predictions', [])
                validate_actual = scatter_data.get('validate_actual', [])  
                validate_predictions = scatter_data.get('validate_predictions', [])
                
                print(f"Train actual length: {len(train_actual)}")
                print(f"Train predictions length: {len(train_predictions)}")
                print(f"Validate actual length: {len(validate_actual)}")
                print(f"Validate predictions length: {len(validate_predictions)}")
                
                if train_actual:
                    print(f"First 5 train actual: {train_actual[:5]}")
                    print(f"First 5 train predictions: {train_predictions[:5]}")
                    print("✅ SUCCESS: API scatter_data is working correctly!")
                    return True
                else:
                    print("❌ FAIL: scatter_data arrays are empty")
                    return False
            else:
                print("❌ FAIL: scatter_data is empty or missing")
                return False
                
        else:
            print(f"❌ FAIL: HTTP {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ FAIL: Cannot connect to API. Make sure the API server is running on localhost:8000")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    print("Testing API displacement-analysis endpoint...")
    success = test_api_analyze_whole()
    exit(0 if success else 1)