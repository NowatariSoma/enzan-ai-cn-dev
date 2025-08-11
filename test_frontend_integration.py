#!/usr/bin/env python3
"""
フロントエンドとバックエンドの統合テスト
LearningDashboardでprocess-eachエンドポイントを使用できることを確認
"""

import requests
import json
import time

# APIベースURL
BASE_URL = "http://localhost:8000/api/v1"

def test_process_each_for_frontend():
    """
    フロントエンドが使用する形式でprocess-eachエンドポイントをテスト
    """
    print("=" * 50)
    print("Frontend Integration Test")
    print("=" * 50)
    
    test_cases = [
        {
            "model_name": "Random Forest",
            "folder_name": "01-hokkaido-akan",
            "max_distance_from_face": 100.0,
            "data_type": "settlement",
            "td": 500,
            "predict_final": True,
        },
        {
            "model_name": "Random Forest",
            "folder_name": "01-hokkaido-akan",
            "max_distance_from_face": 100.0,
            "data_type": "convergence",
            "td": 500,
            "predict_final": True,
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['model_name']} - {test_case['data_type']}")
        print("-" * 30)
        
        # process-eachエンドポイントを呼び出し
        response = requests.post(
            f"{BASE_URL}/models/process-each",
            json=test_case,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # 必要なデータが含まれているか確認
            print(f"✓ Model: {data.get('model_name')}")
            print(f"✓ Data Type: {data.get('data_type')}")
            
            metrics = data.get('metrics', {})
            print(f"✓ Train MSE: {metrics.get('mse_train', 0):.2f}")
            print(f"✓ Train R²: {metrics.get('r2_train', 0):.2f}")
            print(f"✓ Validation MSE: {metrics.get('mse_validate', 0):.2f}")
            print(f"✓ Validation R²: {metrics.get('r2_validate', 0):.2f}")
            
            # 散布図データの確認
            train_predictions = data.get('train_predictions', [])
            train_actual = data.get('train_actual', [])
            validate_predictions = data.get('validate_predictions', [])
            validate_actual = data.get('validate_actual', [])
            
            print(f"✓ Train data points: {len(train_predictions)}")
            print(f"✓ Validation data points: {len(validate_predictions)}")
            
            # 特徴量重要度の確認
            feature_importance = data.get('feature_importance', {})
            if feature_importance:
                top_features = sorted(
                    feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]
                print("✓ Top 3 features:")
                for feature, importance in top_features:
                    print(f"  - {feature}: {importance:.4f}")
            
            # フロントエンドが必要とする形式に変換可能か確認
            scatter_data = []
            for j in range(min(5, len(train_actual))):
                scatter_data.append({
                    "actual": train_actual[j],
                    "predicted": train_predictions[j]
                })
            
            print(f"✓ Sample scatter data (first 5 points):")
            for j, point in enumerate(scatter_data):
                print(f"  {j+1}. Actual: {point['actual']:.2f}, Predicted: {point['predicted']:.2f}")
            
            print(f"\n✅ Test Case {i} PASSED")
        else:
            print(f"❌ Test Case {i} FAILED - Status Code: {response.status_code}")
            print(f"Error: {response.text}")
        
        time.sleep(1)  # APIに負荷をかけないように待機
    
    print("\n" + "=" * 50)
    print("Integration Test Complete")
    print("=" * 50)
    print("\nFrontend can now use the following flow:")
    print("1. User selects model, folder, data type in UI")
    print("2. Frontend calls POST /api/v1/models/process-each")
    print("3. Response contains:")
    print("   - train_actual & train_predictions for train scatter plot")
    print("   - validate_actual & validate_predictions for validation scatter plot")
    print("   - metrics.mse_train & metrics.r2_train for train metrics")
    print("   - metrics.mse_validate & metrics.r2_validate for validation metrics")
    print("   - feature_importance for feature importance chart")

if __name__ == "__main__":
    test_process_each_for_frontend()