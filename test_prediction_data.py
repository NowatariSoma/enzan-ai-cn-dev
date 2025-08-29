#!/usr/bin/env python3
"""
新しいprediction_dataをテスト
"""

import requests
import json
import pandas as pd

def test_prediction_data():
    """新しいprediction_dataが正しく返されるかテスト"""
    
    print("=== Prediction Data テスト ===")
    
    # テストパラメータ
    test_params = {
        "folder_name": "01-hokkaido-akan",
        "ameasure_file": "measurements_A_00004.csv", 
        "distance_from_face": 1.0,
        "daily_advance": 5.0,
        "max_distance_from_face": 200.0
    }
    
    try:
        # APIを呼び出し
        api_url = "http://localhost:8000/api/v1/simulation/local-displacement"
        response = requests.post(api_url, json=test_params, timeout=30)
        
        if response.status_code != 200:
            print(f"APIエラー: {response.text}")
            return False
        
        api_result_raw = response.json()
        
        # prediction_dataが含まれているか確認
        print(f"prediction_data が含まれているか: {'prediction_data' in api_result_raw}")
        
        if 'prediction_data' in api_result_raw:
            prediction_data = api_result_raw['prediction_data']
            prediction_df = pd.DataFrame(prediction_data)
            
            print(f"prediction_data 行数: {len(prediction_df)}")
            print(f"prediction_data 列: {list(prediction_df.columns)}")
            
            print("\nprediction_data 最初の3行:")
            print(prediction_df.head(3).to_string())
            
            # 実際の測定データ（沈下量1, 変位量A など）が含まれているか確認
            actual_measurement_cols = [col for col in prediction_df.columns if not col.endswith('_prediction') and col != '切羽からの距離']
            prediction_cols = [col for col in prediction_df.columns if col.endswith('_prediction')]
            
            print(f"\n実際の測定列: {actual_measurement_cols}")
            print(f"予測列: {prediction_cols}")
            
            if len(actual_measurement_cols) > 0 and len(prediction_cols) > 0:
                print("✅ 実際の測定データと予測データの両方が含まれています")
                
                # 最初の行のサンプル値を確認
                if len(prediction_df) > 0:
                    first_row = prediction_df.iloc[0]
                    print(f"\n最初の行のサンプル:")
                    print(f"切羽からの距離: {first_row['切羽からの距離']}")
                    if '沈下量1' in first_row:
                        print(f"沈下量1（実測値）: {first_row['沈下量1']}")
                    if '沈下量1_prediction' in first_row:
                        print(f"沈下量1_prediction（予測値）: {first_row['沈下量1_prediction']}")
                    if '変位量A' in first_row:
                        print(f"変位量A（実測値）: {first_row['変位量A']}")
                    if '変位量A_prediction' in first_row:
                        print(f"変位量A_prediction（予測値）: {first_row['変位量A_prediction']}")
                
                return True
            else:
                print("❌ 実際の測定データまたは予測データが不足しています")
                return False
        else:
            print("❌ prediction_data がAPIレスポンスに含まれていません")
            return False
        
    except Exception as e:
        print(f"APIエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_prediction_data()
    if success:
        print("\n✅ テスト成功: prediction_dataが正しく返されています")
    else:
        print("\n❌ テスト失敗: prediction_dataに問題があります")