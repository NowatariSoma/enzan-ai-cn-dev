#!/usr/bin/env python3
"""
修正されたAPIのテスト
"""

import requests
import json
import pandas as pd

def test_fixed_api():
    """修正されたAPIをテスト"""
    
    print("=== 修正されたAPI（ポート8001）テスト ===")
    
    # テストパラメータ
    test_params = {
        "folder_name": "01-hokkaido-akan",
        "ameasure_file": "measurements_A_00004.csv", 
        "distance_from_face": 1.0,
        "daily_advance": 5.0,
        "max_distance_from_face": 200.0
    }
    
    print(f"テストパラメータ: {test_params}")
    
    try:
        # 修正されたAPI（ポート8000）を呼び出し
        api_url = "http://localhost:8000/api/v1/simulation/local-displacement"
        response = requests.post(api_url, json=test_params, timeout=30)
        
        print(f"レスポンスステータス: {response.status_code}")
        
        if response.status_code != 200:
            print(f"APIエラー: {response.text}")
            return False
        
        api_result_raw = response.json()
        api_data = pd.DataFrame(api_result_raw["simulation_data"])
        
        print(f"API結果: {len(api_data)}行")
        print("API結果 (最初の5行):")
        print(api_data.head().to_string())
        
        # Streamlitの期待値と比較
        print("\n=== 期待値との比較 ===")
        print("期待値（Streamlitの結果）:")
        print("1.3125m: 変位量A=0.568, 変位量B=0.489, 沈下量1=-14.475")
        
        if len(api_data) > 0:
            first_row = api_data.iloc[0]
            print(f"API結果:")
            print(f"1.3125m: 変位量A={first_row['変位量A_prediction']:.3f}, 変位量B={first_row['変位量B_prediction']:.3f}, 沈下量1={first_row['沈下量1_prediction']:.3f}")
            
            # 許容誤差での比較
            expected_A = 0.568
            expected_B = 0.489  
            expected_settlement = -14.475
            
            actual_A = first_row['変位量A_prediction']
            actual_B = first_row['変位量B_prediction']
            actual_settlement = first_row['沈下量1_prediction']
            
            tolerance = 0.1  # 10%の許容誤差
            
            a_match = abs(actual_A - expected_A) < tolerance
            b_match = abs(actual_B - expected_B) < tolerance
            settlement_match = abs(actual_settlement - expected_settlement) < tolerance
            
            print(f"\n比較結果:")
            print(f"変位量A: {'✅' if a_match else '❌'} (期待値: {expected_A}, 実際: {actual_A:.3f})")
            print(f"変位量B: {'✅' if b_match else '❌'} (期待値: {expected_B}, 実際: {actual_B:.3f})")
            print(f"沈下量1: {'✅' if settlement_match else '❌'} (期待値: {expected_settlement}, 実際: {actual_settlement:.3f})")
            
            if a_match and b_match and settlement_match:
                print("\n🎉 成功! APIがStreamlitの結果と一致しました!")
                return True
            else:
                print("\n❌ まだ一致していません")
                return False
        
        return False
        
    except Exception as e:
        print(f"APIエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_api()
    if success:
        print("\n✅ テスト成功: APIがStreamlitの結果と一致します")
    else:
        print("\n❌ テスト失敗: まだ不一致があります")