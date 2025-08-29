#!/usr/bin/env python3
"""
シミュレーションデータの詳細確認 - 実測、シミュレーション、予測データの内容確認
"""

import requests
import json

def test_detailed_simulation():
    """シミュレーションデータの詳細確認"""
    
    print("=== 詳細シミュレーションテスト ===")
    
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
        
        result = response.json()
        
        print(f"=== データ分析 (distance_from_face={test_params['distance_from_face']}m) ===")
        
        # simulation_data の分析
        sim_data = result.get("simulation_data", [])
        pred_data = result.get("prediction_data", [])
        table_data = result.get("table_data", [])
        
        # 距離の範囲確認
        if sim_data:
            sim_distances = [p.get('切羽からの距離', 0) for p in sim_data]
            print(f"simulation_data 距離範囲: {min(sim_distances)} - {max(sim_distances)}m ({len(sim_data)}ポイント)")
            
            # 1.0m以下のデータがあるか確認
            below_threshold = [d for d in sim_distances if d <= test_params['distance_from_face']]
            above_threshold = [d for d in sim_distances if d > test_params['distance_from_face']]
            print(f"  - ≤{test_params['distance_from_face']}m (実測想定): {len(below_threshold)}ポイント {below_threshold[:5]}")
            print(f"  - >{test_params['distance_from_face']}m (シミュレーション想定): {len(above_threshold)}ポイント {above_threshold[:5]}")
            
            # 実測データが変化しているか確認
            if sim_data:
                first_point = sim_data[0]
                last_point = sim_data[-1] if len(sim_data) > 1 else first_point
                
                print(f"\n実測データの変化確認:")
                for col in ['沈下量1', '変位量A', '変位量B']:
                    if col in first_point:
                        first_val = first_point[col]
                        last_val = last_point[col]
                        print(f"  - {col}: 開始={first_val}, 終了={last_val}, 変化={'あり' if first_val != last_val else 'なし'}")
                
                print(f"\n予測データの変化確認:")
                for col in ['沈下量1_prediction', '変位量A_prediction', '変位量B_prediction']:
                    if col in first_point:
                        first_val = first_point[col]
                        last_val = last_point[col]
                        print(f"  - {col}: 開始={first_val:.3f}, 終了={last_val:.3f}, 変化={'あり' if abs(first_val - last_val) > 0.001 else 'なし'}")
        
        if pred_data:
            pred_distances = [p.get('切羽からの距離', 0) for p in pred_data]
            print(f"\nprediction_data 距離範囲: {min(pred_distances)} - {max(pred_distances)}m ({len(pred_data)}ポイント)")
        
        if table_data:
            table_distances = [p.get('切羽からの距離', 0) for p in table_data]
            print(f"table_data 距離範囲: {min(table_distances)} - {max(table_distances)}m ({len(table_data)}ポイント)")
            
        # 期待される動作:
        # 1. simulation_data: distance_from_face以下は実測値、以降はシミュレーション値
        # 2. prediction_data: 全範囲で実測値+予測値
        # 3. table_data: 予測値のみ
        
        print(f"\n=== 期待動作との比較 ===")
        print(f"✅ simulation_data: {len(below_threshold) if sim_data else 0}個の実測ポイント + {len(above_threshold) if sim_data else 0}個のシミュレーションポイント")
        print(f"✅ prediction_data: {len(pred_data)}個の予測ポイント")  
        print(f"✅ table_data: {len(table_data)}個のテーブルポイント")
        
        # 実測データ不足の問題があるか確認
        if sim_data and not below_threshold:
            print(f"❌ 問題: simulation_dataに実測データ（≤{test_params['distance_from_face']}m）が含まれていません")
            return False
        
        return True
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_detailed_simulation()