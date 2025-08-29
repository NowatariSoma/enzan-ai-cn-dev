#!/usr/bin/env python3
"""
APIのデータ構造確認テスト - 実測、シミュレーション、予測データが全て含まれているかチェック
"""

import requests
import json
from pprint import pprint

def test_api_data_structure():
    """APIのデータ構造を詳細に確認"""
    
    print("=== APIデータ構造確認テスト ===")
    
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
        # APIを呼び出し
        api_url = "http://localhost:8000/api/v1/simulation/local-displacement"
        response = requests.post(api_url, json=test_params, timeout=30)
        
        print(f"レスポンスステータス: {response.status_code}")
        
        if response.status_code != 200:
            print(f"APIエラー: {response.text}")
            return False
        
        api_result = response.json()
        
        print("\n=== レスポンス構造 ===")
        print(f"レスポンスキー: {list(api_result.keys())}")
        
        # 各データセクションの詳細確認
        print("\n=== simulation_data (シミュレーションチャート用) ===")
        if "simulation_data" in api_result:
            sim_data = api_result["simulation_data"]
            print(f"データポイント数: {len(sim_data)}")
            if len(sim_data) > 0:
                print(f"カラム: {list(sim_data[0].keys())}")
                print("最初の3ポイント:")
                for i, point in enumerate(sim_data[:3]):
                    print(f"  [{i}] distance: {point.get('切羽からの距離', 'N/A')}")
                    # 実測データ確認
                    actual_cols = [k for k in point.keys() if not k.endswith('_prediction') and k != '切羽からの距離']
                    prediction_cols = [k for k in point.keys() if k.endswith('_prediction')]
                    print(f"      実測カラム: {actual_cols}")
                    print(f"      予測カラム: {prediction_cols}")
                    
                    # 実測値と予測値の例
                    if actual_cols:
                        print(f"      実測例 - {actual_cols[0]}: {point.get(actual_cols[0], 'N/A')}")
                    if prediction_cols:
                        print(f"      予測例 - {prediction_cols[0]}: {point.get(prediction_cols[0], 'N/A')}")
        
        print("\n=== prediction_data (予測チャート用) ===")
        if "prediction_data" in api_result:
            pred_data = api_result["prediction_data"]
            print(f"データポイント数: {len(pred_data)}")
            if len(pred_data) > 0:
                print(f"カラム: {list(pred_data[0].keys())}")
                print("最初の3ポイント:")
                for i, point in enumerate(pred_data[:3]):
                    print(f"  [{i}] distance: {point.get('切羽からの距離', 'N/A')}")
                    # 実測データ確認
                    actual_cols = [k for k in point.keys() if not k.endswith('_prediction') and k != '切羽からの距離']
                    prediction_cols = [k for k in point.keys() if k.endswith('_prediction')]
                    print(f"      実測カラム: {actual_cols}")
                    print(f"      予測カラム: {prediction_cols}")
        
        print("\n=== table_data (テーブル用) ===")
        if "table_data" in api_result:
            table_data = api_result["table_data"]
            print(f"データポイント数: {len(table_data)}")
            if len(table_data) > 0:
                print(f"カラム: {list(table_data[0].keys())}")
                print("最初の3ポイント:")
                for i, point in enumerate(table_data[:3]):
                    print(f"  [{i}] distance: {point.get('切羽からの距離', 'N/A')}")
                    # 予測データのみか確認
                    prediction_cols = [k for k in point.keys() if k.endswith('_prediction')]
                    other_cols = [k for k in point.keys() if not k.endswith('_prediction') and k != '切羽からの距離']
                    print(f"      予測カラム: {prediction_cols}")
                    print(f"      その他カラム: {other_cols}")
        else:
            print("table_dataが存在しません")
        
        print("\n=== データ内容分析 ===")
        
        # データ範囲の確認
        if "simulation_data" in api_result and len(api_result["simulation_data"]) > 0:
            sim_data = api_result["simulation_data"]
            distances = [p.get('切羽からの距離', 0) for p in sim_data]
            print(f"切羽からの距離範囲: {min(distances)} - {max(distances)}m")
            
            # 実測データと予測データの存在確認
            first_point = sim_data[0]
            has_actual = any(not k.endswith('_prediction') and k != '切羽からの距離' for k in first_point.keys())
            has_prediction = any(k.endswith('_prediction') for k in first_point.keys())
            
            print(f"実測データ含有: {'✅' if has_actual else '❌'}")
            print(f"予測データ含有: {'✅' if has_prediction else '❌'}")
            
            # 具体的なデータ値の確認
            print("\n実測データ例:")
            for k, v in first_point.items():
                if not k.endswith('_prediction') and k != '切羽からの距離':
                    print(f"  {k}: {v}")
            
            print("\n予測データ例:")
            for k, v in first_point.items():
                if k.endswith('_prediction'):
                    print(f"  {k}: {v}")
        
        return True
        
    except Exception as e:
        print(f"APIエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_api_data_structure()