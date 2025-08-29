#!/usr/bin/env python3
"""
フロントエンド（API）とStreamlit（GUI）の結果差異をデバッグ
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import requests

# プロジェクトルートをパスに追加
sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev')
sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure')

# GUI関数をインポートするために作業ディレクトリを変更
original_cwd = os.getcwd()
os.chdir('/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure')

from gui_displacement_temporal_spacial_analysis import simulate_displacement

# 作業ディレクトリを戻す
os.chdir(original_cwd)

def compare_frontend_streamlit():
    """フロントエンド（API経由）とStreamlit（GUI直接）の結果を詳しく比較"""
    
    print("=== フロントエンド vs Streamlit 詳細比較 ===")
    
    # テストパラメータ
    test_params = {
        "folder_name": "01-hokkaido-akan",
        "ameasure_file": "measurements_A_00004.csv", 
        "distance_from_face": 1.0,
        "daily_advance": 5.0,
        "max_distance_from_face": 200.0
    }
    
    print(f"テストパラメータ: {test_params}")
    
    # 1. API（フロントエンドと同じルート）を呼び出し
    print("\n=== 1. API呼び出し（フロントエンドと同じルート）===")
    try:
        api_url = "http://localhost:8000/api/v1/simulation/local-displacement"
        response = requests.post(api_url, json=test_params, timeout=30)
        
        if response.status_code != 200:
            print(f"APIエラー: {response.status_code} - {response.text}")
            return False
        
        api_result_raw = response.json()
        api_data = pd.DataFrame(api_result_raw["simulation_data"])
        
        print(f"API結果: {len(api_data)}行")
        print("API結果 (最初の5行):")
        print(api_data.head().to_string())
        
    except Exception as e:
        print(f"APIエラー: {e}")
        return False
    
    # 2. Streamlit GUI関数を直接実行
    print("\n=== 2. Streamlit GUI関数直接実行 ===")
    try:
        # GUI用のパス設定
        INPUT_FOLDER = '/home/nowatari/repos/enzan-ai-cn-dev/data_folder'
        input_folder = os.path.join(INPUT_FOLDER, test_params["folder_name"], 'main_tunnel', 'CN_measurement_data')
        a_measure_path = os.path.join(INPUT_FOLDER, test_params["folder_name"], 'main_tunnel', 'CN_measurement_data', 'measurements_A', test_params["ameasure_file"])
        
        print(f"input_folder: {input_folder}")
        print(f"a_measure_path: {a_measure_path}")
        
        # 作業ディレクトリを変更
        os.chdir('/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure')
        
        # prediction phase
        df_all_gui_pred, settlements, convergences = simulate_displacement(
            input_folder, a_measure_path, test_params["max_distance_from_face"]
        )
        
        # simulation phase  
        df_all_gui_sim, _, _ = simulate_displacement(
            input_folder, a_measure_path, test_params["max_distance_from_face"],
            test_params["daily_advance"], test_params["distance_from_face"], recursive=True
        )
        
        # 元の作業ディレクトリに戻す
        os.chdir(original_cwd)
        
        # prediction列を抽出
        gui_prediction_cols = [col for col in df_all_gui_sim.columns if col.endswith('_prediction')]
        gui_result = df_all_gui_sim[['切羽からの距離'] + gui_prediction_cols].copy()
        
        print(f"GUI結果: {len(gui_result)}行")
        print("GUI結果 (最初の5行):")
        print(gui_result.head().to_string())
        
    except Exception as e:
        print(f"GUIエラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 詳細比較
    print("\n=== 3. 詳細比較 ===")
    
    # データの行数確認
    print(f"行数: API={len(api_data)}, GUI={len(gui_result)}")
    
    # 列名の確認
    api_cols = set(api_data.columns)
    gui_cols = set(gui_result.columns) 
    
    print(f"API列名: {sorted(api_cols)}")
    print(f"GUI列名: {sorted(gui_cols)}")
    
    common_cols = api_cols.intersection(gui_cols)
    print(f"共通列: {sorted(common_cols)}")
    
    if not common_cols:
        print("❌ 共通列がありません")
        return False
    
    # 数値の詳細比較（最初の10行）
    print("\n--- 数値比較（最初の10行） ---")
    
    for i in range(min(10, len(api_data), len(gui_result))):
        print(f"\n行 {i}:")
        print(f"  切羽からの距離: API={api_data.iloc[i]['切羽からの距離']:.4f}, GUI={gui_result.iloc[i]['切羽からの距離']:.4f}")
        
        for col in sorted(common_cols):
            if col != '切羽からの距離':
                api_val = api_data.iloc[i][col]
                gui_val = gui_result.iloc[i][col] 
                diff = abs(api_val - gui_val)
                status = "✅" if diff < 1e-6 else "❌"
                print(f"  {col}: API={api_val:.6f}, GUI={gui_val:.6f}, diff={diff:.6f} {status}")
    
    # 統計サマリー
    print("\n--- 統計サマリー ---")
    
    for col in sorted(common_cols):
        api_vals = api_data[col].values
        gui_vals = gui_result[col].values
        
        if len(api_vals) == len(gui_vals):
            diff_vals = np.abs(api_vals - gui_vals)
            max_diff = np.max(diff_vals)
            mean_diff = np.mean(diff_vals)
            identical = np.allclose(api_vals, gui_vals, atol=1e-6)
            
            status = "✅" if identical else "❌"
            print(f"  {col}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} {status}")
        else:
            print(f"  {col}: 長さが異なる API={len(api_vals)}, GUI={len(gui_vals)} ❌")

if __name__ == "__main__":
    compare_frontend_streamlit()