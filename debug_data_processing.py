#!/usr/bin/env python3
"""
GUIとAPIのデータ処理過程比較調査
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev')
sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure')

# GUI関数をインポートするために作業ディレクトリを変更
original_cwd = os.getcwd()
os.chdir('/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure')

from gui_displacement_temporal_spacial_analysis import simulate_displacement

# 作業ディレクトリを戻す
os.chdir(original_cwd)

# API関数をインポート
import importlib.util
spec = importlib.util.spec_from_file_location("simulation", "/home/nowatari/repos/enzan-ai-cn-dev/microservices/ai_ameasure/app/api/endpoints/simulation.py")
simulation_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(simulation_module)
simulate_displacement_logic = simulation_module.simulate_displacement_logic

def debug_data_processing():
    """データ処理過程の詳細デバッグ"""
    print("=== データ処理過程比較調査 ===")
    
    # テストパラメータ
    test_params = {
        "folder_name": "01-hokkaido-akan",
        "ameasure_file": "measurements_A_00004.csv", 
        "distance_from_face": 1.0,
        "daily_advance": 5.0,
        "max_distance_from_face": 200.0
    }
    
    print(f"テストパラメータ: {test_params}")
    
    # 共通のパス設定
    INPUT_FOLDER = '/home/nowatari/repos/enzan-ai-cn-dev/data_folder'
    input_folder = os.path.join(INPUT_FOLDER, test_params["folder_name"], 'main_tunnel', 'CN_measurement_data')
    a_measure_path = os.path.join(INPUT_FOLDER, test_params["folder_name"], 'main_tunnel', 'CN_measurement_data', 'measurements_A', test_params["ameasure_file"])
    
    print(f"\n入力フォルダ: {input_folder}")
    print(f"測定ファイル: {a_measure_path}")
    
    # 1. GUI関数の結果
    print("\n1. GUI関数の結果を取得中...")
    try:
        df_all_gui_sim, settlements_gui, convergences_gui = simulate_displacement(
            input_folder, a_measure_path, test_params["max_distance_from_face"],
            test_params["daily_advance"], test_params["distance_from_face"], recursive=True
        )
        
        print(f"GUI settlements: {settlements_gui}")
        print(f"GUI convergences: {convergences_gui}")
        print(f"GUI DataFrame shape: {df_all_gui_sim.shape}")
        print(f"GUI DataFrame columns: {list(df_all_gui_sim.columns)}")
        
        # 最初の3行の詳細を表示
        print("\nGUI DataFrame 最初の3行:")
        for idx in range(min(3, len(df_all_gui_sim))):
            row = df_all_gui_sim.iloc[idx]
            print(f"  行{idx}: 切羽からの距離={row['切羽からの距離']:.4f}")
            for col in settlements_gui + convergences_gui:
                if f"{col}_prediction" in df_all_gui_sim.columns:
                    print(f"    {col}_prediction: {row[f'{col}_prediction']:.6f}")
        
    except Exception as e:
        print(f"GUI実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. API関数の結果  
    print("\n2. API関数の結果を取得中...")
    try:
        df_all_api_sim, settlements_api, convergences_api = simulate_displacement_logic(
            input_folder, a_measure_path, test_params["max_distance_from_face"],
            test_params["daily_advance"], test_params["distance_from_face"], recursive=True
        )
        
        print(f"API settlements: {settlements_api}")
        print(f"API convergences: {convergences_api}")
        print(f"API DataFrame shape: {df_all_api_sim.shape}")
        print(f"API DataFrame columns: {list(df_all_api_sim.columns)}")
        
        # 最初の3行の詳細を表示
        print("\nAPI DataFrame 最初の3行:")
        for idx in range(min(3, len(df_all_api_sim))):
            row = df_all_api_sim.iloc[idx]
            print(f"  行{idx}: 切羽からの距離={row['切羽からの距離']:.4f}")
            for col in settlements_api + convergences_api:
                if f"{col}_prediction" in df_all_api_sim.columns:
                    print(f"    {col}_prediction: {row[f'{col}_prediction']:.6f}")
        
    except Exception as e:
        print(f"API実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 入力データの比較
    print("\n3. 中間データの比較...")
    
    # prediction phaseのデータも比較
    print("\n予測フェーズの結果も比較:")
    
    try:
        df_all_gui_pred, _, _ = simulate_displacement(
            input_folder, a_measure_path, test_params["max_distance_from_face"]
        )
        
        df_all_api_pred, _, _ = simulate_displacement_logic(
            input_folder, a_measure_path, test_params["max_distance_from_face"]
        )
        
        print(f"GUI prediction shape: {df_all_gui_pred.shape}")
        print(f"API prediction shape: {df_all_api_pred.shape}")
        
        # 基本的な測定値を比較
        common_cols = ['切羽からの距離']
        if '変位量A' in df_all_gui_pred.columns and '変位量A' in df_all_api_pred.columns:
            common_cols.append('変位量A')
        if '沈下量1' in df_all_gui_pred.columns and '沈下量1' in df_all_api_pred.columns:
            common_cols.append('沈下量1')
            
        print(f"\n共通列での比較: {common_cols}")
        for col in common_cols:
            if col in df_all_gui_pred.columns and col in df_all_api_pred.columns:
                gui_vals = df_all_gui_pred[col][:5].values
                api_vals = df_all_api_pred[col][:5].values
                
                print(f"{col}:")
                print(f"  GUI: {gui_vals}")
                print(f"  API: {api_vals}")
                print(f"  同一: {np.allclose(gui_vals, api_vals, atol=1e-10)}")
        
    except Exception as e:
        print(f"予測フェーズ比較エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_data_processing()