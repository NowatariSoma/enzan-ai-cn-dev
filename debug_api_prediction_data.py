#!/usr/bin/env python3
"""
APIのPrediction phase のデータをデバッグ
"""

import os
import sys
import json
import pandas as pd
import requests

def debug_api_prediction_data():
    """APIのPrediction phaseのデータを詳しく確認"""
    
    print("=== APIのPrediction Dataデバッグ ===")
    
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
        # APIを呼び出してログを確認
        api_url = "http://localhost:8000/api/v1/simulation/local-displacement"
        response = requests.post(api_url, json=test_params, timeout=30)
        
        if response.status_code != 200:
            print(f"APIエラー: {response.text}")
            return False
        
        api_result_raw = response.json()
        
        # チャートファイルのパスを確認
        prediction_charts = api_result_raw.get("prediction_charts", {})
        simulation_charts = api_result_raw.get("simulation_charts", {})
        
        print(f"\n=== チャートファイルパス ===")
        print(f"Settlement prediction: {prediction_charts.get('settlement')}")
        print(f"Convergence prediction: {prediction_charts.get('convergence')}")
        print(f"Settlement simulation: {simulation_charts.get('settlement')}")
        print(f"Convergence simulation: {simulation_charts.get('convergence')}")
        
        # simulation_dataを確認
        simulation_data = api_result_raw["simulation_data"]
        simulation_df = pd.DataFrame(simulation_data)
        
        print(f"\n=== Simulation Data（テーブル用データ）===")
        print(f"行数: {len(simulation_df)}")
        print("最初の3行:")
        print(simulation_df.head(3).to_string())
        
        # 期待値と比較
        if len(simulation_df) > 0:
            first_row = simulation_df.iloc[0]
            print(f"\n最初の行のデータ:")
            print(f"切羽からの距離: {first_row['切羽からの距離']}")
            print(f"変位量A_prediction: {first_row['変位量A_prediction']}")
            print(f"変位量B_prediction: {first_row['変位量B_prediction']}")
            print(f"沈下量1_prediction: {first_row['沈下量1_prediction']}")
        
        return True
        
    except Exception as e:
        print(f"APIエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_streamlit_directly():
    """Streamlitのsimulate_displacement関数を直接実行してデータを確認"""
    
    print("\n=== Streamlit関数直接実行 ===")
    
    # プロジェクトルートをパスに追加
    sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev')
    sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure')

    # 作業ディレクトリを変更
    original_cwd = os.getcwd()
    os.chdir('/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure')
    
    try:
        from gui_displacement_temporal_spacial_analysis import simulate_displacement
        from app.displacement_temporal_spacial_analysis import DISTANCE_FROM_FACE, TD_NO
        
        # GUI用のパス設定
        INPUT_FOLDER = '/home/nowatari/repos/enzan-ai-cn-dev/data_folder'
        folder_name = "01-hokkaido-akan"
        ameasure_file = "measurements_A_00004.csv"
        max_distance_from_face = 200.0
        
        input_folder = os.path.join(INPUT_FOLDER, folder_name, 'main_tunnel', 'CN_measurement_data')
        a_measure_path = os.path.join(INPUT_FOLDER, folder_name, 'main_tunnel', 'CN_measurement_data', 'measurements_A', ameasure_file)
        
        print(f"input_folder: {input_folder}")
        print(f"a_measure_path: {a_measure_path}")
        
        # prediction phase (recursive=False)
        print("\n--- Prediction Phase ---")
        df_all_pred, settlements, convergences = simulate_displacement(
            input_folder, a_measure_path, max_distance_from_face
        )
        
        print(f"df_all_pred shape: {df_all_pred.shape}")
        print(f"settlements: {settlements}")
        print(f"convergences: {convergences}")
        
        print(f"df_all_pred columns: {list(df_all_pred.columns)}")
        
        # 実際の測定値を確認
        print(f"\n--- 実際の測定値（最初の5行）---")
        print(f"切羽からの距離: {df_all_pred[DISTANCE_FROM_FACE].head().values}")
        if settlements:
            for s in settlements:
                if s in df_all_pred.columns:
                    print(f"{s}: {df_all_pred[s].head().values}")
        if convergences:
            for c in convergences:
                if c in df_all_pred.columns:
                    print(f"{c}: {df_all_pred[c].head().values}")
        
        # 予測値を確認
        print(f"\n--- 予測値（最初の5行）---")
        pred_cols = [col for col in df_all_pred.columns if col.endswith('_prediction')]
        print(f"予測列: {pred_cols}")
        
        for col in pred_cols:
            if col in df_all_pred.columns:
                print(f"{col}: {df_all_pred[col].head().values}")
        
        return df_all_pred, settlements, convergences
        
    except Exception as e:
        print(f"Streamlit関数実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    finally:
        # 作業ディレクトリを戻す
        os.chdir(original_cwd)

if __name__ == "__main__":
    # APIのデータを確認
    debug_api_prediction_data()
    
    # Streamlit関数を直接実行してデータを確認
    debug_streamlit_directly()