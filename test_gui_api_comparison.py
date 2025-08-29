#!/usr/bin/env python3
"""
GUIアプリケーションとAPIの出力結果同一性テスト
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import requests
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev')
sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure')

# GUI関数をインポートするために作業ディレクトリを変更
import os
original_cwd = os.getcwd()
os.chdir('/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure')

# GUI関数をインポート
from gui_displacement_temporal_spacial_analysis import simulate_displacement
from app.displacement_temporal_spacial_analysis import generate_additional_info_df, generate_dataframes, create_dataset, STA

# 作業ディレクトリを戻す
os.chdir(original_cwd)

def debug_data_processing(test_params):
    """中間データ処理をデバッグ"""
    print("\n=== データ処理デバッグ ===")
    
    # GUI用のパス設定
    INPUT_FOLDER = '/home/nowatari/repos/enzan-ai-cn-dev/data_folder'
    input_folder = os.path.join(INPUT_FOLDER, test_params["folder_name"], 'main_tunnel', 'CN_measurement_data')
    a_measure_path = os.path.join(INPUT_FOLDER, test_params["folder_name"], 'main_tunnel', 'CN_measurement_data', 'measurements_A', test_params["ameasure_file"])
    
    print(f"input_folder: {input_folder}")
    print(f"a_measure_path: {a_measure_path}")
    
    # 1. additional_info データを確認
    print("\n--- 1. Additional Info DataFrame ---")
    cycle_support_csv = os.path.join(input_folder, 'cycle_support/cycle_support.csv')
    observation_of_face_csv = os.path.join(input_folder, 'observation_of_face/observation_of_face.csv')
    
    df_additional_info = generate_additional_info_df(cycle_support_csv, observation_of_face_csv)
    print(f"Original df_additional_info shape: {df_additional_info.shape}")
    df_additional_info.drop(columns=[STA], inplace=True)
    print(f"After drop STA shape: {df_additional_info.shape}")
    
    # 2. generate_dataframes の出力を確認
    print("\n--- 2. Generate Dataframes ---")
    df_all, _, _, _, settlements, convergences = generate_dataframes([a_measure_path], test_params["max_distance_from_face"])
    print(f"df_all shape: {df_all.shape}")
    print(f"settlements: {settlements}")
    print(f"convergences: {convergences}")
    
    # 3. create_dataset の出力を確認
    print("\n--- 3. Create Dataset ---")
    settlement_data, convergence_data = create_dataset(df_all, df_additional_info)
    
    if isinstance(settlement_data, tuple) and len(settlement_data) >= 3:
        settlement_df, settlement_x_cols, settlement_y_col = settlement_data
        print(f"settlement_df shape: {settlement_df.shape}")
        print(f"settlement_x_cols: {settlement_x_cols}")
        print(f"settlement_df first few rows unique position_ids: {settlement_df['position_id'].head(10).values}")
        
        # 具体的な予測値の計算過程を確認
        import joblib
        output_folder = "./output"
        final_model_path = os.path.join(output_folder, "model_final_settlement.pkl")
        if os.path.exists(final_model_path):
            final_model = joblib.load(final_model_path)
            y_hat = final_model.predict(settlement_df[settlement_x_cols])
            print(f"settlement y_hat first 5 values: {y_hat[:5]}")
            print(f"settlement df_all values for settlements first position: {df_all[settlements[0]].head().values}")
    
    if isinstance(convergence_data, tuple) and len(convergence_data) >= 3:
        convergence_df, convergence_x_cols, convergence_y_col = convergence_data
        print(f"convergence_df shape: {convergence_df.shape}")
        print(f"convergence_x_cols: {convergence_x_cols}")
        print(f"convergence_df first few rows unique position_ids: {convergence_df['position_id'].head(10).values}")

def test_gui_api_comparison():
    """GUIとAPIの出力結果を比較"""
    print("=== GUIとAPIの出力結果同一性テスト ===")
    
    # テストパラメータ
    test_params = {
        "folder_name": "01-hokkaido-akan",
        "ameasure_file": "measurements_A_00004.csv", 
        "distance_from_face": 1.0,
        "daily_advance": 5.0,
        "max_distance_from_face": 200.0
    }
    
    print(f"テストパラメータ: {test_params}")
    
    # デバッグ情報を出力
    debug_data_processing(test_params)
    
    # 1. GUI関数を直接呼び出し
    print("\n1. GUI関数を実行中...")
    try:
        # GUI用のパス設定 (絶対パスを使用)
        INPUT_FOLDER = '/home/nowatari/repos/enzan-ai-cn-dev/data_folder'
        input_folder = os.path.join(INPUT_FOLDER, test_params["folder_name"], 'main_tunnel', 'CN_measurement_data')
        a_measure_path = os.path.join(INPUT_FOLDER, test_params["folder_name"], 'main_tunnel', 'CN_measurement_data', 'measurements_A', test_params["ameasure_file"])
        
        # prediction phase
        print("  予測フェーズを実行中...")
        df_all_gui_pred, settlements, convergences = simulate_displacement(
            input_folder, a_measure_path, test_params["max_distance_from_face"]
        )
        
        # simulation phase  
        print("  シミュレーションフェーズを実行中...")
        df_all_gui_sim, _, _ = simulate_displacement(
            input_folder, a_measure_path, test_params["max_distance_from_face"],
            test_params["daily_advance"], test_params["distance_from_face"], recursive=True
        )
        
        print(f"  GUI結果: prediction={len(df_all_gui_pred)}行, simulation={len(df_all_gui_sim)}行")
        
        # prediction列を抽出
        gui_prediction_cols = [col for col in df_all_gui_sim.columns if col.endswith('_prediction')]
        gui_result = df_all_gui_sim[['切羽からの距離'] + gui_prediction_cols].copy()
        
        print(f"  GUI予測列: {gui_prediction_cols}")
        
    except Exception as e:
        print(f"  GUI実行エラー: {e}")
        return False
    
    # 2. APIを呼び出し
    print("\n2. API呼び出し中...")
    try:
        api_url = "http://localhost:8000/api/v1/simulation/local-displacement"
        api_params = {
            "folder_name": test_params["folder_name"],
            "ameasure_file": test_params["ameasure_file"],
            "distance_from_face": test_params["distance_from_face"],
            "daily_advance": test_params["daily_advance"],
            "max_distance_from_face": test_params["max_distance_from_face"]
        }
        
        print(f"  リクエストURL: {api_url}")
        print(f"  リクエストパラメータ: {json.dumps(api_params, indent=2, ensure_ascii=False)}")
        
        response = requests.post(api_url, json=api_params, timeout=30)
        
        print(f"  レスポンスステータス: {response.status_code}")
        print(f"  レスポンスヘッダー: {dict(response.headers)}")
        
        if response.status_code != 200:
            print(f"  レスポンステキスト: {response.text}")
            response.raise_for_status()
        
        api_result_raw = response.json()
        
        # APIレスポンス全体を出力
        print("\n--- API完全レスポンス ---")
        print(f"  folder_name: {api_result_raw.get('folder_name')}")
        print(f"  cycle_no: {api_result_raw.get('cycle_no')}")
        print(f"  td: {api_result_raw.get('td')}")
        print(f"  distance_from_face: {api_result_raw.get('distance_from_face')}")
        print(f"  daily_advance: {api_result_raw.get('daily_advance')}")
        print(f"  timestamp: {api_result_raw.get('timestamp')}")
        
        if 'prediction_charts' in api_result_raw:
            print(f"  prediction_charts: {api_result_raw['prediction_charts']}")
        if 'simulation_charts' in api_result_raw:
            print(f"  simulation_charts: {api_result_raw['simulation_charts']}")
        if 'simulation_csv' in api_result_raw:
            print(f"  simulation_csv: {api_result_raw['simulation_csv']}")
        
        # APIレスポンスからDataFrameを構築
        simulation_data = api_result_raw["simulation_data"]
        print(f"  simulation_data length: {len(simulation_data)}")
        
        if len(simulation_data) > 0:
            print(f"  simulation_data first item keys: {list(simulation_data[0].keys())}")
            print(f"  simulation_data first 3 items:")
            for i, item in enumerate(simulation_data[:3]):
                print(f"    [{i}]: {item}")
        
        api_result = pd.DataFrame(simulation_data)
        
        print(f"  API結果: {len(api_result)}行")
        print(f"  API列: {list(api_result.columns)}")
        
    except requests.exceptions.RequestException as e:
        print(f"  APIリクエストエラー: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  レスポンスステータス: {e.response.status_code}")
            print(f"  レスポンステキスト: {e.response.text}")
        return False
    except Exception as e:
        print(f"  API呼び出しエラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 結果を比較
    print("\n3. 結果比較中...")
    
    try:
        # データサンプルを表示
        print("\n--- GUI結果サンプル (最初の3行) ---")
        print(gui_result.head(3).to_string())
        
        print("\n--- API結果サンプル (最初の3行) ---")
        print(api_result.head(3).to_string())
        
        # 行数比較
        print(f"\n行数比較: GUI={len(gui_result)}, API={len(api_result)}")
        if len(gui_result) != len(api_result):
            print("❌ 行数が一致しません")
            return False
        else:
            print("✅ 行数が一致")
        
        # 列名比較 (API側の列名をGUI形式に変換)
        gui_cols = set(gui_result.columns)
        api_cols = set(api_result.columns)
        
        print(f"\nGUI列: {sorted(gui_cols)}")
        print(f"API列: {sorted(api_cols)}")
        
        # 共通列を見つける
        common_cols = gui_cols.intersection(api_cols)
        print(f"共通列: {sorted(common_cols)}")
        
        if not common_cols:
            print("❌ 共通列がありません")
            return False
        
        # 数値比較 (許容誤差1e-10)
        tolerance = 1e-10
        all_close = True
        
        for col in common_cols:
            gui_values = gui_result[col].values
            api_values = api_result[col].values
            
            try:
                if np.allclose(gui_values, api_values, atol=tolerance, rtol=tolerance):
                    print(f"✅ 列 '{col}': 数値が一致 (誤差範囲内)")
                else:
                    print(f"❌ 列 '{col}': 数値が一致しません")
                    
                    # 詳細な差分を表示
                    diff = np.abs(gui_values - api_values)
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)
                    
                    print(f"   最大差分: {max_diff}")
                    print(f"   平均差分: {mean_diff}")
                    
                    # 最初の5つの値を比較
                    print(f"   GUI最初の5値: {gui_values[:5]}")
                    print(f"   API最初の5値: {api_values[:5]}")
                    print(f"   差分最初の5値: {diff[:5]}")
                    
                    all_close = False
            except Exception as e:
                print(f"❌ 列 '{col}': 比較エラー {e}")
                all_close = False
        
        if all_close:
            print("\n🎉 全ての数値が一致しました！")
            return True
        else:
            print("\n❌ 一部の数値が一致しませんでした")
            return False
            
    except Exception as e:
        print(f"  比較エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gui_api_comparison()
    if success:
        print("\n✅ テスト成功: GUIとAPIの出力は同一です")
        sys.exit(0)
    else:
        print("\n❌ テスト失敗: GUIとAPIの出力が異なります")
        sys.exit(1)