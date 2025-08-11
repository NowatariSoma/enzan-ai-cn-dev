#!/usr/bin/env python3
"""
FastAPIのmake-datasetエンドポイントとStreamlitアプリのcreate_dataset関数の結果を比較するスクリプト
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# 両方のモジュールをインポート
sys.path.append('/home/nowatari/repos/tomosigoto/enzan-ai-cn-dev/ai_ameasure')
sys.path.append('/home/nowatari/repos/tomosigoto/enzan-ai-cn-dev/microservices/ai_ameasure')

# StreamlitアプリのDisplacement modules
try:
    from displacement_temporal_spacial_analysis import (
        create_dataset as streamlit_create_dataset,
        generate_additional_info_df as streamlit_generate_additional_info_df,
        generate_dataframes as streamlit_generate_dataframes
    )
    from displacement import DATE, CYCLE_NO, SECTION_TD, FACE_TD, TD_NO, CONVERGENCES, SETTLEMENTS, STA, DISTANCE_FROM_FACE, DAYS_FROM_START, DIFFERENCE_FROM_FINAL_CONVERGENCES, DIFFERENCE_FROM_FINAL_SETTLEMENTS
    print("✓ Streamlitアプリのモジュールをインポートしました")
except ImportError as e:
    print(f"✗ Streamlitアプリのモジュールインポートエラー: {e}")
    # 別のパスを試す
    try:
        sys.path.append('/home/nowatari/repos/tomosigoto/enzan-ai-cn-dev/ai_ameasure/app')
        from displacement_temporal_spacial_analysis import (
            create_dataset as streamlit_create_dataset,
            generate_additional_info_df as streamlit_generate_additional_info_df,
            generate_dataframes as streamlit_generate_dataframes
        )
        from displacement import DATE, CYCLE_NO, SECTION_TD, FACE_TD, TD_NO, CONVERGENCES, SETTLEMENTS, STA, DISTANCE_FROM_FACE, DAYS_FROM_START, DIFFERENCE_FROM_FINAL_CONVERGENCES, DIFFERENCE_FROM_FINAL_SETTLEMENTS
        print("✓ Streamlitアプリのモジュールをインポートしました（app/パス）")
    except ImportError as e2:
        print(f"✗ Streamlitアプリのモジュール再インポートエラー: {e2}")
        sys.exit(1)

# FastAPIのモジュール
try:
    from app.api.endpoints.measurements import (
        create_dataset as fastapi_create_dataset,
        generate_additional_info_df as fastapi_generate_additional_info_df
    )
    from app.core.csv_loader import CSVDataLoader
    from app.core.config import settings
    print("✓ FastAPIのモジュールをインポートしました")
except ImportError as e:
    print(f"✗ FastAPIのモジュールインポートエラー: {e}")
    sys.exit(1)

def load_config():
    """設定ファイルを読み込み"""
    config_path = '/home/nowatari/repos/tomosigoto/enzan-ai-cn-dev/ai_ameasure/config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        print(f"設定ファイルが見つかりません: {config_path}")
        return None

def compare_dataframes(df1, df2, name1, name2):
    """2つのDataFrameを比較"""
    print(f"\n=== {name1} vs {name2} の比較 ===")
    
    # 基本情報の比較
    print(f"{name1} shape: {df1.shape if df1 is not None and not df1.empty else 'Empty/None'}")
    print(f"{name2} shape: {df2.shape if df2 is not None and not df2.empty else 'Empty/None'}")
    
    if df1 is None or df1.empty or df2 is None or df2.empty:
        print("どちらかのデータフレームが空またはNoneです")
        return False
    
    # 列名の比較
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    print(f"{name1} 列数: {len(cols1)}")
    print(f"{name2} 列数: {len(cols2)}")
    
    common_cols = cols1.intersection(cols2)
    only_in_1 = cols1 - cols2
    only_in_2 = cols2 - cols1
    
    print(f"共通列: {len(common_cols)}")
    if only_in_1:
        print(f"{name1}のみ: {list(only_in_1)}")
    if only_in_2:
        print(f"{name2}のみ: {list(only_in_2)}")
    
    # 共通列のデータ比較
    if common_cols:
        print(f"\n共通列の最初の5行データ比較:")
        for col in list(common_cols)[:5]:  # 最初の5列のみ
            if col in df1.columns and col in df2.columns:
                val1 = df1[col].head(3).values
                val2 = df2[col].head(3).values
                print(f"  {col}:")
                print(f"    {name1}: {val1}")
                print(f"    {name2}: {val2}")
    
    return len(common_cols) > 0

def compare_dataset_results(result1, result2):
    """create_dataset関数の結果を比較"""
    print("\n" + "="*60)
    print("CREATE_DATASET 結果比較")
    print("="*60)
    
    # Streamlitの結果
    if isinstance(result1, tuple) and len(result1) == 2:
        settlement_data_1, convergence_data_1 = result1
        print(f"Streamlit結果: settlement_data={type(settlement_data_1)}, convergence_data={type(convergence_data_1)}")
        
        if isinstance(settlement_data_1, tuple) and len(settlement_data_1) == 3:
            df_s1, x_cols_s1, y_col_s1 = settlement_data_1
            print(f"  Settlement: df={df_s1.shape if hasattr(df_s1, 'shape') else type(df_s1)}, x_cols={len(x_cols_s1) if x_cols_s1 else 0}, y_col={y_col_s1}")
        
        if isinstance(convergence_data_1, tuple) and len(convergence_data_1) == 3:
            df_c1, x_cols_c1, y_col_c1 = convergence_data_1
            print(f"  Convergence: df={df_c1.shape if hasattr(df_c1, 'shape') else type(df_c1)}, x_cols={len(x_cols_c1) if x_cols_c1 else 0}, y_col={y_col_c1}")
    
    # FastAPIの結果
    if isinstance(result2, tuple) and len(result2) == 2:
        settlement_data_2, convergence_data_2 = result2
        print(f"FastAPI結果: settlement_data={type(settlement_data_2)}, convergence_data={type(convergence_data_2)}")
        
        if isinstance(settlement_data_2, tuple) and len(settlement_data_2) == 3:
            df_s2, x_cols_s2, y_col_s2 = settlement_data_2
            print(f"  Settlement: df={df_s2.shape if hasattr(df_s2, 'shape') else type(df_s2)}, x_cols={len(x_cols_s2) if x_cols_s2 else 0}, y_col={y_col_s2}")
        
        if isinstance(convergence_data_2, tuple) and len(convergence_data_2) == 3:
            df_c2, x_cols_c2, y_col_c2 = convergence_data_2
            print(f"  Convergence: df={df_c2.shape if hasattr(df_c2, 'shape') else type(df_c2)}, x_cols={len(x_cols_c2) if x_cols_c2 else 0}, y_col={y_col_c2}")
    
    # 詳細比較
    if (isinstance(result1, tuple) and len(result1) == 2 and 
        isinstance(result2, tuple) and len(result2) == 2):
        
        settlement_data_1, convergence_data_1 = result1
        settlement_data_2, convergence_data_2 = result2
        
        # Settlement data comparison
        if (isinstance(settlement_data_1, tuple) and len(settlement_data_1) == 3 and
            isinstance(settlement_data_2, tuple) and len(settlement_data_2) == 3):
            df_s1, x_cols_s1, y_col_s1 = settlement_data_1
            df_s2, x_cols_s2, y_col_s2 = settlement_data_2
            compare_dataframes(df_s1, df_s2, "Streamlit Settlement", "FastAPI Settlement")
            
            print(f"\nSettlement X columns比較:")
            print(f"  Streamlit: {x_cols_s1}")
            print(f"  FastAPI: {x_cols_s2}")
            print(f"  Y column - Streamlit: {y_col_s1}, FastAPI: {y_col_s2}")
        
        # Convergence data comparison
        if (isinstance(convergence_data_1, tuple) and len(convergence_data_1) == 3 and
            isinstance(convergence_data_2, tuple) and len(convergence_data_2) == 3):
            df_c1, x_cols_c1, y_col_c1 = convergence_data_1
            df_c2, x_cols_c2, y_col_c2 = convergence_data_2
            compare_dataframes(df_c1, df_c2, "Streamlit Convergence", "FastAPI Convergence")
            
            print(f"\nConvergence X columns比較:")
            print(f"  Streamlit: {x_cols_c1}")
            print(f"  FastAPI: {x_cols_c2}")
            print(f"  Y column - Streamlit: {y_col_c1}, FastAPI: {y_col_c2}")

def main():
    print("データセット比較テストを開始します...")
    
    # 設定読み込み
    config = load_config()
    if not config:
        return
    
    # テスト用のパラメータ
    folder_name = "01-hokkaido-akan"
    max_distance_from_face = 100
    
    # Streamlitアプリの方法でデータを生成
    print(f"\n{'='*60}")
    print("STREAMLIT アプリのデータ生成")
    print(f"{'='*60}")
    
    try:
        input_folder = Path(config['input_folder']) / folder_name / 'main_tunnel' / 'CN_measurement_data'
        print(f"入力フォルダ: {input_folder}")
        
        # CSVファイルのリスト取得
        measurement_a_csvs = list((input_folder / 'measurements_A').glob('*.csv'))
        print(f"測定CSVファイル数: {len(measurement_a_csvs)}")
        
        if not measurement_a_csvs:
            print("測定CSVファイルが見つかりません")
            return
        
        # 追加情報ファイル
        cycle_support_csv = input_folder / 'cycle_support' / 'cycle_support.csv'
        observation_of_face_csv = input_folder / 'observation_of_face' / 'observation_of_face.csv'
        
        print(f"Cycle support CSV: {cycle_support_csv.exists()}")
        print(f"Observation face CSV: {observation_of_face_csv.exists()}")
        
        # Streamlitアプリの処理
        df_additional_info_streamlit = streamlit_generate_additional_info_df(
            str(cycle_support_csv), str(observation_of_face_csv)
        )
        df_additional_info_streamlit.drop(columns=[STA], inplace=True)
        
        df_all_streamlit, _, _, _, settlements, convergences = streamlit_generate_dataframes(
            [str(f) for f in measurement_a_csvs], max_distance_from_face
        )
        
        print(f"Streamlit df_all shape: {df_all_streamlit.shape}")
        print(f"Streamlit additional_info shape: {df_additional_info_streamlit.shape}")
        
        # Streamlitのcreate_dataset実行
        streamlit_result = streamlit_create_dataset(df_all_streamlit, df_additional_info_streamlit)
        
    except Exception as e:
        print(f"Streamlitデータ生成エラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # FastAPIアプリの方法でデータを生成  
    print(f"\n{'='*60}")
    print("FASTAPI アプリのデータ生成")
    print(f"{'='*60}")
    
    try:
        # FastAPIの設定に合わせてデータパス調整
        data_folder = Path('/home/nowatari/repos/tomosigoto/enzan-ai-cn-dev/data')
        input_folder_fastapi = data_folder / folder_name / "main_tunnel" / "CN_measurement_data"
        
        # CSVローダーでデータ読み込み
        csv_loader = CSVDataLoader()
        df_all_fastapi = csv_loader.load_all_measurement_data(data_folder, folder_name)
        
        print(f"FastAPI df_all shape: {df_all_fastapi.shape}")
        
        # 追加情報ファイル
        cycle_support_csv_fastapi = input_folder_fastapi / 'cycle_support' / 'cycle_support.csv'
        observation_of_face_csv_fastapi = input_folder_fastapi / 'observation_of_face' / 'observation_of_face.csv'
        
        if cycle_support_csv_fastapi.exists() and observation_of_face_csv_fastapi.exists():
            df_additional_info_fastapi = fastapi_generate_additional_info_df(
                cycle_support_csv_fastapi, observation_of_face_csv_fastapi
            )
            print(f"FastAPI additional_info shape: {df_additional_info_fastapi.shape}")
            
            # FastAPIのcreate_dataset実行
            fastapi_result = fastapi_create_dataset(df_all_fastapi, df_additional_info_fastapi)
        else:
            print("FastAPI用の追加情報ファイルが見つかりません")
            return
        
    except Exception as e:
        print(f"FastAPIデータ生成エラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 結果比較
    compare_dataset_results(streamlit_result, fastapi_result)
    
    print(f"\n{'='*60}")
    print("比較テスト完了")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 