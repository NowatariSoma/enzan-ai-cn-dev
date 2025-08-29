#!/usr/bin/env python3
"""
レコード単位でのデータ処理の詳細デバッグ
"""

import os
import sys
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev')
sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure')

# GUI用のパス設定
os.chdir('/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure')

from app.displacement_temporal_spacial_analysis import generate_additional_info_df, generate_dataframes, create_dataset, STA

def debug_record_level():
    """レコード単位でデータ処理をデバッグ"""
    
    # テストパラメータ
    INPUT_FOLDER = '/home/nowatari/repos/enzan-ai-cn-dev/data_folder'
    folder_name = "01-hokkaido-akan"
    ameasure_file = "measurements_A_00004.csv"
    max_distance_from_face = 200.0
    
    input_folder = os.path.join(INPUT_FOLDER, folder_name, 'main_tunnel', 'CN_measurement_data')
    a_measure_path = os.path.join(INPUT_FOLDER, folder_name, 'main_tunnel', 'CN_measurement_data', 'measurements_A', ameasure_file)
    
    print("=== レコード単位デバッグ ===")
    print(f"input_folder: {input_folder}")
    print(f"a_measure_path: {a_measure_path}")
    
    # 1. additional_info データフレーム
    cycle_support_csv = os.path.join(input_folder, 'cycle_support/cycle_support.csv')
    observation_of_face_csv = os.path.join(input_folder, 'observation_of_face/observation_of_face.csv')
    
    df_additional_info = generate_additional_info_df(cycle_support_csv, observation_of_face_csv)
    print(f"\n=== df_additional_info ===")
    print(f"Shape: {df_additional_info.shape}")
    print(f"Columns: {list(df_additional_info.columns)[:10]}...")  # 最初の10列だけ表示
    if STA in df_additional_info.columns:
        print(f"STA values: {df_additional_info[STA].unique()[:10]}")  # 最初の10個のユニーク値
    
    df_additional_info.drop(columns=[STA], inplace=True)
    print(f"After drop STA shape: {df_additional_info.shape}")
    
    # 2. generate_dataframes の詳細確認
    print(f"\n=== generate_dataframes ===")
    df_all, _, _, _, settlements, convergences = generate_dataframes([a_measure_path], max_distance_from_face)
    print(f"df_all shape: {df_all.shape}")
    print(f"df_all columns: {list(df_all.columns)}")
    print(f"settlements: {settlements}")
    print(f"convergences: {convergences}")
    
    # df_allの詳細確認
    print(f"\n--- df_all詳細 ---")
    print(f"切羽からの距離の範囲: {df_all['切羽からの距離'].min():.2f} ~ {df_all['切羽からの距離'].max():.2f}")
    print(f"計測経過日数の範囲: {df_all['計測経過日数'].min():.2f} ~ {df_all['計測経過日数'].max():.2f}")
    
    # position_id が存在するか確認
    if 'position_id' in df_all.columns:
        print(f"position_id unique values: {sorted(df_all['position_id'].unique())}")
    else:
        print("position_id column not found in df_all")
    
    # 沈下量と変位量の値を確認
    for col in settlements + convergences:
        if col in df_all.columns:
            values = df_all[col].dropna()
            print(f"{col}: min={values.min():.2f}, max={values.max():.2f}, count={len(values)}")
    
    # 3. create_dataset の詳細確認  
    print(f"\n=== create_dataset ===")
    settlement_data, convergence_data = create_dataset(df_all, df_additional_info)
    
    # settlement_data 詳細確認
    if isinstance(settlement_data, tuple) and len(settlement_data) >= 3:
        settlement_df, settlement_x_cols, settlement_y_col = settlement_data
        print(f"\n--- settlement_data ---")
        print(f"settlement_df shape: {settlement_df.shape}")
        print(f"settlement_y_col: {settlement_y_col}")
        
        # position_idの確認
        if 'position_id' in settlement_df.columns:
            unique_ids = settlement_df['position_id'].unique()
            print(f"settlement position_id unique count: {len(unique_ids)}")
            print(f"settlement position_id unique values: {sorted(unique_ids)[:20]}")  # 最初の20個
            
            # 各position_idでのレコード数を確認
            id_counts = settlement_df['position_id'].value_counts()
            print(f"Records per position_id: {id_counts.head(10)}")
        
        # 実際の予測対象値（y）の確認
        if settlement_y_col in settlement_df.columns:
            y_values = settlement_df[settlement_y_col].values
            print(f"settlement y values range: {y_values.min():.2f} ~ {y_values.max():.2f}")
            print(f"settlement y first 10 values: {y_values[:10]}")
        
        # 切羽からの距離と計測経過日数の関係を確認
        if '切羽からの距離' in settlement_df.columns and '計測経過日数' in settlement_df.columns:
            print(f"\n--- settlement_df: 距離と日数の関係 ---")
            sample_df = settlement_df[['position_id', '切羽からの距離', '計測経過日数', settlement_y_col]].head(20)
            print(sample_df.to_string())
    
    # convergence_data 詳細確認  
    if isinstance(convergence_data, tuple) and len(convergence_data) >= 3:
        convergence_df, convergence_x_cols, convergence_y_col = convergence_data
        print(f"\n--- convergence_data ---")
        print(f"convergence_df shape: {convergence_df.shape}")
        print(f"convergence_y_col: {convergence_y_col}")
        
        # position_idの確認
        if 'position_id' in convergence_df.columns:
            unique_ids = convergence_df['position_id'].unique()
            print(f"convergence position_id unique count: {len(unique_ids)}")
            print(f"convergence position_id unique values: {sorted(unique_ids)[:20]}")  # 最初の20個
        
        # 実際の予測対象値（y）の確認
        if convergence_y_col in convergence_df.columns:
            y_values = convergence_df[convergence_y_col].values
            print(f"convergence y values range: {y_values.min():.2f} ~ {y_values.max():.2f}")
            print(f"convergence y first 10 values: {y_values[:10]}")

if __name__ == "__main__":
    debug_record_level()