#!/usr/bin/env python3
"""
元のai_ameasureとマイクロサービス版の学習プロセスを詳細に比較するスクリプト
"""
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def debug_training_process():
    print("=" * 80)
    print("学習プロセスの詳細比較")
    print("=" * 80)
    
    # 1. 元のai_ameasureでの学習プロセス
    print("\n1. 元のai_ameasureでの学習プロセス...")
    sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure/app')
    
    from displacement_temporal_spacial_analysis import (
        generate_dataframes, 
        generate_additional_info_df,
        create_dataset,
        analyize_ml
    )
    from displacement import SECTION_TD, DATE
    import os
    
    # データ読み込み
    input_folder = '/home/nowatari/repos/enzan-ai-cn-dev/data_folder/01-hokkaido-akan/main_tunnel/CN_measurement_data'
    measurement_a_csvs = [
        os.path.join(input_folder, 'measurements_A', f)
        for f in os.listdir(os.path.join(input_folder, 'measurements_A'))
        if f.endswith('.csv')
    ]
    cycle_support_csv = os.path.join(input_folder, 'cycle_support/cycle_support.csv')
    observation_of_face_csv = os.path.join(input_folder, 'observation_of_face/observation_of_face.csv')
    
    df_additional_info = generate_additional_info_df(cycle_support_csv, observation_of_face_csv)
    df_all_original, dct_df_settlement, dct_df_convergence, dct_df_td, settlements, convergences = generate_dataframes(measurement_a_csvs, 100.0)
    
    print(f"   df_all shape: {df_all_original.shape}")
    print(f"   settlements: {settlements}")
    print(f"   convergences: {convergences}")
    
    # create_datasetの実行
    settlement_data, convergence_data = create_dataset(df_all_original, df_additional_info)
    
    print(f"   settlement_data shape: {settlement_data[0].shape}")
    print(f"   settlement_data x_columns count: {len(settlement_data[1])}")
    print(f"   settlement_data y_column: {settlement_data[2]}")
    
    # 沈下量データでの学習詳細
    df_settlement, x_columns_settlement, y_column_settlement = settlement_data
    
    print(f"\\n   詳細分析 - 沈下量データ:")
    print(f"   データ形状: {df_settlement.shape}")
    print(f"   特徴量数: {len(x_columns_settlement)}")
    print(f"   目的変数: {y_column_settlement}")
    
    # データ分割の詳細
    td = df_settlement[SECTION_TD].max()
    train_date = df_settlement[df_settlement[SECTION_TD] < td][DATE].max()
    df_train_orig = df_settlement[df_settlement[DATE] <= train_date]
    df_validate_orig = df_settlement[df_settlement[SECTION_TD] >= td]
    
    print(f"   TD閾値: {td}")
    print(f"   Train日付閾値: {train_date}")
    print(f"   Train samples: {len(df_train_orig)}")
    print(f"   Validation samples: {len(df_validate_orig)}")
    
    # 実際の学習
    model_orig = RandomForestRegressor(random_state=42)
    df_train_result, df_validate_result, model_trained, metrics_orig = analyize_ml(
        model_orig, df_train_orig, df_validate_orig, x_columns_settlement, y_column_settlement
    )
    
    print(f"   元のai_ameasure結果:")
    print(f"     Train R²: {metrics_orig['r2_train']:.6f}")
    print(f"     Validation R²: {metrics_orig['r2_validate']:.6f}")
    
    # マイクロサービス版との比較
    print("\\n2. マイクロサービス版との詳細比較...")
    sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev/microservices/ai_ameasure')
    
    # PredictionEngineでの学習
    from app.core.prediction_engine import PredictionEngine
    
    engine = PredictionEngine()
    result_micro = engine.train_model(
        model_name="random_forest",
        folder_name="01-hokkaido-akan",
        max_distance_from_face=100.0,
        td=None
    )
    
    print(f"   マイクロサービス結果:")
    training_metrics = result_micro.get('training_metrics', {})
    settlement_metrics = training_metrics.get('沈下量', {})
    if settlement_metrics:
        print(f"     Train R²: {settlement_metrics['r2_train']:.6f}")
        print(f"     Validation R²: {settlement_metrics['r2_validate']:.6f}")
        print(f"\\n   差異:")
        print(f"     Train R² 差異: {abs(metrics_orig['r2_train'] - settlement_metrics['r2_train']):.6f}")
        print(f"     Validation R² 差異: {abs(metrics_orig['r2_validate'] - settlement_metrics['r2_validate']):.6f}")
    else:
        print("     メトリクスが見つかりません")
        
    # 訓練・検証データの詳細比較
    print("\\n3. 訓練・検証データの詳細比較...")
    print(f"   元版 - Train: {len(df_train_orig)}, Validation: {len(df_validate_orig)}")
    print(f"   元版 - Train目的変数統計: min={df_train_orig[y_column_settlement].min():.3f}, max={df_train_orig[y_column_settlement].max():.3f}, mean={df_train_orig[y_column_settlement].mean():.3f}")
    print(f"   元版 - Validation目的変数統計: min={df_validate_orig[y_column_settlement].min():.3f}, max={df_validate_orig[y_column_settlement].max():.3f}, mean={df_validate_orig[y_column_settlement].mean():.3f}")
    
    print("\\n" + "=" * 80)
    print("比較完了")
    print("=" * 80)

if __name__ == "__main__":
    debug_training_process()