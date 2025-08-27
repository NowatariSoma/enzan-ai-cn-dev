#!/usr/bin/env python3
"""
データ一致性検証スクリプト

元のgui_displacement_temporal_spacial_analysis.pyファイルのcreate_dataset関数と
API実装のmeasurements.pyファイルのcreate_dataset関数の出力データを比較する
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json

# パスの設定
sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure')
sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev/microservices/ai_ameasure')

def test_data_consistency():
    """両方のcreate_dataset関数の出力を比較する"""
    
    print("=" * 60)
    print("データ一致性検証テスト")
    print("=" * 60)
    
    try:
        # 元データファイルからのインポート
        print("1. 元データファイルからのインポート...")
        sys.path.insert(0, '/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure/app')
        from displacement_temporal_spacial_analysis import (
            create_dataset as original_create_dataset,
            generate_additional_info_df as original_generate_additional_info_df,
            generate_dataframes as original_generate_dataframes,
            SETTLEMENTS, CONVERGENCES, DIFFERENCE_FROM_FINAL_SETTLEMENTS, 
            DIFFERENCE_FROM_FINAL_CONVERGENCES
        )
        print("✓ 元データファイルのインポート成功")
        
        # API実装からのインポート  
        print("2. API実装からのインポート...")
        sys.path.insert(0, '/home/nowatari/repos/enzan-ai-cn-dev/microservices/ai_ameasure/app/api/endpoints')
        from measurements import (
            create_dataset as api_create_dataset,
            generate_additional_info_df as api_generate_additional_info_df
        )
        print("✓ API実装のインポート成功")
        
        # テストデータのパス設定
        print("3. テストデータパスの設定...")
        base_folder = "/home/nowatari/repos/enzan-ai-cn-dev/data_folder/01-hokkaido-akan/main_tunnel/CN_measurement_data"
        measurements_folder = os.path.join(base_folder, "measurements_A")
        cycle_support_csv = os.path.join(base_folder, "cycle_support", "cycle_support.csv")
        observation_of_face_csv = os.path.join(base_folder, "observation_of_face", "observation_of_face.csv")
        
        # ファイルの存在確認
        if not os.path.exists(measurements_folder):
            print(f"❌ 計測フォルダが見つかりません: {measurements_folder}")
            return False
            
        if not os.path.exists(cycle_support_csv):
            print(f"❌ cycle_support.csvが見つかりません: {cycle_support_csv}")
            return False
            
        if not os.path.exists(observation_of_face_csv):
            print(f"❌ observation_of_face.csvが見つかりません: {observation_of_face_csv}")
            return False
        
        print("✓ テストデータファイルの存在確認完了")
        
        # CSVファイル一覧の取得
        print("4. 計測データファイルの取得...")
        csv_files = [os.path.join(measurements_folder, f) for f in os.listdir(measurements_folder) if f.endswith('.csv')]
        csv_files = sorted(csv_files)[:5]  # 最初の5ファイルのみでテスト
        print(f"✓ テスト対象ファイル数: {len(csv_files)}")
        
        # 元データでのデータ生成
        print("5. 元データでのデータ生成...")
        max_distance_from_face = 100.0
        original_df_all, _, _, _, original_settlements, original_convergences = original_generate_dataframes(csv_files, max_distance_from_face)
        original_df_additional_info = original_generate_additional_info_df(cycle_support_csv, observation_of_face_csv)
        
        print(f"   - 元データ df_all shape: {original_df_all.shape}")
        print(f"   - 元データ df_additional_info shape: {original_df_additional_info.shape}")
        
        original_settlement_data, original_convergence_data = original_create_dataset(original_df_all, original_df_additional_info)
        print("✓ 元データでのデータセット作成完了")
        
        # API実装でのデータ生成
        print("6. API実装でのデータ生成...")
        # API実装では generate_dataframes が異なるため、同じデータを使用
        api_df_additional_info = api_generate_additional_info_df(cycle_support_csv, observation_of_face_csv)
        
        print(f"   - API実装 df_additional_info shape: {api_df_additional_info.shape}")
        
        api_settlement_data, api_convergence_data = api_create_dataset(original_df_all, api_df_additional_info)
        print("✓ API実装でのデータセット作成完了")
        
        # データ構造の比較
        print("7. データ構造の比較...")
        print("\n--- Settlement Data 比較 ---")
        compare_dataset_results("Settlement", original_settlement_data, api_settlement_data)
        
        print("\n--- Convergence Data 比較 ---") 
        compare_dataset_results("Convergence", original_convergence_data, api_convergence_data)
        
        # データ項目名の詳細表示
        print("\n" + "=" * 80)
        print("データ項目名（列名）の詳細")
        print("=" * 80)
        print_column_details(original_settlement_data, api_settlement_data, "Settlement")
        print_column_details(original_convergence_data, api_convergence_data, "Convergence")
        
        return True
        
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return False
    except FileNotFoundError as e:
        print(f"❌ ファイルが見つかりません: {e}")
        return False
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_dataset_results(name, original_data, api_data):
    """データセット結果を詳細比較"""
    
    print(f"\n{name} データ比較:")
    
    # データタイプの確認
    print(f"  元データタイプ: {type(original_data)}")
    print(f"  APIデータタイプ: {type(api_data)}")
    
    # タプル構造の確認
    if isinstance(original_data, tuple) and isinstance(api_data, tuple):
        print(f"  元データタプル長: {len(original_data)}")
        print(f"  APIデータタプル長: {len(api_data)}")
        
        if len(original_data) == 3 and len(api_data) == 3:
            original_df, original_x_cols, original_y_col = original_data
            api_df, api_x_cols, api_y_col = api_data
            
            # DataFrame比較
            print(f"  元データFrame形状: {original_df.shape if not original_df.empty else 'empty'}")
            print(f"  APIデータFrame形状: {api_df.shape if not api_df.empty else 'empty'}")
            
            # X列比較
            print(f"  元データX列数: {len(original_x_cols) if original_x_cols else 0}")
            print(f"  APIX列数: {len(api_x_cols) if api_x_cols else 0}")
            
            if original_x_cols and api_x_cols:
                common_x_cols = set(original_x_cols) & set(api_x_cols)
                original_only = set(original_x_cols) - set(api_x_cols) 
                api_only = set(api_x_cols) - set(original_x_cols)
                
                print(f"  共通X列数: {len(common_x_cols)}")
                if original_only:
                    print(f"  元データのみの列: {list(original_only)[:5]}{'...' if len(original_only) > 5 else ''}")
                if api_only:
                    print(f"  APIのみの列: {list(api_only)[:5]}{'...' if len(api_only) > 5 else ''}")
            
            # Y列比較
            print(f"  元データY列: {original_y_col}")
            print(f"  APIY列: {api_y_col}")
            
            # データフレームの詳細比較（空でない場合のみ）
            if not original_df.empty and not api_df.empty:
                # 共通列での数値比較
                common_cols = list(set(original_df.columns) & set(api_df.columns))
                if common_cols:
                    print(f"  共通列数: {len(common_cols)}")
                    
                    # 数値列のみを選択して比較
                    numeric_cols = []
                    for col in common_cols:
                        if (original_df[col].dtype in ['float64', 'int64'] and 
                            api_df[col].dtype in ['float64', 'int64']):
                            numeric_cols.append(col)
                    
                    if numeric_cols and len(numeric_cols) > 0:
                        print(f"  比較可能な数値列数: {len(numeric_cols)}")
                        
                        # より多くの列について詳細な統計値を比較
                        sample_cols = numeric_cols[:8]  # 最初の8列に拡張
                        
                        print(f"\n  === 詳細統計値比較 ({len(sample_cols)}列) ===")
                        for i, col in enumerate(sample_cols, 1):
                            if col in original_df.columns and col in api_df.columns:
                                print(f"\n  [{i}] 列: {col}")
                                
                                # 基本統計値を計算
                                orig_series = original_df[col].dropna()
                                api_series = api_df[col].dropna()
                                
                                # データ数の比較
                                print(f"      データ数: 元={len(orig_series)}, API={len(api_series)}")
                                
                                # 欠損値数の比較
                                orig_na = original_df[col].isna().sum()
                                api_na = api_df[col].isna().sum()
                                print(f"      欠損値数: 元={orig_na}, API={api_na}")
                                
                                if len(orig_series) > 0 and len(api_series) > 0:
                                    # 平均値
                                    orig_mean = orig_series.mean()
                                    api_mean = api_series.mean()
                                    mean_diff = abs(orig_mean - api_mean) if pd.notna(orig_mean) and pd.notna(api_mean) else float('inf')
                                    print(f"      平均値: 元={orig_mean:.6f}, API={api_mean:.6f}, 差分={mean_diff:.6f}")
                                    
                                    # 標準偏差
                                    orig_std = orig_series.std()
                                    api_std = api_series.std()
                                    std_diff = abs(orig_std - api_std) if pd.notna(orig_std) and pd.notna(api_std) else float('inf')
                                    print(f"      標準偏差: 元={orig_std:.6f}, API={api_std:.6f}, 差分={std_diff:.6f}")
                                    
                                    # 最小値・最大値
                                    orig_min, orig_max = orig_series.min(), orig_series.max()
                                    api_min, api_max = api_series.min(), api_series.max()
                                    print(f"      最小値: 元={orig_min:.6f}, API={api_min:.6f}, 差分={abs(orig_min-api_min):.6f}")
                                    print(f"      最大値: 元={orig_max:.6f}, API={api_max:.6f}, 差分={abs(orig_max-api_max):.6f}")
                                    
                                    # 中央値
                                    orig_median = orig_series.median()
                                    api_median = api_series.median()
                                    median_diff = abs(orig_median - api_median) if pd.notna(orig_median) and pd.notna(api_median) else float('inf')
                                    print(f"      中央値: 元={orig_median:.6f}, API={api_median:.6f}, 差分={median_diff:.6f}")
                                    
                                    # 25%・75%分位点
                                    orig_q25, orig_q75 = orig_series.quantile(0.25), orig_series.quantile(0.75)
                                    api_q25, api_q75 = api_series.quantile(0.25), api_series.quantile(0.75)
                                    print(f"      25%分位: 元={orig_q25:.6f}, API={api_q25:.6f}, 差分={abs(orig_q25-api_q25):.6f}")
                                    print(f"      75%分位: 元={orig_q75:.6f}, API={api_q75:.6f}, 差分={abs(orig_q75-api_q75):.6f}")
                                    
                                    # 歪度と尖度
                                    try:
                                        orig_skew = orig_series.skew()
                                        api_skew = api_series.skew()
                                        orig_kurt = orig_series.kurtosis()
                                        api_kurt = api_series.kurtosis()
                                        
                                        if pd.notna(orig_skew) and pd.notna(api_skew):
                                            print(f"      歪度: 元={orig_skew:.6f}, API={api_skew:.6f}, 差分={abs(orig_skew-api_skew):.6f}")
                                        if pd.notna(orig_kurt) and pd.notna(api_kurt):
                                            print(f"      尖度: 元={orig_kurt:.6f}, API={api_kurt:.6f}, 差分={abs(orig_kurt-api_kurt):.6f}")
                                    except:
                                        pass
                                    
                                    # データ型の比較
                                    print(f"      データ型: 元={original_df[col].dtype}, API={api_df[col].dtype}")
                                    
                                    # 完全一致判定
                                    if mean_diff < 1e-10 and std_diff < 1e-10:
                                        print(f"      判定: ✅ 完全一致")
                                    elif mean_diff < 1e-6 and std_diff < 1e-6:
                                        print(f"      判定: ✓ ほぼ一致")
                                    else:
                                        print(f"      判定: ❌ 不一致")
                                else:
                                    print(f"      ❌ データが空です")
                        
                        # 全体的な分析
                        print(f"\n  === 全体的な統計比較 ===")
                        
                        # 全数値列の統計サマリー
                        identical_cols = []
                        nearly_identical_cols = []
                        different_cols = []
                        
                        for col in numeric_cols:
                            if col in original_df.columns and col in api_df.columns:
                                orig_mean = original_df[col].mean()
                                api_mean = api_df[col].mean()
                                
                                if pd.notna(orig_mean) and pd.notna(api_mean):
                                    diff = abs(orig_mean - api_mean)
                                    if diff < 1e-10:
                                        identical_cols.append(col)
                                    elif diff < 1e-6:
                                        nearly_identical_cols.append(col)
                                    else:
                                        different_cols.append(col)
                        
                        print(f"  完全一致列数: {len(identical_cols)}/{len(numeric_cols)}")
                        print(f"  ほぼ一致列数: {len(nearly_identical_cols)}/{len(numeric_cols)}")
                        print(f"  不一致列数: {len(different_cols)}/{len(numeric_cols)}")
                        
                        if different_cols:
                            print(f"  不一致列: {different_cols[:5]}{'...' if len(different_cols) > 5 else ''}")
                        
                        # 完全一致しなかった列の詳細調査
                        non_identical_cols = []
                        for col in numeric_cols:
                            if col in original_df.columns and col in api_df.columns:
                                orig_mean = original_df[col].mean()
                                api_mean = api_df[col].mean()
                                
                                # NaNの場合や計算できない場合の処理
                                if pd.isna(orig_mean) and pd.isna(api_mean):
                                    # 両方ともNaNの場合は特別扱い
                                    non_identical_cols.append((col, "両方NaN"))
                                elif pd.isna(orig_mean) or pd.isna(api_mean):
                                    # 片方だけNaNの場合
                                    non_identical_cols.append((col, f"片方NaN: 元={orig_mean}, API={api_mean}"))
                                elif abs(orig_mean - api_mean) >= 1e-10:
                                    # 数値的に異なる場合
                                    non_identical_cols.append((col, f"数値差分: {abs(orig_mean - api_mean):.10f}"))
                        
                        if non_identical_cols:
                            print(f"\n  === 完全一致しなかった列の詳細調査 ===")
                            for i, (col, reason) in enumerate(non_identical_cols, 1):
                                print(f"  [{i}] {col}: {reason}")
                                
                                # さらに詳細な調査
                                orig_series = original_df[col]
                                api_series = api_df[col]
                                
                                # 欠損値の分布を調査
                                orig_na_count = orig_series.isna().sum()
                                api_na_count = api_series.isna().sum()
                                orig_total = len(orig_series)
                                api_total = len(api_series)
                                
                                print(f"      欠損値: 元={orig_na_count}/{orig_total}, API={api_na_count}/{api_total}")
                                
                                # 非欠損値の統計
                                orig_valid = orig_series.dropna()
                                api_valid = api_series.dropna()
                                
                                print(f"      有効値数: 元={len(orig_valid)}, API={len(api_valid)}")
                                
                                if len(orig_valid) == 0 and len(api_valid) == 0:
                                    print(f"      → 両方とも全て欠損値のため統計計算不可")
                                elif len(orig_valid) == 0:
                                    print(f"      → 元データが全て欠損値")
                                elif len(api_valid) == 0:
                                    print(f"      → APIデータが全て欠損値")
                                elif len(orig_valid) > 0 and len(api_valid) > 0:
                                    # 実際の値の差を調査
                                    if len(orig_valid) == len(api_valid):
                                        # サンプルを比較
                                        sample_size = min(5, len(orig_valid))
                                        orig_sample = orig_valid.iloc[:sample_size].values
                                        api_sample = api_valid.iloc[:sample_size].values
                                        print(f"      サンプル値比較(最初{sample_size}個):")
                                        for j in range(sample_size):
                                            diff = abs(orig_sample[j] - api_sample[j])
                                            print(f"        [{j+1}] 元={orig_sample[j]:.6f}, API={api_sample[j]:.6f}, 差分={diff:.10f}")
                        else:
                            print(f"  → 実際には全列が完全一致している可能性があります")
                        
                        # DataFrame全体の形状・サイズ比較
                        print(f"  DataFrame比較:")
                        print(f"    行数: 元={len(original_df)}, API={len(api_df)}")
                        print(f"    列数: 元={len(original_df.columns)}, API={len(api_df.columns)}")
                        print(f"    メモリ使用量: 元={original_df.memory_usage(deep=True).sum()//1024}KB, API={api_df.memory_usage(deep=True).sum()//1024}KB")
    else:
        print("  ❌ データがタプル形式ではありません")
        
    print(f"  比較結果: {'✓ 構造一致' if are_structures_similar(original_data, api_data) else '❌ 構造不一致'}")

def print_column_details(original_data, api_data, data_name):
    """データ項目名の詳細を表示"""
    print(f"\n=== {data_name} Data の全項目名 ===")
    
    if isinstance(original_data, tuple) and len(original_data) == 3:
        orig_df, orig_x_cols, orig_y_col = original_data
        api_df, api_x_cols, api_y_col = api_data
        
        print(f"\n【DataFrame 全列名】（計{len(orig_df.columns)}列）:")
        for i, col in enumerate(orig_df.columns, 1):
            print(f"  {i:2d}. {col}")
            
        print(f"\n【X列（特徴量）】（計{len(orig_x_cols)}列）:")
        for i, col in enumerate(orig_x_cols, 1):
            print(f"  {i:2d}. {col}")
            
        print(f"\n【Y列（目的変数）】:")
        print(f"  1. {orig_y_col}")
        
        # 列の分類
        categorical_cols = []
        numeric_cols = []
        bit_cols = []
        
        for col in orig_df.columns:
            if col.endswith('_bit'):
                bit_cols.append(col)
            elif col.endswith('_numeric'):
                categorical_cols.append(col)
            elif orig_df[col].dtype in ['float64', 'int64']:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        
        print(f"\n【列の分類】:")
        print(f"  数値列（{len(numeric_cols)}列）: {numeric_cols[:5]}{'...' if len(numeric_cols) > 5 else ''}")
        print(f"  カテゴリ列（{len(categorical_cols)}列）: {categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''}")
        print(f"  ビット列（{len(bit_cols)}列）: {bit_cols[:5]}{'...' if len(bit_cols) > 5 else ''}")
        
        # データ型の詳細
        print(f"\n【データ型の詳細】:")
        dtype_counts = orig_df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count}列")
            
        # 欠損値の状況
        print(f"\n【欠損値の状況】:")
        missing_counts = orig_df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0].sort_values(ascending=False)
        
        if len(missing_cols) > 0:
            print(f"  欠損値がある列（{len(missing_cols)}列）:")
            for col, count in missing_cols.items():
                percentage = (count / len(orig_df)) * 100
                print(f"    {col}: {count}個 ({percentage:.1f}%)")
        else:
            print(f"  欠損値なし（全列完全データ）")
            
        # 統計サマリー（数値列のみ）
        if len(numeric_cols) > 0:
            print(f"\n【数値列の統計サマリー】:")
            desc = orig_df[numeric_cols].describe()
            print(f"  平均値の範囲: {desc.loc['mean'].min():.3f} ～ {desc.loc['mean'].max():.3f}")
            print(f"  標準偏差の範囲: {desc.loc['std'].min():.3f} ～ {desc.loc['std'].max():.3f}")
            print(f"  最小値の範囲: {desc.loc['min'].min():.3f} ～ {desc.loc['min'].max():.3f}")
            print(f"  最大値の範囲: {desc.loc['max'].min():.3f} ～ {desc.loc['max'].max():.3f}")

def are_structures_similar(original_data, api_data):
    """データ構造が類似しているかチェック"""
    if type(original_data) != type(api_data):
        return False
        
    if isinstance(original_data, tuple) and isinstance(api_data, tuple):
        if len(original_data) != len(api_data):
            return False
            
        if len(original_data) == 3:
            orig_df, orig_x_cols, orig_y_col = original_data
            api_df, api_x_cols, api_y_col = api_data
            
            # Y列名の一致
            if orig_y_col != api_y_col:
                return False
            
            # データフレームの行数チェック（大きく異なる場合は不一致とする）
            if not orig_df.empty and not api_df.empty:
                if abs(len(orig_df) - len(api_df)) > len(orig_df) * 0.5:  # 50%以上差がある場合
                    return False
                    
            # X列数の大きな差をチェック
            if orig_x_cols and api_x_cols:
                if abs(len(orig_x_cols) - len(api_x_cols)) > max(len(orig_x_cols), len(api_x_cols)) * 0.5:
                    return False
    
    return True

if __name__ == "__main__":
    print("データ一致性検証テスト開始")
    success = test_data_consistency()
    if success:
        print("\n✓ テスト完了")
    else:
        print("\n❌ テスト失敗")
        sys.exit(1)