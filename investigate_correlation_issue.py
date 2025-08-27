#!/usr/bin/env python3
"""
NaN値だけの列の相関行列での表現問題を調査
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# パスの設定
sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure')

def investigate_correlation_issue():
    """NaN値だけの列の相関行列問題を調査"""
    
    print("=" * 60)
    print("NaN値だけの列の相関行列問題調査")
    print("=" * 60)
    
    try:
        # 元データファイルからのインポート
        sys.path.insert(0, '/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure/app')
        from displacement_temporal_spacial_analysis import (
            create_dataset as original_create_dataset,
            generate_additional_info_df as original_generate_additional_info_df,
            generate_dataframes as original_generate_dataframes
        )
        
        # テストデータの取得
        base_folder = "/home/nowatari/repos/enzan-ai-cn-dev/data_folder/01-hokkaido-akan/main_tunnel/CN_measurement_data"
        measurements_folder = os.path.join(base_folder, "measurements_A")
        cycle_support_csv = os.path.join(base_folder, "cycle_support", "cycle_support.csv")
        observation_of_face_csv = os.path.join(base_folder, "observation_of_face", "observation_of_face.csv")
        
        # CSVファイル一覧の取得
        csv_files = [os.path.join(measurements_folder, f) for f in os.listdir(measurements_folder) if f.endswith('.csv')]
        csv_files = sorted(csv_files)[:5]  # 最初の5ファイル
        
        # データ生成
        max_distance_from_face = 100.0
        df_all, _, _, _, _, _ = original_generate_dataframes(csv_files, max_distance_from_face)
        df_additional_info = original_generate_additional_info_df(cycle_support_csv, observation_of_face_csv)
        
        settlement_data, convergence_data = original_create_dataset(df_all, df_additional_info)
        
        # Settlement dataを使って調査
        df, x_columns, y_column = settlement_data
        
        print(f"DataFrame shape: {df.shape}")
        print(f"X列数: {len(x_columns)}")
        
        # 欠損値状況の詳細調査
        print("\n=== 欠損値状況の詳細調査 ===")
        missing_info = {}
        for col in df.columns:
            total = len(df)
            missing = df[col].isnull().sum()
            missing_pct = (missing / total) * 100
            missing_info[col] = {
                'missing_count': missing,
                'missing_pct': missing_pct,
                'dtype': df[col].dtype
            }
            
        # 100%欠損値の列を特定
        fully_missing_cols = [col for col, info in missing_info.items() if info['missing_pct'] == 100.0]
        print(f"\n100%欠損値の列（{len(fully_missing_cols)}列）:")
        for col in fully_missing_cols:
            print(f"  - {col} ({missing_info[col]['dtype']})")
            
        # 相関行列の計算と調査
        print(f"\n=== 相関行列の調査 ===")
        
        # 数値列のみを抽出
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"数値列数: {len(numeric_cols)}")
        
        # 相関行列計算
        corr_matrix = df[numeric_cols].corr()
        print(f"相関行列のshape: {corr_matrix.shape}")
        
        # 100%欠損列の相関値を詳しく調査
        print(f"\n=== 100%欠損列の相関値詳細調査 ===")
        numeric_fully_missing = [col for col in fully_missing_cols if col in numeric_cols]
        print(f"数値型で100%欠損の列: {numeric_fully_missing}")
        
        for col in numeric_fully_missing:
            if col in corr_matrix.columns:
                print(f"\n列: {col}")
                print(f"  対角要素（自己相関）: {corr_matrix.loc[col, col]}")
                print(f"  他列との相関:")
                other_corrs = corr_matrix.loc[col, corr_matrix.columns != col]
                print(f"    NaN以外の相関値数: {other_corrs.notna().sum()}")
                print(f"    NaNの相関値数: {other_corrs.isna().sum()}")
                
                if other_corrs.notna().sum() > 0:
                    print(f"    NaN以外の値（最初の5つ）: {other_corrs.dropna().head().values}")
        
        # 小さなテストケースで問題を再現
        print(f"\n=== テストケース：完全NaN列の相関計算 ===")
        test_df = pd.DataFrame({
            'normal_col': [1, 2, 3, 4, 5],
            'full_nan_col': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'another_normal': [5, 4, 3, 2, 1]
        })
        
        print("テストDataFrame:")
        print(test_df)
        
        test_corr = test_df.corr()
        print(f"\nテスト相関行列:")
        print(test_corr)
        
        print(f"\n完全NaN列の対角要素: {test_corr.loc['full_nan_col', 'full_nan_col']}")
        
        # pandasのバージョン確認
        print(f"\n=== 環境情報 ===")
        print(f"pandas version: {pd.__version__}")
        print(f"numpy version: {np.__version__}")
        
        # seabornのheatmap動作確認
        print(f"\n=== seaborn heatmap での表現確認 ===")
        
        plt.figure(figsize=(8, 6))
        # NaNを含む小さな相関行列でテスト
        small_test_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [np.nan, np.nan, np.nan],  # 完全NaN
            'C': [3, 2, 1]
        })
        
        small_corr = small_test_df.corr()
        print("小テスト相関行列:")
        print(small_corr)
        
        # seabornでの描画
        sns.heatmap(small_corr, annot=True, fmt=".2f", cmap="coolwarm", 
                   cbar=True, square=True, center=0)
        plt.title("NaN列を含む相関行列のheatmap")
        plt.tight_layout()
        plt.savefig("/home/nowatari/repos/enzan-ai-cn-dev/correlation_test.png")
        plt.close()
        
        print("テスト用heatmapを保存: /home/nowatari/repos/enzan-ai-cn-dev/correlation_test.png")
        
        # 実際のデータでの問題を調査
        print(f"\n=== 実データでの相関行列heatmap問題調査 ===")
        
        # 問題のある列を含む小さなサブセットで確認
        problem_cols = numeric_fully_missing + ['切羽からの距離', '計測経過日数', 'ｻｲｸﾙNo']
        problem_cols = [col for col in problem_cols if col in df.columns][:8]  # 最大8列
        
        if len(problem_cols) > 0:
            print(f"問題調査対象列: {problem_cols}")
            subset_df = df[problem_cols]
            subset_corr = subset_df.corr()
            
            print(f"\nサブセット相関行列:")
            print(subset_corr)
            
            # サブセットでheatmap作成
            plt.figure(figsize=(10, 8))
            mask = subset_corr.isna()  # NaN値をマスク
            
            sns.heatmap(subset_corr, annot=True, fmt=".2f", cmap="coolwarm", 
                       cbar=True, square=True, center=0, mask=mask,
                       cbar_kws={'label': 'Correlation'})
            plt.title("実データサブセット相関行列（NaN値マスク付き）")
            plt.tight_layout()
            plt.savefig("/home/nowatari/repos/enzan-ai-cn-dev/real_data_correlation_masked.png")
            plt.close()
            
            # マスクなし版も作成
            plt.figure(figsize=(10, 8))
            sns.heatmap(subset_corr, annot=True, fmt=".2f", cmap="coolwarm", 
                       cbar=True, square=True, center=0,
                       cbar_kws={'label': 'Correlation'})
            plt.title("実データサブセット相関行列（マスクなし）")
            plt.tight_layout()
            plt.savefig("/home/nowatari/repos/enzan-ai-cn-dev/real_data_correlation_no_mask.png")
            plt.close()
            
            print("実データheatmapを保存:")
            print("  - マスク付き: /home/nowatari/repos/enzan-ai-cn-dev/real_data_correlation_masked.png")
            print("  - マスクなし: /home/nowatari/repos/enzan-ai-cn-dev/real_data_correlation_no_mask.png")
            
        return True
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = investigate_correlation_issue()
    if success:
        print("\n✓ 調査完了")
    else:
        print("\n❌ 調査失敗")