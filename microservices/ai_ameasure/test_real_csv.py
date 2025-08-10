#!/usr/bin/env python3
"""
実際のCSVファイルを使用したCSVDataLoaderのテストスクリプト
"""

import sys
import os
from pathlib import Path
import traceback

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from core.csv_loader import CSVDataLoader

def test_real_csv_data():
    """実際のデータフォルダのCSVファイルでテスト"""
    
    print("=" * 80)
    print("実際のCSVデータでCSVDataLoaderをテスト")
    print("=" * 80)
    
    # CSVDataLoaderのインスタンスを作成
    loader = CSVDataLoader()
    
    # 実際のCSVファイルのパスを設定
    data_dir = Path("/home/nowatari/repos/tomosigoto/enzan-ai-cn-dev/data_folder/01-hokkaido-akan/main_tunnel/CN_measurement_data/measurements_A")
    
    # CSVファイルのリストを取得
    csv_files = sorted(list(data_dir.glob("measurements_A_*.csv")))
    
    print(f"\n見つかったCSVファイル数: {len(csv_files)}")
    print(f"最初の5ファイル:")
    for csv_file in csv_files[:5]:
        print(f"  - {csv_file.name}")
    
    print("\n" + "=" * 80)
    print("テスト1: 単一ファイルの処理")
    print("=" * 80)
    
    # 最初のファイルでテスト
    test_file = csv_files[0]
    print(f"\nテストファイル: {test_file.name}")
    
    try:
        max_distance = 100
        df = loader.proccess_a_measure_file(test_file, max_distance)
        
        print(f"✓ ファイル処理成功")
        print(f"\nデータフレーム情報:")
        print(f"  - 形状: {df.shape}")
        print(f"  - カラム数: {len(df.columns)}")
        print(f"  - 行数: {len(df)}")
        
        print(f"\nカラム一覧:")
        for col in df.columns:
            print(f"  - {col}")
        
        print(f"\n最初の5行のデータ:")
        print(df.head())
        
        # データの統計情報
        if '切羽からの距離' in df.columns:
            print(f"\n切羽からの距離:")
            print(f"  - 最小値: {df['切羽からの距離'].min():.2f}")
            print(f"  - 最大値: {df['切羽からの距離'].max():.2f}")
            print(f"  - 平均値: {df['切羽からの距離'].mean():.2f}")
        
        if '計測経過日数' in df.columns:
            print(f"\n計測経過日数:")
            print(f"  - 最小値: {df['計測経過日数'].min()}")
            print(f"  - 最大値: {df['計測経過日数'].max()}")
        
    except Exception as e:
        print(f"✗ エラーが発生しました: {e}")
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("テスト2: 複数ファイルの処理")
    print("=" * 80)
    
    # 最初の5ファイルでテスト
    test_files = csv_files[:5]
    print(f"\nテストファイル数: {len(test_files)}")
    
    try:
        max_distance = 100
        result = loader.generate_dataframes(test_files, max_distance)
        df_all, dct_df_settlement, dct_df_convergence, dct_df_td, settlements, convergences = result
        
        print(f"✓ 複数ファイル処理成功")
        print(f"\n統合データフレーム情報:")
        print(f"  - 形状: {df_all.shape}")
        print(f"  - ユニークなTD数: {df_all['TD(m)'].nunique() if 'TD(m)' in df_all.columns else 'N/A'}")
        
        print(f"\n検出された測定項目:")
        print(f"  - 沈下量カラム: {settlements}")
        print(f"  - 変位量カラム: {convergences}")
        
        print(f"\n距離ごとのデータ件数:")
        for distance_key in sorted(dct_df_settlement.keys()):
            settlement_count = len(dct_df_settlement[distance_key]) if dct_df_settlement[distance_key] else 0
            convergence_count = len(dct_df_convergence[distance_key]) if dct_df_convergence[distance_key] else 0
            print(f"  - {distance_key}:")
            print(f"      沈下量データ数: {settlement_count}")
            print(f"      変位量データ数: {convergence_count}")
            
            if distance_key in dct_df_td and not dct_df_td[distance_key].empty:
                td_df = dct_df_td[distance_key]
                print(f"      TDデータフレーム形状: {td_df.shape}")
        
        # データの品質チェック
        print(f"\nデータ品質チェック:")
        
        # 欠損値のチェック
        missing_counts = df_all.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        if len(cols_with_missing) > 0:
            print(f"  ⚠ 欠損値があるカラム:")
            for col, count in cols_with_missing.items():
                print(f"      {col}: {count}件")
        else:
            print(f"  ✓ 欠損値なし")
        
        # データ範囲のチェック
        if '切羽からの距離' in df_all.columns:
            distance_range = df_all['切羽からの距離']
            print(f"\n  切羽からの距離の範囲:")
            print(f"    最小: {distance_range.min():.2f}m")
            print(f"    最大: {distance_range.max():.2f}m")
            print(f"    データ件数: {len(distance_range)}")
        
    except Exception as e:
        print(f"✗ エラーが発生しました: {e}")
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("テスト3: 全ファイルの処理（パフォーマンステスト）")
    print("=" * 80)
    
    print(f"\n全{len(csv_files)}ファイルを処理中...")
    
    try:
        import time
        start_time = time.time()
        
        max_distance = 200
        result = loader.generate_dataframes(csv_files, max_distance)
        df_all, dct_df_settlement, dct_df_convergence, dct_df_td, settlements, convergences = result
        
        elapsed_time = time.time() - start_time
        
        print(f"✓ 全ファイル処理成功")
        print(f"\n処理時間: {elapsed_time:.2f}秒")
        print(f"処理速度: {len(csv_files)/elapsed_time:.2f}ファイル/秒")
        
        print(f"\n最終データフレーム情報:")
        print(f"  - 総行数: {len(df_all)}")
        print(f"  - 総カラム数: {len(df_all.columns)}")
        print(f"  - メモリ使用量: {df_all.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # データの要約統計
        print(f"\n要約統計:")
        numeric_cols = df_all.select_dtypes(include=['float64', 'int64']).columns
        for col in ['沈下量1', '沈下量2', '沈下量3', '変位量A', '変位量B', '変位量C']:
            if col in numeric_cols:
                print(f"\n  {col}:")
                print(f"    平均: {df_all[col].mean():.3f}")
                print(f"    標準偏差: {df_all[col].std():.3f}")
                print(f"    最小: {df_all[col].min():.3f}")
                print(f"    最大: {df_all[col].max():.3f}")
        
    except Exception as e:
        print(f"✗ エラーが発生しました: {e}")
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("テスト完了")
    print("=" * 80)

if __name__ == "__main__":
    test_real_csv_data()