#!/usr/bin/env python3
"""
元のai_ameasureとマイクロサービス版のデータソースを詳細比較するスクリプト
"""
import sys
import pandas as pd

def compare_data_sources():
    print("=" * 60)
    print("元のai_ameasureとマイクロサービス版のデータソース比較")
    print("=" * 60)
    
    # 1. 元のai_ameasureでのデータ読み込み
    print("\n1. 元のai_ameasureでのデータ読み込み...")
    sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure/app')
    from displacement_temporal_spacial_analysis import generate_dataframes
    import os

    input_folder = '/home/nowatari/repos/enzan-ai-cn-dev/data_folder/01-hokkaido-akan/main_tunnel/CN_measurement_data'
    measurement_a_csvs = [
        os.path.join(input_folder, 'measurements_A', f)
        for f in os.listdir(os.path.join(input_folder, 'measurements_A'))
        if f.endswith('.csv')
    ]

    print(f"   CSVファイル数: {len(measurement_a_csvs)}")
    
    df_all_original, dct_df_settlement, dct_df_convergence, dct_df_td, settlements, convergences = generate_dataframes(measurement_a_csvs, 100.0)
    print(f"   処理済みサンプル数: {len(df_all_original)}")
    print(f"   カラム数: {len(df_all_original.columns)}")
    print(f"   沈下量カラム: {settlements}")
    print(f"   変位量カラム: {convergences}")
    
    # 2. マイクロサービス版でのデータ確認
    print("\n2. マイクロサービス版でのデータ確認...")
    sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev/microservices/ai_ameasure')
    
    try:
        from app.core.dataframe_cache import get_dataframe_cache
        cache = get_dataframe_cache()
        cached_data = cache.get_cached_data('01-hokkaido-akan', 100.0)
        
        if cached_data:
            df_microservice = cached_data['df_all']
            print(f"   マイクロサービス版サンプル数: {len(df_microservice)}")
            print(f"   マイクロサービス版カラム数: {len(df_microservice.columns)}")
            print(f"   マイクロサービス版カラム名: {list(df_microservice.columns)}")
            
            # データの同一性確認
            print("\n3. データの同一性確認...")
            
            # サンプル数比較
            if len(df_all_original) == len(df_microservice):
                print("   ✓ サンプル数: 同じ")
            else:
                print(f"   ✗ サンプル数: 異なる (元: {len(df_all_original)}, マイクロ: {len(df_microservice)})")
                
            # カラム数比較
            if len(df_all_original.columns) == len(df_microservice.columns):
                print("   ✓ カラム数: 同じ")
            else:
                print(f"   ✗ カラム数: 異なる (元: {len(df_all_original.columns)}, マイクロ: {len(df_microservice.columns)})")
                
            # カラム名比較
            original_cols = set(df_all_original.columns)
            micro_cols = set(df_microservice.columns)
            
            if original_cols == micro_cols:
                print("   ✓ カラム名: 同じ")
            else:
                print("   ✗ カラム名: 異なる")
                print(f"      元のみ: {original_cols - micro_cols}")
                print(f"      マイクロのみ: {micro_cols - original_cols}")
                
            # 共通カラムでの値の比較（最初の10行）
            common_cols = list(original_cols & micro_cols)
            if common_cols:
                print(f"\n4. 共通カラムでの値比較（先頭10行）...")
                
                # 同じ順序でソート
                df_orig_sorted = df_all_original.sort_values(['計測日時', 'TD(m)'] if '計測日時' in common_cols and 'TD(m)' in common_cols else common_cols[:2]).reset_index(drop=True)
                df_micro_sorted = df_microservice.sort_values(['計測日時', 'TD(m)'] if '計測日時' in common_cols and 'TD(m)' in common_cols else common_cols[:2]).reset_index(drop=True)
                
                are_same = True
                for col in common_cols[:5]:  # 最初の5カラムをチェック
                    if col in df_orig_sorted.columns and col in df_micro_sorted.columns:
                        try:
                            orig_values = df_orig_sorted[col].head(10).fillna('NaN').astype(str)
                            micro_values = df_micro_sorted[col].head(10).fillna('NaN').astype(str)
                            
                            if not orig_values.equals(micro_values):
                                print(f"      ✗ {col}: 値が異なる")
                                are_same = False
                            else:
                                print(f"      ✓ {col}: 値が同じ")
                        except Exception as e:
                            print(f"      ? {col}: 比較エラー ({e})")
                            are_same = False
                
                if are_same:
                    print("   ✓ 主要な値: 同じ")
                else:
                    print("   ✗ 主要な値: 異なる")
            
        else:
            print("   マイクロサービス版: キャッシュにデータがありません")
            print("   これが問題の原因の可能性があります！")
            
    except Exception as e:
        print(f"   マイクロサービス版データ確認エラー: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "=" * 60)
    print("比較完了")
    print("=" * 60)

if __name__ == "__main__":
    compare_data_sources()