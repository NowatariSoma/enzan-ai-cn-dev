#!/usr/bin/env python3
"""
実際のデータを使用したデータセット比較テスト
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Streamlitアプリのパスを追加
sys.path.insert(0, '/home/nowatari/repos/tomosigoto/enzan-ai-cn-dev/ai_ameasure/app')

# FastAPIアプリのパスを追加  
sys.path.insert(0, '/home/nowatari/repos/tomosigoto/enzan-ai-cn-dev/microservices/ai_ameasure')

def main():
    print("🚀 実際のデータを使用したデータセット比較テスト開始")
    
    # 1. Streamlitアプリからのインポート
    try:
        from displacement_temporal_spacial_analysis import (
            create_dataset as streamlit_create_dataset,
            generate_additional_info_df as streamlit_generate_additional_info_df,
            generate_dataframes as streamlit_generate_dataframes
        )
        from displacement import DATE, CYCLE_NO, SECTION_TD, FACE_TD, TD_NO, CONVERGENCES, SETTLEMENTS, STA, DISTANCE_FROM_FACE, DAYS_FROM_START, DIFFERENCE_FROM_FINAL_CONVERGENCES, DIFFERENCE_FROM_FINAL_SETTLEMENTS
        print("✅ Streamlitアプリのモジュールをインポート成功")
    except ImportError as e:
        print(f"❌ Streamlitインポートエラー: {e}")
        return
    
    # 2. FastAPIアプリからのインポート
    try:
        from app.api.endpoints.measurements import (
            create_dataset as fastapi_create_dataset,
            generate_additional_info_df as fastapi_generate_additional_info_df
        )
        from app.core.csv_loader import CSVDataLoader
        print("✅ FastAPIアプリのモジュールをインポート成功")
    except ImportError as e:
        print(f"❌ FastAPIインポートエラー: {e}")
        return
    
    # 3. 実際のデータパス設定
    folder_name = "01-hokkaido-akan"
    max_distance_from_face = 100
    
    print(f"\n📁 データフォルダ: {folder_name}")
    print(f"📏 最大距離: {max_distance_from_face}m")
    
    # 4. Streamlitアプリの方法でデータを生成
    print(f"\n{'='*60}")
    print("🔄 STREAMLIT アプリのデータ生成")
    print(f"{'='*60}")
    
    try:
        # Streamlitアプリ用のパス（data_folderを使用）
        input_folder = Path('/home/nowatari/repos/tomosigoto/enzan-ai-cn-dev/data_folder') / folder_name / 'main_tunnel' / 'CN_measurement_data'
        print(f"📂 入力フォルダ: {input_folder}")
        
        # CSVファイルのリスト取得
        measurement_a_csvs = list((input_folder / 'measurements_A').glob('*.csv'))
        print(f"📊 測定CSVファイル数: {len(measurement_a_csvs)}")
        
        if not measurement_a_csvs:
            print("❌ 測定CSVファイルが見つかりません")
            return
        
        # 追加情報ファイル
        cycle_support_csv = input_folder / 'cycle_support' / 'cycle_support.csv'
        observation_of_face_csv = input_folder / 'observation_of_face' / 'observation_of_face.csv'
        
        print(f"📋 Cycle support CSV: {cycle_support_csv.exists()}")
        print(f"📋 Observation face CSV: {observation_of_face_csv.exists()}")
        
        if not (cycle_support_csv.exists() and observation_of_face_csv.exists()):
            print("❌ 必要なCSVファイルが見つかりません")
            return
        
        # Streamlitアプリの処理
        df_additional_info_streamlit = streamlit_generate_additional_info_df(
            str(cycle_support_csv), str(observation_of_face_csv)
        )
        if STA in df_additional_info_streamlit.columns:
            df_additional_info_streamlit.drop(columns=[STA], inplace=True)
        
        df_all_streamlit, _, _, _, settlements, convergences = streamlit_generate_dataframes(
            [str(f) for f in measurement_a_csvs], max_distance_from_face
        )
        
        print(f"📊 Streamlit df_all shape: {df_all_streamlit.shape}")
        print(f"📊 Streamlit additional_info shape: {df_additional_info_streamlit.shape}")
        
        # Streamlitのcreate_dataset実行
        print("🔄 Streamlit create_dataset実行中...")
        streamlit_result = streamlit_create_dataset(df_all_streamlit, df_additional_info_streamlit)
        print(f"✅ Streamlit結果タイプ: {type(streamlit_result)}")
        
    except Exception as e:
        print(f"❌ Streamlitデータ生成エラー: {e}")
        import traceback
        traceback.print_exc()
        streamlit_result = None
    
    # 5. FastAPIアプリの方法でデータを生成  
    print(f"\n{'='*60}")
    print("🔄 FASTAPI アプリのデータ生成")
    print(f"{'='*60}")
    
    try:
        # FastAPIの設定に合わせてデータパス調整
        data_folder = Path('/home/nowatari/repos/tomosigoto/enzan-ai-cn-dev/data_folder')
        input_folder_fastapi = data_folder / folder_name / "main_tunnel" / "CN_measurement_data"
        
        # CSVローダーでデータ読み込み
        csv_loader = CSVDataLoader()
        df_all_fastapi = csv_loader.load_all_measurement_data(data_folder, folder_name)
        
        print(f"📊 FastAPI df_all shape: {df_all_fastapi.shape}")
        
        # 追加情報ファイル
        cycle_support_csv_fastapi = input_folder_fastapi / 'cycle_support' / 'cycle_support.csv'
        observation_of_face_csv_fastapi = input_folder_fastapi / 'observation_of_face' / 'observation_of_face.csv'
        
        if cycle_support_csv_fastapi.exists() and observation_of_face_csv_fastapi.exists():
            df_additional_info_fastapi = fastapi_generate_additional_info_df(
                cycle_support_csv_fastapi, observation_of_face_csv_fastapi
            )
            print(f"📊 FastAPI additional_info shape: {df_additional_info_fastapi.shape}")
            
            # FastAPIのcreate_dataset実行
            print("🔄 FastAPI create_dataset実行中...")
            fastapi_result = fastapi_create_dataset(df_all_fastapi, df_additional_info_fastapi)
            print(f"✅ FastAPI結果タイプ: {type(fastapi_result)}")
        else:
            print("❌ FastAPI用の追加情報ファイルが見つかりません")
            fastapi_result = None
        
    except Exception as e:
        print(f"❌ FastAPIデータ生成エラー: {e}")
        import traceback
        traceback.print_exc()
        fastapi_result = None
    
    # 6. 結果比較
    print(f"\n{'='*60}")
    print("📊 結果比較")
    print(f"{'='*60}")
    
    if streamlit_result is None:
        print("❌ Streamlit結果がNullです")
    elif fastapi_result is None:
        print("❌ FastAPI結果がNullです")
    else:
        print("✅ 両方の関数が正常に実行されました")
        
        # 詳細比較
        if (isinstance(streamlit_result, tuple) and len(streamlit_result) == 2 and 
            isinstance(fastapi_result, tuple) and len(fastapi_result) == 2):
            
            settlement_s, convergence_s = streamlit_result
            settlement_f, convergence_f = fastapi_result
            
            print("\n📈 Settlement データ比較:")
            if isinstance(settlement_s, tuple) and isinstance(settlement_f, tuple):
                print(f"   🔹 Streamlit Settlement: {len(settlement_s)} 要素")
                print(f"   🔹 FastAPI Settlement: {len(settlement_f)} 要素")
                
                if len(settlement_s) >= 3 and len(settlement_f) >= 3:
                    df_s, x_cols_s, y_col_s = settlement_s[:3]
                    df_f, x_cols_f, y_col_f = settlement_f[:3]
                    
                    print(f"      📊 Streamlit DF形状: {df_s.shape if hasattr(df_s, 'shape') else 'N/A'}")
                    print(f"      📊 FastAPI DF形状: {df_f.shape if hasattr(df_f, 'shape') else 'N/A'}")
                    print(f"      📝 Streamlit X列数: {len(x_cols_s) if x_cols_s else 0}")
                    print(f"      📝 FastAPI X列数: {len(x_cols_f) if x_cols_f else 0}")
                    print(f"      🎯 Streamlit Y列: {y_col_s}")
                    print(f"      🎯 FastAPI Y列: {y_col_f}")
                    
                    # 列名の比較
                    if x_cols_s and x_cols_f:
                        common_cols = set(x_cols_s) & set(x_cols_f)
                        streamlit_only = set(x_cols_s) - set(x_cols_f)
                        fastapi_only = set(x_cols_f) - set(x_cols_s)
                        
                        print(f"      🤝 共通X列: {len(common_cols)}個")
                        if streamlit_only:
                            print(f"      🔸 Streamlitのみ: {list(streamlit_only)[:5]}...")  # 最初の5個のみ表示
                        if fastapi_only:
                            print(f"      🔹 FastAPIのみ: {list(fastapi_only)[:5]}...")  # 最初の5個のみ表示
            
            print("\n📉 Convergence データ比較:")
            if isinstance(convergence_s, tuple) and isinstance(convergence_f, tuple):
                print(f"   🔹 Streamlit Convergence: {len(convergence_s)} 要素")  
                print(f"   🔹 FastAPI Convergence: {len(convergence_f)} 要素")
                
                if len(convergence_s) >= 3 and len(convergence_f) >= 3:
                    df_s, x_cols_s, y_col_s = convergence_s[:3]
                    df_f, x_cols_f, y_col_f = convergence_f[:3]
                    
                    print(f"      📊 Streamlit DF形状: {df_s.shape if hasattr(df_s, 'shape') else 'N/A'}")
                    print(f"      📊 FastAPI DF形状: {df_f.shape if hasattr(df_f, 'shape') else 'N/A'}")
                    print(f"      📝 Streamlit X列数: {len(x_cols_s) if x_cols_s else 0}")
                    print(f"      📝 FastAPI X列数: {len(x_cols_f) if x_cols_f else 0}")
                    print(f"      🎯 Streamlit Y列: {y_col_s}")
                    print(f"      🎯 FastAPI Y列: {y_col_f}")
                    
                    # 列名の比較
                    if x_cols_s and x_cols_f:
                        common_cols = set(x_cols_s) & set(x_cols_f)
                        streamlit_only = set(x_cols_s) - set(x_cols_f)
                        fastapi_only = set(x_cols_f) - set(x_cols_s)
                        
                        print(f"      🤝 共通X列: {len(common_cols)}個")
                        if streamlit_only:
                            print(f"      🔸 Streamlitのみ: {list(streamlit_only)[:5]}...")  # 最初の5個のみ表示
                        if fastapi_only:
                            print(f"      🔹 FastAPIのみ: {list(fastapi_only)[:5]}...")  # 最初の5個のみ表示
        
        # 同一性の判定
        print(f"\n{'='*40}")
        print("🔍 同一性判定")
        print(f"{'='*40}")
        
        try:
            if streamlit_result == fastapi_result:
                print("🎉 結果は完全に同一です！")
            else:
                print("⚠️  結果に差異があります")
                
                # より詳細な比較
                if (isinstance(streamlit_result, tuple) and len(streamlit_result) == 2 and 
                    isinstance(fastapi_result, tuple) and len(fastapi_result) == 2):
                    
                    settlement_s, convergence_s = streamlit_result
                    settlement_f, convergence_f = fastapi_result
                    
                    # Settlement比較
                    if (isinstance(settlement_s, tuple) and len(settlement_s) >= 3 and
                        isinstance(settlement_f, tuple) and len(settlement_f) >= 3):
                        df_s, x_cols_s, y_col_s = settlement_s[:3]
                        df_f, x_cols_f, y_col_f = settlement_f[:3]
                        
                        shape_same = (hasattr(df_s, 'shape') and hasattr(df_f, 'shape') and df_s.shape == df_f.shape)
                        y_same = (y_col_s == y_col_f)
                        x_same = (set(x_cols_s) == set(x_cols_f) if x_cols_s and x_cols_f else False)
                        
                        print(f"Settlement: 形状同じ={shape_same}, Y列同じ={y_same}, X列同じ={x_same}")
                    
                    # Convergence比較
                    if (isinstance(convergence_s, tuple) and len(convergence_s) >= 3 and
                        isinstance(convergence_f, tuple) and len(convergence_f) >= 3):
                        df_s, x_cols_s, y_col_s = convergence_s[:3]
                        df_f, x_cols_f, y_col_f = convergence_f[:3]
                        
                        shape_same = (hasattr(df_s, 'shape') and hasattr(df_f, 'shape') and df_s.shape == df_f.shape)
                        y_same = (y_col_s == y_col_f)
                        x_same = (set(x_cols_s) == set(x_cols_f) if x_cols_s and x_cols_f else False)
                        
                        print(f"Convergence: 形状同じ={shape_same}, Y列同じ={y_same}, X列同じ={x_same}")
                        
        except Exception as e:
            print(f"⚠️  比較中にエラー: {e}")
    
    print(f"\n{'='*60}")
    print("✨ 実データ比較テスト完了")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 