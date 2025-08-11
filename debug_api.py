#!/usr/bin/env python3
"""
FastAPIのmake-datasetエンドポイントを直接テストしてエラーを詳細に確認（修正版）
"""

import sys
import traceback
from pathlib import Path

# FastAPIアプリのパスを追加  
sys.path.insert(0, '/home/nowatari/repos/tomosigoto/enzan-ai-cn-dev/microservices/ai_ameasure')

try:
    print("🔧 FastAPIモジュールをインポート中...")
    
    from app.api.endpoints.measurements import (
        create_dataset as fastapi_create_dataset,
        generate_additional_info_df as fastapi_generate_additional_info_df
    )
    from app.core.csv_loader import CSVDataLoader
    print("✅ FastAPIモジュールのインポート成功")
    
    # 実際にデータを処理してみる
    print("📊 データ処理開始...")
    
    folder_name = "01-hokkaido-akan"
    max_distance_from_face = 100
    
    # CSVローダーでデータ読み込み
    csv_loader = CSVDataLoader()
    data_folder = Path('/home/nowatari/repos/tomosigoto/enzan-ai-cn-dev/data_folder')
    
    print(f"📂 データフォルダ: {data_folder}")
    print(f"📁 サブフォルダ: {folder_name}")
    
    # 正しい方法：measurements_A CSVファイルのリストを取得してgenerate_dataframesを使用
    input_folder = data_folder / folder_name / "main_tunnel" / "CN_measurement_data"
    measurements_path = input_folder / "measurements_A"
    measurement_a_csvs = list(measurements_path.glob("*.csv"))
    
    print(f"📊 CSVファイル数: {len(measurement_a_csvs)}")
    
    if not measurement_a_csvs:
        print("❌ CSVファイルが見つかりません")
        exit(1)
    
    df_all_fastapi, _, _, _, _, _ = csv_loader.generate_dataframes(measurement_a_csvs, max_distance_from_face)
    print(f"📊 読み込み完了 - df_all shape: {df_all_fastapi.shape}")
    
    # 追加情報ファイル
    cycle_support_csv_fastapi = input_folder / 'cycle_support' / 'cycle_support.csv'
    observation_of_face_csv_fastapi = input_folder / 'observation_of_face' / 'observation_of_face.csv'
    
    print(f"📋 Cycle support CSV: {cycle_support_csv_fastapi.exists()}")
    print(f"📋 Observation face CSV: {observation_of_face_csv_fastapi.exists()}")
    
    if cycle_support_csv_fastapi.exists() and observation_of_face_csv_fastapi.exists():
        df_additional_info_fastapi = fastapi_generate_additional_info_df(
            cycle_support_csv_fastapi, observation_of_face_csv_fastapi
        )
        print(f"📊 Additional info shape: {df_additional_info_fastapi.shape}")
        
        # FastAPIのcreate_dataset実行
        print("🔄 FastAPI create_dataset実行中...")
        fastapi_result = fastapi_create_dataset(df_all_fastapi, df_additional_info_fastapi)
        
        if fastapi_result and len(fastapi_result) == 2:
            settlement_data, convergence_data = fastapi_result
            
            print(f"✅ FastAPI処理成功!")
            print(f"📊 Settlement データ形状: {len(settlement_data) if settlement_data else 0}")
            print(f"📊 Convergence データ形状: {len(convergence_data) if convergence_data else 0}")
            
            # 最初のサンプルデータを確認
            if settlement_data and len(settlement_data) > 0:
                if isinstance(settlement_data, tuple) and len(settlement_data) == 3:
                    df_s, x_cols_s, y_col_s = settlement_data
                    print(f"🔍 Settlement タプル形式: DF shape={df_s.shape}, X列数={len(x_cols_s)}, Y列={y_col_s}")
                    # 辞書リストに変換
                    settlement_records = df_s.to_dict('records') if not df_s.empty else []
                    print(f"🔍 Settlement 辞書リスト形状: {len(settlement_records)}")
                    if settlement_records:
                        print(f"🔍 Settlement サンプルキー: {list(settlement_records[0].keys())[:10]}")
                else:
                    print(f"🔍 Settlement sample keys: {list(settlement_data[0].keys())[:10] if settlement_data else []}")
                
            if convergence_data and len(convergence_data) > 0:
                if isinstance(convergence_data, tuple) and len(convergence_data) == 3:
                    df_c, x_cols_c, y_col_c = convergence_data
                    print(f"🔍 Convergence タプル形式: DF shape={df_c.shape}, X列数={len(x_cols_c)}, Y列={y_col_c}")
                    # 辞書リストに変換
                    convergence_records = df_c.to_dict('records') if not df_c.empty else []
                    print(f"🔍 Convergence 辞書リスト形状: {len(convergence_records)}")
                    if convergence_records:
                        print(f"🔍 Convergence サンプルキー: {list(convergence_records[0].keys())[:10]}")
                else:
                    print(f"🔍 Convergence sample keys: {list(convergence_data[0].keys())[:10] if convergence_data else []}")
                
        else:
            print(f"❌ FastAPI結果が期待される形式ではありません: {type(fastapi_result)}")
    else:
        print("❌ FastAPI用の追加情報ファイルが見つかりません")
        
except Exception as e:
    print(f"❌ エラーが発生しました: {e}")
    print("\n詳細なトレースバック:")
    traceback.print_exc() 