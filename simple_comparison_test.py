#!/usr/bin/env python3
"""
簡単なデータセット比較テスト
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
    print("📊 データセット比較テスト開始")
    
    # 1. Streamlitアプリからのインポート
    try:
        from displacement_temporal_spacial_analysis import create_dataset as streamlit_create_dataset
        from displacement import TD_NO, DISTANCE_FROM_FACE, DAYS_FROM_START
        print("✅ Streamlitアプリのモジュールをインポート成功")
    except ImportError as e:
        print(f"❌ Streamlitインポートエラー: {e}")
        return
    
    # 2. FastAPIアプリからのインポート
    try:
        from app.api.endpoints.measurements import create_dataset as fastapi_create_dataset
        print("✅ FastAPIアプリのモジュールをインポート成功")
    except ImportError as e:
        print(f"❌ FastAPIインポートエラー: {e}")
        return
    
    # 3. 簡単なテストデータを作成
    print("\n📋 テストデータ作成中...")
    
    # 基本的なDataFrameを作成
    test_data = {
        'TD(m)': [100, 150, 200, 250, 300],
        '切羽からの距離': [10, 20, 30, 40, 50],  
        '計測経過日数': [1, 5, 10, 15, 20],
        '計測日時': pd.date_range('2023-01-01', periods=5),
        'ｻｲｸﾙNo': [1, 2, 3, 4, 5],
        'STA': ['STA1'] * 5,
        '実TD': [100, 150, 200, 250, 300],
        '切羽TD': [110, 170, 230, 290, 350],
        '沈下量1': [1.2, 2.3, 3.1, 4.0, 5.2],
        '沈下量2': [1.0, 2.0, 2.8, 3.5, 4.8],
        '沈下量3': [0.8, 1.8, 2.5, 3.2, 4.5],
        '変位量A': [2.1, 3.2, 4.1, 5.0, 6.2],
        '変位量B': [1.9, 3.0, 3.8, 4.7, 5.9],
        '変位量C': [1.7, 2.8, 3.6, 4.5, 5.7],
        '最終沈下量との差分1': [0.5, 1.0, 1.5, 2.0, 2.5],
        '最終沈下量との差分2': [0.4, 0.9, 1.4, 1.9, 2.4], 
        '最終沈下量との差分3': [0.3, 0.8, 1.3, 1.8, 2.3],
        '最終変位量との差分A': [0.8, 1.3, 1.8, 2.3, 2.8],
        '最終変位量との差分B': [0.7, 1.2, 1.7, 2.2, 2.7],
        '最終変位量との差分C': [0.6, 1.1, 1.6, 2.1, 2.6]
    }
    
    df_test = pd.DataFrame(test_data)
    
    # 追加情報DataFrameを作成
    additional_info = {
        'ｻｲｸﾙ': [1, 2, 3, 4, 5],  # Streamlitアプリで必要な列名
        '支保寸法': [1.0, 1.1, 1.2, 1.3, 1.4],
        '吹付厚': [10, 12, 14, 16, 18],
        'ﾛｯｸﾎﾞﾙﾄ数': [5, 6, 7, 8, 9],
        'ﾛｯｸﾎﾞﾙﾄ長': [3.0, 3.2, 3.4, 3.6, 3.8],
        '覆工厚': [20, 22, 24, 26, 28],
        '土被り高さ': [50, 55, 60, 65, 70],
        '岩石グループ': [1, 2, 1, 2, 1],
        '岩石名コード': [10, 20, 10, 20, 10],
        '加重平均評価点': [3.5, 4.0, 3.8, 4.2, 3.9],
        '支保工種': ['A型', 'B型', 'A型', 'B型', 'A型'],
        '支保パターン2': ['Pa1', 'Pb1', 'Pa2', 'Pb2', 'Pa3']
    }
    df_additional_info = pd.DataFrame(additional_info)
    
    print(f"   テストデータ形状: {df_test.shape}")
    print(f"   追加情報形状: {df_additional_info.shape}")
    
    # 4. 両方の関数を実行
    print("\n🔄 Streamlit create_dataset実行中...")
    try:
        streamlit_result = streamlit_create_dataset(df_test, df_additional_info)
        print(f"   Streamlit結果タイプ: {type(streamlit_result)}")
        if isinstance(streamlit_result, tuple) and len(streamlit_result) == 2:
            settlement_s, convergence_s = streamlit_result
            print(f"   Settlement結果: {type(settlement_s)}")
            print(f"   Convergence結果: {type(convergence_s)}")
    except Exception as e:
        print(f"   ❌ Streamlitエラー: {e}")
        streamlit_result = None
    
    print("\n🔄 FastAPI create_dataset実行中...")
    try:
        fastapi_result = fastapi_create_dataset(df_test, df_additional_info)
        print(f"   FastAPI結果タイプ: {type(fastapi_result)}")
        if isinstance(fastapi_result, tuple) and len(fastapi_result) == 2:
            settlement_f, convergence_f = fastapi_result
            print(f"   Settlement結果: {type(settlement_f)}")
            print(f"   Convergence結果: {type(convergence_f)}")
    except Exception as e:
        print(f"   ❌ FastAPIエラー: {e}")
        fastapi_result = None
    
    # 5. 結果比較
    print("\n📊 結果比較:")
    
    if streamlit_result is None:
        print("   ❌ Streamlit結果がNullです")
    elif fastapi_result is None:
        print("   ❌ FastAPI結果がNullです")
    else:
        print("   ✅ 両方の関数が正常に実行されました")
        
        # 詳細比較
        if (isinstance(streamlit_result, tuple) and len(streamlit_result) == 2 and 
            isinstance(fastapi_result, tuple) and len(fastapi_result) == 2):
            
            settlement_s, convergence_s = streamlit_result
            settlement_f, convergence_f = fastapi_result
            
            print("\n   📈 Settlement データ比較:")
            if isinstance(settlement_s, tuple) and isinstance(settlement_f, tuple):
                print(f"      Streamlit Settlement: {len(settlement_s)} 要素")
                print(f"      FastAPI Settlement: {len(settlement_f)} 要素")
                
                if len(settlement_s) >= 3 and len(settlement_f) >= 3:
                    df_s, x_cols_s, y_col_s = settlement_s[:3]
                    df_f, x_cols_f, y_col_f = settlement_f[:3]
                    
                    print(f"         Streamlit DF形状: {df_s.shape if hasattr(df_s, 'shape') else 'N/A'}")
                    print(f"         FastAPI DF形状: {df_f.shape if hasattr(df_f, 'shape') else 'N/A'}")
                    print(f"         Streamlit X列数: {len(x_cols_s) if x_cols_s else 0}")
                    print(f"         FastAPI X列数: {len(x_cols_f) if x_cols_f else 0}")
                    print(f"         Streamlit Y列: {y_col_s}")
                    print(f"         FastAPI Y列: {y_col_f}")
            
            print("\n   📉 Convergence データ比較:")
            if isinstance(convergence_s, tuple) and isinstance(convergence_f, tuple):
                print(f"      Streamlit Convergence: {len(convergence_s)} 要素")  
                print(f"      FastAPI Convergence: {len(convergence_f)} 要素")
                
                if len(convergence_s) >= 3 and len(convergence_f) >= 3:
                    df_s, x_cols_s, y_col_s = convergence_s[:3]
                    df_f, x_cols_f, y_col_f = convergence_f[:3]
                    
                    print(f"         Streamlit DF形状: {df_s.shape if hasattr(df_s, 'shape') else 'N/A'}")
                    print(f"         FastAPI DF形状: {df_f.shape if hasattr(df_f, 'shape') else 'N/A'}")
                    print(f"         Streamlit X列数: {len(x_cols_s) if x_cols_s else 0}")
                    print(f"         FastAPI X列数: {len(x_cols_f) if x_cols_f else 0}")
                    print(f"         Streamlit Y列: {y_col_s}")
                    print(f"         FastAPI Y列: {y_col_f}")
        
        # 同一性の判定
        if streamlit_result == fastapi_result:
            print("\n   ✅ 結果は完全に同一です")
        else:
            print("\n   ⚠️  結果に差異があります")
    
    print("\n✨ 比較テスト完了")

if __name__ == "__main__":
    main() 