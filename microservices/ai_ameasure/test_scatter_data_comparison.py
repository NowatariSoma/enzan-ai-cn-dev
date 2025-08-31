#!/usr/bin/env python3
"""
APIとStreamlitのscatter_data結果を比較するテストコード
"""

import os
import sys
import json
from pathlib import Path

# StreamlitとAPI両方のパスを追加
sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure')
sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev/microservices/ai_ameasure/app')

# Streamlit側のインポート
from app.displacement_temporal_spacial_analysis import analyze_displacement as streamlit_analyze

# API側のインポート
from api.endpoints.displacement_analysis import get_models

# 学習モデル
from sklearn.ensemble import RandomForestRegressor

def test_streamlit_function(test_data_path):
    """Streamlitのanalyze_displacement関数を直接テスト"""
    print("=== Streamlit Function Test ===")
    
    # テストパラメータ
    input_folder = test_data_path
    output_folder = "/tmp/test_output_streamlit"
    
    # 出力フォルダを作成
    Path(output_folder).mkdir(exist_ok=True)
    
    # モデルパス設定
    model_paths = {
        "final_value_prediction_model": [
            os.path.join(output_folder, "model_final_settlement.pkl"),
            os.path.join(output_folder, "model_final_convergence.pkl")
        ],
        "prediction_model": [
            os.path.join(output_folder, "model_settlement.pkl"), 
            os.path.join(output_folder, "model_convergence.pkl")
        ]
    }
    
    # モデル作成
    model = RandomForestRegressor(random_state=42)
    
    try:
        result = streamlit_analyze(
            input_folder,
            output_folder,
            model_paths,
            model,
            max_distance_from_face=100,
            td=None
        )
        
        print(f"Streamlit result type: {type(result)}")
        
        if isinstance(result, tuple):
            print(f"Result tuple length: {len(result)}")
            
            if len(result) == 3:
                df_all, training_metrics, scatter_data = result
                print(f"Training metrics keys: {list(training_metrics.keys()) if training_metrics else 'None'}")
                print(f"Scatter data keys: {list(scatter_data.keys()) if scatter_data else 'None'}")
                if scatter_data:
                    print(f"Train actual length: {len(scatter_data.get('train_actual', []))}")
                    print(f"Train predictions length: {len(scatter_data.get('train_predictions', []))}")
                    print(f"Validate actual length: {len(scatter_data.get('validate_actual', []))}")
                    print(f"Validate predictions length: {len(scatter_data.get('validate_predictions', []))}")
                    
                    # 散布図データの一部を表示
                    if scatter_data.get('train_actual'):
                        print(f"First 5 train actual values: {scatter_data['train_actual'][:5]}")
                        print(f"First 5 train prediction values: {scatter_data['train_predictions'][:5]}")
                    
                    return scatter_data
                else:
                    print("Warning: scatter_data is empty or None")
                    return {}
            elif len(result) == 2:
                df_all, training_metrics = result
                print("Result contains only df_all and training_metrics")
                return {}
            else:
                print(f"Unexpected result tuple length: {len(result)}")
                return {}
        else:
            print("Result is not a tuple")
            return {}
            
    except Exception as e:
        print(f"Streamlit function error: {e}")
        import traceback
        traceback.print_exc()
        return {}

def test_api_implementation(test_data_path):
    """API実装での同じ処理をテスト"""
    print("\n=== API Implementation Test ===")
    
    # API側と同じパラメータでテスト
    input_folder = test_data_path
    output_folder = "/tmp/test_output_api"
    
    # 出力フォルダを作成
    Path(output_folder).mkdir(exist_ok=True)
    
    # モデルパス設定（APIと同じ）
    model_paths = {
        "final_value_prediction_model": [
            Path(output_folder) / "model_final_settlement.pkl",
            Path(output_folder) / "model_final_convergence.pkl"
        ],
        "prediction_model": [
            Path(output_folder) / "model_settlement.pkl", 
            Path(output_folder) / "model_convergence.pkl"
        ]
    }
    
    # APIと同じモデル取得方法
    models = get_models()
    model = models.get("Random Forest")
    
    try:
        print(f"DEBUG: Calling streamlit_analyze with arguments:")
        print(f"  input_folder: {input_folder}")
        print(f"  output_folder: {output_folder}")
        print(f"  model_paths: {model_paths}")
        print(f"  model: {type(model)}")
        print(f"  max_distance_from_face: 100")
        print(f"  td: None")
        
        # APIと同じanalyze_displacement関数を呼び出し
        result = streamlit_analyze(
            str(input_folder),
            str(output_folder), 
            model_paths,
            model,
            max_distance_from_face=100,
            td=None
        )
        
        # APIと同じ戻り値処理
        if isinstance(result, tuple):
            if len(result) == 3:
                df_all, training_metrics, scatter_data = result
            elif len(result) == 2:
                df_all, training_metrics = result
                scatter_data = {}
            else:
                df_all = result[0] if result else None
                training_metrics = {}
                scatter_data = {}
        else:
            df_all = result
            training_metrics = {}
            scatter_data = {}
        
        print(f"API-style processing result:")
        print(f"Training metrics keys: {list(training_metrics.keys()) if training_metrics else 'None'}")
        print(f"Scatter data keys: {list(scatter_data.keys()) if scatter_data else 'None'}")
        if scatter_data:
            print(f"Train actual length: {len(scatter_data.get('train_actual', []))}")
            print(f"Train predictions length: {len(scatter_data.get('train_predictions', []))}")
            print(f"Validate actual length: {len(scatter_data.get('validate_actual', []))}")
            print(f"Validate predictions length: {len(scatter_data.get('validate_predictions', []))}")
            
            return scatter_data
        else:
            print("Warning: scatter_data is empty after API-style processing")
            return {}
            
    except Exception as e:
        print(f"API implementation test error: {e}")
        import traceback
        traceback.print_exc()
        return {}

def compare_results(streamlit_scatter, api_scatter):
    """結果を比較"""
    print("\n=== Results Comparison ===")
    
    print("Streamlit scatter_data:")
    print(f"  Keys: {list(streamlit_scatter.keys())}")
    for key, value in streamlit_scatter.items():
        if isinstance(value, list):
            print(f"  {key}: {len(value)} items")
        else:
            print(f"  {key}: {value}")
    
    print("\nAPI scatter_data:")  
    print(f"  Keys: {list(api_scatter.keys())}")
    for key, value in api_scatter.items():
        if isinstance(value, list):
            print(f"  {key}: {len(value)} items")
        else:
            print(f"  {key}: {value}")
    
    # 数値比較
    if streamlit_scatter and api_scatter:
        keys_to_compare = ['train_actual', 'train_predictions', 'validate_actual', 'validate_predictions']
        all_match = True
        
        for key in keys_to_compare:
            s_data = streamlit_scatter.get(key, [])
            a_data = api_scatter.get(key, [])
            
            if len(s_data) != len(a_data):
                print(f"  {key}: Length mismatch - Streamlit: {len(s_data)}, API: {len(a_data)}")
                all_match = False
            elif s_data == a_data:
                print(f"  {key}: ✅ Match ({len(s_data)} items)")
            else:
                print(f"  {key}: ❌ Data mismatch ({len(s_data)} items)")
                all_match = False
                
        if all_match:
            print("\n🎉 All scatter data matches between Streamlit and API!")
        else:
            print("\n⚠️ Scatter data does not match between Streamlit and API")
    else:
        print("\n❌ One or both scatter_data results are empty")

if __name__ == "__main__":
    print("Testing scatter_data consistency between Streamlit and API implementations\n")
    
    # テスト用データフォルダの存在確認
    test_data_path = "/home/nowatari/repos/enzan-ai-cn-dev/data_folder/01-hokkaido-akan/main_tunnel/CN_measurement_data"
    if not os.path.exists(test_data_path):
        print(f"Warning: Test data path not found: {test_data_path}")
        print("Using alternative path...")
        # データフォルダを探す
        alternative_paths = [
            "/home/nowatari/repos/enzan-ai-cn-dev/data_folder/01-hokkaido-atsuga/main_tunnel/CN_measurement_data",
            "/home/nowatari/repos/enzan-ai-cn-dev/data/01-hokkaido-akan/main_tunnel/CN_measurement_data",
            "/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure/data/01-hokkaido-akan/main_tunnel/CN_measurement_data"
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"Found alternative data path: {alt_path}")
                test_data_path = alt_path
                break
        else:
            print("No valid test data path found. Exiting...")
            sys.exit(1)
    
    print(f"Using test data path: {test_data_path}")
    
    # テスト実行
    streamlit_result = test_streamlit_function(test_data_path)
    api_result = test_api_implementation(test_data_path)
    
    # 結果比較
    compare_results(streamlit_result, api_result)