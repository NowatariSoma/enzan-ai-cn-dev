#!/usr/bin/env python3
"""
StreamlitとAPIの実装を詳細に比較し、同一の値が返されることを確認
"""

import sys
import os
import requests
from pathlib import Path

# パス追加
sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure')
sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev/microservices/ai_ameasure/app')

# インポート
from app.displacement_temporal_spacial_analysis import analyze_displacement
from sklearn.ensemble import RandomForestRegressor

def detailed_test_streamlit():
    """Streamlit関数の詳細テスト"""
    print("=== Detailed Streamlit Function Test ===")
    
    # 完全に同じ設定を使用
    input_folder = '/home/nowatari/repos/enzan-ai-cn-dev/data_folder/01-hokkaido-akan/main_tunnel/CN_measurement_data'
    output_folder = '/tmp/test_streamlit_detailed'
    Path(output_folder).mkdir(exist_ok=True)
    
    model_paths = {
        'final_value_prediction_model': [
            os.path.join(output_folder, 'model_final_settlement.pkl'),
            os.path.join(output_folder, 'model_final_convergence.pkl')
        ],
        'prediction_model': [
            os.path.join(output_folder, 'model_settlement.pkl'), 
            os.path.join(output_folder, 'model_convergence.pkl')
        ]
    }
    
    # 同じrandom_stateを使用
    model = RandomForestRegressor(random_state=42)
    
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Model type: {type(model)}")
    print(f"Model random_state: {model.random_state}")
    
    try:
        # analyze_displacement関数を直接呼び出し
        result = analyze_displacement(input_folder, output_folder, model_paths, model, 100, None)
        
        print(f"Raw result type: {type(result)}")
        print(f"Raw result is tuple: {isinstance(result, tuple)}")
        
        if isinstance(result, tuple):
            print(f"Result tuple length: {len(result)}")
            
            # 全ての要素を展開して確認
            for i, item in enumerate(result):
                print(f"Result[{i}] type: {type(item)}")
                if hasattr(item, 'keys'):
                    print(f"Result[{i}] keys: {list(item.keys())}")
                elif hasattr(item, 'shape'):
                    print(f"Result[{i}] shape: {item.shape}")
            
            if len(result) >= 3:
                df_all, training_metrics, scatter_data = result
                
                print(f"\n--- Streamlit Results ---")
                print(f"df_all shape: {df_all.shape if hasattr(df_all, 'shape') else 'No shape'}")
                print(f"training_metrics keys: {list(training_metrics.keys()) if training_metrics else 'None'}")
                print(f"scatter_data keys: {list(scatter_data.keys()) if scatter_data else 'None'}")
                
                if scatter_data:
                    return {
                        'df_all': df_all,
                        'training_metrics': training_metrics,
                        'scatter_data': scatter_data,
                        'status': 'success'
                    }
                else:
                    print("❌ scatter_data is empty")
                    return {'status': 'scatter_data_empty'}
            else:
                print(f"❌ Expected 3 results, got {len(result)}")
                return {'status': 'wrong_result_count', 'count': len(result)}
        else:
            print(f"❌ Result is not tuple: {type(result)}")
            return {'status': 'not_tuple'}
            
    except Exception as e:
        print(f"❌ ERROR in Streamlit test: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}

def detailed_test_real_api():
    """実際のAPIエンドポイントをテスト"""
    print("\n=== Real API Endpoint Test ===")
    
    url = "http://localhost:8000/api/v1/displacement-analysis/analyze-whole"
    
    payload = {
        "folder_name": "01-hokkaido-akan",
        "model_name": "Random Forest", 
        "max_distance_from_face": 100,
        "td": None
    }
    
    print(f"API URL: {url}")
    print(f"Payload: {payload}")
    
    try:
        print("Sending HTTP POST request to API...")
        response = requests.post(url, json=payload, timeout=300)  # 5分タイムアウト
        
        print(f"HTTP Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n--- Real API Results ---")
            print(f"API Status: {data.get('status')}")
            print(f"API Message: {data.get('message')}")
            
            # training_metricsチェック
            training_metrics = data.get('training_metrics', {})
            print(f"training_metrics keys: {list(training_metrics.keys())}")
            
            # scatter_dataチェック  
            scatter_data = data.get('scatter_data', {})
            print(f"scatter_data keys: {list(scatter_data.keys()) if scatter_data else 'None'}")
            
            if scatter_data and scatter_data.get('train_actual'):
                train_actual = scatter_data.get('train_actual', [])
                train_predictions = scatter_data.get('train_predictions', [])
                validate_actual = scatter_data.get('validate_actual', [])  
                validate_predictions = scatter_data.get('validate_predictions', [])
                
                print(f"Train actual length: {len(train_actual)}")
                print(f"Train predictions length: {len(train_predictions)}")
                print(f"Validate actual length: {len(validate_actual)}")
                print(f"Validate predictions length: {len(validate_predictions)}")
                
                return {
                    'training_metrics': training_metrics,
                    'scatter_data': scatter_data,
                    'status': 'success'
                }
            else:
                print("❌ Real API: scatter_data is empty or missing")
                return {'status': 'scatter_data_empty', 'scatter_data': scatter_data, 'training_metrics': training_metrics}
                
        else:
            print(f"❌ API Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return {'status': 'http_error', 'code': response.status_code, 'text': response.text}
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the API server is running on localhost:8000")
        print("   Start the API server with: cd microservices/ai_ameasure && uvicorn app.main:app --host 0.0.0.0 --port 8000")
        return {'status': 'connection_error'}
    except requests.exceptions.Timeout:
        print("❌ API request timed out (>5 minutes)")
        return {'status': 'timeout'}
    except Exception as e:
        print(f"❌ ERROR calling real API: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}

def compare_results(streamlit_result, api_result):
    """結果を詳細に比較"""
    print("\n=== Detailed Results Comparison ===")
    
    print(f"Streamlit status: {streamlit_result['status']}")
    print(f"API status: {api_result['status']}")
    
    if streamlit_result['status'] != 'success':
        print(f"❌ Streamlit test failed: {streamlit_result['status']}")
        return False
        
    if api_result['status'] != 'success':
        print(f"❌ API test failed: {api_result['status']}")
        if api_result['status'] == 'connection_error':
            print("   Make sure to start the API server before running this test!")
        elif api_result['status'] == 'scatter_data_empty':
            print("   This indicates the scatter_data fix is not working in the real API!")
        return False
    
    s_scatter = streamlit_result['scatter_data']
    a_scatter = api_result['scatter_data']
    
    print(f"Streamlit scatter_data length: train={len(s_scatter.get('train_actual', []))}, validate={len(s_scatter.get('validate_actual', []))}")
    print(f"API scatter_data length: train={len(a_scatter.get('train_actual', []))}, validate={len(a_scatter.get('validate_actual', []))}")
    
    # データ長の比較
    s_train_len = len(s_scatter.get('train_actual', []))
    a_train_len = len(a_scatter.get('train_actual', []))
    s_val_len = len(s_scatter.get('validate_actual', []))
    a_val_len = len(a_scatter.get('validate_actual', []))
    
    length_match = (s_train_len == a_train_len) and (s_val_len == a_val_len)
    print(f"Data length match: {length_match}")
    
    if length_match and s_train_len > 0:
        # 実際の値を比較（最初の10個）
        s_train_actual = s_scatter['train_actual'][:10]
        a_train_actual = a_scatter['train_actual'][:10]
        
        values_match = s_train_actual == a_train_actual
        print(f"First 10 train actual values match: {values_match}")
        
        if values_match:
            # 予測値も比較
            s_train_pred = s_scatter['train_predictions'][:10]
            a_train_pred = a_scatter['train_predictions'][:10]
            
            predictions_match = s_train_pred == a_train_pred
            print(f"First 10 train prediction values match: {predictions_match}")
            
            if predictions_match:
                print("✅ SUCCESS: Streamlit and Real API return identical values!")
                return True
            else:
                print("❌ FAIL: Prediction values do not match")
                print(f"Streamlit predictions: {s_train_pred}")
                print(f"API predictions: {a_train_pred}")
                return False
        else:
            print("❌ FAIL: Actual values do not match")  
            print(f"Streamlit actual: {s_train_actual}")
            print(f"API actual: {a_train_actual}")
            return False
    else:
        print("❌ FAIL: Data lengths do not match or data is empty")
        print(f"  Streamlit: train={s_train_len}, validate={s_val_len}")
        print(f"  API: train={a_train_len}, validate={a_val_len}")
        return False

def main():
    """メイン実行"""
    print("Comprehensive test to ensure Streamlit and Real API return identical values\n")
    print("⚠️  IMPORTANT: Make sure the API server is running on localhost:8000")
    print("   Start with: cd microservices/ai_ameasure && uvicorn app.main:app --host 0.0.0.0 --port 8000\n")
    
    # Streamlitテスト
    streamlit_result = detailed_test_streamlit()
    
    # 実際のAPIテスト  
    api_result = detailed_test_real_api()
    
    # 結果比較
    success = compare_results(streamlit_result, api_result)
    
    if success:
        print("\n🎉 完了！StreamlitとReal APIで同一の値が返されています。")
        print("   修正が正常に適用され、実際のAPIエンドポイントでも動作しています。")
    else:
        print("\n❌ まだ差異があります。")
        if api_result.get('status') == 'connection_error':
            print("   APIサーバーを起動してから再実行してください。")
        elif api_result.get('status') == 'scatter_data_empty':
            print("   実際のAPIでscatter_dataが空です。修正が正しく適用されていない可能性があります。")
        else:
            print("   追加の修正が必要です。")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)