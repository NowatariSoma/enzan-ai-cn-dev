#!/usr/bin/env python3
"""
沈下量と変位量に分けたscatter_dataのテスト
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

def test_streamlit_separated_scatter_data():
    """Streamlitで分離されたscatter_dataをテスト"""
    print("=== Testing Separated Scatter Data (Streamlit) ===")
    
    input_folder = '/home/nowatari/repos/enzan-ai-cn-dev/data_folder/01-hokkaido-akan/main_tunnel/CN_measurement_data'
    output_folder = '/tmp/test_separated_streamlit'
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
    
    model = RandomForestRegressor(random_state=42)
    
    try:
        result = analyze_displacement(input_folder, output_folder, model_paths, model, 100, None)
        
        if isinstance(result, tuple) and len(result) >= 3:
            df_all, training_metrics, scatter_data = result
            
            print(f"Scatter data keys: {list(scatter_data.keys())}")
            
            # 沈下量と変位量に分かれているかチェック
            if 'settlement' in scatter_data and 'convergence' in scatter_data:
                print("✅ Scatter data successfully separated into settlement and convergence")
                
                # 各カテゴリのデータ構造をチェック
                for category in ['settlement', 'convergence']:
                    cat_data = scatter_data[category]
                    print(f"\n{category.upper()} data:")
                    print(f"  Keys: {list(cat_data.keys())}")
                    print(f"  Train actual length: {len(cat_data.get('train_actual', []))}")
                    print(f"  Train predictions length: {len(cat_data.get('train_predictions', []))}")
                    print(f"  Validate actual length: {len(cat_data.get('validate_actual', []))}")
                    print(f"  Validate predictions length: {len(cat_data.get('validate_predictions', []))}")
                    print(f"  Metrics keys: {list(cat_data.get('metrics', {}).keys())}")
                
                return {
                    'status': 'success',
                    'scatter_data': scatter_data,
                    'training_metrics': training_metrics
                }
            else:
                print("❌ Scatter data not properly separated")
                return {'status': 'not_separated', 'keys': list(scatter_data.keys())}
        else:
            print("❌ Invalid result structure")
            return {'status': 'invalid_structure'}
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}

def test_api_separated_scatter_data():
    """APIで分離されたscatter_dataをテスト"""
    print("\n=== Testing Separated Scatter Data (API) ===")
    
    url = "http://localhost:8000/api/v1/displacement-analysis/analyze-whole"
    
    payload = {
        "folder_name": "01-hokkaido-akan",
        "model_name": "Random Forest", 
        "max_distance_from_face": 100,
        "td": None
    }
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        
        if response.status_code == 200:
            data = response.json()
            scatter_data = data.get('scatter_data', {})
            
            print(f"API scatter data keys: {list(scatter_data.keys())}")
            
            if 'settlement' in scatter_data and 'convergence' in scatter_data:
                print("✅ API scatter data successfully separated into settlement and convergence")
                
                # 各カテゴリのデータ構造をチェック
                for category in ['settlement', 'convergence']:
                    cat_data = scatter_data[category]
                    print(f"\n{category.upper()} data:")
                    print(f"  Keys: {list(cat_data.keys())}")
                    print(f"  Train actual length: {len(cat_data.get('train_actual', []))}")
                    print(f"  Train predictions length: {len(cat_data.get('train_predictions', []))}")
                    print(f"  Validate actual length: {len(cat_data.get('validate_actual', []))}")
                    print(f"  Validate predictions length: {len(cat_data.get('validate_predictions', []))}")
                    print(f"  Metrics keys: {list(cat_data.get('metrics', {}).keys())}")
                
                return {
                    'status': 'success',
                    'scatter_data': scatter_data,
                    'training_metrics': data.get('training_metrics', {})
                }
            else:
                print("❌ API scatter data not properly separated")
                return {'status': 'not_separated', 'keys': list(scatter_data.keys())}
        else:
            print(f"❌ API Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return {'status': 'http_error', 'code': response.status_code}
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the API server is running on localhost:8000")
        return {'status': 'connection_error'}
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {'status': 'error', 'error': str(e)}

def compare_separated_scatter_data(streamlit_result, api_result):
    """分離されたscatter_dataを比較"""
    print("\n=== Comparing Separated Scatter Data ===")
    
    if streamlit_result['status'] != 'success' or api_result['status'] != 'success':
        print("❌ One or both tests failed")
        return False
    
    s_scatter = streamlit_result['scatter_data']
    a_scatter = api_result['scatter_data']
    
    # 各カテゴリで比較
    for category in ['settlement', 'convergence']:
        print(f"\nComparing {category.upper()}:")
        
        s_data = s_scatter[category]
        a_data = a_scatter[category]
        
        # データ長を比較
        s_train_len = len(s_data['train_actual'])
        a_train_len = len(a_data['train_actual'])
        s_val_len = len(s_data['validate_actual'])
        a_val_len = len(a_data['validate_actual'])
        
        print(f"  Train data length - Streamlit: {s_train_len}, API: {a_train_len}")
        print(f"  Validate data length - Streamlit: {s_val_len}, API: {a_val_len}")
        
        if s_train_len == a_train_len and s_val_len == a_val_len and s_train_len > 0:
            # 実際の値を比較（最初の10個）
            s_train_actual = s_data['train_actual'][:10]
            a_train_actual = a_data['train_actual'][:10]
            
            if s_train_actual == a_train_actual:
                print(f"  ✅ {category} train actual values match")
            else:
                print(f"  ❌ {category} train actual values don't match")
                return False
        else:
            print(f"  ❌ {category} data lengths don't match or empty")
            return False
    
    print("\n✅ All separated scatter data matches between Streamlit and API!")
    return True

def test_scatter_plot_data_structure():
    """散布図描画用のデータ構造をテスト"""
    print("\n=== Testing Scatter Plot Data Structure ===")
    
    # Streamlitテスト
    streamlit_result = test_streamlit_separated_scatter_data()
    
    if streamlit_result['status'] == 'success':
        scatter_data = streamlit_result['scatter_data']
        
        # 散布図描画に必要なデータがあるかチェック
        print("\nChecking data for scatter plot generation:")
        
        for category in ['settlement', 'convergence']:
            cat_data = scatter_data[category]
            train_actual = cat_data.get('train_actual', [])
            train_pred = cat_data.get('train_predictions', [])
            val_actual = cat_data.get('validate_actual', [])
            val_pred = cat_data.get('validate_predictions', [])
            
            if len(train_actual) > 0 and len(train_pred) > 0:
                print(f"  ✅ {category} has sufficient train data for scatter plot ({len(train_actual)} points)")
            else:
                print(f"  ❌ {category} missing train data for scatter plot")
                
            if len(val_actual) > 0 and len(val_pred) > 0:
                print(f"  ✅ {category} has sufficient validate data for scatter plot ({len(val_actual)} points)")
            else:
                print(f"  ❌ {category} missing validate data for scatter plot")
                
            # メトリクスがあるかチェック
            metrics = cat_data.get('metrics', {})
            if metrics:
                print(f"  ✅ {category} has metrics for scatter plot text")
            else:
                print(f"  ❌ {category} missing metrics for scatter plot text")

def main():
    """メイン実行"""
    print("Testing separated scatter data for settlement and convergence\n")
    print("⚠️  IMPORTANT: Make sure the API server is running on localhost:8000\n")
    
    # Streamlitテスト
    streamlit_result = test_streamlit_separated_scatter_data()
    
    # APIテスト
    api_result = test_api_separated_scatter_data()
    
    # 比較
    if streamlit_result['status'] == 'success' and api_result['status'] == 'success':
        success = compare_separated_scatter_data(streamlit_result, api_result)
    else:
        success = False
    
    # 散布図データ構造テスト
    test_scatter_plot_data_structure()
    
    if success:
        print("\n🎉 完了！沈下量と変位量の分離されたscatter_dataが正常に動作しています。")
        print("   これで散布図を個別に描画できます。")
    else:
        print("\n❌ 分離されたscatter_dataにまだ問題があります。")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)