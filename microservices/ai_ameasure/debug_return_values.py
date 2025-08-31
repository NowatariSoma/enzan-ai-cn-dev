#!/usr/bin/env python3
"""
analyze_displacement関数の戻り値を詳細にデバッグ
"""

import sys
import os
from pathlib import Path

# パス追加
sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure')

from app.displacement_temporal_spacial_analysis import analyze_displacement
from sklearn.ensemble import RandomForestRegressor

def debug_analyze_displacement():
    """analyze_displacement関数の戻り値をデバッグ"""
    
    # 設定
    input_folder = '/home/nowatari/repos/enzan-ai-cn-dev/data_folder/01-hokkaido-akan/main_tunnel/CN_measurement_data'
    output_folder = '/tmp/debug_return'
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
    
    print("Calling analyze_displacement and capturing return values...")
    
    # 戻り値を直接取得
    result = analyze_displacement(input_folder, output_folder, model_paths, model, 100, None)
    
    print(f"\n=== RETURN VALUE DEBUG ===")
    print(f"Result type: {type(result)}")
    print(f"Result is tuple: {isinstance(result, tuple)}")
    
    if isinstance(result, tuple):
        print(f"Tuple length: {len(result)}")
        
        for i, item in enumerate(result):
            print(f"\nResult[{i}]:")
            print(f"  Type: {type(item)}")
            print(f"  Is None: {item is None}")
            
            if hasattr(item, 'shape'):
                print(f"  Shape: {item.shape}")
            elif hasattr(item, '__len__'):
                print(f"  Length: {len(item)}")
                if hasattr(item, 'keys'):
                    print(f"  Keys: {list(item.keys())}")
                    
        # 3つの戻り値があるかチェック                    
        if len(result) == 3:
            df_all, training_metrics, scatter_data = result
            
            print(f"\n=== UNPACKED VALUES ===")
            print(f"df_all: {type(df_all)}, shape: {df_all.shape if hasattr(df_all, 'shape') else 'No shape'}")
            print(f"training_metrics: {type(training_metrics)}, keys: {list(training_metrics.keys()) if hasattr(training_metrics, 'keys') else 'No keys'}")
            print(f"scatter_data: {type(scatter_data)}, keys: {list(scatter_data.keys()) if hasattr(scatter_data, 'keys') else 'No keys'}")
            
            if scatter_data and hasattr(scatter_data, 'keys'):
                for key in scatter_data.keys():
                    if isinstance(scatter_data[key], list):
                        print(f"  scatter_data['{key}']: {len(scatter_data[key])} items")
                    else:
                        print(f"  scatter_data['{key}']: {type(scatter_data[key])}")
            
            return True
        elif len(result) == 2:
            print(f"\n=== ONLY 2 VALUES RETURNED ===")
            df_all, training_metrics = result
            print(f"df_all: {type(df_all)}")
            print(f"training_metrics: {type(training_metrics)}")
            print("❌ Missing scatter_data!")
            return False
        else:
            print(f"❌ Unexpected number of return values: {len(result)}")
            return False
    else:
        print(f"❌ Result is not a tuple: {type(result)}")
        return False

if __name__ == "__main__":
    success = debug_analyze_displacement()
    
    if success:
        print("\n✅ Function returns 3 values as expected")
    else:
        print("\n❌ Function does not return 3 values correctly")
    
    exit(0 if success else 1)