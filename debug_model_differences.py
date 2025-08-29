#!/usr/bin/env python3
"""
GUIとAPIが使用するモデルファイルの詳細比較
"""

import os
import sys
import joblib
import hashlib
from pathlib import Path

def get_file_hash(filepath):
    """ファイルのハッシュ値を取得"""
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def compare_models():
    """モデルファイルの比較"""
    print("=== モデルファイル比較調査 ===")
    
    # GUIが使用するパス
    gui_output = "/home/nowatari/repos/enzan-ai-cn-dev/output"
    
    # APIが使用するパス (設定ファイルから確認)
    sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev/microservices/ai_ameasure')
    from app.core.config import settings
    api_output = str(settings.OUTPUT_FOLDER)
    
    print(f"GUI Output Path: {gui_output}")
    print(f"API Output Path: {api_output}")
    
    model_files = [
        "model_final_settlement.pkl",
        "model_final_convergence.pkl", 
        "model_settlement.pkl",
        "model_convergence.pkl"
    ]
    
    print("\n--- ファイル存在確認 ---")
    for model_file in model_files:
        gui_path = os.path.join(gui_output, model_file)
        api_path = os.path.join(api_output, model_file)
        
        print(f"\n{model_file}:")
        print(f"  GUI: {gui_path} ({'存在' if os.path.exists(gui_path) else '不在'})")
        print(f"  API: {api_path} ({'存在' if os.path.exists(api_path) else '不在'})")
        
        if os.path.exists(gui_path) and os.path.exists(api_path):
            gui_hash = get_file_hash(gui_path)
            api_hash = get_file_hash(api_path)
            
            print(f"  GUI Hash: {gui_hash}")
            print(f"  API Hash: {api_hash}")
            print(f"  同一: {'✅' if gui_hash == api_hash else '❌'}")
            
            # ファイルサイズも比較
            gui_size = os.path.getsize(gui_path)
            api_size = os.path.getsize(api_path)
            print(f"  GUI Size: {gui_size} bytes")
            print(f"  API Size: {api_size} bytes")

def inspect_model_content():
    """モデル内容の詳細調査"""
    print("\n=== モデル内容調査 ===")
    
    gui_model_path = "/home/nowatari/repos/enzan-ai-cn-dev/output/model_final_settlement.pkl"
    api_model_path = "/home/nowatari/repos/enzan-ai-cn-dev/output/model_final_settlement.pkl"  # 同じパスを使用
    
    if os.path.exists(gui_model_path):
        print(f"\nモデル詳細: {gui_model_path}")
        try:
            model = joblib.load(gui_model_path)
            print(f"  タイプ: {type(model)}")
            print(f"  属性: {dir(model)}")
            
            if hasattr(model, 'random_state'):
                print(f"  random_state: {model.random_state}")
            if hasattr(model, 'n_estimators'):
                print(f"  n_estimators: {model.n_estimators}")
            if hasattr(model, 'feature_importances_'):
                print(f"  feature_importances shape: {model.feature_importances_.shape}")
                print(f"  feature_importances: {model.feature_importances_[:5]}...")
                
        except Exception as e:
            print(f"  エラー: {e}")

if __name__ == "__main__":
    compare_models()
    inspect_model_content()