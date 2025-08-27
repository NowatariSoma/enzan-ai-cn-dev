#!/usr/bin/env python3
"""
PredictionEngineを直接実行して生成されるメトリクスを詳細確認
"""
import sys
sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev/microservices/ai_ameasure')

from app.core.prediction_engine import PredictionEngine

def debug_prediction_engine():
    print("=" * 80)
    print("PredictionEngine直接実行 - メトリクス確認")
    print("=" * 80)
    
    engine = PredictionEngine()
    result = engine.train_model(
        model_name="random_forest",
        folder_name="01-hokkaido-akan",
        max_distance_from_face=100.0,
        td=None
    )
    
    print(f"\nResult keys: {list(result.keys())}")
    
    training_metrics = result.get('training_metrics', {})
    print(f"\nTraining metrics keys: {list(training_metrics.keys())}")
    
    for key, metrics in training_metrics.items():
        print(f"\n{key}:")
        print(f"  Train R²: {metrics.get('r2_train', 'N/A')}")
        print(f"  Validation R²: {metrics.get('r2_validate', 'N/A')}")
    
    # settlementとconvergenceの詳細確認
    settlement_metrics = training_metrics.get("沈下量", {})
    final_settlement_metrics = training_metrics.get("最終沈下量との差分", {})
    
    print(f"\n沈下量メトリクス:")
    print(f"  Train R²: {settlement_metrics.get('r2_train', 'N/A')}")
    print(f"  Validation R²: {settlement_metrics.get('r2_validate', 'N/A')}")
    
    print(f"\n最終沈下量との差分メトリクス:")
    print(f"  Train R²: {final_settlement_metrics.get('r2_train', 'N/A')}")
    print(f"  Validation R²: {final_settlement_metrics.get('r2_validate', 'N/A')}")

if __name__ == "__main__":
    debug_prediction_engine()