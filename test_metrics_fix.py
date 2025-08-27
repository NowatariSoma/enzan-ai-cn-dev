#!/usr/bin/env python3
"""
Test script to verify that PredictionEngine returns actual training metrics
"""
import sys
import os

# Add the microservices path
sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev/microservices/ai_ameasure')

from app.core.prediction_engine import PredictionEngine

def test_metrics_extraction():
    print("Testing PredictionEngine metrics extraction...")
    
    try:
        engine = PredictionEngine()
        result = engine.train_model(
            model_name="random_forest",
            folder_name="01-hokkaido-akan",
            max_distance_from_face=100.0
        )
        
        print("\n=== Training Result ===")
        for key, value in result.items():
            if key == 'training_metrics':
                print(f"{key}:")
                for metric_name, metrics in value.items():
                    print(f"  {metric_name}: {metrics}")
            else:
                print(f"{key}: {value}")
                
        # Check if we got the expected metrics
        training_metrics = result.get('training_metrics', {})
        settlement_metrics = training_metrics.get('沈下量1', {})
        
        if settlement_metrics:
            print(f"\n=== Settlement Metrics Found ===")
            print(f"Train R²: {settlement_metrics.get('r2_train')}")
            print(f"Validation R²: {settlement_metrics.get('r2_validate')}")
            print(f"Train MSE: {settlement_metrics.get('mse_train')}")
            print(f"Validation MSE: {settlement_metrics.get('mse_validate')}")
        else:
            print("\n❌ No settlement metrics found!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_metrics_extraction()