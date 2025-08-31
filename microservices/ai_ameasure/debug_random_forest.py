#!/usr/bin/env python3
"""
Debug script to investigate RandomForest 'estimators_' attribute issue
"""

import sys
import os
sys.path.append('/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure')

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

from app.displacement_temporal_spacial_analysis import (
    generate_dataframes,
    generate_additional_info_df,
    create_dataset,
    analyize_ml,
    SECTION_TD,
    DATE
)

def test_model_training():
    """Test model training with both RandomForest and LinearRegression"""
    
    # Setup data paths
    input_folder = '/home/nowatari/repos/enzan-ai-cn-dev/data_folder/01-hokkaido-akan/main_tunnel/CN_measurement_data'
    measurement_a_csvs = [os.path.join(input_folder, 'measurements_A', f) for f in os.listdir(os.path.join(input_folder, 'measurements_A')) if f.endswith('.csv')]
    cycle_support_csv = os.path.join(input_folder, 'cycle_support/cycle_support.csv')
    observation_of_face_csv = os.path.join(input_folder, 'observation_of_face/observation_of_face.csv')

    print(f"Found {len(measurement_a_csvs)} measurement files")
    
    # Generate additional info
    df_additional_info = generate_additional_info_df(cycle_support_csv, observation_of_face_csv)
    # Drop STA column if it exists
    if 'ＳＴＡ' in df_additional_info.columns:
        df_additional_info.drop(columns=['ＳＴＡ'], inplace=True)
    elif 'STA' in df_additional_info.columns:
        df_additional_info.drop(columns=['STA'], inplace=True)
    
    # Generate dataframes
    df_all, _, _, _, settlements, convergences = generate_dataframes(measurement_a_csvs, 100)
    settlement_data, convergence_data = create_dataset(df_all, df_additional_info)
    
    print(f"Settlement data shape: {settlement_data[0].shape if settlement_data else 'None'}")
    print(f"Convergence data shape: {convergence_data[0].shape if convergence_data else 'None'}")
    
    # Test both models
    models_to_test = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42)
    }
    
    for model_name, model in models_to_test.items():
        print(f"\n=== Testing {model_name} ===")
        
        for dataset_name, (df, x_columns, y_column) in [("Settlement", settlement_data), ("Convergence", convergence_data)]:
            if df is None:
                print(f"  {dataset_name}: No data available")
                continue
                
            print(f"  {dataset_name} - Data shape: {df.shape}")
            print(f"  {dataset_name} - Features: {len(x_columns)}")
            print(f"  {dataset_name} - Target: {y_column}")
            
            # Split data
            td = 500  # Use same TD as in comparison
            train_date = df[df[SECTION_TD] < td][DATE].max()
            df_train = df[df[DATE] <= train_date]
            df_validate = df[df[SECTION_TD] >= td]
            
            print(f"  {dataset_name} - Train shape: {df_train.shape}")
            print(f"  {dataset_name} - Validate shape: {df_validate.shape}")
            
            if df_train.empty or df_validate.empty:
                print(f"  {dataset_name}: Empty train or validation set")
                continue
                
            try:
                # Test model fitting and prediction
                model_copy = type(model)(**model.get_params()) if hasattr(model, 'get_params') else type(model)()
                df_train_result, df_validate_result, trained_model, metrics = analyize_ml(
                    model_copy, df_train, df_validate, x_columns, y_column
                )
                
                print(f"  {dataset_name} - Training successful!")
                print(f"  {dataset_name} - R2 train: {metrics['r2_train']:.6f}")
                print(f"  {dataset_name} - R2 validate: {metrics['r2_validate']:.6f}")
                
                # Test feature importance access
                if hasattr(trained_model, 'feature_importances_'):
                    print(f"  {dataset_name} - Feature importances shape: {trained_model.feature_importances_.shape}")
                    print(f"  {dataset_name} - Feature importances available: YES")
                else:
                    print(f"  {dataset_name} - Feature importances available: NO")
                
                # For RandomForest, check estimators_
                if model_name == "Random Forest":
                    if hasattr(trained_model, 'estimators_'):
                        print(f"  {dataset_name} - Estimators available: YES ({len(trained_model.estimators_)})")
                    else:
                        print(f"  {dataset_name} - Estimators available: NO")
                        
            except Exception as e:
                print(f"  {dataset_name} - ERROR: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    test_model_training()