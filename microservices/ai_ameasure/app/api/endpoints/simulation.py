import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from app.core.config import settings
from app.displacement import DURATION_DAYS

# 代わりにGUI実装からインポート（作業ディレクトリ変更後）
from app.schemas import simulation as schemas
from fastapi import APIRouter, HTTPException, status

# GUI関数をインポートするためのパス設定
gui_path = '/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure'
if gui_path not in sys.path:
    sys.path.append(gui_path)

# 作業ディレクトリを変更してGUI関数をインポート
original_import_cwd = os.getcwd()
os.chdir(gui_path)

try:
    # GUI関数をインポート（名前を変えて衝突を回避）
    from gui_displacement_temporal_spacial_analysis import simulate_displacement as gui_simulate_displacement
    # GUI実装の関数もインポート
    from app.displacement_temporal_spacial_analysis import (
        DATE,
        DISTANCE_FROM_FACE,
        STA,
        TD_NO,
        generate_additional_info_df,
        generate_dataframes,
        create_dataset,
    )
finally:
    # 作業ディレクトリを戻す
    os.chdir(original_import_cwd)

router = APIRouter()


def draw_local_prediction_chart(
    output_path: str, x_measure, df_measure_y, x_predict, df_predict_y, title: str
):
    """
    Generate prediction chart and save to file
    """
    import japanize_matplotlib  # noqa: F401
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")

    for i, c in enumerate(df_measure_y.columns):
        plt.plot(
            x_measure,
            df_measure_y[c],
            label=c,
            marker="x",
            linestyle="--",
            markersize=4,
            alpha=0.5,
            color=cmap(i),
        )

    for i, c in enumerate(df_predict_y.columns):
        plt.plot(
            x_predict,
            df_predict_y[c],
            label=["予測最終" + c],
            marker="o",
            linestyle="-",
            markersize=4,
            color=cmap(i),
        )

    plt.title(title)
    plt.xlabel(DISTANCE_FROM_FACE)
    plt.ylabel(f"(mm)")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.close()


# GUI関数で使用されるmodel_pathsを動的に設定するためのヘルパー
def setup_gui_environment():
    """GUI環境をAPI用に設定"""
    import json
    import gui_displacement_temporal_spacial_analysis
    
    # GUI関数が依存するconfig.jsonを設定
    gui_config = {
        'input_folder': str(settings.DATA_FOLDER)
    }
    
    config_path = os.path.join(gui_path, 'config.json')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(gui_config, f)
    
    # GUI関数で使用されるmodel_pathsを設定（作業ディレクトリがai_ameasureに変更されるので）
    gui_output_folder = os.path.join(gui_path, "output")  # ai_ameasure/output
    model_paths = {
        # 最終沈下量、変位量予測モデルのパス
        "final_value_prediction_model": [
            os.path.join(gui_output_folder, "model_final_settlement.pkl"),
            os.path.join(gui_output_folder, "model_final_convergence.pkl")
        ],
        # 沈下量、変位量予測モデルのパス
        "prediction_model": [
            os.path.join(gui_output_folder, "model_settlement.pkl"),
            os.path.join(gui_output_folder, "model_convergence.pkl")
        ]
    }
    
    # GUIモジュールにmodel_pathsを設定
    gui_displacement_temporal_spacial_analysis.model_paths = model_paths
    
    # 作業ディレクトリを一時的に変更してGUI関数を使用
    original_cwd = os.getcwd()
    os.chdir(gui_path)
    return original_cwd

def restore_environment(original_cwd):
    """元の作業ディレクトリに戻す"""
    os.chdir(original_cwd)


from pydantic import BaseModel

class LocalDisplacementRequest(BaseModel):
    folder_name: str
    ameasure_file: str
    distance_from_face: float
    daily_advance: float
    max_distance_from_face: float = 200.0

@router.post("/local-displacement", response_model=Dict[str, Any])
async def analyze_local_displacement_gui_style(
    request: LocalDisplacementRequest
) -> Dict[str, Any]:
    """
    Analyze local displacement based on GUI tab2 functionality

    Args:
        request: LocalDisplacementRequest containing all parameters

    Returns:
        Dictionary containing prediction results, simulation results, and chart paths
    """
    try:
        # Extract parameters from request
        folder_name = request.folder_name
        ameasure_file = request.ameasure_file
        distance_from_face = request.distance_from_face
        daily_advance = request.daily_advance
        max_distance_from_face = request.max_distance_from_face
        
        # Setup paths using settings
        input_base = str(settings.DATA_FOLDER)
        input_folder = os.path.join(input_base, folder_name, "main_tunnel", "CN_measurement_data")
        a_measure_path = os.path.join(input_folder, "measurements_A", ameasure_file)
        output_folder = str(settings.OUTPUT_FOLDER)

        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)

        # Check if input files exist
        if not os.path.exists(a_measure_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Measurement file not found: {a_measure_path}",
            )

        # GUI環境をセットアップ
        original_cwd = setup_gui_environment()
        
        try:
            # Prediction phase - GUI関数を直接使用
            df_all_prediction, settlements, convergences = gui_simulate_displacement(
                input_folder, a_measure_path, max_distance_from_face
            )
            
            # Simulation phase with recursive prediction - GUI関数を直接使用
            df_all_simulated, _, _ = gui_simulate_displacement(
                input_folder,
                a_measure_path,
                max_distance_from_face,
                daily_advance,
                distance_from_face,
                recursive=True,
            )
        
        finally:
            # 環境を復元
            restore_environment(original_cwd)

        td = float(df_all_prediction[TD_NO].values[0])
        cycle_no = float(os.path.basename(a_measure_path).split("_")[2].split(".")[0])

        # Save simulation results to CSV
        simulation_csv_path = os.path.join(
            output_folder, f"{folder_name}_{os.path.basename(a_measure_path)}"
        )
        df_all_simulated.to_csv(simulation_csv_path, index=False)

        # Create combined simulation data: use simulated data as the base (it includes the recursive simulation)
        # The simulated data already contains the extended simulation values
        df_combined_simulation = df_all_simulated.copy()
        
        if not df_all_simulated.empty:
            distances = df_all_simulated[DISTANCE_FROM_FACE].tolist()
            
            # Check if we have data at or below distance_from_face
            below_threshold = [d for d in distances if d <= distance_from_face]
            above_threshold = [d for d in distances if d > distance_from_face]
            
            # If we don't have data at the exact distance_from_face, add it from prediction data
            if not below_threshold and not df_all_prediction.empty:
                # Find the closest point in prediction data
                pred_distances = df_all_prediction[DISTANCE_FROM_FACE].tolist()
                closest_distance = min(pred_distances, key=lambda x: abs(x - distance_from_face))
                print(f"  - Adding closest prediction point at {closest_distance}m")
                
                closest_point = df_all_prediction[df_all_prediction[DISTANCE_FROM_FACE] == closest_distance].copy()
                if not closest_point.empty:
                    # Adjust the distance to match exactly
                    closest_point.loc[:, DISTANCE_FROM_FACE] = distance_from_face
                    df_combined_simulation = pd.concat([closest_point, df_combined_simulation], ignore_index=True)
                    df_combined_simulation = df_combined_simulation.sort_values(DISTANCE_FROM_FACE).reset_index(drop=True)
        
        # Prepare columns for simulation data
        simulation_cols = [DISTANCE_FROM_FACE]
        
        # Add actual measurement columns 
        for col in settlements + convergences:
            if col in df_combined_simulation.columns:
                simulation_cols.append(col)
        
        # Add prediction columns
        for col in settlements + convergences:
            pred_col = col + "_prediction"
            if pred_col in df_combined_simulation.columns:
                simulation_cols.append(pred_col)
        
        simulation_data_df = df_combined_simulation[simulation_cols]
        simulation_data_records = simulation_data_df.to_dict(orient="records")
        
        # Prepare prediction data (actual measurements + predictions for charts)
        prediction_data_df = df_all_prediction[
            [DISTANCE_FROM_FACE] + settlements + convergences + [l + "_prediction" for l in settlements + convergences]
        ]
        prediction_data_records = prediction_data_df.to_dict(orient="records")
        
        # Prepare table data - only prediction columns for the data table
        table_prediction_cols = [DISTANCE_FROM_FACE]
        for col in settlements + convergences:
            pred_col = col + "_prediction"
            if pred_col in df_all_simulated.columns:
                table_prediction_cols.append(pred_col)
        
        table_data_df = df_all_simulated[table_prediction_cols]
        table_data_records = table_data_df.to_dict(orient="records")
        
        response_data = {
            "folder_name": folder_name,
            "cycle_no": cycle_no,
            "td": td,
            "distance_from_face": distance_from_face,
            "daily_advance": daily_advance,
            "simulation_csv": simulation_csv_path,
            "simulation_data": simulation_data_records,
            "prediction_data": prediction_data_records,
            "table_data": table_data_records,
            "timestamp": datetime.now().isoformat(),
        }
        
        return response_data

    except Exception as e:
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in local displacement analysis: {str(e)}",
        )


@router.post("/simulate", response_model=schemas.SimulationResponse)
async def simulate_displacement(request: schemas.SimulationRequest) -> schemas.SimulationResponse:
    """
    Displacement simulation endpoint using trained models
    """
    try:
        # Setup paths
        input_base = str(settings.DATA_FOLDER)
        input_folder = os.path.join(input_base, request.folder_name, "main_tunnel", "CN_measurement_data")
        
        # Find latest measurement file if not specified
        measurements_path = os.path.join(input_folder, "measurements_A")
        if not os.path.exists(measurements_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Measurements folder not found: {measurements_path}",
            )
        
        # Get the latest measurement file
        csv_files = sorted([f for f in os.listdir(measurements_path) if f.endswith(".csv")])
        if not csv_files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No measurement files found in: {measurements_path}",
            )
        
        latest_measurement_file = csv_files[-1]  # Get latest file
        a_measure_path = os.path.join(measurements_path, latest_measurement_file)
        
        # GUI環境をセットアップ
        original_cwd = setup_gui_environment()
        
        try:
            # GUI関数を直接使用
            df_all_simulated, settlements, convergences = gui_simulate_displacement(
                input_folder,
                a_measure_path,
                request.max_distance,
                request.daily_advance,
                request.distance_from_face,
                recursive=request.recursive,
            )
        finally:
            # 環境を復元
            restore_environment(original_cwd)
        
        # Convert simulation results to response format
        simulation_data = []
        
        for _, row in df_all_simulated.iterrows():
            # Get position IDs from settlements and convergences
            for settlement_col in settlements:
                position_id = settlement_col  # Use column name as position ID
                
                # Get corresponding convergence column
                convergence_col = None
                for conv_col in convergences:
                    if conv_col.split('_')[0] == settlement_col.split('_')[0]:  # Match prefix
                        convergence_col = conv_col
                        break
                
                if convergence_col is None:
                    continue
                
                simulation_data.append(
                    schemas.SimulationDataPoint(
                        td_no=int(row.get(TD_NO, 0)),
                        date=pd.to_datetime(row.get(DATE, datetime.now())),
                        distance_from_face=float(row[DISTANCE_FROM_FACE]),
                        position_id=position_id,
                        settlement=float(row.get(settlement_col, 0.0)),
                        settlement_prediction=float(row.get(f"{settlement_col}_prediction", 0.0)),
                        convergence=float(row.get(convergence_col, 0.0)),
                        convergence_prediction=float(row.get(f"{convergence_col}_prediction", 0.0)),
                    )
                )

        return schemas.SimulationResponse(
            folder_name=request.folder_name,
            simulation_data=simulation_data,
            daily_advance=request.daily_advance,
            distance_from_face=request.distance_from_face,
            recursive=request.recursive,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in displacement simulation: {str(e)}"
        )


@router.get("/folders")
async def list_folders() -> Dict[str, List[str]]:
    """
    List available folders for analysis
    """
    input_base = str(settings.DATA_FOLDER)

    if not os.path.exists(input_base):
        return {"folders": []}

    folders = [f for f in os.listdir(input_base) if os.path.isdir(os.path.join(input_base, f))]

    return {"folders": sorted(folders)}


@router.get("/measurements/{folder_name}")
async def list_measurement_files(folder_name: str) -> Dict[str, List[str]]:
    """
    List CSV measurement files for a given folder
    """
    input_base = str(settings.DATA_FOLDER)
    measurements_path = os.path.join(
        input_base, folder_name, "main_tunnel", "CN_measurement_data", "measurements_A"
    )

    if not os.path.exists(measurements_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Measurements folder not found: {measurements_path}",
        )

    csv_files = [f for f in os.listdir(measurements_path) if f.endswith(".csv")]

    return {"measurement_files": sorted(csv_files)}
