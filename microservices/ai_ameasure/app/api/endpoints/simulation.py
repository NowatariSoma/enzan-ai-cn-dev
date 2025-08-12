import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from app.core.config import settings
from app.displacement import DURATION_DAYS
from app.displacement_temporal_spacial_analysis import (
    DATE,
    DISTANCE_FROM_FACE,
    STA,
    TD_NO,
    create_dataset,
    generate_additional_info_df,
    generate_dataframes,
)
from app.schemas import simulation as schemas
from fastapi import APIRouter, HTTPException, status

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


def simulate_displacement_logic(
    input_folder: str,
    a_measure_path: str,
    max_distance_from_face: float,
    daily_advance: Optional[float] = None,
    distance_from_face: Optional[float] = None,
    recursive: bool = False,
):
    """
    Core displacement simulation logic extracted from GUI
    """
    cycle_support_csv = os.path.join(input_folder, "cycle_support/cycle_support.csv")
    observation_of_face_csv = os.path.join(
        input_folder, "observation_of_face/observation_of_face.csv"
    )

    df_additional_info = generate_additional_info_df(cycle_support_csv, observation_of_face_csv)
    df_additional_info.drop(columns=[STA], inplace=True)

    # Process each CSV file using the preprocess function
    df_all, _, _, _, settlements, convergences = generate_dataframes(
        [a_measure_path], max_distance_from_face
    )

    if daily_advance and distance_from_face:
        # Create a new dataframe with the specified length and interval
        max_record = math.ceil(min(max_distance_from_face / daily_advance, DURATION_DAYS))
        df_all_actual = df_all[df_all[DISTANCE_FROM_FACE] < distance_from_face]

        if df_all_actual.empty:
            df_all_new = pd.DataFrame([df_all.iloc[0]] * max_record).reset_index()
        else:
            df_all_new = pd.DataFrame([df_all_actual.iloc[-1]] * max_record).reset_index()

        df_all_new[DATE] = pd.to_datetime(df_all.iloc[0][DATE]) + pd.to_timedelta(
            range(max_record), unit="D"
        )
        df_all_new[DISTANCE_FROM_FACE] = df_all.iloc[0][
            DISTANCE_FROM_FACE
        ] + daily_advance * pd.Series(range(max_record))
        df_all = pd.concat(
            [df_all_actual, df_all_new[distance_from_face <= df_all_new[DISTANCE_FROM_FACE]]],
            ignore_index=True,
        ).reset_index()

    settlement_data, convergence_data = create_dataset(df_all, df_additional_info)

    # Load models
    output_folder = str(settings.OUTPUT_FOLDER)
    model_paths = {
        "final_value_prediction_model": [
            os.path.join(output_folder, "model_final_settlement.pkl"),
            os.path.join(output_folder, "model_final_convergence.pkl"),
        ],
        "prediction_model": [
            os.path.join(output_folder, "model_settlement.pkl"),
            os.path.join(output_folder, "model_convergence.pkl"),
        ],
    }

    for i, ((df, x_columns, y_column), target) in enumerate(
        zip([settlement_data, convergence_data], [settlements, convergences])
    ):
        if not os.path.exists(model_paths["final_value_prediction_model"][i]):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model file not found: {model_paths['final_value_prediction_model'][i]}",
            )

        final_model_data = joblib.load(model_paths["final_value_prediction_model"][i])
        model_data = joblib.load(model_paths["prediction_model"][i])
        
        # Extract the actual model from the dictionary
        final_model = final_model_data["model"] if isinstance(final_model_data, dict) else final_model_data
        model = model_data["model"] if isinstance(model_data, dict) else model_data

        if recursive:
            # まずは沈下量・変位量を予測し、それに基づき最終沈下量・変位量を予測する
            _y_column = x_columns[2]
            _x_columns = [x for x in x_columns if x != _y_column]
            _x_columns = [x for x in _x_columns if x != y_column]
            _y_hat = model.predict(pd.DataFrame(df[_x_columns]))
            df.loc[df[DISTANCE_FROM_FACE] > distance_from_face, _y_column] = _y_hat[
                df[DISTANCE_FROM_FACE] > distance_from_face
            ]

        y_hat = final_model.predict(df[x_columns])
        for position_id in df["position_id"].unique():
            df_all[f"{target[position_id]}_prediction"] = (
                y_hat[df["position_id"] == position_id] + df_all[target[position_id]]
            )

    return df_all, settlements, convergences


from pydantic import BaseModel

class LocalDisplacementRequest(BaseModel):
    folder_name: str
    ameasure_file: str
    distance_from_face: float
    daily_advance: float
    max_distance_from_face: float = 200.0

@router.post("/local-displacement", response_model=Dict[str, Any])
async def analyze_local_displacement(
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

        # Prediction phase
        df_all, settlements, convergences = simulate_displacement_logic(
            input_folder, a_measure_path, max_distance_from_face
        )

        td = float(df_all[TD_NO].values[0])
        cycle_no = float(os.path.basename(a_measure_path).split("_")[2].split(".")[0])

        # Generate prediction charts
        settlement_prediction_path = os.path.join(
            output_folder, f"settlement_prediction_{cycle_no}.png"
        )
        convergence_prediction_path = os.path.join(
            output_folder, f"convergence_prediction_{cycle_no}.png"
        )

        draw_local_prediction_chart(
            settlement_prediction_path,
            df_all[DISTANCE_FROM_FACE],
            df_all[settlements],
            df_all[DISTANCE_FROM_FACE],
            df_all[[l + "_prediction" for l in settlements]],
            f"最終沈下量予測 for Cycle {cycle_no} (TD: {td})",
        )

        draw_local_prediction_chart(
            convergence_prediction_path,
            df_all[DISTANCE_FROM_FACE],
            df_all[convergences],
            df_all[DISTANCE_FROM_FACE],
            df_all[[l + "_prediction" for l in convergences]],
            f"最終変位量予測 for Cycle {cycle_no} (TD: {td})",
        )

        # Simulation phase with recursive prediction
        df_all_simulated, _, _ = simulate_displacement_logic(
            input_folder,
            a_measure_path,
            max_distance_from_face,
            daily_advance,
            distance_from_face,
            recursive=True,
        )

        # Save simulation results to CSV
        simulation_csv_path = os.path.join(
            output_folder, f"{folder_name}_{os.path.basename(a_measure_path)}"
        )
        df_all_simulated.to_csv(simulation_csv_path, index=False)

        # Generate simulation charts
        settlement_simulation_path = os.path.join(
            output_folder, f"settlement_simulation_{cycle_no}.png"
        )
        convergence_simulation_path = os.path.join(
            output_folder, f"convergence_simulation_{cycle_no}.png"
        )

        draw_local_prediction_chart(
            settlement_simulation_path,
            df_all[DISTANCE_FROM_FACE],
            df_all[settlements],
            df_all_simulated[DISTANCE_FROM_FACE],
            df_all_simulated[[l + "_prediction" for l in settlements]],
            f"最終沈下量予測 for Cycle {cycle_no} (TD: {td})",
        )

        draw_local_prediction_chart(
            convergence_simulation_path,
            df_all[DISTANCE_FROM_FACE],
            df_all[convergences],
            df_all_simulated[DISTANCE_FROM_FACE],
            df_all_simulated[[l + "_prediction" for l in convergences]],
            f"最終変位量予測 for Cycle {cycle_no} (TD: {td})",
        )

        # Prepare response data
        response_data = {
            "folder_name": folder_name,
            "cycle_no": cycle_no,
            "td": td,
            "distance_from_face": distance_from_face,
            "daily_advance": daily_advance,
            "prediction_charts": {
                "settlement": settlement_prediction_path,
                "convergence": convergence_prediction_path,
            },
            "simulation_charts": {
                "settlement": settlement_simulation_path,
                "convergence": convergence_simulation_path,
            },
            "simulation_csv": simulation_csv_path,
            "simulation_data": df_all_simulated[
                [DISTANCE_FROM_FACE] + [l + "_prediction" for l in convergences + settlements]
            ].to_dict(orient="records"),
            "timestamp": datetime.now(),
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
    General displacement simulation endpoint (keeping existing functionality)
    """
    try:
        # This maintains the existing mock functionality for compatibility
        # You can extend this to use the real simulate_displacement_logic if needed
        simulation_data = []
        base_date = datetime.now()
        position_ids = ["A-1", "B-1", "C-1"]

        max_record = math.ceil(min(request.max_distance / request.daily_advance, DURATION_DAYS))

        for i in range(max_record):
            current_date = base_date + pd.Timedelta(days=i)
            current_distance = request.distance_from_face + (request.daily_advance * i)

            if current_distance > request.max_distance:
                break

            for position_id in position_ids:
                simulation_data.append(
                    schemas.SimulationDataPoint(
                        td_no=100 + i,
                        date=current_date,
                        distance_from_face=current_distance,
                        position_id=position_id,
                        settlement=0.0,  # Will be populated by real logic
                        settlement_prediction=0.0,
                        convergence=0.0,
                        convergence_prediction=0.0,
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
        raise HTTPException(status_code=500, detail=str(e))


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
