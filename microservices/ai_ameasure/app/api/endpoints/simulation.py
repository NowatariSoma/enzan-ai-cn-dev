import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from app import schemas
from app.api import deps
from app.core.config import settings
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter()


def generate_simulation_data(
    folder_name: str,
    daily_advance: float,
    distance_from_face: float,
    max_distance: float,
    recursive: bool,
) -> List[schemas.SimulationDataPoint]:
    """
    シミュレーションデータを生成
    GUIのsimulate_displacement関数をAPIで実装
    """
    # モックデータ生成のパラメータ
    DURATION_DAYS = 365  # 最大シミュレーション日数
    max_record = math.ceil(min(max_distance / daily_advance, DURATION_DAYS))

    simulation_data = []
    base_date = datetime.now()

    # 複数の位置IDでシミュレーション
    position_ids = ["A-1", "B-1", "C-1"]

    for i in range(max_record):
        current_date = base_date + timedelta(days=i)
        current_distance = distance_from_face + (daily_advance * i)

        if current_distance > max_distance:
            break

        for position_id in position_ids:
            # ランダムノイズを追加してリアルなデータを生成
            noise = (np.random.random() - 0.5) * 0.1

            # 実測値（切羽までの距離のみ）
            if current_distance <= distance_from_face:
                settlement = np.sin(current_distance * 0.05) * 2.0 + noise
                convergence = np.cos(current_distance * 0.04) * 1.5 + noise * 0.8
            else:
                settlement = 0
                convergence = 0

            # 予測値
            settlement_prediction = np.sin(current_distance * 0.05) * 2.2 + noise * 1.2
            convergence_prediction = np.cos(current_distance * 0.04) * 1.7 + noise * 1.0

            # 再帰的予測の場合、予測値を調整
            if recursive and current_distance > distance_from_face:
                settlement_prediction *= 0.9
                convergence_prediction *= 0.85

            simulation_data.append(
                schemas.SimulationDataPoint(
                    td_no=100 + i,
                    date=current_date,
                    distance_from_face=current_distance,
                    position_id=position_id,
                    settlement=settlement,
                    settlement_prediction=settlement_prediction,
                    convergence=convergence,
                    convergence_prediction=convergence_prediction,
                )
            )

    return simulation_data


@router.post("/simulate", response_model=schemas.SimulationResponse)
async def simulate_displacement(request: schemas.SimulationRequest) -> schemas.SimulationResponse:
    """
    変位予測シミュレーションを実行

    - 日進量と現在の切羽からの距離を指定してシミュレーション
    - 再帰的予測オプションあり
    """
    try:
        simulation_data = generate_simulation_data(
            request.folder_name,
            request.daily_advance,
            request.distance_from_face,
            request.max_distance,
            request.recursive,
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


@router.post("/chart-data", response_model=schemas.ChartDataResponse)
async def get_chart_data(request: schemas.ChartDataRequest) -> schemas.ChartDataResponse:
    """
    チャート描画用のデータを生成
    """
    chart_data = []

    # チャートタイプに応じたデータ生成
    if request.chart_type == "displacement":
        # 変位量チャート（フロントエンドが期待する形式に合わせる）
        for i in range(50):
            x_val = i * 4.0  # 0から200mまで

            # 実測値
            chart_data.extend(
                [
                    schemas.ChartDataPoint(
                        x=x_val,
                        y=np.sin(x_val * 0.1) * 0.5 + (np.random.random() - 0.5) * 0.2,
                        series="変位量A",
                        label=f"変位量A at {x_val}m",
                    ),
                    schemas.ChartDataPoint(
                        x=x_val,
                        y=np.cos(x_val * 0.08) * 0.3 + (np.random.random() - 0.5) * 0.1,
                        series="変位量B",
                        label=f"変位量B at {x_val}m",
                    ),
                    schemas.ChartDataPoint(
                        x=x_val,
                        y=np.sin(x_val * 0.12) * 0.4 + (np.random.random() - 0.5) * 0.15,
                        series="変位量C",
                        label=f"変位量C at {x_val}m",
                    ),
                ]
            )

            if request.include_predictions:
                # 予測値
                chart_data.extend(
                    [
                        schemas.ChartDataPoint(
                            x=x_val,
                            y=np.cos(x_val * 0.1) * 0.6 + (np.random.random() - 0.5) * 0.2,
                            series="変位量A_prediction",
                            label=f"変位量A予測 at {x_val}m",
                        ),
                        schemas.ChartDataPoint(
                            x=x_val,
                            y=np.sin(x_val * 0.09) * 0.35 + (np.random.random() - 0.5) * 0.1,
                            series="変位量B_prediction",
                            label=f"変位量B予測 at {x_val}m",
                        ),
                        schemas.ChartDataPoint(
                            x=x_val,
                            y=np.cos(x_val * 0.11) * 0.45 + (np.random.random() - 0.5) * 0.15,
                            series="変位量C_prediction",
                            label=f"変位量C予測 at {x_val}m",
                        ),
                    ]
                )

        return schemas.ChartDataResponse(
            chart_type=request.chart_type,
            data=chart_data,
            x_label="切羽からの距離 (m)",
            y_label="変位量 (mm)",
            title="変位量の推移",
        )

    elif request.chart_type == "settlement":
        # 沈下量チャート
        for i in range(50):
            x_val = i * 4.0
            chart_data.extend(
                [
                    schemas.ChartDataPoint(
                        x=x_val,
                        y=np.sin(x_val * 0.05) * 2.0,
                        series="沈下量実測",
                        label=f"実測 at {x_val}m",
                    )
                ]
            )

            if request.include_predictions:
                chart_data.append(
                    schemas.ChartDataPoint(
                        x=x_val,
                        y=np.sin(x_val * 0.05) * 2.2,
                        series="沈下量予測",
                        label=f"予測 at {x_val}m",
                    )
                )

        return schemas.ChartDataResponse(
            chart_type=request.chart_type,
            data=chart_data,
            x_label="切羽からの距離 (m)",
            y_label="沈下量 (mm)",
            title="沈下量の推移",
        )

    else:
        raise HTTPException(status_code=400, detail=f"Unknown chart type: {request.chart_type}")


@router.get("/config", response_model=schemas.ModelConfigListResponse)
async def get_model_config() -> schemas.ModelConfigListResponse:
    """
    現在のモデル設定を取得
    """
    # モックデータとしてデフォルト設定を返す
    configs = {
        "settlement": schemas.ModelConfigResponse(
            model_name="settlement",
            model_type="RandomForest",
            parameters={"n_estimators": 100, "random_state": 42},
            is_fitted=True,
        ),
        "final_settlement": schemas.ModelConfigResponse(
            model_name="final_settlement",
            model_type="RandomForest",
            parameters={"n_estimators": 100, "random_state": 42},
            is_fitted=True,
        ),
        "convergence": schemas.ModelConfigResponse(
            model_name="convergence",
            model_type="HistGradientBoostingRegressor",
            parameters={"random_state": 42},
            is_fitted=True,
        ),
        "final_convergence": schemas.ModelConfigResponse(
            model_name="final_convergence",
            model_type="HistGradientBoostingRegressor",
            parameters={"random_state": 42},
            is_fitted=True,
        ),
    }

    return schemas.ModelConfigListResponse(configs=configs)


@router.post("/config", response_model=schemas.ModelConfigResponse)
async def update_model_config(request: schemas.ModelConfigRequest) -> schemas.ModelConfigResponse:
    """
    モデル設定を更新
    """
    valid_models = ["settlement", "final_settlement", "convergence", "final_convergence"]
    valid_types = [
        "RandomForest",
        "LinearRegression",
        "SVR",
        "HistGradientBoostingRegressor",
        "MLPRegressor",
    ]

    if request.model_name not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model name: {request.model_name}")

    if request.model_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {request.model_type}")

    # デフォルトパラメータ
    default_params = {
        "RandomForest": {"n_estimators": 100, "random_state": 42},
        "LinearRegression": {},
        "SVR": {"kernel": "linear", "C": 1.0},
        "HistGradientBoostingRegressor": {"random_state": 42},
        "MLPRegressor": {"hidden_layer_sizes": (100,), "max_iter": 1000},
    }

    parameters = request.parameters or default_params.get(request.model_type, {})

    return schemas.ModelConfigResponse(
        model_name=request.model_name,
        model_type=request.model_type,
        parameters=parameters,
        is_fitted=False,  # 設定更新後は未訓練状態
    )


@router.post("/batch-process", response_model=schemas.BatchProcessResponse)
async def batch_process_folders(
    request: schemas.BatchProcessRequest,
) -> schemas.BatchProcessResponse:
    """
    複数フォルダをバッチ処理
    """
    results = []
    total_time = 0

    for folder_name in request.folder_names:
        start_time = datetime.now()

        try:
            # モック処理結果
            processing_time = np.random.uniform(1.0, 3.0)  # 1-3秒のランダム処理時間

            result = schemas.BatchProcessResult(
                folder_name=folder_name,
                success=True,
                message=f"Successfully processed {folder_name}",
                processing_time=processing_time,
                result_data={
                    "train_score": 0.85 + np.random.random() * 0.1,
                    "validation_score": 0.80 + np.random.random() * 0.1,
                    "num_samples": np.random.randint(1000, 5000),
                },
            )
            results.append(result)

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            result = schemas.BatchProcessResult(
                folder_name=folder_name,
                success=False,
                message=f"Failed to process {folder_name}: {str(e)}",
                processing_time=processing_time,
            )
            results.append(result)

        total_time += processing_time

    successful_count = sum(1 for r in results if r.success)
    failed_count = len(results) - successful_count

    return schemas.BatchProcessResponse(
        results=results,
        total_folders=len(request.folder_names),
        successful_folders=successful_count,
        failed_folders=failed_count,
        total_processing_time=total_time,
    )


@router.post("/additional-data", response_model=schemas.AdditionalDataResponse)
async def get_additional_data(
    request: schemas.AdditionalDataRequest,
) -> schemas.AdditionalDataResponse:
    """
    追加データ（cycle_support, observation_of_face）を生成
    """
    response_data = {"folder_name": request.folder_name}

    if request.include_cycle_support:
        # サイクルサポートのモックデータ
        response_data["cycle_support_data"] = {
            "support_pattern": ["P1", "P2", "P3", "P1", "P2"],
            "support_timing": [0, 24, 48, 72, 96],
            "support_strength": [100, 120, 110, 105, 115],
        }

    if request.include_observation:
        # 観測データのモックデータ
        response_data["observation_data"] = {
            "face_condition": ["Good", "Fair", "Good", "Poor", "Fair"],
            "groundwater_level": [0.5, 0.8, 0.6, 1.2, 0.9],
            "rock_quality": [85, 70, 80, 55, 65],
        }

    if request.include_cycle_support and request.include_observation:
        # 結合データ
        response_data["combined_data"] = {
            "analysis_ready": True,
            "total_records": 5,
            "quality_score": 0.85,
        }

    return schemas.AdditionalDataResponse(**response_data)
