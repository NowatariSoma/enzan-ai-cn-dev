"""
機械学習予測・シミュレーション関連のAPIエンドポイント
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from app import schemas
from app.core.prediction_engine import PredictionEngine
from app.schemas.prediction import (
    BatchProcessRequest,
    BatchProcessResult,
    ModelConfigRequest,
    ModelInfo,
    ModelListResponse,
    PredictionRequest,
    PredictionResult,
    SimulationRequest,
    SimulationResult,
    TrainingRequest,
    TrainingResult,
)
from fastapi import APIRouter, BackgroundTasks, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter()

# 予測エンジンのインスタンス
prediction_engine = PredictionEngine()


@router.get("/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """
    利用可能なモデル一覧を取得
    """
    try:
        result = prediction_engine.list_models()
        return ModelListResponse(**result)
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str) -> ModelInfo:
    """
    特定のモデル情報を取得
    """
    try:
        result = prediction_engine.get_model_info(model_name)
        return ModelInfo(**result)
    except Exception as e:
        logger.error(f"Error getting model info for {model_name}: {e}")
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")


@router.post("/models/{model_name}/train", response_model=TrainingResult)
async def train_model(
    model_name: str, request: TrainingRequest, background_tasks: BackgroundTasks
) -> TrainingResult:
    """
    モデルを訓練
    """
    try:
        # ターゲットタイプを決定
        target_type = "settlement" if "settlement" in model_name else "convergence"

        result = prediction_engine.train_model(
            model_name=model_name,
            folder_name=request.folder_name,
            target_type=target_type,
            test_size=request.test_size,
        )

        return TrainingResult(**result)

    except Exception as e:
        logger.error(f"Error training model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_name}/predict", response_model=PredictionResult)
async def predict(model_name: str, request: PredictionRequest) -> PredictionResult:
    """
    モデルで予測を実行
    """
    try:
        result = prediction_engine.predict(
            model_name=model_name, features=request.features, folder_name=request.folder_name
        )

        return PredictionResult(**result)

    except Exception as e:
        logger.error(f"Error during prediction with {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate", response_model=SimulationResult)
async def simulate_displacement(request: SimulationRequest) -> SimulationResult:
    """
    変位・沈下シミュレーションを実行
    """
    try:
        result = prediction_engine.simulate_displacement(
            folder_name=request.folder_name,
            daily_advance=request.daily_advance,
            distance_from_face=request.distance_from_face,
            max_distance=request.max_distance,
            prediction_days=request.prediction_days,
            recursive=request.recursive,
            use_models=request.use_models,
        )

        return SimulationResult(**result)

    except Exception as e:
        logger.error(f"Error during simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-process", response_model=BatchProcessResult)
async def batch_process(
    request: BatchProcessRequest, background_tasks: BackgroundTasks
) -> BatchProcessResult:
    """
    バッチ処理（複数フォルダの一括処理）
    """
    try:
        import time

        start_time = time.time()

        processed_folders = []
        failed_folders = []
        training_results = []

        for folder_name in request.folder_names:
            try:
                logger.info(f"Processing folder: {folder_name}")

                # 各モデルを訓練（再訓練が有効な場合）
                if request.retrain_models:
                    models_to_train = [
                        "settlement",
                        "convergence",
                        "final_settlement",
                        "final_convergence",
                    ]

                    for model_name in models_to_train:
                        try:
                            target_type = (
                                "settlement" if "settlement" in model_name else "convergence"
                            )

                            result = prediction_engine.train_model(
                                model_name=f"{folder_name}_{model_name}",
                                folder_name=folder_name,
                                target_type=target_type,
                            )

                            training_results.append(TrainingResult(**result))

                        except Exception as e:
                            logger.warning(f"Failed to train {model_name} for {folder_name}: {e}")

                processed_folders.append(folder_name)
                logger.info(f"Successfully processed {folder_name}")

            except Exception as e:
                logger.error(f"Failed to process {folder_name}: {e}")
                failed_folders.append(folder_name)

        total_processing_time = time.time() - start_time
        success_rate = (
            len(processed_folders) / len(request.folder_names) if request.folder_names else 0.0
        )

        result = {
            "processed_folders": processed_folders,
            "failed_folders": failed_folders,
            "training_results": [result.dict() for result in training_results],
            "total_processing_time": total_processing_time,
            "success_rate": success_rate,
        }

        return BatchProcessResult(**result)

    except Exception as e:
        logger.error(f"Error during batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_name}/update-config")
async def update_model_config(model_name: str, request: ModelConfigRequest):
    """
    モデル設定を更新
    """
    try:
        model = prediction_engine.model_manager.update_model_type(
            model_name=request.model_name, new_type=request.model_type, new_params=request.params
        )

        # 設定を保存
        prediction_engine.model_manager.save_config()

        return {"message": f"Model {model_name} configuration updated successfully"}

    except Exception as e:
        logger.error(f"Error updating model config for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """
    モデルを削除
    """
    try:
        # モデルファイルを削除
        model_path = prediction_engine.model_manager.config.get_model_save_path(model_name)
        if Path(model_path).exists():
            Path(model_path).unlink()

        # メモリからも削除
        if model_name in prediction_engine.model_manager.models:
            del prediction_engine.model_manager.models[model_name]

        return {"message": f"Model {model_name} deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    予測エンジンのヘルスチェック
    """
    try:
        # 簡単なテスト予測を実行
        test_features = {"TD(m)": 100.0, "切羽TD": 100.0, "実TD": 100.0, "ｻｲｸﾙNo": 1}

        status = {
            "status": "healthy",
            "available_models": len(prediction_engine.list_models()["models"]),
            "available_types": prediction_engine.model_manager.get_available_model_types(),
        }

        return status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


# Quick prediction endpoints for common use cases
@router.get("/quick-predict/settlement")
async def quick_predict_settlement(
    td: float = 100.0, cycle: int = 1, folder_name: str = "01-hokkaido-akan"
):
    """
    沈下量の簡易予測
    """
    try:
        features = {"TD(m)": td, "切羽TD": td, "実TD": td, "ｻｲｸﾙNo": cycle}

        result = prediction_engine.predict("settlement", features, folder_name)
        return result

    except Exception as e:
        logger.error(f"Error in quick settlement prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quick-predict/convergence")
async def quick_predict_convergence(
    td: float = 100.0, cycle: int = 1, folder_name: str = "01-hokkaido-akan"
):
    """
    変位量の簡易予測
    """
    try:
        features = {"TD(m)": td, "切羽TD": td, "実TD": td, "ｻｲｸﾙNo": cycle}

        result = prediction_engine.predict("convergence", features, folder_name)
        return result

    except Exception as e:
        logger.error(f"Error in quick convergence prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
