from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import numpy as np
from pathlib import Path

from app import schemas
from app.core.config import settings

router = APIRouter()


# モック用のモデル情報
MOCK_MODELS = {
    "Random Forest": {
        "type": "RandomForest",
        "params": {"n_estimators": 100, "random_state": 42},
        "is_fitted": True
    },
    "Linear Regression": {
        "type": "LinearRegression",
        "params": {},
        "is_fitted": True
    },
    "SVR": {
        "type": "SVR",
        "params": {"kernel": "linear", "C": 1.0},
        "is_fitted": True
    },
    "HistGradientBoosting": {
        "type": "HistGradientBoostingRegressor",
        "params": {"random_state": 42},
        "is_fitted": False
    },
    "MLP": {
        "type": "MLPRegressor",
        "params": {"hidden_layer_sizes": (100,), "max_iter": 1000},
        "is_fitted": False
    }
}


@router.get("/", response_model=schemas.ModelListResponse)
async def get_models() -> schemas.ModelListResponse:
    """
    利用可能なモデル一覧を取得
    """
    models = []
    for name, info in MOCK_MODELS.items():
        models.append(schemas.ModelInfo(
            name=name,
            type=info["type"],
            params=info["params"],
            is_fitted=info["is_fitted"]
        ))
    
    return schemas.ModelListResponse(models=models)


@router.post("/train", response_model=schemas.ModelTrainResponse)
async def train_model(
    request: schemas.ModelTrainRequest
) -> schemas.ModelTrainResponse:
    """
    モデルを訓練する
    """
    if request.model_name not in MOCK_MODELS:
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
    
    # モック訓練結果を生成
    train_score = 0.85 + np.random.random() * 0.1  # 0.85-0.95の間
    validation_score = train_score - 0.05 - np.random.random() * 0.05  # 訓練スコアより少し低い
    
    # モック特徴量重要度
    features = request.feature_columns or [
        "TD", "Distance_from_face", "Excavation_advance", 
        "Ground_condition", "Support_type"
    ]
    feature_importance = {}
    for feature in features:
        feature_importance[feature] = np.random.random() * 0.2
    
    # 正規化
    total = sum(feature_importance.values())
    if total > 0:
        feature_importance = {k: v/total for k, v in feature_importance.items()}
    
    # モデルを訓練済みに更新
    MOCK_MODELS[request.model_name]["is_fitted"] = True
    
    return schemas.ModelTrainResponse(
        model_name=request.model_name,
        train_score=train_score,
        validation_score=validation_score,
        feature_importance=feature_importance
    )


@router.post("/predict", response_model=schemas.ModelPredictResponse)
async def predict(
    request: schemas.ModelPredictRequest
) -> schemas.ModelPredictResponse:
    """
    モデルで予測を実行
    """
    if request.model_name not in MOCK_MODELS:
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
    
    if not MOCK_MODELS[request.model_name]["is_fitted"]:
        raise HTTPException(status_code=400, detail=f"Model {request.model_name} is not trained yet")
    
    # モック予測結果を生成
    predictions = []
    for data_point in request.data:
        # 入力値に基づいて何らかの計算をシミュレート
        base_value = sum(data_point.values()) / len(data_point)
        noise = (np.random.random() - 0.5) * 0.2
        prediction = base_value * 0.8 + noise
        predictions.append(prediction)
    
    return schemas.ModelPredictResponse(
        predictions=predictions,
        model_name=request.model_name
    )


@router.get("/types", response_model=List[str])
async def get_model_types() -> List[str]:
    """
    利用可能なモデルタイプ一覧を取得
    """
    return list(set(info["type"] for info in MOCK_MODELS.values()))