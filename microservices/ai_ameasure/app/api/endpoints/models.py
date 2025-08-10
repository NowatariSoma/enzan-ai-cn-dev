from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from app import schemas
from app.core.config import settings

router = APIRouter()


# 実際のモデルインスタンス
MOCK_MODELS = {
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42),
        "type": "RandomForestRegressor",
        "is_fitted": False
    },
    "Linear Regression": {
        "model": LinearRegression(),
        "type": "LinearRegression",
        "is_fitted": False
    },
    "SVR": {
        "model": SVR(kernel='linear', C=1.0, epsilon=0.2),
        "type": "SVR",
        "is_fitted": False
    },
    "HistGradientBoostingRegressor": {
        "model": HistGradientBoostingRegressor(random_state=42),
        "type": "HistGradientBoostingRegressor",
        "is_fitted": False
    },
    "MLP": {
        "model": MLPRegressor(hidden_layer_sizes=(100, 100, 50), max_iter=1000, random_state=42),
        "type": "MLPRegressor",
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
        # 実際のモデルインスタンスからパラメータを取得
        params = info["model"].get_params()
        models.append(schemas.ModelInfo(
            name=name,
            type=info["type"],
            params=params,
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
    
    model_info = MOCK_MODELS[request.model_name]
    model = model_info["model"]
    
    # モック訓練データを生成（実際の使用時はrequestから取得）
    # ここでは簡単なモックデータを使用
    n_samples = 100
    n_features = len(request.feature_columns) if request.feature_columns else 5
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    
    # モデルを訓練
    model.fit(X, y)
    
    # 訓練スコアを計算
    train_score = model.score(X, y)
    
    # 検証スコア（簡易版）
    validation_score = train_score - 0.05 - np.random.random() * 0.05
    
    # 特徴量重要度（対応するモデルの場合のみ）
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        features = request.feature_columns or [f"feature_{i}" for i in range(n_features)]
        for i, feature in enumerate(features):
            feature_importance[feature] = float(model.feature_importances_[i])
    else:
        # 特徴量重要度が利用できないモデルの場合
        features = request.feature_columns or [f"feature_{i}" for i in range(n_features)]
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
    
    model_info = MOCK_MODELS[request.model_name]
    model = model_info["model"]
    
    # 入力データを適切な形式に変換
    predictions = []
    for data_point in request.data:
        # 辞書形式のデータを配列に変換
        features = list(data_point.values())
        X = np.array(features).reshape(1, -1)
        
        # 予測を実行
        prediction = model.predict(X)[0]
        predictions.append(float(prediction))
    
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