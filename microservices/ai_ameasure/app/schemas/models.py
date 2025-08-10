from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ModelInfo(BaseModel):
    name: str = Field(..., description="モデル名")
    type: str = Field(..., description="モデルタイプ")
    params: Dict[str, Any] = Field(default_factory=dict, description="モデルパラメータ")
    is_fitted: bool = Field(False, description="学習済みかどうか")


class ModelListResponse(BaseModel):
    models: List[ModelInfo] = Field(..., description="利用可能なモデル一覧")


class ModelTrainRequest(BaseModel):
    model_name: str = Field(..., description="モデル名")
    data_path: str = Field(..., description="訓練データのパス")
    target_column: str = Field(..., description="予測対象のカラム名")
    feature_columns: Optional[List[str]] = Field(None, description="特徴量として使用するカラム名のリスト")


class ModelTrainResponse(BaseModel):
    model_name: str = Field(..., description="モデル名")
    train_score: float = Field(..., description="訓練スコア")
    validation_score: Optional[float] = Field(None, description="検証スコア")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="特徴量重要度")


class ModelPredictRequest(BaseModel):
    model_name: str = Field(..., description="使用するモデル名")
    data: List[Dict[str, float]] = Field(..., description="予測対象データ")


class ModelPredictResponse(BaseModel):
    predictions: List[float] = Field(..., description="予測結果")
    model_name: str = Field(..., description="使用したモデル名")