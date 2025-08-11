from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class ModelInfo(BaseModel):
    name: str = Field(..., description="モデル名")
    type: str = Field(..., description="モデルの種類")
    params: Dict[str, Any] = Field(..., description="モデルのパラメータ")
    is_fitted: bool = Field(..., description="学習済みかどうか")


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
    validation_score: float = Field(..., description="検証スコア")
    feature_importance: Dict[str, float] = Field(..., description="特徴量重要度")
    metrics: Dict[str, float] = Field(..., description="評価指標")
    scatter_train: str = Field(..., description="訓練データの散布図（Base64）")
    scatter_validate: str = Field(..., description="検証データの散布図（Base64）")
    feature_importance_plot: str = Field(..., description="特徴量重要度プロット（Base64）")
    train_predictions: Optional[List[float]] = Field(None, description="訓練データの予測値")
    validate_predictions: Optional[List[float]] = Field(None, description="検証データの予測値")


class ModelPredictRequest(BaseModel):
    model_name: str = Field(..., description="モデル名")
    data: List[Dict[str, float]] = Field(..., description="予測対象データ")


class ModelPredictResponse(BaseModel):
    predictions: List[float] = Field(..., description="予測結果")
    model_name: str = Field(..., description="使用したモデル名")


class ProcessEachRequest(BaseModel):
    model_name: str = Field(..., description="使用するモデル名")
    folder_name: str = Field(default="01-hokkaido-akan", description="データフォルダ名")
    max_distance_from_face: float = Field(default=100.0, description="切羽からの最大距離")
    data_type: str = Field(..., description="データタイプ：settlement または convergence")
    td: Optional[float] = Field(None, description="訓練・検証データ分割用のTD値")
    predict_final: bool = Field(default=True, description="True: 最終変位量を予測, False: 現在の変位量を予測")


class ProcessEachResponse(BaseModel):
    model_name: str = Field(..., description="使用したモデル名")
    data_type: str = Field(..., description="処理したデータタイプ")
    metrics: Dict[str, float] = Field(..., description="評価指標")
    scatter_train: str = Field(..., description="訓練データ散布図（Base64）")
    scatter_validate: str = Field(..., description="検証データ散布図（Base64）")
    feature_importance_plot: str = Field(..., description="特徴量重要度プロット（Base64）")
    feature_importance: Dict[str, float] = Field(..., description="特徴量重要度")
    train_count: int = Field(..., description="訓練データ数")
    validate_count: int = Field(..., description="検証データ数")
    train_predictions: List[float] = Field(..., description="訓練データ予測値")
    validate_predictions: List[float] = Field(..., description="検証データ予測値")
    train_actual: List[float] = Field(..., description="訓練データ実測値")
    validate_actual: List[float] = Field(..., description="検証データ実測値")