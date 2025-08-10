"""
予測・シミュレーション関連のスキーマ
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum


class ModelType(str, Enum):
    """利用可能なモデルタイプ"""
    random_forest = "random_forest"
    linear_regression = "linear_regression"
    svr = "svr"
    hist_gradient_boosting = "hist_gradient_boosting"
    mlp = "mlp"


class PredictionTarget(str, Enum):
    """予測対象"""
    settlement = "settlement"
    convergence = "convergence"
    final_settlement = "final_settlement"
    final_convergence = "final_convergence"


# リクエストスキーマ
class ModelConfigRequest(BaseModel):
    """モデル設定リクエスト"""
    model_name: str = Field(..., description="モデル名")
    model_type: ModelType = Field(..., description="モデルタイプ")
    params: Optional[Dict[str, Any]] = Field(default={}, description="モデルパラメータ")


class TrainingRequest(BaseModel):
    """モデル訓練リクエスト"""
    model_name: str = Field(..., description="モデル名")
    folder_name: str = Field(default="01-hokkaido-akan", description="データフォルダ名")
    target_columns: Optional[List[str]] = Field(default=None, description="ターゲット列")
    feature_columns: Optional[List[str]] = Field(default=None, description="特徴量列")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="テストデータの割合")


class PredictionRequest(BaseModel):
    """予測リクエスト"""
    model_name: str = Field(..., description="モデル名")
    features: Dict[str, Union[float, int]] = Field(..., description="特徴量データ")
    folder_name: str = Field(default="01-hokkaido-akan", description="データフォルダ名")


class SimulationRequest(BaseModel):
    """シミュレーションリクエスト"""
    folder_name: str = Field(default="01-hokkaido-akan", description="データフォルダ名")
    daily_advance: float = Field(default=2.0, ge=0.1, le=10.0, description="日進量 (m/day)")
    distance_from_face: float = Field(default=50.0, ge=0.0, description="現在の切羽からの距離 (m)")
    max_distance: float = Field(default=200.0, ge=10.0, description="最大距離 (m)")
    prediction_days: int = Field(default=30, ge=1, le=365, description="予測日数")
    recursive: bool = Field(default=True, description="再帰的予測を使用")
    use_models: Dict[str, str] = Field(
        default={
            "settlement": "settlement",
            "convergence": "convergence", 
            "final_settlement": "final_settlement",
            "final_convergence": "final_convergence"
        },
        description="使用するモデル"
    )


class BatchProcessRequest(BaseModel):
    """バッチ処理リクエスト"""
    folder_names: List[str] = Field(..., description="処理対象フォルダ名リスト")
    max_distance_from_face: float = Field(default=200.0, description="最大距離")
    retrain_models: bool = Field(default=True, description="モデルを再訓練するか")


# レスポンススキーマ
class ModelInfo(BaseModel):
    """モデル情報"""
    name: str = Field(..., description="モデル名")
    type: str = Field(..., description="モデルタイプ")
    params: Dict[str, Any] = Field(..., description="モデルパラメータ")
    is_trained: bool = Field(..., description="訓練済みかどうか")
    save_path: Optional[str] = Field(None, description="保存パス")


class TrainingResult(BaseModel):
    """訓練結果"""
    model_name: str = Field(..., description="モデル名")
    training_score: float = Field(..., description="訓練スコア (R²)")
    validation_score: float = Field(..., description="検証スコア (R²)")
    test_score: Optional[float] = Field(None, description="テストスコア (R²)")
    feature_importance: Optional[List[Dict[str, Union[str, float]]]] = Field(None, description="特徴量重要度")
    training_samples: int = Field(..., description="訓練サンプル数")
    validation_samples: int = Field(..., description="検証サンプル数")
    processing_time: float = Field(..., description="処理時間（秒）")


class PredictionResult(BaseModel):
    """予測結果"""
    model_name: str = Field(..., description="モデル名")
    prediction: Union[float, List[float]] = Field(..., description="予測値")
    features: Dict[str, Union[float, int]] = Field(..., description="入力特徴量")
    confidence: Optional[float] = Field(None, description="信頼度")


class SimulationDataPoint(BaseModel):
    """シミュレーション結果の1データポイント"""
    day: int = Field(..., description="日数")
    distance_from_face: float = Field(..., description="切羽からの距離")
    td_position: float = Field(..., description="TD位置")
    settlement_prediction: Dict[str, float] = Field(..., description="沈下量予測")
    convergence_prediction: Dict[str, float] = Field(..., description="変位量予測")
    final_settlement: Optional[float] = Field(None, description="最終沈下量予測")
    final_convergence: Optional[float] = Field(None, description="最終変位量予測")


class SimulationResult(BaseModel):
    """シミュレーション結果"""
    folder_name: str = Field(..., description="データフォルダ名")
    simulation_params: SimulationRequest = Field(..., description="シミュレーションパラメータ")
    data_points: List[SimulationDataPoint] = Field(..., description="シミュレーション結果データ")
    summary: Dict[str, Any] = Field(..., description="結果サマリー")
    processing_time: float = Field(..., description="処理時間（秒）")


class BatchProcessResult(BaseModel):
    """バッチ処理結果"""
    processed_folders: List[str] = Field(..., description="処理済みフォルダ")
    failed_folders: List[str] = Field(..., description="処理失敗フォルダ")
    training_results: List[TrainingResult] = Field(..., description="訓練結果")
    total_processing_time: float = Field(..., description="総処理時間（秒）")
    success_rate: float = Field(..., description="成功率")


class ModelListResponse(BaseModel):
    """モデル一覧レスポンス"""
    models: List[ModelInfo] = Field(..., description="モデル一覧")
    available_types: List[str] = Field(..., description="利用可能なモデルタイプ")


class FeatureImportance(BaseModel):
    """特徴量重要度"""
    feature: str = Field(..., description="特徴量名")
    importance: float = Field(..., description="重要度")
    rank: int = Field(..., description="ランク")


class ModelAnalysis(BaseModel):
    """モデル分析結果"""
    model_name: str = Field(..., description="モデル名")
    feature_importance: List[FeatureImportance] = Field(..., description="特徴量重要度")
    performance_metrics: Dict[str, float] = Field(..., description="性能指標")
    training_info: Dict[str, Any] = Field(..., description="訓練情報")