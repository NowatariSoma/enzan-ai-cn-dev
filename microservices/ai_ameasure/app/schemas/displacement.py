from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DisplacementData(BaseModel):
    distance_from_face: float = Field(..., description="距離（切羽からの距離）")
    displacement_a: float = Field(..., description="変位量A")
    displacement_b: float = Field(..., description="変位量B")
    displacement_c: float = Field(..., description="変位量C")
    displacement_a_prediction: Optional[float] = Field(None, description="変位量A予測値")
    displacement_b_prediction: Optional[float] = Field(None, description="変位量B予測値")
    displacement_c_prediction: Optional[float] = Field(None, description="変位量C予測値")


class DisplacementAnalysisRequest(BaseModel):
    folder: str = Field(
        ..., description="フォルダ名", json_schema_extra={"example": "01-hokkaido-akan"}
    )
    model: str = Field("Random Forest", description="使用するモデル")
    prediction_td: int = Field(500, description="予測TD")
    max_distance: float = Field(100.0, description="最大距離")


class DisplacementAnalysisResponse(BaseModel):
    chart_data: List[DisplacementData] = Field(..., description="チャートデータ")
    train_r_squared_a: float = Field(..., description="訓練データR²値（A）")
    train_r_squared_b: float = Field(..., description="訓練データR²値（B）")
    validation_r_squared_a: float = Field(..., description="検証データR²値（A）")
    validation_r_squared_b: float = Field(..., description="検証データR²値（B）")
    feature_importance_a: List[Dict[str, Any]] = Field(..., description="特徴量重要度（A）")
    feature_importance_b: List[Dict[str, Any]] = Field(..., description="特徴量重要度（B）")


class LocalAnalysisRequest(BaseModel):
    folder: str = Field(..., description="フォルダ名")
    cycle_number: str = Field(..., description="サイクル番号（CSVファイル名）")
    distance_from_face: float = Field(..., description="切羽からの距離")
    daily_advance: float = Field(..., description="日進量 (m/day)")
    max_distance: float = Field(100.0, description="最大距離")


class LocalAnalysisResponse(BaseModel):
    cycle_number: str = Field(..., description="サイクル番号")
    td: float = Field(..., description="TD値")
    prediction_data: List[Dict[str, Any]] = Field(..., description="予測データ")
    simulation_data: List[Dict[str, Any]] = Field(..., description="シミュレーションデータ")
    charts: Dict[str, str] = Field(..., description="生成されたチャートのパス")


class ScatterData(BaseModel):
    actual: float
    predicted: float


class HeatmapData(BaseModel):
    x: str
    y: str
    value: float


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class ChartGenerationRequest(BaseModel):
    folder: str = Field(..., description="フォルダ名")
    chart_type: str = Field(..., description="チャートタイプ (settlement/convergence)")
    cycle_number: Optional[str] = Field(None, description="サイクル番号")
    include_predictions: bool = Field(True, description="予測値を含めるか")


class ChartGenerationResponse(BaseModel):
    chart_path: str = Field(..., description="チャートファイルのパス")
    chart_type: str = Field(..., description="チャートタイプ")
    data_points: int = Field(..., description="データポイント数")
    generated_at: datetime = Field(default_factory=datetime.now)
