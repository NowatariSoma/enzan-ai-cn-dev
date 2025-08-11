from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    """変位予測シミュレーションのリクエスト"""

    folder_name: str = Field(..., description="シミュレーション対象フォルダ名")
    daily_advance: float = Field(..., gt=0, le=10.0, description="日進量 (m/day)")
    distance_from_face: float = Field(..., ge=0, description="現在の切羽からの距離 (m)")
    max_distance: float = Field(default=200.0, gt=0, description="最大距離 (m)")
    recursive: bool = Field(default=True, description="再帰的予測を使用するか")


class SimulationDataPoint(BaseModel):
    """シミュレーション結果の各データポイント"""

    td_no: int = Field(..., description="TD番号")
    date: datetime = Field(..., description="日付")
    distance_from_face: float = Field(..., description="切羽からの距離")
    position_id: str = Field(..., description="位置ID")
    settlement: float = Field(..., description="沈下量")
    settlement_prediction: float = Field(..., description="沈下量予測値")
    convergence: float = Field(..., description="変位量")
    convergence_prediction: float = Field(..., description="変位量予測値")


class SimulationResponse(BaseModel):
    """変位予測シミュレーションのレスポンス"""

    folder_name: str
    simulation_data: List[SimulationDataPoint]
    daily_advance: float
    distance_from_face: float
    recursive: bool
    timestamp: datetime = Field(default_factory=datetime.now)


class ChartDataRequest(BaseModel):
    """チャートデータ生成のリクエスト"""

    folder_name: str = Field(..., description="対象フォルダ名")
    chart_type: str = Field(
        ...,
        pattern="^(displacement|settlement|convergence|combined)$",
        description="チャートタイプ",
    )
    include_predictions: bool = Field(default=True, description="予測値を含めるか")


class ChartDataPoint(BaseModel):
    """チャートデータのポイント"""

    x: float
    y: float
    series: str
    label: Optional[str] = None


class ChartDataResponse(BaseModel):
    """チャートデータのレスポンス"""

    chart_type: str
    data: List[ChartDataPoint]
    x_label: str
    y_label: str
    title: str


class ModelConfigRequest(BaseModel):
    """モデル設定更新のリクエスト"""

    model_name: str = Field(
        ..., description="モデル名 (settlement, final_settlement, convergence, final_convergence)"
    )
    model_type: str = Field(
        ..., description="モデルタイプ (RandomForest, LinearRegression, SVR, etc.)"
    )
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="モデルパラメータ")


class ModelConfigResponse(BaseModel):
    """モデル設定のレスポンス"""

    model_name: str
    model_type: str
    parameters: Dict[str, Any]
    is_fitted: bool
    updated_at: datetime = Field(default_factory=datetime.now)


class ModelConfigListResponse(BaseModel):
    """全モデル設定のレスポンス"""

    configs: Dict[str, ModelConfigResponse]


class BatchProcessRequest(BaseModel):
    """複数フォルダのバッチ処理リクエスト"""

    folder_names: List[str] = Field(..., min_items=1, description="処理対象フォルダ名のリスト")
    max_distance_from_face: float = Field(default=200.0, gt=0, description="最大距離")
    include_charts: bool = Field(default=True, description="チャートを生成するか")


class BatchProcessResult(BaseModel):
    """バッチ処理の個別結果"""

    folder_name: str
    success: bool
    message: str
    processing_time: float
    result_data: Optional[Dict[str, Any]] = None


class BatchProcessResponse(BaseModel):
    """バッチ処理のレスポンス"""

    results: List[BatchProcessResult]
    total_folders: int
    successful_folders: int
    failed_folders: int
    total_processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)


class AdditionalDataRequest(BaseModel):
    """追加データ生成のリクエスト"""

    folder_name: str = Field(..., description="対象フォルダ名")
    include_cycle_support: bool = Field(
        default=True, description="サイクルサポートデータを含めるか"
    )
    include_observation: bool = Field(default=True, description="観測データを含めるか")


class AdditionalDataResponse(BaseModel):
    """追加データのレスポンス"""

    folder_name: str
    cycle_support_data: Optional[Dict[str, Any]] = None
    observation_data: Optional[Dict[str, Any]] = None
    combined_data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
