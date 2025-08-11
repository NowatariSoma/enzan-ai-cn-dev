from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class TimeSeriesDataPoint(BaseModel):
    """時系列データポイント"""
    td: float = Field(..., description="TD (Tunnel Distance)")
    series3m: float = Field(..., description="3m地点のデータ")
    series5m: float = Field(..., description="5m地点のデータ")
    series10m: float = Field(..., description="10m地点のデータ")
    series20m: float = Field(..., description="20m地点のデータ")
    series50m: float = Field(..., description="50m地点のデータ")
    series100m: float = Field(..., description="100m地点のデータ")


class DisplacementSeriesResponse(BaseModel):
    """変位時系列データのレスポンス"""
    data: List[TimeSeriesDataPoint]
    unit: str = Field(default="mm", description="データの単位")
    measurement_type: str = Field(default="displacement", description="計測タイプ")


class SettlementSeriesResponse(BaseModel):
    """沈下時系列データのレスポンス"""
    data: List[TimeSeriesDataPoint]
    unit: str = Field(default="mm", description="データの単位")
    measurement_type: str = Field(default="settlement", description="計測タイプ")


class DistributionDataPoint(BaseModel):
    """分布データポイント"""
    range: str = Field(..., description="値の範囲")
    series3m: int = Field(..., description="3m地点の頻度")
    series5m: int = Field(..., description="5m地点の頻度")
    series10m: int = Field(..., description="10m地点の頻度")
    series20m: int = Field(..., description="20m地点の頻度")
    series50m: int = Field(..., description="50m地点の頻度")
    series100m: int = Field(..., description="100m地点の頻度")


class DisplacementDistributionResponse(BaseModel):
    """変位分布データのレスポンス"""
    data: List[DistributionDataPoint]
    bin_size: float = Field(default=1.0, description="ビンのサイズ")
    measurement_type: str = Field(default="displacement", description="計測タイプ")


class SettlementDistributionResponse(BaseModel):
    """沈下分布データのレスポンス"""
    data: List[DistributionDataPoint]
    bin_size: float = Field(default=1.0, description="ビンのサイズ")
    measurement_type: str = Field(default="settlement", description="計測タイプ")


class TunnelScatterPoint(BaseModel):
    """トンネル散布図データポイント"""
    x: float = Field(..., description="切羽からの距離")
    y: float = Field(..., description="計測日数")
    depth: float = Field(..., description="深度")
    color: str = Field(..., description="カラーコード")


class TunnelScatterResponse(BaseModel):
    """トンネル散布図データのレスポンス"""
    data: List[TunnelScatterPoint]
    x_label: str = Field(default="Distance from Face (m)", description="X軸ラベル")
    y_label: str = Field(default="Measurement Days", description="Y軸ラベル")
    color_scale: str = Field(default="depth", description="カラースケールの基準")


class MeasurementFileInfo(BaseModel):
    """計測ファイル情報"""
    id: str = Field(..., description="ファイルID")
    name: str = Field(..., description="ファイル名")
    description: str = Field(..., description="ファイルの説明")
    created_at: Optional[datetime] = None
    size: Optional[int] = None


class MeasurementFilesResponse(BaseModel):
    """計測ファイル一覧のレスポンス"""
    files: List[MeasurementFileInfo]
    total_count: int


class MeasurementAnalysisRequest(BaseModel):
    """計測データ解析のリクエスト"""
    cycleNumber: str = Field(..., description="計測サイクル番号またはファイル名")
    distanceFromFace: float = Field(..., ge=0, description="切羽からの距離")
    excavationAdvance: float = Field(..., gt=0, description="掘進進捗")


class MeasurementPrediction(BaseModel):
    """計測予測データ"""
    step: int = Field(..., description="ステップ数")
    days: int = Field(..., description="日数")
    prediction1: str = Field(..., description="予測値1")
    prediction2: str = Field(..., description="予測値2")
    prediction3: str = Field(..., description="予測値3")


class MeasurementPredictionsResponse(BaseModel):
    """計測予測データのレスポンス"""
    predictions: List[MeasurementPrediction]
    excavationAdvance: float
    distanceFromFace: float


class ProcessMeasurementRequest(BaseModel):
    """計測ファイル処理のリクエスト"""
    file_path: str = Field(..., description="処理するCSVファイルのパス")
    folder_name: str = Field(default="01-hokkaido-akan", description="データフォルダ名")
    max_distance_from_face: float = Field(default=100.0, gt=0, description="切羽からの最大距離")
    duration_days: int = Field(default=90, gt=0, description="解析対象期間（日数）")


class ProcessedMeasurementResponse(BaseModel):
    """処理済み計測データのレスポンス"""
    data: List[Dict[str, Any]] = Field(..., description="処理済みデータ")
    columns: List[str] = Field(..., description="データのカラム名リスト")
    stats: Dict[str, Any] = Field(..., description="統計情報")
    file_path: str = Field(..., description="処理したファイルのパス")
    processing_params: Dict[str, Any] = Field(..., description="処理パラメータ")


class TDDataPoint(BaseModel):
    """TD別のデータポイント"""
    td: float = Field(..., description="TD値")
    settlements: List[float] = Field(..., description="沈下量のリスト")
    convergences: List[float] = Field(..., description="変位量のリスト")


class DistanceDataResponse(BaseModel):
    """距離別データのレスポンス"""
    dct_df_td: Dict[str, List[TDDataPoint]] = Field(..., description="距離別のTDデータ")
    settlements: Dict[str, List[float]] = Field(..., description="距離別の沈下量データ")
    convergences: Dict[str, List[float]] = Field(..., description="距離別の変位量データ")
    settlements_columns: List[str] = Field(..., description="沈下量のカラム名")
    convergences_columns: List[str] = Field(..., description="変位量のカラム名")
    distances: List[str] = Field(..., description="距離のリスト")
    df_all: List[Dict[str, Any]] = Field(..., description="全計測データ")


class MLDatasetResponse(BaseModel):
    settlement_data: List[Dict[str, float]] = Field(..., description="沈下量データ")
    convergence_data: List[Dict[str, float]] = Field(..., description="変位量データ")


class MLDatasetRequest(BaseModel):
    folder_name: str = Field(..., description="データフォルダ名")
    max_distance_from_face: float = Field(..., description="切羽からの最大距離")