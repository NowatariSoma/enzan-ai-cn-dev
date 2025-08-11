from typing import List, Dict, Optional, Any
from pydantic import BaseModel


class ChartRequest(BaseModel):
    """チャート生成リクエスト"""
    measurement_data: List[Dict[str, Any]]
    chart_type: str  # "settlement" or "convergence"
    max_distance_from_face: Optional[float] = 200.0


class HistogramChartRequest(BaseModel):
    """ヒストグラムチャート生成リクエスト"""
    measurement_data: List[Dict[str, Any]]
    chart_type: str  # "settlement" or "convergence"
    max_distance_from_face: Optional[float] = 200.0


class MultiChartRequest(BaseModel):
    """複数チャート生成リクエスト"""
    measurement_data: List[Dict[str, Any]]
    chart_types: List[str]  # ["settle", "conv", "settle_hist", "conv_hist"]
    max_distance_from_face: Optional[float] = 200.0


class ChartResponse(BaseModel):
    """チャート生成レスポンス"""
    success: bool
    message: str
    file_path: Optional[str] = None
    files: Optional[List[str]] = None


class DataRequest(BaseModel):
    """データ取得リクエスト"""
    measurement_data: List[Dict[str, Any]]
    max_distance_from_face: Optional[float] = 200.0
    include_distance_data: Optional[bool] = False


class DataResponse(BaseModel):
    """データ取得レスポンス"""
    success: bool
    message: str
    data: Dict[str, Any]


class HeatmapRequest(BaseModel):
    """ヒートマップ生成リクエスト"""
    folder_name: str
    x_columns: List[str]
    y_column: str
    correlation_method: Optional[str] = "pearson"  # "pearson", "spearman", "kendall"


class HeatmapDataRequest(BaseModel):
    """ヒートマップデータ取得リクエスト"""
    folder_name: str
    features: Optional[List[str]] = None