from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class FolderListResponse(BaseModel):
    """フォルダ一覧レスポンス"""
    folders: List[str]

class MeasurementFileListResponse(BaseModel):
    """計測ファイル一覧レスポンス"""
    files: List[str]

class WholeAnalysisRequest(BaseModel):
    """全体分析リクエスト"""
    folder_name: str
    model_name: str
    td: Optional[float] = None
    max_distance_from_face: float = 100

class WholeAnalysisResponse(BaseModel):
    """全体分析レスポンス"""
    status: str
    message: str
    training_metrics: Optional[Dict[str, Any]] = None  # 学習メトリクス
    scatter_data: Optional[Dict[str, Any]] = None  # 散布図データ
    feature_importance: Optional[Dict[str, Any]] = None  # 特徴量重要度
    model_files_saved: bool = False  # モデルファイル保存状況

class LocalAnalysisRequest(BaseModel):
    """局所分析リクエスト"""
    folder_name: str
    ameasure_file: str
    max_distance_from_face: float = 100
    distance_from_face: Optional[float] = None
    daily_advance: Optional[float] = None

class LocalAnalysisResponse(BaseModel):
    """局所分析レスポンス"""
    cycle_no: int
    td: float
    prediction_charts: Dict[str, str]  # settlement, convergence charts (Base64)
    simulation_charts: Dict[str, str]  # simulation charts (Base64)
    prediction_data: List[Dict[str, Any]]
    csv_path: Optional[str] = None

class ModelListResponse(BaseModel):
    """モデル一覧レスポンス"""
    models: List[str]