from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    csv_files: List[str] = Field(..., description="解析対象のCSVファイルリスト")
    model_type: str = Field("Random Forest", description="使用するモデルタイプ")
    max_distance_from_face: float = Field(100.0, description="切羽からの最大距離")
    should_train: bool = Field(True, description="モデルを訓練するかどうか")


class AnalysisResult(BaseModel):
    folder_name: str = Field(..., description="解析フォルダ名")
    model_type: str = Field(..., description="使用したモデルタイプ")
    train_score: float = Field(..., description="訓練スコア")
    validation_score: float = Field(..., description="検証スコア")
    feature_importance: Dict[str, float] = Field(..., description="特徴量重要度")
    predictions: List[Dict[str, float]] = Field(..., description="予測結果")
    timestamp: datetime = Field(default_factory=datetime.now, description="解析実行時刻")


class FileUploadResponse(BaseModel):
    filename: str = Field(..., description="アップロードされたファイル名")
    file_path: str = Field(..., description="ファイルの保存パス")
    size: int = Field(..., description="ファイルサイズ（バイト）")


class CorrelationData(BaseModel):
    features: List[str] = Field(..., description="特徴量名のリスト")
    correlation_matrix: List[List[float]] = Field(..., description="相関行列")
    heatmap_data: List[Dict[str, Any]] = Field(..., description="ヒートマップ用データ")
