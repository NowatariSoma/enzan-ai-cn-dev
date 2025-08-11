from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path
import os


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI A-Measure API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # CORS設定
    BACKEND_CORS_ORIGINS: list[str] = ["*"]  # 開発環境のため全てのオリジンを許可
    
    # ファイルアップロード設定
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    UPLOAD_DIR: Path = Path("/tmp/ai_ameasure_uploads")
    
    # データフォルダ設定（環境変数でオーバーライド可能）
    DATA_FOLDER: Path = Path(os.getenv("DATA_FOLDER", "/app/data"))
    OUTPUT_FOLDER: Path = Path(os.getenv("OUTPUT_FOLDER", "/app/output"))
    
    # モデル設定
    MODELS_DIR: Path = Path(__file__).parent.parent.parent.parent.parent / "ai_ameasure" / "app" / "models"
    CONFIG_DIR: Path = Path(__file__).parent.parent.parent.parent.parent / "ai_ameasure" / "config"
    MASTER_DATA_DIR: Path = Path(__file__).parent.parent.parent.parent.parent / "ai_ameasure" / "app" / "master"
    
    # 計算設定
    DEFAULT_MAX_DISTANCE_FROM_FACE: float = 100.0
    DEFAULT_PREDICTION_TD: int = 500
    
    model_config = {
        "case_sensitive": True,
        "env_file": ".env"
    }


settings = Settings()

# 必要なディレクトリの作成
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_FOLDER.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)