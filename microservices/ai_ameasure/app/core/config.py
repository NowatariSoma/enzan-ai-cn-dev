from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI A-Measure API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # CORS設定
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:8000"]
    
    # ファイルアップロード設定
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    UPLOAD_DIR: Path = Path("/tmp/ai_ameasure_uploads")
    
    # データフォルダ設定
    DATA_FOLDER: Path = Path("/home/nowatari/repos/enzan-koubou/ai-cn/data_folder")
    OUTPUT_FOLDER: Path = Path("/home/nowatari/repos/enzan-koubou/ai-cn/output")
    
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