import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI A-Measure API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # CORS設定
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:3004",
        "http://localhost:3005", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3004",
        "http://127.0.0.1:3005",
        "*"
    ]  # 開発環境のため全てのオリジンを許可

    # データベース設定（Django管理者画面のMySQLと同じ）
    MYSQL_ENGINE: str = os.getenv("MYSQL_ENGINE", "mysql")
    MYSQL_DB: str = os.getenv("MYSQL_DB", "mlp_db")
    MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "mysql_pass")
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT: str = os.getenv("MYSQL_PORT", "3306")

    @property
    def DATABASE_URL(self) -> str:
        return f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DB}"

    # ファイルアップロード設定
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    UPLOAD_DIR: Path = Path("/tmp/ai_ameasure_uploads")

    # データフォルダ設定（環境変数でオーバーライド可能）
    DATA_FOLDER: Path = Path(os.getenv("DATA_FOLDER", "/home/nowatari/repos/enzan-ai-cn-dev/data_folder"))
    OUTPUT_FOLDER: Path = Path(os.getenv("OUTPUT_FOLDER", "/home/nowatari/repos/enzan-ai-cn-dev/microservices/ai_ameasure/output"))

    # モデル設定
    MODELS_DIR: Path = (
        Path(__file__).parent.parent.parent.parent.parent / "ai_ameasure" / "app" / "models"
    )
    CONFIG_DIR: Path = Path(__file__).parent.parent.parent.parent.parent / "ai_ameasure" / "config"
    MASTER_DATA_DIR: Path = (
        Path(__file__).parent.parent.parent.parent.parent / "ai_ameasure" / "app" / "master"
    )

    # 計算設定
    DEFAULT_MAX_DISTANCE_FROM_FACE: float = 100.0
    DEFAULT_PREDICTION_TD: int = 500

    # その他の設定
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    model_config = {"case_sensitive": True, "env_file": ".env"}


settings = Settings()

# 必要なディレクトリの作成
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_FOLDER.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
