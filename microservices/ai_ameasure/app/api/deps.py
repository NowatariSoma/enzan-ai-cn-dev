from typing import Generator
from pathlib import Path

from app.core.config import settings


def get_upload_path() -> Path:
    """
    アップロードディレクトリのパスを取得
    """
    return settings.UPLOAD_DIR


def ensure_upload_dir() -> None:
    """
    アップロードディレクトリが存在することを確認
    """
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)