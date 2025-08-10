import pytest
import tempfile
import os
from pathlib import Path
from fastapi.testclient import TestClient

# テスト環境変数を設定
os.environ["ENV"] = "test"
if os.path.exists(".env.test"):
    from dotenv import load_dotenv
    load_dotenv(".env.test")

from app.main import app
from app.core.config import settings


@pytest.fixture(autouse=True)
def setup_test_env():
    """テスト環境のセットアップ"""
    # テスト用の一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as tmpdir:
        original_upload_dir = settings.UPLOAD_DIR
        settings.UPLOAD_DIR = Path(tmpdir) / "uploads"
        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        yield
        # テスト後にリセット
        settings.UPLOAD_DIR = original_upload_dir


@pytest.fixture
def client():
    """FastAPIテストクライアントのフィクスチャ"""
    return TestClient(app)


@pytest.fixture
def sample_displacement_request():
    """変位解析リクエストのサンプルデータ"""
    return {
        "folder": "01-hokkaido-akan",
        "model": "Random Forest",
        "prediction_td": 500,
        "max_distance": 100.0
    }


@pytest.fixture
def sample_model_train_request():
    """モデル訓練リクエストのサンプルデータ"""
    return {
        "model_name": "Random Forest",
        "data_path": "/path/to/data.csv",
        "target_column": "displacement_a",
        "feature_columns": ["TD", "Distance_from_face", "Ground_condition"]
    }


@pytest.fixture
def sample_model_predict_request():
    """モデル予測リクエストのサンプルデータ"""
    return {
        "model_name": "Random Forest",
        "data": [
            {"TD": 100, "Distance_from_face": 50, "Ground_condition": 1},
            {"TD": 200, "Distance_from_face": 75, "Ground_condition": 2}
        ]
    }


@pytest.fixture
def sample_analysis_request():
    """解析リクエストのサンプルデータ"""
    return {
        "csv_files": ["file1.csv", "file2.csv"],
        "model_type": "Random Forest",
        "max_distance_from_face": 100.0,
        "should_train": True
    }