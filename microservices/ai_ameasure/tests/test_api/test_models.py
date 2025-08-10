import pytest
from fastapi import status


class TestModelsEndpoints:
    """モデル管理エンドポイントのテスト"""
    
    def test_get_models_list(self, client):
        """モデル一覧の取得"""
        response = client.get("/api/v1/models/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0
        
        # 各モデルの構造を確認
        for model in data["models"]:
            assert "name" in model
            assert "type" in model
            assert "params" in model
            assert "is_fitted" in model
    
    def test_train_model_success(self, client, sample_model_train_request):
        """モデル訓練の成功"""
        response = client.post(
            "/api/v1/models/train",
            json=sample_model_train_request
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "model_name" in data
        assert "train_score" in data
        assert "validation_score" in data
        assert "feature_importance" in data
        
        assert data["model_name"] == sample_model_train_request["model_name"]
        assert 0 <= data["train_score"] <= 1
        assert 0 <= data["validation_score"] <= 1
        assert isinstance(data["feature_importance"], dict)
    
    def test_train_nonexistent_model(self, client):
        """存在しないモデルの訓練"""
        request_data = {
            "model_name": "NonExistentModel",
            "data_path": "/path/to/data.csv",
            "target_column": "displacement_a"
        }
        
        response = client.post(
            "/api/v1/models/train",
            json=request_data
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "NonExistentModel" in response.json()["detail"]
    
    def test_predict_with_trained_model(self, client, sample_model_predict_request):
        """訓練済みモデルでの予測"""
        # まずモデルを訓練
        train_request = {
            "model_name": "Random Forest",
            "data_path": "/path/to/data.csv",
            "target_column": "displacement_a"
        }
        client.post("/api/v1/models/train", json=train_request)
        
        # 予測を実行
        response = client.post(
            "/api/v1/models/predict",
            json=sample_model_predict_request
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "predictions" in data
        assert "model_name" in data
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) == len(sample_model_predict_request["data"])
        assert data["model_name"] == sample_model_predict_request["model_name"]
    
    def test_predict_with_untrained_model(self, client):
        """未訓練モデルでの予測（エラー）"""
        request_data = {
            "model_name": "MLP",  # 未訓練のモデル
            "data": [
                {"TD": 100, "Distance_from_face": 50}
            ]
        }
        
        response = client.post(
            "/api/v1/models/predict",
            json=request_data
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "not trained" in response.json()["detail"]
    
    def test_get_model_types(self, client):
        """モデルタイプ一覧の取得"""
        response = client.get("/api/v1/models/types")
        
        assert response.status_code == status.HTTP_200_OK
        types = response.json()
        
        assert isinstance(types, list)
        assert len(types) > 0
        assert "RandomForest" in types or "LinearRegression" in types