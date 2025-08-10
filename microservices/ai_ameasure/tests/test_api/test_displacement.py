import pytest
from fastapi import status


class TestDisplacementEndpoints:
    """変位解析エンドポイントのテスト"""
    
    def test_analyze_displacement_success(self, client, sample_displacement_request):
        """変位解析が正常に実行されることを確認"""
        response = client.post(
            "/api/v1/displacement/analyze",
            json=sample_displacement_request
        )
        
        if response.status_code != status.HTTP_200_OK:
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # レスポンスの構造を確認
        assert "chart_data" in data
        assert "train_r_squared_a" in data
        assert "train_r_squared_b" in data
        assert "validation_r_squared_a" in data
        assert "validation_r_squared_b" in data
        assert "feature_importance_a" in data
        assert "feature_importance_b" in data
        
        # チャートデータの確認
        assert len(data["chart_data"]) > 0
        first_point = data["chart_data"][0]
        assert "distance_from_face" in first_point
        assert "displacement_a" in first_point
        assert "displacement_b" in first_point
        assert "displacement_c" in first_point
        
        # R²値の確認（0から1の間）
        assert 0 <= data["train_r_squared_a"] <= 1
        assert 0 <= data["train_r_squared_b"] <= 1
        assert 0 <= data["validation_r_squared_a"] <= 1
        assert 0 <= data["validation_r_squared_b"] <= 1
        
        # 特徴量重要度の確認
        assert len(data["feature_importance_a"]) > 0
        assert len(data["feature_importance_b"]) > 0
        
    def test_analyze_displacement_with_different_parameters(self, client):
        """異なるパラメータでの変位解析"""
        request_data = {
            "folder": "02-tohoku-sendai",
            "model": "Linear Regression",
            "prediction_td": 1000,
            "max_distance": 200.0
        }
        
        response = client.post(
            "/api/v1/displacement/analyze",
            json=request_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # max_distanceが反映されていることを確認
        chart_data = data["chart_data"]
        max_distance_in_data = max(point["distance_from_face"] for point in chart_data)
        assert max_distance_in_data <= request_data["max_distance"]
    
    def test_analyze_displacement_invalid_request(self, client):
        """無効なリクエストでのエラー確認"""
        invalid_request = {
            "folder": "01-hokkaido-akan",
            # modelフィールドが欠けている
            "prediction_td": "invalid",  # 数値でなければならない
            "max_distance": -100  # 負の値
        }
        
        response = client.post(
            "/api/v1/displacement/analyze",
            json=invalid_request
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_get_available_folders(self, client):
        """利用可能なフォルダ一覧の取得"""
        response = client.get("/api/v1/displacement/folders")
        
        assert response.status_code == status.HTTP_200_OK
        folders = response.json()
        
        assert isinstance(folders, list)
        assert len(folders) > 0
        assert "01-hokkaido-akan" in folders