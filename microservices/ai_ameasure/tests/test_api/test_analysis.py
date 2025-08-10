import pytest
from fastapi import status
import io


class TestAnalysisEndpoints:
    """解析エンドポイントのテスト"""
    
    def test_analyze_displacement_time_space(self, client, sample_analysis_request):
        """変位の時空間解析"""
        response = client.post(
            "/api/v1/analysis/displacement",
            json=sample_analysis_request
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "folder_name" in data
        assert "model_type" in data
        assert "train_score" in data
        assert "validation_score" in data
        assert "feature_importance" in data
        assert "predictions" in data
        assert "timestamp" in data
        
        assert data["model_type"] == sample_analysis_request["model_type"]
        assert 0 <= data["train_score"] <= 1
        assert 0 <= data["validation_score"] <= 1
        assert isinstance(data["feature_importance"], dict)
        assert isinstance(data["predictions"], list)
    
    def test_upload_csv_file(self, client):
        """CSVファイルのアップロード"""
        # テスト用のCSVデータを作成
        csv_content = """distance,displacement_a,displacement_b
0,0.1,0.2
10,0.3,0.4
20,0.5,0.6"""
        
        files = {
            "file": ("test_data.csv", io.BytesIO(csv_content.encode()), "text/csv")
        }
        
        response = client.post(
            "/api/v1/analysis/upload",
            files=files
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "filename" in data
        assert "file_path" in data
        assert "size" in data
        
        assert data["filename"] == "test_data.csv"
        assert data["size"] > 0
    
    def test_upload_non_csv_file(self, client):
        """CSV以外のファイルアップロード（エラー）"""
        files = {
            "file": ("test.txt", io.BytesIO(b"This is not a CSV file"), "text/plain")
        }
        
        response = client.post(
            "/api/v1/analysis/upload",
            files=files
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "CSV files" in response.json()["detail"]
    
    def test_upload_large_file(self, client):
        """大きすぎるファイルのアップロード（エラー）"""
        # 101MBのダミーデータ（制限は100MB）
        large_content = b"x" * (101 * 1024 * 1024)
        
        files = {
            "file": ("large_file.csv", io.BytesIO(large_content), "text/csv")
        }
        
        response = client.post(
            "/api/v1/analysis/upload",
            files=files
        )
        
        assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    
    def test_get_correlation_data(self, client):
        """相関データの取得"""
        folder_name = "01-hokkaido-akan"
        response = client.get(f"/api/v1/analysis/correlation/{folder_name}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "features" in data
        assert "correlation_matrix" in data
        assert "heatmap_data" in data
        
        assert isinstance(data["features"], list)
        assert isinstance(data["correlation_matrix"], list)
        assert isinstance(data["heatmap_data"], list)
        
        # 相関行列が正方行列であることを確認
        n_features = len(data["features"])
        assert len(data["correlation_matrix"]) == n_features
        assert all(len(row) == n_features for row in data["correlation_matrix"])
        
        # ヒートマップデータの構造を確認
        if data["heatmap_data"]:
            first_item = data["heatmap_data"][0]
            assert "x" in first_item
            assert "y" in first_item
            assert "value" in first_item
            assert -1 <= first_item["value"] <= 1  # 相関係数は-1から1の間