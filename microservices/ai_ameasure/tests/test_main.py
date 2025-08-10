import pytest
from fastapi import status


class TestMainApp:
    """メインアプリケーションのテスト"""
    
    def test_root_endpoint(self, client):
        """ルートエンドポイントのテスト"""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        
        assert data["message"] == "Welcome to AI A-Measure API"
        assert data["version"] in ["1.0.0", "1.0.0-test"]  # テスト環境と本番環境の両方を許可
        assert data["docs"] == "/api/v1/docs"
    
    def test_openapi_endpoint(self, client):
        """OpenAPIドキュメントエンドポイントのテスト"""
        response = client.get("/api/v1/openapi.json")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "info" in data
        assert "paths" in data
        assert "components" in data
        
        # API情報の確認
        assert data["info"]["title"] in ["AI A-Measure API", "AI A-Measure API Test"]  # テスト環境と本番環境の両方を許可
        assert data["info"]["version"] in ["1.0.0", "1.0.0-test"]  # テスト環境と本番環境の両方を許可
        
        # エンドポイントが含まれていることを確認
        assert "/api/v1/displacement/analyze" in data["paths"]
        assert "/api/v1/models/" in data["paths"]
        assert "/api/v1/analysis/displacement" in data["paths"]