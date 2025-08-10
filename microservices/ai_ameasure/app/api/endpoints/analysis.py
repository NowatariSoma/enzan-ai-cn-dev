from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime

from app import schemas
from app.core.config import settings

router = APIRouter()


def generate_correlation_matrix(features: List[str]) -> Dict[str, Any]:
    """相関行列とヒートマップデータを生成"""
    n = len(features)
    correlation_matrix = np.zeros((n, n))
    heatmap_data = []
    
    for i in range(n):
        for j in range(n):
            if i == j:
                correlation = 1.0
            else:
                # リアルな相関を生成
                base_correlation = (np.random.random() - 0.5) * 2
                
                # 隣接する特徴量は高い相関を持つ可能性
                if abs(i - j) == 1:
                    correlation = base_correlation * 0.7
                elif abs(i - j) <= 3:
                    correlation = base_correlation * 0.5
                else:
                    correlation = base_correlation * 0.3
                
                correlation = max(-0.95, min(0.95, correlation))
            
            correlation_matrix[i][j] = correlation
            heatmap_data.append({
                "x": features[j],
                "y": features[i],
                "value": correlation
            })
    
    return {
        "features": features,
        "correlation_matrix": correlation_matrix.tolist(),
        "heatmap_data": heatmap_data
    }


@router.post("/displacement", response_model=schemas.AnalysisResult)
async def analyze_displacement(
    request: schemas.AnalysisRequest
) -> schemas.AnalysisResult:
    """
    変位の時空間解析を実行
    """
    try:
        # モック解析結果を生成
        train_score = 0.85 + np.random.random() * 0.1
        validation_score = train_score - 0.05 - np.random.random() * 0.05
        
        # 特徴量重要度
        features = [
            'TD', 'Distance_from_face', 'Excavation_advance', 'Ground_condition',
            'Support_type', 'Overburden', 'Groundwater', 'Rock_strength'
        ]
        feature_importance = {}
        for feature in features:
            feature_importance[feature] = np.random.random() * 0.2
        
        # 正規化
        total = sum(feature_importance.values())
        if total > 0:
            feature_importance = {k: v/total for k, v in feature_importance.items()}
        
        # モック予測結果
        predictions = []
        for i in range(10):
            predictions.append({
                "td": 100 + i * 50,
                "displacement_a": np.random.random() * 2,
                "displacement_b": np.random.random() * 1.5,
                "displacement_c": np.random.random() * 1.8
            })
        
        return schemas.AnalysisResult(
            folder_name=request.csv_files[0] if request.csv_files else "default",
            model_type=request.model_type,
            train_score=train_score,
            validation_score=validation_score,
            feature_importance=feature_importance,
            predictions=predictions,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=schemas.FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...)
) -> schemas.FileUploadResponse:
    """
    CSVファイルをアップロード
    """
    # ファイルサイズチェック
    contents = await file.read()
    file_size = len(contents)
    
    if file_size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE} bytes"
        )
    
    # ファイル拡張子チェック
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are allowed"
        )
    
    # ファイル保存
    file_path = settings.UPLOAD_DIR / file.filename
    with open(file_path, 'wb') as f:
        f.write(contents)
    
    return schemas.FileUploadResponse(
        filename=file.filename,
        file_path=str(file_path),
        size=file_size
    )


@router.get("/correlation/{folder_name}", response_model=schemas.CorrelationData)
async def get_correlation_data(
    folder_name: str
) -> schemas.CorrelationData:
    """
    指定フォルダの相関データを取得
    """
    # モック特徴量リスト
    features = [
        'TD', 'Distance_from_face', 'Excavation_advance', 'Ground_condition',
        'Support_type', 'Overburden', 'Groundwater', 'Rock_strength',
        'Tunnel_diameter', 'Depth', 'Geological_formation', 'Weather_condition',
        'Equipment_type', 'Advance_rate', 'Face_stability', 'Convergence_rate'
    ]
    
    correlation_data = generate_correlation_matrix(features)
    
    return schemas.CorrelationData(**correlation_data)