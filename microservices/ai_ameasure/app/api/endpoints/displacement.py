from fastapi import APIRouter, HTTPException, Depends
from typing import List
import numpy as np
import pandas as pd
from pathlib import Path

from app import schemas
from app.core.config import settings
from app.api import deps

router = APIRouter()


def generate_mock_displacement_data(max_distance: float, prediction_td: int) -> List[schemas.DisplacementData]:
    """モックの変位データを生成"""
    data = []
    num_points = 50
    
    for i in range(num_points + 1):
        distance_from_face = (i / num_points) * max_distance
        noise = (np.random.random() - 0.5) * 0.2
        
        # 実測値
        displacement_a = np.sin(distance_from_face * 0.1) * 0.5 + noise
        displacement_b = np.cos(distance_from_face * 0.08) * 0.3 + noise * 0.5
        displacement_c = np.sin(distance_from_face * 0.12) * 0.4 + noise * 0.3
        
        # 予測値（TDの影響を含める）
        td_factor = prediction_td / 500.0
        displacement_a_pred = np.cos(distance_from_face * 0.1) * 0.6 * td_factor + noise * 0.4
        displacement_b_pred = np.sin(distance_from_face * 0.09) * 0.35 * td_factor + noise * 0.6
        displacement_c_pred = np.cos(distance_from_face * 0.11) * 0.45 * td_factor + noise * 0.5
        
        data.append(schemas.DisplacementData(
            distance_from_face=distance_from_face,
            displacement_a=displacement_a,
            displacement_b=displacement_b,
            displacement_c=displacement_c,
            displacement_a_prediction=displacement_a_pred,
            displacement_b_prediction=displacement_b_pred,
            displacement_c_prediction=displacement_c_pred
        ))
    
    return data


def generate_mock_scatter_data(num_points: int = 200, noise_level: float = 2.0) -> List[schemas.ScatterData]:
    """モックの散布図データを生成"""
    data = []
    
    for _ in range(num_points):
        actual = (np.random.random() - 0.5) * 10  # -5から5の間のランダム値
        noise = (np.random.random() - 0.5) * noise_level
        predicted = actual + noise
        
        data.append(schemas.ScatterData(actual=actual, predicted=predicted))
    
    return data


def generate_mock_feature_importance() -> List[schemas.FeatureImportance]:
    """モックの特徴量重要度データを生成"""
    features = [
        'TD', 'Distance_from_face', 'Excavation_advance', 'Ground_condition',
        'Support_type', 'Overburden', 'Groundwater', 'Rock_strength',
        'Tunnel_diameter', 'Depth', 'Geological_formation', 'Weather_condition',
        'Equipment_type', 'Advance_rate', 'Face_stability', 'Convergence_rate',
        'Stress_level', 'Deformation_history', 'Support_pressure', 'Time_factor'
    ]
    
    data = []
    for feature in features:
        importance = np.random.random() * 0.15
        data.append(schemas.FeatureImportance(feature=feature, importance=importance))
    
    # 重要度でソートして上位の値を調整
    data.sort(key=lambda x: x.importance, reverse=True)
    if len(data) > 0:
        data[0].importance = 0.12 + np.random.random() * 0.03
    if len(data) > 1:
        data[1].importance = 0.08 + np.random.random() * 0.03
    if len(data) > 2:
        data[2].importance = 0.05 + np.random.random() * 0.03
    
    return data


def calculate_r_squared(scatter_data: List[schemas.ScatterData]) -> float:
    """R²値を計算"""
    if not scatter_data:
        return 0.0
    
    actuals = [d.actual for d in scatter_data]
    predicteds = [d.predicted for d in scatter_data]
    
    actual_mean = np.mean(actuals)
    total_sum_squares = sum((a - actual_mean) ** 2 for a in actuals)
    residual_sum_squares = sum((a - p) ** 2 for a, p in zip(actuals, predicteds))
    
    if total_sum_squares == 0:
        return 0.0
    
    return max(0, 1 - (residual_sum_squares / total_sum_squares))


@router.post("/analyze", response_model=schemas.DisplacementAnalysisResponse)
async def analyze_displacement(
    request: schemas.DisplacementAnalysisRequest
) -> schemas.DisplacementAnalysisResponse:
    """
    変位解析を実行し、結果を返す
    """
    try:
        # モックデータの生成
        chart_data = generate_mock_displacement_data(
            request.max_distance,
            request.prediction_td
        )
        
        # 訓練データと検証データの散布図データ生成
        train_scatter_a = generate_mock_scatter_data(200, 2.0)
        train_scatter_b = generate_mock_scatter_data(200, 2.0)
        validation_scatter_a = generate_mock_scatter_data(150, 2.5)
        validation_scatter_b = generate_mock_scatter_data(150, 2.5)
        
        # R²値の計算
        train_r2_a = calculate_r_squared(train_scatter_a)
        train_r2_b = calculate_r_squared(train_scatter_b)
        val_r2_a = calculate_r_squared(validation_scatter_a)
        val_r2_b = calculate_r_squared(validation_scatter_b)
        
        # 特徴量重要度の生成
        feature_importance_a = generate_mock_feature_importance()
        feature_importance_b = generate_mock_feature_importance()
        
        # レスポンスの作成
        response = schemas.DisplacementAnalysisResponse(
            chart_data=chart_data,
            train_r_squared_a=train_r2_a,
            train_r_squared_b=train_r2_b,
            validation_r_squared_a=val_r2_a,
            validation_r_squared_b=val_r2_b,
            feature_importance_a=[
                {"feature": fi.feature, "importance": fi.importance}
                for fi in feature_importance_a
            ],
            feature_importance_b=[
                {"feature": fi.feature, "importance": fi.importance}
                for fi in feature_importance_b
            ]
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/folders", response_model=List[str])
async def get_available_folders() -> List[str]:
    """
    利用可能なフォルダ一覧を取得
    """
    # 実際のデータフォルダから取得
    data_folder = settings.DATA_FOLDER
    
    if not data_folder.exists():
        # フォルダが存在しない場合はモックデータを返す
        return [
            "01-hokkaido-akan",
            "02-tohoku-sendai",
            "03-kanto-tokyo",
            "04-chubu-nagoya",
            "05-kinki-osaka",
            "06-chugoku-hiroshima",
            "07-shikoku-takamatsu",
            "08-kyushu-fukuoka"
        ]
    
    # 実際のフォルダ一覧を取得
    folders = []
    for item in data_folder.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            folders.append(item.name)
    
    # フォルダがない場合はモックデータを返す
    if not folders:
        folders = [
            "01-hokkaido-akan",
            "02-tohoku-sendai",
            "03-kanto-tokyo"
        ]
    
    return sorted(folders)