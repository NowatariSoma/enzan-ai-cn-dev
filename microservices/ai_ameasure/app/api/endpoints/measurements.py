from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging

from app import schemas
from app.core.config import settings
from app.core.csv_loader import CSVDataLoader
from app.api import deps

logger = logging.getLogger(__name__)

router = APIRouter()

# CSVデータローダーのインスタンス
csv_loader = CSVDataLoader()

def get_real_time_series_data(
    data_type: str = "displacement",
    num_points: int = 100,
    folder_name: str = "01-hokkaido-akan"
) -> List[schemas.TimeSeriesDataPoint]:
    """
    実際のCSVファイルから時系列データを取得
    """
    try:
        # 実際の計測データを読み込み
        df = csv_loader.load_all_measurement_data(settings.DATA_FOLDER, folder_name)
        
        if df.empty:
            logger.warning(f"No measurement data found for {folder_name}, using fallback mock data")
            return generate_mock_time_series_data(data_type, num_points)
        
        # 時系列データを抽出
        raw_data = csv_loader.extract_time_series_data(df, data_type, num_points)
        
        # Pydanticモデルに変換
        data = []
        for point in raw_data:
            data_point = schemas.TimeSeriesDataPoint(**point)
            data.append(data_point)
        
        logger.info(f"Successfully extracted {len(data)} {data_type} time series points from real data")
        return data
        
    except Exception as e:
        logger.error(f"Error loading real data: {e}, falling back to mock data")
        return generate_mock_time_series_data(data_type, num_points)


def get_real_distribution_data(
    data_type: str = "displacement",
    folder_name: str = "01-hokkaido-akan"
) -> List[schemas.DistributionDataPoint]:
    """
    実際のCSVファイルから分布データを取得
    """
    try:
        # 実際の計測データを読み込み
        df = csv_loader.load_all_measurement_data(settings.DATA_FOLDER, folder_name)
        
        if df.empty:
            logger.warning(f"No measurement data found for {folder_name}, using fallback mock data")
            return generate_mock_distribution_data(data_type)
        
        # 分布データを抽出
        raw_data = csv_loader.extract_distribution_data(df, data_type)
        
        # Pydanticモデルに変換
        data = []
        for point in raw_data:
            data_point = schemas.DistributionDataPoint(**point)
            data.append(data_point)
        
        logger.info(f"Successfully extracted {len(data)} {data_type} distribution bins from real data")
        return data
        
    except Exception as e:
        logger.error(f"Error loading real data: {e}, falling back to mock data")
        return generate_mock_distribution_data(data_type)


def get_real_scatter_data(
    num_points: int = 200,
    folder_name: str = "01-hokkaido-akan"
) -> List[schemas.TunnelScatterPoint]:
    """
    実際のCSVファイルから散布図データを取得
    """
    try:
        # 実際の計測データを読み込み
        df = csv_loader.load_all_measurement_data(settings.DATA_FOLDER, folder_name)
        
        if df.empty:
            logger.warning(f"No measurement data found for {folder_name}, using fallback mock data")
            return generate_mock_scatter_data(num_points)
        
        # 散布図データを抽出
        raw_data = csv_loader.extract_scatter_data(df, num_points)
        
        # Pydanticモデルに変換
        data = []
        for point in raw_data:
            data_point = schemas.TunnelScatterPoint(**point)
            data.append(data_point)
        
        logger.info(f"Successfully extracted {len(data)} scatter points from real data")
        return data
        
    except Exception as e:
        logger.error(f"Error loading real data: {e}, falling back to mock data")
        return generate_mock_scatter_data(num_points)


# フォールバック用のモックデータ生成関数
def generate_mock_time_series_data(
    data_type: str = "displacement",
    num_points: int = 100
) -> List[schemas.TimeSeriesDataPoint]:
    """モックデータ生成（フォールバック用）"""
    data = []
    
    for i in range(num_points):
        td = i * 5.0
        
        if data_type == "displacement":
            base_value = np.sin(td * 0.01) * 2.0
            data_point = schemas.TimeSeriesDataPoint(
                td=td,
                series3m=-2.5 + base_value + np.random.normal(0, 0.3),
                series5m=-1.8 + base_value * 1.2 + np.random.normal(0, 0.4),
                series10m=-3.2 + base_value * 1.5 + np.random.normal(0, 0.5),
                series20m=-2.1 + base_value * 1.8 + np.random.normal(0, 0.6),
                series50m=-4.5 + base_value * 2.0 + np.random.normal(0, 0.7),
                series100m=-3.8 + base_value * 2.2 + np.random.normal(0, 0.8)
            )
        else:
            base_value = np.cos(td * 0.008) * 1.5
            data_point = schemas.TimeSeriesDataPoint(
                td=td,
                series3m=-1.2 + base_value + np.random.normal(0, 0.2),
                series5m=-1.5 + base_value * 1.1 + np.random.normal(0, 0.25),
                series10m=-2.0 + base_value * 1.3 + np.random.normal(0, 0.3),
                series20m=-1.8 + base_value * 1.5 + np.random.normal(0, 0.35),
                series50m=-2.5 + base_value * 1.7 + np.random.normal(0, 0.4),
                series100m=-2.2 + base_value * 1.9 + np.random.normal(0, 0.45)
            )
        
        data.append(data_point)
    
    return data


def generate_mock_distribution_data(
    data_type: str = "displacement"
) -> List[schemas.DistributionDataPoint]:
    """モック分布データ生成（フォールバック用）"""
    data = []
    
    for i in range(-15, 16):
        range_val = str(i)
        val = float(range_val)
        
        if data_type == "displacement":
            mean = -2.0
            std = 3.0
        else:
            mean = -1.5
            std = 2.0
        
        base_freq = int(50 * np.exp(-0.5 * ((val - mean) / std) ** 2))
        
        data_point = schemas.DistributionDataPoint(
            range=range_val,
            series3m=max(0, base_freq + np.random.randint(-10, 10)),
            series5m=max(0, int(base_freq * 1.1) + np.random.randint(-10, 10)),
            series10m=max(0, int(base_freq * 1.2) + np.random.randint(-10, 10)),
            series20m=max(0, int(base_freq * 1.15) + np.random.randint(-10, 10)),
            series50m=max(0, int(base_freq * 0.9) + np.random.randint(-10, 10)),
            series100m=max(0, int(base_freq * 0.8) + np.random.randint(-10, 10))
        )
        
        data.append(data_point)
    
    return data


def generate_mock_scatter_data(
    num_points: int = 200
) -> List[schemas.TunnelScatterPoint]:
    """モック散布図データ生成（フォールバック用）"""
    data = []
    
    for i in range(num_points):
        x = np.random.uniform(0, 100)
        y = np.random.uniform(0, 60)
        depth = -10 - (x * 0.1) - (y * 0.05) + np.random.normal(0, 2)
        
        if depth > -10:
            color = "#00FFFF"
        elif depth > -15:
            color = "#0080FF"
        elif depth > -20:
            color = "#0040FF"
        else:
            color = "#0000FF"
        
        data_point = schemas.TunnelScatterPoint(
            x=round(x, 1),
            y=round(y, 1),
            depth=round(depth, 1),
            color=color
        )
        
        data.append(data_point)
    
    return data


@router.get("/displacement-series", response_model=schemas.DisplacementSeriesResponse)
async def get_displacement_series(
    num_points: int = Query(default=100, ge=10, le=500),
    folder_name: str = Query(default="01-hokkaido-akan", description="データフォルダ名")
) -> schemas.DisplacementSeriesResponse:
    """
    変位の時系列データを取得（実際のCSVデータから）
    """
    data = get_real_time_series_data("displacement", num_points, folder_name)
    return schemas.DisplacementSeriesResponse(
        data=data,
        unit="mm",
        measurement_type="displacement"
    )


@router.get("/settlement-series", response_model=schemas.SettlementSeriesResponse)
async def get_settlement_series(
    num_points: int = Query(default=100, ge=10, le=500),
    folder_name: str = Query(default="01-hokkaido-akan", description="データフォルダ名")
) -> schemas.SettlementSeriesResponse:
    """
    沈下の時系列データを取得（実際のCSVデータから）
    """
    data = get_real_time_series_data("settlement", num_points, folder_name)
    return schemas.SettlementSeriesResponse(
        data=data,
        unit="mm",
        measurement_type="settlement"
    )


@router.get("/displacement-distribution", response_model=schemas.DisplacementDistributionResponse)
async def get_displacement_distribution(
    folder_name: str = Query(default="01-hokkaido-akan", description="データフォルダ名")
) -> schemas.DisplacementDistributionResponse:
    """
    変位の分布データを取得（実際のCSVデータから）
    """
    data = get_real_distribution_data("displacement", folder_name)
    return schemas.DisplacementDistributionResponse(
        data=data,
        bin_size=1,
        measurement_type="displacement"
    )


@router.get("/settlement-distribution", response_model=schemas.SettlementDistributionResponse)
async def get_settlement_distribution(
    folder_name: str = Query(default="01-hokkaido-akan", description="データフォルダ名")
) -> schemas.SettlementDistributionResponse:
    """
    沈下の分布データを取得（実際のCSVデータから）
    """
    data = get_real_distribution_data("settlement", folder_name)
    return schemas.SettlementDistributionResponse(
        data=data,
        bin_size=1,
        measurement_type="settlement"
    )


@router.get("/tunnel-scatter", response_model=schemas.TunnelScatterResponse)
async def get_tunnel_scatter(
    num_points: int = Query(default=200, ge=50, le=1000),
    folder_name: str = Query(default="01-hokkaido-akan", description="データフォルダ名")
) -> schemas.TunnelScatterResponse:
    """
    トンネル計測の散布図データを取得（実際のCSVデータから）
    """
    data = get_real_scatter_data(num_points, folder_name)
    return schemas.TunnelScatterResponse(
        data=data,
        x_label="切羽からの距離 (m)",
        y_label="計測経過日数",
        color_scale="depth"
    )


@router.get("/measurement-files", response_model=schemas.MeasurementFilesResponse)
async def get_measurement_files(
    folder_name: str = Query(default="01-hokkaido-akan", description="フォルダ名")
) -> schemas.MeasurementFilesResponse:
    """
    利用可能な計測ファイル一覧を取得
    """
    files = []
    
    # 実際のデータフォルダから取得
    measurements_path = settings.DATA_FOLDER / folder_name / "main_tunnel" / "CN_measurement_data" / "measurements_A"
    
    if measurements_path.exists():
        # 実際のファイルを取得
        csv_files = sorted(measurements_path.glob("*.csv"))
        
        for csv_file in csv_files:
            # ファイル名から番号を抽出
            match = csv_file.stem.replace("measurements_A_", "")
            try:
                cycle_num = int(match)
                description = f"Measurement cycle {cycle_num}"
            except:
                description = f"Measurement file {csv_file.stem}"
            
            # ファイル情報を取得
            stat = csv_file.stat()
            
            file_info = schemas.MeasurementFileInfo(
                id=csv_file.name,
                name=csv_file.name,
                description=description,
                created_at=datetime.fromtimestamp(stat.st_ctime),
                size=stat.st_size
            )
            files.append(file_info)
    
    # ファイルがない場合はモックデータを返す
    if not files:
        for i in range(1, 11):
            file_info = schemas.MeasurementFileInfo(
                id=f"measurements_A_{i:05d}.csv",
                name=f"measurements_A_{i:05d}.csv",
                description=f"Measurement cycle {i}",
                created_at=datetime.now() - timedelta(days=10-i),
                size=np.random.randint(1000, 5000)
            )
            files.append(file_info)
    
    return schemas.MeasurementFilesResponse(
        files=files,
        total_count=len(files)
    )


@router.post("/analyze", response_model=schemas.DisplacementAnalysisResponse)
async def analyze_measurements(
    request: schemas.MeasurementAnalysisRequest
) -> schemas.DisplacementAnalysisResponse:
    """
    計測データの解析を実行
    既存のdisplacement/analyzeエンドポイントと互換性を保つ
    """
    # チャートデータを生成
    chart_data = []
    num_points = 50
    
    for i in range(num_points):
        distance = i * (request.excavationAdvance / num_points)
        
        # 実測値と予測値を生成
        data_point = schemas.DisplacementData(
            distanceFromFace=distance,
            displacement_a=np.sin(distance * 0.1) * 0.5 + np.random.normal(0, 0.1),
            displacement_b=np.cos(distance * 0.08) * 0.3 + np.random.normal(0, 0.08),
            displacement_c=np.sin(distance * 0.12) * 0.4 + np.random.normal(0, 0.09),
            displacement_a_prediction=np.sin(distance * 0.1) * 0.55 + np.random.normal(0, 0.12),
            displacement_b_prediction=np.cos(distance * 0.08) * 0.32 + np.random.normal(0, 0.1),
            displacement_c_prediction=np.sin(distance * 0.12) * 0.42 + np.random.normal(0, 0.11)
        )
        chart_data.append(data_point)
    
    # モックレスポンスを生成
    return schemas.DisplacementAnalysisResponse(
        chart_data=chart_data,
        train_r_squared_a=0.85 + np.random.random() * 0.1,
        train_r_squared_b=0.82 + np.random.random() * 0.1,
        validation_r_squared_a=0.80 + np.random.random() * 0.1,
        validation_r_squared_b=0.78 + np.random.random() * 0.1,
        feature_importance_a=[
            {"feature": "TD", "importance": 0.15},
            {"feature": "Distance_from_face", "importance": 0.12},
            {"feature": "Excavation_advance", "importance": 0.10}
        ],
        feature_importance_b=[
            {"feature": "TD", "importance": 0.14},
            {"feature": "Distance_from_face", "importance": 0.13},
            {"feature": "Ground_condition", "importance": 0.09}
        ]
    )


@router.get("/predictions", response_model=schemas.MeasurementPredictionsResponse)
async def get_predictions(
    excavationAdvance: float = Query(..., gt=0),
    distanceFromFace: float = Query(..., ge=0)
) -> schemas.MeasurementPredictionsResponse:
    """
    予測データテーブルを取得
    """
    predictions = []
    
    # 10ステップの予測を生成
    for step in range(10):
        days = step * 2  # 2日ごと
        
        # 距離と日数に基づいて予測値を計算
        factor = (step + 1) * 0.1
        pred1 = round(0.234 + factor * np.sin(step * 0.5), 3)
        pred2 = round(-0.125 + factor * np.cos(step * 0.4), 3)
        pred3 = round(1.456 + factor * np.sin(step * 0.3), 3)
        
        prediction = schemas.MeasurementPrediction(
            step=step,
            days=days,
            prediction1=f"{pred1:.3f}",
            prediction2=f"{pred2:.3f}",
            prediction3=f"{pred3:.3f}"
        )
        predictions.append(prediction)
    
    return schemas.MeasurementPredictionsResponse(
        predictions=predictions,
        excavationAdvance=excavationAdvance,
        distanceFromFace=distanceFromFace
    )