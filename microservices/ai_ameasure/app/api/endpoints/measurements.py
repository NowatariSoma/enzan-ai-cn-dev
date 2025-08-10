from fastapi import APIRouter, HTTPException, Query, File, UploadFile
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from fastapi.responses import FileResponse, StreamingResponse
import tempfile
import json

from app import schemas
from app.schemas.measurements import TDDataPoint, DistanceDataResponse
from app.core.config import settings
from app.core.csv_loader import CSVDataLoader

logger = logging.getLogger(__name__)

router = APIRouter()

# CSVデータローダーのインスタンス
csv_loader = CSVDataLoader()

# 定数定義（analyze_displacement関数から）
DATE = '計測日時'
CYCLE_NO = 'サイクル番号'
SECTION_TD = '切羽TD'
FACE_TD = '切羽位置'
TD_NO = 'TD'
CONVERGENCES = ['変位量A', '変位量B', '変位量C']
SETTLEMENTS = ['沈下量1', '沈下量2', '沈下量3']
STA = 'STA'
DISTANCE_FROM_FACE = '切羽からの距離'
DAYS_FROM_START = '計測経過日数'
DIFFERENCE_FROM_FINAL_CONVERGENCES = ['変位量A差分', '変位量B差分', '変位量C差分']
DIFFERENCE_FROM_FINAL_SETTLEMENTS = ['沈下量1差分', '沈下量2差分', '沈下量3差分']
DISTANCES_FROM_FACE = [3, 5, 10, 20, 50, 100]

# ヘルパー関数: NaN/Inf対策
def safe_float(value):
    """NaNやInfinityをNoneに変換"""
    if pd.isna(value) or np.isinf(value):
        return None
    return float(value)

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
            logger.warning(f"No measurement data found for {folder_name}")
            return []
        
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
        logger.error(f"Error loading real data: {e}")
        return []


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
            logger.warning(f"No measurement data found for {folder_name}")
            return []
        
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
        logger.error(f"Error loading real data: {e}")
        return []


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
            logger.warning(f"No measurement data found for {folder_name}")
            return []
        
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
        logger.error(f"Error loading real data: {e}")
        return []

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
    
    # ファイルがない場合は空のリストを返す
    if not files:
        logger.warning(f"No measurement files found in {measurements_path}")
    
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
    try:
        # 実際のCSVファイルから計測データを読み込み
        file_path = settings.DATA_FOLDER / "01-hokkaido-akan" / "main_tunnel" / "CN_measurement_data" / "measurements_A" / request.cycleNumber
        
        if not file_path.exists():
            # ファイル名として渡されなかった場合は、番号として処理
            file_path = settings.DATA_FOLDER / "01-hokkaido-akan" / "main_tunnel" / "CN_measurement_data" / "measurements_A" / f"measurements_A_{request.cycleNumber:05d}.csv"
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Measurement file not found: {request.cycleNumber}")
        
        # CSVファイルを処理
        df = csv_loader.process_measurement_file(
            file_path=file_path,
            max_distance_from_face=request.distanceFromFace,
            duration_days=90
        )
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found in the measurement file")
        
        # チャートデータを実データから生成
        chart_data = []
        
        # 切羽からの距離でデータを抽出
        if '切羽からの距離' in df.columns:
            df_filtered = df[df['切羽からの距離'] <= request.distanceFromFace]
        else:
            df_filtered = df
        
        # 変位量カラムを特定
        displacement_cols = [col for col in df.columns if '変位量' in col and '差分' not in col and 'ｵﾌｾｯﾄ' not in col]
        
        for _, row in df_filtered.iterrows():
            distance = row.get('切羽からの距離', 0.0) if '切羽からの距離' in row else 0.0
            
            # 実際の変位量データを取得（最大3つ）
            displacement_a = float(row[displacement_cols[0]]) if len(displacement_cols) > 0 and pd.notna(row[displacement_cols[0]]) else 0.0
            displacement_b = float(row[displacement_cols[1]]) if len(displacement_cols) > 1 and pd.notna(row[displacement_cols[1]]) else 0.0
            displacement_c = float(row[displacement_cols[2]]) if len(displacement_cols) > 2 and pd.notna(row[displacement_cols[2]]) else 0.0
            
            # 予測値は実測値に基づいて生成（簡易的な予測）
            data_point = schemas.DisplacementData(
                distanceFromFace=float(distance),
                displacement_a=displacement_a,
                displacement_b=displacement_b,
                displacement_c=displacement_c,
                displacement_a_prediction=displacement_a * 1.05,  # 実測値の105%
                displacement_b_prediction=displacement_b * 1.03,  # 実測値の103%
                displacement_c_prediction=displacement_c * 1.04   # 実測値の104%
            )
            chart_data.append(data_point)
        
        # 実データに基づく統計値を計算
        return schemas.DisplacementAnalysisResponse(
            chart_data=chart_data,
            train_r_squared_a=0.0,  # 実際のモデル学習がないため0
            train_r_squared_b=0.0,
            validation_r_squared_a=0.0,
            validation_r_squared_b=0.0,
            feature_importance_a=[],
            feature_importance_b=[]
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing measurements: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions", response_model=schemas.MeasurementPredictionsResponse)
async def get_predictions(
    excavationAdvance: float = Query(..., gt=0),
    distanceFromFace: float = Query(..., ge=0)
) -> schemas.MeasurementPredictionsResponse:
    """
    予測データテーブルを取得
    注：実際の予測モデルが未実装のため、空のリストを返す
    """
    # 実際の予測モデルが実装されるまで空のリストを返す
    predictions = []
    
    logger.warning("Prediction model not implemented yet")
    
    return schemas.MeasurementPredictionsResponse(
        predictions=predictions,
        excavationAdvance=excavationAdvance,
        distanceFromFace=distanceFromFace
    )


def generate_additional_info_df(cycle_support_csv: Path, observation_of_face_csv: Path) -> pd.DataFrame:
    """
    cycle_supportとobservation_of_faceデータを統合
    """
    try:
        df_cycle_support = pd.read_csv(cycle_support_csv).iloc[1:]
    except:
        df_cycle_support = pd.read_csv(cycle_support_csv, encoding='cp932').iloc[1:]
    try:
        df_observation_of_face = pd.read_csv(observation_of_face_csv)
    except:
        df_observation_of_face = pd.read_csv(observation_of_face_csv, encoding='cp932')
    
    # 最初のカラムで結合
    df_additional_info = pd.merge(
        df_cycle_support, 
        df_observation_of_face, 
        left_on=df_cycle_support.columns[0], 
        right_on=df_observation_of_face.columns[0], 
        how='inner'
    )
    return df_additional_info


def process_measurement_file(file_path: Path, max_distance_from_face: float = 100) -> pd.DataFrame:
    """
    単一の計測ファイルを処理
    """
    df = csv_loader.process_measurement_file(
        file_path=file_path,
        max_distance_from_face=max_distance_from_face,
        duration_days=90
    )
    
    # 切羽からの距離でフィルタリング
    if DISTANCE_FROM_FACE in df.columns:
        df = df[df[DISTANCE_FROM_FACE] >= -1]
        df = df[df[DISTANCE_FROM_FACE] <= max_distance_from_face]
    
    return df


def generate_dataframes(measurement_a_csvs: List[Path], max_distance_from_face: float):
    """
    複数の計測CSVファイルを統合処理
    """
    df_all = []
    for csv_file in sorted(measurement_a_csvs):
        try:
            df = process_measurement_file(csv_file, max_distance_from_face)
            df_all.append(df)
        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")
            continue
    
    if not df_all:
        return pd.DataFrame(), {}, {}, {}, [], []
    
    df_all = pd.concat(df_all)
    
    # 変位量と沈下量のカラムを特定
    settlements = [s for s in SETTLEMENTS if s in df_all.columns]
    convergences = [c for c in CONVERGENCES if c in df_all.columns]
    
    dct_df_settlement = {}
    dct_df_convergence = {}
    dct_df_td = {}
    
    for distance_from_face in DISTANCES_FROM_FACE:
        if max_distance_from_face < distance_from_face:
            continue
        
        dct_df_settlement[f"{distance_from_face}m"] = []
        dct_df_convergence[f"{distance_from_face}m"] = []
        
        # TD毎にグループ化して処理
        dfs = []
        if TD_NO in df_all.columns:
            for td, _df in df_all.groupby(TD_NO):
                rows = _df[_df[DISTANCE_FROM_FACE] <= distance_from_face]
                if rows.empty:
                    continue
                dfs.append(rows.iloc[-1][[TD_NO] + settlements + convergences])
                dct_df_settlement[f"{distance_from_face}m"] += rows.iloc[-1][settlements].values.tolist()
                dct_df_convergence[f"{distance_from_face}m"] += rows.iloc[-1][convergences].values.tolist()
            
            if dfs:
                dct_df_td[f"{distance_from_face}m"] = pd.DataFrame(dfs).reset_index()
    
    return df_all, dct_df_settlement, dct_df_convergence, dct_df_td, settlements, convergences


@router.post("/analyze-displacement")
async def analyze_displacement(
    folder_name: str = Query(default="01-hokkaido-akan", description="データフォルダ名"),
    max_distance_from_face: float = Query(default=100.0, gt=0, description="切羽からの最大距離"),
    generate_charts: bool = Query(default=False, description="チャート生成の有無")
) -> Dict[str, Any]:
    """
    analyze_displacement関数の主要機能をAPI化
    複数の計測ファイルを統合解析し、変位・沈下データを分析
    """
    try:
        # 入力フォルダのパス設定
        input_folder = settings.DATA_FOLDER / folder_name / "main_tunnel" / "CN_measurement_data"
        
        # measurements_Aフォルダ内のすべてのCSVファイルを取得
        measurements_path = input_folder / "measurements_A"
        if not measurements_path.exists():
            raise HTTPException(status_code=404, detail=f"Measurements folder not found: {measurements_path}")
        
        measurement_a_csvs = list(measurements_path.glob("*.csv"))
        if not measurement_a_csvs:
            raise HTTPException(status_code=404, detail="No measurement CSV files found")
        
        # 追加情報ファイルのパス
        cycle_support_csv = input_folder / "cycle_support" / "cycle_support.csv"
        observation_of_face_csv = input_folder / "observation_of_face" / "observation_of_face.csv"
        
        # データフレームの生成
        df_all, dct_df_settlement, dct_df_convergence, dct_df_td, settlements, convergences = generate_dataframes(
            measurement_a_csvs, max_distance_from_face
        )
        
        if df_all.empty:
            raise HTTPException(status_code=404, detail="No valid data found in measurement files")
        
        # 追加情報の統合（ファイルが存在する場合のみ）
        additional_info = None
        if cycle_support_csv.exists() and observation_of_face_csv.exists():
            try:
                df_additional_info = generate_additional_info_df(cycle_support_csv, observation_of_face_csv)
                if STA in df_additional_info.columns:
                    df_additional_info.drop(columns=[STA], inplace=True)
                additional_info = {
                    "columns": df_additional_info.columns.tolist(),
                    "shape": df_additional_info.shape,
                    "sample": df_additional_info.head(5).to_dict(orient='records')
                }
            except Exception as e:
                logger.warning(f"Could not load additional info: {e}")
        
        # 統計情報の計算
        stats = {
            "total_records": len(df_all),
            "unique_td_values": df_all[TD_NO].nunique() if TD_NO in df_all.columns else 0,
            "distance_range": {
                "min": safe_float(df_all[DISTANCE_FROM_FACE].min()) if DISTANCE_FROM_FACE in df_all.columns else None,
                "max": safe_float(df_all[DISTANCE_FROM_FACE].max()) if DISTANCE_FROM_FACE in df_all.columns else None
            },
            "days_range": {
                "min": safe_float(df_all[DAYS_FROM_START].min()) if DAYS_FROM_START in df_all.columns else None,
                "max": safe_float(df_all[DAYS_FROM_START].max()) if DAYS_FROM_START in df_all.columns else None
            },
            "settlements_columns": settlements,
            "convergences_columns": convergences
        }
        
        # 距離ごとの統計データ
        distance_stats = {}
        for distance_key in dct_df_settlement.keys():
            settlement_values = dct_df_settlement[distance_key]
            convergence_values = dct_df_convergence[distance_key]
            
            distance_stats[distance_key] = {
                "settlement": {
                    "count": len(settlement_values),
                    "mean": safe_float(np.mean(settlement_values)) if settlement_values else None,
                    "std": safe_float(np.std(settlement_values)) if settlement_values else None,
                    "min": safe_float(np.min(settlement_values)) if settlement_values else None,
                    "max": safe_float(np.max(settlement_values)) if settlement_values else None
                },
                "convergence": {
                    "count": len(convergence_values),
                    "mean": safe_float(np.mean(convergence_values)) if convergence_values else None,
                    "std": safe_float(np.std(convergence_values)) if convergence_values else None,
                    "min": safe_float(np.min(convergence_values)) if convergence_values else None,
                    "max": safe_float(np.max(convergence_values)) if convergence_values else None
                }
            }
        
        # チャート生成（オプション）
        charts = {}
        if generate_charts:
            charts = await generate_analysis_charts(
                df_all, dct_df_settlement, dct_df_convergence, 
                dct_df_td, settlements, convergences
            )
        
        return {
            "status": "success",
            "stats": stats,
            "distance_stats": distance_stats,
            "additional_info": additional_info,
            "charts": charts,
            "processed_files_count": len(measurement_a_csvs)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_displacement: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_analysis_charts(
    df_all: pd.DataFrame,
    dct_df_settlement: Dict,
    dct_df_convergence: Dict,
    dct_df_td: Dict,
    settlements: List[str],
    convergences: List[str]
) -> Dict[str, str]:
    """
    解析用のチャートを生成し、Base64エンコードして返す
    """
    charts = {}
    
    try:
        # 散布図1: 距離 vs 日数 (変位量)
        plt.figure(figsize=(10, 6))
        for conv in convergences:
            if conv in df_all.columns:
                scatter = plt.scatter(
                    df_all[DISTANCE_FROM_FACE], 
                    df_all[DAYS_FROM_START], 
                    c=df_all[conv], 
                    cmap='jet', 
                    alpha=0.5, 
                    s=5
                )
        plt.colorbar(scatter, label="変位量")
        plt.title(f"切羽からの距離 vs 計測経過日数 (変位量)")
        plt.xlabel(f"{DISTANCE_FROM_FACE} (m)")
        plt.ylabel(DAYS_FROM_START)
        plt.grid()
        
        # チャートをBase64エンコード
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        charts["scatter_distance_days_convergence"] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # 散布図2: 距離 vs 日数 (沈下量)
        plt.figure(figsize=(10, 6))
        for settle in settlements:
            if settle in df_all.columns:
                scatter = plt.scatter(
                    df_all[DISTANCE_FROM_FACE], 
                    df_all[DAYS_FROM_START], 
                    c=df_all[settle], 
                    cmap='jet', 
                    alpha=0.5, 
                    s=5
                )
        plt.colorbar(scatter, label="沈下量")
        plt.title(f"切羽からの距離 vs 計測経過日数 (沈下量)")
        plt.xlabel(f"{DISTANCE_FROM_FACE} (m)")
        plt.ylabel(DAYS_FROM_START)
        plt.grid()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        charts["scatter_distance_days_settlement"] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # ヒストグラム1: 変位量分布
        if dct_df_convergence:
            plt.figure(figsize=(10, 6))
            for name, values in dct_df_convergence.items():
                if values:
                    data = np.array(values)
                    sns.histplot(
                        data, 
                        bins=20, 
                        alpha=0.5, 
                        label=name, 
                        kde=True
                    )
            plt.title("変位量分布")
            plt.xlabel("変位量 (mm)")
            plt.ylabel("頻度")
            plt.legend()
            plt.grid()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            charts["histogram_convergence"] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
        
        # ヒストグラム2: 沈下量分布
        if dct_df_settlement:
            plt.figure(figsize=(10, 6))
            for name, values in dct_df_settlement.items():
                if values:
                    data = np.array(values)
                    sns.histplot(
                        data, 
                        bins=20, 
                        alpha=0.5, 
                        label=name, 
                        kde=True
                    )
            plt.title("沈下量分布")
            plt.xlabel("沈下量 (mm)")
            plt.ylabel("頻度")
            plt.legend()
            plt.grid()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            charts["histogram_settlement"] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
        
        # 距離vs変位量チャート（draw_charts_distance_displace相当）
        if dct_df_td and convergences:
            plt.figure(figsize=(10, 6))
            for name, df in dct_df_td.items():
                if not df.empty and TD_NO in df.columns:
                    # 各収束値の平均を計算
                    conv_mean = df[convergences].mean(axis=1) if all(c in df.columns for c in convergences) else pd.Series()
                    if not conv_mean.empty:
                        plt.plot(df[TD_NO], conv_mean, label=name, marker='o', linestyle='-', markersize=4)
            
            plt.title("TD vs 変位量")
            plt.xlabel('TD (m)')
            plt.ylabel('変位量 (mm)')
            plt.legend()
            plt.grid()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            charts["distance_convergence"] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
        
        # 距離vs沈下量チャート
        if dct_df_td and settlements:
            plt.figure(figsize=(10, 6))
            for name, df in dct_df_td.items():
                if not df.empty and TD_NO in df.columns:
                    # 各沈下値の平均を計算
                    settle_mean = df[settlements].mean(axis=1) if all(s in df.columns for s in settlements) else pd.Series()
                    if not settle_mean.empty:
                        plt.plot(df[TD_NO], settle_mean, label=name, marker='o', linestyle='-', markersize=4)
            
            plt.title("TD vs 沈下量")
            plt.xlabel('TD (m)')
            plt.ylabel('沈下量 (mm)')
            plt.legend()
            plt.grid()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            charts["distance_settlement"] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
        
    except Exception as e:
        logger.error(f"Error generating charts: {e}")
    
    return charts


@router.get("/distance-data", response_model=DistanceDataResponse)
async def get_distance_data(
    folder_name: str = Query(default="01-hokkaido-akan", description="データフォルダ名"),
    max_distance_from_face: float = Query(default=100.0, gt=0, description="切羽からの最大距離")
) -> DistanceDataResponse:
    """
    dct_df_tdとsettlementsデータを取得してフロントエンドに表示
    キャッシュされたデータフレームを利用して高速レスポンス
    """
    try:
        from app.core.dataframe_cache import get_dataframe_cache
        
        # キャッシュからデータを取得
        cache = get_dataframe_cache()
        cached_data = cache.get_cached_data(folder_name, max_distance_from_face)
        
        if not cached_data:
            raise HTTPException(status_code=404, detail=f"Failed to load data for folder: {folder_name}")
        
        # キャッシュからデータを展開
        dct_df_td = cached_data['dct_df_td']
        dct_df_settlement = cached_data['dct_df_settlement']
        dct_df_convergence = cached_data['dct_df_convergence']
        settlements = cached_data['settlements']
        convergences = cached_data['convergences']
        
        # dct_df_tdをTDDataPointのリストに変換（パフォーマンス改善）
        formatted_dct_df_td = {}
        for distance_key, df in dct_df_td.items():
            td_data_points = []
            if not df.empty and csv_loader.TD_NO in df.columns:
                # to_dict('records')でパフォーマンス向上
                for row in df.to_dict('records'):
                    # 各行から沈下量と変位量を抽出（リスト内包表記で高速化）
                    settlement_values = [
                        float(row[settle]) 
                        for settle in settlements 
                        if settle in row and pd.notna(row[settle])
                    ]
                    
                    convergence_values = [
                        float(row[conv])
                        for conv in convergences
                        if conv in row and pd.notna(row[conv])
                    ]
                    
                    td_point = TDDataPoint(
                        td=float(row[csv_loader.TD_NO]),
                        settlements=settlement_values,
                        convergences=convergence_values
                    )
                    td_data_points.append(td_point)
            
            formatted_dct_df_td[distance_key] = td_data_points
        
        # レスポンスを作成
        return DistanceDataResponse(
            dct_df_td=formatted_dct_df_td,
            settlements=dct_df_settlement,
            convergences=dct_df_convergence,
            settlements_columns=settlements,
            convergences_columns=convergences,
            distances=list(dct_df_settlement.keys())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_distance_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-dataset")
async def create_dataset(
    folder_name: str = Query(default="01-hokkaido-akan", description="データフォルダ名"),
    max_distance_from_face: float = Query(default=100.0, gt=0, description="切羽からの最大距離")
) -> Dict[str, Any]:
    """
    機械学習用のデータセットを作成
    settlement/convergenceデータと追加情報を統合
    """
    try:
        # 入力フォルダのパス設定
        input_folder = settings.DATA_FOLDER / folder_name / "main_tunnel" / "CN_measurement_data"
        
        # measurements_Aフォルダ内のすべてのCSVファイルを取得
        measurements_path = input_folder / "measurements_A"
        measurement_a_csvs = list(measurements_path.glob("*.csv"))
        
        # データフレームの生成
        df_all, _, _, _, settlements, convergences = generate_dataframes(
            measurement_a_csvs, max_distance_from_face
        )
        
        if df_all.empty:
            raise HTTPException(status_code=404, detail="No valid data found")
        
        # 追加情報の統合
        cycle_support_csv = input_folder / "cycle_support" / "cycle_support.csv"
        observation_of_face_csv = input_folder / "observation_of_face" / "observation_of_face.csv"
        
        dataset_info = {
            "main_data_shape": df_all.shape,
            "settlements_columns": settlements,
            "convergences_columns": convergences,
            "total_records": len(df_all)
        }
        
        # 追加情報が存在する場合は統合
        if cycle_support_csv.exists() and observation_of_face_csv.exists():
            df_additional_info = generate_additional_info_df(cycle_support_csv, observation_of_face_csv)
            if STA in df_additional_info.columns:
                df_additional_info.drop(columns=[STA], inplace=True)
            
            dataset_info["additional_info_shape"] = df_additional_info.shape
            dataset_info["additional_info_columns"] = df_additional_info.columns.tolist()
            
            # サンプルデータを含める
            dataset_info["sample_data"] = {
                "main": df_all.head(10).to_dict(orient='records'),
                "additional": df_additional_info.head(10).to_dict(orient='records')
            }
        
        return {
            "status": "success",
            "dataset_info": dataset_info
        }
        
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


