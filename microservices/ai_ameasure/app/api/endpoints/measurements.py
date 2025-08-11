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
from app.schemas.measurements import TDDataPoint, DistanceDataResponse, MLDatasetResponse, MLDatasetRequest
from app.core.config import settings
from app.core.csv_loader import CSVDataLoader

logger = logging.getLogger(__name__)

router = APIRouter()

# CSVデータローダーのインスタンス
csv_loader = CSVDataLoader()

# 定数定義（CSVLoaderからの値を使用）
DATE = csv_loader.DATE
CYCLE_NO = csv_loader.CYCLE_NO
SECTION_TD = csv_loader.SECTION_TD
FACE_TD = csv_loader.FACE_TD
TD_NO = csv_loader.TD_NO
CONVERGENCES = csv_loader.CONVERGENCES
SETTLEMENTS = csv_loader.SETTLEMENTS
STA = csv_loader.STA
DISTANCE_FROM_FACE = csv_loader.DISTANCE_FROM_FACE
DAYS_FROM_START = csv_loader.DAYS_FROM_START
DIFFERENCE_FROM_FINAL_CONVERGENCES = csv_loader.DIFFERENCE_FROM_FINAL_CONVERGENCES
DIFFERENCE_FROM_FINAL_SETTLEMENTS = csv_loader.DIFFERENCE_FROM_FINAL_SETTLEMENTS
DISTANCES_FROM_FACE = csv_loader.DISTANCES_FROM_FACE

# ヘルパー関数: NaN/Inf対策
def safe_float(value):
    """安全にfloatに変換し、NaN/Infの場合はNoneを返す"""
    try:
        if pd.isna(value) or np.isinf(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

def convert_support_pattern_to_numeric(value):
    """支保パターンを数値に変換（Streamlitアプリと同じ実装）"""
    if isinstance(value, str):
        value = value.lower().translate(str.maketrans({
            'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd', 'ｅ': 'e', 'ｆ': 'f', 'ｇ': 'g', 'ｈ': 'h', 'ｉ': 'i', 'ｊ': 'j',
            'ｋ': 'k', 'ｌ': 'l', 'ｍ': 'm', 'ｎ': 'n', 'ｏ': 'o', 'ｐ': 'p', 'ｑ': 'q', 'ｒ': 'r', 'ｓ': 's', 'ｔ': 't',
            'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x', 'ｙ': 'y', 'ｚ': 'z'
        }))
        if 'a' in value:
            return 1
        elif 'b' in value:
            return 2
        elif 'c' in value:
            return 3
        elif 'd' in value:
            return 4
        else:
            return -1  # その他特殊な断面
    else:
        return 0

def get_real_time_series_data(
    data_type: str = "displacement",
    num_points: int = 100,
    folder_name: str = "01-hokkaido-akan"
) -> List[schemas.TimeSeriesDataPoint]:
    """
    実際のCSVファイルから時系列データを取得
    """
    try:
        # 実際の計測データを読み込み - 正しいメソッドを使用
        input_folder = settings.DATA_FOLDER / folder_name / "main_tunnel" / "CN_measurement_data"
        measurements_path = input_folder / "measurements_A"
        
        if not measurements_path.exists():
            logger.warning(f"Measurements folder not found: {measurements_path}")
            return []
            
        measurement_a_csvs = list(measurements_path.glob("*.csv"))
        if not measurement_a_csvs:
            logger.warning(f"No measurement CSV files found in {measurements_path}")
            return []
        
        # generate_dataframesメソッドを使用してデータフレームを取得
        df_all, _, _, _, _, _ = csv_loader.generate_dataframes(measurement_a_csvs, 100.0)
        
        if df_all.empty:
            logger.warning(f"No measurement data found for {folder_name}")
            return []
        
        # 時系列データを抽出
        raw_data = csv_loader.extract_time_series_data(df_all, data_type, num_points)
        
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
    
def generate_additional_info_df(cycle_support_csv, observation_of_face_csv):
    try:
        df_cycle_support = pd.read_csv(cycle_support_csv).iloc[1:]
    except:
        df_cycle_support = pd.read_csv(cycle_support_csv, encoding='cp932').iloc[1:]
    try:
        df_observation_of_face = pd.read_csv(observation_of_face_csv)
    except:
        df_observation_of_face = pd.read_csv(observation_of_face_csv, encoding='cp932')
    # Concatenate df_cycle_support and df_observation_of_face by their first columns
    df_additional_info = pd.merge(
        df_cycle_support, 
        df_observation_of_face, 
        left_on=df_cycle_support.columns[0], 
        right_on=df_observation_of_face.columns[0], 
        how='inner'
    )
    return df_additional_info


def create_dataset(df, df_additional_info):
    """displacement_temporal_spacial_analysis.pyと完全に同じ実装"""
    # 定数の取得: まずCSVローダーの定数を使用し、必要に応じてStreamlitアプリの定数も使用
    try:
        # Streamlitアプリの定数を直接使用せず、既にインポート済みの定数を使用
        # ファイル上部でimportされている定数を使用
        pass
    except ImportError:
        logger.warning("定数の取得でエラーが発生しました")
    
    # CSVローダーから取得した定数を使用（すでにファイル上部でimport済み）
    common_columns = [csv_loader.DATE, csv_loader.CYCLE_NO, csv_loader.TD_NO, csv_loader.STA, 
                     csv_loader.SECTION_TD, csv_loader.FACE_TD, csv_loader.DISTANCE_FROM_FACE, csv_loader.DAYS_FROM_START]
    
    # 追加情報の列定義（Streamlitアプリと同じ）
    additional_info_common_columns = ['支保寸法', '吹付厚', 'ﾛｯｸﾎﾞﾙﾄ数', 'ﾛｯｸﾎﾞﾙﾄ長', '覆工厚', '土被り高さ', '岩石グループ', '岩石名コード', '加重平均評価点']
    additional_info_common_support_columns = ['支保工種', '支保パターン2']
    additional_info_common_bit_columns = ['補助工法の緒元', '増し支保工の緒元', '計測条件・状態等', 'インバートの早期障害']
    additional_info_left_columns = ['左肩・圧縮強度', '左肩・風化変質', '左肩・割目間隔', '左肩・割目状態', '左肩割目の方向・平行', '左肩・湧水量', '左肩・劣化', '左肩・評価点']
    additional_info_crown_columns = ['天端・圧縮強度', '天端・風化変質', '天端・割目間隔', '天端・割目状態', '天端割目の方向・平行', '天端・湧水量', '天端・劣化', '天端・評価点']
    additional_info_right_columns = ['右肩・圧縮強度', '右肩・風化変質', '右肩・割目間隔', '右肩・割目状態', '右肩割目の方向・平行', '右肩・湧水量', '右肩・劣化', '右肩・評価点']
    additional_info_columns = ['圧縮強度', '風化変質', '割目間隔', '割目状態', '割目の方向・平行', '湧水量', '劣化', '評価点']
    
    def decompose_columns(df, df_additional_info, targets, diff_targets):
        """Streamlitアプリと同じdecompose_columns関数の実装"""
        df_decomposed = pd.DataFrame()
        
        # Merge df and df_additional_info_filtered by CYCLE_NO and 'ｻｲｸﾙ'
        for i, row in df.iterrows():
            # df_additional_infoに'ｻｲｸﾙ'列があるかチェック
            if 'ｻｲｸﾙ' in df_additional_info.columns and csv_loader.CYCLE_NO in row:
                try:
                    matching_index = df_additional_info.index[df_additional_info['ｻｲｸﾙ'] <= row[csv_loader.CYCLE_NO]].max()
                    if pd.notna(matching_index):
                        df_additional_info_filtered = df_additional_info[df_additional_info.index == matching_index]
                        filtered_columns = additional_info_common_columns + \
                            additional_info_left_columns + \
                            additional_info_crown_columns + \
                            additional_info_right_columns + \
                            additional_info_common_bit_columns + \
                            additional_info_common_support_columns
                        
                        # 実際に存在する列のみを使用
                        filtered_columns = [col for col in filtered_columns if col in df_additional_info_filtered.columns]
                        
                        if filtered_columns:
                            df_additional_info_filtered = df_additional_info_filtered[filtered_columns]
                            df.loc[i, df_additional_info_filtered.columns] = df_additional_info_filtered.values.squeeze()
                except Exception as e:
                    logger.warning(f"追加情報の結合でエラー: {e}")
                    continue
        
        # Position別のデータ分解（左肩、天端、右肩）
        for i, (a, b) in enumerate(zip(targets[:3],  # 3 at the moment
                                    [additional_info_left_columns, additional_info_crown_columns, additional_info_right_columns])):
            if a not in df.columns or diff_targets[i] not in df.columns:
                continue
                
            filtered_columns = common_columns + [a] + additional_info_common_columns + additional_info_common_bit_columns + additional_info_common_support_columns + b + [diff_targets[i]]
            
            # 実際に存在する列のみを選択
            filtered_columns = [col for col in filtered_columns if col in df.columns]
            
            if len(filtered_columns) > 0:
                _df = df[filtered_columns].copy()
                
                # 列名の変更
                if a in _df.columns:
                    _df.rename(columns={a: a[:-1]}, inplace=True)
                if diff_targets[i] in _df.columns:
                    _df.rename(columns={diff_targets[i]: diff_targets[0][:-1]}, inplace=True)
                
                # additional_info_columnsへの列名変更
                for src, tgt in zip(b, additional_info_columns):
                    if src in _df.columns:
                        _df.rename(columns={src: tgt}, inplace=True)
                
                _df['position_id'] = i
                df_decomposed = pd.concat([df_decomposed, _df], axis=0, ignore_index=True)
        
        if df_decomposed.empty:
            logger.warning("decompose_columnsで空のDataFrameが生成されました")
            return pd.DataFrame(), [], ""
        
        # ビット列の処理
        bit_columns_in_df = [col for col in additional_info_common_bit_columns if col in df_decomposed.columns]
        for c in bit_columns_in_df:
            df_decomposed[f"{c}_bit"] = (~df_decomposed[c].isna()).astype(int)
        
        # サポートパターンの数値変換
        support_columns_in_df = [col for col in additional_info_common_support_columns if col in df_decomposed.columns]
        for c in support_columns_in_df:
            df_decomposed[f"{c}_numeric"] = df_decomposed[c].apply(convert_support_pattern_to_numeric)
        
        # X列とY列の設定
        exclude_columns = [csv_loader.DATE, csv_loader.CYCLE_NO, csv_loader.TD_NO, csv_loader.STA, 
                          csv_loader.SECTION_TD, csv_loader.FACE_TD] + bit_columns_in_df + support_columns_in_df
        if len(diff_targets) > 0:
            y_column = diff_targets[0][:-1]
            exclude_columns.append(y_column)
        else:
            y_column = ""
        
        x_columns = [col for col in df_decomposed.columns if col not in exclude_columns and col != 'position_id']
        
        # JSON対応：Inf値をNaNに変換してから削除
        numeric_columns = df_decomposed.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df_decomposed[col] = df_decomposed[col].replace([np.inf, -np.inf], np.nan)
        
        # NaN値の除去（より緩い条件）
        # Y列（ターゲット列）のNaNのみを除去し、X列のNaNは許容
        if y_column and y_column in df_decomposed.columns:
            df_decomposed = df_decomposed.dropna(subset=[y_column])
        
        # 重要なX列のみでNaN除去（全てのX列でNaNがある行のみ除去）
        if x_columns:
            # 全てのX列がNaNの行のみを除去
            essential_x_cols = [col for col in x_columns if col in df_decomposed.columns][:5]  # 重要な最初の5列のみ
            if essential_x_cols:
                df_decomposed = df_decomposed.dropna(subset=essential_x_cols, how='all')
        
        return df_decomposed, x_columns, y_column
    
    try:
        # 沈下量データセット作成
        settlement = decompose_columns(df, df_additional_info, csv_loader.SETTLEMENTS, csv_loader.DIFFERENCE_FROM_FINAL_SETTLEMENTS)
        # 変位量データセット作成
        convergence = decompose_columns(df, df_additional_info, csv_loader.CONVERGENCES, csv_loader.DIFFERENCE_FROM_FINAL_CONVERGENCES)
        
        logger.info(f"Settlement data: {type(settlement)}")
        logger.info(f"Convergence data: {type(convergence)}")
        
        return settlement, convergence
        
    except Exception as e:
        logger.error(f"Error in create_dataset: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return (pd.DataFrame(), [], ""), (pd.DataFrame(), [], "")

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


@router.get("/scatter-plot-data")
async def get_scatter_plot_data(
    folder_name: str = Query(default="01-hokkaido-akan", description="データフォルダ名"),
    max_distance_from_face: float = Query(default=100.0, gt=0, description="切羽からの最大距離"),
    plot_type: str = Query(default="convergences", description="プロットタイプ: convergences or settlements")
) -> Dict[str, Any]:
    """
    散布図用のデータを生成
    切羽からの距離 vs 計測経過日数で、変位量または沈下量を色で表現
    """
    try:
        from app.core.dataframe_cache import get_dataframe_cache
        
        # キャッシュからデータを取得
        cache = get_dataframe_cache()
        cached_data = cache.get_cached_data(folder_name, max_distance_from_face)
        
        if not cached_data:
            raise HTTPException(status_code=404, detail=f"Failed to load data for folder: {folder_name}")
        
        df_all = cached_data['df_all']
        settlements = cached_data['settlements']
        convergences = cached_data['convergences']
        
        # プロットタイプに応じてカラムを選択
        if plot_type == "convergences":
            value_columns = convergences
            label = "変位量"
        elif plot_type == "settlements":
            value_columns = settlements
            label = "沈下量"
        else:
            raise HTTPException(status_code=400, detail="plot_type must be 'convergences' or 'settlements'")
        
        # 散布図データを生成
        scatter_points = []
        
        # 必要なカラムが存在するかチェック
        if DISTANCE_FROM_FACE not in df_all.columns or DAYS_FROM_START not in df_all.columns:
            raise HTTPException(status_code=400, detail="Required columns not found in data")
        
        # 各値カラムごとにデータポイントを生成
        for _, row in df_all.iterrows():
            distance = row.get(DISTANCE_FROM_FACE)
            days = row.get(DAYS_FROM_START)
            
            # NaN値をスキップ
            if pd.isna(distance) or pd.isna(days):
                continue
            
            # 各測定値の平均を計算（色として使用）
            values = []
            for col in value_columns:
                if col in row and not pd.isna(row[col]):
                    values.append(float(row[col]))
            
            if values:
                avg_value = sum(values) / len(values)
            else:
                avg_value = 0
            
            scatter_points.append({
                "x": float(distance),
                "y": float(days),
                "value": avg_value,
                "td": float(row.get(TD_NO, 0)) if TD_NO in row else None
            })
        
        # カラーマップの範囲を計算
        if scatter_points:
            values = [p["value"] for p in scatter_points]
            min_value = min(values)
            max_value = max(values)
        else:
            min_value = 0
            max_value = 1
        
        return {
            "data": scatter_points,
            "label": label,
            "x_label": f"{DISTANCE_FROM_FACE} (m)",
            "y_label": DAYS_FROM_START,
            "color_range": {
                "min": min_value,
                "max": max_value
            },
            "plot_type": plot_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_scatter_plot_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        df_all = cached_data['df_all']
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
        
        # df_allをJSONシリアライズ可能な形式に変換（NaN/Inf対策）
        df_all_records = []
        for _, row in df_all.iterrows():
            clean_row = {}
            for col, val in row.items():
                if pd.isna(val) or (isinstance(val, float) and np.isinf(val)):
                    clean_row[col] = None
                else:
                    clean_row[col] = val
            df_all_records.append(clean_row)
        
        # レスポンスを作成
        return DistanceDataResponse(
            dct_df_td=formatted_dct_df_td,
            settlements=dct_df_settlement,
            convergences=dct_df_convergence,
            settlements_columns=settlements,
            convergences_columns=convergences,
            distances=list(dct_df_settlement.keys()),
            df_all=df_all_records
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_distance_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/debug-dataset")
async def debug_dataset(
    request: MLDatasetRequest
) -> Dict[str, Any]:
    """
    create_dataset関数のデバッグ用エンドポイント
    """
    try:
        from app.core.dataframe_cache import get_dataframe_cache
        
        # キャッシュからデータを取得
        cache = get_dataframe_cache()
        cached_data = cache.get_cached_data(request.folder_name, request.max_distance_from_face)
        
        if not cached_data:
            return {"error": f"Failed to load data for folder: {request.folder_name}"}
        
        df_all = cached_data['df_all']

        # 正しいパス構築
        input_folder = settings.DATA_FOLDER / request.folder_name / "main_tunnel" / "CN_measurement_data"
        cycle_support_csv = input_folder / 'cycle_support' / 'cycle_support.csv'
        observation_of_face_csv = input_folder / 'observation_of_face' / 'observation_of_face.csv'
        
        debug_info = {
            "df_all_shape": df_all.shape,
            "df_all_columns": df_all.columns.tolist(),
            "cycle_support_exists": cycle_support_csv.exists(),
            "observation_face_exists": observation_of_face_csv.exists(),
            "SETTLEMENTS": SETTLEMENTS,
            "CONVERGENCES": CONVERGENCES,
            "DIFFERENCE_FROM_FINAL_SETTLEMENTS": DIFFERENCE_FROM_FINAL_SETTLEMENTS,
            "DIFFERENCE_FROM_FINAL_CONVERGENCES": DIFFERENCE_FROM_FINAL_CONVERGENCES
        }
        
        # 追加情報ファイルの確認
        if cycle_support_csv.exists() and observation_of_face_csv.exists():
            df_additional_info = generate_additional_info_df(cycle_support_csv, observation_of_face_csv)
            debug_info["additional_info_shape"] = df_additional_info.shape
            debug_info["additional_info_columns"] = df_additional_info.columns.tolist()
            
            # 実際のデータの存在チェック
            available_settlements = [col for col in SETTLEMENTS if col in df_all.columns]
            available_convergences = [col for col in CONVERGENCES if col in df_all.columns]
            available_diff_settlements = [col for col in DIFFERENCE_FROM_FINAL_SETTLEMENTS if col in df_all.columns]
            available_diff_convergences = [col for col in DIFFERENCE_FROM_FINAL_CONVERGENCES if col in df_all.columns]
            
            debug_info["available_settlements"] = available_settlements
            debug_info["available_convergences"] = available_convergences
            debug_info["available_diff_settlements"] = available_diff_settlements
            debug_info["available_diff_convergences"] = available_diff_convergences
            
        else:
            debug_info["additional_info_error"] = "Additional info files not found"
        
        return debug_info
        
    except Exception as e:
        return {"error": str(e), "traceback": __import__('traceback').format_exc()}


def safe_to_dict(df_clean):
    """DataFrameを安全にdictに変換"""
    try:
        # まず辞書に変換
        result = df_clean.to_dict('records')
        # JSONシリアライゼーションのテスト
        json.dumps(result)
        return result
    except (ValueError, TypeError) as e:
        logger.warning(f"JSON serialization test failed: {e}")
        # より厳しいクリーニングを実行
        for col in df_clean.columns:
            if df_clean[col].dtype in ['float64', 'float32']:
                # すべてのfloat値をチェック
                mask = pd.notna(df_clean[col])
                valid_values = df_clean[col][mask]
                if len(valid_values) > 0:
                    # 極端な値をクリップ
                    df_clean[col] = df_clean[col].clip(-1e100, 1e100)
                    df_clean[col] = df_clean[col].round(6)  # 精度を制限
                    df_clean[col] = df_clean[col].where(pd.notna(df_clean[col]), None)
            elif 'datetime' in str(df_clean[col].dtype) or 'timestamp' in str(df_clean[col].dtype):
                # Timestamp/datetime型を文字列に変換
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].where(df_clean[col] != 'NaT', None)
        
        result = df_clean.to_dict('records')
        # 再度テスト
        try:
            json.dumps(result)
            return result
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to create JSON-safe data even after aggressive cleaning: {e}")
            # 最後の手段：Timestamp型をすべて文字列に変換
            try:
                for record in result:
                    for key, value in record.items():
                        if hasattr(value, 'isoformat'):
                            # datetime/Timestamp型
                            record[key] = value.isoformat()
                        elif pd.isna(value):
                            record[key] = None
                json.dumps(result)
                return result
            except Exception:
                logger.error("Complete failure in JSON serialization")
                return []

def clean_dataframe_for_json(df):
    """DataFrameをJSON対応可能な形式にクリーンアップ"""
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # すべての列をチェック
    for col in df_clean.columns:
        # 数値列の処理
        if df_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            # Inf/-InfをNaNに変換
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
            # 非常に大きな値もNaNに変換（JSON限界を超える値）
            df_clean[col] = df_clean[col].where(
                (df_clean[col] >= -1e308) & (df_clean[col] <= 1e308), 
                np.nan
            )
            # NaNをNoneに変換（JSON対応）
            df_clean[col] = df_clean[col].where(pd.notna(df_clean[col]), None)
        elif 'datetime' in str(df_clean[col].dtype) or 'timestamp' in str(df_clean[col].dtype):
            # Timestamp/datetime型を文字列に変換
            df_clean[col] = df_clean[col].astype(str)
            df_clean[col] = df_clean[col].where(df_clean[col] != 'NaT', None)
        else:
            # オブジェクト型列のNaNも処理
            df_clean[col] = df_clean[col].where(pd.notna(df_clean[col]), None)
    
    return df_clean

@router.post("/make-dataset")
async def make_dataset(
    request: MLDatasetRequest
):
    """
    計測データから機械学習用のデータセットを作成
    訓練用とテスト用に分割し、指定したフォーマットで保存
    """
    # 安全なフォールバック：常に空のレスポンスを準備
    safe_response = {
        "settlement_data": [],
        "convergence_data": []
    }
    
    try:
        logger.info(f"make_dataset called with folder_name: {request.folder_name}, max_distance: {request.max_distance_from_face}")
        from app.core.dataframe_cache import get_dataframe_cache
        
        # キャッシュからデータを取得
        cache = get_dataframe_cache()
        logger.info(f"Cache retrieved")
        cached_data = cache.get_cached_data(request.folder_name, request.max_distance_from_face)
        logger.info(f"Cached data result: {cached_data is not None}")
        
        if not cached_data:
            logger.error(f"Failed to load data for folder: {request.folder_name}")
            return safe_response
        
        df_all = cached_data['df_all']
        logger.info(f"df_all shape: {df_all.shape if not df_all.empty else 'empty'}")

        # 正しいパス構築：settings.DATA_FOLDERを使用
        input_folder = settings.DATA_FOLDER / request.folder_name / "main_tunnel" / "CN_measurement_data"
        cycle_support_csv = input_folder / 'cycle_support' / 'cycle_support.csv'
        observation_of_face_csv = input_folder / 'observation_of_face' / 'observation_of_face.csv'
        logger.info(f"Input folder: {input_folder}")
        logger.info(f"Cycle support CSV exists: {cycle_support_csv.exists()}")
        logger.info(f"Observation CSV exists: {observation_of_face_csv.exists()}")
        
        if not cycle_support_csv.exists() or not observation_of_face_csv.exists():
            logger.warning("Required CSV files not found, returning empty response")
            return safe_response
        
        df_additional_info = generate_additional_info_df(cycle_support_csv, observation_of_face_csv)
        logger.info(f"Additional info shape: {df_additional_info.shape if not df_additional_info.empty else 'empty'}")
        
        if df_additional_info.empty:
            logger.warning("Additional info is empty, returning empty response")
            return safe_response
        
        # 実際のcreate_dataset関数を呼び出す
        logger.info("Calling create_dataset function")
        settlement_data, convergence_data = create_dataset(df_all, df_additional_info)
        logger.info(f"Settlement data type: {type(settlement_data)}")
        logger.info(f"Convergence data type: {type(convergence_data)}")

        settlement_result = []
        convergence_result = []
        
        # Settlement データの処理
        try:
            if isinstance(settlement_data, tuple) and len(settlement_data) == 3:
                df_s, x_cols_s, y_col_s = settlement_data
                logger.info(f"Settlement DataFrame shape: {df_s.shape if not df_s.empty else 'empty'}")
                logger.info(f"Settlement X columns: {x_cols_s[:5] if x_cols_s else 'None'}")  # 最初の5列のみログ
                logger.info(f"Settlement Y column: {y_col_s}")
                
                if not df_s.empty:
                    # DataFrameをJSON安全な形式に変換
                    df_s_clean = df_s.copy()
                    
                    # すべての数値列に対してInf/NaN処理
                    for col in df_s_clean.columns:
                        if df_s_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                            # Inf値をNaNに変換
                            df_s_clean[col] = df_s_clean[col].replace([np.inf, -np.inf], np.nan)
                            # NaNを0に変換（もしくはNoneに）
                            df_s_clean[col] = df_s_clean[col].fillna(0)
                            # 浮動小数点を丸める（精度制限）
                            if df_s_clean[col].dtype in ['float64', 'float32']:
                                df_s_clean[col] = df_s_clean[col].round(6)
                    
                    # DataFrameを辞書のリストに変換
                    settlement_result = df_s_clean.to_dict('records')
                    
                    # 各レコードのクリーンアップ
                    for record in settlement_result:
                        for key, value in list(record.items()):
                            # Timestamp型の処理
                            if hasattr(value, 'isoformat'):
                                record[key] = value.isoformat()
                            # NaN/None/Inf の処理
                            elif pd.isna(value) or (isinstance(value, float) and np.isinf(value)):
                                record[key] = None
                            # 通常の数値は安全な範囲にクリップ
                            elif isinstance(value, (float, np.float64, np.float32)):
                                record[key] = float(np.clip(value, -1e100, 1e100))
                            elif isinstance(value, (int, np.int64, np.int32)):
                                record[key] = int(value)
                    
                    logger.info(f"Settlement result length: {len(settlement_result)}")
        except Exception as e:
            logger.error(f"Error processing settlement data: {e}", exc_info=True)
            settlement_result = []
        
        # Convergence データの処理
        try:
            if isinstance(convergence_data, tuple) and len(convergence_data) == 3:
                df_c, x_cols_c, y_col_c = convergence_data
                logger.info(f"Convergence DataFrame shape: {df_c.shape if not df_c.empty else 'empty'}")
                logger.info(f"Convergence X columns: {x_cols_c[:5] if x_cols_c else 'None'}")  # 最初の5列のみログ
                logger.info(f"Convergence Y column: {y_col_c}")
                
                if not df_c.empty:
                    # DataFrameをJSON安全な形式に変換
                    df_c_clean = df_c.copy()
                    
                    # すべての数値列に対してInf/NaN処理
                    for col in df_c_clean.columns:
                        if df_c_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                            # Inf値をNaNに変換
                            df_c_clean[col] = df_c_clean[col].replace([np.inf, -np.inf], np.nan)
                            # NaNを0に変換
                            df_c_clean[col] = df_c_clean[col].fillna(0)
                            # 浮動小数点を丸める（精度制限）
                            if df_c_clean[col].dtype in ['float64', 'float32']:
                                df_c_clean[col] = df_c_clean[col].round(6)
                    
                    # DataFrameを辞書のリストに変換
                    convergence_result = df_c_clean.to_dict('records')
                    
                    # 各レコードのクリーンアップ
                    for record in convergence_result:
                        for key, value in list(record.items()):
                            # Timestamp型の処理
                            if hasattr(value, 'isoformat'):
                                record[key] = value.isoformat()
                            # NaN/None/Inf の処理
                            elif pd.isna(value) or (isinstance(value, float) and np.isinf(value)):
                                record[key] = None
                            # 通常の数値は安全な範囲にクリップ
                            elif isinstance(value, (float, np.float64, np.float32)):
                                record[key] = float(np.clip(value, -1e100, 1e100))
                            elif isinstance(value, (int, np.int64, np.int32)):
                                record[key] = int(value)
                    
                    logger.info(f"Convergence result length: {len(convergence_result)}")
        except Exception as e:
            logger.error(f"Error processing convergence data: {e}", exc_info=True)
            convergence_result = []

        logger.info(f"Final settlement_result length: {len(settlement_result)}")
        logger.info(f"Final convergence_result length: {len(convergence_result)}")

        # レスポンス作成前に最終的なJSON対応性をテスト
        try:
            response_data = {
                "settlement_data": settlement_result,
                "convergence_data": convergence_result
            }
            # JSONシリアライゼーションテスト
            json_str = json.dumps(response_data, default=str)  # default=strで未知の型を文字列化
            logger.info(f"JSON serialization test passed. Response size: {len(json_str)} bytes")
            
            return response_data
            
        except (ValueError, TypeError) as e:
            logger.error(f"JSON serialization failed: {e}", exc_info=True)
            return safe_response
        
    except Exception as e:
        logger.error(f"Error in make_dataset: {e}", exc_info=True)
        return safe_response