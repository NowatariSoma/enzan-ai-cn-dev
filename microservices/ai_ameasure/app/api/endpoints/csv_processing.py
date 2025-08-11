"""
CSV処理関連のAPIエンドポイント
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from app import schemas
from app.core.config import settings
from app.core.csv_loader import CSVDataLoader
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

logger = logging.getLogger(__name__)

router = APIRouter()

# CSVデータローダーのインスタンス
csv_loader = CSVDataLoader()


@router.post("/process-file", response_model=schemas.ProcessedMeasurementResponse)
async def process_measurement_file(
    request: schemas.ProcessMeasurementRequest,
) -> schemas.ProcessedMeasurementResponse:
    """
    計測ファイルを処理し、切羽からの距離や差分を計算

    指定されたCSVファイルを読み込み、以下の処理を実行:
    - 切羽からの距離の計算
    - 計測経過日数の計算
    - 最終値との差分計算
    - 日次平均の計算
    """
    try:
        # ファイルパスを構築
        file_path = Path(request.file_path)
        if not file_path.is_absolute():
            # 相対パスの場合はデータフォルダからの相対パスとして扱う
            file_path = (
                settings.DATA_FOLDER
                / request.folder_name
                / "main_tunnel"
                / "CN_measurement_data"
                / "measurements_A"
                / request.file_path
            )

        # CSVローダーで処理
        processed_df = csv_loader.process_measurement_file(
            file_path=file_path,
            max_distance_from_face=request.max_distance_from_face,
            duration_days=request.duration_days,
        )

        if processed_df.empty:
            raise HTTPException(status_code=404, detail="No data found in the file")

        # DataFrameをレスポンス形式に変換
        data = processed_df.to_dict(orient="records")

        # カラム情報を収集
        columns = list(processed_df.columns)

        # 統計情報を計算
        stats = {
            "total_rows": len(processed_df),
            "total_columns": len(columns),
            "date_range": None,
            "distance_range": None,
        }

        # 日付範囲を取得
        if "計測日時" in processed_df.columns:
            date_col = processed_df["計測日時"]
            if not date_col.isna().all():
                stats["date_range"] = {
                    "start": date_col.min().isoformat() if pd.notna(date_col.min()) else None,
                    "end": date_col.max().isoformat() if pd.notna(date_col.max()) else None,
                }

        # 距離範囲を取得
        if "切羽からの距離" in processed_df.columns:
            distance_col = processed_df["切羽からの距離"]
            if not distance_col.isna().all():
                stats["distance_range"] = {
                    "min": float(distance_col.min()) if pd.notna(distance_col.min()) else None,
                    "max": float(distance_col.max()) if pd.notna(distance_col.max()) else None,
                }

        return schemas.ProcessedMeasurementResponse(
            data=data,
            columns=columns,
            stats=stats,
            file_path=str(file_path),
            processing_params={
                "max_distance_from_face": request.max_distance_from_face,
                "duration_days": request.duration_days,
            },
        )

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
    except Exception as e:
        logger.error(f"Error processing measurement file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-upload")
async def process_uploaded_file(
    file: UploadFile = File(...),
    max_distance_from_face: float = Query(default=100.0, gt=0, description="切羽からの最大距離"),
    duration_days: int = Query(default=90, gt=0, description="解析対象期間（日数）"),
) -> Dict[str, Any]:
    """
    アップロードされたCSVファイルを処理

    CSVファイルをアップロードして、直接処理を実行
    """
    try:
        # ファイルの内容を読み込み
        contents = await file.read()

        # 一時ファイルとして保存
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name

        try:
            # CSVローダーで処理
            processed_df = csv_loader.process_measurement_file(
                file_path=tmp_file_path,
                max_distance_from_face=max_distance_from_face,
                duration_days=duration_days,
            )

            if processed_df.empty:
                raise HTTPException(
                    status_code=400, detail="No valid data found in the uploaded file"
                )

            # DataFrameをレスポンス形式に変換
            data = processed_df.to_dict(orient="records")

            # カラム情報を収集
            columns = list(processed_df.columns)

            # 統計情報を計算
            stats = {
                "total_rows": len(processed_df),
                "total_columns": len(columns),
                "filename": file.filename,
            }

            return {
                "status": "success",
                "data": data,
                "columns": columns,
                "stats": stats,
                "processing_params": {
                    "max_distance_from_face": max_distance_from_face,
                    "duration_days": duration_days,
                },
            }

        finally:
            # 一時ファイルを削除
            Path(tmp_file_path).unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available-columns")
async def get_available_columns(
    folder_name: str = Query(default="01-hokkaido-akan", description="データフォルダ名"),
    file_name: str = Query(default="measurements_A_00001.csv", description="サンプルファイル名"),
) -> Dict[str, List[str]]:
    """
    利用可能なカラム名を取得

    指定されたファイルから利用可能なカラム名のリストを返す
    """
    try:
        # ファイルパスを構築
        file_path = (
            settings.DATA_FOLDER
            / folder_name
            / "main_tunnel"
            / "CN_measurement_data"
            / "measurements_A"
            / file_name
        )

        if not file_path.exists():
            # ファイルが見つからない場合は、フォルダ内の最初のファイルを使用
            measurements_path = (
                settings.DATA_FOLDER
                / folder_name
                / "main_tunnel"
                / "CN_measurement_data"
                / "measurements_A"
            )
            csv_files = list(measurements_path.glob("*.csv"))

            if not csv_files:
                raise HTTPException(status_code=404, detail="No measurement files found")

            file_path = csv_files[0]

        # CSVファイルを読み込み（ヘッダーのみ）
        df = csv_loader.load_measurement_data(file_path)

        if df.empty:
            raise HTTPException(status_code=404, detail="No data found in the file")

        # カラムを分類
        columns_by_type = {
            "convergences": [],
            "settlements": [],
            "metadata": [],
            "calculated": [],
            "all": list(df.columns),
        }

        for col in df.columns:
            if "変位量" in col:
                if "オフセット" in col:
                    columns_by_type["calculated"].append(col)
                elif "差分" in col:
                    columns_by_type["calculated"].append(col)
                else:
                    columns_by_type["convergences"].append(col)
            elif "沈下量" in col:
                if "オフセット" in col:
                    columns_by_type["calculated"].append(col)
                elif "差分" in col:
                    columns_by_type["calculated"].append(col)
                else:
                    columns_by_type["settlements"].append(col)
            elif any(key in col for key in ["TD", "STA", "サイクル", "日時", "切羽"]):
                columns_by_type["metadata"].append(col)
            else:
                columns_by_type["calculated"].append(col)

        return columns_by_type

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting available columns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-data")
async def extract_specific_data(
    file_path: str = Query(..., description="CSVファイルパス"),
    columns: List[str] = Query(..., description="抽出するカラム名のリスト"),
    folder_name: str = Query(default="01-hokkaido-akan", description="データフォルダ名"),
) -> Dict[str, Any]:
    """
    指定されたカラムのデータを抽出

    CSVファイルから指定されたカラムのみを抽出して返す
    """
    try:
        # ファイルパスを構築
        full_path = Path(file_path)
        if not full_path.is_absolute():
            full_path = (
                settings.DATA_FOLDER
                / folder_name
                / "main_tunnel"
                / "CN_measurement_data"
                / "measurements_A"
                / file_path
            )

        # CSVファイルを読み込み
        df = csv_loader.load_measurement_data(full_path)

        if df.empty:
            raise HTTPException(status_code=404, detail="No data found in the file")

        # 指定されたカラムが存在するか確認
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Columns not found: {missing_columns}")

        # 指定されたカラムのみを抽出
        extracted_df = df[columns]

        # データを辞書形式に変換
        data = extracted_df.to_dict(orient="records")

        # 統計情報を計算
        stats = {}
        for col in columns:
            if pd.api.types.is_numeric_dtype(extracted_df[col]):
                stats[col] = {
                    "mean": (
                        float(extracted_df[col].mean())
                        if not extracted_df[col].isna().all()
                        else None
                    ),
                    "min": (
                        float(extracted_df[col].min())
                        if not extracted_df[col].isna().all()
                        else None
                    ),
                    "max": (
                        float(extracted_df[col].max())
                        if not extracted_df[col].isna().all()
                        else None
                    ),
                    "std": (
                        float(extracted_df[col].std())
                        if not extracted_df[col].isna().all()
                        else None
                    ),
                }

        return {"data": data, "columns": columns, "stats": stats, "row_count": len(data)}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"Error extracting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
