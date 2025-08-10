import os
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import tempfile
from io import StringIO

from app.schemas.charts import (
    ChartRequest,
    ChartResponse,
    HistogramChartRequest,
    MultiChartRequest,
    DataRequest,
    DataResponse
)
from app.core.csv_loader import CSVDataLoader
from app.api.endpoints.displacement import proccess_a_measure_file
from app.api.endpoints.measurements import (
    DATE, CYCLE_NO, SECTION_TD, FACE_TD, TD_NO, 
    CONVERGENCES, SETTLEMENTS, STA, DISTANCE_FROM_FACE, 
    DAYS_FROM_START, DIFFERENCE_FROM_FINAL_CONVERGENCES, 
    DIFFERENCE_FROM_FINAL_SETTLEMENTS
)

router = APIRouter()

DISTANCES_FROM_FACE = [3, 5, 10, 20, 50, 100]


def draw_charts_distance_displace(output_path: str, dict_df: Dict, columns: List[str]):
    """距離vs変位チャートの描画"""
    plt.figure(figsize=(10, 6))
    axis_name = columns[0].replace("1", "")
    
    for name, df in dict_df.items():
        try:
            df[axis_name] = df[columns].mean(axis=1)
            plt.plot(df[TD_NO], df[axis_name], label=name, marker='o', linestyle='-', markersize=4)
        except Exception as e:
            print(f"Error: {e}")
    
    plt.title(f"{axis_name} over TD")
    plt.xlabel('TD (m)')
    plt.ylabel(f"{axis_name} (mm)")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.close()


def draw_charts_histram_displace(output_path: str, dct_df: Dict, columns: List[str]):
    """変位分布のヒストグラム描画"""
    plt.figure(figsize=(10, 6))
    axis_name = columns[0].replace("1", "")
    
    for name, values in dct_df.items():
        data = np.array(values)
        try:
            sns.histplot(data, bins=range(int(data.min()), int(data.max()) + 2), 
                        alpha=0.5, label=name, kde=True)
        except Exception as e:
            print(f"Error: {e}")
    
    plt.title(f"{axis_name} Distribution")
    plt.xlabel(f"{axis_name} (mm)")
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.close()


def generate_dataframes(measurement_data: List[Dict], max_distance_from_face: float):
    """測定データからDataFrameを生成"""
    df_all = []
    
    for data in measurement_data:
        try:
            # CSV文字列からDataFrameを作成
            df = pd.read_csv(StringIO(data['content']))
            df_all.append(df)
        except Exception as e:
            print(f"Error processing data: {e}")
            continue
    
    if not df_all:
        raise ValueError("No valid data to process")
    
    df_all = pd.concat(df_all)
    df_all = df_all[df_all[DISTANCE_FROM_FACE] >= -1]
    df_all = df_all[df_all[DISTANCE_FROM_FACE] <= max_distance_from_face]
    
    settlements = [settle for settle in SETTLEMENTS if settle in df_all.columns]
    convergences = [conv for conv in CONVERGENCES if conv in df_all.columns]
    
    dct_df_settlement = {}
    dct_df_convergence = {}
    dct_df_td = {}
    
    for distance_from_face in DISTANCES_FROM_FACE:
        if max_distance_from_face < distance_from_face:
            continue
        
        dct_df_settlement[f"{distance_from_face}m"] = []
        dct_df_convergence[f"{distance_from_face}m"] = []
        dfs = []
        
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


@router.post("/draw-distance-chart", response_model=ChartResponse)
async def draw_distance_chart(request: ChartRequest):
    """距離vs変位チャートを生成"""
    try:
        # データ生成
        df_all, dct_df_settlement, dct_df_convergence, dct_df_td, settlements, convergences = \
            generate_dataframes(request.measurement_data, request.max_distance_from_face or 200)
        
        # チャートタイプによって処理を分岐
        if request.chart_type == "settlement":
            columns = settlements
            dict_df = dct_df_td
        elif request.chart_type == "convergence":
            columns = convergences
            dict_df = dct_df_td
        else:
            raise ValueError(f"Invalid chart type: {request.chart_type}")
        
        # 一時ファイルに描画
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            draw_charts_distance_displace(tmp.name, dict_df, columns)
            
            return ChartResponse(
                success=True,
                message="Chart generated successfully",
                file_path=tmp.name
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/draw-histogram-chart", response_model=ChartResponse)
async def draw_histogram_chart(request: HistogramChartRequest):
    """変位分布のヒストグラムを生成"""
    try:
        # データ生成
        df_all, dct_df_settlement, dct_df_convergence, dct_df_td, settlements, convergences = \
            generate_dataframes(request.measurement_data, request.max_distance_from_face or 200)
        
        # チャートタイプによって処理を分岐
        if request.chart_type == "settlement":
            columns = settlements
            dct_df = dct_df_settlement
        elif request.chart_type == "convergence":
            columns = convergences
            dct_df = dct_df_convergence
        else:
            raise ValueError(f"Invalid chart type: {request.chart_type}")
        
        # 一時ファイルに描画
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            draw_charts_histram_displace(tmp.name, dct_df, columns)
            
            return ChartResponse(
                success=True,
                message="Histogram generated successfully",
                file_path=tmp.name
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/draw-multiple-charts", response_model=ChartResponse)
async def draw_multiple_charts(request: MultiChartRequest):
    """複数のチャートを一度に生成"""
    try:
        # データ生成
        df_all, dct_df_settlement, dct_df_convergence, dct_df_td, settlements, convergences = \
            generate_dataframes(request.measurement_data, request.max_distance_from_face or 200)
        
        # 出力ディレクトリ作成
        output_dir = tempfile.mkdtemp()
        generated_files = []
        
        # 各チャートを生成
        if "settle" in request.chart_types:
            file_path = os.path.join(output_dir, 'settle.png')
            draw_charts_distance_displace(file_path, dct_df_td, settlements)
            generated_files.append(file_path)
        
        if "conv" in request.chart_types:
            file_path = os.path.join(output_dir, 'conv.png')
            draw_charts_distance_displace(file_path, dct_df_td, convergences)
            generated_files.append(file_path)
        
        if "settle_hist" in request.chart_types:
            file_path = os.path.join(output_dir, 'settle_hist2.png')
            draw_charts_histram_displace(file_path, dct_df_settlement, settlements)
            generated_files.append(file_path)
        
        if "conv_hist" in request.chart_types:
            file_path = os.path.join(output_dir, 'conv_hist2.png')
            draw_charts_histram_displace(file_path, dct_df_convergence, convergences)
            generated_files.append(file_path)
        
        return ChartResponse(
            success=True,
            message=f"Generated {len(generated_files)} charts successfully",
            file_path=output_dir,
            files=generated_files
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download-chart/{file_path:path}")
async def download_chart(file_path: str):
    """生成されたチャートファイルをダウンロード"""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        media_type="image/png",
        filename=os.path.basename(file_path)
    )


@router.post("/get-chart-data", response_model=DataResponse)
async def get_chart_data(request: DataRequest):
    """チャート用のデータを取得"""
    try:
        # データ生成
        df_all, dct_df_settlement, dct_df_convergence, dct_df_td, settlements, convergences = \
            generate_dataframes(request.measurement_data, request.max_distance_from_face or 200)
        
        # 返却用データ構造の作成
        response_data = {
            "settlements": settlements,
            "convergences": convergences,
            "settlement_data": {},
            "convergence_data": {},
            "td_data": {}
        }
        
        # 沈下量データ
        for key, values in dct_df_settlement.items():
            response_data["settlement_data"][key] = values
        
        # 変位量データ
        for key, values in dct_df_convergence.items():
            response_data["convergence_data"][key] = values
        
        # TD データ（DataFrame to dict）
        for key, df in dct_df_td.items():
            response_data["td_data"][key] = df.to_dict(orient='records')
        
        # 全データの統計情報を追加
        if not df_all.empty:
            response_data["statistics"] = {
                "total_records": len(df_all),
                "td_range": [float(df_all[TD_NO].min()), float(df_all[TD_NO].max())],
                "distance_range": [float(df_all[DISTANCE_FROM_FACE].min()), 
                                 float(df_all[DISTANCE_FROM_FACE].max())],
                "available_distances": list(dct_df_td.keys())
            }
        
        return DataResponse(
            success=True,
            message="Data retrieved successfully",
            data=response_data
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get-processed-dataframe", response_model=DataResponse)
async def get_processed_dataframe(request: DataRequest):
    """処理済みDataFrameを取得"""
    try:
        # データ生成
        df_all, dct_df_settlement, dct_df_convergence, dct_df_td, settlements, convergences = \
            generate_dataframes(request.measurement_data, request.max_distance_from_face or 200)
        
        # DataFrameをJSON形式に変換
        result = {
            "df_all": df_all.to_dict(orient='records'),
            "columns": {
                "settlements": settlements,
                "convergences": convergences,
                "all_columns": df_all.columns.tolist()
            },
            "shape": {
                "rows": len(df_all),
                "columns": len(df_all.columns)
            }
        }
        
        # 指定された距離ごとのデータも含める
        if request.include_distance_data:
            result["distance_data"] = {}
            for distance in DISTANCES_FROM_FACE:
                if distance <= (request.max_distance_from_face or 200):
                    distance_key = f"{distance}m"
                    if distance_key in dct_df_td:
                        result["distance_data"][distance_key] = {
                            "dataframe": dct_df_td[distance_key].to_dict(orient='records'),
                            "settlement_values": dct_df_settlement.get(distance_key, []),
                            "convergence_values": dct_df_convergence.get(distance_key, [])
                        }
        
        return DataResponse(
            success=True,
            message="Processed dataframe retrieved successfully",
            data=result
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))