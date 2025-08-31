import logging
import math
import os
import json
import base64
import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd
import numpy as np
import joblib
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

# 学習モデル
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from app.core.config import settings
from app.schemas import displacement_analysis as schemas

# Streamlitプロジェクトから直接インポート
import importlib.util
spec = importlib.util.spec_from_file_location("streamlit_displacement", "/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure/app/displacement_temporal_spacial_analysis.py")
streamlit_displacement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(streamlit_displacement)

# Streamlitの関数を直接参照
analyze_displacement = streamlit_displacement.analyze_displacement
generate_dataframes = streamlit_displacement.generate_dataframes
generate_additional_info_df = streamlit_displacement.generate_additional_info_df
create_dataset = streamlit_displacement.create_dataset
Y_COLUMNS = streamlit_displacement.Y_COLUMNS
STA = streamlit_displacement.STA
DISTANCE_FROM_FACE = streamlit_displacement.DISTANCE_FROM_FACE
TD_NO = streamlit_displacement.TD_NO
DATE = streamlit_displacement.DATE
DISTANCES_FROM_FACE = streamlit_displacement.DISTANCES_FROM_FACE
SETTLEMENTS = streamlit_displacement.SETTLEMENTS
CONVERGENCES = streamlit_displacement.CONVERGENCES

# displacement.pyもStreamlitから
spec2 = importlib.util.spec_from_file_location("streamlit_displ", "/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure/app/displacement.py")
streamlit_displ = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(streamlit_displ)
DURATION_DAYS = streamlit_displ.DURATION_DAYS


logger = logging.getLogger(__name__)
router = APIRouter()

def get_models():
    """学習モデルの定義（GUIと完全に同じ設定）- 毎回新しいインスタンスを返す"""
    return {
        "Random Forest": RandomForestRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "SVR": SVR(kernel='linear', C=1.0, epsilon=0.2),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=42),
        "MLP": MLPRegressor(hidden_layer_sizes=(100, 100, 50), max_iter=1000, random_state=42),
    }

def extract_feature_importance(model, x_columns):
    """モデルから特徴量重要度を安全に抽出（キーと値をペアで返す）"""
    if hasattr(model, 'feature_importances_'):
        # 特徴量名と重要度をペアにしたdictを作成
        features = {}
        if len(x_columns) == len(model.feature_importances_):
            for name, importance in zip(x_columns, model.feature_importances_):
                features[name] = float(importance)
        
        # 重要度順にソート
        features_sorted = dict(sorted(features.items(), key=lambda item: item[1], reverse=True))
        
        return {
            'features': features_sorted,
            'available': True,
            'total_features': len(features),
            'model_type': type(model).__name__
        }
    else:
        return {
            'features': {},
            'available': False,
            'reason': f"{type(model).__name__} does not support feature importance",
            'total_features': 0,
            'model_type': type(model).__name__
        }

def extract_metrics_from_output(output_path):
    """出力フォルダからメトリクスを抽出"""
    metrics = {}
    
    # 沈下量結果
    settlement_csv = os.path.join(output_path, 'result最終沈下量との差分.csv')
    if os.path.exists(settlement_csv):
        try:
            df = pd.read_csv(settlement_csv)
            train_df = df[df['mode'] == 'train']
            val_df = df[df['mode'] == 'validate']
            y_col = '最終沈下量との差分'
            
            if not train_df.empty and not val_df.empty:
                from sklearn.metrics import r2_score, mean_squared_error
                metrics['settlement'] = {
                    'r2_train': float(r2_score(train_df[y_col], train_df['pred'])),
                    'r2_validate': float(r2_score(val_df[y_col], val_df['pred'])),
                    'mse_train': float(mean_squared_error(train_df[y_col], train_df['pred'])),
                    'mse_validate': float(mean_squared_error(val_df[y_col], val_df['pred']))
                }
        except Exception as e:
            logger.error(f"Error extracting settlement metrics: {e}")
    
    # 変位量結果  
    convergence_csv = os.path.join(output_path, 'result最終変位量との差分.csv')
    if os.path.exists(convergence_csv):
        try:
            df = pd.read_csv(convergence_csv)
            train_df = df[df['mode'] == 'train']
            val_df = df[df['mode'] == 'validate']
            y_col = '最終変位量との差分'
            
            if not train_df.empty and not val_df.empty:
                from sklearn.metrics import r2_score, mean_squared_error
                metrics['convergence'] = {
                    'r2_train': float(r2_score(train_df[y_col], train_df['pred'])),
                    'r2_validate': float(r2_score(val_df[y_col], val_df['pred'])),
                    'mse_train': float(mean_squared_error(train_df[y_col], train_df['pred'])),
                    'mse_validate': float(mean_squared_error(val_df[y_col], val_df['pred']))
                }
        except Exception as e:
            logger.error(f"Error extracting convergence metrics: {e}")
            
    return metrics

# アウトプットフォルダ
OUTPUT_FOLDER = "./output"
MAX_DISTANCE_M = 200

def list_folders() -> List[str]:
    """List all folder names in the DATA_FOLDER."""
    input_folder = settings.DATA_FOLDER
    if not input_folder.exists():
        return []
    return sorted([f.name for f in input_folder.iterdir() if f.is_dir()])

def draw_local_prediction_chart(x_measure: List[float], df_measure_y: pd.DataFrame, 
                               x_predict: List[float], df_predict_y: pd.DataFrame, 
                               title: str) -> str:
    """
    局所的予測チャートを描画し、Base64エンコードされた画像を返す
    GUIのdraw_local_prediction_chart関数を完全に模倣
    """
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')
    
    # 実測値をプロット
    for i, c in enumerate(df_measure_y.columns):
        plt.plot(x_measure, df_measure_y[c], label=c, marker='x', linestyle='--', 
                markersize=4, alpha=0.5, color=cmap(i))
    
    # 予測値をプロット
    for i, c in enumerate(df_predict_y.columns):
        plt.plot(x_predict, df_predict_y[c], label=['予測最終' + c], marker='o', 
                linestyle='-', markersize=4, color=cmap(i))
    
    plt.title(title)
    plt.xlabel(DISTANCE_FROM_FACE)
    plt.ylabel("(mm)")
    plt.legend()
    plt.grid()
    
    # 画像をBase64エンコード
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

def simulate_displacement(input_folder: Path, a_measure_path: Path, max_distance_from_face: float, 
                         daily_advance: Optional[float] = None, distance_from_face: Optional[float] = None, 
                         recursive: bool = False) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    GUIのsimulate_displacement関数を完全に模倣（displacement_temporal_spacial_analysis.pyから直接使用）
    """
    # GUIと全く同じように処理
    cycle_support_csv = str(input_folder / 'cycle_support' / 'cycle_support.csv')
    observation_of_face_csv = str(input_folder / 'observation_of_face' / 'observation_of_face.csv')

    df_additional_info = generate_additional_info_df(cycle_support_csv, observation_of_face_csv)
    df_additional_info.drop(columns=[STA], inplace=True)
    
    # Process each CSV file using the preprocess function
    df_all, _, _, _, settlements, convergences = generate_dataframes([str(a_measure_path)], max_distance_from_face)
    
    if daily_advance and distance_from_face:
        # Create a new dataframe with the specified length and interval
        remaining_distance = max_distance_from_face - distance_from_face
        max_record = math.ceil(min(remaining_distance / daily_advance, DURATION_DAYS))
        max_record = max(max_record, 1)  # Ensure at least 1 record
        
        print(f"🔍 SIMULATION DEBUG:")
        print(f"  - distance_from_face: {distance_from_face}")
        print(f"  - max_distance_from_face: {max_distance_from_face}")
        print(f"  - daily_advance: {daily_advance}")
        print(f"  - remaining_distance: {remaining_distance}")
        print(f"  - max_record: {max_record}")
        df_all_actual = df_all[df_all[DISTANCE_FROM_FACE] < distance_from_face]
        if df_all_actual.empty:
            df_all_new = pd.DataFrame([df_all.iloc[0]] * max_record).reset_index()
        else:
            df_all_new = pd.DataFrame([df_all_actual.iloc[-1]] * max_record).reset_index()
        df_all_new[DATE] = pd.to_datetime(df_all.iloc[0][DATE]) + pd.to_timedelta(range(max_record), unit='D')

        df_all_new[DISTANCE_FROM_FACE] = distance_from_face + daily_advance * pd.Series(range(max_record))
        df_all = pd.concat([df_all_actual, df_all_new[df_all_new[DISTANCE_FROM_FACE] >= distance_from_face]], ignore_index=True).reset_index()

    settlement_data, convergence_data = create_dataset(df_all, df_additional_info)

    # モデルパスはGUIと完全に同じ構造
    model_paths = {
        "final_value_prediction_model": [
            os.path.join(OUTPUT_FOLDER, "model_final_settlement.pkl"),
            os.path.join(OUTPUT_FOLDER, "model_final_convergence.pkl")
        ],
        "prediction_model": [
            os.path.join(OUTPUT_FOLDER, "model_settlement.pkl"),
            os.path.join(OUTPUT_FOLDER, "model_convergence.pkl")
        ]
    }
    
    for i, ((df, x_columns, y_column), target) in enumerate(zip([settlement_data, convergence_data], [settlements, convergences])):
        final_model_path = model_paths["final_value_prediction_model"][i]
        model_path = model_paths["prediction_model"][i]
        
        if os.path.exists(final_model_path):
            final_model = joblib.load(final_model_path)
            
            if recursive and os.path.exists(model_path):
                model = joblib.load(model_path)
                # まずは沈下量・変位量を予測し、それに基づき最終沈下量・変位量を予測する
                _y_column = x_columns[2]
                _x_columns = [x for x in x_columns if x != _y_column]
                _x_columns = [x for x in _x_columns if x != y_column]
                _y_hat = model.predict(pd.DataFrame(df[_x_columns]))
                df.loc[df[DISTANCE_FROM_FACE] > distance_from_face, _y_column] = _y_hat[df[DISTANCE_FROM_FACE] > distance_from_face]

            y_hat = final_model.predict(df[x_columns])
            for position_id in df['position_id'].unique():
                df_all[f"{target[position_id]}_prediction"] = y_hat[df['position_id'] == position_id] + df_all[target[position_id]]
        else:
            logger.warning(f"Model file not found: {final_model_path}")
            
    return df_all, settlements, convergences

@router.get("/folders")
async def get_folders() -> schemas.FolderListResponse:
    """利用可能なフォルダ一覧を取得"""
    folders = list_folders()
    return schemas.FolderListResponse(folders=folders)

@router.get("/measurement-files/{folder_name}")
async def get_measurement_files(folder_name: str) -> schemas.MeasurementFileListResponse:
    """指定フォルダの計測ファイル一覧を取得"""
    measurements_path = (settings.DATA_FOLDER / folder_name / "main_tunnel" / 
                        "CN_measurement_data" / "measurements_A")
    
    if not measurements_path.exists():
        raise HTTPException(status_code=404, detail=f"Measurements folder not found: {measurements_path}")
    
    csv_files = sorted([f.name for f in measurements_path.glob("*.csv")])
    return schemas.MeasurementFileListResponse(files=csv_files)

@router.post("/analyze-whole")
async def analyze_whole_displacement(request: schemas.WholeAnalysisRequest) -> schemas.WholeAnalysisResponse:
    """
    全体分析を実行（GUIのanalyze_displacement機能を模倣）
    """
    try:
        input_folder = (settings.DATA_FOLDER / request.folder_name / "main_tunnel" / 
                       "CN_measurement_data")
        output_folder = Path(OUTPUT_FOLDER)
        output_folder.mkdir(exist_ok=True)
        
        # モデルパス設定
        model_paths = {
            "final_value_prediction_model": [
                output_folder / "model_final_settlement.pkl",
                output_folder / "model_final_convergence.pkl"
            ],
            "prediction_model": [
                output_folder / "model_settlement.pkl", 
                output_folder / "model_convergence.pkl"
            ]
        }
        
        # 選択されたモデルを取得（毎回新しいインスタンス）
        models = get_models()
        model = models.get(request.model_name)
        if model is None:
            raise HTTPException(status_code=400, detail=f"Unknown model: {request.model_name}")
        
        # Streamlitと完全に同じ実装を使用
        try:
            logger.info(f"Starting complete analysis with model: {request.model_name}")
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"Input folder: {input_folder}")
            logger.info(f"Output folder: {output_folder}")
            logger.info(f"Model type: {type(model)}")
            
            # analyze_displacement関数を直接呼び出し（Streamlitと同じ）
            result = analyze_displacement(
                str(input_folder),
                str(output_folder), 
                model_paths,
                model,
                request.max_distance_from_face,
                td=request.td
            )
            
            # 戻り値の処理（新しい形式に対応）
            if isinstance(result, tuple):
                if len(result) == 4:
                    df_all, training_metrics, scatter_data, feature_importance_from_analysis = result
                elif len(result) == 3:
                    df_all, training_metrics, scatter_data = result
                    feature_importance_from_analysis = {}
                elif len(result) == 2:
                    df_all, training_metrics = result
                    scatter_data = {}
                    feature_importance_from_analysis = {}
                else:
                    df_all = result[0] if result else None
                    training_metrics = {}
                    scatter_data = {}
                    feature_importance_from_analysis = {}
            else:
                df_all = result
                training_metrics = {}
                scatter_data = {}
                feature_importance_from_analysis = {}
            
            logger.info(f"Complete analysis finished. Training metrics: {list(training_metrics.keys()) if training_metrics else 'None'}")
            logger.info(f"Result type: {type(result)}, length: {len(result) if isinstance(result, tuple) else 'N/A'}")
            logger.info(f"Scatter data keys: {list(scatter_data.keys()) if scatter_data else 'None'}")
            if scatter_data:
                for category in ['settlement', 'convergence']:
                    if category in scatter_data:
                        train_len = len(scatter_data[category].get('train_actual', []))
                        logger.info(f"Scatter data {category} train length: {train_len}")
            
            # 新しい特徴量重要度データを使用
            feature_importance = {
                'available': bool(feature_importance_from_analysis),
                'model_type': type(model).__name__ if model else 'Unknown',
                'features_by_category': feature_importance_from_analysis
            }
            
            # モデルファイルの保存状況を確認
            required_files = ['model_final_settlement.pkl', 'model_final_convergence.pkl', 
                            'model_settlement.pkl', 'model_convergence.pkl']
            model_files_saved = all(os.path.exists(os.path.join(output_folder, f)) for f in required_files)
            
            return schemas.WholeAnalysisResponse(
                status="completed",
                message="Analysis completed successfully",
                training_metrics=training_metrics,
                scatter_data=scatter_data,
                feature_importance=feature_importance,
                model_files_saved=model_files_saved
            )
            
        except Exception as e:
            logger.error(f"Complete analysis failed: {e}")
            import traceback
            traceback.print_exc()
            
            # エラーが発生してもメトリクスだけでも取得を試みる
            try:
                extracted_metrics = extract_metrics_from_output(str(output_folder))
                if extracted_metrics:
                    return schemas.WholeAnalysisResponse(
                        status="partial_success",
                        message=f"Analysis partially completed. Error: {str(e)}",
                        training_metrics=extracted_metrics,
                        scatter_data={},
                        feature_importance={},
                        model_files_saved=False
                    )
            except:
                pass
            
            return schemas.WholeAnalysisResponse(
                status="failed",
                message=f"Analysis failed: {str(e)}",
                training_metrics={},
                scatter_data={},
                feature_importance={},
                model_files_saved=False
            )
        
    except Exception as e:
        logger.error(f"Error in whole analysis: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # エラーが発生してもメトリクスだけでも取得を試みる
        try:
            extracted_metrics = extract_metrics_from_output(str(output_folder))
            if extracted_metrics:
                return schemas.WholeAnalysisResponse(
                    status="partial_success",
                    message=f"Analysis partially completed. Error: {str(e)}",
                    training_metrics=extracted_metrics,
                    scatter_data={},
                    feature_importance={},
                    model_files_saved=False
                )
        except Exception as inner_e:
            logger.error(f"Error extracting metrics: {inner_e}")
        
        return schemas.WholeAnalysisResponse(
            status="failed",
            message=f"Analysis failed: {str(e)}",
            training_metrics={},
            scatter_data={},
            feature_importance={},
            model_files_saved=False
        )

@router.post("/analyze-local")
async def analyze_local_displacement(request: schemas.LocalAnalysisRequest) -> schemas.LocalAnalysisResponse:
    """
    局所分析を実行（GUIのlocal analysis機能を模倣）
    """
    try:
        input_folder = (settings.DATA_FOLDER / request.folder_name / "main_tunnel" / 
                       "CN_measurement_data")
        a_measure_path = input_folder / "measurements_A" / request.ameasure_file
        
        if not a_measure_path.exists():
            raise HTTPException(status_code=404, detail=f"Measurement file not found: {request.ameasure_file}")
        
        # 予測実行
        df_all, settlements, convergences = simulate_displacement(
            input_folder, a_measure_path, request.max_distance_from_face
        )
        
        cycle_no = float(os.path.basename(a_measure_path).split('_')[2].split('.')[0])
        td = float(df_all[TD_NO].values[0]) if TD_NO in df_all.columns else 0.0
        
        # 予測チャートを生成
        settlement_chart = draw_local_prediction_chart(
            df_all[DISTANCE_FROM_FACE].tolist(),
            df_all[settlements],
            df_all[DISTANCE_FROM_FACE].tolist(),
            df_all[[l + "_prediction" for l in settlements]],
            f"最終沈下量予測 for Cycle {cycle_no} (TD: {td})"
        )
        
        convergence_chart = draw_local_prediction_chart(
            df_all[DISTANCE_FROM_FACE].tolist(),
            df_all[convergences],
            df_all[DISTANCE_FROM_FACE].tolist(),
            df_all[[l + "_prediction" for l in convergences]],
            f"最終変位量予測 for Cycle {cycle_no} (TD: {td})"
        )
        
        # シミュレーション実行（日次進行がある場合）
        simulation_charts = {}
        if request.daily_advance and request.distance_from_face:
            df_all_simulated, _, _ = simulate_displacement(
                input_folder, a_measure_path, request.max_distance_from_face,
                request.daily_advance, request.distance_from_face, recursive=True
            )
            
            # シミュレーション結果をCSVで保存
            output_path = Path(OUTPUT_FOLDER) / f"{request.folder_name}_{request.ameasure_file}.csv"
            output_path.parent.mkdir(exist_ok=True)
            df_all_simulated.to_csv(output_path, index=False)
            
            # シミュレーションチャート生成
            simulation_charts['settlement'] = draw_local_prediction_chart(
                df_all[DISTANCE_FROM_FACE].tolist(),
                df_all[settlements],
                df_all_simulated[DISTANCE_FROM_FACE].tolist(),
                df_all_simulated[[l + "_prediction" for l in settlements]],
                f"最終沈下量予測 for Cycle {cycle_no} (TD: {td})"
            )
            
            simulation_charts['convergence'] = draw_local_prediction_chart(
                df_all[DISTANCE_FROM_FACE].tolist(),
                df_all[convergences],
                df_all_simulated[DISTANCE_FROM_FACE].tolist(),
                df_all_simulated[[l + "_prediction" for l in convergences]],
                f"最終変位量予測 for Cycle {cycle_no} (TD: {td})"
            )
        
        # 予測データテーブル作成
        prediction_data = []
        prediction_columns = [DISTANCE_FROM_FACE] + [l + "_prediction" for l in convergences + settlements]
        available_columns = [col for col in prediction_columns if col in df_all.columns]
        
        for _, row in df_all[available_columns].iterrows():
            prediction_data.append({col: float(row[col]) if pd.notna(row[col]) else None for col in available_columns})
        
        return schemas.LocalAnalysisResponse(
            cycle_no=int(cycle_no),
            td=td,
            prediction_charts={
                'settlement': settlement_chart,
                'convergence': convergence_chart
            },
            simulation_charts=simulation_charts,
            prediction_data=prediction_data,
            csv_path=str(output_path) if request.daily_advance and request.distance_from_face else None
        )
        
    except Exception as e:
        logger.error(f"Error in local analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def get_available_models() -> schemas.ModelListResponse:
    """利用可能なモデル一覧を取得"""
    models = get_models()
    model_list = list(models.keys())
    return schemas.ModelListResponse(models=model_list)

@router.get("/output/{filename}")
async def get_output_file(filename: str):
    """アウトプットファイル（画像、CSV等）を取得"""
    file_path = Path(OUTPUT_FOLDER) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    return FileResponse(file_path)