"""
機械学習予測・シミュレーションエンジン（ai_ameasureのオリジナルアルゴリズム統合版）
"""

import logging
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 全ての警告を抑制（高精度処理のパフォーマンス向上）
warnings.filterwarnings('ignore')

import joblib
import matplotlib
import numpy as np
import pandas as pd

# パンダス警告を個別に抑制
pd.options.mode.chained_assignment = None

import seaborn as sns
from app.core.config import settings
from app.core.csv_loader import CSVDataLoader
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# 元のai_ameasureのパスを動的に取得（Docker環境対応）
current_file_path = Path(__file__).resolve()
original_ai_ameasure_path = None

# Docker環境とローカル環境の複数パターンを試行
possible_paths = [
    Path("/app/ai_ameasure"),  # Docker環境
    Path("/home/nowatari/repos/enzan-ai-cn-dev/ai_ameasure"),  # ローカル環境
    Path(__file__).parent.parent.parent.parent / "ai_ameasure",  # 相対パス
]

for path in possible_paths:
    if path.exists():
        original_ai_ameasure_path = path
        logger.info(f"Found ai_ameasure directory at: {original_ai_ameasure_path}")
        break

if original_ai_ameasure_path is None:
    logger.error(f"Cannot find ai_ameasure directory in any of these locations: {possible_paths}")
    # フォールバック: 現在のmicroservices内の実装を使用
    logger.warning("Using fallback implementation from microservices")
    analyze_displacement = None
    create_dataset = None
    generate_additional_info_df = None
    generate_dataframes = None
else:
    try:
        sys.path.insert(0, str(original_ai_ameasure_path))
        from app.displacement_temporal_spacial_analysis import analyze_displacement
        from app.displacement_temporal_spacial_analysis import (
            create_dataset,
            generate_additional_info_df, 
            generate_dataframes,
        )
        logger.info(f"Successfully imported from ai_ameasure: {original_ai_ameasure_path}")
    except ImportError as e:
        logger.error(f"Failed to import from ai_ameasure: {e}")
        # フォールバック設定
        analyze_displacement = None
        create_dataset = None
        generate_additional_info_df = None
        generate_dataframes = None

try:
    from app.models.manager import ModelManager
except ImportError:
    logger.warning("ModelManager not available, using fallback")
    ModelManager = None
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib

logger = logging.getLogger(__name__)


class PredictionEngine:
    """機械学習予測・シミュレーションエンジン（オリジナルアルゴリズム統合版）"""

    def __init__(self):
        if ModelManager:
            try:
                self.model_manager = ModelManager(
                    settings.DATA_FOLDER.parent / "microservices" / "ai_ameasure" / "config" / "models.yaml"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize ModelManager: {e}")
                self.model_manager = None
        else:
            self.model_manager = None
            
        self.csv_loader = CSVDataLoader()

        # オリジナルの特徴量・ターゲット列定義（高度な特徴エンジニアリング対応）
        self.feature_columns = ["TD(m)", "切羽TD", "実TD", "ｻｲｸﾙNo"]
        self.settlement_columns = [
            "沈下量1",
            "沈下量2",
            "沈下量3",
            "沈下量4",
            "沈下量5",
            "沈下量6",
            "沈下量7",
        ]
        self.convergence_columns = [
            "変位量A",
            "変位量B",
            "変位量C",
            "変位量D",
            "変位量E",
            "変位量F",
            "変位量G",
            "変位量H",
            "変位量I",
        ]

    def load_training_data(self, folder_name: str = "01-hokkaido-akan") -> pd.DataFrame:
        """
        訓練用データを読み込む (dataframe_cacheを使用)

        Args:
            folder_name: データフォルダ名

        Returns:
            DataFrame: 訓練用データ
        """
        from app.core.dataframe_cache import get_dataframe_cache
        
        # キャッシュからデータを取得
        cache = get_dataframe_cache()
        cached_data = cache.get_cached_data(folder_name, 100.0)  # デフォルト最大距離100m
        
        if not cached_data:
            raise ValueError(f"No training data found for {folder_name} in cache")
            
        df = cached_data["df_all"]

        if df.empty:
            raise ValueError(f"Empty training data for {folder_name}")

        # 数値列のみを抽出
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_columns].copy()

        # 欠損値を除去
        df_clean = df_numeric.dropna()

        logger.info(f"Loaded training data from cache: {df_clean.shape}")
        return df_clean

    def prepare_features_targets(
        self, df: pd.DataFrame, target_type: str = "settlement"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        特徴量とターゲットを準備

        Args:
            df: データフレーム
            target_type: ターゲットタイプ ("settlement" or "convergence")

        Returns:
            Tuple[DataFrame, DataFrame]: (特徴量, ターゲット)
        """
        # 利用可能な特徴量を選択
        available_features = [col for col in self.feature_columns if col in df.columns]
        if not available_features:
            # フォールバック特徴量: TD(m)列を優先して選択
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            # TD列を最優先
            if "TD(m)" in numeric_cols:
                available_features = ["TD(m)"]
            else:
                available_features = list(numeric_cols)[:1]  # 最初の1列のみ使用

        # ターゲット列を選択
        if target_type == "settlement":
            target_columns = [col for col in self.settlement_columns if col in df.columns]
        else:  # convergence
            target_columns = [col for col in self.convergence_columns if col in df.columns]

        if not target_columns:
            raise ValueError(f"No target columns found for {target_type}")

        X = df[available_features].copy()
        y = df[target_columns].copy()

        # NaNを除去
        valid_indices = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
        X = X[valid_indices]
        y = y[valid_indices]

        logger.info(f"Features: {list(X.columns)}")
        logger.info(f"Targets: {list(y.columns)}")
        logger.info(f"Training samples: {len(X)}")

        return X, y

    def train_model(
        self,
        model_name: str,
        folder_name: str = "01-hokkaido-akan",
        max_distance_from_face: float = 100.0,
        td: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        オリジナルアルゴリズムによるモデル訓練

        Args:
            model_name: モデル名
            folder_name: データフォルダ名
            max_distance_from_face: 切羽からの最大距離
            td: TD値でのフィルタ

        Returns:
            Dict: 訓練結果
        """
        start_time = time.time()

        try:
            # データフォルダパスを環境変数から取得（Docker対応）
            input_folder = settings.DATA_FOLDER / folder_name / "main_tunnel" / "CN_measurement_data"
            output_path = Path("./output")
            output_path.mkdir(exist_ok=True)

            # モデルパス設定
            model_paths = {
                "final_value_prediction_model": [
                    output_path / "model_final_settlement.pkl",
                    output_path / "model_final_convergence.pkl",
                ],
                "prediction_model": [
                    output_path / "model_settlement.pkl",
                    output_path / "model_convergence.pkl",
                ],
            }

            # オリジナルと完全に同じモデルを使用
            from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.svm import SVR
            from sklearn.neural_network import MLPRegressor
            
            original_models = {
                "random_forest": RandomForestRegressor(random_state=42),
                "linear_regression": LinearRegression(),
                "svr": SVR(kernel='linear', C=1.0, epsilon=0.2),
                "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42),
                "mlp": MLPRegressor(hidden_layer_sizes=(100, 100, 50), max_iter=1000, random_state=42),
            }
            
            # オリジナルと完全に同じモデルインスタンスを使用
            model_instance = original_models.get(model_name, RandomForestRegressor(random_state=42))

            # オリジナルの高精度学習処理を実行（フォールバック無し）
            if analyze_displacement is None:
                raise ImportError("High-precision analyze_displacement function is not available. Please check ai_ameasure integration.")
                
            logger.info("Using original ai_ameasure high-precision analyze_displacement function")
            df_all, training_metrics, scatter_data = analyze_displacement(
                str(input_folder),
                str(output_path),
                model_paths,
                model_instance,
                max_distance_from_face,
                td=td,
            )

            # 訓練結果の統計を計算
            processing_time = time.time() - start_time

            result = {
                "model_name": model_name,
                "folder_name": folder_name,
                "max_distance_from_face": max_distance_from_face,
                "td": td,
                "training_samples": len(df_all),
                "processing_time": processing_time,
                "training_metrics": training_metrics,  # 実際の学習メトリクスを含める
                "scatter_data": scatter_data,  # 散布図データを含める
                "models_saved": {
                    "settlement": str(model_paths["prediction_model"][0]),
                    "convergence": str(model_paths["prediction_model"][1]),
                    "final_settlement": str(model_paths["final_value_prediction_model"][0]),
                    "final_convergence": str(model_paths["final_value_prediction_model"][1]),
                },
                "status": "success",
            }

            logger.info(
                f"Model training completed for {folder_name}. "
                f"Processed {len(df_all)} samples in {processing_time:.2f}s"
            )
            return result

        except Exception as e:
            logger.error(f"Error training model {model_name}: {e}")
            raise

    # フォールバック実装は削除 - 高精度アルゴリズムのみ使用

    def predict(
        self, model_name: str, features: Dict[str, float], folder_name: str = "01-hokkaido-akan"
    ) -> Dict[str, Any]:
        """
        訓練済みモデルによる予測実行

        Args:
            model_name: モデル名
            features: 特徴量
            folder_name: データフォルダ名

        Returns:
            Dict: 予測結果
        """
        try:
            # 保存済みモデルを直接読み込み
            model_path = Path("./output") / f"model_{model_name}.pkl"

            if not model_path.exists():
                # モデルが存在しない場合は訓練
                logger.warning(f"Model {model_name} not found, training new model")
                self.train_model("random_forest", folder_name)

            # joblibでモデルを直接読み込み
            model = joblib.load(str(model_path))

            # 特徴量をDataFrameに変換（オリジナルの特徴量に合わせる）
            feature_df = pd.DataFrame([features])

            # 予測実行
            prediction = model.predict(feature_df)

            # 結果を整形
            if prediction.ndim > 1 and prediction.shape[1] > 1:
                pred_values = prediction[0].tolist()
            else:
                pred_values = float(prediction[0]) if hasattr(prediction[0], "item") else float(prediction[0])

            result = {
                "model_name": model_name,
                "prediction": pred_values,
                "features": features,
                "model_path": str(model_path),
            }

            return result

        except Exception as e:
            logger.error(f"Error during prediction with {model_name}: {e}")
            raise

    def simulate_displacement(
        self,
        folder_name: str,
        daily_advance: float = 2.0,
        distance_from_face: float = 50.0,
        max_distance: float = 200.0,
        prediction_days: int = 30,
        recursive: bool = True,
        use_models: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """
        変位・沈下シミュレーション

        Args:
            folder_name: データフォルダ名
            daily_advance: 日進量
            distance_from_face: 現在の切羽からの距離
            max_distance: 最大距離
            prediction_days: 予測日数
            recursive: 再帰的予測
            use_models: 使用するモデル

        Returns:
            Dict: シミュレーション結果
        """
        start_time = time.time()

        if use_models is None:
            use_models = {
                "settlement": "settlement",
                "convergence": "convergence",
                "final_settlement": "final_settlement",
                "final_convergence": "final_convergence",
            }

        try:
            # 基準データを読み込み
            df = self.load_training_data(folder_name)

            simulation_data = []
            current_distance = distance_from_face
            current_td = 0.0

            for day in range(prediction_days):
                # TD位置を更新
                current_td += daily_advance

                # 基本特徴量を設定
                features = {
                    "TD(m)": current_td,
                    "切羽TD": current_td,
                    "実TD": current_td,
                    "ｻｲｸﾙNo": day + 1,
                }

                # 沈下量予測
                settlement_pred = {}
                try:
                    settlement_result = self.predict(
                        use_models["settlement"], features, folder_name
                    )
                    settlement_pred = {
                        "series1": (
                            settlement_result["prediction"][0]
                            if isinstance(settlement_result["prediction"], list)
                            else settlement_result["prediction"]
                        ),
                        "series2": (
                            settlement_result["prediction"][1]
                            if isinstance(settlement_result["prediction"], list)
                            and len(settlement_result["prediction"]) > 1
                            else 0.0
                        ),
                        "series3": (
                            settlement_result["prediction"][2]
                            if isinstance(settlement_result["prediction"], list)
                            and len(settlement_result["prediction"]) > 2
                            else 0.0
                        ),
                    }
                except Exception as e:
                    logger.warning(f"Settlement prediction failed: {e}")
                    settlement_pred = {"series1": 0.0, "series2": 0.0, "series3": 0.0}

                # 変位量予測
                convergence_pred = {}
                try:
                    convergence_result = self.predict(
                        use_models["convergence"], features, folder_name
                    )
                    convergence_pred = {
                        "seriesA": (
                            convergence_result["prediction"][0]
                            if isinstance(convergence_result["prediction"], list)
                            else convergence_result["prediction"]
                        ),
                        "seriesB": (
                            convergence_result["prediction"][1]
                            if isinstance(convergence_result["prediction"], list)
                            and len(convergence_result["prediction"]) > 1
                            else 0.0
                        ),
                        "seriesC": (
                            convergence_result["prediction"][2]
                            if isinstance(convergence_result["prediction"], list)
                            and len(convergence_result["prediction"]) > 2
                            else 0.0
                        ),
                    }
                except Exception as e:
                    logger.warning(f"Convergence prediction failed: {e}")
                    convergence_pred = {"seriesA": 0.0, "seriesB": 0.0, "seriesC": 0.0}

                # 最終予測（オプション）
                final_settlement = None
                final_convergence = None

                try:
                    if "final_settlement" in use_models:
                        final_sett_result = self.predict(
                            use_models["final_settlement"], features, folder_name
                        )
                        final_settlement = (
                            final_sett_result["prediction"]
                            if isinstance(final_sett_result["prediction"], (int, float))
                            else final_sett_result["prediction"][0]
                        )
                except:
                    pass

                try:
                    if "final_convergence" in use_models:
                        final_conv_result = self.predict(
                            use_models["final_convergence"], features, folder_name
                        )
                        final_convergence = (
                            final_conv_result["prediction"]
                            if isinstance(final_conv_result["prediction"], (int, float))
                            else final_conv_result["prediction"][0]
                        )
                except:
                    pass

                data_point = {
                    "day": day + 1,
                    "distance_from_face": current_distance,
                    "td_position": current_td,
                    "settlement_prediction": settlement_pred,
                    "convergence_prediction": convergence_pred,
                    "final_settlement": final_settlement,
                    "final_convergence": final_convergence,
                }

                simulation_data.append(data_point)

                # 再帰的予測の場合、距離を更新
                if recursive:
                    current_distance = max(0, current_distance - daily_advance)

            # サマリー統計
            summary = {
                "total_days": prediction_days,
                "total_advance": prediction_days * daily_advance,
                "final_td": current_td,
                "max_settlement": max(
                    [max(d["settlement_prediction"].values()) for d in simulation_data]
                ),
                "max_convergence": max(
                    [max(d["convergence_prediction"].values()) for d in simulation_data]
                ),
            }

            processing_time = time.time() - start_time

            result = {
                "folder_name": folder_name,
                "simulation_params": {
                    "daily_advance": daily_advance,
                    "distance_from_face": distance_from_face,
                    "max_distance": max_distance,
                    "prediction_days": prediction_days,
                    "recursive": recursive,
                    "use_models": use_models,
                },
                "data_points": simulation_data,
                "summary": summary,
                "processing_time": processing_time,
            }

            logger.info(f"Simulation completed for {folder_name}: {prediction_days} days")
            return result

        except Exception as e:
            logger.error(f"Error during simulation: {e}")
            raise

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        モデル情報を取得

        Args:
            model_name: モデル名

        Returns:
            Dict: モデル情報
        """
        try:
            model_config = self.model_manager.config.get_model_config(model_name)
            save_path = self.model_manager.config.get_model_save_path(model_name)

            # モデルが訓練済みかチェック
            is_trained = False
            try:
                if Path(save_path).exists():
                    is_trained = True
            except:
                pass

            return {
                "name": model_name,
                "type": model_config.get("type", "unknown"),
                "params": model_config.get("params", {}),
                "is_trained": is_trained,
                "save_path": str(save_path),
            }

        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            raise

    def list_models(self) -> Dict[str, Any]:
        """
        利用可能なモデル一覧を取得

        Returns:
            Dict: モデル一覧情報
        """
        try:
            model_names = ["settlement", "convergence", "final_settlement", "final_convergence"]
            models = []

            for name in model_names:
                try:
                    info = self.get_model_info(name)
                    models.append(info)
                except:
                    # 設定にないモデルはスキップ
                    pass

            available_types = self.model_manager.get_available_model_types()

            return {"models": models, "available_types": available_types}

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise
