"""
機械学習予測・シミュレーションエンジン
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from app.core.config import settings
from app.core.csv_loader import CSVDataLoader
from app.models.manager import ModelManager
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class PredictionEngine:
    """機械学習予測・シミュレーションエンジン"""

    def __init__(self):
        self.model_manager = ModelManager(
            settings.DATA_FOLDER.parent / "microservices" / "ai_ameasure" / "config" / "models.yaml"
        )
        self.csv_loader = CSVDataLoader()

        # 特徴量・ターゲット列の定義
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
        訓練用データを読み込む

        Args:
            folder_name: データフォルダ名

        Returns:
            DataFrame: 訓練用データ
        """
        df = self.csv_loader.load_all_measurement_data(settings.DATA_FOLDER, folder_name)

        if df.empty:
            raise ValueError(f"No training data found for {folder_name}")

        # 数値列のみを抽出
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_columns].copy()

        # 欠損値を除去
        df_clean = df_numeric.dropna()

        logger.info(f"Loaded training data: {df_clean.shape}")
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
        target_type: str = "settlement",
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """
        モデルを訓練

        Args:
            model_name: モデル名
            folder_name: データフォルダ名
            target_type: ターゲットタイプ
            test_size: テストデータの割合

        Returns:
            Dict: 訓練結果
        """
        start_time = time.time()

        try:
            # データ読み込み
            df = self.load_training_data(folder_name)
            X, y = self.prepare_features_targets(df, target_type)

            if len(X) < 10:
                raise ValueError(f"Insufficient training data: {len(X)} samples")

            # データ分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # モデル訓練
            model = self.model_manager.train_model(model_name, X_train, y_train)

            # 評価
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            # 多出力の場合は平均スコアを計算
            if y_train.shape[1] > 1:
                train_score = np.mean(
                    [
                        r2_score(y_train.iloc[:, i], train_pred[:, i])
                        for i in range(y_train.shape[1])
                    ]
                )
                test_score = np.mean(
                    [r2_score(y_test.iloc[:, i], test_pred[:, i]) for i in range(y_test.shape[1])]
                )
            else:
                train_score = r2_score(y_train, train_pred)
                test_score = r2_score(y_test, test_pred)

            # 特徴量重要度（利用可能な場合）
            feature_importance = None
            try:
                if hasattr(model.model, "feature_importances_"):
                    importances = model.model.feature_importances_
                    feature_importance = [
                        {"feature": col, "importance": float(imp)}
                        for col, imp in zip(X.columns, importances)
                    ]
                    feature_importance.sort(key=lambda x: x["importance"], reverse=True)
            except:
                pass

            # モデル保存
            self.model_manager.save_model(model_name)

            processing_time = time.time() - start_time

            result = {
                "model_name": model_name,
                "training_score": float(train_score),
                "validation_score": float(test_score),
                "test_score": float(test_score),
                "feature_importance": feature_importance,
                "training_samples": len(X_train),
                "validation_samples": len(X_test),
                "processing_time": processing_time,
            }

            logger.info(f"Model {model_name} trained successfully. Score: {test_score:.3f}")
            return result

        except Exception as e:
            logger.error(f"Error training model {model_name}: {e}")
            raise

    def predict(
        self, model_name: str, features: Dict[str, float], folder_name: str = "01-hokkaido-akan"
    ) -> Dict[str, Any]:
        """
        予測を実行

        Args:
            model_name: モデル名
            features: 特徴量
            folder_name: データフォルダ名（参考用）

        Returns:
            Dict: 予測結果
        """
        try:
            # モデル読み込み（必要に応じて）
            try:
                model = self.model_manager.load_model(model_name)
            except:
                # モデルが存在しない場合は訓練
                logger.warning(f"Model {model_name} not found, training new model")
                self.train_model(model_name, folder_name)
                model = self.model_manager.get_model(model_name)

            # 訓練時に使用した特徴量のみを使用
            # まず訓練データを読み込んで特徴量を確認
            df = self.load_training_data(folder_name)
            target_type = "settlement" if "settlement" in model_name else "convergence"
            X_train, _ = self.prepare_features_targets(df, target_type)
            training_features = list(X_train.columns)

            # 入力特徴量を訓練時の特徴量に合わせる
            aligned_features = {}
            for col in training_features:
                if col in features:
                    aligned_features[col] = features[col]
                else:
                    # デフォルト値を設定
                    aligned_features[col] = 0.0

            # 特徴量をDataFrameに変換
            feature_df = pd.DataFrame([aligned_features])

            # 予測実行
            prediction = model.predict(feature_df)

            # 結果を整形
            if prediction.ndim > 1 and prediction.shape[1] > 1:
                pred_values = prediction[0].tolist()
            else:
                pred_values = (
                    float(prediction[0]) if hasattr(prediction[0], "item") else float(prediction[0])
                )

            result = {
                "model_name": model_name,
                "prediction": pred_values,
                "features": features,
                "confidence": None,  # 実装可能であれば追加
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
