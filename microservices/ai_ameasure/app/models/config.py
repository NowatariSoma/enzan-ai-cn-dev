import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class ModelConfig:
    """
    モデル設定を管理するクラス
    YAMLまたはJSON形式の設定ファイルをサポート
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込む"""
        # 環境変数から設定ファイルパスを取得
        if not self.config_path:
            env_config_path = os.getenv("MODEL_CONFIG_PATH")
            if env_config_path:
                self.config_path = Path(env_config_path)

        # デフォルト設定
        default_config = self._get_default_config()

        if not self.config_path or not self.config_path.exists():
            return default_config

        # ファイル拡張子に応じて読み込み
        if self.config_path.suffix in [".yaml", ".yml"]:
            with open(self.config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f)
        elif self.config_path.suffix == ".json":
            with open(self.config_path, "r", encoding="utf-8") as f:
                file_config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {self.config_path.suffix}")

        # デフォルト設定とマージ
        default_config.update(file_config)
        return default_config

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を取得"""
        return {
            "models": {
                "settlement": {
                    "type": "random_forest",
                    "params": {"n_estimators": 100, "random_state": 42},
                    "save_path": "./output/model_settlement.pkl",
                },
                "convergence": {
                    "type": "random_forest",
                    "params": {"n_estimators": 100, "random_state": 42},
                    "save_path": "./output/model_convergence.pkl",
                },
                "final_settlement": {
                    "type": "hist_gradient_boosting",
                    "params": {"random_state": 42},
                    "save_path": "./output/model_final_settlement.pkl",
                },
                "final_convergence": {
                    "type": "hist_gradient_boosting",
                    "params": {"random_state": 42},
                    "save_path": "./output/model_final_convergence.pkl",
                },
            },
            "available_model_types": [
                "random_forest",
                "linear_regression",
                "svr",
                "hist_gradient_boosting",
                "mlp",
            ],
            "output_dir": "./output",
            "model_version": "1.0.0",
        }

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        特定のモデルの設定を取得

        Args:
            model_name: モデル名

        Returns:
            モデル設定
        """
        if model_name not in self.config["models"]:
            raise ValueError(f"Model {model_name} not found in configuration")

        return self.config["models"][model_name]

    def set_model_config(self, model_name: str, config: Dict[str, Any]) -> None:
        """
        特定のモデルの設定を更新

        Args:
            model_name: モデル名
            config: 新しい設定
        """
        self.config["models"][model_name] = config

    def save_config(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        設定をファイルに保存

        Args:
            path: 保存先パス（指定しない場合は元のパスに保存）
        """
        save_path = Path(path) if path else self.config_path
        if not save_path:
            raise ValueError("No path specified for saving configuration")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.suffix in [".yaml", ".yml"]:
            with open(save_path, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        elif save_path.suffix == ".json":
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported config file format: {save_path.suffix}")

    def get_model_type(self, model_name: str) -> str:
        """モデルタイプを取得"""
        return self.get_model_config(model_name)["type"]

    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """モデルパラメータを取得"""
        return self.get_model_config(model_name).get("params", {})

    def get_model_save_path(self, model_name: str) -> str:
        """モデル保存パスを取得"""
        return self.get_model_config(model_name)["save_path"]

    def get_available_model_types(self) -> list[str]:
        """利用可能なモデルタイプのリストを取得"""
        return self.config.get("available_model_types", [])
