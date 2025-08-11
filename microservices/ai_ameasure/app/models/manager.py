from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from .base import BaseModel
from .config import ModelConfig
from .factory import ModelFactory


class ModelManager:
    """
    モデルを統合的に管理するクラス
    設定ファイルに基づいてモデルの作成、学習、保存、読み込みを行う
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Args:
            config_path: 設定ファイルのパス
        """
        self.config = ModelConfig(config_path)
        self.models: Dict[str, BaseModel] = {}

    def create_model(self, model_name: str) -> BaseModel:
        """
        設定に基づいてモデルを作成

        Args:
            model_name: モデル名

        Returns:
            作成されたモデル
        """
        model_type = self.config.get_model_type(model_name)
        model_params = self.config.get_model_params(model_name)

        model = ModelFactory.create_model(model_type, model_params)
        self.models[model_name] = model

        return model

    def get_model(self, model_name: str) -> BaseModel:
        """
        モデルを取得（存在しない場合は作成）

        Args:
            model_name: モデル名

        Returns:
            モデルインスタンス
        """
        if model_name not in self.models:
            self.create_model(model_name)

        return self.models[model_name]

    def train_model(
        self,
        model_name: str,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs,
    ) -> BaseModel:
        """
        モデルを学習

        Args:
            model_name: モデル名
            X: 特徴量
            y: ターゲット
            **kwargs: 追加の学習パラメータ

        Returns:
            学習済みモデル
        """
        model = self.get_model(model_name)
        model.fit(X, y, **kwargs)

        return model

    def predict(self, model_name: str, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> np.ndarray:
        """
        予測を実行

        Args:
            model_name: モデル名
            X: 特徴量
            **kwargs: 追加の予測パラメータ

        Returns:
            予測結果
        """
        model = self.get_model(model_name)
        return model.predict(X, **kwargs)

    def save_model(self, model_name: str, path: Optional[Union[str, Path]] = None) -> None:
        """
        モデルを保存

        Args:
            model_name: モデル名
            path: 保存先パス（指定しない場合は設定ファイルのパスを使用）
        """
        model = self.get_model(model_name)

        if path is None:
            path = self.config.get_model_save_path(model_name)

        model.save(path)

    def load_model(self, model_name: str, path: Optional[Union[str, Path]] = None) -> BaseModel:
        """
        モデルを読み込み

        Args:
            model_name: モデル名
            path: 読み込み元パス（指定しない場合は設定ファイルのパスを使用）

        Returns:
            読み込まれたモデル
        """
        if path is None:
            path = self.config.get_model_save_path(model_name)

        # モデルがまだ作成されていない場合は作成
        if model_name not in self.models:
            self.create_model(model_name)

        model = self.models[model_name]
        model.load(path)

        return model

    def update_model_type(
        self, model_name: str, new_type: str, new_params: Optional[Dict[str, Any]] = None
    ) -> BaseModel:
        """
        モデルタイプを変更

        Args:
            model_name: モデル名
            new_type: 新しいモデルタイプ
            new_params: 新しいモデルパラメータ

        Returns:
            新しいモデルインスタンス
        """
        # 設定を更新
        model_config = self.config.get_model_config(model_name)
        model_config["type"] = new_type
        if new_params:
            model_config["params"] = new_params

        self.config.set_model_config(model_name, model_config)

        # 新しいモデルを作成
        model = ModelFactory.create_model(new_type, new_params)
        self.models[model_name] = model

        return model

    def get_available_model_types(self) -> list[str]:
        """利用可能なモデルタイプのリストを取得"""
        return ModelFactory.get_available_models()

    def save_config(self, path: Optional[Union[str, Path]] = None) -> None:
        """設定を保存"""
        self.config.save_config(path)
