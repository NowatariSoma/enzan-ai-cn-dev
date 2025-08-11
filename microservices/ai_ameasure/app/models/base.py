from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    機械学習モデルの基底クラス
    全てのモデルはこのクラスを継承して実装する
    """

    def __init__(self, model_name: str, model_params: Optional[Dict[str, Any]] = None):
        """
        Args:
            model_name: モデルの名前
            model_params: モデルのパラメータ
        """
        self.model_name = model_name
        self.model_params = model_params or {}
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def build_model(self) -> Any:
        """モデルのインスタンスを構築する"""
        pass

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], **kwargs
    ) -> "BaseModel":
        """
        モデルを学習する

        Args:
            X: 特徴量
            y: ターゲット
            **kwargs: 追加の学習パラメータ

        Returns:
            self
        """
        if self.model is None:
            self.model = self.build_model()

        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> np.ndarray:
        """
        予測を行う

        Args:
            X: 特徴量
            **kwargs: 追加の予測パラメータ

        Returns:
            予測結果
        """
        if not self.is_fitted:
            raise ValueError(f"Model {self.model_name} is not fitted yet. Call fit() first.")

        return self.model.predict(X, **kwargs)

    def save(self, path: Union[str, Path]) -> None:
        """
        モデルを保存する

        Args:
            path: 保存先のパス
        """
        if not self.is_fitted:
            raise ValueError(f"Model {self.model_name} is not fitted yet. Cannot save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "model_name": self.model_name,
            "model_params": self.model_params,
            "is_fitted": self.is_fitted,
        }

        joblib.dump(model_data, path)

    def load(self, path: Union[str, Path]) -> "BaseModel":
        """
        モデルを読み込む

        Args:
            path: 読み込み元のパス

        Returns:
            self
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model_data = joblib.load(path)

        self.model = model_data["model"]
        self.model_name = model_data["model_name"]
        self.model_params = model_data["model_params"]
        self.is_fitted = model_data["is_fitted"]

        return self

    def get_params(self) -> Dict[str, Any]:
        """モデルのパラメータを取得する"""
        if self.model is None:
            return self.model_params
        return self.model.get_params()

    def set_params(self, **params) -> "BaseModel":
        """モデルのパラメータを設定する"""
        self.model_params.update(params)
        if self.model is not None:
            self.model.set_params(**params)
        return self

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(model_name='{self.model_name}', is_fitted={self.is_fitted})"
        )
