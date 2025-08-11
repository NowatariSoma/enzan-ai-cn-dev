from typing import Any, Dict, Optional, Type

from .base import BaseModel
from .sklearn_models import (
    HistGradientBoostingModel,
    LinearRegressionModel,
    MLPModel,
    RandomForestModel,
    SVRModel,
)


class ModelFactory:
    """
    モデルファクトリークラス
    登録されたモデルタイプから適切なモデルインスタンスを生成する
    """

    _models: Dict[str, Type[BaseModel]] = {
        "random_forest": RandomForestModel,
        "linear_regression": LinearRegressionModel,
        "svr": SVRModel,
        "hist_gradient_boosting": HistGradientBoostingModel,
        "mlp": MLPModel,
    }

    @classmethod
    def create_model(
        cls, model_type: str, model_params: Optional[Dict[str, Any]] = None
    ) -> BaseModel:
        """
        指定されたタイプのモデルインスタンスを作成する

        Args:
            model_type: モデルのタイプ
            model_params: モデルのパラメータ

        Returns:
            モデルインスタンス

        Raises:
            ValueError: 指定されたモデルタイプが登録されていない場合
        """
        if model_type not in cls._models:
            available_models = ", ".join(cls._models.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. " f"Available models: {available_models}"
            )

        model_class = cls._models[model_type]
        return model_class(model_params)

    @classmethod
    def register_model(cls, model_type: str, model_class: Type[BaseModel]) -> None:
        """
        新しいモデルタイプを登録する

        Args:
            model_type: モデルのタイプ名
            model_class: モデルのクラス
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"{model_class} must be a subclass of BaseModel")

        cls._models[model_type] = model_class

    @classmethod
    def get_available_models(cls) -> list[str]:
        """利用可能なモデルタイプのリストを取得する"""
        return list(cls._models.keys())

    @classmethod
    def get_model_class(cls, model_type: str) -> Type[BaseModel]:
        """指定されたタイプのモデルクラスを取得する"""
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls._models[model_type]
