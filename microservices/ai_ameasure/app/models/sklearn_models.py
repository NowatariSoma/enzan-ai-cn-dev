from typing import Any, Dict, Optional

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from .base import BaseModel


class RandomForestModel(BaseModel):
    """ランダムフォレストモデル"""

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        default_params = {"random_state": 42}
        if model_params:
            default_params.update(model_params)
        super().__init__("Random Forest", default_params)

    def build_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(**self.model_params)


class LinearRegressionModel(BaseModel):
    """線形回帰モデル"""

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        super().__init__("Linear Regression", model_params or {})

    def build_model(self) -> LinearRegression:
        return LinearRegression(**self.model_params)


class SVRModel(BaseModel):
    """サポートベクター回帰モデル"""

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        default_params = {"kernel": "linear", "C": 1.0, "epsilon": 0.2}
        if model_params:
            default_params.update(model_params)
        super().__init__("SVR", default_params)

    def build_model(self) -> SVR:
        return SVR(**self.model_params)


class HistGradientBoostingModel(BaseModel):
    """ヒストグラム勾配ブースティングモデル"""

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        default_params = {"random_state": 42}
        if model_params:
            default_params.update(model_params)
        super().__init__("HistGradientBoostingRegressor", default_params)

    def build_model(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(**self.model_params)


class MLPModel(BaseModel):
    """多層パーセプトロンモデル"""

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        default_params = {
            "hidden_layer_sizes": (100, 100, 50),
            "max_iter": 1000,
            "random_state": 42,
        }
        if model_params:
            default_params.update(model_params)
        super().__init__("MLP", default_params)

    def build_model(self) -> MLPRegressor:
        return MLPRegressor(**self.model_params)
