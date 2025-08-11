from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

from app import schemas

router = APIRouter()


# 実際のモデルインスタンス
MOCK_MODELS = {
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42),
        "type": "RandomForestRegressor",
        "is_fitted": False
    },
    "Linear Regression": {
        "model": LinearRegression(),
        "type": "LinearRegression",
        "is_fitted": False
    },
    "SVR": {
        "model": SVR(kernel='linear', C=1.0, epsilon=0.2),
        "type": "SVR",
        "is_fitted": False
    },
    "HistGradientBoostingRegressor": {
        "model": HistGradientBoostingRegressor(random_state=42),
        "type": "HistGradientBoostingRegressor",
        "is_fitted": False
    },
    "MLP": {
        "model": MLPRegressor(hidden_layer_sizes=(100, 100, 50), max_iter=1000, random_state=42),
        "type": "MLPRegressor",
        "is_fitted": False
    }
}


@router.get("/", response_model=schemas.ModelListResponse)
async def get_models() -> schemas.ModelListResponse:
    """
    利用可能なモデル一覧を取得
    """
    models = []
    for name, info in MOCK_MODELS.items():
        # 実際のモデルインスタンスからパラメータを取得
        params = info["model"].get_params()
        models.append(schemas.ModelInfo(
            name=name,
            type=info["type"],
            params=params,
            is_fitted=info["is_fitted"]
        ))
    
    return schemas.ModelListResponse(models=models)


def analyze_ml(model, df_train: pd.DataFrame, df_validate: pd.DataFrame, 
               x_columns: List[str], y_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, Any, Dict]:
    """
    機械学習モデルの学習と評価を行う
    """
    try:
        model.fit(df_train[x_columns], df_train[y_column])
        y_pred_train = model.predict(df_train[x_columns])
        y_pred_validate = model.predict(df_validate[x_columns])
    except ValueError as e:
        print(f"Error fitting model: {e}")
        raise
        
    # Evaluate the model
    mse_train = mean_squared_error(df_train[y_column].values, y_pred_train)
    r2_train = r2_score(df_train[y_column].values, y_pred_train)
    mse_validate = mean_squared_error(df_validate[y_column].values, y_pred_validate)
    r2_validate = r2_score(df_validate[y_column].values, y_pred_validate)

    df_train = df_train.copy()
    df_validate = df_validate.copy()
    df_train['pred'] = y_pred_train
    df_validate['pred'] = y_pred_validate
    
    print(f"Mean Squared Error for train: {mse_train}")
    print(f"R2 Score for train: {r2_train}")
    print(f"Mean Squared Error for validate: {mse_validate}")
    print(f"R2 Score for validate: {r2_validate}")

    metrics = {
        'mse_train': mse_train,
        'r2_train': r2_train,
        'mse_validate': mse_validate,
        'r2_validate': r2_validate
    }
    return df_train, df_validate, model, metrics


def draw_scatter_plot(gt: pd.Series, pred: pd.Series, label: str, metrics: Dict) -> str:
    """
    実測値と予測値の散布図を描画し、Base64エンコードされた画像を返す
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(gt, pred, alpha=0.5, label=label)
    plt.plot([gt.min(), gt.max()],
            [gt.min(), gt.max()],
            color='red', linestyle='--', label='Ideal Fit')
    plt.title(f"Actual vs Predicted ({label})")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    
    # メトリクスのテキストを追加
    if 'train' in label.lower():
        text = f"MSE: {metrics['mse_train']:.2f}\nR2: {metrics['r2_train']:.2f}"
    else:
        text = f"MSE: {metrics['mse_validate']:.2f}\nR2: {metrics['r2_validate']:.2f}"
    
    plt.text(0.05, 0.95, text, 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
    plt.legend()
    plt.grid()
    
    # 画像をBase64にエンコード
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64


def draw_feature_importance(model, x_columns: List[str]) -> str:
    """
    特徴量重要度のグラフを描画し、Base64エンコードされた画像を返す
    """
    if not hasattr(model, 'feature_importances_'):
        return ""
        
    plt.figure(figsize=(10, 6))
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    plt.bar(range(len(x_columns)), feature_importances[indices], align='center')
    plt.xticks(range(len(x_columns)), [x_columns[i] for i in indices], rotation=90)
    plt.title("Feature Importances")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    
    # 画像をBase64にエンコード
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64


@router.post("/train", response_model=schemas.ModelTrainResponse)
async def train_model(
    request: schemas.ModelTrainRequest
) -> schemas.ModelTrainResponse:
    """
    モデルを訓練する
    """
    if request.model_name not in MOCK_MODELS:
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
    
    model_info = MOCK_MODELS[request.model_name]
    model = model_info["model"]
    
    # データの読み込み（実際の実装では request.data_path から読み込む）
    # ここではモックデータを使用
    n_samples = 1000
    n_features = len(request.feature_columns) if request.feature_columns else 5
    feature_columns = request.feature_columns or [f"feature_{i}" for i in range(n_features)]
    
    # モックデータフレームを作成
    data = {col: np.random.randn(n_samples) for col in feature_columns}
    data[request.target_column] = np.random.randn(n_samples)
    df = pd.DataFrame(data)
    
    # データを訓練用と検証用に分割
    train_idx = int(n_samples * 0.8)
    df_train = df.iloc[:train_idx]
    df_validate = df.iloc[train_idx:]
    
    # モデルの学習と評価
    df_train, df_validate, model, metrics = analyze_ml(
        model, df_train, df_validate, feature_columns, request.target_column
    )
    
    # 散布図の生成
    scatter_train = draw_scatter_plot(
        df_train[request.target_column], 
        df_train['pred'], 
        'Train Data', 
        metrics
    )
    scatter_validate = draw_scatter_plot(
        df_validate[request.target_column],
        df_validate['pred'],
        'Validate Data',
        metrics
    )
    
    # 特徴量重要度の取得
    feature_importance = {}
    feature_importance_plot = ""
    if hasattr(model, 'feature_importances_'):
        for i, feature in enumerate(feature_columns):
            feature_importance[feature] = float(model.feature_importances_[i])
        feature_importance_plot = draw_feature_importance(model, feature_columns)
    
    # モデルを訓練済みに更新
    MOCK_MODELS[request.model_name]["is_fitted"] = True
    
    return schemas.ModelTrainResponse(
        model_name=request.model_name,
        train_score=metrics['r2_train'],
        validation_score=metrics['r2_validate'],
        feature_importance=feature_importance,
        metrics=metrics,
        scatter_train=scatter_train,
        scatter_validate=scatter_validate,
        feature_importance_plot=feature_importance_plot,
        train_predictions=df_train['pred'].tolist(),
        validate_predictions=df_validate['pred'].tolist()
    )


@router.post("/predict", response_model=schemas.ModelPredictResponse)
async def predict(
    request: schemas.ModelPredictRequest
) -> schemas.ModelPredictResponse:
    """
    モデルで予測を実行
    """
    if request.model_name not in MOCK_MODELS:
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
    
    if not MOCK_MODELS[request.model_name]["is_fitted"]:
        raise HTTPException(status_code=400, detail=f"Model {request.model_name} is not trained yet")
    
    model_info = MOCK_MODELS[request.model_name]
    model = model_info["model"]
    
    # 入力データを適切な形式に変換
    predictions = []
    for data_point in request.data:
        # 辞書形式のデータを配列に変換
        features = list(data_point.values())
        X = np.array(features).reshape(1, -1)
        
        # 予測を実行
        prediction = model.predict(X)[0]
        predictions.append(float(prediction))
    
    return schemas.ModelPredictResponse(
        predictions=predictions,
        model_name=request.model_name
    )


@router.get("/types", response_model=List[str])
async def get_model_types() -> List[str]:
    """
    利用可能なモデルタイプ一覧を取得
    """
    return list(set(info["type"] for info in MOCK_MODELS.values()))