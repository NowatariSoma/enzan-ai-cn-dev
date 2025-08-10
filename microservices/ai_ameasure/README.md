# AI A-Measure API

[![Tests](https://github.com/enzan-koubou/ai-cn/actions/workflows/ai-ameasure-test.yml/badge.svg)](https://github.com/enzan-koubou/ai-cn/actions/workflows/ai-ameasure-test.yml)
[![codecov](https://codecov.io/gh/enzan-koubou/ai-cn/branch/main/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/enzan-koubou/ai-cn)

AI A-Measure機能をRESTful APIとして提供するFastAPIアプリケーション

## 概要

AI A-Measure APIは、トンネル工事における計測データの解析・可視化・予測を行うためのRESTful APIです。実際のCSV計測データを読み込み、機械学習による予測・シミュレーション機能を提供します。

## 主要機能

### 📊 データ可視化
- **実CSVデータ読み込み**: 実際のトンネル計測データ（1,460行、63ファイル）
- **時系列可視化**: 変位量・沈下量の時系列グラフ
- **分布分析**: ヒストグラム・散布図による分布可視化
- **多次元表示**: 切羽距離・計測日数・深度の3次元データ

### 🤖 機械学習予測
- **5種類のアルゴリズム**: RandomForest、SVR、線形回帰、HistGradientBoosting、MLP
- **高精度予測**: R²スコア 0.66-0.72の性能
- **多出力予測**: 沈下量（7センサー）・変位量（9センサー）同時予測
- **特徴量重要度**: 予測に重要な要素の分析

### 🔮 シミュレーション
- **日進量ベース予測**: 日々の掘進進捗に基づく将来予測
- **再帰的予測**: 過去の予測結果を活用した高精度シミュレーション
- **長期予測**: 最大365日間の予測シミュレーション
- **複数シナリオ**: 異なる掘進速度での比較分析

### ⚙️ システム管理
- **バッチ処理**: 複数プロジェクトの一括分析
- **モデル管理**: 訓練・保存・読み込み・更新
- **設定管理**: YAMLベースの設定ファイル
- **ヘルスモニタリング**: システム状態の監視

## セットアップ

### 必要な環境

- Python 3.11+
- Docker (オプション)

### ローカル開発

1. **依存関係のインストール**:
```bash
pip install -r requirements.txt
```

2. **環境変数の設定**:
```bash
cp .env.example .env
# 必要に応じて.envを編集
# 特にDATA_FOLDERとOUTPUT_FOLDERのパスを環境に合わせて設定
```

3. **アプリケーションの起動**:
```bash
uvicorn app.main:app --reload --port 8000
```

### Dockerを使用する場合

1. **イメージのビルド**:
```bash
docker build -t ai-ameasure-api .
```

2. **コンテナの起動**:
```bash
docker run -p 8000:8000 -v $(pwd)/.env:/app/.env ai-ameasure-api
```

## API仕様

APIドキュメントは以下のURLで確認できます:
- **Swagger UI**: http://localhost:8000/api/v1/docs
- **ReDoc**: http://localhost:8000/api/v1/redoc

### 主要エンドポイント

#### 📊 計測データ可視化 (`/api/v1/measurements`)
- `GET /displacement-series` - 変位量時系列データ
- `GET /settlement-series` - 沈下量時系列データ
- `GET /displacement-distribution` - 変位量分布データ
- `GET /settlement-distribution` - 沈下量分布データ
- `GET /tunnel-scatter` - トンネル計測散布図データ
- `GET /measurement-files` - 利用可能な計測ファイル一覧
- `POST /analyze` - 計測データの解析実行
- `GET /predictions` - 予測データテーブル

#### 🤖 機械学習予測 (`/api/v1/prediction`)
- `GET /models` - モデル一覧取得
- `GET /models/{model_name}` - モデル情報取得
- `POST /models/{model_name}/train` - モデル訓練
- `POST /models/{model_name}/predict` - 予測実行
- `POST /models/{model_name}/update-config` - モデル設定更新
- `DELETE /models/{model_name}` - モデル削除
- `GET /quick-predict/settlement` - 沈下量簡易予測
- `GET /quick-predict/convergence` - 変位量簡易予測

#### 🔮 シミュレーション (`/api/v1/prediction`)
- `POST /simulate` - 変位・沈下シミュレーション
- `POST /batch-process` - バッチ処理（複数フォルダ一括処理）
- `GET /health` - 予測エンジンヘルスチェック

#### 🛠️ 変位解析 (`/api/v1/displacement`)
- `POST /analyze` - 変位解析の実行
- `GET /folders` - 利用可能なフォルダ一覧

#### 📈 分析機能 (`/api/v1/analysis`)
- `POST /displacement` - 変位の時空間解析
- `POST /upload` - CSVファイルのアップロード
- `GET /correlation/{folder_name}` - 相関データの取得

### API使用例

#### 1. モデル訓練
```bash
curl -X POST "http://localhost:8000/api/v1/prediction/models/settlement/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "settlement",
    "folder_name": "01-hokkaido-akan",
    "test_size": 0.2
  }'
```

#### 2. 予測実行
```bash
curl -X GET "http://localhost:8000/api/v1/prediction/quick-predict/settlement?td=150.0&cycle=5"
```

#### 3. シミュレーション実行
```bash
curl -X POST "http://localhost:8000/api/v1/prediction/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "folder_name": "01-hokkaido-akan",
    "daily_advance": 2.0,
    "distance_from_face": 50.0,
    "prediction_days": 30,
    "recursive": true
  }'
```

#### 4. 計測データ取得
```bash
curl -X GET "http://localhost:8000/api/v1/measurements/displacement-series?num_points=100"
```

### 機能詳細

#### 📊 計測データAPI

**時系列データ取得**
- TD（トンネル距離）に対する変位・沈下の時系列データ
- 3m, 5m, 10m, 20m, 50m, 100m地点での計測値
- グラフ描画用に最適化されたデータ形式

**分布データ取得**
- 変位・沈下値の頻度分布（ヒストグラム用）
- 各距離地点での値の分布を表示
- ビンサイズ調整可能

**散布図データ**
- 切羽からの距離 vs 計測日数の散布図
- 深度情報を色で表現
- 3次元的なデータ可視化をサポート

#### 🤖 機械学習API

**モデル管理**
- 5種類のアルゴリズム（RandomForest、SVR、線形回帰、HistGradientBoosting、MLP）
- YAML設定ファイルによる柔軟な設定管理
- モデルの保存・読み込み・更新機能

**予測機能**
- 沈下量（7センサー）・変位量（9センサー）の多出力予測
- 特徴量重要度分析
- 予測信頼度評価

#### 🔮 シミュレーションAPI

**変位予測シミュレーション**
- 日進量と現在の切羽からの距離を指定して将来の変位を予測
- 再帰的予測オプションで精度向上
- 最大365日間の長期予測

**バッチ処理**
- 複数のトンネルフォルダを一括処理
- 各フォルダの処理結果と実行時間を返却
- 成功/失敗の統計情報を提供

## データ形式

### 入力データ
- **CSVファイル**: Shift-JIS エンコーディング
- **計測データ**: `measurements_A_XXXXX.csv`
- **データ量**: 1,460行、63ファイル
- **期間**: 2007年〜2024年の実際の計測データ

### 出力データ
- **時系列データ**: JSON形式、グラフ描画対応
- **予測結果**: 信頼度付きの予測値
- **シミュレーション結果**: 日次進捗データ

## ディレクトリ構成

```
microservices/ai_ameasure/
├── app/
│   ├── api/
│   │   ├── endpoints/           # APIエンドポイント
│   │   │   ├── measurements.py  # 計測データAPI
│   │   │   ├── prediction.py    # 機械学習・予測API
│   │   │   ├── displacement.py  # 変位解析API
│   │   │   ├── analysis.py      # 分析機能API
│   │   │   ├── models.py        # モデル管理API
│   │   │   └── simulation.py    # シミュレーションAPI
│   │   ├── deps.py              # 依存関係
│   │   └── api.py               # ルーター統合
│   ├── core/
│   │   ├── config.py            # 設定管理
│   │   ├── csv_loader.py        # CSVデータローダー
│   │   └── prediction_engine.py # 機械学習エンジン
│   ├── models/                  # 機械学習モデル
│   │   ├── manager.py           # モデルマネージャー
│   │   ├── factory.py           # モデルファクトリー
│   │   ├── config.py            # モデル設定
│   │   └── sklearn_models.py    # scikit-learnラッパー
│   ├── schemas/                 # Pydanticスキーマ
│   │   ├── measurements.py      # 計測データスキーマ
│   │   └── prediction.py        # 予測スキーマ
│   └── main.py                  # アプリケーションエントリポイント
├── config/
│   └── models.yaml              # モデル設定ファイル
├── requirements.txt
├── Dockerfile
└── README.md
```

## パフォーマンス

### 機械学習性能
- **沈下量予測**: R² = 0.72 (良好)
- **変位量予測**: R² = 0.67 (良好)
- **訓練時間**: 約3秒/モデル
- **予測時間**: <100ms/リクエスト

### API応答時間
- **計測データ取得**: <1秒
- **簡易予測**: <1秒
- **シミュレーション**: 5-30秒（予測日数による）
- **モデル訓練**: 3-10秒

## 開発

### テストの実行

```bash
# 単体テストの実行
make test

# カバレッジレポート付きテスト
make test-cov

# Docker内でテスト
make docker-test
```

### コードフォーマットとリント

```bash
# コードフォーマット
make format

# リントチェック
make lint
```

### CI/CD

このプロジェクトはGitHub Actionsを使用して自動テストを実行します：

- **プッシュ時**: `main`, `develop`, `feature/**` ブランチへのプッシュでテストを実行
- **プルリクエスト時**: `main`, `develop` ブランチへのPRでテストを実行
- **テスト内容**:
  - Python 3.10と3.11での単体テスト
  - コードカバレッジレポート
  - コードフォーマットチェック（black, isort）
  - リントチェック（flake8）
  - 型チェック（mypy）
  - Dockerビルドテスト

## トラブルシューティング

### よくある問題

**1. CSVファイル読み込みエラー**
```
UnicodeDecodeError: 'utf-8' codec can't decode
```
- 原因: CSVファイルがShift-JISエンコーディング
- 解決策: CSVローダーが自動的にエンコーディングを検出

**2. モデル訓練失敗**
```
Insufficient training data: X samples
```
- 原因: 訓練データが不足（<10サンプル）
- 解決策: より多くのCSVファイルを配置

**3. 予測エラー**
```
Feature names should match those that were passed during fit
```
- 原因: 特徴量名の不一致
- 解決策: APIが自動的に特徴量を調整

### ログレベル設定

```bash
# デバッグログを有効化
export LOG_LEVEL=DEBUG
uvicorn app.main:app --reload
```