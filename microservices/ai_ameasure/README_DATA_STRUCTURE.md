# AI A-Measure API データ構造ガイド

## 必要なデータ構造

APIが正常に動作するために、以下のディレクトリ構造でデータを配置する必要があります：

```
data_folder/                       # データルートフォルダ
└── 01-hokkaido-akan/             # プロジェクトフォルダ
    └── main_tunnel/
        └── CN_measurement_data/
            ├── measurements_A/    # 計測データフォルダ
            │   ├── measurements_A_00001.csv
            │   ├── measurements_A_00002.csv
            │   └── ...
            ├── cycle_support/     # サイクル支保データ
            │   └── cycle_support.csv
            └── observation_of_face/  # 切羽観察データ
                └── observation_of_face.csv
```

## Docker環境での設定

### 1. docker-compose.ymlの設定

```yaml
volumes:
  # データフォルダをマウント
  - ../../data_folder:/app/data:ro   # プロジェクトのdata_folderをコンテナの/app/dataにマウント
  - ../../output:/app/output         # 出力用フォルダ
environment:
  - DATA_FOLDER=/app/data            # コンテナ内のデータパス
  - OUTPUT_FOLDER=/app/output        # コンテナ内の出力パス
```

### 2. .envファイルの設定

Docker環境用：
```env
DATA_FOLDER="/app/data"
OUTPUT_FOLDER="/app/output"
```

ローカル環境用：
```env
DATA_FOLDER="/path/to/your/data"
OUTPUT_FOLDER="/path/to/your/output"
```

## データファイルの配置

プロジェクトには既に`data_folder`ディレクトリにデータが配置されています。
新しいデータを追加する場合は、上記の構造に従って配置してください。

```bash
# Docker起動（microservices/ai_ameasureディレクトリから）
cd microservices/ai_ameasure
docker-compose up

# または、バックグラウンドで起動
docker-compose up -d
```

## APIエンドポイントの使用

### analyze-displacementエンドポイント

```bash
curl -X POST "http://localhost:8000/api/v1/measurements/analyze-displacement" \
  -H "Content-Type: application/json" \
  -d '{
    "folder_name": "01-hokkaido-akan",
    "max_distance_from_face": 100,
    "generate_charts": true
  }'
```

## トラブルシューティング

### エラー: "Measurements folder not found"

このエラーが発生する場合、以下を確認してください：

1. データフォルダが正しい構造で配置されているか
2. docker-compose.ymlでボリュームが正しくマウントされているか
3. 環境変数DATA_FOLDERが正しく設定されているか

### 確認コマンド

```bash
# コンテナ内のファイル構造を確認
docker-compose exec api ls -la /app/data/01-hokkaido-akan/main_tunnel/CN_measurement_data/

# 環境変数を確認
docker-compose exec api env | grep DATA_FOLDER
```