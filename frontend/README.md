# WorkerVision Frontend

Next.jsベースのフロントエンドアプリケーションで、複数カメラからのリアルタイム作業員監視を提供します。

## 機能

- **リアルタイム監視**: 2つのカメラからのライブフィード（1秒間隔で更新）
- **作業員カウント**: ヘルメット種別による作業員の分類と集計
- **接続状態表示**: 各カメラの接続状態をリアルタイムで監視
- **日本語表示**: 設定可能な日本語表示名（管理者、作業員、ヘルメットなし）

## 環境設定

`.env.` ファイルを作成して以下の環境変数を設定してください：

```.env
# Camera API URLs
NEXT_PUBLIC_CAMERA1_URL=http://localhost:8001
NEXT_PUBLIC_CAMERA2_URL=http://localhost:8002

# Legacy API URLs (for backward compatibility)
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_BACKEND2_URL=http://localhost:8003
```

## API エンドポイント

### カメラ1 (ポート 8001)
- 接続状態: `GET http://localhost:8001/status`
- カウント取得: `GET http://localhost:8001/get-data`
- 画像取得: `GET http://localhost:8001/get-annotated-frame`

### カメラ2 (ポート 8002)
- 接続状態: `GET http://localhost:8002/status`
- カウント取得: `GET http://localhost:8002/get-data`
- 画像取得: `GET http://localhost:8002/get-annotated-frame`

## ヘルメット種別の表示名

以下の表示名が設定されています：
- `red_helmet` → `管理者`
- `other_helmet` → `作業員`
- `no_helmet` → `ヘルメットなし`

表示名は `frontend/types/worker.ts` の `HELMET_LABELS` で変更可能です。

## 開発環境での実行

```bash
# 依存関係のインストール
npm install

# 開発サーバー起動
npm run dev
```

アプリケーションは `http://localhost:3000` で利用可能になります。

## 更新間隔

- **カメラフィード**: 1秒間隔で自動更新
- **カウントデータ**: 1秒間隔で自動更新
- **接続状態**: リアルタイム監視