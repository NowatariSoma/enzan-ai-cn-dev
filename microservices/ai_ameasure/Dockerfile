FROM python:3.11-slim

WORKDIR /app

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# 環境変数の設定
ENV GDAL_CONFIG=/usr/bin/gdal-config

# 依存関係のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードのコピー
COPY ./app /app/app

# テストコードのコピー（テスト実行用）
COPY ./tests /app/tests
COPY ./pytest.ini /app/pytest.ini
COPY ./pyproject.toml /app/pyproject.toml

# マスターデータのコピー（元のai_ameasureから）
# 注: docker-compose.ymlでボリュームマウントするため、ここではコピーしない
# COPY ../../ai_ameasure/app/master /app/app/master

# アップロードディレクトリの作成
RUN mkdir -p /tmp/ai_ameasure_uploads

# ポート公開
EXPOSE 8000

# アプリケーション起動
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]