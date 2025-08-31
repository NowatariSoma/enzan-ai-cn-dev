# Python 3.11のベースイメージを使用
FROM python:3.11-slim

# 作業ディレクトリを設定
WORKDIR /app

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    default-libmysqlclient-dev \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Poetryをインストール
RUN pip install --no-cache-dir poetry

# プロジェクトファイルをコピー
COPY pyproject.toml poetry.lock* ./

# Poetry設定を最適化
RUN poetry config virtualenvs.create false

# 依存関係をインストール（プロジェクト自体はインストールしない）
RUN poetry install --only=main --no-interaction --no-ansi --no-root

# アプリケーションコードをコピー
COPY . .

# 静的ファイル用のディレクトリを作成
RUN mkdir -p staticfiles

# ポート8080を公開
EXPOSE 8080

# 環境変数を設定
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=config.settings

# ヘルスチェック用のスクリプト
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/')"

# 起動スクリプト
CMD ["python", "run.py", "runserver", "0.0.0.0:8080"]
