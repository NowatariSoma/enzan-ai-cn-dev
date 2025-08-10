.PHONY: help install test lint format clean docker-build docker-run docker-test

help:
	@echo "利用可能なコマンド:"
	@echo "  make install      - 依存関係をインストール"
	@echo "  make test         - テストを実行"
	@echo "  make lint         - コードのリントを実行"
	@echo "  make format       - コードをフォーマット"
	@echo "  make clean        - キャッシュファイルを削除"
	@echo "  make docker-build - Dockerイメージをビルド"
	@echo "  make docker-run   - Dockerコンテナを起動"
	@echo "  make docker-test  - Dockerコンテナ内でテストを実行"

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ --cov=app --cov-report=html --cov-report=term

lint:
	flake8 app/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 app/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	black app/ tests/
	isort app/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +

docker-build:
	docker build -t ai-ameasure-api .

docker-run:
	docker-compose up

docker-test:
	docker run --rm ai-ameasure-api python -m pytest tests/ -v