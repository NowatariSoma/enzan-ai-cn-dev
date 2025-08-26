# Admin Panel - Backend

## Project Overview

管理者画面のバックエンドAPI。Djangoで構築されたREST APIサーバー。

## Quick Start

### 1. 依存関係のインストール
```bash
poetry install
```

### 2. データベース起動
```bash
# MySQL 8.0コンテナを起動
docker-compose up -d db
```

### 3. データベース設定
```bash
# マイグレーション実行
poetry run python manage.py migrate

# 管理者ユーザー作成
poetry run python manage.py create_superuser
# Username: admin, Password: admin123456

# 初期データ投入
poetry run python manage.py seed_locations
poetry run python manage.py seed_features
```

### 4. 開発サーバー起動
```bash
# デフォルトで8080番ポートで起動
poetry run python run.py runserver
```

### 5. アクセス
- API: http://localhost:8080/
- 管理画面: http://localhost:8080/admin/
- API文書: http://localhost:8080/swagger/

**管理画面ログイン情報:**
- ユーザー名: `admin`
- パスワード: `admin123456`

## Project Structure

```
admin/
├── accounts/          # ユーザー認証・管理
│   ├── management/
│   │   └── commands/
│   │       └── create_superuser.py
│   ├── migrations/
│   ├── models.py
│   ├── serializers.py
│   ├── urls.py
│   └── views.py
├── common/            # 共通ユーティリティ
│   ├── constant.py
│   ├── exceptions_handler.py
│   ├── format_response.py
│   ├── helper.py
│   ├── mail_services.py
│   ├── messages.py
│   ├── ordering_filter.py
│   ├── pagination.py
│   ├── services.py
│   ├── swagger.py
│   ├── tests/
│   └── validators.py
├── config/            # Django設定
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── locations/         # 拠点管理
│   ├── management/
│   │   └── commands/
│   │       ├── seed_features.py
│   │       └── seed_locations.py
│   ├── migrations/
│   ├── models.py
│   ├── serializers.py
│   ├── urls.py
│   └── views.py
├── manage.py
├── pyproject.toml
└── README.md
```

## Testing

### Running Tests

```bash
# Run all tests
poetry run pytest -v

# Run specific test file
poetry run pytest accounts/tests.py -v

# Run specific test class or method
poetry run pytest accounts/tests.py::TestUserModel -v
```

### Test Options

- `-v`: Verbose output
- `-k "expression"`: Run tests matching expression
- `-x`: Stop after first failure
- `--pdb`: Enter debugger on failures
- `-s`: Show print statements

## API Endpoints

### Authentication

- `POST /api/auth/login/` - ログイン
- `POST /api/auth/logout/` - ログアウト
- `POST /api/auth/refresh/` - トークンリフレッシュ

### Locations

- `GET /api/locations/` - 拠点一覧取得
- `POST /api/locations/` - 拠点作成
- `GET /api/locations/{id}/` - 拠点詳細取得
- `PUT /api/locations/{id}/` - 拠点更新
- `DELETE /api/locations/{id}/` - 拠点削除

### Users

- `GET /api/users/` - ユーザー一覧取得
- `POST /api/users/` - ユーザー作成
- `GET /api/users/{id}/` - ユーザー詳細取得
- `PUT /api/users/{id}/` - ユーザー更新
- `DELETE /api/users/{id}/` - ユーザー削除

## Management Commands

### Create Superuser

```bash
poetry run python manage.py create_superuser
```

### Seed Data

```bash
# Seed locations data
poetry run python manage.py seed_locations

# Seed features data  
poetry run python manage.py seed_features
```

## Architecture

### Models

- **User**: カスタムユーザーモデル（認証・権限管理）
- **Location**: 拠点情報（名前、説明、アラートレベルなど）

### Authentication

JWT（JSON Web Token）ベースの認証システム。

### API Response Format

```json
{
    "success": true,
    "data": {},
    "message": "Success",
    "errors": null
}
```

### Error Handling

統一されたエラーレスポンス形式。バリデーションエラー、認証エラー、サーバーエラーを適切にハンドリング。
