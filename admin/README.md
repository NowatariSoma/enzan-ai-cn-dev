# Admin Panel - Backend

## Project Overview

管理者画面のバックエンドAPI。Djangoで構築されたREST APIサーバー。

## Development Setup

### Prerequisites

- Python 3.10以上
- Poetry 2.0以上

### Install Dependencies

```bash
poetry install
```

### Environment Setup

1. Create `.env` file:
```bash
cp .env.example .env
```

2. Configure environment variables in `.env`

### Database Setup

```bash
# Run migrations
poetry run python manage.py migrate

# Create superuser
poetry run python manage.py create_superuser

# Seed initial data
poetry run python manage.py seed_locations
poetry run python manage.py seed_features
```

### Development Server

```bash
# Using custom run script (defaults to port 8080)
poetry run python run.py runserver

# Or manually specify port 8080
poetry run python manage.py runserver 8080
```

- API: http://localhost:8080/
- Admin panel: http://localhost:8080/admin/
- API documentation: http://localhost:8080/swagger/

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
