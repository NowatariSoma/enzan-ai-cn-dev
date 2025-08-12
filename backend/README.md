# Support-PF - Backend

## Setting up for development

### Create environment file

Create a `.env` file follow structure of `.env.example` file.

1. Run script `cp .env.example .env`
1. Replace variables of `.env` file with your correct environment variables

## Setting up for deployment with Docker

### Download and install Docker desktop, then start Docker desktop

Download [`Docker desktop`](https://www.docker.com/products/docker-desktop/).

### Run deploy script

Run `docker compose up -d --build` will create containers:
- `mlp_postgres_db`
- `mlp_django_app`

### Access the application

- API: http://localhost:8000/
- Admin panel: http://localhost:8000/admin/

### Migrate database

#### PostgreSQL
```bash
docker compose exec backend python manage.py migrate
```


### Create admin account

```bash
docker compose exec backend python manage.py createsuperuser
```

### Stopping and Removing Containers

To stop the running containers:
```bash
docker compose down
```
To remove all containers and volumes:
```bash
docker compose down -v
```

## Project structure

<pre>
<code>
backend/
├── accounts/
├── common/
├── docker/
├── mlp/
├── ....
├── mlp_kms/
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── docker-compose.yml
├── README.md
├── manage.py
</code>
</pre>

## Testing Guide

### Prerequisites

Before running tests, ensure you have:
- Python 3.12 or higher installed
- Poetry installed (version 2.1.2 or higher)
- All dependencies installed via `poetry install`

### Running Tests

The project uses pytest for testing. Here are the different ways to run tests:

1. **Run all tests**:
```bash
cd backend
poetry run pytest -v
```

2. **Run specific test file**:
```bash
poetry run pytest projects/tests/test_auth.py -v
```

3. **Run specific test class**:
```bash
poetry run pytest projects/tests/test_auth.py::TestAuthentication -v
```

4. **Run specific test method**:
```bash
poetry run pytest projects/tests/test_auth.py::TestAuthentication::test_login_success -v
```

5. **Run multiple test files**:
```bash
poetry run pytest projects/tests/test_auth.py projects/tests/test_project_list.py -v
```

### Useful Test Options

- `-v`: Verbose output
- `-k "expression"`: Run tests matching the expression
- `-x`: Stop after first failure
- `--pdb`: Enter debugger on failures
- `-s`: Show print statements
- `--cov`: Generate coverage report (requires pytest-cov)

### Test Structure

The test suite is organized as follows:

- `projects/tests/`: Contains all test files
  - `test_auth.py`: Authentication tests
  - `test_project_list.py`: Project listing API tests
  - `test_project_detail.py`: Project detail API tests
  - `conftest.py`: Shared test fixtures
  - `__init__.py`: Test package initialization

### Writing New Tests

1. Create test files with prefix `test_`
2. Use descriptive test names
3. Follow the existing test structure
4. Use fixtures from `conftest.py` when needed

Example test structure:
```python
def test_something():
    # Arrange
    # Act
    # Assert
```

### Troubleshooting

Common issues and solutions:

1. **Database Issues**:
   - Tests use a test database :
     - Need to create a database for test before run the test and set variable in .env POSTGRES_TEST_DB
   - Database is reset between test runs
   - Use fixtures for test data

2. **Authentication Issues**:
   - Use the provided auth fixtures
   - Check token expiration
   - Verify user permissions

3. **Test Failures**:
   - Check the test output for detailed error messages
   - Use `-v` flag for verbose output
   - Use `--pdb` to debug failures
