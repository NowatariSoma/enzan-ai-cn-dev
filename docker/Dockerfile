FROM python:3.12.10-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    dos2unix \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy pyproject files
COPY backend/pyproject.toml backend/poetry.lock ./

# Copy entire backend
COPY backend/ ./

# Copy and fix entrypoint script
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
RUN dos2unix /app/docker/entrypoint.sh && chmod +x /app/docker/entrypoint.sh

# Install dependencies
WORKDIR /app/backend
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

# Set the working directory back to backend for manage.py
WORKDIR /app/backend

# Below supervisord settings

# Create static directory to avoid Django staticfiles warning
RUN mkdir -p /app/backend/static

# Copy supervisor configuration
COPY docker/conf/scheduler.conf /etc/supervisor/conf.d/scheduler.conf

# Entrypoint
ENTRYPOINT ["/bin/sh", "/app/docker/entrypoint.sh"]
