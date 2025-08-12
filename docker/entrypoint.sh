#!/bin/bash

set -e

# Wait for database to be ready
echo "Waiting for database to be ready..."
while ! python manage.py check --database default 2>&1; do
  echo "Database is not ready yet. Waiting..."
  sleep 2
done
echo "Database is ready!"

# Make migrations
echo "Making migrations..."
python manage.py makemigrations

# Apply migrations
echo "Applying migrations postgres..."
python manage.py migrate

# Create a superuser if it doesn't exist
# echo "Creating superuser..."
# python manage.py custom_create_super_user

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Starting Supervisord..."
# Start Supervisord in the foreground here, which is PID 1 for the container.
# # # -n: Running in the foreground.
# # # -c: specify the main supervisord
exec /usr/bin/supervisord -n -c /etc/supervisor/supervisord.conf

echo "Initialization complete!"
