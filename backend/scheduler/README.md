
**Solutions**:
- Ensure `SCHEDULER_DEFAULT = True` in settings
- Check if `django-apscheduler` is installed
- Verify database migrations for APScheduler

### Debug Mode

Enable debug logging for more detailed information:

```python
# In settings.py
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "DEBUG",  # Change to DEBUG
    },
}
```

## Advanced Configuration

### Custom Retry Settings

Modify retry behavior in the sync service:

```python
# In scheduler/services.py
sync_service = LabelStudioSyncService(
    max_retries=5,      # Increase retry attempts
    retry_delay=2.0     # Increase initial delay
)
```

### Custom Page Size

Adjust the number of projects fetched per page:

```python
# In scheduler/services.py
stats = sync_service.sync_projects(page_size=50)  # Smaller page size
```

### Custom Job ID

Use a custom job ID for multiple sync jobs:

```bash
python manage.py start_label_studio_sync_job --job-id "label_studio_sync_prod"
```

## Production Deployment

### 1. Environment Configuration

For production, use environment variables:

```bash
LABEL_STUDIO_URL=https://your-label-studio-domain.com
LABEL_STUDIO_API_KEY=your_production_api_key
LABEL_STUDIO_SYNC_INTERVAL_MINUTES=60
```

### 2. Process Management

Use a process manager like `supervisord` or `systemd` to keep the sync job running:

```ini
# supervisord.conf
[program:label_studio_sync]
command=python manage.py start_label_studio_sync_job
directory=/path/to/your/django/app
user=www-data
autostart=true
autorestart=true
```

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the logs for detailed error messages
3. Verify your configuration settings
4. Test with the `--run-once` command first

## Related Files

- `scheduler/services.py`: Main sync service implementation
- `scheduler/management/commands/start_label_studio_sync_job.py`: Management command
- `projects/models.py`: Project and Label models
- `config/settings.py`: Configuration settings
- `docker-compose.yml`: Docker setup with Label Studio