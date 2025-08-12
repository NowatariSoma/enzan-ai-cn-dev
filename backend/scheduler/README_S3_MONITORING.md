# S3 Prediction Monitoring Job

This module provides a robust, scheduled S3 monitoring solution for your Django project. It uses APScheduler to periodically scan S3 for new prediction files and process them per project.

---

## Quick Start

**1. Ensure prerequisites:**
- `SCHEDULER_DEFAULT = True` is set in your Django `settings.py`
- `django-apscheduler` is installed and migrations are applied
- Your AWS/LocalStack S3 credentials are configured

**2. Add the job:**

```bash
python manage.py start_s3_monitoring_job --all-projects
```

**3. Run once for testing:**

```bash
python manage.py start_s3_monitoring_job --run-once --all-projects
```

**4. Remove the job:**

```bash
python manage.py start_s3_monitoring_job --remove
```

**5. List all jobs:**

```bash
python manage.py start_s3_monitoring_job --list
```

---

## Command Options

| Option                      | Description                                                        |
|-----------------------------|--------------------------------------------------------------------|
| `--all-projects`            | Monitor all projects (recommended)                                 |
| `--project-id <id>`         | Monitor a specific project by `ls_project_id`                      |
| `--max-files-per-project`   | Max files to process per project (default: 1000)                   |
| `--dry-run`                 | List files that would be processed, but do not process them        |
| `--verbose`                 | Enable verbose logging                                             |
| `--interval-minutes`        | Set the interval (in minutes) for the scheduled job (default: 5)   |
| `--force`                   | Force add the job even if it already exists                        |
| `--job-id <id>`             | Unique job ID (default: `s3_monitoring`)                           |
| `--remove`                  | Remove the S3 monitoring job instead of adding it                  |
| `--list`                    | List all existing jobs                                             |
| `--run-once`                | Run the job once and exit (don't start scheduler)                  |

---

## Debug Mode

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

---

## Advanced Configuration

### Custom Interval

Set a custom interval for the scheduled job:

```bash
python manage.py start_s3_monitoring_job --all-projects --interval-minutes 10
```

### Custom Job ID

Use a custom job ID for multiple monitoring jobs:

```bash
python manage.py start_s3_monitoring_job --all-projects --job-id "s3_monitoring_prod"
```

---

## Production Deployment

### 1. Environment Configuration

Set your S3 and monitoring settings via environment variables or `settings.py`:

```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_STORAGE_BUCKET_NAME=your_bucket
S3_MONITORING_INTERVAL_MINUTES=5
AWS_S3_ENDPOINT_URL=localhost:4556 -- this config when you want to use another storage to testing
```

### 2. Process Management

Use a process manager like `supervisord` or `systemd` to keep the monitoring job running:

```ini
# supervisord.conf
[program:s3_monitoring]
command=python manage.py start_s3_monitoring_job --all-projects
directory=/path/to/your/django/app
user=www-data
autostart=true
autorestart=true
```

---

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the logs for detailed error messages
3. Verify your configuration settings
4. Test with the `--run-once` command first

---

## Related Files

- `scheduler/services.py`: S3 monitoring scheduling logic
- `scheduler/management/commands/start_s3_monitoring_job.py`: Management command
- `projects/services.py`: S3 monitoring and processing logic
- `projects/models.py`: Project and Prediction models
- `config/settings.py`: Configuration settings
- `docker-compose.yml`: Docker setup (if using LocalStack) 