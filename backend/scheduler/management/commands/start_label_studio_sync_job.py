from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from django_apscheduler.jobstores import DjangoJobStore
from django_apscheduler.models import DjangoJob
import logging

from scheduler.services import LabelStudioSyncService

logger = logging.getLogger(__name__)


def run_label_studio_sync_job():
    """
    Standalone function to run the Label Studio sync job.
    This function is called by the APScheduler.
    """
    try:
        logger.info("Starting scheduled Label Studio sync...")
        
        # Create sync service instance
        sync_service = LabelStudioSyncService()
        
        # Run the sync
        stats = sync_service.sync_projects()
        
        # Log success
        logger.info(
            f"Scheduled Label Studio sync completed: "
            f"{stats['total']} total, {stats['created']} created, "
            f"{stats['updated']} updated, {stats['errors']} errors"
        )
        
        # Log detailed stats if there are errors
        if stats['errors'] > 0:
            logger.warning(
                f"Label Studio sync completed with {stats['errors']} errors. "
                f"Check logs for details."
            )
        
    except Exception as e:
        logger.error(f"Scheduled Label Studio sync failed: {str(e)}")
        # Don't re-raise - let the job complete so it can be retried
        raise


class Command(BaseCommand):
    help = 'Add Label Studio sync job to APScheduler with configurable interval'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force add the job even if it already exists',
            default=True
        )
        parser.add_argument(
            '--interval-minutes',
            type=int,
            help='Override interval from settings (in minutes)'
        )
        parser.add_argument(
            '--job-id',
            type=str,
            default='label_studio_sync',
            help='Unique job ID (default: label_studio_sync)'
        )
        parser.add_argument(
            '--remove',
            action='store_true',
            help='Remove the sync job instead of adding it'
        )
        parser.add_argument(
            '--list',
            action='store_true',
            help='List all existing jobs'
        )
        parser.add_argument(
            '--run-once',
            action='store_true',
            help='Run the job once and exit (don\'t start scheduler)'
        )

    def handle(self, *args, **options):
        # Check if APScheduler is configured
        if not hasattr(settings, 'SCHEDULER_DEFAULT') or not settings.SCHEDULER_DEFAULT:
            raise CommandError('APScheduler is not configured. Add SCHEDULER_DEFAULT = True to settings.')

        # Handle list command
        if options['list']:
            self._list_jobs()
            return

        # Handle remove command
        if options['remove']:
            self._remove_job(options['job_id'])
            return

        # Handle run-once command
        if options['run_once']:
            self._run_job_once()
            return

        # Get interval from settings or command line
        interval_minutes = self._get_interval_minutes(options['interval_minutes'])
        
        # Add the job and start scheduler
        self._add_job_and_start(options['job_id'], interval_minutes, options['force'])

    def _get_interval_minutes(self, override_interval=None):
        """Get interval minutes from settings or override."""
        if override_interval is not None:
            return override_interval
        
        # Get from settings with fallback
        interval_minutes = getattr(settings, 'LABEL_STUDIO_SYNC_INTERVAL_MINUTES', 60)
        
        if not isinstance(interval_minutes, int) or interval_minutes <= 0:
            raise CommandError(
                f'Invalid LABEL_STUDIO_SYNC_INTERVAL_MINUTES setting: {interval_minutes}. '
                'Must be a positive integer.'
            )
        
        return interval_minutes

    def _add_job_and_start(self, job_id, interval_minutes, force=False):
        """Add the Label Studio sync job to APScheduler and start the scheduler."""
        try:
            # Check if job already exists
            existing_job = DjangoJob.objects.filter(id=job_id).first()
            
            if existing_job and not force:
                self.stdout.write(
                    self.style.WARNING(
                        f'Job "{job_id}" already exists. Use --force to replace it.'
                    )
                )
                return
            
            # Remove existing job if force is True
            if existing_job and force:
                self._remove_job(job_id)
                self.stdout.write(f'Removed existing job "{job_id}"')
            
            # Create blocking scheduler
            scheduler = BlockingScheduler(timezone=settings.TIME_ZONE)
            scheduler.add_jobstore(DjangoJobStore(), "default")
            
            # Add the job using the standalone function
            trigger = IntervalTrigger(minutes=interval_minutes)
            
            scheduler.add_job(
                func=run_label_studio_sync_job,  # Use the standalone function
                trigger=trigger,
                id=job_id,
                name=f'Label Studio Sync (every {interval_minutes} minutes)',
                replace_existing=True,
                jobstore='default',
                max_instances=1,  # Prevent multiple instances running simultaneously
                coalesce=True,    # Combine missed executions
                misfire_grace_time=300  # 5 minutes grace time for missed executions
            )
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'Successfully added Label Studio sync job "{job_id}" '
                    f'with {interval_minutes}-minute interval'
                )
            )
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'Starting scheduler... Job will run every {interval_minutes} minutes.'
                )
            )
            self.stdout.write('Press Ctrl+C to stop the scheduler.')
            
            try:
                scheduler.start()
            except KeyboardInterrupt:
                self.stdout.write('\nStopping scheduler...')
                scheduler.shutdown()
                self.stdout.write(self.style.SUCCESS('Scheduler stopped.'))
            
        except Exception as e:
            raise CommandError(f'Failed to add job: {str(e)}')

    def _run_job_once(self):
        """Run the Label Studio sync job once and exit."""
        try:
            self.stdout.write('Running Label Studio sync job once...')
            run_label_studio_sync_job()
            self.stdout.write(self.style.SUCCESS('Job completed successfully.'))
        except Exception as e:
            raise CommandError(f'Failed to run job: {str(e)}')

    def _remove_job(self, job_id):
        """Remove the Label Studio sync job from APScheduler."""
        try:
            # Check if job exists
            existing_job = DjangoJob.objects.filter(id=job_id).first()
            if not existing_job:
                self.stdout.write(
                    self.style.WARNING(f'Job "{job_id}" does not exist in database.')
                )
                return
            
            # Create scheduler and remove job
            scheduler = BlockingScheduler(timezone=settings.TIME_ZONE)
            scheduler.add_jobstore(DjangoJobStore(), "default")
            
            try:
                # Try to remove from scheduler
                scheduler.remove_job(job_id, jobstore='default')
                self.stdout.write(
                    self.style.SUCCESS(f'Successfully removed job "{job_id}" from scheduler')
                )
            except Exception:
            #     # Job exists in database but not in scheduler - clean up database
            #     self.stdout.write(
            #         self.style.WARNING(
            #             f'Job "{job_id}" exists in database but not in scheduler. '
            #             'Cleaning up database entry...'
            #         )
            #     )
                existing_job.delete()
                self.stdout.write(
                    self.style.SUCCESS(f'Successfully removed job "{job_id}" from database')
                )
            
        except Exception as e:
            raise CommandError(f'Failed to remove job: {str(e)}')

    def _list_jobs(self):
        """List all existing jobs."""
        try:
            scheduler = BlockingScheduler(timezone=settings.TIME_ZONE)
            scheduler.add_jobstore(DjangoJobStore(), "default")
            
            jobs = scheduler.get_jobs()
            
            if not jobs:
                self.stdout.write('No jobs found.')
                return
            
            self.stdout.write('\n' + '='*80)
            self.stdout.write('EXISTING APSCHEDULER JOBS')
            self.stdout.write('='*80)
            
            for job in jobs:
                self.stdout.write(f'\nJob ID: {job.id}')
                self.stdout.write(f'Name: {job.name}')
                self.stdout.write(f'Function: {job.func}')
                self.stdout.write(f'Trigger: {job.trigger}')
                self.stdout.write(f'Next Run: {job.next_run_time}')
                self.stdout.write(f'Max Instances: {job.max_instances}')
                self.stdout.write(f'Coalesce: {job.coalesce}')
                self.stdout.write(f'Misfire Grace Time: {job.misfire_grace_time}s')
                self.stdout.write('-' * 40)
            
            self.stdout.write('='*80)
            
        except Exception as e:
            raise CommandError(f'Failed to list jobs: {str(e)}')