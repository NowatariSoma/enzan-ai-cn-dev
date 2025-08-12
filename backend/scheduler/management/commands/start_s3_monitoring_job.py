import logging
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from django_apscheduler.jobstores import DjangoJobStore
from django_apscheduler.models import DjangoJob
from django.core.management import call_command
from projects.services import S3PredictionMonitor
from projects.models import Project

logger = logging.getLogger(__name__)


def run_s3_monitoring_job(all_projects=True, project_id=None, max_files_per_project=1000, dry_run=False, verbose=False):
    """
    Standalone function to run the S3 monitoring job. Called by APScheduler.
    """
    try:
        logger.info("Starting scheduled S3 prediction monitoring...")
        monitor = S3PredictionMonitor()
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        if dry_run:
            logger.info('DRY RUN MODE - No files will be processed')
        if project_id:
            try:
                project = Project.objects.get(ls_project_id=project_id)
                if dry_run:
                    json_files = monitor.get_project_json_files(
                        project, max_keys=max_files_per_project * 2)
                    logger.info(
                        f'Found {len(json_files)} prediction JSON files for project {project_id}')
                    for i, file_info in enumerate(json_files[:max_files_per_project], 1):
                        logger.info(
                            f'  {i}. {file_info["key"]} (Last modified: {file_info["last_modified"]}, Size: {file_info["size"]} bytes)')
                else:
                    processed_count, failed_count = monitor.monitor_project(
                        project, max_files=max_files_per_project)
                    logger.info(
                        f'Successfully processed {processed_count} prediction files for project {project_id}')
                    if failed_count > 0:
                        logger.warning(
                            f'Failed to process {failed_count} prediction files for project {project_id}')
            except Project.DoesNotExist:
                logger.error(
                    f"Project with ls_project_id {project_id} not found")
        else:
            if dry_run:
                projects = monitor.get_projects_for_monitoring()
                total_files = 0
                for project in projects:
                    json_files = monitor.get_project_json_files(
                        project, max_keys=max_files_per_project * 2)
                    total_files += len(json_files)
                    logger.info(
                        f'\nProject {project.ls_project_id} ({project.title}):')
                    logger.info(f'  Found {len(json_files)} files')
                    for i, file_info in enumerate(json_files[:max_files_per_project], 1):
                        logger.info(
                            f'    {i}. {file_info["key"]} (Last modified: {file_info["last_modified"]})')
                logger.info(
                    f'\nTotal files across all projects: {total_files}')
            else:
                results = monitor.monitor_all_projects(max_files_per_project)
                total_processed = 0
                total_failed = 0
                for project_id, (processed, failed) in results.items():
                    if processed > 0:
                        logger.info(
                            f'Project {project_id}: {processed} processed, {failed} failed')
                    elif failed > 0:
                        logger.warning(
                            f'Project {project_id}: {processed} processed, {failed} failed')
                    else:
                        logger.info(
                            f'Project {project_id}: No new files found')
                    total_processed += processed
                    total_failed += failed
                logger.info(
                    f'All projects monitoring complete: {total_processed} total processed, {total_failed} total failed')
    except Exception as e:
        logger.error(f"Error in S3 prediction monitoring: {e}")
        raise


class Command(BaseCommand):
    help = 'Add S3 monitoring job to APScheduler with configurable interval.'

    def add_arguments(self, parser):
        parser.add_argument('--interval-minutes', type=int,
                            help='Override interval from settings (in minutes)')
        parser.add_argument('--job-id', type=str, default='s3_monitoring',
                            help='Unique job ID (default: s3_monitoring)')
        parser.add_argument('--remove', action='store_true',
                            help='Remove the S3 monitoring job instead of adding it')
        parser.add_argument('--list', action='store_true',
                            help='List all existing jobs')
        parser.add_argument('--run-once', action='store_true',
                            help='Run the job once and exit (don\'t start scheduler)')
        parser.add_argument('--all-projects', action='store_true',
                            help='Monitor all projects individually (recommended for project-specific folders)')
        parser.add_argument('--project-id', type=int,
                            help='Monitor only a specific project by ls_project_id')
        parser.add_argument('--max-files-per-project', type=int, default=1000,
                            help='Maximum number of files to process per project (default: 1000)')
        parser.add_argument('--dry-run', action='store_true',
                            help='List files that would be processed without actually processing them')
        parser.add_argument('--verbose', action='store_true',
                            help='Enable verbose logging')

    def handle(self, *args, **options):
        if not hasattr(settings, 'SCHEDULER_DEFAULT') or not settings.SCHEDULER_DEFAULT:
            raise CommandError(
                'APScheduler is not configured. Add SCHEDULER_DEFAULT = True to settings.')

        if options['list']:
            self._list_jobs()
            return
        if options['remove']:
            self._remove_job(options['job_id'])
            return
        if options['run_once']:
            self._run_job_once(options)
            return

        interval_minutes = options['interval_minutes'] or getattr(
            settings, 'S3_MONITORING_INTERVAL_MINUTES', 5)
        self._add_job_and_start(
            options['job_id'], interval_minutes, options)

    def _add_job_and_start(self, job_id, interval_minutes, options):
        try:
            existing_job = DjangoJob.objects.filter(id=job_id).first()
            if existing_job:
                self._remove_job(job_id)
                self.stdout.write(f'Removed existing job "{job_id}"')
            scheduler = BlockingScheduler(timezone=settings.TIME_ZONE)
            scheduler.add_jobstore(DjangoJobStore(), "default")
            trigger = IntervalTrigger(minutes=interval_minutes)
            scheduler.add_job(
                func=run_s3_monitoring_job,
                trigger=trigger,
                id=job_id,
                name=f'S3 Monitoring (every {interval_minutes} minutes)',
                replace_existing=True,
                jobstore='default',
                max_instances=1,
                coalesce=True,
                misfire_grace_time=300,
                kwargs={
                    'all_projects': options['all_projects'],
                    'project_id': options['project_id'],
                    'max_files_per_project': options['max_files_per_project'],
                    'dry_run': options['dry_run'],
                    'verbose': options['verbose'],
                }
            )
            self.stdout.write(self.style.SUCCESS(
                f'Successfully added S3 monitoring job "{job_id}" with {interval_minutes}-minute interval'))
            self.stdout.write(self.style.SUCCESS(
                f'Starting scheduler... Job will run every {interval_minutes} minutes.'))
            self.stdout.write('Press Ctrl+C to stop the scheduler.')
            try:
                scheduler.start()
            except KeyboardInterrupt:
                self.stdout.write('\nStopping scheduler...')
                scheduler.shutdown()
                self.stdout.write(self.style.SUCCESS('Scheduler stopped.'))
        except Exception as e:
            raise CommandError(f'Failed to add job: {str(e)}')

    def _run_job_once(self, options):
        try:
            self.stdout.write('Running S3 monitoring job once...')
            run_s3_monitoring_job(
                all_projects=options['all_projects'],
                project_id=options['project_id'],
                max_files_per_project=options['max_files_per_project'],
                dry_run=options['dry_run'],
                verbose=options['verbose']
            )
            self.stdout.write(self.style.SUCCESS(
                'Job completed successfully.'))
        except Exception as e:
            raise CommandError(f'Failed to run job: {str(e)}')

    def _remove_job(self, job_id):
        try:
            existing_job = DjangoJob.objects.filter(id=job_id).first()
            if not existing_job:
                self.stdout.write(self.style.WARNING(
                    f'Job "{job_id}" does not exist in database.'))
                return
            scheduler = BlockingScheduler(timezone=settings.TIME_ZONE)
            scheduler.add_jobstore(DjangoJobStore(), "default")
            try:
                scheduler.remove_job(job_id, jobstore='default')
                self.stdout.write(self.style.SUCCESS(
                    f'Successfully removed job "{job_id}" from scheduler'))
            except Exception:
                existing_job.delete()
                self.stdout.write(self.style.SUCCESS(
                    f'Successfully removed job "{job_id}" from database'))
        except Exception as e:
            raise CommandError(f'Failed to remove job: {str(e)}')

    def _list_jobs(self):
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
                self.stdout.write(
                    f'Misfire Grace Time: {job.misfire_grace_time}s')
                self.stdout.write('-' * 40)
            self.stdout.write('='*80)
        except Exception as e:
            raise CommandError(f'Failed to list jobs: {str(e)}')
