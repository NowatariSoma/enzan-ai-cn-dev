import logging
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

from projects.services import S3PredictionMonitor
from projects.models import Project

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Monitor S3 for new prediction JSON files and process them per project'

    def add_arguments(self, parser):
        parser.add_argument(
            '--max-files-per-project',
            type=int,
            default=1000,
            help='Maximum number of files to process per project (default: 1000)'
        )
        parser.add_argument(
            '--project-id',
            type=int,
            help='Monitor only a specific project by ls_project_id'
        )
        parser.add_argument(
            '--all-projects',
            action='store_true',
            help='Monitor all projects individually (recommended for project-specific folders)'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='List files that would be processed without actually processing them'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose logging'
        )

    def handle(self, *args, **options):
        if options['verbose']:
            logging.getLogger().setLevel(logging.DEBUG)

        max_files_per_project = options['max_files_per_project']
        project_id = options['project_id']
        all_projects = options['all_projects']
        dry_run = options['dry_run']

        self.stdout.write(
            self.style.SUCCESS(f'Starting S3 prediction monitoring...')
        )

        try:
            # Initialize the monitor
            monitor = S3PredictionMonitor()

            if dry_run:
                self.stdout.write('DRY RUN MODE - No files will be processed')

                if project_id:
                    # Dry run for specific project
                    try:
                        project = Project.objects.get(ls_project_id=project_id)
                        json_files = monitor.get_project_json_files(
                            project, max_keys=max_files_per_project * 2)

                        self.stdout.write(
                            f'Found {len(json_files)} prediction JSON files for project {project_id}:')
                        for i, file_info in enumerate(json_files[:max_files_per_project], 1):
                            self.stdout.write(
                                f'  {i}. {file_info["key"]} '
                                f'(Last modified: {file_info["last_modified"]}, '
                                f'Size: {file_info["size"]} bytes)'
                            )
                    except Project.DoesNotExist:
                        raise CommandError(
                            f"Project with ls_project_id {project_id} not found")

                elif all_projects:
                    # Dry run for all projects
                    projects = monitor.get_projects_for_monitoring()
                    total_files = 0

                    for project in projects:
                        json_files = monitor.get_project_json_files(
                            project, max_keys=max_files_per_project * 2)
                        total_files += len(json_files)

                        self.stdout.write(
                            f'\nProject {project.ls_project_id} ({project.title}):')
                        self.stdout.write(f'  Found {len(json_files)} files')

                        for i, file_info in enumerate(json_files[:max_files_per_project], 1):
                            self.stdout.write(
                                f'    {i}. {file_info["key"]} '
                                f'(Last modified: {file_info["last_modified"]})'
                            )

                    self.stdout.write(
                        f'\nTotal files across all projects: {total_files}')

                else:
                    self.stdout.write(
                        self.style.ERROR(
                            'Please specify either --project-id or --all-projects for dry run')
                    )
            else:
                # Process the files
                if project_id:
                    # Monitor specific project
                    try:
                        project = Project.objects.get(ls_project_id=project_id)
                        processed_count, failed_count = monitor.monitor_project(
                            project, max_files_per_project)

                        if processed_count > 0:
                            self.stdout.write(
                                self.style.SUCCESS(
                                    f'Successfully processed {processed_count} prediction files for project {project_id}'
                                )
                            )

                        if failed_count > 0:
                            self.stdout.write(
                                self.style.WARNING(
                                    f'Failed to process {failed_count} prediction files for project {project_id}'
                                )
                            )

                        if processed_count == 0 and failed_count == 0:
                            self.stdout.write(
                                self.style.SUCCESS(
                                    f'No new prediction files found for project {project_id}')
                            )

                    except Project.DoesNotExist:
                        raise CommandError(
                            f"Project with ls_project_id {project_id} not found")

                else:
                    # Monitor all projects individually
                    results = monitor.monitor_all_projects(
                        max_files_per_project)

                    total_processed = 0
                    total_failed = 0

                    for project_id, (processed, failed) in results.items():
                        if processed > 0:
                            self.stdout.write(
                                self.style.SUCCESS(
                                    f'Project {project_id}: {processed} processed, {failed} failed'
                                )
                            )
                        elif failed > 0:
                            self.stdout.write(
                                self.style.WARNING(
                                    f'Project {project_id}: {processed} processed, {failed} failed'
                                )
                            )
                        else:
                            self.stdout.write(
                                f'Project {project_id}: No new files found'
                            )

                        total_processed += processed
                        total_failed += failed

                    self.stdout.write(
                        self.style.SUCCESS(
                            f'All projects monitoring complete: {total_processed} total processed, {total_failed} total failed'
                        )
                    )

                self.stdout.write(
                    self.style.SUCCESS('S3 prediction monitoring completed')
                )

        except Exception as e:
            logger.error(f"Error in S3 prediction monitoring: {e}")
            raise CommandError(f"Failed to monitor S3 predictions: {e}")
