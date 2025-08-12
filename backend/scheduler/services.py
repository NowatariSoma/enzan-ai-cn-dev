import logging
import time
from typing import Any, Dict, List, Optional

from django.conf import settings
from django.db import transaction
from label_studio_sdk.client import LabelStudio
from projects.models import Label, Project

logger = logging.getLogger(__name__)


class LabelStudioSyncService:
    """Service for syncing data from Label Studio with pagination support."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the Label Studio sync service.

        Args:
            max_retries: Maximum number of retry attempts for failed API calls
            retry_delay: Delay between retries in seconds (will be exponential)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = LabelStudio(
            base_url=settings.LABEL_STUDIO_URL,
            api_key=settings.LABEL_STUDIO_API_KEY
        )
        logger.info(
            f"Initialized Label Studio client for {settings.LABEL_STUDIO_URL}")

    def sync_projects(self, page_size: int = 100) -> Dict[str, Any]:
        """
        Sync projects from Label Studio with pagination support.

        Args:
            page_size: Number of projects to fetch per page (default: 100)

        Returns:
            Dictionary containing sync statistics
        """
        stats = {
            'total': 0,
            'created': 0,
            'updated': 0,
            'skipped': 0,
            'errors': 0,
            'pages_processed': 0,
            'start_time': time.time()
        }

        try:
            logger.info("Starting project sync from Label Studio")

            # Process projects page by page
            page = 1
            has_more = True

            while has_more:
                try:
                    logger.info(
                        f"Processing page {page} with page_size={page_size}")

                    # Fetch projects for current page
                    ls_projects = self._fetch_projects_page(page, page_size)

                    if not ls_projects:
                        logger.info(f"No more projects found on page {page}")
                        has_more = False
                        break

                    # Process projects in current page
                    page_stats = self._process_projects_page(ls_projects)

                    # Update overall stats
                    stats['total'] += page_stats['total']
                    stats['created'] += page_stats['created']
                    stats['updated'] += page_stats['updated']
                    stats['skipped'] += page_stats['skipped']
                    stats['errors'] += page_stats['errors']
                    stats['pages_processed'] += 1

                    logger.info(f"Page {page} processed: {page_stats['total']} projects, "
                                f"{page_stats['created']} created, {page_stats['updated']} updated")

                    # Check if we have more pages
                    if len(ls_projects.items) < page_size:
                        has_more = False
                        logger.info(f"Reached last page (page {page})")
                    else:
                        page += 1

                except Exception as e:
                    stats['errors'] += 1
                    logger.error(f"Error processing page {page}: {str(e)}")
                    # Continue with next page instead of failing completely
                    page += 1

        except Exception as e:
            stats['errors'] += 1
            logger.error(f"Critical error during project sync: {str(e)}")
            raise

        finally:
            stats['end_time'] = time.time()
            stats['duration'] = stats['end_time'] - stats['start_time']
            logger.info(f"Project sync completed in {stats['duration']:.2f}s: "
                        f"{stats['total']} total, {stats['created']} created, "
                        f"{stats['updated']} updated, {stats['errors']} errors")

        return stats

    def _fetch_projects_page(self, page: int, page_size: int) -> List[Any]:
        """
        Fetch a single page of projects from Label Studio with retry logic.

        Args:
            page: Page number to fetch
            page_size: Number of projects per page

        Returns:
            List of project objects from Label Studio
        """
        for attempt in range(self.max_retries + 1):
            try:
                # Use pagination parameters if supported by the SDK
                # Note: The exact parameter names may vary based on Label Studio version
                projects = self.client.projects.list(
                    page=page,
                    page_size=page_size
                )
                return projects
            except Exception as e:
                # 404エラー（Invalid page）なら空リストを返して終了
                if 'Invalid page' in str(e) or '404' in str(e):
                    logger.info(f"Page {page} does not exist (404). Ending pagination.")
                    return []
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries + 1}): "
                                   f"{str(e)}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Failed to fetch projects page {page} after {self.max_retries + 1} attempts")
                    raise

    def _process_projects_page(self, ls_projects: List[Any]) -> Dict[str, int]:
        """
        Process a single page of projects.

        Args:
            ls_projects: List of project objects from Label Studio

        Returns:
            Dictionary containing page processing statistics
        """
        page_stats = {
            'total': len(ls_projects.items),
            'created': 0,
            'updated': 0,
            'skipped': 0,
            'errors': 0
        }

        with transaction.atomic():
            for ls_project in ls_projects:
                try:
                    project_stats = self._process_single_project(ls_project)

                    # Update page stats
                    for key in ['created', 'updated', 'skipped', 'errors']:
                        page_stats[key] += project_stats.get(key, 0)

                except Exception as e:
                    page_stats['errors'] += 1
                    logger.error(
                        f"Error processing project {getattr(ls_project, 'id', 'unknown')}: {str(e)}")
                    continue

        return page_stats

    def _process_single_project(self, ls_project: Any) -> Dict[str, int]:
        """
        Process a single project from Label Studio.

        Args:
            ls_project: Project object from Label Studio

        Returns:
            Dictionary containing processing statistics for this project
        """
        stats = {'created': 0, 'updated': 0, 'skipped': 0, 'errors': 0}

        try:
            # Get detailed project information with retry logic
            project_detail = self._get_project_detail(ls_project.id)

            if not project_detail:
                stats['skipped'] += 1
                logger.warning(
                    f"Skipping project {ls_project.id}: Could not fetch details")
                return stats

            # Prepare project data
            project_data = self._prepare_project_data(project_detail)

            # Create or update project
            project, created = self._create_or_update_project(project_data)

            if created:
                stats['created'] += 1
                logger.info(
                    f"Created project: {project.title} (ID: {project.id})")
            else:
                stats['updated'] += 1
                logger.info(
                    f"Updated project: {project.title} (ID: {project.id})")

            # Sync project labels if available
            # self._sync_project_labels(project, project_detail)

        except Exception as e:
            stats['errors'] += 1
            logger.error(
                f"Error processing project {getattr(ls_project, 'id', 'unknown')}: {str(e)}")

        return stats

    def _get_project_detail(self, project_id: int) -> Optional[Any]:
        """
        Get detailed project information with retry logic.

        Args:
            project_id: Label Studio project ID

        Returns:
            Project detail object or None if failed
        """
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.projects.get(project_id)

            except Exception as e:
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Failed to get project {project_id} details "
                                   f"(attempt {attempt + 1}/{self.max_retries + 1}): "
                                   f"{str(e)}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Failed to get project {project_id} details after {self.max_retries + 1} attempts")
                    return None

    def _prepare_project_data(self, project_detail: Any) -> Dict[str, Any]:
        """
        Prepare project data for database storage.

        Args:
            project_detail: Project detail object from Label Studio

        Returns:
            Dictionary containing project data
        """
        return {
            'title': getattr(project_detail, 'title', ''),
            'description': getattr(project_detail, 'description', '') or '',
            'organization': getattr(project_detail, 'organization'),
            'ls_project_id': project_detail.id,
            'is_published': getattr(project_detail, 'is_published', False),
            'model_version': getattr(project_detail, 'model_version', '1.0.0'),
            'is_draft': not getattr(project_detail, 'is_draft', False),
            'total_predictions_number': getattr(project_detail, 'total_predictions_number', 0),
            'ml_type_name': getattr(project_detail, 'ml_type_name', '')
        }

    def _create_or_update_project(self, project_data: Dict[str, Any]) -> tuple[Project, bool]:
        """
        Create or update a project in the database.

        Args:
            project_data: Project data dictionary

        Returns:
            Tuple of (project_object, created_boolean)
        """
        try:
            project = Project.objects.get(
                ls_project_id=project_data['ls_project_id'])

            # Update existing project
            for key, value in project_data.items():
                setattr(project, key, value)
            project.save()

            return project, False  # Updated

        except Project.DoesNotExist:
            # Create new project
            project = Project.objects.create(**project_data)
            return project, True  # Created

    # TODO : labels will get from predict system use s3
    def _sync_project_labels(self, project: Project, project_detail: Any) -> None:
        pass
