import itertools
import json
import logging
import re
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from .models import Prediction, PredictionResult, Project

logger = logging.getLogger(__name__)


class S3PredictionMonitor:
    """Service for monitoring S3 for new prediction JSON files per project."""

    def __init__(self):
        self.s3_client = None
        self.bucket_name = settings.AWS_STORAGE_BUCKET_NAME
        self._initialize_s3_client()

    def _initialize_s3_client(self):
        """Initialize S3 client with credentials from environment."""
        try:
            client_kwargs = {
                'region_name': settings.AWS_S3_REGION_NAME,
            }
            if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
                client_kwargs['aws_access_key_id'] = settings.AWS_ACCESS_KEY_ID
                client_kwargs['aws_secret_access_key'] = settings.AWS_SECRET_ACCESS_KEY
            if settings.DEBUG:
                client_kwargs['endpoint_url'] = settings.AWS_S3_ENDPOINT_URL
                client_kwargs['verify'] = getattr(settings, "AWS_S3_VERIFY", True)
            print(
                f"Connecting to S3 bucket: {self.bucket_name} "
                f"with config: {client_kwargs}"
            )
            self.s3_client = boto3.client('s3', **client_kwargs)
            self.s3_client.list_objects_v2(Bucket=self.bucket_name, MaxKeys=1)
            logger.info(
                f"Successfully connected to S3 bucket: {self.bucket_name}"
            )
        except NoCredentialsError:
            logger.error(
                "AWS credentials not found. Please configure AWS credentials."
            )
            raise
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(
                    f"S3 bucket '{self.bucket_name}' not found."
                )
            else:
                logger.error(f"Failed to connect to S3: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error initializing S3 client: {e}"
            )
            raise

    def get_projects_for_monitoring(self) -> List[Project]:
        """
        Get all projects that should be monitored.
        Currently returns all published projects, but you can customize this logic.
        """
        return Project.objects.all()

    def get_project_json_files(self, project: Project, max_keys: int = 1000) -> List[Dict]:
        """
        Get JSON files for a specific project from S3.

        Args:
            project: Project instance to monitor
            max_keys: Maximum number of keys to retrieve from S3

        Returns:
            List of dictionaries containing file information
        """
        try:

            current_time = timezone.now()
            folder_path = f"{project.ls_project_id}/{current_time.strftime('%Y')}/{int(current_time.strftime('%m'))}/"
            logger.info(
                f"Listing S3 objects for project {project.ls_project_id} in folder {folder_path}")
            # List objects in the project's folder
            project_prefix = folder_path
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=project_prefix,
                MaxKeys=max_keys
            )

            if 'Contents' not in response:
                logger.info(
                    f"No objects found in S3 for project {project.ls_project_id}")
                return []

            # Filter for JSON files that match our patterns
            json_files = []
            for obj in response['Contents']:
                key = obj['Key']
                # Check if it's a JSON file and matches our patterns
                if self._is_prediction_json_file(key):
                    json_files.append({
                        'key': key,
                        'last_modified': obj['LastModified'],
                        'size': obj['Size'],
                        'project': project
                    })

            # Sort by last modified (newest first)
            json_files.sort(key=lambda x: x['last_modified'], reverse=True)

            logger.info(
                f"Found {len(json_files)} prediction JSON files for project {project.ls_project_id}")
            return json_files

        except ClientError as e:
            logger.error(
                f"Error listing S3 objects for project {project.ls_project_id}: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error listing S3 objects for project {project.ls_project_id}: {e}")
            raise

    def get_all_json_files(self, max_keys_per_project: int = 1000) -> List[Dict]:
        """
        Get JSON files from all projects.

        Args:
            max_keys_per_project: Maximum number of keys to retrieve per project

        Returns:
            List of dictionaries containing file information from all projects
        """
        all_files = []
        projects = self.get_projects_for_monitoring()
        print(projects)
        for project in projects:
            try:
                project_files = self.get_project_json_files(
                    project, max_keys_per_project)
                all_files.extend(project_files)
            except Exception as e:
                logger.error(
                    f"Error getting files for project {project.ls_project_id}: {e}")
                continue

        # Sort all files by last modified (newest first)
        all_files.sort(key=lambda x: x['last_modified'], reverse=True)

        logger.info(
            f"Found total of {len(all_files)} prediction JSON files across all projects")
        return all_files

    def _is_prediction_json_file(self, key: str) -> bool:
        """
        Check if the S3 key matches our prediction JSON file patterns.

        Patterns:
         /<project_id>/YYYY/M/YYYY-MM-DD-HH-MM-SS.json (hierarchical structure)
        """

        # Pattern : /<project_id>/YYYY/M/YYYY-MM-DD-HH-MM-SS.json
        # ディレクトリのmonthは0埋めなし（1～12）、ファイル名は従来通り0埋め2桁
        hierarchical_pattern = r'^\d+/\d{4}/(\d{1,2})/\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\.json$'

        return (re.match(hierarchical_pattern, key) is not None)

    def download_and_parse_json(self, key: str) -> Optional[Dict]:
        """
        Download and parse a JSON file from S3.

        Args:
            key: S3 object key

        Returns:
            Parsed JSON data or None if failed
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name, Key=key)
            json_content = response['Body'].read().decode('utf-8')
            data = json.loads(json_content)

            # Validate required fields
            if not self._validate_prediction_data(data):
                logger.warning(f"Invalid prediction data in file {key}")
                return None

            return data

        except ClientError as e:
            logger.error(f"Error downloading file {key}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON in file {key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing file {key}: {e}")
            return None

    def _validate_prediction_data(self, data: Dict) -> bool:
        """
        Validate that the prediction JSON data has required fields.

        Args:
            data: Parsed JSON data

        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            'project_id',
            'predicted_at',
            'image_url',
            'results'
        ]

        # Check required fields exist
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field: {field}")
                return False

        # Validate project_id
        if not isinstance(data['project_id'], int):
            logger.warning(
                "project_id must be an integer"
            )
            return False

        # Validate predicted_at (ISO8601 format)
        try:
            datetime.fromisoformat(
                data['predicted_at'].replace('Z', '+00:00')
            )
        except (ValueError, AttributeError):
            logger.warning("predicted_at must be in ISO8601 format")
            return False

        # Validate image_url
        if not isinstance(data['image_url'], str) or not data['image_url'].startswith(
            ('http://', 'https://')
        ):
            logger.warning("image_url must be a valid HTTP/HTTPS URL")
            return False

        # Validate results
        if not isinstance(data['results'], list) or len(data['results']) == 0:
            logger.warning(
                "results must be a non-empty list"
            )
            return False

        # Validate each result object
        for result in data['results']:
            if not isinstance(result, dict) or len(result) != 1:
                logger.warning(
                    "Each result must be an object with exactly one key-value pair"
                )
                return False

            # Check that the value is a number
            label, score = list(result.items())[0]
            if not isinstance(score, (int, float)):
                logger.warning(
                    f"Score for label '{label}' must be a number"
                )
                return False

        return True

    def process_prediction_file(self, key: str, data: Dict, project: Optional[Project] = None) -> bool:
        """
        Process a prediction JSON file and store data in database.

        Args:
            key: S3 object key
            data: Parsed JSON data
            project: Optional project instance (if known from folder structure)

        Returns:
            True if successfully processed, False otherwise
        """
        try:
            with transaction.atomic():
                # Check if this file has already been processed
                if Prediction.objects.filter(json_file_name=key).exists():
                    logger.info(
                        f"File {key} already processed, skipping"
                    )
                    return True

                # Parse predicted_at timestamp
                predicted_at_str = data['predicted_at']
                if predicted_at_str.endswith('Z'):
                    predicted_at_str = predicted_at_str.replace('Z', '+00:00')
                predicted_at = datetime.fromisoformat(predicted_at_str)

                # Convert to timezone-aware datetime
                if timezone.is_naive(predicted_at):
                    predicted_at = timezone.make_aware(predicted_at)

                # Try to find the corresponding project
                if project is None:
                    try:
                        project = Project.objects.get(
                            ls_project_id=data['project_id']
                        )
                    except Project.DoesNotExist:
                        logger.warning(
                            f"Project with ls_project_id {data['project_id']} not found"
                        )

                # Create prediction record
                prediction = Prediction.objects.create(
                    ls_project_id=data['project_id'],
                    image_url=data['image_url'],
                    predicted_at=predicted_at,
                    json_file_name=key
                )

                # Create prediction results
                for result_obj in data['results']:
                    label, score = list(result_obj.items())[0]
                    PredictionResult.objects.create(
                        prediction=prediction,
                        label=label,
                        score=Decimal(str(score))
                    )

                logger.info(f"Successfully processed prediction file {key}")
                return True

        except Exception as e:
            logger.error(f"Error processing prediction file {key}: {e}")
            return False

    def monitor_project(self, project: Project, max_files: int = 10000, batch_size: int = 500) -> tuple[int, int]:
        """
        Efficiently process only new S3 files for a project, even with 10,000+ files.
        Args:
            project: Project to monitor
            max_files: Maximum number of files to process for this project
            batch_size: Number of S3 objects to check against DB in each batch
        Returns:
            Tuple of (files_processed, files_failed)
        """
        logger.info(
            f"Starting monitoring for project {project.ls_project_id} ({project.title})"
        )

        # Ensure S3 client is initialized
        if self.s3_client is None:
            self._initialize_s3_client()

        processed_count = 0
        failed_count = 0
        processed_files = 0

        paginator = self.s3_client.get_paginator('list_objects_v2')
        current_time = timezone.now()
        prefix = f"{project.ls_project_id}/{current_time.strftime('%Y')}/{int(current_time.strftime('%m'))}/"

        # Helper to batch an iterable
        def batcher(iterable, n):
            it = iter(iterable)
            while True:
                batch = list(itertools.islice(it, n))
                if not batch:
                    break
                yield batch

        for page in paginator.paginate(
            Bucket=self.bucket_name, Prefix=prefix, StartAfter=prefix
        ):
            objects = page.get('Contents', [])
            if not objects:
                print(
                    f"No objects found for project {project.ls_project_id} in S3"
                )
                continue

            # Process in batches for DB lookup
            for obj_batch in batcher(objects, batch_size):
                keys = [obj['Key'] for obj in obj_batch]

                # Query DB for existing keys in this batch
                existing_keys = set(
                    Prediction.objects.filter(
                        json_file_name__in=keys
                    ).values_list('json_file_name', flat=True)
                )
                for obj in obj_batch:
                    key = obj['Key']
                    if not self._is_prediction_json_file(key):
                        logger.info(f"Skipping non-prediction file: {key}")
                        continue
                    if key in existing_keys:
                        logger.info(
                             f"File {key} already processed, skipping"
                        )
                        continue

                    # Download and parse JSON
                    data = self.download_and_parse_json(key)
                    if data is None:
                        failed_count += 1
                        continue

                    # Process the prediction data (save to DB)
                    if self.process_prediction_file(key, data, project):
                        processed_count += 1
                    else:
                        failed_count += 1

                    processed_files += 1
                    if processed_files >= max_files:
                        logger.info(
                            f"Reached max_files limit: {max_files}"
                        )
                        return processed_count, failed_count

        logger.info(
            f"Project {project.ls_project_id} monitoring complete: "
            f"{processed_count} processed, {failed_count} failed"
        )
        return processed_count, failed_count

    def monitor_all_projects(self, max_files_per_project: int = 50) -> Dict[int, Tuple[int, int]]:
        """
        Monitor S3 for new prediction files for all projects.

        Args:
            max_files_per_project: Maximum number of files to process per project

        Returns:
            Dictionary mapping project_id to (files_processed, files_failed)
        """
        results = {}
        projects = self.get_projects_for_monitoring()

        logger.info(f"Starting monitoring for {projects.count()} projects")

        for project in projects:
            try:
                processed, failed = self.monitor_project(
                    project, max_files_per_project)
                results[project.ls_project_id] = (processed, failed)
            except Exception as e:
                logger.error(
                    f"Error monitoring project {project.ls_project_id}: {e}")
                results[project.ls_project_id] = (0, 0)

        total_processed = sum(result[0] for result in results.values())
        total_failed = sum(result[1] for result in results.values())

        logger.info(
            f"All projects monitoring complete: {total_processed} total processed, {total_failed} total failed")
        return results
