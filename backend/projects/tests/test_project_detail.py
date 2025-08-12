from common import messages
from django.test import TestCase
from django.urls import reverse
from group_users.models import Group
from rest_framework.test import APIClient
from rest_framework import status
from django.contrib.auth import get_user_model
from projects.models import Project
import json
import pytest

User = get_user_model()

pytestmark = pytest.mark.django_db

class TestProjectDetailAPI:
    def test_project_detail_authenticated(self, authenticated_client, project_with_labels, project_urls):
        """Test authenticated access to project detail"""
        response = authenticated_client.get(project_urls['detail'](project_with_labels.id))
        assert response.status_code == status.HTTP_200_OK
        assert 'data' in response.data
        assert response.data['data']['title'] == 'Test Project'
        assert response.data['data']['description'] == 'Test Description'
        assert response.data['data']['ml_type_name'] == 'Classification'
        assert len(response.data['data']['labels']) == 2
        assert response.data['data']['labels'][0]['name'] == 'Label 1'
        assert response.data['data']['labels'][0]['percentage'] == 60.0

    def test_project_detail_nonexistent(self, authenticated_client, project_urls):
        """Test accessing non-existent project"""
        response = authenticated_client.get(project_urls['detail'](99999))
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_project_detail_unauthenticated(self, api_client, project_with_group, project_urls):
        """Test project detail without authentication"""
        response = api_client.get(project_urls['detail'](project_with_group.id))
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_project_detail_other_group(self, authenticated_client, other_group, project_urls):
        """Test accessing project from another group"""
        # Create a project in another group
        from projects.models import Project
        other_project = Project.objects.create(
            title='Other Group Project',
            description='Test Description',
            ml_type_name='Classification',
            organization=1,
            total_predictions_number=100,
            is_published=True,
            model_version='v1.0',
            is_draft=False,
            ls_project_id=999
        )
        other_project.group_projects.add(other_group)

        # Try to access the project
        response = authenticated_client.get(project_urls['detail'](other_project.id))
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_project_detail_invalid_token(self, api_client, project_with_group, project_urls):
        """Test project detail with invalid token"""
        api_client.credentials(HTTP_AUTHORIZATION='Bearer invalid_token')
        response = api_client.get(project_urls['detail'](project_with_group.id))
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_project_detail_with_labels(self, authenticated_client, project_with_labels, project_urls):
        """Test project detail with labels"""
        response = authenticated_client.get(project_urls['detail'](project_with_labels.id))
        assert response.status_code == status.HTTP_200_OK
        assert 'data' in response.data
        assert 'labels' in response.data['data']
        assert len(response.data['data']['labels']) == 2
        
        # Verify label data
        labels = sorted(response.data['data']['labels'], key=lambda x: x['name'])
        assert labels[0]['name'] == 'Label 1'
        assert labels[0]['percentage'] == 60.0
        assert labels[1]['name'] == 'Label 2'
        assert labels[1]['percentage'] == 40.0

    def test_project_detail_without_labels(self, authenticated_client, project_with_group, project_urls):
        """Test project detail without labels"""
        response = authenticated_client.get(project_urls['detail'](project_with_group.id))
        assert response.status_code == status.HTTP_200_OK
        assert 'data' in response.data
        assert 'labels' in response.data['data']
        assert len(response.data['data']['labels']) == 0

    def test_project_detail_response_structure(self, authenticated_client, project_with_labels, project_urls):
        """Test project detail response structure"""
        response = authenticated_client.get(project_urls['detail'](project_with_labels.id))
        assert response.status_code == status.HTTP_200_OK
        assert 'data' in response.data
        
        # Check all required fields are present
        required_fields = [
            'id', 'title', 'description', 'ml_type_name', 'organization',
            'total_predictions_number', 'is_published', 'model_version',
            'is_draft', 'labels', 'created_at', 'updated_at'
        ]
        for field in required_fields:
            assert field in response.data['data']

        # Check field types
        assert isinstance(response.data['data']['id'], int)
        assert isinstance(response.data['data']['title'], str)
        assert isinstance(response.data['data']['description'], str)
        assert isinstance(response.data['data']['ml_type_name'], str)
        assert isinstance(response.data['data']['organization'], int)
        assert isinstance(response.data['data']['total_predictions_number'], int)
        assert isinstance(response.data['data']['is_published'], bool)
        assert isinstance(response.data['data']['model_version'], str)
        assert isinstance(response.data['data']['is_draft'], bool)
        assert isinstance(response.data['data']['labels'], list)
        assert isinstance(response.data['data']['created_at'], str)
        assert isinstance(response.data['data']['updated_at'], str)


