from django.test import TestCase
from django.urls import reverse
from group_users.models import Group
from rest_framework.test import APIClient
from rest_framework import status
from django.contrib.auth import get_user_model
from projects.models import Project
import json
from common import messages
from rest_framework_simplejwt.tokens import RefreshToken
import pytest

User = get_user_model()

pytestmark = pytest.mark.django_db

class TestProjectListAPI:
    def test_project_list_authenticated(self, authenticated_client, project_with_group, project_urls):
        """Test authenticated access to project list"""
        response = authenticated_client.get(project_urls['list'])
        assert response.status_code == status.HTTP_200_OK
        assert 'data' in response.data
        assert 'projects' in response.data['data']
        assert 'pagination' in response.data['data']
        assert len(response.data['data']['projects']) == 1
        assert response.data['data']['projects'][0]['title'] == 'Test Project'

    def test_project_list_search(self, authenticated_client, multiple_projects, project_urls):
        """Test project list search functionality"""
        # Search by title
        response = authenticated_client.get(project_urls['list'], {'search': 'Project 19'})
        assert response.status_code == status.HTTP_200_OK
        assert 'data' in response.data
        assert 'projects' in response.data['data']
        assert len(response.data['data']['projects']) == 1
        assert response.data['data']['projects'][0]['title'] == 'Project 19'

    def test_project_list_sorting(self, authenticated_client, multiple_projects, project_urls):
        """Test project list sorting"""
        # Sort by title ascending
        response = authenticated_client.get(project_urls['list'], {'order_by': 'title'})
        assert response.status_code == status.HTTP_200_OK
        titles = [p['title'] for p in response.data['data']['projects']]
        assert titles[0] == 'Project 19'

    def test_project_list_pagination(self, authenticated_client, multiple_projects, project_urls):
        """Test project list pagination"""
        # Test default pagination
        response = authenticated_client.get(project_urls['list'])
        assert response.status_code == status.HTTP_200_OK
        assert 'data' in response.data
        assert 'pagination' in response.data['data']
        pagination = response.data['data']['pagination']
        assert 'current_page' in pagination
        assert 'total_pages' in pagination
        assert 'total_items' in pagination
        assert 'items_per_page' in pagination

    def test_project_list_filtering(self, authenticated_client, multiple_projects, project_urls):
        """Test project list filtering"""
        # Filter by is_published
        response = authenticated_client.get(project_urls['list'], {'is_published': 'true'})
        assert response.status_code == status.HTTP_200_OK
        assert all(p['is_published'] for p in response.data['data']['projects'])

    def test_project_list_group_filtering(self, authenticated_client, project_with_group, other_group, project_urls):
        """Test project list filtering by group"""
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

        # Should only see projects from user's group
        response = authenticated_client.get(project_urls['list'])
        assert response.status_code == status.HTTP_200_OK
        assert 'data' in response.data
        assert 'projects' in response.data['data']
        assert len(response.data['data']['projects']) == 1
        assert response.data['data']['projects'][0]['title'] == 'Test Project'


    def test_project_list_unauthenticated(self, api_client, project_urls):
        """Test project list without authentication"""
        response = api_client.get(project_urls['list'])
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

