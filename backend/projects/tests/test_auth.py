import pytest
from rest_framework import status
from rest_framework_simplejwt.tokens import RefreshToken
from django.urls import reverse
from datetime import timedelta

pytestmark = pytest.mark.django_db


class TestAuthentication:
    def test_login_success(self, api_client, staff_user, project_urls):
        """Test successful login and token generation"""
        response = api_client.post(project_urls['login'], {
            'email': 'staff@example.com',
            'password': 'staffpass123'
        })
        assert response.status_code == status.HTTP_200_OK
        assert 'data' in response.data
        assert 'access' in response.data['data']
        assert 'refresh' in response.data['data']
        assert 'user' in response.data['data']

    def test_login_invalid_credentials(self, api_client, project_urls):
        """Test login with invalid credentials"""
        response = api_client.post(project_urls['login'], {
            'email': 'staff@example.com',
            'password': 'wrongpass'
        })
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_authentication_required(self, api_client, project_urls):
        """Test that authentication is required for protected endpoints"""
        # Try to access project list without authentication
        response = api_client.get(project_urls['list'])
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_invalid_token(self, api_client, project_urls):
        """Test API behavior with invalid token"""
        api_client.credentials(HTTP_AUTHORIZATION='Bearer invalid_token')
        response = api_client.get(project_urls['list'])
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_expired_token(self, api_client, staff_user, project_urls):
        from rest_framework_simplejwt.tokens import AccessToken
        from datetime import timedelta
        from django.utils import timezone
        # Generate token and force expiry
        token = AccessToken.for_user(staff_user)
        token.set_exp(from_time=timezone.now(),
                      lifetime=timedelta(seconds=-60))

        api_client.credentials(HTTP_AUTHORIZATION=f'Bearer {str(token)}')
        response = api_client.get(project_urls['list'])

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_cross_group_access(self, api_client, staff_user, other_group, project_urls):
        """Test that users cannot access projects from other groups"""
        # Create a project in another group
        from projects.models import Project
        project = Project.objects.create(
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
        project.group_projects.add(other_group)

        # Login and try to access the project
        login_response = api_client.post(project_urls['login'], {
            'email': 'staff@example.com',
            'password': 'staffpass123'
        })
        api_client.credentials(
            HTTP_AUTHORIZATION=f'Bearer {login_response.data["data"]["access"]}')

        response = api_client.get(project_urls['detail'](project.id))
        assert response.status_code == status.HTTP_403_FORBIDDEN
