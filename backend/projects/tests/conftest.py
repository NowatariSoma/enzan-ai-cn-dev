import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient
from rest_framework_simplejwt.tokens import RefreshToken
from group_users.models import Group
from projects.models import Project, Label
from django.urls import reverse

User = get_user_model()

@pytest.fixture
def api_client():
    """Fixture for API client"""
    return APIClient()

@pytest.fixture
def staff_user(db):
    """Fixture for staff user"""
    user = User.objects.create_user(
        email='staff@example.com',
        password='staffpass123',
        first_name='Staff',
        last_name='User',
        role=User.Role.STAFF
    )
    return user

@pytest.fixture
def admin_user(db):
    """Fixture for admin user"""
    user = User.objects.create_superuser(
        email='admin@example.com',
        password='adminpass123',
        first_name='Admin',
        last_name='User'
    )
    return user

@pytest.fixture
def other_staff_user(db):
    """Fixture for another staff user"""
    user = User.objects.create_user(
        email='other@example.com',
        password='testpass123',
        first_name='Other',
        last_name='User',
        role=User.Role.STAFF
    )
    return user

@pytest.fixture
def authenticated_client(api_client, staff_user):
    """Fixture for authenticated API client with staff user"""
    refresh = RefreshToken.for_user(staff_user)
    api_client.credentials(HTTP_AUTHORIZATION=f'Bearer {refresh.access_token}')
    return api_client

@pytest.fixture
def admin_client(api_client, admin_user):
    """Fixture for authenticated API client with admin user"""
    refresh = RefreshToken.for_user(admin_user)
    api_client.credentials(HTTP_AUTHORIZATION=f'Bearer {refresh.access_token}')
    return api_client

@pytest.fixture
def test_group(db, staff_user):
    """Fixture for test group"""
    group = Group.objects.create(name='Test Group')
    group.user_ids.add(staff_user)
    return group

@pytest.fixture
def other_group(db):
    """Fixture for another group"""
    return Group.objects.create(name='Other Group')

@pytest.fixture
def project_with_group(db, test_group):
    """Fixture for project with test group"""
    project = Project.objects.create(
        title='Test Project',
        description='Test Description',
        ml_type_name='Classification',
        organization=1,
        total_predictions_number=100,
        is_published=True,
        model_version='v1.0',
        is_draft=False,
        ls_project_id=123
    )
    project.group_projects.add(test_group)
    return project

@pytest.fixture
def project_with_labels(db, project_with_group):
    """Fixture for project with labels"""
    Label.objects.create(
        project=project_with_group,
        name='Label 1',
        percentage=60.0
    )
    Label.objects.create(
        project=project_with_group,
        name='Label 2',
        percentage=40.0
    )
    return project_with_group

@pytest.fixture
def multiple_projects(db, test_group):
    """Fixture for multiple projects"""
    projects = []
    for i in range(20):
        project = Project.objects.create(
            title=f'Project {i}',
            description=f'Description {i}',
            ml_type_name='Classification',
            organization=1,
            total_predictions_number=100,
            is_published=True,
            model_version='v1.0',
            is_draft=False,
            ls_project_id=100 + i
        )
        project.group_projects.add(test_group)
        projects.append(project)
    return projects

@pytest.fixture
def project_urls():
    """Fixture for project URLs"""
    return {
        'list': reverse('project-list'),
        'detail': lambda pk: reverse('project-detail', kwargs={'pk': pk}),
        'login': reverse('login'),
    } 