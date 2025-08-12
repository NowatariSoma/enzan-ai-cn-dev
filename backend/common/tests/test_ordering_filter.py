from common.ordering_filter import CustomOrderingFilter
import pytest
from unittest.mock import Mock, patch
from rest_framework.request import Request
from rest_framework.test import APIRequestFactory

# Fixtures
@pytest.fixture
def factory():
    return APIRequestFactory()

@pytest.fixture
def filter_instance():
    return CustomOrderingFilter()

@pytest.fixture
def view():
    return Mock()

@pytest.fixture
def queryset():
    return Mock()

@pytest.fixture
def mock_super_get_ordering():
    """Fixture to mock the parent class's get_ordering method"""
    with patch('rest_framework.filters.OrderingFilter.get_ordering') as mock:
        yield mock

# Test cases
def test_get_ordering_with_default_sort_order(factory, filter_instance, view, queryset, mock_super_get_ordering):
    """Test with default sort_order (desc)"""
    request = factory.get('/api/test/?sort_by=name')
    request = Request(request)
    
    # Mock the parent class's get_ordering to return ['name']
    mock_super_get_ordering.return_value = ['name']
    result = filter_instance.get_ordering(request, queryset, view)
    assert result == ['-name']

def test_get_ordering_with_asc_sort_order(factory, filter_instance, view, queryset, mock_super_get_ordering):
    """Test with explicit asc sort_order"""
    request = factory.get('/api/test/?sort_by=name&sort_order=asc')
    request = Request(request)
    
    mock_super_get_ordering.return_value = ['name']
    result = filter_instance.get_ordering(request, queryset, view)
    assert result == ['name']

@pytest.mark.parametrize("query_params,expected_fields,expected_result", [
    # Test multiple fields with default sort_order (desc)
    ("sort_by=name,created_at", ['name', 'created_at'], ['-name', '-created_at']),
    # Test fields with explicit sort_order=asc
    ("sort_by=name,created_at&sort_order=asc", ['name', 'created_at'], ['name', 'created_at']),
    # Test fields with explicit sort_order=asc and minus sign
    ("sort_by=-name,-created_at&sort_order=asc", ['-name', '-created_at'], ['name', 'created_at']),
    # Test fields that already have minus sign
    ("sort_by=-name,created_at", ['-name', 'created_at'], ['-name', '-created_at']),
])
def test_get_ordering_with_different_field_combinations(
    factory, filter_instance, view, queryset, mock_super_get_ordering,
    query_params, expected_fields, expected_result
):
    """Test different combinations of field ordering"""
    request = factory.get(f'/api/test/?{query_params}')
    request = Request(request)
    
    mock_super_get_ordering.return_value = expected_fields
    result = filter_instance.get_ordering(request, queryset, view)
    assert result == expected_result

def test_get_ordering_with_no_ordering(factory, filter_instance, view, queryset, mock_super_get_ordering):
    """Test when no ordering is specified"""
    request = factory.get('/api/test/')
    request = Request(request)
    
    mock_super_get_ordering.return_value = None
    result = filter_instance.get_ordering(request, queryset, view)
    assert result is None

@pytest.mark.parametrize("sort_order", ["asc", "desc", "ASC", "DESC", "Asc", "Desc"])
def test_get_ordering_case_insensitive_sort_order(
    factory, filter_instance, view, queryset, mock_super_get_ordering, sort_order
):
    """Test case insensitivity of sort_order parameter"""
    request = factory.get(f'/api/test/?sort_by=name&sort_order={sort_order}')
    request = Request(request)
    
    mock_super_get_ordering.return_value = ['name']
    result = filter_instance.get_ordering(request, queryset, view)
    
    expected = ['name'] if sort_order.lower() == 'asc' else ['-name']
    assert result == expected 