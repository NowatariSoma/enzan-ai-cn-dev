"""
Base service class that provides common functionality for all services.
This class serves as a foundation for service classes in the application.
"""

from typing import Any, Optional, Dict, List, Type
from django.db import models
from django.db.models import QuerySet


class BaseService:
    """
    Base service class that provides common functionality for all services.

    This class serves as a foundation for service classes in the application.
    It provides common methods and utilities that can be used across different services.
    """

    @staticmethod
    def get_object_or_none(model: models.Model, **kwargs) -> Optional[Any]:
        """
        Get an object from the database or return None if it doesn't exist.

        Args:
            model: The Django model class to query
            **kwargs: Filter parameters for the query

        Returns:
            Optional[Any]: The model instance if found, None otherwise
        """
        try:
            return model.objects.get(**kwargs)
        except model.DoesNotExist:
            return None

    @staticmethod
    def get_objects(model: models.Model, **kwargs) -> List[Any]:
        """
        Get a list of objects from the database.

        Args:
            model: The Django model class to query
            **kwargs: Filter parameters for the query

        Returns:
            List[Any]: List of model instances
        """
        return list(model.objects.filter(**kwargs))

    @staticmethod
    def get_queryset(model: models.Model, **kwargs) -> QuerySet:
        """
        Get a queryset from the database.

        Args:
            model: The Django model class to query
            **kwargs: Filter parameters for the query

        Returns:
            QuerySet: Django queryset of model instances
        """
        return model.objects.filter(**kwargs)

    @staticmethod
    def create_object(model: models.Model, **kwargs) -> Any:
        """
        Create a new object in the database.

        Args:
            model: The Django model class to create
            **kwargs: Fields and values for the new object

        Returns:
            Any: The created model instance
        """
        return model.objects.create(**kwargs)

    @staticmethod
    def update_object(instance: models.Model, **kwargs) -> Any:
        """
        Update an existing object in the database.

        Args:
            instance: The model instance to update
            **kwargs: Fields and values to update

        Returns:
            Any: The updated model instance
        """
        for key, value in kwargs.items():
            setattr(instance, key, value)
        instance.save()
        return instance

    @staticmethod
    def delete_object(instance: models.Model) -> None:
        """
        Delete an object from the database.

        Args:
            instance: The model instance to delete
        """
        instance.delete()

    @staticmethod
    def bulk_create_objects(model: Type[models.Model], objects: List[Dict[str, Any]]) -> List[Any]:
        """
        Create multiple objects in the database in a single query.

        Args:
            model: The Django model class to create objects for
            objects: List of dictionaries containing field values for each object

        Returns:
            List[Any]: List of created model instances
        """
        instances = [model(**obj) for obj in objects]
        return model.objects.bulk_create(instances)

    @staticmethod
    def bulk_update_objects(instances: List[models.Model], fields: List[str], batch_size: int = 100) -> None:
        """
        Update multiple objects in the database in a single query.

        Args:
            instances: List of model instances to update
            fields: List of field names to update
            batch_size: Number of objects to update in a single query (default: 100)

        Raises:
            ValueError: If instances list is empty or if no fields are provided
        """
        if not instances:
            raise ValueError("No instances provided for bulk update")
        if not fields:
            raise ValueError("No fields provided for bulk update")

        # Get the model class from the first instance
        model = instances[0].__class__

        # Perform bulk update in batches
        model.objects.bulk_update(instances, fields, batch_size=batch_size)
