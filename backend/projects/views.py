import logging
from common import messages
from common.ordering_filter import CustomOrderingFilter
from projects.throttle import ProjectsRateThrottle
from rest_framework import filters
from rest_framework.generics import ListAPIView, RetrieveAPIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.exceptions import PermissionDenied, NotFound
from common.pagination import CustomPagination
from common.format_response import create_response
from .models import Project
from .serializers import (
    ProjectSerializer, ProjectDetailSerializer,
)
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from common.swagger import standard_response_schema

from rest_framework import status
logger = logging.getLogger(__name__)


class ProjectApiView(ListAPIView):
    """
    ViewSet for listing projects.
    Supports pagination, search, and sorting as per specification.
    Rate limited to 1000 requests per hour per user.
    Projects are filtered based on user's group access.
    """
    serializer_class = ProjectSerializer
    permission_classes = [IsAuthenticated]
    throttle_classes = [ProjectsRateThrottle]
    pagination_class = CustomPagination
    filter_backends = [filters.SearchFilter, CustomOrderingFilter]
    search_fields = ['title', 'description', 'ml_type_name']
    ordering_fields = ['created_at', 'updated_at', 'title']
    ordering = ['-created_at']

    def get_queryset(self):
        user = self.request.user
        user_groups = user.group_users.all()
        return Project.objects.filter(
            group_projects__in=user_groups
        ).distinct()

    @swagger_auto_schema(
        operation_description="Get a list of projects with pagination, search, and sorting",
        manual_parameters=[
            openapi.Parameter(
                'page',
                openapi.IN_QUERY,
                description="Page number (default: 1)",
                type=openapi.TYPE_INTEGER,
                required=False
            ),
            openapi.Parameter(
                'limit',
                openapi.IN_QUERY,
                description="Number of items per page (default: 20, max: 100)",
                type=openapi.TYPE_INTEGER,
                required=False
            ),
            openapi.Parameter(
                'search',
                openapi.IN_QUERY,
                description="Search query (targets title, description, and ml_type_name fields)",
                type=openapi.TYPE_STRING,
                required=False
            ),
            openapi.Parameter(
                'sort_by',
                openapi.IN_QUERY,
                description="Sort field (created_at, updated_at, title)",
                type=openapi.TYPE_STRING,
                required=False,
                enum=['created_at', 'updated_at', 'title']
            ),
            openapi.Parameter(
                'sort_order',
                openapi.IN_QUERY,
                description="Sort order (asc, desc) (default: desc)",
                type=openapi.TYPE_STRING,
                required=False,
                enum=['asc', 'desc']
            ),
        ],
        responses={
            200: openapi.Response(
                description="List of projects",
                schema=standard_response_schema(
                    data_properties={
                        "projects": openapi.Schema(
                            type=openapi.TYPE_ARRAY,
                            items=openapi.Schema(
                                type=openapi.TYPE_OBJECT,
                                properties={
                                    'id': openapi.Schema(type=openapi.TYPE_INTEGER, example=123),
                                    'title': openapi.Schema(type=openapi.TYPE_STRING, example="Medical X-Ray Classification"),
                                    'description': openapi.Schema(type=openapi.TYPE_STRING, example="Deep learning model for automated medical imaging analysis"),
                                    'created_at': openapi.Schema(type=openapi.TYPE_STRING, format='date-time', example="2024-01-15T09:30:00Z"),
                                    'updated_at': openapi.Schema(type=openapi.TYPE_STRING, format='date-time', example="2024-01-20T14:22:00Z"),
                                    'organization': openapi.Schema(type=openapi.TYPE_INTEGER, example=1),
                                    'total_predictions_number': openapi.Schema(type=openapi.TYPE_INTEGER, example=1542),
                                    'is_published': openapi.Schema(type=openapi.TYPE_BOOLEAN, example=True),
                                    'model_version': openapi.Schema(type=openapi.TYPE_STRING, example="Resnet v2.1.3"),
                                    'is_draft': openapi.Schema(type=openapi.TYPE_BOOLEAN, example=False),
                                    'ml_type_name': openapi.Schema(type=openapi.TYPE_STRING, example="Soil Property Prediction")
                                }
                            )
                        ),
                        "pagination": openapi.Schema(
                            type=openapi.TYPE_OBJECT,
                            properties={
                                "current_page": openapi.Schema(type=openapi.TYPE_INTEGER, example=1),
                                "total_pages": openapi.Schema(type=openapi.TYPE_INTEGER, example=3),
                                "total_items": openapi.Schema(type=openapi.TYPE_INTEGER, example=5),
                                "items_per_page": openapi.Schema(type=openapi.TYPE_INTEGER, example=20),
                                "has_next": openapi.Schema(type=openapi.TYPE_BOOLEAN, example=False),
                                "has_previous": openapi.Schema(type=openapi.TYPE_BOOLEAN, example=False)
                            }
                        )
                    },
                    message_example="Projects list has been fetched successfully.",
                    status_code_example=200
                )
            ),
            401: openapi.Response(
                description="Authentication required",
                schema=standard_response_schema(
                    error_code_example="AUTHENTICATION_REQUIRED",
                    message_example="Authentication credentials were not provided.",
                    status_code_example=401
                )
            ),
            400: openapi.Response(
                description="Invalid request parameters",
                schema=standard_response_schema(
                    error_code_example="INVALID_PARAMETERS",
                    message_example="Invalid request parameters",
                    status_code_example=400
                )
            ),
            500: openapi.Response(
                description="Internal server error",
                schema=standard_response_schema(
                    error_code_example="INTERNAL_SERVER_ERROR",
                    message_example="An internal server error occurred",
                    status_code_example=500
                )
            )
        }
    )
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)


class ProjectDetailApiView(RetrieveAPIView):
    serializer_class = ProjectDetailSerializer
    permission_classes = [IsAuthenticated]
    throttle_classes = [ProjectsRateThrottle]

    def get_object(self):
        try:
            obj = Project.objects.get(pk=self.kwargs['pk'])
        except Project.DoesNotExist:
            raise NotFound(messages.NOT_FOUND)

        user_groups = self.request.user.group_users.all()
        if not obj.group_projects.filter(id__in=user_groups.values_list('id', flat=True)).exists():
            raise PermissionDenied(messages.PERMISSION_DENIED)

        return obj

    @swagger_auto_schema(
        operation_description="Get detailed information about a specific project including its labels and latest prediction data. Rate limited to 1000 requests per hour per user. Access is restricted to users who belong to groups associated with this project.",
        manual_parameters=[
            openapi.Parameter(
                'pk',
                openapi.IN_PATH,
                description="Project ID (primary key)",
                type=openapi.TYPE_INTEGER,
                required=True,
                example=123
            ),
        ],
        responses={
            200: openapi.Response(
                description="Project details with labels and latest prediction",
                schema=standard_response_schema(
                    data_properties={
                        "project": openapi.Schema(
                            type=openapi.TYPE_OBJECT,
                            properties={
                                'id': openapi.Schema(type=openapi.TYPE_INTEGER, example=123, description="Unique project identifier"),
                                'title': openapi.Schema(type=openapi.TYPE_STRING, example="Medical X-Ray Classification", description="Project title"),
                                'description': openapi.Schema(type=openapi.TYPE_STRING, example="Deep learning model for automated medical imaging analysis", description="Detailed project description"),
                                'created_at': openapi.Schema(type=openapi.TYPE_STRING, format='date-time', example="2024-01-15T09:30:00Z", description="Project creation timestamp"),
                                'updated_at': openapi.Schema(type=openapi.TYPE_STRING, format='date-time', example="2024-01-20T14:22:00Z", description="Last update timestamp"),
                                'organization': openapi.Schema(type=openapi.TYPE_INTEGER, example=1, description="Organization ID"),
                                'total_predictions_number': openapi.Schema(type=openapi.TYPE_INTEGER, example=1542, description="Total number of predictions made for this project"),
                                'is_published': openapi.Schema(type=openapi.TYPE_BOOLEAN, example=True, description="Whether the project is published"),
                                'model_version': openapi.Schema(type=openapi.TYPE_STRING, example="Resnet v2.1.3", description="ML model version used"),
                                'is_draft': openapi.Schema(type=openapi.TYPE_BOOLEAN, example=False, description="Whether the project is in draft status"),
                                'ml_type_name': openapi.Schema(type=openapi.TYPE_STRING, example="Soil Property Prediction", description="Type of machine learning task"),
                                'labels': openapi.Schema(
                                    type=openapi.TYPE_ARRAY,
                                    description="Classification labels with their percentages",
                                    items=openapi.Schema(
                                        type=openapi.TYPE_OBJECT,
                                        properties={
                                            'name': openapi.Schema(type=openapi.TYPE_STRING, example="Normal", description="Label name"),
                                            'percentage': openapi.Schema(type=openapi.TYPE_NUMBER, example=75.5, description="Percentage value (0-100)")
                                        }
                                    ),
                                    example=[
                                        {"name": "Normal", "percentage": 75.5},
                                        {"name": "Abnormal", "percentage": 24.5}
                                    ]
                                ),
                                "prediction": openapi.Schema(
                                    type=openapi.TYPE_OBJECT,
                                    description="Latest prediction data for the project",
                                    properties={
                                        'id': openapi.Schema(type=openapi.TYPE_INTEGER, example=456, description="Prediction ID"),
                                        'image_url': openapi.Schema(type=openapi.TYPE_STRING, example="https://s3.amazonaws.com/bucket/image.jpg", description="S3 URL of the predicted image"),
                                        'predicted_at': openapi.Schema(type=openapi.TYPE_STRING, format='date-time', example="2024-01-20T15:30:00Z", description="Timestamp when prediction was made"),
                                        'created_at': openapi.Schema(type=openapi.TYPE_STRING, format='date-time', example="2024-01-20T15:30:00Z", description="Prediction record creation timestamp"),
                                        'updated_at': openapi.Schema(type=openapi.TYPE_STRING, format='date-time', example="2024-01-20T15:30:00Z", description="Prediction record last update timestamp"),
                                        'results': openapi.Schema(
                                            type=openapi.TYPE_ARRAY,
                                            description="Individual prediction results with scores",
                                            items=openapi.Schema(
                                                type=openapi.TYPE_OBJECT,
                                                properties={
                                                    'id': openapi.Schema(type=openapi.TYPE_INTEGER, example=789, description="Result ID"),
                                                    'label': openapi.Schema(type=openapi.TYPE_STRING, example="Normal", description="Predicted class/label"),
                                                    'score': openapi.Schema(type=openapi.TYPE_NUMBER, example=85.50, description="Prediction confidence score (0-100)"),
                                                    'created_at': openapi.Schema(type=openapi.TYPE_STRING, format='date-time', example="2024-01-20T15:30:00Z", description="Result creation timestamp")
                                                }
                                            ),
                                            example=[
                                                {"id": 789, "label": "Normal", "score": 85.50,
                                                 "created_at": "2024-01-20T15:30:00Z"},
                                                {"id": 790, "label": "Abnormal", "score": 14.50,
                                                 "created_at": "2024-01-20T15:30:00Z"}
                                            ]
                                        )
                                    }
                                )
                            }
                        ),

                    },
                    message_example="Project details has been fetched successfully.",
                    status_code_example=200
                )
            ),
            401: openapi.Response(
                description="Authentication required",
                schema=standard_response_schema(
                    error_code_example="AUTHENTICATION_REQUIRED",
                    message_example="Authentication credentials were not provided.",
                    status_code_example=401
                )
            ),
            403: openapi.Response(
                description="Permission denied - User does not have access to this project",
                schema=standard_response_schema(
                    error_code_example="PERMISSION_DENIED",
                    message_example="You do not have permission to access this project.",
                    status_code_example=403
                )
            ),
            404: openapi.Response(
                description="Project not found",
                schema=standard_response_schema(
                    error_code_example="NOT_FOUND",
                    message_example="Project not found.",
                    status_code_example=404
                )
            ),
            500: openapi.Response(
                description="Internal server error",
                schema=standard_response_schema(
                    error_code_example="INTERNAL_SERVER_ERROR",
                    message_example="An internal server error occurred",
                    status_code_example=500
                )
            )
        }
    )
    def retrieve(self, request, *args, **kwargs):
        """
        Retrieve project details with latest prediction data.
        
        Returns:
            Response: Serialized project data with prediction information
        """
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        
        return create_response(
            message=messages.SUCCESSFUL,
            status_code=status.HTTP_200_OK,
            data=serializer.data
        )
