from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response


class CustomPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "limit"
    max_page_size = 100
    page_query_param = "page"
    include_standard_format = True  # Flag to control response format

    def get_paginated_response(self, data, message="Data has been fetched successfully."):
        pagination_data = {
            "current_page": self.page.number,
            "total_pages": self.page.paginator.num_pages,
            "total_items": self.page.paginator.count,
            "items_per_page": self.get_page_size(self.request),
            "has_next": self.page.has_next(),
            "has_previous": self.page.has_previous()
        }

        if self.include_standard_format:
            return Response({
                "message": message,
                "status_code": 200,
                "data": {
                    "projects": data,
                    "pagination": pagination_data
                }
            })
        
        return Response({
            "projects": data,
            "pagination": pagination_data
        })

    def get_paginated_response_schema(self, schema):
        return openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                "message": openapi.Schema(
                    type=openapi.TYPE_STRING, 
                    example="Projects list has been fetched successfully."
                ),
                "status_code": openapi.Schema(
                    type=openapi.TYPE_INTEGER, 
                    example=200
                ),
                "data": openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        "projects": schema,
                        "pagination": openapi.Schema(
                            type=openapi.TYPE_OBJECT,
                            properties={
                                "current_page": openapi.Schema(type=openapi.TYPE_INTEGER, example=1),
                                "total_pages": openapi.Schema(type=openapi.TYPE_INTEGER, example=3),
                                "total_items": openapi.Schema(type=openapi.TYPE_INTEGER, example=5),
                                "items_per_page": openapi.Schema(type=openapi.TYPE_INTEGER, example=20),
                                "has_next": openapi.Schema(type=openapi.TYPE_BOOLEAN, example=False),
                                "has_previous": openapi.Schema(type=openapi.TYPE_BOOLEAN, example=False)
                            },
                        ),
                    },
                ),
            },
        )

    def get_paginated_meta_response(self):
        return Response(
            {
                "pagination": {
                    "page": self.page.number,
                    "per_page": self.page_size,
                    "total_pages": self.page.paginator.num_pages,
                    "total_items": self.page.paginator.count,
                }
            }
        )
