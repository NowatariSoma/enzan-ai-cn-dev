from typing import Any

from common import messages
from common.format_response import create_response
from django.http import Http404
from django.utils.translation import gettext_lazy as _
from rest_framework import status
from rest_framework.exceptions import (
    NotAuthenticated,
    PermissionDenied,
    ValidationError,
    NotFound,
)
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework.views import exception_handler


def handle_exception(exc: Exception, context: Any) -> Any:
    """
    Handle exceptions and return a response.
    """
    response = exception_handler(exc, context)
    
    if isinstance(exc, TokenError):
        return create_response(
            message=messages.REFRESH_TOKEN_INVALID,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code=exc.__class__.__name__,
            error_messages=str(exc),
        )

    if response is None:
        return create_response(
            message=messages.INTERNAL_SERVER_ERROR,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=exc.__class__.__name__,
            error_messages=str(exc),
        )

    if response.status_code == 400 and isinstance(exc, ValidationError):
        return create_response(
            message=messages.VALIDATION_FAILED,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code=exc.__class__.__name__,
            error_messages=exc.detail,
        )

    if response.status_code == 404 and isinstance(exc, Http404) or isinstance(exc, NotFound):
        return create_response(
            message=messages.NOT_FOUND,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=exc.__class__.__name__,
            error_messages=str(exc),
        )

    if response.status_code == 401 and isinstance(exc, NotAuthenticated):
        return create_response(
            message=messages.AUTHENTICATION_REQUIRED,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code=exc.__class__.__name__,
            error_messages=str(exc),
        )

    if response.status_code == 403 and isinstance(exc, PermissionDenied):
        return create_response(
            message=messages.FORBIDDEN,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code=exc.__class__.__name__,
            error_messages=str(exc),
        )

    return response
