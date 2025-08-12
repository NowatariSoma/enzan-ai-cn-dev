from datetime import datetime, timedelta
import logging
import random
import string
from typing import Optional, Tuple

from common import constant, messages
from common.format_response import create_response
from common.helper import decode_base64
from common.mail_services import generate_url, send_email_to_user
from common.swagger import standard_response_schema
from django.conf import settings
from django.contrib.auth import authenticate
from django.contrib.auth.tokens import default_token_generator
from django.utils.translation import gettext_lazy as _
from django.template.loader import render_to_string

from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from rest_framework import status
from rest_framework.exceptions import ValidationError
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.serializers import (TokenBlacklistSerializer
                                                  )
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import (TokenBlacklistView,
                                            TokenObtainPairView)

from .models import Invitation, User, Token
from .serializers import (ForgotPasswordSerializer, LoginSerializer,
                          RegistrationSerializer, ResetPasswordSerializer)
from .services import InvitationService, TokenService, UserService

logger = logging.getLogger(__name__)


class ConfirmRegistrationAPIView(APIView):
    """
    API View for confirming user registration and handling password set.

    This view handles both GET requests for validating registration tokens
    and POST requests for completing the registration process.
    """

    token_service = TokenService()
    invitation_service = InvitationService()

    def _generate_random_password(self, length=12):
        characters = string.ascii_letters + string.digits
        return "".join(random.choice(characters) for _ in range(length))

    def _validate_registration_data(
        self,
        token: str,
        invitation_id: int
    ) -> Tuple[bool, Optional[Invitation]]:
        """
        Validate registration data including token and invitation.

        Args:
            token (str): The registration token
            invitation_id (int): The invitation ID

        Returns:
            Tuple[bool, Optional[Invitation]]: Tuple containing validation result and invitation
        """
        invitation = self.invitation_service.get_invitation(invitation_id)
        if not self.invitation_service.validate_invitation(invitation):
            return False, None
        key = f"{constant.INVITE_USER}:{invitation.email}"
        if not self.token_service.validate_token(token, key):
            return False, None

        return True, invitation

    @swagger_auto_schema(
        operation_description="Confirm invitation url is valid",
        manual_parameters=[
            openapi.Parameter(
                "uid64",
                openapi.IN_QUERY,
                description="Invitation (base64 encoded)",
                type=openapi.TYPE_STRING,
            ),
            openapi.Parameter(
                "token",
                openapi.IN_QUERY,
                description="Token for confirmation",
                type=openapi.TYPE_STRING,
            ),
        ],
        responses={
            200: openapi.Response(description="The invitation is valid"),
            401: openapi.Response(description="The invitation is invalid or expired"),
        },
    )
    def get(self, request: Request) -> Response:
        """
        Validate registration token and invitation.

        Args:
            request (Request): The HTTP request

        Returns:
            Response: HTTP response indicating validation result
        """
        try:
            token = request.GET.get("token")
            uidb64 = request.GET.get("uid64")
            invitation_id = decode_base64(uidb64)
            is_valid, _ = self._validate_registration_data(
                token, invitation_id)

            if is_valid:
                return create_response(
                    messages.TOKEN_IS_VALID,
                    status_code=status.HTTP_200_OK,
                )

            return create_response(
                messages.INVALID_REGISTRATION_OR_EXPIRED_TOKEN,
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

        except Exception as e:
            return create_response(
                str(e),
                status_code=status.HTTP_400_BAD_REQUEST,
            )

    @swagger_auto_schema(
        operation_description="Complete user registration and set password",
        request_body=RegistrationSerializer,
        responses={
            200: openapi.Response(description="Registration completed successfully"),
            400: openapi.Response(description="Invalid parameters or validation failed"),
            401: openapi.Response(description="Invalid or expired token"),
        },
    )
    def post(self, request: Request) -> Response:
        """
        Complete user registration by creating user account and setting password.

        Args:
            request (Request): The HTTP request containing registration data

        Returns:
            Response: HTTP response indicating registration result
        """

        serializer = RegistrationSerializer(data=request.data)
        if not serializer.is_valid():
            return create_response(
                messages.VALIDATION_FAILED,
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code="VALIDATION_FAILED",
                error_messages=serializer.errors,
            )

        token = serializer.validated_data["token"]
        new_password = serializer.validated_data["new_password"]
        invitation_id = decode_base64(serializer.validated_data["uidb64"])

        is_valid, invitation = self._validate_registration_data(
            token, invitation_id)
        if not is_valid:
            return create_response(
                messages.INVALID_REGISTRATION_OR_EXPIRED_TOKEN,
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

        user = User.objects.create_user(
            email=invitation.email,
            first_name=invitation.first_name,
            last_name=invitation.last_name,
            role=User.Role.STAFF,
            password=self._generate_random_password(),  # Password will be set later
        )
        user.set_password(new_password)
        user.save()

        invitation.registered = True
        invitation.save()

        # Mark token as used
        self.token_service.mark_token_as_used(token)
        logger.info(
            f"User {user.email} registered successfully with invitation ID {invitation_id}"
        )
        return create_response(
            messages.REGISTRATION_SUCCESSFUL,
            status_code=status.HTTP_200_OK,
        )


class LoginAPIView(TokenObtainPairView):
    """API View for handling user login authentication."""

    serializer_class = LoginSerializer

    @swagger_auto_schema(
        operation_description="Login API for user authentication",
        request_body=LoginSerializer,
        responses={
            status.HTTP_200_OK: openapi.Response(
                "Login successful",
                schema=standard_response_schema({
                    "refresh": openapi.Schema(type=openapi.TYPE_STRING),
                    "access": openapi.Schema(type=openapi.TYPE_STRING),
                    "user": openapi.Schema(
                        type=openapi.TYPE_OBJECT,
                        properties={
                            "email": openapi.Schema(type=openapi.TYPE_STRING),
                            "first_name": openapi.Schema(type=openapi.TYPE_STRING),
                            "last_name": openapi.Schema(type=openapi.TYPE_STRING),
                            "group": openapi.Schema(
                                type=openapi.TYPE_ARRAY,
                                items=openapi.Schema(type=openapi.TYPE_STRING),
                            ),
                        },
                    ),
                }, message_example=messages.LOGIN_SUCCESSFUL, status_code_example=status.HTTP_200_OK),
            ),
            status.HTTP_401_UNAUTHORIZED: openapi.Response(
                "Login Failed",
                schema=standard_response_schema(
                    error_code_example="INVALID_CREDENTIALS",
                    message_example=messages.INVALID_CREDENTIALS,
                    status_code_example=status.HTTP_401_UNAUTHORIZED
                ),
            ),
        },
    )
    def post(self, request: Request, *args, **kwargs) -> Response:
        """
        Handle user login authentication.

        Args:
            request: The HTTP request containing login credentials

        Returns:
            Response: HTTP response with authentication tokens and user data
        """

        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid(raise_exception=True):
            return create_response(
                messages.VALIDATION_FAILED,
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code="VALIDATION_FAILED",
                error_messages=serializer.errors,
            )

        email = serializer.validated_data["email"]
        password = serializer.validated_data["password"]

        user = authenticate(request, email=email, password=password)

        if user is not None:
            refresh = RefreshToken.for_user(user)
            access = str(refresh.access_token)

            return create_response(
                messages.LOGIN_SUCCESSFUL,
                status_code=status.HTTP_200_OK,
                data={
                    "refresh": str(refresh),
                    "access": access,
                    "user": {
                        "email": user.email,
                        "first_name": user.first_name,
                        "last_name": user.last_name,
                        "group": [group.name for group in user.group_users.all()],
                    },
                },
            )
        return create_response(
            message=messages.INVALID_CREDENTIALS,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="INVALID_CREDENTIALS",
        )


class LogoutAPIView(TokenBlacklistView):
    """API View for handling user logout and token blacklisting."""

    serializer_class = TokenBlacklistSerializer

    @swagger_auto_schema(
        operation_description="Logout API for user authentication",
        responses={
            status.HTTP_200_OK: openapi.Response(
                description="Logout successful",
                schema=standard_response_schema(
                    message_example=messages.LOGOUT_SUCCESSFUL,
                    status_code_example=status.HTTP_200_OK
                ),
            ),
            status.HTTP_401_UNAUTHORIZED: openapi.Response(
                description="Logout failed",
                schema=standard_response_schema(
                    error_code_example="LOGOUT_FAILED",
                    message_example=messages.LOGOUT_FAILED,
                    status_code_example=status.HTTP_401_UNAUTHORIZED,
                ),
            ),
        },
    )
    def post(self, request: Request) -> Response:
        """
        Handle user logout by blacklisting the refresh token.

        Args:
            request: The HTTP request containing the refresh token

        Returns:
            Response: HTTP response indicating logout status
        """
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        return create_response(
            message=messages.LOGOUT_SUCCESSFUL,
            status_code=status.HTTP_200_OK
        )


class ForgotPasswordAPIView(APIView):
    """API View for handling forgot password requests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_service = UserService()

    def _send_reset_email(self, request: any, user: User, token: str, forgot_password_url: str) -> None:
        subject = constant.FORGOT_PASSWORD_EMAIL_SUBJECT
        template_mail = constant.TEMPLATE_FORGOT_PASSWORD
        forgot_password_url = settings.FORGOT_PASSWORD_URL

        expiry_date = datetime.now() + timedelta(minutes=30)
        Token.objects.create(
            key=f"{constant.FORGOT_PASSWORD_KEY}:{user.email}",
            value=token,
            is_used=False,
            expires_at=expiry_date,
        )
        format_expiry_date = expiry_date.strftime("%Y-%m-%d %H:%M:%S")
        url = generate_url(user, token, forgot_password_url)

        message = render_to_string(
            template_mail,
            {
                "user": user,
                "service_name": settings.SERVICE_NAME,
                "expiry_date": format_expiry_date,
                "confirm_url": url,
            },
        )

        send_email_to_user(request, subject, message, user.email, url)

    @swagger_auto_schema(
        operation_description="Send password reset email to user",
        request_body=ForgotPasswordSerializer,
        responses={
            status.HTTP_200_OK: openapi.Response(
                description="Reset email sent successfully",
                schema=standard_response_schema(
                    message_example=messages.RESET_EMAIL_SENT,
                    status_code_example=status.HTTP_200_OK
                ),
            ),
            status.HTTP_400_BAD_REQUEST: openapi.Response(
                description="Invalid email",
                schema=standard_response_schema(
                    error_code_example="INVALID_EMAIL",
                    message_example="Invalid email provided",
                    status_code_example=status.HTTP_400_BAD_REQUEST,
                ),
            ),
            status.HTTP_404_NOT_FOUND: openapi.Response(
                description="User not found",
                schema=standard_response_schema(
                    error_code_example="USER_NOT_FOUND",
                    message_example=messages.USER_NOT_FOUND,
                    status_code_example=status.HTTP_404_NOT_FOUND,
                ),
            ),
        },
    )
    def post(self, request: Request) -> Response:
        """
        Handle forgot password request by sending reset email.

        Args:
            request: The HTTP request containing user email

        Returns:
            Response: HTTP response indicating email send status
        """

        serializer = ForgotPasswordSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        email = serializer.validated_data["email"]
        user = self.user_service.get_user_by_email(email)

        if not user:
            return create_response(
                message=messages.USER_NOT_FOUND,
                status_code=status.HTTP_404_NOT_FOUND
            )

        token = default_token_generator.make_token(user)
        forgot_password_url = settings.FORGOT_PASSWORD_URL

        self._send_reset_email(request, user, token, forgot_password_url)

        return create_response(
            message=messages.RESET_EMAIL_SENT,
            status_code=status.HTTP_200_OK
        )


class ResetPasswordAPIView(APIView):
    """API View for handling password reset requests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_service = TokenService()
        self.user_service = UserService()

    def _validate_token_params(self, uidb64: str, token: str) -> tuple[Optional[User], Optional[str]]:
        """
        Validate token parameters and return user if valid.

        Args:
            uidb64: Base64 encoded user ID
            token: Password reset token

        Returns:
            tuple: (User object if valid, error message if invalid)
        """
        if not uidb64 or not token:
            return None, messages.INVALID_REQUEST_PARAMETERS

        try:
            user_id = decode_base64(uidb64)
            user = self.user_service.get_user_by_id(user_id)

            if not user:
                return None, messages.USER_NOT_FOUND

            if not self.token_service.validate_token_reset_password(token, f"{constant.FORGOT_PASSWORD_KEY}:{user.email}"):
                return None, messages.INVALID_OR_EXPIRED_TOKEN

            return user, None
        except Exception as e:
            return None, str(e)

    @swagger_auto_schema(
        operation_description="Validate password reset token",
        manual_parameters=[
            openapi.Parameter(
                "uidb64",
                openapi.IN_QUERY,
                description="User ID (base64 encoded)",
                type=openapi.TYPE_STRING,
                required=True,
            ),
            openapi.Parameter(
                "token",
                openapi.IN_QUERY,
                description="Password reset token",
                type=openapi.TYPE_STRING,
                required=True,
            ),
        ],
        responses={
            status.HTTP_200_OK: openapi.Response(
                description="Token is valid",
                schema=standard_response_schema(
                    message_example=messages.TOKEN_IS_VALID,
                    status_code_example=status.HTTP_200_OK
                ),
            ),
            status.HTTP_400_BAD_REQUEST: openapi.Response(
                description="Invalid parameters",
                schema=standard_response_schema(
                    error_code_example="INVALID_PARAMETERS",
                    message_example=messages.INVALID_REQUEST_PARAMETERS,
                    status_code_example=status.HTTP_400_BAD_REQUEST,
                ),
            ),
            status.HTTP_401_UNAUTHORIZED: openapi.Response(
                description="Invalid or expired token",
                schema=standard_response_schema(
                    error_code_example="INVALID_TOKEN",
                    message_example=messages.INVALID_OR_EXPIRED_TOKEN,
                    status_code_example=status.HTTP_401_UNAUTHORIZED,
                ),
            ),
        },
    )
    def get(self, request: Request) -> Response:
        """
        Validate password reset token.

        Args:
            request: The HTTP request containing token parameters

        Returns:
            Response: HTTP response indicating token validity
        """
        uidb64 = request.GET.get("uidb64")
        token = request.GET.get("token")
        user, error = self._validate_token_params(uidb64, token)
        if error:
            status_code = (
                status.HTTP_401_UNAUTHORIZED
                if error == messages.INVALID_OR_EXPIRED_TOKEN
                else status.HTTP_400_BAD_REQUEST
            )
            return create_response(message=error, status_code=status_code)

        return create_response(
            message=messages.TOKEN_IS_VALID,
            status_code=status.HTTP_200_OK
        )

    @swagger_auto_schema(
        operation_description="Reset user password",
        request_body=ResetPasswordSerializer,
        responses={
            status.HTTP_200_OK: openapi.Response(
                description="Password reset successful",
                schema=standard_response_schema(
                    message_example=messages.PASSWORD_RESET_SUCCESSFUL,
                    status_code_example=status.HTTP_200_OK
                ),
            ),
            status.HTTP_400_BAD_REQUEST: openapi.Response(
                description="Invalid parameters",
                schema=standard_response_schema(
                    error_code_example="INVALID_PARAMETERS",
                    message_example=messages.INVALID_REQUEST_PARAMETERS,
                    status_code_example=status.HTTP_400_BAD_REQUEST,
                ),
            ),
            status.HTTP_401_UNAUTHORIZED: openapi.Response(
                description="Invalid or expired token",
                schema=standard_response_schema(
                    error_code_example="INVALID_TOKEN",
                    message_example=messages.INVALID_OR_EXPIRED_TOKEN,
                    status_code_example=status.HTTP_401_UNAUTHORIZED,
                ),
            ),
        },
    )
    def post(self, request: Request) -> Response:
        """
        Reset user password using provided token.

        Args:
            request: The HTTP request containing reset parameters

        Returns:
            Response: HTTP response indicating reset status
        """

        serializer = ResetPasswordSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        data = serializer.validated_data
        token = data["token"]
        uidb64 = data["uidb64"]
        user, error = self._validate_token_params(
            uidb64, token)

        if error:
            status_code = (
                status.HTTP_401_UNAUTHORIZED
                if error == messages.INVALID_OR_EXPIRED_TOKEN
                else status.HTTP_400_BAD_REQUEST
            )
            return create_response(message=error, status_code=status_code)

        user.set_password(data["new_password"])
        user.save()
        self.token_service.mark_token_as_used(token)

        return create_response(
            message=messages.PASSWORD_RESET_SUCCESSFUL,
            status_code=status.HTTP_200_OK
        )
