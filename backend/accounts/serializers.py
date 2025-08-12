
from accounts.models import User
from rest_framework.serializers import (
    Serializer,
    EmailField,
    CharField,
    ValidationError,
)
from common import messages
from common.validators import validate_password
from django.utils.translation import gettext_lazy as _


class RegistrationSerializer(Serializer):
    uidb64 = CharField()
    token = CharField()
    new_password = CharField(min_length=8, max_length=64)

    def validate_new_password(self, value):
        validate_password(value)
        return value


class LoginSerializer(Serializer):
    email = EmailField(
        error_messages={"invalid": _(messages.INVALID_CREDENTIALS)})
    password = CharField(write_only=True)

    def validate(self, data):
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            raise ValidationError(messages.EMAIL_AND_PASSWORD_REQUIRED)
        return data


class ForgotPasswordSerializer(Serializer):
    email = EmailField(
        error_messages={"invalid": _(messages.EMAIL_NOT_REGISTERED_MESSAGE)}
    )

    def validate_email(self, value):
        user = User.objects.filter(email=value).first()
        if not user:
            raise ValidationError(messages.EMAIL_NOT_REGISTERED_MESSAGE)
        if not user.is_active:
            raise ValidationError(messages.EMAIL_INVALID)
        return value


class ResetPasswordSerializer(RegistrationSerializer):
    def validate(self, data):
        return super().validate(data)
