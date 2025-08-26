
from accounts.models import User
from rest_framework.serializers import (
    Serializer,
    ModelSerializer,
    EmailField,
    CharField,
    ValidationError,
    BooleanField,
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
    username = CharField()
    password = CharField(write_only=True)

    def validate(self, data):
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            raise ValidationError("ユーザーネームとパスワードが必要です")
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


class UserListSerializer(ModelSerializer):
    full_name = CharField(source='get_full_name', read_only=True)
    
    class Meta:
        model = User
        fields = ['id', 'email', 'first_name', 'last_name', 'full_name', 'role', 'is_active', 'created_at', 'updated_at']
        read_only_fields = ['created_at', 'updated_at']


class UserDetailSerializer(ModelSerializer):
    full_name = CharField(source='get_full_name', read_only=True)
    location_names = CharField(source='get_locations', read_only=True)
    
    class Meta:
        model = User
        fields = ['id', 'email', 'first_name', 'last_name', 'full_name', 'role', 'is_active', 'location_names', 'created_at', 'updated_at']
        read_only_fields = ['created_at', 'updated_at']


class UserCreateSerializer(ModelSerializer):
    password = CharField(write_only=True, min_length=8)
    
    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name', 'role', 'password', 'is_active']
    
    def validate_password(self, value):
        validate_password(value)
        return value
    
    def create(self, validated_data):
        password = validated_data.pop('password')
        user = User(**validated_data)
        user.set_password(password)
        user.save()
        return user


class UserUpdateSerializer(ModelSerializer):
    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name', 'role', 'is_active']
