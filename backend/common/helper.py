from django.conf import settings
import jwt
from django.utils.http import urlsafe_base64_decode
from drf_yasg import openapi


def decode_token(token):
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
    except jwt.InvalidTokenError:
        return None


def decode_base64(uidb64):
    return urlsafe_base64_decode(uidb64).decode()
