import re
from typing import Optional
from django.core.exceptions import ValidationError
from common import messages


def validate_password(value, field_name: Optional[str] = None):
    password_regex = r'^(?=(.*[a-z]))(?=(.*[A-Z]))(?=(.*\d))[\w\d!@#$%^&*()_+={}\[\]:;"\'<>,.?/|\\`~\-]{8,64}$'

    if not re.match(password_regex, value):
        error_message = (
            messages.PASSWORD_VALIDATION
            if field_name
            else messages.PASSWORD_VALIDATION_EN
        )
        raise ValidationError(
            {field_name: error_message} if field_name else error_message
        )

    return value
