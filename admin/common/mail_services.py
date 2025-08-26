import logging
from django.core.mail import send_mail
from django.contrib import messages as django_messages
from django.utils.translation import gettext_lazy as _
from django.utils.http import urlsafe_base64_encode
from django.utils.http import urlencode
from django.conf import settings


def send_email_to_user(request, subject, message, to_email, confirm_url=None):
    environment = settings.ENVIRONMENT
    if environment == "production":
        logging.info(
            f"Sending email to {to_email} with subject: {subject} and message: {message}")
        send_mail(
            subject,
            message,
            settings.NO_REPLY_EMAIL,
            [to_email],
            html_message=message,
        )
    else:
        django_messages.success(
            request,
            _(
                f"The user was successfully invited! The link sent to the email is: {confirm_url}"
            ),
        )


def generate_url(object, token, confirm_url):
    uid = urlsafe_base64_encode(str(object.pk).encode())
    query_params = urlencode({"uid": uid, "token": token})
    return f"{confirm_url}?{query_params}"
