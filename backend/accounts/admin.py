from django.conf import settings
from django.contrib import admin
from django.contrib.auth.forms import UserChangeForm
from django.utils.translation import gettext_lazy as _
from common.validators import validate_password
from .models import Invitation, User, Token
from django import forms
from common import messages, constant
from common.mail_services import generate_url, send_email_to_user
from django.template.loader import render_to_string
import jwt
from django.urls import reverse
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import Group
from datetime import datetime, timedelta
from django.contrib import messages as django_messages


class CustomUserChangeForm(UserChangeForm):
    class Meta:
        model = User
        fields = ("email", "first_name", "last_name",
                  "password", "role", "is_active")
        read_only_fields = ["email", "created_at", "updated_at"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        user_role = self.instance.role
        if user_role != User.Role.ADMIN:
            self.fields["role"].choices = [
                choice for choice in User.Role.choices if choice[0] != User.Role.ADMIN
            ]
        else:
            self.fields["role"].disabled = True
        self.fields["password"].required = False
        self.fields["password"].widget = forms.PasswordInput(
            attrs={"placeholder": "******************************"}
        )
        self.fields["password"].help_text = "You cannot change the password here."


class CustomUserAdmin(UserAdmin):
    exclude = ('username', 'is_staff', 'is_superuser', 'date_joined',)

    def delete_queryset(self, request, queryset):
        admin_users = queryset.filter(role="admin")
        admin_count = admin_users.count()

        if admin_count > 0:
            admin_emails = ", ".join(
                admin_users.values_list("email", flat=True))
            self.message_user(
                request,
                f"Cannot delete users with role Admin: {admin_emails}",
                level=django_messages.ERROR,
            )
            queryset = queryset.exclude(role="admin")

        deleted_count = queryset.count()
        if deleted_count > 0:
            for obj in queryset:
                self.log_deletion(request, obj, str(obj))
            queryset.delete()
            django_messages.add_message(
                request,
                django_messages.SUCCESS,
                f"{deleted_count} User{'s' if deleted_count > 1 else ''} have been deleted.",
            )

    def message_user(
        self,
        request,
        message,
        level=django_messages.INFO,
        extra_tags="",
        fail_silently=False,
    ):
        if "Successfully deleted" in message:
            return
        super().message_user(request, message, level, extra_tags, fail_silently)

    def get_form(self, request, obj=None, **kwargs):
        if obj is not None:
            kwargs["form"] = CustomUserChangeForm
        return super().get_form(request, obj, **kwargs)

    fieldsets = (
        (
            None,
            {
                "fields": (
                    "email",
                    "first_name",
                    "last_name",
                    "password",
                    "role",
                    "is_active",
                )
            },
        ),
    )

    readonly_fields = ("created_at", "updated_at")
    ordering = ["-created_at"]
    list_filter = ["role"]
    list_display = ("email", "first_name", "last_name", "role", "is_active")
    search_fields = ("email", "first_name", "last_name")

    def has_add_permission(self, request):
        return False


class InvitationAdmin(admin.ModelAdmin):
    list_display = ("email", "first_name", "last_name",
                    "expires_at", "registered",)
    search_fields = ("email", "first_name", "last_name")
    ordering = ["-id"]

    def get_fieldsets(self, request, obj=None):
        if obj is None:
            return (
                (None, {
                    'fields': ('email', 'first_name', 'last_name')
                }),
            )
        else:
            return (
                (None, {
                    'fields': ('email', 'first_name', 'last_name', 'expires_at', 'registered')
                }),
            )

    def save_model(self, request, obj, form, change):
        is_new = not change
        if is_new:
            obj.expires_at = datetime.now() + timedelta(hours=48)
            obj.registered = False
        super().save_model(request, obj, form, change)
        if is_new:
            self._send_invitation_email(request, obj)

    def _send_invitation_email(self, request, invitation):
        current_timestamp = datetime.now().timestamp()
        user_data = {
            "invitation_id": invitation.id,
            "email": invitation.email,
            "first_name": invitation.first_name,
            "last_name": invitation.last_name,
            "timestamp": current_timestamp,  # Add timestamp to make token unique
        }
        token = jwt.encode(
            user_data,
            settings.SECRET_KEY,
            algorithm="HS256",
        )

        subject = constant.REGISTRATION_EMAIL_SUBJECT
        template_mail = constant.TEMPLATE_CONFIRM_PASSWORD
        password_confirm_url = settings.REGISTRATION_URL

        expiry_date = datetime.now() + timedelta(days=settings.TOKEN_LIFETIME)

        Token.objects.create(
            key=f"{constant.INVITE_USER}:{invitation.email}",
            value=token,
            is_used=False,
            expires_at=expiry_date,
        )

        format_expiry_date = expiry_date.strftime("%Y-%m-%d %H:%M:%S")
        url = generate_url(
            invitation, token, password_confirm_url)

        message = render_to_string(
            template_mail,
            {
                "inviation": invitation,
                "service_name": settings.SERVICE_NAME,
                "expiry_date": format_expiry_date,
                "confirm_url": url,
            },
        )
        send_email_to_user(request, subject, message, invitation.email, url)


admin.site.register(Invitation, InvitationAdmin)
admin.site.unregister(Group)
admin.site.register(User, CustomUserAdmin)
