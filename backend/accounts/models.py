from django.db import models

from django.contrib.auth.models import AbstractUser
from django.contrib.auth.models import PermissionsMixin, BaseUserManager
from django.db import models
from common import messages
from django.utils.timezone import now
from django.utils.translation import gettext_lazy as _


class CustomUserManager(BaseUserManager):

    def validate_email_and_password(self, email, password):
        if not email:
            raise ValueError(messages.EMAIL_FIELD_REQUIRED)
        if not password:
            raise ValueError(messages.PASSWORD_FIELD_REQUIRED)

    def validate_role(self, role):
        allowed_roles = [User.Role.STAFF]
        if role not in allowed_roles:
            raise ValueError(messages.USER_ROLE_ERROR)

    def create_user(self, email, password, **extra_fields):
        self.validate_email_and_password(email, password)

        email = self.normalize_email(email)
        role = extra_fields.get("role")
        self.validate_role(role)

        user = self.model(email=email, **extra_fields)
        user.set_password(password)

        user.save(using=self._db)
        return user

    def create_superuser(self, email, password, **extra_fields):
        self.validate_email_and_password(email, password)
        extra_fields.setdefault("role", User.Role.ADMIN)
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)

        user.save(using=self._db)
        return user


class User(AbstractUser, PermissionsMixin):
    class Role(models.TextChoices):
        ADMIN = "admin", _("Admin")
        STAFF = "staff", _("Staff")

    username = None
    email = models.EmailField(_("email address"), unique=True)
    first_name = models.CharField(_("First Name"), max_length=256)
    last_name = models.CharField(_("Last Name"), max_length=256)
    role = models.CharField(
        _("Role"),
        max_length=50,
        choices=Role.choices,
        default=Role.STAFF,
    )
    date_joined = None
    created_at = models.DateTimeField(
        _("Created At"), auto_now_add=True, editable=False
    )
    updated_at = models.DateTimeField(_("Updated At"), auto_now=True)

    EMAIL_FIELD = "email"
    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = []

    objects = CustomUserManager()

    @property
    def is_staff(self):
        return self.role in [self.Role.ADMIN]

    @property
    def is_superuser(self):
        return self.role in [self.Role.ADMIN]

    class Meta:
        verbose_name = _("User")
        verbose_name_plural = _("Users")

    def get_groups(self):
        return self.group_users.all()


class Token(models.Model):
    key = models.TextField(null=False)
    value = models.TextField(unique=True, null=False)
    is_used = models.BooleanField(default=False)
    expires_at = models.DateTimeField(null=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def is_expired(self):
        return now() > self.expires_at

    class Meta:
        db_table = "tokens"
        ordering = ["-id"]
        verbose_name = "Token"
        verbose_name_plural = "Tokens"


class Invitation(models.Model):
    first_name = models.CharField(_("First Name"), max_length=256)
    last_name = models.CharField(_("Last Name"), max_length=256)
    email = models.EmailField(_("Email Address"), null=False)
    expires_at = models.DateTimeField(_("Expires At"), null=False)
    registered = models.BooleanField(_("Registered"), default=False)

    def is_expired(self):
        return now() > self.expires_at
    
    class Meta:
        db_table = "invitations"
        ordering = ["-id"]
        verbose_name = "Invitation"
        verbose_name_plural = "Invitations"
