from django.db import models

from django.contrib.auth.models import AbstractUser
from django.contrib.auth.models import PermissionsMixin, BaseUserManager
from django.db import models
from common import messages
from django.utils.timezone import now
from django.utils.translation import gettext_lazy as _


class CustomUserManager(BaseUserManager):

    def validate_username_email_and_password(self, username, email, password):
        if not username:
            raise ValueError("ユーザーネームは必須です")
        if not email:
            raise ValueError(messages.EMAIL_FIELD_REQUIRED)
        if not password:
            raise ValueError(messages.PASSWORD_FIELD_REQUIRED)

    def validate_role(self, role):
        allowed_roles = [User.Role.ADMIN, User.Role.STAFF, User.Role.USER]
        if role not in allowed_roles:
            raise ValueError(messages.USER_ROLE_ERROR)

    def create_user(self, username, email, password, **extra_fields):
        self.validate_username_email_and_password(username, email, password)

        email = self.normalize_email(email)
        role = extra_fields.get("role")
        self.validate_role(role)

        user = self.model(username=username, email=email, **extra_fields)
        user.set_password(password)

        user.save(using=self._db)
        return user

    def create_superuser(self, username, email, password, **extra_fields):
        self.validate_username_email_and_password(username, email, password)
        extra_fields.setdefault("role", User.Role.ADMIN)
        email = self.normalize_email(email)
        user = self.model(username=username, email=email, **extra_fields)
        user.set_password(password)

        user.save(using=self._db)
        return user


class User(AbstractUser, PermissionsMixin):
    class Role(models.TextChoices):
        ADMIN = "admin", _("管理者")
        STAFF = "staff", _("スタッフ")
        USER = "user", _("一般ユーザー")

    username = models.CharField(_("ユーザーネーム"), max_length=150, unique=True)
    email = models.EmailField(_("email address"), unique=True)
    first_name = models.CharField(_("名前"), max_length=256)
    last_name = models.CharField(_("姓"), max_length=256)
    role = models.CharField(
        _("役割"),
        max_length=50,
        choices=Role.choices,
        default=Role.USER,
    )
    is_active = models.BooleanField(_("有効"), default=True)
    date_joined = None
    created_at = models.DateTimeField(
        _("Created At"), auto_now_add=True, editable=False
    )
    updated_at = models.DateTimeField(_("Updated At"), auto_now=True)

    EMAIL_FIELD = "email"
    USERNAME_FIELD = "username"
    REQUIRED_FIELDS = ["email"]

    objects = CustomUserManager()

    @property
    def is_staff(self):
        return self.role in [self.Role.ADMIN]

    @property
    def is_superuser(self):
        return self.role in [self.Role.ADMIN]

    class Meta:
        verbose_name = _("ユーザー")
        verbose_name_plural = _("ユーザー")

    def get_locations(self):
        return self.locations.all()
    
    def get_full_name(self):
        return f"{self.last_name} {self.first_name}".strip()
    
    def is_admin_user(self):
        return self.role == self.Role.ADMIN
    
    def is_staff_user(self):
        return self.role in [self.Role.ADMIN, self.Role.STAFF]




