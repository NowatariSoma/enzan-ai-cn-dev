from django.contrib import admin
from django.contrib.auth.forms import UserChangeForm
from django.utils.translation import gettext_lazy as _
from .models import User
from django import forms
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import Group
from django.contrib import messages as django_messages


class CustomUserChangeForm(UserChangeForm):
    class Meta:
        model = User
        fields = ("username", "email", "first_name", "last_name",
                  "password", "role", "is_active")
        read_only_fields = ["username", "email", "created_at", "updated_at"]

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
    exclude = ('is_staff', 'is_superuser', 'date_joined',)

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
        (None, {
            'fields': ('username', 'password')
        }),
        ('Personal info', {
            'fields': ('email', 'first_name', 'last_name')
        }),
        ('Permissions', {
            'fields': ('role', 'is_active')
        }),
    )
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'email', 'first_name', 'last_name', 'role', 'password1', 'password2'),
        }),
    )

    readonly_fields = ("created_at", "updated_at")
    ordering = ["-created_at"]
    list_filter = ["role"]
    list_display = ("username", "email", "first_name", "last_name", "role", "is_active")
    search_fields = ("username", "email", "first_name", "last_name")

    def has_add_permission(self, request):
        return True


admin.site.unregister(Group)
admin.site.register(User, CustomUserAdmin)
