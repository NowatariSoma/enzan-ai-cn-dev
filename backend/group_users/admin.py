from django.contrib import admin

# Register your models here.
from .models import Group




@admin.register(Group)
class GroupAdmin(admin.ModelAdmin):
    list_display = ("name", "created_at", "updated_at")
    list_filter = ("created_at", "updated_at")
    search_fields = ("name",)
    filter_horizontal = ('user_ids', 'projects')
    readonly_fields = ('created_at', 'updated_at')
