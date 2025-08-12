from django.contrib import admin
from projects.models import Project
from .models import Project, Label, Prediction, PredictionResult


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ['title', 'organization', 'ml_type_name',
                    'is_published', 'is_draft', 'created_at']
    list_filter = ['is_published', 'is_draft', 'ml_type_name', 'organization']
    search_fields = ['title', 'description']
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['-created_at']


class PredictionResultInline(admin.TabularInline):
    model = PredictionResult
    extra = 0
    readonly_fields = ['created_at', 'updated_at']


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['id', 'ls_project_id',
                    'predicted_at', 'result_count', 'created_at']
    list_filter = ['predicted_at', 'created_at']
    search_fields = ['ls_project_id', 'json_file_name',]
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['-predicted_at']
    inlines = [PredictionResultInline]


    def result_count(self, obj):
        return obj.results.count()
    result_count.short_description = 'Results Count'


@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    list_display = ['id', 'prediction', 'label', 'score', 'created_at']
    list_filter = ['label', 'created_at']
    search_fields = ['label', 'prediction__ls_project_id']
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['-score']
