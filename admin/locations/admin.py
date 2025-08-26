from django.contrib import admin
from .models import Feature, Location, LocationFeature


@admin.register(Feature)
class FeatureAdmin(admin.ModelAdmin):
    """機能マスター管理画面"""
    list_display = ("name", "feature_type", "is_active", "created_at", "updated_at")
    list_filter = ("feature_type", "is_active", "created_at", "updated_at")
    search_fields = ("name", "description")
    readonly_fields = ('created_at', 'updated_at')
    ordering = ('feature_type', 'name')


class LocationFeatureInline(admin.TabularInline):
    """拠点詳細画面で機能設定をインライン編集"""
    model = LocationFeature
    extra = 0
    fields = ('feature', 'is_enabled', 'settings')
    readonly_fields = ('created_at', 'updated_at')


@admin.register(Location)
class LocationAdmin(admin.ModelAdmin):
    """拠点管理画面"""
    list_display = (
        "name", "location_id", "region", "prefecture", "tunnel_name", 
        "status", "alert_level", "progress", "user_count", "feature_count", 
        "enabled_feature_count", "updated_at"
    )
    list_filter = ("status", "alert_level", "region", "prefecture", "created_at", "updated_at")
    search_fields = ("name", "location_id", "description", "address", "tunnel_name", "folder_name")
    filter_horizontal = ('users',)
    readonly_fields = ('created_at', 'updated_at')
    inlines = [LocationFeatureInline]
    
    fieldsets = (
        ('基本情報', {
            'fields': ('location_id', 'name', 'description', 'address')
        }),
        ('地域情報', {
            'fields': ('region', 'prefecture', 'tunnel_name')
        }),
        ('プロジェクト情報', {
            'fields': ('folder_name', 'status', 'start_date', 'total_length', 'progress', 'measurement_count', 'alert_level')
        }),
        ('座標情報', {
            'fields': ('latitude', 'longitude'),
            'classes': ('collapse',)
        }),
        ('関連情報', {
            'fields': ('users',)
        }),
        ('システム情報', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def user_count(self, obj):
        return obj.users.count()
    user_count.short_description = '利用ユーザー数'
    
    def feature_count(self, obj):
        return obj.features.count()
    feature_count.short_description = '機能数'
    
    def enabled_feature_count(self, obj):
        return obj.location_features.filter(is_enabled=True).count()
    enabled_feature_count.short_description = '有効機能数'
    
    def get_readonly_fields(self, request, obj=None):
        readonly_fields = list(self.readonly_fields)
        # 編集時はlocation_idを読み取り専用にする（重複を避けるため）
        if obj:
            readonly_fields.append('location_id')
        return readonly_fields


# LocationFeature は拠点の管理画面で直接編集するため、独立した管理画面は不要
