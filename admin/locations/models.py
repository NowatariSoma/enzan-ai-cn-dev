from django.db import models
from django.contrib.auth import get_user_model
from django.utils.translation import gettext_lazy as _
from django.db.models.signals import post_save
from django.dispatch import receiver


User = get_user_model()


class Feature(models.Model):
    """機能マスター"""
    FEATURE_TYPES = [
        ('ai_a_measurement', 'AI-A計測'),  # 親機能
        ('ai_measurement', 'AI計測集計'),
        ('prediction', '予測モデル作成'),
        ('simulation', '最終変位・沈下予測'),
        ('data_analysis', 'データ分析'),
        ('reporting', 'レポート機能'),
        ('user_management', 'ユーザー管理'),
        ('location_management', '拠点管理'),
        ('custom', 'カスタム機能'),
    ]
    
    name = models.CharField(_("機能名"), max_length=255, unique=True)
    feature_type = models.CharField(_("機能タイプ"), max_length=50, choices=FEATURE_TYPES)
    parent_feature = models.ForeignKey(
        'self', 
        on_delete=models.CASCADE, 
        null=True, 
        blank=True,
        related_name='child_features',
        verbose_name=_("親機能"),
        help_text=_("この機能の親機能（指定した場合、親機能が有効になると自動で有効になります）")
    )
    description = models.TextField(_("説明"), blank=True, null=True)
    is_active = models.BooleanField(_("利用可能"), default=True)
    display_order = models.IntegerField(_("表示順"), default=0, help_text="表示順序（小さい値が先に表示）")
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        if self.parent_feature:
            return f"{self.parent_feature.name} > {self.name}"
        return self.name

    @property
    def is_parent(self):
        """親機能かどうか"""
        return self.child_features.exists()
    
    @property
    def is_child(self):
        """子機能かどうか"""
        return self.parent_feature is not None

    class Meta:
        verbose_name = _("機能")
        verbose_name_plural = _("機能")
        ordering = ['display_order', 'id']


class Location(models.Model):
    STATUS_CHOICES = [
        ('active', 'アクティブ'),
        ('monitoring', 'モニタリング中'),
        ('completed', '完了'),
        ('planning', '計画中'),
    ]
    
    ALERT_LEVEL_CHOICES = [
        ('normal', '正常'),
        ('warning', '警告'),
        ('danger', '危険'),
    ]

    # 基本情報
    location_id = models.CharField(_("拠点ID"), max_length=100, unique=True, help_text="フロントエンド用の識別子", null=True, blank=True)
    name = models.CharField(_("拠点名"), max_length=255)
    description = models.TextField(_("説明"), blank=True, null=True)
    address = models.TextField(_("住所"), blank=True, null=True)
    
    # 地域情報
    region = models.CharField(_("地域"), max_length=100, blank=True, null=True)
    prefecture = models.CharField(_("都道府県"), max_length=100, blank=True, null=True)
    tunnel_name = models.CharField(_("トンネル名"), max_length=255, blank=True, null=True)
    
    # プロジェクト情報
    folder_name = models.CharField(_("フォルダ名"), max_length=100, blank=True, null=True, help_text="バックエンドのフォルダ名")
    status = models.CharField(_("ステータス"), max_length=20, choices=STATUS_CHOICES, default='planning')
    start_date = models.DateField(_("開始日"), blank=True, null=True)
    total_length = models.IntegerField(_("トンネル全長（m）"), blank=True, null=True)
    progress = models.DecimalField(_("進捗率（%）"), max_digits=5, decimal_places=2, default=0.0)
    measurement_count = models.IntegerField(_("計測ポイント数"), default=0)
    alert_level = models.CharField(_("アラートレベル"), max_length=20, choices=ALERT_LEVEL_CHOICES, default='normal')
    
    # 座標情報
    latitude = models.DecimalField(_("緯度"), max_digits=10, decimal_places=8, blank=True, null=True)
    longitude = models.DecimalField(_("経度"), max_digits=11, decimal_places=8, blank=True, null=True)
    
    # 関連情報
    users = models.ManyToManyField(User, related_name="locations", blank=True, verbose_name=_("利用可能ユーザー"))
    features = models.ManyToManyField(Feature, through='LocationFeature', verbose_name=_("利用機能"))
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name
    
    @property
    def coordinates(self):
        """座標情報をdict形式で返す（フロントエンド互換性）"""
        if self.latitude and self.longitude:
            return {
                'lat': float(self.latitude),
                'lng': float(self.longitude)
            }
        return None

    class Meta:
        verbose_name = _("拠点")
        verbose_name_plural = _("拠点")
        ordering = ['-updated_at']


class LocationFeature(models.Model):
    """拠点と機能の中間テーブル"""
    location = models.ForeignKey(Location, related_name="location_features", on_delete=models.CASCADE, verbose_name=_("拠点"))
    feature = models.ForeignKey(Feature, related_name="location_features", on_delete=models.CASCADE, verbose_name=_("機能"))
    is_enabled = models.BooleanField(_("有効"), default=True)
    settings = models.JSONField(_("拠点固有設定"), blank=True, null=True, default=dict)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.location.name} - {self.feature.name}"

    class Meta:
        verbose_name = _("拠点機能設定")
        verbose_name_plural = _("拠点機能設定")
        unique_together = ['location', 'feature']


@receiver(post_save, sender=LocationFeature)
def handle_parent_feature_activation(sender, instance, created, **kwargs):
    """
    親機能が有効になった場合、子機能も自動で有効にする
    """
    if instance.is_enabled and instance.feature.is_parent:
        # 親機能が有効化された場合、子機能も自動で有効にする
        child_features = instance.feature.child_features.filter(is_active=True)
        
        for child_feature in child_features:
            child_location_feature, created = LocationFeature.objects.get_or_create(
                location=instance.location,
                feature=child_feature,
                defaults={'is_enabled': True}
            )
            if not created and not child_location_feature.is_enabled:
                child_location_feature.is_enabled = True
                child_location_feature.save()
                
        print(f"親機能 '{instance.feature.name}' が有効化されました。{child_features.count()}個の子機能も有効になりました。")
    
    elif not instance.is_enabled and instance.feature.is_parent:
        # 親機能が無効化された場合、子機能も無効にする
        child_features = instance.feature.child_features.all()
        LocationFeature.objects.filter(
            location=instance.location,
            feature__in=child_features,
            is_enabled=True
        ).update(is_enabled=False)
        
        print(f"親機能 '{instance.feature.name}' が無効化されました。{child_features.count()}個の子機能も無効になりました。")
