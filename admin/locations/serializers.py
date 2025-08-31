from rest_framework.serializers import ModelSerializer, CharField, SerializerMethodField
from .models import Location, Feature, LocationFeature
from accounts.serializers import UserListSerializer


class FeatureSerializer(ModelSerializer):
    """機能マスター用シリアライザー"""
    class Meta:
        model = Feature
        fields = ['id', 'name', 'feature_type', 'description', 'is_active', 'created_at', 'updated_at']
        read_only_fields = ['created_at', 'updated_at']


class FeatureListSerializer(ModelSerializer):
    """機能一覧用シリアライザー"""
    class Meta:
        model = Feature
        fields = ['id', 'name', 'feature_type', 'description', 'is_active']


class LocationFeatureSerializer(ModelSerializer):
    """拠点機能設定用シリアライザー"""
    feature = FeatureListSerializer(read_only=True)
    feature_id = CharField(write_only=True)
    
    class Meta:
        model = LocationFeature
        fields = ['id', 'feature', 'feature_id', 'is_enabled', 'settings', 'created_at', 'updated_at']
        read_only_fields = ['created_at', 'updated_at']


class LocationListSerializer(ModelSerializer):
    """拠点一覧用シリアライザー"""
    user_count = SerializerMethodField()
    feature_count = SerializerMethodField()
    enabled_feature_count = SerializerMethodField()
    coordinates = SerializerMethodField()
    available_features = SerializerMethodField()
    lastUpdated = SerializerMethodField()
    
    class Meta:
        model = Location
        fields = [
            'id', 'location_id', 'name', 'description', 'address',
            'region', 'prefecture', 'tunnel_name', 'folder_name',
            'status', 'start_date', 'total_length', 'progress', 
            'measurement_count', 'alert_level', 'coordinates',
            'available_features', 'lastUpdated', 'user_count', 'feature_count', 
            'enabled_feature_count', 'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']
    
    def get_user_count(self, obj):
        return obj.users.count()
    
    def get_feature_count(self, obj):
        return obj.features.count()
    
    def get_enabled_feature_count(self, obj):
        return obj.location_features.filter(is_enabled=True).count()
    
    def get_coordinates(self, obj):
        return obj.coordinates
    
    def get_available_features(self, obj):
        """利用可能な機能一覧（フロントエンド形式で返す）"""
        features_dict = {}
        
        # すべての機能をfalseで初期化
        for feature in Feature.objects.filter(is_active=True):
            # フロントエンドの機能名にマッピング
            feature_key = self._get_frontend_feature_key(feature.feature_type)
            if feature_key:
                features_dict[feature_key] = False
        
        # 拠点で有効な機能をtrueに設定
        for location_feature in obj.location_features.filter(is_enabled=True):
            feature_key = self._get_frontend_feature_key(location_feature.feature.feature_type)
            if feature_key:
                features_dict[feature_key] = True
        
        return features_dict
    
    def get_lastUpdated(self, obj):
        return obj.updated_at.isoformat()
    
    def _get_frontend_feature_key(self, feature_type):
        """バックエンドの機能タイプをフロントエンドのキーにマッピング"""
        mapping = {
            'ai_a_measurement': 'aiMeasurement',  # 親のAI-A計測機能
            'measurement': 'measurement',  # A計測集計
            'simulation': 'simulation',  # 最終変位・沈下予測
            'modelCreation': 'modelCreation',  # 予測モデル作成
            'reporting': 'reportGeneration',
            'user_management': 'userManagement',
            'location_management': 'locationManagement',
        }
        return mapping.get(feature_type)


class LocationDetailSerializer(ModelSerializer):
    """拠点詳細用シリアライザー"""
    users = UserListSerializer(many=True, read_only=True)
    location_features = LocationFeatureSerializer(many=True, read_only=True)
    available_features = SerializerMethodField()
    user_count = SerializerMethodField()
    feature_count = SerializerMethodField()
    enabled_feature_count = SerializerMethodField()
    coordinates = SerializerMethodField()
    
    class Meta:
        model = Location
        fields = [
            'id', 'location_id', 'name', 'description', 'address',
            'region', 'prefecture', 'tunnel_name', 'folder_name',
            'status', 'start_date', 'total_length', 'progress', 
            'measurement_count', 'alert_level', 'coordinates',
            'users', 'location_features', 'available_features',
            'user_count', 'feature_count', 'enabled_feature_count', 
            'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']
    
    def get_available_features(self, obj):
        """利用可能な機能一覧（フロントエンド形式で返す）"""
        features_dict = {}
        
        # すべての機能をfalseで初期化
        for feature in Feature.objects.filter(is_active=True):
            # フロントエンドの機能名にマッピング
            feature_key = self._get_frontend_feature_key(feature.feature_type)
            if feature_key:
                features_dict[feature_key] = False
        
        # 拠点で有効な機能をtrueに設定
        for location_feature in obj.location_features.filter(is_enabled=True):
            feature_key = self._get_frontend_feature_key(location_feature.feature.feature_type)
            if feature_key:
                features_dict[feature_key] = True
        
        return features_dict
    
    def _get_frontend_feature_key(self, feature_type):
        """バックエンドの機能タイプをフロントエンドのキーにマッピング"""
        mapping = {
            'ai_a_measurement': 'aiMeasurement',  # 親のAI-A計測機能
            'measurement': 'measurement',  # A計測集計
            'simulation': 'simulation',  # 最終変位・沈下予測
            'modelCreation': 'modelCreation',  # 予測モデル作成
            'reporting': 'reportGeneration',
            'user_management': 'userManagement',
            'location_management': 'locationManagement',
        }
        return mapping.get(feature_type)
    
    def get_user_count(self, obj):
        return obj.users.count()
    
    def get_feature_count(self, obj):
        return obj.features.count()
    
    def get_enabled_feature_count(self, obj):
        return obj.location_features.filter(is_enabled=True).count()
    
    def get_coordinates(self, obj):
        return obj.coordinates


class LocationCreateUpdateSerializer(ModelSerializer):
    """拠点作成・更新用シリアライザー"""
    class Meta:
        model = Location
        fields = [
            'location_id', 'name', 'description', 'address',
            'region', 'prefecture', 'tunnel_name', 'folder_name',
            'status', 'start_date', 'total_length', 'progress',
            'measurement_count', 'alert_level', 'latitude', 'longitude'
        ]


class LocationFeatureCreateUpdateSerializer(ModelSerializer):
    """拠点機能設定作成・更新用シリアライザー"""
    class Meta:
        model = LocationFeature
        fields = ['feature', 'is_enabled', 'settings']