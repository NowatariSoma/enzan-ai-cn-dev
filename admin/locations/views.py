from rest_framework.viewsets import ModelViewSet
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.decorators import action, permission_classes
from rest_framework.response import Response
from rest_framework import status
from django.db.models import Q
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from common.format_response import create_response
from .models import Location, Feature, LocationFeature
from .serializers import (
    FeatureSerializer, LocationListSerializer, LocationDetailSerializer, 
    LocationCreateUpdateSerializer, LocationFeatureSerializer,
    LocationFeatureCreateUpdateSerializer
)
from accounts.models import User


class FeatureViewSet(ModelViewSet):
    """機能マスター管理API"""
    
    queryset = Feature.objects.all()
    serializer_class = FeatureSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        queryset = Feature.objects.all().order_by('feature_type', 'name')
        
        # 管理者のみがアクセス可能
        if not self.request.user.is_admin_user():
            return Feature.objects.none()
        
        # 機能タイプでフィルタ
        feature_type = self.request.query_params.get('feature_type', None)
        if feature_type:
            queryset = queryset.filter(feature_type=feature_type)
        
        # 有効性でフィルタ
        is_active = self.request.query_params.get('is_active', None)
        if is_active is not None:
            queryset = queryset.filter(is_active=is_active.lower() == 'true')
        
        # 検索機能
        search = self.request.query_params.get('search', None)
        if search:
            queryset = queryset.filter(
                Q(name__icontains=search) |
                Q(description__icontains=search)
            )
            
        return queryset
    
    @swagger_auto_schema(
        operation_description="機能一覧取得",
        manual_parameters=[
            openapi.Parameter('feature_type', openapi.IN_QUERY, description="機能タイプ", type=openapi.TYPE_STRING),
            openapi.Parameter('is_active', openapi.IN_QUERY, description="有効性フィルタ", type=openapi.TYPE_BOOLEAN),
            openapi.Parameter('search', openapi.IN_QUERY, description="検索キーワード", type=openapi.TYPE_STRING),
        ],
        responses={200: FeatureSerializer(many=True)}
    )
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)
    
    @action(detail=True, methods=['post'])
    @swagger_auto_schema(
        operation_description="機能の有効/無効切り替え",
        responses={200: "切り替え成功"}
    )
    def toggle_active(self, request, pk=None):
        feature = self.get_object()
        feature.is_active = not feature.is_active
        feature.save()
        return create_response(
            message=f"機能「{feature.name}」を{'有効' if feature.is_active else '無効'}に設定しました",
            status_code=status.HTTP_200_OK,
            data={'is_active': feature.is_active}
        )


class LocationViewSet(ModelViewSet):
    """拠点管理API"""
    
    queryset = Location.objects.all()
    
    def get_permissions(self):
        """
        認証不要のアクションと認証必要のアクションを分離
        """
        if self.action in ['list', 'retrieve']:
            # 読み取り専用のアクションは認証不要（フロントエンド用）
            self.permission_classes = [AllowAny]
        else:
            # 作成・更新・削除は認証必要
            self.permission_classes = [IsAuthenticated]
        return super().get_permissions()
    
    def get_serializer_class(self):
        if self.action == 'list':
            return LocationListSerializer
        elif self.action in ['create', 'update', 'partial_update']:
            return LocationCreateUpdateSerializer
        else:
            return LocationDetailSerializer
    
    def get_queryset(self):
        queryset = Location.objects.all().order_by('-created_at')
        
        # 認証が必要なアクション（管理系）では管理者のみアクセス可能
        if self.action not in ['list', 'retrieve'] and hasattr(self.request, 'user') and self.request.user.is_authenticated:
            if not self.request.user.is_admin_user():
                return Location.objects.none()
        
        # 検索機能
        search = self.request.query_params.get('search', None)
        if search:
            queryset = queryset.filter(
                Q(name__icontains=search) |
                Q(description__icontains=search) |
                Q(address__icontains=search)
            )
            
        return queryset
    
    @swagger_auto_schema(
        operation_description="拠点一覧取得",
        manual_parameters=[
            openapi.Parameter('search', openapi.IN_QUERY, description="検索キーワード", type=openapi.TYPE_STRING),
        ],
        responses={200: LocationListSerializer(many=True)}
    )
    def list(self, request, *args, **kwargs):
        # フロントエンド用に直接データを返す（create_responseを使わない）
        queryset = self.filter_queryset(self.get_queryset())
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)
    
    @swagger_auto_schema(
        operation_description="拠点詳細取得",
        responses={200: LocationDetailSerializer}
    )
    def retrieve(self, request, *args, **kwargs):
        # フロントエンド用に直接データを返す（create_responseを使わない）
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        return Response(serializer.data)
    
    @swagger_auto_schema(
        operation_description="新規拠点作成",
        request_body=LocationCreateUpdateSerializer,
        responses={201: LocationDetailSerializer}
    )
    def create(self, request, *args, **kwargs):
        return super().create(request, *args, **kwargs)
    
    @swagger_auto_schema(
        operation_description="拠点情報更新",
        request_body=LocationCreateUpdateSerializer,
        responses={200: LocationDetailSerializer}
    )
    def update(self, request, *args, **kwargs):
        return super().update(request, *args, **kwargs)
    
    @swagger_auto_schema(
        operation_description="拠点削除",
        responses={204: "削除成功"}
    )
    def destroy(self, request, *args, **kwargs):
        return super().destroy(request, *args, **kwargs)
    
    @action(detail=True, methods=['post'])
    @swagger_auto_schema(
        operation_description="拠点にユーザーを追加",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'user_ids': openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Schema(type=openapi.TYPE_INTEGER))
            }
        ),
        responses={200: "追加成功"}
    )
    def add_users(self, request, pk=None):
        location = self.get_object()
        user_ids = request.data.get('user_ids', [])
        
        users = User.objects.filter(id__in=user_ids)
        location.users.add(*users)
        
        return create_response(
            message=f"{len(users)}人のユーザーを拠点に追加しました",
            status_code=status.HTTP_200_OK
        )
    
    @action(detail=True, methods=['post'])
    @swagger_auto_schema(
        operation_description="拠点からユーザーを削除",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'user_ids': openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Schema(type=openapi.TYPE_INTEGER))
            }
        ),
        responses={200: "削除成功"}
    )
    def remove_users(self, request, pk=None):
        location = self.get_object()
        user_ids = request.data.get('user_ids', [])
        
        users = User.objects.filter(id__in=user_ids)
        location.users.remove(*users)
        
        return create_response(
            message=f"{len(users)}人のユーザーを拠点から削除しました",
            status_code=status.HTTP_200_OK
        )
    
    @action(detail=True, methods=['post'])
    @swagger_auto_schema(
        operation_description="拠点に機能を追加",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'feature_ids': openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Schema(type=openapi.TYPE_INTEGER)),
                'is_enabled': openapi.Schema(type=openapi.TYPE_BOOLEAN, default=True)
            }
        ),
        responses={200: "追加成功"}
    )
    def add_features(self, request, pk=None):
        location = self.get_object()
        feature_ids = request.data.get('feature_ids', [])
        is_enabled = request.data.get('is_enabled', True)
        
        features = Feature.objects.filter(id__in=feature_ids, is_active=True)
        created_count = 0
        
        for feature in features:
            location_feature, created = LocationFeature.objects.get_or_create(
                location=location,
                feature=feature,
                defaults={'is_enabled': is_enabled}
            )
            if created:
                created_count += 1
        
        return create_response(
            message=f"{created_count}個の機能を拠点に追加しました",
            status_code=status.HTTP_200_OK
        )
    
    @action(detail=True, methods=['post'])
    @swagger_auto_schema(
        operation_description="拠点から機能を削除",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'feature_ids': openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Schema(type=openapi.TYPE_INTEGER))
            }
        ),
        responses={200: "削除成功"}
    )
    def remove_features(self, request, pk=None):
        location = self.get_object()
        feature_ids = request.data.get('feature_ids', [])
        
        deleted_count = LocationFeature.objects.filter(
            location=location,
            feature_id__in=feature_ids
        ).delete()[0]
        
        return create_response(
            message=f"{deleted_count}個の機能を拠点から削除しました",
            status_code=status.HTTP_200_OK
        )
    
    @action(detail=True, methods=['post'], url_path='features/(?P<feature_id>[^/.]+)/toggle')
    @swagger_auto_schema(
        operation_description="拠点の特定機能の有効/無効切り替え",
        responses={200: "切り替え成功"}
    )
    def toggle_feature(self, request, pk=None, feature_id=None):
        location = self.get_object()
        try:
            location_feature = LocationFeature.objects.get(location=location, feature_id=feature_id)
            location_feature.is_enabled = not location_feature.is_enabled
            location_feature.save()
            return create_response(
                message=f"拠点「{location.name}」の機能「{location_feature.feature.name}」を{'有効' if location_feature.is_enabled else '無効'}に設定しました",
                status_code=status.HTTP_200_OK,
                data={'is_enabled': location_feature.is_enabled}
            )
        except LocationFeature.DoesNotExist:
            return create_response(
                message="指定された機能が見つかりません",
                status_code=status.HTTP_404_NOT_FOUND
            )
    
    @action(detail=True, methods=['post'], url_path='features/(?P<feature_id>[^/.]+)/settings')
    @swagger_auto_schema(
        operation_description="拠点の特定機能の設定更新",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'settings': openapi.Schema(type=openapi.TYPE_OBJECT)
            }
        ),
        responses={200: "設定更新成功"}
    )
    def update_feature_settings(self, request, pk=None, feature_id=None):
        location = self.get_object()
        try:
            location_feature = LocationFeature.objects.get(location=location, feature_id=feature_id)
            new_settings = request.data.get('settings', {})
            
            # 既存の設定とマージ
            current_settings = location_feature.settings or {}
            current_settings.update(new_settings)
            location_feature.settings = current_settings
            location_feature.save()
            
            return create_response(
                message=f"拠点「{location.name}」の機能「{location_feature.feature.name}」の設定を更新しました",
                status_code=status.HTTP_200_OK,
                data={'settings': location_feature.settings}
            )
        except LocationFeature.DoesNotExist:
            return create_response(
                message="指定された機能が見つかりません",
                status_code=status.HTTP_404_NOT_FOUND
            )


