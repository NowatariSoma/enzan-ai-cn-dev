from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import FeatureViewSet, LocationViewSet

router = DefaultRouter()
router.register(r'features-master', FeatureViewSet, basename='feature-master')
router.register(r'', LocationViewSet, basename='location')

urlpatterns = [
    path('', include(router.urls)),
]
