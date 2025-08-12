from django.urls import path
from projects.views import ProjectApiView, ProjectDetailApiView


urlpatterns = [
    path('', ProjectApiView.as_view(), name='project-list'),
    path('<int:pk>', ProjectDetailApiView.as_view(), name='project-detail'),
]
