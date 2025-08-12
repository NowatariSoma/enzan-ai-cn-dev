from accounts.views import ConfirmRegistrationAPIView, ForgotPasswordAPIView, LoginAPIView, LogoutAPIView, ResetPasswordAPIView
from django.urls import path


urlpatterns = [
    path(
        "confirm-registration/",
        ConfirmRegistrationAPIView.as_view(),
        name="confirm-registration",
    ),
    path('login', LoginAPIView.as_view(), name='login'),
    path('logout', LogoutAPIView.as_view(), name='logout'),
    path('forgot-password', ForgotPasswordAPIView.as_view(), name='forgot-password'),
    path('reset-password', ResetPasswordAPIView.as_view(), name='reset-password'),
]
