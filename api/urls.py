# api/urls.py — make sure these exist
from django.urls import path
from .views import (
    RegisterView, LoginView, AnalyzeView,
    SessionHistoryView, ProfileView,
    ScriptListView, ScriptUploadView,
)

urlpatterns = [
    path('register/',       RegisterView.as_view()),
    path('login/',          LoginView.as_view()),
    path('analyze/',        AnalyzeView.as_view()),
    path('sessions/',       SessionHistoryView.as_view()),
    path('profile/',        ProfileView.as_view()),
    path('scripts/',        ScriptListView.as_view()),
    path('scripts/upload/', ScriptUploadView.as_view()),
]