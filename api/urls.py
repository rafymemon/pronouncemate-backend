# api/urls.py
"""
API URL Configuration for PronounceMate
Includes main pronunciation + Practice Mistakes endpoints
"""

from django.urls import path
from .views import (
    # Auth
    RegisterView,
    LoginView,
    # Core
    AnalyzeView,
    # Practice Mistakes
    PracticeScriptView,
    PracticeMistakesView,
    PracticeProgressView,
    # History & Profile
    SessionHistoryView,
    ProfileView,
    # Scripts
    ScriptListView,
    ScriptUploadView,
    # Dashboard & Analytics
    dashboard_summary,
    progress_analytics,
    # Script Rotation
    ScriptRotationView,
)

urlpatterns = [
    # Auth
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),

    # Core Analysis
    path('analyze/', AnalyzeView.as_view(), name='analyze'),

    # Practice Mistakes
    path('practice/script/', PracticeScriptView.as_view(), name='practice_script'),
    path('practice/', PracticeMistakesView.as_view(), name='practice_mistakes'),
    path('practice/progress/', PracticeProgressView.as_view(), name='practice_progress'),

    # Session History
    path('sessions/', SessionHistoryView.as_view(), name='session_history'),

    # User Profile
    path('profile/', ProfileView.as_view(), name='profile'),

    # Scripts
    path('scripts/', ScriptListView.as_view(), name='script_list'),
    path('scripts/upload/', ScriptUploadView.as_view(), name='script_upload'),

    # Dashboard & Analytics
    path('dashboard/', dashboard_summary, name='dashboard'),
    path('analytics/', progress_analytics, name='analytics'),

    path('scripts/rotation/', ScriptRotationView.as_view(), name='script-rotation'),
]