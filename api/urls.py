# api/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # Auth
    path('register/', views.RegisterView.as_view(),      name='register'),
    path('login/',    views.LoginView.as_view(),         name='login'),
    # Core feature
    path('analyze/',  views.AnalyzeView.as_view(),       name='analyze'),
    # User data
    path('sessions/', views.SessionHistoryView.as_view(),name='sessions'),
    path('profile/',  views.ProfileView.as_view(),       name='profile'),
    path('scripts/',  views.ScriptListView.as_view(),    name='scripts'),
    path('scripts/upload/', views.ScriptUploadView.as_view(), name='script-upload'),
]