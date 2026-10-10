# api/urls.py
"""
API URL Configuration for PronounceMate
Includes main pronunciation + Practice Mistakes endpoints
"""

from django.urls import path
from .views import (
    # Core
    AnalyzeView,
    # Practice Mistakes
    PracticeScriptView,
    PracticeMistakesView,
    PracticeProgressView,
    # History
    SessionHistoryView,
    # Scripts
    ScriptListView,
    ScriptUploadView,
    GenerateScriptView,
    # Dashboard & Analytics
    dashboard_summary,
    progress_analytics,
    # Script Rotation
    ScriptRotationView,
    # Improvement Phase
    ProgressionStatusView,
    ConfirmPromotionView,
    WordMasteryListView,
    FocusScriptListView,
    LatestFocusScriptView,
    FocusScriptStatsView,

    NotificationListView,
    MarkNotificationsReadView,
    NotificationDeleteView,
)
# Authentication (register / email verification / login / password reset)
from .auth_views import (
    RegisterView,
    VerifyEmailView,
    ResendVerificationView,
    LoginView,
    LogoutView,
    ForgotPasswordView,
    VerifyResetOTPView,
    ResetPasswordView,
)
# Signed-in account management (profile, change password, change email)
from .account_views import (
    ProfileView,
    ChangePasswordView,
    EmailChangeRequestView,
    EmailChangeConfirmView,
)
# Focus Script before/after flow lives in its own module
from .focus_views import FocusScriptDetailView, CompleteFocusScriptView
# Stat-card drill-downs + single-session detail
from .stats_views import StatDetailView, SessionDetailView

urlpatterns = [
    # Auth
    path('register/', RegisterView.as_view(), name='register'),
    path('verify-email/', VerifyEmailView.as_view(), name='verify-email'),
    path('resend-verification/', ResendVerificationView.as_view(), name='resend-verification'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('forgot-password/', ForgotPasswordView.as_view(), name='forgot-password'),
    path('verify-reset-otp/', VerifyResetOTPView.as_view(), name='verify-reset-otp'),
    path('reset-password/', ResetPasswordView.as_view(), name='reset-password'),
    path('change-password/', ChangePasswordView.as_view(), name='change-password'),
    path('email-change/request/', EmailChangeRequestView.as_view(), name='email-change-request'),
    path('email-change/confirm/', EmailChangeConfirmView.as_view(), name='email-change-confirm'),

    # Core Analysis
    path('analyze/', AnalyzeView.as_view(), name='analyze'),

    # Practice Mistakes
    path('practice/script/', PracticeScriptView.as_view(), name='practice_script'),
    path('practice/', PracticeMistakesView.as_view(), name='practice_mistakes'),
    path('practice/progress/', PracticeProgressView.as_view(), name='practice_progress'),

    # Session History
    path('sessions/', SessionHistoryView.as_view(), name='session_history'),
    path('sessions/<int:session_id>/', SessionDetailView.as_view(), name='session_detail'),

    # Stat-card drill-downs (metric = sessions|accuracy|words|streak|errors)
    path('stats/<str:metric>/', StatDetailView.as_view(), name='stat_detail'),

    # User Profile (email is read-only here; see email-change/ above)
    path('profile/', ProfileView.as_view(), name='profile'),

    # Scripts
    path('scripts/', ScriptListView.as_view(), name='script_list'),
    path('scripts/upload/', ScriptUploadView.as_view(), name='script_upload'),
    path('scripts/generate/', GenerateScriptView.as_view(), name='script_generate'),

    # Dashboard & Analytics
    path('dashboard/', dashboard_summary, name='dashboard'),
    path('analytics/', progress_analytics, name='analytics'),

    path('scripts/rotation/', ScriptRotationView.as_view(), name='script-rotation'),

    # ── Improvement Phase ────────────────────────────────
    path('progression/', ProgressionStatusView.as_view(), name='progression-status'),
    path('progression/confirm/', ConfirmPromotionView.as_view(), name='progression-confirm'),

    path('mastery/', WordMasteryListView.as_view(), name='word-mastery-list'),
    path('focus-scripts/latest/', LatestFocusScriptView.as_view(), name='focus-script-latest'),
    path('focus-scripts/stats/', FocusScriptStatsView.as_view(), name='focus-script-stats'),
    path('focus-scripts/<int:script_id>/complete/', CompleteFocusScriptView.as_view(), name='focus-script-complete'),
    path('focus-scripts/<int:script_id>/', FocusScriptDetailView.as_view(), name='focus-script-detail'),
    path('focus-scripts/', FocusScriptListView.as_view(), name='focus-script-list'),

    path('notifications/', NotificationListView.as_view(), name='notifications'),
    path('notifications/read/', MarkNotificationsReadView.as_view(), name='notifications-read'),
    path('notifications/<int:notification_id>/', NotificationDeleteView.as_view(), name='notification-delete'),
]