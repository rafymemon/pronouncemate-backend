# api/notifications.py
"""
Notification generation. Kept separate from views.py/progression.py to
avoid a circular import (views.py already imports from progression.py,
and both would need to import Notification-creation helpers).
"""
from django.utils import timezone
from datetime import timedelta

from .models import Notification, PracticeSession, UserProfile

# Milestones that trigger a "streak achievement" notification. Chosen to
# feel earned without spamming — 3-day is the first meaningful one, then
# roughly doubling.
STREAK_MILESTONES = {3, 7, 14, 30, 60, 100}


def _current_streak(user) -> int:
    """
    Same consecutive-day-streak logic as views._calculate_streak(), kept
    as its own small copy here rather than imported, to avoid a circular
    import between views.py and this module.
    """
    session_dates = PracticeSession.objects.filter(user=user).dates(
        'created_at', 'day', order='DESC'
    )
    if not session_dates:
        return 0

    today = timezone.now().date()
    if session_dates[0] < today - timedelta(days=1):
        return 0

    streak = 1
    for i in range(1, len(session_dates)):
        if session_dates[i] == session_dates[i - 1] - timedelta(days=1):
            streak += 1
        else:
            break
    return streak


def check_streak_notifications(user) -> None:
    """
    Called once per completed session (from AnalyzeView). Fires:
      - a milestone notification when the current streak lands exactly
        on one of STREAK_MILESTONES
      - a "new longest streak" notification when the current streak
        beats the user's previously recorded best (UserProfile.longest_streak)
    """
    streak = _current_streak(user)
    if streak == 0:
        return

    if streak in STREAK_MILESTONES:
        Notification.objects.create(
            user=user,
            notif_type='streak_milestone',
            title=f'{streak}-Day Streak!',
            message=f"You've practiced {streak} days in a row. Keep it going!",
        )

    profile, _ = UserProfile.objects.get_or_create(
        user=user, defaults={'proficiency_level': 'beginner'}
    )
    if streak > profile.longest_streak:
        profile.longest_streak = streak
        profile.save(update_fields=['longest_streak'])
        if streak > 1:
            Notification.objects.create(
                user=user,
                notif_type='longest_streak',
                title='New Longest Streak!',
                message=f'{streak} days is your new personal best streak.',
            )


def notify_focus_script_ready(user, focus_script) -> None:
    """focus_script is the FocusScript instance just created, or None if
    check_and_maybe_generate_focus_script() didn't generate one this call."""
    if not focus_script:
        return
    preview_words = ', '.join(f'"{w}"' for w in (focus_script.source_words or [])[:3])
    Notification.objects.create(
        user=user,
        notif_type='focus_script',
        title='Focus Script Ready',
        message=f'A new practice script targeting {preview_words} is ready for you.',
    )


def notify_level_up(user, new_level: str) -> None:
    Notification.objects.create(
        user=user,
        notif_type='level_up',
        title='Level Up!',
        message=f"You've been promoted to {new_level.title()}. Great work!",
    )