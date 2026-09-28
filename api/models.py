# api/models.py

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


LEVEL_CHOICES = [
    ('beginner',     'Beginner'),
    ('intermediate', 'Intermediate'),
    ('expert',       'Expert'),
]


class Script(models.Model):

    MODE_CHOICES = [
        ('public_speaking',    'Public Speaking'),
        ('daily_conversation', 'Daily Conversation'),
        ('script_based',       'Script-Based'),
    ]

    title       = models.CharField(max_length=200)
    description = models.CharField(max_length=400, blank=True)
    level       = models.CharField(max_length=15, choices=LEVEL_CHOICES)
    mode        = models.CharField(max_length=25, choices=MODE_CHOICES)
    content     = models.TextField()
    is_builtin  = models.BooleanField(default=False)
    uploaded_by = models.ForeignKey(
        User, null=True, blank=True, on_delete=models.SET_NULL
    )
    created_at  = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['level', 'title']

    def word_count(self):
        return len(self.content.split())

    def __str__(self):
        return f'[{self.mode}][{self.level}] {self.title}'


class PracticeSession(models.Model):
    """Stores one complete practice session result per user."""
    user           = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sessions')
    created_at     = models.DateTimeField(auto_now_add=True)

    script = models.ForeignKey(
        Script,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
    )
    transcript     = models.TextField(blank=True, help_text='What Whisper heard')
    accuracy       = models.FloatField(help_text='0.0 to 100.0')
    grade          = models.CharField(max_length=4,  help_text='A+, A, B, C, D')
    grade_label    = models.CharField(max_length=20, help_text='Excellent, Good, etc.')
    severity_total = models.IntegerField(default=0)
    total_words    = models.IntegerField(default=0)
    flagged_words  = models.IntegerField(default=0)
    skipped_words  = models.IntegerField(default=0)
    result_json    = models.JSONField(help_text='Full pipeline output stored as JSON')
    audio_file     = models.FileField(upload_to='sessions/', blank=True, null=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Practice Session'

    def __str__(self):
        return f'{self.user.username} | {self.accuracy:.1f}% | {self.created_at:%Y-%m-%d}'


class PhonemeError(models.Model):
    """Individual phoneme error record for analytics."""
    ERROR_TYPES = [
        ('substitution', 'Substitution'),
        ('omission',     'Omission'),
        ('insertion',    'Insertion'),
        ('skipped',      'Skipped Word'),
    ]
    session    = models.ForeignKey(PracticeSession, on_delete=models.CASCADE, related_name='errors')
    word       = models.CharField(max_length=100)
    error_type = models.CharField(max_length=20, choices=ERROR_TYPES)
    expected   = models.CharField(max_length=20, blank=True, help_text='Expected phoneme')
    spoken     = models.CharField(max_length=20, blank=True, null=True)
    severity   = models.IntegerField(default=0)
    position   = models.IntegerField(default=0, help_text='Position in phoneme sequence')

    class Meta:
        ordering = ['session', 'word']

    def __str__(self):
        return f'{self.word}: {self.expected} -> {self.spoken} ({self.error_type})'


class UserProfile(models.Model):
    """Extended user info for PronounceMate."""
    ACCENTS = [('american','American English'),('british','British English')]

    user             = models.OneToOneField(User, on_delete=models.CASCADE)
    proficiency_level= models.CharField(max_length=15, choices=LEVEL_CHOICES, default='beginner')
    reference_accent = models.CharField(max_length=20, choices=ACCENTS, default='american')
    voice_feedback   = models.BooleanField(default=True)
    daily_reminders       = models.BooleanField(default=True)
    auto_save_recordings  = models.BooleanField(default=False)

    # NEW: tracks the user's best-ever consecutive-day streak, so the
    # notification system can tell "you're on a streak" (already known
    # via PracticeSession dates) apart from "you just beat your personal
    # record" (which needs a persisted high-water mark).
    longest_streak   = models.IntegerField(default=0)

    total_sessions   = models.IntegerField(default=0)
    best_score       = models.FloatField(default=0.0)
    created_at       = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Profile: {self.user.username}'


class UserScriptProgress(models.Model):
    """
    Tracks how a specific user has engaged with a specific script.
    Powers Smart Rotation: unpracticed scripts are prioritized first;
    once a mode/level pool is exhausted, the oldest-practiced script
    resurfaces first (spaced-repetition-lite, no extra infra needed).
    """
    user              = models.ForeignKey(User, on_delete=models.CASCADE, related_name='script_progress')
    script            = models.ForeignKey(Script, on_delete=models.CASCADE, related_name='user_progress')
    times_practiced   = models.IntegerField(default=0)
    last_practiced_at = models.DateTimeField(null=True, blank=True)
    best_score        = models.FloatField(null=True, blank=True)
    last_score        = models.FloatField(null=True, blank=True)
    created_at        = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'script')
        ordering = ['-last_practiced_at']
        verbose_name = 'User Script Progress'
        verbose_name_plural = 'User Script Progress'

    def __str__(self):
        return f'{self.user.username} × {self.script.title} ({self.times_practiced}x)'



from django.utils import timezone


class WordMastery(models.Model):
    """
    Tracks an individual mispronounced word until the user demonstrates
    consistent, retained correctness (3 consecutive correct attempts
    across separate sessions).
    """
    STATUS_CHOICES = [('active', 'Active'), ('mastered', 'Mastered')]

    user               = models.ForeignKey(User, on_delete=models.CASCADE, related_name='word_mastery')
    word               = models.CharField(max_length=100)
    error_type         = models.CharField(max_length=20, blank=True)
    consecutive_correct= models.IntegerField(default=0)
    status             = models.CharField(max_length=10, choices=STATUS_CHOICES, default='active')
    last_seen_at       = models.DateTimeField(auto_now=True)
    mastered_at        = models.DateTimeField(null=True, blank=True)
    created_at         = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'word')
        ordering = ['-last_seen_at']

    def __str__(self):
        return f'{self.user.username}: {self.word} ({self.status})'


class SessionErrorLog(models.Model):
    """Per-word error events, feeding both progression gates and Focus Script generation."""
    session             = models.ForeignKey(PracticeSession, on_delete=models.CASCADE, related_name='error_log_entries')
    user                = models.ForeignKey(User, on_delete=models.CASCADE, related_name='error_logs')
    word                = models.CharField(max_length=100)
    error_type          = models.CharField(max_length=20)
    mode                = models.CharField(max_length=25, blank=True)
    level_at_time       = models.CharField(max_length=15, blank=True)
    used_in_focus_script= models.BooleanField(default=False)
    created_at          = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.user.username}: {self.word} ({self.error_type})'


class LevelProgressionStatus(models.Model):
    """Single row per user tracking global level-progression eligibility."""
    user                 = models.OneToOneField(User, on_delete=models.CASCADE, related_name='progression_status')
    current_level        = models.CharField(max_length=15, choices=LEVEL_CHOICES, default='beginner')
    level_changed_at     = models.DateTimeField(default=timezone.now)
    rolling_avg_score    = models.FloatField(default=0.0)
    rolling_error_rate   = models.FloatField(default=0.0)
    mastery_ratio        = models.FloatField(default=0.0)
    flagged_phoneme_rate = models.FloatField(default=0.0)
    eligible_for_promotion = models.BooleanField(default=False)
    last_evaluated_at    = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f'{self.user.username}: {self.current_level} (eligible={self.eligible_for_promotion})'




class FocusScript(models.Model):
    """AI-generated remediation script compiled from a user's recurring error words."""
    user             = models.ForeignKey(User, on_delete=models.CASCADE, related_name='focus_scripts')
    generated_at     = models.DateTimeField(auto_now_add=True)
    source_words     = models.JSONField(default=list)
    regression_words = models.JSONField(default=list, blank=True)
    script_text      = models.TextField()
    completed        = models.BooleanField(default=False)
    completed_at     = models.DateTimeField(null=True, blank=True)
    pre_error_rate   = models.FloatField(null=True, blank=True)
    post_error_rate  = models.FloatField(null=True, blank=True)


class Notification(models.Model):
    """
    In-app notification feed. Populated by:
      - notifications.notify_level_up()          — ConfirmPromotionView
      - notifications.check_streak_notifications() — AnalyzeView, per session
      - notifications.notify_focus_script_ready() — AnalyzeView, per session
    """
    TYPE_CHOICES = [
        ('level_up',         'Level Up'),
        ('streak_milestone', 'Streak Milestone'),
        ('longest_streak',   'Longest Streak'),
        ('focus_script',     'Focus Script Ready'),
    ]

    user       = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    notif_type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    title      = models.CharField(max_length=120)
    message    = models.CharField(max_length=300)
    is_read    = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.user.username}: {self.title} ({"read" if self.is_read else "unread"})'