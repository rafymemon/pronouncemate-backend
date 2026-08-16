# api/models.py
# api/models.py

from django.db import models
from django.contrib.auth.models import User


class Script(models.Model):

    LEVEL_CHOICES = [
        ('beginner',          'Beginner'),
        ('intermediate',      'Intermediate'),
        ('upper_intermediate','Upper Intermediate'),
        ('advanced',          'Advanced'),
        ('fluent',            'Fluent'),
    ]

    MODE_CHOICES = [
        ('public_speaking',    'Public Speaking'),
        ('daily_conversation', 'Daily Conversation'),
        ('script_based',       'Script-Based'),
    ]

    title       = models.CharField(max_length=200)
    description = models.CharField(max_length=400, blank=True)
    level       = models.CharField(max_length=25, choices=LEVEL_CHOICES)
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
        null=True,      # ← ADD THIS
        blank=True,     # ← ADD THIS
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
    LEVELS = [('beginner','Beginner'),('intermediate','Intermediate'),('advanced','Advanced')]
    ACCENTS = [('american','American English'),('british','British English')]

    user             = models.OneToOneField(User, on_delete=models.CASCADE)
    proficiency_level= models.CharField(max_length=20, choices=LEVELS, default='beginner')
    reference_accent = models.CharField(max_length=20, choices=ACCENTS, default='american')
    voice_feedback   = models.BooleanField(default=True)
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