# api/signals.py
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone

from .models import PracticeSession, UserScriptProgress


@receiver(post_save, sender=PracticeSession)
def update_script_progress(sender, instance, created, **kwargs):
    """
    Whenever a PracticeSession is created, update (or create) the
    UserScriptProgress row for that user + script pair. Fires only on
    creation (not on later saves/updates) to avoid double-counting.
    """
    if not created:
        return
    if instance.script_id is None:
        # Ad-hoc script-based sessions without a saved Script row —
        # nothing to track for rotation purposes.
        return

    progress, _ = UserScriptProgress.objects.get_or_create(
        user=instance.user,
        script_id=instance.script_id,
    )
    progress.times_practiced += 1
    progress.last_practiced_at = instance.created_at or timezone.now()
    progress.last_score = instance.accuracy

    if progress.best_score is None or instance.accuracy > progress.best_score:
        progress.best_score = instance.accuracy

    progress.save()