# api/rotation.py
"""
Smart Rotation logic for PronounceMate.

Every rotation request generates FRESH scripts live via Gemini
(api/gemini_service.py) so users get new content each time instead of
recycling the same fixed library. Each generated script is persisted as a
new Script row (is_builtin=False), so UserScriptProgress / history
tracking keeps working exactly as it does for any other script.

If Gemini fails (network/API error, malformed response), rotation falls
back to the original curated-pool tiered logic so a Gemini outage never
blocks a practice session:
  Tier 1 — built-in scripts the user has never practiced. Picked randomly.
  Tier 2 — once Tier 1 is exhausted, top up with practiced built-ins,
           oldest last_practiced_at first.
"""
import logging
import random

from django.contrib.auth.models import User

from .models import Script, UserScriptProgress
from . import gemini_service

logger = logging.getLogger(__name__)


def get_rotated_scripts(user: User, mode: str = None, level: str = None, count: int = 5):
    """
    Returns up to `count` fresh Script instances, generated live by Gemini
    for this user's mode/level and saved as is_builtin=False rows. Falls
    back to the curated built-in pool if generation fails or mode/level
    aren't both provided (generation needs both to target the content).
    """
    if not mode or not level:
        return _get_pool_fallback_scripts(user, mode=mode, level=level, count=count)

    generated = gemini_service.generate_rotation_scripts(mode=mode, level=level, count=count)

    if not generated:
        logger.warning(
            f'Smart Rotation: Gemini returned no scripts for '
            f'user={user.username}, mode={mode}, level={level}; falling back to pool.'
        )
        return _get_pool_fallback_scripts(user, mode=mode, level=level, count=count)

    saved_scripts = []
    for item in generated:
        script = Script.objects.create(
            title=item['title'],
            description='AI-generated rotation script',
            level=level,
            mode=mode,
            content=item['content'],
            is_builtin=False,
            uploaded_by=None,
        )
        saved_scripts.append(script)

    return saved_scripts


def get_single_rotated_script(user: User, mode: str, level: str):
    """Convenience wrapper for flows that just want ONE next script
    (e.g. a 'Surprise me' button) instead of a list of picks."""
    results = get_rotated_scripts(user, mode=mode, level=level, count=1)
    return results[0] if results else None


def _get_pool_fallback_scripts(user: User, mode: str = None, level: str = None, count: int = 5):
    """
    Original curated-pool tiered logic. Used only when live Gemini
    generation is unavailable, or when mode/level aren't both supplied.
    """
    base_qs = Script.objects.filter(is_builtin=True)
    if mode:
        base_qs = base_qs.filter(mode=mode)
    if level:
        base_qs = base_qs.filter(level=level)

    practiced_script_ids = UserScriptProgress.objects.filter(
        user=user
    ).values_list('script_id', flat=True)

    # ── Tier 1: never practiced ──────────────────────────
    unpracticed = list(base_qs.exclude(id__in=practiced_script_ids))
    random.shuffle(unpracticed)

    if len(unpracticed) >= count:
        return unpracticed[:count]

    # ── Tier 2: top up with oldest-practiced ─────────────
    remaining_needed = count - len(unpracticed)

    stale_progress = (
        UserScriptProgress.objects
        .filter(user=user, script__in=base_qs)
        .select_related('script')
        .order_by('last_practiced_at')[:remaining_needed]
    )
    stale_scripts = [p.script for p in stale_progress]

    return unpracticed + stale_scripts