# api/rotation.py
"""
Smart Rotation logic for PronounceMate.

Tier 1 — scripts the user has never practiced (no UserScriptProgress row,
          or times_practiced == 0). Picked randomly.
Tier 2 — once Tier 1 is exhausted for a mode/level, fall back to scripts
          the user HAS practiced, oldest last_practiced_at first.

Scoped to is_builtin=True scripts only — the rotation pool is the curated
library, not a user's own uploads (those already live in "My Scripts").
"""
import random

from .models import Script, UserScriptProgress


def get_rotated_scripts(user, mode: str = None, level: str = None, count: int = 5):
    """
    Returns up to `count` Script instances for the given user, prioritizing
    never-practiced scripts, then topping up with the least-recently
    practiced ones if the pool runs short.
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


def get_single_rotated_script(user, mode: str, level: str):
    """Convenience wrapper for flows that just want ONE next script
    (e.g. a 'Surprise me' button) instead of a list of picks."""
    results = get_rotated_scripts(user, mode=mode, level=level, count=1)
    return results[0] if results else None