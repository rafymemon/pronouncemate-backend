# api/progression.py
"""
Improvement Phase core logic: Word Mastery tracking, Level Progression
gating, and Focus Script triggering. Called from AnalyzeView after each
completed session.
"""
import logging
import random
from collections import Counter
from django.utils import timezone

from .models import (
    PracticeSession, PhonemeError, UserProfile,
    WordMastery, SessionErrorLog, LevelProgressionStatus, FocusScript,
)
from .gemini_service import generate_focus_script_text

logger = logging.getLogger(__name__)

LEVEL_ORDER = ['beginner', 'intermediate', 'expert']

MASTERY_REQUIRED_STREAK = 3
FOCUS_SCRIPT_TRIGGER_THRESHOLD = 5
FOCUS_SCRIPT_MAX_WORDS = None  # None = include every error word


REGRESSION_CHECK_PROBABILITY = 0.1  # ~1-in-10, per design doc Section 3
REGRESSION_MAX_WORDS = 2

# ── Progression gate thresholds (Section 2 of the design doc) ──────
GATE_MIN_SESSIONS   = 5
GATE_MIN_AVG_SCORE   = 80
GATE_MAX_ERROR_RATE  = 15
GATE_MIN_MASTERY_PCT = 70
GATE_MAX_PHONEME_PCT = 10


def log_session_errors(session, user, word_results):
    """Persist one SessionErrorLog row per non-correct word in this session."""
    profile = UserProfile.objects.filter(user=user).first()
    level = profile.proficiency_level if profile else ''
    mode = session.script.mode if session.script else ''

    entries = []
    for w in word_results:
        if w.get('type') in ('correct', 'inserted'):
            continue
        word = (w.get('word') or '').strip().lower()
        if not word:
            continue
        entries.append(SessionErrorLog(
            session=session, user=user, word=word,
            error_type=w.get('type', 'unknown'),
            mode=mode, level_at_time=level,
        ))
    if entries:
        SessionErrorLog.objects.bulk_create(entries)


def update_word_mastery(user, word_results):
    """
    A word enters tracking the first time it's mispronounced. Needs
    3 consecutive correct attempts across separate sessions to be
    marked mastered. Mastered words that are re-tested (via regression
    re-insertion) and errored again revert to active.
    """
    for w in word_results:
        word = (w.get('word') or '').strip().lower()
        if not word:
            continue
        is_correct = w.get('type') == 'correct'

        mastery = WordMastery.objects.filter(user=user, word=word).first()

        if mastery is None:
            if is_correct:
                continue  # never flagged before — nothing to track
            WordMastery.objects.create(
                user=user, word=word, error_type=w.get('type', ''),
                status='active', consecutive_correct=0,
            )
            continue

        if mastery.status == 'mastered':
            if not is_correct:
                mastery.status = 'active'
                mastery.consecutive_correct = 0
                mastery.mastered_at = None
                mastery.error_type = w.get('type', mastery.error_type)
                mastery.save()
            # if correct, leave mastered row untouched (last_seen_at auto-updates)
            continue

        if is_correct:
            mastery.consecutive_correct += 1
            if mastery.consecutive_correct >= MASTERY_REQUIRED_STREAK:
                mastery.status = 'mastered'
                mastery.mastered_at = timezone.now()
        else:
            mastery.consecutive_correct = 0
            mastery.error_type = w.get('type', mastery.error_type)

        mastery.save()


def evaluate_progression(user):
    """Recompute the five progression gates and update LevelProgressionStatus."""
    profile, _ = UserProfile.objects.get_or_create(user=user, defaults={'proficiency_level': 'beginner'})
    status_obj, _ = LevelProgressionStatus.objects.get_or_create(
        user=user, defaults={'current_level': profile.proficiency_level}
    )

    recent = list(PracticeSession.objects.filter(user=user).order_by('-created_at')[:5])
    sessions_at_level = PracticeSession.objects.filter(
        user=user, created_at__gte=status_obj.level_changed_at
    ).count()

    if not recent:
        status_obj.eligible_for_promotion = False
        status_obj.save()
        return status_obj, {}

    rolling_avg_score = sum(s.accuracy for s in recent) / len(recent)
    total_words = sum(s.total_words for s in recent) or 1
    total_flagged = sum(s.flagged_words for s in recent)
    rolling_error_rate = (total_flagged / total_words) * 100

    mastery_qs = WordMastery.objects.filter(user=user)
    total_tracked = mastery_qs.count()
    mastered_count = mastery_qs.filter(status='mastered').count()
    mastery_ratio = (mastered_count / total_tracked * 100) if total_tracked else 100.0

    recent_ids = [s.id for s in recent]
    substitution_count = PhonemeError.objects.filter(
        session_id__in=recent_ids, error_type='substitution'
    ).count()
    flagged_phoneme_rate = (substitution_count / total_words) * 100

    gates = {
        'sessions':  sessions_at_level >= GATE_MIN_SESSIONS,
        'score':     rolling_avg_score >= GATE_MIN_AVG_SCORE,
        'error_rate':rolling_error_rate <= GATE_MAX_ERROR_RATE,
        'mastery':   mastery_ratio >= GATE_MIN_MASTERY_PCT,
        'phoneme':   flagged_phoneme_rate <= GATE_MAX_PHONEME_PCT,
    }

    current_idx = LEVEL_ORDER.index(status_obj.current_level)
    at_top_level = current_idx >= len(LEVEL_ORDER) - 1
    eligible = all(gates.values()) and not at_top_level

    status_obj.rolling_avg_score    = round(rolling_avg_score, 1)
    status_obj.rolling_error_rate   = round(rolling_error_rate, 1)
    status_obj.mastery_ratio        = round(mastery_ratio, 1)
    status_obj.flagged_phoneme_rate = round(flagged_phoneme_rate, 1)
    status_obj.eligible_for_promotion = eligible
    status_obj.save()

    return status_obj, gates


def promote_if_eligible(user):
    """
    Auto-applies the level-up when all gates pass. (Open design decision:
    could instead surface eligible=True and require user confirmation —
    easy to swap later, see Improvement Phase doc Section 7.)
    """
    status_obj, gates = evaluate_progression(user)
    if not status_obj.eligible_for_promotion:
        return None

    current_idx = LEVEL_ORDER.index(status_obj.current_level)
    if current_idx >= len(LEVEL_ORDER) - 1:
        return None

    new_level = LEVEL_ORDER[current_idx + 1]
    status_obj.current_level = new_level
    status_obj.level_changed_at = timezone.now()
    status_obj.eligible_for_promotion = False
    status_obj.save()

    profile = UserProfile.objects.get(user=user)
    profile.proficiency_level = new_level
    profile.save(update_fields=['proficiency_level'])

    logger.info(f'{user.username} promoted to {new_level}')
    return new_level


def _pick_regression_words(user):
    """
    With low probability, select mastered words to re-test for retention.
    If a regression word is mispronounced in the resulting script,
    update_word_mastery() will revert its status back to active.
    """
    if random.random() > REGRESSION_CHECK_PROBABILITY:
        return []
    mastered = list(
        WordMastery.objects.filter(user=user, status='mastered')
        .values_list('word', flat=True)
    )
    if not mastered:
        return []
    random.shuffle(mastered)
    return mastered[:REGRESSION_MAX_WORDS]


def check_and_maybe_generate_focus_script(user):
    """
    Rolling trigger: once >= FOCUS_SCRIPT_TRIGGER_THRESHOLD new distinct
    error words have accumulated since the last Focus Script (or ever,
    if none exists), generate a new one via Gemini.
    """
    last_focus = FocusScript.objects.filter(user=user).order_by('-generated_at').first()
    since = last_focus.generated_at if last_focus else None

    error_qs = SessionErrorLog.objects.filter(user=user)
    if since:
        error_qs = error_qs.filter(created_at__gt=since)

    distinct_words = list(error_qs.values_list('word', flat=True).distinct())
    if len(distinct_words) < FOCUS_SCRIPT_TRIGGER_THRESHOLD:
        return None

    counts = Counter(error_qs.values_list('word', flat=True))
    top_words = [w for w, _ in counts.most_common(FOCUS_SCRIPT_MAX_WORDS)]

    regression_words = _pick_regression_words(user)
    # Don't double-count a word that's somehow in both lists
    regression_words = [w for w in regression_words if w not in top_words]
    combined_words = top_words + regression_words

    profile = UserProfile.objects.filter(user=user).first()
    level = profile.proficiency_level if profile else 'beginner'

    script_text = generate_focus_script_text(combined_words, level)

    focus_script = FocusScript.objects.create(
        user=user, source_words=top_words, regression_words=regression_words,
        script_text=script_text,
        pre_error_rate=100.0,  # baseline: every source word was a logged error
    )
    error_qs.update(used_in_focus_script=True)

    logger.info(
        f'Generated Focus Script for {user.username}: {top_words}'
        + (f' (+ regression: {regression_words})' if regression_words else '')
    )
    return focus_script