# api/focus_results.py
"""
Builds the before/after comparison for a completed Focus Script.

BEFORE = the error words (and how they were wrong) that triggered the
         Focus Script, reconstructed from SessionErrorLog.
AFTER  = how those same words were scored in the practice session.
"""
import logging
from collections import Counter
from django.utils import timezone

from .models import FocusScript, SessionErrorLog, WordMastery

logger = logging.getLogger(__name__)


def _rank(error_type):
    """0 = correct, 1 = minor slip, 2 = hard error / skipped."""
    if error_type in ('correct', 'accent'):
        return 0
    if error_type == 'minor':
        return 1
    return 2


def compute_baseline(focus_script):
    """
    Per source word: how it was wrong before and how many times.

    Uses the same window that generated the Focus Script: error logs
    created after the previous Focus Script and up to this one's
    generation time.
    """
    prev = (
        FocusScript.objects
        .filter(user=focus_script.user, generated_at__lt=focus_script.generated_at)
        .order_by('-generated_at')
        .first()
    )
    qs = SessionErrorLog.objects.filter(
        user=focus_script.user, created_at__lte=focus_script.generated_at
    )
    if prev:
        qs = qs.filter(created_at__gt=prev.generated_at)

    words = []
    for w in focus_script.source_words or []:
        w = str(w).strip().lower()
        if w and w not in words:
            words.append(w)

    counts = Counter()
    last_type = {}
    rows = qs.filter(word__in=words).order_by('created_at').values('word', 'error_type')
    for row in rows:
        counts[row['word']] += 1
        last_type[row['word']] = row['error_type']

    return [
        {
            'word': w,
            'error_type': last_type.get(w, 'unknown'),
            'error_count': counts.get(w, 0),
        }
        for w in words
    ]


def build_focus_result(focus_script, session):
    """Compare the session's word results against the Focus Script baseline."""
    baseline = compute_baseline(focus_script)
    session_words = (session.result_json or {}).get('words', [])

    # Worst occurrence wins if a word appears more than once in the script.
    by_word = {}
    for w in session_words:
        if w.get('type') == 'inserted':
            continue
        key = (w.get('word') or '').strip().lower()
        if not key:
            continue
        current = by_word.get(key)
        if current is None or _rank(w.get('type')) > _rank(current.get('type')):
            by_word[key] = w

    mastery_map = {
        m.word: m for m in WordMastery.objects.filter(
            user=focus_script.user, word__in=[b['word'] for b in baseline]
        )
    }

    rows = []
    corrected = improved = unchanged = worse = not_practiced = 0
    still_hard = 0

    for b in baseline:
        word = b['word']
        before_type = b['error_type']
        before_rank = 2 if before_type == 'unknown' else _rank(before_type)
        after = by_word.get(word)
        mastery = mastery_map.get(word)

        row = {
            'word': word,
            'before_type': before_type,
            'before_count': b['error_count'],
            'after_type': None,
            'after_score': None,
            'tip': None,
            'mastered': bool(mastery and mastery.status == 'mastered'),
            'mastery_streak': mastery.consecutive_correct if mastery else 0,
        }

        if after is None:
            row['status'] = 'not_practiced'
            not_practiced += 1
            rows.append(row)
            continue

        after_type = after.get('type') or 'correct'
        after_rank = _rank(after_type)
        score = after.get('per_word_score')
        row['after_type'] = after_type
        row['after_score'] = round(float(score), 1) if score is not None else None
        row['tip'] = after.get('explanation')

        if after_rank == 0:
            row['status'] = 'corrected'
            corrected += 1
        elif after_rank < before_rank:
            row['status'] = 'improved'
            improved += 1
        elif after_rank == before_rank:
            row['status'] = 'unchanged'
            unchanged += 1
        else:
            row['status'] = 'worse'
            worse += 1

        if after_rank == 2:
            still_hard += 1
        rows.append(row)

    measured = len(rows) - not_practiced
    progressed = corrected + improved

    pre_rate = focus_script.pre_error_rate if focus_script.pre_error_rate is not None else 100.0
    post_rate = round(still_hard / measured * 100, 1) if measured else None

    if measured == 0:
        verdict = 'none'
        headline = 'None of the target words were detected in this recording.'
    elif corrected == measured:
        verdict = 'excellent'
        headline = f'Excellent! All {measured} error words are now correct.'
    elif progressed / measured >= 0.6:
        verdict = 'good'
        headline = (f'Good progress — {corrected} corrected and {improved} '
                    f'improved out of {measured} words.')
    elif progressed > 0:
        verdict = 'some'
        headline = (f'Some progress — {progressed} of {measured} words got better. '
                    f'Keep practicing the rest.')
    else:
        verdict = 'none'
        headline = ('No improvement yet on these words. Listen to the correct '
                    'pronunciation and try again.')
    if worse and measured:
        headline += f' {worse} word(s) got worse.'

    retention = []
    for w in focus_script.regression_words or []:
        key = str(w).strip().lower()
        after = by_word.get(key)
        if after is None:
            continue
        after_type = after.get('type') or 'correct'
        retention.append({
            'word': key,
            'after_type': after_type,
            'retained': _rank(after_type) == 0,
        })

    return {
        'session_id': session.id,
        'session_score': round(float(session.accuracy), 1),
        'verdict': verdict,
        'headline': headline,
        'total_words': measured,
        'corrected': corrected,
        'improved': improved,
        'unchanged': unchanged,
        'worse': worse,
        'not_practiced': not_practiced,
        'pre_error_rate': pre_rate,
        'post_error_rate': post_rate,
        'improvement_points': (
            round(pre_rate - post_rate, 1) if post_rate is not None else None
        ),
        'words': rows,
        'retention': retention,
        'completed_at': timezone.now().isoformat(),
    }