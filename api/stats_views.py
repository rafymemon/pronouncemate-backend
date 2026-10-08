# api/stats_views.py
"""
Drill-down endpoints for the Home / Dashboard stat cards and session rows.

  GET /api/stats/<metric>/?scope=week|all
        metric = sessions | accuracy | words | streak | errors
        scope  = week (current calendar week, Monday -> Sunday; default)
               = all  (everything since the user's first session)

  GET /api/sessions/<id>/
        Full detail of ONE saved practice session.

Nothing new is stored: every number is derived from data that is already
saved by AnalyzeView (PracticeSession, its `result_json`, PhonemeError,
SessionErrorLog). No migration is needed.
"""
import logging
from datetime import timedelta

from django.conf import settings
from django.db.models import Avg, Count, Max, Min, Sum
from django.utils import timezone
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import PracticeSession, PhonemeError, SessionErrorLog

logger = logging.getLogger(__name__)

VALID_METRICS = ('sessions', 'accuracy', 'words', 'streak', 'errors')

# Word `type` values coming out of the pipeline.
_OK_TYPES = ('correct', 'accent')   # accent = forgiven Pakistani-accent variant
_MINOR_TYPES = ('minor',)


# ───────────────────────── helpers ─────────────────────────

def _today():
    return timezone.localdate() if settings.USE_TZ else timezone.now().date()


def _local_date(dt):
    return timezone.localtime(dt).date() if settings.USE_TZ else dt.date()


def _week_start():
    """Monday of the current calendar week."""
    today = _today()
    return today - timedelta(days=today.weekday())


def _session_title(s):
    if s.script:
        t = s.script.title or 'Practice Session'
        return t if len(t) <= 50 else t[:47] + '...'
    return 'Custom Script'


def _session_row(s):
    return {
        'id': s.id,
        'title': _session_title(s),
        'mode': s.script.mode if s.script else 'script_based',
        'level': s.script.level if s.script else None,
        'created_at': s.created_at.isoformat(),
        'accuracy': round(float(s.accuracy), 1),
        'grade': s.grade,
        'grade_label': s.grade_label,
        'total_words': s.total_words,
        'flagged_words': s.flagged_words,
        'skipped_words': s.skipped_words,
    }


def _by_day(sessions, week_start=None):
    """
    [{date, count, avg_accuracy}] ascending. With week_start, always returns
    all 7 days of that week (empty days have count 0).
    """
    buckets = {}
    for s in sessions:
        d = _local_date(s.created_at)
        b = buckets.setdefault(d, [0, 0.0])
        b[0] += 1
        b[1] += float(s.accuracy)

    if week_start is not None:
        days = [week_start + timedelta(days=i) for i in range(7)]
    else:
        days = sorted(buckets.keys())

    out = []
    for d in days:
        count, total = buckets.get(d, [0, 0.0])
        out.append({
            'date': d.isoformat(),
            'count': count,
            'avg_accuracy': round(total / count, 1) if count else 0.0,
        })
    return out


def _streaks(day_set):
    """Returns (current_streak, longest_streak, current_streak_dates)."""
    if not day_set:
        return 0, 0, []

    today = _today()
    # Same rule as the rest of the app: the streak is alive if the user
    # practiced today OR yesterday.
    if today in day_set:
        cursor = today
    elif (today - timedelta(days=1)) in day_set:
        cursor = today - timedelta(days=1)
    else:
        cursor = None

    current_dates = []
    while cursor is not None and cursor in day_set:
        current_dates.append(cursor)
        cursor -= timedelta(days=1)

    longest = run = 0
    prev = None
    for d in sorted(day_set):
        run = run + 1 if (prev is not None and (d - prev).days == 1) else 1
        longest = max(longest, run)
        prev = d

    return len(current_dates), longest, [d.isoformat() for d in current_dates]


# ───────────────────────── stat drill-down ─────────────────────────

class StatDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, metric):
        if metric not in VALID_METRICS:
            return Response({'error': f'Unknown metric "{metric}"'}, status=400)

        scope = request.query_params.get('scope', 'week')
        if scope not in ('week', 'all'):
            scope = 'week'

        week_start = _week_start()
        base_qs = PracticeSession.objects.filter(
            user=request.user
        ).select_related('script')
        qs = (
            base_qs.filter(created_at__date__gte=week_start)
            if scope == 'week' else base_qs
        ).order_by('-created_at')

        payload = {
            'metric': metric,
            'scope': scope,
            'week_start': week_start.isoformat(),
            'week_end': (week_start + timedelta(days=6)).isoformat(),
        }
        payload.update(
            getattr(self, f'_{metric}')(request.user, qs, base_qs, scope, week_start)
        )
        return Response(payload)

    # ── sessions ─────────────────────────────────────────
    def _sessions(self, user, qs, base_qs, scope, week_start):
        total = qs.count()
        sessions = list(qs[:500])
        return {
            'total': total,
            'sessions': [_session_row(s) for s in sessions],
            'by_day': _by_day(sessions, week_start if scope == 'week' else None),
        }

    # ── accuracy ─────────────────────────────────────────
    def _accuracy(self, user, qs, base_qs, scope, week_start):
        agg = qs.aggregate(
            avg=Avg('accuracy'), best=Max('accuracy'), low=Min('accuracy')
        )
        sessions = list(qs[:500])
        return {
            'total': qs.count(),
            'average': round(float(agg['avg']), 1) if agg['avg'] is not None else 0.0,
            'best': round(float(agg['best']), 1) if agg['best'] is not None else 0.0,
            'lowest': round(float(agg['low']), 1) if agg['low'] is not None else 0.0,
            'sessions': [_session_row(s) for s in sessions],
        }

    # ── words ────────────────────────────────────────────
    def _words(self, user, qs, base_qs, scope, week_start):
        stats = {}
        # qs is newest-first, so the first time we see a word is its most
        # recent practice.
        for s in qs:
            result = s.result_json if isinstance(s.result_json, dict) else {}
            for w in result.get('words') or []:
                if not isinstance(w, dict):
                    continue
                wtype = w.get('type') or 'correct'
                if wtype == 'inserted':
                    continue  # extra spoken word, not a script word
                word = (w.get('word') or '').strip().lower()
                if not word:
                    continue
                e = stats.setdefault(word, {
                    'word': word, 'times': 0, 'correct': 0, 'minor': 0,
                    'errors': 0, 'last_practiced': s.created_at.isoformat(),
                })
                e['times'] += 1
                if wtype in _OK_TYPES:
                    e['correct'] += 1
                elif wtype in _MINOR_TYPES:
                    e['minor'] += 1
                else:
                    e['errors'] += 1

        words = []
        for e in stats.values():
            e['status'] = 'needs_work' if (e['errors'] or e['minor']) else 'clean'
            words.append(e)
        words.sort(key=lambda e: (-e['times'], e['word']))

        return {
            # Same figure as the "Words Practiced" card (sum of total_words).
            'total_words': qs.aggregate(t=Sum('total_words'))['t'] or 0,
            'unique_words': len(words),
            'words_with_errors': sum(1 for e in words if e['status'] == 'needs_work'),
            'words': words,
        }

    # ── streak ───────────────────────────────────────────
    def _streak(self, user, qs, base_qs, scope, week_start):
        # The streak itself is always computed over the user's FULL history.
        per_day = {}
        for created_at, accuracy in base_qs.values_list('created_at', 'accuracy'):
            d = _local_date(created_at)
            b = per_day.setdefault(d, [0, 0.0])
            b[0] += 1
            b[1] += float(accuracy)

        current, longest, streak_dates = _streaks(set(per_day.keys()))

        def day_row(d):
            count, total = per_day[d]
            return {
                'date': d.isoformat(),
                'session_count': count,
                'avg_accuracy': round(total / count, 1),
            }

        if scope == 'week':
            days = [d for d in per_day if d >= week_start]
        else:
            days = list(per_day.keys())
        days.sort(reverse=True)

        week_days = []
        for i in range(7):
            d = week_start + timedelta(days=i)
            week_days.append({
                'date': d.isoformat(),
                'practiced': d in per_day,
                'session_count': per_day[d][0] if d in per_day else 0,
            })

        return {
            'current_streak': current,
            'longest_streak': longest,
            'streak_days': streak_dates,
            'days_practiced': len(days),
            'practiced_days': [day_row(d) for d in days[:366]],
            'week_days': week_days,
        }

    # ── errors ───────────────────────────────────────────
    def _errors(self, user, qs, base_qs, scope, week_start):
        ep = PhonemeError.objects.filter(session__user=user)
        if scope == 'week':
            ep = ep.filter(session__created_at__date__gte=week_start)

        by_type = {
            row['error_type']: row['c']
            for row in ep.values('error_type').annotate(c=Count('id'))
        }
        words = ep.values('word').annotate(
            count=Count('id'),
            avg_severity=Avg('severity'),
            last_seen=Max('session__created_at'),
        ).order_by('-count', 'word')[:150]

        return {
            'total_errors': ep.count(),
            'by_type': by_type,
            'words': [{
                'word': w['word'],
                'count': w['count'],
                'avg_severity': round(float(w['avg_severity'] or 0), 1),
                'last_seen': w['last_seen'].isoformat() if w['last_seen'] else None,
            } for w in words],
        }


# ───────────────────────── single session ─────────────────────────

class SessionDetailView(APIView):
    """GET /api/sessions/<id>/ — everything we saved about one session."""
    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        try:
            s = PracticeSession.objects.select_related('script').get(
                id=session_id, user=request.user
            )
        except PracticeSession.DoesNotExist:
            return Response({'error': 'Session not found'}, status=404)

        result = s.result_json if isinstance(s.result_json, dict) else {}
        words = [w for w in (result.get('words') or []) if isinstance(w, dict)]
        script_words = [w for w in words if w.get('type') != 'inserted']
        inserted = [
            w.get('word') for w in words
            if w.get('type') == 'inserted' and w.get('word')
        ]

        # Script text: the Script row if there is one; custom / pasted text
        # is not stored on the session, so rebuild it from the analysis.
        if s.script:
            script_text, script_source = s.script.content, 'script'
        else:
            script_text = ' '.join(
                str(w.get('word')) for w in script_words if w.get('word')
            )
            script_source = 'reconstructed' if script_text else 'unavailable'

        saved_stats = result.get('stats') if isinstance(result.get('stats'), dict) else {}
        correct_words = saved_stats.get('correct_words')
        if correct_words is None and script_words:
            correct_words = sum(1 for w in script_words if w.get('type') in _OK_TYPES)
        wrong_words = saved_stats.get('wrong_words')

        errors = [{
            'word': e.word,
            'error_type': e.error_type,
            'severity': e.severity,
            'expected': e.expected,
            'spoken': e.spoken,
        } for e in s.errors.all()]

        log = SessionErrorLog.objects.filter(session=s).first()

        feedback = [f for f in (result.get('feedback') or []) if isinstance(f, dict)]

        return Response({
            'id': s.id,
            'created_at': s.created_at.isoformat(),
            'title': _session_title(s),
            'mode': s.script.mode if s.script else 'script_based',
            'level': s.script.level if s.script else None,
            'level_at_time': log.level_at_time if log and log.level_at_time else None,
            'script_id': s.script_id,
            'script_text': script_text,
            'script_source': script_source,
            'accuracy': round(float(s.accuracy), 1),
            'grade': s.grade,
            'grade_label': s.grade_label,
            'stats': {
                'total_words': saved_stats.get('total_words', s.total_words),
                'correct_words': correct_words,
                'wrong_words': wrong_words,
                'flagged_words': s.flagged_words,
                'skipped_words': s.skipped_words,
                'word_accuracy': saved_stats.get('word_accuracy'),
                'severity_total': s.severity_total,
            },
            'transcript': s.transcript,
            'words': words,
            'inserted_words': inserted,
            'errors': errors,
            'feedback': feedback,
            'mistake_summary': result.get('mistake_summary'),
            'recording_url': result.get('recording_url'),
        })