# api/views.py
import os, tempfile, logging, json, uuid
from rest_framework.views      import APIView
from rest_framework.response   import Response
from rest_framework.parsers    import MultiPartParser, JSONParser
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from django.conf import settings
from .models      import PracticeSession, PhonemeError, UserProfile, Script
from .serializers import RegisterSerializer, SessionSerializer, UserProfileSerializer, ScriptSerializer
from Pipeline.pipeline import run_pipeline
from django.contrib.auth.models import User

from django.db.models import Avg, Count, Max, F
from django.utils import timezone
from datetime import timedelta
from rest_framework.decorators import api_view, permission_classes


# NEW IMPORT — add this line
from Pipeline.feedback import get_word_feedback_audio_url

# Practice Mistakes imports
from Pipeline.practice_mistakes import (
    extract_mistake_words,
    generate_practice_script,
    run_practice_session,
    should_offer_practice,
    get_mistake_summary,
    MistakeWord,
    MIN_WORDS_FOR_PRACTICE,
    PRACTICE_PASS_THRESHOLD,
)



logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# AUTH VIEWS
# ═══════════════════════════════════════════════════════════════

class RegisterView(APIView):
    permission_classes = [AllowAny]
    def post(self, request):
        s = RegisterSerializer(data=request.data)
        if s.is_valid():
            user  = s.save()
            token, _ = Token.objects.get_or_create(user=user)
            return Response({'token': token.key, 'username': user.username}, status=201)
        return Response(s.errors, status=400)

class LoginView(APIView):
    permission_classes = [AllowAny]
    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')
        user = authenticate(username=username, password=password)
        if user:
            token, _ = Token.objects.get_or_create(user=user)
            return Response({'token': token.key, 'username': user.username})
        return Response({'error': 'Invalid credentials'}, status=401)

# ═══════════════════════════════════════════════════════════════
# CORE ANALYSIS ENDPOINT (Enhanced with Practice Mistakes + Audio)
# ═══════════════════════════════════════════════════════════════

class AnalyzeView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes     = [MultiPartParser]

    def post(self, request):
        script_text = request.data.get('script_text', '').strip()
        audio_file  = request.FILES.get('audio')

        if not script_text:
            return Response({'error': 'script_text is required'}, status=400)
        if not audio_file:
            return Response({'error': 'audio file is required'}, status=400)
        if len(script_text.split()) < 2:
            return Response({'error': 'Script must have at least 2 words'}, status=400)

        ext = '.wav'
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            for chunk in audio_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            logger.info(f'Analyzing for user {request.user.username}: {script_text[:50]}')

            result = run_pipeline(tmp_path, script_text)

            if not result.get('success', False):
                os.unlink(tmp_path)  # ← CHANGED: delete temp file on failure only
                return Response(result, status=400)

            # ── Save recording for playback ──────────────────────────
            # ← ADDED: Move temp WAV to permanent media storage
            recordings_dir = os.path.join(settings.MEDIA_ROOT, 'recordings')
            os.makedirs(recordings_dir, exist_ok=True)
            
            recording_filename = f'rec_{request.user.id}_{uuid.uuid4().hex[:12]}.wav'
            recording_path = os.path.join(recordings_dir, recording_filename)
            os.rename(tmp_path, recording_path)  # move from temp to media
            
            recording_url = f'/media/recordings/{recording_filename}'
            result['recording_url'] = recording_url  # ← ADDED: Flutter receives this

            # ── Generate TTS audio for every error word ─────────────
            for word_data in result.get('words', []):
                if word_data.get('type') not in ('correct', 'inserted'):
                    w = word_data.get('word', '')
                    if w:
                        word_data['audio_path'] = get_word_feedback_audio_url(w)

            # Build mistake summary for Practice Mistakes button
            errors = result.get('words', [])
            mistake_summary = get_mistake_summary(errors)
            result['mistake_summary'] = mistake_summary

            session = PracticeSession.objects.create(
                user          = request.user,
                script        = script_text,
                transcript    = ' '.join(result.get('transcript', [])),
                accuracy      = result['score'],
                grade         = result['grade'],
                grade_label   = result['grade_label'],
                severity_total= result.get('stats', {}).get('severity_total', 0),
                total_words   = result.get('stats', {}).get('total_words', 0),
                flagged_words = result.get('stats', {}).get('flagged_words', 0),
                skipped_words = result.get('stats', {}).get('skipped_words', 0),
                result_json   = result
            )

            for w in result.get('words', []):
                if w.get('type') not in ('correct', 'inserted'):
                    PhonemeError.objects.create(
                        session    = session,
                        word       = w['word'],
                        error_type = w['type'],
                        severity   = w.get('severity', 0)
                    )

            profile = request.user.userprofile
            profile.total_sessions += 1
            if result['score'] > profile.best_score:
                profile.best_score = result['score']
            profile.save()

            result['session_id'] = session.id
            return Response(result, status=200)

        except Exception as e:
            logger.exception(f'Pipeline error: {e}')
            # ← CHANGED: Clean up temp file on error only (success = moved to media)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return Response({'error': f'Analysis failed: {str(e)}'}, status=500)

# PRACTICE MISTAKES ENDPOINTS (DRF APIView)

class PracticeScriptView(APIView):
    """
    POST /api/practice/script/
    Get generated practice script from mistake words (no audio needed).
    """
    permission_classes = [IsAuthenticated]
    parser_classes = [JSONParser]

    def post(self, request):
        mistake_data = request.data.get('mistake_data', [])
        mode = request.data.get('mode', 'sentences')

        if not mistake_data:
            return Response(
                {'success': False, 'error': 'No mistake words provided'},
                status=400
            )

        mistake_words = []
        for md in mistake_data:
            mw = MistakeWord(
                word=md.get('word', ''),
                error_type=md.get('error_type', ''),
                severity=md.get('severity', 0),
                per_word_score=md.get('per_word_score', 0.0),
                ref_phonemes=md.get('ref_phonemes', []),
                pred_phonemes=md.get('pred_phonemes', []),
                explanation=md.get('explanation', ''),
                audio_path=md.get('audio_path', None)
            )
            mistake_words.append(mw)

        practice_script = generate_practice_script(mistake_words, mode=mode)

        return Response({
            'success': True,
            'practice_script': practice_script,
            'word_count': len(mistake_words),
            'mode': mode,
            'words': [mw.word for mw in mistake_words]
        })


class PracticeMistakesView(APIView):
    """
    POST /api/practice/
    Submit audio for focused practice on mistake words.
    """
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser]

    def post(self, request):
        audio_file = request.FILES.get('audio')
        original_session_id = request.data.get('original_session_id', '')
        mode = request.data.get('mode', 'sentences')
        mistake_data = request.data.get('mistake_data', [])

        if not audio_file:
            return Response(
                {'success': False, 'error': 'No audio file provided'},
                status=400
            )

        if not mistake_data:
            return Response(
                {'success': False, 'error': 'No mistake words provided'},
                status=400
            )

        mistake_words = []
        for md in mistake_data:
            mw = MistakeWord(
                word=md.get('word', ''),
                error_type=md.get('error_type', 'unknown'),
                severity=md.get('severity', 0),
                per_word_score=md.get('per_word_score', 0.0),
                ref_phonemes=md.get('ref_phonemes', []),
                pred_phonemes=md.get('pred_phonemes', []),
                explanation=md.get('explanation', ''),
                audio_path=md.get('audio_path', None)
            )
            mistake_words.append(mw)

        ext = '.wav'
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            for chunk in audio_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            logger.info(
                f'Practice session for user {request.user.username}: '
                f'{len(mistake_words)} words, mode={mode}'
            )

            result = run_practice_session(
                raw_audio_path=tmp_path,
                mistake_words=mistake_words,
                original_session_id=original_session_id,
                mode=mode
            )

            result['practice_session_id'] = uuid.uuid4().hex[:12]

            return Response(result, status=200)

        except Exception as e:
            logger.exception(f'Practice session error: {e}')
            return Response(
                {'success': False, 'error': f'Practice failed: {str(e)}'},
                status=500
            )

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class PracticeProgressView(APIView):
    """
    GET /api/practice/progress/
    Get practice progress for a specific word or all words.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        word = request.query_params.get('word')
        user = request.user

        errors_qs = PhonemeError.objects.filter(session__user=user)
        if word:
            errors_qs = errors_qs.filter(word=word)

        error_stats = errors_qs.values('word').annotate(
            total_errors=Count('id'),
            avg_severity=Avg('severity'),
            last_occurrence=Max('session__created_at')
        ).order_by('-total_errors')

        return Response({
            'success': True,
            'user_id': user.id,
            'word': word,
            'stats': list(error_stats),
            'message': 'Practice progress retrieved'
        })

# ═══════════════════════════════════════════════════════════════
# SESSION HISTORY & PROFILE
# ═══════════════════════════════════════════════════════════════

class SessionHistoryView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request):
        sessions = PracticeSession.objects.filter(user=request.user)[:20]
        return Response(SessionSerializer(sessions, many=True).data)

# api/views.py — replace or add ProfileView



class ProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user    = request.user
        profile = self._get_or_create_profile(user)

        # ── Aggregate session stats ──────────────────────
        sessions = PracticeSession.objects.filter(user=user)
        total_sessions   = sessions.count()
        scores           = [s.accuracy for s in sessions if s.accuracy is not None]
        avg_accuracy     = round(sum(scores) / len(scores), 1) if scores else 0.0
        best_score       = round(max(scores), 1) if scores else 0.0
        total_words      = PhonemeError.objects.filter(
            session__user=user).count()
        correct_words    = PhonemeError.objects.filter(
            session__user=user, error_type='correct').count()

        # ── Auto-derive proficiency from performance ─────
        computed_level = self._compute_level(avg_accuracy, total_sessions)

        # ── Streak calculation ───────────────────────────
        streak = self._compute_streak(user)

        # ── Most practiced mode ──────────────────────────
        from django.db.models import Count
        mode_counts = sessions.values('script__mode').annotate(
            count=Count('id')).order_by('-count')
        top_mode = mode_counts.first()
        favorite_mode = top_mode['script__mode'] if top_mode else 'None yet'

        # ── Most mispronounced words (top 5) ─────────────
        from django.db.models import Count
        top_errors = PhonemeError.objects.filter(
            session__user=user
        ).exclude(
            error_type='correct'
        ).values('expected').annotate(
            count=Count('id')
        ).order_by('-count')[:5]

        return Response({
            # Identity
            'username':    user.username,
            'email':       user.email,
            'first_name':  user.first_name,
            'member_since': user.date_joined.strftime('%B %Y'),
            'member_since_full': user.date_joined.isoformat(),

            # Level
            'proficiency_level':    profile.proficiency_level,
            'computed_level':       computed_level,
            'level_reason':         self._level_reason(avg_accuracy, total_sessions),

            # Stats
            'total_sessions':  total_sessions,
            'avg_accuracy':    avg_accuracy,
            'best_score':      best_score,
            'total_words':     total_words,
            'correct_words':   correct_words,
            'streak_days':     streak,
            'favorite_mode':   favorite_mode.replace('_', ' ').title(),

            # Top errors
            'top_errors': [
                {'word': e['expected_word'], 'count': e['count']}
                for e in top_errors
            ],
        })

    def patch(self, request):
        """Update profile fields."""
        user    = request.user
        profile = self._get_or_create_profile(user)
        data    = request.data

        if 'first_name' in data:
            user.first_name = data['first_name'].strip()
            user.save(update_fields=['first_name'])

        if 'proficiency_level' in data:
            valid = ['beginner','intermediate','upper_intermediate',
                     'advanced','fluent']
            level = data['proficiency_level']
            if level not in valid:
                return Response({'error': 'Invalid level'}, status=400)
            profile.proficiency_level = level
            profile.save(update_fields=['proficiency_level'])

        if 'email' in data:
            email = data['email'].strip()
            if User.objects.exclude(id=user.id).filter(email=email).exists():
                return Response({'error': 'Email already in use'}, status=400)
            user.email = email
            user.save(update_fields=['email'])

        return Response({'message': 'Profile updated successfully'})

    def delete(self, request):
        """Clear session history."""
        PracticeSession.objects.filter(user=request.user).delete()
        return Response({'message': 'Session history cleared'})

    # ── Helpers ───────────────────────────────────────────

    def _get_or_create_profile(self, user):
        profile, _ = UserProfile.objects.get_or_create(
            user=user,
            defaults={'proficiency_level': 'beginner'}
        )
        return profile

    def _compute_level(self, avg_accuracy: float, total_sessions: int) -> str:
        """
        Automatically derive level from actual performance.
        Requires at least 3 sessions to assess reliably.
        """
        if total_sessions < 3:
            return 'beginner'
        if avg_accuracy >= 88:
            return 'fluent'
        if avg_accuracy >= 78:
            return 'advanced'
        if avg_accuracy >= 66:
            return 'upper_intermediate'
        if avg_accuracy >= 52:
            return 'intermediate'
        return 'beginner'

    def _level_reason(self, avg: float, sessions: int) -> str:
        if sessions < 3:
            return f'Complete at least 3 sessions to unlock your computed level. ({sessions}/3 done)'
        return f'Based on your average accuracy of {avg}% across {sessions} sessions.'

    def _compute_streak(self, user) -> int:
        from django.utils import timezone
        from datetime import timedelta

        sessions = PracticeSession.objects.filter(
            user=user
        ).order_by('-created_at').values_list('created_at', flat=True)

        if not sessions:
            return 0

        streak   = 0
        today    = timezone.now().date()
        last_day = today

        seen_days = sorted(
            set(s.date() for s in sessions), reverse=True)

        for day in seen_days:
            if (last_day - day).days <= 1:
                streak  += 1
                last_day = day
            else:
                break

        return streak



class ScriptListView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        mode  = request.query_params.get('mode')
        level = request.query_params.get('level')

        builtin_qs = Script.objects.filter(is_builtin=True)
        if mode:
            builtin_qs = builtin_qs.filter(mode=mode)
        if level:
            builtin_qs = builtin_qs.filter(level=level)

        uploaded_qs = Script.objects.none()
        if request.user and request.user.is_authenticated:
            uploaded_qs = Script.objects.filter(
                is_builtin=False,
                uploaded_by=request.user
            )
            if mode and mode != 'script_based':
                uploaded_qs = uploaded_qs.filter(mode=mode)

        return Response({
            'builtin':  ScriptSerializer(builtin_qs,  many=True).data,
            'uploaded': ScriptSerializer(uploaded_qs, many=True).data,
        })


class ScriptUploadView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes     = [MultiPartParser, JSONParser]

    def post(self, request):
        title   = request.data.get('title', '').strip()
        content = request.data.get('content', '').strip()
        level   = request.data.get('level', 'intermediate')
        mode    = request.data.get('mode', 'script_based')
        file    = request.FILES.get('file')

        if file:
            filename = file.name.lower()
            if filename.endswith('.txt'):
                content = file.read().decode('utf-8').strip()
            elif filename.endswith('.docx'):
                content = self._read_docx(file)
            else:
                return Response(
                    {'error': 'Only .txt and .docx files are supported'},
                    status=400
                )
            if not title:
                title = file.name.rsplit('.', 1)[0]
                title = title.replace('_', ' ').replace('-', ' ').title()

        if not content:
            return Response({'error': 'No content provided'}, status=400)
        if not title:
            return Response({'error': 'Title is required'}, status=400)
        if len(content.split()) < 5:
            return Response(
                {'error': 'Script must have at least 5 words'}, status=400
            )

        script = Script.objects.create(
            title=title, level=level, mode=mode,
            content=content, is_builtin=False,
            uploaded_by=request.user,
            description=f'Uploaded by {request.user.username}',
        )
        return Response(ScriptSerializer(script).data, status=201)

    def _read_docx(self, file) -> str:
        import docx, tempfile, os
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        doc  = docx.Document(tmp_path)
        text = ' '.join(p.text for p in doc.paragraphs if p.text.strip())
        os.unlink(tmp_path)
        return text.strip()

# ═══════════════════════════════════════════════════════════════
# DASHBOARD & ANALYTICS
# ═══════════════════════════════════════════════════════════════

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def dashboard_summary(request):
    user = request.user
    sessions = PracticeSession.objects.filter(user=user)
    all_ordered = sessions.order_by('created_at')
    
    total_sessions = sessions.count()
    best_score = sessions.aggregate(max=Max('accuracy'))['max'] or 0
    streak = _calculate_streak(user)
    
    profile, _ = UserProfile.objects.get_or_create(user=user)
    profile.total_sessions = total_sessions
    profile.best_score = best_score
    profile.save(update_fields=['total_sessions', 'best_score'])
    
    recent = sessions.order_by('-created_at')[:5]
    recent_sessions = [{
        'id': s.id,
        'title': s.script[:50] + '...' if len(s.script) > 50 else s.script,
        'score': float(s.accuracy),
        'grade': s.grade,
        'date_label': _format_date(s.created_at),
        'mode': _detect_mode(s.script),
        'total_words': s.total_words,
        'flagged_words': s.flagged_words,
    } for s in recent]
    
    total_words = sessions.aggregate(total=Count('total_words'))['total'] or 0
    total_errors = PhonemeError.objects.filter(session__user=user).count()
    total_minutes = sum(s.total_words * 3 // 60 for s in all_ordered)
    
    sparkline_sessions = list(all_ordered.values('accuracy', 'created_at')[:7])
    sparkline = [float(s['accuracy']) for s in sparkline_sessions]
    
    improvement = _calculate_improvement(all_ordered)
    
    top_errors = PhonemeError.objects.filter(session__user=user).values('word').annotate(error_count=Count('id')).order_by('-error_count')[:3]
    
    focus_words = [{
        'word': w['word'],
        'error_count': w['error_count'],
    } for w in top_errors]
    
    mode_scores = _infer_mode_scores(sessions)
    
    error_types = PhonemeError.objects.filter(session__user=user).values('error_type').annotate(count=Count('id'))
    
    error_breakdown = {'substitution': 0, 'omission': 0, 'insertion': 0, 'skipped': 0}
    for et in error_types:
        if et['error_type'] in error_breakdown:
            error_breakdown[et['error_type']] = et['count']
    
    return Response({
        'user': {
            'name': user.first_name or user.username,
            'email': user.email,
        },
        'streak_days': streak,
        'total_sessions': total_sessions,
        'best_score': round(float(best_score), 1),
        'recent_sessions': recent_sessions,
        'quick_stats': {
            'words': total_words,
            'errors_corrected': total_errors,
            'minutes': total_minutes,
        },
        'sparkline': sparkline,
        'improvement': improvement,
        'focus_words': focus_words,
        'mode_scores': mode_scores,
        'error_breakdown': error_breakdown,
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def progress_analytics(request):
    user = request.user
    sessions = PracticeSession.objects.filter(user=user).order_by('created_at')
    
    if sessions.count() < 1:
        return Response({
            'sessions': [],
            'top_errors': [],
            'error_breakdown': {'substitution': 0, 'omission': 0, 'stress': 0, 'insertion': 0},
            'mode_scores': {},
            'improvement': {'first_avg': 0, 'last_avg': 0, 'percentage_change': 0},
            'total_stats': {'sessions': 0, 'words': 0, 'errors': 0, 'minutes': 0},
        })
    
    session_data = [{
        'id': s.id,
        'accuracy': float(s.accuracy),
        'word_count': s.total_words,
        'duration_minutes': 0,
        'mode': _detect_mode(s.script),
        'created_at': s.created_at.isoformat(),
    } for s in sessions]
    
    top_errors = PhonemeError.objects.filter(session__user=user).values('word').annotate(error_count=Count('id')).order_by('-error_count')[:10]
    
    focus_words = [{
        'word': w['word'],
        'error_count': w['error_count'],
        'total_count': max(1, sessions.count()),
    } for w in top_errors]
    
    error_types = PhonemeError.objects.filter(session__user=user).values('error_type').annotate(count=Count('id'))
    
    error_breakdown = {'substitution': 0, 'omission': 0, 'insertion': 0, 'skipped': 0}
    for et in error_types:
        if et['error_type'] in error_breakdown:
            error_breakdown[et['error_type']] = et['count']
    
    mode_scores = _infer_mode_scores(sessions)
    improvement = _calculate_improvement(sessions)
    
    total_words = sessions.aggregate(total=Count('total_words'))['total'] or 0
    total_errors = PhonemeError.objects.filter(session__user=user).count()
    total_minutes = sum(s.total_words * 3 // 60 for s in sessions)
    
    return Response({
        'sessions': session_data,
        'top_errors': focus_words,
        'error_breakdown': error_breakdown,
        'mode_scores': mode_scores,
        'improvement': improvement,
        'total_stats': {
            'sessions': sessions.count(),
            'words': total_words,
            'errors': total_errors,
            'minutes': total_minutes,
        }
    })


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _calculate_streak(user):
    session_dates = PracticeSession.objects.filter(user=user).dates('created_at', 'day', order='DESC')
    
    if not session_dates:
        return 0
    
    streak = 1
    today = timezone.now().date()
    
    if session_dates[0] < today - timedelta(days=1):
        return 0
    
    for i in range(1, len(session_dates)):
        if session_dates[i] == session_dates[i-1] - timedelta(days=1):
            streak += 1
        else:
            break
    
    return streak


def _format_date(dt):
    now = timezone.now()
    diff = now - dt
    
    if diff.days == 0:
        if diff.seconds < 3600:
            return f'{diff.seconds // 60}m ago'
        return f'{diff.seconds // 3600}h ago'
    elif diff.days == 1:
        return 'Yesterday'
    elif diff.days < 7:
        return dt.strftime('%A')
    return dt.strftime('%b %d')


def _calculate_improvement(sessions_qs):
    sessions_list = list(sessions_qs)
    if len(sessions_list) < 2:
        return {
            'first_avg': 0,
            'last_avg': 0,
            'percentage_change': 0,
            'trend': 'neutral'
        }
    
    first_five = sessions_list[:min(5, len(sessions_list))]
    last_five = sessions_list[max(0, len(sessions_list)-5):]
    
    first_avg = sum(s.accuracy for s in first_five) / len(first_five)
    last_avg = sum(s.accuracy for s in last_five) / len(last_five)
    pct_change = ((last_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
    
    trend = 'up' if pct_change > 5 else ('down' if pct_change < -5 else 'neutral')
    
    return {
        'first_avg': round(float(first_avg), 1),
        'last_avg': round(float(last_avg), 1),
        'percentage_change': round(float(pct_change), 1),
        'trend': trend,
    }


def _detect_mode(script_text):
    text_lower = script_text.lower()
    
    speech_keywords = ['ladies and gentlemen', 'thank you', 'welcome', 'honored', 'present', 'audience']
    if any(kw in text_lower for kw in speech_keywords):
        return 'public_speaking'
    
    conv_keywords = ['hi', 'hello', 'how are you', 'nice to meet', 'chat', 'talk', 'conversation']
    if any(kw in text_lower for kw in conv_keywords):
        return 'daily_conversation'
    
    return 'script_based'


def _infer_mode_scores(sessions_qs):
    overall_avg = sessions_qs.aggregate(avg=Avg('accuracy'))['avg'] or 0
    
    return {
        'public_speaking': round(float(overall_avg) * 0.95, 1),
        'daily_conversation': round(float(overall_avg) * 1.05, 1),
        'script_based': round(float(overall_avg), 1),
    }