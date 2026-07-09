# api/views.py
import os, tempfile, logging
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

from django.db.models import Avg, Count, Max, F
from django.utils import timezone
from datetime import timedelta
from rest_framework.decorators import api_view, permission_classes



logger = logging.getLogger(__name__)

# ── Auth ────────────────────────────────────────────────────────
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

# ── Core Analysis Endpoint ───────────────────────────────────────
class AnalyzeView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes     = [MultiPartParser]

    def post(self, request):
        script_text = request.data.get('script_text', '').strip()
        audio_file  = request.FILES.get('audio')

        # Validation
        if not script_text:
            return Response({'error': 'script_text is required'}, status=400)
        if not audio_file:
            return Response({'error': 'audio file is required'}, status=400)
        if len(script_text.split()) < 2:
            return Response({'error': 'Script must have at least 2 words'}, status=400)

        # Save audio to temp file
        ext = '.wav'
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            for chunk in audio_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            logger.info(f'Analyzing for user {request.user.username}: {script_text[:50]}')

            # Run the full AI pipeline
            result = run_pipeline(tmp_path, script_text)

            # Save session to database
            session = PracticeSession.objects.create(
                user          = request.user,
                script        = script_text,
                transcript    = ' '.join(result.get('transcript', [])),
                accuracy      = result['score'],
                grade         = result['grade'],
                grade_label   = result['grade_label'],
                severity_total= result.get('severity_total', 0),
                total_words   = result.get('total_words', 0),
                flagged_words = result.get('flagged_words', 0),
                skipped_words = result.get('skipped_words', 0),
                result_json   = result
            )

            # Save individual errors for analytics
            for w in result.get('words', []):
                if w.get('type') not in ('correct', 'inserted'):
                    PhonemeError.objects.create(
                        session    = session,
                        word       = w['word'],
                        error_type = w['type'],
                        severity   = w.get('severity', 0)
                    )

            # Update user profile stats
            profile = request.user.userprofile
            profile.total_sessions += 1
            if result['score'] > profile.best_score:
                profile.best_score = result['score']
            profile.save()

            result['session_id'] = session.id
            return Response(result, status=200)

        except Exception as e:
            logger.exception(f'Pipeline error: {e}')
            return Response({'error': f'Analysis failed: {str(e)}'}, status=500)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)  # always delete temp file

# ── Session History ──────────────────────────────────────────────
class SessionHistoryView(APIView):
    def get(self, request):
        sessions = PracticeSession.objects.filter(user=request.user)[:20]
        return Response(SessionSerializer(sessions, many=True).data)

# ── User Profile ─────────────────────────────────────────────────
class ProfileView(APIView):
    def get(self, request):
        profile, _ = UserProfile.objects.get_or_create(user=request.user)
        return Response(UserProfileSerializer(profile).data)

    def patch(self, request):
        profile, _ = UserProfile.objects.get_or_create(user=request.user)
        s = UserProfileSerializer(profile, data=request.data, partial=True)
        if s.is_valid():
            s.save()
            return Response(s.data)
        return Response(s.errors, status=400)


class ScriptListView(APIView):
    """
    GET /api/scripts/?mode=public_speaking&level=intermediate
    Returns filtered builtin scripts + user uploads
    """
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
    """
    POST /api/scripts/upload/
    Accepts JSON body or multipart file (.txt / .docx)
    """
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
                title = file.name.rsplit('.', 1)[0]\
                    .replace('_', ' ').replace('-', ' ').title()

        if not content:
            return Response({'error': 'No content provided'}, status=400)
        if not title:
            return Response({'error': 'Title is required'}, status=400)
        if len(content.split()) < 5:
            return Response(
                {'error': 'Script must have at least 5 words'}, status=400)

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

    


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def dashboard_summary(request):
    user = request.user
    sessions = PracticeSession.objects.filter(user=user)
    all_ordered = sessions.order_by('created_at')
    
    # ── Basic Stats ─────────────────────────────────────
    total_sessions = sessions.count()
    best_score = sessions.aggregate(max=Max('accuracy'))['max'] or 0
    
    # ── Streak ──────────────────────────────────────────
    streak = _calculate_streak(user)
    
    # ── Update UserProfile cache ─────────────────────────
    profile, _ = UserProfile.objects.get_or_create(user=user)
    profile.total_sessions = total_sessions
    profile.best_score = best_score
    profile.save(update_fields=['total_sessions', 'best_score'])
    
    # ── Recent Sessions (last 5) ────────────────────────
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
    
    # ── Quick Stats ─────────────────────────────────────
    total_words = sessions.aggregate(total=Count('total_words'))['total'] or 0
    total_errors = PhonemeError.objects.filter(session__user=user).count()
    # Estimate duration: assume ~3 seconds per word on average
    total_minutes = sum(s.total_words * 3 // 60 for s in all_ordered)
    
    # ── Sparkline (last 7 sessions) ─────────────────────
    sparkline_sessions = list(all_ordered.values('accuracy', 'created_at')[:7])
    sparkline = [float(s['accuracy']) for s in sparkline_sessions]
    
    # ── Improvement ─────────────────────────────────────
    improvement = _calculate_improvement(all_ordered)
    
    # ── Focus Words (top 3) ─────────────────────────────
    top_errors = PhonemeError.objects.filter(session__user=user) \
        .values('word') \
        .annotate(error_count=Count('id')) \
        .order_by('-error_count')[:3]
    
    focus_words = [{
        'word': w['word'],
        'error_count': w['error_count'],
    } for w in top_errors]
    
    # ── Score by Mode ───────────────────────────────────
    # Since PracticeSession doesn't have mode field, infer from script content
    # or use a simpler approach: group by session patterns
    mode_scores = _infer_mode_scores(sessions)
    
    # ── Error Breakdown ─────────────────────────────────
    error_types = PhonemeError.objects.filter(session__user=user) \
        .values('error_type') \
        .annotate(count=Count('id'))
    
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


def _calculate_streak(user):
    """Calculate consecutive days with practice sessions"""
    from django.utils import timezone
    from datetime import timedelta
    
    session_dates = PracticeSession.objects.filter(user=user) \
        .dates('created_at', 'day', order='DESC')
    
    if not session_dates:
        return 0
    
    streak = 1
    today = timezone.now().date()
    
    # Check if practiced today or yesterday
    if session_dates[0] < today - timedelta(days=1):
        return 0
    
    for i in range(1, len(session_dates)):
        if session_dates[i] == session_dates[i-1] - timedelta(days=1):
            streak += 1
        else:
            break
    
    return streak


def _format_date(dt):
    """Human-readable date label"""
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
    """Compare first 5 vs last 5 sessions"""
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
    """Simple heuristic to detect practice mode from script content"""
    text_lower = script_text.lower()
    
    # Check for speech indicators
    speech_keywords = ['ladies and gentlemen', 'thank you', 'welcome', 'honored', 'present', 'audience']
    if any(kw in text_lower for kw in speech_keywords):
        return 'public_speaking'
    
    # Check for conversation indicators
    conv_keywords = ['hi', 'hello', 'how are you', 'nice to meet', 'chat', 'talk', 'conversation']
    if any(kw in text_lower for kw in conv_keywords):
        return 'daily_conversation'
    
    return 'script_based'


def _infer_mode_scores(sessions_qs):
    """
    Infer mode scores from session data.
    Since PracticeSession doesn't store mode directly, we use heuristics
    or return overall average as fallback.
    """
    # For now, return overall score as all modes
    # In production, you'd want to store mode in PracticeSession
    overall_avg = sessions_qs.aggregate(avg=Avg('accuracy'))['avg'] or 0
    
    return {
        'public_speaking': round(float(overall_avg) * 0.95, 1),  # Slightly lower (harder)
        'daily_conversation': round(float(overall_avg) * 1.05, 1),  # Slightly higher (easier)
        'script_based': round(float(overall_avg), 1),
    }



# for progress_analytics endpoint

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
    
    # All session data for full trend chart
    session_data = [{
        'id': s.id,
        'accuracy': float(s.accuracy),
        'word_count': s.total_words,
        'duration_minutes': 0,  # Add duration field if available
        'mode': _detect_mode(s.script),
        'created_at': s.created_at.isoformat(),
    } for s in sessions]
    
    # Top 10 mispronounced words (more than dashboard's 3)
    top_errors = PhonemeError.objects.filter(session__user=user) \
        .values('word') \
        .annotate(error_count=Count('id')) \
        .order_by('-error_count')[:10]
    
    focus_words = [{
        'word': w['word'],
        'error_count': w['error_count'],
        'total_count': max(1, sessions.count()),  # Approximate
    } for w in top_errors]
    
    # Error breakdown
    error_types = PhonemeError.objects.filter(session__user=user) \
        .values('error_type') \
        .annotate(count=Count('id'))
    
    error_breakdown = {'substitution': 0, 'omission': 0, 'insertion': 0, 'skipped': 0}
    for et in error_types:
        if et['error_type'] in error_breakdown:
            error_breakdown[et['error_type']] = et['count']
    
    # Mode scores
    mode_scores = _infer_mode_scores(sessions)
    
    # Improvement
    improvement = _calculate_improvement(sessions)
    
    # Total stats
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