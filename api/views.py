# api/views.py
import os, tempfile, logging, json, uuid
from rest_framework.views      import APIView
from rest_framework.response   import Response
from rest_framework.parsers    import MultiPartParser, JSONParser
from rest_framework import status
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from django.conf import settings
from .models import PracticeSession, PhonemeError, UserProfile, Script, LEVEL_CHOICES, Notification
from .serializers import (
    RegisterSerializer, SessionSerializer, UserProfileSerializer,
    ScriptSerializer, NotificationSerializer,
)
from Pipeline.pipeline import run_pipeline
from django.contrib.auth.models import User

from django.db.models import Avg, Count, Max, F, Sum, Value
from django.db.models.functions import Coalesce
from django.utils import timezone
from datetime import timedelta
from rest_framework.decorators import api_view, permission_classes

from .rotation import get_rotated_scripts
from . import gemini_service
from . import notifications as notif
from .serializers import RotatedScriptSerializer
from .progression import (
    log_session_errors, update_word_mastery,
    evaluate_progression, promote_if_eligible,
    check_and_maybe_generate_focus_script,
)

from .models import WordMastery, LevelProgressionStatus, FocusScript
from .serializers import (
    WordMasterySerializer, LevelProgressionStatusSerializer, FocusScriptSerializer,
)
from .progression import evaluate_progression, promote_if_eligible
from django.db.models.functions import TruncDate


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
        print("🚨 FLUTTER SENT THIS DATA:", request.data)
        script_text = request.data.get('script_text', '').strip()
        script_id   = request.data.get('script_id')  # ← ADDED: Get script_id from Flutter
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
                os.unlink(tmp_path)
                return Response(result, status=400)

            # ── Save recording for playback ──────────────────────────
            recordings_dir = os.path.join(settings.MEDIA_ROOT, 'recordings')
            os.makedirs(recordings_dir, exist_ok=True)
            
            recording_filename = f'rec_{request.user.id}_{uuid.uuid4().hex[:12]}.wav'
            recording_path = os.path.join(recordings_dir, recording_filename)
            os.rename(tmp_path, recording_path)
            
            recording_url = f'/media/recordings/{recording_filename}'
            result['recording_url'] = recording_url

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

            # ── FIX: Handle script_id properly ──────────────────────
            script = None
            if script_id:
                try:
                    script = Script.objects.get(id=int(script_id))
                except (Script.DoesNotExist, ValueError):
                    logger.warning(f'Script {script_id} not found, saving without script')
                    script = None

            session = PracticeSession.objects.create(
                user          = request.user,
                script        = script,  # ← FIXED: Can be null now
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

            # ── Improvement Phase hooks ──────────────────────────────
            focus_script_obj = None
            try:
                word_results = result.get('words', [])
                log_session_errors(session, request.user, word_results)
                update_word_mastery(request.user, word_results)
                evaluate_progression(request.user)  # flags eligibility only — user confirms the actual level-up
                focus_script_obj = check_and_maybe_generate_focus_script(request.user)
            except Exception:
                logger.exception('Improvement Phase hooks failed (non-fatal)')

            # ── Notifications: streak milestones + new Focus Script ──
            # Own try block so a notification failure can never break the
            # analysis response.
            try:
                notif.check_streak_notifications(request.user)
                notif.notify_focus_script_ready(request.user, focus_script_obj)
            except Exception:
                logger.exception('Notification generation failed (non-fatal)')

            # ── Attach the Focus Script the Dashboard shows ──────────
            # Same query as LatestFocusScriptView (most recent uncompleted),
            # run AFTER the hooks above so a script generated by this very
            # session is included. The Flutter "Practice Mistakes" button
            # opens exactly this script, so the Session Summary and the
            # Dashboard card can never disagree. Kept in its own try block
            # so a failure here can never break the analysis response.
            try:
                latest_focus = FocusScript.objects.filter(
                    user=request.user, completed=False
                ).order_by('-generated_at').first()
                result['focus_script'] = (
                    FocusScriptSerializer(latest_focus).data if latest_focus else None
                )
            except Exception:
                logger.exception('Attaching focus script failed (non-fatal)')
                result['focus_script'] = None

            result['session_id'] = session.id
            return Response(result, status=200)

        except Exception as e:
            logger.exception(f'Pipeline error: {e}')
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

        mistake_data_raw = request.data.get('mistake_data', '[]')
        if isinstance(mistake_data_raw, str):
            try:
                mistake_data = json.loads(mistake_data_raw)
            except json.JSONDecodeError:
                return Response(
                    {'success': False, 'error': 'mistake_data is not valid JSON'},
                    status=400
                )
        else:
            mistake_data = mistake_data_raw

        if not isinstance(mistake_data, list):
            return Response(
                {'success': False, 'error': 'mistake_data must be a JSON array'},
                status=400
            )

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
            if not isinstance(md, dict):
                return Response(
                    {'success': False, 'error': 'Each mistake_data entry must be an object'},
                    status=400
                )
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
    """
    GET /api/sessions/
    GET /api/sessions/?days=7 — restricts to the last N days. Powers the
    Home screen's "Recent Sessions" without shipping the user's whole
    history to filter on-device.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        sessions = PracticeSession.objects.filter(user=request.user)

        days_param = request.query_params.get('days')
        if days_param:
            try:
                days = max(1, int(days_param))
                cutoff = timezone.now() - timedelta(days=days)
                sessions = sessions.filter(created_at__gte=cutoff)
            except (TypeError, ValueError):
                pass  # malformed ?days= -> fall back to unfiltered, capped list

        sessions = list(sessions.order_by('-created_at')[:50])
        serialized = SessionSerializer(sessions, many=True).data

        # Make sure every row carries `total_words` (the Home screen's
        # "Words Practiced" sums it). setdefault keeps the serializer's own
        # value if it already exposes the field.
        payload = []
        for session, item in zip(sessions, serialized):
            row = dict(item)
            row.setdefault('total_words', session.total_words)
            payload.append(row)
        return Response(payload)


class ProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user    = request.user
        profile = self._get_or_create_profile(user)

        sessions = PracticeSession.objects.filter(user=user)
        total_sessions   = sessions.count()
        scores           = [s.accuracy for s in sessions if s.accuracy is not None]
        avg_accuracy     = round(sum(scores) / len(scores), 1) if scores else 0.0
        best_score       = round(max(scores), 1) if scores else 0.0
        total_words      = PhonemeError.objects.filter(
            session__user=user).count()
        correct_words    = PhonemeError.objects.filter(
            session__user=user, error_type='correct').count()

        words_practiced = sessions.aggregate(
            total=Sum('total_words')
        )['total'] or 0

        computed_level = self._compute_level(avg_accuracy, total_sessions)
        streak = self._compute_streak(user)
        weekly_accuracy = _compute_weekly_accuracy(user)

        # FIX: sessions saved without a Script (custom/pasted text, or a
        # script_id that no longer exists) have script__mode = NULL, which
        # used to make favorite_mode None and crash the .replace() below.
        # Treat a missing script as 'script_based', matching what
        # dashboard_summary and progress_analytics already do. 'None yet'
        # is now only returned when the user has no sessions at all.
        mode_counts = sessions.annotate(
            mode_value=Coalesce('script__mode', Value('script_based'))
        ).values('mode_value').annotate(
            count=Count('id')).order_by('-count')
        top_mode = mode_counts.first()
        favorite_mode = (top_mode['mode_value'] if top_mode else None) or 'None yet'

        top_errors = PhonemeError.objects.filter(
            session__user=user
        ).exclude(
            error_type='correct'
        ).values('word').annotate(
            count=Count('id')
        ).order_by('-count')[:5]

        return Response({
            'username':    user.username,
            'email':       user.email,
            'first_name':  user.first_name,
            'member_since': user.date_joined.strftime('%B %Y'),
            'member_since_full': user.date_joined.isoformat(),

            'proficiency_level':    profile.proficiency_level,
            'computed_level':       computed_level,
            'level_reason':         self._level_reason(avg_accuracy, total_sessions),

            'voice_feedback':       profile.voice_feedback,
            'reference_accent':     profile.reference_accent,
            'daily_reminders':      profile.daily_reminders,
            'auto_save_recordings': profile.auto_save_recordings,

            'total_sessions':  total_sessions,
            'avg_accuracy':    avg_accuracy,
            'best_score':      best_score,
            'total_words':     total_words,
            'correct_words':   correct_words,
            'words_practiced': words_practiced,
            'streak_days':     streak,
            'favorite_mode':   favorite_mode.replace('_', ' ').title(),
            'weekly_accuracy': weekly_accuracy,

            'top_errors': [
            {'word': e['word'], 'count': e['count']}
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
            valid = ['beginner', 'intermediate', 'expert']
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

        if 'voice_feedback' in data:
            value = data['voice_feedback']
            if not isinstance(value, bool):
                return Response({'error': 'voice_feedback must be a boolean'}, status=400)
            profile.voice_feedback = value
            profile.save(update_fields=['voice_feedback'])

        if 'reference_accent' in data:
            valid_accents = ['american', 'british']
            accent = data['reference_accent']
            if accent not in valid_accents:
                return Response({'error': 'Invalid reference_accent'}, status=400)
            profile.reference_accent = accent
            profile.save(update_fields=['reference_accent'])

        if 'daily_reminders' in data:
            value = data['daily_reminders']
            if not isinstance(value, bool):
                return Response({'error': 'daily_reminders must be a boolean'}, status=400)
            profile.daily_reminders = value
            profile.save(update_fields=['daily_reminders'])

        if 'auto_save_recordings' in data:
            value = data['auto_save_recordings']
            if not isinstance(value, bool):
                return Response({'error': 'auto_save_recordings must be a boolean'}, status=400)
            profile.auto_save_recordings = value
            profile.save(update_fields=['auto_save_recordings'])

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
        if total_sessions < 3:
            return 'beginner'
        if avg_accuracy >= 78:
            return 'expert'
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


def _auto_level_from_word_count(word_count: int) -> str:
    """
    Assigns a difficulty level automatically from content length, used
    when the caller doesn't supply one (see ScriptUploadView below —
    currently only the "type or paste text" flow omits it; file uploads
    still let the user pick explicitly via the level picker dialog).

    Thresholds: beginner ≤ 60 words, intermediate 61-150, expert > 150.
    """
    if word_count <= 60:
        return 'beginner'
    if word_count <= 150:
        return 'intermediate'
    return 'expert'


class ScriptUploadView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes     = [MultiPartParser, JSONParser]

    def post(self, request):
        title   = request.data.get('title', '').strip()
        content = request.data.get('content', '').strip()
        level   = (request.data.get('level') or '').strip() or None
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

        # FIX: level is now optional. When the caller doesn't supply one
        # (the "type or paste text" dialog no longer shows a difficulty
        # picker), assign it automatically from word count instead of
        # silently defaulting to 'intermediate' regardless of length.
        if not level:
            level = _auto_level_from_word_count(len(content.split()))

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


class GenerateScriptView(APIView):
    """
    POST /api/scripts/generate/
    Body: {"topic": str, "word_limit": int, "mode": str, "level": str}
    """
    permission_classes = [IsAuthenticated]
    parser_classes = [JSONParser]

    MIN_WORD_LIMIT = 20
    MAX_WORD_LIMIT = 400
    MAX_TOPIC_LEN = 200

    def post(self, request):
        topic = (request.data.get('topic') or '').strip()
        mode  = request.data.get('mode')
        level = request.data.get('level')

        try:
            word_limit = int(request.data.get('word_limit', 60))
        except (TypeError, ValueError):
            return Response({'error': 'word_limit must be a number'}, status=400)

        if not topic:
            return Response({'error': 'topic is required'}, status=400)
        if len(topic) > self.MAX_TOPIC_LEN:
            return Response(
                {'error': f'topic must be under {self.MAX_TOPIC_LEN} characters'},
                status=400,
            )
        if mode not in dict(Script.MODE_CHOICES):
            return Response({'error': f'Invalid mode "{mode}"'}, status=400)
        if level not in dict(LEVEL_CHOICES):
            return Response({'error': f'Invalid level "{level}"'}, status=400)

        word_limit = max(self.MIN_WORD_LIMIT, min(word_limit, self.MAX_WORD_LIMIT))

        content = gemini_service.generate_custom_script(
            topic=topic, word_limit=word_limit, level=level, mode=mode,
        )

        if not content:
            return Response(
                {'error': 'Script generation failed. Please try again.'},
                status=502,
            )

        title = topic if len(topic) <= 80 else topic[:77] + '...'

        script = Script.objects.create(
            title=title,
            description=f'AI-generated · topic: "{topic}"',
            level=level,
            mode=mode,
            content=content,
            is_builtin=False,
            uploaded_by=request.user,
        )
        return Response(ScriptSerializer(script).data, status=201)

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
        'title': (s.script.title[:50] + '...') if s.script and len(s.script.title) > 50 else (s.script.title if s.script else 'Custom Script'),
        'score': float(s.accuracy),
        'grade': s.grade,
        'date_label': _format_date(s.created_at),
        'mode': s.script.mode if s.script else 'script_based',
        'total_words': s.total_words,
        'flagged_words': s.flagged_words,
    } for s in recent]
    
    total_words = sessions.aggregate(total=Sum('total_words'))['total'] or 0
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
        'mode': s.script.mode if s.script else 'script_based',
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
    
    total_words = sessions.aggregate(total=Sum('total_words'))['total'] or 0
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


def _current_week_start():
    """Monday of the current calendar week (local date)."""
    today = timezone.localdate() if settings.USE_TZ else timezone.now().date()
    return today - timedelta(days=today.weekday())


def _compute_weekly_accuracy(user):
    """
    Calendar-week heatmap data: Monday -> Sunday of the CURRENT week.
    Resets to empty on Monday. Days that haven't happened yet are
    returned with 0 sessions so the client always gets 7 entries.
    """
    start_date = _current_week_start()

    rows = (
        PracticeSession.objects
        .filter(user=user, created_at__date__gte=start_date)
        .annotate(day=TruncDate('created_at'))
        .values('day')
        .annotate(avg_accuracy=Avg('accuracy'), session_count=Count('id'))
    )
    by_day = {row['day']: row for row in rows}

    result = []
    for i in range(7):
        day = start_date + timedelta(days=i)
        row = by_day.get(day)
        result.append({
            'date': day.isoformat(),
            'average_accuracy': round(float(row['avg_accuracy']), 1) if row else 0.0,
            'session_count': row['session_count'] if row else 0,
        })
    return result


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


def _infer_mode_scores(sessions_qs):
    overall_avg = sessions_qs.aggregate(avg=Avg('accuracy'))['avg'] or 0

    result = {}
    for mode_value, _label in Script.MODE_CHOICES:
        mode_avg = sessions_qs.filter(script__mode=mode_value).aggregate(
            avg=Avg('accuracy')
        )['avg']
        result[mode_value] = round(
            float(mode_avg if mode_avg is not None else overall_avg), 1
        )
    return result



class ScriptRotationView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        mode = request.query_params.get('mode')
        level = request.query_params.get('level')

        try:
            count = int(request.query_params.get('count', 5))
        except (TypeError, ValueError):
            count = 5
        count = max(1, min(count, 20))

        if mode and mode not in dict(Script.MODE_CHOICES):
            return Response(
                {'error': f'Invalid mode "{mode}"'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if level and level not in dict(LEVEL_CHOICES):
            return Response(
                {'error': f'Invalid level "{level}"'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        scripts = get_rotated_scripts(request.user, mode=mode, level=level, count=count)
        serializer = RotatedScriptSerializer(scripts, many=True)
        return Response({'scripts': serializer.data})


# ═══════════════════════════════════════════════════════════════
# IMPROVEMENT PHASE ENDPOINTS
# ═══════════════════════════════════════════════════════════════

class ProgressionStatusView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        status_obj, _ = evaluate_progression(request.user)
        return Response(LevelProgressionStatusSerializer(status_obj).data)


class WordMasteryListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        qs = WordMastery.objects.filter(user=request.user)
        status_filter = request.query_params.get('status')
        if status_filter in ('active', 'mastered'):
            qs = qs.filter(status=status_filter)
        return Response(WordMasterySerializer(qs, many=True).data)


class FocusScriptListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        qs = FocusScript.objects.filter(user=request.user)
        return Response(FocusScriptSerializer(qs, many=True).data)


class LatestFocusScriptView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        focus_script = FocusScript.objects.filter(
            user=request.user, completed=False
        ).order_by('-generated_at').first()

        if not focus_script:
            return Response({'focus_script': None})

        return Response({'focus_script': FocusScriptSerializer(focus_script).data})


class CompleteFocusScriptView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request, script_id):
        try:
            focus_script = FocusScript.objects.get(id=script_id, user=request.user)
        except FocusScript.DoesNotExist:
            return Response({'error': 'Focus Script not found'}, status=404)

        post_error_rate = request.data.get('post_error_rate')
        if post_error_rate is None:
            return Response({'error': 'post_error_rate is required'}, status=400)

        focus_script.completed = True
        focus_script.completed_at = timezone.now()
        focus_script.post_error_rate = float(post_error_rate)
        focus_script.save()

        return Response(FocusScriptSerializer(focus_script).data)



class ConfirmPromotionView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        new_level = promote_if_eligible(request.user)
        if new_level:
            try:
                notif.notify_level_up(request.user, new_level)
            except Exception:
                logger.exception('Level-up notification failed (non-fatal)')
            return Response({'promoted': True, 'new_level': new_level})
        return Response(
            {'promoted': False, 'message': 'Not currently eligible for promotion'},
            status=400,
        )


class FocusScriptStatsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        completed = FocusScript.objects.filter(
            user=request.user, completed=True, post_error_rate__isnull=False
        )
        total = completed.count()

        if total == 0:
            return Response({
                'total_completed': 0,
                'avg_pre_error_rate': None,
                'avg_post_error_rate': None,
                'avg_improvement_pct': None,
            })

        deltas = [
            (fs.pre_error_rate or 100.0) - fs.post_error_rate
            for fs in completed
        ]
        avg_pre = sum(fs.pre_error_rate or 100.0 for fs in completed) / total
        avg_post = sum(fs.post_error_rate for fs in completed) / total
        avg_improvement = sum(deltas) / total

        return Response({
            'total_completed': total,
            'avg_pre_error_rate': round(avg_pre, 1),
            'avg_post_error_rate': round(avg_post, 1),
            'avg_improvement_pct': round(avg_improvement, 1),
        })


# ═══════════════════════════════════════════════════════════════
# NOTIFICATIONS
# ═══════════════════════════════════════════════════════════════

class NotificationListView(APIView):
    """
    GET /api/notifications/
    Most recent 50 notifications for the user, plus an unread count for
    the Home screen's bell badge.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        qs = Notification.objects.filter(user=request.user)[:50]
        unread_count = Notification.objects.filter(user=request.user, is_read=False).count()
        return Response({
            'notifications': NotificationSerializer(qs, many=True).data,
            'unread_count': unread_count,
        })


class MarkNotificationsReadView(APIView):
    """
    POST /api/notifications/read/
    Marks all of the user's unread notifications as read. Called when the
    notifications screen opens.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        Notification.objects.filter(user=request.user, is_read=False).update(is_read=True)
        return Response({'message': 'Notifications marked as read'})


class NotificationDeleteView(APIView):
    """
    DELETE /api/notifications/<id>/
    Removes one of the user's own notifications (swipe-to-dismiss in the
    notifications feed). Scoped to request.user so nobody can delete
    someone else's notification by guessing an id.
    """
    permission_classes = [IsAuthenticated]

    def delete(self, request, notification_id):
        deleted, _ = Notification.objects.filter(
            id=notification_id, user=request.user
        ).delete()
        if not deleted:
            return Response({'error': 'Notification not found'}, status=404)
        return Response({'message': 'Notification deleted'})