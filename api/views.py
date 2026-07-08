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