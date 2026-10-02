# api/focus_views.py
import logging
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .models import FocusScript, PracticeSession
from .serializers import FocusScriptSerializer
from .focus_results import build_focus_result

logger = logging.getLogger(__name__)


class FocusScriptDetailView(APIView):
    """
    GET /api/focus-scripts/<id>/
    One Focus Script including its 'baseline' (how each target word was
    wrong before) — the app uses it to pre-highlight the error words.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, script_id):
        focus_script = FocusScript.objects.filter(
            id=script_id, user=request.user
        ).first()
        if not focus_script:
            return Response({'error': 'Focus Script not found'}, status=404)
        return Response(FocusScriptSerializer(focus_script).data)


class CompleteFocusScriptView(APIView):
    """
    PATCH /api/focus-scripts/<id>/complete/

    New flow:    body {"session_id": <PracticeSession id>}
                 -> compares that session against the script's baseline,
                    stores the result, returns the Focus Script with
                    a full 'result' comparison.
    Legacy flow: body {"post_error_rate": <float>} (kept so older app
                 builds keep working).
    """
    permission_classes = [IsAuthenticated]

    def patch(self, request, script_id):
        try:
            focus_script = FocusScript.objects.get(id=script_id, user=request.user)
        except FocusScript.DoesNotExist:
            return Response({'error': 'Focus Script not found'}, status=404)

        session_id = request.data.get('session_id')

        if session_id is not None:
            try:
                session = PracticeSession.objects.get(
                    id=int(session_id), user=request.user
                )
            except (PracticeSession.DoesNotExist, TypeError, ValueError):
                return Response({'error': 'Session not found'}, status=404)

            if session.created_at < focus_script.generated_at:
                return Response(
                    {'error': 'Session predates this Focus Script'}, status=400
                )

            result = build_focus_result(focus_script, session)

            focus_script.completed = True
            focus_script.completed_at = timezone.now()
            focus_script.result = result
            if result['post_error_rate'] is not None:
                focus_script.post_error_rate = result['post_error_rate']
            focus_script.save()

            return Response(FocusScriptSerializer(focus_script).data)

        # ── Legacy path ──
        post_error_rate = request.data.get('post_error_rate')
        if post_error_rate is None:
            return Response(
                {'error': 'session_id or post_error_rate is required'}, status=400
            )

        focus_script.completed = True
        focus_script.completed_at = timezone.now()
        focus_script.post_error_rate = float(post_error_rate)
        focus_script.save()
        return Response(FocusScriptSerializer(focus_script).data)