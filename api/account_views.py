# api/account_views.py
"""Endpoints for a signed-in user managing their own account."""
import hmac
import logging

from django.conf import settings
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError as DjangoValidationError
from django.core.mail import send_mail
from django.core.validators import validate_email as django_validate_email
from django.db import transaction
from django.db.models import F
from django.utils import timezone
from rest_framework import serializers
from rest_framework.authtoken.models import Token
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

from .auth_serializers import normalize_email, validate_email_domain
from .auth_views import (
    BAD_CODE_MESSAGE,
    MAX_OTP_ATTEMPTS,
    OTP_TTL,
    RESEND_COOLDOWN,
    _bad_request,
    _clean_code,
    _cooldown_remaining,
    _delete_users,
    _generate_code,
    _hash,
    _pending_unverified,
)
from .models import EmailOTP
from .views import ProfileView as BaseProfileView

logger = logging.getLogger(__name__)

# Stored in EmailOTP.purpose. (Not listed in the model's `choices`, which are
# only used by forms/admin, so no migration is needed.)
PURPOSE_CHANGE_EMAIL = 'change_email'


# ───────────────────────── change password ─────────────────────────

class ChangePasswordView(APIView):
    """
    POST /api/change-password/   Body: {current_password, new_password}

    On success every existing token is revoked (other devices are signed
    out) and a fresh token is returned so the calling device stays signed in.
    A wrong current password answers 400, not 401, so the app can tell
    "wrong password" apart from "session expired".
    """
    permission_classes = [IsAuthenticated]
    throttle_classes   = [ScopedRateThrottle]
    throttle_scope     = 'auth_login'   # same limit as sign-in attempts

    def post(self, request):
        user    = request.user
        current = request.data.get('current_password') or ''
        new     = request.data.get('new_password') or ''

        if not user.check_password(current):
            return _bad_request({'current_password': ['Your current password is incorrect.']})

        try:
            validate_password(new, user=user)
        except DjangoValidationError as exc:
            return _bad_request({'new_password': list(exc.messages)})

        if current == new:
            return _bad_request({'new_password': ['New password must be different from your current one.']})

        with transaction.atomic():
            user.set_password(new)
            user.save(update_fields=['password'])
            Token.objects.filter(user=user).delete()
            token = Token.objects.create(user=user)

        return Response({'message': 'Password updated.', 'token': token.key})


# ───────────────────────── change email (OTP-verified) ─────────────────────────

def _bind(new_email: str) -> str:
    """The code is cryptographically tied to the exact address it was sent to."""
    return f'{PURPOSE_CHANGE_EMAIL}:{new_email}'


def _email_problem(new_email: str):
    """Returns a message if the address is unusable, otherwise None."""
    if not new_email:
        return 'Enter the new email address.'
    try:
        django_validate_email(new_email)
    except DjangoValidationError:
        return 'Enter a valid email address.'
    try:
        validate_email_domain(new_email)
    except serializers.ValidationError as exc:
        detail = exc.detail
        return str(detail[0] if isinstance(detail, list) else detail)
    return None


def _email_in_use(new_email: str, user) -> bool:
    """True if a real (non-pending) account other than `user` owns the address."""
    pending_ids = _pending_unverified().values('pk')
    return (User.objects
            .filter(email__iexact=new_email)
            .exclude(pk=user.pk)
            .exclude(pk__in=pending_ids)
            .exists())


def _send_change_email_code(user, new_email: str, code: str) -> bool:
    minutes = int(OTP_TTL.total_seconds() // 60)
    name = user.first_name or user.username
    body = (
        f'Hi {name},\n\n'
        f'Use this code to confirm your new PronounceMate email address:\n\n'
        f'    {code}\n\n'
        f'This code expires in {minutes} minutes. '
        f'If you did not request this, you can safely ignore this email.\n\n'
        f'- PronounceMate'
    )
    try:
        send_mail('Confirm your new PronounceMate email', body,
                  settings.DEFAULT_FROM_EMAIL, [new_email], fail_silently=False)
        return True
    except Exception:
        logger.exception('Failed to send change-email code for user %s', user.pk)
        return False


def _notify_old_email(old_email: str, new_email: str, user) -> None:
    """Best-effort security notice to the address that was just replaced."""
    if not old_email:
        return
    body = (
        f'Hi {user.first_name or user.username},\n\n'
        f'The email address on your PronounceMate account was changed to {new_email}.\n'
        f'If this was not you, reset your password immediately.\n\n'
        f'- PronounceMate'
    )
    try:
        send_mail('Your PronounceMate email was changed', body,
                  settings.DEFAULT_FROM_EMAIL, [old_email], fail_silently=True)
    except Exception:
        logger.exception('Failed to notify old email for user %s', user.pk)


class EmailChangeRequestView(APIView):
    """
    POST /api/email-change/request/   Body: {new_email, password}
    Sends a 6-digit code to the NEW address. The email is not changed yet.
    """
    permission_classes = [IsAuthenticated]
    throttle_classes   = [ScopedRateThrottle]
    throttle_scope     = 'auth_email'

    def post(self, request):
        user      = request.user
        password  = request.data.get('password') or ''
        new_email = normalize_email(request.data.get('new_email'))

        if not user.check_password(password):
            return _bad_request({'password': ['Your password is incorrect.']})

        problem = _email_problem(new_email)
        if problem:
            return _bad_request({'new_email': [problem]})

        if new_email == (user.email or '').strip().lower():
            return _bad_request({'new_email': ['This is already your email address.']})

        if _email_in_use(new_email, user):
            return _bad_request({'new_email': ['This email is already in use by another account.']})

        wait = _cooldown_remaining(user, PURPOSE_CHANGE_EMAIL)
        if wait:
            return _bad_request(f'Please wait {wait} seconds before requesting another code.')

        code = _generate_code()
        with transaction.atomic():
            EmailOTP.objects.filter(
                user=user, purpose=PURPOSE_CHANGE_EMAIL, is_used=False
            ).update(is_used=True)
            EmailOTP.objects.create(
                user=user,
                purpose=PURPOSE_CHANGE_EMAIL,
                code_hash=_hash(code, user.pk, _bind(new_email)),
                expires_at=timezone.now() + OTP_TTL,
            )

        if not _send_change_email_code(user, new_email, code):
            return Response(
                {'error': 'We could not send the email. Please try again in a moment.'},
                status=503,
            )

        return Response({
            'message': f'We sent a 6-digit code to {new_email}.',
            'new_email': new_email,
            'expires_in_minutes': int(OTP_TTL.total_seconds() // 60),
            'resend_after_seconds': RESEND_COOLDOWN,
        })


class EmailChangeConfirmView(APIView):
    """
    POST /api/email-change/confirm/   Body: {new_email, otp}
    Verifies the code and switches the account to the new address.
    """
    permission_classes = [IsAuthenticated]
    throttle_classes   = [ScopedRateThrottle]
    throttle_scope     = 'auth_otp'

    def post(self, request):
        user      = request.user
        new_email = normalize_email(request.data.get('new_email'))
        code      = _clean_code(request.data.get('otp'))

        if not new_email or len(code) != 6:
            return _bad_request('Enter the 6-digit code.')

        otp = (EmailOTP.objects
               .filter(user=user, purpose=PURPOSE_CHANGE_EMAIL, is_used=False)
               .order_by('-created_at').first())

        if otp is None or otp.expires_at <= timezone.now():
            return _bad_request(BAD_CODE_MESSAGE)
        if otp.attempts >= MAX_OTP_ATTEMPTS:
            return _bad_request('Too many incorrect attempts. Please request a new code.')

        if not hmac.compare_digest(otp.code_hash, _hash(code, user.pk, _bind(new_email))):
            EmailOTP.objects.filter(pk=otp.pk).update(attempts=F('attempts') + 1)
            left = MAX_OTP_ATTEMPTS - (otp.attempts + 1)
            if left <= 0:
                return _bad_request('Too many incorrect attempts. Please request a new code.')
            return _bad_request(f'Incorrect code. {left} attempt(s) left.')

        old_email = user.email
        with transaction.atomic():
            # The code proves ownership of the address, so any never-verified
            # signup still holding it is released.
            _delete_users(_pending_unverified().filter(email__iexact=new_email).exclude(pk=user.pk))
            if User.objects.filter(email__iexact=new_email).exclude(pk=user.pk).exists():
                return _bad_request('This email is already in use by another account.')

            user.email = new_email
            user.save(update_fields=['email'])
            otp.is_used = True
            otp.verified_at = timezone.now()
            otp.save(update_fields=['is_used', 'verified_at'])

        _notify_old_email(old_email, new_email, user)
        return Response({'message': 'Email updated.', 'email': new_email})


# ───────────────────────── profile (email can no longer be set directly) ─────────────────────────

class ProfileView(BaseProfileView):
    """
    Same as views.ProfileView, except the email can no longer be changed
    through PATCH /profile/ — it must go through the OTP-verified
    /email-change/ endpoints above.
    """

    def patch(self, request):
        if 'email' in request.data:
            return Response(
                {'error': 'To change your email, verify the new address with a code '
                          '(Profile > Email Address).'},
                status=400,
            )
        return super().patch(request)