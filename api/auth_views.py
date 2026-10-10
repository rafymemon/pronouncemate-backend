# api/auth_views.py
"""
Authentication endpoints for PronounceMate:
register, verify-email, resend-verification, login, logout,
forgot-password, verify-reset-otp, reset-password.
"""
import hashlib
import hmac
import logging
import math
import secrets
from datetime import timedelta

from django.conf import settings
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError as DjangoValidationError
from django.core.mail import send_mail
from django.db import transaction
from django.db.models import F, Q
from django.utils import timezone
from rest_framework.authtoken.models import Token
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

from .auth_serializers import RegisterSerializer, normalize_email
from .models import EmailOTP

logger = logging.getLogger(__name__)

OTP_TTL            = timedelta(minutes=10)
RESET_TOKEN_TTL    = timedelta(minutes=10)
MAX_OTP_ATTEMPTS   = 5
RESEND_COOLDOWN    = 60  # seconds
UNVERIFIED_MAX_AGE = timedelta(hours=24)

GENERIC_SENT_MESSAGE = 'If an account exists for this email, a 6-digit code has been sent.'
BAD_CODE_MESSAGE     = 'Invalid or expired code. Please request a new one.'


# ───────────────────────── helpers ─────────────────────────

def _hash(value: str, *parts) -> str:
    """HMAC-SHA256 keyed with SECRET_KEY; `parts` bind the hash to user/purpose."""
    message = ':'.join(str(p) for p in parts) + ':' + value
    return hmac.new(
        settings.SECRET_KEY.encode(), message.encode(), hashlib.sha256
    ).hexdigest()


def _generate_code() -> str:
    return f'{secrets.randbelow(1_000_000):06d}'


def _first_error(errors) -> str:
    """Pulls one human-readable message out of a DRF errors dict."""
    if isinstance(errors, dict):
        for value in errors.values():
            msg = _first_error(value)
            if msg:
                return msg
    elif isinstance(errors, (list, tuple)):
        for value in errors:
            msg = _first_error(value)
            if msg:
                return msg
    elif errors:
        return str(errors)
    return 'Invalid input.'


def _bad_request(errors):
    """400 response: `error` (one message for the UI) + `errors` (per field)."""
    return Response({'error': _first_error(errors), 'errors': errors}, status=400)


def _issue_otp(user, purpose) -> str:
    """Invalidates older open codes for this purpose and creates a fresh one."""
    code = _generate_code()
    with transaction.atomic():
        EmailOTP.objects.filter(user=user, purpose=purpose, is_used=False).update(is_used=True)
        EmailOTP.objects.create(
            user=user,
            purpose=purpose,
            code_hash=_hash(code, user.pk, purpose),
            expires_at=timezone.now() + OTP_TTL,
        )
    return code


def _cooldown_remaining(user, purpose) -> int:
    last = EmailOTP.objects.filter(user=user, purpose=purpose).order_by('-created_at').first()
    if not last:
        return 0
    elapsed = (timezone.now() - last.created_at).total_seconds()
    return max(0, math.ceil(RESEND_COOLDOWN - elapsed))


def _send_otp_email(user, code, purpose) -> bool:
    minutes = int(OTP_TTL.total_seconds() // 60)
    name = user.first_name or user.username
    if purpose == EmailOTP.PURPOSE_VERIFY:
        subject = 'Verify your PronounceMate email'
        intro = 'Use this code to verify your email address:'
    else:
        subject = 'Your PronounceMate password reset code'
        intro = 'Use this code to reset your password:'

    body = (
        f'Hi {name},\n\n'
        f'{intro}\n\n'
        f'    {code}\n\n'
        f'This code expires in {minutes} minutes. '
        f'If you did not request it, you can safely ignore this email.\n\n'
        f'- PronounceMate'
    )
    try:
        send_mail(subject, body, settings.DEFAULT_FROM_EMAIL, [user.email], fail_silently=False)
        return True
    except Exception:
        logger.exception('Failed to send OTP email to user %s', user.pk)
        return False


def _check_otp(user, purpose, code):
    """
    Returns (otp, None) when `code` matches the newest open OTP, otherwise
    (None, error_message). Wrong guesses are counted; after MAX_OTP_ATTEMPTS
    the code is dead and the user must request a new one.
    """
    otp = (EmailOTP.objects
           .filter(user=user, purpose=purpose, is_used=False)
           .order_by('-created_at').first())

    if otp is None or otp.expires_at <= timezone.now():
        return None, BAD_CODE_MESSAGE
    if otp.attempts >= MAX_OTP_ATTEMPTS:
        return None, 'Too many incorrect attempts. Please request a new code.'

    if not hmac.compare_digest(otp.code_hash, _hash(code, user.pk, purpose)):
        EmailOTP.objects.filter(pk=otp.pk).update(attempts=F('attempts') + 1)
        left = MAX_OTP_ATTEMPTS - (otp.attempts + 1)
        if left <= 0:
            return None, 'Too many incorrect attempts. Please request a new code.'
        return None, f'Incorrect code. {left} attempt(s) left.'

    return otp, None


def _clean_code(raw) -> str:
    return ''.join(ch for ch in str(raw or '') if ch.isdigit())


def _pending_unverified():
    """
    Accounts that registered but never completed email verification.
    Anyone who was ever verified (e.g. later deactivated by an admin) and
    all staff are excluded, so they are never treated as "pending".
    """
    return (User.objects
            .filter(is_active=False, is_staff=False, is_superuser=False,
                    email_otps__purpose=EmailOTP.PURPOSE_VERIFY)
            .exclude(email_otps__verified_at__isnull=False))


def _delete_users(queryset):
    ids = list(queryset.values_list('pk', flat=True))
    if ids:
        User.objects.filter(pk__in=ids).delete()


def _purge_stale_unverified():
    """Deletes accounts that never verified within 24h."""
    cutoff = timezone.now() - UNVERIFIED_MAX_AGE
    _delete_users(_pending_unverified().filter(date_joined__lt=cutoff))


def _release_pending_claims(data):
    """
    If someone signs up again with a username/email that belongs to a
    still-UNVERIFIED account (e.g. they mistyped their email the first time),
    drop the unverified account so the new signup can take over. Verified
    accounts are never touched.
    """
    username = str(data.get('username') or '').strip()
    email = normalize_email(str(data.get('email') or ''))
    conditions = Q()
    if username:
        conditions |= Q(username__iexact=username)
    if email:
        conditions |= Q(email__iexact=email)
    if conditions:
        _delete_users(_pending_unverified().filter(conditions))


def _unverified_user_by_email(email):
    return (_pending_unverified()
            .filter(email__iexact=email)
            .order_by('-date_joined').first())


def _active_user_by_email(email):
    return User.objects.filter(email__iexact=email, is_active=True).order_by('id').first()


# ───────────────────────── registration ─────────────────────────

class RegisterView(APIView):
    """
    POST /api/register/
    Body: {username, email, password, first_name?, last_name?}
    Creates an INACTIVE account and emails a 6-digit verification code.
    """
    permission_classes = [AllowAny]
    throttle_classes   = [ScopedRateThrottle]
    throttle_scope     = 'auth_register'

    def post(self, request):
        _purge_stale_unverified()
        _release_pending_claims(request.data)

        serializer = RegisterSerializer(data=request.data)
        if not serializer.is_valid():
            return _bad_request(serializer.errors)

        user = serializer.save()
        code = _issue_otp(user, EmailOTP.PURPOSE_VERIFY)
        sent = _send_otp_email(user, code, EmailOTP.PURPOSE_VERIFY)

        return Response({
            'message': 'Account created. Enter the 6-digit code we emailed you to verify your address.'
                       if sent else
                       'Account created, but we could not send the email. Tap "Resend code" to try again.',
            'email': user.email,
            'verification_required': True,
            'email_sent': sent,
            'expires_in_minutes': int(OTP_TTL.total_seconds() // 60),
            'resend_after_seconds': RESEND_COOLDOWN,
        }, status=201)


class VerifyEmailView(APIView):
    """POST /api/verify-email/   Body: {email, otp}"""
    permission_classes = [AllowAny]
    throttle_classes   = [ScopedRateThrottle]
    throttle_scope     = 'auth_otp'

    def post(self, request):
        email = normalize_email(request.data.get('email'))
        code  = _clean_code(request.data.get('otp'))
        if not email or len(code) != 6:
            return _bad_request('Enter the 6-digit code.')

        user = _unverified_user_by_email(email)
        if user is None:
            return _bad_request(BAD_CODE_MESSAGE)

        otp, error = _check_otp(user, EmailOTP.PURPOSE_VERIFY, code)
        if error:
            return _bad_request(error)

        with transaction.atomic():
            user.is_active = True
            user.save(update_fields=['is_active'])
            otp.is_used = True
            otp.verified_at = timezone.now()
            otp.save(update_fields=['is_used', 'verified_at'])

        return Response({'message': 'Email verified. You can now sign in.', 'verified': True})


class ResendVerificationView(APIView):
    """POST /api/resend-verification/   Body: {email}  (always answers 200)"""
    permission_classes = [AllowAny]
    throttle_classes   = [ScopedRateThrottle]
    throttle_scope     = 'auth_email'

    def post(self, request):
        email = normalize_email(request.data.get('email'))
        if not email or '@' not in email:
            return _bad_request('Enter a valid email address.')

        user = _unverified_user_by_email(email)
        if user and _cooldown_remaining(user, EmailOTP.PURPOSE_VERIFY) == 0:
            code = _issue_otp(user, EmailOTP.PURPOSE_VERIFY)
            _send_otp_email(user, code, EmailOTP.PURPOSE_VERIFY)

        return Response({
            'message': 'If this email is awaiting verification, a new code has been sent.',
            'resend_after_seconds': RESEND_COOLDOWN,
        })


# ───────────────────────── login / logout ─────────────────────────

class LoginView(APIView):
    """
    POST /api/login/
    Body: {username, password}   ("username" may also be the account's email)
    """
    permission_classes = [AllowAny]
    throttle_classes   = [ScopedRateThrottle]
    throttle_scope     = 'auth_login'

    def post(self, request):
        identifier = (request.data.get('identifier') or request.data.get('username') or '').strip()
        password   = request.data.get('password') or ''
        if not identifier or not password:
            return Response({'error': 'Enter your username and password.'}, status=400)

        user = (
            User.objects.filter(username=identifier).first()
            or User.objects.filter(email__iexact=identifier).order_by('-is_active', 'id').first()
            or User.objects.filter(username__iexact=identifier).first()
        )

        # Same message for "no such user" and "wrong password".
        if user is None or not user.check_password(password):
            return Response({'error': 'Invalid username or password.'}, status=401)

        if not user.is_active:
            if _pending_unverified().filter(pk=user.pk).exists():
                return Response({
                    'error': 'Please verify your email before signing in.',
                    'code': 'email_not_verified',
                    'email': user.email,
                }, status=403)
            return Response({'error': 'Invalid username or password.'}, status=401)

        token, _ = Token.objects.get_or_create(user=user)
        return Response({'token': token.key, 'username': user.username, 'email': user.email})


class LogoutView(APIView):
    """POST /api/logout/ — invalidates the caller's token on the server."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        Token.objects.filter(user=request.user).delete()
        return Response({'message': 'Signed out.'})


# ───────────────────────── forgot / reset password ─────────────────────────

class ForgotPasswordView(APIView):
    """
    POST /api/forgot-password/   Body: {email}
    Always answers 200 with the same message, whether or not the email
    belongs to an account, so it cannot be used to discover registered emails.
    """
    permission_classes = [AllowAny]
    throttle_classes   = [ScopedRateThrottle]
    throttle_scope     = 'auth_email'

    def post(self, request):
        email = normalize_email(request.data.get('email'))
        if not email or '@' not in email or '.' not in email.rsplit('@', 1)[-1]:
            return _bad_request('Enter a valid email address.')

        user = _active_user_by_email(email)
        if user and _cooldown_remaining(user, EmailOTP.PURPOSE_RESET) == 0:
            code = _issue_otp(user, EmailOTP.PURPOSE_RESET)
            _send_otp_email(user, code, EmailOTP.PURPOSE_RESET)

        return Response({
            'message': GENERIC_SENT_MESSAGE,
            'expires_in_minutes': int(OTP_TTL.total_seconds() // 60),
            'resend_after_seconds': RESEND_COOLDOWN,
        })


class VerifyResetOTPView(APIView):
    """
    POST /api/verify-reset-otp/   Body: {email, otp}
    On success returns a short-lived `reset_token` required by reset-password.
    """
    permission_classes = [AllowAny]
    throttle_classes   = [ScopedRateThrottle]
    throttle_scope     = 'auth_otp'

    def post(self, request):
        email = normalize_email(request.data.get('email'))
        code  = _clean_code(request.data.get('otp'))
        if not email or len(code) != 6:
            return _bad_request('Enter the 6-digit code.')

        user = _active_user_by_email(email)
        if user is None:
            return _bad_request(BAD_CODE_MESSAGE)

        otp, error = _check_otp(user, EmailOTP.PURPOSE_RESET, code)
        if error:
            return _bad_request(error)

        reset_token = secrets.token_urlsafe(32)
        now = timezone.now()
        otp.reset_token_hash = _hash(reset_token, user.pk, 'reset_token')
        otp.verified_at = now
        otp.expires_at = now + RESET_TOKEN_TTL
        otp.save(update_fields=['reset_token_hash', 'verified_at', 'expires_at'])

        return Response({
            'message': 'Code verified. Choose a new password.',
            'reset_token': reset_token,
            'expires_in_minutes': int(RESET_TOKEN_TTL.total_seconds() // 60),
        })


class ResetPasswordView(APIView):
    """
    POST /api/reset-password/   Body: {email, reset_token, new_password}
    Sets the new password and signs the user out everywhere.
    """
    permission_classes = [AllowAny]
    throttle_classes   = [ScopedRateThrottle]
    throttle_scope     = 'auth_otp'

    def post(self, request):
        email        = normalize_email(request.data.get('email'))
        reset_token  = request.data.get('reset_token') or ''
        new_password = request.data.get('new_password') or ''
        expired_msg  = 'Your reset session has expired. Please start again.'

        if not email or not reset_token:
            return _bad_request(expired_msg)

        user = _active_user_by_email(email)
        if user is None:
            return _bad_request(expired_msg)

        otp = (EmailOTP.objects
               .filter(user=user, purpose=EmailOTP.PURPOSE_RESET,
                       is_used=False, verified_at__isnull=False)
               .order_by('-created_at').first())

        if (otp is None
                or otp.expires_at <= timezone.now()
                or not otp.reset_token_hash
                or not hmac.compare_digest(
                    otp.reset_token_hash, _hash(reset_token, user.pk, 'reset_token'))):
            return _bad_request(expired_msg)

        try:
            validate_password(new_password, user=user)
        except DjangoValidationError as exc:
            return _bad_request({'new_password': list(exc.messages)})

        if user.check_password(new_password):
            return _bad_request({'new_password': ['New password must be different from your old one.']})

        with transaction.atomic():
            user.set_password(new_password)
            user.save(update_fields=['password'])
            EmailOTP.objects.filter(
                user=user, purpose=EmailOTP.PURPOSE_RESET, is_used=False
            ).update(is_used=True)
            Token.objects.filter(user=user).delete()  # sign out all devices

        return Response({'message': 'Password updated. Please sign in with your new password.'})