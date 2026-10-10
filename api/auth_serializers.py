# api/auth_serializers.py
import re

from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError as DjangoValidationError
from django.db import transaction
from rest_framework import serializers

from .models import UserProfile

# Throwaway-inbox providers. A code sent to these proves nothing useful,
# so we refuse them. Extend this list as needed.
DISPOSABLE_DOMAINS = {
    'mailinator.com', 'guerrillamail.com', 'guerrillamail.net', '10minutemail.com',
    '10minutemail.net', 'tempmail.com', 'temp-mail.org', 'temp-mail.io',
    'throwawaymail.com', 'yopmail.com', 'yopmail.net', 'trashmail.com',
    'getnada.com', 'sharklasers.com', 'dispostable.com', 'maildrop.cc',
    'fakeinbox.com', 'mailnesia.com', 'mintemail.com', 'emailondeck.com',
}

USERNAME_RE = re.compile(r'^[A-Za-z0-9_.]{3,30}$')


def normalize_email(value: str) -> str:
    return (value or '').strip().lower()


def validate_email_domain(email: str) -> None:
    """Raises serializers.ValidationError for obviously unusable domains."""
    domain = email.rsplit('@', 1)[-1]
    if '.' not in domain or domain.startswith('.') or domain.endswith('.'):
        raise serializers.ValidationError('Enter a valid email address.')
    if domain in DISPOSABLE_DOMAINS:
        raise serializers.ValidationError(
            'Disposable email addresses are not allowed. Please use your real email.'
        )


class RegisterSerializer(serializers.Serializer):
    username   = serializers.CharField(max_length=30)
    email      = serializers.EmailField(max_length=254)
    password   = serializers.CharField(write_only=True, trim_whitespace=False, max_length=128)
    first_name = serializers.CharField(required=False, allow_blank=True, max_length=30)
    last_name  = serializers.CharField(required=False, allow_blank=True, max_length=30)

    def validate_username(self, value):
        value = value.strip()
        if not USERNAME_RE.match(value):
            raise serializers.ValidationError(
                'Username must be 3-30 characters: letters, numbers, underscore or dot.'
            )
        if User.objects.filter(username__iexact=value).exists():
            raise serializers.ValidationError('This username is already taken.')
        return value

    def validate_email(self, value):
        value = normalize_email(value)
        validate_email_domain(value)
        if User.objects.filter(email__iexact=value).exists():
            raise serializers.ValidationError('An account with this email already exists.')
        return value

    def validate(self, attrs):
        candidate = User(
            username=attrs.get('username', ''),
            email=attrs.get('email', ''),
            first_name=attrs.get('first_name', ''),
            last_name=attrs.get('last_name', ''),
        )
        try:
            validate_password(attrs['password'], user=candidate)
        except DjangoValidationError as exc:
            raise serializers.ValidationError({'password': list(exc.messages)})
        return attrs

    def create(self, validated_data):
        with transaction.atomic():
            user = User.objects.create_user(
                username=validated_data['username'],
                email=validated_data['email'],
                password=validated_data['password'],
                first_name=validated_data.get('first_name', '').strip(),
                last_name=validated_data.get('last_name', '').strip(),
                is_active=False,  # activated only after the email code is verified
            )
            UserProfile.objects.create(user=user)
        return user