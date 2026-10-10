# api/management/commands/check_email.py
"""
Usage:  python manage.py check_email you@gmail.com

Shows how OTP emails will be delivered, sends a real test email, and lists
accounts whose email is empty or duplicated (those can never receive codes).
"""
import smtplib
from collections import Counter

from django.conf import settings
from django.contrib.auth.models import User
from django.core.mail import send_mail
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Checks the OTP email setup and sends a test email.'

    def add_arguments(self, parser):
        parser.add_argument('to', nargs='?', help='Address to send the test email to.')

    def handle(self, *args, **options):
        out = self.stdout.write
        ok = self.style.SUCCESS
        bad = self.style.ERROR
        warn = self.style.WARNING

        out('── Email settings ──')
        out(f'  Backend : {settings.EMAIL_BACKEND}')
        out(f'  Host    : {settings.EMAIL_HOST}:{settings.EMAIL_PORT} (TLS={settings.EMAIL_USE_TLS})')
        out(f'  User    : {settings.EMAIL_HOST_USER or "(empty)"}')
        out(f'  Password: {"set (" + str(len(settings.EMAIL_HOST_PASSWORD)) + " chars)" if settings.EMAIL_HOST_PASSWORD else "(empty)"}')
        out(f'  From    : {settings.DEFAULT_FROM_EMAIL}')

        if settings.EMAIL_BACKEND.endswith('console.EmailBackend'):
            out(warn('\n  Console mode: emails are only printed, never sent.'))
            out(warn('  Add EMAIL_HOST_USER and EMAIL_HOST_PASSWORD to .env, then restart.'))
        elif len(settings.EMAIL_HOST_PASSWORD) != 16 and 'gmail' in settings.EMAIL_HOST:
            out(warn('\n  A Gmail App Password has exactly 16 characters. Yours does not.'))

        to = options.get('to') or settings.EMAIL_HOST_USER
        if to:
            out(f'\n── Sending test email to {to} ──')
            try:
                send_mail(
                    'PronounceMate test email',
                    'If you can read this, OTP emails will be delivered.',
                    settings.DEFAULT_FROM_EMAIL, [to], fail_silently=False,
                )
                out(ok('  Sent. Check the inbox (and the spam folder).'))
            except smtplib.SMTPAuthenticationError:
                out(bad('  Gmail rejected the login. Use a 16-character App Password '
                        '(2-Step Verification must be on), and the full Gmail address as EMAIL_HOST_USER.'))
            except Exception as exc:
                out(bad(f'  Failed: {exc!r}'))
        else:
            out(warn('\nPass an address to send a test:  python manage.py check_email you@gmail.com'))

        out('\n── Accounts ──')
        users = list(User.objects.all().order_by('id'))
        counts = Counter(u.email.strip().lower() for u in users if u.email.strip())
        for u in users:
            email = u.email.strip()
            notes = []
            if not email:
                notes.append('NO EMAIL - cannot receive codes')
            elif counts[email.lower()] > 1:
                notes.append('DUPLICATE email')
            if not u.is_active:
                notes.append('inactive (not verified yet)')
            line = f'  {u.username:<20} {email or "-":<35} {"; ".join(notes)}'
            out(bad(line) if notes else line)