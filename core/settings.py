# core/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / '.env')  # Load variables from .env (project root, next to manage.py)

# ── Security ───────────────────────────────────────────────────
SECRET_KEY = os.getenv('SECRET_KEY', 'fallback-insecure-key')
DEBUG = os.getenv('DEBUG', 'True') == 'True'
ALLOWED_HOSTS = ['*']

# OTP hashes are keyed with SECRET_KEY, so never run production on the fallback.
if not DEBUG and SECRET_KEY == 'fallback-insecure-key':
    raise RuntimeError('Set SECRET_KEY in the environment when DEBUG is False.')

# ── Installed Apps ─────────────────────────────────────────────

INSTALLED_APPS = [
    'daphne',                     # ← ADD, must be first
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'channels',                   # ← ADD
    'rest_framework',
    'rest_framework.authtoken',
    'corsheaders',
    'api.apps.ApiConfig',
    'Pipeline.apps.PipelineConfig',
]

# ── Middleware ──────────────────────────────────────────────────
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # MUST be first
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'core.urls'

TEMPLATES = [{'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [], 'APP_DIRS': True, 'OPTIONS': {'context_processors': [
    'django.template.context_processors.debug',
    'django.template.context_processors.request',
    'django.contrib.auth.context_processors.auth',
    'django.contrib.messages.context_processors.messages',
]}}]

ASGI_APPLICATION = 'core.asgi.application'

# ── Database (PostgreSQL) ───────────────────────────────────────
DATABASES = {
    'default': {
        'ENGINE':   'django.db.backends.postgresql',
        'NAME':     os.getenv('DB_NAME',     'pronouncemate_db'),
        'USER':     os.getenv('DB_USER',     'pm_user'),
        'PASSWORD': os.getenv('DB_PASSWORD', ''),
        'HOST':     os.getenv('DB_HOST',     'localhost'),
        'PORT':     os.getenv('DB_PORT',     '5432'),
    }
}

# ── Django REST Framework ───────────────────────────────────────
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.TokenAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.MultiPartParser',
        'rest_framework.parsers.JSONParser',
    ],
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    # Only views that set throttle_scope (the auth endpoints) are throttled.
    # Limits are per client IP for anonymous requests.
    'DEFAULT_THROTTLE_RATES': {
        'auth_login':    '10/min',
        'auth_register': '20/hour',
        'auth_otp':      '15/min',
        'auth_email':    '20/hour',
    },
}

# ── CORS (allows Flutter app to call this API) ──────────────────
CORS_ALLOWED_ORIGINS = [
    'http://localhost:3000',
    'http://127.0.0.1:8000',
    'http://10.0.2.2:8000',  # Android emulator -> host machine
]
CORS_ALLOW_ALL_ORIGINS = DEBUG  # Allow all origins in development

# ── Email (OTP codes) ───────────────────────────────────────────
# Put these two lines in .env to send REAL emails (Gmail App Password):
#   EMAIL_HOST_USER=yourgmail@gmail.com
#   EMAIL_HOST_PASSWORD=<16-character Google App Password>
# Without them, emails are only PRINTED in the server terminal.
EMAIL_HOST          = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT          = int(os.getenv('EMAIL_PORT', '587'))
EMAIL_USE_TLS       = os.getenv('EMAIL_USE_TLS', 'True') == 'True'
EMAIL_HOST_USER     = os.getenv('EMAIL_HOST_USER', '').strip()
EMAIL_HOST_PASSWORD = os.getenv('EMAIL_HOST_PASSWORD', '').replace(' ', '')  # Google shows it with spaces
EMAIL_TIMEOUT       = 15
DEFAULT_FROM_EMAIL  = os.getenv(
    'DEFAULT_FROM_EMAIL',
    f'PronounceMate <{EMAIL_HOST_USER}>' if EMAIL_HOST_USER
    else 'PronounceMate <noreply@pronouncemate.local>',
)

_smtp_ready = bool(EMAIL_HOST_USER and EMAIL_HOST_PASSWORD)
EMAIL_BACKEND = os.getenv('EMAIL_BACKEND') or (
    'django.core.mail.backends.smtp.EmailBackend' if _smtp_ready
    else 'django.core.mail.backends.console.EmailBackend'
)

# Say clearly, at every startup, how OTP emails will be delivered.
if EMAIL_BACKEND.endswith('console.EmailBackend'):
    print('⚠️  EMAIL: console mode - OTP emails are only PRINTED in this terminal, '
          'NOT sent. Set EMAIL_HOST_USER and EMAIL_HOST_PASSWORD in .env.')
else:
    print(f'📧 EMAIL: sending via {EMAIL_HOST}:{EMAIL_PORT} as {EMAIL_HOST_USER or "(no user)"}')

# ── Media Files (gTTS audio, uploaded WAV files) ────────────────
MEDIA_ROOT = BASE_DIR / os.getenv('MEDIA_ROOT', 'media')
MEDIA_URL  = os.getenv('MEDIA_URL', '/media/')

# ── Static Files ────────────────────────────────────────────────
STATIC_URL  = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

# ── Password Validation ─────────────────────────────────────────
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
     'OPTIONS': {'min_length': 8}},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# ── Localisation ────────────────────────────────────────────────
LANGUAGE_CODE = 'en-us'
TIME_ZONE     = 'Asia/Karachi'
USE_I18N      = True
USE_TZ        = True

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ── Logging ─────────────────────────────────────────────────────
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {'console': {'class': 'logging.StreamHandler'}},
    'root': {'handlers': ['console'], 'level': 'INFO'},
    'loggers': {
        'Pipeline': {'handlers': ['console'], 'level': 'INFO', 'propagate': False},
    },
}

# ── Whisper Model Selection ──────────────────────────────────────
WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'base.en')


# ← ADD: Channels layer (in-memory, no Redis needed for dev/FYP demo)
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels.layers.InMemoryChannelLayer',
    }
}


# GEMINI_API_KEY: os.getenv() takes the NAME of an env var to look up, not
# the key value itself. The previous version passed the raw key string as
# the lookup name, so this always evaluated to None. Put the real key in
# .env as GEMINI_API_KEY=... — never here.
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')