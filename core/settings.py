# core/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Security ───────────────────────────────────────────────────
SECRET_KEY = os.getenv('SECRET_KEY', 'fallback-insecure-key')
DEBUG = os.getenv('DEBUG', 'True') == 'True'
ALLOWED_HOSTS = ['*']

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
}

# ── CORS (allows Flutter app to call this API) ──────────────────
CORS_ALLOWED_ORIGINS = [
    'http://localhost:3000',
    'http://127.0.0.1:8000',
    'http://10.0.2.2:8000',  # Android emulator -> host machine
]
CORS_ALLOW_ALL_ORIGINS = DEBUG  # Allow all origins in development

# ── Media Files (gTTS audio, uploaded WAV files) ────────────────
MEDIA_ROOT = BASE_DIR / os.getenv('MEDIA_ROOT', 'media')
MEDIA_URL  = os.getenv('MEDIA_URL', '/media/')

# ── Static Files ────────────────────────────────────────────────
STATIC_URL  = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

# ── Password Validation ─────────────────────────────────────────
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
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
        'pipeline': {'handlers': ['console'], 'level': 'INFO', 'propagate': False},
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