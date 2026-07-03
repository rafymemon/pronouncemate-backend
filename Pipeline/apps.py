# pipeline/apps.py
from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)

class PipelineConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'pipeline'

    def ready(self):
        """
        Called once when Django server starts.
        Pre-loads all AI models into memory so first request is fast.
        """
        import os
        # Don't load models during manage.py migrate, makemigrations etc.
        # Only load when actual server is running
        if os.environ.get('RUN_MAIN') != 'true' and not os.environ.get('LOAD_MODELS'):
            return

        try:
            logger.info('=== PronounceMate: Loading AI models ===')

            logger.info('Loading g2p-en and CMU Dict...')
            from Pipeline.g2p_module import g2p, cmu
            logger.info(f'  g2p-en loaded. CMU Dict: {len(cmu)} words.')

            logger.info('Loading Faster-Whisper...')
            from Pipeline.whisper_asr import get_model
            get_model()  # triggers download if not cached
            logger.info('  Faster-Whisper loaded.')

            logger.info('Loading Wav2Vec2 phoneme model...')
            from Pipeline.wav2vec_phonemes import load_wav2vec
            load_wav2vec()  # triggers download if not cached (~1.2GB first time)
            logger.info('  Wav2Vec2 loaded.')

            logger.info('=== All models ready. Server accepting requests. ===')

        except Exception as e:
            logger.error(f'Model loading failed: {e}')
            logger.error('Server will start but /api/analyze/ will return errors')