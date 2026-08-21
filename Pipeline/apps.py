# Pipeline/apps.py
from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)

class PipelineConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Pipeline'

    def ready(self):
        """
        Called once when Django server starts.
        Pre-loads all AI models into memory so first request is fast.
        """
        import os
        if os.environ.get('RUN_MAIN') != 'true' and not os.environ.get('LOAD_MODELS'):
            return

        try:
            logger.info('=== PronounceMate: Loading AI models ===')

            logger.info('Loading g2p-en and CMU Dict...')
            from Pipeline.g2p_module import g2p, cmu
            logger.info(f'  g2p-en loaded. CMU Dict: {len(cmu)} words.')

            logger.info('Loading Faster-Whisper (live + analyze)...')
            from Pipeline.whisper_asr import get_live_model, get_analyze_model
            get_live_model()      # tiny.en — used by the WS live-coloring path
            get_analyze_model()   # small.en — used by /api/analyze/
            logger.info('  Faster-Whisper (both models) loaded.')

            logger.info('Loading Wav2Vec2 phoneme model...')
            from Pipeline.wav2vec_phonemes import load_wav2vec
            load_wav2vec()
            logger.info('  Wav2Vec2 loaded.')

            logger.info('=== All models ready. Server accepting requests. ===')

        except Exception as e:
            logger.exception('Model loading failed — server will start but AI endpoints will error')