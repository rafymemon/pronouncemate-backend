# whisper_asr.py
from faster_whisper import WhisperModel
import re

# Load ONCE at server startup inside AppConfig.ready()
# Do NOT load per-request — takes 3-5 seconds each time
_model = None

def get_model():
    global _model
    if _model is None:
        # 'base.en' for development, 'small.en' for production
        # compute_type='int8' = integer quantization: 2x faster, <0.5% accuracy drop
        _model = WhisperModel('small.en', device='cpu', compute_type='int8')
    return _model

def transcribe(audio_array, sr=16000) -> list:
    """
    Returns list of {word, start, end, prob} dicts.
    audio_array: float32 numpy array at 16kHz
    """
    model = get_model()
    segments, info = model.transcribe(
        audio_array,
        language='en',
        word_timestamps=True,          # CRITICAL for per-word slicing in Stage 5
        condition_on_previous_text=False,  # prevents hallucination carry-over
        temperature=0.0,               # greedy decoding — deterministic output
        vad_filter=True,               # skip silent segments automatically
    )
    words = []
    for seg in segments:
        if seg.words is None: continue
        for w in seg.words:
            word = re.sub(r"[^a-z'-]", '', w.word.strip().lower())
            if word:
                words.append({
                    'word':  word,
                    'start': round(w.start, 3),
                    'end':   round(w.end, 3),
                    'prob':  round(w.probability, 3)
                })
    return words