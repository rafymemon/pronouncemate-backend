from faster_whisper import WhisperModel
import re
import io
import numpy as np
from scipy.io import wavfile

# Cache of loaded models, keyed by model_size (e.g. 'tiny.en', 'small.en').
# Each size is loaded ONCE and reused — this is what makes both the live
# and final-analysis paths fast after the first call for each size.
_models: dict[str, WhisperModel] = {}

LIVE_MODEL_SIZE = 'tiny.en'      # fast, low-latency — used for live word coloring
ANALYZE_MODEL_SIZE = 'small.en'  # slower, more accurate — used for final scoring


def get_model(model_size: str = ANALYZE_MODEL_SIZE, device: str = 'cpu', compute_type: str = 'int8'):
    """
    Returns a cached WhisperModel for the given size, loading it on first
    use. Different sizes are cached independently, so calling this with
    'tiny.en' and 'small.en' will end up with two models resident in
    memory at once — that's intentional (see LIVE_MODEL_SIZE / ANALYZE_MODEL_SIZE).
    """
    if model_size in _models:
        return _models[model_size]

    model = None
    for ct in [compute_type, 'float16', 'float32']:
        try:
            print(f"[Whisper] Loading {model_size} on {device} with {ct}...")
            model = WhisperModel(model_size, device=device, compute_type=ct)
            print(f"[Whisper] Model '{model_size}' loaded successfully with {ct}")
            break
        except Exception as e:
            print(f"[Whisper] Failed loading '{model_size}' with {ct}: {e}")
            continue

    if model is None:
        raise RuntimeError(f"Failed to load Whisper model '{model_size}' with any compute type")

    _models[model_size] = model
    return model


def get_live_model():
    """Small, fast model dedicated to the live WebSocket coloring path."""
    return get_model(LIVE_MODEL_SIZE)


def get_analyze_model():
    """Larger, more accurate model dedicated to the final /api/analyze/ pass."""
    return get_model(ANALYZE_MODEL_SIZE)


def _numpy_to_wav_bytes(audio_array: np.ndarray, sr: int = 16000) -> io.BytesIO:
    """Convert float32 numpy array to WAV bytes for faster_whisper."""
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)

    audio_array = np.clip(audio_array, -1.0, 1.0)
    audio_int16 = (audio_array * 32767).astype(np.int16)

    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sr, audio_int16)
    wav_buffer.seek(0)
    return wav_buffer


def transcribe(audio_array: np.ndarray, sr: int = 16000,
               min_word_prob: float = 0.5,
               min_word_duration: float = 0.05,
               model: WhisperModel | None = None) -> list:
    """
    Returns list of {word, start, end, prob} dicts.
    audio_array: float32 numpy array at 16kHz

    `model`: optionally pass an already-resolved model instance (e.g. from
    get_live_model() or get_analyze_model()) to control which model does
    the work. Defaults to the analyze model for backward compatibility
    with any existing callers that don't pass one.
    """
    if model is None:
        model = get_analyze_model()

    wav_buffer = _numpy_to_wav_bytes(audio_array, sr)

    segments, info = model.transcribe(
        wav_buffer,
        language='en',
        word_timestamps=True,
        condition_on_previous_text=True,
        temperature=0.0,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    words = []
    prev_word = None

    for seg in segments:
        if hasattr(seg, 'avg_logprob') and seg.avg_logprob < -1.0:
            continue

        if seg.words is None:
            continue

        for w in seg.words:
            if w.probability < min_word_prob:
                continue

            duration = w.end - w.start
            if duration < min_word_duration:
                continue

            raw_word = w.word.strip().lower()
            word = re.sub(r"[^a-z']", '', raw_word)
            word = _normalize_contractions(word)

            if not word or len(word) < 1:
                continue

            if prev_word and word == prev_word['word']:
                prev_word['end'] = round(w.end, 3)
                prev_word['prob'] = max(prev_word['prob'], round(w.probability, 3))
                continue

            word_dict = {
                'word':  word,
                'start': round(w.start, 3),
                'end':   round(w.end, 3),
                'prob':  round(w.probability, 3),
                'duration': round(duration, 3)
            }
            words.append(word_dict)
            prev_word = word_dict

    return words


def transcribe_live(audio_array: np.ndarray, sr: int = 16000,
                     min_word_prob: float = 0.3,
                     min_word_duration: float = 0.02) -> list:
    """
    Convenience wrapper for the live WebSocket path: always uses the
    small/fast LIVE_MODEL_SIZE model, so consumers.py doesn't need to
    know about model selection at all.
    """
    return transcribe(
        audio_array, sr,
        min_word_prob=min_word_prob,
        min_word_duration=min_word_duration,
        model=get_live_model(),
    )


def _normalize_contractions(word: str) -> str:
    """
    Normalize contractions to match g2p_module.py expectations.
    g2p_module expands contractions to base words.
    """
    contractions = {
        "won't": "will", "can't": "can", "don't": "do", "didn't": "did",
        "wouldn't": "would", "couldn't": "could", "shouldn't": "should",
        "isn't": "is", "aren't": "are", "wasn't": "was", "weren't": "were",
        "hasn't": "has", "haven't": "have", "hadn't": "had",
        "i'm": "i", "you're": "you", "he's": "he", "she's": "she", "it's": "it",
        "we're": "we", "they're": "they",
        "i've": "i", "you've": "you", "we've": "we", "they've": "they",
        "i'll": "i", "you'll": "you", "he'll": "he", "she'll": "she",
        "it'll": "it", "we'll": "we", "they'll": "they",
        "i'd": "i", "you'd": "you", "he'd": "he", "she'd": "she",
        "it'd": "it", "we'd": "we", "they'd": "they",
        "let's": "let", "that's": "that", "there's": "there", "what's": "what",
        "who's": "who", "where's": "where", "when's": "when", "why's": "why",
        "how's": "how",
    }
    return contractions.get(word, word)


def transcribe_with_fallback(audio_array: np.ndarray, sr: int = 16000) -> list:
    """
    Try primary (analyze) model, fallback to base if it fails.
    """
    try:
        return transcribe(audio_array, sr)
    except Exception as e:
        print(f"[Whisper] Primary transcription failed: {e}")
        print("[Whisper] Trying fallback with base model...")
        _models.pop(ANALYZE_MODEL_SIZE, None)  # force reload of just the analyze model
        fallback_model = get_model(model_size='base.en', compute_type='float32')
        return transcribe(audio_array, sr, model=fallback_model)