from faster_whisper import WhisperModel
import re
import io
import numpy as np
from scipy.io import wavfile

# Load ONCE at server startup
_model = None

def get_model(model_size='small.en', device='cpu', compute_type='int8'):
    global _model
    if _model is None:
        # Fallback compute types if int8 fails
        for ct in [compute_type, 'float16', 'float32']:
            try:
                print(f"[Whisper] Loading {model_size} on {device} with {ct}...")
                _model = WhisperModel(model_size, device=device, compute_type=ct)
                print(f"[Whisper] Model loaded successfully with {ct}")
                break
            except Exception as e:
                print(f"[Whisper] Failed with {ct}: {e}")
                continue
        if _model is None:
            raise RuntimeError("Failed to load Whisper model with any compute type")
    return _model


def _numpy_to_wav_bytes(audio_array: np.ndarray, sr: int = 16000) -> io.BytesIO:
    """Convert float32 numpy array to WAV bytes for faster_whisper."""
    # Ensure correct range and dtype
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    
    # Clip to [-1, 1] to prevent distortion
    audio_array = np.clip(audio_array, -1.0, 1.0)
    
    # Convert to int16 for WAV
    audio_int16 = (audio_array * 32767).astype(np.int16)
    
    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sr, audio_int16)
    wav_buffer.seek(0)
    return wav_buffer


def transcribe(audio_array: np.ndarray, sr: int = 16000,
               min_word_prob: float = 0.5,
               min_word_duration: float = 0.05) -> list:
    """
    Returns list of {word, start, end, prob} dicts.
    audio_array: float32 numpy array at 16kHz
    """
    model = get_model()
    
    # Convert numpy to WAV bytes
    wav_buffer = _numpy_to_wav_bytes(audio_array, sr)
    
    segments, info = model.transcribe(
        wav_buffer,
        language='en',
        word_timestamps=True,
        condition_on_previous_text=True,  # Better for connected speech
        temperature=0.0,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),  # Configurable VAD
    )
    
    words = []
    prev_word = None
    
    for seg in segments:
        # Skip low-confidence segments
        if hasattr(seg, 'avg_logprob') and seg.avg_logprob < -1.0:
            continue
            
        if seg.words is None:
            continue
            
        for w in seg.words:
            # Skip very low-probability words (likely hallucinations)
            if w.probability < min_word_prob:
                continue
            
            # Skip very short words (likely noise)
            duration = w.end - w.start
            if duration < min_word_duration:
                continue
            
            # Clean word: lowercase, strip punctuation except apostrophe
            raw_word = w.word.strip().lower()
            word = re.sub(r"[^a-z']", '', raw_word)
            
            # Handle contractions consistently with G2P module
            # Expand common contractions to match g2p_module expectations
            word = _normalize_contractions(word)
            
            if not word or len(word) < 1:
                continue
            
            # Deduplicate consecutive identical words (Whisper repetition bug)
            if prev_word and word == prev_word['word']:
                # Merge timing instead of adding duplicate
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


def _normalize_contractions(word: str) -> str:
    """
    Normalize contractions to match g2p_module.py expectations.
    g2p_module expands contractions to base words.
    """
    contractions = {
        "won't": "will",
        "can't": "can",
        "don't": "do",
        "didn't": "did",
        "wouldn't": "would",
        "couldn't": "could",
        "shouldn't": "should",
        "isn't": "is",
        "aren't": "are",
        "wasn't": "was",
        "weren't": "were",
        "hasn't": "has",
        "haven't": "have",
        "hadn't": "had",
        "i'm": "i",
        "you're": "you",
        "he's": "he",
        "she's": "she",
        "it's": "it",
        "we're": "we",
        "they're": "they",
        "i've": "i",
        "you've": "you",
        "we've": "we",
        "they've": "they",
        "i'll": "i",
        "you'll": "you",
        "he'll": "he",
        "she'll": "she",
        "it'll": "it",
        "we'll": "we",
        "they'll": "they",
        "i'd": "i",
        "you'd": "you",
        "he'd": "he",
        "she'd": "she",
        "it'd": "it",
        "we'd": "we",
        "they'd": "they",
        "let's": "let",
        "that's": "that",
        "there's": "there",
        "what's": "what",
        "who's": "who",
        "where's": "where",
        "when's": "when",
        "why's": "why",
        "how's": "how",
    }
    
    return contractions.get(word, word)


def transcribe_with_fallback(audio_array: np.ndarray, sr: int = 16000) -> list:
    """
    Try primary model, fallback to base if small fails.
    """
    try:
        return transcribe(audio_array, sr)
    except Exception as e:
        print(f"[Whisper] Primary transcription failed: {e}")
        print("[Whisper] Trying fallback with base model...")
        global _model
        _model = None  # Force reload
        model = get_model(model_size='base.en', compute_type='float32')
        return transcribe(audio_array, sr)