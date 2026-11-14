from faster_whisper import WhisperModel
import inspect
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

# ── Final-analysis (/api/analyze/) tuning ─────────────────────
# The reference script is known, so a word Whisper is unsure about is far
# more likely to be a real, accented word than a hallucination. Dropping it
# turns it into a "skipped" word (severity 15), so the analysis pass keeps
# lower-confidence words. A wrongly kept word just becomes an 'inserted'
# word (severity 0). The live path keeps its own, stricter defaults.
ANALYZE_MIN_WORD_PROB = 0.15
ANALYZE_MIN_SEGMENT_LOGPROB = -1.5   # whole segments below this are discarded
MAX_HOTWORDS = 20

# Which optional arguments the installed faster-whisper supports
# (`hotwords` only exists in newer releases).
_TRANSCRIBE_PARAMS = set(inspect.signature(WhisperModel.transcribe).parameters)


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


# ── Hotwords (names / acronyms from the script) ───────────────
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]*")
_NOT_NAMES = {"i", "i'm", "i've", "i'll", "i'd"}


def extract_hotwords(script_text: str, limit: int = MAX_HOTWORDS) -> list:
    """
    Pull names and acronyms out of the reference script so Whisper can be
    biased toward spelling them correctly: ALL-CAPS tokens ("MUET") and
    capitalized words that are not at the start of a sentence ("Ahmed",
    "Python", "Flutter"). Ordinary words are deliberately NOT included —
    biasing toward the whole script would make Whisper "complete" words the
    user never said and hide genuinely skipped words.
    """
    if not script_text:
        return []

    found = []
    seen = set()
    for m in _TOKEN_RE.finditer(script_text):
        tok = m.group(0)
        if tok.lower() in _NOT_NAMES:
            continue

        prefix = script_text[:m.start()]
        stripped = prefix.rstrip()
        sentence_initial = (
            not stripped
            or stripped[-1] in '.!?:'
            or '\n' in prefix[len(stripped):]
        )

        is_acronym = tok.isupper() and len(tok) > 1
        is_proper = tok[0].isupper() and not sentence_initial and not tok.isupper()

        if (is_acronym or is_proper) and tok.lower() not in seen:
            seen.add(tok.lower())
            found.append(tok)
            if len(found) >= limit:
                break

    return found


def transcribe(audio_array: np.ndarray, sr: int = 16000,
               min_word_prob: float = 0.5,
               min_word_duration: float = 0.05,
               model: WhisperModel | None = None,
               min_segment_logprob: float = -1.0,
               hotwords: list | None = None,
               log_drops: bool = False,
               vad_filter: bool = True) -> list:
    """
    Returns list of {word, start, end, prob} dicts.
    audio_array: float32 numpy array at 16kHz

    `model`: optionally pass an already-resolved model instance (e.g. from
    get_live_model() or get_analyze_model()) to control which model does
    the work. Defaults to the analyze model for backward compatibility
    with any existing callers that don't pass one.

    `min_segment_logprob`: segments whose average log-probability is below
    this are discarded whole (default -1.0 = previous behaviour).
    `hotwords`: optional list of names/acronyms to bias recognition toward.
    `log_drops`: print each word/segment that gets filtered out and why.

    `vad_filter`: faster-whisper's own internal Silero VAD, run BEFORE
    decoding, independent of any trimming audio_processor.py already did.
    It carries the same "there's a silence gap before speech" assumption
    as audio_processor.preprocess_audio's Step 2/Step 3 — on corpora with
    no real lead-in silence (e.g. SpeechOcean762), it may be misjudging
    where speech actually starts and swallowing part of the first word.
    HYPOTHESIS, being tested via transcribe_for_analysis's own
    `vad_filter` passthrough — not yet confirmed. Default True = original
    behaviour, so every existing caller is unaffected unless it opts in
    to vad_filter=False.
    """
    if model is None:
        model = get_analyze_model()

    wav_buffer = _numpy_to_wav_bytes(audio_array, sr)

    kwargs = dict(
        language='en',
        word_timestamps=True,
        condition_on_previous_text=True,
        temperature=0.0,
        vad_filter=vad_filter,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    if hotwords:
        joined = ', '.join(hotwords)
        if 'hotwords' in _TRANSCRIBE_PARAMS:
            kwargs['hotwords'] = joined
        elif 'initial_prompt' in _TRANSCRIBE_PARAMS:
            kwargs['initial_prompt'] = f'Glossary: {joined}.'

    segments, info = model.transcribe(wav_buffer, **kwargs)

    def _drop(word: str, reason: str) -> None:
        if log_drops:
            print(f"[Whisper] dropped '{word}': {reason}")

    words = []
    prev_word = None

    for seg in segments:
        if hasattr(seg, 'avg_logprob') and seg.avg_logprob < min_segment_logprob:
            if log_drops:
                text = getattr(seg, 'text', '').strip()
                print(f"[Whisper] dropped whole segment (avg_logprob={seg.avg_logprob:.2f} "
                      f"< {min_segment_logprob}): {text[:80]!r}")
            continue

        if seg.words is None:
            continue

        for w in seg.words:
            raw_word = w.word.strip().lower()
            word = re.sub(r"[^a-z']", '', raw_word)
            word = _normalize_contractions(word)

            if w.probability < min_word_prob:
                if word:
                    _drop(word, f"probability {w.probability:.2f} < {min_word_prob}")
                continue

            start = round(w.start, 3)
            end = round(w.end, 3)

            # ── Clamp against overlap with the previous word ──
            # Faster-Whisper's word-level timestamps come from a
            # DTW/cross-attention alignment and occasionally report a
            # word's start slightly BEFORE the previous word's end,
            # especially in fast/continuous speech. Left uncorrected,
            # downstream bucketing ends up including part of the previous
            # word's audio in this word's phoneme bucket.
            #
            # If simply moving this word's start to the previous word's end
            # would leave it too short to keep (typical for short function
            # words like "a"), split the overlap at its midpoint instead of
            # dropping the word.
            if prev_word and start < prev_word['end']:
                orig_start = start
                clamped_start = prev_word['end']
                if round(end - clamped_start, 3) < min_word_duration and round(end - orig_start, 3) >= min_word_duration:
                    mid = round((orig_start + prev_word['end']) / 2, 3)
                    if round(mid - prev_word['start'], 3) >= min_word_duration and round(end - mid, 3) >= min_word_duration:
                        prev_word['end'] = mid
                        prev_word['duration'] = round(mid - prev_word['start'], 3)
                        start = mid
                    else:
                        start = clamped_start
                else:
                    start = clamped_start

            duration = round(end - start, 3)
            if duration < min_word_duration:
                if word:
                    _drop(word, f"duration {duration:.3f}s < {min_word_duration}s after overlap clamp")
                continue

            if not word or len(word) < 1:
                continue

            if prev_word and word == prev_word['word']:
                prev_word['end'] = end
                prev_word['prob'] = max(prev_word['prob'], round(w.probability, 3))
                continue

            word_dict = {
                'word':  word,
                'start': start,
                'end':   end,
                'prob':  round(w.probability, 3),
                'duration': round(duration, 3)
            }
            words.append(word_dict)
            prev_word = word_dict

    return words


def transcribe_for_analysis(audio_array: np.ndarray, sr: int = 16000,
                            reference_script: str | None = None,
                            vad_filter: bool = True) -> list:
    """
    Transcription for the final /api/analyze/ pass: analyze model, lower
    confidence thresholds (see ANALYZE_* constants), names/acronyms from the
    reference script passed as hotwords, and every dropped word/segment
    logged so a "skipped" word can be traced to its cause.

    `vad_filter`: passthrough to transcribe()'s own vad_filter — see its
    docstring. pipeline.run_pipeline sets this to `not skip_denoise`, so
    benchmark-corpus runs (skip_denoise=True) also get Whisper's internal
    VAD disabled, since both flags are testing the same "no real lead-in
    silence" condition. Live app calls (skip_denoise=False, the default)
    are completely unaffected — vad_filter stays True there.
    """
    hotwords = extract_hotwords(reference_script) if reference_script else None
    if hotwords:
        print(f"[Whisper] hotwords: {hotwords}")

    return transcribe(
        audio_array, sr,
        min_word_prob=ANALYZE_MIN_WORD_PROB,
        min_word_duration=0.05,
        model=get_analyze_model(),
        min_segment_logprob=ANALYZE_MIN_SEGMENT_LOGPROB,
        hotwords=hotwords,
        log_drops=True,
        vad_filter=vad_filter,
    )


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