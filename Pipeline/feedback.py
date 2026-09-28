import os
import hashlib
import re
import time
import uuid
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Optional
from gtts import gTTS, gTTSError

# ── Templates ─────────────────────────────────────────────────
TEMPLATES = {
    'correct':      "Good job! '{word}' was pronounced correctly.",
    'substitution': "You said '{user_said}' but the correct word is '{word}'.",
    'omission':     "You dropped one or more sounds in '{word}'.",
    'insertion':    "You added an extra sound in '{word}'.",
    'skipped':      "You skipped the word '{word}'. Please include it.",
    'stress':       "The word '{word}' has incorrect stress placement.",
}

# ── Phoneme-level hints (ARPAbet → IPA mapping included) ──────
# Keys: (expected_phoneme, user_phoneme) in normalized form
# Supports both ARPAbet and IPA representations
PK_HINTS = {
    # TH sounds (ARPAbet and IPA)
    ('th', 't'):  "Tip: Place your tongue between your teeth for the 'th' sound.",
    ('th', 'f'):  "Tip: The 'th' sound uses the tongue tip, not the lips.",
    ('θ',  't'):  "Tip: Place your tongue between your teeth for the 'th' sound.",
    ('θ',  'f'):  "Tip: The 'th' sound uses the tongue tip, not the lips.",
    ('dh', 'd'):  "Tip: For 'th' in 'this', touch tongue to upper teeth.",
    ('dh', 'z'):  "Tip: For 'th' in 'this', touch tongue to upper teeth.",
    ('ð',  'd'):  "Tip: For 'th' in 'this', touch tongue to upper teeth.",
    ('ð',  'z'):  "Tip: For 'th' in 'this', touch tongue to upper teeth.",

    # W/V confusion
    ('w',  'v'):  "Tip: Round your lips fully for the 'w' sound.",
    ('v',  'w'):  "Tip: Touch upper teeth to lower lip for 'v'.",

    # Vowel length
    ('ih', 'iy'): "Tip: The vowel in 'ship' is shorter than in 'sheep'.",
    ('iy', 'ih'): "Tip: The vowel in 'sheep' is longer than in 'ship'.",
    ('ɪ',  'i'):  "Tip: The vowel in 'ship' is shorter than in 'sheep'.",
    ('i',  'ɪ'):  "Tip: The vowel in 'sheep' is longer than in 'ship'.",

    # R/L confusion (common in some accents)
    ('r',  'l'):  "Tip: Curl your tongue back for 'r', touch the roof for 'l'.",
    ('l',  'r'):  "Tip: Touch the roof of your mouth for 'l'.",

    # P/B confusion (unaspirated vs voiced)
    ('p',  'b'):  "Tip: 'p' is voiceless — no vibration in your throat.",
    ('b',  'p'):  "Tip: 'b' is voiced — feel vibration in your throat.",
}

# ── Audio Cache ───────────────────────────────────────────────
CACHE_DIR = 'media/feedback/'
os.makedirs(CACHE_DIR, exist_ok=True)

# ── TTS performance / resilience settings ─────────────────────
# Every error word needs a gTTS network call the first time it is seen.
# Doing those one after another made /api/analyze/ take minutes on a
# session with many flagged words, so generation is now:
#   - deduplicated (each distinct word is fetched once per batch),
#   - run in parallel (TTS_MAX_WORKERS threads),
#   - bounded (TTS_BATCH_TIMEOUT seconds for a whole batch, and a per-request
#     HTTP timeout when the installed gTTS supports one),
#   - fail-fast: after a network-type failure, TTS is skipped for
#     TTS_COOLDOWN_SECONDS instead of every remaining word waiting to fail.
# Words that miss the deadline simply come back with audio_path = None and
# get their audio on a later request (the cache is keyed by word).
TTS_MAX_WORKERS = 6
TTS_BATCH_TIMEOUT = 45.0
TTS_COOLDOWN_SECONDS = 60.0
TTS_HTTP_TIMEOUT = 8.0

# Older gTTS releases have no `timeout` argument; only pass it if supported.
_GTTS_SUPPORTS_TIMEOUT = 'timeout' in inspect.signature(gTTS.__init__).parameters

_tts_disabled_until = 0.0
_tts_lock = threading.Lock()


def _tts_available() -> bool:
    return time.monotonic() >= _tts_disabled_until


def _trip_tts_breaker() -> None:
    """Pause TTS generation after a network-type failure."""
    global _tts_disabled_until
    with _tts_lock:
        _tts_disabled_until = time.monotonic() + TTS_COOLDOWN_SECONDS


def generate_explanation(error: dict) -> str:
    """
    Generate human-readable feedback for a pronunciation error.

    Args:
        error: dict with keys: type, word, ref_phon, pred_phon,
               ref_phon_norm, pred_phon_norm, editops

    Returns:
        Formatted explanation string with phoneme details and hints.
    """
    err_type = error.get('type', 'unknown')
    word = error.get('word', 'unknown')

    # Get what user actually said (word, not phoneme)
    pred_phon = error.get('pred_phon', [])
    user_said = word  # Default: assume they said the word (but wrong)

    # If we have phonemes, try to reconstruct or use first as hint
    if pred_phon:
        user_said = ' '.join(pred_phon) if len(pred_phon) > 1 else pred_phon[0]

    # Build base message from template
    template = TEMPLATES.get(err_type, "Check your pronunciation of '{word}'.")
    try:
        base = template.format(word=word, user_said=user_said)
    except KeyError:
        base = f"Check your pronunciation of '{word}'."

    # Find phoneme-level hint from editops
    hint = _find_phoneme_hint(error)

    # Build phoneme comparison text
    ref_phon = error.get('ref_phon', [])
    phon_text = ""
    if ref_phon and pred_phon:
        phon_text = (f"Expected phonemes: {' '.join(ref_phon)}. "
                     f"You produced: {' '.join(pred_phon)}. ")
    elif ref_phon:
        phon_text = f"Expected phonemes: {' '.join(ref_phon)}. "

    # Build per-phoneme breakdown for UI
    breakdown = _build_phoneme_breakdown(error)

    explanation = base
    if phon_text:
        explanation += ' ' + phon_text
    if hint:
        explanation += ' ' + hint
    if breakdown:
        explanation += ' ' + breakdown

    return explanation.strip()


def _find_phoneme_hint(error: dict) -> str:
    """
    Find a relevant pronunciation hint based on the error's editops.
    """
    editops = error.get('editops', [])
    ref_norm = error.get('ref_phon_norm', [])
    pred_norm = error.get('pred_phon_norm', [])

    if not editops or not ref_norm or not pred_norm:
        return ""

    for op, i, j in editops:
        if op == 'replace':
            # Guard against out-of-range indices
            if i < len(ref_norm) and j < len(pred_norm):
                expected = ref_norm[i]
                actual = pred_norm[j]

                # Try direct match
                key = (expected, actual)
                if key in PK_HINTS:
                    return PK_HINTS[key]

                # Try reverse match
                key_rev = (actual, expected)
                if key_rev in PK_HINTS:
                    return PK_HINTS[key_rev]

    return ""


def _build_phoneme_breakdown(error: dict) -> str:
    """
    Build a detailed phoneme-by-phoneme comparison string.
    Example: "Expected: DH-AH0 | Got: D-AH0"
    """
    ref_phon = error.get('ref_phon', [])
    pred_phon = error.get('pred_phon', [])
    editops = error.get('editops', [])
    ref_norm = error.get('ref_phon_norm', [])
    pred_norm = error.get('pred_phon_norm', [])

    if not ref_phon:
        return ""

    # Simple format: show expected vs got
    expected_str = '-'.join(ref_phon)
    got_str = '-'.join(pred_phon) if pred_phon else "(none)"

    return f"Breakdown: [{expected_str}] vs [{got_str}]"


def _cache_path(word: str, lang: str, tld: str, slow: bool) -> str:
    """Deterministic cache file path for a word/voice combination."""
    safe_word = re.sub(r'[^\w]', '_', word.strip().lower())
    key = hashlib.md5(f'{word}{lang}{tld}{slow}'.encode()).hexdigest()[:8]
    return os.path.join(CACHE_DIR, f'{safe_word}_{key}.mp3')


def get_feedback_audio(word: str, lang: str = 'en',
                       tld: str = 'com', slow: bool = True) -> Optional[str]:
    """
    Generate or retrieve cached TTS audio for a word.

    Args:
        word: the word to speak
        lang: language code
        tld: top-level domain for accent ('com'=American, 'co.uk'=British)
        slow: speak slowly for clarity

    Returns:
        Path to MP3 file, or None if generation failed.
    """
    if not word or not word.strip():
        return None

    path = _cache_path(word, lang, tld, slow)

    # Return cached file if exists
    if os.path.exists(path):
        return path

    # Recent network failure: don't make every remaining word wait to fail too
    if not _tts_available():
        return None

    # Write to a temp file first so an interrupted download can never leave a
    # truncated MP3 that would later be served as a "cached" file.
    tmp_path = f'{path}.{uuid.uuid4().hex[:8]}.part'

    try:
        kwargs = {'text': word, 'lang': lang, 'tld': tld, 'slow': slow}
        if _GTTS_SUPPORTS_TIMEOUT:
            kwargs['timeout'] = TTS_HTTP_TIMEOUT
        tts = gTTS(**kwargs)
        tts.save(tmp_path)
        os.replace(tmp_path, path)
        return path
    except (gTTSError, OSError) as e:
        # Network / filesystem-type failure: pause TTS briefly.
        _trip_tts_breaker()
        print(f"[Feedback] gTTS failed for '{word}': {e} "
              f"(pausing TTS for {TTS_COOLDOWN_SECONDS:.0f}s)")
        return None
    except Exception as e:
        # Word-specific failure (e.g. nothing speakable): don't pause TTS.
        print(f"[Feedback] gTTS failed for '{word}': {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def get_feedback_audio_batch(words: list, lang: str = 'en',
                             tld: str = 'com', slow: bool = True,
                             max_workers: int = TTS_MAX_WORKERS,
                             timeout: float = TTS_BATCH_TIMEOUT) -> dict:
    """
    Generate audio for multiple words: deduplicated, in parallel, and bounded
    by `timeout` seconds overall. Cached words are resolved immediately and
    never touch the network.

    Returns dict of {word: path_or_None}. Words that fail or miss the
    deadline map to None.
    """
    results = {w: None for w in words}
    unique = list(dict.fromkeys(w for w in words if w and w.strip()))

    to_generate = []
    for w in unique:
        path = _cache_path(w, lang, tld, slow)
        if os.path.exists(path):
            results[w] = path
        else:
            to_generate.append(w)

    if not to_generate or not _tts_available():
        return results

    executor = ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(to_generate))))
    futures = {executor.submit(get_feedback_audio, w, lang, tld, slow): w
               for w in to_generate}
    done, not_done = wait(futures, timeout=timeout)

    for f in done:
        try:
            results[futures[f]] = f.result()
        except Exception as e:
            print(f"[Feedback] TTS worker error for '{futures[f]}': {e}")

    if not_done:
        print(f"[Feedback] {len(not_done)} TTS request(s) missed the "
              f"{timeout:.0f}s deadline; returning without audio for them")

    # Don't block the request on stragglers.
    executor.shutdown(wait=False, cancel_futures=True)
    return results


def clear_audio_cache():
    """Clear all cached feedback audio files."""
    import glob
    files = glob.glob(os.path.join(CACHE_DIR, '*.mp3'))
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"[Feedback] Failed to remove {f}: {e}")
    return len(files)


def get_word_feedback_audio_url(word: str, lang: str = 'en') -> Optional[str]:
    """
    Generate TTS audio for a word and return the HTTP URL path.
    e.g., '/media/feedback/hello_abc123.mp3'
    """
    local_path = get_feedback_audio(word, lang=lang)
    if not local_path:
        return None
    # Convert filesystem path to URL path
    return '/' + local_path.replace('\\', '/')