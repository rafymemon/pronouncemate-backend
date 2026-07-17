import os
import hashlib
import re
from typing import Optional
from gtts import gTTS

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
    
    # Sanitize filename
    safe_word = re.sub(r'[^\w]', '_', word.strip().lower())
    key = hashlib.md5(f'{word}{lang}{tld}{slow}'.encode()).hexdigest()[:8]
    filename = f'{safe_word}_{key}.mp3'
    path = os.path.join(CACHE_DIR, filename)
    
    # Return cached file if exists
    if os.path.exists(path):
        return path
    
    # Generate new audio with error handling
    try:
        tts = gTTS(text=word, lang=lang, tld=tld, slow=slow)
        tts.save(path)
        return path
    except Exception as e:
        print(f"[Feedback] gTTS failed for '{word}': {e}")
        return None


def get_feedback_audio_batch(words: list, lang: str = 'en',
                             tld: str = 'com', slow: bool = True) -> dict:
    """
    Pre-generate audio for multiple words. Useful for warm-up/caching.
    Returns dict of {word: path_or_None}.
    """
    results = {}
    for word in words:
        results[word] = get_feedback_audio(word, lang, tld, slow)
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