# error_detector.py
import Levenshtein
import re
from .g2p_module import build_answer_key
from .wav2vec_phonemes import recognize_phonemes
from .audio_processor import slice_word_audio

# ── Normalization: strip stress digits and lowercase ──
def normalize(phonemes: list) -> list:
    """
    Convert ARPAbet or IPA to a comparable form.
    'AE1' -> 'ae',  'TH' -> 'th',  'k' -> 'k'
    Stress digits are stripped because Wav2Vec2 does not output them.
    """
    return [re.sub(r'[012]', '', p).lower() for p in phonemes if p.strip()]

# ── Layer 1 threshold ──
# Words with normalized Levenshtein distance above this go to Layer 2.
# Start at 0.0 (all words go to Layer 2) during testing.
# Calibrate using calibrate.py (Section 11) to find the best value.
L1_THRESHOLD = 0.15

# ── Minimum clip duration for Wav2Vec2 ──
MIN_CLIP_DURATION = 0.15  # seconds  (FIX 7: guard added)

def detect_errors(alignment: list, audio, sr: int,
                  answer_key: dict) -> list:
    """
    For each aligned (spoken_word, script_word) pair:
      1. Run Layer 1 (text-level Levenshtein on G2P phonemes)
      2. If Layer 1 flags word, run Layer 2 (Wav2Vec2 on audio clip)
      3. Classify error type from Layer 2 editops backtrace
    Returns list of error dicts.
    """
    results = []

    for spoken_word, script_word in alignment:

        # ── Skipped word ──
        if spoken_word is None:
            results.append({
                'word': script_word, 'type': 'skipped',
                'severity': SEVERITY['skipped'],
                'ref_phon': answer_key.get(script_word, []),
                'pred_phon': []
            })
            continue

        # ── Inserted word (user added, not in script) ──
        if script_word is None:
            results.append({'word': spoken_word, 'type': 'inserted', 'severity': 0})
            continue

        ref_phon  = answer_key.get(script_word, [])
        ref_norm  = normalize(ref_phon)

        # ── LAYER 1: Text-level check ──
        # G2P on what Whisper heard (text, not audio)
        from .g2p_module import build_answer_key as g2p_word
        user_key  = g2p_word(spoken_word)
        user_phon_text = user_key.get(spoken_word, ref_phon)
        user_norm_text = normalize(user_phon_text)

        dist_l1 = Levenshtein.distance(ref_norm, user_norm_text)
        max_len = max(len(ref_norm), len(user_norm_text), 1)
        norm_dist_l1 = dist_l1 / max_len

        if norm_dist_l1 == 0.0:
            # Layer 1 says identical — mark correct, skip Layer 2 (fast path)
            results.append({
                'word': script_word, 'type': 'correct', 'severity': 0,
                'ref_phon': ref_phon, 'pred_phon': user_phon_text,
                'per_word_score': 100.0
            })
            continue

        # ── LAYER 2: Acoustic check (Wav2Vec2) ──
        # Find timing for this spoken word
        word_timing = spoken_word  # caller provides timing dict; see pipeline.py

        # FIX 7: minimum clip length guard
        timing = word_timing if isinstance(word_timing, dict) else {}
        start, end = timing.get('start', 0), timing.get('end', 0)
        clip_duration = end - start

        if clip_duration < MIN_CLIP_DURATION:
            # Clip too short for reliable Wav2Vec2 output
            # Fall back to Layer 1 result
            error_type = classify_from_dist(norm_dist_l1)
            results.append(build_result(script_word, error_type, ref_phon,
                                        user_phon_text, norm_dist_l1))
            continue

        clip = slice_word_audio(audio, start, end)
        pred_phon_ipa = recognize_phonemes(clip)   # IPA from Wav2Vec2
        pred_norm = normalize(pred_phon_ipa)

        # Compare canonical (ARPAbet normalized) vs recognized (IPA normalized)
        dist_l2 = Levenshtein.distance(ref_norm, pred_norm)
        max_len2 = max(len(ref_norm), len(pred_norm), 1)
        norm_dist_l2 = dist_l2 / max_len2

        if norm_dist_l2 == 0.0:
            results.append({
                'word': script_word, 'type': 'correct', 'severity': 0,
                'ref_phon': ref_phon, 'pred_phon': pred_phon_ipa,
                'per_word_score': 100.0
            })
            continue

        # ── Classify error type from editops backtrace ──
        ops = Levenshtein.editops(ref_norm, pred_norm)
        error_type = classify_ops(ops, ref_norm, pred_norm)
        result = build_result(script_word, error_type, ref_phon,
                              pred_phon_ipa, norm_dist_l2)
        results.append(result)

    return results


def classify_ops(ops: list, ref: list, pred: list) -> str:
    """Determine dominant error type from Levenshtein editops."""
    if not ops: return 'correct'
    counts = {'replace': 0, 'delete': 0, 'insert': 0}
    for op, i, j in ops:
        counts[op] += 1
    # Dominant type = whichever has most operations
    dominant = max(counts, key=counts.get)
    if dominant == 'replace': return 'substitution'
    if dominant == 'delete':  return 'omission'
    if dominant == 'insert':  return 'insertion'
    return 'substitution'

def classify_from_dist(norm_dist: float) -> str:
    """Fallback classifier when audio clip is too short for Wav2Vec2."""
    if norm_dist <= 0.0: return 'correct'
    if norm_dist <= 0.2: return 'substitution'
    return 'omission'  # large distance suggests dropped phonemes



# In error_detector.py

SEVERITY = {
    'correct':      0,
    'substitution': 6,   # will be overridden by count-based logic below
    'omission':     7,   # will be overridden by count-based logic below
    'insertion':    4,   # will be overridden by count-based logic below
    'skipped':      15,
    'stress':       3,
}

def assign_severity(error_type: str, ops: list) -> int:
    """
    Assign severity based on error type AND number of operations.
    A word with 2 substitutions is penalized more than one with 1.
    """
    if error_type == 'correct': return 0
    if error_type == 'skipped': return SEVERITY['skipped']
    if error_type == 'stress':  return SEVERITY['stress']

    n = len(ops)  # number of edit operations

    if error_type == 'substitution':
        return 6 if n == 1 else 10
    if error_type == 'omission':
        return 7 if n == 1 else 12
    if error_type == 'insertion':
        return 4 if n == 1 else 8

    return 6  # default


# In error_detector.py

def per_word_score(error: dict) -> float:
    """
    Returns a score from 0.0 to 100.0 for a single word.
    Uses normalized Levenshtein distance between canonical and recognized phonemes.
    This is the value shown in the Flutter word tooltip.
    """
    if error.get('type') == 'correct':
        return 100.0
    if error.get('type') == 'skipped':
        return 0.0

    ref  = error.get('ref_phon_norm',  [])
    pred = error.get('pred_phon_norm', [])

    if not ref and not pred:
        return 100.0

    max_len = max(len(ref), len(pred))
    if max_len == 0:
        return 100.0

    dist  = Levenshtein.distance(ref, pred)
    score = max(0.0, 1.0 - dist / max_len) * 100.0
    return round(score, 1)

def build_result(word, error_type, ref_phon, pred_phon, norm_dist) -> dict:
    """Build the standard result dict for one word."""
    ref_norm  = normalize(ref_phon)
    pred_norm = normalize(pred_phon)
    ops = Levenshtein.editops(ref_norm, pred_norm)
    severity = assign_severity(error_type, ops)
    result = {
        'word':           word,
        'type':           error_type,
        'severity':       severity,
        'ref_phon':       ref_phon,
        'pred_phon':      pred_phon,
        'ref_phon_norm':  ref_norm,
        'pred_phon_norm': pred_norm,
        'editops':        ops
    }
    result['per_word_score'] = per_word_score(result)  # FIX 3
    return result