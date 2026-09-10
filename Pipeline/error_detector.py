import Levenshtein
import re
import numpy as np
from typing import List, Dict, Tuple, Optional

from .g2p_module import get_phonemes, build_answer_key
from .wav2vec_phonemes import recognize_phonemes, get_spoken_phonemes, text_to_phonemes
from .audio_processor import slice_word_audio


# ── Severity Scale ─────────────────────────────────────────────
SEVERITY = {
    'correct':      0,
    'substitution': 6,
    'omission':     7,
    'insertion':    4,
    'skipped':      15,
    'stress':       3,
}


# ── Normalization ─────────────────────────────────────────────
_ARPABET_RE = re.compile(r'^[A-Z]+[0-2]?$')

def normalize(phonemes: list) -> list:
    """
    Convert ARPAbet or IPA to a comparable form.
    Strips stress digits (ARPAbet: AE1 -> ae) and IPA stress markers (ˈ, ˌ).
    """
    normalized = []
    for p in phonemes:
        if not p or not p.strip():
            continue
        p = re.sub(r'[012]', '', p)
        p = re.sub(r'[ˈˌ]', '', p)
        normalized.append(p.lower().strip())
    return normalized


# ── Configuration ─────────────────────────────────────────────
L1_THRESHOLD = 0.15
MIN_CLIP_DURATION = 0.15


# ── Main Detection ────────────────────────────────────────────
def detect_errors(alignment: list, audio: np.ndarray, sr: int = 16000,
                  answer_key: dict = None, script_text: str = None) -> list:
    """
    For each aligned (spoken_word_info, script_word) pair:
      1. Try the ACOUSTIC check first (Wav2Vec2 on the audio clip) —
         this is the real pronunciation verification, since it's derived
         from what was actually spoken rather than Whisper's ASR text,
         which tends to auto-correct mispronounced-but-recognizable words
         back to the intended dictionary spelling.
      2. Fall back to the text-level G2P/CMU comparison only when the
         audio clip is unusable (too short, bad timestamps, or the
         Wav2Vec2 call fails / returns low confidence).
      3. Classify error type from editops backtrace.

    IMPORTANT — phoneme alphabets are NOT interchangeable:
      - answer_key (from g2p_module/CMU dict) is ARPAbet.
      - Wav2Vec2 (facebook/wav2vec2-lv-60-espeak-cv-ft) outputs
        espeak/IPA phonemes.
      Comparing ARPAbet against IPA directly (e.g. 'hh' vs 'h', 'ow' vs
      'oʊ') is meaningless — they almost never match even for correct
      pronunciation. So the acoustic branch below builds its OWN IPA
      reference via text_to_phonemes() (espeak), completely separate
      from answer_key. The text-fallback branch keeps using
      answer_key/CMU (ARPAbet-vs-ARPAbet, internally consistent) since
      it never touches Wav2Vec2 output.
    """
    if answer_key is None and script_text is not None:
        answer_key = build_answer_key(script_text)
    if answer_key is None:
        raise ValueError("Must provide either answer_key or script_text")

    results = []

    for spoken_info, script_word in alignment:

        # ── Skipped word ──
        if spoken_info is None:
            results.append({
                'word': script_word,
                'type': 'skipped',
                'severity': SEVERITY['skipped'],
                'ref_phon': answer_key.get(script_word, []),
                'pred_phon': [],
                'ref_phon_norm': normalize(answer_key.get(script_word, [])),
                'pred_phon_norm': [],
                'per_word_score': 0.0
            })
            continue

        # ── Extract spoken word info ──
        spoken_word = spoken_info.get('word', '') if isinstance(spoken_info, dict) else str(spoken_info)
        timing = spoken_info if isinstance(spoken_info, dict) else {}

        # ── Inserted word ──
        if script_word is None or script_word == '':
            results.append({
                'word': spoken_word,
                'type': 'inserted',
                'severity': 0,
                'ref_phon': [],
                'pred_phon': get_phonemes(spoken_word) if spoken_word else [],
                'ref_phon_norm': [],
                'pred_phon_norm': normalize(get_phonemes(spoken_word)) if spoken_word else [],
                'per_word_score': 0.0
            })
            continue

        # ── Get ARPAbet reference phonemes (used only by text-fallback) ──
        ref_phon = answer_key.get(script_word, [])
        if not ref_phon:
            ref_phon = get_phonemes(script_word)
        ref_norm = normalize(ref_phon)

        # ── Try ACOUSTIC check first ──
        start = timing.get('start', 0.0)
        end = timing.get('end', 0.0)
        clip_duration = end - start
        clip_usable = clip_duration >= MIN_CLIP_DURATION and start < end

        acoustic_result = None
        if clip_usable:
            # IPA reference for THIS branch only — never mix with the
            # ARPAbet ref_phon/ref_norm computed above.
            ref_phon_ipa = text_to_phonemes(script_word)
            ref_norm_ipa = normalize(ref_phon_ipa)

            clip = slice_word_audio(audio, start, end)
            try:
                pred_phon_ipa, avg_conf, _ = get_spoken_phonemes(clip, sr)
            except Exception as e:
                print(f"Warning: Wav2Vec2 failed for '{script_word}': {e}")
                pred_phon_ipa, avg_conf = None, 0.0

            if pred_phon_ipa and avg_conf >= 0.3:
                pred_norm = normalize(pred_phon_ipa)
                dist_l2 = Levenshtein.distance(ref_norm_ipa, pred_norm)
                max_len_l2 = max(len(ref_norm_ipa), len(pred_norm), 1)
                norm_dist_l2 = dist_l2 / max_len_l2

                if norm_dist_l2 < 0.001:
                    acoustic_result = {
                        'word': script_word,
                        'type': 'correct',
                        'severity': 0,
                        'ref_phon': ref_phon_ipa,
                        'pred_phon': pred_phon_ipa,
                        'ref_phon_norm': ref_norm_ipa,
                        'pred_phon_norm': pred_norm,
                        'per_word_score': 100.0
                    }
                else:
                    ops = Levenshtein.editops(ref_norm_ipa, pred_norm)
                    error_type = classify_ops(ops, ref_norm_ipa, pred_norm)
                    acoustic_result = build_result(script_word, error_type, ref_phon_ipa,
                                                    pred_phon_ipa, norm_dist_l2)
            elif pred_phon_ipa == [] or pred_phon_ipa is None:
                # Wav2Vec2 ran but detected no usable phonemes in the clip
                acoustic_result = build_result(script_word, 'omission', ref_phon_ipa,
                                                [], 1.0)
            # else: low confidence but non-empty prediction — fall through
            # to the text-based fallback below rather than trusting a
            # low-confidence acoustic read.

        if acoustic_result is not None:
            results.append(acoustic_result)
            continue

        # ── FALLBACK: text-level G2P/CMU comparison (ARPAbet vs ARPAbet) ──
        # Used when the audio clip is unusable, Wav2Vec2 failed, or
        # confidence was too low to trust the acoustic read.
        user_phon_text = get_phonemes(spoken_word) if spoken_word else ref_phon
        user_norm_text = normalize(user_phon_text)

        dist_l1 = Levenshtein.distance(ref_norm, user_norm_text)
        max_len_l1 = max(len(ref_norm), len(user_norm_text), 1)
        norm_dist_l1 = dist_l1 / max_len_l1

        if norm_dist_l1 < 0.001:
            results.append({
                'word': script_word,
                'type': 'correct',
                'severity': 0,
                'ref_phon': ref_phon,
                'pred_phon': user_phon_text,
                'ref_phon_norm': ref_norm,
                'pred_phon_norm': user_norm_text,
                'per_word_score': 100.0
            })
        else:
            error_type = classify_from_dist(norm_dist_l1)
            result = build_result(script_word, error_type, ref_phon,
                                  user_phon_text, norm_dist_l1)
            results.append(result)

    return results


def classify_ops(ops: list, ref: list, pred: list) -> str:
    """Determine dominant error type from Levenshtein editops."""
    if not ops:
        return 'correct'

    counts = {'replace': 0, 'delete': 0, 'insert': 0}
    for op, i, j in ops:
        counts[op] += 1

    total = len(ops)

    if counts['replace'] == total:
        return 'substitution'
    if counts['delete'] == total:
        return 'omission'
    if counts['insert'] == total:
        return 'insertion'

    replace_ratio = counts['replace'] / total
    delete_ratio = counts['delete'] / total
    insert_ratio = counts['insert'] / total

    if replace_ratio >= 0.5:
        return 'substitution'
    elif delete_ratio >= 0.5:
        return 'omission'
    elif insert_ratio >= 0.5:
        return 'insertion'

    return 'substitution'


def classify_from_dist(norm_dist: float) -> str:
    """Fallback classifier when audio clip is too short for Wav2Vec2."""
    if norm_dist < 0.001:
        return 'correct'
    if norm_dist <= 0.2:
        return 'substitution'
    if norm_dist <= 0.5:
        return 'omission'
    return 'omission'


def assign_severity(error_type: str, ops: list) -> int:
    """Assign severity based on error type AND number of operations."""
    if error_type == 'correct':
        return 0
    if error_type == 'skipped':
        return SEVERITY['skipped']
    if error_type == 'stress':
        return SEVERITY['stress']

    n = len(ops)

    if error_type == 'substitution':
        return 6 if n == 1 else 10
    if error_type == 'omission':
        return 7 if n == 1 else 12
    if error_type == 'insertion':
        return 4 if n == 1 else 8

    return 6


def per_word_score(error: dict) -> float:
    """Returns 0.0-100.0 for a single word from error dict."""
    if error.get('type') == 'correct':
        return 100.0
    if error.get('type') == 'skipped':
        return 0.0

    ref = error.get('ref_phon_norm', [])
    pred = error.get('pred_phon_norm', [])

    if not ref and not pred:
        return 100.0

    max_len = max(len(ref), len(pred))
    if max_len == 0:
        return 100.0

    dist = Levenshtein.distance(ref, pred)
    score = max(0.0, 1.0 - dist / max_len) * 100.0
    return round(score, 1)


def per_word_score_from_lists(ref: list, pred: list) -> float:
    """Standalone utility: score from two phoneme lists directly."""
    if not ref and not pred:
        return 100.0

    ref_norm = normalize(ref)
    pred_norm = normalize(pred)

    max_len = max(len(ref_norm), len(pred_norm))
    if max_len == 0:
        return 100.0

    dist = Levenshtein.distance(ref_norm, pred_norm)
    score = max(0.0, 1.0 - dist / max_len) * 100.0
    return round(score, 1)


def build_result(word: str, error_type: str, ref_phon: list,
                 pred_phon: list, norm_dist: float) -> dict:
    """Build the standard result dict for one word."""
    ref_norm = normalize(ref_phon)
    pred_norm = normalize(pred_phon)
    ops = Levenshtein.editops(ref_norm, pred_norm)
    severity = assign_severity(error_type, ops)

    result = {
        'word': word,
        'type': error_type,
        'severity': severity,
        'ref_phon': ref_phon,
        'pred_phon': pred_phon,
        'ref_phon_norm': ref_norm,
        'pred_phon_norm': pred_norm,
        'editops': ops,
        'norm_distance': round(norm_dist, 3)
    }
    result['per_word_score'] = per_word_score(result)
    return result