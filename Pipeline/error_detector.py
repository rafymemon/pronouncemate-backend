import Levenshtein
import re
import numpy as np
from typing import List, Dict, Tuple, Optional

from .g2p_module import get_phonemes, build_answer_key
from .wav2vec_phonemes import (
    recognize_phonemes,
    recognize_phonemes_with_timestamps,
    bucket_phonemes_by_word,
    text_to_phonemes,
)
from .accent_rules import check_accent_pattern, accent_score


# ── Severity Scale ─────────────────────────────────────────────
SEVERITY = {
    'correct':      0,
    'accent':       0,   # forgiven Pakistani-accent variant — never penalized
    'minor':        2,   # close-but-not-exact acoustic read — see MINOR_DIST_THRESHOLD
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

# A word's Whisper timing must be at least this long to be treated as
# acoustically usable at all — below this, timing is too unreliable to
# trust for bucketing and the word falls straight to the text fallback.
MIN_CLIP_DURATION = 0.15

# A substitution/omission/insertion whose normalized phoneme edit distance
# is at or below this is downgraded to 'minor' instead of a hard error —
# e.g. a dropped plural -s, a slightly shortened vowel, a light final-
# consonant softening. These are real, technically-imperfect reads, but
# categorically different from a word that came out almost unrecognizable.
# TUNABLE: pick this value empirically against a labeled eval set once
# you have one for the thesis — 0.35 is a reasonable starting point based
# on observed session data, not a derived constant.
MINOR_DIST_THRESHOLD = 0.35

# TEMPORARY — diagnostic flag. Prints ref/pred phonemes and which code
# path each word took. Flip to True when validating this change; leave
# False otherwise.
DEBUG_DETECTION = True


def _find_prev_end(timings: list, i: int) -> Optional[float]:
    """Walk backward from index i for the nearest word with real timing."""
    for j in range(i - 1, -1, -1):
        if timings[j] is not None:
            return timings[j][1]
    return None


def _find_next_start(timings: list, i: int) -> Optional[float]:
    """Walk forward from index i for the nearest word with real timing."""
    for j in range(i + 1, len(timings)):
        if timings[j] is not None:
            return timings[j][0]
    return None


def _maybe_downgrade_to_minor(error_type: str, norm_dist: float) -> str:
    """
    Downgrade a hard substitution/omission/insertion verdict to 'minor'
    when the underlying phoneme distance is small enough that this was a
    close, largely-correct read rather than a genuinely different word.
    Never touches 'correct', 'accent', 'skipped', or 'inserted' — those
    are decided elsewhere and never pass through here.
    """
    if error_type in ('substitution', 'omission', 'insertion') and norm_dist <= MINOR_DIST_THRESHOLD:
        return 'minor'
    return error_type


# ── Main Detection ────────────────────────────────────────────
def detect_errors(alignment: list, audio: np.ndarray, sr: int = 16000,
                  answer_key: dict = None, script_text: str = None) -> list:
    """
    For each aligned (spoken_word_info, script_word) pair:
      1. Compare against a per-word bucket of phonemes drawn from ONE
         continuous Wav2Vec2 decode of the whole utterance (see below) —
         this is the real pronunciation verification, since it's derived
         from what was actually spoken rather than Whisper's ASR text,
         which tends to auto-correct mispronounced-but-recognizable words
         back to the intended dictionary spelling.
      2. If the acoustic reading isn't an exact match, check it against
         the Pakistani-accent tolerance patterns (accent_rules.py) BEFORE
         falling back to generic substitution/omission classification.
         Only a clean match against the forgiven pattern list is scored
         as 'accent'; anything else — including the flagged f/p case —
         falls through to normal classification.
      3. If it's not an accent match either, classify from editops, then
         downgrade to 'minor' if the phoneme distance is small enough
         that this was a close read rather than a genuinely wrong word
         (see MINOR_DIST_THRESHOLD).
      4. Fall back to the text-level G2P/CMU comparison only when the
         word's timing is unusable (too short, missing) or the bucket
         came back empty/low-confidence. Same accent- and minor-downgrade
         logic applies there too.

    METHODOLOGY NOTE — continuous decode + bucketing, not per-word slicing:
    Earlier versions of this pipeline sliced a separate audio clip per
    word (via slice_word_audio) and ran Wav2Vec2 on each clip in
    isolation. That approach removes exactly the phonetic context
    (coarticulation with neighboring words) Wav2Vec2 needs to recognize
    short/fast speech correctly — testing showed heavy contamination
    (leading/trailing phoneme fragments from neighboring words) even
    after correcting Whisper's own word-boundary timestamps. This version
    instead runs Wav2Vec2 ONCE on the full utterance audio
    (recognize_phonemes_with_timestamps), then splits that single
    continuous phoneme timeline into per-word groups purely by timestamp
    (bucket_phonemes_by_word) — the audio itself is never re-cut, so the
    model always has full context. This is also strictly cheaper: one
    forward pass per utterance instead of one per word.

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
      it never touches Wav2Vec2 output. The accent-tolerance check runs
      in BOTH branches, but always against phonemes in that branch's own
      alphabet — never mixed.
    """
    if answer_key is None and script_text is not None:
        answer_key = build_answer_key(script_text)
    if answer_key is None:
        raise ValueError("Must provide either answer_key or script_text")

    # Precompute (start, end) or None for every alignment position, so
    # each word can look up its actual timing — skipped/inserted entries
    # have no real audio timing.
    timings = []
    for spoken_info, _ in alignment:
        if isinstance(spoken_info, dict) and 'start' in spoken_info and 'end' in spoken_info:
            timings.append((spoken_info['start'], spoken_info['end']))
        else:
            timings.append(None)

    # ── ONE continuous Wav2Vec2 decode for the whole utterance ──
    # Replaces the old per-word slice_word_audio() + get_spoken_phonemes()
    # calls. See the methodology note in the docstring above.
    try:
        phoneme_timeline = recognize_phonemes_with_timestamps(audio, sr)
    except Exception as e:
        print(f"Warning: whole-utterance Wav2Vec2 decode failed: {e}")
        phoneme_timeline = []

    word_phoneme_buckets = bucket_phonemes_by_word(phoneme_timeline, timings)

    if DEBUG_DETECTION:
        print(f"[DEBUG] utterance-level phoneme timeline: {len(phoneme_timeline)} phonemes decoded")

    results = []

    for i, (spoken_info, script_word) in enumerate(alignment):

        # ── Skipped word ──
        if spoken_info is None:
            if DEBUG_DETECTION:
                print(f"[DEBUG] '{script_word}': SKIPPED (no spoken word aligned)")
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
            if DEBUG_DETECTION:
                print(f"[DEBUG] '{spoken_word}': INSERTED (no script word aligned)")
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

        # ── Try ACOUSTIC check first, using this word's phoneme bucket ──
        start = timing.get('start', 0.0)
        end = timing.get('end', 0.0)
        clip_duration = end - start
        timing_usable = clip_duration >= MIN_CLIP_DURATION and start < end

        bucket = word_phoneme_buckets[i]
        pred_phon_ipa = bucket['phonemes']
        avg_conf = bucket['avg_confidence']

        if DEBUG_DETECTION:
            print(f"[DEBUG] '{script_word}': spoken='{spoken_word}' "
                  f"duration={clip_duration:.3f}s timing_usable={timing_usable} "
                  f"bucket_phonemes={pred_phon_ipa} avg_conf={avg_conf:.3f}")

        acoustic_result = None
        if timing_usable:
            # IPA reference for THIS branch only — never mix with the
            # ARPAbet ref_phon/ref_norm computed above.
            ref_phon_ipa = text_to_phonemes(script_word)
            ref_norm_ipa = normalize(ref_phon_ipa)

            if pred_phon_ipa and avg_conf >= 0.3:
                pred_norm = normalize(pred_phon_ipa)
                dist_l2 = Levenshtein.distance(ref_norm_ipa, pred_norm)
                max_len_l2 = max(len(ref_norm_ipa), len(pred_norm), 1)
                norm_dist_l2 = dist_l2 / max_len_l2

                if norm_dist_l2 < 0.001:
                    if DEBUG_DETECTION:
                        print(f"[DEBUG]   -> CORRECT (exact acoustic match)")
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
                    accent_match = check_accent_pattern(ref_norm_ipa, pred_norm)
                    if accent_match:
                        if DEBUG_DETECTION:
                            print(f"[DEBUG]   -> ACCENT ({accent_match}), dist={norm_dist_l2:.3f}")
                        acoustic_result = {
                            'word': script_word,
                            'type': 'accent',
                            'severity': SEVERITY['accent'],
                            'ref_phon': ref_phon_ipa,
                            'pred_phon': pred_phon_ipa,
                            'ref_phon_norm': ref_norm_ipa,
                            'pred_phon_norm': pred_norm,
                            'accent_pattern': accent_match,
                            'per_word_score': accent_score(norm_dist_l2)
                        }
                    else:
                        ops = Levenshtein.editops(ref_norm_ipa, pred_norm)
                        error_type = classify_ops(ops, ref_norm_ipa, pred_norm)
                        error_type = _maybe_downgrade_to_minor(error_type, norm_dist_l2)
                        if DEBUG_DETECTION:
                            print(f"[DEBUG]   -> {error_type.upper()} (no accent match), "
                                  f"dist={norm_dist_l2:.3f}, ops={ops}")
                        acoustic_result = build_result(script_word, error_type, ref_phon_ipa,
                                                        pred_phon_ipa, norm_dist_l2)
            elif pred_phon_ipa == [] or pred_phon_ipa is None:
                # Bucket came back empty — no phonemes from the utterance
                # decode landed in this word's time range at all.
                if DEBUG_DETECTION:
                    print(f"[DEBUG]   -> OMISSION (empty bucket)")
                acoustic_result = build_result(script_word, 'omission', ref_phon_ipa,
                                                [], 1.0)
            elif DEBUG_DETECTION:
                print(f"[DEBUG]   -> low confidence ({avg_conf:.3f}), falling through to text fallback")
            # else: low confidence but non-empty bucket — fall through to
            # the text-based fallback below rather than trusting a
            # low-confidence acoustic read.

        if acoustic_result is not None:
            results.append(acoustic_result)
            continue

        # ── FALLBACK: text-level G2P/CMU comparison (ARPAbet vs ARPAbet) ──
        # Used when the word's timing is unusable, the bucket was empty,
        # or confidence was too low to trust the acoustic read.
        if DEBUG_DETECTION:
            print(f"[DEBUG]   -> using TEXT FALLBACK (Whisper transcript vs answer key)")

        user_phon_text = get_phonemes(spoken_word) if spoken_word else ref_phon
        user_norm_text = normalize(user_phon_text)

        dist_l1 = Levenshtein.distance(ref_norm, user_norm_text)
        max_len_l1 = max(len(ref_norm), len(user_norm_text), 1)
        norm_dist_l1 = dist_l1 / max_len_l1

        if norm_dist_l1 < 0.001:
            if DEBUG_DETECTION:
                print(f"[DEBUG]   -> CORRECT (text fallback exact match)")
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
            accent_match = check_accent_pattern(ref_norm, user_norm_text)
            if accent_match:
                if DEBUG_DETECTION:
                    print(f"[DEBUG]   -> ACCENT ({accent_match}) via text fallback, dist={norm_dist_l1:.3f}")
                results.append({
                    'word': script_word,
                    'type': 'accent',
                    'severity': SEVERITY['accent'],
                    'ref_phon': ref_phon,
                    'pred_phon': user_phon_text,
                    'ref_phon_norm': ref_norm,
                    'pred_phon_norm': user_norm_text,
                    'accent_pattern': accent_match,
                    'per_word_score': accent_score(norm_dist_l1)
                })
            else:
                error_type = classify_from_dist(norm_dist_l1)
                error_type = _maybe_downgrade_to_minor(error_type, norm_dist_l1)
                if DEBUG_DETECTION:
                    print(f"[DEBUG]   -> {error_type.upper()} via text fallback, dist={norm_dist_l1:.3f}, "
                          f"ref={ref_norm} vs spoken_word_g2p={user_norm_text}")
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
    """Fallback classifier when word timing is too short/unreliable for the acoustic path."""
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
    if error_type == 'accent':
        return SEVERITY['accent']
    if error_type == 'minor':
        return SEVERITY['minor']
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