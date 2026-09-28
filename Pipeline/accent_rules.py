# Pipeline/accent_rules.py
"""
Pakistani-accent tolerance rules for PronounceMate.

This module encodes the project's core original contribution: a set of
documented Urdu/Punjabi-influenced English phoneme substitution patterns
that should be FORGIVEN (scored generously, not flagged as errors) versus
one superficially similar pattern that must remain FLAGGED because it
collapses real English minimal pairs rather than just sounding "accented".

All phoneme comparisons here operate in the SAME IPA/espeak alphabet that
Wav2Vec2 (facebook/wav2vec2-lv-60-espeak-cv-ft) produces — never ARPAbet.
Callers must pass already-normalized phoneme lists (via error_detector's
normalize()) so stress marks are stripped and casing is consistent.
"""

from typing import Dict, List, Optional, Set, Tuple
import Levenshtein


# ── Forgiven substitution patterns ──────────────────────────────
# pattern_id -> set of (reference_phoneme, spoken_phoneme) pairs that are
# tolerated as accent variation rather than scored as a pronunciation error.
FORGIVEN_SUBSTITUTIONS: Dict[str, Set[Tuple[str, str]]] = {
    'dental_stop_voiceless_th': {('θ', 't')},
    'dental_stop_voiced_th':    {('ð', 'd')},
    'v_w_merger':               {('v', 'w'), ('w', 'v')},
    'retroflex_t':              {('t', 'ʈ')},
    'retroflex_d':              {('d', 'ɖ')},
    'retroflex_r':              {('ɹ', 'ɽ'), ('r', 'ɽ'), ('ɹ', 'r')},
    'vowel_i_merger':           {('ɪ', 'i')},
}

# ── Flagged pattern — explicitly NOT forgiven ───────────────────
# f -> p is phonetically "close" (both labial) in the same way the forgiven
# pairs above are, but it collapses genuine English minimal pairs
# (fan/pan, fine/pine, coffee/copy). This is a real pronunciation error,
# not an accent feature — kept here explicitly so it's never accidentally
# folded into FORGIVEN_SUBSTITUTIONS later.
FLAGGED_SUBSTITUTIONS: Set[Tuple[str, str]] = {('f', 'p')}

# ── Epenthetic vowel insertion ──────────────────────────────────
# Word-initial /s/ + consonant clusters commonly get a short vowel inserted
# before them: "school" -> /ɪskuːl/, "speak" -> /ɪspiːk/.
EPENTHETIC_VOWELS: Set[str] = {'ɪ', 'ə', 'i'}
CLUSTER_TRIGGER_CONSONANT = 's'


def check_accent_pattern(ref_norm: List[str], pred_norm: List[str]) -> Optional[str]:
    """
    Check whether the ENTIRE difference between ref_norm and pred_norm is
    explained by forgiven Pakistani-accent patterns (substitution or the
    epenthetic-vowel insertion).

    Deliberately conservative: if there's a single unexplained edit op —
    including the flagged f/p case, or any insert/delete other than the
    specific epenthesis case — this returns None and the caller should
    fall through to normal error classification.

    Args:
        ref_norm: normalized reference IPA phonemes (script word)
        pred_norm: normalized predicted IPA phonemes (what was spoken)

    Returns:
        Comma-joined pattern id(s) matched (e.g. 'v_w_merger'), or None
        if this isn't a clean accent-pattern match.
    """
    if ref_norm == pred_norm:
        return None  # exact match — handled as 'correct' upstream, not here

    ops = Levenshtein.editops(ref_norm, pred_norm)
    if not ops:
        return None

    # ── Special case: single word-initial epenthetic vowel insertion ──
    if (len(ops) == 1 and ops[0][0] == 'insert' and ops[0][1] == 0
            and pred_norm and pred_norm[0] in EPENTHETIC_VOWELS
            and ref_norm and ref_norm[0] == CLUSTER_TRIGGER_CONSONANT):
        return 'epenthetic_vowel'

    # ── General case: every op must be a 'replace' matching a forgiven pair ──
    matched_patterns = []

    for op, i, j in ops:
        if op != 'replace':
            return None  # any insert/delete outside the epenthesis case
                         # above is too structurally different to call "accent"

        pair = (ref_norm[i], pred_norm[j])

        if pair in FLAGGED_SUBSTITUTIONS:
            return None  # explicitly excluded, even though it's a clean
                         # single substitution

        found = False
        for pattern_id, allowed_pairs in FORGIVEN_SUBSTITUTIONS.items():
            if pair in allowed_pairs:
                matched_patterns.append(pattern_id)
                found = True
                break

        if not found:
            return None  # some substitution not in the tolerance list at all

    return ','.join(matched_patterns) if matched_patterns else None


def accent_score(norm_dist: float) -> float:
    """
    Generous scoring for a confirmed accent-pattern match, per the design
    documented in scorer.py: min(similarity + 8, 95). Never returns a
    perfect 100 — an accent variant is still a variant, not an exact match,
    and 95 is the ceiling so it stays visibly distinct from 'correct' in
    analytics even though it's not penalized as an error.
    """
    similarity_pct = (1.0 - norm_dist) * 100.0
    return round(min(similarity_pct + 8.0, 95.0), 1)