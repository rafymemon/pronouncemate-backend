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

# ── Acoustic confidence gates ─────────────────────────────────
# Below ACOUSTIC_MIN_CONF the bucket is not used at all (unchanged).
# Between MIN and TRUST the bucket may still CONFIRM a word (exact match /
# accent match / variant match) but may NOT condemn it: a non-matching read
# at that confidence is far more often wav2vec noise or neighbouring-word
# leakage than a real mispronunciation (observed: 'my' conf 0.46 -> ['n','ə'],
# 'during' conf 0.51, 'quick' conf 0.58 -> 'oɪ'), so the word goes to the
# text fallback (Whisper transcript vs answer key) instead.
# TUNABLE: raise to be stricter, lower to be stricter-on-acoustics.
ACOUSTIC_MIN_CONF = 0.3
ACOUSTIC_TRUST_CONF = 0.6

# ── Boundary refinement (see refine_bucket_boundaries) ────────
# Whisper's word timestamps are only accurate to a few tens of ms, and
# short function words are often off by more than that. Because the
# acoustic path buckets phonemes purely by those timestamps, edge phonemes
# routinely land in the neighbouring word's bucket ("my" gets the /n/ of
# "name", "a" gets the "soft" of a word Whisper dropped, etc.). Refinement
# re-assigns edge phonemes between ADJACENT words whenever doing so
# strictly lowers the combined edit distance to the two words' references.
# Set REFINE_BOUNDARIES = False to get the previous behaviour back.
REFINE_BOUNDARIES = True
MAX_BOUNDARY_SHIFT = 4      # max phonemes moved across one boundary per step
BOUNDARY_PASSES = 3         # max left-to-right sweeps over the utterance

# TEMPORARY — diagnostic flag. Prints ref/pred phonemes and which code
# path each word took. Flip to True when validating this change; leave
# False otherwise.
DEBUG_DETECTION = True


# ── Phoneme canonicalization (acoustic / IPA branch ONLY) ─────
# The reference comes from espeak (phonemizer); the prediction comes from
# wav2vec2 fine-tuned on espeak labels. They use slightly different symbols
# for the SAME sound, which used to be scored as substitutions. This map
# folds those into one form. It is applied ONLY after the exact-match and
# accent checks (which keep using plain normalize()), so it can never change
# what accent_rules.py sees or reports.
#
#  notation only (same sound, different symbol):
#     ᵻ ᵿ ɐ  oːɹ/oɹ/ɔɹ  ɚ ɝ  g  ɫ  r  ɜ ɑ
#  dialect-standard (valid in British / Pakistani / Indian English):
#     ɒ ɔ -> ɑː   (LOT vowel)       ɾ -> t   (flap allophone of /t/)
_SYMBOL_EQUIV = {
    'ᵻ': ['ɪ'], 'ᵿ': ['ʊ'], 'ɐ': ['ə'],
    'oːɹ': ['ɔːɹ'], 'oɹ': ['ɔːɹ'], 'ɔɹ': ['ɔːɹ'],
    'ɚ': ['ə', 'ɹ'], 'ɝ': ['ɜː', 'ɹ'],
    'g': ['ɡ'], 'ɫ': ['l'], 'r': ['ɹ'],
    'ɜ': ['ɜː'], 'ɑ': ['ɑː'],
    'ɒ': ['ɑː'], 'ɔ': ['ɑː'],
    'ɾ': ['t'],
}


def _canon(phonemes: list) -> list:
    """normalize() + fold equivalent IPA symbols. IPA branch only."""
    out: list = []
    for p in normalize(phonemes):
        out.extend(_SYMBOL_EQUIV.get(p, [p]))
    return out


def _norm_dist(a: list, b: list) -> float:
    return Levenshtein.distance(a, b) / max(len(a), len(b), 1)


# ── Alternative valid pronunciations ──────────────────────────
# espeak gives ONE reference per word, but many words have several correct
# pronunciations. A read is judged against the closest one.
# Written as token lists in the same IPA alphabet as the model output; they
# pass through _canon() so notation differences don't matter. Extend freely.

# Weak/strong forms of function words. Natural English reduces these
# ("to" -> /tə/), so reading them in the strong form OR the weak form is fine.
FUNCTION_WORD_VARIANTS = {
    'a':      [['ə'], ['eɪ']],
    'an':     [['ə', 'n'], ['æ', 'n']],
    'the':    [['ð', 'ə'], ['ð', 'ɪ'], ['ð', 'iː']],
    'to':     [['t', 'ə'], ['t', 'ʊ'], ['t', 'uː']],
    'of':     [['ə', 'v'], ['ʌ', 'v'], ['ɒ', 'v']],
    'and':    [['ə', 'n', 'd'], ['ə', 'n'], ['æ', 'n', 'd'], ['æ', 'n']],
    'for':    [['f', 'ə'], ['f', 'ɔːɹ'], ['f', 'ɔː']],
    'from':   [['f', 'ɹ', 'ə', 'm'], ['f', 'ɹ', 'ʌ', 'm'], ['f', 'ɹ', 'ɒ', 'm']],
    'at':     [['ə', 't'], ['æ', 't']],
    'as':     [['ə', 'z'], ['æ', 'z']],
    'that':   [['ð', 'ə', 't'], ['ð', 'æ', 't']],
    'than':   [['ð', 'ə', 'n'], ['ð', 'æ', 'n']],
    'can':    [['k', 'ə', 'n'], ['k', 'æ', 'n']],
    'was':    [['w', 'ə', 'z'], ['w', 'ɒ', 'z'], ['w', 'ʌ', 'z']],
    'were':   [['w', 'ə'], ['w', 'ə', 'ɹ'], ['w', 'ɜː']],
    'have':   [['h', 'ə', 'v'], ['ə', 'v'], ['h', 'æ', 'v']],
    'has':    [['h', 'ə', 'z'], ['ə', 'z'], ['h', 'æ', 'z']],
    'had':    [['h', 'ə', 'd'], ['ə', 'd'], ['h', 'æ', 'd']],
    'be':     [['b', 'ɪ'], ['b', 'iː']],
    'been':   [['b', 'ɪ', 'n'], ['b', 'iː', 'n']],
    'you':    [['j', 'ə'], ['j', 'ʊ'], ['j', 'uː']],
    'your':   [['j', 'ə'], ['j', 'ɔːɹ'], ['j', 'ʊɹ']],
    'he':     [['h', 'ɪ'], ['ɪ'], ['h', 'iː']],
    'she':    [['ʃ', 'ɪ'], ['ʃ', 'iː']],
    'her':    [['h', 'ə'], ['ə'], ['h', 'ɜː']],
    'him':    [['ɪ', 'm'], ['h', 'ɪ', 'm']],
    'his':    [['ɪ', 'z'], ['h', 'ɪ', 'z']],
    'them':   [['ð', 'ə', 'm'], ['ð', 'ɛ', 'm']],
    'us':     [['ə', 's'], ['ʌ', 's']],
    'we':     [['w', 'ɪ'], ['w', 'iː']],
    'but':    [['b', 'ə', 't'], ['b', 'ʌ', 't']],
    'or':     [['ɔːɹ'], ['ə']],
    'some':   [['s', 'ə', 'm'], ['s', 'ʌ', 'm']],
    'would':  [['w', 'ə', 'd'], ['w', 'ʊ', 'd']],
    'could':  [['k', 'ə', 'd'], ['k', 'ʊ', 'd']],
    'should': [['ʃ', 'ə', 'd'], ['ʃ', 'ʊ', 'd']],
    'will':   [['w', 'ɪ', 'l'], ['ə', 'l']],
    'my':     [['m', 'aɪ'], ['m', 'ɪ'], ['m', 'ə']],
    'with':   [['w', 'ɪ', 'ð'], ['w', 'ɪ', 'θ']],
}

# British / American (and other well-known) variants of content words.
# Pakistani English follows British norms, so these matter for this app.
WORD_VARIANTS = {
    'schedule':   [['ʃ', 'ɛ', 'd', 'j', 'uː', 'l'], ['ʃ', 'ɛ', 'd', 'uː', 'l'],
                   ['ʃ', 'ɛ', 'dʒ', 'uː', 'l'],
                   ['s', 'k', 'ɛ', 'dʒ', 'uː', 'l'], ['s', 'k', 'ɛ', 'dʒ', 'ʊ', 'l']],
    'ensure':     [['ɪ', 'n', 'ʃ', 'ɔːɹ'], ['ɛ', 'n', 'ʃ', 'ɔːɹ'],
                   ['ɪ', 'n', 'ʃ', 'ʊɹ'], ['ɛ', 'n', 'ʃ', 'ʊɹ']],
    'transition': [['t', 'ɹ', v, 'n', c, 'ɪ', s, 'ə', 'n']
                   for v in ('æ', 'ɑː') for c in ('s', 'z') for s in ('ʃ', 'ʒ')],
    'either':     [['aɪ', 'ð', 'ə'], ['iː', 'ð', 'ə'], ['aɪ', 'ð', 'ə', 'ɹ'], ['iː', 'ð', 'ə', 'ɹ']],
    'neither':    [['n', 'aɪ', 'ð', 'ə'], ['n', 'iː', 'ð', 'ə'],
                   ['n', 'aɪ', 'ð', 'ə', 'ɹ'], ['n', 'iː', 'ð', 'ə', 'ɹ']],
    'privacy':    [['p', 'ɹ', 'ɪ', 'v', 'ə', 's', 'i'], ['p', 'ɹ', 'aɪ', 'v', 'ə', 's', 'i']],
}

_VARIANTS_CANON: Dict[str, List[list]] = {}
for _w, _vs in list(FUNCTION_WORD_VARIANTS.items()) + list(WORD_VARIANTS.items()):
    _VARIANTS_CANON.setdefault(_w, []).extend(_canon(v) for v in _vs)

# Short function words whose acoustic bucket is often legitimately empty
# (a ~50 ms schwa that CTC decoding simply doesn't emit). Whisper DID hear
# them, so an empty bucket is not evidence of omission.
_WEAK_WORDS = frozenset(FUNCTION_WORD_VARIANTS)


# ── espeak reference cache ────────────────────────────────────
# text_to_phonemes() is an espeak call per word and was re-run for every
# distinct word of every session (it dominates detect_errors time). The
# reference for a word never changes, so cache it for the process lifetime.
_IPA_CACHE: Dict[str, tuple] = {}
_IPA_CACHE_MAX = 20000


def _ipa_cached(word: str) -> list:
    hit = _IPA_CACHE.get(word)
    if hit is None:
        hit = tuple(text_to_phonemes(word) or [])
        if len(_IPA_CACHE) < _IPA_CACHE_MAX:
            _IPA_CACHE[word] = hit
    return list(hit)


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


# ── Boundary refinement ───────────────────────────────────────
def refine_bucket_boundaries(phoneme_lists: List[list],
                             ref_lists: List[list],
                             confs: List[float],
                             max_shift: int = MAX_BOUNDARY_SHIFT,
                             max_passes: int = BOUNDARY_PASSES
                             ) -> Tuple[List[list], List[float], List[tuple]]:
    """
    Re-assign edge phonemes between adjacent words' buckets.

    For every adjacent pair (i, i+1) we try moving the last k phonemes of
    word i to the front of word i+1, and the first k phonemes of word i+1
    to the end of word i (k = 1..max_shift). A move is applied only if it
    STRICTLY lowers  dist(bucket_i, ref_i) + dist(bucket_i+1, ref_i+1),
    where dist is the raw phoneme edit distance (computed on canonicalized
    phonemes, so espeak-vs-wav2vec notation differences don't cause or
    block moves). Sweeps repeat (up to max_passes) so a phoneme can travel
    across a short word in a chain.

    Only bucket edges are ever touched and only when a neighbour's reference
    explains the phonemes better, so a phoneme that genuinely belongs to a
    mispronounced word stays with it: moving it would not reduce the total.

    Words with no reference (inserted words) have an empty reference list;
    skipped words keep their real reference and start with an empty bucket.
    Both can therefore donate/absorb phonemes without special-casing.

    Returns (new_phoneme_lists, new_confs, moves), where each move is
    (left_index, right_index, direction, moved_phonemes) and direction is
    'right' (left -> right) or 'left' (right -> left).

    avg_confidence is not recomputable from the bucket alone, so it is
    approximated: a bucket keeps its own value, a previously-empty bucket
    that receives phonemes inherits the donor's value, and a bucket that is
    emptied drops to 0.0.
    """
    P = [list(p) for p in phoneme_lists]
    C = list(confs)
    R = [_canon(r) for r in ref_lists]
    moves: List[tuple] = []

    def cost(pred: list, idx: int) -> int:
        return Levenshtein.distance(_canon(pred), R[idx])

    for _ in range(max_passes):
        changed = False
        for i in range(len(P) - 1):
            left, right = P[i], P[i + 1]
            base = cost(left, i) + cost(right, i + 1)
            if base == 0:
                continue

            best_cost = base
            best = None  # (new_left, new_right, direction, k)

            for k in range(1, min(max_shift, len(left)) + 1):
                new_left, new_right = left[:-k], left[-k:] + right
                c = cost(new_left, i) + cost(new_right, i + 1)
                if c < best_cost:
                    best_cost, best = c, (new_left, new_right, 'right', k)

            for k in range(1, min(max_shift, len(right)) + 1):
                new_left, new_right = left + right[:k], right[k:]
                c = cost(new_left, i) + cost(new_right, i + 1)
                if c < best_cost:
                    best_cost, best = c, (new_left, new_right, 'left', k)

            if best is None:
                continue

            new_left, new_right, direction, k = best
            if direction == 'right':
                moved = left[-k:]
                if not right:
                    C[i + 1] = C[i]
                if not new_left:
                    C[i] = 0.0
            else:
                moved = right[:k]
                if not left:
                    C[i] = C[i + 1]
                if not new_right:
                    C[i + 1] = 0.0

            P[i], P[i + 1] = new_left, new_right
            moves.append((i, i + 1, direction, list(moved)))
            changed = True

        if not changed:
            break

    return P, C, moves


# ── Acoustic classification of ONE word ───────────────────────
def _classify_acoustic(script_word: str, ref_phon_ipa: list, pred_phon_ipa: list,
                       avg_conf: float, allow_negative: bool = True) -> Optional[dict]:
    """
    Judge one word from its wav2vec phoneme bucket.

    Order (each step only runs if the previous one didn't decide):
      1. exact match on plain-normalized phonemes            -> correct
      2. Pakistani-accent pattern (accent_rules.py, plain)   -> accent
      3. exact match after canonicalization against the
         primary reference OR any alternative valid
         pronunciation (FUNCTION_WORD_VARIANTS / WORD_VARIANTS) -> correct
      4. confidence gate: below ACOUSTIC_TRUST_CONF a
         non-matching read is not trusted                    -> None
      5. classify from edit operations vs the CLOSEST
         reference, downgrade to 'minor' if close            -> substitution / omission /
                                                                insertion / minor

    Returns None when the acoustic evidence can't decide (empty bucket, too
    low confidence, or — with allow_negative=False — no positive verdict).
    The caller then falls back to the text path (or leaves a skipped word
    skipped).
    """
    if not pred_phon_ipa or avg_conf < ACOUSTIC_MIN_CONF:
        return None

    ref_plain = normalize(ref_phon_ipa)
    pred_plain = normalize(pred_phon_ipa)
    plain_dist = _norm_dist(ref_plain, pred_plain)

    if DEBUG_DETECTION:
        print(f"[DEBUG]   ref={ref_plain} pred={pred_plain}")

    # 1. exact
    if plain_dist < 0.001:
        if DEBUG_DETECTION:
            print(f"[DEBUG]   -> CORRECT (exact acoustic match)")
        return {
            'word': script_word,
            'type': 'correct',
            'severity': 0,
            'ref_phon': ref_phon_ipa,
            'pred_phon': pred_phon_ipa,
            'ref_phon_norm': ref_plain,
            'pred_phon_norm': pred_plain,
            'per_word_score': 100.0
        }

    # 2. accent (always against PLAIN normalized phonemes, as before)
    accent_match = check_accent_pattern(ref_plain, pred_plain)
    if accent_match:
        if DEBUG_DETECTION:
            print(f"[DEBUG]   -> ACCENT ({accent_match}), dist={plain_dist:.3f}")
        return {
            'word': script_word,
            'type': 'accent',
            'severity': SEVERITY['accent'],
            'ref_phon': ref_phon_ipa,
            'pred_phon': pred_phon_ipa,
            'ref_phon_norm': ref_plain,
            'pred_phon_norm': pred_plain,
            'accent_pattern': accent_match,
            'per_word_score': accent_score(plain_dist)
        }

    # 3. canonical / alternative-pronunciation match
    pred_canon = _canon(pred_phon_ipa)
    candidates = [_canon(ref_phon_ipa)] + _VARIANTS_CANON.get(script_word.lower().strip(), [])
    best_ref, best_dist = min(
        ((c, _norm_dist(c, pred_canon)) for c in candidates),
        key=lambda cd: cd[1]
    )

    if best_dist < 0.001:
        if DEBUG_DETECTION:
            print(f"[DEBUG]   -> CORRECT (equivalent symbols / valid variant) ref_used={best_ref}")
        return {
            'word': script_word,
            'type': 'correct',
            'severity': 0,
            'ref_phon': ref_phon_ipa,
            'pred_phon': pred_phon_ipa,
            'ref_phon_norm': best_ref,
            'pred_phon_norm': pred_canon,
            'variant_match': True,
            'per_word_score': 100.0
        }

    if not allow_negative:
        return None

    # 4. confidence gate — confirm freely, condemn only when confident
    if avg_conf < ACOUSTIC_TRUST_CONF:
        if DEBUG_DETECTION:
            print(f"[DEBUG]   -> low confidence ({avg_conf:.3f}) and no match; "
                  f"not trusting acoustic verdict")
        return None

    # 5. genuine difference
    ops = Levenshtein.editops(best_ref, pred_canon)
    error_type = classify_ops(ops, best_ref, pred_canon)
    error_type = _maybe_downgrade_to_minor(error_type, best_dist)
    if DEBUG_DETECTION:
        print(f"[DEBUG]   -> {error_type.upper()} (no accent match), "
              f"dist={best_dist:.3f}, ops={ops}, ref_used={best_ref}")
    return build_result(script_word, error_type, ref_phon_ipa, pred_phon_ipa,
                        best_dist, ref_norm=best_ref, pred_norm=pred_canon)


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
      3. Then compare against ALL valid pronunciations of the word
         (symbol-equivalence + weak forms + British/American variants).
      4. If it's still not a match, classify from editops, then
         downgrade to 'minor' if the phoneme distance is small enough
         that this was a close read rather than a genuinely wrong word
         (see MINOR_DIST_THRESHOLD). Low-confidence reads are never used
         to condemn a word (see ACOUSTIC_TRUST_CONF).
      5. Fall back to the text-level G2P/CMU comparison only when the
         word's timing is unusable (too short, missing) or the bucket
         came back empty/low-confidence. Same accent- and minor-downgrade
         logic applies there too.

    SKIPPED words: a word Whisper dropped (e.g. very short 'I', 'it') has no
    timing, but boundary refinement may still have moved the matching
    phonemes into its bucket. If that bucket matches the word, the word was
    spoken and is returned as correct/accent with 'recovered': True instead
    of 'skipped'.

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

    METHODOLOGY NOTE — boundary refinement:
    Timestamp bucketing still leaks edge phonemes into neighbouring words
    when Whisper's word times are slightly off (common for short function
    words, and for any word Whisper dropped or renamed, e.g. names and
    acronyms). refine_bucket_boundaries() runs after bucketing and moves
    edge phonemes between adjacent buckets only when that strictly lowers
    the combined edit distance to the two words' IPA references. See the
    REFINE_BOUNDARIES switch in the configuration block.

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
      alphabet — never mixed. Canonicalization (_canon) is IPA-only.
    """
    if answer_key is None and script_text is not None:
        answer_key = build_answer_key(script_text)
    if answer_key is None:
        raise ValueError("Must provide either answer_key or script_text")

    _ipa = _ipa_cached

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

    # ── Boundary refinement ──
    if REFINE_BOUNDARIES and len(alignment) > 1:
        try:
            n = len(alignment)
            raw = list(word_phoneme_buckets)
            if len(raw) < n:
                raw += [None] * (n - len(raw))

            phoneme_lists = [list((b or {}).get('phonemes') or []) for b in raw[:n]]
            confs = [float((b or {}).get('avg_confidence') or 0.0) for b in raw[:n]]
            ref_lists = [_ipa(sw) if sw else [] for _, sw in alignment]

            new_lists, new_confs, moves = refine_bucket_boundaries(
                phoneme_lists, ref_lists, confs
            )

            if moves:
                if DEBUG_DETECTION:
                    labels = []
                    for spoken_info, sw in alignment:
                        if sw:
                            labels.append(sw)
                        elif isinstance(spoken_info, dict):
                            labels.append(f"+{spoken_info.get('word', '?')}")
                        else:
                            labels.append('?')
                    print(f"[DEBUG] boundary refinement: {len(moves)} move(s)")
                    for li, ri, direction, moved in moves:
                        src, dst = (li, ri) if direction == 'right' else (ri, li)
                        print(f"[DEBUG]   moved {moved} from '{labels[src]}' -> '{labels[dst]}'")

                word_phoneme_buckets = [
                    {'phonemes': p, 'avg_confidence': c}
                    for p, c in zip(new_lists, new_confs)
                ]
        except Exception as e:
            # Refinement is an improvement, never a requirement: on any
            # failure fall back to the raw timestamp buckets.
            print(f"Warning: boundary refinement failed, using raw buckets: {e}")

    def _bucket(i: int) -> Tuple[list, float]:
        if i < len(word_phoneme_buckets):
            b = word_phoneme_buckets[i]
            if b:
                return list(b.get('phonemes') or []), float(b.get('avg_confidence') or 0.0)
        return [], 0.0

    results = []

    for i, (spoken_info, script_word) in enumerate(alignment):

        # ── Skipped word ──
        if spoken_info is None:
            # Whisper dropped this word, but boundary refinement may have
            # moved its phonemes into the (otherwise empty) bucket. Only a
            # POSITIVE acoustic verdict rescues it — a partial/garbled
            # bucket still counts as skipped.
            recovered = None
            if script_word:
                pred_b, conf_b = _bucket(i)
                if pred_b:
                    if DEBUG_DETECTION:
                        print(f"[DEBUG] '{script_word}': no spoken word aligned, "
                              f"checking refined bucket {pred_b} conf={conf_b:.3f}")
                    recovered = _classify_acoustic(script_word, _ipa(script_word),
                                                   pred_b, conf_b, allow_negative=False)
            if recovered is not None:
                recovered['recovered'] = True
                if DEBUG_DETECTION:
                    print(f"[DEBUG]   -> RECOVERED from acoustic evidence "
                          f"(Whisper dropped it, but it was spoken)")
                results.append(recovered)
                continue

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

        pred_phon_ipa, avg_conf = _bucket(i)

        if DEBUG_DETECTION:
            print(f"[DEBUG] '{script_word}': spoken='{spoken_word}' "
                  f"duration={clip_duration:.3f}s timing_usable={timing_usable} "
                  f"bucket_phonemes={pred_phon_ipa} avg_conf={avg_conf:.3f}")

        acoustic_result = None
        if timing_usable:
            # IPA reference for THIS branch only — never mix with the
            # ARPAbet ref_phon/ref_norm computed above.
            ref_phon_ipa = _ipa(script_word)

            if pred_phon_ipa:
                acoustic_result = _classify_acoustic(script_word, ref_phon_ipa,
                                                     pred_phon_ipa, avg_conf)
                # None => too low confidence / untrusted: fall through to
                # the text-based fallback below.
            elif script_word.lower().strip() in _WEAK_WORDS:
                # Short function word with an empty bucket: Whisper heard
                # it, CTC just didn't emit a ~50ms schwa. Not an omission.
                if DEBUG_DETECTION:
                    print(f"[DEBUG]   -> empty bucket on weak word, using text fallback")
            else:
                # Bucket came back empty — no phonemes from the utterance
                # decode landed in this word's time range at all.
                if DEBUG_DETECTION:
                    print(f"[DEBUG]   -> OMISSION (empty bucket)")
                acoustic_result = build_result(script_word, 'omission', ref_phon_ipa,
                                                [], 1.0)

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
                 pred_phon: list, norm_dist: float,
                 ref_norm: Optional[list] = None,
                 pred_norm: Optional[list] = None) -> dict:
    """
    Build the standard result dict for one word.

    ref_norm / pred_norm let the acoustic branch pass its canonicalized
    (closest-variant) lists so editops, severity and per_word_score are all
    computed on the same lists the verdict was based on. When omitted they
    default to plain normalize(), exactly as before (text-fallback branch).
    """
    if ref_norm is None:
        ref_norm = normalize(ref_phon)
    if pred_norm is None:
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