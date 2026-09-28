from Levenshtein import distance as lev_dist
from typing import Callable, Dict, List, Optional, Tuple
import re


# ── Word similarity ───────────────────────────────────────────

def _osa_distance(a: str, b: str) -> int:
    """
    Optimal-string-alignment (restricted Damerau-Levenshtein) distance:
    like Levenshtein, but swapping two ADJACENT characters costs 1 instead
    of 2. ASR output very often differs from the script by exactly such a
    swap ("mute" vs "muet"), and plain Levenshtein scored that pair 0.5,
    below the alignment's match threshold.
    """
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j

    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,          # deletion
                d[i][j - 1] + 1,          # insertion
                d[i - 1][j - 1] + cost,   # substitution
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)   # adjacent swap
    return d[la][lb]


def word_similarity(a: str, b: str) -> float:
    """
    Returns 1.0 for identical words, lower for similar words.
    Character-level: OSA distance (Levenshtein + adjacent swaps) normalized
    by max length.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    max_len = max(len(a), len(b))
    dist = _osa_distance(a, b)
    return 1.0 - (dist / max_len)


# ── Phonetic similarity (used inside needleman_wunsch only) ───
# ASR often returns a different SPELLING of what was said ("mute" for
# "MUET", "eye" for "I"). Comparing the words' phonemes catches those,
# which character similarity alone can't. Phonemes come from g2p_module
# (CMU dict / g2p-en) and are compared as stress-stripped ARPAbet.
PHONETIC_DISCOUNT = 0.95   # exact-spelling matches always beat phonetic ones

_STRESS_RE = re.compile(r'[012]')


def _load_phoneme_fn() -> Optional[Callable[[str], list]]:
    """Lazy import so this module still works standalone / in unit tests."""
    try:
        from .g2p_module import get_phonemes
        return get_phonemes
    except Exception:
        return None


def _build_similarity_fn(phoneme_fn: Optional[Callable[[str], list]]
                         ) -> Callable[[str, str], float]:
    """
    Returns a cached similarity(a, b) = max(character similarity,
    discounted phonetic similarity). Falls back to character similarity
    alone if the phoneme function is unavailable or fails for a word.
    """
    phon_cache: Dict[str, Optional[list]] = {}
    pair_cache: Dict[Tuple[str, str], float] = {}

    def phonemes(word: str) -> Optional[list]:
        if word not in phon_cache:
            result = None
            if phoneme_fn is not None:
                try:
                    raw = phoneme_fn(word) or []
                    result = [_STRESS_RE.sub('', p).lower().strip() for p in raw if p and p.strip()]
                except Exception:
                    result = None
            phon_cache[word] = result or None
        return phon_cache[word]

    def similarity(a: str, b: str) -> float:
        key = (a, b)
        if key in pair_cache:
            return pair_cache[key]

        sim = word_similarity(a, b)
        if sim < 1.0 and phoneme_fn is not None:
            pa, pb = phonemes(a), phonemes(b)
            if pa and pb:
                max_len = max(len(pa), len(pb))
                phon_sim = 1.0 - (lev_dist(pa, pb) / max_len)
                sim = max(sim, phon_sim * PHONETIC_DISCOUNT)

        pair_cache[key] = sim
        return sim

    return similarity


# ── Global alignment ──────────────────────────────────────────

def needleman_wunsch(
    spoken_words: List[str],
    script_words: List[str],
    match: float = 2.0,
    mismatch: float = -1.5,
    gap_spoken: float = -1.0,   # user added extra word (less severe)
    gap_script: float = -2.5,   # user missed script word (more severe)
    min_similarity: float = 0.6,  # minimum to consider an alignment
    use_phonetic_similarity: bool = True
) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Global sequence alignment of spoken words to script words.

    Args:
        spoken_words: words from Whisper transcription
        script_words: words from reference script
        match: score for identical words
        mismatch: base penalty for different words
        gap_spoken: penalty for skipping a spoken word (user inserted)
        gap_script: penalty for skipping a script word (user missed)
        min_similarity: minimum similarity to use match score
        use_phonetic_similarity: also compare the words' phonemes, so an ASR
            respelling of a correctly spoken word still aligns to it

    Returns:
        List of (spoken_word, script_word) pairs.
        None in script_word = user added an extra word.
        None in spoken_word = user skipped a script word.
    """
    n, m = len(spoken_words), len(script_words)

    # Edge cases
    if n == 0 and m == 0:
        return []
    if n == 0:
        return [(None, w) for w in script_words]
    if m == 0:
        return [(w, None) for w in spoken_words]

    similarity = _build_similarity_fn(_load_phoneme_fn() if use_phonetic_similarity else None)

    def diag(i: int, j: int) -> float:
        sim = similarity(spoken_words[i - 1], script_words[j - 1])
        # Continuous scoring: higher similarity = higher score
        # (partial credit for near-matches); full penalty for very
        # different words.
        return match * sim if sim >= min_similarity else mismatch

    # Initialize DP table
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_spoken
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap_script

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = max(
                dp[i - 1][j - 1] + diag(i, j),  # align these two words
                dp[i - 1][j] + gap_spoken,      # skip spoken word (user added)
                dp[i][j - 1] + gap_script       # skip script word (user missed)
            )

    # Backtrace
    pairs = []
    i, j = n, m

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            # Check if we came from diagonal
            if abs(dp[i][j] - (dp[i - 1][j - 1] + diag(i, j))) < 0.001:
                pairs.append((spoken_words[i - 1], script_words[j - 1]))
                i -= 1
                j -= 1
                continue

        # Check if we came from top (skipped spoken word)
        if i > 0 and abs(dp[i][j] - (dp[i - 1][j] + gap_spoken)) < 0.001:
            pairs.append((spoken_words[i - 1], None))
            i -= 1
            continue

        # Must have come from left (skipped script word)
        if j > 0:
            pairs.append((None, script_words[j - 1]))
            j -= 1

    pairs.reverse()
    return pairs


def align_words(
    spoken_words: List[dict],
    script_words: List[str],
    use_phonetic_similarity: bool = True
) -> List[Tuple[Optional[dict], Optional[str]]]:
    """
    Higher-level alignment that works with Whisper output dicts.

    Args:
        spoken_words: list of {word, start, end, prob} from Whisper
        script_words: list of words from reference script
        use_phonetic_similarity: if True, also use phoneme similarity

    Returns:
        List of (spoken_info_dict, script_word) pairs.
        None in either position indicates a gap.
    """
    spoken_strs = [w['word'] for w in spoken_words]
    alignment = needleman_wunsch(
        spoken_strs, script_words,
        use_phonetic_similarity=use_phonetic_similarity
    )

    # Map back to full dicts
    spoken_idx = 0
    result = []

    for spoken_str, script_str in alignment:
        if spoken_str is not None:
            # Find the corresponding dict (handles duplicates via index)
            while spoken_idx < len(spoken_words) and spoken_words[spoken_idx]['word'] != spoken_str:
                spoken_idx += 1
            if spoken_idx < len(spoken_words):
                result.append((spoken_words[spoken_idx], script_str))
                spoken_idx += 1
            else:
                # Shouldn't happen, but fallback
                result.append(({'word': spoken_str, 'start': 0, 'end': 0, 'prob': 0}, script_str))
        else:
            result.append((None, script_str))

    return result


def get_alignment_stats(alignment: List[Tuple]) -> dict:
    """Get statistics about the alignment for debugging/tuning."""
    stats = {
        'total_pairs': len(alignment),
        'correct': 0,
        'substitutions': 0,
        'insertions': 0,
        'deletions': 0,
        'similarity_sum': 0.0
    }

    for spoken, script in alignment:
        if spoken is None:
            stats['deletions'] += 1
        elif script is None:
            stats['insertions'] += 1
        else:
            spoken_word = spoken['word'] if isinstance(spoken, dict) else spoken
            sim = word_similarity(spoken_word, script)
            stats['similarity_sum'] += sim
            if sim >= 0.95:
                stats['correct'] += 1
            else:
                stats['substitutions'] += 1

    aligned = stats['correct'] + stats['substitutions']
    stats['avg_similarity'] = round(stats['similarity_sum'] / aligned, 3) if aligned > 0 else 0.0

    return stats