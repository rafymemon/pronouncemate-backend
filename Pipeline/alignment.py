from Levenshtein import distance as lev_dist
from typing import List, Tuple, Optional


def word_similarity(a: str, b: str) -> float:
    """
    Returns 1.0 for identical words, lower for similar words.
    Uses Levenshtein distance normalized by max length.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    
    max_len = max(len(a), len(b))
    dist = lev_dist(a, b)
    return 1.0 - (dist / max_len)


def needleman_wunsch(
    spoken_words: List[str],
    script_words: List[str],
    match: float = 2.0,
    mismatch: float = -1.5,
    gap_spoken: float = -1.0,   # user added extra word (less severe)
    gap_script: float = -2.5,   # user missed script word (more severe)
    min_similarity: float = 0.6  # minimum to consider an alignment
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
    
    # Initialize DP table
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_spoken
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap_script
    
    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = word_similarity(spoken_words[i - 1], script_words[j - 1])
            
            # Continuous scoring: higher similarity = higher score
            if sim >= min_similarity:
                # Scale match score by similarity (partial credit for near-matches)
                diag_score = match * sim
            else:
                # Full mismatch penalty for very different words
                diag_score = mismatch
            
            dp[i][j] = max(
                dp[i - 1][j - 1] + diag_score,  # align these two words
                dp[i - 1][j] + gap_spoken,      # skip spoken word (user added)
                dp[i][j - 1] + gap_script       # skip script word (user missed)
            )
    
    # Backtrace
    pairs = []
    i, j = n, m
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sim = word_similarity(spoken_words[i - 1], script_words[j - 1])
            diag_score = match * sim if sim >= min_similarity else mismatch
            
            # Check if we came from diagonal
            if abs(dp[i][j] - (dp[i - 1][j - 1] + diag_score)) < 0.001:
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
    use_phonetic_similarity: bool = False
) -> List[Tuple[Optional[dict], Optional[str]]]:
    """
    Higher-level alignment that works with Whisper output dicts.
    
    Args:
        spoken_words: list of {word, start, end, prob} from Whisper
        script_words: list of words from reference script
        use_phonetic_similarity: if True, use phoneme similarity instead of string
    
    Returns:
        List of (spoken_info_dict, script_word) pairs.
        None in either position indicates a gap.
    """
    spoken_strs = [w['word'] for w in spoken_words]
    alignment = needleman_wunsch(spoken_strs, script_words)
    
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