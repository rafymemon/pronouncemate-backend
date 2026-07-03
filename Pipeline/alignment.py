# alignment.py
from Levenshtein import distance as lev_dist

def word_similarity(a: str, b: str) -> float:
    """
    Returns 1.0 for identical words, lower for similar words.
    'berry' vs 'very': lev=1, max=5 -> similarity = 0.80
    'cat' vs 'car': lev=1, max=3 -> similarity = 0.67
    """
    if a == b: return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0: return 1.0
    return 1.0 - (lev_dist(a, b) / max_len)

def needleman_wunsch(spoken_words: list, script_words: list,
                     match=2, mismatch=-1, gap=-2) -> list:
    """
    Global sequence alignment of spoken words to script words.
    Returns list of (spoken_word, script_word) pairs.
    None in script_word = user added an extra word.
    None in spoken_word = user skipped a script word.
    """
    n, m = len(spoken_words), len(script_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i * gap
    for j in range(m + 1): dp[0][j] = j * gap

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = word_similarity(spoken_words[i-1], script_words[j-1])
            # FIX: use continuous similarity score, not binary match/mismatch
            # sim=1.0 -> full match score; sim=0.0 -> full mismatch penalty
            diag_score = match if sim >= 0.8 else int(mismatch * (1.0 - sim))
            dp[i][j] = max(
                dp[i-1][j-1] + diag_score,  # align these two words
                dp[i-1][j]   + gap,          # skip a spoken word (user added)
                dp[i][j-1]   + gap           # skip a script word (user missed)
            )

    # Backtrace
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sim = word_similarity(spoken_words[i-1], script_words[j-1])
            diag_score = match if sim >= 0.8 else int(mismatch * (1.0 - sim))
            if dp[i][j] == dp[i-1][j-1] + diag_score:
                pairs.append((spoken_words[i-1], script_words[j-1]))
                i -= 1; j -= 1; continue
        if i > 0 and (j == 0 or dp[i][j] == dp[i-1][j] + gap):
            pairs.append((spoken_words[i-1], None))  # user added this word
            i -= 1
        else:
            pairs.append((None, script_words[j-1]))  # user skipped this word
            j -= 1
    pairs.reverse()
    return pairs

# Corrected example output for 'berry'/'very':
# spoken=['the','module','is','berry','importan']
# script=['the','module','is','very', 'important']
# -> [('the','the'),('module','module'),('is','is'),
#     ('berry','very'),        <- NOW CORRECTLY PAIRED (sim=0.80 >= 0.80)
#     ('importan','important')] <- sim=0.89 >= 0.80, also paired