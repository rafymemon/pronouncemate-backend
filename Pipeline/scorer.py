# scorer.py
import Levenshtein
from .error_detector import normalize, per_word_score

GRADE_MAP = [
    (90, 'Excellent',      'A+'),
    (80, 'Very Good',      'A'),
    (70, 'Good',           'B'),
    (60, 'Needs Practice', 'C'),
    (0,  'Focus Required', 'D'),
]

def calculate_score(errors: list) -> dict:
    """
    Session score = weighted average of per_word_score across all scored words.
    Weighting by phoneme count: longer words contribute more to the score.
    Skipped words count as 0 but are included in the denominator.
    Returns dict with accuracy, grade, severity_total, and breakdown.
    """
    total_weight = 0
    weighted_sum = 0.0
    severity_total = 0
    skipped = 0
    flagged = 0

    for e in errors:
        if e['type'] == 'inserted':
            continue  # user-added words don’t count against score

        phon_count = max(len(e.get('ref_phon', [])), 1)
        
        # --- THE FIX ---
        if e['type'] == 'skipped':
            word_score = 0.0  # Skipped words strictly get 0 points
        else:
            # For substituted/correct words, trust the upstream score. 
            # If it's missing, default to 100, but log a warning if you want to track bugs.
            word_score = e.get('per_word_score', 100.0)

        weighted_sum  += word_score * phon_count
        total_weight  += phon_count
        severity_total += e.get('severity', 0)

        if e['type'] == 'skipped': skipped += 1
        elif e['type'] != 'correct': flagged += 1

    accuracy = round(weighted_sum / total_weight, 1) if total_weight > 0 else 0.0

    grade_label, grade = 'Focus Required', 'D'
    for threshold, label, g in GRADE_MAP:
        if accuracy >= threshold:
            grade_label, grade = label, g
            break

    return {
        'score':         accuracy,
        'grade':         grade,
        'grade_label':   grade_label,
        'severity_total': severity_total,  # for UI breakdown display
        'total_words':   len(errors),
        'skipped_words': skipped,
        'flagged_words': flagged
    }

def letter_grade(score: float) -> str:
    """Convenience wrapper returning just the letter grade."""
    return calculate_score([{'type':'correct','per_word_score':score,
                             'ref_phon':['X'],'severity':0}])['grade']