import Levenshtein
from typing import List, Dict

# Grade thresholds: (min_score, label, letter)
GRADE_MAP = [
    (90, 'Excellent',      'A+'),
    (80, 'Very Good',      'A'),
    (70, 'Good',           'B'),
    (60, 'Needs Practice', 'C'),
    (0,  'Focus Required', 'D'),
]


def _get_grade(accuracy: float) -> tuple:
    """Get grade label and letter for a given accuracy score."""
    for threshold, label, grade in GRADE_MAP:
        if accuracy >= threshold:
            return label, grade
    return 'Focus Required', 'D'


def calculate_score(errors: List[Dict]) -> Dict:
    """
    Session score = weighted average of per_word_score across all SCOREABLE words.

    IMPORTANT: 'inserted' words (extra words the user said that aren't in the
    script) are NEVER part of the weighted average. They are tallied separately
    in `inserted_words` for display purposes only. Including them in the
    weighted sum was the root cause of the earlier score/word-count mismatch
    bug (score dropping / total inflating whenever a user rambled extra words).
    """
    if not errors:
        return {
            'score': 0.0, 'grade': 'D', 'grade_label': 'Focus Required',
            'severity_total': 0, 'severity_avg': 0.0,
            'total_words': 0, 'scored_words': 0,
            'skipped_words': 0, 'flagged_words': 0, 'inserted_words': 0,
            'breakdown': []
        }

    total_weight = 0
    weighted_sum = 0.0
    severity_total = 0
    skipped = 0
    flagged = 0
    inserted = 0
    breakdown = []

    for e in errors:
        word = e.get('word', 'unknown')
        err_type = e.get('type', 'unknown')
        ref_phon = e.get('ref_phon', [])
        # NOTE: error_detector.build_result() stores predicted phonemes under
        # 'pred_phon', not 'hyp_phon'. The old key was always empty, so every
        # breakdown entry silently reported the user's phonemes as [].
        pred_phon = e.get('pred_phon', [])
        severity = e.get('severity', 0)

        # ── Inserted words: tally only, NEVER enter the weighted average ──
        if err_type == 'inserted':
            inserted += 1
            breakdown.append({
                'word': word,
                'type': err_type,
                'score': 0.0,
                'ref_phonemes': [],
                'hyp_phonemes': pred_phon,
                'severity': 0,
                'phoneme_count': 0
            })
            continue

        phon_count = max(len(ref_phon), 1)

        if err_type == 'skipped':
            skipped += 1
            word_score = 0.0
        elif err_type == 'correct':
            word_score = e.get('per_word_score', 100.0)
        elif err_type == 'accent':
            # Forgiven Pakistani-accent variant — already scored generously
            # by error_detector (min(sim+8, 95%)). Do not re-flag as an error.
            word_score = e.get('per_word_score', 90.0)
        elif err_type in ('substituted', 'partial', 'substitution', 'omission', 'insertion', 'stress'):
            word_score = e.get('per_word_score')
            if word_score is None:
                from .error_detector import per_word_score_from_lists
                word_score = per_word_score_from_lists(ref_phon, pred_phon)
            flagged += 1
        else:
            word_score = e.get('per_word_score', 0.0)
            flagged += 1

        weighted_sum += word_score * phon_count
        total_weight += phon_count
        severity_total += severity

        breakdown.append({
            'word': word,
            'type': err_type,
            'score': round(word_score, 1),
            'ref_phonemes': ref_phon,
            'hyp_phonemes': pred_phon,
            'severity': severity,
            'phoneme_count': phon_count
        })

    accuracy = round(weighted_sum / total_weight, 1) if total_weight > 0 else 0.0
    grade_label, grade = _get_grade(accuracy)

    # total_words / scored_words EXCLUDE inserted words by construction now,
    # since inserted entries never reach the phon_count/weighted_sum logic.
    scored_words = len(errors) - inserted
    severity_avg = round(severity_total / scored_words, 2) if scored_words > 0 else 0.0

    return {
        'score': accuracy,
        'grade': grade,
        'grade_label': grade_label,
        'severity_total': severity_total,
        'severity_avg': severity_avg,
        'total_words': scored_words,
        'scored_words': scored_words,
        'skipped_words': skipped,
        'flagged_words': flagged,
        'inserted_words': inserted,
        'breakdown': breakdown
    }


def letter_grade(score: float) -> str:
    """Get letter grade from a raw score (0-100)."""
    _, grade = _get_grade(score)
    return grade


def grade_label(score: float) -> str:
    """Get descriptive label from a raw score (0-100)."""
    label, _ = _get_grade(score)
    return label


def quick_score(errors: List[Dict]) -> float:
    """Return just the numeric score without full breakdown."""
    return calculate_score(errors)['score']