"""
practice_mistakes.py

Retry with Mistake Words Feature
================================
After a session, allow the user to practice only the words they got wrong.
Creates a mini-script from flagged words and runs a focused practice session.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from .pipeline import run_pipeline
from .scorer import calculate_score

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────
MIN_WORDS_FOR_PRACTICE = 3
MAX_WORDS_FOR_PRACTICE = 15
PRACTICE_PASS_THRESHOLD = 85.0


@dataclass
class MistakeWord:
    """Represents a single word the user got wrong."""
    word: str
    error_type: str
    severity: int
    per_word_score: float
    ref_phonemes: List[str]
    pred_phonemes: List[str]
    explanation: str
    audio_path: Optional[str]
    attempt_count: int = 1
    best_score: float = 0.0
    cleared: bool = False


def extract_mistake_words(errors: List[Dict],
                          min_severity: int = 0,
                          exclude_types: tuple = ('inserted',)) -> List[MistakeWord]:
    """
    Extract words that were flagged from a session's error list.
    """
    mistakes = []

    for e in errors:
        err_type = e.get('type', '')
        severity = e.get('severity', 0)

        if err_type == 'correct' or err_type in exclude_types:
            continue

        if severity < min_severity:
            continue

        mw = MistakeWord(
            word=e.get('word', ''),
            error_type=err_type,
            severity=severity,
            per_word_score=e.get('per_word_score', 0.0),
            ref_phonemes=e.get('ref_phon', []),
            pred_phonemes=e.get('pred_phon', []),
            explanation=e.get('explanation', ''),
            audio_path=e.get('audio_path', None)
        )
        mistakes.append(mw)

    mistakes.sort(key=lambda x: (-x.severity, x.per_word_score))
    return mistakes


def generate_practice_script(mistake_words: List[MistakeWord],
                             mode: str = 'sentences') -> str:
    """
    Generate a practice script from mistake words.

    Args:
        mistake_words: List of words to practice
        mode: 'sentences' | 'words' | 'minimal_pairs'

    Returns:
        A practice script string
    """
    words = [mw.word for mw in mistake_words[:MAX_WORDS_FOR_PRACTICE]]

    if mode == 'words':
        return ' '.join(words)

    elif mode == 'sentences':
        carriers = [
            "Please say the word {word}.",
            "I want to hear you say {word}.",
            "Can you pronounce {word} clearly?",
            "Repeat after me: {word}.",
            "Let's practice the word {word}.",
        ]

        sentences = []
        for i, word in enumerate(words):
            carrier = carriers[i % len(carriers)]
            sentences.append(carrier.format(word=word))

        return ' '.join(sentences)

    elif mode == 'minimal_pairs':
        sentences = []
        for mw in mistake_words[:MAX_WORDS_FOR_PRACTICE]:
            if mw.error_type == 'substitution' and len(mw.pred_phonemes) > 0:
                sentences.append(f"Say '{mw.word}' clearly: {mw.word}.")
            else:
                sentences.append(f"Please say: {mw.word}.")
        return ' '.join(sentences)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_practice_session(raw_audio_path: str,
                         mistake_words: List[MistakeWord],
                         original_session_id: str = None,
                         mode: str = 'sentences') -> Dict:
    """
    Run a focused practice session on mistake words.
    """
    practice_script = generate_practice_script(mistake_words, mode=mode)

    logger.info(f"Practice session starting with {len(mistake_words)} words, mode='{mode}'")
    logger.info(f"Practice script: {practice_script[:100]}...")

    result = run_pipeline(raw_audio_path, practice_script)

    if not result.get('success', False):
        logger.error("Practice session pipeline failed")
        return {
            'success': False,
            'error': result.get('error', 'Practice session failed'),
            'practice_script': practice_script,
            'original_session_id': original_session_id
        }

    # Analyze improvement per word
    word_results = result.get('words', [])
    cleared_words = []
    still_struggling = []
    improved_words = []

    for mw in mistake_words:
        practice_word = next(
            (w for w in word_results if w['word'] == mw.word),
            None
        )

        if practice_word is None:
            still_struggling.append({
                'word': mw.word,
                'status': 'missing',
                'previous_score': mw.per_word_score,
                'current_score': 0.0,
                'improvement': 0.0
            })
            continue

        current_score = practice_word.get('per_word_score', 0.0)
        previous_score = mw.per_word_score
        improvement = current_score - previous_score

        word_status = {
            'word': mw.word,
            'previous_score': previous_score,
            'current_score': current_score,
            'improvement': round(improvement, 1),
            'error_type': practice_word.get('type', 'unknown'),
            'severity': practice_word.get('severity', 0)
        }

        if current_score >= PRACTICE_PASS_THRESHOLD:
            word_status['status'] = 'cleared'
            cleared_words.append(word_status)
        elif improvement > 5.0:
            word_status['status'] = 'improved'
            improved_words.append(word_status)
        else:
            word_status['status'] = 'still_struggling'
            still_struggling.append(word_status)

    practice_result = {
        'success': True,
        'session_type': 'practice',
        'original_session_id': original_session_id,
        'practice_script': practice_script,
        'mode': mode,
        'score': result['score'],
        'grade': result['grade'],
        'grade_label': result['grade_label'],
        'cleared_words': cleared_words,
        'improved_words': improved_words,
        'still_struggling': still_struggling,
        'cleared_count': len(cleared_words),
        'improved_count': len(improved_words),
        'struggling_count': len(still_struggling),
        'total_practiced': len(mistake_words),
        'words': word_results,
        'feedback': result.get('feedback', []),
        'stats': result.get('stats', {}),
        'message': _generate_encouragement(
            len(cleared_words),
            len(improved_words),
            len(still_struggling)
        )
    }

    logger.info(
        f"Practice complete: {len(cleared_words)} cleared, "
        f"{len(improved_words)} improved, {len(still_struggling)} still struggling"
    )

    return practice_result


def _generate_encouragement(cleared: int, improved: int, struggling: int) -> str:
    """Generate an encouraging message based on practice results."""
    total = cleared + improved + struggling

    if cleared == total and total > 0:
        return "Excellent! You cleared all words. You are making great progress!"
    elif cleared >= total * 0.5:
        return f"Great job! You cleared {cleared} out of {total} words. Keep practicing the rest!"
    elif improved > 0:
        return f"Good effort! {improved} words improved. Practice the remaining {struggling} words again."
    elif struggling > 0:
        return "Keep trying! Focus on the tricky sounds. Listen to the feedback audio carefully."
    else:
        return "Practice complete! Review your results and try again."


def should_offer_practice(errors: List[Dict]) -> bool:
    """Determine if the 'Practice Mistakes' button should be shown."""
    mistake_words = extract_mistake_words(errors)
    return len(mistake_words) >= MIN_WORDS_FOR_PRACTICE


def get_mistake_summary(errors: List[Dict]) -> Dict:
    """Get a quick summary of mistakes for the UI 'Practice Mistakes' button."""
    mistakes = extract_mistake_words(errors)

    severity_breakdown = {
        'critical': len([m for m in mistakes if m.severity >= 10]),
        'major': len([m for m in mistakes if 7 <= m.severity < 10]),
        'minor': len([m for m in mistakes if 3 <= m.severity < 7]),
        'notice': len([m for m in mistakes if m.severity < 3])
    }

    type_counts = {}
    for m in mistakes:
        type_counts[m.error_type] = type_counts.get(m.error_type, 0) + 1
    most_common_type = max(type_counts, key=type_counts.get) if type_counts else None

    return {
        'mistake_count': len(mistakes),
        'offer_practice': len(mistakes) >= MIN_WORDS_FOR_PRACTICE,
        'severity_breakdown': severity_breakdown,
        'most_common_error': most_common_type,
        'words': [m.word for m in mistakes[:5]],
        'top_mistakes': [
            {
                'word': m.word,
                'error_type': m.error_type,
                'score': m.per_word_score,
                'explanation': m.explanation
            }
            for m in mistakes[:3]
        ]
    }