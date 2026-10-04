import logging
import time
from typing import List, Dict
import uuid
from .audio_processor import preprocess_audio, slice_word_audio
from .whisper_asr import transcribe, transcribe_for_analysis
from .g2p_module import build_answer_key, build_script_word_list
from .alignment import needleman_wunsch
from .error_detector import detect_errors
from .scorer import calculate_score
from .feedback import generate_explanation, get_feedback_audio, get_feedback_audio_batch
from .phoneme_display import to_display_ipa

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────
MIN_AUDIO_DURATION = 0.5   # seconds — reject if too short
MAX_AUDIO_DURATION = 300.0  # seconds — 5 minutes max

# 'minor' = a close-but-not-exact read (see error_detector.MINOR_DIST_THRESHOLD).
# MINOR_COUNTS_AS_CORRECT: if True, minor words are counted as correct in the
#   headline tallies (stats.correct_words / wrong_words / word_accuracy), so a
#   mostly-right read isn't shown as a pile of "errors". They are still listed
#   (amber) and still lower the weighted score via their per_word_score.
# INCLUDE_MINOR_IN_PRACTICE: if True, minor words also feed the Practice
#   Mistakes list and the Focus Script. Imported by practice_mistakes.py so
#   both places always agree.
MINOR_COUNTS_AS_CORRECT = True
INCLUDE_MINOR_IN_PRACTICE = False


def run_pipeline(raw_audio_path: str, reference_script: str) -> dict:
    """
    Full pipeline: audio -> transcript -> alignment -> errors -> score -> feedback.

    Args:
        raw_audio_path: path to WAV file from Flutter upload
        reference_script: the script the user was asked to read

    Returns:
        dict with score, grade, transcript, words list, feedback list, mistake_summary
    """
    t_start = time.perf_counter()
    timings: Dict[str, float] = {}

    def _lap(name: str, since: float) -> float:
        """Record the elapsed time for a stage and return the new 'now'."""
        now = time.perf_counter()
        timings[name] = now - since
        return now

    # ── Stage 1: Preprocess audio ──
    t = time.perf_counter()
    try:
        audio = preprocess_audio(raw_audio_path)
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        return _error_response("Audio preprocessing failed")
    t = _lap('preprocess', t)

    audio_duration = len(audio) / 16000
    logger.info(f'Preprocessed: {audio_duration:.2f}s audio')

    if audio_duration < MIN_AUDIO_DURATION:
        return _error_response(f"Audio too short ({audio_duration:.1f}s). Please speak for at least {MIN_AUDIO_DURATION}s.")
    if audio_duration > MAX_AUDIO_DURATION:
        return _error_response(f"Audio too long ({audio_duration:.1f}s). Maximum is {MAX_AUDIO_DURATION/60:.0f} minutes.")

    # ── Stage 2: Transcribe with timestamps ──
    try:
        transcript = transcribe_for_analysis(audio, reference_script=reference_script)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return _error_response("Transcription failed. Please try again with clearer audio.")
    t = _lap('transcribe', t)

    logger.info(f'Transcribed: {len(transcript)} words')

    if not transcript:
        return _error_response("No speech detected. Please try again.")

    # ── Stage 3: Build answer key from reference script ──
    try:
        answer_key = build_answer_key(reference_script)
    except Exception as e:
        logger.error(f"Answer key generation failed: {e}")
        return _error_response("Failed to process reference script")
    t = _lap('answer_key', t)

    logger.info(f'Answer key: {len(answer_key)} words')

    if not answer_key:
        return _error_response("Reference script is empty or invalid")

    # ── Stage 4: Align spoken words to script words ──
    spoken_words = [w['word'] for w in transcript]
    script_words = build_script_word_list(reference_script)   # ← was list(answer_key.keys())
    try:
        alignment = needleman_wunsch(spoken_words, script_words)
    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        return _error_response("Word alignment failed")

    alignment_with_timing = _build_alignment_with_timing(alignment, transcript)
    t = _lap('align', t)

    # ── Stage 5: Detect errors ──
    try:
        errors = detect_errors(
            alignment=alignment_with_timing,
            audio=audio,
            sr=16000,
            answer_key=answer_key
        )
    except Exception as e:
        logger.error(f"Error detection failed: {e}")
        return _error_response("Error detection failed")
    t = _lap('detect_errors', t)

    # Log honest counts: the old line counted every non-'correct' word
    # (minor slips, forgiven accent variants, ...) as an "error".
    _types = [e['type'] for e in errors if e['type'] != 'inserted']
    minor_n = _types.count('minor')
    accent_n = _types.count('accent')
    hard_n = len([x for x in _types if x not in ('correct', 'minor', 'accent')])
    logger.info(
        f'Detected {hard_n} errors, {minor_n} minor, {accent_n} accent '
        f'out of {len(_types)} words'
    )

    # ── Stage 6: Score and grade ──
    try:
        score_result = calculate_score(errors)
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        return _error_response("Scoring failed")
    t = _lap('score', t)

    # ── Stage 7: Build feedback ──
    # IMPORTANT: this list still includes 'inserted' entries (Flutter filters
    # them client-side via script-word matching), but every COUNT below is
    # computed from the same scoreable set the scorer used, so score and
    # word-count are always consistent with each other.
    words_output, feedback = _build_feedback(errors)
    t = _lap('feedback_tts', t)

    # ── Stage 8: Build mistake summary for Practice Mistakes feature ──
    mistake_summary = _build_mistake_summary(errors)

    # ── Stage 9: Compute the exact stats Flutter's summary screen expects ──
    # These MUST be derived from the same `errors` list the scorer consumed,
    # never recomputed independently — that divergence was the original bug.
    scoreable = [e for e in errors if e['type'] != 'inserted']
    passing_types = ('correct', 'accent') + (('minor',) if MINOR_COUNTS_AS_CORRECT else ())
    correct_count = len([e for e in scoreable if e['type'] in passing_types])
    total_scoreable = len(scoreable)
    wrong_count = total_scoreable - correct_count
    word_accuracy = round((correct_count / total_scoreable) * 100, 1) if total_scoreable else 0.0

    timings['total'] = time.perf_counter() - t_start
    logger.info(
        '[timing] ' + ' '.join(f'{name}={secs:.2f}s' for name, secs in timings.items())
    )

    return {
        'success': True,
        'session_id': uuid.uuid4().hex[:12],
        'score': score_result['score'],
        'grade': score_result['grade'],
        'grade_label': score_result['grade_label'],
        'transcript': [w['word'] for w in transcript],
        'script_words': script_words,
        'words': words_output,
        'feedback': feedback,
        'mistake_summary': mistake_summary,
        'stats': {
            # Existing fields (kept for backward compatibility with any
            # other consumers relying on these names)
            'total_words': score_result.get('total_words', 0),
            'skipped_words': score_result.get('skipped_words', 0),
            'flagged_words': score_result.get('flagged_words', 0),
            'inserted_words': score_result.get('inserted_words', 0),
            'minor_words': score_result.get('minor_words', 0),
            'severity_total': score_result.get('severity_total', 0),
            'audio_duration': round(audio_duration, 2),

            # Fields active_session_screen.dart / session_summary_screen.dart
            # actually read. Without these, Flutter's `?? 0` fallback masked
            # the missing keys with a literal 0, so the summary screen always
            # displayed "0 correct / 0 errors" regardless of real performance.
            'correct_words': correct_count,
            'wrong_words': wrong_count,
            'word_accuracy': word_accuracy,
        }
    }


def _build_alignment_with_timing(alignment: list, transcript: list) -> list:
    """Build alignment with proper timing info, handling duplicate words."""
    used_indices = set()
    alignment_with_timing = []

    for spoken_word, script_word in alignment:
        if spoken_word is None:
            alignment_with_timing.append((None, script_word))
            continue

        if script_word is None:
            timing = _find_unused_timing(spoken_word, transcript, used_indices)
            alignment_with_timing.append((timing, None))
            continue

        timing = _find_unused_timing(spoken_word, transcript, used_indices)
        alignment_with_timing.append((timing, script_word))

    return alignment_with_timing



def _find_unused_timing(word: str, transcript: list, used_indices: set) -> dict:
    """Find the next unused transcript entry matching this word."""
    for i, entry in enumerate(transcript):
        if i not in used_indices and entry['word'] == word:
            used_indices.add(i)
            return entry

    logger.warning(f"Word '{word}' not found in unused transcript entries")

    return {'word': word, 'start': 0.0, 'end': 0.0, 'prob': 0.0}



def _build_feedback(errors: list) -> tuple:
    """Build words_output and feedback lists from error detections."""
    words_output = []
    feedback = []

    # Generate TTS audio for all flagged words up front: deduplicated,
    # in parallel, and time-bounded (see feedback.get_feedback_audio_batch).
    # Previously this happened one word at a time inside the loop below,
    # which dominated /api/analyze/ latency on sessions with many errors.
    # 'accent' words are forgiven, so they get no TTS "correction" audio.
    words_needing_audio = [
        e['word'] for e in errors
        if e['type'] not in ('correct', 'inserted', 'accent') and e.get('word')
    ]
    try:
        audio_map = get_feedback_audio_batch(words_needing_audio)
    except Exception as ex:
        logger.warning(f"TTS batch generation failed: {ex}")
        audio_map = {}

    for e in errors:
        # Display-only conversion — scoring already happened upstream
        # using whichever alphabet (ARPAbet or IPA) each path produced.
        ref_display = to_display_ipa(e.get('ref_phon', []))
        pred_display = to_display_ipa(e.get('pred_phon', []))

        word_entry = {
            'word': e['word'],
            'type': e['type'],
            'per_word_score': e.get('per_word_score', 100.0),
            'severity': e.get('severity', 0),
            'ref_phonemes': ref_display,
            'pred_phonemes': pred_display,
            'explanation': None,
            'audio_path': None
        }

        if e['type'] not in ('correct', 'inserted'):
            try:
                explanation = generate_explanation(e)
                audio_path = audio_map.get(e['word'])
                word_entry['explanation'] = explanation
                word_entry['audio_path'] = audio_path

                # Forgiven accent variants keep their explanation (shown when
                # the word is tapped) but are NOT correction feedback, so they
                # stay out of the feedback list that Flutter's red warning
                # banner reads from.
                if e['type'] != 'accent':
                    feedback.append({
                        'word': e['word'],
                        'type': e['type'],
                        'per_word_score': e.get('per_word_score', 0.0),
                        'severity': e.get('severity', 0),
                        'explanation': explanation,
                        'audio_path': audio_path,
                        'ref_phonemes': ref_display,
                        'pred_phonemes': pred_display
                    })
            except Exception as ex:
                logger.warning(f"Feedback generation failed for '{e['word']}': {ex}")
                word_entry['explanation'] = f"Error generating feedback: {str(ex)}"

        words_output.append(word_entry)

    # Worst first: Flutter shows feedback[0] in its banner, which used to be
    # simply the first flagged word in reading order (often a trivial slip).
    feedback.sort(key=lambda f: -(f.get('severity') or 0))

    return words_output, feedback



def _build_mistake_summary(errors: list) -> dict:
    """Build mistake summary for Practice Mistakes button."""
    excluded = ('correct', 'inserted', 'accent') + (() if INCLUDE_MINOR_IN_PRACTICE else ('minor',))

    mistakes = []
    for e in errors:
        err_type = e.get('type', '')
        if err_type in excluded:
            continue

        mistakes.append({
            'word': e.get('word', ''),
            'error_type': err_type,
            'severity': e.get('severity', 0),
            'per_word_score': e.get('per_word_score', 0.0),
            'ref_phonemes': e.get('ref_phon', []),
            'pred_phonemes': e.get('pred_phon', []),
            'explanation': e.get('explanation', ''),
            'audio_path': e.get('audio_path', None)
        })

    mistakes.sort(key=lambda x: (-x['severity'], x['per_word_score']))

    severity_breakdown = {
        'critical': len([m for m in mistakes if m['severity'] >= 10]),
        'major': len([m for m in mistakes if 7 <= m['severity'] < 10]),
        'minor': len([m for m in mistakes if 3 <= m['severity'] < 7]),
        'notice': len([m for m in mistakes if m['severity'] < 3])
    }

    type_counts = {}
    for m in mistakes:
        type_counts[m['error_type']] = type_counts.get(m['error_type'], 0) + 1
    most_common_type = max(type_counts, key=type_counts.get) if type_counts else None

    return {
        'mistake_count': len(mistakes),
        'offer_practice': len(mistakes) >= 3,
        'severity_breakdown': severity_breakdown,
        'most_common_error': most_common_type,
        'words': [m['word'] for m in mistakes[:5]],
        'top_mistakes': mistakes[:3],
        'all_mistakes': mistakes
    }







def _error_response(message: str) -> dict:
    """Return standardized error response."""
    return {
        'success': False,
        'error': message,
        'score': 0.0,
        'grade': 'D',
        'grade_label': 'Focus Required',
        'transcript': [],
        'words': [],
        'feedback': [],
        'mistake_summary': {
            'mistake_count': 0,
            'offer_practice': False,
            'severity_breakdown': {},
            'words': [],
            'top_mistakes': [],
            'all_mistakes': []
        },
        'stats': {
            'total_words': 0,
            'correct_words': 0,
            'wrong_words': 0,
            'word_accuracy': 0.0,
            'skipped_words': 0,
            'flagged_words': 0,
            'inserted_words': 0,
            'minor_words': 0,
            'severity_total': 0,
            'audio_duration': 0.0,
        }
    }