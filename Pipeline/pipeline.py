import logging
from typing import List, Dict

from .audio_processor import preprocess_audio, slice_word_audio
from .whisper_asr import transcribe
from .g2p_module import build_answer_key
from .alignment import needleman_wunsch
from .error_detector import detect_errors
from .scorer import calculate_score
from .feedback import generate_explanation, get_feedback_audio

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────
MIN_AUDIO_DURATION = 0.5   # seconds — reject if too short
MAX_AUDIO_DURATION = 300.0  # seconds — 5 minutes max


def run_pipeline(raw_audio_path: str, reference_script: str) -> dict:
    """
    Full pipeline: audio → transcript → alignment → errors → score → feedback.
    
    Args:
        raw_audio_path: path to WAV file from Flutter upload
        reference_script: the script the user was asked to read
    
    Returns:
        dict with score, grade, transcript, words list, feedback list
    """
    
    # ── Stage 1: Preprocess audio ──
    try:
        audio = preprocess_audio(raw_audio_path)
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        return _error_response("Audio preprocessing failed")
    
    audio_duration = len(audio) / 16000
    logger.info(f'Preprocessed: {audio_duration:.2f}s audio')
    
    # Validate audio duration
    if audio_duration < MIN_AUDIO_DURATION:
        return _error_response(f"Audio too short ({audio_duration:.1f}s). Please speak for at least {MIN_AUDIO_DURATION}s.")
    if audio_duration > MAX_AUDIO_DURATION:
        return _error_response(f"Audio too long ({audio_duration:.1f}s). Maximum is {MAX_AUDIO_DURATION/60:.0f} minutes.")
    
    # ── Stage 2: Transcribe with timestamps ──
    try:
        transcript = transcribe(audio)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return _error_response("Transcription failed. Please try again with clearer audio.")
    
    logger.info(f'Transcribed: {len(transcript)} words')
    
    if not transcript:
        return _error_response("No speech detected. Please try again.")
    
    # ── Stage 3: Build answer key from reference script ──
    try:
        answer_key = build_answer_key(reference_script)
    except Exception as e:
        logger.error(f"Answer key generation failed: {e}")
        return _error_response("Failed to process reference script")
    
    logger.info(f'Answer key: {len(answer_key)} words')
    
    if not answer_key:
        return _error_response("Reference script is empty or invalid")
    
    # ── Stage 4: Align spoken words to script words ──
    spoken_words = [w['word'] for w in transcript]
    script_words = list(answer_key.keys())
    
    try:
        alignment = needleman_wunsch(spoken_words, script_words)
    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        return _error_response("Word alignment failed")
    
    # Build timing lookup that handles duplicates via index
    # Use index-based lookup instead of word-based to handle duplicates
    alignment_with_timing = _build_alignment_with_timing(alignment, transcript)
    
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
    
    error_count = len([e for e in errors if e["type"] != "correct"])
    logger.info(f'Detected {error_count} errors out of {len(errors)} words')
    
    # ── Stage 6: Score and grade ──
    try:
        score_result = calculate_score(errors)
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        return _error_response("Scoring failed")
    
    # ── Stage 7: Build feedback ──
    words_output, feedback = _build_feedback(errors)
    
    return {
        'success': True,
        'score': score_result['score'],
        'grade': score_result['grade'],
        'grade_label': score_result['grade_label'],
        'transcript': [w['word'] for w in transcript],
        'script_words': script_words,
        'words': words_output,
        'feedback': feedback,
        'stats': {
            'total_words': score_result.get('total_words', 0),
            'skipped_words': score_result.get('skipped_words', 0),
            'flagged_words': score_result.get('flagged_words', 0),
            'inserted_words': score_result.get('inserted_words', 0),
            'audio_duration': round(audio_duration, 2)
        }
    }


def _build_alignment_with_timing(alignment: list, transcript: list) -> list:
    """
    Build alignment with proper timing info, handling duplicate words.
    
    alignment: [(spoken_word_str, script_word_str), ...] from needleman_wunsch
    transcript: list of {word, start, end, prob} from Whisper
    
    Returns: [(spoken_info_dict_or_None, script_word_str), ...]
    """
    # Create index-based mapping: track which transcript entry we've used
    used_indices = set()
    alignment_with_timing = []
    
    for spoken_word, script_word in alignment:
        if spoken_word is None:
            # Skipped word
            alignment_with_timing.append((None, script_word))
            continue
        
        if script_word is None:
            # Inserted word — find unused transcript entry
            timing = _find_unused_timing(spoken_word, transcript, used_indices)
            alignment_with_timing.append((timing, None))
            continue
        
        # Normal aligned pair — find unused transcript entry for this word
        timing = _find_unused_timing(spoken_word, transcript, used_indices)
        alignment_with_timing.append((timing, script_word))
    
    return alignment_with_timing


def _find_unused_timing(word: str, transcript: list, used_indices: set) -> dict:
    """
    Find the next unused transcript entry matching this word.
    Handles duplicate words correctly.
    """
    for i, entry in enumerate(transcript):
        if i not in used_indices and entry['word'] == word:
            used_indices.add(i)
            return entry
    
    # Fallback: word not found (shouldn't happen with proper alignment)
    logger.warning(f"Word '{word}' not found in unused transcript entries")
    return {'word': word, 'start': 0.0, 'end': 0.0, 'prob': 0.0}


def _build_feedback(errors: list) -> tuple:
    """
    Build words_output and feedback lists from error detections.
    """
    words_output = []
    feedback = []
    
    for e in errors:
        word_entry = {
            'word': e['word'],
            'type': e['type'],
            'per_word_score': e.get('per_word_score', 100.0),
            'severity': e.get('severity', 0),
            'ref_phonemes': e.get('ref_phon', []),
            'pred_phonemes': e.get('pred_phon', []),
            'explanation': None,
            'audio_path': None
        }
        
        # Generate feedback for non-correct, non-inserted words
        if e['type'] not in ('correct', 'inserted'):
            try:
                explanation = generate_explanation(e)
                audio_path = get_feedback_audio(e['word'])
                word_entry['explanation'] = explanation
                word_entry['audio_path'] = audio_path
                
                feedback.append({
                    'word': e['word'],
                    'type': e['type'],
                    'per_word_score': e.get('per_word_score', 0.0),
                    'severity': e.get('severity', 0),
                    'explanation': explanation,
                    'audio_path': audio_path,
                    'ref_phonemes': e.get('ref_phon', []),
                    'pred_phonemes': e.get('pred_phon', [])
                })
            except Exception as ex:
                logger.warning(f"Feedback generation failed for '{e['word']}': {ex}")
                word_entry['explanation'] = f"Error generating feedback: {str(ex)}"
        
        words_output.append(word_entry)
    
    return words_output, feedback


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
        'stats': {}
    }