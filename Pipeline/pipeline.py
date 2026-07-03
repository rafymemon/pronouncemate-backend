# pipeline.py  —  Complete corrected master pipeline
from .audio_processor  import preprocess_audio, slice_word_audio
from .whisper_asr      import transcribe
from .g2p_module       import build_answer_key
from .alignment        import needleman_wunsch
from .error_detector   import detect_errors, per_word_score
from .scorer           import calculate_score
from .feedback         import generate_explanation, get_feedback_audio
import logging

logger = logging.getLogger(__name__)

def run_pipeline(raw_audio_path: str, reference_script: str) -> dict:
    """
    Args:
        raw_audio_path   : path to WAV file (saved from Flutter upload)
        reference_script : string (the script the user was asked to read)
    Returns dict with score, grade, transcript, words list, feedback list.
    """
    # Stage 1: Clean audio
    audio = preprocess_audio(raw_audio_path)
    logger.info(f'Preprocessed: {len(audio)/16000:.2f}s audio')

    # Stage 2: What the user actually said (with timestamps)
    transcript = transcribe(audio)
    logger.info(f'Transcribed: {len(transcript)} words')

    # Stage 3: What the user should have said
    answer_key  = build_answer_key(reference_script)
    logger.info(f'Answer key: {len(answer_key)} words')

    # Stage 4: Align spoken words to script words
    spoken_words = [w['word'] for w in transcript]
    script_words = list(answer_key.keys())
    alignment    = needleman_wunsch(spoken_words, script_words)
    # Build lookup: word -> timing dict
    timing_map = {w['word']: w for w in transcript}

    # Stage 5: Detect errors (with clip duration guard)
    # Pass timing into alignment so detect_errors can slice audio
    alignment_with_timing = []
    for spoken_word, script_word in alignment:
        timing = timing_map.get(spoken_word, {}) if spoken_word else {}
        alignment_with_timing.append((timing or spoken_word, script_word))

    errors = detect_errors(alignment_with_timing, audio, 16000, answer_key)
    logger.info(f'Detected {len([e for e in errors if e["type"]!="correct"])} errors')

    # Stage 6: Score and grade
    score_result = calculate_score(errors)

    # Build feedback for flagged words
    words_output = []
    feedback     = []
    for e in errors:
        word_entry = {
            'word':          e['word'],
            'type':          e['type'],
            'per_word_score': e.get('per_word_score', 100.0),
            'explanation':   None,
            'audio':         None
        }
        if e['type'] not in ('correct', 'inserted'):
            explanation = generate_explanation(e)
            audio_path  = get_feedback_audio(e['word'])  # American English
            word_entry['explanation'] = explanation
            word_entry['audio']       = audio_path
            feedback.append({
                'word':          e['word'],
                'type':          e['type'],
                'per_word_score': e.get('per_word_score', 0.0),
                'explanation':   explanation,
                'audio_path':    audio_path
            })
        words_output.append(word_entry)

    return {
        'score':      score_result['score'],
        'grade':      score_result['grade'],
        'grade_label':score_result['grade_label'],
        'transcript': [w['word'] for w in transcript],
        'words':      words_output,
        'feedback':   feedback
    }