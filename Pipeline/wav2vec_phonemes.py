from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from phonemizer import phonemize
from phonemizer.separator import Separator
import torch
import numpy as np
import warnings
import re

# ── Configuration ─────────────────────────────────────────────
PHONEME_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"  # Audio → IPA phonemes
TARGET_SR = 16000

# Singletons
_model = None
_processor = None
_device = None
_allowed_token_ids = None  # cached English-only decode mask

_PHONE_SEPARATOR = Separator(phone=' ', word='|', syllable='')

# Canonical General American English phoneme inventory as produced by
# espeak's en-us backend (verified against this project's own
# text_to_phonemes() output — see e.g. 'hello' -> h ə l oʊ,
# 'world' -> w ɜː l d, 'beautiful' -> b j uː ɾ i f əl). This model
# (wav2vec2-lv-60-espeak-cv-ft) was fine-tuned across MANY languages
# sharing one output vocabulary (Mandarin tones, Hindi retroflexes,
# German/French vowels, etc.) — without restricting decode to this
# set, CTC argmax is free to pick a foreign-language symbol for any
# ambiguous frame, which is unusable for English pronunciation scoring.
#
# DELIBERATE EXCEPTION — ʈ, ɖ, ɽ (retroflex t/d/r):
# These are NOT part of General American English and would normally be
# excluded under the rule above. They're included on purpose so that
# accent_rules.py's Pakistani-accent tolerance system (retroflex_t,
# retroflex_d, retroflex_r patterns) can actually see a retroflex
# realization when a speaker produces one, rather than the mask forcing
# it to collapse into plain t/d/r before it ever reaches the tolerance
# check. This is a deliberate loosening of the "English-only" guarantee
# for these three symbols specifically — accent_rules.py is what decides
# whether a retroflex reading gets forgiven or flagged; this mask only
# controls whether the model is ALLOWED to report one at all.
#
# RISK / TODO: loosening the mask for these three tokens means CTC
# argmax can now pick them for frames that were never retroflex at all
# (it's still choosing the locally-likeliest ALLOWED symbol, and these
# three are now allowed everywhere, not just where a Pakistani speaker
# actually produced a retroflex). Before trusting this in scoring,
# spot-check non-retroflex speech (e.g. a General American reference
# recording of the same script) to confirm the false-positive rate for
# ʈ/ɖ/ɽ appearing where they shouldn't hasn't measurably increased. If
# it has, this needs a narrower fix (e.g. only allow these three tokens
# for words your answer key already expects to contain a coronal
# stop/rhotic) rather than a blanket global allow.
_ENGLISH_PHONEME_BASE = {
    # Vowels / diphthongs
    'aɪ', 'aʊ', 'aː', 'æ',
    'e', 'eɪ', 'eə', 'eʊ', 'eː',
    'i', 'iː',
    'o', 'oʊ', 'oɪ', 'oː', 'oːɹ',
    'u', 'uː',
    'ɑ', 'ɑː', 'ɑːɹ',
    'ɔ', 'ɔː', 'ɔːɹ', 'ɔɪ',
    'ɚ', 'ɛ', 'ɛɹ', 'ɜː',
    'ɪ', 'ɪɹ',
    'ʊ', 'ʊɹ',
    'ʌ', 'ə', 'əl',
    # Consonants
    'b', 'tʃ', 'd', 'ð', 'f', 'ɡ', 'h', 'dʒ', 'k', 'l', 'm', 'n', 'ŋ',
    'p', 'ɹ', 's', 'ʃ', 't', 'θ', 'v', 'w', 'j', 'z', 'ʒ', 'ɾ',
    # Retroflex consonants — accent-tolerance exception, see comment above
    'ʈ', 'ɖ', 'ɽ',
}

# Strips CTC/tokenizer stress or length artifacts so we can match a raw
# vocab token (which may carry stress marks or duration colons) against
# the base inventory above.
_STRIP_RE = re.compile(r'[ˈˌ]')


# ── Model Loading ─────────────────────────────────────────────
def load_wav2vec():
    """Lazy-load Wav2Vec2 phoneme model. Thread-safe enough for single-user."""
    global _model, _processor, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Wav2Vec2] Loading phoneme model on {_device}...")
        
        _processor = Wav2Vec2Processor.from_pretrained(PHONEME_MODEL)
        _model = Wav2Vec2ForCTC.from_pretrained(PHONEME_MODEL).to(_device)
        _model.eval()
        print("[Wav2Vec2] Model ready.")
        
    return _model, _processor, _device


def _get_allowed_token_ids(processor) -> set:
    """
    Builds (once, cached) the set of vocab token IDs the CTC decode is
    allowed to choose from: the pad/blank token (REQUIRED — CTC collapse
    breaks without it) plus every vocab entry whose phoneme content,
    after stripping stress marks, is in the English inventory above.
    """
    global _allowed_token_ids
    if _allowed_token_ids is not None:
        return _allowed_token_ids

    vocab = processor.tokenizer.get_vocab()
    allowed = set()

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        allowed.add(pad_id)

    for token, tid in vocab.items():
        stripped = _STRIP_RE.sub('', token)
        if stripped in _ENGLISH_PHONEME_BASE:
            allowed.add(tid)

    print(f"[Wav2Vec2] English-only decode mask: {len(allowed)} / {len(vocab)} tokens allowed")
    _allowed_token_ids = allowed
    return _allowed_token_ids


def _mask_logits_to_english(logits: torch.Tensor, processor) -> torch.Tensor:
    """
    Sets logits for every non-English-allowed vocab id to -inf, so
    argmax can never select a foreign-language phoneme. Operates on the
    full [1, time, vocab] logits tensor in place (on a clone).
    """
    allowed_ids = _get_allowed_token_ids(processor)
    vocab_size = logits.shape[-1]

    mask = torch.full((vocab_size,), float('-inf'), device=logits.device)
    allowed_list = [i for i in allowed_ids if i < vocab_size]
    mask[allowed_list] = 0.0

    return logits + mask  # broadcasts over [1, time, vocab]


# ── Audio → Phonemes (Primary: Wav2Vec2) ─────────────────────
def recognize_phonemes(audio_clip: np.ndarray, sr: int = 16000) -> tuple:
    """
    Primary method: Wav2Vec2 audio → IPA phonemes, decode constrained
    to English-only phoneme tokens (see _ENGLISH_PHONEME_BASE).
    Returns: (phonemes_list, avg_confidence, per_phoneme_confidences)

    NOTE: this single-clip method is still used by callers that want a
    quick word-in-isolation read (e.g. ad-hoc lookups outside the main
    detection pipeline). The main detect_errors() pipeline no longer
    calls this per word — see recognize_phonemes_with_timestamps() +
    bucket_phonemes_by_word() below, which decode the whole utterance
    once and avoid re-cutting audio at word boundaries.
    """
    model, processor, device = load_wav2vec()
    
    if audio_clip.dtype != np.float32:
        audio_clip = audio_clip.astype(np.float32)
    
    inputs = processor(audio_clip, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits  # [1, time, vocab]

    logits = _mask_logits_to_english(logits, processor)

    probs = torch.softmax(logits, dim=-1)
    predicted_ids = torch.argmax(logits, dim=-1).squeeze(0)  # [time]
    confidence_values = torch.max(probs, dim=-1).values.squeeze(0)  # [time]
    
    # Collapse CTC: merge repeats, remove blanks
    phonemes = []
    confidences = []
    prev_id = -1
    
    for t, pid in enumerate(predicted_ids.cpu().numpy()):
        if pid != prev_id and pid != processor.tokenizer.pad_token_id:
            phoneme = processor.decode([pid]).strip()
            if phoneme and phoneme != "|":  # "|" is often used as word separator
                phonemes.append(phoneme)
                confidences.append(float(confidence_values[t].cpu()))
        prev_id = pid
    
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    
    return phonemes, avg_conf, confidences


def recognize_phonemes_with_timestamps(audio_clip: np.ndarray, sr: int = 16000) -> list:
    """
    Returns [{'phoneme': str, 'start': float, 'end': float, 'confidence': float}, ...]
    for the given audio, decode constrained to English-only phoneme
    tokens (same mask as recognize_phonemes()).

    Called ONCE per utterance on the full audio (never on a re-cut
    per-word clip), then bucketed into per-word groups by
    bucket_phonemes_by_word() below. Timestamps are in seconds, relative
    to the start of whatever audio array is passed in — callers MUST
    pass the same audio array (and therefore the same time reference)
    that produced the word-level Whisper timestamps being used for
    bucketing, or the two timelines won't line up.
    """
    model, processor, device = load_wav2vec()
    
    inputs = processor(audio_clip, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits

    logits = _mask_logits_to_english(logits, processor)
    
    probs = torch.softmax(logits, dim=-1).squeeze(0)  # [time, vocab]
    predicted_ids = torch.argmax(logits, dim=-1).squeeze(0)  # [time]
    
    hop_duration = 320 / sr  # ~20ms per frame at 16kHz
    
    phonemes = []
    prev_id = -1
    current_start = 0.0
    
    for t, pid in enumerate(predicted_ids.cpu().numpy()):
        if pid != prev_id and pid != processor.tokenizer.pad_token_id:
            phoneme = processor.decode([pid]).strip()
            if phoneme and phoneme != "|":
                end_time = (t + 1) * hop_duration
                phonemes.append({
                    'phoneme': phoneme,
                    'start': current_start,
                    'end': end_time,
                    'confidence': float(probs[t, pid].cpu())
                })
                current_start = end_time
        prev_id = pid
    
    return phonemes


# Maximum distance (seconds) a word's phoneme bucket boundary is allowed
# to extend beyond its OWN Whisper start/end timestamp, whether at the
# very edges of the utterance (first/last real-timed word) or toward a
# gap containing a word Whisper failed to transcribe at all.
#
# WHY THIS EXISTS: a word Whisper completely misses (not mistimed —
# fully absent from the transcript) still has real, physically-spoken
# audio in that time range, but no entry in word_timings. Without a cap,
# the boundary logic below extends a neighbor's bucket all the way to
# 0.0 (utterance start), +inf (utterance end), or the full midpoint with
# the next real neighbor — which silently vacuums that missed word's
# entire phoneme content into whichever neighbor's boundary happens to
# reach it. Capping the extension bounds how much of a missed word's
# audio can leak into a neighbor; anything beyond the cap is simply
# unassigned (dropped from every bucket) rather than attributed to the
# wrong word. This is a bound, not a perfect fix — a missed word shorter
# than this cap still leaks in — but it turns systematic whole-word
# bleed into a much smaller, capped amount.
# TUNABLE: 0.15s matches this project's other short-clip conventions
# (MIN_CLIP_DURATION in error_detector.py); revisit empirically if
# missed-word bleed is still visible in testing.
MAX_BOUNDARY_EXTENSION = 0.15


def bucket_phonemes_by_word(phoneme_timeline: list, word_timings: list) -> list:
    """
    Split a single continuous phoneme timeline (from
    recognize_phonemes_with_timestamps(), decoded ONCE on the whole
    utterance) into per-word phoneme groups, WITHOUT ever re-cutting the
    underlying audio.

    Args:
        phoneme_timeline: output of recognize_phonemes_with_timestamps()
            for the full utterance.
        word_timings: list of (start, end) tuples OR None, one entry per
            alignment position, in the same order as the alignment list
            detect_errors() is processing. A None entry (a skipped or
            inserted alignment slot with no real Whisper timing) always
            gets an empty bucket — it's never eligible to receive
            phonemes.

    Returns:
        A list, same length and order as word_timings, of
        {'phonemes': [ipa_str, ...], 'avg_confidence': float}. Every
        phoneme in the input timeline is assigned to AT MOST ONE word
        bucket, or dropped if it falls in a gap too large to be claimed
        by either neighbor (see MAX_BOUNDARY_EXTENSION) or outside every
        word's boundary entirely.

    Method: for each word with real timing, its bucket boundary normally
    extends from the midpoint between it and its previous real-timed
    neighbor, to the midpoint between it and its next real-timed
    neighbor — but that extension is CAPPED at MAX_BOUNDARY_EXTENSION
    beyond the word's own start/end on each side (this also governs the
    very first word's left edge and the very last word's right edge,
    replacing what would otherwise be an unconditional 0.0 / +inf). When
    the gap to a neighbor is normal (natural word spacing), the cap
    never engages and this behaves exactly like a plain midpoint split.
    When the gap is unusually large — typically because a word Whisper
    failed to transcribe at all sits physically in between — the capped
    boundary leaves a middle strip of that gap unclaimed by both
    neighbors, so a missed word's audio is only partially, not fully,
    attributed to either side.
    """
    n = len(word_timings)
    buckets = [{'phonemes': [], 'avg_confidence': 0.0} for _ in range(n)]

    real_indices = [i for i, t in enumerate(word_timings) if t is not None]
    if not real_indices:
        return buckets

    boundaries = {}
    for pos, idx in enumerate(real_indices):
        start, end = word_timings[idx]

        if pos == 0:
            left = max(0.0, start - MAX_BOUNDARY_EXTENSION)
        else:
            prev_idx = real_indices[pos - 1]
            _, prev_end = word_timings[prev_idx]
            midpoint_left = (prev_end + start) / 2.0
            left = max(midpoint_left, start - MAX_BOUNDARY_EXTENSION)

        if pos == len(real_indices) - 1:
            right = end + MAX_BOUNDARY_EXTENSION
        else:
            next_idx = real_indices[pos + 1]
            next_start, _ = word_timings[next_idx]
            midpoint_right = (end + next_start) / 2.0
            right = min(midpoint_right, end + MAX_BOUNDARY_EXTENSION)

        boundaries[idx] = (left, right)

    confidences = {idx: [] for idx in real_indices}

    for p in phoneme_timeline:
        midpoint = (p['start'] + p['end']) / 2.0
        for idx in real_indices:
            left, right = boundaries[idx]
            if left <= midpoint < right:
                buckets[idx]['phonemes'].append(p['phoneme'])
                confidences[idx].append(p['confidence'])
                break
        # A phoneme matching no bucket — either it falls in a capped-off
        # gap (likely a missed word) or genuinely outside every
        # boundary — is dropped rather than crashing or being forced
        # into the nearest word.

    for idx in real_indices:
        if confidences[idx]:
            buckets[idx]['avg_confidence'] = sum(confidences[idx]) / len(confidences[idx])

    return buckets


# ── Text → Phonemes (Fallback: Phonemizer) ───────────────────
def text_to_phonemes(text: str, language: str = "en-us") -> list:
    """
    Fallback/Expected: Convert text to IPA phonemes using eSpeak backend.
    Returns a list of INDIVIDUAL phoneme tokens (uses the phone
    Separator — without it phonemize() returns the whole word as one
    unsplit string).
    """
    try:
        ipa = phonemize(
            text,
            language=language,
            backend="espeak",
            separator=_PHONE_SEPARATOR,
            strip=True,
            preserve_punctuation=False,
            with_stress=True
        )
        phonemes = ipa.replace('|', ' ').strip().split()
        return phonemes
    except Exception as e:
        warnings.warn(f"Phonemizer failed: {e}. Install with: pip install phonemizer")
        return []


def text_to_phonemes_arpabet(text: str) -> list:
    """
    Alternative: Use g2p_en (ARPAbet) for expected text phonemes.
    Import from your g2p_module.py instead if preferred.
    """
    from g2p_en import G2p
    g2p = G2p()
    return g2p(text)


# ── Unified Interface ────────────────────────────────────────
def get_spoken_phonemes(audio_clip: np.ndarray, sr: int = 16000, 
                        return_timestamps: bool = False) -> tuple:
    """
    Main entry for spoken audio → phonemes.
    Always uses Wav2Vec2 (primary method), English-only constrained.

    NOTE: kept for callers that still want a single-clip read. The main
    detect_errors() pipeline no longer calls this — see
    recognize_phonemes_with_timestamps() + bucket_phonemes_by_word().
    """
    if return_timestamps:
        return recognize_phonemes_with_timestamps(audio_clip, sr)
    return recognize_phonemes(audio_clip, sr)


def get_expected_phonemes(text: str, backend: str = "phonemizer") -> list:
    """
    Main entry for expected text → phonemes.
    backend: 'phonemizer' (IPA) or 'g2p_en' (ARPAbet)
    """
    if backend == "phonemizer":
        return text_to_phonemes(text)
    elif backend == "g2p_en":
        return text_to_phonemes_arpabet(text)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'phonemizer' or 'g2p_en'")


# ── Example / Test ───────────────────────────────────────────
if __name__ == "__main__":
    text = "hello world"
    print(f"Text: {text}")
    print(f"Phonemizer (IPA): {get_expected_phonemes(text, 'phonemizer')}")
    print(f"g2p_en (ARPAbet): {get_expected_phonemes(text, 'g2p_en')}")