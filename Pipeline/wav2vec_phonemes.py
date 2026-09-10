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
    Advanced: Returns [(phoneme, start_time, end_time, confidence), ...]
    Useful for detailed feedback on *when* a mispronunciation occurred.
    Decode is constrained to English-only phoneme tokens, same as
    recognize_phonemes().
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