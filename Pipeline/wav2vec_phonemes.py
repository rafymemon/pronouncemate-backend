from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from phonemizer import phonemize
import torch
import numpy as np
import warnings

# ── Configuration ─────────────────────────────────────────────
PHONEME_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"  # Audio → IPA phonemes
TARGET_SR = 16000

# Singletons
_model = None
_processor = None
_device = None


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


# ── Audio → Phonemes (Primary: Wav2Vec2) ─────────────────────
def recognize_phonemes(audio_clip: np.ndarray, sr: int = 16000) -> tuple:
    """
    Primary method: Wav2Vec2 audio → IPA phonemes.
    Returns: (phonemes_list, avg_confidence, per_phoneme_confidences)
    """
    model, processor, device = load_wav2vec()
    
    # Ensure correct format
    if audio_clip.dtype != np.float32:
        audio_clip = audio_clip.astype(np.float32)
    
    # Preprocess
    inputs = processor(audio_clip, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits  # [1, time, vocab]
    
    # Get probabilities and predictions
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
    """
    model, processor, device = load_wav2vec()
    
    inputs = processor(audio_clip, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    probs = torch.softmax(logits, dim=-1).squeeze(0)  # [time, vocab]
    predicted_ids = torch.argmax(logits, dim=-1).squeeze(0)  # [time]
    
    # Time per frame (Wav2Vec2 downsamples by factor of 320)
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
    Use this for the 'expected' pronunciation from script text.
    """
    try:
        ipa = phonemize(
            text,
            language=language,
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
            with_stress=True  # Includes stress markers like ˈ, ˌ
        )
        # Split into phoneme tokens
        phonemes = ipa.strip().split()
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
    Always uses Wav2Vec2 (primary method).
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
    # Test text → phonemes
    text = "hello world"
    print(f"Text: {text}")
    print(f"Phonemizer (IPA): {get_expected_phonemes(text, 'phonemizer')}")
    print(f"g2p_en (ARPAbet): {get_expected_phonemes(text, 'g2p_en')}")
    
    # Note: Audio test requires actual audio file
    # from audio_processor import preprocess_audio
    # audio = preprocess_audio("test.wav")
    # phonemes, conf, _ = get_spoken_phonemes(audio)
    # print(f"Spoken: {phonemes} (confidence: {conf:.2f})")