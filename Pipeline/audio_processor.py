# audio_processor.py
import librosa, noisereduce as nr, numpy as np

TARGET_SR = 16000  # Both Whisper and Wav2Vec2 require exactly 16kHz

def preprocess_audio(file_path: str) -> np.ndarray:
    """
    Run all four steps in sequence.
    Returns clean float32 numpy array at 16kHz mono.
    """
    # Step 1: Load and resample to 16kHz mono
    audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

    # Step 2: Noise reduction
    # Use first 0.5s as noise profile (assumes user hasn't started speaking yet)
    noise_sample = audio[:int(TARGET_SR * 0.5)]
    if np.max(np.abs(noise_sample)) > 0.01:  # only if noise sample has content
        audio = nr.reduce_noise(
            y=audio, sr=TARGET_SR,
            y_noise=noise_sample,
            prop_decrease=0.75   # 75% noise removed; >0.9 distorts consonants
        )

    # Step 3: Trim silence from start and end
    # top_db=40: anything 40dB below peak is silence
    # Increase to 50 if mid-sentence pauses are being cut
    audio, _ = librosa.effects.trim(audio, top_db=40)

    # Step 4: Normalize volume to peak 0.9
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9

    return audio.astype(np.float32)

def slice_word_audio(audio: np.ndarray, start_sec: float,
                     end_sec: float, pad_ms: int = 50) -> np.ndarray:
    """
    Slice per-word audio clip using Whisper timestamps.
    pad_ms: milliseconds of context added each side (helps Wav2Vec2 accuracy).
    Returns minimum 0.15s clip (zero-padded if shorter).
    """
    pad = int(pad_ms * TARGET_SR / 1000)
    s = max(0, int(start_sec * TARGET_SR) - pad)
    e = min(len(audio), int(end_sec * TARGET_SR) + pad)
    clip = audio[s:e]
    # Pad short clips to minimum 0.15s (2400 samples)
    min_samples = int(0.15 * TARGET_SR)
    if len(clip) < min_samples:
        clip = np.pad(clip, (0, min_samples - len(clip)))
    return clip