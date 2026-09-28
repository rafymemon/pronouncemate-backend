import librosa
import noisereduce as nr
import numpy as np

TARGET_SR = 16000  # Both Whisper and Wav2Vec2 require exactly 16kHz


def preprocess_audio(file_path: str) -> np.ndarray:
    """
    Run all four steps in sequence.
    Returns clean float32 numpy array at 16kHz mono.
    """
    # Step 1: Load and resample to 16kHz mono
    audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

    # Step 2: Noise reduction
    noise_duration = min(0.5, len(audio) / TARGET_SR)  # Handle short audio
    noise_samples = int(noise_duration * TARGET_SR)
    noise_sample = audio[:noise_samples]
    
    if np.max(np.abs(noise_sample)) > 0.01:  # only if noise sample has content
        try:
            audio = nr.reduce_noise(
                y=audio,
                sr=TARGET_SR,
                y_noise=noise_sample,
                stationary=True,  # Use stationary for consistent noise profile
                prop_decrease=0.75  # 75% noise removed; >0.9 distorts consonants
            )
        except Exception as e:
            print(f"Warning: Noise reduction failed ({e}), using original audio")

    # Step 3: Trim silence from start and end
    # top_db=40: anything 40dB below peak is silence
    # Increase to 50 if mid-sentence pauses are being cut
    audio, _ = librosa.effects.trim(audio, top_db=40, frame_length=512, hop_length=128)

    # Step 4: Normalize volume to peak 0.9
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9

    return audio.astype(np.float32)


def slice_word_audio(audio: np.ndarray, start_sec: float,
                     end_sec: float, pad_ms: int = 50,
                     prev_end_sec: float | None = None,
                     next_start_sec: float | None = None) -> np.ndarray:
    """
    Slice per-word audio clip using Whisper timestamps.
    pad_ms: milliseconds of context added each side (helps Wav2Vec2 accuracy).
    Returns minimum 0.15s clip (zero-padded if shorter).

    prev_end_sec / next_start_sec: optional timestamps of the NEIGHBORING
    words in the utterance. When provided, padding is clamped so it can
    never cross into a neighboring word's own timestamp range. Without
    this, the pad_ms context window (meant to help Wav2Vec2 by including
    a bit of natural onset/release) can instead pull in the tail of the
    previous word or the onset of the next one whenever words are spoken
    close together — this was the root cause of acoustic predictions
    coming back contaminated with fragments of adjacent words.
    """
    pad = int(pad_ms * TARGET_SR / 1000)
    s = max(0, int(start_sec * TARGET_SR) - pad)
    e = min(len(audio), int(end_sec * TARGET_SR) + pad)

    if prev_end_sec is not None:
        s = max(s, int(prev_end_sec * TARGET_SR))
    if next_start_sec is not None:
        e = min(e, int(next_start_sec * TARGET_SR))

    # If clamping collapsed the window (e.g. Whisper still reported
    # touching/overlapping timestamps for adjacent words even after the
    # monotonic clamp in whisper_asr.py), fall back to the unpadded
    # original start/end rather than producing an empty or negative-length
    # clip.
    if e <= s:
        s = max(0, int(start_sec * TARGET_SR))
        e = min(len(audio), int(end_sec * TARGET_SR))

    clip = audio[s:e]
    
    # Pad short clips to minimum 0.15s (2400 samples)
    min_samples = int(0.15 * TARGET_SR)
    if len(clip) < min_samples:
        clip = np.pad(clip, (0, min_samples - len(clip)), mode='constant')
    
    return clip