"""
Throwaway diagnostic — saves the first ~1.2s of the preprocessed audio
(where 'mark' should be) to a standalone WAV file so we can actually listen
to it, plus prints RMS energy in 50ms buckets to check whether there's real
signal there or near-silence.

Run from pronouncemate-backend/ root:
    python extract_mark_segment.py
"""

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

import numpy as np
import soundfile as sf
from Pipeline.audio_processor import preprocess_audio

WAV_PATH = "/home/developer/FYP-PronounceMate/Data/speechocean762/WAVE/SPEAKER0003/000030012.WAV"
SR = 16000

audio = preprocess_audio(WAV_PATH, skip_denoise=True)
print(f"Total preprocessed length: {len(audio)/SR:.3f}s")

# First 1.2s — should cover "MARK" plus a bit of "IS"
segment = audio[:int(1.2 * SR)]
out_path = "mark_segment.wav"
sf.write(out_path, segment, SR)
print(f"Saved first 1.2s to {out_path} — download and listen to it")

# RMS energy per 50ms bucket, to see where real signal starts
bucket_samples = int(0.05 * SR)
print("\nRMS energy per 50ms bucket (first 1.2s):")
for i in range(0, len(segment), bucket_samples):
    bucket = segment[i:i+bucket_samples]
    if len(bucket) == 0:
        continue
    rms = np.sqrt(np.mean(bucket.astype(np.float64) ** 2))
    t = i / SR
    bar = '#' * int(rms * 200)
    print(f"  {t:5.2f}s: rms={rms:.4f} {bar}")