"""
Throwaway diagnostic — NOT a production file, just to confirm/deny the
"Whisper has no left-context for word 1 of a short clip" hypothesis before
touching whisper_asr.py for real.

Tests one known-bad utterance (MARK IS GOING TO SEE ELEPHANT) three ways:
  (a) as-is (reproduces the bug)
  (b) with ~250ms of leading silence prepended before transcription
  (c) with a generic initial_prompt alongside hotwords

Run from pronouncemate-backend/ root:
    python test_padding_hypothesis.py
"""

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

import numpy as np
from Pipeline.audio_processor import preprocess_audio
from Pipeline.whisper_asr import transcribe, extract_hotwords

WAV_PATH = "/home/developer/FYP-PronounceMate/Data/speechocean762/WAVE/SPEAKER0003/000030012.WAV"
REFERENCE = "MARK IS GOING TO SEE ELEPHANT"
PAD_SEC = 0.25

audio = preprocess_audio(WAV_PATH, skip_denoise=True)
hotwords = extract_hotwords(REFERENCE)

print("=" * 60)
print("(a) AS-IS")
print("=" * 60)
words_a = transcribe(audio, hotwords=hotwords, log_drops=True)
for w in words_a:
    print(f"  {w['word']!r}: prob={w['prob']}")

print()
print("=" * 60)
print(f"(b) WITH {int(PAD_SEC*1000)}ms LEADING SILENCE")
print("=" * 60)
pad_samples = int(PAD_SEC * 16000)
padded_audio = np.concatenate([np.zeros(pad_samples, dtype=np.float32), audio])
words_b = transcribe(padded_audio, hotwords=hotwords, log_drops=True)
for w in words_b:
    # subtract pad back out so timestamps are comparable to (a)
    print(f"  {w['word']!r}: prob={w['prob']}  (raw start={w['start']:.3f}, would be {w['start']-PAD_SEC:.3f} after un-padding)")

print()
print("=" * 60)
print("(c) WITH GENERIC initial_prompt (no hotwords)")
print("=" * 60)
# Bypass the hotwords branch entirely by calling with hotwords=None and
# manually checking if initial_prompt alone (without the hotwords glossary
# format) changes anything. This requires a quick monkeypatch since
# transcribe() only sets initial_prompt when hotwords isn't supported —
# so this block just re-runs (a) for now; if (b) confirms the hypothesis,
# we'll wire up a real initial_prompt path in the production fix instead.
print("  (skipped — only meaningful once we know if (b) worked)")