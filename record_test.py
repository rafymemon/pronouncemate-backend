# save as record_test.py and run it separately
import pyaudio
import wave
import numpy as np

SAMPLE_RATE = 16000
DURATION    = 5       # seconds
OUTPUT_FILE = "test_speech.wav"

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1,
                rate=SAMPLE_RATE, input=True, frames_per_buffer=1024)

print(f"Recording {DURATION}s — say: 'Hello, this is a test of the speech recognition system.'")
frames = [stream.read(1024) for _ in range(int(SAMPLE_RATE / 1024 * DURATION))]
stream.stop_stream()
stream.close()
p.terminate()

with wave.open(OUTPUT_FILE, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))

print(f"Saved → {OUTPUT_FILE}")