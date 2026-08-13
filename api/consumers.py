import json
import logging
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer

from Pipeline.whisper_asr import transcribe

logger = logging.getLogger(__name__)

SILENCE_RMS_THRESHOLD = 500
SILENCE_CHUNKS_TO_CLOSE_WORD = 2
MIN_VOICED_BYTES = 3200


def _pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return audio / 32768.0


def _rms(pcm_bytes: bytes) -> float:
    if not pcm_bytes:
        return 0.0
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0


def _clean_word(w: str) -> str:
    return ''.join(c for c in w.lower() if c.isalnum())


class PracticeConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.expected_words = []
        self.word_index = 0
        self.voice_buffer = bytearray()
        self.silence_streak = 0
        self.initialized = False
        self._chunk_count = 0
        print("🎤 [WS] Client connected")

    async def disconnect(self, close_code):
        print(f"🎤 [WS] Client disconnected: {close_code}")

    async def receive(self, text_data=None, bytes_data=None):
        if text_data is not None:
            print(f"🎤 [WS] Control message: {text_data}")
            await self._handle_control(text_data)
            return
        if bytes_data is not None:
            if not self.initialized:
                print("🎤 [WS] Ignoring audio — not initialized yet")
                return
            await self._handle_audio_chunk(bytes_data)

    async def _handle_control(self, text_data):
        try:
            data = json.loads(text_data)
        except json.JSONDecodeError:
            print("🎤 [WS] Failed to parse control message")
            return

        if data.get('type') == 'init':
            script_text = data.get('script_text', '')
            self.expected_words = script_text.split()
            self.word_index = 0
            self.voice_buffer = bytearray()
            self.silence_streak = 0
            self.initialized = True
            print(f"🎤 [WS] Initialized with {len(self.expected_words)} words")
            await self.send(text_data=json.dumps({
                'type': 'ready',
                'total_words': len(self.expected_words),
            }))

    async def _handle_audio_chunk(self, chunk: bytes):
        self._chunk_count += 1
        loudness = _rms(chunk)

        if self._chunk_count % 10 == 0:
            print(f"🎤 [WS] Chunk #{self._chunk_count}, size={len(chunk)}B, "
                  f"rms={loudness:.1f}, silence_streak={self.silence_streak}, "
                  f"buffer={len(self.voice_buffer)}B, word_index={self.word_index}")

        if loudness < SILENCE_RMS_THRESHOLD:
            self.silence_streak += 1
        else:
            self.silence_streak = 0
            self.voice_buffer.extend(chunk)

        if (self.silence_streak >= SILENCE_CHUNKS_TO_CLOSE_WORD
                and len(self.voice_buffer) >= MIN_VOICED_BYTES
                and self.word_index < len(self.expected_words)):
            print(f"🎤 [WS] Closing word {self.word_index} "
                  f"({len(self.voice_buffer)}B buffered)")
            await self._close_current_word()

    async def _close_current_word(self):
        pcm_bytes = bytes(self.voice_buffer)
        self.voice_buffer = bytearray()
        self.silence_streak = 0

        expected = self.expected_words[self.word_index]
        audio_array = _pcm16_to_float32(pcm_bytes)

        try:
            heard = transcribe(audio_array, sr=16000, min_word_prob=0.3, min_word_duration=0.02)
            heard_words = [w['word'] for w in heard]
        except Exception as e:
            print(f"🎤 [WS] Transcription failed: {e}")
            heard_words = []

        correct = _clean_word(expected) in [_clean_word(w) for w in heard_words]
        print(f"🎤 [WS] word_index={self.word_index} expected='{expected}' "
              f"heard={heard_words} correct={correct}")

        await self.send(text_data=json.dumps({
            'type': 'word_result',
            'word_index': self.word_index,
            'expected': expected,
            'heard': ' '.join(heard_words),
            'correct': correct,
        }))

        self.word_index += 1