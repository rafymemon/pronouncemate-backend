import json
import logging
import time
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async

from Pipeline.whisper_asr import transcribe_live

logger = logging.getLogger(__name__)

SILENCE_RMS_THRESHOLD = 500
SILENCE_CHUNKS_TO_CLOSE_WORD = 2
MIN_VOICED_BYTES = 3200
LOOKAHEAD = 3

# Force-close a segment even without silence once the buffer holds this
# much audio, so continuous fluent speech still gets periodic live
# feedback instead of stalling until the next natural pause.
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
MAX_BUFFER_SECONDS = 2.5
MAX_BUFFER_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * MAX_BUFFER_SECONDS)


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
            await self._handle_control(text_data)
            return
        if bytes_data is not None:
            if not self.initialized:
                return
            await self._handle_audio_chunk(bytes_data)

    async def _handle_control(self, text_data):
        try:
            data = json.loads(text_data)
        except json.JSONDecodeError:
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

        if loudness < SILENCE_RMS_THRESHOLD:
            self.silence_streak += 1
        else:
            self.silence_streak = 0
            self.voice_buffer.extend(chunk)

        if self._chunk_count % 10 == 0:
            print(f"🎤 [WS] Chunk #{self._chunk_count}, size={len(chunk)}B, "
                  f"rms={loudness:.1f}, silence_streak={self.silence_streak}, "
                  f"buffer={len(self.voice_buffer)}B, word_index={self.word_index}")

        should_close_on_silence = (
            self.silence_streak >= SILENCE_CHUNKS_TO_CLOSE_WORD
            and len(self.voice_buffer) >= MIN_VOICED_BYTES
        )
        should_force_close = len(self.voice_buffer) >= MAX_BUFFER_BYTES

        if (should_close_on_silence or should_force_close) and self.word_index < len(self.expected_words):
            reason = 'silence' if should_close_on_silence else 'max_buffer'
            print(f"🎤 [WS] Closing segment ({reason}), {len(self.voice_buffer)}B buffered")
            await self._close_current_segment()

    async def _close_current_segment(self):
        pcm_bytes = bytes(self.voice_buffer)
        self.voice_buffer = bytearray()
        self.silence_streak = 0

        audio_array = _pcm16_to_float32(pcm_bytes)

        t0 = time.monotonic()
        try:
            heard = await sync_to_async(transcribe_live, thread_sensitive=False)(
                audio_array, sr=16000, min_word_prob=0.3, min_word_duration=0.02
            )
            heard_words = [w['word'] for w in heard]
        except Exception as e:
            logger.warning(f'Live transcription failed: {e}')
            heard_words = []
        finally:
            elapsed = time.monotonic() - t0
            print(f"🎤 [WS] transcribe_live() took {elapsed:.2f}s for {len(pcm_bytes)}B")

        if not heard_words:
            if self.word_index < len(self.expected_words):
                await self._emit(self.word_index, 'error')
                self.word_index += 1
            return

        for hw in heard_words:
            if self.word_index >= len(self.expected_words):
                break
            await self._process_heard_word(hw)

    async def _process_heard_word(self, heard_word: str):
        clean_heard = _clean_word(heard_word)

        if clean_heard == _clean_word(self.expected_words[self.word_index]):
            await self._emit(self.word_index, 'correct')
            self.word_index += 1
            return

        for offset in range(1, LOOKAHEAD + 1):
            idx = self.word_index + offset
            if idx < len(self.expected_words) and clean_heard == _clean_word(self.expected_words[idx]):
                for skip_i in range(self.word_index, idx):
                    await self._emit(skip_i, 'skipped')
                await self._emit(idx, 'correct')
                self.word_index = idx + 1
                return

        await self._emit(self.word_index, 'error')
        self.word_index += 1

    async def _emit(self, word_index: int, status: str):
        expected = self.expected_words[word_index]
        print(f"🎤 [WS] word_index={word_index} expected='{expected}' status={status}")
        await self.send(text_data=json.dumps({
            'type': 'word_result',
            'word_index': word_index,
            'expected': expected,
            'status': status,
        }))