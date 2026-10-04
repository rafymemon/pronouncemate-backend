# api/consumers.py

import json
import logging
import time
import numpy as np
from Levenshtein import ratio as _lev_ratio
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async

from Pipeline.whisper_asr import transcribe_live

logger = logging.getLogger(__name__)

SILENCE_RMS_THRESHOLD = 500
SILENCE_CHUNKS_TO_CLOSE_WORD = 2
MIN_VOICED_BYTES = 3200
LOOKAHEAD = 3
LOOKBACK = 2   # how far back a late/repeated word may still "heal" an earlier guess

# Force-close a segment even without silence once the buffer holds this
# much audio, so continuous fluent speech still gets periodic live
# feedback instead of stalling until the next natural pause.
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
MAX_BUFFER_SECONDS = 2.5
MAX_BUFFER_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * MAX_BUFFER_SECONDS)

# Live statuses sent to Flutter:
#   correct - tiny.en heard the expected word
#   skipped - tiny.en jumped over a content word (user may really have skipped it)
#   unsure  - tiny.en could not confirm the word (ASR noise, a dropped short
#             function word, a mishear). NOT a verdict: the final /api/analyze/
#             pass (small.en + wav2vec) always has the last word. Flutter shows
#             it neutral/amber instead of red.
# There is intentionally no live 'error': tiny.en text alone is not reliable
# enough to call a pronunciation wrong.

# Short words tiny.en routinely drops or merges; a gap here is 'unsure',
# never 'skipped'.
_WEAK_WORDS = frozenset({
    'a', 'an', 'the', 'to', 'of', 'and', 'or', 'but', 'for', 'from', 'at', 'as',
    'in', 'on', 'is', 'it', 'its', 'i', 'my', 'me', 'we', 'he', 'she', 'you',
    'us', 'be', 'was', 'are', 'that', 'with', 'than', 'so', 'if', 'do', 'am',
})


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


def _words_match(heard: str, expected: str) -> bool:
    """
    Lenient live match: exact after cleaning, or (for words of 4+ letters)
    a near-identical spelling — tiny.en often returns plural/tense variants
    or small spelling slips ('schedual', 'meetings') for a correctly read word.
    """
    h, e = _clean_word(heard), _clean_word(expected)
    if not h or not e:
        return False
    if h == e:
        return True
    if len(h) >= 4 and len(e) >= 4 and _lev_ratio(h, e) >= 0.8:
        return True
    return False


class PracticeConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.expected_words = []
        self.word_index = 0
        self.voice_buffer = bytearray()
        self.silence_streak = 0
        self.initialized = False
        self._chunk_count = 0
        self.statuses = {}
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
            self.statuses = {}
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

        print(f"🎤 [WS] heard={heard_words}")

        if not heard_words:
            # Nothing intelligible in this segment (noise / breath). Do NOT
            # advance: the old code painted the next expected word red and
            # moved on, which desynced the pointer from where the user
            # actually was and caused long red cascades.
            return

        for hw in heard_words:
            if self.word_index >= len(self.expected_words):
                break
            await self._process_heard_word(hw)

    def _gap_status(self, idx: int) -> str:
        """Status for an expected word the live ASR jumped over."""
        if _clean_word(self.expected_words[idx]) in _WEAK_WORDS:
            return 'unsure'
        return 'skipped'

    async def _process_heard_word(self, heard_word: str):
        if not _clean_word(heard_word):
            return

        # 1) Heard the word we were waiting for.
        if _words_match(heard_word, self.expected_words[self.word_index]):
            await self._emit(self.word_index, 'correct')
            self.word_index += 1
            return

        # 2) Heard a word a little further ahead: the words in between were
        #    dropped by the ASR (short function words) or genuinely skipped.
        for offset in range(1, LOOKAHEAD + 1):
            idx = self.word_index + offset
            if idx < len(self.expected_words) and _words_match(heard_word, self.expected_words[idx]):
                for skip_i in range(self.word_index, idx):
                    await self._emit(skip_i, self._gap_status(skip_i))
                await self._emit(idx, 'correct')
                self.word_index = idx + 1
                return

        # 3) Heard a word we already gave up on (an earlier 'unsure'/'skipped'
        #    word arriving late, or an extra word that had pushed the pointer
        #    forward). Heal it and DON'T advance — this is what stops one
        #    stray word from turning the rest of the sentence red.
        for back in range(1, LOOKBACK + 1):
            j = self.word_index - back
            if j < 0:
                break
            if (self.statuses.get(j) in ('unsure', 'skipped')
                    and _words_match(heard_word, self.expected_words[j])):
                await self._emit(j, 'correct')
                return

        # 4) Unmatched: ASR noise or a mishear. Mark the current word
        #    'unsure' (neutral), move on, and let steps 2-3 resync on the
        #    next heard word.
        await self._emit(self.word_index, 'unsure')
        self.word_index += 1

    async def _emit(self, word_index: int, status: str):
        expected = self.expected_words[word_index]
        self.statuses[word_index] = status
        print(f"🎤 [WS] word_index={word_index} expected='{expected}' status={status}")
        await self.send(text_data=json.dumps({
            'type': 'word_result',
            'word_index': word_index,
            'expected': expected,
            'status': status,
        }))