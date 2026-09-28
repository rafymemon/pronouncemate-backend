# api/gemini_service.py
import json
import logging

from google import genai
from google.genai import types
from django.conf import settings

logger = logging.getLogger(__name__)

_client = None

# Google retired gemini-2.5-flash-lite for new API keys; the current
# equivalent is gemini-3.5-flash-lite. If Google deprecates this one too,
# check https://ai.google.dev/gemini-api/docs/models for the current name
# before assuming the string below is still correct.
MODEL_NAME = "gemini-3.5-flash-lite"

MODE_LABELS = {
    'public_speaking':    'Public Speaking',
    'daily_conversation': 'Daily Conversation',
    'script_based':       'Script-Based',
}

# Timeouts are in milliseconds. Without these the SDK keeps retrying 503s
# for 15-30s, which makes the Flutter client time out before the server's
# pool fallback can respond.
DEFAULT_TIMEOUT_MS = 10_000      # per attempt, client-wide default
DEFAULT_ATTEMPTS = 2             # initial call + 1 retry
ROTATION_TIMEOUT_MS = 7_000      # rotation must fail fast -> pool fallback
ROTATION_ATTEMPTS = 1            # no retries; the pool is the fallback


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(
            api_key=settings.GEMINI_API_KEY,
            http_options=types.HttpOptions(
                timeout=DEFAULT_TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(attempts=DEFAULT_ATTEMPTS),
            ),
        )
    return _client


def generate_focus_script_text(words: list[str], level: str, mode: str = 'daily_conversation') -> str:
    """
    Generates a short practice script naturally embedding the given
    target words, matched to the user's current level.
    """
    prompt = f"""You are generating a short English pronunciation practice script.

Target words to naturally embed (each should appear at least once): {", ".join(words)}
Learner level: {level}
Practice mode: {mode}

Write a natural, coherent script (4-6 sentences) suitable for reading aloud,
that includes all target words in context. Return ONLY the script text,
no preamble or explanation."""

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=400,
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.exception(f'Gemini Focus Script generation failed: {e}')
        # Graceful fallback so a Gemini outage never breaks session completion
        return 'Practice these words in your own sentences: ' + ', '.join(words) + '.'


def generate_rotation_scripts(mode: str, level: str, count: int = 1) -> list[dict]:
    """
    Generates `count` fresh, level-appropriate practice scripts for Smart
    Rotation via a single Gemini call (batched, not one call per script).

    Returns a list of dicts shaped {'title': str, 'content': str}. Returns
    an empty list on any failure (API error, timeout, malformed JSON, etc.)
    so the caller (api/rotation.py) can fall back to the curated built-in pool.

    This call uses a short timeout and no retries so a Gemini outage
    (e.g. 503 high demand) falls back to the pool in seconds, not tens of
    seconds.
    """
    mode_label = MODE_LABELS.get(mode, mode)

    prompt = f"""You are generating English pronunciation practice scripts for
a language-learning app.

Learner level: {level}
Practice mode: {mode_label}
Number of distinct scripts needed: {count}

For each script, write a natural, coherent passage (4-6 sentences) suitable
for reading aloud, appropriate in vocabulary and sentence complexity for a
"{level}" English learner practicing "{mode_label}".

Respond with ONLY a JSON array, no markdown fences, no preamble, in this
exact shape:
[
  {{"title": "Short 3-6 word title", "content": "The full script text."}}
]
The array must contain exactly {count} item(s)."""

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=250 * count + 150,
                http_options=types.HttpOptions(
                    timeout=ROTATION_TIMEOUT_MS,
                    retry_options=types.HttpRetryOptions(attempts=ROTATION_ATTEMPTS),
                ),
            ),
        )

        raw = response.text.strip()
        # Defensive strip in case the model wraps the array in ```json fences
        if raw.startswith('```'):
            raw = raw.strip('`')
            if raw.lower().startswith('json'):
                raw = raw[4:].strip()

        scripts = json.loads(raw)
        if not isinstance(scripts, list):
            raise ValueError('Expected a JSON array of scripts')

        cleaned = []
        for item in scripts:
            title = (item.get('title') or '').strip()
            content = (item.get('content') or '').strip()
            if title and content:
                cleaned.append({'title': title, 'content': content})
        return cleaned

    except Exception as e:
        logger.exception(
            f'Gemini Rotation Script generation failed '
            f'(mode={mode}, level={level}, count={count}): {e}'
        )
        return []


def generate_custom_script(topic: str, word_limit: int, level: str, mode: str) -> str | None:
    """
    Generates a single practice script on a user-supplied topic, targeting
    approximately `word_limit` words. Powers the "Generate with AI" button
    on the practice screen (api/views.py::GenerateScriptView).

    Unlike the other two generators, this returns None on failure rather
    than a silent fallback string — this is a live, user-initiated action,
    so the caller should surface a clear error instead of quietly serving
    filler text the user didn't ask for.
    """
    mode_label = MODE_LABELS.get(mode, mode)

    prompt = f"""You are generating an English pronunciation practice script.

Topic: {topic}
Target length: approximately {word_limit} words (stay within about 20% of this)
Learner level: {level}
Practice mode: {mode_label}

Write a natural, coherent script suitable for reading aloud, on the given
topic, with vocabulary and sentence complexity appropriate for a
"{level}" English learner practicing "{mode_label}". Return ONLY the
script text — no title, no preamble, no explanation, no markdown."""

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=int(word_limit * 2.5) + 100,
            ),
        )
        text = response.text.strip()
        return text if text else None
    except Exception as e:
        logger.exception(
            f'Gemini custom script generation failed '
            f'(topic={topic!r}, word_limit={word_limit}, level={level}, mode={mode}): {e}'
        )
        return None