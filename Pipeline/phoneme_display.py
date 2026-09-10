"""
Display-only ARPAbet -> IPA conversion.

Purpose: after scoring is complete, words scored via the acoustic path
carry IPA phonemes (from Wav2Vec2 / espeak) while words scored via the
text-fallback path carry ARPAbet phonemes (from CMU dict / g2p_en).
Both are CORRECT for their own scoring comparison — this module does
NOT touch scoring. It only normalizes what gets shown to the user, so
the same UI list doesn't mix 'OW1' next to 'ˈɜː' depending on which
internal path happened to score that particular word.

Call to_display_ipa() ONLY when building the API/UI-facing word list —
never before or during scoring.
"""

import re

# CMU/ARPAbet -> IPA (General American), matched against the same
# phoneme inventory produced by g2p_module.py / CMU dict.
_ARPABET_TO_IPA = {
    # Vowels
    'aa': 'ɑ',   'ae': 'æ',   'ah': 'ʌ',   'ao': 'ɔ',
    'aw': 'aʊ',  'ay': 'aɪ',  'eh': 'ɛ',   'er': 'ɜː',
    'ey': 'eɪ',  'ih': 'ɪ',   'iy': 'i',   'ow': 'oʊ',
    'oy': 'ɔɪ',  'uh': 'ʊ',   'uw': 'u',
    # Consonants
    'b': 'b',   'ch': 'tʃ',  'd': 'd',   'dh': 'ð',
    'f': 'f',   'g': 'ɡ',    'hh': 'h',  'jh': 'dʒ',
    'k': 'k',   'l': 'l',    'm': 'm',   'n': 'n',
    'ng': 'ŋ',  'p': 'p',    'r': 'ɹ',   's': 's',
    'sh': 'ʃ',  't': 't',    'th': 'θ',  'v': 'v',
    'w': 'w',   'y': 'j',    'z': 'z',   'zh': 'ʒ',
}

# Matches raw ARPAbet tokens as they appear BEFORE normalize() strips
# stress digits, e.g. 'OW1', 'HH', 'AE0'. Used to detect which alphabet
# a phoneme list is in — IPA tokens contain non-ASCII characters and
# will never match this.
_ARPABET_TOKEN_RE = re.compile(r'^[A-Z]+[0-2]?$')


def _is_arpabet_list(phonemes: list) -> bool:
    """True if every non-empty token looks like a raw ARPAbet symbol."""
    tokens = [p for p in phonemes if p and p.strip()]
    if not tokens:
        return False
    return all(_ARPABET_TOKEN_RE.match(p.strip()) for p in tokens)


def _arpabet_token_to_ipa(token: str) -> str:
    """Convert a single raw ARPAbet token (e.g. 'OW1') to IPA (e.g. 'ˈoʊ')."""
    token = token.strip()
    stress = ''
    base = token

    # Primary stress digit '1' -> IPA primary stress mark ˈ.
    # Secondary stress '2' and unstressed '0' are dropped for display —
    # CMU's stress annotation doesn't map cleanly 1:1 onto IPA stress
    # placement rules, so we only surface primary stress, which is the
    # one pronunciation learners most need to see.
    if token and token[-1] in '012':
        base = token[:-1]
        if token[-1] == '1':
            stress = 'ˈ'

    ipa = _ARPABET_TO_IPA.get(base.lower())
    if ipa is None:
        # Unknown symbol (shouldn't normally happen with CMU/g2p_en
        # output) — fall back to the lowercased original rather than
        # silently dropping it, so a gap in the table is visible.
        return base.lower()
    return f'{stress}{ipa}'


def to_display_ipa(phonemes: list) -> list:
    """
    Convert a phoneme list to IPA for display purposes only.

    If `phonemes` is already IPA (e.g. from the acoustic/Wav2Vec2 path),
    it's returned unchanged. If it's ARPAbet (from the text-fallback/CMU
    path), each token is converted. Never call this before or during
    scoring — scoring must keep comparing within a single alphabet, per
    the acoustic-path vs text-fallback-path split in error_detector.py.
    """
    if not phonemes:
        return []
    if _is_arpabet_list(phonemes):
        return [_arpabet_token_to_ipa(p) for p in phonemes if p and p.strip()]
    return phonemes