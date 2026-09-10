import nltk
import re
from functools import lru_cache
from g2p_en import G2p

# Download CMU dict once
nltk.download('cmudict', quiet=True)
cmu = nltk.corpus.cmudict.dict()  # ~134,000 words — loaded once
g2p = G2p()                        # neural fallback for unknown words

# Valid ARPAbet phoneme regex (with optional stress 0-2)
_ARPABET_RE = re.compile(r'^[A-Z]+[0-2]?$')


def _number_to_words(text: str) -> str:
    """Convert digits to words before G2P. Basic implementation."""
    import inflect
    p = inflect.engine()
    
    def replace_num(match):
        num = match.group()
        try:
            return p.number_to_words(int(num))
        except (ValueError, OverflowError):
            return num
    
    return re.sub(r'\b\d+\b', replace_num, text)


def clean_word(word: str) -> str:
    """
    Strip punctuation and lowercase.
    Handles contractions carefully: preserves base word.
    """
    word = word.lower().strip()
    
    # Handle common contractions by expanding them
    contractions = {
        "won't": "will",
        "can't": "can",
        "don't": "do",
        "didn't": "did",
        "wouldn't": "would",
        "couldn't": "could",
        "shouldn't": "should",
        "isn't": "is",
        "aren't": "are",
        "wasn't": "was",
        "weren't": "were",
        "hasn't": "has",
        "haven't": "have",
        "hadn't": "had",
        "i'm": "i",
        "you're": "you",
        "he's": "he",
        "she's": "she",
        "it's": "it",
        "we're": "we",
        "they're": "they",
        "i've": "i",
        "you've": "you",
        "we've": "we",
        "they've": "they",
        "i'll": "i",
        "you'll": "you",
        "he'll": "he",
        "she'll": "she",
        "it'll": "it",
        "we'll": "we",
        "they'll": "they",
        "i'd": "i",
        "you'd": "you",
        "he'd": "he",
        "she'd": "she",
        "it'd": "it",
        "we'd": "we",
        "they'd": "they",
        "let's": "let",
        "that's": "that",
        "there's": "there",
        "what's": "what",
        "who's": "who",
        "where's": "where",
        "when's": "when",
        "why's": "why",
        "how's": "how",
    }
    
    # Check for contraction first
    if word in contractions:
        word = contractions[word]
    else:
        # Strip apostrophe and anything after for other contractions
        word = word.split("'")[0]
    
    # Remove all non-alphabetic characters
    word = re.sub(r"[^a-z]", '', word)
    
    return word


@lru_cache(maxsize=10000)
def _g2p_cached(word: str) -> list:
    """Cached neural G2P for OOV words."""
    phonemes = g2p(word)
    # Validate and clean ARPAbet output
    cleaned = []
    for p in phonemes:
        p = p.strip().upper()
        if _ARPABET_RE.match(p):
            cleaned.append(p)
    return cleaned if cleaned else ['<UNK>']


def build_answer_key(script_text: str) -> dict:
    """
    Returns {word: [ARPAbet phonemes]} for every word in script.
    Uses CMU Dict first (fast, exact), g2p-en neural fallback for OOV.
    """
    # Convert numbers to words first
    script_text = _number_to_words(script_text)
    
    tokens = script_text.split()
    key = {}
    
    for raw in tokens:
        word = clean_word(raw)
        if not word:
            continue
        if word in key:
            continue
            
        if word in cmu:
            # Use first pronunciation; could be enhanced with context-aware selection
            phonemes = cmu[word][0]
            # Ensure stress markers are present
            phonemes = [p.upper() for p in phonemes]
        else:
            phonemes = _g2p_cached(word)
        
        key[word] = phonemes
    
    return key

def build_script_word_list(script_text: str) -> list:
    """
    Returns the ordered list of cleaned words in the script, WITH
    duplicates preserved (unlike build_answer_key's dict keys, which
    dedupe). This is what must be passed to the aligner — the aligner
    needs the actual word sequence, not a unique-word set.
    """
    script_text = _number_to_words(script_text)
    tokens = script_text.split()
    words = []
    for raw in tokens:
        word = clean_word(raw)
        if not word:
            continue
        words.append(word)
    return words


def get_phonemes(word: str) -> list:
    """Get phonemes for a single word. Returns empty list if unknown."""
    word = clean_word(word)
    if not word:
        return []
    if word in cmu:
        return [p.upper() for p in cmu[word][0]]
    return _g2p_cached(word)


# Example usage
if __name__ == "__main__":
    test = build_answer_key("The module is very important")
    for w, p in test.items():
        print(f"{w}: {p}")