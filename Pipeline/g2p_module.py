# g2p_module.py
import nltk, re
from g2p_en import G2p

nltk.download('cmudict', quiet=True)
cmu = nltk.corpus.cmudict.dict()  # ~134,000 words — loaded once
g2p = G2p()                        # neural fallback for unknown words

def clean_word(word: str) -> str:
    """Strip punctuation and lowercase."""
    # Handle contractions: don't -> dont, I'm -> im
    word = re.sub(r"'", '', word)
    return re.sub(r"[^a-z]", '', word.lower())

def build_answer_key(script_text: str) -> dict:
    """
    Returns {word: [ARPAbet phonemes]} for every word in script.
    Uses CMU Dict first (fast, exact), g2p-en neural fallback for OOV.
    """
    tokens = script_text.split()
    key = {}
    for raw in tokens:
        word = clean_word(raw)
        if not word or word in key:
            continue
        if word in cmu:
            phonemes = cmu[word][0]         # first pronunciation variant
        else:
            phonemes = g2p(word)            # neural G2P for names, new words
            phonemes = [p for p in phonemes if p.strip()]
        key[word] = phonemes
    return key

# Example:
# build_answer_key('The module is very important')
# -> {'the':['DH','AH0'], 'module':['M','AA1','JH','UW0','L'],
#     'is':['IH0','Z'], 'very':['V','EH1','R','IY0'],
#     'important':['IH0','M','P','AO1','R','T','AH0','N','T']}