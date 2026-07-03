# In feedback.py

TEMPLATES = {
    'substitution': "You said '{user_said}' but the correct word is '{word}'.",
    'omission':     "You dropped one or more sounds in '{word}'.",
    'insertion':    "You added an extra sound in '{word}'.",
    'skipped':      "You skipped the word '{word}'. Please include it.",
    'stress':       "The word '{word}' has incorrect stress placement.",
}

# Pakistani-English specific hints
PK_HINTS = {
    ('th', 't'):  "Tip: Place your tongue between your teeth for the 'th' sound.",
    ('th', 'f'):  "Tip: The 'th' sound uses the tongue tip, not the lips.",
    ('dh', 'd'):  "Tip: For 'th' in 'this', touch tongue to upper teeth.",
    ('w',  'v'):  "Tip: Round your lips fully for the 'w' sound.",
    ('v',  'w'):  "Tip: Touch upper teeth to lower lip for 'v'.",
    ('ih', 'iy'): "Tip: The vowel in 'ship' is shorter than in 'sheep'.",
}

def generate_explanation(error: dict) -> str:
    template  = TEMPLATES.get(error['type'], '')
    base = template.format(
        word=error['word'],
        user_said=error.get('pred_phon', [error['word']])[0]
               if error.get('pred_phon') else error['word']
    )
    hint = ''
    for op, i, j in error.get('editops', []):
        if op == 'replace':
            key = (error['ref_phon_norm'][i], error['pred_phon_norm'][j])
            if key in PK_HINTS:
                hint = '  ' + PK_HINTS[key]
                break
    phon_text = (f"Expected phonemes: {' '.join(error.get('ref_phon',[]))}. "
                 f"You produced: {' '.join(error.get('pred_phon',[]))}. ")
    return base + '  ' + phon_text + hint


# In feedback.py
from gtts import gTTS
import os, hashlib

CACHE_DIR = 'media/feedback/'
os.makedirs(CACHE_DIR, exist_ok=True)

def get_feedback_audio(word: str, lang: str = 'en',
                       tld: str = 'com') -> str:
    """
    FIX 5: tld changed from 'co.uk' (British) to 'com' (American English).
    Pakistani English curriculum targets General American English,
    not British English. Use tld='com' for American accent.
    tld='co.uk' = British | tld='com' = American | tld='com.au' = Australian
    """
    key  = hashlib.md5(f'{word}{lang}{tld}'.encode()).hexdigest()[:8]
    path = os.path.join(CACHE_DIR, f'{word}_{key}.mp3')
    if not os.path.exists(path):
        gTTS(text=word, lang=lang, tld=tld, slow=True).save(path)
    return path

# Cache note: after first call per word, subsequent calls are instant
# and do not need internet. Pre-generate top 500 words at deployment time.