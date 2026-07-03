# pipeline/wav2vec_phonemes.py
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch, numpy as np

_model = None
_processor = None

def load_wav2vec():
    global _model, _processor
    if _model is None:
        _processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        _model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        _model.eval()
    return _model, _processor

def recognize_phonemes(audio_clip: np.ndarray, sr: int = 16000) -> list:
    model, processor = load_wav2vec()
    inputs = processor(audio_clip, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    ids = torch.argmax(logits, dim=-1)
    transcript = processor.batch_decode(ids)[0]
    return list(transcript.lower().split())