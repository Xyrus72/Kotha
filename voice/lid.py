"""English-vs-Bangla decision for one utterance.

Measured on 901 test clips (tools/lang_eval.py, tools/lid_fit.py; FLEURS English + Bangla and South-Asian-accented
English from Common Voice, full sentences and 2 s cuts), 5-fold cross-validated:

    Whisper language guess alone (old rule)             32 wrong
    Whisper + VoxLingua107 + English-decode fit (this)   8 wrong   (0 on full Bangla sentences)

Why three signals:
  * Whisper large-v3's language guess is good on full sentences, but on 2 s Bangla it often spreads its guess
    over unrelated languages, and South Asian English sometimes looks like Urdu/Hindi to it.
  * VoxLingua107 (SpeechBrain ECAPA, CPU, ~70 ms) never missed Bangla, even 2 s phrases, but calls about half of
    South Asian English "Bengali" - an accent detector more than a language detector.
  * How well large-v3 can transcribe the audio as English (average token log-probability) separates them:
    real English, accented or not, decodes cleanly; Bangla forced into English does not.
A logistic rule over the three (weights fitted on all 901 clips) makes the call.
"""
from __future__ import annotations

import math
import threading

import numpy as np

from .config import ROOT

# tools/lid_fit.py "wh_en_bn,wh_en_near,wh_en_other,vl,en_lp" (Nov 12 2026)
_MU = np.array([3.5639, 2.8451, 1.4053, -6.1014, -0.8452])
_SD = np.array([7.2023, 4.2665, 2.5196, 16.4951, 1.7884])
_W = np.array([1.3701, 1.0044, 1.0515, 1.5477, 0.5754])
_B = 1.1255

_NEAR = ("hi", "ur", "ne", "mr", "sa", "gu", "pa", "or", "si")
MODEL_DIR = ROOT / "models" / "lid-voxlingua107"


def _l(p: float) -> float:
    return math.log(max(p, 1e-6))


def score(whisper_probs: dict, vl_margin: float | None, en_lp: float) -> float:
    """> 0 means English, < 0 Bangla. vl_margin None (VoxLingua unavailable) falls back to its average."""
    p = whisper_probs
    en, bn = p.get("en", 0.0), p.get("bn", 0.0) + p.get("as", 0.0)
    near = sum(p.get(k, 0.0) for k in _NEAR)
    other = max(1e-6, 1.0 - en - bn - near)
    vl = _MU[3] if vl_margin is None else max(-30.0, min(30.0, vl_margin))
    x = np.array([_l(en) - _l(bn), _l(en) - _l(near), _l(en) - _l(other), vl, en_lp])
    return float(((x - _MU) / _SD) @ _W + _B)


class VoxLingua:
    """SpeechBrain VoxLingua107 ECAPA on the CPU. available False when torch/speechbrain or the model is missing."""

    def __init__(self):
        self.clf = None
        self.error: str | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        try:
            import os
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch and ctranslate2 both ship OpenMP
            import torch
            from speechbrain.inference.classifiers import EncoderClassifier
            torch.set_num_threads(4)
            self.clf = EncoderClassifier.from_hparams(source=str(MODEL_DIR), savedir=str(MODEL_DIR),
                                                      run_opts={"device": "cpu"})
            enc = self.clf.hparams.label_encoder
            names = [enc.decode_ndim(i).split(":")[0] for i in range(len(enc))]
            self.en_i, self.bn_i = names.index("en"), names.index("bn")
        except Exception as e:                       # noqa: BLE001 - detection still works without it
            self.clf, self.error = None, repr(e)

    @property
    def available(self) -> bool:
        return self.clf is not None

    def margin(self, speech: np.ndarray) -> float | None:
        """log P(English) - log P(Bengali) for speech (float32, 16 kHz); None when unavailable."""
        if self.clf is None:
            return None
        import torch
        with self._lock, torch.no_grad():
            out = self.clf.classify_batch(torch.from_numpy(np.ascontiguousarray(speech[: 30 * 16000])).unsqueeze(0))
        logp = out[0][0].numpy()
        return float(logp[self.en_i] - logp[self.bn_i])
