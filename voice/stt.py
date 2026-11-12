"""Speech -> text for English and Bangla.

Two Whisper models run on the GPU through faster-whisper (CTranslate2):
  * the multilingual large-v3 decides which of the two languages was spoken and transcribes English;
  * a Bangla fine-tune transcribes Bangla.

In Auto mode the language is decided between English and Bangla only, by voice/lid.py: Whisper's language
guess, the VoxLingua107 language-ID model and how well the audio transcribes as English, combined by a rule
measured on 901 test clips (see that file). Choosing the language in the app skips all of this.
"""
from __future__ import annotations

import threading
import time
import unicodedata
from dataclasses import dataclass, field

import numpy as np

from . import lid, runtime
from .config import model_path

RATE = 16000
BANGLA_FAMILY = ("bn", "as", "hi", "ne", "mr", "sa", "ur", "gu", "pa", "or")
CLOSE_CALL = 4.0                 # decode both languages when one is less than this many times likelier
CHUNK_S = 24.0                   # Bangla needs many tokens per word: a full 30 s can overflow the decoder
PAD_S = 0.25                     # audio kept around each stretch of speech so first/last sounds survive
MIN_PIECE_S = 0.4                # shorter leftovers are glued to their neighbour, never decoded alone


@dataclass
class Result:
    text: str
    language: str                # "en" or "bn"
    seconds: float               # time spent transcribing
    duration: float              # length of the audio
    avg_logprob: float           # mean token log-probability (closer to 0 = more confident)
    language_probs: dict = field(default_factory=dict)


def _clean(text: str, language: str) -> str:
    text = unicodedata.normalize("NFC", " ".join(text.split()))
    if language == "bn":
        # the two ways of writing য় / ড় / ঢ় and stray zero-width marks that fine-tunes emit
        text = (text.replace("য়", "য়").replace("ড়", "ড়")
                .replace("ঢ়", "ঢ়").replace("‌", "").replace("‍", ""))
    return text.strip()


def speech_chunks(audio: np.ndarray, limit_s: float = CHUNK_S) -> list[np.ndarray]:
    """Speech-only pieces of at most `limit_s` seconds, cut where the speaker paused."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    spans = get_speech_timestamps(audio, VadOptions(
        threshold=0.35, min_speech_duration_ms=100, min_silence_duration_ms=300, speech_pad_ms=0,
        max_speech_duration_s=limit_s - 2 * PAD_S))      # Silero itself splits long speech at its dips
    if not spans:
        return []
    pad, limit = int(PAD_S * RATE), int(limit_s * RATE)
    padded = []
    for i, s in enumerate(spans):                         # pad, but never into the neighbouring span
        before = s["start"] - (spans[i - 1]["end"] if i else 0)
        after = (spans[i + 1]["start"] if i + 1 < len(spans) else len(audio)) - s["end"]
        padded.append((s["start"] - min(pad, before // 2), s["end"] + min(pad, after // 2)))
    groups: list[list[int]] = [list(padded[0])]
    for start, end in padded[1:]:
        if end - groups[-1][0] <= limit:
            groups[-1][1] = end
        else:
            groups.append([start, end])
    pieces: list[np.ndarray] = []
    for start, end in groups:
        for a in range(start, end, limit):                # safety net; Silero has already split long speech
            piece = audio[a: min(end, a + limit)]
            if pieces and len(piece) < MIN_PIECE_S * RATE:
                pieces[-1] = np.concatenate([pieces[-1], piece])
            else:
                pieces.append(piece)
    return pieces


def _dedupe(text: str, max_repeat: int = 3) -> str:
    """Whisper can get stuck repeating a word or phrase; keep at most `max_repeat` copies in a row."""
    words = text.split()
    for size in range(1, 7):
        out: list[str] = []
        i = 0
        while i < len(words):
            unit = words[i: i + size]
            reps = 1
            if len(unit) == size:
                while words[i + reps * size: i + (reps + 1) * size] == unit:
                    reps += 1
            if reps > max_repeat:
                out.extend(unit * max_repeat)
                i += size * reps
            else:
                out.append(words[i])
                i += 1
        words = out
    return " ".join(words)


class Transcriber:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.ready = threading.Event()
        self.error: str | None = None
        self.en = None
        self.bn = None
        self.en_path = self.bn_path = ""
        self.bn_limit_s = CHUNK_S
        self.vox = None                      # VoxLingua107 language-ID helper (voice/lid.py)
        self._lock = threading.Lock()        # one decode at a time on the GPU

    def load(self) -> None:
        """Load both models (call on the thread that will also decode, or a long-lived worker)."""
        try:
            runtime.setup()
            from faster_whisper import WhisperModel
            dev, ct = self.cfg["device"], self.cfg["compute_type"]
            self.en_path = model_path(self.cfg["english_model"])
            self.bn_path = model_path(self.cfg["bangla_model"])
            self.en = WhisperModel(self.en_path, device=dev, compute_type=ct)
            self.bn = WhisperModel(self.bn_path, device=dev, compute_type=ct)
            # A model without a Bangla-aware tokenizer spends ~3x the tokens per word; give it shorter
            # pieces so a dense one cannot run into the 448-token decoder limit and lose its ending.
            sample = "আমি আজ সকালে বাজারে গিয়ে অনেক কিছু কিনেছি এবং বাড়ি ফিরে রান্না করেছি"
            per_word = len(self.bn.hf_tokenizer.encode(sample, add_special_tokens=False).ids) / len(sample.split())
            self.bn_limit_s = CHUNK_S if per_word <= 4 else 14.0
            warm = np.zeros(RATE, dtype=np.float32)
            for m, lang in ((self.en, "en"), (self.bn, "bn")):
                list(m.transcribe(warm, language=lang, beam_size=1)[0])
            if self.cfg.get("voxlingua", True):
                self.vox = lid.VoxLingua()
                self.vox.load()
                if not self.vox.available:
                    print("language-ID helper unavailable, using Whisper alone:", self.vox.error, flush=True)
                    self.vox = None
                else:
                    self.vox.margin(np.zeros(RATE, dtype=np.float32) + 1e-4)       # warm-up
        except Exception as e:               # noqa: BLE001
            self.error = repr(e)
            raise
        finally:
            self.ready.set()

    # -- language ---------------------------------------------------------------------------------
    def language_probs(self, speech: np.ndarray) -> dict:
        """speech: audio with the silence already removed (the same audio that will be decoded)."""
        _, _, probs = self.en.detect_language(speech[: 30 * RATE], vad_filter=False,
                                              language_detection_segments=1)
        probs = dict(probs)
        return {"en": probs.get("en", 0.0), "bn": sum(probs.get(k, 0.0) for k in BANGLA_FAMILY)}

    # -- decoding ---------------------------------------------------------------------------------
    def _decode(self, model, pieces: list[np.ndarray], language: str) -> tuple[str, float]:
        """Decode each piece on its own.

        Fixed 30 s windows cut words in half and let one bad window derail the next; pieces that end in a
        pause, decoded independently, are what the best long-form Bangla systems use. Silence is dropped
        before decoding because that is where Whisper invents text."""
        texts, lp, n = [], 0.0, 0
        for piece in pieces:
            segments, _ = model.transcribe(
                piece,
                language=language,
                task="transcribe",
                beam_size=self.cfg["beam_size"],
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                condition_on_previous_text=False,
                without_timestamps=True,           # one piece = one window: no timestamp tokens to get wrong
                vad_filter=False,
                # the ratio is bytes/zlib bytes: correct Bangla (3 bytes per letter) reaches ~2.9, so the
                # English threshold would throw away good long sentences and re-decode them by sampling
                compression_ratio_threshold=2.4 if language == "en" else 3.5,
            )
            for s in segments:
                texts.append(s.text)
                k = max(1, len(s.tokens))
                lp += s.avg_logprob * k
                n += k
        return _clean(_dedupe(" ".join(texts)), language), (lp / n if n else -9.0)

    def _pieces(self, audio: np.ndarray, language: str) -> list[np.ndarray]:
        return speech_chunks(audio, self.bn_limit_s if language == "bn" else CHUNK_S)

    def transcribe(self, audio: np.ndarray, language: str = "auto") -> Result:
        """audio: float32 mono 16 kHz. language: "auto", "en" or "bn"."""
        self.ready.wait()
        if self.error:
            raise RuntimeError(self.error)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        t0 = time.perf_counter()
        duration = len(audio) / RATE

        def result(text, lang, lp, probs):
            return Result(text, lang, time.perf_counter() - t0, duration, lp, probs)

        with self._lock:
            pieces = speech_chunks(audio)
            if not pieces:                                   # no speech at all: nothing to guess from
                return result("", language if language in ("en", "bn") else "en", -9.0, {})
            probs: dict = {}
            if language not in ("en", "bn"):
                # see voice/lid.py: Whisper's guess + VoxLingua107 + how well the audio decodes as English
                speech = np.concatenate(pieces)
                _, _, all_probs = self.en.detect_language(speech[: 30 * RATE], vad_filter=False,
                                                          language_detection_segments=1)
                all_probs = dict(all_probs)
                vl = self.vox.margin(speech) if self.vox else None
                en_text, en_lp = self._decode(self.en, pieces, "en")
                s = lid.score(all_probs, vl, en_lp)
                probs = {"en": all_probs.get("en", 0.0), "bn": all_probs.get("bn", 0.0),
                         "voxlingua_en_minus_bn": vl, "score": s}
                if s >= 0:
                    return result(en_text, "en", en_lp, probs)
                language = "bn"
            if language == "bn":
                text, lp = self._decode(self.bn, self._pieces(audio, "bn"), "bn")
            else:
                text, lp = self._decode(self.en, pieces, "en")
        return result(text, language, lp, probs)


def load_audio(source) -> np.ndarray:
    """A path or file-like object in any container/codec -> float32 mono 16 kHz."""
    runtime.setup()
    from faster_whisper import decode_audio
    return decode_audio(source, sampling_rate=RATE)
