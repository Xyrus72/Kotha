"""Text -> speech, fully offline on the CPU (sherpa-onnx).

English: Kokoro-82M (int8). Bangla: a VITS voice trained on Bangla (Coqui "bn custom female").
Text is spoken sentence by sentence so the first words play while the rest is still being written.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np

from .config import ROOT

TTS_DIR = ROOT / "models" / "tts"
BN_CHARS = re.compile(r"[ঀ-৿]")

# Sentence ends: . ! ? and the Bangla danda, followed by space or end. Newlines also end a sentence.
_SENT_END = re.compile(r"(?<=[.!?।॥])\s+|\n+")


def detect_language(text: str) -> str:
    """'bn' when the text is mostly Bengali script, else 'en'."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    bn = sum(1 for c in letters if BN_CHARS.match(c))
    return "bn" if bn >= len(letters) * 0.3 else "en"


def clean_for_speech(text: str) -> str:
    """Drop what should not be read aloud: markdown marks, URLs, emoji, bullet symbols."""
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[*_#`>|~]+", " ", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    text = re.sub(r"[\U0001F000-\U0001FAFF☀-➿️]", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_END.split(text) if s and s.strip()]


class SentenceBuffer:
    """Feed streamed text in; get complete sentences out as soon as each one ends."""

    def __init__(self):
        self.buf = ""

    def push(self, piece: str) -> list[str]:
        self.buf += piece
        parts = _SENT_END.split(self.buf)
        if len(parts) == 1:
            return []
        self.buf = parts[-1]
        return [p.strip() for p in parts[:-1] if p.strip()]

    def flush(self) -> list[str]:
        rest, self.buf = self.buf.strip(), ""
        return [rest] if rest else []


class Speaker:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.en = None
        self.bn = None
        self.error: str | None = None
        self.ready = threading.Event()
        self._locks = {"en": threading.Lock(), "bn": threading.Lock()}

    def load(self) -> None:
        try:
            import sherpa_onnx as so
            threads = int(self.cfg.get("tts_threads", 4))
            k = TTS_DIR / self.cfg.get("tts_english", "kokoro-int8-multi-lang-v1_0")
            self.en = so.OfflineTts(so.OfflineTtsConfig(
                model=so.OfflineTtsModelConfig(
                    kokoro=so.OfflineTtsKokoroModelConfig(
                        model=str(next(k.glob("model*.onnx"))), voices=str(k / "voices.bin"),
                        tokens=str(k / "tokens.txt"), data_dir=str(k / "espeak-ng-data"),
                        dict_dir=str(k / "dict"),
                        lexicon=f"{k / 'lexicon-us-en.txt'},{k / 'lexicon-zh.txt'}", lang="en-us"),
                    num_threads=threads, provider="cpu"),
                max_num_sentences=1))
            b = TTS_DIR / self.cfg.get("tts_bangla", "vits-coqui-bn-custom_female")
            self.bn = so.OfflineTts(so.OfflineTtsConfig(
                model=so.OfflineTtsModelConfig(
                    vits=so.OfflineTtsVitsModelConfig(
                        model=str(next(b.glob("*.onnx"))), tokens=str(b / "tokens.txt"),
                        data_dir=str(b / "espeak-ng-data") if (b / "espeak-ng-data").is_dir() else "",
                        length_scale=float(self.cfg.get("tts_bangla_length", 1.0))),
                    num_threads=threads, provider="cpu"),
                max_num_sentences=1))
        except Exception as e:                       # noqa: BLE001
            self.error = repr(e)
        finally:
            self.ready.set()

    RATE = 24000                                     # every clip is returned at this rate

    def runs(self, text: str, language: str | None = None) -> list[tuple[str, str]]:
        """Split a sentence into (language, text) parts, each read by the right voice.

        The Bangla voice reads only Bengali letters, so in a Bangla sentence numbers become Bangla words and
        English words (common in Bangladeshi speech: 'laptop', 'RAM', 'online') go to the English voice.
        In an English sentence, a Bengali-script word goes to the Bangla voice."""
        from .bn_numbers import spell_numbers
        text = clean_for_speech(text)
        if not re.search(r"\w", text):
            return []
        main = language if language in ("en", "bn") else detect_language(text)
        if main == "bn":
            # "রাজধানী (Capital)": an English gloss in brackets repeats the Bangla word - skip it
            text = re.sub(r"\(\s*[A-Za-z][^)ঀ-৿]*\)", " ", text)
            text = spell_numbers(text)
        else:
            text = text.translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789"))
        text = re.sub(r"[()\[\]{}\"]", " ", text)
        # Bengali-script stretches vs everything else; spaces/punctuation stay with the run they follow
        parts = re.findall(r"[ঀ-৿‌‍]+(?:[^\wঀ-৿]+[ঀ-৿‌‍]+)*[^\wঀ-৿]*"
                           r"|[^ঀ-৿]+", text)
        out: list[tuple[str, str]] = []
        for p in parts:
            lang = "bn" if BN_CHARS.search(p) else "en"
            if not re.search(r"[^\W_]", p):              # punctuation only
                continue
            if out and out[-1][0] == lang:
                out[-1] = (lang, out[-1][1] + " " + p)
            else:
                out.append((lang, p))
        return [(lang, " ".join(t.split())) for lang, t in out]

    def _bn_chars(self, text: str) -> str:
        """Only characters the Bangla voice knows; the danda becomes a pause."""
        if not hasattr(self, "_bn_tokens"):
            path = TTS_DIR / self.cfg.get("tts_bangla", "vits-coqui-bn-custom_female") / "tokens.txt"
            self._bn_tokens = {line.rsplit(" ", 1)[0] for line in path.read_text(encoding="utf8").splitlines()}
        out = []
        for ch in text:
            if ch in self._bn_tokens or ch == " ":
                out.append(ch)
            else:
                import unicodedata
                dec = unicodedata.normalize("NFD", ch)
                out.append(dec if all(c in self._bn_tokens for c in dec) else " ")
        return " ".join("".join(out).split())

    def _one(self, lang: str, text: str) -> np.ndarray:
        if lang == "bn":
            text = self._bn_chars(text)
            model, sid, speed = self.bn, 0, float(self.cfg.get("tts_bangla_speed", 1.0))
        else:
            model, sid, speed = self.en, int(self.cfg.get("tts_english_voice", 3)), \
                float(self.cfg.get("tts_english_speed", 1.0))
        if not re.search(r"[^\W_]", text):
            return np.zeros(0, dtype=np.float32)
        with self._locks[lang]:
            audio = model.generate(text, sid=sid, speed=speed)
        samples = np.asarray(audio.samples, dtype=np.float32)
        if audio.sample_rate != self.RATE and len(samples):
            n = int(round(len(samples) * self.RATE / audio.sample_rate))
            samples = np.interp(np.linspace(0, len(samples) - 1, n), np.arange(len(samples)), samples)
        return samples.astype(np.float32)

    def synth(self, text: str, language: str | None = None) -> tuple[np.ndarray, int]:
        """-> (float32 samples, sample rate). Empty array when there is nothing to say."""
        self.ready.wait()
        if self.error:
            raise RuntimeError(self.error)
        gap = np.zeros(int(0.06 * self.RATE), dtype=np.float32)
        clips = []
        for lang, part in self.runs(text, language):
            a = self._one(lang, part)
            if len(a):
                clips += [a, gap]
        if not clips:
            return np.zeros(0, dtype=np.float32), self.RATE
        return np.concatenate(clips[:-1]), self.RATE


def to_wav_bytes(samples: np.ndarray, rate: int) -> bytes:
    import io
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, samples, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
