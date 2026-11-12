"""Settings: defaults here, overridden by config.json next to the project root."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    # speech -> text
    "english_model": "models/whisper-large-v3",        # also decides which language was spoken
    "bangla_model": ["models/bn-tugstugi-medium", "models/bn-mozilla-large-v3"],
    "device": "cuda",
    "compute_type": "float16",
    "beam_size": 5,
    # reply
    "llm_model": "gemma4:12b-it-qat",
    "ollama_host": "http://127.0.0.1:11434",
    "llm_context": 8192,
    "llm_think": None,                                  # None = model default, False = no thinking phase
    # speaking the reply (offline, CPU)
    "speak_replies": True,
    "tts_english": "kokoro-multi-lang-v1_0",            # folder under models/tts (fp32: 3x faster than int8 on CPU)
    "tts_english_voice": 3,                             # Kokoro speaker id (3 = af_heart)
    "tts_english_speed": 1.0,
    "tts_bangla": "vits-coqui-bn-custom_female",
    "tts_bangla_speed": 1.0,
    "tts_threads": 8,
    # microphone (terminal mode)
    "mic_device": None,
    "end_silence_s": 1.2,
    # web app
    "host": "127.0.0.1",
    "port": 8765,
}


def load() -> dict:
    cfg = dict(DEFAULTS)
    path = ROOT / "config.json"
    if path.exists():
        cfg.update(json.loads(path.read_text(encoding="utf8")))
    return cfg


def model_path(value) -> str:
    """A model folder, relative to the project or absolute. A list means: the first one that exists."""
    values = [value] if isinstance(value, str) else list(value)
    paths = [p if p.is_absolute() else ROOT / p for p in map(Path, values)]
    for p in paths:
        if (p / "model.bin").exists():
            return str(p)
    raise FileNotFoundError(f"no model found at {', '.join(map(str, paths))}")
