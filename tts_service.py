"""Expressive voice service (OmniVoice, GPU) - run with the separate environment:

    .venv-tts\\Scripts\\python.exe tts_service.py            (port 8766)

POST /speak {"text": "...", "speed": 1.0}  -> audio/wav (24 kHz)
GET  /health                                -> {"ready": true, ...}

The agent's voice is one fixed reference clip (voices/mitu.wav + voices/mitu.txt), so every sentence, in Bangla
or English, sounds like the same person. If the clip does not exist it is DESIGNED once from a text description
(no real person's voice is copied): see make_reference(). Replace voices/mitu.wav with a recording of a consenting
speaker (and mitu.txt with exactly what they said) to use a real voice instead.
"""
from __future__ import annotations

import io
import os
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "omnivoice"
VOICE_DIR = ROOT / "voices"
REF_WAV, REF_TXT = VOICE_DIR / "mitu.wav", VOICE_DIR / "mitu.txt"
RATE = 24000
DESIGN = os.environ.get("VOICE_DESIGN", "female, young adult, moderate pitch")
DESIGN_TEXT = "আসসালামু আলাইকুম, নির্ভানা গ্যাজেটস থেকে মিতু বলছি। বলুন, আজকে আপনাকে কীভাবে সাহায্য করতে পারি?"
STEPS = int(os.environ.get("VOICE_STEPS", "16"))          # diffusion steps: 16 is the documented fast setting

app = FastAPI(title="Voice service")
model = None
lock = threading.Lock()
state = {"ready": False, "error": None}


def make_reference() -> None:
    """Design the agent's voice once and keep it, so later sentences can be spoken in that same voice."""
    VOICE_DIR.mkdir(exist_ok=True)
    audio = model.generate(text=DESIGN_TEXT, instruct=DESIGN, num_step=32)[0]
    sf.write(REF_WAV, audio, RATE)
    REF_TXT.write_text(DESIGN_TEXT, encoding="utf8")


def load() -> None:
    global model
    try:
        from omnivoice import OmniVoice
        source = str(MODEL_DIR) if (MODEL_DIR / "config.json").exists() else "k2-fsa/OmniVoice"
        model = OmniVoice.from_pretrained(source, device_map="cuda:0", dtype=torch.float16)
        if not REF_WAV.exists():
            make_reference()
        speak("আচ্ছা।")                                           # warm-up (first call compiles kernels)
        state["ready"] = True
    except Exception as e:                                          # noqa: BLE001
        state["error"] = repr(e)
        print("voice service failed to load:", e, flush=True)


def speak(text: str, speed: float = 1.0) -> np.ndarray:
    with lock, torch.inference_mode():
        audio = model.generate(text=text, ref_audio=str(REF_WAV), ref_text=REF_TXT.read_text(encoding="utf8"),
                               num_step=STEPS, speed=speed)[0]
    return np.asarray(audio, dtype=np.float32)


class SpeakIn(BaseModel):
    text: str
    speed: float = 1.0


@app.get("/health")
def health():
    mem = torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else 0
    return {**state, "voice": DESIGN, "gpu_mb": round(mem)}


@app.post("/speak")
def api_speak(body: SpeakIn):
    if not state["ready"]:
        return JSONResponse({"error": state["error"] or "loading"}, status_code=503)
    t0 = time.perf_counter()
    audio = speak(body.text.strip(), body.speed)
    buf = io.BytesIO()
    sf.write(buf, audio, RATE, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav",
                    headers={"X-Synth-Seconds": f"{time.perf_counter() - t0:.3f}"})


if __name__ == "__main__":
    threading.Thread(target=load, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("VOICE_PORT", "8766")), log_level="warning")
