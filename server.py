"""Voice Chat web app: speak in the browser -> text -> the local LLM replies.

    .venv\\Scripts\\python.exe server.py        then open http://127.0.0.1:8765
"""
from __future__ import annotations

import io
import json
import queue
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from voice import config
from voice.llm import Chat
from voice.stt import Transcriber, load_audio
from voice.tts import Speaker, to_wav_bytes

ROOT = Path(__file__).resolve().parent
cfg = config.load()
stt = Transcriber(cfg)
speaker = Speaker(cfg)
chat = Chat.from_config(cfg)
app = FastAPI(title="Voice Chat")

# All GPU speech work happens on one long-lived thread: CTranslate2's CUDA handles are per-thread, and a
# model loaded on one thread and used on another crashes the process at exit.
_jobs: queue.Queue = queue.Queue()


def _stt_worker() -> None:
    try:
        stt.load()
    except Exception as e:                          # noqa: BLE001
        print("speech models failed to load:", e, flush=True)
    while True:
        audio, language, out = _jobs.get()
        try:
            out.put(stt.transcribe(audio, language))
        except Exception as e:                      # noqa: BLE001
            out.put(e)


def transcribe(audio, language: str):
    out: queue.Queue = queue.Queue(1)
    _jobs.put((audio, language, out))
    result = out.get()
    if isinstance(result, Exception):
        raise result
    return result


def _line(**event) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf8")


def _reply_events(text: str, language: str | None):
    problem = chat.available()
    if problem:
        yield _line(type="error", message=problem)
        return
    try:
        for piece in chat.reply(text, language):
            yield _line(type="delta", text=piece)
    except Exception as e:                          # noqa: BLE001
        yield _line(type="error", message=f"LLM error: {e}")
    yield _line(type="done")


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/ready")
def ready():
    """Speech models only - never waits on Ollama (the launcher polls this)."""
    return {"stt_ready": stt.ready.is_set() and not stt.error, "stt_error": stt.error}


@app.get("/api/status")
def status():
    return {
        "stt_ready": stt.ready.is_set() and not stt.error,
        "stt_error": stt.error,
        "stt": f"{Path(getattr(stt, 'en_path', '')).name} + {Path(getattr(stt, 'bn_path', '')).name}",
        "llm": cfg["llm_model"],
        "llm_error": chat.available(),
        "tts_ready": speaker.ready.is_set() and not speaker.error,
        "tts_error": speaker.error,
        "speak_replies": bool(cfg["speak_replies"]),
    }


class SpeakIn(BaseModel):
    text: str
    language: str | None = None


@app.post("/api/speak")
def api_speak(body: SpeakIn):
    """One sentence of text -> WAV (English or Bangla, picked from the script when not given)."""
    try:
        samples, rate = speaker.synth(body.text, body.language)
    except Exception as e:                          # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=503)
    if not len(samples):
        return Response(status_code=204)
    return Response(to_wav_bytes(samples, rate), media_type="audio/wav")


@app.post("/api/transcribe")
def api_transcribe(audio: UploadFile = File(...), language: str = Form("auto")):
    """Speech -> text only (no reply)."""
    r = transcribe(load_audio(io.BytesIO(audio.file.read())), language)
    return {"text": r.text, "language": r.language, "seconds": round(r.seconds, 3),
            "duration": round(r.duration, 2), "avg_logprob": round(r.avg_logprob, 3),
            "language_probs": r.language_probs}


@app.post("/api/talk")
def api_talk(audio: UploadFile = File(...), language: str = Form("auto"), redo: bool = Form(False)):
    """redo: the previous message was heard in the wrong language - forget that exchange first."""
    data = audio.file.read()
    if redo:
        chat.drop_last()

    def events():
        try:
            r = transcribe(load_audio(io.BytesIO(data)), language)
        except Exception as e:                      # noqa: BLE001
            yield _line(type="error", message=f"Could not transcribe: {e}")
            return
        yield _line(type="transcript", text=r.text, language=r.language, seconds=r.seconds,
                    avg_logprob=r.avg_logprob)
        if r.text:
            yield from _reply_events(r.text, r.language)

    return StreamingResponse(events(), media_type="application/x-ndjson")


class ChatIn(BaseModel):
    text: str


@app.post("/api/chat")
def api_chat(body: ChatIn):
    return StreamingResponse(_reply_events(body.text.strip(), None), media_type="application/x-ndjson")


@app.post("/api/reset")
def api_reset():
    """New chat. In customer-care mode the agent opens the call with its greeting, like a real agent."""
    greeting = cfg.get("greeting", "") if cfg.get("mode", "care") == "care" else ""
    if greeting:
        chat.greet(greeting)
    else:
        chat.reset()
    return JSONResponse({"ok": True, "greeting": greeting})


def _warm_llm() -> None:
    if chat.available() is None:
        try:
            chat.warm()
        except Exception as e:                      # noqa: BLE001
            print("LLM warm-up failed:", e, flush=True)


if __name__ == "__main__":
    threading.Thread(target=_stt_worker, name="stt", daemon=True).start()
    threading.Thread(target=_warm_llm, name="llm-warm", daemon=True).start()
    threading.Thread(target=speaker.load, name="tts-load", daemon=True).start()
    print(f"Voice Chat: http://{cfg['host']}:{cfg['port']}", flush=True)
    uvicorn.run(app, host=cfg["host"], port=cfg["port"], log_level="warning")
