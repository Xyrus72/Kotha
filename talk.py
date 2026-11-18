"""Hands-free voice chat in the terminal: just talk; each time you stop, your words are printed and the
local LLM answers out loud.

    .venv\\Scripts\\python.exe talk.py                  auto-detect English / Bangla
    .venv\\Scripts\\python.exe talk.py --lang bn        always Bangla (most accurate when you know the language)
    .venv\\Scripts\\python.exe talk.py --quiet          print the reply without speaking it
    .venv\\Scripts\\python.exe talk.py --no-llm         dictation only (speech -> text)
    .venv\\Scripts\\python.exe talk.py --file clip.wav  transcribe a recording instead of the microphone
    .venv\\Scripts\\python.exe talk.py --devices        list microphones
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

from voice import config


class Voice:
    """Speaks sentences in order on a background thread; `busy` is set while anything is queued or playing
    (and for a short tail after), so the microphone can be ignored meanwhile."""

    TAIL_S = 0.4                                    # room echo after the last word

    def __init__(self, speaker):
        self.speaker = speaker
        self.q: queue.Queue = queue.Queue()
        self.busy = threading.Event()
        self.pending = 0
        self.lock = threading.Lock()
        threading.Thread(target=self._run, name="voice", daemon=True).start()

    def say(self, sentence: str) -> None:
        with self.lock:
            self.pending += 1
            self.busy.set()
        self.q.put(sentence)

    def _run(self) -> None:
        import sounddevice as sd
        while True:
            sentence = self.q.get()
            try:
                samples, rate = self.speaker.synth(sentence)
                if len(samples):
                    sd.play(samples, rate)
                    sd.wait()
            except Exception as e:                  # noqa: BLE001
                print(f"\n(could not speak: {e})", flush=True)
            with self.lock:
                self.pending -= 1
                last = self.pending == 0
            if last:
                time.sleep(self.TAIL_S)
                with self.lock:
                    if self.pending == 0:
                        self.busy.clear()

    def wait(self) -> None:
        while self.busy.is_set():
            time.sleep(0.05)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=("auto", "en", "bn"), default="auto")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="do not speak the replies")
    ap.add_argument("--file")
    ap.add_argument("--devices", action="store_true")
    ap.add_argument("--device", type=int)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf8")
    cfg = config.load()

    if args.devices:
        from voice.mic import list_inputs
        for i, name in list_inputs():
            print(f"{i:3}  {name}")
        return 0

    device = args.device if args.device is not None else cfg["mic_device"]
    if not args.file:
        from voice.mic import check_input
        problem = check_input(device)
        if problem:
            print(f"That microphone cannot be used ({problem}). Run with --devices to see the usable ones.")
            return 1
    try:                                    # two copies of the speech models do not fit on the GPU
        import urllib.request
        urllib.request.urlopen(f"http://{cfg['host']}:{cfg['port']}/api/ready", timeout=1)
        print("Voice Chat (the browser app) is already running and is using the GPU.\n"
              f"Close its window first, or use the page at http://{cfg['host']}:{cfg['port']}")
        return 1
    except OSError:
        pass

    from voice.stt import Transcriber, load_audio
    print("loading speech models…", flush=True)
    stt = Transcriber(cfg)
    stt.load()

    chat = None
    if not args.no_llm:
        from voice.llm import Chat
        chat = Chat.from_config(cfg)
        problem = chat.available()
        if problem:
            print(f"(no replies: {problem})")
            chat = None
        else:
            try:
                chat.warm()
            except Exception as e:                      # noqa: BLE001
                print(f"(model not warmed up: {e}; it will load on the first reply)")

    voice = None
    if chat and not args.quiet and cfg.get("speak_replies", True):
        from voice.tts import Speaker
        speaker = Speaker(cfg)
        speaker.load()
        if speaker.error:
            print(f"(replies will not be spoken: {speaker.error})")
        else:
            voice = Voice(speaker)

    def handle(audio) -> None:
        try:
            r = stt.transcribe(audio, args.lang)
        except Exception as e:                          # noqa: BLE001
            print(f"\n(transcription failed: {e})", flush=True)
            return
        if not r.text:
            return
        print(f"\nYou [{r.language}, {r.seconds:.1f}s]: {r.text}", flush=True)
        if chat:
            from voice.tts import SentenceBuffer
            sentences = SentenceBuffer()
            print("AI: ", end="", flush=True)
            try:
                for piece in chat.reply(r.text, r.language):
                    print(piece, end="", flush=True)
                    if voice:
                        for s in sentences.push(piece):
                            voice.say(s)
            except Exception as e:                      # noqa: BLE001
                print(f"(no reply: {e})", end="")
            if voice:
                for s in sentences.flush():
                    voice.say(s)
            print(flush=True)

    if args.file:
        handle(load_audio(args.file))
        if voice:
            voice.wait()
        return 0

    from voice.mic import utterances
    print("Listening — speak in English or Bangla. Ctrl+C to stop.", flush=True)
    try:
        for clip in utterances(device, end_silence_s=cfg["end_silence_s"],
                               muted=(voice.busy.is_set if voice else None)):
            handle(clip)
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
