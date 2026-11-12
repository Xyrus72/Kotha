"""Microphone capture with Silero endpointing: yields one float32 16 kHz clip per spoken utterance.

An utterance starts when speech is detected and ends after `end_silence_s` of quiet, so pauses in the middle
of a sentence do not cut it. A little audio from before the onset is kept so first syllables are not lost.
Capture keeps running while the caller is busy with an utterance: what is said meanwhile is queued, not lost.
"""
from __future__ import annotations

import queue
from collections import deque
from typing import Callable, Iterator

import numpy as np

from .vad import RATE, WINDOW, SileroStream

BLOCK = WINDOW * 4                 # 128 ms per callback


def list_inputs() -> list[tuple[int, str]]:
    """Input devices that can really be opened at 16 kHz mono, labelled with their Windows audio API."""
    import sounddevice as sd
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 1:
            continue
        try:
            sd.check_input_settings(device=i, samplerate=RATE, channels=1, dtype="float32")
        except Exception:                               # noqa: BLE001 - e.g. WASAPI refuses 16 kHz
            continue
        out.append((i, f"{' '.join(d['name'].split())} [{sd.query_hostapis(d['hostapi'])['name']}]"))
    return out


def check_input(device: int | None) -> str | None:
    """None when the device can be opened, otherwise why not."""
    import sounddevice as sd
    try:
        sd.check_input_settings(device=device, samplerate=RATE, channels=1, dtype="float32")
    except Exception as e:                              # noqa: BLE001
        return str(e)
    return None


def utterances(device: int | None = None, *, onset_p: float = 0.5, offset_p: float = 0.35,
               end_silence_s: float = 1.2, pre_roll_s: float = 0.4, min_speech_s: float = 0.25,
               max_s: float = 120.0, on_state: Callable[[str], None] | None = None,
               stop: Callable[[], bool] | None = None,
               muted: Callable[[], bool] | None = None) -> Iterator[np.ndarray]:
    """muted: while it returns True the microphone is ignored (e.g. while the reply is being spoken
    through the speakers, so the assistant does not hear itself)."""
    import sounddevice as sd

    q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    vad = SileroStream()
    win_s = WINDOW / RATE
    pre = deque(maxlen=max(1, round(pre_roll_s / win_s)))
    speech: list[np.ndarray] = []
    in_speech = False
    onset = quiet = n_voiced = 0
    rest = np.zeros(0, dtype=np.float32)

    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=BLOCK,
                        device=device, callback=callback):
        if on_state:
            on_state("listening")
        while not (stop and stop()):
            try:
                block = q.get(timeout=0.2)
            except queue.Empty:
                continue
            if muted and muted():
                in_speech, onset, n_voiced, speech = False, 0, 0, []
                pre.clear()
                rest = np.zeros(0, dtype=np.float32)
                vad.reset()
                continue
            probs = vad.push(block)                   # vad and `rest` carry the same remainder
            data = np.concatenate([rest, block])
            n = len(data) // WINDOW
            rest = data[n * WINDOW:]
            for w, p in zip(data[: n * WINDOW].reshape(n, WINDOW), probs):
                if not in_speech:
                    pre.append(w)
                    onset = onset + 1 if p >= onset_p else 0
                    if onset >= 2:
                        in_speech, quiet, n_voiced = True, 0, 2
                        speech = list(pre)
                        pre.clear()
                        if on_state:
                            on_state("speech")
                    continue
                speech.append(w)
                if p >= offset_p:
                    n_voiced += 1
                    quiet = 0
                else:
                    quiet += 1
                if quiet * win_s >= end_silence_s or len(speech) * win_s >= max_s:
                    clip = np.concatenate(speech)
                    long_enough = n_voiced * win_s >= min_speech_s      # a click or cough is not speech
                    in_speech, onset, n_voiced, speech = False, 0, 0, []
                    if long_enough:
                        if on_state:
                            on_state("processing")
                        yield clip
                    if on_state:
                        on_state("listening")
