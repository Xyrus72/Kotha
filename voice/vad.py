"""Silero VAD as a stream: one speech probability per 512-sample window (32 ms at 16 kHz).

Uses the model file that ships inside faster-whisper (onnxruntime, CPU). The LSTM state is carried between
pushes so chunked audio scores the same as a whole clip, and forgotten after two quiet seconds: carried
forever it drifts in a real room until quiet speech stops registering.
"""
from __future__ import annotations

import threading

import numpy as np

from . import runtime

RATE = 16000
WINDOW = 512
CONTEXT = 64
QUIET_P = 0.2
QUIET_RESET_S = 2.0

_session = None
_state_names: tuple[str, ...] = ("h", "c")
_lock = threading.Lock()


def _load():
    global _session, _state_names
    with _lock:
        if _session is not None:
            return
        runtime.setup()
        from faster_whisper.vad import get_vad_model
        session = get_vad_model().session
        names = {i.name for i in session.get_inputs()}
        _state_names = ("h", "c") if {"h", "c"} <= names else ("state",)
        _session = session


class SileroStream:
    def __init__(self):
        _load()
        self._quiet_max = max(1, round(QUIET_RESET_S * RATE / WINDOW))
        self.reset()

    def reset(self) -> None:
        self._rest = np.zeros(0, dtype=np.float32)
        self._forget()

    def _forget(self) -> None:
        self._quiet = 0
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)
        if _state_names == ("state",):
            self._state = {"state": np.zeros((2, 1, 128), dtype=np.float32)}
        else:
            self._state = {n: np.zeros((1, 1, 128), dtype=np.float32) for n in _state_names}

    def push(self, audio: np.ndarray) -> list[float]:
        """audio: float32 mono 16 kHz in [-1, 1]. Returns one probability per complete window."""
        data = np.concatenate([self._rest, audio.astype(np.float32, copy=False)])
        n = len(data) // WINDOW
        self._rest = data[n * WINDOW:]
        out: list[float] = []
        for w in data[: n * WINDOW].reshape(n, WINDOW):
            x = np.concatenate([self._context, w[None, :]], axis=1)
            with _lock:
                res = _session.run(None, {"input": x, **self._state})
            p = float(np.asarray(res[0]).reshape(-1)[0])
            out.append(p)
            self._state = dict(zip(_state_names, res[1:]))
            self._context = w[None, -CONTEXT:]
            self._quiet = self._quiet + 1 if p < QUIET_P else 0
            if self._quiet >= self._quiet_max:
                self._forget()
        return out
