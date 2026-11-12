"""The reply side: a local LLM behind Ollama, streamed, with conversation history."""
from __future__ import annotations

from typing import Iterator

SYSTEM = (
    "You are a helpful voice assistant running locally on the user's computer. The user speaks to you and "
    "their speech is transcribed to text, so a word may occasionally be misheard: answer what they most "
    "plausibly meant. Always reply in the language the user used: Bangla (in Bengali script) for Bangla, "
    "English for English. Be clear and direct, like a knowledgeable friend. Write plain text without "
    "markdown symbols such as ** or #."
)


class Chat:
    def __init__(self, model: str, host: str = "http://127.0.0.1:11434", system: str = SYSTEM,
                 num_ctx: int = 8192, keep_alive: str = "30m", max_turns: int = 20,
                 think: bool | None = None):
        import ollama
        self.extra = {} if think is None else {"think": think}
        self.client = ollama.Client(host=host)
        self.probe = ollama.Client(host=host, timeout=2)     # status checks must not hang when Ollama is down
        self.model = model
        self.system = system
        self.options = {"num_ctx": num_ctx}
        self.keep_alive = keep_alive
        self.max_turns = max_turns
        self.history: list[dict] = []

    def available(self) -> str | None:
        """None when the model can be used, otherwise a message saying what is missing."""
        try:
            names = [m.model for m in self.probe.list().models]
        except Exception as e:                      # noqa: BLE001
            return "Ollama is not running (no replies yet)"
        if not any(n == self.model or n.split(":")[0] == self.model for n in names):
            return f"model {self.model!r} is not pulled (have: {', '.join(names) or 'none'})"
        return None

    def warm(self) -> None:
        self.client.generate(model=self.model, prompt="", keep_alive=self.keep_alive, options=self.options)

    def reset(self) -> None:
        self.history.clear()

    def drop_last(self) -> None:
        """Forget the last exchange (used when the user says it was heard in the wrong language)."""
        if self.history and self.history[-1]["role"] == "assistant":
            self.history.pop()
        if self.history and self.history[-1]["role"] == "user":
            self.history.pop()

    def reply(self, text: str, language: str | None = None) -> Iterator[str]:
        """Stream the reply to `text`; the exchange is added to the history when the stream ends."""
        self.history.append({"role": "user", "content": text})
        self.history = self.history[-2 * self.max_turns:]
        system = self.system
        if language == "bn":
            system += " The user just spoke Bangla: reply in Bangla."
        elif language == "en":
            system += " The user just spoke English: reply in English."
        messages = [{"role": "system", "content": system}, *self.history]
        parts: list[str] = []
        try:
            for chunk in self.client.chat(model=self.model, messages=messages, stream=True,
                                          keep_alive=self.keep_alive, options=self.options, **self.extra):
                piece = chunk.message.content or ""
                if piece:
                    parts.append(piece)
                    yield piece
        finally:
            answer = "".join(parts).strip()
            if answer:
                self.history.append({"role": "assistant", "content": answer})
            else:
                self.history.pop()
