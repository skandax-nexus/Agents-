"""Deterministic backends for tests and dry runs."""
from __future__ import annotations

from collections import deque
from typing import Callable, Sequence

from .base import ChatMessage, LLMBackend


class ScriptedBackend(LLMBackend):
    """Returns canned responses in order; the last one repeats once exhausted."""

    name = "mock"
    supports_offline = True

    def __init__(self, responses: Sequence[str] | None = None,
                 handler: Callable[[str, Sequence[ChatMessage]], str] | None = None) -> None:
        self.model = "scripted"
        self._responses = deque(responses or [])
        self._handler = handler
        self.calls: list[tuple[str, list[ChatMessage]]] = []

    def available(self) -> bool:
        return True

    def chat(self, system: str, messages: Sequence[ChatMessage], *, temperature=None, max_tokens=None) -> str:
        self.calls.append((system, list(messages)))
        if self._handler is not None:
            return self._handler(system, messages)
        if not self._responses:
            return '{"action": "final", "content": "No scripted response left."}'
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.popleft()


class EchoBackend(LLMBackend):
    """Answers every prompt with a short summary of the request - useful for `--backend mock` dry runs."""

    name = "mock"
    supports_offline = True
    model = "echo"

    def available(self) -> bool:
        return True

    def chat(self, system: str, messages: Sequence[ChatMessage], *, temperature=None, max_tokens=None) -> str:
        last = messages[-1].content if messages else ""
        snippet = last.strip().splitlines()[0][:160] if last.strip() else ""
        return '{"action": "final", "content": "[mock backend] Received task: ' + snippet.replace('"', "'") + '"}'
