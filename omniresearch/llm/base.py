"""Backend-neutral chat interface.

All agents talk to the model through ``LLMBackend.chat``.  Tool use is handled
at the agent layer with a plain-text JSON protocol, which means the exact same
agents run on a 7B model in Ollama, a llama.cpp server, or the Claude API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


class LLMError(RuntimeError):
    """Raised when a backend cannot produce a completion."""


@dataclass
class ChatMessage:
    role: str   # "user" | "assistant"
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMBackend:
    """Abstract chat backend."""

    name: str = "abstract"
    model: str = ""
    supports_offline: bool = False

    def chat(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        raise NotImplementedError

    def available(self) -> bool:
        """Cheap liveness check used by auto-detection and `doctor`."""
        return False

    def describe(self) -> str:
        return f"{self.name} ({self.model})"
