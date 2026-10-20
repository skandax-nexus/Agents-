"""Ollama backend - the default offline engine (https://ollama.com)."""
from __future__ import annotations

from typing import Sequence

from .base import ChatMessage, LLMBackend, LLMError
from .http import get_json, post_json

DEFAULT_MODEL = "llama3.1:8b"


class OllamaBackend(LLMBackend):
    name = "ollama"
    supports_offline = True

    def __init__(self, base_url: str = "http://localhost:11434", model: str | None = None,
                 *, temperature: float = 0.2, max_tokens: int = 4096, num_ctx: int = 16384,
                 timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model or DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.timeout = timeout

    # -- liveness -----------------------------------------------------------
    def available(self) -> bool:
        try:
            get_json(f"{self.base_url}/api/tags", timeout=2.0)
            return True
        except Exception:
            return False

    def installed_models(self) -> list[str]:
        try:
            data = get_json(f"{self.base_url}/api/tags", timeout=3.0)
        except Exception:
            return []
        if isinstance(data, dict):
            return [m.get("name", "") for m in data.get("models", [])]
        return []

    def pick_model(self) -> str:
        """If the configured model is not pulled, fall back to any installed one."""
        installed = self.installed_models()
        if not installed:
            return self.model
        if self.model in installed or any(m.startswith(self.model) for m in installed):
            return self.model
        return installed[0]

    # -- chat ---------------------------------------------------------------
    def chat(self, system: str, messages: Sequence[ChatMessage], *, temperature=None, max_tokens=None) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}] + [m.as_dict() for m in messages],
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": self.max_tokens if max_tokens is None else max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        data = post_json(f"{self.base_url}/api/chat", payload, timeout=self.timeout)
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected Ollama response: {str(data)[:300]}") from exc
