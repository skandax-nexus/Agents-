"""OpenAI-compatible chat backend.

Works offline with llama.cpp's ``llama-server``, LM Studio, vLLM, text-generation-webui,
LocalAI, Jan, etc.  Anything that exposes ``POST /v1/chat/completions``.
"""
from __future__ import annotations

from typing import Sequence

from .base import ChatMessage, LLMBackend, LLMError
from .http import get_json, post_json


class OpenAICompatBackend(LLMBackend):
    name = "openai-compatible"
    supports_offline = True

    def __init__(self, base_url: str = "http://localhost:8080/v1", model: str | None = None,
                 api_key: str = "not-needed", *, temperature: float = 0.2, max_tokens: int = 4096,
                 timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model or "local-model"
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def available(self) -> bool:
        try:
            get_json(f"{self.base_url}/models", timeout=2.0)
            return True
        except Exception:
            return False

    def installed_models(self) -> list[str]:
        try:
            data = get_json(f"{self.base_url}/models", timeout=3.0)
        except Exception:
            return []
        if isinstance(data, dict):
            return [m.get("id", "") for m in data.get("data", [])]
        return []

    def pick_model(self) -> str:
        installed = self.installed_models()
        if installed and self.model not in installed:
            return installed[0]
        return self.model

    def chat(self, system: str, messages: Sequence[ChatMessage], *, temperature=None, max_tokens=None) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + [m.as_dict() for m in messages],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        data = post_json(f"{self.base_url}/chat/completions", payload, headers=self._headers(),
                         timeout=self.timeout)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response: {str(data)[:300]}") from exc
