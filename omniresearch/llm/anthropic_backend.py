"""Claude API backend (online).  Requires ``pip install anthropic`` and credentials
(ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an ``ant auth login`` profile)."""
from __future__ import annotations

import os
from typing import Sequence

from .base import ChatMessage, LLMBackend, LLMError

DEFAULT_MODEL = "claude-opus-5-5"


class AnthropicBackend(LLMBackend):
    name = "anthropic"
    supports_offline = False

    def __init__(self, model: str | None = None, *, effort: str = "high", max_tokens: int = 16000,
                 timeout: float = 600.0) -> None:
        self.model = model or DEFAULT_MODEL
        self.effort = effort
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = None

    def _client_or_raise(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise LLMError("The 'anthropic' package is not installed: pip install anthropic") from exc
            self._client = anthropic.Anthropic(timeout=self.timeout)
        return self._client

    def available(self) -> bool:
        if any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")):
            try:
                import anthropic  # noqa: F401
                return True
            except ImportError:
                return False
        return False

    def chat(self, system: str, messages: Sequence[ChatMessage], *, temperature=None, max_tokens=None) -> str:
        import anthropic

        client = self._client_or_raise()
        try:
            # Thinking is adaptive by default on current models; depth is set with output_config.effort.
            # Server-side refusal fallbacks are enabled so a policy decline is retried on another model
            # inside the same request instead of ending the research run.
            with client.beta.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[m.as_dict() for m in messages],
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                response = stream.get_final_message()
        except anthropic.RateLimitError as exc:
            raise LLMError(f"Rate limited by the Claude API: {exc.message}") from exc
        except anthropic.AuthenticationError as exc:
            raise LLMError("Claude API authentication failed - check ANTHROPIC_API_KEY") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Cannot reach the Claude API (offline?). Use an offline backend.") from exc

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMError(f"The model declined this request (category: {category}).")
        return "".join(block.text for block in response.content if block.type == "text")
