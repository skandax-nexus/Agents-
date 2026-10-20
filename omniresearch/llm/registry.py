"""Backend construction and auto-detection."""
from __future__ import annotations

from ..config import Config
from .anthropic_backend import AnthropicBackend
from .base import LLMBackend, LLMError
from .mock import EchoBackend
from .ollama import OllamaBackend
from .openai_compat import OpenAICompatBackend


def _student_model(cfg: Config) -> str | None:
    """Name of the self-trained student model, if one was adopted (see omniresearch train)."""
    try:
        import json
        record = json.loads((cfg.home / "training" / "student.json").read_text())
        return str(record.get("ollama_model") or "") or None
    except (OSError, ValueError):
        return None


def _ollama(cfg: Config) -> OllamaBackend:
    backend = OllamaBackend(cfg.ollama_url, cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                            num_ctx=cfg.num_ctx, timeout=cfg.request_timeout)
    if cfg.model is None:
        student = _student_model(cfg)
        if student and any(m.split(":")[0] == student.split(":")[0] for m in backend.installed_models()):
            backend.model = student
    return backend


def _openai(cfg: Config) -> OpenAICompatBackend:
    return OpenAICompatBackend(cfg.openai_base_url, cfg.model, cfg.openai_api_key, temperature=cfg.temperature,
                               max_tokens=cfg.max_tokens, timeout=cfg.request_timeout)


def _anthropic(cfg: Config) -> AnthropicBackend:
    return AnthropicBackend(cfg.model or cfg.anthropic_model, effort=cfg.anthropic_effort,
                            max_tokens=max(cfg.max_tokens, 16000), timeout=cfg.request_timeout)


def probe_backends(cfg: Config) -> list[tuple[str, LLMBackend, bool]]:
    """Return (name, backend, available) for every known backend - used by `doctor`."""
    out = []
    for name, factory in (("ollama", _ollama), ("openai", _openai), ("anthropic", _anthropic)):
        backend = factory(cfg)
        out.append((name, backend, backend.available()))
    out.append(("mock", EchoBackend(), True))
    return out


def build_backend(cfg: Config) -> LLMBackend:
    """Construct the backend named in the config, or auto-detect one.

    Auto order prefers offline engines: Ollama, then any OpenAI-compatible local server,
    then the Claude API if credentials are present.
    """
    choice = (cfg.backend or "auto").lower()
    if choice == "ollama":
        backend = _ollama(cfg)
        if cfg.model is None and backend.model == backend.pick_model() or cfg.model is None and not backend.installed_models():
            backend.model = backend.pick_model()
        return backend
    if choice in {"openai", "llamacpp", "llama.cpp", "lmstudio", "vllm"}:
        backend = _openai(cfg)
        if cfg.model is None:
            backend.model = backend.pick_model()
        return backend
    if choice in {"anthropic", "claude"}:
        return _anthropic(cfg)
    if choice in {"mock", "echo"}:
        return EchoBackend()
    if choice != "auto":
        raise LLMError(f"Unknown backend '{cfg.backend}'. Choose ollama, openai, anthropic, or mock.")

    ollama = _ollama(cfg)
    if ollama.available():
        if cfg.model is None:
            ollama.model = ollama.pick_model()
        return ollama
    local = _openai(cfg)
    if local.available():
        if cfg.model is None:
            local.model = local.pick_model()
        return local
    claude = _anthropic(cfg)
    if claude.available():
        return claude
    raise LLMError(
        "No model backend found. Start Ollama (`ollama serve && ollama pull llama3.1:8b`), "
        "run a llama.cpp/LM Studio server on http://localhost:8080, or set ANTHROPIC_API_KEY. "
        "Run `omniresearch doctor` for details."
    )
