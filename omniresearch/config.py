"""Configuration for OmniResearch.

Resolution order (highest priority first):
  1. explicit keyword arguments / CLI flags
  2. environment variables (OMNIRESEARCH_*)
  3. config file  (./config.toml, then ~/.omniresearch/config.toml)
  4. built-in defaults

Everything works with zero configuration when a local Ollama server is running.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

DEFAULT_HOME = Path(os.environ.get("OMNIRESEARCH_HOME", Path.home() / ".omniresearch"))

DEPTH_PRESETS: dict[str, dict[str, int]] = {
    # quick: one pass, few tool steps.  deep: more sub-questions, more steps, a critique round.
    "quick": {"max_subquestions": 3, "max_steps": 4, "critique_rounds": 0, "max_domains": 2},
    "standard": {"max_subquestions": 5, "max_steps": 8, "critique_rounds": 1, "max_domains": 4},
    "deep": {"max_subquestions": 8, "max_steps": 14, "critique_rounds": 2, "max_domains": 6},
}


@dataclass
class Config:
    # --- model backend -------------------------------------------------
    backend: str = "auto"              # auto | ollama | openai | anthropic | mock
    model: str | None = None           # backend-specific model name
    ollama_url: str = "http://localhost:11434"
    openai_base_url: str = "http://localhost:8080/v1"   # llama.cpp server, LM Studio, vLLM ...
    openai_api_key: str = "not-needed"
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: str = "high"     # low | medium | high | xhigh | max
    temperature: float = 0.2
    max_tokens: int = 4096
    num_ctx: int = 16384               # context window requested from local backends
    request_timeout: float = 600.0

    # --- research behaviour ---------------------------------------------
    depth: str = "standard"            # quick | standard | deep
    online: bool | None = None         # None = auto-detect connectivity; False = strictly offline
    parallel_agents: int = 2
    language: str = "English"

    # --- storage ----------------------------------------------------------
    home: Path = field(default_factory=lambda: DEFAULT_HOME)
    output_dir: Path = field(default_factory=lambda: Path("research_output"))
    corpus_dirs: list[str] = field(default_factory=list)  # directories to index for offline knowledge
    kiwix_url: str | None = None       # optional offline Wikipedia (kiwix-serve), e.g. http://localhost:8081

    # --- safety of local execution ----------------------------------------
    python_timeout: int = 60
    allow_python: bool = True

    def preset(self) -> dict[str, int]:
        return DEPTH_PRESETS.get(self.depth, DEPTH_PRESETS["standard"])

    @property
    def knowledge_db(self) -> Path:
        return self.home / "knowledge.db"

    @property
    def memory_db(self) -> Path:
        return self.home / "memory.db"

    def ensure_dirs(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["home"] = str(self.home)
        d["output_dir"] = str(self.output_dir)
        return d


_ENV_MAP = {
    "OMNIRESEARCH_BACKEND": ("backend", str),
    "OMNIRESEARCH_MODEL": ("model", str),
    "OMNIRESEARCH_OLLAMA_URL": ("ollama_url", str),
    "OMNIRESEARCH_OPENAI_BASE_URL": ("openai_base_url", str),
    "OMNIRESEARCH_OPENAI_API_KEY": ("openai_api_key", str),
    "OMNIRESEARCH_ANTHROPIC_MODEL": ("anthropic_model", str),
    "OMNIRESEARCH_ANTHROPIC_EFFORT": ("anthropic_effort", str),
    "OMNIRESEARCH_DEPTH": ("depth", str),
    "OMNIRESEARCH_ONLINE": ("online", "bool"),
    "OMNIRESEARCH_OUTPUT_DIR": ("output_dir", Path),
    "OMNIRESEARCH_KIWIX_URL": ("kiwix_url", str),
    "OMNIRESEARCH_TEMPERATURE": ("temperature", float),
    "OMNIRESEARCH_MAX_TOKENS": ("max_tokens", int),
    "OMNIRESEARCH_NUM_CTX": ("num_ctx", int),
    "OMNIRESEARCH_PARALLEL": ("parallel_agents", int),
}


def _coerce(value: str, kind: Any) -> Any:
    if kind == "bool":
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return kind(value)


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        return {}


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Build a Config from file + environment + explicit overrides."""
    data: dict[str, Any] = {}
    candidates = [Path(path)] if path else [Path("config.toml"), DEFAULT_HOME / "config.toml"]
    for candidate in candidates:
        file_data = _read_toml(candidate)
        if file_data:
            data.update(file_data)
            break

    for env_name, (key, kind) in _ENV_MAP.items():
        raw = os.environ.get(env_name)
        if raw is not None and raw != "":
            data[key] = _coerce(raw, kind)

    for key, value in overrides.items():
        if value is not None:
            data[key] = value

    if "home" in data:
        data["home"] = Path(data["home"]).expanduser()
    if "output_dir" in data:
        data["output_dir"] = Path(data["output_dir"]).expanduser()

    valid = {f for f in Config.__dataclass_fields__}
    unknown = set(data) - valid
    for key in unknown:
        data.pop(key)
    return Config(**data)
