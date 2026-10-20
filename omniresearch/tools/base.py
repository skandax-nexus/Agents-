from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


class ToolError(RuntimeError):
    pass


@dataclass
class ToolResult:
    output: str
    sources: list[str] = field(default_factory=list)   # citations surfaced by the tool
    error: bool = False


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, str]           # param name -> human description (incl. type)
    func: Callable[..., ToolResult]
    offline: bool = True                 # usable without network
    required: tuple[str, ...] = ()

    def spec(self) -> str:
        params = ", ".join(f'"{k}": <{v}>' for k, v in self.parameters.items())
        return f'- {self.name}: {self.description}\n  input: {{{params}}}'

    def run(self, **kwargs: Any) -> ToolResult:
        missing = [p for p in self.required if p not in kwargs or kwargs[p] in (None, "")]
        if missing:
            return ToolResult(f"Missing required input(s): {', '.join(missing)}", error=True)
        allowed = {k: v for k, v in kwargs.items() if k in self.parameters}
        try:
            return self.func(**allowed)
        except ToolError as exc:
            return ToolResult(str(exc), error=True)
        except Exception as exc:  # noqa: BLE001 - a tool crash must not kill the research run
            return ToolResult(f"{type(exc).__name__}: {exc}", error=True)
