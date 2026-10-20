from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..config import Config
from .base import Tool, ToolResult
from .files import file_tools
from .knowledge import KnowledgeBase, knowledge_tools
from .memory import Memory, memory_tools
from .python_exec import python_tool
from .web import has_connectivity, web_tools


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools:
            self.add(t)

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def subset(self, names: Iterable[str]) -> "ToolRegistry":
        return ToolRegistry(self._tools[n] for n in names if n in self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def spec_text(self) -> str:
        return "\n".join(t.spec() for t in self._tools.values())

    def run(self, tool_name: str, /, **kwargs) -> ToolResult:
        tool = self.get(tool_name)
        if tool is None:
            return ToolResult(f"Unknown tool '{tool_name}'. Available: {', '.join(self.names())}", error=True)
        return tool.run(**kwargs)


def build_tools(cfg: Config, session_id: str, *, kb: KnowledgeBase | None = None,
                memory: Memory | None = None, experience=None) -> tuple[ToolRegistry, bool]:
    """Assemble the full tool set for a run. Returns (registry, online_flag)."""
    from ..cognition.experience import Experience, experience_tools  # local import avoids an import cycle

    cfg.ensure_dirs()
    kb = kb or KnowledgeBase(cfg.knowledge_db)
    memory = memory or Memory(cfg.memory_db)
    experience = experience or Experience(cfg.experience_db)
    online = cfg.online if cfg.online is not None else has_connectivity()

    tools: list[Tool] = []
    tools += knowledge_tools(kb)
    tools += file_tools(Path.cwd())
    if cfg.allow_python:
        tools.append(python_tool(cfg.home / "sandbox", timeout=cfg.python_timeout))
    tools += memory_tools(memory, session_id)
    tools += experience_tools(experience, session_id)
    tools += web_tools(kb, kiwix_url=cfg.kiwix_url, online=online)
    return ToolRegistry(tools), online
