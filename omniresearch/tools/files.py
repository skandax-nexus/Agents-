"""Workspace file tools: read local files and list directories (offline)."""
from __future__ import annotations

from pathlib import Path

from .base import Tool, ToolError, ToolResult
from .textutil import extract_text


def _resolve(root: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def file_tools(root: Path) -> list[Tool]:
    root = Path(root).resolve()

    def read_file(path: str, max_chars: int = 8000) -> ToolResult:
        p = _resolve(root, path)
        if not p.exists() or not p.is_file():
            raise ToolError(f"File not found: {path}")
        try:
            limit = max(500, min(int(max_chars), 40000))
        except (TypeError, ValueError):
            limit = 8000
        text = extract_text(p)
        if not text:
            raise ToolError(f"Could not extract text from {path} (unsupported type or empty).")
        suffix = "" if len(text) <= limit else f"\n... [truncated, {len(text)} chars total]"
        return ToolResult(text[:limit] + suffix, sources=[str(p)])

    def list_dir(path: str = ".") -> ToolResult:
        p = _resolve(root, path or ".")
        if not p.exists() or not p.is_dir():
            raise ToolError(f"Directory not found: {path}")
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))[:300]
        lines = [f"{'[dir] ' if e.is_dir() else ''}{e.name}" for e in entries]
        return ToolResult("\n".join(lines) or "(empty)")

    return [
        Tool("read_file", "Read a local file (text, markdown, csv, json, html, pdf) from disk.",
             {"path": "string - file path", "max_chars": "int, optional"}, read_file, required=("path",)),
        Tool("list_dir", "List the contents of a local directory.",
             {"path": "string - directory path (default '.')"}, list_dir),
    ]
