"""The model <-> agent action protocol.

Agents ask the model to answer with a single JSON object:

    {"thought": "...", "action": "tool", "tool": "<name>", "input": {...}}
    {"thought": "...", "action": "final", "content": "..."}

Small local models are sloppy, so parsing is deliberately forgiving: it finds
the first balanced JSON object in the reply, tolerates code fences, and treats
a reply with no JSON at all as a final answer.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


@dataclass
class Action:
    kind: str                       # "tool" | "final"
    tool: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    content: str = ""
    thought: str = ""
    raw: str = ""


def _balanced_objects(text: str):
    """Yield every top-level {...} substring, respecting strings and escapes."""
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start:i + 1]
                    start = -1


def extract_json(text: str) -> dict[str, Any] | None:
    """Return the first JSON object that has an 'action' key (or any object as fallback)."""
    candidates = []
    for fenced in _FENCE.findall(text):
        candidates.append(fenced)
    candidates.append(text)
    fallback = None
    for blob in candidates:
        for obj in _balanced_objects(blob):
            try:
                data = json.loads(obj)
            except json.JSONDecodeError:
                # common small-model slips: trailing commas, single quotes
                try:
                    data = json.loads(re.sub(r",\s*([}\]])", r"\1", obj))
                except json.JSONDecodeError:
                    continue
            if not isinstance(data, dict):
                continue
            if "action" in data or "tool" in data or "content" in data:
                return data
            fallback = fallback or data
    return fallback


def parse_action(text: str) -> Action:
    data = extract_json(text)
    if not data:
        return Action(kind="final", content=text.strip(), raw=text)
    thought = str(data.get("thought", "") or data.get("reasoning", ""))
    kind = str(data.get("action", "")).lower().strip()
    tool = data.get("tool") or data.get("tool_name") or data.get("name")
    if kind in {"tool", "call", "use_tool", "tool_call"} or (kind == "" and tool):
        tool_input = data.get("input") or data.get("arguments") or data.get("args") or data.get("parameters") or {}
        if not isinstance(tool_input, dict):
            tool_input = {"query": str(tool_input)}
        return Action(kind="tool", tool=str(tool or ""), input=tool_input, thought=thought, raw=text)
    content = data.get("content") or data.get("answer") or data.get("final") or data.get("response") or ""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, indent=1)
    if not content.strip():
        # the model sent an empty final; keep whatever text surrounded the JSON
        content = text.strip()
    return Action(kind="final", content=content, thought=thought, raw=text)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a plan/critique style reply where the whole answer is one JSON object."""
    data = extract_json(text)
    return data if isinstance(data, dict) else None
