"""Generic tool-using agent (a ReAct-style loop over the JSON protocol)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

from ..llm import ChatMessage, LLMBackend, LLMError
from ..tools import ToolRegistry
from .protocol import parse_action

EventHook = Callable[[str, dict], None]


@dataclass
class AgentResult:
    agent: str
    task: str
    content: str
    steps: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    truncated: bool = False
    # filled by ThinkingAgent (metacognition)
    thinking: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)
    confidence: float | None = None
    attempts: int = 1
    attempt_history: list[dict] = field(default_factory=list)   # earlier attempts that failed verification

    def summary(self, limit: int = 4000) -> str:
        text = self.content
        return text if len(text) <= limit else text[:limit] + "\n... [truncated]"


class Agent:
    def __init__(self, name: str, system_prompt: str, backend: LLMBackend, tools: ToolRegistry | None = None,
                 *, max_steps: int = 8, max_tool_output: int = 6000, on_event: EventHook | None = None,
                 temperature: float | None = None) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.backend = backend
        self.tools = tools or ToolRegistry()
        self.max_steps = max_steps
        self.max_tool_output = max_tool_output
        self.on_event = on_event or (lambda kind, data: None)
        self.temperature = temperature

    # ------------------------------------------------------------------
    def _emit(self, kind: str, **data) -> None:
        try:
            self.on_event(kind, {"agent": self.name, **data})
        except Exception:  # noqa: BLE001 - a UI hook must never break a run
            pass

    def _chat(self, messages: list[ChatMessage]) -> str:
        return self.backend.chat(self.system_prompt, messages, temperature=self.temperature)

    # ------------------------------------------------------------------
    def run(self, task: str, context: str = "") -> AgentResult:
        started = time.time()
        prompt = task if not context else f"{task}\n\n## Context\n{context}"
        messages = [ChatMessage("user", prompt)]
        result = AgentResult(agent=self.name, task=task, content="")
        seen_calls: dict[str, int] = {}
        self._emit("agent_start", task=task)

        for step in range(1, self.max_steps + 1):
            result.steps = step
            try:
                reply = self._chat(messages)
            except LLMError as exc:
                result.content = f"(model error: {exc})"
                result.truncated = True
                break
            action = parse_action(reply)
            if action.kind == "final":
                result.content = action.content
                self._emit("agent_final", steps=step)
                break

            messages.append(ChatMessage("assistant", reply))
            key = action.tool + json.dumps(action.input, sort_keys=True, default=str)
            seen_calls[key] = seen_calls.get(key, 0) + 1
            if seen_calls[key] > 2:
                tool_output, is_error = ("You already made this exact call. Use a different query/tool or give your final answer.", True)
                sources: list[str] = []
            else:
                self._emit("tool_call", tool=action.tool, input=action.input, thought=action.thought)
                res = self.tools.run(action.tool, **action.input) if action.tool else None
                if res is None:
                    tool_output, is_error, sources = ("No tool named. Use one of: " + ", ".join(self.tools.names()), True, [])
                else:
                    tool_output, is_error, sources = res.output, res.error, res.sources
            if len(tool_output) > self.max_tool_output:
                tool_output = tool_output[: self.max_tool_output] + "\n... [tool output truncated]"
            result.tool_calls.append({"tool": action.tool, "input": action.input, "error": is_error,
                                      "output_preview": tool_output[:300]})
            for s in sources:
                if s not in result.sources:
                    result.sources.append(s)
            self._emit("tool_result", tool=action.tool, error=is_error, preview=tool_output[:200])
            remaining = self.max_steps - step
            label = "TOOL ERROR" if is_error else "TOOL RESULT"
            messages.append(ChatMessage(
                "user",
                f"{label} ({action.tool}):\n{tool_output}\n\n"
                f"({remaining} step(s) left before you must answer.) Continue with the next JSON action."))
        else:
            # Out of steps: force a final answer without tools.
            result.truncated = True
            messages.append(ChatMessage("user", "You are out of tool steps. Give your final answer now as "
                                                '{"action": "final", "content": "..."} using what you have.'))
            try:
                reply = self._chat(messages)
                result.content = parse_action(reply).content
            except LLMError as exc:
                result.content = f"(model error: {exc})"
            self._emit("agent_final", steps=self.max_steps, forced=True)

        result.elapsed = time.time() - started
        return result
