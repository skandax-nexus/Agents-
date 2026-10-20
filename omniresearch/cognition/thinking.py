"""ThinkingAgent: an Agent that deliberates before acting, verifies afterwards, and retries.

    recall experience -> deliberate (understand, hypothesise, choose strategy, set success criteria)
    -> act (tool loop) -> verify against the criteria -> if it fails and attempts remain, act again
    with the verifier's fix strategy.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..agents.base import Agent, AgentResult
from ..agents.protocol import parse_json_object
from ..llm import ChatMessage, LLMError
from .prompts import DELIBERATE_PROMPT, META_SYSTEM, VERIFY_PROMPT

if TYPE_CHECKING:
    from .experience import Experience


@dataclass
class Thought:
    understanding: str = ""
    knowns: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    strategies: list[str] = field(default_factory=list)
    chosen_strategy: str = ""
    first_steps: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    difficulty: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Thought":
        def lst(key):
            v = d.get(key) or []
            return [str(x) for x in v] if isinstance(v, list) else [str(v)]
        return cls(understanding=str(d.get("understanding", "")), knowns=lst("knowns"), unknowns=lst("unknowns"),
                   assumptions=lst("assumptions"), hypotheses=lst("hypotheses"), strategies=lst("strategies"),
                   chosen_strategy=str(d.get("chosen_strategy", "")), first_steps=lst("first_steps"),
                   success_criteria=lst("success_criteria"), risks=lst("risks"), difficulty=str(d.get("difficulty", "")))

    def as_dict(self) -> dict:
        return self.__dict__.copy()

    def briefing(self) -> str:
        """What the acting agent is told about its own prior thinking."""
        lines = []
        if self.understanding:
            lines.append(f"Understanding: {self.understanding}")
        if self.assumptions:
            lines.append("Assumptions: " + "; ".join(self.assumptions))
        if self.hypotheses:
            lines.append("Hypotheses to test: " + "; ".join(self.hypotheses))
        if self.chosen_strategy:
            lines.append(f"Strategy: {self.chosen_strategy}")
        if self.first_steps:
            lines.append("First steps: " + " -> ".join(self.first_steps))
        if self.success_criteria:
            lines.append("Success criteria: " + "; ".join(self.success_criteria))
        if self.risks:
            lines.append("Watch out for: " + "; ".join(self.risks))
        return "\n".join(lines)


@dataclass
class Verification:
    passes: bool = True
    confidence: float = 0.5
    problems: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    fix_strategy: str = ""
    inconclusive: bool = False   # the verifier produced nothing usable; do not retry on it

    @classmethod
    def from_dict(cls, d: dict) -> "Verification":
        try:
            conf = max(0.0, min(1.0, float(d.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        passes = d.get("passes", True)
        if isinstance(passes, str):
            passes = passes.strip().lower() in {"true", "yes", "pass", "passes"}
        return cls(passes=bool(passes), confidence=conf,
                   problems=[str(x) for x in (d.get("problems") or [])],
                   unsupported_claims=[str(x) for x in (d.get("unsupported_claims") or [])],
                   fix_strategy=str(d.get("fix_strategy", "")))

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class ThinkingAgent(Agent):
    def __init__(self, *args, deliberate: bool = True, verify: bool = True, max_attempts: int = 2,
                 experience: "Experience | None" = None, domains: tuple[str, ...] = (),
                 min_confidence: float = 0.6, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.deliberate_enabled = deliberate
        self.verify_enabled = verify
        self.max_attempts = max(1, max_attempts)
        self.experience = experience
        self.domains = domains
        self.min_confidence = min_confidence

    # -- metacognitive calls (separate system prompt, no tools) ----------------
    def _meta(self, prompt: str) -> dict:
        try:
            reply = self.backend.chat(META_SYSTEM, [ChatMessage("user", prompt)], temperature=0.1)
        except LLMError as exc:
            return {"error": str(exc)}
        return parse_json_object(reply) or {}

    def recall(self, task: str) -> str:
        if self.experience is None:
            return "(no experience store)"
        return self.experience.recall_text(task, domains=self.domains)

    def deliberate(self, task: str, context: str, experience: str) -> Thought:
        data = self._meta(DELIBERATE_PROMPT.format(task=task, context=context or "(none)", experience=experience))
        thought = Thought.from_dict(data)
        self._emit("deliberation", strategy=thought.chosen_strategy, difficulty=thought.difficulty,
                   hypotheses=len(thought.hypotheses))
        return thought

    def verify(self, task: str, result: AgentResult, thought: Thought) -> Verification:
        criteria = "\n".join(f"- {c}" for c in thought.success_criteria) or "- answers the task completely and correctly"
        evidence = "\n".join(
            f"- {c['tool']}({json.dumps(c['input'], ensure_ascii=False)[:120]}) -> "
            f"{'ERROR ' if c.get('error') else ''}{c.get('output_preview', '')[:160]!r}"
            for c in result.tool_calls) or "(no tools were used)"
        data = self._meta(VERIFY_PROMPT.format(task=task, criteria=criteria, answer=result.content[:7000],
                                               evidence=evidence[:4000]))
        if data and "error" not in data and "passes" in data:
            ver = Verification.from_dict(data)
        else:
            ver = Verification(inconclusive=True)
        self._emit("verification", passes=ver.passes, confidence=ver.confidence, problems=len(ver.problems),
                   inconclusive=ver.inconclusive)
        return ver

    # -- the cognitive cycle ----------------------------------------------------
    def run(self, task: str, context: str = "") -> AgentResult:
        started = time.time()
        experience = self.recall(task)
        thought = Thought()
        if self.deliberate_enabled:
            thought = self.deliberate(task, context, experience)

        full_context = context
        if experience and not experience.startswith("("):
            full_context += f"\n\n## Relevant experience from past work\n{experience}"
        briefing = thought.briefing()
        if briefing:
            full_context += f"\n\n## Your own prior thinking about this task\n{briefing}"

        result = AgentResult(agent=self.name, task=task, content="")
        verification = Verification()
        for attempt in range(1, self.max_attempts + 1):
            result = super().run(task, full_context.strip())
            result.attempts = attempt
            if not self.verify_enabled:
                break
            verification = self.verify(task, result, thought)
            ok = verification.inconclusive or (verification.passes and verification.confidence >= self.min_confidence)
            if ok or attempt == self.max_attempts:
                break
            problems = "; ".join(verification.problems) or "the answer did not meet the success criteria"
            self._emit("retry", attempt=attempt, reason=problems[:200])
            full_context += (f"\n\n## Previous attempt (#{attempt}) failed verification\nProblems: {problems}\n"
                             f"Do differently: {verification.fix_strategy or 'address every problem above'}\n"
                             f"Previous answer (do not repeat its mistakes):\n{result.content[:3000]}")

        result.thinking = thought.as_dict()
        if self.verify_enabled:
            result.verification = verification.as_dict()
            result.confidence = None if verification.inconclusive else verification.confidence
            if verification.unsupported_claims or (not verification.passes and verification.problems):
                note = "\n\n> Verifier notes: " + "; ".join((verification.problems + verification.unsupported_claims)[:5])
                result.content = result.content.rstrip() + note
        result.elapsed = time.time() - started
        return result
