"""ProblemSolver: general problem solving beyond research reports.

    classify (type, strategy) -> ThinkingAgent with every tool (plus `research`, which runs
    the multi-agent orchestrator for substantial evidence gathering) -> verified solution
    -> reflection & learning.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..agents.orchestrator import ResearchOrchestrator
from ..agents.prompts import base_system
from ..agents.protocol import parse_json_object
from ..config import Config
from ..llm import ChatMessage, LLMError
from ..tools.base import Tool, ToolResult
from .experience import Reflector, trace_for_reflection
from .prompts import META_SYSTEM, SOLVER_CLASSIFY_PROMPT, SOLVER_ROLE
from .thinking import ThinkingAgent

PROBLEM_TYPES = ("research", "calculation", "design", "decision", "troubleshooting", "planning", "creative",
                 "logic", "howto", "coding")
STRATEGIES = ("decomposition", "first_principles", "analogy", "working_backwards", "constraint_analysis",
              "simplify", "trial_and_verification", "inversion", "simulation")


@dataclass
class Solution:
    session_id: str
    problem: str
    restated: str
    ptype: str
    strategy: str
    answer: str
    confidence: float | None
    attempts: int
    thinking: dict
    verification: dict
    tool_calls: list[dict]
    sources: list[str]
    elapsed: float
    learned: dict = field(default_factory=dict)
    output_path: Path | None = None
    attempt_history: list[dict] = field(default_factory=list)
    backend: str = ""


class ProblemSolver:
    def __init__(self, config: Config | None = None, orchestrator: ResearchOrchestrator | None = None, *,
                 on_event=None) -> None:
        self.orch = orchestrator or ResearchOrchestrator(config, on_event=on_event)
        self.config = self.orch.config
        self.backend = self.orch.backend
        self.on_event = on_event or (lambda kind, data: None)
        self.preset = self.orch.preset
        self.tools = self.orch.tools
        self._add_research_tool()

    def _emit(self, kind: str, **data) -> None:
        try:
            self.on_event(kind, data)
        except Exception:  # noqa: BLE001
            pass

    def _add_research_tool(self) -> None:
        if self.tools.get("research") is not None:
            return
        orch = self.orch

        def research(question: str) -> ToolResult:
            saved_depth = orch.config.depth
            saved_preset = orch.preset
            saved_reflect = orch.config.reflect
            orch.config.depth, orch.config.reflect = "quick", False
            orch.preset = orch.config.preset()
            try:
                report = orch.research(str(question), save=False)
            finally:
                orch.config.depth, orch.config.reflect, orch.preset = saved_depth, saved_reflect, saved_preset
            return ToolResult(report.report[:12000], sources=report.sources)

        self.tools.add(Tool("research", "Run a full multi-agent research pass on a question that needs substantial "
                                        "evidence gathering across sources. Slow; use for genuine knowledge gaps.",
                            {"question": "string"}, research, required=("question",)))

    # ------------------------------------------------------------------
    def classify(self, problem: str) -> dict:
        self._emit("phase", name="understanding the problem")
        experience = self.orch.experience.recall_text(problem, limit=5)
        try:
            reply = self.backend.chat(META_SYSTEM, [ChatMessage("user", SOLVER_CLASSIFY_PROMPT.format(
                problem=problem, experience=experience))], temperature=0.1)
            data = parse_json_object(reply) or {}
        except LLMError:
            data = {}
        ptype = str(data.get("type", "")).lower().strip()
        strategy = str(data.get("strategy", "")).lower().strip().replace(" ", "_")
        return {
            "restated": str(data.get("restated") or problem),
            "type": ptype if ptype in PROBLEM_TYPES else "research",
            "difficulty": str(data.get("difficulty", "moderate")),
            "strategy": strategy if strategy in STRATEGIES else "decomposition",
            "why": str(data.get("why", "")),
            "needs_research": bool(data.get("needs_research", False)),
            "deliverable": str(data.get("deliverable") or "a correct, complete, verified solution"),
        }

    def solve(self, problem: str, *, save: bool = True, fresh: bool = False) -> Solution:
        started = time.time()
        problem = problem.strip()
        self._emit("start", question=problem, backend=self.backend.describe(), online=self.orch.online,
                   depth=self.config.depth)
        if self.config.instant_recall and not fresh:
            hit = self.orch.learner.recall(problem, min_similarity=self.config.recall_similarity)
            if hit:
                when = time.strftime("%Y-%m-%d %H:%M", time.localtime(hit["created"]))
                self._emit("instant_recall", similarity=hit["similarity"], learned_at=when)
                answer = (f"{hit['answer']}\n\n_(answered instantly from what I learned on {when}; similarity "
                          f"{hit['similarity']:.2f}. Use --fresh to solve it again.)_")
                return Solution(self.orch.session_id, problem, hit["question"], "recalled", "instant_recall", answer,
                                hit["confidence"], 0, {}, {}, [], hit["sources"], time.time() - started,
                                backend=self.backend.describe())
        klass = self.classify(problem)
        self._emit("classified", **klass)

        role = SOLVER_ROLE.format(ptype=klass["type"], strategy=klass["strategy"], why=klass["why"],
                                  deliverable=klass["deliverable"])
        tool_names = [n for n in self.tools.names() if n not in {"list_dir"}]
        tools = self.tools.subset(tool_names)
        system = base_system("the problem solver", role, tools.spec_text(), language=self.config.language)
        agent = ThinkingAgent("solver", system, self.backend, tools,
                              max_steps=max(6, self.preset["max_steps"] + 2), on_event=self.on_event,
                              deliberate=bool(self.preset.get("deliberate", 1)), verify=bool(self.preset.get("verify", 1)),
                              max_attempts=int(self.preset.get("max_attempts", 2)), experience=self.orch.experience,
                              min_confidence=self.config.min_confidence)
        hint = ""
        if klass["needs_research"] and self.preset["max_steps"] >= 6:
            hint = "This problem needs evidence: consider the `research` tool for the knowledge-heavy part."
        self._emit("phase", name=f"solving ({klass['type']}, {klass['strategy']})")
        result = agent.run(f"Problem: {klass['restated']}\n(Original wording: {problem})", hint)

        solution = Solution(self.orch.session_id, problem, klass["restated"], klass["type"], klass["strategy"],
                            result.content, result.confidence, result.attempts, result.thinking, result.verification,
                            result.tool_calls, result.sources, time.time() - started,
                            attempt_history=result.attempt_history, backend=self.backend.describe())
        if self.config.reflect:
            self._emit("phase", name="reflecting & learning")
            reflector = Reflector(self.backend, self.orch.experience)
            solution.learned = reflector.reflect(problem, trace_for_reflection([result]), result.content,
                                                 session=self.orch.session_id)
            self._emit("learned", **{k: v for k, v in solution.learned.items() if k != "error"})
        trustworthy = result.confidence is None or result.confidence >= self.config.min_confidence
        if trustworthy and result.content and not result.content.startswith("(model error"):
            self.orch.learner.remember(problem, result.content, confidence=result.confidence, sources=result.sources,
                                       session=self.orch.session_id)
        if save:
            solution.output_path = self._save(solution)
        self._emit("done", elapsed=solution.elapsed, path=str(solution.output_path or ""))
        return solution

    def _save(self, sol: Solution) -> Path:
        out = self.config.output_dir
        out.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", sol.problem.lower()).strip("-")[:60] or "problem"
        base = f"{time.strftime('%Y%m%d-%H%M%S')}-solve-{slug}"
        md = out / f"{base}.md"
        conf = f"{sol.confidence:.2f}" if sol.confidence is not None else "n/a"
        header = (f"<!-- OmniResearch solve {sol.session_id} | type: {sol.ptype} | strategy: {sol.strategy} | "
                  f"confidence: {conf} | attempts: {sol.attempts} | {sol.elapsed:.0f}s -->\n\n"
                  f"# {sol.restated}\n\n")
        md.write_text(header + sol.answer.strip() + "\n", encoding="utf-8")
        (out / f"{base}.trace.json").write_text(json.dumps({
            "session_id": sol.session_id, "problem": sol.problem, "restated": sol.restated, "type": sol.ptype,
            "strategy": sol.strategy, "confidence": sol.confidence, "attempts": sol.attempts,
            "thinking": sol.thinking, "verification": sol.verification, "tool_calls": sol.tool_calls,
            "sources": sol.sources, "learned": sol.learned, "elapsed": round(sol.elapsed, 1),
            "attempt_history": sol.attempt_history, "backend": sol.backend, "kind": "solve",
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        return md
