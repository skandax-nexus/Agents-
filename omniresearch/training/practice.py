"""Self-practice: the system sets itself problems, solves them with verification, and keeps
only the verified solutions as training data (STaR-style self-training).

    pick domains -> generate N practice problems per domain (varied, concrete, checkable)
    -> ProblemSolver.solve() each (deliberate / act / verify / retry) -> traces saved
    -> DatasetBuilder keeps the ones that passed verification above the confidence bar
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..agents.protocol import parse_json_object
from ..cognition.solver import ProblemSolver
from ..domains import DOMAINS, get_domain
from ..llm import ChatMessage, LLMError

GENERATE_SYSTEM = """You design practice problems for a research and problem-solving assistant so it can train itself.
Reply with exactly ONE JSON object and nothing else."""

GENERATE_PROMPT = """Domain: {name} ({category})
Scope: {description}
Methods used in this field: {methods}
Difficulty: {difficulty}
Already used (do not repeat): {seen}

Write {n} varied practice problems in this domain. Each must be concrete, self-contained, and CHECKABLE:
a definite answer or a design whose correctness can be verified (prefer problems with numbers, units, mechanisms,
or clear criteria). Mix types: calculation, explanation of mechanism, comparison, design, troubleshooting, data
interpretation. Avoid anything requiring live data.
{{"problems": ["<problem 1>", "<problem 2>", ...]}}"""


@dataclass
class PracticeResult:
    domain: str
    problem: str
    confidence: float | None
    attempts: int
    passed: bool
    trace_path: str = ""
    elapsed: float = 0.0


@dataclass
class PracticeSummary:
    results: list[PracticeResult] = field(default_factory=list)

    @property
    def kept(self) -> int:
        return sum(1 for r in self.results if r.passed)

    def as_dict(self) -> dict:
        return {"problems": len(self.results), "kept": self.kept,
                "by_domain": {d: sum(1 for r in self.results if r.domain == d and r.passed) for d in
                              sorted({r.domain for r in self.results})}}


class PracticeLoop:
    def __init__(self, solver: ProblemSolver, *, min_confidence: float = 0.7, on_event=None) -> None:
        self.solver = solver
        self.backend = solver.backend
        self.min_confidence = min_confidence
        self.on_event = on_event or (lambda kind, data: None)
        self.seen_path = solver.config.home / "training" / "practice_seen.json"
        self.seen: list[str] = []
        if self.seen_path.exists():
            try:
                self.seen = json.loads(self.seen_path.read_text())
            except (OSError, ValueError):
                self.seen = []

    def _emit(self, kind: str, **data) -> None:
        try:
            self.on_event(kind, data)
        except Exception:  # noqa: BLE001
            pass

    def generate(self, domain_id: str, n: int, difficulty: str = "moderate") -> list[str]:
        d = get_domain(domain_id)
        recent = [s for s in self.seen if s.startswith(f"{d.id}:")][-15:]
        prompt = GENERATE_PROMPT.format(name=d.name, category=d.category, description=d.description,
                                        methods=d.methods, difficulty=difficulty, n=n,
                                        seen="; ".join(s.split(":", 1)[1][:80] for s in recent) or "(none)")
        try:
            reply = self.backend.chat(GENERATE_SYSTEM, [ChatMessage("user", prompt)], temperature=0.8)
        except LLMError:
            return []
        data = parse_json_object(reply) or {}
        problems = [str(p).strip() for p in (data.get("problems") or []) if str(p).strip()]
        return problems[:n]

    def run(self, domains: list[str] | None = None, *, per_domain: int = 3, difficulty: str = "moderate",
            max_problems: int | None = None) -> PracticeSummary:
        chosen = domains or [d.id for d in DOMAINS if d.id != "general"]
        summary = PracticeSummary()
        total = 0
        for domain_id in chosen:
            problems = self.generate(domain_id, per_domain, difficulty)
            self._emit("practice_generated", domain=domain_id, count=len(problems))
            for problem in problems:
                if max_problems is not None and total >= max_problems:
                    return self._finish(summary)
                total += 1
                started = time.time()
                sol = self.solver.solve(problem, save=True)
                passed = (sol.confidence is not None and sol.confidence >= self.min_confidence
                          and bool(sol.verification.get("passes", False)))
                trace = str(sol.output_path.with_suffix("").with_suffix(".trace.json")) if sol.output_path else ""
                summary.results.append(PracticeResult(domain_id, problem, sol.confidence, sol.attempts, passed, trace,
                                                      time.time() - started))
                self.seen.append(f"{domain_id}:{problem}")
                self._emit("practice_solved", domain=domain_id, passed=passed, confidence=sol.confidence,
                           attempts=sol.attempts)
        return self._finish(summary)

    def _finish(self, summary: PracticeSummary) -> PracticeSummary:
        self.seen_path.parent.mkdir(parents=True, exist_ok=True)
        self.seen_path.write_text(json.dumps(self.seen[-2000:]), encoding="utf-8")
        return summary
