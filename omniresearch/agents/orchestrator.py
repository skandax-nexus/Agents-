"""The main agent.

ResearchOrchestrator drives a full research run:

    question
      -> Planner        (domains + sub-questions)
      -> Scout          (what evidence exists locally / online)
      -> Specialists    (one per sub-question, in parallel)      ─┐
      -> Analyst        (sub-questions that need computation)     │ repeated for
      -> Critic         (gaps, unsupported claims, follow-ups)   ─┘ critique_rounds
      -> Synthesizer    (final report)
      -> saved to disk + remembered

Everything runs on whichever backend is configured, including fully offline ones.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from ..domains import Domain, get_domain
from ..domains.router import route_domains, validate_domain_ids
from ..llm import LLMBackend, build_backend
from ..tools import ToolRegistry, build_tools
from ..tools.knowledge import KnowledgeBase
from ..tools.memory import Memory
from ..cognition.experience import Experience, Reflector, trace_for_reflection
from ..cognition.instant import InstantLearner
from .base import AgentResult, EventHook
from .protocol import parse_json_object
from .subagents import AnalystAgent, CriticAgent, PlannerAgent, ScoutAgent, SpecialistAgent, SynthesizerAgent


@dataclass
class SubQuestion:
    question: str
    domain: Domain
    needs_computation: bool = False


@dataclass
class ResearchPlan:
    question: str
    restated: str
    domains: list[Domain]
    subquestions: list[SubQuestion]
    deliverable: str = ""
    raw: dict = field(default_factory=dict)

    def describe(self) -> str:
        lines = [f"Question: {self.restated}", "Domains: " + ", ".join(d.name for d in self.domains), "Sub-questions:"]
        for i, sq in enumerate(self.subquestions, 1):
            flag = " (computation)" if sq.needs_computation else ""
            lines.append(f"  {i}. [{sq.domain.name}] {sq.question}{flag}")
        if self.deliverable:
            lines.append(f"Deliverable: {self.deliverable}")
        return "\n".join(lines)


@dataclass
class ResearchReport:
    session_id: str
    question: str
    report: str
    plan: ResearchPlan
    findings: list[AgentResult]
    critiques: list[dict]
    sources: list[str]
    elapsed: float
    backend: str
    online: bool
    output_path: Path | None = None
    learned: dict = field(default_factory=dict)

    def trace(self) -> dict:
        return {
            "session_id": self.session_id,
            "question": self.question,
            "backend": self.backend,
            "online": self.online,
            "elapsed_seconds": round(self.elapsed, 1),
            "plan": {
                "restated": self.plan.restated,
                "domains": [d.id for d in self.plan.domains],
                "subquestions": [{"question": s.question, "domain": s.domain.id, "needs_computation": s.needs_computation}
                                 for s in self.plan.subquestions],
            },
            "findings": [
                {"agent": f.agent, "task": f.task, "steps": f.steps, "attempts": f.attempts,
                 "confidence": f.confidence, "elapsed": round(f.elapsed, 1), "thinking": f.thinking,
                 "verification": f.verification, "tool_calls": f.tool_calls, "sources": f.sources,
                 "attempt_history": f.attempt_history, "content": f.content}
                for f in self.findings
            ],
            "critiques": self.critiques,
            "sources": self.sources,
            "learned": self.learned,
        }


def _slug(text: str, n: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:n] or "research"


class ResearchOrchestrator:
    """The main research agent."""

    def __init__(self, config: Config | None = None, backend: LLMBackend | None = None,
                 tools: ToolRegistry | None = None, *, on_event: EventHook | None = None,
                 kb: KnowledgeBase | None = None, memory: Memory | None = None,
                 experience: Experience | None = None, learner: InstantLearner | None = None) -> None:
        self.config = config or Config()
        self.config.ensure_dirs()
        self.backend = backend or build_backend(self.config)
        self.session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.kb = kb or KnowledgeBase(self.config.knowledge_db)
        self.memory = memory or Memory(self.config.memory_db)
        self.experience = experience or Experience(self.config.experience_db)
        self.learner = learner or InstantLearner(self.kb, self.experience, self.config.answers_db)
        if tools is None:
            tools, online = build_tools(self.config, self.session_id, kb=self.kb, memory=self.memory,
                                        experience=self.experience, learner=self.learner)
        else:
            online = any(not t.offline for t in tools)
        self.tools = tools
        self.online = online
        self.on_event = on_event or (lambda kind, data: None)
        self.preset = self.config.preset()
        self.language = self.config.language

    def _cognition(self) -> dict:
        """ThinkingAgent options derived from the depth preset / config."""
        return {"deliberate": bool(self.preset.get("deliberate", 1)), "verify": bool(self.preset.get("verify", 1)),
                "max_attempts": int(self.preset.get("max_attempts", 2)), "experience": self.experience,
                "min_confidence": self.config.min_confidence}

    # ------------------------------------------------------------------ events
    def _emit(self, kind: str, **data) -> None:
        try:
            self.on_event(kind, data)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ phases
    def plan(self, question: str) -> ResearchPlan:
        self._emit("phase", name="planning")
        planner = PlannerAgent(self.backend, max_domains=self.preset["max_domains"],
                               max_subquestions=self.preset["max_subquestions"], language=self.language,
                               on_event=self.on_event)
        prior = self.memory.search(question, limit=4)
        context = ""
        if prior:
            context = "Relevant notes from earlier research runs:\n" + "\n".join(
                f"- {p['topic']}: {p['content'][:300]}" for p in prior)
        lessons = self.experience.recall_text(question, kinds=("lesson", "feedback"), limit=5)
        if not lessons.startswith("("):
            context += f"\n\nLessons and user feedback from past work (apply them):\n{lessons}"
        result = planner.run(f"Research question: {question}", context)
        data = parse_json_object(result.content) or {}
        max_d, max_q = self.preset["max_domains"], self.preset["max_subquestions"]

        domains = validate_domain_ids(list(data.get("domains") or []), question, max_d)
        subqs: list[SubQuestion] = []
        for item in (data.get("subquestions") or [])[:max_q]:
            if isinstance(item, str):
                item = {"question": item}
            if not isinstance(item, dict) or not str(item.get("question", "")).strip():
                continue
            dom_id = str(item.get("domain", "") or "")
            domain = get_domain(dom_id) if dom_id and get_domain(dom_id).id != "general" else None
            if domain is None:
                routed = route_domains(str(item["question"]), max_domains=1)
                domain = routed[0] if routed[0].id != "general" else domains[0]
            subqs.append(SubQuestion(str(item["question"]).strip(), domain, bool(item.get("needs_computation", False))))
        if not subqs:
            # Fallback plan: one sub-question per routed domain.
            for d in domains:
                subqs.append(SubQuestion(f"{question} (from the perspective of {d.name})", d))
        plan = ResearchPlan(question=question, restated=str(data.get("restated_question") or question),
                            domains=domains, subquestions=subqs, deliverable=str(data.get("deliverable") or ""),
                            raw=data)
        self._emit("plan", plan=plan.describe())
        return plan

    def scout(self, plan: ResearchPlan) -> AgentResult:
        self._emit("phase", name="scouting sources")
        agent = ScoutAgent(self.backend, self.tools, max_steps=self.preset["max_steps"], language=self.language,
                           on_event=self.on_event)
        return agent.run(f"Find the best available evidence for: {plan.restated}", plan.describe())

    def _run_specialist(self, sq: SubQuestion, plan: ResearchPlan, scout_notes: str) -> AgentResult:
        agent = SpecialistAgent(sq.domain, self.backend, self.tools, max_steps=self.preset["max_steps"],
                                language=self.language, on_event=self.on_event, **self._cognition())
        context = f"Overall research question: {plan.restated}\n\nPlan:\n{plan.describe()}"
        if scout_notes:
            context += f"\n\nSource scout notes:\n{scout_notes[:3000]}"
        return agent.run(f"Sub-question: {sq.question}", context)

    def investigate(self, subqs: list[SubQuestion], plan: ResearchPlan, scout_notes: str) -> list[AgentResult]:
        self._emit("phase", name=f"investigating {len(subqs)} sub-question(s)")
        workers = max(1, min(self.config.parallel_agents, len(subqs)))
        if workers == 1:
            results = [self._run_specialist(sq, plan, scout_notes) for sq in subqs]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(lambda sq: self._run_specialist(sq, plan, scout_notes), subqs))
        comp = [sq for sq in subqs if sq.needs_computation]
        if comp:
            self._emit("phase", name="quantitative analysis")
            analyst = AnalystAgent(self.backend, self.tools, max_steps=self.preset["max_steps"],
                                   language=self.language, on_event=self.on_event, **self._cognition())
            tasks = "\n".join(f"- {sq.question}" for sq in comp)
            context = f"Overall question: {plan.restated}\n\nSpecialist findings so far:\n" + "\n\n".join(
                f"[{r.agent}] {r.summary(1500)}" for r in results)
            results.append(analyst.run(f"Work out the quantitative parts of these sub-questions:\n{tasks}", context))
        return results

    def critique(self, plan: ResearchPlan, findings: list[AgentResult], round_no: int) -> dict:
        self._emit("phase", name=f"critique round {round_no}")
        critic = CriticAgent(self.backend, self.tools, max_steps=max(2, self.preset["max_steps"] // 2),
                             max_followups=max(1, self.preset["max_subquestions"] // 2), language=self.language,
                             on_event=self.on_event)
        bundle = "\n\n".join(f"### {r.agent} - {r.task}\n{r.summary(3500)}" for r in findings)
        result = critic.run(f"Review the draft findings for: {plan.restated}", bundle)
        data = parse_json_object(result.content) or {"verdict": result.content[:1000], "followups": []}
        data.setdefault("followups", [])
        data.setdefault("issues", [])
        data.setdefault("missing", [])
        data.setdefault("unsupported_claims", [])
        data["round"] = round_no
        self._emit("critique", round=round_no, verdict=str(data.get("verdict", ""))[:300],
                   followups=len(data["followups"]))
        return data

    def synthesize(self, plan: ResearchPlan, findings: list[AgentResult], critiques: list[dict],
                   scout: AgentResult | None) -> AgentResult:
        self._emit("phase", name="writing report")
        writer = SynthesizerAgent(self.backend, self.tools, max_steps=3, language=self.language, on_event=self.on_event)
        parts = [f"## Plan\n{plan.describe()}"]
        if scout is not None:
            parts.append(f"## Source scout\n{scout.summary(4000)}")
        for r in findings:
            parts.append(f"## {r.agent} - {r.task}\n{r.summary(6000)}\nSources used: {', '.join(r.sources) or '(none recorded)'}")
        for c in critiques:
            parts.append(f"## Critic review (round {c.get('round')})\n{json.dumps(c, ensure_ascii=False, indent=1)[:4000]}")
        return writer.run(f"Write the final research report answering: {plan.restated}\nOriginal wording: {plan.question}",
                          "\n\n".join(parts))

    # ------------------------------------------------------------------ main entry point
    def research(self, question: str, *, save: bool = True) -> ResearchReport:
        started = time.time()
        question = question.strip()
        self._emit("start", question=question, backend=self.backend.describe(), online=self.online,
                   depth=self.config.depth)
        plan = self.plan(question)
        scout = self.scout(plan) if self.preset["max_steps"] >= 4 else None
        scout_notes = scout.content if scout else ""
        findings = self.investigate(plan.subquestions, plan, scout_notes)
        critiques: list[dict] = []
        for round_no in range(1, self.preset["critique_rounds"] + 1):
            review = self.critique(plan, findings, round_no)
            critiques.append(review)
            followups: list[SubQuestion] = []
            for item in review.get("followups") or []:
                if isinstance(item, str):
                    item = {"question": item}
                if not isinstance(item, dict) or not str(item.get("question", "")).strip():
                    continue
                dom = get_domain(str(item.get("domain", "")))
                if dom.id == "general":
                    dom = route_domains(str(item["question"]), max_domains=1)[0]
                    if dom.id == "general":
                        dom = plan.domains[0]
                followups.append(SubQuestion(str(item["question"]).strip(), dom, bool(item.get("needs_computation"))))
            if not followups:
                break
            findings += self.investigate(followups[: max(1, self.preset["max_subquestions"] // 2)], plan, scout_notes)

        final = self.synthesize(plan, findings, critiques, scout)
        sources: list[str] = []
        for r in ([scout] if scout else []) + findings + [final]:
            for s in r.sources:
                if s not in sources:
                    sources.append(s)
        report = ResearchReport(self.session_id, question, final.content, plan, findings, critiques, sources,
                                time.time() - started, self.backend.describe(), self.online)
        if self.config.reflect:
            report.learned = self.learn_from(report)
        if save:
            report.output_path = self._save(report)
            self._remember(report)
        self.learner.remember(question, report.report, sources=report.sources, session=self.session_id)
        self._emit("done", elapsed=report.elapsed, path=str(report.output_path) if report.output_path else "")
        return report

    # ------------------------------------------------------------------ learning
    def learn_from(self, report: ResearchReport) -> dict:
        """Reflect on the finished run and store lessons, facts and skills for future runs."""
        self._emit("phase", name="reflecting & learning")
        reflector = Reflector(self.backend, self.experience)
        learned = reflector.reflect(report.question, trace_for_reflection(report.findings), report.report,
                                    session=report.session_id, domains=",".join(d.id for d in report.plan.domains))
        self._emit("learned", **{k: v for k, v in learned.items() if k != "error"})
        return learned

    # ------------------------------------------------------------------ persistence
    def _save(self, report: ResearchReport) -> Path:
        out_dir = self.config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        base = f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(report.question)}"
        md_path = out_dir / f"{base}.md"
        header = (f"<!-- OmniResearch session {report.session_id} | backend: {report.backend} | "
                  f"online: {report.online} | {report.elapsed:.0f}s -->\n\n")
        md_path.write_text(header + report.report.strip() + "\n", encoding="utf-8")
        (out_dir / f"{base}.trace.json").write_text(json.dumps(report.trace(), ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
        return md_path

    def _remember(self, report: ResearchReport) -> None:
        summary = report.report.strip()
        m = re.search(r"## Executive summary\s*(.+?)(?:\n## |\Z)", summary, re.S | re.I)
        if m:
            summary = m.group(1).strip()
        self.memory.save(report.question, summary[:2000], tags=",".join(d.id for d in report.plan.domains),
                         session=report.session_id)

    # ------------------------------------------------------------------ follow-up chat
    def ask(self, question: str, report: ResearchReport | None = None, *, fresh: bool = False) -> str:
        """Quick answer: instantly from what was learned before if a close match exists, else one specialist."""
        question = question.strip()
        if self.config.instant_recall and not fresh and report is None:
            hit = self.learner.recall(question, min_similarity=self.config.recall_similarity)
            if hit:
                when = time.strftime("%Y-%m-%d %H:%M", time.localtime(hit["created"]))
                self._emit("instant_recall", similarity=hit["similarity"], learned_at=when)
                return (f"{hit['answer']}\n\n_(answered instantly from what I learned on {when}; "
                        f"similarity {hit['similarity']:.2f}. Ask with --fresh to work it out again.)_")
        domain = route_domains(question, max_domains=1)[0]
        agent = SpecialistAgent(domain, self.backend, self.tools, max_steps=self.preset["max_steps"],
                                language=self.language, on_event=self.on_event, **self._cognition())
        context = f"Previous report:\n{report.report[:12000]}" if report else ""
        result = agent.run(question, context)
        trustworthy = result.confidence is None or result.confidence >= self.config.min_confidence
        if trustworthy and result.content and not result.content.startswith("(model error"):
            self.learner.remember(question, result.content, confidence=result.confidence, sources=result.sources,
                                  session=self.session_id)
        return result.content
