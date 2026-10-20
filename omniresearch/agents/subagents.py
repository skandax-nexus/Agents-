"""The sub-agents: planner, domain specialist, source scout, analyst, critic, synthesizer."""
from __future__ import annotations

from ..domains import Domain
from ..domains.registry import catalogue_text
from ..llm import LLMBackend
from ..tools import ToolRegistry
from .base import Agent, EventHook
from ..cognition.thinking import ThinkingAgent
from .prompts import (ANALYST_ROLE, CRITIC_FORMAT, CRITIC_ROLE, PLANNER_FORMAT, PLANNER_ROLE, SCOUT_ROLE,
                      SPECIALIST_ROLE, SYNTHESIZER_ROLE, base_system)

LEARNING_TOOLS = ("experience_recall", "skill_search", "skill_save", "lesson_save")
RESEARCH_TOOLS = ("knowledge_search", "knowledge_read", "memory_recall", "memory_save", "read_file", "list_dir",
                  "python", "web_search", "web_fetch", "wikipedia") + LEARNING_TOOLS
SCOUT_TOOLS = ("knowledge_search", "knowledge_read", "memory_recall", "read_file", "list_dir",
               "web_search", "web_fetch", "wikipedia", "experience_recall")
ANALYST_TOOLS = ("python", "knowledge_search", "knowledge_read", "read_file") + LEARNING_TOOLS
CRITIC_TOOLS = ("python", "knowledge_search", "memory_recall", "experience_recall")
WRITER_TOOLS = ("knowledge_search", "knowledge_read", "python", "experience_recall")


def _cog(kwargs: dict) -> dict:
    """Pull ThinkingAgent options out of a kwargs dict (all optional)."""
    return {k: kwargs.pop(k) for k in ("deliberate", "verify", "max_attempts", "experience", "min_confidence")
            if k in kwargs}


class PlannerAgent(Agent):
    def __init__(self, backend: LLMBackend, *, max_domains: int, max_subquestions: int,
                 language: str = "English", on_event: EventHook | None = None) -> None:
        fmt = PLANNER_FORMAT.replace("{max_domains}", str(max_domains)).replace("{max_subquestions}", str(max_subquestions))
        extra = f"## Domain catalogue (use these ids)\n{catalogue_text()}\n{fmt}"
        system = (
            f"You are the research planner inside OmniResearch.\n{PLANNER_ROLE}\n"
            f"Any topic is in scope - if nothing fits, use the id 'general'.\n{extra}\nWrite in {language}."
        )
        super().__init__("planner", system, backend, ToolRegistry(), max_steps=1, on_event=on_event)


class SpecialistAgent(ThinkingAgent):
    def __init__(self, domain: Domain, backend: LLMBackend, tools: ToolRegistry, *, max_steps: int,
                 language: str = "English", on_event: EventHook | None = None, **cog) -> None:
        role = SPECIALIST_ROLE.format(name=domain.name, category=domain.category, description=domain.description,
                                      methods=domain.methods, sources=domain.sources)
        system = base_system(f"the {domain.name} specialist", role, tools.subset(RESEARCH_TOOLS).spec_text(),
                             language=language)
        super().__init__(f"specialist:{domain.id}", system, backend, tools.subset(RESEARCH_TOOLS),
                         max_steps=max_steps, on_event=on_event, domains=(domain.id,), **_cog(cog))
        self.domain = domain


class ScoutAgent(Agent):
    def __init__(self, backend: LLMBackend, tools: ToolRegistry, *, max_steps: int, language: str = "English",
                 on_event: EventHook | None = None) -> None:
        sub = tools.subset(SCOUT_TOOLS)
        system = base_system("the source scout", SCOUT_ROLE, sub.spec_text(), language=language)
        super().__init__("scout", system, backend, sub, max_steps=max_steps, on_event=on_event)


class AnalystAgent(ThinkingAgent):
    def __init__(self, backend: LLMBackend, tools: ToolRegistry, *, max_steps: int, language: str = "English",
                 on_event: EventHook | None = None, **cog) -> None:
        sub = tools.subset(ANALYST_TOOLS)
        system = base_system("the quantitative analyst", ANALYST_ROLE, sub.spec_text(), language=language)
        super().__init__("analyst", system, backend, sub, max_steps=max_steps, on_event=on_event,
                         domains=("statistics", "mathematics"), **_cog(cog))


class CriticAgent(Agent):
    def __init__(self, backend: LLMBackend, tools: ToolRegistry, *, max_steps: int, max_followups: int,
                 language: str = "English", on_event: EventHook | None = None) -> None:
        sub = tools.subset(CRITIC_TOOLS)
        role = CRITIC_ROLE.format(max_followups=max_followups)
        extra = ("When you are done checking, your FINAL answer's \"content\" must itself be the JSON review object "
                 "described here (as a string):" + CRITIC_FORMAT)
        system = base_system("the critic", role, sub.spec_text(), extra=extra, language=language)
        super().__init__("critic", system, backend, sub, max_steps=max_steps, on_event=on_event)


class SynthesizerAgent(Agent):
    def __init__(self, backend: LLMBackend, tools: ToolRegistry, *, max_steps: int, language: str = "English",
                 on_event: EventHook | None = None) -> None:
        sub = tools.subset(WRITER_TOOLS)
        system = base_system("the report writer", SYNTHESIZER_ROLE, sub.spec_text(), language=language)
        super().__init__("synthesizer", system, backend, sub, max_steps=max_steps, on_event=on_event)
