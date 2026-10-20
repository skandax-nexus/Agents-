"""End-to-end run of the orchestrator on a scripted backend - no model, no network."""
import json
from pathlib import Path

from omniresearch.agents.orchestrator import ResearchOrchestrator
from omniresearch.config import Config
from omniresearch.llm import ChatMessage
from omniresearch.llm.mock import ScriptedBackend
from omniresearch.tools.knowledge import KnowledgeBase
from omniresearch.tools.memory import Memory
from omniresearch.tools.registry import build_tools


def _handler(system: str, messages: list[ChatMessage]) -> str:
    """Role-aware fake model: returns a plan, tool calls, a critique or a report."""
    last = messages[-1].content
    if system.startswith("You are the research planner"):
        return json.dumps({
            "restated_question": "How much does thawing permafrost add to warming?",
            "domains": ["cryosphere", "climate_science"],
            "subquestions": [
                {"question": "What carbon stocks are held in permafrost?", "domain": "cryosphere", "needs_computation": False},
                {"question": "How much CO2 would 10% of that release?", "domain": "climate_science", "needs_computation": True},
            ],
            "deliverable": "A short evidence-based brief.",
        })
    if system.startswith("You are the critic"):
        if last.startswith("TOOL"):
            return json.dumps({"action": "final", "content": json.dumps({
                "overall_quality": "adequate", "issues": ["units unclear"], "unsupported_claims": [],
                "missing": [], "followups": [{"question": "What is the permafrost feedback range in CMIP6?", "domain": "climate_science"}],
                "verdict": "ok"})})
        return json.dumps({"action": "tool", "tool": "python", "input": {"code": "print(1500*0.1)"}})
    if system.startswith("You are the report writer"):
        return json.dumps({"action": "final", "content": "# Permafrost\n## Executive summary\nAbout 1500 GtC are stored [1].\n## Sources\n1. local doc"})
    if system.startswith("You are the quantitative analyst"):
        if last.startswith("TOOL"):
            return json.dumps({"action": "final", "content": "### Results\n150 GtC -> ~550 GtCO2"})
        return json.dumps({"action": "tool", "tool": "python", "input": {"code": "print(150*44/12)"}})
    # specialists & scout: search once, then answer
    if last.startswith("TOOL"):
        return json.dumps({"action": "final", "content": "### Findings\nPermafrost holds ~1500 GtC.\n### Evidence & sources\nlocal\n### Confidence & gaps\nmedium"})
    return json.dumps({"action": "tool", "tool": "knowledge_search", "input": {"query": "permafrost carbon"}})


def test_full_pipeline(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "permafrost.md").write_text("Northern permafrost soils store roughly 1500 gigatonnes of carbon.")
    cfg = Config(home=tmp_path / "home", output_dir=tmp_path / "out", depth="standard", online=False, parallel_agents=2)
    kb = KnowledgeBase(cfg.knowledge_db)
    kb.index_path(docs)
    memory = Memory(cfg.memory_db)
    tools, online = build_tools(cfg, "test", kb=kb, memory=memory)
    assert online is False and "web_search" not in tools.names()

    backend = ScriptedBackend(handler=_handler)
    events = []
    orch = ResearchOrchestrator(cfg, backend=backend, tools=tools, kb=kb, memory=memory,
                                on_event=lambda k, d: events.append(k))
    report = orch.research("permafrost thaw warming")

    assert report.report.startswith("# Permafrost")
    assert [d.id for d in report.plan.domains] == ["cryosphere", "climate_science"]
    agents = [f.agent for f in report.findings]
    assert "specialist:cryosphere" in agents and "specialist:climate_science" in agents and "analyst" in agents
    assert len(report.critiques) == 1 and report.critiques[0]["overall_quality"] == "adequate"
    # follow-up from the critic was investigated
    assert agents.count("specialist:climate_science") == 2
    assert any("permafrost.md" in s for s in report.sources)
    assert report.output_path and report.output_path.exists()
    trace = json.loads(report.output_path.with_suffix("").with_suffix(".trace.json").read_text())
    assert trace["plan"]["domains"] == ["cryosphere", "climate_science"]
    assert memory.count() == 1 and "1500 GtC" in memory.recent()[0]["content"]
    assert "plan" in events and "done" in events


def test_plan_fallback_on_garbage(tmp_path: Path):
    cfg = Config(home=tmp_path / "home", output_dir=tmp_path / "out", depth="quick", online=False)
    backend = ScriptedBackend(responses=["I cannot make JSON today."])
    tools, _ = build_tools(cfg, "t")
    orch = ResearchOrchestrator(cfg, backend=backend, tools=tools)
    plan = orch.plan("Why are honeybee pollinator populations declining in farmland?")
    assert plan.domains and plan.domains[0].id in {"ecology", "agriculture"}
    assert plan.subquestions


def test_agent_step_limit(tmp_path: Path):
    """An agent that never finishes is forced to answer after max_steps."""
    from omniresearch.agents.base import Agent
    from omniresearch.tools.registry import ToolRegistry
    from omniresearch.tools.python_exec import python_tool

    calls = {"n": 0}

    def handler(system, messages):
        calls["n"] += 1
        if "out of tool steps" in messages[-1].content:
            return '{"action": "final", "content": "forced"}'
        return '{"action": "tool", "tool": "python", "input": {"code": "print(%d)"}}' % calls["n"]

    agent = Agent("t", "sys", ScriptedBackend(handler=handler), ToolRegistry([python_tool(tmp_path)]), max_steps=3)
    res = agent.run("loop forever")
    assert res.content == "forced" and res.truncated and res.steps == 3 and len(res.tool_calls) == 3
