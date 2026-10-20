import json
from pathlib import Path

from omniresearch.cognition.experience import Experience, Reflector, experience_tools
from omniresearch.cognition.solver import ProblemSolver
from omniresearch.cognition.thinking import ThinkingAgent, Thought, Verification
from omniresearch.config import Config
from omniresearch.llm.mock import ScriptedBackend
from omniresearch.tools.python_exec import python_tool
from omniresearch.tools.registry import ToolRegistry, build_tools


def test_experience_store(tmp_path: Path):
    exp = Experience(tmp_path / "e.db")
    a = exp.add("lesson", "Use basin names", "For hydrology, search by basin name.", domains="hydrology")
    assert exp.add("lesson", "Use basin names", "For hydrology, search by basin name.", domains="hydrology") == a  # dedup
    exp.add("fact", "Ocean pH", "Surface pH fell ~0.1 since pre-industrial.", domains="oceanography")
    exp.add_skill("aragonite", "compute omega", "def omega(): return 1.0\nprint(omega())")
    fb = exp.add_feedback("Lead with numbers.")
    assert exp.stats() == {"lesson": 1, "fact": 1, "skill": 1, "feedback": 1, "total": 4}
    rows = exp.recall("hydrology basin search")
    assert rows and rows[0].kind == "lesson" and rows[0].uses == 0
    assert exp.recall("basin")[0].uses == 1  # use counts tick up
    # feedback surfaces first when it matches
    got = exp.recall("numbers lead")
    assert got[0].kind == "feedback"
    assert "omega" in exp.recall_text("compute omega", kinds=("skill",))
    assert exp.recall_text("zzzz qqqq").startswith("(")
    exp.delete(fb)
    assert exp.stats()["feedback"] == 0


def test_experience_tools(tmp_path: Path):
    exp = Experience(tmp_path / "e.db")
    reg = ToolRegistry(experience_tools(exp, "s"))
    assert "No saved skill" in reg.run("skill_search", query="anything").output
    reg.run("skill_save", name="c_to_f", description="celsius to fahrenheit", code="def c_to_f(c): return c*9/5+32")
    assert "c_to_f" in reg.run("skill_search", query="fahrenheit").output
    reg.run("lesson_save", title="Check units", content="Always convert to SI first.", domains="physics")
    assert "Check units" in reg.run("experience_recall", query="units SI").output


def test_reflector_tolerates_garbage(tmp_path: Path):
    exp = Experience(tmp_path / "e.db")
    out = Reflector(ScriptedBackend(responses=["not json"]), exp).reflect("t", "trace", "answer")
    assert out["lessons"] == 0 and exp.stats()["total"] == 0


def test_thinking_agent_retries_until_verified(tmp_path: Path):
    """Attempt 1 fails the self-check, attempt 2 passes; the retry context carries the fix strategy."""
    calls = {"verify": 0}

    def handler(system, messages):
        last = messages[-1].content
        if system.startswith("You are the metacognition module"):
            if last.startswith("Check this answer"):
                calls["verify"] += 1
                if calls["verify"] == 1:
                    return json.dumps({"passes": False, "confidence": 0.3, "problems": ["wrong units"],
                                       "fix_strategy": "convert to km"})
                return json.dumps({"passes": True, "confidence": 0.95, "problems": []})
            return json.dumps({"understanding": "distance", "chosen_strategy": "compute", "success_criteria": ["km"]})
        if "failed verification" in last:
            assert "convert to km" in last
            return '{"action": "final", "content": "42 km"}'
        return '{"action": "final", "content": "42 miles"}'

    exp = Experience(tmp_path / "e.db")
    exp.add("lesson", "Units", "State units explicitly.")
    agent = ThinkingAgent("t", "sys", ScriptedBackend(handler=handler), ToolRegistry([python_tool(tmp_path)]),
                          max_steps=3, experience=exp, max_attempts=3)
    res = agent.run("How far is it, in distance units?")
    assert res.content.startswith("42 km") and res.attempts == 2 and res.confidence == 0.95
    assert res.thinking["chosen_strategy"] == "compute"
    # experience was injected into the acting prompt
    first_prompt = agent.backend.calls[1][1][0].content
    assert "State units explicitly" in first_prompt and "Success criteria: km" in first_prompt


def test_thinking_agent_without_metacognition(tmp_path: Path):
    agent = ThinkingAgent("t", "sys", ScriptedBackend(responses=['{"action":"final","content":"ok"}']),
                          ToolRegistry(), deliberate=False, verify=False)
    res = agent.run("x")
    assert res.content == "ok" and res.confidence is None and len(agent.backend.calls) == 1


def test_inconclusive_verification_does_not_retry():
    backend = ScriptedBackend(responses=['{"action":"final","content":"answer"}'])  # same reply for every call
    agent = ThinkingAgent("t", "sys", backend, ToolRegistry(), deliberate=False, verify=True, max_attempts=3)
    res = agent.run("x")
    assert res.attempts == 1 and res.confidence is None and res.verification["inconclusive"] is True


def test_thought_and_verification_parsing():
    t = Thought.from_dict({"hypotheses": "single", "success_criteria": ["a", "b"], "difficulty": "hard"})
    assert t.hypotheses == ["single"] and "Success criteria: a; b" in t.briefing()
    v = Verification.from_dict({"passes": "no", "confidence": "2"})
    assert v.passes is False and v.confidence == 1.0
    assert Verification.from_dict({"confidence": "bad"}).confidence == 0.5


def test_problem_solver_end_to_end(tmp_path: Path):
    def handler(system, messages):
        last = messages[-1].content
        if system.startswith("You are the metacognition module"):
            if last.startswith("Classify this problem"):
                return json.dumps({"restated": "Energy to heat 2 L of water by 50 K", "type": "calculation",
                                   "strategy": "first_principles", "why": "Q=mcdT", "needs_research": False,
                                   "deliverable": "a number in kJ"})
            if last.startswith("Check this answer"):
                return json.dumps({"passes": True, "confidence": 0.97})
            return json.dumps({"understanding": "Q=mcdT", "chosen_strategy": "compute", "success_criteria": ["kJ"]})
        if system.startswith("You are the reflection module"):
            return json.dumps({"lessons": [], "facts": [],
                               "skills": [{"name": "heat_energy", "description": "Q=mcdT", "code": "def q(m,c,dT): return m*c*dT\nprint(q(2,4186,50))"}]})
        if last.startswith("TOOL RESULT"):
            return json.dumps({"action": "final", "content": "## Solution\n418.6 kJ\n## Checks performed\npython"})
        return json.dumps({"action": "tool", "tool": "python", "input": {"code": "print(2*4186*50/1000)"}})

    cfg = Config(home=tmp_path / "home", output_dir=tmp_path / "out", depth="standard", online=False)
    tools, _ = build_tools(cfg, "t")
    from omniresearch.agents.orchestrator import ResearchOrchestrator
    orch = ResearchOrchestrator(cfg, backend=ScriptedBackend(handler=handler), tools=tools)
    events = []
    solver = ProblemSolver(cfg, orch, on_event=lambda k, d: events.append(k))
    assert "research" in solver.tools.names()
    sol = solver.solve("how much energy to heat 2 litres of water by 50 degrees")
    assert sol.ptype == "calculation" and sol.strategy == "first_principles"
    assert "418.6 kJ" in sol.answer and sol.confidence == 0.97
    assert sol.tool_calls and sol.tool_calls[0]["tool"] == "python" and "418.6" in sol.tool_calls[0]["output_preview"]
    assert sol.learned["skills"] == 1 and orch.experience.stats()["skill"] == 1
    assert sol.output_path and sol.output_path.exists()
    trace = json.loads(sol.output_path.with_suffix("").with_suffix(".trace.json").read_text())
    assert trace["type"] == "calculation" and trace["thinking"]["chosen_strategy"] == "compute"
    assert "classified" in events and "learned" in events


def test_config_cognition_overrides():
    assert Config(depth="quick").preset()["deliberate"] == 0
    p = Config(depth="quick", deliberate=True, verify=True, max_attempts=5).preset()
    assert p["deliberate"] == 1 and p["verify"] == 1 and p["max_attempts"] == 5
