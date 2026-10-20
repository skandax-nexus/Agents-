"""Instant learning: teach it once, it answers from storage next time - no model needed."""
import json
from pathlib import Path

from omniresearch.agents.orchestrator import ResearchOrchestrator
from omniresearch.cognition.experience import Experience
from omniresearch.cognition.instant import InstantLearner, instant_tools, similarity
from omniresearch.cognition.solver import ProblemSolver
from omniresearch.config import Config
from omniresearch.llm.mock import ScriptedBackend
from omniresearch.tools.knowledge import KnowledgeBase
from omniresearch.tools.registry import ToolRegistry, build_tools


def _learner(tmp_path: Path) -> InstantLearner:
    kb = KnowledgeBase(tmp_path / "kb.db")
    return InstantLearner(kb, Experience(tmp_path / "e.db"), tmp_path / "a.db")


def test_similarity():
    assert similarity("How tall is Mount Everest?", "what is the height of mount everest") > 0.3
    assert similarity("How tall is Mount Everest?", "best pasta recipe") == 0.0


def test_learn_text_is_searchable_immediately(tmp_path: Path):
    L = _learner(tmp_path)
    info = L.learn_text("The Rhine is about 1,233 km long and drains into the North Sea.", title="Rhine length")
    assert info["stored"] and info["fact"] > 0
    assert "1,233 km" in L.kb.search("rhine length")[0]["content"]
    assert L.experience.recall("rhine")[0].kind == "fact"
    # long text goes to the knowledge base only
    big = L.learn_text("word " * 1000, title="big")
    assert big["stored"] and big["fact"] == 0 and L.kb.stats()["documents"] == 2


def test_learn_file(tmp_path: Path):
    L = _learner(tmp_path)
    f = tmp_path / "notes.md"
    f.write_text("Coral bleaching begins above a 1 C anomaly sustained for weeks.")
    assert L.learn_file(f)["stored"]
    assert not L.learn_file(tmp_path / "missing.txt")["stored"]
    assert L.kb.search("coral bleaching anomaly")


def test_remember_and_recall(tmp_path: Path):
    L = _learner(tmp_path)
    L.remember("What is the boiling point of water at sea level?", "100 C (212 F) at 1 atm.", confidence=0.95,
               sources=["kb://physics"])
    hit = L.recall("boiling point of water at sea level")
    assert hit and "100 C" in hit["answer"] and hit["sources"] == ["kb://physics"] and hit["similarity"] >= 0.6
    assert L.recall("boiling point of water at sea level")["id"] == hit["id"]
    assert L.stats()["instant_hits"] == 2
    assert L.recall("how do volcanoes form") is None
    # re-remembering the same question updates in place
    L.remember("What is the boiling point of water at sea level?", "100 C at 101.325 kPa.")
    assert L.stats()["answers"] == 1 and "kPa" in L.recall("water boiling point sea level")["answer"]
    L.forget(hit["id"])
    assert L.stats()["answers"] == 0


def test_placeholders_are_never_learned(tmp_path: Path):
    L = _learner(tmp_path)
    assert L.remember("q", "[mock backend] Received task: q") == 0
    assert L.remember("q", "(model error: timeout)") == 0
    assert L.stats()["answers"] == 0


def test_agent_learn_tool(tmp_path: Path):
    L = _learner(tmp_path)
    reg = ToolRegistry(instant_tools(L, "s"))
    out = reg.run("learn", text="Permafrost covers ~15% of Northern Hemisphere land.", title="Permafrost extent")
    assert not out.error and "Permafrost extent" in out.output
    assert "15%" in L.kb.search("permafrost extent")[0]["content"]
    assert reg.run("recall_answer", question="anything").output.startswith("No previously")
    L.remember("How much land is permafrost?", "About 15% of NH land.")
    assert "15%" in reg.run("recall_answer", question="how much land is permafrost").output


def test_ask_answers_instantly_second_time(tmp_path: Path):
    cfg = Config(home=tmp_path / "home", output_dir=tmp_path / "out", depth="quick", online=False, reflect=False)
    backend = ScriptedBackend(responses=['{"action":"final","content":"Everest is 8,849 m (2020 survey)."}'])
    tools, _ = build_tools(cfg, "t")
    orch = ResearchOrchestrator(cfg, backend=backend, tools=tools)
    first = orch.ask("How tall is Mount Everest?")
    assert first.startswith("Everest is 8,849 m")
    n_calls = len(backend.calls)
    second = orch.ask("how tall is mount everest")
    assert "8,849 m" in second and "answered instantly" in second
    assert len(backend.calls) == n_calls              # no model call at all
    assert "8,849" in orch.ask("How tall is Mount Everest?", fresh=True) and len(backend.calls) > n_calls


def test_solver_instant_recall(tmp_path: Path):
    cfg = Config(home=tmp_path / "home", output_dir=tmp_path / "out", depth="quick", online=False, reflect=False)
    backend = ScriptedBackend(handler=lambda s, m: '{"action":"final","content":"## Solution\\n418.6 kJ"}')
    tools, _ = build_tools(cfg, "t")
    orch = ResearchOrchestrator(cfg, backend=backend, tools=tools)
    solver = ProblemSolver(cfg, orch)
    s1 = solver.solve("energy to heat 2 litres of water by 50 degrees", save=False)
    assert "418.6" in s1.answer and s1.strategy != "instant_recall"
    s2 = solver.solve("energy to heat 2 litres of water by 50 degrees?", save=False)
    assert s2.strategy == "instant_recall" and "418.6" in s2.answer and s2.attempts == 0
    assert solver.solve("energy to heat 2 litres of water by 50 degrees", save=False, fresh=True).strategy != "instant_recall"
