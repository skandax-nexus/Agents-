"""Preview of the cognitive loop with no model and no network.

A scripted model stands in for the LLM so the mechanics are visible:
recall -> deliberate -> act (python tool) -> self-check FAILS (unit error) -> retry with
the fix strategy -> self-check passes -> reflect -> lesson + skill stored -> second run
recalls the skill.

    python3 examples/demo_cognitive_loop.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omniresearch.cognition.solver import ProblemSolver  # noqa: E402
from omniresearch.config import Config  # noqa: E402
from omniresearch.agents.orchestrator import ResearchOrchestrator  # noqa: E402
from omniresearch.llm.mock import ScriptedBackend  # noqa: E402
from omniresearch.tools.registry import build_tools  # noqa: E402

PROBLEM = "How much energy does it take to heat 2 litres of water from 20 C to 70 C, in kJ and kWh?"
state = {"verify_calls": 0}


def fake_model(system: str, messages) -> str:
    """Plays the roles a real local model would play, with one deliberate mistake."""
    last = messages[-1].content
    if system.startswith("You are the metacognition module"):
        if last.startswith("Classify this problem"):
            return json.dumps({"restated": "Energy to raise 2 L of water by 50 K, reported in kJ and kWh",
                               "type": "calculation", "difficulty": "easy", "strategy": "first_principles",
                               "why": "Q = m c dT follows directly from the definition of specific heat",
                               "needs_research": False, "deliverable": "both numbers with units and a check"})
        if last.startswith("Check this answer"):
            state["verify_calls"] += 1
            if state["verify_calls"] == 1:
                return json.dumps({"passes": False, "confidence": 0.35,
                                   "problems": ["kWh value is wrong: 418.6 kJ is 0.116 kWh, not 1.16 kWh"],
                                   "unsupported_claims": [],
                                   "fix_strategy": "recompute kWh as kJ / 3600 in python and show the code"})
            return json.dumps({"passes": True, "confidence": 0.96, "problems": [], "unsupported_claims": []})
        return json.dumps({"understanding": "Sensible heat of 2 kg water over a 50 K rise",
                           "knowns": ["2 L water ~ 2 kg", "dT = 50 K", "c = 4186 J/kg/K"],
                           "unknowns": ["Q in kJ", "Q in kWh"],
                           "assumptions": ["no heat loss", "density 1 kg/L"],
                           "hypotheses": ["about 400 kJ, i.e. roughly 0.1 kWh"],
                           "strategies": ["first principles Q=mc dT", "look up a heating calculator"],
                           "chosen_strategy": "first principles with python, then convert units",
                           "first_steps": ["python: 2*4186*50"], "success_criteria": ["kJ and kWh both given", "units consistent"],
                           "risks": ["J vs kJ vs kWh conversion slip"], "difficulty": "easy"})
    if system.startswith("You are the reflection module"):
        return json.dumps({"lessons": [{"title": "Unit conversions in code", "content": "Do every unit conversion in the python tool, never by hand; the verifier caught a 10x kWh slip.", "domains": "physics"}],
                           "facts": [{"title": "Water specific heat", "content": "c_water = 4186 J/(kg K) (background knowledge)", "domains": "physics"}],
                           "skills": [{"name": "sensible_heat", "description": "Energy to heat a mass of water: Q = m c dT, returns kJ and kWh",
                                       "code": "def sensible_heat(mass_kg, dT_K, c=4186):\n    q_kj = mass_kg*c*dT_K/1000\n    return q_kj, q_kj/3600\nprint(sensible_heat(2, 50))"}],
                           "self_assessment": "Should have converted to kWh in code on the first attempt."})
    # --- the acting agent
    if "failed verification" in last and not last.startswith("TOOL"):
        return json.dumps({"action": "tool", "tool": "python", "input": {"code": "q=2*4186*50/1000; print(q, 'kJ', q/3600, 'kWh')"}})
    if last.startswith("TOOL RESULT"):
        if "kWh" in last:
            return json.dumps({"action": "final", "content": "## Solution\n**418.6 kJ = 0.116 kWh** (2 kg x 4186 J/kg/K x 50 K)\n\n## How I got there\n`q=2*4186*50/1000; q/3600`\n\n## Checks performed\nRecomputed in python after the first self-check flagged the kWh figure.\n\n## Caveats & confidence\nNo heat loss assumed."})
        return json.dumps({"action": "final", "content": "## Solution\n418.6 kJ, which is about 1.16 kWh.\n\n## How I got there\n2 kg x 4186 x 50 = 418,600 J"})
    return json.dumps({"action": "tool", "tool": "python", "input": {"code": "print(2*4186*50)"}})


def show(kind: str, data: dict) -> None:
    agent = data.get("agent", "")
    if kind == "classified":
        print(f"[understand]   type={data['type']} strategy={data['strategy']}  ({data['why']})")
    elif kind == "deliberation":
        print(f"[deliberate]   {agent}: {data['hypotheses']} hypothesis, difficulty={data['difficulty']}, strategy: {data['strategy']}")
    elif kind == "tool_call":
        print(f"[act]          {agent}: {data['tool']}({data['input'].get('code', '')})")
    elif kind == "verification":
        print(f"[verify]       {agent}: {'PASS' if data['passes'] else 'FAIL'} confidence={data['confidence']:.2f} problems={data['problems']}")
    elif kind == "retry":
        print(f"[retry]        {agent}: attempt {data['attempt']} rejected -> {data['reason']}")
    elif kind == "learned":
        print(f"[learn]        {data['lessons']} lesson, {data['facts']} fact, {data['skills']} skill | {data['self_assessment']}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(home=Path(tmp) / "home", output_dir=Path(tmp) / "out", depth="standard", online=False)
        tools, _ = build_tools(cfg, "demo")
        orch = ResearchOrchestrator(cfg, backend=ScriptedBackend(handler=fake_model), tools=tools)
        solver = ProblemSolver(cfg, orch, on_event=show)

        print(f"PROBLEM: {PROBLEM}\n")
        sol = solver.solve(PROBLEM, save=False)
        print(f"\nRESULT after {sol.attempts} attempt(s), confidence {sol.confidence:.2f}:\n{sol.answer}\n")

        print("--- what the system now remembers (available offline to every future run) ---")
        for e in orch.experience.recent():
            print(f"  [{e.kind}] {e.title}: {e.content.splitlines()[0][:90]}")
        print("\n--- a new task recalls the saved skill ---")
        print(tools.run("skill_search", query="heat water energy").output[:300])


if __name__ == "__main__":
    main()
