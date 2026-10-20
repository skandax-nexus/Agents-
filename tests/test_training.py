"""Self-training pipeline: dataset mining, practice loop, evaluation, and (if torch is installed) a toy LoRA run."""
import json
from pathlib import Path

import pytest

from omniresearch.cognition.experience import Experience
from omniresearch.training.dataset import DatasetBuilder, Example, PreferencePair, load_jsonl
from omniresearch.training.evaluate import evaluate_backend, numeric_match, score, token_f1
from omniresearch.llm.mock import ScriptedBackend


def _trace(path: Path, findings: list[dict], backend="ollama (qwen)", kind=None, **extra) -> Path:
    data = {"backend": backend, "plan": {"domains": ["hydrology"]}, "findings": findings, **extra}
    if kind:
        data["kind"] = kind
    path.write_text(json.dumps(data))
    return path


def test_dataset_builder_filters_and_pairs(tmp_path: Path):
    good = {"agent": "specialist:hydrology", "task": "Estimate the Rhine's mean discharge at the Dutch border.",
            "content": "### Findings\nAbout 2,300 m3/s on average at Lobith, with floods above 10,000 m3/s.\n### Evidence & sources\nlocal\n### Confidence & gaps\nhigh\n\n> Verifier notes: fine",
            "confidence": 0.9, "verification": {"passes": True},
            "attempt_history": [{"attempt": 1, "content": "About 23,000 m3/s.", "problems": ["order of magnitude"]}]}
    low = {**good, "task": "q2", "confidence": 0.4, "attempt_history": []}
    failed = {**good, "task": "q3", "verification": {"passes": False}, "attempt_history": []}
    unverified = {"agent": "scout", "task": "q4", "content": "x" * 200}
    _trace(tmp_path / "a.trace.json", [good, low, failed, unverified])
    _trace(tmp_path / "b.trace.json", [good])   # duplicate -> deduplicated
    # solver trace
    (tmp_path / "c.trace.json").write_text(json.dumps({
        "kind": "solve", "backend": "anthropic (claude-opus-5-5)", "problem": "heat 2 L water by 50 K",
        "restated": "Energy to heat 2 L of water by 50 K", "type": "calculation", "strategy": "first_principles",
        "answer": "## Solution\n418.6 kJ = 0.116 kWh\n## Checks\npython recomputation agreed.",
        "confidence": 0.97, "verification": {"passes": True}, "attempt_history": []}))
    exp = Experience(tmp_path / "e.db")
    exp.add("fact", "Water specific heat", "c = 4186 J/(kg K), standard reference tables.", domains="physics")
    exp.add_skill("sensible_heat", "Q = m c dT", "def q(m,c,dT):\n    return m*c*dT\nprint(q(2,4186,50))")

    b = DatasetBuilder(min_confidence=0.7, min_chars=20)
    assert b.add_traces(tmp_path) == 3
    b.add_experience(exp)
    s = b.write(tmp_path / "data")
    assert s["findings_seen"] == 6 and s["accepted"] == 2 and s["rejected_low_confidence"] == 2
    assert s["rejected_unverified"] == 1 and s["pairs"] == 1 and s["experience"] == 2
    rows = load_jsonl(tmp_path / "data" / "sft.jsonl")
    assert len(rows) == 4 and s["eval"] == 0            # < 10 examples: no held-out split
    hydro = next(r for r in rows if r["meta"]["agent"] == "specialist:hydrology")
    assert "Verifier notes" not in hydro["messages"][-1]["content"]
    assert "Hydrology" in hydro["messages"][0]["content"]
    solver = next(r for r in rows if r["meta"]["agent"] == "solver")
    assert solver["meta"]["teacher"].startswith("anthropic") and "calculation" in solver["meta"]["tags"]
    pair = load_jsonl(tmp_path / "data" / "dpo.jsonl")[0]
    assert "23,000" in pair["rejected"] and "2,300" in pair["chosen"] and pair["reason"] == "order of magnitude"
    skill = next(r for r in rows if r["meta"]["agent"] == "skill")
    assert skill["messages"][-1]["content"].startswith("```python")


def test_dataset_split(tmp_path: Path):
    b = DatasetBuilder()
    for i in range(20):
        b.add_example(Example("s", f"question {i} about topic {i}", f"answer {i} " * 20))
    train, evals = b.split()
    assert len(train) == 18 and len(evals) == 2
    assert not b.add_pair(PreferencePair("s", "p", "same", "same"))


def test_scores():
    assert numeric_match("The answer is 418.6 kJ or 0.116 kWh", "418.6 kJ = 0.1163 kWh") == 1.0
    assert numeric_match("about 400 kJ", "418.6 kJ") == 0.0
    assert numeric_match("no numbers", "none here") is None
    assert token_f1("ocean acidification lowers aragonite saturation", "aragonite saturation falls with ocean acidification") > 0.7
    s = score('{"a": 1}', '{"a": 1}')
    assert s["json"] == 1.0 and s["token_f1"] is None and s["combined"] == 1.0


def test_evaluate_backend():
    rows = [{"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "2 L water by 50 K?"},
                          {"role": "assistant", "content": "418.6 kJ"}]},
            {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "capital of France"},
                          {"role": "assistant", "content": "Paris is the capital."}]}]
    backend = ScriptedBackend(handler=lambda s, m: "418.6 kJ" if "water" in m[-1].content else "Lyon")
    rep = evaluate_backend(backend, rows)
    assert rep.n == 2 and rep.numeric == 1.0 and 0 < rep.combined < 1


def test_practice_loop_keeps_only_verified(tmp_path: Path):
    from omniresearch.agents.orchestrator import ResearchOrchestrator
    from omniresearch.cognition.solver import ProblemSolver
    from omniresearch.config import Config
    from omniresearch.tools.registry import build_tools
    from omniresearch.training.practice import PracticeLoop

    state = {"n": 0}

    def handler(system, messages):
        last = messages[-1].content
        if system.startswith("You design practice problems"):
            return json.dumps({"problems": ["How much rain falls on 100 m2 in a 20 mm event?", "Why do rivers meander?"]})
        if system.startswith("You are the metacognition module"):
            if last.startswith("Classify"):
                return json.dumps({"type": "calculation", "strategy": "first_principles"})
            if last.startswith("Check this answer"):
                state["n"] += 1
                return json.dumps({"passes": state["n"] % 2 == 1, "confidence": 0.9 if state["n"] % 2 == 1 else 0.2,
                                   "problems": [] if state["n"] % 2 == 1 else ["vague"]})
            return json.dumps({"chosen_strategy": "compute"})
        if system.startswith("You are the reflection module"):
            return json.dumps({"lessons": [], "facts": [], "skills": []})
        return json.dumps({"action": "final", "content": "## Solution\n2 m3 (2,000 L) of water. " * 5})

    cfg = Config(home=tmp_path / "home", output_dir=tmp_path / "out", depth="standard", online=False,
                 max_attempts=1, instant_recall=False)
    tools, _ = build_tools(cfg, "t")
    orch = ResearchOrchestrator(cfg, backend=ScriptedBackend(handler=handler), tools=tools)
    loop = PracticeLoop(ProblemSolver(cfg, orch), min_confidence=0.7)
    summary = loop.run(["hydrology"], per_domain=2)
    assert len(summary.results) == 2 and summary.kept == 1
    assert len(list((tmp_path / "out").glob("*.trace.json"))) == 2
    assert (tmp_path / "home" / "training" / "practice_seen.json").exists()
    # the kept one becomes a training example
    b = DatasetBuilder(min_confidence=0.7, min_chars=20)
    b.add_traces(tmp_path / "out")
    assert b.stats["accepted"] == 1


def test_self_training_loop_dry_run(tmp_path: Path):
    from omniresearch.agents.orchestrator import ResearchOrchestrator
    from omniresearch.cognition.solver import ProblemSolver
    from omniresearch.config import Config
    from omniresearch.tools.registry import build_tools
    from omniresearch.training.loop import SelfTrainingLoop

    def handler(system, messages):
        if system.startswith("You design practice problems"):
            return json.dumps({"problems": ["p1"]})
        if system.startswith("You are the metacognition module"):
            if messages[-1].content.startswith("Check this answer"):
                return json.dumps({"passes": True, "confidence": 0.9})
            return json.dumps({"type": "calculation"})
        if system.startswith("You are the reflection module"):
            return "{}"
        return json.dumps({"action": "final", "content": "## Solution\nA verified answer with numbers 42 and 7." * 3})

    cfg = Config(home=tmp_path / "home", output_dir=tmp_path / "out", depth="standard", online=False, max_attempts=1)
    tools, _ = build_tools(cfg, "t")
    orch = ResearchOrchestrator(cfg, backend=ScriptedBackend(handler=handler), tools=tools)
    loop = SelfTrainingLoop(cfg, ProblemSolver(cfg, orch), train=False)
    res = loop.run(1, domains=["physics"], per_domain=1)[0]
    assert res.practice["kept"] == 1 and res.dataset["train"] == 1 and "dry run" in res.note
    assert (tmp_path / "home" / "training" / "rounds.jsonl").exists()


def test_lora_trainer_on_toy_model(tmp_path: Path):
    """End-to-end LoRA fine-tune on a tiny randomly initialised model (no download needed)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from omniresearch.training.trainer import LoRATrainer, TrainConfig

    rows = [{"messages": [{"role": "system", "content": "You are OmniResearch."},
                          {"role": "user", "content": f"What is {i} plus {i}?"},
                          {"role": "assistant", "content": f"{i} plus {i} is {2 * i}."}]} for i in range(12)]
    corpus = [m["content"] for r in rows for m in r["messages"]] + ["<|system|> <|user|> <|assistant|>"]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=300, special_tokens=["<unk>", "<s>", "</s>", "<pad>"]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
                                        pad_token="<pad>")
    base_dir = tmp_path / "base"
    tokenizer.save_pretrained(str(base_dir))
    cfg = LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256)
    LlamaForCausalLM(cfg).save_pretrained(str(base_dir))
    data = tmp_path / "data"
    data.mkdir()
    with (data / "sft.jsonl").open("w") as fh:
        for r in rows[:10]:
            fh.write(json.dumps(r) + "\n")
    with (data / "eval.jsonl").open("w") as fh:
        for r in rows[10:]:
            fh.write(json.dumps(r) + "\n")

    events = []
    trainer = LoRATrainer(TrainConfig(base_model=str(base_dir), output_dir=tmp_path / "run", epochs=8, batch_size=2,
                                      grad_accum=1, learning_rate=1e-2, device="cpu", lora_r=8, warmup_steps=2,
                                      log_every=10), on_event=lambda k, d: events.append(k))
    stats = trainer.train_sft(data / "sft.jsonl", data / "eval.jsonl")
    assert stats["examples"] == 10 and stats["updates"] == 40 and (tmp_path / "run" / "adapter").exists()
    assert stats["train_loss_last"] < stats["train_loss_first"]       # it learned something
    assert stats["eval_loss_after"] < stats["eval_loss_before"]
    merged = trainer.merge_and_save()
    assert (merged / "config.json").exists()
    info = trainer.export_ollama("toy-student", merged)
    assert Path(info["modelfile"]).read_text().startswith("FROM ")
    assert "model_loaded" in events and "train_done" in events
