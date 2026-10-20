"""Build training data from what the system has already done and verified.

Sources
-------
1. Run traces (``research_output/*.trace.json``): every agent answer that passed its own
   verification with confidence >= ``min_confidence`` becomes a supervised example
   (task -> answer).  When the run used a stronger online model, those examples distil
   the teacher into the local student.
2. Retry history: an answer that failed verification followed by one that passed gives a
   preference pair (rejected, chosen) for DPO-style training.
3. The experience store: facts become question/answer pairs, skills become
   "write a procedure" -> code examples.
4. Self-practice runs (see ``practice.py``) are ordinary traces and flow through (1).

Everything is deduplicated, split into train/eval, and written as JSONL in the chat
format ``{"messages": [{"role": ...}, ...]}`` that the trainer and most tools accept.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from ..domains import get_domain

STUDENT_SYSTEM = (
    "You are OmniResearch, a careful research and problem-solving assistant for any domain. "
    "Think step by step, be specific (numbers, units, sources), separate evidence from interpretation, "
    "and state uncertainty honestly."
)


@dataclass
class Example:
    system: str
    prompt: str
    response: str
    source: str = ""          # trace path / "experience:fact" / "practice"
    agent: str = ""
    confidence: float | None = None
    teacher: str = ""         # backend that produced the response
    tags: list[str] = field(default_factory=list)

    def key(self) -> str:
        return hashlib.sha1((self.prompt.strip() + "\x00" + self.response.strip()).encode()).hexdigest()

    def to_chat(self) -> dict:
        return {"messages": [{"role": "system", "content": self.system},
                             {"role": "user", "content": self.prompt},
                             {"role": "assistant", "content": self.response}],
                "meta": {"source": self.source, "agent": self.agent, "confidence": self.confidence,
                         "teacher": self.teacher, "tags": self.tags}}


@dataclass
class PreferencePair:
    system: str
    prompt: str
    chosen: str
    rejected: str
    reason: str = ""
    source: str = ""

    def key(self) -> str:
        return hashlib.sha1((self.prompt + "\x00" + self.chosen + "\x00" + self.rejected).encode()).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)


def _role_system(agent: str) -> str:
    if agent.startswith("specialist:"):
        d = get_domain(agent.split(":", 1)[1])
        return (f"{STUDENT_SYSTEM} You are acting as the {d.name} specialist. {d.methods}")
    if agent == "analyst":
        return STUDENT_SYSTEM + " You are acting as the quantitative analyst: show the method, results with units, and assumptions."
    if agent == "solver":
        return STUDENT_SYSTEM + " You are acting as a general problem solver: understand, plan, solve, check."
    if agent == "synthesizer":
        return STUDENT_SYSTEM + " You are writing a complete, cited research report."
    return STUDENT_SYSTEM


_PLACEHOLDER = re.compile(r"\[mock backend\]|\(model error:|No scripted response", re.I)


def _usable(text: str, min_chars: int) -> bool:
    return bool(text) and len(text.strip()) >= min_chars and not _PLACEHOLDER.search(text)


class DatasetBuilder:
    def __init__(self, *, min_confidence: float = 0.7, min_chars: int = 80, eval_fraction: float = 0.1,
                 seed: int = 7) -> None:
        self.min_confidence = min_confidence
        self.min_chars = min_chars
        self.eval_fraction = eval_fraction
        self.seed = seed
        self.examples: dict[str, Example] = {}
        self.pairs: dict[str, PreferencePair] = {}
        self.stats = {"traces": 0, "findings_seen": 0, "accepted": 0, "rejected_low_confidence": 0,
                      "rejected_unverified": 0, "pairs": 0, "experience": 0}

    # -- sources -------------------------------------------------------------
    def add_example(self, ex: Example) -> bool:
        if not _usable(ex.response, self.min_chars) or not ex.prompt.strip():
            return False
        k = ex.key()
        if k in self.examples:
            return False
        self.examples[k] = ex
        return True

    def add_pair(self, pair: PreferencePair) -> bool:
        if not _usable(pair.chosen, self.min_chars) or not _usable(pair.rejected, 1) or pair.chosen == pair.rejected:
            return False
        k = pair.key()
        if k in self.pairs:
            return False
        self.pairs[k] = pair
        self.stats["pairs"] += 1
        return True

    def _finding(self, f: dict, trace_path: str, backend: str, extra_tags: list[str]) -> None:
        self.stats["findings_seen"] += 1
        conf = f.get("confidence")
        verification = f.get("verification") or {}
        passes = verification.get("passes", None)
        if conf is None or passes is None:
            # unverified work is not trusted as a training target
            self.stats["rejected_unverified"] += 1
            return
        if not passes or float(conf) < self.min_confidence:
            self.stats["rejected_low_confidence"] += 1
            return
        agent = str(f.get("agent", ""))
        task = str(f.get("task", "")).strip()
        content = str(f.get("content", "")).strip()
        content = re.sub(r"\n\n> Verifier notes:.*$", "", content, flags=re.S)  # strip appended reviewer notes
        ex = Example(_role_system(agent), task, content, source=trace_path, agent=agent, confidence=float(conf),
                     teacher=backend, tags=extra_tags + [agent.split(":")[0]])
        if self.add_example(ex):
            self.stats["accepted"] += 1
        for prev in f.get("attempt_history") or []:
            self.add_pair(PreferencePair(_role_system(agent), task, content, str(prev.get("content", "")),
                                         reason="; ".join(prev.get("problems") or []), source=trace_path))

    def add_trace(self, path: Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.stats["traces"] += 1
        backend = str(data.get("backend", ""))
        if data.get("kind") == "solve":
            finding = {"agent": "solver", "task": data.get("restated") or data.get("problem", ""),
                       "content": data.get("answer", ""), "confidence": data.get("confidence"),
                       "verification": data.get("verification") or {}, "attempt_history": data.get("attempt_history")}
            # the solver's saved markdown is the answer; the trace keeps the fields above
            md = Path(path).with_suffix("").with_suffix(".md")
            if not finding["content"] and md.exists():
                finding["content"] = md.read_text(encoding="utf-8").split("\n\n", 2)[-1]
            self._finding(finding, str(path), backend, [str(data.get("type", "solve")), str(data.get("strategy", ""))])
            return
        domains = [str(d) for d in (data.get("plan") or {}).get("domains") or []]
        for f in data.get("findings") or []:
            self._finding(f, str(path), backend, domains)

    def add_traces(self, directory: Path) -> int:
        n = 0
        for p in sorted(Path(directory).glob("*.trace.json")):
            try:
                self.add_trace(p)
                n += 1
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return n

    def add_experience(self, experience) -> int:
        """Facts -> Q/A, skills -> coding examples. Lessons and feedback shape the system prompt instead."""
        n = 0
        for e in experience.recent(limit=100000, kind="fact"):
            q = f"What is known about {e.title}? Give the specific figures and the source."
            if self.add_example(Example(STUDENT_SYSTEM, q, e.content, source="experience:fact", agent="fact",
                                        confidence=1.0, tags=["fact"] + [d for d in e.domains.split(",") if d])):
                n += 1
        for e in experience.recent(limit=100000, kind="skill"):
            desc = e.content.split("```", 1)[0].strip()
            code = e.content.split("```python", 1)[1].split("```", 1)[0].strip() if "```python" in e.content else ""
            if code and self.add_example(Example(STUDENT_SYSTEM + " Answer with working Python.",
                                                 f"Write a reusable Python procedure: {desc or e.title}",
                                                 f"```python\n{code}\n```", source="experience:skill", agent="skill",
                                                 confidence=1.0, tags=["skill"])):
                n += 1
        self.stats["experience"] += n
        return n

    # -- output --------------------------------------------------------------
    def split(self) -> tuple[list[Example], list[Example]]:
        items = sorted(self.examples.values(), key=lambda e: e.key())
        rng = random.Random(self.seed)
        rng.shuffle(items)
        n_eval = int(len(items) * self.eval_fraction) if len(items) >= 10 else 0
        return items[n_eval:], items[:n_eval]

    def write(self, out_dir: Path) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        train, evals = self.split()
        with (out_dir / "sft.jsonl").open("w", encoding="utf-8") as fh:
            for ex in train:
                fh.write(json.dumps(ex.to_chat(), ensure_ascii=False) + "\n")
        with (out_dir / "eval.jsonl").open("w", encoding="utf-8") as fh:
            for ex in evals:
                fh.write(json.dumps(ex.to_chat(), ensure_ascii=False) + "\n")
        with (out_dir / "dpo.jsonl").open("w", encoding="utf-8") as fh:
            for pair in sorted(self.pairs.values(), key=lambda p: p.key()):
                fh.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
        summary = {**self.stats, "train": len(train), "eval": len(evals), "dir": str(out_dir)}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        return summary


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
