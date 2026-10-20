"""The self-training cycle.

    round:  practice (self-set problems, verified)  ->  dataset  ->  LoRA train  ->  evaluate
            -> keep the student only if it beats the current model on the held-out set
            -> export to Ollama and make it the default offline model

Every step is optional and resumable, so the cycle also works piecewise from the CLI.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..cognition.solver import ProblemSolver
from ..config import Config
from ..llm import LLMBackend
from .dataset import DatasetBuilder, STUDENT_SYSTEM, load_jsonl
from .evaluate import evaluate_backend
from .practice import PracticeLoop
from .trainer import LoRATrainer, TrainConfig, TrainingUnavailable


@dataclass
class RoundResult:
    round: int
    practice: dict = field(default_factory=dict)
    dataset: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    eval_base: dict = field(default_factory=dict)
    eval_student: dict = field(default_factory=dict)
    adopted: bool = False
    export: dict = field(default_factory=dict)
    note: str = ""


def student_record_path(cfg: Config) -> Path:
    return cfg.home / "training" / "student.json"


def current_student(cfg: Config) -> dict | None:
    p = student_record_path(cfg)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return None
    return None


class SelfTrainingLoop:
    def __init__(self, cfg: Config, solver: ProblemSolver, *, base_model: str | None = None,
                 student_backend_factory=None, on_event=None, train: bool = True) -> None:
        self.cfg = cfg
        self.solver = solver
        self.base_model = base_model
        self.student_backend_factory = student_backend_factory   # callable(merged_dir|name) -> LLMBackend
        self.on_event = on_event or (lambda kind, data: None)
        self.train_enabled = train
        self.dir = cfg.home / "training"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _emit(self, kind: str, **data) -> None:
        try:
            self.on_event(kind, data)
        except Exception:  # noqa: BLE001
            pass

    # -- individual steps -----------------------------------------------------------
    def practice(self, domains: list[str] | None, per_domain: int, difficulty: str = "moderate",
                 max_problems: int | None = None) -> dict:
        loop = PracticeLoop(self.solver, min_confidence=self.cfg.min_confidence, on_event=self.on_event)
        return loop.run(domains, per_domain=per_domain, difficulty=difficulty, max_problems=max_problems).as_dict()

    def build_dataset(self) -> dict:
        builder = DatasetBuilder(min_confidence=max(0.7, self.cfg.min_confidence))
        builder.add_traces(self.cfg.output_dir)
        builder.add_experience(self.solver.orch.experience)
        summary = builder.write(self.dir / "data")
        self._emit("dataset", **summary)
        return summary

    def train(self, base_model: str, *, epochs: float = 2.0, max_examples: int | None = None) -> tuple[dict, LoRATrainer]:
        out = self.dir / "runs" / time.strftime("%Y%m%d-%H%M%S")
        trainer = LoRATrainer(TrainConfig(base_model=base_model, output_dir=out, epochs=epochs,
                                          max_examples=max_examples), on_event=self.on_event)
        stats = trainer.train_sft(self.dir / "data" / "sft.jsonl", self.dir / "data" / "eval.jsonl")
        dpo = trainer.train_dpo(self.dir / "data" / "dpo.jsonl")
        if dpo:
            stats["dpo"] = dpo
        return stats, trainer

    def evaluate(self, backend: LLMBackend, limit: int | None = None) -> dict:
        rows = load_jsonl(self.dir / "data" / "eval.jsonl") if (self.dir / "data" / "eval.jsonl").exists() else []
        if not rows:
            return {"n": 0, "note": "no held-out examples yet (need >= 10 training examples)"}
        return evaluate_backend(backend, rows, limit=limit, on_event=self.on_event).as_dict()

    # -- the full cycle -------------------------------------------------------------
    def run_round(self, round_no: int, *, domains: list[str] | None, per_domain: int, epochs: float = 2.0,
                  eval_limit: int | None = 40, student_name: str = "omniresearch-student") -> RoundResult:
        res = RoundResult(round=round_no)
        self._emit("round_start", round=round_no)
        res.practice = self.practice(domains, per_domain)
        res.dataset = self.build_dataset()
        if not self.train_enabled:
            res.note = "training disabled (dry run)"
            return self._finish(res)
        if res.dataset.get("train", 0) < 1:
            res.note = "nothing to train on yet"
            return self._finish(res)
        base = self.base_model
        if not base:
            res.note = "no base model given (--base <hf-id-or-dir>)"
            return self._finish(res)
        try:
            res.training, trainer = self.train(base, epochs=epochs)
        except TrainingUnavailable as exc:
            res.note = str(exc)
            return self._finish(res)
        res.eval_base = self.evaluate(self.solver.backend, eval_limit)
        merged = trainer.merge_and_save()
        if self.student_backend_factory is not None:
            student = self.student_backend_factory(merged)
            res.eval_student = self.evaluate(student, eval_limit)
            better = res.eval_student.get("n", 0) and res.eval_student.get("combined", 0) >= res.eval_base.get("combined", 0)
        else:
            # no way to run the student here; adopt on eval-loss improvement
            before, after = res.training.get("eval_loss_before"), res.training.get("eval_loss_after")
            better = before is None or after is None or after <= before
        res.adopted = bool(better)
        if res.adopted:
            res.export = trainer.export_ollama(student_name, merged, system_prompt=STUDENT_SYSTEM)
            record = {"ollama_model": student_name, "merged_dir": str(merged), "base_model": base,
                      "round": round_no, "created_in_ollama": res.export.get("created", False),
                      "eval_student": res.eval_student, "eval_base": res.eval_base, "time": time.time()}
            student_record_path(self.cfg).write_text(json.dumps(record, indent=1), encoding="utf-8")
            res.note = "student adopted" + ("" if res.export.get("created") else " (import into Ollama manually, see export)")
        else:
            res.note = "student did not beat the current model; kept the current one"
        return self._finish(res)

    def _finish(self, res: RoundResult) -> RoundResult:
        log = self.dir / "rounds.jsonl"
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(res.__dict__, ensure_ascii=False, default=str) + "\n")
        self._emit("round_done", round=res.round, adopted=res.adopted, note=res.note)
        return res

    def run(self, rounds: int = 1, **kw) -> list[RoundResult]:
        return [self.run_round(i + 1, **kw) for i in range(rounds)]
